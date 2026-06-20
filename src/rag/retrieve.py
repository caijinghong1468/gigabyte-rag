"""Hybrid retrieval = dense (vectors) + lexical (BM25), fused with RRF.

Why hybrid for a spec sheet? Dense vectors capture meaning ("螢幕多大" ~ Display)
but can miss exact tokens, while spec questions are full of literal strings that
*must* match: "Thunderbolt 5", "DDR5", "RTX 5090", "240Hz", "99Wh". A small
hand-written BM25 nails those, and Reciprocal Rank Fusion (RRF) merges the two
ranked lists without needing to calibrate score scales.

Everything here -- the tokenizer, BM25, and the fusion -- is plain Python.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .config import cfg
from .embed import Embedder
from .index import VectorIndex

# Latin/number runs become one token each ("rtx", "5090", "ddr5", "240hz"),
# and every CJK character becomes its own token (good enough for zh matching
# without a segmenter dependency like jieba).
_LATIN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = _LATIN.findall(text)
    tokens += _CJK.findall(text)
    return tokens


class BM25:
    """Textbook Okapi BM25."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]

        df: Counter = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n = len(docs)
        # BM25+ style idf, floored at a small positive value so common terms
        # still contribute a little rather than going negative.
        self.idf = {
            term: math.log(1 + (n - dfi + 0.5) / (dfi + 0.5)) for term, dfi in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q_terms = tokenize(query)
        out = [0.0] * len(self.doc_tokens)
        for i, tf in enumerate(self.tf):
            dl = self.doc_len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                s += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + denom_norm)
            out[i] = s
        return out


def _ranked_positions(scores: list[float], k: int) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [i for i in order[:k] if scores[i] > 0] or order[:k]


class HybridRetriever:
    def __init__(self, index: VectorIndex, embedder: Embedder, use_hybrid: bool | None = None):
        self.index = index
        self.embedder = embedder
        self.chunks = index.chunks
        self.use_hybrid = cfg.use_hybrid if use_hybrid is None else use_hybrid
        self._id2pos = {c["id"]: i for i, c in enumerate(self.chunks)}
        # source_index -> position of that record's row-level chunk, used to pull
        # the full enumeration back when only a single line-level chunk ranked.
        self._row_by_source = {
            c["source_index"]: i
            for i, c in enumerate(self.chunks)
            if c.get("granularity") == "row"
        }
        self.bm25 = BM25([c["text"] for c in self.chunks]) if self.use_hybrid else None

    def _augment_with_parent_rows(self, retrieved: list[dict]) -> list[dict]:
        """Ensure the full row-level chunk is present for any line-level hit.

        Line-level chunking is great for precise facts, but on "list all" questions
        (ports, OS options, networking) a single line can rank while the complete
        list (the row chunk) does not — leaving the model to answer with only one
        item. We append the parent row of every retrieved line so the whole
        enumeration is in context. Appended as extras, the ranked top_k is intact.
        """
        present = {r["chunk"]["id"] for r in retrieved}
        extra: list[dict] = []
        for r in retrieved:
            c = r["chunk"]
            if c.get("granularity") != "line":
                continue
            pos = self._row_by_source.get(c["source_index"])
            if pos is None:
                continue
            row = self.chunks[pos]
            if row["id"] in present:
                continue
            present.add(row["id"])
            extra.append({
                "chunk": row, "score": r["score"],
                "dense_rank": None, "lexical_rank": None, "method": "row-augment",
            })
        return retrieved + extra

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or cfg.top_k

        # --- dense ----------------------------------------------------------
        qv = self.embedder.encode_query(query)
        dense_hits = self.index.search(qv, cfg.dense_top_k)
        dense_pos = [self._id2pos[h["chunk"]["id"]] for h in dense_hits]
        dense_score = {self._id2pos[h["chunk"]["id"]]: h["score"] for h in dense_hits}

        if not self.use_hybrid:
            return self._augment_with_parent_rows([
                {"chunk": self.chunks[p], "score": dense_score[p],
                 "dense_rank": r, "lexical_rank": None, "method": "dense"}
                for r, p in enumerate(dense_pos[:top_k])
            ])

        # --- lexical (BM25) -------------------------------------------------
        lex_scores = self.bm25.scores(query)
        lex_pos = _ranked_positions(lex_scores, cfg.lexical_top_k)

        # --- Reciprocal Rank Fusion ----------------------------------------
        rrf: dict[int, float] = {}
        dense_rank = {p: r for r, p in enumerate(dense_pos)}
        lex_rank = {p: r for r, p in enumerate(lex_pos)}
        for p, r in dense_rank.items():
            rrf[p] = rrf.get(p, 0.0) + 1.0 / (cfg.rrf_k + r + 1)
        for p, r in lex_rank.items():
            rrf[p] = rrf.get(p, 0.0) + 1.0 / (cfg.rrf_k + r + 1)

        fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return self._augment_with_parent_rows([
            {
                "chunk": self.chunks[p],
                "score": score,
                "dense_rank": dense_rank.get(p),
                "lexical_rank": lex_rank.get(p),
                "dense_score": dense_score.get(p),
                "lexical_score": lex_scores[p],
                "method": "hybrid",
            }
            for p, score in fused
        ])
