"""A tiny hand-written vector index (NumPy only -- no FAISS, no framework).

The corpus is small (tens of chunks), so exact brute-force cosine similarity is
both fastest-to-implement and instant at query time. Embeddings are stored
unit-normalized, so cosine similarity is a single matrix-vector dot product.
This is the "implement retrieval yourself" core of the assignment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import cfg


class VectorIndex:
    def __init__(self, embeddings: np.ndarray, chunks: list[dict]):
        assert embeddings.shape[0] == len(chunks), "embeddings/chunks length mismatch"
        self.embeddings = embeddings.astype("float32")
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return the top_k chunks by cosine similarity.

        query_vec must be L2-normalized (Embedder does this). Because the stored
        embeddings are normalized too, ``embeddings @ query_vec`` *is* the cosine
        similarity for every chunk at once.
        """
        q = query_vec.astype("float32").reshape(-1)
        sims = self.embeddings @ q                       # (N,)
        k = min(top_k, len(self.chunks))
        # argpartition for top-k, then sort just those k by score (desc).
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [
            {"rank": r, "score": float(sims[i]), "chunk": self.chunks[i]}
            for r, i in enumerate(top)
        ]

    # --- persistence --------------------------------------------------------
    def save(self, index_dir: Path | None = None) -> Path:
        index_dir = Path(index_dir) if index_dir else cfg.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        return index_dir

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "VectorIndex":
        index_dir = Path(index_dir) if index_dir else cfg.index_dir
        emb_path = index_dir / "embeddings.npy"
        chunk_path = index_dir / "chunks.jsonl"
        if not emb_path.exists() or not chunk_path.exists():
            raise FileNotFoundError(
                f"Index not found in {index_dir}. Run scripts/build_index.py first."
            )
        embeddings = np.load(emb_path)
        with chunk_path.open(encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        return cls(embeddings, chunks)
