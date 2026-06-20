"""End-to-end RAG pipeline: retrieve -> build prompt -> stream generation.

This is the object the CLI, the HTTP server, and the evaluator all use. After
each answer, ``self.last`` holds the retrieved chunks, retrieval latency, and
the LLM metrics (TTFT / TPS) so callers can display or log them.
"""

from __future__ import annotations

import time

from .config import cfg
from .embed import Embedder
from .index import VectorIndex
from .llm import LlamaLLM
from .prompt import build_prompt, detect_lang
from .retrieve import HybridRetriever


class RagPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LlamaLLM | None = None,
        enable_thinking: bool | None = None,
    ):
        self.retriever = retriever
        self.llm = llm or LlamaLLM()
        self.enable_thinking = cfg.enable_thinking if enable_thinking is None else enable_thinking
        self.last: dict = {}

    @classmethod
    def from_artifacts(cls, **kwargs) -> "RagPipeline":
        """Load the prebuilt index + embedder and assemble the pipeline."""
        index = VectorIndex.load()
        embedder = Embedder()
        retriever = HybridRetriever(index, embedder)
        return cls(retriever, **kwargs)

    # Leave a little slack between (prompt + max_tokens) and n_ctx so we never
    # bump the context window even if the tokenizer estimate is off by a few.
    _CTX_SAFETY = 64

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict]:
        return self.retriever.search(question, top_k=top_k)

    def _retrieval_query(self, question: str, history: list[dict] | None) -> str:
        """Prepend the most recent user turn so elliptical follow-ups still
        retrieve the right chunk (e.g. "那 BYH 呢？" after a BZH question)."""
        if not history:
            return question
        prev_user = next(
            (h["content"] for h in reversed(history) if h.get("role") == "user"), None
        )
        return f"{prev_user}\n{question}" if prev_user else question

    def _fit_history(
        self, question: str, retrieved: list[dict], history: list[dict] | None
    ) -> list[dict]:
        """Keep the most recent turns that fit the leftover context budget:
        n_ctx − max_tokens − (system + spec context + question) − safety.

        Walks history newest→oldest, dropping older turns once the budget is
        spent, so a long back-and-forth degrades gracefully instead of
        overflowing n_ctx."""
        if not history:
            return []
        base = build_prompt(question, retrieved, enable_thinking=self.enable_thinking)
        budget = cfg.n_ctx - cfg.max_tokens - self.llm.count_tokens(base) - self._CTX_SAFETY
        if budget <= 0:
            return []
        kept: list[dict] = []
        used = 0
        for turn in reversed(history):
            # +8 ≈ per-turn role/template wrapper tokens (<|im_start|>role…<|im_end|>).
            cost = self.llm.count_tokens(turn["content"]) + 8
            if used + cost > budget:
                break
            kept.append(turn)
            used += cost
        kept.reverse()
        return kept

    def answer_stream(
        self, question: str, top_k: int | None = None, history: list[dict] | None = None
    ):
        """Generator yielding answer tokens. Fills self.last afterwards.

        ``history`` is the prior conversation as ``{role, content}`` turns; it is
        token-budgeted here and threaded into the prompt for multi-turn memory.
        """
        t0 = time.perf_counter()
        retrieved = self.retrieve(self._retrieval_query(question, history), top_k=top_k)
        retrieval_s = time.perf_counter() - t0

        history = self._fit_history(question, retrieved, history)
        prompt = build_prompt(
            question, retrieved, enable_thinking=self.enable_thinking, history=history
        )

        answer_parts: list[str] = []
        for token in self.llm.stream(prompt):
            answer_parts.append(token)
            yield token

        self.last = {
            "question": question,
            "lang": detect_lang(question),
            "retrieved": retrieved,
            "retrieved_categories": [r["chunk"]["category"] for r in retrieved],
            "answer": "".join(answer_parts),
            "retrieval_s": round(retrieval_s, 4),
            "metrics": self.llm.last_metrics,
        }

    def answer(
        self, question: str, top_k: int | None = None, history: list[dict] | None = None
    ) -> dict:
        """Non-streaming: run to completion and return self.last."""
        for _ in self.answer_stream(question, top_k=top_k, history=history):
            pass
        return self.last
