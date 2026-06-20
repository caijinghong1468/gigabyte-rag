"""Embedding model wrapper — GGUF via llama.cpp (no torch / transformers).

Uses the same inference engine as the LLM. The Qwen3-Embedding-0.6B GGUF runs on
CPU (EMBED_N_GPU_LAYERS=0) so the 4GB of VRAM stays reserved for the LLM.
Qwen3-Embedding is a last-token-pooled model: queries carry a short retrieval
instruction, documents are embedded as-is. Vectors are L2-normalized so the
index can use a plain dot product as cosine similarity.
"""

from __future__ import annotations

import numpy as np

from .config import cfg

# Qwen3-Embedding retrieval format: queries get an instruction; docs stay raw.
QUERY_INSTRUCT = (
    "Given a question about the GIGABYTE AORUS MASTER 16 AM6H laptop "
    "specifications, retrieve the spec entry that answers it."
)


class Embedder:
    def __init__(self, model_path=None, n_ctx: int | None = None, n_gpu_layers: int | None = None):
        from pathlib import Path

        import llama_cpp
        from llama_cpp import Llama

        path = str(model_path or cfg.embed_model_path)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Embedding GGUF not found at {path}. "
                f"Run: uv run python scripts/download_model.py"
            )

        # Force last-token pooling (Qwen3-Embedding) even if GGUF metadata omits
        # it, so embed() returns one pooled vector per input.
        kwargs = {}
        pooling_last = getattr(llama_cpp, "LLAMA_POOLING_TYPE_LAST", None)
        if pooling_last is not None:
            kwargs["pooling_type"] = pooling_last

        self._llm = Llama(
            model_path=path,
            embedding=True,
            n_ctx=n_ctx if n_ctx is not None else cfg.embed_n_ctx,
            n_gpu_layers=n_gpu_layers if n_gpu_layers is not None else cfg.embed_n_gpu_layers,
            verbose=False,
            **kwargs,
        )

    def _format(self, texts: list[str], kind: str) -> list[str]:
        if kind == "query":
            return [f"Instruct: {QUERY_INSTRUCT}\nQuery:{t}" for t in texts]
        return list(texts)

    def encode(self, texts: list[str], kind: str = "passage", batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized float32 embeddings, shape (len(texts), dim)."""
        raw = self._llm.embed(self._format(texts, kind))
        arr = np.asarray(raw, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / np.clip(norms, 1e-9, None)).astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        """Embed a single query -> shape (dim,)."""
        return self.encode([text], kind="query")[0]

    @property
    def dim(self) -> int:
        return int(self._llm.n_embd())
