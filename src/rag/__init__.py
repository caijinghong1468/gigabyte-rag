"""Pure-Python RAG core for the GIGABYTE AORUS MASTER 16 AM6H spec assistant.

No RAG frameworks (no LangChain / LlamaIndex). Every stage of the pipeline
-- chunking, retrieval (dense + BM25 hybrid), and generation -- is hand-written
here so the logic is fully inspectable.
"""

from .config import Config, cfg

__all__ = ["Config", "cfg"]
