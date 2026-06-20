"""Central configuration for the RAG system.

Every value has a sane default and can be overridden with an environment
variable, which is how the Docker images and docker-compose inject runtime
settings (e.g. ``N_GPU_LAYERS=0`` for the CPU image, ``-1`` for the CUDA image).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Repo root = two levels up from this file (src/rag/config.py -> repo/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _int(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


def _bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _path(key: str, default: Path) -> Path:
    val = os.environ.get(key)
    return Path(val) if val else default


# --------------------------------------------------------------------------- #
# Model presets — pick a whole LLM stack with one env var: MODEL=qwen | llama.
# Each preset bundles the GGUF source + the chat/prompt format that model needs.
# Individual MODEL_REPO / MODEL_FILE / PROMPT_FORMAT env vars still override the
# chosen preset. The embedding model is SHARED across presets (see below), so the
# vector index is identical for both and an A/B compares only generation.
# --------------------------------------------------------------------------- #
MODEL_PRESETS: dict[str, dict[str, str]] = {
    "qwen": {
        "repo": "MaziyarPanahi/Qwen3-1.7B-GGUF",
        "file": "Qwen3-1.7B.Q4_K_M.gguf",
        "prompt_format": "qwen",     # ChatML + Qwen3 thinking toggle
    },
    "llama": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "prompt_format": "llama3",   # Llama-3 header template (<|start_header_id|>)
    },
}

_MODEL_KEY = _str("MODEL", "qwen").strip().lower()
_PRESET = MODEL_PRESETS.get(_MODEL_KEY, MODEL_PRESETS["qwen"])


@dataclass
class Config:
    # --- Source data ---------------------------------------------------------
    # The single source of truth: the AORUS MASTER 16 AM6H spec page, saved as
    # local HTML (the live page returns HTTP 403 to non-browser clients).
    spec_url: str = _str(
        "SPEC_URL",
        "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp",
    )
    raw_html: Path = _path("RAW_HTML", PROJECT_ROOT / "data" / "raw" / "am6h_spec.html")
    spec_jsonl: Path = _path("SPEC_JSONL", PROJECT_ROOT / "data" / "processed" / "spec.jsonl")
    # Authored variant (BZH/BYH/BXH GPU) data — the scraped page is a single
    # configuration, so the per-variant breakdown lives here and is merged into
    # the index at build time (see rag.variants.build_variant_chunks).
    variants_json: Path = _path("VARIANTS_JSON", PROJECT_ROOT / "data" / "raw" / "variants.json")

    # --- Vector index --------------------------------------------------------
    index_dir: Path = _path("INDEX_DIR", PROJECT_ROOT / "data" / "index")

    @property
    def embeddings_npy(self) -> Path:
        return self.index_dir / "embeddings.npy"

    @property
    def chunks_jsonl(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    # --- Embedding model (GGUF via llama.cpp, on CPU to leave VRAM for the LLM)
    # Qwen3-Embedding-0.6B: multilingual (zh + en). SHARED by every LLM preset so
    # retrieval is identical across models; runs on the same llama.cpp engine, so
    # no torch / transformers are needed. Lives in models/embedding/.
    embed_repo: str = _str("EMBED_REPO", "Qwen/Qwen3-Embedding-0.6B-GGUF")
    embed_file: str = _str("EMBED_FILE", "Qwen3-Embedding-0.6B-Q8_0.gguf")
    embed_n_ctx: int = _int("EMBED_N_CTX", 1024)
    embed_n_gpu_layers: int = _int("EMBED_N_GPU_LAYERS", 0)  # 0 = CPU

    # --- LLM (llama.cpp + GGUF) ---------------------------------------------
    # MODEL picks a preset (qwen|llama); MODEL_REPO/MODEL_FILE/PROMPT_FORMAT
    # override it. The GGUF lives in models/<model_key>/.
    models_dir: Path = _path("MODELS_DIR", PROJECT_ROOT / "models")
    model_key: str = _MODEL_KEY
    model_repo: str = _str("MODEL_REPO", _PRESET["repo"])
    model_file: str = _str("MODEL_FILE", _PRESET["file"])
    prompt_format: str = _str("PROMPT_FORMAT", _PRESET["prompt_format"])
    # Optional Hugging Face token for gated repos (read by download_model.py).
    hf_token: str = _str("HF_TOKEN", "") or _str("HUGGING_FACE_HUB_TOKEN", "")

    n_ctx: int = _int("N_CTX", 8192)
    # -1 = offload all layers to GPU 0 (fits easily for a 1.7B Q4 model in 4GB).
    n_gpu_layers: int = _int("N_GPU_LAYERS", -1)
    # Which physical GPU to use. Defaults to GPU 0; combine with
    # CUDA_VISIBLE_DEVICES=0 (set in compose) to pin a single device.
    main_gpu: int = _int("MAIN_GPU", 0)
    n_threads: int = _int("N_THREADS", os.cpu_count() or 4)
    flash_attn: bool = _bool("FLASH_ATTN", True)

    max_tokens: int = _int("MAX_TOKENS", 1024)
    temperature: float = _float("TEMPERATURE", 0.2)
    top_p: float = _float("TOP_P", 0.9)
    repeat_penalty: float = _float("REPEAT_PENALTY", 1.05)

    # Qwen3-1.7B is a hybrid reasoning model; we keep thinking OFF for a fast,
    # clean spec QA bot (see prompt.build_prompt -> enable_thinking).
    enable_thinking: bool = _bool("ENABLE_THINKING", False)

    # --- Retrieval -----------------------------------------------------------
    top_k: int = _int("TOP_K", 5)              # chunks fed to the LLM
    dense_top_k: int = _int("DENSE_TOP_K", 10)  # candidates from dense search
    lexical_top_k: int = _int("LEXICAL_TOP_K", 10)  # candidates from BM25
    use_hybrid: bool = _bool("USE_HYBRID", True)
    rrf_k: int = _int("RRF_K", 60)             # Reciprocal Rank Fusion constant

    # --- Server --------------------------------------------------------------
    host: str = _str("HOST", "0.0.0.0")
    port: int = _int("PORT", 8000)

    @property
    def model_dir(self) -> Path:
        """Per-model GGUF directory, e.g. models/qwen or models/llama."""
        return self.models_dir / self.model_key

    @property
    def embed_dir(self) -> Path:
        """Shared embedding GGUF directory: models/embedding."""
        return self.models_dir / "embedding"

    @property
    def results_dir(self) -> Path:
        """Per-model eval output directory: eval/results/<model_key>."""
        return PROJECT_ROOT / "eval" / "results" / self.model_key

    @property
    def model_path(self) -> Path:
        """Resolved path to the LLM GGUF file (MODEL_PATH wins if set)."""
        override = os.environ.get("MODEL_PATH")
        if override:
            return Path(override)
        return self.model_dir / self.model_file

    @property
    def embed_model_path(self) -> Path:
        """Resolved path to the embedding GGUF (EMBED_MODEL_PATH wins if set)."""
        override = os.environ.get("EMBED_MODEL_PATH")
        if override:
            return Path(override)
        return self.embed_dir / self.embed_file


# Importable singleton.
cfg = Config()
