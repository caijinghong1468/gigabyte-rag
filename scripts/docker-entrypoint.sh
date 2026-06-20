#!/usr/bin/env bash
# Self-bootstrap on container start, then serve.
#   1. Ensure both GGUF models (LLM + embedding) are present.
#   2. Build the vector index from the baked-in spec HTML (if missing).
#   3. Launch the streaming FastAPI server.
# models/ and data/ are volume-mounted, so steps 1-2 only run once.
#
# We run through `uv run --no-sync`: this uses the uv-managed venv built at image
# time and does NOT re-sync or hit the network at startup.
set -euo pipefail

# 1. Ensure both GGUF models (LLM + embedding) are present (idempotent).
echo "[entrypoint] Ensuring GGUF models are present..."
uv run --no-sync python scripts/download_model.py

# 2. Build the vector index from the baked-in spec HTML (needs the embedder).
if [ ! -f /app/data/processed/spec.jsonl ] \
    || [ ! -f /app/data/index/embeddings.npy ] \
    || [ ! -f /app/data/index/chunks.jsonl ]; then
    echo "[entrypoint] Incomplete index artifacts -- scraping spec HTML and building index..."
    uv run --no-sync python scripts/scrape.py
    uv run --no-sync python scripts/build_index.py
fi

echo "[entrypoint] Starting server on :${PORT:-8000} (N_GPU_LAYERS=${N_GPU_LAYERS})"
exec uv run --no-sync uvicorn app.server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"
