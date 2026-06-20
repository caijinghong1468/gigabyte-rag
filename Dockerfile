# ---------------------------------------------------------------------------
# Single image — the app on GPU 0.
#   * llama.cpp is pulled prebuilt from the CUDA (cu124) wheel index (pinned in
#     pyproject [tool.uv.sources]) so all layers offload to the GPU.
#   * Embeddings run on CPU on purpose, so the 4GB of VRAM is reserved entirely
#     for the LLM.
#   * Python comes from uv (managed CPython 3.11). We deliberately do NOT use the
#     distro's python3.11: Ubuntu 22.04 can ship a pre-release 3.11 that lacks
#     sys.get_int_max_str_digits, which crashes torch._dynamo on import.
# Requires the NVIDIA Container Toolkit on the host.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=hardlink \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_SYNC=1 \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    HF_HOME=/app/.hf \
    N_GPU_LAYERS=-1 \
    MAIN_GPU=0 \
    CUDA_VISIBLE_DEVICES=0 \
    MODEL=qwen

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv binary from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install a managed CPython 3.11, then resolve dependencies (cached layer).
COPY .python-version pyproject.toml ./
RUN UV_NO_SYNC=0 uv python install 3.11 \
    && UV_NO_SYNC=0 uv sync --no-install-project --no-dev

# Copy source and install the project itself, then drop the uv cache (the
# 1.7GB llama.cpp wheel archive) to reclaim space inside the image.
COPY . .
RUN UV_NO_SYNC=0 uv sync --no-dev \
    && uv cache clean \
    && chmod +x scripts/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["bash", "scripts/docker-entrypoint.sh"]
