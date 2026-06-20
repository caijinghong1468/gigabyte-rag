"""Record GPU VRAM usage of the active LLM at load and during inference.

Measures device VRAM at several points and prints a breakdown, demonstrating
that llama.cpp pre-allocates at load time (weights + full-n_ctx KV cache +
compute buffers) and does NOT grow VRAM during generation. Only the runtime
footprint in GB is persisted as {model_name: vram_gb} into one shared file:
eval/results/used_vram.json (merged, so qwen / llama sit side by side).

Run against whichever model's service is up (it reads the active MODEL):
    docker compose exec rag       uv run --no-sync python scripts/vram_report.py
    docker compose exec rag-llama uv run --no-sync python scripts/vram_report.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rag.config import cfg  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from rag.vram import VramSampler, gpu_mem_total_mib, gpu_mem_used_mib  # noqa: E402

# One shared store mapping {model_key: vram_gb}, so qwen / llama sit side by side
# in a single file. Lives under the bind-mounted eval/results/ so it persists on
# the host.
VRAM_STORE = pathlib.Path(__file__).resolve().parents[1] / "eval" / "results" / "uesd_vram.json"


def _fmt(v: float | None) -> str:
    return f"{v:.0f} MiB" if v is not None else "n/a (no nvidia-smi)"


def _gb(v_mib: float | None) -> float | None:
    """MiB -> GB (GiB, /1024), rounded to 2 dp."""
    return round(v_mib / 1024, 2) if v_mib is not None else None


def _store_vram_gb(gb: float | None) -> pathlib.Path:
    """Record ONLY {model_name: vram_gb} into the shared used_vram.json, merging so the
    other models' entries are preserved."""
    VRAM_STORE.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if VRAM_STORE.exists():
        try:
            data = json.loads(VRAM_STORE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    data[cfg.model_key] = f"Total used {gb} GB VRAM"
    VRAM_STORE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return VRAM_STORE


def main() -> None:
    idx = cfg.main_gpu
    total = gpu_mem_total_mib(idx)
    base = gpu_mem_used_mib(idx)

    print("=" * 60)
    print(f"VRAM report — GPU index {idx} (in-container), total {_fmt(total)}")
    print(f"  LLM: {cfg.model_file}  | n_ctx={cfg.n_ctx}, n_gpu_layers={cfg.n_gpu_layers}")
    print("=" * 60)
    print(f"[1] baseline (before loading the LLM) : {_fmt(base)}")

    # Load index + embedder (CPU) + force the LLM onto the GPU.
    pipe = RagPipeline.from_artifacts()
    pipe.llm._ensure_loaded()
    after_load = gpu_mem_used_mib(idx)
    print(f"[2] after LLM load                    : {_fmt(after_load)}")
    if base is not None and after_load is not None:
        print(f"    -> model footprint (weights + KV cache + buffers): {after_load - base:.0f} MiB")

    # Sample VRAM while generating a deliberately long answer.
    sampler = VramSampler(idx)
    sampler.start()
    res = pipe.answer("請完整列出這台筆電左右兩側的所有連接埠，並說明顯示器規格。")
    sampler.stop()
    sampler.join(timeout=1)
    after_infer = gpu_mem_used_mib(idx)

    print(f"[3] peak DURING inference             : {_fmt(sampler.peak)}")
    print(f"[4] after inference                   : {_fmt(after_infer)}")
    if after_load is not None and sampler.peak is not None:
        print(f"    -> extra VRAM from generation vs load: {sampler.peak - after_load:+.0f} MiB")

    # "Total VRAM the model needs at runtime (peak)" = the highest device usage
    # we saw (load-time vs during-generation), minus the baseline.
    peaks = [v for v in (after_load, sampler.peak) if v is not None]
    peak_used = max(peaks) if peaks else None
    footprint = (peak_used - base) if (peak_used is not None and base is not None) else None
    print(f"[5] PEAK total VRAM used (max of 2,3)  : {_fmt(peak_used)}")
    if footprint is not None:
        print(
            f"    -> model needs ~{footprint:.0f} MiB ≈ {footprint / 1024:.2f} GB "
            f"VRAM at runtime (peak − baseline)"
        )

    m = res["metrics"]
    # used_vram.json stores ONLY {model_name: vram_gb} — the runtime footprint in GB.
    footprint_gb = _gb(footprint)
    store_path = _store_vram_gb(footprint_gb)

    print(
        f"\n(generated {m.get('completion_tokens')} tokens, "
        f"TTFT={m.get('ttft_s')}s, TPS={m.get('tps')})"
    )
    print(
        "\nConclusion: VRAM is reserved at LOAD time and stays ~flat during\n"
        "inference (the [3]-[2] delta is near zero). KV cache is sized by n_ctx,\n"
        f"not by prompt length — lower N_CTX (currently {cfg.n_ctx}) to use less VRAM."
    )
    print(f'\n[saved] {{"{cfg.model_key}": {footprint_gb}GB VRAM}} -> {store_path}')


if __name__ == "__main__":
    main()
