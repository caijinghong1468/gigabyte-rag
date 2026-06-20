"""Download the GGUF models from Hugging Face.

Fetches, for the ACTIVE model (MODEL=qwen|llama):
  * the LLM            (MODEL_REPO / MODEL_FILE)  -> models/<model>/
  * the embedding model (EMBED_REPO / EMBED_FILE) -> models/embedding/  (shared)

Idempotent: a file already present is skipped. The exact GGUF filename inside a
repo varies between uploaders, so if MODEL_FILE is not found we auto-resolve the
best matching .gguf (preferring the same quant) and save it under the expected
name, keeping rag.config authoritative.

Gated repos: set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) in .env; it is passed to
every Hugging Face call below.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

from rag.config import cfg  # noqa: E402

_QUANTS = ("q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "q4_0", "q5_0")


def _token() -> str | None:
    return cfg.hf_token or None


def _resolve_filename(repo: str, wanted: str) -> str | None:
    """Find the real .gguf in `repo` that best matches `wanted`.

    Returns the repo-relative path to download, or None if it can't decide.
    """
    files = [f for f in list_repo_files(repo, token=_token()) if f.lower().endswith(".gguf")]
    if not files:
        return None
    # 1. exact path, or same basename anywhere in the repo
    if wanted in files:
        return wanted
    for f in files:
        if f.split("/")[-1].lower() == wanted.lower():
            return f
    # 2. same quant level as requested (e.g. q4_k_m)
    low = wanted.lower()
    quant = next((q for q in _QUANTS if q in low), None)
    if quant:
        match = [f for f in files if quant in f.lower()]
        if match:
            return min(match, key=len)  # prefer the plain single-file variant
    # 3. a single .gguf in the repo is unambiguous
    if len(files) == 1:
        return files[0]
    return None


def _download(repo: str, filename: str, dest_dir: pathlib.Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists():
        print(f"[skip] {dest.relative_to(cfg.models_dir.parent)} already present")
        return

    resolved = filename
    try:
        path = hf_hub_download(
            repo_id=repo, filename=filename, local_dir=str(dest_dir), token=_token()
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[..] '{filename}' not found directly in {repo}; resolving best match...")
        resolved = _resolve_filename(repo, filename)
        if not resolved:
            print(f"\n[!] Could not resolve a .gguf in {repo}: {exc}\n")
            try:
                ggufs = [f for f in list_repo_files(repo, token=_token()) if f.lower().endswith(".gguf")]
                print(f"Available .gguf files in {repo}:")
                for f in ggufs:
                    print(f"  - {f}")
                print("\nSet MODEL_FILE to one of the above (or fix HF_TOKEN if gated).")
            except Exception as exc2:  # noqa: BLE001
                print(f"(could not list repo files: {exc2})")
            sys.exit(1)
        path = hf_hub_download(
            repo_id=repo, filename=resolved, local_dir=str(dest_dir), token=_token()
        )

    # Normalise the on-disk name to what rag.config expects (model_file).
    got = pathlib.Path(path)
    if got.name != filename:
        got.replace(dest)
        path = str(dest)
    if resolved != filename:
        print(f"[i] resolved '{filename}' <- repo file '{resolved}'")
    print(f"[OK] {path}")


def main() -> None:
    print(f"[i] Active model: MODEL={cfg.model_key} ({cfg.model_repo} / {cfg.model_file})")
    _download(cfg.model_repo, cfg.model_file, cfg.model_dir)       # LLM -> models/<model>/
    _download(cfg.embed_repo, cfg.embed_file, cfg.embed_dir)       # embedding -> models/embedding/
    print("\n[done] models ready under", cfg.models_dir)


if __name__ == "__main__":
    main()
