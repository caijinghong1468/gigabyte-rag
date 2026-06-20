"""Step 2: chunk the specs, embed them, and save the vector index.

Input : data/processed/spec.jsonl
Output: data/index/embeddings.npy + data/index/chunks.jsonl
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rag.chunk import build_chunks  # noqa: E402
from rag.embed import Embedder  # noqa: E402
from rag.index import VectorIndex  # noqa: E402
from rag.scrape import load_specs  # noqa: E402
from rag.variants import build_variant_chunks  # noqa: E402


def main() -> None:
    records = load_specs()
    chunks = build_chunks(records)
    # Merge authored BZH/BYH/BXH variant chunks (GPU differences + the platform
    # relationship), which the single-config scraped page does not contain.
    variant_chunks = build_variant_chunks(start_index=len(records))
    chunks += variant_chunks
    n_row = sum(1 for c in chunks if c.granularity == "row")
    n_line = sum(1 for c in chunks if c.granularity == "line")
    print(
        f"[i] {len(records)} records + {len(variant_chunks)} variant chunks "
        f"-> {len(chunks)} chunks ({n_row} row, {n_line} line)"
    )

    print("[i] Loading embedding model (Qwen3-Embedding GGUF via llama.cpp, CPU)...")
    embedder = Embedder()
    vectors = embedder.encode([c.text for c in chunks], kind="passage")
    print(f"[i] Embedded {vectors.shape[0]} chunks, dim={vectors.shape[1]}")

    index = VectorIndex(vectors, [c.to_dict() for c in chunks])
    out = index.save()
    print(f"[OK] Index saved -> {out}")


if __name__ == "__main__":
    main()
