"""Variant (sub-model) corpus for the AORUS MASTER 16 platform.

AM6H is the *platform* (chassis) model. BZH / BYH / BXH are three specific
configuration models under it that differ ONLY in the discrete GPU (chip, video
memory, graphics power). The scraped spec page (``am6h_spec.html``) describes a
single configuration and contains no per-variant breakdown, so questions like
"what GPU does BYH have?" or "how does AM6H relate to BZH/BYH/BXH?" are
unanswerable from it — the model would have to guess.

This module turns ``data/raw/variants.json`` into extra retrieval chunks so the
RAG corpus actually contains:

  * a bilingual "model relationship" chunk (AM6H = platform, BZH/BYH/BXH = GPU
    sub-models) — answers the relationship / clarification questions;
  * a single GPU comparison chunk listing all three side by side — answers
    "what's the difference between BZH and BYH?";
  * one GPU chunk per variant, each carrying its model code inline so the
    retriever binds "BXH" → 140W instead of grabbing the generic GPU row.
"""

from __future__ import annotations

import json
from pathlib import Path

from .chunk import Chunk, _is_footnote
from .config import cfg

_GPU_CATEGORY = "顯示晶片"
_GPU_ALIASES = ["GPU", "Graphics", "Graphics Card"]
_REL_CATEGORY = "型號關係"
_REL_ALIASES = ["Model Relationship", "Platform", "Variants", "Sub-model"]


def load_variants(path: Path | None = None) -> dict:
    path = Path(path) if path else cfg.variants_json
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_summary(v: dict) -> str:
    """One-line GPU summary (chip / vram / power) for a variant, footnotes off."""
    lines = [ln for ln in v["lines"] if not _is_footnote(ln)]
    return " / ".join(lines)


def build_variant_chunks(start_index: int = 100, path: Path | None = None) -> list[Chunk]:
    data = load_variants(path)
    if not data:
        return []

    platform = data.get("platform", "AORUS MASTER 16 AM6H")
    variants = data.get("variants", [])
    chunks: list[Chunk] = []
    idx = start_index

    # 1) Model-relationship chunk (bilingual) + explicit code→GPU mapping. One
    #    retrieval should be enough to answer the platform/clarification Qs.
    rel = data.get("relationship", {})
    mapping = "；".join(
        f"{v['code']} = {v['lines'][0]}" for v in variants if v.get("lines")
    )
    rel_text = (
        f"{_REL_CATEGORY}（Model Relationship / Platform）："
        f"{rel.get('zh', '')}\n{rel.get('en', '')}\n"
        f"型號對應 / Model mapping: {mapping}"
    )
    chunks.append(
        Chunk(
            id=f"variant-rel-{idx}",
            text=rel_text,
            category=_REL_CATEGORY,
            aliases=_REL_ALIASES + [v["code"] for v in variants],
            granularity="row",
            source_index=idx,
        )
    )
    idx += 1

    # 2) Side-by-side GPU comparison chunk (helps "BZH vs BYH" style questions).
    if variants:
        cmp_parts = [
            f"{platform} {v['code']}：{_gpu_summary(v)}" for v in variants
        ]
        chunks.append(
            Chunk(
                id=f"variant-gpu-cmp-{idx}",
                text=(
                    "顯示晶片型號比較（GPU comparison across variants）：\n"
                    + "\n".join(cmp_parts)
                ),
                category=_GPU_CATEGORY,
                aliases=_GPU_ALIASES + [v["code"] for v in variants],
                granularity="row",
                source_index=idx,
            )
        )
        idx += 1

    # 3) One GPU chunk per variant, model code inline so retrieval binds tightly.
    for v in variants:
        code = v["code"]
        content = [ln for ln in v["lines"] if not _is_footnote(ln)]
        chunks.append(
            Chunk(
                id=f"variant-{code}-{idx}",
                text=(
                    f"{platform} {code} 顯示晶片（GPU / Graphics Card）："
                    + " / ".join(content)
                ),
                category=_GPU_CATEGORY,
                aliases=_GPU_ALIASES + [code],
                granularity="row",
                source_index=idx,
            )
        )
        idx += 1

    return chunks
