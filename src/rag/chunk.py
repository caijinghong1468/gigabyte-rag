"""Chunking for structured Key-Value spec data.

Generic fixed-size character chunking is wrong for a spec sheet: it would split
"2560×1600 240Hz" away from "Display". Instead we chunk along the natural
structure with two granularities:

  * row-level  -- one chunk per spec category (中央處理器, 顯示晶片, ...). Good
                  for broad questions ("what's the screen like?").
  * line-level -- one chunk per sub-line of multi-line values (each port, each
                  display feature). Good for precise questions ("does it have
                  Thunderbolt 5?") so the exact fact sits alone in context.

Every chunk carries its category label plus English aliases inline, so a query
in either language can match a Chinese-labelled row.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    id: str
    text: str            # text that gets embedded AND shown to the LLM
    category: str        # Chinese category label (e.g. "連接埠")
    aliases: list[str]   # English aliases (e.g. ["Ports", "I/O"])
    granularity: str     # "row" | "line"
    source_index: int    # index of the originating spec record

    def to_dict(self) -> dict:
        return asdict(self)


# Footnotes / disclaimers add embedding noise and are never the answer.
_FOOTNOTE_MARKERS = (
    "may vary",
    "may differ",
    "please refer",
    "please contact",
    "contact your local",
    "warranty",
    "more info",
)


def _is_footnote(line: str) -> bool:
    # A leading "*" alone is NOT enough: GIGABYTE marks real spec content with
    # asterisks too (e.g. "* 2x SO-DIMM sockets for expansion"). Only drop lines
    # that actually read like a disclaimer (matched by the markers above).
    low = line.lower()
    return any(m in low for m in _FOOTNOTE_MARKERS)


def _clean(line: str) -> str:
    """Strip leading footnote asterisks from a kept spec line."""
    return line.lstrip("*").strip()


def _is_subheader(line: str) -> bool:
    # e.g. "Left Side:" / "Right Side:" -- useful as row context, not as a
    # standalone precise fact.
    return line.endswith(":") and len(line) <= 16


def _label(category: str, aliases: list[str]) -> str:
    return f"{category}（{' / '.join(aliases)}）" if aliases else category


def build_chunks(records: list[dict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for rec in records:
        idx = rec["index"]
        category = rec["key"]
        aliases = rec.get("aliases", [])
        label = _label(category, aliases)

        content_lines = [_clean(ln) for ln in rec["lines"] if not _is_footnote(ln)]
        content_lines = [ln for ln in content_lines if ln]
        if not content_lines:
            content_lines = rec["lines"]  # keep something rather than nothing

        # Row-level chunk: the whole category.
        chunks.append(
            Chunk(
                id=f"row-{idx}",
                text=f"{label}：{' / '.join(content_lines)}",
                category=category,
                aliases=aliases,
                granularity="row",
                source_index=idx,
            )
        )

        # Line-level chunks: only when the value has multiple meaningful lines.
        precise = [ln for ln in content_lines if not _is_subheader(ln)]
        if len(precise) > 1:
            for j, line in enumerate(precise):
                chunks.append(
                    Chunk(
                        id=f"line-{idx}-{j}",
                        text=f"{label}：{line}",
                        category=category,
                        aliases=aliases,
                        granularity="line",
                        source_index=idx,
                    )
                )
    return chunks
