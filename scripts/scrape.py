"""Step 1: parse the saved spec HTML into structured Key-Value JSONL.

Input : data/raw/am6h_spec.html  (a browser-saved copy of the AM6H /sp page)
Output: data/processed/spec.jsonl
"""

import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rag import scrape  # noqa: E402
from rag.config import PROJECT_ROOT, cfg  # noqa: E402


def main() -> None:
    # Convenience: if the canonical raw file is missing but a web.html was
    # dropped at the repo root, adopt it.
    if not cfg.raw_html.exists():
        fallback = PROJECT_ROOT / "web.html"
        if fallback.exists():
            cfg.raw_html.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(fallback, cfg.raw_html)
            print(f"[i] Adopted {fallback} -> {cfg.raw_html}")

    html = scrape.load_html()
    records = scrape.parse_specs(html)
    out = scrape.save_jsonl(records)

    print(f"[OK] Parsed {len(records)} spec rows -> {out}\n")
    for r in records:
        n = len(r.lines)
        print(f"  • {r.key:6s} ({n} line{'s' if n != 1 else ''}): {r.value[:70]}")


if __name__ == "__main__":
    main()
