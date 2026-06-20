"""Parse the AORUS MASTER 16 AM6H spec page into structured Key-Value records.

The live page (cfg.spec_url) returns HTTP 403 to non-browser clients, so the
canonical input is a locally saved copy of the page HTML
(``data/raw/am6h_spec.html``). The real page renders each spec as:

    <ul class="spec-item-list">
        <li class="spec-title"><div>中央處理器</div></li>
        <li class="spec-desc"><div>Intel® Core™ Ultra 9 ... <br> ...</div></li>
    </ul>

so one ``<ul class="spec-item-list">`` == one Key→Value pair. ``<br>`` tags
inside the value separate sub-lines (e.g. the long port list).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

from .config import cfg

# Browser-ish headers for the optional live fetch. The page usually still 403s
# from a datacenter IP; the offline HTML is the reliable path.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

# The page labels every category in Traditional Chinese while most values are in
# English. These English aliases are injected into the indexed text so an
# English question ("what GPU does it have?") can still match a zh-labelled row.
KEY_ALIASES: dict[str, list[str]] = {
    "作業系統": ["Operating System", "OS"],
    "中央處理器": ["CPU", "Processor"],
    "顯示晶片": ["GPU", "Graphics", "Graphics Card"],
    "顯示器": ["Display", "Screen", "Panel", "螢幕", "螢幕面板"],
    "記憶體": ["Memory", "RAM"],
    "儲存裝置": ["Storage", "SSD", "Disk"],
    "鍵盤種類": ["Keyboard"],
    "連接埠": ["Ports", "I/O", "Connectivity", "Interface"],
    "音效": ["Audio", "Sound", "Speakers"],
    "通訊": ["Networking", "Wireless", "WiFi", "Wi-Fi", "Bluetooth", "LAN"],
    "視訊鏡頭": ["Webcam", "Camera"],
    "安全裝置": ["Security", "TPM"],
    "電池": ["Battery"],
    "變壓器": ["Adapter", "Power Adapter", "Charger"],
    "尺寸": ["Dimensions", "Size"],
    "重量": ["Weight"],
    "顏色": ["Color", "Colour"],
}


@dataclass
class SpecRecord:
    index: int
    key: str                       # Chinese category label, e.g. "中央處理器"
    aliases: list[str] = field(default_factory=list)  # English aliases
    lines: list[str] = field(default_factory=list)    # value, split on <br>
    value: str = ""                # value lines joined with " / "

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# HTML acquisition
# --------------------------------------------------------------------------- #
def load_html(path: Path | None = None) -> str:
    path = Path(path) if path else cfg.raw_html
    if not path.exists():
        raise FileNotFoundError(
            f"Spec HTML not found at {path}. Save the page from a browser to "
            f"this location (the live URL blocks non-browser clients), or run "
            f"fetch_html() if your network is allowed to reach it."
        )
    return path.read_text(encoding="utf-8", errors="ignore")


def fetch_html(url: str | None = None, timeout: int = 30) -> str:
    """Best-effort live fetch. Often blocked (403); prefer the saved HTML."""
    import requests

    url = url or cfg.spec_url
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _div_lines(div) -> list[str]:
    """Turn a <div>…<br>…</div> value cell into a list of clean text lines."""
    for br in div.find_all("br"):
        br.replace_with("\n")
    text = div.get_text()
    lines = [ln.strip() for ln in text.split("\n")]
    return [ln for ln in lines if ln]  # drop empty separator segments


def parse_specs(html: str) -> list[SpecRecord]:
    soup = BeautifulSoup(html, "lxml")
    records = _parse_spec_item_lists(soup)
    if not records:
        # Fallback for unexpected markup changes: any 2-column-ish rows.
        records = _parse_generic(soup)
    return _dedupe(records)


def _parse_spec_item_lists(soup) -> list[SpecRecord]:
    """Primary parser for the real GIGABYTE structure."""
    records: list[SpecRecord] = []
    for ul in soup.select("ul.spec-item-list"):
        title = ul.select_one("li.spec-title div") or ul.select_one("li.spec-title")
        desc = ul.select_one("li.spec-desc div") or ul.select_one("li.spec-desc")
        if not title or not desc:
            continue
        key = title.get_text(strip=True)
        lines = _div_lines(desc)
        if not key or not lines:
            continue
        records.append(
            SpecRecord(
                index=len(records),
                key=key,
                aliases=KEY_ALIASES.get(key, []),
                lines=lines,
                value=" / ".join(lines),
            )
        )
    return records


def _parse_generic(soup) -> list[SpecRecord]:
    """Heuristic fallback: tables / definition lists with key+value pairs."""
    records: list[SpecRecord] = []
    # <table> rows with two cells
    for tr in soup.select("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) == 2:
            key = cells[0].get_text(strip=True)
            for br in cells[1].find_all("br"):
                br.replace_with("\n")
            lines = [ln.strip() for ln in cells[1].get_text().split("\n") if ln.strip()]
            if key and lines:
                records.append(
                    SpecRecord(
                        index=len(records),
                        key=key,
                        aliases=KEY_ALIASES.get(key, []),
                        lines=lines,
                        value=" / ".join(lines),
                    )
                )
    return records


def _dedupe(records: list[SpecRecord]) -> list[SpecRecord]:
    """Drop duplicate (key, value) pairs (variant tabs repeat the same specs)."""
    seen: set[tuple[str, str]] = set()
    out: list[SpecRecord] = []
    for r in records:
        sig = (r.key, r.value)
        if sig in seen:
            continue
        seen.add(sig)
        r.index = len(out)
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_jsonl(records: list[SpecRecord], path: Path | None = None) -> Path:
    path = Path(path) if path else cfg.spec_jsonl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_specs(path: Path | None = None) -> list[dict]:
    path = Path(path) if path else cfg.spec_jsonl
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
