"""Evaluate the RAG pipeline: quantitative (TTFT/TPS) + qualitative (quality).

Metrics produced:
  * TTFT and TPS  -- per question + avg / p50 / p95 (the quantitative ask).
  * Retrieval Hit@k -- did the expected spec category appear in the retrieved
    chunks? (spec questions only).
  * Answer correctness -- do the required keyword(s) appear in the answer?
  * Refusal correctness -- for out-of-scope questions, did the bot decline
    instead of hallucinating?
  * Language match -- zh question -> zh answer, en -> en.
  * <think> leakage -- sanity check that thinking really is disabled.

Writes eval/results/<model>/benchmark.md and prints a summary.

    uv run python eval/run_eval.py
"""

import json
import pathlib
import platform
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from rag.config import cfg  # noqa: E402
from rag.pipeline import RagPipeline  # noqa: E402
from rag.prompt import detect_lang  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.jsonl"
# Per-model output so the two models' runs don't overwrite each other.
RESULTS = HERE / "results" / cfg.model_key / "benchmark.md"

REFUSAL_MARKERS = [
    # zh
    "未提供", "未列", "沒有提供", "查無", "找不到", "無法在規格", "規格表未",
    "沒有相關", "未標示", "沒有說明", "未在規格", "規格未", "並未列出", "沒有列出",
    # en
    "not listed", "not specified", "not in the spec", "no information",
    "cannot find", "does not list", "does not provide", "doesn't provide",
    "do not provide", "not provided", "not available", "no mention",
]


# An answer counts as correct when it covers at least this fraction of the
# expected keyword groups. A hard all() is unfair against a "be concise" system
# prompt (a 7-certification question would need every badge verbatim) — partial
# coverage reflects answer quality far better.
COVERAGE_THRESHOLD = 0.7


def _norm(text: str) -> str:
    text = text.replace("®", "").replace("™", "").replace("©", "")
    # Normalise so bilingual / formatting variants still match: drop thousands
    # separators and unify the multiplication sign used in resolutions.
    text = text.replace(",", "").replace("×", "x")
    return text.lower()


def keyword_hit(answer: str, group: str) -> bool:
    """A keyword 'group' may list alternatives separated by '|'; any one counts.

    Use this to attach bilingual synonyms, e.g. ``"24 cores|24 核心|24核"`` so a
    Chinese answer is not penalised for not echoing an English keyword verbatim.
    """
    a = _norm(answer)
    return any(_norm(alt.strip()) in a for alt in group.split("|"))


def cjk_ratio(text: str) -> float:
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    letters = sum(1 for ch in text if ch.isalpha() or "一" <= ch <= "鿿")
    return cjk / letters if letters else 0.0


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else "CPU only"
    except Exception:  # noqa: BLE001
        return "CPU only"


def pct(values: list[float], p: float) -> float:
    return round(float(np.percentile(values, p)), 3) if values else 0.0


def main() -> None:
    questions = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[i] Loading pipeline and evaluating {len(questions)} questions...\n")
    pipe = RagPipeline.from_artifacts()

    rows = []
    for q in questions:
        res = pipe.answer(q["question"])
        answer = res["answer"]
        cats = res["retrieved_categories"]
        m = res["metrics"]

        # Hit@k is meaningful for any question that names an expected category.
        has_cat = q.get("expected_category") is not None
        hit = (q["expected_category"] in cats) if has_cat else None
        if q["type"] == "refusal":
            # Out-of-scope: correct == declined instead of hallucinating.
            correct = any(_norm(mk) in _norm(answer) for mk in REFUSAL_MARKERS)
            coverage = 1.0 if correct else 0.0
        else:
            kws = q["expected_keywords"]
            coverage = (sum(keyword_hit(answer, g) for g in kws) / len(kws)) if kws else 0.0
            correct = coverage >= COVERAGE_THRESHOLD

        want = q["lang"]
        # zh answers legitimately carry many English spec tokens (port names,
        # model numbers), so a low CJK threshold is enough to confirm framing.
        lang_ok = (cjk_ratio(answer) > 0.05) if want == "zh" else (cjk_ratio(answer) < 0.15)
        think_leak = "<think>" in answer

        rows.append({
            **q, "answer": answer, "hit": hit, "correct": correct, "coverage": coverage,
            "lang_ok": lang_ok, "think_leak": think_leak,
            "ttft": m.get("ttft_s", 0.0), "tps": m.get("tps", 0.0),
            "tokens": m.get("completion_tokens", 0), "retrieval_s": res["retrieval_s"],
        })
        flag = "OK " if correct else "XX "
        print(f"  [{flag}] {q['id']} ({q['lang']}/{q['type']}) "
              f"hit={hit} correct={correct} cov={coverage:.2f} ttft={m.get('ttft_s')}s tps={m.get('tps')}")

    # ---- aggregates --------------------------------------------------------
    answerable = [r for r in rows if r["type"] != "refusal"]
    refusal = [r for r in rows if r["type"] == "refusal"]
    with_cat = [r for r in rows if r["hit"] is not None]
    ttfts = [r["ttft"] for r in rows]
    tpss = [r["tps"] for r in rows if r["tps"] > 0]

    def rate(items, key):
        return f"{sum(1 for r in items if r[key])}/{len(items)}" if items else "n/a"

    summary = {
        "retrieval_hit": rate(with_cat, "hit"),
        "answer_correct": rate(answerable, "correct"),
        "answer_coverage_avg": round(float(np.mean([r["coverage"] for r in answerable])), 2) if answerable else 0.0,
        "refusal_correct": rate(refusal, "correct"),
        "lang_match": rate(rows, "lang_ok"),
        "think_leaks": sum(1 for r in rows if r["think_leak"]),
        "ttft_avg": round(float(np.mean(ttfts)), 3), "ttft_p50": pct(ttfts, 50), "ttft_p95": pct(ttfts, 95),
        "tps_avg": round(float(np.mean(tpss)), 2) if tpss else 0.0, "tps_p50": pct(tpss, 50), "tps_p95": pct(tpss, 95),
    }

    _write_report(rows, summary)
    print("\n==== SUMMARY ====")
    for k, v in summary.items():
        print(f"  {k:16s}: {v}")
    print(f"\n[OK] Report written -> {RESULTS}")


def _write_report(rows: list[dict], s: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append(f"# AORUS MASTER 16 AM6H — RAG Benchmark ({cfg.model_key})\n")
    L.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")

    L.append("## Environment & Config\n")
    L.append(f"- Platform: `{platform.platform()}` | Python `{platform.python_version()}`")
    L.append(f"- GPU: `{gpu_name()}`")
    L.append(f"- MODEL: `{cfg.model_key}` | prompt format: `{cfg.prompt_format}`")
    L.append(f"- LLM: `{cfg.model_repo}` / `{cfg.model_file}` (n_ctx={cfg.n_ctx}, n_gpu_layers={cfg.n_gpu_layers}, flash_attn={cfg.flash_attn})")
    L.append(f"- Thinking mode: `{cfg.enable_thinking}` | Embedding: `{cfg.embed_file}` (llama.cpp, CPU)")
    L.append(f"- Retrieval: hybrid={cfg.use_hybrid}, top_k={cfg.top_k}, dense_k={cfg.dense_top_k}, lexical_k={cfg.lexical_top_k}, rrf_k={cfg.rrf_k}\n")

    L.append("## Quantitative\n")
    L.append("| Metric | avg | p50 | p95 |")
    L.append("|---|---|---|---|")
    L.append(f"| TTFT (s) | {s['ttft_avg']} | {s['ttft_p50']} | {s['ttft_p95']} |")
    L.append(f"| TPS (tok/s) | {s['tps_avg']} | {s['tps_p50']} | {s['tps_p95']} |\n")

    L.append("## Qualitative\n")
    L.append(f"- Retrieval Hit@{cfg.top_k} (Qs with expected category): **{s['retrieval_hit']}**")
    L.append(f"- Answer correctness (answerable Qs, ≥{int(COVERAGE_THRESHOLD * 100)}% keyword coverage): **{s['answer_correct']}**")
    L.append(f"- Avg keyword coverage (answerable Qs): **{s['answer_coverage_avg']}**")
    L.append(f"- Refusal correctness (out-of-scope Qs): **{s['refusal_correct']}**")
    L.append(f"- Language match: **{s['lang_match']}**")
    L.append(f"- `<think>` leaks (should be 0): **{s['think_leaks']}**\n")

    L.append("## Per-question detail\n")
    L.append("| id | lang | type | hit | correct | cov | lang_ok | TTFT | TPS | answer |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ans = r["answer"].replace("\n", " ").strip()
        ans = (ans[:80] + "…") if len(ans) > 80 else ans
        L.append(
            f"| {r['id']} | {r['lang']} | {r['type']} | {r['hit']} | {r['correct']} "
            f"| {r['coverage']:.2f} | {r['lang_ok']} | {r['ttft']} | {r['tps']} | {ans} |"
        )

    # Full, untruncated model outputs (the table above clips answers to 80 chars).
    L.append("\n## Full answers\n")
    for r in rows:
        L.append(
            f"### {r['id']} ({r['lang']}/{r['type']}) — "
            f"hit={r['hit']} correct={r['correct']} cov={r['coverage']:.2f}\n"
        )
        L.append(f"**Q:** {r['question']}\n")
        L.append(f"**A:** {r['answer'].strip()}\n")

    RESULTS.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
