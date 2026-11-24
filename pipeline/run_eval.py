"""
Phase 3.2 — Extended evaluation runner
---------------------------------------
Run the full RAG pipeline against `evaluation/test_questions.json` and
write a metrics report.

Usage:
    python -m pipeline.run_eval
    python -m pipeline.run_eval --k 3 --out evaluation/results.json
    python -m pipeline.run_eval --questions custom_set.json

Output JSON shape:
    {
      "summary":      {...aggregates...},
      "per_question": [ {...} ],
      "config":       {"k": 5, "questions_path": "...", "ran_at": "...Z"}
    }

This is meant to be wired into CI: a regression in precision_at_k or
keyword_coverage between PRs should fail the build.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Make `app` importable when this is invoked as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.eval_harness import evaluate  # noqa: E402
from app.rag_service import ask        # noqa: E402


DEFAULT_QUESTIONS = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "test_questions.json"
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__), "..", "evaluation", "results.json"
)


def _load_questions(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of question objects")
    return data


def _print_summary(summary: dict) -> None:
    print()
    print("=" * 64)
    print(f"  RAG Evaluation Summary  (n={summary['n_questions']}, k={summary['k']})")
    print("=" * 64)
    rows = [
        ("Precision@k",         summary["precision_at_k"]),
        ("Recall@k",            summary["recall_at_k"]),
        ("MRR",                 summary["mrr"]),
        ("Keyword coverage",    summary["keyword_coverage"]),
        ("Answer relevance",    summary["answer_relevance"]),
        ("Context utilization", summary["context_utilization"]),
    ]
    for name, val in rows:
        bar = "█" * int(round(val * 30))
        print(f"  {name:<22} {val:>6.4f}  {bar}")
    lat = summary["latency_ms"]
    print(f"  Latency p50/p95/p99    {lat['p50']:>4} / {lat['p95']:>4} / {lat['p99']:>4} ms")
    print(f"  Cost total / per-q     ${summary['cost_usd_total']:.6f} / ${summary['cost_usd_per_question']:.6f}")
    print("=" * 64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the extended RAG eval harness.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS,
                        help="Path to test_questions.json (default: evaluation/test_questions.json)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output JSON path (default: evaluation/results.json)")
    parser.add_argument("--k", type=int, default=5, help="K for precision/recall/hit (default 5)")
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    print(f"Loaded {len(questions)} questions from {args.questions}")
    print(f"Running pipeline (k={args.k})...")

    report = evaluate(questions, ask, k=args.k)
    report["config"] = {
        "k":              args.k,
        "questions_path": args.questions,
        "ran_at":         datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {args.out}")

    _print_summary(report["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
