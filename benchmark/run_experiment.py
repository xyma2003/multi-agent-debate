#!/usr/bin/env python3
# benchmark/run_experiment.py
"""
Experiment runner for A/B evaluation and ablation study.

Runs selected variants on selected questions, computes evaluation metrics,
and saves results to results/<variant>.json.

Supports:
  - Resuming interrupted runs (skips already-saved results)
  - Pilot mode (first N questions only)
  - Selecting specific variants

Usage examples:
    # Pilot: run full_system + single_llm on first 5 questions
    uv run python benchmark/run_experiment.py --variants full_system single_llm --limit 5

    # Full ablation: all 5 multi-agent variants on all 30 questions
    uv run python benchmark/run_experiment.py --variants full_system no_prohibition sequential fixed_rounds fulltext_embedding

    # Single-LLM baseline only
    uv run python benchmark/run_experiment.py --variants single_llm

    # Everything
    uv run python benchmark/run_experiment.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
import uuid
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.baseline import run_single_llm
from benchmark.evaluator import evaluate_debate, evaluate_single_llm
from benchmark.variants import build_variant_graph, VARIANT_BUILDERS

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"

ALL_VARIANTS = list(VARIANT_BUILDERS.keys()) + ["single_llm"]
# Note: nli_detection is included via VARIANT_BUILDERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(limit: int | None = None) -> list[dict]:
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]
    return questions


def load_existing_results(variant: str) -> dict[int, dict]:
    """Load already-computed results for a variant. Keyed by question_id."""
    path = RESULTS_DIR / f"{variant}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["question_id"]: r for r in data.get("results", [])}


def save_results(variant: str, results: list[dict], meta: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{variant}.json"
    payload = {
        "variant": variant,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------

def _is_daily_limit(err: Exception) -> bool:
    """Return True if this 429 is a daily quota exhaustion (not retryable today)."""
    return "PerDay" in str(err)


def _retry_delay_seconds(err: Exception, default: float = 60.0) -> float:
    """Extract suggested retry delay from error message, fallback to default."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(err))
    return float(match.group(1)) + 2 if match else default


def _run_with_retry(fn, label: str, max_retries: int = 3):
    """Call fn(); on per-minute 429, wait and retry up to max_retries times.
    On daily limit or non-429 errors, return None immediately."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            err_str = str(e)
            if "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str:
                print(f"  {label}: ERROR — {e}")
                return None
            if _is_daily_limit(e):
                print(f"  {label}: DAILY LIMIT reached — skipping remaining questions")
                raise  # re-raise so caller can stop the loop
            delay = _retry_delay_seconds(e)
            if attempt < max_retries:
                print(f"  {label}: rate limited, waiting {delay:.0f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"  {label}: rate limit retries exhausted — skipping")
                return None
    return None


# ---------------------------------------------------------------------------
# Run one question for a multi-agent variant
# ---------------------------------------------------------------------------

def run_debate_variant(
    variant: str,
    question_id: int,
    question: str,
    max_rounds: int = 3,
) -> dict | None:
    """Run one question through a multi-agent variant. Returns serialized EvaluationResult."""
    label = f"[{variant}] q{question_id}"
    print(f"  {label}: {question[:60]}...")

    def _run():
        graph = build_variant_graph(variant)
        thread_id = str(uuid.uuid4())
        state = graph.invoke(
            {"topic": question, "max_rounds": max_rounds},
            config={"configurable": {"thread_id": thread_id}},
        )
        report = state.get("final_report")
        if report is None:
            print(f"  {label}: WARNING — no final_report in state")
            return None
        return evaluate_debate(
            report=report,
            question_id=question_id,
            question=question,
            system=variant,
        ).as_dict()

    return _run_with_retry(_run, label)


# ---------------------------------------------------------------------------
# Run one question for single-LLM baseline
# ---------------------------------------------------------------------------

def run_single_llm_variant(question_id: int, question: str) -> dict | None:
    """Run one question through single-LLM baseline. Returns serialized EvaluationResult."""
    label = f"[single_llm] q{question_id}"
    print(f"  {label}: {question[:60]}...")

    def _run():
        llm_result = run_single_llm(question_id=question_id, question=question)
        return evaluate_single_llm(llm_result).as_dict()

    return _run_with_retry(_run, label)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_variant(
    variant: str,
    questions: list[dict],
    max_rounds: int = 3,
    delay: int = 0,
) -> None:
    """Run all questions for one variant, resuming from last saved point."""
    print(f"\n{'='*60}")
    print(f"VARIANT: {variant}  ({len(questions)} questions, max_rounds={max_rounds})")
    print(f"{'='*60}")

    existing = load_existing_results(variant)
    results: list[dict] = list(existing.values())
    skipped = 0

    for q in questions:
        qid = q["id"]
        if qid in existing:
            skipped += 1
            continue

        try:
            if variant == "single_llm":
                result = run_single_llm_variant(qid, q["question"])
            else:
                result = run_debate_variant(variant, qid, q["question"], max_rounds)
        except Exception as e:
            if _is_daily_limit(e):
                print(f"\n  Daily quota exhausted. Stopping variant '{variant}'.")
                break
            print(f"  Skipping q{qid} (unexpected error: {e})")
            continue

        if result is not None:
            results.append(result)
            save_results(variant, results, {"max_rounds": max_rounds, "total_questions": len(questions)})
        else:
            print(f"  Skipping q{qid} (failed)")

        if delay > 0:
            print(f"  (waiting {delay}s before next question...)")
            time.sleep(delay)

    if skipped:
        print(f"\n  (Skipped {skipped} already-completed questions)")
    print(f"\n  Done. {len(results)} results saved for variant '{variant}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run debate evaluation experiments"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=ALL_VARIANTS,
        choices=ALL_VARIANTS,
        help="Which variants to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N questions (for pilot runs)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="Max debate rounds for multi-agent variants (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Seconds to sleep between questions (default: 0, use 30+ for NLI variant)",
    )
    args = parser.parse_args()

    questions = load_questions(limit=args.limit)
    print(f"\nLoaded {len(questions)} questions.")
    print(f"Variants to run: {args.variants}")

    for variant in args.variants:
        run_variant(variant, questions, max_rounds=args.max_rounds, delay=args.delay)

    print("\n\nAll variants complete. Run analysis/analysis.ipynb to view results.")


if __name__ == "__main__":
    main()
