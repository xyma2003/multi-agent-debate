# benchmark/run_quality_eval.py
"""
Run LLM-as-judge quality evaluation across debate system variants.

Usage:
    python benchmark/run_quality_eval.py               # all overlapping questions
    python benchmark/run_quality_eval.py --limit 3     # first 3 questions only
    python benchmark/run_quality_eval.py --delay 10    # seconds between calls

Outputs: results/quality_scores.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
QUESTIONS_PATH = Path(__file__).parent / "questions.json"

SYSTEMS = ["single_llm", "full_system", "nli_detection"]


def load_results(variant: str) -> dict[int, dict]:
    """Load debate results keyed by question_id."""
    path = RESULTS_DIR / f"{variant}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["question_id"]: r for r in data["results"]}


def find_overlap(systems: list[str]) -> list[int]:
    """Find question IDs present in ALL systems (excluding high-conflict q31+)."""
    id_sets = []
    for s in systems:
        results = load_results(s)
        id_sets.append({qid for qid in results if qid < 31})
    if not id_sets:
        return []
    overlap = id_sets[0]
    for s in id_sets[1:]:
        overlap &= s
    return sorted(overlap)


def load_existing_scores() -> dict:
    """Load already-computed quality scores."""
    path = RESULTS_DIR / "quality_scores.json"
    if not path.exists():
        return {"scores": []}
    with open(path) as f:
        return json.load(f)


def save_scores(data: dict) -> None:
    path = RESULTS_DIR / "quality_scores.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def already_evaluated(existing: dict, question_id: int, system: str, judge: str) -> bool:
    for entry in existing.get("scores", []):
        if (entry["question_id"] == question_id
                and entry["system"] == system
                and entry.get("judge") == judge):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality evaluation on debate outputs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N questions (default: all overlapping)")
    parser.add_argument("--delay", type=int, default=5,
                        help="Seconds between LLM calls (default: 5)")
    parser.add_argument("--systems", nargs="+", default=SYSTEMS,
                        choices=SYSTEMS, help="Which systems to evaluate (default: all)")
    parser.add_argument("--judge", default="qwen",
                        choices=["qwen", "openai", "anthropic", "groq"],
                        help="Judge model backend (default: qwen/Qwen3-32b — different arch from debaters)")
    args = parser.parse_args()

    # Load questions
    with open(QUESTIONS_PATH) as f:
        questions = {q["id"]: q["question"] for q in json.load(f)}

    # Find overlapping question IDs
    qids = find_overlap(args.systems)
    if args.limit:
        qids = qids[:args.limit]
    print(f"Evaluating {len(qids)} questions × {len(args.systems)} systems "
          f"= {len(qids) * len(args.systems)} evaluations")
    print(f"Questions: {qids}")
    print(f"Systems:   {args.systems}\n")

    # Load all results upfront
    all_results = {s: load_results(s) for s in args.systems}

    # Load existing scores (resume support)
    existing = load_existing_scores()
    skipped = 0

    # LLM for judging — use a DIFFERENT model than the debaters (Groq llama-3.3-70b)
    import os
    os.environ["LLM_BACKEND"] = args.judge
    from debate.llm import _make_llm
    from benchmark.quality_evaluator import evaluate_analysis, extract_analysis_text  # noqa: E402
    llm = _make_llm()
    print(f"Judge: {args.judge}\n")

    for qid in qids:
        question_text = questions.get(qid, f"Question {qid}")
        print(f"\n{'='*60}")
        print(f"Q{qid}: {question_text[:70]}...")

        for system in args.systems:
            if already_evaluated(existing, qid, system, args.judge):
                print(f"  [{system}] already evaluated — skip")
                skipped += 1
                continue

            report = all_results[system].get(qid)
            if not report:
                print(f"  [{system}] no result found — skip")
                continue

            analysis_text = extract_analysis_text(report)
            print(f"  [{system}] evaluating...", end=" ", flush=True)

            score = evaluate_analysis(question_text, analysis_text, llm)
            if score is None:
                print("FAILED")
                continue

            entry = {
                "question_id": qid,
                "question": question_text,
                "system": system,
                "judge": args.judge,
                "total": round(score.total(), 3),
                "dimensions": {
                    "perspective_diversity": score.perspective_diversity.score,
                    "analytical_depth": score.analytical_depth.score,
                    "claim_specificity": score.claim_specificity.score,
                    "honest_uncertainty": score.honest_uncertainty.score,
                    "practical_utility": score.practical_utility.score,
                },
                "reasoning": {
                    "perspective_diversity": score.perspective_diversity.reasoning,
                    "analytical_depth": score.analytical_depth.reasoning,
                    "claim_specificity": score.claim_specificity.reasoning,
                    "honest_uncertainty": score.honest_uncertainty.reasoning,
                    "practical_utility": score.practical_utility.reasoning,
                },
            }
            existing["scores"].append(entry)
            save_scores(existing)

            dims = entry["dimensions"]
            print(f"total={entry['total']:.1f}  "
                  f"div={dims['perspective_diversity']}  "
                  f"depth={dims['analytical_depth']}  "
                  f"spec={dims['claim_specificity']}  "
                  f"uncert={dims['honest_uncertainty']}  "
                  f"util={dims['practical_utility']}")

            if args.delay:
                time.sleep(args.delay)

    print(f"\n{'='*60}")
    print(f"Done. {len(existing['scores'])} evaluations saved "
          f"(skipped {skipped} already-completed).")
    print_summary(existing)


def print_summary(data: dict) -> None:
    import statistics
    from collections import defaultdict

    scores_by_system: dict[str, list[float]] = defaultdict(list)
    dims_by_system: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for entry in data["scores"]:
        s = entry["system"]
        scores_by_system[s].append(entry["total"])
        for dim, val in entry["dimensions"].items():
            dims_by_system[s][dim].append(val)

    print("\n=== Quality Score Summary ===")
    print(f"{'System':<22} {'Total':>6}  {'Diversity':>9}  {'Depth':>7}  "
          f"{'Specific':>9}  {'Uncert':>8}  {'Utility':>8}")
    print("-" * 78)
    for system in SYSTEMS:
        if system not in scores_by_system:
            continue
        total = statistics.mean(scores_by_system[system])
        d = dims_by_system[system]
        print(f"{system:<22} {total:>6.2f}  "
              f"{statistics.mean(d['perspective_diversity']):>9.2f}  "
              f"{statistics.mean(d['analytical_depth']):>7.2f}  "
              f"{statistics.mean(d['claim_specificity']):>9.2f}  "
              f"{statistics.mean(d['honest_uncertainty']):>8.2f}  "
              f"{statistics.mean(d['practical_utility']):>8.2f}")


if __name__ == "__main__":
    main()
