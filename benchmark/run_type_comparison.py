# benchmark/run_type_comparison.py
"""
3-type PROHIBITION comparison experiment.

Tests the hypothesis: adaptive PROHIBITION should outperform full_system on
context_dependent questions, match it on values_based, and be competitive on binary.

Question sets:
  binary            → benchmark/questions_binary.json (q71-q80, n=10)
  values_based      → benchmark/questions_high_conflict.json (q31-q40, n=10)
  context_dependent → benchmark/questions.json (q1-q7, already labeled, n=7)

Systems compared:
  full_system          — full PROHIBITION on every question
  adaptive_prohibition — classifies question type, applies calibrated PROHIBITION

Evaluation: Qwen3-32b quality judge (5 dimensions). Type-specific focus:
  binary:             analytical_depth + claim_specificity
  values_based:       perspective_diversity + analytical_depth
  context_dependent:  claim_specificity + practical_utility (condition mapping)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent.parent / "results"
BENCHMARK_DIR = Path(__file__).parent

QUESTION_FILES = {
    "binary":             BENCHMARK_DIR / "questions_binary.json",
    "values_based":       BENCHMARK_DIR / "questions_high_conflict.json",
    "context_dependent":  BENCHMARK_DIR / "questions.json",
}

QUESTION_LIMITS = {
    "binary": 10,
    "values_based": 10,
    "context_dependent": 6,   # q1,q4,q5,q7,q8,q9 have context_dependent label
}

SYSTEMS = ["full_system", "adaptive_prohibition"]

TYPE_FOCUS_DIMS = {
    "binary":            ["analytical_depth", "claim_specificity"],
    "values_based":      ["perspective_diversity", "analytical_depth"],
    "context_dependent": ["claim_specificity", "practical_utility"],
}


# ---------------------------------------------------------------------------
# Debate runner (reuses existing variants)
# ---------------------------------------------------------------------------

def run_debate(question: str, system: str) -> dict:
    """Run a debate and return agent_positions dict."""
    if system == "single_llm":
        return {}  # not testing single_llm here

    from benchmark.variants import build_variant_graph
    graph = build_variant_graph(system)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    try:
        result = graph.invoke({"topic": question, "max_rounds": 3}, config=config)
        report = result.get("final_report")
        if report and report.reasoning_trace:
            last_round = report.reasoning_trace[-1]
            return {arg.agent_role: arg.position for arg in last_round.arguments}
    except Exception as e:
        print(f"\n  [error] {str(e)[:80]}")
    return {}


# ---------------------------------------------------------------------------
# Quality evaluator (reuses existing rubric)
# ---------------------------------------------------------------------------

def evaluate_quality(question: str, positions: dict, llm) -> dict | None:
    """Run Qwen quality judge and return dimension scores."""
    from benchmark.quality_evaluator import evaluate_analysis, extract_analysis_text
    # Build a fake result dict compatible with extract_analysis_text
    fake_result = {"question": question, "agent_positions": positions}
    analysis_text = extract_analysis_text(fake_result)
    score = evaluate_analysis(question, analysis_text, llm)
    if score is None:
        return None
    return {
        "perspective_diversity": score.perspective_diversity.score,
        "analytical_depth":      score.analytical_depth.score,
        "claim_specificity":     score.claim_specificity.score,
        "honest_uncertainty":    score.honest_uncertainty.score,
        "practical_utility":     score.practical_utility.score,
        "total":                 round(score.total(), 3),
    }


# ---------------------------------------------------------------------------
# Resume-safe save
# ---------------------------------------------------------------------------

OUT_PATH = RESULTS_DIR / "type_comparison.json"

def load_state() -> dict:
    if OUT_PATH.exists():
        with open(OUT_PATH) as f:
            return json.load(f)
    return {"debates": {}, "scores": []}

def save_state(state: dict) -> None:
    with open(OUT_PATH, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=int, default=10)
    parser.add_argument("--skip-debates", action="store_true",
                        help="Skip running debates, only re-evaluate existing positions")
    args = parser.parse_args()

    os.environ["LLM_BACKEND"] = "groq"
    from debate.llm import _make_llm as _groq_llm

    os.environ["LLM_BACKEND"] = "qwen"
    from debate.llm import _make_llm as _qwen_llm
    judge_llm = _qwen_llm()

    state = load_state()
    done_debates = set(state["debates"].keys())
    done_scores  = {(s["question_id"], s["system"]) for s in state["scores"]}

    print("=== 3-Type PROHIBITION Comparison ===")
    print(f"Systems: {SYSTEMS}")
    print(f"Question types: {list(QUESTION_FILES.keys())}\n")

    for qtype, qfile in QUESTION_FILES.items():
        with open(qfile) as f:
            all_qs = json.load(f)

        # Filter to questions labeled (or expected) as this type
        if qtype == "context_dependent":
            qs = [q for q in all_qs if q.get("question_type") == "context_dependent"][:QUESTION_LIMITS[qtype]]
        else:
            qs = all_qs[:QUESTION_LIMITS[qtype]]

        print(f"\n{'='*60}")
        print(f"Type: {qtype} ({len(qs)} questions)")

        for q in qs:
            qid   = q["id"]
            qtext = q["question"]
            print(f"\n  Q{qid}: {qtext[:65]}...")

            for system in SYSTEMS:
                debate_key = f"{qid}_{system}"

                # --- Step 1: Run debate ---
                if debate_key in done_debates:
                    positions = state["debates"][debate_key]
                    print(f"    [{system}] debate cached")
                elif args.skip_debates:
                    positions = {}
                    print(f"    [{system}] skip-debates mode")
                else:
                    print(f"    [{system}] running...", end=" ", flush=True)
                    os.environ["LLM_BACKEND"] = "groq"
                    positions = run_debate(qtext, system)
                    state["debates"][debate_key] = positions
                    save_state(state)
                    print("done")
                    if args.delay:
                        time.sleep(args.delay)

                if not positions:
                    print(f"    [{system}] no positions — skip evaluation")
                    continue

                # --- Step 2: Quality evaluation ---
                if (qid, system) in done_scores:
                    cached = next(s for s in state["scores"]
                                  if s["question_id"]==qid and s["system"]==system)
                    print(f"    [{system}] eval cached: total={cached['scores']['total']:.2f}")
                    continue

                print(f"    [{system}] evaluating...", end=" ", flush=True)
                os.environ["LLM_BACKEND"] = "qwen"
                scores = evaluate_quality(qtext, positions, judge_llm)
                if scores is None:
                    print("FAILED")
                    continue

                entry = {
                    "question_id":   qid,
                    "question_type": qtype,
                    "system":        system,
                    "scores":        scores,
                    "focus_dims":    {d: scores[d] for d in TYPE_FOCUS_DIMS[qtype]},
                }
                state["scores"].append(entry)
                done_scores.add((qid, system))
                save_state(state)

                focus_str = "  ".join(f"{d}={scores[d]}" for d in TYPE_FOCUS_DIMS[qtype])
                print(f"total={scores['total']:.2f}  [{focus_str}]")
                if args.delay:
                    time.sleep(args.delay)

    print_summary(state["scores"])


def print_summary(scores: list) -> None:
    import statistics
    from collections import defaultdict

    print(f"\n{'='*65}")
    print("=== Per-type Quality Summary (focus dimensions) ===\n")

    by_type_system: dict = defaultdict(lambda: defaultdict(list))
    for s in scores:
        for dim in TYPE_FOCUS_DIMS[s["question_type"]]:
            by_type_system[s["question_type"]][s["system"]].append(s["scores"][dim])

    for qtype in ["binary", "values_based", "context_dependent"]:
        print(f"[{qtype}]  focus: {TYPE_FOCUS_DIMS[qtype]}")
        for system in SYSTEMS:
            vals = by_type_system[qtype].get(system, [])
            if not vals:
                continue
            avg = statistics.mean(vals)
            print(f"  {system:<28}: {avg:.2f}  (n={len(vals)//len(TYPE_FOCUS_DIMS[qtype])})")
        print()

    print("Hypothesis:")
    print("  context_dependent: adaptive > full_system  ← should see this")
    print("  values_based:      adaptive ≈ full_system  ← both use full PROHIBITION")
    print("  binary:            adaptive ≈ full_system  ← moderate vs full, unclear")


if __name__ == "__main__":
    main()
