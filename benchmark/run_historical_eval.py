# benchmark/run_historical_eval.py
"""
Path B: Historical decisions with ground truth.

Runs three systems on historical business decisions (questions_historical.json)
WITHOUT revealing the outcome. Evaluates whether each system's analysis
correctly identified the KEY FACTOR that determined the actual result.

Evaluation is NOT "was the analysis good?" but "did it predict the right thing?"
The judge is given the KEY FACTOR (ground truth) and scores whether the system's
analysis identified it.

Scoring per system per question:
  2 = clearly identified the key factor as a critical consideration
  1 = partially or tangentially mentioned it
  0 = missed entirely

This is a factual evaluation (did you predict the right thing?) not a style
evaluation (was the writing good?), making it harder to game and more credible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

QUESTIONS_PATH = Path(__file__).parent / "questions_historical.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"
HISTORICAL_RESULTS = RESULTS_DIR / "historical"
HISTORICAL_RESULTS.mkdir(exist_ok=True)

SYSTEMS = ["single_llm", "full_system", "adaptive_prohibition"]


# ---------------------------------------------------------------------------
# Run a single system on a question
# ---------------------------------------------------------------------------

def run_single_llm(question_text: str, llm) -> dict:
    """Ask one LLM for multi-perspective analysis. Returns agent_positions dict."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    class MultiPerspective(BaseModel):
        optimist: str = Field(description="The strongest case FOR proceeding with this decision")
        pessimist: str = Field(description="The strongest case AGAINST proceeding")
        devil: str = Field(description="The critical assumption or overlooked factor that both sides might be missing")

    structured = llm.with_structured_output(MultiPerspective, include_raw=True)
    system = (
        "You are a strategic analyst. For the business decision described, "
        "provide three distinct analytical perspectives: optimist (strongest case for), "
        "pessimist (strongest case against), and devil's advocate (key assumption both sides miss). "
        "Each perspective should be ONE specific, concrete sentence. Do not hedge."
    )
    for attempt in range(3):
        try:
            result = structured.invoke([
                SystemMessage(content=system),
                HumanMessage(content=question_text),
            ])
            if result.get("parsed"):
                p = result["parsed"]
                return {"optimist": p.optimist, "pessimist": p.pessimist, "devil": p.devil}
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                print(f"\n  [rate limit] waiting 60s...", end=" ", flush=True)
                time.sleep(60)
    return {}


def run_multi_agent(question_text: str, system: str = "full_system") -> dict:
    """Run a multi-agent debate graph and return agent final positions."""
    from benchmark.variants import build_variant_graph
    import uuid

    graph = build_variant_graph(system)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"topic": question_text, "max_rounds": 3},
        config=config,
    )
    # Extract final positions from round_history
    report = result.get("final_report")
    if report:
        positions = {}
        if report.reasoning_trace:
            last_round = report.reasoning_trace[-1]
            for arg in last_round.arguments:
                positions[arg.agent_role] = arg.position
        return positions
    return {}


# ---------------------------------------------------------------------------
# Evaluate: did the analysis identify the key factor?
# ---------------------------------------------------------------------------

def evaluate_key_factor(
    question: str,
    analysis: dict,
    key_factor: str,
    actual_outcome: str,
    llm,
) -> dict:
    """Score (0-2) whether the analysis identified the key factor."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel, Field

    class KeyFactorScore(BaseModel):
        score: int = Field(
            ge=0, le=2,
            description=(
                "2 = the analysis CLEARLY identified this factor as a critical consideration. "
                "1 = the analysis PARTIALLY or TANGENTIALLY mentioned something related. "
                "0 = the analysis MISSED this factor entirely."
            )
        )
        evidence: str = Field(
            description="Quote the specific text from the analysis that earned this score, or 'Not found' if score=0"
        )
        reasoning: str = Field(description="One sentence explaining the score")

    # Build the analysis text from positions
    analysis_text = "\n".join(
        f"{role.upper()}: {pos}" for role, pos in analysis.items()
    )

    prompt = (
        f"Business decision: {question[:200]}\n\n"
        f"Analysis to evaluate:\n{analysis_text}\n\n"
        f"Key factor that actually determined the outcome:\n\"{key_factor}\"\n\n"
        f"(For reference, the actual outcome was: {actual_outcome[:150]})\n\n"
        "Did the analysis identify this key factor? Score 0/1/2 and quote evidence."
    )

    system = (
        "You are evaluating whether a business analysis correctly identified the factor "
        "that actually determined the outcome of a historical decision. "
        "You already know the outcome — your job is to check if the analysis mentioned "
        "the key factor BEFORE the outcome was revealed."
    )

    structured = llm.with_structured_output(KeyFactorScore, include_raw=True)
    for attempt in range(3):
        try:
            result = structured.invoke([
                SystemMessage(content=system),
                HumanMessage(content=prompt),
            ])
            if result.get("parsed"):
                p = result["parsed"]
                return {"score": p.score, "evidence": p.evidence, "reasoning": p.reasoning}
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                print(f"\n  [rate limit] waiting 60s...", end=" ", flush=True)
                time.sleep(60)
    return {"score": -1, "evidence": "FAILED", "reasoning": "Parse error"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Historical decisions ground-truth evaluation")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of historical questions to run (default: 5)")
    parser.add_argument("--delay", type=int, default=5,
                        help="Seconds between API calls (default: 5)")
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip running debates, only re-evaluate existing results")
    args = parser.parse_args()

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)[:args.limit]

    # Load existing results (resume support)
    results_path = HISTORICAL_RESULTS / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            saved = json.load(f)
    else:
        saved = {"debates": {}, "scores": []}

    # LLM setup
    os.environ["LLM_BACKEND"] = "groq"
    from debate.llm import _make_llm
    llm_debate = _make_llm()

    os.environ["LLM_BACKEND"] = "qwen"
    llm_judge = _make_llm()

    print(f"Running {len(questions)} historical questions on {len(SYSTEMS)} systems")
    print(f"Judge: Qwen3-32b | Debaters: Groq llama-3.3-70b\n")

    for q in questions:
        qid = q["id"]
        question_text = q["question"]
        key_factor = q["key_factor"]
        actual_outcome = q["actual_outcome"]

        print(f"\n{'='*65}")
        print(f"Q{qid} ({q['year']}): {question_text[:80]}...")

        for system in SYSTEMS:
            debate_key = f"{qid}_{system}"

            # Step 1: Run debate (or load cached)
            if args.skip_run and debate_key in saved["debates"]:
                positions = saved["debates"][debate_key]
                print(f"  [{system}] loaded cached debate")
            elif debate_key in saved["debates"]:
                positions = saved["debates"][debate_key]
                print(f"  [{system}] using cached debate")
            else:
                print(f"  [{system}] running debate...", end=" ", flush=True)
                if system == "single_llm":
                    os.environ["LLM_BACKEND"] = "groq"
                    positions = run_single_llm(question_text, _make_llm())
                else:
                    positions = run_multi_agent(question_text, system)
                saved["debates"][debate_key] = positions
                print("done")
                with open(results_path, "w") as f:
                    json.dump(saved, f, indent=2, ensure_ascii=False)
                if args.delay:
                    time.sleep(args.delay)

            # Step 2: Evaluate against key factor
            already_scored = any(
                s["question_id"] == qid and s["system"] == system
                for s in saved["scores"]
            )
            if already_scored:
                score_entry = next(
                    s for s in saved["scores"]
                    if s["question_id"] == qid and s["system"] == system
                )
                print(f"  [{system}] score={score_entry['score']}/2 (cached)")
            else:
                print(f"  [{system}] evaluating key factor...", end=" ", flush=True)
                os.environ["LLM_BACKEND"] = "qwen"
                eval_result = evaluate_key_factor(
                    question_text, positions, key_factor, actual_outcome, llm_judge
                )
                score_entry = {
                    "question_id": qid,
                    "year": q["year"],
                    "question_short": question_text[:60],
                    "system": system,
                    "score": eval_result["score"],
                    "evidence": eval_result["evidence"],
                    "reasoning": eval_result["reasoning"],
                    "key_factor": key_factor[:100],
                }
                saved["scores"].append(score_entry)
                with open(results_path, "w") as f:
                    json.dump(saved, f, indent=2, ensure_ascii=False)
                print(f"score={eval_result['score']}/2 — {eval_result['reasoning'][:60]}")
                if args.delay:
                    time.sleep(args.delay)

    # Print summary
    print_summary(saved["scores"])


def print_summary(scores: list) -> None:
    import statistics
    from collections import defaultdict

    if not scores:
        return

    by_system = defaultdict(list)
    for s in scores:
        if s["score"] >= 0:
            by_system[s["system"]].append(s["score"])

    print(f"\n{'='*65}")
    print("=== Historical Decisions: Key Factor Identification ===")
    print(f"{'System':<22} {'n':>3} {'Avg score':>10} {'Score dist (0/1/2)':>20}")
    print("-" * 60)
    for system in SYSTEMS:
        sc = by_system.get(system, [])
        if not sc:
            continue
        avg = statistics.mean(sc)
        dist = f"{sc.count(0)}/{sc.count(1)}/{sc.count(2)}"
        print(f"{system:<22} {len(sc):>3} {avg:>10.2f} {dist:>20}")

    print()
    print("Score: 0=missed, 1=partial, 2=clearly identified the key factor")
    print("Higher score = better prediction of what actually determined the outcome")


if __name__ == "__main__":
    main()
