# benchmark/prohibition_analysis.py
"""
Path A: Is PROHIBITION creating false certainty or just clear stances?

Qwen3-32b (independent judge) gave uncertainty=1 to full_system agents.
Two possible explanations:
  (A) Bad: Agent falsely claims certainty — "There are ZERO risks" ignoring real risks
  (B) Good: Agent takes a role-appropriate committed stance — Pessimist's job IS to
      assert risk, that's not miscalibration, that's the design

This script tests each agent position and asks Qwen to classify it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

RESULTS_DIR = Path(__file__).parent.parent / "results"
QUESTIONS_PATH = Path(__file__).parent / "questions.json"


class PositionAnalysis(BaseModel):
    """Judge's analysis of one agent position."""
    false_certainty_score: int = Field(
        ge=1, le=5,
        description=(
            "How much is this agent FALSELY claiming certainty about genuinely uncertain things? "
            "5 = clearly overclaiming (e.g. 'there are absolutely NO risks'), "
            "1 = appropriately confident given the agent's stated analytical role"
        )
    )
    commitment_score: int = Field(
        ge=1, le=5,
        description=(
            "How clearly committed is this position? "
            "5 = clear unambiguous stance, "
            "1 = vague, hedged, or non-committal"
        )
    )
    verdict: str = Field(
        description=(
            "One of: 'false_certainty' (agent is overclaiming beyond what is defensible), "
            "'role_appropriate_commitment' (agent is taking a clear stance consistent with its "
            "analytical role — this is good design, not miscalibration), "
            "'appropriate_hedge' (single-LLM style balanced response)"
        )
    )
    reasoning: str = Field(description="One sentence explaining the verdict")


_JUDGE_SYSTEM = """\
You are evaluating whether an AI analyst's statement represents:
  (A) FALSE CERTAINTY — claiming something is definitely true/false when it's genuinely uncertain
  (B) ROLE-APPROPRIATE COMMITMENT — taking a clear position because that is the analyst's assigned role
      (e.g. a Risk Analyst who says 'this investment WILL fail' is doing their job — they're
       applying bear-case analysis, not claiming omniscience)
  (C) APPROPRIATE HEDGE — acknowledging multiple sides without committing (single-LLM style)

The key distinction between (A) and (B):
  FALSE CERTAINTY ignores counterevidence that any reasonable analyst would acknowledge.
  ROLE-APPROPRIATE COMMITMENT takes a stance within a defined analytical framework.

A Risk Analyst saying "VC destroys founder control" is (B) — that's their job.
A Risk Analyst saying "VC NEVER creates value for anyone" is (A) — that ignores obvious evidence."""


def analyze_position(question: str, position: str, llm) -> PositionAnalysis | None:
    import time
    from langchain_core.messages import HumanMessage, SystemMessage
    structured = llm.with_structured_output(PositionAnalysis, include_raw=True)
    prompt = (
        f"Question being analyzed: \"{question}\"\n\n"
        f"Analyst statement: \"{position}\"\n\n"
        "Is this statement false certainty, role-appropriate commitment, or appropriate hedge?"
    )
    for attempt in range(4):
        try:
            result = structured.invoke([
                SystemMessage(content=_JUDGE_SYSTEM),
                HumanMessage(content=prompt),
            ])
            if result.get("parsed"):
                return result["parsed"]
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg or "limit" in msg:
                wait = 60 * (attempt + 1)
                print(f"\n  [rate limit] waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"\n  [error] {str(e)[:80]}")
                return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=7)
    parser.add_argument("--delay", type=int, default=2)
    args = parser.parse_args()

    with open(QUESTIONS_PATH) as f:
        q_map = {q["id"]: q["question"] for q in json.load(f)}

    os.environ["LLM_BACKEND"] = "qwen"
    from debate.llm import _make_llm
    llm = _make_llm()

    out_path = RESULTS_DIR / "prohibition_analysis.json"

    # Resume support: load existing results
    if out_path.exists():
        with open(out_path) as f:
            saved = json.load(f)
        results = saved.get("results", [])
        counts = saved.get("counts", {})
        print(f"Resuming — {len(results)} positions already evaluated.")
    else:
        results = []
        counts = {}

    done_keys = {(r["system"], r["question_id"], r["role"]) for r in results}

    def save_progress():
        # Recompute counts from results
        c = {"full_system": {}, "single_llm": {}}
        for r in results:
            s, v = r["system"], r["verdict"]
            c[s][v] = c[s].get(v, 0) + 1
        with open(out_path, "w") as f:
            json.dump({"results": results, "counts": c}, f, indent=2, ensure_ascii=False)

    for system in ["full_system", "single_llm"]:
        path = RESULTS_DIR / f"{system}.json"
        with open(path) as f:
            data = json.load(f)["results"]

        rows = [r for r in data if r["question_id"] <= args.limit]
        print(f"\n{'='*60}")
        print(f"System: {system} ({len(rows)} questions)")

        for row in rows:
            qid = row["question_id"]
            question = q_map.get(qid, "?")
            positions = row.get("agent_positions", {})

            for role, position in positions.items():
                key = (system, qid, role)
                if key in done_keys:
                    print(f"  q{qid} [{role}] already done — skip")
                    continue

                print(f"  q{qid} [{role}]...", end=" ", flush=True)
                analysis = analyze_position(question, position, llm)
                if analysis is None:
                    print("FAILED")
                    continue

                verdict = analysis.verdict
                results.append({
                    "system": system, "question_id": qid, "role": role,
                    "position": position[:80],
                    "false_certainty_score": analysis.false_certainty_score,
                    "commitment_score": analysis.commitment_score,
                    "verdict": verdict,
                    "reasoning": analysis.reasoning,
                })
                done_keys.add(key)
                save_progress()  # incremental save after each position

                print(f"{verdict[:25]}  (false={analysis.false_certainty_score} commit={analysis.commitment_score})")
                if args.delay:
                    time.sleep(args.delay)

    save_progress()

    # Summary
    with open(out_path) as f:
        final = json.load(f)
    counts = final["counts"]

    print(f"\n{'='*60}")
    print("=== PROHIBITION Analysis Summary ===\n")
    for system in ["full_system", "single_llm"]:
        c = counts.get(system, {})
        t = sum(c.values())
        if t == 0: continue
        fc = c.get("false_certainty", 0)
        rc = c.get("role_appropriate_commitment", 0)
        ah = c.get("appropriate_hedge", 0)
        print(f"{system} (n={t} positions):")
        print(f"  false_certainty:           {fc:2d} ({fc/t*100:.0f}%)")
        print(f"  role_appropriate_commit:   {rc:2d} ({rc/t*100:.0f}%)")
        print(f"  appropriate_hedge:         {ah:2d} ({ah/t*100:.0f}%)")
        print()

    print("Key question: is full_system's false_certainty% >> single_llm's?")
    print("If YES → PROHIBITION creates real miscalibration")
    print("If NO  → Qwen's uncertainty=1 was judging role-appropriate commitment as miscalibration")


if __name__ == "__main__":
    main()
