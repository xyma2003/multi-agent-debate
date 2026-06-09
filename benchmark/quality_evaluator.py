# benchmark/quality_evaluator.py
"""
LLM-as-judge quality evaluator for debate analysis outputs.

Evaluates each system's output (single_llm / full_system / nli_detection)
on 5 dimensions, blind (no system label shown to judge).

Public API:
    evaluate_analysis(question, analysis_text, llm) -> QualityScore
    extract_analysis_text(report: dict) -> str
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5, description="Score from 1 (poor) to 5 (excellent)")
    reasoning: str = Field(description="One sentence explaining the score")


class QualityScore(BaseModel):
    """Structured quality evaluation from the LLM judge."""

    perspective_diversity: DimensionScore = Field(
        description="Are the viewpoints genuinely distinct, or restatements of each other?"
    )
    analytical_depth: DimensionScore = Field(
        description="Does the analysis identify non-obvious risks or opportunities?"
    )
    claim_specificity: DimensionScore = Field(
        description="Are claims concrete and traceable, or vague and generic?"
    )
    honest_uncertainty: DimensionScore = Field(
        description="Does the analysis accurately acknowledge what is genuinely unknown?"
    )
    practical_utility: DimensionScore = Field(
        description="Could a decision-maker act on this analysis?"
    )

    def total(self) -> float:
        dims = [
            self.perspective_diversity,
            self.analytical_depth,
            self.claim_specificity,
            self.honest_uncertainty,
            self.practical_utility,
        ]
        return sum(d.score for d in dims) / len(dims)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an expert analyst evaluating the quality of decision-support analyses.
You will be shown a question and an analysis produced by an AI system.
Your job is to score the analysis on 5 dimensions.

Scoring guide (1-5 for each dimension):
  1 = very poor   2 = below average   3 = adequate   4 = good   5 = excellent

Dimensions:
  perspective_diversity  — Are the viewpoints genuinely distinct, not just restated with
                           different words? Score 5 if 3+ truly independent angles are present.
  analytical_depth       — Does it identify non-obvious risks or opportunities that a casual
                           reader would likely miss? Score 5 for genuine insight, 1 for clichés.
  claim_specificity      — Are claims concrete ("A-round dilution averages 20-25%") or vague
                           ("there are risks")? Score 5 for specific, traceable claims.
  honest_uncertainty     — Does the analysis accurately flag what is genuinely unknown or
                           context-dependent, without over-hedging everything? Score 5 for
                           calibrated uncertainty.
  practical_utility      — Could a real decision-maker act on this analysis? Score 5 if it
                           clearly identifies what to do differently based on context.

Be strict. Do not inflate scores. A score of 3 means "adequate but not impressive"."""


def _build_judge_prompt(question: str, analysis_text: str) -> str:
    return f"""Question being analyzed:
\"{question}\"

Analysis to evaluate:
---
{analysis_text}
---

Score this analysis on all 5 dimensions. For each dimension, provide:
  - score: integer 1-5
  - reasoning: one sentence explaining the score"""


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_analysis_text(report: dict) -> str:
    """Extract evaluable text from a benchmark result dict.

    Uses agent_positions (final one-sentence stance from each analyst) for
    uniform comparison across all systems. Role labels are stripped so the
    judge cannot infer which system produced the output.

    Format shown to judge:
        Perspective 1: <stance>
        Perspective 2: <stance>
        Perspective 3: <stance>
    """
    agent_positions = report.get("agent_positions", {})
    if not agent_positions:
        return str(report)

    parts: list[str] = []
    for i, pos in enumerate(agent_positions.values(), start=1):
        parts.append(f"Perspective {i}: {pos}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate_analysis(
    question: str,
    analysis_text: str,
    llm,
    max_retries: int = 2,
) -> QualityScore | None:
    """Call the judge LLM and return a QualityScore.

    Returns None if all retries fail.
    """
    structured_llm = llm.with_structured_output(QualityScore, include_raw=True)
    messages = [
        SystemMessage(content=_JUDGE_SYSTEM),
        HumanMessage(content=_build_judge_prompt(question, analysis_text)),
    ]
    for attempt in range(max_retries + 1):
        result = structured_llm.invoke(messages)
        if result.get("parsed") is not None:
            return result["parsed"]
        print(f"  [judge] parse failed (attempt {attempt + 1}/{max_retries + 1}): "
              f"{result.get('parsing_error')}")
    return None
