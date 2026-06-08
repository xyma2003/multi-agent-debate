# benchmark/baseline.py
"""
Single-LLM baseline runner.

Asks a single Claude instance to analyze a question from three perspectives
(optimist, pessimist, devil's advocate) in one call — the standard
"multi-perspective analysis" prompt pattern that most users reach for.

This is deliberately NOT given PROHIBITION constraints. The point is to
capture the natural sycophantic hedging behavior that the multi-agent
system is designed to overcome.

The structured output (SingleLLMOutput) mirrors the fields used for
evaluation so the same evaluator.py metrics apply.

Usage:
    from benchmark.baseline import run_single_llm
    result = run_single_llm(question_id=1, question="Should startups raise VC?")
"""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from debate.llm import _make_llm
from benchmark.evaluator import SingleLLMPerspective, SingleLLMResult


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class PerspectiveOutput(BaseModel):
    """One perspective in the single-LLM multi-perspective analysis."""

    role: str = Field(
        description="The analytical role: 'optimist', 'pessimist', or 'devil'"
    )
    position: str = Field(
        description="Core stance in ONE sentence — the single most important claim"
    )
    key_claims: list[str] = Field(
        min_length=3,
        description="3–7 specific, concrete claims supporting this position"
    )
    reasoning: str = Field(
        description="Full argument prose (2–4 sentences) supporting the position"
    )


class SingleLLMOutput(BaseModel):
    """Structured output from the single-LLM multi-perspective prompt."""

    optimist: PerspectiveOutput = Field(
        description="Analysis from the optimist perspective"
    )
    pessimist: PerspectiveOutput = Field(
        description="Analysis from the pessimist perspective"
    )
    devil: PerspectiveOutput = Field(
        description="Analysis from the devil's advocate perspective"
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert analyst. When given a question or topic, analyze it from
three distinct perspectives: optimist, pessimist, and devil's advocate.

For each perspective:
- State the core position clearly in one sentence
- Provide 3–7 specific, concrete claims that support that position
- Write 2–4 sentences of reasoning

Be thorough and give each perspective a fair, substantive treatment."""

_HUMAN_TEMPLATE = """\
Analyze the following question from three perspectives:

Question: {question}

Provide analysis from:
1. The optimist perspective — focus on opportunities, upsides, and reasons for optimism
2. The pessimist perspective — focus on risks, downsides, and reasons for caution
3. The devil's advocate perspective — challenge the most dominant or obvious view

Give each perspective a genuine, substantive argument."""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_single_llm(
    question_id: int,
    question: str,
    max_retries: int = 2,
) -> SingleLLMResult:
    """Run a single-LLM multi-perspective analysis on one question.

    Args:
        question_id: ID from questions.json.
        question:    The question text.
        max_retries: Number of extra attempts on parse failure.

    Returns:
        SingleLLMResult with three perspectives.
        On total failure, returns a result with placeholder perspectives
        (flagged by role prefix "ERROR_").
    """
    llm = _make_llm()
    structured_llm = llm.with_structured_output(SingleLLMOutput, include_raw=True)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_HUMAN_TEMPLATE.format(question=question)),
    ]

    for attempt in range(max_retries + 1):
        result = structured_llm.invoke(messages)
        parsed: SingleLLMOutput | None = result.get("parsed")
        if parsed is not None:
            return _to_result(parsed, question_id, question)
        print(
            f"[single_llm q={question_id}] Parse failed "
            f"(attempt {attempt + 1}/{max_retries + 1}): "
            f"{result.get('parsing_error')}"
        )

    # Fallback: return error sentinels so downstream code doesn't crash
    print(f"[single_llm q={question_id}] All retries exhausted — returning sentinel")
    return _error_result(question_id, question)


def _to_result(output: SingleLLMOutput, question_id: int, question: str) -> SingleLLMResult:
    """Convert SingleLLMOutput to SingleLLMResult."""
    perspectives = []
    for role, perspective in [
        ("optimist", output.optimist),
        ("pessimist", output.pessimist),
        ("devil", output.devil),
    ]:
        full_text = f"{perspective.position}\n\n{perspective.reasoning}"
        perspectives.append(SingleLLMPerspective(
            role=role,
            position=perspective.position,
            key_claims=perspective.key_claims,
            full_text=full_text,
        ))
    return SingleLLMResult(
        question_id=question_id,
        question=question,
        perspectives=perspectives,
    )


def _error_result(question_id: int, question: str) -> SingleLLMResult:
    """Return a sentinel SingleLLMResult on total failure."""
    sentinel_perspective = SingleLLMPerspective(
        role="ERROR",
        position="[Analysis unavailable due to parse failure]",
        key_claims=["parse_error", "sentinel", "no_data"],
        full_text="[Analysis unavailable due to repeated parse failures]",
    )
    return SingleLLMResult(
        question_id=question_id,
        question=question,
        perspectives=[sentinel_perspective] * 3,
    )
