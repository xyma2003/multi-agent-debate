# debate/classify.py
"""
Question type classifier for adaptive PROHIBITION selection.

Classifies a question into one of three types:
  values_based       — fundamental value/ethical disagreement (no objective answer)
  binary             — one answer is likely better, based on evidence or analysis
  context_dependent  — correct answer depends heavily on specific circumstances

Used by the adaptive_prohibition variant to select the right prompt set.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class QuestionClassification(BaseModel):
    question_type: str = Field(
        description=(
            "One of: 'values_based', 'binary', 'context_dependent'. "
            "'values_based' = ethical/moral disagreement where people with different "
            "values will legitimately reach different conclusions. "
            "'binary' = one approach is likely better based on evidence, but reasonable "
            "people could disagree. "
            "'context_dependent' = the correct answer clearly depends on specific "
            "circumstances — 'it depends' is a legitimately correct response."
        )
    )
    confidence: str = Field(
        description="'high', 'medium', or 'low'"
    )
    reasoning: str = Field(
        description="One sentence explaining the classification"
    )


_CLASSIFIER_SYSTEM = """\
Classify the question into exactly one of three types:

  values_based:       The disagreement is fundamentally about VALUES or ETHICS.
                      People with different values will reach different conclusions,
                      and neither is objectively wrong.
                      Examples: "Should AI development be halted?",
                               "Is capitalism compatible with climate action?"

  binary:             One answer is LIKELY BETTER based on evidence, analysis, or
                      commonly accepted principles — but reasonable people could disagree.
                      Examples: "Should startups prioritize growth or profitability early?",
                               "Is move-fast-break-things a sound philosophy?"

  context_dependent:  The correct answer GENUINELY DEPENDS on specific circumstances.
                      'It depends on your situation' is a legitimately correct response.
                      Examples: "Should startups hire specialists or generalists?",
                               "Should you build custom tools or use off-the-shelf?"

When in doubt between binary and context_dependent: if the question has a
recognizable 'default' right answer in most cases, it's binary. If the right
answer genuinely varies by business model, stage, team, or industry, it's
context_dependent."""


def classify_question(question: str, llm=None) -> str:
    """Classify a question and return its type string.

    Returns one of: 'values_based', 'binary', 'context_dependent'.
    Falls back to 'binary' if classification fails.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if llm is None:
        from debate.llm import _make_llm
        llm = _make_llm()

    structured = llm.with_structured_output(QuestionClassification, include_raw=True)
    for attempt in range(3):
        try:
            result = structured.invoke([
                SystemMessage(content=_CLASSIFIER_SYSTEM),
                HumanMessage(content=f"Classify this question:\n\n\"{question}\""),
            ])
            if result.get("parsed"):
                p = result["parsed"]
                qtype = p.question_type.lower().strip()
                if qtype in ("values_based", "binary", "context_dependent"):
                    return qtype
        except Exception:
            pass
    return "binary"  # safe fallback
