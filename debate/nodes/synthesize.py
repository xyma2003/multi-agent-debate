# debate/nodes/synthesize.py
"""
Synthesizer node: produces DebateReport from completed debate state.

Replaces the Phase 2 stub. Node name "synthesize_stub" stays unchanged in graph.py.

Implementation split:
  - LLM call: produces SynthesizerOutput (consensus_points, disputed_points, verdict)
  - Python code: computes confidence_score via formula (SYNTH-03)
  - Python code: determines convergence_status from state values (not LLM)
  - Python code: assembles DebateReport from all of the above + state fields

SYNTH-03 invariant: confidence_score NEVER appears in the LLM prompt or SynthesizerOutput.
It is computed by _compute_confidence_score() after the LLM call.
"""
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from debate.divergence import DIVERGE_THRESHOLD
from debate.llm import _make_llm
from debate.state import Concession, DebateReport, DebateState, DisputedPoint, RoundRecord


# ---------------------------------------------------------------------------
# LLM output schema (confidence_score intentionally absent — SYNTH-03)
# ---------------------------------------------------------------------------

class SynthesizerOutput(BaseModel):
    """Structured output from the Synthesizer LLM call.

    confidence_score is intentionally absent. It is computed in Python
    by _compute_confidence_score() after this call. Including it here
    would allow the LLM to invent a number, violating SYNTH-03.
    """

    consensus_points: list[str] = Field(
        description="Claims all 3 agents genuinely agreed on. Empty list if no consensus."
    )
    disputed_points: list[DisputedPoint] = Field(
        description=(
            "Key points of disagreement. Each entry has 'topic' (the contested claim) "
            "and 'agent_positions' (dict mapping agent_role to their stance on it)."
        )
    )
    verdict: str = Field(
        description=(
            "2-4 sentence synthesis of the debate outcome. "
            "If agents did not reach consensus, this MUST begin with exactly: "
            "'Agents did not reach consensus on this topic.'"
        )
    )


# ---------------------------------------------------------------------------
# Confidence formula constants
# ---------------------------------------------------------------------------

# Round adjustment penalizes debates that needed more rounds to resolve.
# Key: round_num (count of completed rounds, 1-indexed). Default 0.8 for round_num >= 4.
_ROUND_ADJUSTMENTS: dict[int, float] = {1: 1.0, 2: 0.9, 3: 0.8}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYNTHESIZER_SYSTEM = """\
You are the Debate Synthesizer. Your role is to produce an honest, structured
summary of a multi-agent debate.

Rules:
1. Extract only claims explicitly made by agents — do not add new claims or inferences.
2. A consensus point requires evidence that agents actually agreed (shared claim or explicit concession).
3. A disputed point must include each agent's actual position, not a paraphrase.
4. Your verdict summarizes the overall outcome in 2-4 sentences.
5. Do NOT invent confidence scores, numerical assessments, or probabilities.
6. Do NOT fabricate agreement where agents disagreed."""


# ---------------------------------------------------------------------------
# Helper: confidence formula (pure Python, SYNTH-03)
# ---------------------------------------------------------------------------

def _compute_confidence_score(round_history: list[RoundRecord], round_num: int) -> float:
    """Compute confidence score from debate outcome.

    Formula: (1 - max_divergence_score) * round_adjustment

    max_divergence_score: the highest per-round divergence_score across ALL rounds
      (not just the last round's state["divergence_score"], which is last-write-wins).
    round_adjustment: 1.0 for 1 round, 0.9 for 2 rounds, 0.8 for 3+ rounds.

    Edge cases:
      - round_history empty or all divergence_score == 0.0:
        max_divergence = 0.0, result is round_adjustment (e.g. 1.0 for 1 round).
      - round_num 0 (should not occur in normal flow): treated as round 1 (adj=1.0).
    """
    max_divergence = max(
        (r.divergence_score for r in round_history), default=0.0
    )
    round_adjustment = _ROUND_ADJUSTMENTS.get(round_num, 0.8)
    return round((1.0 - max_divergence) * round_adjustment, 4)


# ---------------------------------------------------------------------------
# Helper: convergence status (pure Python)
# ---------------------------------------------------------------------------

def _determine_convergence_status(
    divergence_score: float,
    round_num: int,
    max_rounds: int,
) -> str:
    """Classify how the debate loop terminated.

    Uses DIVERGE_THRESHOLD (0.75) imported from debate.divergence — same value
    used by route_divergence. The stub's hardcoded 0.25 was a bug.

    "partial" is a defensive fallback: it should not occur in normal graph flow
    (route_divergence only routes to synthesize_stub on max_rounds or convergence),
    but is kept for robustness.
    """
    # Mirror route_divergence guard order: max_rounds check comes first.
    if round_num >= max_rounds:
        return "max_rounds"
    elif divergence_score < DIVERGE_THRESHOLD:
        return "converged"
    return "partial"  # defensive fallback; see module docstring


# ---------------------------------------------------------------------------
# Helper: compact synthesis context for LLM
# ---------------------------------------------------------------------------

def _build_synthesis_context(
    topic: str,
    round_history: list[RoundRecord],
    convergence_status: str,
) -> str:
    """Build compact human message for the synthesizer LLM call.

    Includes: per-round per-agent position + key_claims (first 3) + concessions.
    Excludes: raw reasoning prose (too verbose; not needed for synthesis).
    Skips: sentinel arguments (is_sentinel=True carry no real content).

    Keeping this under ~1500 tokens for a 3-round debate avoids context bloat.
    """
    lines = [f"Topic: {topic}\n"]

    for record in round_history:
        lines.append(
            f"--- Round {record.round_num + 1} "
            f"(divergence: {record.divergence_score:.2f}) ---"
        )
        for arg in record.arguments:
            if arg.is_sentinel:
                lines.append(f"[{arg.agent_role.upper()}] <no data — sentinel>")
                continue
            claims_text = "\n".join(
                f"  - {c}" for c in arg.key_claims[:3]
            )
            lines.append(
                f"[{arg.agent_role.upper()}] confidence={arg.confidence:.0%}\n"
                f"Position: {arg.position}\n"
                f"Key claims:\n{claims_text}"
            )
            if arg.concessions:
                for c in arg.concessions:
                    lines.append(
                        f"  CONCESSION: '{c.conceded_point}' "
                        f"(triggered by {c.triggered_by_agent}: {c.triggered_by_claim})"
                    )
        lines.append("")

    # Non-convergence honest-uncertainty path (SYNTH-04)
    if convergence_status == "max_rounds":
        lines.append(
            "\n\nIMPORTANT: The agents did NOT reach consensus — the debate ended "
            "because the maximum number of rounds was reached, not because divergence "
            "dropped below the threshold. Your verdict MUST begin with exactly: "
            "'Agents did not reach consensus on this topic.'"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: LLM invocation with retry
# ---------------------------------------------------------------------------

def _invoke_synthesizer(context: str) -> SynthesizerOutput:
    """Call the LLM and parse into SynthesizerOutput, with up to 2 retries.

    Uses include_raw=True so parse failures surface as result["parsed"] is None
    rather than raising ValidationError. Returns a minimal fallback on exhaustion.
    """
    llm = _make_llm()
    structured_llm = llm.with_structured_output(SynthesizerOutput, include_raw=True)
    messages = [
        SystemMessage(content=_SYNTHESIZER_SYSTEM),
        HumanMessage(content=context),
    ]
    for attempt in range(3):
        result = structured_llm.invoke(messages)
        if result.get("parsed") is not None:
            return result["parsed"]
        print(
            f"[synthesizer] Pydantic parse failed "
            f"(attempt {attempt + 1}/3): {result.get('parsing_error')}"
        )
    # Fallback: return minimal valid SynthesizerOutput so graph does not crash
    return SynthesizerOutput(
        consensus_points=[],
        disputed_points=[],
        verdict="[Synthesis unavailable due to repeated parse failures]",
    )


# ---------------------------------------------------------------------------
# Graph node (name must stay "synthesize_stub" — graph.py registers it by name)
# ---------------------------------------------------------------------------

def synthesize_stub(state: DebateState) -> dict:
    """Phase 3 synthesizer. Replaces the Phase 2 stub.

    Node name stays "synthesize_stub" so graph.py needs no changes.

    Steps:
      1. Determine convergence_status in Python (not LLM)
      2. Build compact synthesis context from round_history
      3. Call LLM → SynthesizerOutput (consensus, disputed, verdict only)
      4. Compute confidence_score in Python (SYNTH-03 — never LLM-invented)
      5. Flatten concession_log from all rounds
      6. Assemble and return DebateReport
    """
    topic = state.get("topic", "")
    debate_id = state.get("debate_id", "")
    round_history: list[RoundRecord] = state.get("round_history", [])
    round_num: int = state.get("round_num", 0)
    max_rounds: int = state.get("max_rounds", 3)
    divergence_score: float = state.get("divergence_score", 0.0)

    # Step 1: convergence status
    convergence_status = _determine_convergence_status(
        divergence_score, round_num, max_rounds
    )

    # Step 2: compact context
    context = _build_synthesis_context(topic, round_history, convergence_status)

    # Step 3: LLM call (consensus / disputed / verdict — no confidence_score)
    synthesis = _invoke_synthesizer(context)

    # Step 4: confidence formula (SYNTH-03)
    confidence_score = _compute_confidence_score(round_history, round_num)

    # Step 5: flatten concessions across all rounds
    concession_log: list[Concession] = [
        c
        for record in round_history
        for arg in record.arguments
        for c in arg.concessions
    ]

    # Step 6: assemble DebateReport
    report = DebateReport(
        debate_id=debate_id,
        topic=topic,
        consensus_points=synthesis.consensus_points,
        disputed_points=synthesis.disputed_points,
        verdict=synthesis.verdict,
        confidence_score=confidence_score,
        convergence_status=convergence_status,
        reasoning_trace=round_history,
        concession_log=concession_log,
        created_at=datetime.now(timezone.utc),
    )

    return {"final_report": report, "status": convergence_status}
