# debate/state.py
"""
Single source of truth for all debate graph schemas.

DebateState: The LangGraph state TypedDict. Only `current_round_arguments` uses
             the `add` reducer — required for fan-in from parallel Send dispatch.
             All other fields are last-write-wins (default).

AgentArgument: Pydantic model for one agent's output in one round.
Concession:    Pydantic model for a point yielded to another agent's argument.
RoundRecord:   Pydantic model collecting all three arguments from one round.
"""
from __future__ import annotations

from datetime import datetime
from operator import add
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


class Concession(BaseModel):
    """A point that an agent concedes in response to another agent's argument."""

    conceded_point: str = Field(description="The specific point being conceded")
    triggered_by_agent: str = Field(
        description="Role of the agent whose argument triggered this concession: "
                    "'optimist' | 'pessimist' | 'devil'"
    )
    triggered_by_claim: str = Field(
        description="The specific claim from the other agent that forced this concession"
    )
    rationale: str = Field(
        description="One-sentence explanation of why this point is conceded"
    )


class AgentArgument(BaseModel):
    """Structured output from one agent node in one debate round."""

    agent_role: str = Field(
        description="Role identifier: 'optimist' | 'pessimist' | 'devil'"
    )
    round_num: int = Field(description="0-indexed debate round number")
    position: str = Field(
        description="Main thesis in ONE sentence — no hedging, no 'however'"
    )
    reasoning: str = Field(
        description="Full argument prose supporting the position"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence in the position, 0.0–1.0; not calibrated accuracy or an input to the report's agreement index",
    )
    key_claims: list[str] = Field(
        min_length=3,
        description="3–7 short extractable claims (used for embedding in Phase 2)",
    )
    concessions: list[Concession] = Field(
        default_factory=list,
        description="Points conceded this round; empty in Round 1",
    )
    is_sentinel: bool = Field(
        default=False,
        description="True if this is a fallback object injected after 3 parse failures",
    )


class DetectorMetadata(BaseModel):
    """Whitelisted settings captured where a round is evaluated, not at replay.

    Model IDs describe the loader's configured model, not a pinned weight revision.
    A zero evaluated_pair_count means no pair was evaluated by the model.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    mode: Literal["nli", "cosine"]
    model_id: str
    input_field: Literal["key_claims", "reasoning"] = "key_claims"
    divergence_threshold: float = Field(ge=0.0, le=1.0)
    cosine_fast_path_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    nli_contradiction_label_index: int | None = Field(default=None, ge=0)
    evaluated_pair_count: int = Field(default=0, ge=0)
    sentinels_excluded: bool = False


class RoundRecord(BaseModel):
    """Container for all three agent arguments from a single debate round."""

    round_num: int = Field(description="0-indexed round number")
    arguments: list[AgentArgument] = Field(
        description="Exactly 3 AgentArguments: optimist, pessimist, devil"
    )
    divergence_score: float = Field(
        default=0.0,
        description="Detector-specific divergence for this round (0.0=none detected, 1.0=maximum). NLI and cosine scores are not directly comparable; zero does not prove agreement. Written by divergence_check_node.",
    )
    detector_metadata: DetectorMetadata | None = Field(
        default=None,
        description="Detector settings captured during this round's evaluation. None means unknown; never infer missing metadata from current settings or synthetic scores.",
    )


class DisputedPoint(BaseModel):
    """One topic where agents took opposing positions."""

    topic: str = Field(description="The specific point of disagreement")
    agent_positions: dict[str, str] = Field(
        description="Map of agent_role to their position on this point: "
                    "{'optimist': '...', 'pessimist': '...', 'devil': '...'}"
    )


class DebateReport(BaseModel):
    """Final output of the debate system. Written to state['final_report'].

    confidence_score is the legacy storage name for the Debate Agreement Index,
    kept on a 0–1 scale for compatibility. Formula (SYNTH-03):
    (1 - max_divergence_score across all rounds) * round_adjustment.
    Python computes it, not the LLM. It is a heuristic, not calibrated accuracy.
    """

    debate_id: str
    topic: str
    consensus_points: list[str] = Field(
        description="Points all agents agreed on (can be empty list)"
    )
    disputed_points: list[DisputedPoint] = Field(
        description="Points where agents held opposing positions"
    )
    verdict: str = Field(
        description="Synthesizer's final 2-4 sentence assessment of the debate outcome"
    )
    synthesis_status: Literal["complete", "failed", "unknown"] = Field(
        default="unknown",
        description=(
            "Whether final synthesis was generated: 'complete' means a structured "
            "response was parsed, 'failed' means the existing attempts were exhausted, "
            "and 'unknown' means no status was recorded. Assigned by application code, "
            "not the model; independent of loop termination, factual correctness, "
            "and whether the report was saved."
        ),
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Debate Agreement Index on a 0–1 scale, retaining the legacy field name: (1 - highest divergence across all rounds) * round_adjustment. A heuristic, not correctness probability, evidence quality, or proof of consensus; excludes agent self-reported confidence.",
    )
    convergence_status: Literal["converged", "plateau", "stalled", "max_rounds", "partial"] = Field(
        description=(
            "How the debate loop terminated: "
            "'converged' = the active detector found no divergent agent pair; "
            "'plateau' = score stopped changing (< PLATEAU_DELTA per round); "
            "'stalled' = no concessions in last round; "
            "'max_rounds' = caller or absolute round cap hit; "
            "'partial' = defensive fallback"
        )
    )
    reasoning_trace: list[RoundRecord] = Field(
        description="All RoundRecords from round_history — full argument provenance"
    )
    concession_log: list[Concession] = Field(
        description="All Concession objects across all rounds, flattened"
    )
    created_at: datetime = Field(
        description="UTC timestamp when this report was assembled"
    )


class DebateState(TypedDict, total=False):
    """
    LangGraph StateGraph state for the multi-agent debate system.

    IMPORTANT — fan-in accumulator:
        `current_round_arguments` uses `Annotated[list[AgentArgument], add]`.
        This is the ONLY field with a reducer. All parallel Send nodes append
        their single-item list here; LangGraph merges them automatically.
        Every other field is last-write-wins.

    Field lifecycle:
        topic, max_rounds  — set by caller at graph.invoke time
        debate_id          — set by initialize_node
        round_num          — initialized to 0 by initialize_node; incremented by collect_round1
        current_round_arguments — accumulates during fan-out; reset to [] by collect_round1
        round_history      — extended by collect_round1 after each round
        divergence_score, diverged_pairs — written by Phase 2 divergence node
        final_report       — written by Phase 3 synthesizer
        status             — "running" | "converged" | "plateau" | "stalled" |
                             "max_rounds" | "partial"
    """

    # --- Input fields (set at invoke time) ---
    topic: str
    max_rounds: int

    # --- Metadata (set by initialize_node) ---
    debate_id: str

    # --- Round tracking ---
    round_num: int

    # --- Fan-in accumulator — ONLY this field uses a reducer ---
    current_round_arguments: Annotated[list[AgentArgument], add]

    # --- Full debate history ---
    round_history: list[RoundRecord]

    # --- Phase 2 fields (written by divergence detector; Phase 1 leaves at defaults) ---
    divergence_score: float
    diverged_pairs: list[tuple[str, str]]

    # --- Phase 3 fields ---
    final_report: Optional["DebateReport"]  # Written by Phase 3 synthesizer
    status: str
