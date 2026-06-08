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

from pydantic import BaseModel, Field
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
        description="Self-reported confidence in the position, 0.0–1.0",
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


class RoundRecord(BaseModel):
    """Container for all three agent arguments from a single debate round."""

    round_num: int = Field(description="0-indexed round number")
    arguments: list[AgentArgument] = Field(
        description="Exactly 3 AgentArguments: optimist, pessimist, devil"
    )
    divergence_score: float = Field(
        default=0.0,
        description="Divergence score computed after this round (0.0=converged, 1.0=max divergence). Written by divergence_check_node.",
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

    confidence_score is formula-derived (SYNTH-03): (1 - max_divergence_score) * round_adjustment.
    It is computed in Python code in synthesize.py, never produced by the LLM.
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
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Formula-derived score: (1 - max_divergence_score) * round_adjustment",
    )
    convergence_status: Literal["converged", "plateau", "stalled", "max_rounds", "partial"] = Field(
        description=(
            "How the debate loop terminated: "
            "'converged' = score dropped below threshold; "
            "'plateau' = score stopped changing (< PLATEAU_DELTA per round); "
            "'stalled' = no concessions in last round; "
            "'max_rounds' = absolute safety cap hit; "
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
        status             — "running" | "converged" | "max_rounds" | "complete"
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
