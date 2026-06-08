# debate/nodes/dispatch.py
"""
Dispatch node: returns list[Send] to fan out the three agent nodes in parallel.

CRITICAL: Each Send payload contains ONLY the fields the agent needs.
Do NOT pass the full DebateState — that would expose round_history from
prior rounds, violating Round 1 isolation (PITFALL 4 in RESEARCH.md).
"""
from langgraph.types import Send

from debate.divergence import (
    ABSOLUTE_MAX_ROUNDS,
    DIVERGE_THRESHOLD,
    PLATEAU_DELTA,
    PLATEAU_MIN_ROUNDS,
)
from debate.state import DebateState


def dispatch_round1(state: DebateState) -> list[Send]:
    """Fan-out: dispatch 3 agents in parallel with no cross-visibility."""
    topic = state.get("topic", "")
    round_num = state.get("round_num", 0)
    return [
        Send(
            "optimist_node",
            {"topic": topic, "agent_role": "optimist", "prior_arguments": [], "round_num": round_num},
        ),
        Send(
            "pessimist_node",
            {"topic": topic, "agent_role": "pessimist", "prior_arguments": [], "round_num": round_num},
        ),
        Send(
            "devil_node",
            {"topic": topic, "agent_role": "devil", "prior_arguments": [], "round_num": round_num},
        ),
    ]


def _build_compact_summaries(round_history: list) -> list[dict]:
    """Build compact per-agent summaries across ALL completed rounds.

    Each entry is ~80 tokens (position + 3 claims + confidence). For a 3-round
    debate with 3 agents, the total context is ~9 entries × 80 tokens = ~720 tokens —
    well within budget and necessary so agents can track how positions evolved and
    rebut original claims that opponents may have dodged in intermediate rounds.

    Returns a flat list of dicts ordered by round, one per agent per round.
    """
    if not round_history:
        return []
    summaries = []
    for record in round_history:
        for arg in record.arguments:
            summaries.append({
                "agent_role": arg.agent_role,
                "round_num": arg.round_num,
                "position": arg.position,
                "key_claims": arg.key_claims[:3],   # top 3 only — ~80 tokens per agent
                "confidence": arg.confidence,
            })
    return summaries


def route_divergence(state: DebateState):
    """Routing function for add_conditional_edges('divergence_check_node', ...).

    Returns list[Send] to fan out rebuttal agents OR 'synthesize_stub' to terminate.

    Four-guard adaptive convergence (evaluated in order):
      Guard 1 — absolute safety cap (ABSOLUTE_MAX_ROUNDS). Prevents infinite loops
                regardless of all other signals. max_rounds from invoke() is ignored
                here; use it as a per-invocation override via graph.invoke.
      Guard 2 — genuine convergence: score dropped below DIVERGE_THRESHOLD.
      Guard 3 — score plateau: score changed less than PLATEAU_DELTA for the last
                two rounds. Agents are stuck — more rounds won't help.
      Guard 4 — no concessions: no agent conceded anything last round. Agents are
                just repeating themselves with different words.

    DO NOT register this as a node. Pass directly to add_conditional_edges.
    (Pitfall: registering returns list[Send] as state update → InvalidUpdateError)
    """
    round_num = state.get("round_num", 0)
    divergence_score = state.get("divergence_score", 0.0)
    topic = state.get("topic", "")
    round_history = state.get("round_history", [])

    # Guard 1: absolute safety cap — always check first
    if round_num >= ABSOLUTE_MAX_ROUNDS:
        return "synthesize_stub"

    # Guard 2: genuine convergence — score dropped below threshold
    if divergence_score < DIVERGE_THRESHOLD:
        return "synthesize_stub"

    # Guard 3: score plateau — no meaningful progress in last two rounds
    if len(round_history) >= PLATEAU_MIN_ROUNDS:
        prev_score = round_history[-2].divergence_score
        curr_score = round_history[-1].divergence_score
        if abs(prev_score - curr_score) < PLATEAU_DELTA:
            return "synthesize_stub"

    # Guard 4: no concessions last round — agents are repeating themselves
    # (only applies after ≥1 rebuttal round so Round 1 is never penalised)
    if len(round_history) >= 2:
        last_round = round_history[-1]
        total_concessions = sum(len(arg.concessions) for arg in last_round.arguments)
        if total_concessions == 0:
            return "synthesize_stub"

    # Still making progress → fan out rebuttal agents
    compact_summaries = _build_compact_summaries(round_history)
    return [
        Send(
            "optimist_node",
            {
                "topic": topic,
                "agent_role": "optimist",
                "prior_arguments": compact_summaries,
                "round_num": round_num,
            },
        ),
        Send(
            "pessimist_node",
            {
                "topic": topic,
                "agent_role": "pessimist",
                "prior_arguments": compact_summaries,
                "round_num": round_num,
            },
        ),
        Send(
            "devil_node",
            {
                "topic": topic,
                "agent_role": "devil",
                "prior_arguments": compact_summaries,
                "round_num": round_num,
            },
        ),
    ]
