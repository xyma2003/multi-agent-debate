# debate/nodes/dispatch.py
"""
Dispatch node: returns list[Send] to fan out the three agent nodes in parallel.

CRITICAL: Each Send payload contains ONLY the fields the agent needs.
Do NOT pass the full DebateState — that would expose round_history from
prior rounds, violating Round 1 isolation (PITFALL 4 in RESEARCH.md).
"""
from langgraph.types import Send

from debate.divergence import DIVERGE_THRESHOLD
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

    CRITICAL ORDERING (Pitfall 2 in RESEARCH.md):
      Guard 1 — max_rounds check MUST come first. If divergence_score is stuck
      (bug or genuinely persistent disagreement), max_rounds is the only escape.
      Checking score first and then max_rounds would allow runaway loops if the
      score check has a bug.

    DO NOT register this as a node. Pass directly to add_conditional_edges.
    (Pitfall 1: registering returns list[Send] as state update → InvalidUpdateError)
    """
    round_num = state.get("round_num", 0)
    max_rounds = state.get("max_rounds", 3)
    divergence_score = state.get("divergence_score", 0.0)
    topic = state.get("topic", "")
    round_history = state.get("round_history", [])

    # Guard 1: hard stop — max rounds reached regardless of divergence
    if round_num >= max_rounds:
        return "synthesize_stub"

    # Guard 2: converged — agents have reached semantic agreement
    if divergence_score < DIVERGE_THRESHOLD:
        return "synthesize_stub"

    # Diverged and within round budget: fan out rebuttal agents
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
