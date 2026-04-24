# debate/nodes/dispatch.py
"""
Dispatch node: returns list[Send] to fan out the three agent nodes in parallel.

CRITICAL: Each Send payload contains ONLY the fields the agent needs.
Do NOT pass the full DebateState — that would expose round_history from
prior rounds, violating Round 1 isolation (PITFALL 4 in RESEARCH.md).
"""
from langgraph.types import Send

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
