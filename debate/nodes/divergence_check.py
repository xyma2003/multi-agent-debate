# debate/nodes/divergence_check.py
"""
Divergence check node: computes semantic divergence from the latest round's
arguments and stores the result in DebateState.

This node is a regular graph node (NOT a routing function). Routing logic
lives separately in route_divergence (debate/nodes/dispatch.py).

Reads:  round_history (last entry's arguments)
Writes: divergence_score (float), diverged_pairs (list[tuple[str,str]])
        Also back-fills round_history[-1].divergence_score for Phase 3 SYNTH-03.
"""
from debate.divergence import compute_divergence
from debate.state import DebateState


def divergence_check_node(state: DebateState) -> dict:
    """Compute divergence from the most recent round and write to state.

    If round_history is empty (should not happen in normal flow), returns
    divergence_score=1.0 (force rebuttal) to prevent silent failures.
    """
    round_history = state.get("round_history", [])
    if not round_history:
        return {"divergence_score": 1.0, "diverged_pairs": []}

    latest_round = round_history[-1]
    score, diverged_pairs = compute_divergence(latest_round.arguments)

    # Back-fill per-round score onto the RoundRecord for Phase 3 SYNTH-03.
    # RoundRecord is a Pydantic model — use model_copy to avoid mutating in place.
    updated_record = latest_round.model_copy(update={"divergence_score": score})
    updated_history = list(round_history[:-1]) + [updated_record]

    return {
        "divergence_score": score,
        "diverged_pairs": diverged_pairs,
        "round_history": updated_history,
    }
