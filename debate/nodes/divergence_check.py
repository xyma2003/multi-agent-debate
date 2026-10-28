# debate/nodes/divergence_check.py
"""
Divergence check node: computes semantic divergence from the latest round's
arguments and stores the result in DebateState.

This node is a regular graph node (NOT a routing function). Routing logic
lives separately in route_divergence (debate/nodes/dispatch.py).

Reads:  round_history (last entry's arguments)
Writes: divergence_score (float), diverged_pairs (list[tuple[str,str]])
        Also back-fills the latest round's score and actual detector metadata.
"""
from debate.divergence import detect_divergence
from debate.state import DebateState


def divergence_check_node(state: DebateState) -> dict:
    """Compute divergence from the most recent round and write to state.

    An empty history indicates a graph wiring error, so fail loudly instead of
    silently labeling the debate as converged or divergent.
    """
    round_history = state.get("round_history", [])
    if not round_history:
        raise RuntimeError("divergence_check_node requires a completed round")

    latest_round = round_history[-1]
    result = detect_divergence(latest_round.arguments)

    # Back-fill per-round score onto the RoundRecord for Phase 3 SYNTH-03.
    # RoundRecord is a Pydantic model — use model_copy to avoid mutating in place.
    updated_record = latest_round.model_copy(update={
        "divergence_score": result.score,
        "detector_metadata": result.metadata,
    })
    updated_history = list(round_history[:-1]) + [updated_record]

    return {
        "divergence_score": result.score,
        "diverged_pairs": result.diverged_pairs,
        "round_history": updated_history,
    }
