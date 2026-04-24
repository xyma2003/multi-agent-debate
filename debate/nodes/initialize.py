# debate/nodes/initialize.py
"""
Initialize node: stamps metadata and sets all DebateState defaults.

Must run before any other node. Sets every field that downstream nodes read
via state.get("field", default) to an explicit value, preventing KeyError.
"""
import uuid

from debate.state import DebateState


def initialize_node(state: DebateState) -> dict:
    """Stamp debate_id and initialize all counters and accumulators."""
    return {
        "debate_id": str(uuid.uuid4()),
        "round_num": 0,
        "max_rounds": state.get("max_rounds", 3),
        "current_round_arguments": [],
        "round_history": [],
        "divergence_score": 0.0,
        "diverged_pairs": [],
        "final_report": None,
        "status": "running",
    }
