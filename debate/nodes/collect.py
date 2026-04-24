# debate/nodes/collect.py
"""
Collect node: fan-in after all three agent nodes complete.

Moves current_round_arguments into round_history as a RoundRecord,
then resets the accumulator to [] so Phase 2's rebuttal rounds start clean.
"""
from debate.state import DebateState, RoundRecord


def collect_round1(state: DebateState) -> dict:
    """Aggregate all three agent arguments from the current round into history."""
    current_args = state.get("current_round_arguments", [])
    round_num = state.get("round_num", 0)

    new_record = RoundRecord(
        round_num=round_num,
        arguments=current_args,
    )
    return {
        "round_history": state.get("round_history", []) + [new_record],
        "current_round_arguments": [],  # reset accumulator for next round
        "round_num": round_num + 1,
        "status": "running",
    }
