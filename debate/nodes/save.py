# debate/nodes/save.py
"""Save node: persists state['final_report'] to SQLite after synthesis.

Returns {} (no state mutation). If final_report is None (defensive), skips silently.
"""
from debate.state import DebateState
from debate.store import save_debate


def save_node(state: DebateState) -> dict:
    """Graph node: read final_report from state, persist to SQLite, return {}."""
    report = state.get("final_report")
    if report is not None:
        save_debate(report)
    return {}
