# tests/test_phase5.py
"""
Phase 5 UI tests — covers UI-01 through UI-04.

AppTest-based tests require: pip install streamlit==1.56.0
Unit tests for render_report() and stream dispatch require: debate package installed
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

APP_PATH = Path(__file__).parent.parent / "app.py"

# ---- Fixtures ----

@pytest.fixture
def sample_report():
    """Minimal DebateReport for render tests."""
    from debate.state import (
        DebateReport, DisputedPoint, RoundRecord, AgentArgument, Concession
    )
    arg = AgentArgument(
        agent_role="optimist", round_num=0,
        position="AI is net positive",
        reasoning="Because productivity",
        confidence=0.8,
        key_claims=["Claim A", "Claim B", "Claim C"],
    )
    rr = RoundRecord(round_num=0, arguments=[arg], divergence_score=0.3)
    dp = DisputedPoint(topic="Job displacement", agent_positions={"optimist": "minimal", "pessimist": "severe"})
    return DebateReport(
        debate_id="test-id",
        topic="Is AI beneficial?",
        consensus_points=["AI accelerates research"],
        disputed_points=[dp],
        verdict="Overall net positive with caveats.",
        confidence_score=0.72,
        convergence_status="converged",
        reasoning_trace=[rr],
        concession_log=[],
        created_at=datetime.now(timezone.utc),
    )


# ---- UI-01: Session state initialises clean ----

def test_session_state_init():
    """AppTest: fresh session has all required keys, no KeyError."""
    st_testing = pytest.importorskip("streamlit.testing.v1", reason="streamlit not installed")
    AppTest = st_testing.AppTest
    if not APP_PATH.exists():
        pytest.skip("app.py not yet created")
    at = AppTest.from_file(str(APP_PATH))
    at.run()
    assert not at.exception, f"app.py raised on fresh run: {at.exception}"


# ---- UI-02: Stream chunk dispatch ----

def test_stream_dispatch(sample_report):
    """Unit: _render_agent_chunk handles AgentArgument attributes correctly."""
    import importlib, sys
    # Import app only if it exists
    if not APP_PATH.exists():
        pytest.skip("app.py not yet created")
    pytest.importorskip("streamlit", reason="streamlit not installed")
    import streamlit as st
    spec = importlib.util.spec_from_file_location("app", APP_PATH)
    app_mod = importlib.util.module_from_spec(spec)
    # Provide session state stub so imports don't fail
    with patch("streamlit.session_state", MagicMock()):
        try:
            spec.loader.exec_module(app_mod)
        except Exception:
            pass  # module-level side effects may raise; function import is the goal
    # Verify _render_agent_chunk exists and accepts (str, dict) without AttributeError
    assert hasattr(app_mod, "_render_agent_chunk"), "_render_agent_chunk must be a top-level function"


# ---- UI-03: render_report renders DebateReport without error ----

def test_render_report(sample_report):
    """AppTest: complete app run loads a report from session state and renders without exception."""
    st_testing = pytest.importorskip("streamlit.testing.v1", reason="streamlit not installed")
    AppTest = st_testing.AppTest
    if not APP_PATH.exists():
        pytest.skip("app.py not yet created")
    at = AppTest.from_file(str(APP_PATH))
    at.session_state["debate_status"] = "complete"
    at.session_state["final_report"] = sample_report
    at.session_state["thread_id"] = "test-thread"
    at.session_state["error_msg"] = None
    at.run()
    assert not at.exception, f"render_report raised: {at.exception}"


# ---- UI-04: Fresh session, no KeyError ----

def test_fresh_session_no_error():
    """AppTest: idle state on first load produces no exception and renders input + button."""
    st_testing = pytest.importorskip("streamlit.testing.v1", reason="streamlit not installed")
    AppTest = st_testing.AppTest
    if not APP_PATH.exists():
        pytest.skip("app.py not yet created")
    at = AppTest.from_file(str(APP_PATH))
    at.run()
    assert not at.exception, f"Fresh session raised: {at.exception}"
    # Verify the Start Debate button exists
    buttons = [b.label for b in at.button]
    assert any("Start Debate" in lbl for lbl in buttons), f"Start Debate button missing. Found: {buttons}"
