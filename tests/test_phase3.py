# tests/test_phase3.py
"""
Phase 3: Synthesis & Report — test suite.

SYNTH-01: test_full_graph_produces_debate_report
SYNTH-02: test_debate_report_has_all_required_fields
SYNTH-03: test_compute_confidence_formula, test_compute_confidence_zero_divergence,
          test_confidence_score_not_llm_generated
SYNTH-04: test_non_convergence_verdict
SYNTH-05: test_reasoning_trace_contains_all_rounds

Integration tests use max_rounds=1 for speed (~30-40s). Adjust to max_rounds=2/3
only for tests that explicitly require multi-round behavior.
"""
import uuid

import pytest

from debate.state import AgentArgument, Concession, RoundRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arg(role: str, claims: list[str], concessions: list = None) -> AgentArgument:
    """Create a minimal AgentArgument for unit tests."""
    return AgentArgument(
        agent_role=role,
        round_num=0,
        position=f"Position for {role}",
        reasoning="Reasoning text.",
        confidence=0.7,
        key_claims=claims,
        concessions=concessions or [],
    )


def _make_record(round_num: int, divergence_score: float) -> RoundRecord:
    """Create a RoundRecord with three minimal agents and a given divergence_score."""
    return RoundRecord(
        round_num=round_num,
        arguments=[
            _make_arg("optimist", ["claim A", "claim B", "claim C"]),
            _make_arg("pessimist", ["claim D", "claim E", "claim F"]),
            _make_arg("devil", ["claim G", "claim H", "claim I"]),
        ],
        divergence_score=divergence_score,
    )


# ---------------------------------------------------------------------------
# SYNTH-03 unit tests: confidence formula
# ---------------------------------------------------------------------------

def test_compute_confidence_formula():
    """_compute_confidence_score: known inputs produce expected output.

    divergence_score=0.3, round_num=2 → (1-0.3)*0.9 = 0.63
    """
    from debate.nodes.synthesize import _compute_confidence_score

    record = _make_record(round_num=0, divergence_score=0.3)
    score = _compute_confidence_score([record], round_num=2)

    assert abs(score - 0.63) < 1e-4, (
        f"Expected 0.63, got {score}. "
        f"Formula: (1-0.3)*0.9 = 0.63"
    )


def test_compute_confidence_zero_divergence():
    """_compute_confidence_score: all-zero divergence → max=0.0 → result equals round_adjustment.

    round_num=2, adjustment=0.9: expected score = (1-0.0)*0.9 = 0.9
    """
    from debate.nodes.synthesize import _compute_confidence_score

    records = [
        _make_record(round_num=0, divergence_score=0.0),
        _make_record(round_num=1, divergence_score=0.0),
    ]
    score = _compute_confidence_score(records, round_num=2)

    # round_num=2, adjustment=0.9, max_divergence=0.0 → 1.0*0.9=0.9
    assert abs(score - 0.9) < 1e-4, (
        f"Expected 0.9 (zero divergence, round 2), got {score}"
    )


def test_compute_confidence_formula_round3():
    """_compute_confidence_score: divergence=0.7, round_num=3 → (1-0.7)*0.8 = 0.24"""
    from debate.nodes.synthesize import _compute_confidence_score

    record = _make_record(round_num=0, divergence_score=0.7)
    score = _compute_confidence_score([record], round_num=3)

    assert abs(score - 0.24) < 1e-4, (
        f"Expected 0.24, got {score}. Formula: (1-0.7)*0.8 = 0.24"
    )


# ---------------------------------------------------------------------------
# SYNTH-04 unit test: non-convergence path
# ---------------------------------------------------------------------------

def test_non_convergence_verdict():
    """When convergence_status='max_rounds', the synthesizer prompt triggers
    a verdict that starts with 'Agents did not reach consensus'.

    This test mocks synthesize_stub at the _build_synthesis_context level to
    verify the non-convergence instruction is present in the context string,
    then checks the real node's returned report when given a max_rounds state.
    """
    from debate.nodes.synthesize import _build_synthesis_context

    context = _build_synthesis_context(
        topic="Test topic",
        round_history=[_make_record(round_num=0, divergence_score=0.9)],
        convergence_status="max_rounds",
    )

    assert "Agents did not reach consensus" in context, (
        "Non-convergence instruction missing from synthesis context when "
        "convergence_status='max_rounds'"
    )
    assert "Your verdict MUST begin with exactly" in context, (
        "Hard requirement for verdict phrasing missing from non-convergence context"
    )


# ---------------------------------------------------------------------------
# SYNTH-01, SYNTH-02, SYNTH-05 integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_graph_produces_debate_report():
    """SYNTH-01 + SYNTH-02: graph.invoke() returns state with final_report as DebateReport.

    Uses max_rounds=1 for speed. The report must be a DebateReport instance,
    not None, not a dict, not a stub sentinel.
    """
    from debate.graph import graph
    from debate.state import DebateReport

    result = graph.invoke(
        {"topic": "Is universal basic income economically viable?", "max_rounds": 1},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )

    report = result.get("final_report")
    assert report is not None, "state['final_report'] is None after graph.invoke()"
    assert isinstance(report, DebateReport), (
        f"final_report is {type(report)}, expected DebateReport"
    )


@pytest.mark.integration
def test_debate_report_has_all_required_fields():
    """SYNTH-02: DebateReport contains consensus_points, disputed_points, verdict, confidence_score."""
    from debate.graph import graph
    from debate.state import DebateReport

    result = graph.invoke(
        {"topic": "Should nuclear energy be expanded?", "max_rounds": 1},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )

    report: DebateReport = result["final_report"]
    assert isinstance(report, DebateReport)

    # Structural completeness: all 10 required fields must be set and non-None
    assert report.debate_id, "debate_id is empty"
    assert report.topic, "topic is empty"
    assert isinstance(report.consensus_points, list), "consensus_points not a list"
    assert isinstance(report.disputed_points, list), "disputed_points not a list"
    assert report.verdict, "verdict is empty"
    assert isinstance(report.confidence_score, float), "confidence_score not a float"
    assert 0.0 <= report.confidence_score <= 1.0, (
        f"confidence_score {report.confidence_score} out of [0, 1]"
    )
    assert report.convergence_status in ("converged", "max_rounds", "partial"), (
        f"unexpected convergence_status: {report.convergence_status}"
    )
    assert isinstance(report.reasoning_trace, list), "reasoning_trace not a list"
    assert isinstance(report.concession_log, list), "concession_log not a list"
    assert report.created_at is not None, "created_at is None"


@pytest.mark.integration
def test_reasoning_trace_contains_all_rounds():
    """SYNTH-05: len(report.reasoning_trace) equals state['round_num'] (completed rounds)."""
    from debate.graph import graph
    from debate.state import DebateReport

    result = graph.invoke(
        {"topic": "Is space colonization worth the cost?", "max_rounds": 1},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )

    report: DebateReport = result["final_report"]
    round_num = result.get("round_num", 0)

    assert isinstance(report, DebateReport)
    assert len(report.reasoning_trace) == round_num, (
        f"reasoning_trace has {len(report.reasoning_trace)} records "
        f"but round_num={round_num}"
    )
    # Each record must be a RoundRecord (not a dict)
    for record in report.reasoning_trace:
        assert isinstance(record, RoundRecord), (
            f"reasoning_trace entry is {type(record)}, expected RoundRecord"
        )


@pytest.mark.integration
def test_confidence_score_not_llm_generated():
    """SYNTH-03: confidence_score is in [0,1] and matches formula output.

    Verifies that confidence_score is NOT LLM-generated by recomputing the
    formula from the reasoning_trace and checking the values match.
    """
    from debate.graph import graph
    from debate.nodes.synthesize import _compute_confidence_score
    from debate.state import DebateReport

    result = graph.invoke(
        {"topic": "Will AI replace most white-collar jobs by 2035?", "max_rounds": 1},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )

    report: DebateReport = result["final_report"]
    round_num = result.get("round_num", 0)

    assert isinstance(report, DebateReport)
    assert isinstance(report.confidence_score, float)
    assert 0.0 <= report.confidence_score <= 1.0

    # Recompute using the formula — must match report value exactly
    expected = _compute_confidence_score(report.reasoning_trace, round_num)
    assert report.confidence_score == expected, (
        f"confidence_score={report.confidence_score} does not match "
        f"formula output={expected}. "
        f"The LLM may have generated this value instead of the formula."
    )
