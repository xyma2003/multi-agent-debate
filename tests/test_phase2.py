# tests/test_phase2.py
"""
Phase 2: Debate Engine — test suite.

DEBATE-04: test_compute_divergence_*  (this plan)
DEBATE-05: test_rebuttal_loop_fires   (Plan 02)
DEBATE-06: test_loop_terminates_*     (Plan 02)
DEBATE-07: test_concession_*          (Plan 02/03)
"""
import pytest

from debate.state import AgentArgument, Concession, RoundRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arg(role: str, claims: list[str]) -> AgentArgument:
    """Create a minimal AgentArgument with the given key_claims."""
    return AgentArgument(
        agent_role=role,
        round_num=0,
        position=f"Position for {role}",
        reasoning="Reasoning text.",
        confidence=0.7,
        key_claims=claims,
        concessions=[],
    )


# ---------------------------------------------------------------------------
# DEBATE-04: DivergeDetector
# ---------------------------------------------------------------------------

def test_compute_divergence_returns_score():
    """compute_divergence returns (float, list) for semantically opposed claims."""
    from debate.divergence import compute_divergence

    args = [
        _make_arg("optimist", [
            "Remote work dramatically boosts deep focus and productivity",
            "Eliminating commutes recovers 10+ hours per week for employees",
            "Async-first culture reduces unnecessary meeting overhead by 40%",
        ]),
        _make_arg("pessimist", [
            "Collaboration collapses without in-person whiteboard sessions",
            "Junior employees lose mentorship access in fully remote settings",
            "Work-life boundary erosion causes long-term burnout and attrition",
        ]),
    ]
    score, pairs = compute_divergence(args)

    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    assert isinstance(pairs, list), f"Expected list, got {type(pairs)}"


def test_compute_divergence_similar_claims():
    """When claims are semantically identical, divergence is near zero."""
    from debate.divergence import compute_divergence

    same_claims = [
        "AI will transform the economy over the next decade",
        "Machine learning models are becoming more capable each year",
        "Automation will displace some jobs while creating others",
    ]
    args = [
        _make_arg("optimist", same_claims),
        _make_arg("pessimist", same_claims),
    ]
    score, pairs = compute_divergence(args)

    assert score < 0.3, f"Expected low divergence for identical claims, got {score:.3f}"
    assert len(pairs) == 0, f"Expected no diverged pairs, got {pairs}"


def test_compute_divergence_empty():
    """Empty argument list returns (0.0, []) without error."""
    from debate.divergence import compute_divergence

    score, pairs = compute_divergence([])
    assert score == 0.0
    assert pairs == []


def test_roundrecord_has_divergence_score():
    """RoundRecord must have divergence_score field defaulting to 0.0."""
    record = RoundRecord(round_num=0, arguments=[])
    assert hasattr(record, "divergence_score"), "RoundRecord missing divergence_score field"
    assert record.divergence_score == 0.0


# ---------------------------------------------------------------------------
# DEBATE-05 / DEBATE-06: Rebuttal loop (Plan 02 — stubs)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implemented in Plan 02")
def test_rebuttal_loop_fires_on_divergence():
    """Graph invokes agent nodes a second time when divergence is above threshold."""
    pass


@pytest.mark.skip(reason="Implemented in Plan 02")
def test_loop_terminates_at_max_rounds():
    """Graph routes to synthesize_stub after max_rounds regardless of divergence."""
    pass


@pytest.mark.skip(reason="Implemented in Plan 02")
def test_loop_exits_on_convergence():
    """If divergence < threshold after round 1, synthesize_stub called immediately."""
    pass


# ---------------------------------------------------------------------------
# DEBATE-07: Concession attribution (Plan 02/03 — stubs)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Implemented in Plan 02/03")
def test_concession_attribution():
    """Concession triggered_by_claim matches an opponent's key_claims entry."""
    pass
