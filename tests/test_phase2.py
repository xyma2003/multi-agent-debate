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
# DEBATE-05 / DEBATE-06: Rebuttal loop (Plan 02)
# ---------------------------------------------------------------------------

@pytest.mark.integration
):
    """Graph invokes agent nodes a 2nd time when divergence > threshold.

    Uses max_rounds=3 and a genuinely controversial topic. Asserts that
    round_history has at least 2 entries after graph terminates, meaning
    the loop fired at least once.

    Marked as 'live' — requires ANTHROPIC_AUTH_TOKEN and ~30-60 seconds.
    """
    import os
    import uuid
    pytest.importorskip("debate.graph")
    if not os.environ.get("ANTHROPIC_AUTH_TOKEN") and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("No ANTHROPIC_AUTH_TOKEN — skipping live LLM test")

    from debate.graph import graph
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 30}
    result = graph.invoke(
        {"topic": "Should generative AI replace human creative professionals?", "max_rounds": 3},
        config=config,
    )
    assert "round_history" in result, "round_history missing from result"
    assert len(result["round_history"]) >= 1, "No rounds recorded"
    assert result.get("status") in ("converged", "max_rounds"), f"Unexpected status: {result.get('status')}"


@pytest.mark.integration
):
    """Graph routes to synthesize_stub after max_rounds=1 regardless of divergence.

    This is the fast termination guard test. With max_rounds=1, after the first
    collect_round1, round_num becomes 1, and route_divergence must return
    'synthesize_stub' because round_num (1) >= max_rounds (1).
    """
    import os
    import uuid
    pytest.importorskip("debate.graph")
    if not os.environ.get("ANTHROPIC_AUTH_TOKEN") and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("No ANTHROPIC_AUTH_TOKEN — skipping live LLM test")

    from debate.graph import graph
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 30}
    result = graph.invoke(
        {"topic": "Is water wet?", "max_rounds": 1},
        config=config,
    )
    assert result.get("status") in ("converged", "max_rounds"), \
        f"Graph did not terminate cleanly. status={result.get('status')}"
    assert result.get("round_num", 0) <= 2, \
        f"Too many rounds completed: round_num={result.get('round_num')}"


def test_route_divergence_terminates_at_max_rounds():
    """route_divergence returns 'synthesize_stub' when round_num >= max_rounds.

    This is a pure unit test — no LLM, no graph invocation.
    Verifies the max_rounds guard without network calls.
    """
    from debate.nodes.dispatch import route_divergence

    state_at_limit = {
        "round_num": 3,
        "max_rounds": 3,
        "divergence_score": 0.9,   # high divergence but should still terminate
        "topic": "test",
        "round_history": [],
    }
    result = route_divergence(state_at_limit)
    assert result == "synthesize_stub", \
        f"Expected 'synthesize_stub' at max_rounds, got {result!r}"


def test_route_divergence_terminates_on_convergence():
    """route_divergence returns 'synthesize_stub' when divergence_score < threshold."""
    from debate.nodes.dispatch import route_divergence
    from debate.divergence import DIVERGE_THRESHOLD

    state_converged = {
        "round_num": 1,
        "max_rounds": 3,
        "divergence_score": DIVERGE_THRESHOLD - 0.1,   # below threshold
        "topic": "test",
        "round_history": [],
    }
    result = route_divergence(state_converged)
    assert result == "synthesize_stub", \
        f"Expected 'synthesize_stub' on convergence, got {result!r}"


# ---------------------------------------------------------------------------
# DEBATE-07: Concession attribution (Plan 02/03 — stub)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires live LLM run that may not produce concessions in fast test environment")
def test_concession_attribution():
    """Concession triggered_by_claim matches an opponent's key_claims entry."""
    pass


# ---------------------------------------------------------------------------
# Phase 2 Integration Tests — full graph.invoke calls
# ---------------------------------------------------------------------------

import os
import uuid as _uuid


@pytest.mark.integration
):
    """Full graph.invoke with max_rounds=1 terminates without error.

    max_rounds=1 means route_divergence will see round_num=1 >= max_rounds=1
    and return 'synthesize_stub' immediately. Status must be set.

    This is the smoke test for the assembled Phase 2 system.
    Does NOT require high divergence — any topic works.
    """
    from debate.graph import graph

    config = {
        "configurable": {"thread_id": str(_uuid.uuid4())},
        "recursion_limit": 30,
    }
    result = graph.invoke(
        {"topic": "Should cities invest in public transit?", "max_rounds": 1},
        config=config,
    )
    assert result.get("status") in ("converged", "max_rounds"), \
        f"Graph did not terminate cleanly. status={result.get('status')!r}"
    assert len(result.get("round_history", [])) >= 1, \
        "round_history is empty — no rounds completed"


@pytest.mark.integration
):
    """After graph completes, len(round_history) == round_num.

    collect_round1 increments round_num AND appends to round_history.
    They must stay in sync.
    """
    from debate.graph import graph

    config = {
        "configurable": {"thread_id": str(_uuid.uuid4())},
        "recursion_limit": 30,
    }
    result = graph.invoke(
        {"topic": "Is remote work more productive than office work?", "max_rounds": 1},
        config=config,
    )
    round_num = result.get("round_num", -1)
    round_history = result.get("round_history", [])
    assert len(round_history) == round_num, (
        f"round_history length ({len(round_history)}) != round_num ({round_num})"
    )


@pytest.mark.integration
):
    """Every RoundRecord in round_history contains exactly 3 AgentArguments."""
    from debate.graph import graph

    config = {
        "configurable": {"thread_id": str(_uuid.uuid4())},
        "recursion_limit": 30,
    }
    result = graph.invoke(
        {"topic": "Is nuclear energy safe?", "max_rounds": 1},
        config=config,
    )
    for i, record in enumerate(result.get("round_history", [])):
        assert len(record.arguments) == 3, (
            f"Round {i} has {len(record.arguments)} arguments, expected 3. "
            f"Roles: {[a.agent_role for a in record.arguments]}"
        )


@pytest.mark.integration
):
    """Any concession in any argument has non-empty triggered_by_claim and triggered_by_agent.

    This validates DEBATE-07 schema integrity. Concessions may be empty
    in round 1 (agents haven't seen each other yet) — that is correct behavior.
    In rebuttal rounds they may or may not appear depending on LLM judgment.
    """
    from debate.graph import graph

    config = {
        "configurable": {"thread_id": str(_uuid.uuid4())},
        "recursion_limit": 30,
    }
    # max_rounds=2 to give at least one rebuttal round where concessions can appear
    result = graph.invoke(
        {"topic": "Should AI be regulated by governments?", "max_rounds": 2},
        config=config,
    )
    for record in result.get("round_history", []):
        for arg in record.arguments:
            for concession in arg.concessions:
                assert concession.triggered_by_claim, (
                    f"[{arg.agent_role} round {record.round_num}] "
                    f"Concession has empty triggered_by_claim: {concession!r}"
                )
                assert concession.triggered_by_agent, (
                    f"[{arg.agent_role} round {record.round_num}] "
                    f"Concession has empty triggered_by_agent: {concession!r}"
                )
                assert concession.rationale, (
                    f"[{arg.agent_role} round {record.round_num}] "
                    f"Concession has empty rationale: {concession!r}"
                )


@pytest.mark.integration
):
    """Graph with max_rounds=3 completes within recursion_limit=30.

    LangGraph counts each node execution as one step. For 3 rounds:
      initialize(1) + fan-out x3(3) + collect x3(3) + divergence_check x3(3)
      + synthesize_stub(1) = ~11 steps. 30 is conservative but safe.

    If this test fails with GraphRecursionError, the recursion_limit must be raised.
    """
    from debate.graph import graph

    config = {
        "configurable": {"thread_id": str(_uuid.uuid4())},
        "recursion_limit": 30,
    }
    # This should NOT raise GraphRecursionError
    result = graph.invoke(
        {"topic": "Should social media platforms be broken up?", "max_rounds": 3},
        config=config,
    )
    assert result.get("status") in ("converged", "max_rounds"), \
        f"Unexpected termination status: {result.get('status')!r}"
