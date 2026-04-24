# tests/test_phase1.py
"""
Phase 1 smoke test: invoke the debate graph and assert 3 AgentArguments are returned.

Run:
    python tests/test_phase1.py          # standalone
    python -m pytest tests/test_phase1.py -v  # with pytest

Requires env vars:
    ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_CUSTOM_HEADERS
"""
import uuid

from debate.graph import graph
from debate.state import AgentArgument


def test_phase1_returns_three_agent_arguments():
    """Invoke the graph with a topic and assert 3 AgentArguments are returned."""
    topic = "Is remote work more productive than office work?"
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 30,
    }

    result = graph.invoke(
        {"topic": topic, "max_rounds": 3},
        config=config,
    )

    # round_history[0].arguments must have exactly 3 items
    round_history = result.get("round_history", [])
    assert len(round_history) >= 1, "round_history is empty -- collect_round1 may not have run"
    args = round_history[0].arguments

    assert len(args) == 3, f"Expected 3 AgentArguments, got {len(args)}: {[a.agent_role for a in args]}"

    roles = {a.agent_role for a in args}
    assert roles == {"optimist", "pessimist", "devil"}, f"Missing roles: roles present = {roles}"

    # Validate structure of each argument
    for arg in args:
        assert isinstance(arg, AgentArgument), f"Expected AgentArgument, got {type(arg)}"
        assert arg.position, f"{arg.agent_role}: position must be non-empty"
        assert arg.reasoning, f"{arg.agent_role}: reasoning must be non-empty"
        assert len(arg.key_claims) >= 3, f"{arg.agent_role}: need >= 3 key_claims, got {len(arg.key_claims)}"
        assert 0.0 <= arg.confidence <= 1.0, f"{arg.agent_role}: confidence out of range: {arg.confidence}"
        assert not arg.is_sentinel, (
            f"{arg.agent_role}: sentinel injected -- parse failure occurred. "
            f"Check ANTHROPIC env vars and LLM connectivity."
        )


def test_persona_compliance():
    """Soft check: warn if agents are producing hedged, balanced responses (persona drift).

    This test NEVER fails -- LLM output is probabilistic.
    Violations are printed as warnings for manual review.
    """
    topic = "Should startups raise venture capital?"
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": 30,
    }

    result = graph.invoke({"topic": topic, "max_rounds": 3}, config=config)
    args = result["round_history"][0].arguments

    optimist = next(a for a in args if a.agent_role == "optimist")
    pessimist = next(a for a in args if a.agent_role == "pessimist")

    OPTIMIST_FORBIDDEN = ["risk", "fail", "problem", "challenge", "concern"]
    PESSIMIST_FORBIDDEN = ["opportunity", "upside", "growth", "potential"]

    optimist_text = (optimist.position + " " + optimist.reasoning).lower()
    pessimist_text = (pessimist.position + " " + pessimist.reasoning).lower()

    opt_violations = [w for w in OPTIMIST_FORBIDDEN if w in optimist_text]
    pess_violations = [w for w in PESSIMIST_FORBIDDEN if w in pessimist_text]

    if opt_violations:
        print(f"\nWARNING: Optimist persona drift -- found forbidden words: {opt_violations}")
    else:
        print("\nOptimist: no persona drift detected")

    if pess_violations:
        print(f"WARNING: Pessimist persona drift -- found forbidden words: {pess_violations}")
    else:
        print("Pessimist: no persona drift detected")


if __name__ == "__main__":
    print("Running Phase 1 smoke test...")
    test_phase1_returns_three_agent_arguments()
    print("Phase 1 smoke test PASSED")
    print("\nRunning persona compliance check...")
    test_persona_compliance()
    print("Persona compliance check complete")
