---
plan: 03
phase: 1
title: "Graph Wiring and Smoke Test"
wave: 3
depends_on: [01, 02]
requirements_addressed: [DEBATE-01, DEBATE-02, DEBATE-03, AGENT-01, AGENT-02, AGENT-03]
files_modified:
  - debate/graph.py
  - tests/test_phase1.py
autonomous: true

must_haves:
  truths:
    - "Developer can run `python tests/test_phase1.py` and see 'Phase 1 smoke test PASSED'"
    - "`graph.invoke({\"topic\": \"...\", \"max_rounds\": 3}, config=...)` returns a state dict where `result['round_history'][0].arguments` has exactly 3 AgentArgument objects with roles optimist, pessimist, devil"
    - "Each returned AgentArgument has non-empty position, non-empty reasoning, and at least 3 key_claims"
    - "No unhandled exception is raised during the invoke — sentinel injection works if parse fails"
  artifacts:
    - path: "debate/graph.py"
      provides: "Compiled StateGraph exposed as module-level `graph` object"
      exports: [graph, build_graph]
    - path: "tests/test_phase1.py"
      provides: "Smoke test: 3 AgentArguments returned + persona compliance warnings"
      contains: "test_phase1_returns_three_agent_arguments"
  key_links:
    - from: "debate/graph.py"
      to: "debate/nodes/dispatch.py"
      via: "builder.add_conditional_edges('dispatch_round1', lambda s: s)"
      pattern: "add_conditional_edges.*lambda s.*s"
    - from: "optimist_node, pessimist_node, devil_node"
      to: "collect_round1"
      via: "builder.add_edge"
    - from: "tests/test_phase1.py"
      to: "debate/graph.py"
      via: "from debate.graph import graph"
---

# Plan 03: Graph Wiring and Smoke Test

## Objective

Wire all nodes from Plan 02 into a compiled StateGraph and run a live smoke test against
the real LLM API. This plan is the Phase 1 acceptance gate: it only passes when
`python tests/test_phase1.py` prints "Phase 1 smoke test PASSED" with three distinct,
non-sentinel AgentArgument objects in the output.

The graph wiring must use `builder.add_conditional_edges("dispatch_round1", lambda s: s)`
for the Send fan-out — any other approach will break parallel dispatch.

## Tasks

### Task 1: Assemble the StateGraph

<read_first>
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Pattern 2: Send Fan-Out, "Wiring in graph.py" code block (add_conditional_edges call)
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Pitfall 5: Wrong add_conditional_edges call for Send (the error to avoid)
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — State of the Art (InMemorySaver correct import: `from langgraph.checkpoint.memory import InMemorySaver`)
</read_first>

<action>
Create `debate/graph.py`:

```python
# debate/graph.py
"""
StateGraph assembly for the multi-agent debate system.

Phase 1 topology (linear + fan-out/fan-in):

  START --> initialize --> dispatch_round1
                              |
             (Send fan-out — 3 parallel branches)
             |               |               |
        optimist_node  pessimist_node   devil_node
             |               |               |
             +-----------+---+---------------+
                         |
                    collect_round1 --> END

Phase 2 will replace the collect_round1 --> END edge with conditional routing
to a divergence detector and rebuttal loop.

Import: `from debate.graph import graph` then `graph.invoke({"topic": ..., "max_rounds": 3}, config=...)`
"""
from langgraph.checkpoint.memory import InMemorySaver  # NOT MemorySaver — deprecated alias
from langgraph.graph import END, START, StateGraph

from debate.nodes.agents import devil_node, optimist_node, pessimist_node
from debate.nodes.collect import collect_round1
from debate.nodes.dispatch import dispatch_round1
from debate.nodes.initialize import initialize_node
from debate.state import DebateState


def build_graph():
    """Build and compile the Phase 1 debate graph."""
    builder = StateGraph(DebateState)

    # Register all nodes
    builder.add_node("initialize", initialize_node)
    builder.add_node("dispatch_round1", dispatch_round1)
    builder.add_node("optimist_node", optimist_node)
    builder.add_node("pessimist_node", pessimist_node)
    builder.add_node("devil_node", devil_node)
    builder.add_node("collect_round1", collect_round1)

    # Linear edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "dispatch_round1")

    # Fan-out: dispatch_round1 returns list[Send] — identity lambda passes it through
    # DO NOT use add_edge here — that creates a single edge, not a parallel fan-out
    builder.add_conditional_edges("dispatch_round1", lambda s: s)

    # Fan-in: all three agent nodes converge to collect_round1
    builder.add_edge("optimist_node", "collect_round1")
    builder.add_edge("pessimist_node", "collect_round1")
    builder.add_edge("devil_node", "collect_round1")

    # Phase 1 terminal edge — Phase 2 will replace this
    builder.add_edge("collect_round1", END)

    # InMemorySaver required for interrupt() support in Phase 4
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Module-level compiled graph — import this directly
graph = build_graph()
```

After writing, verify the import is clean:
```bash
python -c "from debate.graph import graph; print('graph compiled OK')"
```

If an `ImportError` or `ModuleNotFoundError` is raised, check that Plan 01 and Plan 02
files are all present before proceeding.
</action>

<acceptance_criteria>
- [ ] `python -c "from debate.graph import graph; print('OK')"` exits 0 and prints `OK`
- [ ] `grep "add_conditional_edges.*dispatch_round1.*lambda s.*s" debate/graph.py` matches
- [ ] `grep "InMemorySaver" debate/graph.py` matches (not MemorySaver)
- [ ] `grep "from langgraph.checkpoint.memory import InMemorySaver" debate/graph.py` matches (correct import path)
- [ ] `grep "add_edge.*optimist_node.*collect_round1" debate/graph.py` matches
- [ ] `grep "add_edge.*pessimist_node.*collect_round1" debate/graph.py` matches
- [ ] `grep "add_edge.*devil_node.*collect_round1" debate/graph.py` matches
</acceptance_criteria>

---

### Task 2: Write and Run Smoke Test

<read_first>
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Testing Phase 1 section (full test_phase1.py code, "What Phase 1 Done Looks Like")
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Persona compliance checks (forbidden word lists for optimist and pessimist)
</read_first>

<action>
Create `tests/test_phase1.py`. This is a live integration test — it makes real LLM API calls.
Requires the env vars `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_CUSTOM_HEADERS`
to be set. On any import failure, fix the missing module before running.

```python
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
    assert len(round_history) >= 1, "round_history is empty — collect_round1 may not have run"
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
            f"{arg.agent_role}: sentinel injected — parse failure occurred. "
            f"Check ANTHROPIC env vars and LLM connectivity."
        )


def test_persona_compliance():
    """Soft check: warn if agents are producing hedged, balanced responses (persona drift).

    This test NEVER fails — LLM output is probabilistic.
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
        print(f"\nWARNING: Optimist persona drift — found forbidden words: {opt_violations}")
    else:
        print("\nOptimist: no persona drift detected")

    if pess_violations:
        print(f"WARNING: Pessimist persona drift — found forbidden words: {pess_violations}")
    else:
        print("Pessimist: no persona drift detected")


if __name__ == "__main__":
    print("Running Phase 1 smoke test...")
    test_phase1_returns_three_agent_arguments()
    print("Phase 1 smoke test PASSED")
    print("\nRunning persona compliance check...")
    test_persona_compliance()
    print("Persona compliance check complete")
```

After writing `tests/test_phase1.py`, run the smoke test:

```bash
cd /Users/maxinyue09/Downloads/projects/项目/debate-agent
python tests/test_phase1.py
```

Expected terminal output:
```
Running Phase 1 smoke test...
Phase 1 smoke test PASSED

Running persona compliance check...
Optimist: no persona drift detected
Pessimist: no persona drift detected
Persona compliance check complete
```

If the test fails with `AssertionError: sentinel injected`:
- The LLM returned an unparseable response. Check that env vars are set.
- Run `python -c "from debate.llm import _make_llm; llm = _make_llm(); print(llm.model)"` to verify connectivity.

If the test fails with `AssertionError: Expected 3 AgentArguments, got N`:
- The fan-in accumulator is broken. Verify `Annotated[list[AgentArgument], add]` in `debate/state.py`.
- Verify `add_conditional_edges("dispatch_round1", lambda s: s)` in `debate/graph.py`.
</action>

<acceptance_criteria>
- [ ] `python tests/test_phase1.py` exits 0 and final output contains `Phase 1 smoke test PASSED`
- [ ] `python tests/test_phase1.py` output contains `Persona compliance check complete`
- [ ] After test run: `result["round_history"][0].arguments` contains exactly 3 items (asserted inside test)
- [ ] No `AssertionError: sentinel injected` in output (all 3 agents parsed successfully)
- [ ] `grep "test_phase1_returns_three_agent_arguments" tests/test_phase1.py` matches
- [ ] `grep "recursion_limit.*30" tests/test_phase1.py` matches
</acceptance_criteria>

## Verification

```bash
cd /Users/maxinyue09/Downloads/projects/项目/debate-agent

# Import check (no LLM call)
python -c "from debate.graph import graph; print('graph import OK')"

# Structural checks
grep "add_conditional_edges" debate/graph.py
grep "InMemorySaver" debate/graph.py
grep "collect_round1" debate/graph.py | grep "add_edge" | wc -l   # expect: 3 (one per agent)

# Full smoke test (makes LLM calls — requires env vars)
python tests/test_phase1.py

# Pytest alternative
python -m pytest tests/test_phase1.py -v
```

## must_haves

- `python tests/test_phase1.py` prints "Phase 1 smoke test PASSED" — the graph returns 3 non-sentinel AgentArguments
- `result["round_history"][0].arguments` has exactly 3 items with roles optimist, pessimist, devil
- Each argument has non-empty `position`, non-empty `reasoning`, and `len(key_claims) >= 3`
- `debate/graph.py` uses `add_conditional_edges("dispatch_round1", lambda s: s)` — the correct Send fan-out pattern
- `InMemorySaver` is imported from `langgraph.checkpoint.memory` (not the deprecated `MemorySaver` alias)
