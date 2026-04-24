---
phase: 1
plan: 3
subsystem: graph-wiring
tags: [langgraph, stategraph, send-fanout, smoke-test, graph-compilation]
dependency_graph:
  requires: [01-02]
  provides: [compiled-debate-graph, phase1-smoke-test]
  affects: [phase-2-divergence, all-downstream-plans]
tech_stack:
  added: []
  patterns:
    - "add_conditional_edges(source, routing_fn) where routing_fn returns list[Send] for parallel fan-out"
    - "dispatch function passed directly as routing function, NOT registered as a graph node"
    - "InMemorySaver from langgraph.checkpoint.memory (not deprecated MemorySaver)"
key_files:
  created:
    - debate/graph.py
    - tests/test_phase1.py
  modified: []
decisions:
  - "dispatch_round1 must be passed as routing function to add_conditional_edges, NOT registered as a node — registering it as a node causes InvalidUpdateError because LangGraph tries to merge list[Send] as a state update dict"
  - "add_conditional_edges('initialize', dispatch_round1) is the correct fan-out wiring in LangGraph 1.1.9"
metrics:
  duration_seconds: 267
  completed_date: "2026-04-24"
  tasks_completed: 2
  files_created: 2
  files_modified: 0
---

# Phase 1 Plan 3: Graph Wiring and Smoke Test Summary

**One-liner:** StateGraph wired with Send fan-out via `add_conditional_edges("initialize", dispatch_round1)` and smoke-tested live against claude-sonnet-4-6; graph.invoke returns 3 non-sentinel AgentArguments in round_history[0].

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Assemble StateGraph | 6701978 | debate/graph.py |
| 2 | Write and Run Smoke Test | ae39563 | debate/graph.py (fixed), tests/test_phase1.py |

## Outcomes

- `debate/graph.py` created with `build_graph()` function and module-level `graph` object
- `from debate.graph import graph` imports cleanly
- `graph.invoke({"topic": "...", "max_rounds": 3}, config=...)` returns state with `round_history[0].arguments` containing exactly 3 AgentArguments with roles optimist, pessimist, devil
- Each argument has non-empty `position`, non-empty `reasoning`, `len(key_claims) >= 3`, `confidence` in [0.0, 1.0], and `is_sentinel=False`
- `python tests/test_phase1.py` prints "Phase 1 smoke test PASSED" and "Persona compliance check complete"
- `InMemorySaver` imported from `langgraph.checkpoint.memory` (correct; not deprecated `MemorySaver` alias)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] dispatch_round1 must not be registered as a graph node**

- **Found during:** Task 2 smoke test execution
- **Issue:** Plan specified `builder.add_node("dispatch_round1", dispatch_round1)` followed by `builder.add_conditional_edges("dispatch_round1", lambda s: s)`. In LangGraph 1.1.9, when a function is registered as a node, its return value is treated as a state update dict. A `list[Send]` return triggers `InvalidUpdateError: Expected dict, got [Send(...), ...]`.
- **Root cause:** In LangGraph 1.x, `Send` fan-out requires the dispatch function to be passed _directly_ as the routing callable to `add_conditional_edges`. The function must NOT be registered as a node. The `lambda s: s` identity lambda pattern in the plan was incorrect — that would receive the state (not the node's return value) and pass it through as the routing result.
- **Fix:** Removed `builder.add_node("dispatch_round1", dispatch_round1)`. Changed wiring to `builder.add_conditional_edges("initialize", dispatch_round1)` — dispatch_round1 is called with the state after initialize runs, returns `list[Send]`, and LangGraph dispatches the three agent nodes in parallel.
- **Files modified:** `debate/graph.py`
- **Commit:** ae39563

### Persona Compliance (soft warnings, not failures)

The `test_persona_compliance` test is non-failing by design — LLM output is probabilistic. On two separate runs:
- Run 1: Pessimist used words "opportunity", "growth" (2 violations)
- Run 2: Optimist used "risk", "fail"; Pessimist used "growth"

These are expected. The Phase 1 acceptance gate is `test_phase1_returns_three_agent_arguments` (structural validation), not persona purity.

## Key Technical Finding

In LangGraph 1.1.9, `Send` fan-out wiring follows this exact pattern:

```python
# CORRECT: dispatch function is the routing callable, not a node
builder.add_conditional_edges("initialize", dispatch_round1)

# WRONG (plan's suggestion): dispatch function registered as node
# builder.add_node("dispatch_round1", dispatch_round1)       # do NOT do this
# builder.add_conditional_edges("dispatch_round1", lambda s: s)  # identity lambda fails
```

The routing function receives the current state dict and returns `list[Send]`. LangGraph handles the parallel dispatch automatically.

## Known Stubs

None. All data is wired to live LLM calls. No hardcoded placeholders.

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| debate/graph.py exists | FOUND |
| tests/test_phase1.py exists | FOUND |
| 03-SUMMARY.md exists | FOUND |
| commit 6701978 exists | FOUND |
| commit ae39563 exists | FOUND |
