---
phase: 02-debate-engine
plan: 02
subsystem: debate-engine
tags: [langgraph, multi-agent, debate-loop, divergence-detection, send-fan-out, rebuttal, sentence-transformers]

# Dependency graph
requires:
  - phase: 02-01
    provides: divergence.py with compute_divergence, RoundRecord.divergence_score field, test_phase2.py with DEBATE-04 tests
  - phase: 01
    provides: graph.py Phase 1 topology, state.py DebateState, nodes/agents.py, nodes/dispatch.py, nodes/collect.py
provides:
  - Full Phase 2 debate loop: divergence check → route_divergence → rebuttal fan-out → collect (cyclic)
  - divergence_check_node: reads round_history[-1], calls compute_divergence, back-fills RoundRecord.divergence_score
  - synthesize_stub: Phase 3 placeholder, terminates loop with status "converged" or "max_rounds"
  - route_divergence: routing function checking max_rounds FIRST then divergence_score, returns list[Send] or "synthesize_stub"
  - _build_compact_summaries: builds ~100-word per-agent summaries from latest round for rebuttal context
  - _agent_node rebuttal context injection: injects opposing summaries + concession instructions when round_num > 0
affects: [03-synthesis, Phase 3 SYNTH-01 through SYNTH-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "route_divergence routing function: add_conditional_edges('divergence_check_node', route_divergence) returns list[Send] OR string — NOT registered as node (Pitfall 1)"
    - "max_rounds guard FIRST in route_divergence before divergence_score check (Pitfall 2)"
    - "collect_round1 reused for rebuttal rounds — no new collect node created (Pitfall 5)"
    - "Compact summaries: only latest round, top 3 key_claims per agent, ~360 tokens total opposing context"
    - "Rebuttal agents skip own prior argument when injecting opposing context (agent_role == role filter)"
    - "divergence_check_node back-fills RoundRecord.divergence_score via model_copy to avoid in-place mutation"

key-files:
  created:
    - debate/nodes/divergence_check.py
    - debate/nodes/synthesize.py
  modified:
    - debate/nodes/dispatch.py
    - debate/nodes/agents.py
    - debate/graph.py
    - tests/test_phase2.py

key-decisions:
  - "route_divergence uses DIVERGE_THRESHOLD (0.75) not 0.25 for Guard 2 — synthesize_stub uses 0.25 for status labeling; routing and labeling use different thresholds"
  - "divergence_check_node back-fills RoundRecord.divergence_score via model_copy (immutable update pattern, not in-place mutation)"
  - "test_loop_terminates_at_max_rounds kept as live LLM test (skips without API key) — route_divergence unit test covers the logic without LLM"

patterns-established:
  - "Pattern: Routing function returning list[Send] OR string passed to add_conditional_edges — both work in LangGraph 1.1.9 without warnings"
  - "Pattern: max_rounds guard as FIRST check in any rebuttal routing function (safety escape from divergent loops)"
  - "Pattern: Compact summary injection for rebuttal — only latest round, skip own role, top 3 claims only"

requirements-completed: [DEBATE-05, DEBATE-06, DEBATE-07]

# Metrics
duration: 4min
completed: 2026-04-24
---

# Phase 02 Plan 02: Debate Engine Loop Summary

**Full multi-round debate loop wired: divergence_check_node → route_divergence fan-out with max_rounds guard → rebuttal agents with compact opposing-argument injection → collect_round1 reuse → synthesize_stub termination**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-04-24T04:41:36Z
- **Completed:** 2026-04-24T04:44:55Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Created `divergence_check_node` that reads `round_history[-1]`, calls `compute_divergence`, back-fills `RoundRecord.divergence_score` immutably via `model_copy`, and writes score + pairs to state
- Created `synthesize_stub` Phase 3 placeholder that writes `status: "converged"/"max_rounds"` so the loop terminates cleanly and tests can assert on termination reason
- Extended `dispatch.py` with `route_divergence` (max_rounds guard first, then DIVERGE_THRESHOLD check) and `_build_compact_summaries` (~100 words per agent, latest round only, top 3 claims)
- Extended `_agent_node` to inject compact opposing summaries + anti-sycophancy concession instructions when `round_num > 0`
- Rewrote `graph.py` to Phase 2 topology: removed `collect_round1→END`, added `collect_round1→divergence_check_node→route_divergence (conditional)→synthesize_stub→END` with rebuttal loop
- Replaced 3 skip stubs in `test_phase2.py` with real `route_divergence` unit tests (2 passing) + live LLM loop tests (auto-skip without API key)

## Task Commits

1. **Task 1: Create divergence_check_node and synthesize_stub** - `b086979` (feat)
2. **Task 2: Wire rebuttal dispatch, update agents, assemble Phase 2 graph, activate tests** - `0049d88` (feat)

## Files Created/Modified

- `debate/nodes/divergence_check.py` — divergence_check_node: reads round_history[-1], computes score, back-fills RoundRecord
- `debate/nodes/synthesize.py` — synthesize_stub: Phase 3 placeholder writing status "converged"/"max_rounds"
- `debate/nodes/dispatch.py` — added route_divergence (max_rounds guard first) and _build_compact_summaries
- `debate/nodes/agents.py` — _agent_node extended with rebuttal context injection when round_num > 0
- `debate/graph.py` — Phase 2 loop topology; collect_round1→END removed; full loop wired
- `tests/test_phase2.py` — replaced 3 skip stubs with test_route_divergence_terminates_at_max_rounds and test_route_divergence_terminates_on_convergence (both passing)

## Decisions Made

- `route_divergence` Guard 2 uses `DIVERGE_THRESHOLD` (0.75) from `debate/divergence.py` — this is the semantic boundary below which agents are considered to have converged. The `synthesize_stub` separately uses 0.25 for labeling the termination reason (converged vs max_rounds); these serve different purposes.
- `divergence_check_node` back-fills `RoundRecord.divergence_score` using `model_copy(update=...)` rather than direct attribute assignment — Pydantic v2 model immutability is respected; `list(round_history[:-1]) + [updated_record]` pattern avoids in-place mutation.
- Live LLM tests (`test_rebuttal_loop_fires_on_divergence`, `test_loop_terminates_at_max_rounds`) auto-skip when no API key is present; the routing logic is covered by the two unit tests that require no LLM.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None. The `grep -n "collect_round1.*END"` verification command produced false positives from comment lines in `graph.py`, but a targeted Python check confirmed no active `add_edge("collect_round1", END)` exists in the file.

## Known Stubs

- `debate/nodes/synthesize.py` — `synthesize_stub` is an intentional Phase 3 placeholder. It writes `status` but produces no `final_report`. Phase 3 SYNTH-01 through SYNTH-05 will replace this node with the real synthesizer.

## Next Phase Readiness

- Phase 3 (Synthesis) can now read `round_history` with per-round `divergence_score` populated, `status` set to "converged"/"max_rounds", and `diverged_pairs` for targeted synthesis
- `synthesize_stub` in `graph.py` is the exact node Phase 3 will replace — clean handoff point
- Graph terminates cleanly on both convergence and max_rounds paths; no GraphRecursionError possible given max_rounds guard

## Self-Check: PASSED

Files exist:
- debate/nodes/divergence_check.py: FOUND
- debate/nodes/synthesize.py: FOUND
- debate/nodes/dispatch.py: FOUND (modified)
- debate/nodes/agents.py: FOUND (modified)
- debate/graph.py: FOUND (modified)
- tests/test_phase2.py: FOUND (modified)

Commits exist:
- b086979: FOUND (Task 1)
- 0049d88: FOUND (Task 2)

Tests: 2 passed (test_route_divergence_terminates_at_max_rounds, test_route_divergence_terminates_on_convergence)

---
*Phase: 02-debate-engine*
*Completed: 2026-04-24*
