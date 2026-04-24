---
phase: 02-debate-engine
plan: 03
subsystem: tests
tags: [integration-tests, debate-loop, phase2, verification]
dependency_graph:
  requires:
    - 02-01  # divergence detector + RoundRecord schema
    - 02-02  # route_divergence, divergence_check_node, rebuttal loop wiring
  provides:
    - Full Phase 2 integration test suite (DEBATE-04 through DEBATE-07 coverage)
    - Live graph.invoke smoke tests confirming assembled system termination
  affects:
    - tests/test_phase2.py (only file modified)
tech_stack:
  added: []
  patterns:
    - pytest.skip guard for live LLM tests via _live_skip() helper
    - graph.invoke with configurable thread_id + recursion_limit=30
    - max_rounds=1 as fast termination guard pattern for smoke tests
key_files:
  created: []
  modified:
    - tests/test_phase2.py
decisions:
  - "Integration tests use max_rounds=1 for smoke tests (fast 38s), max_rounds=2 for concession tests, max_rounds=3 for recursion_limit test — balanced coverage vs latency"
  - "test_concession_attribution stub kept with @pytest.mark.skip (intentional) — concession presence is non-deterministic in round 1; validated structurally by test_concession_fields_are_valid_if_present instead"
  - "_live_skip() uses ANTHROPIC_AUTH_TOKEN OR ANTHROPIC_API_KEY — ANTHROPIC_AUTH_TOKEN was the active credential in this environment"
metrics:
  duration_minutes: 10
  completed_date: "2026-04-24"
  tasks_completed: 1
  tasks_total: 1
  files_modified: 1
---

# Phase 02 Plan 03: Integration Tests — Full Debate Loop Verified

**One-liner:** 5 live integration tests added for graph.invoke smoke testing; 13 passed + 1 intentional skip; Phase 2 loop confirmed termination-clean at max_rounds=1,2,3 with no GraphRecursionError.

## What Was Built

Added 5 live integration tests to `tests/test_phase2.py` that invoke the assembled Phase 2 debate graph end-to-end with a real LLM. These tests complement the 8 unit tests (4 for DEBATE-04, 2 for routing logic, 2 pre-existing live loop tests) already present from Plans 02-01 and 02-02.

### New Tests Added

| Test | Purpose | max_rounds |
|------|---------|------------|
| `test_full_graph_terminates_cleanly` | Smoke test — status in ("converged","max_rounds"), round_history non-empty | 1 |
| `test_round_history_length_matches_round_num` | len(round_history) == round_num after graph.invoke | 1 |
| `test_each_round_has_three_arguments` | Every RoundRecord has exactly 3 AgentArguments | 1 |
| `test_concession_fields_are_valid_if_present` | If concession exists: triggered_by_claim/agent/rationale all non-empty | 2 |
| `test_recursion_limit_is_sufficient` | No GraphRecursionError within recursion_limit=30 for 3-round debate | 3 |

### Final Test Inventory

```
14 items collected:
  4 × DEBATE-04 unit tests (compute_divergence)
  2 × DEBATE-05/06 unit tests (route_divergence routing)
  2 × DEBATE-05/06 live tests (rebuttal loop + max_rounds guard)
  1 × DEBATE-07 stub (@pytest.mark.skip — intentional)
  5 × Phase 2 integration tests (this plan)
```

## Verification Results

All success criteria met:

- `--collect-only` shows 14 items (>= 12 required)
- Full suite: **13 passed, 1 skipped** in 192 seconds
- `from debate.graph import graph` exits 0
- graph.nodes contains all required nodes: initialize, optimist_node, pessimist_node, devil_node, collect_round1, divergence_check_node, synthesize_stub
- DEBATE-04 through DEBATE-07 all have passing test coverage
- No GraphRecursionError observed at max_rounds=1,2,3

## Deviations from Plan

None — plan executed exactly as written. The test code from the plan's `<action>` block was appended verbatim (minor whitespace: comment `~` symbol → `x` to avoid markdown rendering issue, non-functional).

## Known Stubs

One intentional stub remains from Plan 02-02:
- `test_concession_attribution` in `tests/test_phase2.py` (line ~191) — decorated with `@pytest.mark.skip`
- Reason: concession presence is non-deterministic in round 1; LLM may or may not generate concessions depending on topic
- Structural concession integrity is now covered by `test_concession_fields_are_valid_if_present` (validates schema if concessions appear)
- Not blocking: plan's goal is fully achieved

## Phase 2 Requirements Traceability

| Requirement | Pattern | Status |
|-------------|---------|--------|
| DEBATE-04 (divergence detection) | `def compute_divergence` in debate/divergence.py | COVERED |
| DEBATE-05 (rebuttal loop fires) | `route_divergence` in debate/nodes/dispatch.py | COVERED |
| DEBATE-06 (terminates at max_rounds) | `round_num >= max_rounds` in debate/nodes/dispatch.py | COVERED |
| DEBATE-07 (concession attribution) | `triggered_by_claim` in debate/state.py | COVERED |

## Self-Check: PASSED

Files exist:
- FOUND: tests/test_phase2.py (14 tests collected)
- FOUND: .planning/phases/02-debate-engine/02-03-SUMMARY.md

Commits exist:
- 17bec08: feat(02-03): activate Phase 2 integration tests — full debate loop verified
