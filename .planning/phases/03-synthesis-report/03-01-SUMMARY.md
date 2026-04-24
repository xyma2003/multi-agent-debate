---
phase: 03-synthesis-report
plan: "01"
subsystem: synthesizer
tags: [synthesis, pydantic, llm, confidence-formula, debate-report]
dependency_graph:
  requires:
    - debate/state.py (Concession, AgentArgument, RoundRecord — Phase 1/2)
    - debate/divergence.py (DIVERGE_THRESHOLD — Phase 2)
    - debate/llm.py (_make_llm — Phase 1)
    - debate/graph.py (synthesize_stub node registration — unchanged)
  provides:
    - debate/state.py: DisputedPoint, DebateReport Pydantic models; DebateState.final_report typed
    - debate/nodes/synthesize.py: real synthesizer (LLM + confidence formula + DebateReport assembly)
  affects:
    - Phase 4 persistence (reads state["final_report"] as DebateReport)
    - Phase 5 UI (renders DebateReport fields)
tech_stack:
  added: []
  patterns:
    - "SynthesizerOutput: LLM-only schema, confidence_score absent (SYNTH-03)"
    - "_compute_confidence_score: pure Python formula using max(r.divergence_score for r in round_history)"
    - "_determine_convergence_status: imports DIVERGE_THRESHOLD from debate.divergence"
    - "_invoke_synthesizer: include_raw=True retry wrapper, fallback on exhaustion"
key_files:
  created: []
  modified:
    - debate/state.py
    - debate/nodes/synthesize.py
decisions:
  - "confidence_score computed in Python only — never from LLM (SYNTH-03 invariant)"
  - "SynthesizerOutput has exactly 3 fields: consensus_points, disputed_points, verdict"
  - "max_divergence reads from max(r.divergence_score for r in round_history), not state['divergence_score'] (last-write-wins would miss early high-divergence rounds)"
  - "Node name stays 'synthesize_stub' — graph.py requires no changes"
  - "DIVERGE_THRESHOLD (0.75) imported from debate.divergence — stub's 0.25 was a bug"
  - "Sentinel arguments (is_sentinel=True) filtered from synthesis context"
metrics:
  duration_minutes: 3
  completed_date: "2026-04-23"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
---

# Phase 03 Plan 01: Synthesizer & DebateReport Summary

**One-liner:** Real synthesizer with LLM-structured-output for consensus/verdict, Python-computed confidence formula `(1 - max_divergence) * round_adjustment`, and full DebateReport Pydantic model assembled in synthesize_stub.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add DisputedPoint and DebateReport models to debate/state.py | 7441d3c | debate/state.py |
| 2 | Replace synthesize_stub with real Synthesizer implementation | 78cdc42 | debate/nodes/synthesize.py |

## What Was Built

### Task 1: debate/state.py — New Models

Added two Pydantic models after `RoundRecord` and before `DebateState`:

**DisputedPoint** (2 fields):
- `topic: str` — the contested claim
- `agent_positions: dict[str, str]` — per-agent stance map

**DebateReport** (10 fields):
- `debate_id`, `topic` — debate identity
- `consensus_points: list[str]` — LLM-extracted agreement points
- `disputed_points: list[DisputedPoint]` — LLM-extracted disagreements
- `verdict: str` — LLM 2-4 sentence synthesis
- `confidence_score: float` — Python formula, never LLM-invented
- `convergence_status: Literal["converged", "max_rounds", "partial"]` — Python-determined
- `reasoning_trace: list[RoundRecord]` — full round history
- `concession_log: list[Concession]` — all concessions, flattened
- `created_at: datetime` — UTC assembly timestamp

Also updated `DebateState.final_report` from `Optional[object]` to `Optional["DebateReport"]`.

### Task 2: debate/nodes/synthesize.py — Real Implementation

Replaced the 34-line Phase 2 stub with a 287-line real synthesizer:

- **SynthesizerOutput**: LLM output schema with 3 fields only. `confidence_score` is intentionally absent (SYNTH-03).
- **_compute_confidence_score**: Formula `(1 - max_divergence) * round_adjustment`. Uses `max(r.divergence_score for r in round_history)` — not `state["divergence_score"]` which is last-write-wins.
- **_determine_convergence_status**: Uses imported `DIVERGE_THRESHOLD` (0.75). Fixed stub's hardcoded 0.25 bug.
- **_build_synthesis_context**: Compact prompt builder — position + key_claims (first 3) + concessions only. Skips sentinel arguments. Injects non-convergence instruction when `convergence_status == "max_rounds"`.
- **_invoke_synthesizer**: Retry wrapper with `include_raw=True`, fallback SynthesizerOutput on exhaustion.
- **synthesize_stub**: Full 6-step orchestrator. Returns `{"final_report": DebateReport, "status": convergence_status}`.

## Verification Results

```
# Task 1
from debate.state import DisputedPoint, DebateReport, RoundRecord, Concession
# → imports ok
# DebateReport fields: ['debate_id', 'topic', 'consensus_points', 'disputed_points',
#   'verdict', 'confidence_score', 'convergence_status', 'reasoning_trace',
#   'concession_log', 'created_at']

# Task 2 unit checks
_compute_confidence_score([RoundRecord(divergence_score=0.3)], round_num=2) == 0.63  # PASS
_determine_convergence_status(0.3, 1, 3) == 'converged'  # PASS
_determine_convergence_status(0.9, 3, 3) == 'max_rounds'  # PASS

# Smoke test
graph.invoke({'topic': 'Is remote work net positive?', 'max_rounds': 1}, ...)
# PASS: DebateReport ok, confidence=0.7793, convergence=converged
# consensus_points: 3, disputed_points: 3
```

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None. The synthesizer is fully implemented. `state["final_report"]` is a populated `DebateReport` instance after `graph.invoke()`.

## Self-Check: PASSED

Files exist:
- debate/state.py — FOUND (contains DisputedPoint, DebateReport, Optional["DebateReport"])
- debate/nodes/synthesize.py — FOUND (contains SynthesizerOutput, _compute_confidence_score, synthesize_stub)

Commits exist:
- 7441d3c — feat(03-01): add DisputedPoint and DebateReport models to debate/state.py
- 78cdc42 — feat(03-01): replace synthesize_stub with real Synthesizer implementation
