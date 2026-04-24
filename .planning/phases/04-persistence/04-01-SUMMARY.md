---
phase: 04-persistence
plan: 01
subsystem: database
tags: [sqlite, sqlite3, pydantic, langgraph, persistence]

# Dependency graph
requires:
  - phase: 03-synthesis-report
    provides: DebateReport Pydantic model with model_dump_json/model_validate_json round-trip
provides:
  - debate/store.py: SQLite persistence layer (get_connection, save_debate, load_debate, list_debates)
  - debate/nodes/save.py: save_node graph node for post-synthesis persistence
  - Updated graph topology: synthesize_stub -> save_node -> END
affects: [05-ui, streamlit-ui, debate-replay]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Singleton connection per resolved db_path keyed in module-level dict"
    - "INSERT OR REPLACE for upsert semantics on debate_id primary key"
    - "save_node returns {} (no state mutation) — pure side-effect node pattern"

key-files:
  created:
    - debate/store.py
    - debate/nodes/save.py
  modified:
    - debate/graph.py

key-decisions:
  - "save_node returns {} and does not write to DebateState — persistence is a side effect, not state mutation"
  - "get_connection() uses module-level singleton keyed by resolved absolute path, safe for single-threaded graph runs"
  - "DB_PATH defaults to project root (debates.db) via Path(__file__).parent.parent"
  - "save_debate stores convergence_status as the status column (not a separate enum)"

patterns-established:
  - "Side-effect graph nodes: return {} to signal no state change, perform I/O inside"
  - "SQLite singleton pattern: _connections dict keyed by str(Path.resolve()) avoids redundant connections"

requirements-completed: [STORE-01, STORE-02]

# Metrics
duration: 2min
completed: 2026-04-24
---

# Phase 4 Plan 01: Persistence Summary

**SQLite persistence for debate results via debate/store.py module and save_node graph node wired between synthesize_stub and END**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-24T08:10:53Z
- **Completed:** 2026-04-24T08:12:48Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Created `debate/store.py` with full SQLite persistence API: get_connection (singleton), save_debate (upsert), load_debate (by id), list_debates (summary rows, newest-first)
- Created `debate/nodes/save.py` with save_node that reads final_report from state and persists to SQLite without mutating state
- Updated `debate/graph.py` topology: replaced `synthesize_stub -> END` with `synthesize_stub -> save_node -> END`

## Task Commits

Each task was committed atomically:

1. **Task 1: Create debate/store.py — SQLite persistence module** - `e1d9ea7` (feat)
2. **Task 2: Create debate/nodes/save.py and update debate/graph.py** - `7d214fc` (feat)

**Plan metadata:** (see final commit below)

## Files Created/Modified

- `debate/store.py` - SQLite persistence layer; get_connection/save_debate/load_debate/list_debates
- `debate/nodes/save.py` - save_node graph node; persists state['final_report'], returns {}
- `debate/graph.py` - Added save_node import, registered node, wired synthesize_stub -> save_node -> END

## Decisions Made

- save_node returns {} (no state mutation): persistence is a pure side-effect; DebateState needs no new fields
- get_connection() singleton keyed by resolved absolute path: safe for single-threaded LangGraph runs, avoids re-opening DB on each node execution
- save_debate uses INSERT OR REPLACE so re-running graph with same debate_id updates rather than fails
- No new packages added: stdlib sqlite3 is sufficient; no requirements.txt change needed

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. debates.db is created automatically at project root on first graph.invoke() call.

## Next Phase Readiness

- Phase 5 UI can call `load_debate(debate_id)` to replay any completed debate without re-running agents
- `list_debates()` provides summary rows for a "debate history" UI component
- debates.db will be written to project root; Phase 5 may want to expose its path via config/env var if needed

---
*Phase: 04-persistence*
*Completed: 2026-04-24*

## Self-Check: PASSED

- debate/store.py: FOUND
- debate/nodes/save.py: FOUND
- .planning/phases/04-persistence/04-01-SUMMARY.md: FOUND
- commit e1d9ea7: FOUND
- commit 7d214fc: FOUND
