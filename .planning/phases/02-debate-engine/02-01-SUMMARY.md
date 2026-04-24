---
phase: 02-debate-engine
plan: 01
subsystem: testing
tags: [sentence-transformers, embeddings, cosine-similarity, pydantic, pytest]

# Dependency graph
requires:
  - phase: 01-graph-foundation
    provides: AgentArgument and RoundRecord Pydantic schemas in debate/state.py
provides:
  - compute_divergence() function in debate/divergence.py using BAAI/bge-small-en-v1.5
  - DIVERGE_THRESHOLD=0.75 and CONVERGE_FAST_PATH=0.97 constants
  - RoundRecord.divergence_score field for per-round divergence history
  - tests/test_phase2.py scaffold with 4 active unit tests + 4 skipped Plan 02/03 stubs
affects:
  - 02-02 (rebuttal loop wiring uses compute_divergence)
  - 02-03 (concession tests use test_phase2.py stubs)
  - 03-synthesis (SYNTH-03 confidence formula reads RoundRecord.divergence_score)

# Tech tracking
tech-stack:
  added:
    - sentence-transformers==5.4.1
    - BAAI/bge-small-en-v1.5 model (~130MB, cached in ~/.cache/huggingface/hub/)
  patterns:
    - Lazy singleton _get_model() for SentenceTransformer — avoids repeated model load
    - normalize_embeddings=True mandatory for BAAI/bge-small-en-v1.5 cosine similarity via dot product
    - Batch encode all claims in one model.encode() call before computing sim_matrix = emb_a @ emb_b.T
    - divergence_score = 1.0 - mean(pairwise max_similarities) formula
    - Per-round divergence_score stored in RoundRecord (not just DebateState) for Phase 3 history

key-files:
  created:
    - debate/divergence.py
    - tests/test_phase2.py
  modified:
    - requirements.txt
    - debate/state.py

key-decisions:
  - "BAAI/bge-small-en-v1.5 with normalize_embeddings=True: dot product on normalized vectors equals cosine similarity; scores guaranteed in [0,1]"
  - "Per-round divergence_score stored in RoundRecord (Option A from RESEARCH.md): co-locates score with the round it describes; Phase 3 SYNTH-03 formula can read max across rounds"
  - "Borderline zone 0.75–0.97 treated conservatively as diverged for Phase 2; Claude judge deferred to Phase 3 enhancement if false-positive rate warrants it"
  - "Plan verification command used 2-item key_claims lists — invalid per min_length=3 constraint on AgentArgument; test suite and actual implementation use valid 3+ claims"

patterns-established:
  - "Pattern: compute_divergence() is a pure function (no state, no side effects); test it in isolation without graph or LLM"
  - "Pattern: embed key_claims (NOT reasoning) — reasoning text collapses semantic distance; claims are discriminative"
  - "Pattern: _get_model() lazy singleton — call once per process, cached by HF Hub on disk"

requirements-completed:
  - DEBATE-04

# Metrics
duration: 42min
completed: 2026-04-24
---

# Phase 02 Plan 01: Divergence Detector Foundation Summary

**Pairwise cosine similarity divergence detector using BAAI/bge-small-en-v1.5 embeddings on key_claims, with per-round score storage in RoundRecord and full test_phase2.py scaffold**

## Performance

- **Duration:** 42 min
- **Started:** 2026-04-24T03:56:59Z
- **Completed:** 2026-04-24T04:38:41Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Installed sentence-transformers==5.4.1 and pre-downloaded BAAI/bge-small-en-v1.5 model (~130MB) to HF cache
- Created debate/divergence.py with compute_divergence(), _get_model() lazy singleton, DIVERGE_THRESHOLD=0.75, CONVERGE_FAST_PATH=0.97 — pure function that embeds key_claims and returns (float, list[tuple])
- Extended RoundRecord in debate/state.py with divergence_score: float = Field(default=0.0) for per-round history (RESEARCH.md Pitfall 6)
- Created tests/test_phase2.py: 4 active DEBATE-04 unit tests all passing, 4 skipped stubs for Plans 02/03

## Task Commits

Each task was committed atomically:

1. **Task 1: Install sentence-transformers and pre-download model** - `d786eb8` (chore)
2. **Task 2: Create debate/divergence.py and extend RoundRecord** - `26afbb0` (feat)

## Files Created/Modified

- `requirements.txt` - Added sentence-transformers==5.4.1
- `debate/divergence.py` - compute_divergence(), _get_model() singleton, DIVERGE_THRESHOLD, CONVERGE_FAST_PATH
- `debate/state.py` - RoundRecord extended with divergence_score: float = Field(default=0.0)
- `tests/test_phase2.py` - Phase 2 test suite: 4 active + 4 skipped stubs

## Decisions Made

- Used BAAI/bge-small-en-v1.5 with normalize_embeddings=True: ensures dot product == cosine similarity and scores in [0,1]
- Stored divergence_score per RoundRecord (Option A from RESEARCH.md open question 3): co-located with round data, ready for Phase 3 SYNTH-03 formula
- Borderline zone (0.75–0.97) treated as diverged conservatively; Claude judge deferred to Phase 3 if false-positive rate warrants
- plan verification command used only 2 key_claims (below min_length=3); noted as plan error, implementation and tests are correct

## Deviations from Plan

None — plan executed exactly as written. The plan's inline verification example used only 2 key_claims per AgentArgument which fails the Pydantic min_length=3 constraint, but this is a documentation error in the plan, not a code issue. The actual test suite and divergence.py use valid 3+ claim lists.

## Issues Encountered

- `pip install sentence-transformers==5.4.1` is a large install (~600MB including torch); background job required; verified with `python -c "import sentence_transformers; print(sentence_transformers.__version__)"` returning 5.4.1
- Plan verification snippet used 2-item key_claims lists — incompatible with AgentArgument.key_claims min_length=3 constraint; ran corrected verification with 3 claims each confirming score=0.175 and pairs=[('optimist', 'pessimist')]
- pytest not installed in environment; added via `pip install pytest`

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- compute_divergence() is ready for use by divergence_check_node in Plan 02-02
- RoundRecord.divergence_score field is ready for Plans 02-02 and 03
- test_phase2.py scaffold has stub slots for DEBATE-05/06/07 tests; Plans 02-02 and 02-03 should fill those stubs
- BAAI/bge-small-en-v1.5 is pre-cached at ~/.cache/huggingface/hub/; no download latency on first test run

---
*Phase: 02-debate-engine*
*Completed: 2026-04-24*
