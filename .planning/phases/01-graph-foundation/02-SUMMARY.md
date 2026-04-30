---
phase: 01-graph-foundation
plan: "02"
subsystem: api
tags: [langgraph, langchain-anthropic, pydantic, agent-nodes, persona-prompts, fan-out, fan-in, retry-wrapper, sentinel]

# Dependency graph
requires:
  - phase: 01-graph-foundation plan 01
    provides: DebateState TypedDict, AgentArgument/Concession/RoundRecord Pydantic models in debate/state.py
provides:
  - debate/llm.py: _make_llm() factory with ANTHROPIC_CUSTOM_HEADERS proxy auth
  - debate/prompts.py: AGENT_PROMPTS dict with methodology-based persona prompts for optimist/pessimist/devil
  - debate/nodes/initialize.py: initialize_node stamps debate_id, sets all DebateState defaults
  - debate/nodes/dispatch.py: dispatch_round1 returns list[Send] for parallel fan-out
  - debate/nodes/agents.py: optimist_node, pessimist_node, devil_node with _invoke_with_retry and sentinel injection
  - debate/nodes/collect.py: collect_round1 moves accumulator to round_history, resets accumulator
affects: [01-graph-foundation plan 03 (graph wiring + smoke test), Phase 2 divergence detection, Phase 3 synthesizer]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_make_llm() factory: ANTHROPIC_CUSTOM_HEADERS parsed as newline-separated Key: Value pairs, passed as default_headers to ChatAnthropic"
    - "Methodology-based persona prompts: analytical framework + reference persona + PROHIBITION block + terminal claim instruction"
    - "_invoke_with_retry with include_raw=True: surfaces ValidationError as result['parsed'] is None, retries 2x, injects sentinel on 3rd failure"
    - "Sentinel AgentArgument pattern: is_sentinel=True, confidence=0.0, copied via model_copy(deep=True), never mutates template"
    - "Agent nodes return {'current_round_arguments': [argument]} as single-item list to trigger add reducer fan-in"
    - "dispatch_round1 Send payloads contain only minimal fields — no full DebateState cross-contamination"

key-files:
  created:
    - debate/llm.py
    - debate/prompts.py
    - debate/nodes/initialize.py
    - debate/nodes/dispatch.py
    - debate/nodes/agents.py
    - debate/nodes/collect.py
  modified: []

key-decisions:
  - "Persona prompts are methodology-based (analytical framework + reference persona), not intensity-based adjectives — prevents sycophantic collapsed output"
  - "Each prompt has PROHIBITION block with explicit forbidden phrases and 'Do not concede to avoid conflict' anti-sycophancy instruction"
  - "_invoke_with_retry uses include_raw=True (never raises ValidationError) with 2-retry cap and sentinel injection on total failure"
  - "Sentinel template is a module-level constant; nodes call model_copy(deep=True) to avoid mutating the template"
  - "dispatch_round1 Send payloads contain only topic/agent_role/prior_arguments/round_num — no full DebateState leak"

patterns-established:
  - "Pattern: _make_llm() — use this factory in all agent nodes; never instantiate ChatAnthropic directly in nodes"
  - "Pattern: include_raw=True on with_structured_output — mandatory for all agent nodes to prevent unhandled ValidationError crashes"
  - "Pattern: return single-item list for fan-in — {'current_round_arguments': [argument]} not {'current_round_arguments': argument}"

requirements-completed: [AGENT-01, AGENT-02, AGENT-03]

# Metrics
duration: 8min
completed: 2026-04-24
---

# Phase 1 Plan 02: LLM Helper, Persona Prompts, and Agent Nodes Summary

**Methodology-based persona prompts with PROHIBITION blocks, proxy-aware _make_llm() factory, and 6-function node layer (initialize/dispatch/3 agents/collect) with include_raw=True retry wrapper and sentinel injection**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-24T03:19:00Z
- **Completed:** 2026-04-24T03:27:00Z
- **Tasks:** 2 of 2
- **Files modified:** 6

## Accomplishments

- Created `debate/llm.py` with `_make_llm()` proxy-aware factory reading ANTHROPIC_CUSTOM_HEADERS as newline-separated Key: Value pairs
- Created `debate/prompts.py` with three methodology-based persona prompts (optimist/pessimist/devil), each with an analytical framework, reference persona, PROHIBITION block with explicit forbidden phrases, and anti-sycophancy "Do not concede" instruction
- Created all four node modules: `initialize_node`, `dispatch_round1`, three agent nodes with retry wrapper + sentinel, and `collect_round1`

## Task Commits

Each task was committed atomically:

1. **Task 1: LLM Helper and Persona Prompts** - `5352b42` (feat)
2. **Task 2: Graph Nodes — Initialize, Dispatch, Agents, Collect** - `bfc9732` (feat)

## Files Created/Modified

- `debate/llm.py` - _make_llm() factory for corporate proxy auth with ANTHROPIC_CUSTOM_HEADERS parsing
- `debate/prompts.py` - AGENT_PROMPTS dict with methodology-based prompts for all 3 roles
- `debate/nodes/initialize.py` - initialize_node stamps debate_id, sets all DebateState defaults
- `debate/nodes/dispatch.py` - dispatch_round1 returns list[Send] for parallel fan-out, minimal payloads
- `debate/nodes/agents.py` - optimist/pessimist/devil_node via shared _agent_node(); _invoke_with_retry with include_raw=True; _SENTINEL_TEMPLATE with is_sentinel=True
- `debate/nodes/collect.py` - collect_round1 fan-in: moves accumulator to round_history, resets to [], increments round_num

## Decisions Made

- Persona prompts are methodology-based rather than intensity-based — prevents sycophantic collapsed output where all agents produce "pros and cons" hedged responses
- `include_raw=True` is mandatory from day one to surface ValidationError as `result["parsed"] is None` rather than raising
- Sentinel template is a module-level constant; `model_copy(deep=True)` is used to avoid mutating the template on each injection
- `dispatch_round1` Send payloads intentionally contain only the 4 minimal fields to prevent Round 1 cross-contamination (PITFALL 4)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

None — no data is wired to UI. This plan delivers the node layer only. The graph assembly (Plan 03) wires these nodes together and performs the first live LLM call via smoke test.

## Next Phase Readiness

- All 6 node functions are importable and verified structurally (no LLM call needed)
- Plan 03 (Graph Assembly + Smoke Test) can now import from all these modules to build `debate/graph.py` and run the first real end-to-end LLM call
- No blockers

## Self-Check: PASSED

Files created:
- FOUND: debate/llm.py
- FOUND: debate/prompts.py
- FOUND: debate/nodes/initialize.py
- FOUND: debate/nodes/dispatch.py
- FOUND: debate/nodes/agents.py
- FOUND: debate/nodes/collect.py

Commits verified:
- FOUND: 5352b42 (Task 1 - LLM helper and persona prompts)
- FOUND: bfc9732 (Task 2 - graph node functions)

---
*Phase: 01-graph-foundation*
*Completed: 2026-04-24*
