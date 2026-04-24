# Roadmap: Multi-Agent Debate System

## Overview

Build a LangGraph-based multi-agent debate system from schema and graph skeleton through
divergence detection, synthesis, persistence, and a Streamlit UI — delivering a demo-ready
portfolio app that produces auditable consensus reports from adversarial agent reasoning.

## Phases

- [x] **Phase 1: Graph Foundation** - State schema, graph skeleton, agent nodes, and persona enforcement (completed 2026-04-24)
- [x] **Phase 2: Debate Engine** - Divergence detection, multi-round rebuttal loop, concession tracking (completed 2026-04-24)
- [ ] **Phase 3: Synthesis & Report** - Synthesizer verdict, confidence scoring, full reasoning trace
- [ ] **Phase 4: Persistence** - SQLite debate storage and replay by debate_id
- [ ] **Phase 5: Streamlit UI** - Live debate feed, final report display, demo-ready polish

## Phase Details

### Phase 1: Graph Foundation
**Goal**: A runnable LangGraph debate graph where three agents independently analyze a topic, each enforcing its cognitive bias and producing validated structured output
**Depends on**: Nothing (first phase)
**Requirements**: DEBATE-01, DEBATE-02, DEBATE-03, AGENT-01, AGENT-02, AGENT-03
**Success Criteria** (what must be TRUE):
  1. Developer can invoke the graph with a topic string and receive three independent AgentArgument objects (Optimist, Pessimist, Devil's Advocate) containing position, reasoning, key_claims, confidence, and concessions
  2. Each agent's system prompt enforces its cognitive bias via methodology instructions, not personality adjectives — inspectable in source
  3. Pydantic validation failure triggers up to 2 retries; a sentinel AgentArgument is injected on the third failure without crashing the graph
  4. Anti-sycophancy instructions are present and verifiable in the agent prompt templates
**Plans**: 3 plans

Plans:
- [x] 01-PLAN.md — Project setup, DebateState TypedDict, AgentArgument/Concession/RoundRecord Pydantic models
- [x] 02-PLAN.md — _make_llm() helper, methodology-based persona prompts, all six graph node implementations
- [x] 03-PLAN.md — StateGraph wiring with Send fan-out, smoke test confirming 3 AgentArguments returned

### Phase 2: Debate Engine
**Goal**: A functioning multi-round debate loop where agents rebut each other based on semantically detected divergence and can concede points with traceable attribution
**Depends on**: Phase 1
**Requirements**: DEBATE-04, DEBATE-05, DEBATE-06, DEBATE-07
**Success Criteria** (what must be TRUE):
  1. Given two agent outputs, the divergence detector returns a numeric score based on semantic similarity of key_claims — not raw text comparison
  2. When divergence score exceeds the threshold, the rebuttal loop fires and agents receive compact summaries of opposing arguments in the next round
  3. The loop terminates automatically when divergence drops below threshold or after 3 rounds, whichever comes first
  4. When an agent concedes a point, the concession record names the source agent whose argument triggered it and includes a one-line reason
**Plans**: 3 plans

Plans:
- [x] 02-01-PLAN.md — Install sentence-transformers, create debate/divergence.py (compute_divergence), extend RoundRecord with divergence_score, create tests/test_phase2.py scaffold
- [x] 02-02-PLAN.md — divergence_check_node, synthesize_stub, route_divergence + compact summaries, rebuttal context in agents, full Phase 2 graph loop wiring
- [x] 02-03-PLAN.md — Live integration tests: full graph termination, round_history integrity, concession field validation, recursion limit verification

### Phase 3: Synthesis & Report
**Goal**: A Synthesizer agent that consumes the completed debate state and produces a final report with formula-derived confidence score, explicit consensus/disputed split, and full reasoning trace
**Depends on**: Phase 2
**Requirements**: SYNTH-01, SYNTH-02, SYNTH-03, SYNTH-04, SYNTH-05
**Success Criteria** (what must be TRUE):
  1. After debate completes, the Synthesizer produces a DebateReport containing consensus_points, disputed_points, verdict, and confidence_score
  2. Confidence score is computed as `(1 - max_divergence_score) * round_adjustment` — the formula is in code, not prompted from the LLM
  3. If the debate did not converge, the report's verdict section explicitly states non-convergence rather than fabricating a consensus
  4. The full reasoning trace (all rounds, all arguments, all concessions with attribution) is accessible on the DebateReport object
**Plans**: 2 plans

Plans:
- [ ] 03-01-PLAN.md — DebateReport + DisputedPoint models in state.py; full synthesize_stub replacement with LLM call, confidence formula, DebateReport assembly
- [ ] 03-02-PLAN.md — tests/test_phase3.py: confidence formula unit tests, non-convergence path test, full graph integration tests

### Phase 4: Persistence
**Goal**: Completed debates are saved to SQLite and can be reloaded and displayed by debate_id
**Depends on**: Phase 3
**Requirements**: STORE-01, STORE-02
**Success Criteria** (what must be TRUE):
  1. After a debate completes, a row is written to SQLite containing debate_id, topic, timestamp, and the full DebateReport serialized as JSON — verifiable with a direct DB query
  2. Given a valid debate_id, the system loads the stored JSON and reconstructs the DebateReport without re-running any agents
**Plans**: TBD

### Phase 5: Streamlit UI
**Goal**: A demo-ready Streamlit app where a user enters a topic, watches agents debate round by round, and reads the final structured report — with no broken states on a fresh run
**Depends on**: Phase 4
**Requirements**: UI-01, UI-02, UI-03, UI-04
**Success Criteria** (what must be TRUE):
  1. User can type a topic into the input field, click "Start Debate", and see per-round agent output appear progressively via graph.stream
  2. Final report section renders consensus points and disputed points in separate visual blocks, with confidence score prominently displayed
  3. Full reasoning trace is accessible via an expandable section without cluttering the main report view
  4. A fresh browser session with no prior state completes a full debate end-to-end without errors or blank screens
**Plans**: TBD
**UI hint**: yes

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Graph Foundation | 3/3 | Complete    | 2026-04-24 |
| 2. Debate Engine | 0/3 | Complete    | 2026-04-24 |
| 3. Synthesis & Report | 0/2 | Not started | - |
| 4. Persistence | 0/TBD | Not started | - |
| 5. Streamlit UI | 0/TBD | Not started | - |
