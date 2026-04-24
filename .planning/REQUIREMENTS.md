# Requirements — Multi-Agent Debate System

## v1 Requirements

### Core Debate Flow

- [x] **DEBATE-01**: User can enter any topic/question and trigger a multi-agent debate
- [x] **DEBATE-02**: 3 analysis agents (Optimist, Pessimist, Devil's Advocate) analyze the topic independently in Round 1 with no cross-visibility
- [x] **DEBATE-03**: Each agent produces structured output: position, reasoning, key_claims, confidence score, concessions list
- [x] **DEBATE-04**: Debate engine detects real divergence between agents using semantic similarity on key_claims (not full text)
- [ ] **DEBATE-05**: Multi-round rebuttal loop fires when divergence is detected; agents see compact summaries of opposing arguments
- [ ] **DEBATE-06**: Debate loop terminates on convergence (divergence below threshold) or max 3 rounds
- [ ] **DEBATE-07**: Agents can concede points with structured attribution: which agent's argument triggered the concession and why

### Agent Quality

- [x] **AGENT-01**: Each agent has a structural persona prompt that enforces its cognitive bias via methodology (not just "be pessimistic")
- [x] **AGENT-02**: Anti-sycophancy instructions prevent agents from conceding to avoid conflict rather than on logical grounds
- [x] **AGENT-03**: Pydantic validation errors are handled with 2-retry wrapper; sentinel AgentArgument injected on third failure

### Synthesis & Output

- [ ] **SYNTH-01**: Synthesizer agent produces a final verdict after debate completes
- [ ] **SYNTH-02**: Final report contains: consensus points, disputed points, confidence score (formula-derived), verdict
- [ ] **SYNTH-03**: Confidence score is formula-derived: `(1 - max_divergence_score) * round_adjustment` — never LLM-invented
- [ ] **SYNTH-04**: Synthesizer has honest-uncertainty path: if debate did not converge, report says so explicitly
- [ ] **SYNTH-05**: Full reasoning trace stored: all rounds, all arguments, all concessions with attribution

### Persistence

- [ ] **STORE-01**: Completed debates saved to SQLite with debate_id, topic, timestamp, full DebateReport JSON
- [ ] **STORE-02**: Debates are replayable by debate_id (load from SQLite and display)

### UI

- [ ] **UI-01**: Streamlit app with topic input field and "Start Debate" button
- [ ] **UI-02**: Live debate progress shown as agents complete each round (streaming via graph.stream)
- [ ] **UI-03**: Final report displayed with consensus/disputed split, confidence score, and expandable reasoning trace
- [ ] **UI-04**: Demo-ready: clean layout, no broken states, works end-to-end on first try

## v2 Requirements (Deferred)

- Per-round divergence trend chart
- Confidence-modulated debate (agents condition updates on each other's confidence)
- Domain-specific mode (financial analysis with real data)
- Debate history browser (list past debates)
- Export report as markdown/PDF

## Out of Scope

- Authentication / user accounts — portfolio demo, single user
- Real-time financial data integration — domain-agnostic text input for v1
- Production deployment / scaling — demo-first
- More than 4 agents — keep explainable
- Voting / majority consensus — research shows it's inferior to synthesizer-with-trace
- Per-token streaming in UI — node-level granularity is sufficient and simpler

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DEBATE-01 | Phase 1 | Complete |
| DEBATE-02 | Phase 1 | Complete |
| DEBATE-03 | Phase 1 | Complete |
| AGENT-01 | Phase 1 | Complete |
| AGENT-02 | Phase 1 | Complete |
| AGENT-03 | Phase 1 | Complete |
| DEBATE-04 | Phase 2 | Complete |
| DEBATE-05 | Phase 2 | Pending |
| DEBATE-06 | Phase 2 | Pending |
| DEBATE-07 | Phase 2 | Pending |
| SYNTH-01 | Phase 3 | Pending |
| SYNTH-02 | Phase 3 | Pending |
| SYNTH-03 | Phase 3 | Pending |
| SYNTH-04 | Phase 3 | Pending |
| SYNTH-05 | Phase 3 | Pending |
| STORE-01 | Phase 4 | Pending |
| STORE-02 | Phase 4 | Pending |
| UI-01 | Phase 5 | Pending |
| UI-02 | Phase 5 | Pending |
| UI-03 | Phase 5 | Pending |
| UI-04 | Phase 5 | Pending |
