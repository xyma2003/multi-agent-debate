# Requirements — Multi-Agent Debate System

## v1 Requirements

### Core Debate Flow

- [ ] **DEBATE-01**: User can enter any topic/question and trigger a multi-agent debate
- [ ] **DEBATE-02**: 3 analysis agents (Optimist, Pessimist, Devil's Advocate) analyze the topic independently in Round 1 with no cross-visibility
- [ ] **DEBATE-03**: Each agent produces structured output: position, reasoning, key_claims, confidence score, concessions list
- [ ] **DEBATE-04**: Debate engine detects real divergence between agents using semantic similarity on key_claims (not full text)
- [ ] **DEBATE-05**: Multi-round rebuttal loop fires when divergence is detected; agents see compact summaries of opposing arguments
- [ ] **DEBATE-06**: Debate loop terminates on convergence (divergence below threshold) or max 3 rounds
- [ ] **DEBATE-07**: Agents can concede points with structured attribution: which agent's argument triggered the concession and why

### Agent Quality

- [ ] **AGENT-01**: Each agent has a structural persona prompt that enforces its cognitive bias via methodology (not just "be pessimistic")
- [ ] **AGENT-02**: Anti-sycophancy instructions prevent agents from conceding to avoid conflict rather than on logical grounds
- [ ] **AGENT-03**: Pydantic validation errors are handled with 2-retry wrapper; sentinel AgentArgument injected on third failure

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

| REQ-ID | Phase |
|--------|-------|
| DEBATE-01, DEBATE-02, DEBATE-03, AGENT-01, AGENT-02, AGENT-03 | Phase 1 |
| DEBATE-04, DEBATE-05, DEBATE-06, DEBATE-07 | Phase 2 |
| SYNTH-01, SYNTH-02, SYNTH-03, SYNTH-04, SYNTH-05 | Phase 3 |
| STORE-01, STORE-02 | Phase 4 |
| UI-01, UI-02, UI-03, UI-04 | Phase 5 |
