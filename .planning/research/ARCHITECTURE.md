# Architecture Patterns: Multi-Agent Debate System

**Domain:** Multi-agent LLM orchestration with structured debate loop
**Researched:** 2026-04-23
**Overall confidence:** HIGH (LangGraph patterns verified against existing working code in this repo; streaming verified against official LangGraph docs)

---

## Recommended Architecture

The system is a **single StateGraph** with a **fan-out/fan-in debate loop**. There is no subgraph nesting — it is not needed and adds indirection for a 4-agent system. All coordination state lives in one `DebateState` TypedDict.

```
START
  │
  ▼
[initialize]          ← parse topic, stamp metadata, set round_num=0
  │
  ▼
[dispatch_round1] ──── Send("optimist_node", ...) ──►  [optimist_node]
                  ──── Send("pessimist_node", ...) ──► [pessimist_node]
                  ──── Send("devil_node", ...) ──────► [devil_node]
                       (three parallel superstep)         │
                                                         ▼
                                               [collect_round1]  ← fan-in via add reducer
                                                         │
                                                         ▼
                                               [divergence_check]
                                                         │
                                          ┌──────────────┴──────────────┐
                                    converged                      diverged
                                          │                             │
                                          ▼                             ▼
                                   [synthesize]              [dispatch_rebuttal]
                                          │                  ── Send("optimist_node", ...)
                                          ▼                  ── Send("pessimist_node", ...)
                                       [save]                ── Send("devil_node", ...)
                                          │                             │
                                         END                           ▼
                                                             [collect_rebuttal]
                                                                       │
                                                                       ▼
                                                             [divergence_check]  ← same node, loops
                                                             (max_rounds guard → synthesize)
```

The loop between `divergence_check → dispatch_rebuttal → collect_rebuttal → divergence_check` is an explicit LangGraph cycle, terminated by a max-rounds guard inside `divergence_check`.

---

## Component Boundaries

| Component | File | Responsibility | Communicates With |
|-----------|------|---------------|-------------------|
| `DebateState` | `debate/state.py` | Single source of truth for all graph state | All nodes read/write this |
| `graph.py` | `debate/graph.py` | Graph topology: add_node, add_edge, compile | Imports all nodes |
| `initialize_node` | `debate/nodes/initialize.py` | Parse topic, set metadata, zero round counter | → dispatch_round1 |
| `dispatch_round1` | `debate/nodes/dispatch.py` | Fan-out via Send to 3 agent nodes (no prior arguments visible to agents) | → [optimist, pessimist, devil] in parallel |
| `optimist_node` | `debate/nodes/agents.py` | Generate AgentArgument for current topic/round | → collect node |
| `pessimist_node` | `debate/nodes/agents.py` | Same, different system prompt | → collect node |
| `devil_node` | `debate/nodes/agents.py` | Same, actively attacks majority view | → collect node |
| `collect_round1` | `debate/nodes/collect.py` | Fan-in: accumulate agent arguments via list reducer | → divergence_check |
| `divergence_check` | `debate/nodes/divergence.py` | Embed arguments, compute pairwise cosine sim, decide continue/stop | → dispatch_rebuttal or synthesize |
| `dispatch_rebuttal` | `debate/nodes/dispatch.py` | Fan-out again, now with prior_arguments visible in agent state | → [optimist, pessimist, devil] in parallel |
| `collect_rebuttal` | `debate/nodes/collect.py` | Same as collect_round1, appends to round history | → divergence_check |
| `synthesize_node` | `debate/nodes/synthesize.py` | Synthesizer agent with full trace; produces final report | → save_node |
| `save_node` | `debate/nodes/save.py` | Persist DebateState to SQLite | → END |
| `DivergeDetector` | `debate/divergence.py` | Pure function: list[str] → pairwise similarity matrix → bool | Called by divergence_check node |
| `Streamlit app` | `app.py` | UI: topic input, live debate feed, final report | Calls graph.stream() |

---

## Data Flow

### Round 1 Independence (no cross-contamination)

The critical constraint is that Round 1 agents must not see each other's arguments. This is enforced by what `dispatch_round1` puts in each `Send` payload:

```python
# dispatch_round1 sends ONLY topic + agent identity — no prior_arguments
def dispatch_round1(state: DebateState):
    topic = state["topic"]
    return [
        Send("optimist_node",  {"topic": topic, "agent_role": "optimist",  "prior_arguments": [], "round_num": 0}),
        Send("pessimist_node", {"topic": topic, "agent_role": "pessimist", "prior_arguments": [], "round_num": 0}),
        Send("devil_node",     {"topic": topic, "agent_role": "devil",     "prior_arguments": [], "round_num": 0}),
    ]
```

Each `Send` carries its own isolated sub-state. The agents run in separate supersteps with no shared memory. The `prior_arguments` field is explicitly empty — agents cannot read each other during Round 1.

### Round 2+ Rebuttals (cross-visibility enabled)

`dispatch_rebuttal` sends prior rounds' arguments to each agent so they can rebut:

```python
def dispatch_rebuttal(state: DebateState):
    topic = state["topic"]
    round_num = state["round_num"]
    # All previous round arguments visible to all agents now
    all_prior = state["round_history"]   # list of RoundRecord
    return [
        Send("optimist_node",  {"topic": topic, "agent_role": "optimist",  "prior_arguments": all_prior, "round_num": round_num}),
        ...
    ]
```

### Fan-in: List Reducer on `current_round_arguments`

Parallel nodes write to the same state key. The `add` reducer accumulates them:

```python
from typing import Annotated
from operator import add

class DebateState(TypedDict):
    current_round_arguments: Annotated[list[AgentArgument], add]
    # Each parallel agent appends its AgentArgument
    # LangGraph merges them automatically via the add reducer
```

After all three parallel agents complete their superstep, `current_round_arguments` holds all three results.

### Collect node

`collect_round1` / `collect_rebuttal` is a simple reducer node that moves `current_round_arguments` into `round_history` and resets the accumulator:

```python
def collect_round(state: DebateState) -> dict:
    new_record = RoundRecord(
        round_num=state["round_num"],
        arguments=state["current_round_arguments"],
    )
    return {
        "round_history": state["round_history"] + [new_record],
        "current_round_arguments": [],   # reset for next fan-in
        "round_num": state["round_num"] + 1,
    }
```

---

## Data Models

### AgentArgument (Pydantic)

```python
class AgentArgument(BaseModel):
    agent_role: str          # "optimist" | "pessimist" | "devil"
    round_num: int
    position: str            # main thesis in one sentence
    reasoning: str           # full argument text
    confidence: float        # 0.0–1.0, self-reported
    concessions: list[Concession]   # points yielded this round
    key_claims: list[str]    # extractable claims for embedding
```

### Concession (Pydantic)

```python
class Concession(BaseModel):
    conceded_point: str           # what the agent gave up
    triggered_by_agent: str       # who made the argument that caused this
    triggered_by_claim: str       # the specific claim text
    rationale: str                # why the agent is conceding
```

The concession mechanism is modeled as structured output from the agent itself. The agent's system prompt instructs it to populate `concessions` when it genuinely yields a point. This is auditable: the reasoning trace shows `triggered_by_agent` + `triggered_by_claim` for every concession. Agents are not forced to concede — the LLM decides based on argument quality.

### RoundRecord

```python
class RoundRecord(BaseModel):
    round_num: int
    arguments: list[AgentArgument]
```

### DebateReport (final output)

```python
class DebateReport(BaseModel):
    topic: str
    consensus_points: list[str]      # claims all agents agreed on (or conceded to)
    disputed_points: list[str]       # claims with remaining divergence
    confidence_score: float          # 0.0–1.0 overall confidence
    verdict: str                     # synthesizer's final answer
    reasoning_trace: list[RoundRecord]   # full debate history
    concession_log: list[Concession]     # all concessions from all agents
    created_at: str
```

### DebateState (LangGraph TypedDict)

```python
class DebateState(TypedDict):
    # Input
    topic: str
    debate_id: str

    # Round tracking
    round_num: int
    max_rounds: int          # default 3

    # Fan-in accumulator (reset each round)
    current_round_arguments: Annotated[list[AgentArgument], add]

    # Full history (append-only)
    round_history: list[RoundRecord]

    # Divergence signal
    divergence_score: float
    diverged_pairs: list[tuple[str, str]]   # which agent pairs diverge

    # Terminal state
    final_report: DebateReport | None
    status: str   # "running" | "converged" | "max_rounds" | "complete"
```

**Design note:** `current_round_arguments` uses the `add` reducer so parallel agent nodes can append independently. All other keys use default (last-write-wins) semantics. This is the only state key that needs a reducer.

---

## Divergence Detection

**Where it lives:** `debate/divergence.py` — a pure Python module, not a LangGraph node. The `divergence_check` node calls it.

**Algorithm:**

```
1. For each AgentArgument in current_round_arguments:
   - Extract key_claims (list of strings)
   - Encode all claims with sentence-transformers (all-MiniLM-L6-v2)
2. For each pair of agents (optimist/pessimist, optimist/devil, pessimist/devil):
   - Compute max cosine similarity between claim pairs
   - If max_similarity < DIVERGE_THRESHOLD (0.75): agents are diverged on at least one point
3. divergence_score = 1 - mean(max_similarities across all pairs)
4. Return: is_diverged (bool), divergence_score (float), diverged_pairs (list)
```

**Threshold rationale (MEDIUM confidence):** 0.75 cosine similarity on all-MiniLM-L6-v2 is a reasonable starting point for "these claims are making different points." Values below 0.6 are clearly unrelated; values above 0.85 are near-paraphrases. 0.75 is the practical boundary. This will need empirical tuning after Phase 1.

**Max rounds guard:** Inside `divergence_check`, if `state["round_num"] >= state["max_rounds"]`, route to `synthesize` regardless of divergence score. This prevents infinite loops.

**Conditional routing from `divergence_check`:**

```python
def route_after_divergence(state: DebateState) -> str:
    if state["round_num"] >= state["max_rounds"]:
        return "synthesize"
    if state["divergence_score"] > 0.25:   # 25% = noticeably diverged
        return "dispatch_rebuttal"
    return "synthesize"
```

---

## Debate State Machine

```
INIT
  │
  ▼
ROUND_1_DISPATCH → (parallel) AGENT_ROUND_1 × 3 → COLLECT → DIVERGENCE_CHECK
                                                                    │
                                      ┌────────────────────────────┤
                                  diverged                    converged
                                      │                        (or max rounds)
                                      ▼                             │
                             REBUTTAL_DISPATCH                      ▼
                           → (parallel) AGENT × 3         SYNTHESIZE → SAVE → END
                           → COLLECT → DIVERGENCE_CHECK
                                           ↑
                                     (loop back)
```

State transitions are explicit in the graph topology. There is no hidden state machine logic — the `round_num` counter in `DebateState` is the single source of truth for "where are we."

---

## Streamlit Connection

**Use `graph.stream()` with `stream_mode="updates"`, not polling, not callbacks.**

Rationale: `stream_mode="updates"` delivers one chunk per node completion, with the node name and state delta. This is exactly what the UI needs: "Agent X finished Round Y." No polling loop needed. No async required for the demo.

```python
# app.py pattern
def run_debate(topic: str):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    inputs = {"topic": topic, "max_rounds": 3}

    for chunk in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, state_delta in chunk.items():
            # Dispatch to UI renderers by node name
            if node_name == "collect_round1":
                render_round_complete(state_delta, st.empty())
            elif node_name == "divergence_check":
                render_divergence_signal(state_delta, st.sidebar)
            elif node_name == "synthesize":
                render_final_report(state_delta)
```

Each iteration of the `for` loop is one node completing. The UI updates incrementally. The Streamlit session does not need to `invoke` and then poll — `stream()` blocks until the next chunk arrives, which Streamlit handles in a single synchronous script run via `st.empty()` + `st.write()` inside the loop.

**Thread ID lifecycle:** Generate one UUID per debate session. Store in `st.session_state.debate_thread_id`. The SQLite checkpointer uses this to persist and replay.

**For UI progress display:** Use `st.status()` (Streamlit >= 1.28) as a collapsible container for each agent's round output. Update it as chunks arrive.

---

## Suggested Build Order (Phase Implications)

Build order is dependency-driven. Each phase must be fully working before the next.

### Phase 1 — State + Graph Skeleton (no LLM calls)
Build `DebateState`, `graph.py` topology, stub nodes that return hardcoded data. Verify fan-out/fan-in works (Send API + add reducer). Verify the loop terminates.

**Why first:** Every other component depends on a working graph. Wiring bugs are much harder to debug once LLM calls are involved.

**Deliverable:** `graph.stream({"topic": "test", "max_rounds": 2})` completes without error, all nodes called in correct order, round_history populated.

### Phase 2 — Agent Nodes (LLM calls, structured output)
Implement `AgentArgument` Pydantic model. Implement `optimist_node`, `pessimist_node`, `devil_node` with real system prompts and `llm.with_structured_output(AgentArgument)`. Verify Round 1 isolation (no prior arguments leak). Verify Pydantic parsing succeeds.

**Why second:** Agents are the core value unit. Everything downstream (divergence, synthesis) depends on real agent output.

**Deliverable:** Single run produces 3 real `AgentArgument` objects with populated `key_claims`.

### Phase 3 — Divergence Detection
Implement `DivergeDetector` with sentence-transformers. Implement `divergence_check` node. Wire conditional routing. Tune threshold empirically on 3-5 test topics.

**Why third:** Can only tune against real agent outputs from Phase 2. Don't build divergence logic against stub data.

**Deliverable:** Different topics route correctly (contentious → loop, clear consensus → synthesize).

### Phase 4 — Synthesizer + Final Report
Implement `synthesize_node` with `DebateReport` structured output. Implement `save_node` with SQLite persistence.

**Why fourth:** Synthesizer reads full `round_history` — only useful once debates run real rounds.

**Deliverable:** `DebateReport` is written to SQLite and is readable.

### Phase 5 — Streamlit UI
Wire `graph.stream()` into `app.py`. Build live debate feed, round cards, final report view.

**Why last:** UI is a wrapper over working logic. Building UI early creates pressure to fake data.

**Deliverable:** Demo-ready app showing live agent outputs, divergence signal, final report.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Shared Message History Across Round 1 Agents
**What goes wrong:** Using a single `messages: Annotated[list[BaseMessage], add_messages]` in DebateState and letting all agents read/append to it. Round 1 agents will see each other's outputs if the list is not isolated per-agent.

**Prevention:** `current_round_arguments` is the accumulator, not `messages`. Agent nodes receive the topic string, not the full state. The `Send` payload is the isolation boundary.

### Anti-Pattern 2: Putting Divergence Logic in the Agent Prompt
**What goes wrong:** Asking the LLM to "check if you agree with the other agents." This is unreliable — LLMs tend toward agreement when asked to compare. It also makes divergence non-deterministic and non-auditable.

**Prevention:** Divergence detection is a deterministic function over embeddings. The LLM's job is to make arguments; the graph's job is to measure divergence.

### Anti-Pattern 3: Subgraph Nesting for Agents
**What goes wrong:** Creating a separate sub-StateGraph for each agent, then composing them. For 4 agents with simple node logic, this adds complexity without benefit. Subgraph state merging introduces bugs.

**Prevention:** All 4 agents are nodes in the same flat graph. Use Send for parallelism. Only introduce subgraphs if an agent itself becomes a multi-step workflow (not needed in v1).

### Anti-Pattern 4: Running Graph in Streamlit Thread Without session_state
**What goes wrong:** Calling `graph.invoke()` directly inside a Streamlit button handler. Streamlit reruns the entire script on each interaction — the graph object is recreated, `thread_id` is lost, and the debate restarts.

**Prevention:** Store the compiled graph and `thread_id` in `st.session_state` at module level. Use `graph.stream()` inside a `with st.spinner()` block. Initialize `session_state` keys once with a guard: `if "graph" not in st.session_state`.

### Anti-Pattern 5: Embedding Entire Arguments Instead of Key Claims
**What goes wrong:** Embedding the full `reasoning` text (500-1000 tokens) for cosine similarity. Long texts tend toward semantic middling — everything looks similar because all arguments are about the same topic. Divergence signal collapses.

**Prevention:** Agents output `key_claims: list[str]` (5-10 short claims each). Embed the claims, not the full reasoning. Compare agent positions at claim granularity.

---

## Sources

- LangGraph graph-api documentation (`https://docs.langchain.com/oss/python/langgraph/graph-api`): StateGraph, Send API, add reducer, conditional edges — HIGH confidence (official docs, verified live)
- LangGraph streaming documentation (`https://docs.langchain.com/oss/python/langgraph/streaming`): stream_mode options, updates format, node_name in chunks — HIGH confidence (official docs, verified live)
- LangGraph interrupt documentation (`https://docs.langchain.com/oss/python/langgraph/interrupts`): interrupt/Command pattern, Streamlit session_state integration — HIGH confidence (official docs)
- Existing codebase (`桌面动物园/agent/graph.py`, `桌面动物园/agent/state.py`): confirmed working LangGraph patterns for TypedDict state, conditional edges, loop routing — HIGH confidence (working production code in this repo)
- Existing codebase (`.claude/worktrees/agent-a38a74af/workdiary_agent/graph.py`): InMemorySaver pattern, START/END imports, add_node/add_edge/add_conditional_edges usage — HIGH confidence (working production code)
- sentence-transformers all-MiniLM-L6-v2: semantic similarity threshold (0.75) — MEDIUM confidence (training data; needs empirical validation on actual debate outputs)
