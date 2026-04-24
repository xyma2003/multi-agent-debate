# Technology Stack

**Project:** Multi-Agent Debate System
**Researched:** 2026-04-23
**Research mode:** Ecosystem / Verification

---

## Recommended Stack

### Core Orchestration

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| langgraph | 1.1.9 | Multi-agent state machine, graph execution | Latest stable. The 1.x series is a production-ready rewrite with clean `interrupt()`/`Command` API, `Send` for parallel fan-out, and first-class multi-agent support via `Command.PARENT` handoffs. |
| langgraph-checkpoint | 4.0.2 | Checkpoint base classes (`InMemorySaver`) | Auto-installed by langgraph 1.1.9. |
| langgraph-prebuilt | 1.0.10 | `ToolNode`, `create_react_agent` | Auto-installed. Not central to debate graph but available if an agent needs tool-calling. |
| langchain-core | 1.3.1 | Message types, Runnable protocol | Required by langgraph. Provides `HumanMessage`, `AIMessage`, `BaseMessage`, and the `add_messages` reducer used by `MessagesState`. |

**Confidence: HIGH** — verified against PyPI live JSON feeds.

---

### LLM Integration

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| langchain-anthropic | 1.4.1 | `ChatAnthropic` wrapper for LangGraph nodes | Bridges Anthropic SDK to LangChain Runnable protocol. Required for `llm.with_structured_output(Model)` and `llm.bind_tools(tools)` patterns inside graph nodes. The 1.x line (released Oct 2025) stabilized the API; 1.4.1 is current as of Apr 2026. |
| anthropic | 0.97.0 | Direct Claude SDK; pulled in by langchain-anthropic | Latest as of Apr 23 2026. Contains native Structured Outputs API (`output_config` with `json_schema` type, added v0.77.0), Claude Managed Agents (v0.92.0), and CMA Memory public beta (v0.97.0). |

**Model to use:** `claude-sonnet-4-6` — strong reasoning, reliable structured output, confirmed in project charter. Claude Opus 4.7 (v0.96.0) exists but is overkill for a portfolio demo.

**Confidence: HIGH** — verified against PyPI and anthropic SDK changelog.

---

### Structured Outputs

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pydantic | 2.13.3 | Schema definitions for agent outputs | langchain-anthropic 1.4.1 requires `pydantic>=2.7.4`. Use `BaseModel` + `Field` to define per-round schemas (e.g., `AgentPosition`, `DebateRound`, `FinalVerdict`). Pass to `llm.with_structured_output(AgentPosition)`. |

**Pattern to follow:** Use `llm.with_structured_output(MyModel)` via `langchain-anthropic`, NOT the raw Anthropic SDK `output_config` dict. The former integrates with LangGraph node return types natively.

**Confidence: HIGH** — verified against PyPI.

---

### Semantic Divergence Detection

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| sentence-transformers | 5.4.1 | Compute embeddings for agent positions; cosine similarity to detect real vs surface disagreements | Still the standard library for local embedding computation. 210M+ downloads/month on HuggingFace as of Apr 2026. Version 5.4.1 released Apr 14, 2026. |

**Model to use:** `BAAI/bge-small-en-v1.5` (33.4M params, 384-dim, MTEB avg 62.17) — preferred over the historically popular `all-MiniLM-L6-v2` (22.7M params, MTEB avg lower). Both are fast for CPU inference. `bge-small-en-v1.5` scores higher on MTEB semantic similarity tasks for roughly similar size and speed. Download once at startup; no API calls at inference time.

**Why sentence-transformers over alternatives:**

| Alternative | Why Not |
|-------------|---------|
| OpenAI `text-embedding-3-small` | Requires API call per divergence check — adds latency and cost to every debate round; not suitable for tight inner loops |
| fastembed 0.8.0 | Lighter runtime, good for RAG retrieval tasks, but sentence-transformers has wider model support and is the direct integration path for SBERT models |
| Claude API embeddings | Anthropic does not expose an embeddings endpoint; not available |

**Confidence: MEDIUM** — MTEB scores verified from HuggingFace model cards; CPU speed benchmarks from SBERT docs (170 sentences/sec for MiniLM-L6); bge-small-v1.5 MTEB score from HuggingFace model card. No head-to-head benchmark for this exact use case found.

---

### Persistence

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| langgraph-checkpoint-sqlite | 3.0.3 | `SqliteSaver` for LangGraph state checkpointing (HITL resume, debate replay) | Zero-infrastructure SQLite persistence. Required for `interrupt()`-based human-in-the-loop and debate replay. Uses `aiosqlite>=0.20` internally. |
| aiosqlite | >=0.20 | Async SQLite driver | Pulled in by langgraph-checkpoint-sqlite. |

**Pattern:** Use `SqliteSaver.from_conn_string("debate_history.db")` for the graph checkpointer. Debate rounds are checkpointed automatically at each node boundary — no manual write logic needed.

**Why SQLite over Postgres:** Portfolio demo is single-user, single-machine. No ops overhead, no docker compose. `langgraph-checkpoint-postgres` is the right move only when deploying for concurrent users.

**Confidence: HIGH** — verified against PyPI.

---

### Frontend

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| streamlit | 1.56.0 | Web UI: topic input, live debate progress, final report | Latest stable. `st.chat_message()` and `st.chat_input()` (available since 1.23) give a debate-feed UI out of the box. `st.session_state` is the correct mechanism for persisting `graph` and `thread_id` across Streamlit reruns. |

**Confidence: HIGH** — verified against PyPI.

---

## LangGraph Patterns for This System

### Pattern 1: Parallel Fan-Out with `Send` (Round 1 Analysis)

Round 1 requires 4 agents analyzing independently with no cross-visibility. Use `Send` for true parallel dispatch:

```python
from langgraph.types import Send

def dispatch_to_agents(state: DebateState):
    # Each agent gets the topic but not each other's output
    return [
        Send("agent_node", {"agent_id": agent_id, "topic": state["topic"]})
        for agent_id in ["optimist", "pessimist", "devils_advocate", "synthesizer"]
    ]

graph.add_conditional_edges("start", dispatch_to_agents)
```

Results from all 4 nodes are collected by the reducer on the shared state list before the next node runs. This is the canonical map-reduce pattern in LangGraph 1.x.

### Pattern 2: `Command` for Debate Routing (Multi-Round Loop)

After Round 1, agents enter debate rounds where they respond to each other. Use `Command` to combine state update + routing:

```python
def debate_router(state: DebateState) -> Command:
    if state["round"] >= MAX_ROUNDS or state["divergence_score"] < THRESHOLD:
        return Command(goto="synthesizer")
    return Command(
        update={"round": state["round"] + 1},
        goto="debate_round"
    )
```

### Pattern 3: Shared State with Typed Reducers

```python
from typing import Annotated
from operator import add

class DebateState(TypedDict):
    topic: str
    positions: Annotated[list[AgentPosition], add]  # agents append; never overwrite
    rebuttals: Annotated[list[Rebuttal], add]
    concessions: Annotated[list[Concession], add]
    round: int
    divergence_score: float
    final_verdict: FinalVerdict | None
```

Use `Annotated[list, add]` for append-only channels. Multiple parallel nodes writing to `positions` will all be merged correctly by LangGraph's superstep resolution.

### Pattern 4: `interrupt()` for Human Review (Optional)

If demo includes a "pause and review" mode:

```python
from langgraph.types import interrupt

def review_checkpoint(state: DebateState):
    human_input = interrupt({"positions_so_far": state["positions"]})
    return {"human_override": human_input}
```

Requires `SqliteSaver` as the checkpointer (not `InMemorySaver`) for the `graph.invoke()` → pause → `graph.invoke(Command(resume=...))` pattern to survive across Streamlit reruns.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Orchestration | LangGraph 1.1.9 | AutoGen | AutoGen's multi-agent is conversation-centric; LangGraph gives explicit state control and checkpointing needed for an auditable reasoning trace |
| Orchestration | LangGraph 1.1.9 | CrewAI | CrewAI is higher-level and opinionated; less flexibility for custom debate loop logic; LangGraph gives full control |
| LLM wrapper | langchain-anthropic 1.4.1 | anthropic SDK directly | Direct SDK skips LangChain Runnable protocol; `with_structured_output` and `bind_tools` not available without a custom adapter |
| Divergence | sentence-transformers + bge-small-en-v1.5 | Claude API comparison | Would require 2 extra API round-trips per divergence check; adds latency to every debate loop iteration |
| Divergence | bge-small-en-v1.5 | all-MiniLM-L6-v2 | bge-small-en-v1.5 scores higher on MTEB at similar size (~33M vs ~22M params); both fast for CPU |
| State type | TypedDict | Pydantic BaseModel for graph state | TypedDict is simpler; LangGraph reducers work natively with `Annotated[T, reducer]` on TypedDict; Pydantic state adds validation overhead without benefit here |
| Checkpointer | SqliteSaver | InMemorySaver | InMemorySaver loses all state on process restart; SQLite is required for debate replay feature |
| Supervisor lib | Custom conditional edges | langgraph-supervisor 0.0.31 | The library's own docs now recommend "supervisor pattern directly via tools" over the library itself. For a debate graph, custom routing logic via `Command` is cleaner and more flexible. |
| Frontend | Streamlit 1.56.0 | Gradio | Streamlit has better `session_state` management for stateful graph interactions across reruns; more readable code for portfolio reviewers |

---

## What NOT to Use and Why

**Do NOT install `langchain` (the full meta-package).** Only install `langchain-core` and `langchain-anthropic`. The `langchain` package pulls in ~40 extras (SQL chains, document loaders, PDF parsers) unused here and inflates the environment.

**Do NOT use `langgraph-supervisor` library.** Version 0.0.31, last released Nov 2025. The library's own README now says: "We now recommend using the supervisor pattern directly via tools rather than this library." Use LangGraph primitives directly.

**Do NOT use the old `interrupt_before=["node"]` compile pattern.** Use `interrupt()` inside nodes (LangGraph 1.x style). Gives finer control — you can include context (e.g., current debate positions) in the interrupt value for the UI to display.

**Do NOT use `MemorySaver` as a synonym for `InMemorySaver`.** The correct import is `from langgraph.checkpoint.memory import InMemorySaver`. `MemorySaver` is a deprecated alias.

**Do NOT use OpenAI or any other LLM.** The project charter commits to Claude. Mixing providers adds environment complexity and defeats the portfolio narrative of "built end-to-end on Claude."

**Do NOT poll for streaming output in Streamlit by calling `graph.invoke()` and then refreshing.** Use `graph.stream()` with `stream_mode="updates"` and pipe results into `st.write_stream()` or a `st.empty()` placeholder for live debate progress.

---

## Installation

```bash
# Core LangGraph stack
# langgraph 1.1.9 auto-installs: langgraph-checkpoint>=4.0, langgraph-prebuilt>=1.0.9, langchain-core>=1.3.1
pip install "langgraph==1.1.9"

# Claude integration
# langchain-anthropic 1.4.1 auto-installs: anthropic>=0.96.0
pip install "langchain-anthropic==1.4.1"

# SQLite checkpointing (for graph interrupt/resume persistence + debate replay)
# auto-installs: aiosqlite>=0.20, sqlite-vec>=0.1.6
pip install "langgraph-checkpoint-sqlite==3.0.3"

# UI
pip install "streamlit==1.56.0"

# Structured output (pydantic 2.13.3 already satisfied by langgraph requirements)
# No explicit install needed — langgraph pulls it in.
# Verify: python -c "import pydantic; print(pydantic.VERSION)"

# Semantic divergence detection
pip install "sentence-transformers==5.4.1"
```

Note: `pydantic>=2.7.4` is pulled in by langgraph. Do not pin it separately unless you hit a conflict.

---

## Version Compatibility Matrix

| Package | Version | Requires |
|---------|---------|---------|
| langgraph | 1.1.9 | python>=3.10, langchain-core>=1.3.0, pydantic>=2.7.4 |
| langchain-anthropic | 1.4.1 | anthropic>=0.96.0, langchain-core>=1.3.0, pydantic>=2.7.4 |
| anthropic | 0.97.0 | python>=3.8 |
| pydantic | 2.13.3 | python>=3.8 |
| langgraph-checkpoint-sqlite | 3.0.3 | langgraph-checkpoint>=3,<5, aiosqlite>=0.20, sqlite-vec>=0.1.6 |
| streamlit | 1.56.0 | python>=3.9 |
| sentence-transformers | 5.4.1 | python>=3.10, torch or onnx backend |

---

## Sources

- PyPI JSON API (live queries): langgraph 1.1.9, anthropic 0.97.0, streamlit 1.56.0, langchain-anthropic 1.4.1, langgraph-checkpoint-sqlite 3.0.3, pydantic 2.13.3, sentence-transformers 5.4.1, fastembed 0.8.0, langgraph-supervisor 0.0.31, langchain-core 1.3.1 — HIGH confidence
- Anthropic SDK CHANGELOG (github.com/anthropics/anthropic-sdk-python): structured outputs added v0.77.0, Claude Managed Agents v0.92.0, CMA Memory v0.97.0 — HIGH confidence
- LangGraph multi-agent concepts (docs.langchain.com): Send API, Command primitive, StateGraph pattern, shared state reducers — HIGH confidence
- HuggingFace model cards: all-MiniLM-L6-v2 (210M downloads/month), BAAI/bge-small-en-v1.5 (MTEB avg 62.17, 33.4M params) — MEDIUM confidence (no direct head-to-head benchmark for debate use case)
- SBERT pretrained models docs (sbert.net): all-MiniLM-L6-v2 750 sentences/sec CPU — MEDIUM confidence (benchmark from sbert.net docs, hardware-dependent)
- langgraph-supervisor PyPI / README: "We now recommend using the supervisor pattern directly via tools" — HIGH confidence
