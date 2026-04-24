<!-- GSD:project-start source:PROJECT.md -->
## Project

**Project: Multi-Agent Debate System**

A multi-agent debate system where specialized LLM agents with distinct cognitive biases analyze a topic independently, then engage in structured argumentation with divergence detection and concession tracking, producing auditable consensus reports with confidence scoring.

**Core value:** Given any topic or question, produce a more reliable, multi-perspective analysis than a single LLM can — by having agents with different "personalities" challenge each other, detect real disagreements, and reach a traceable consensus.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Recommended Stack
### Core Orchestration
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| langgraph | 1.1.9 | Multi-agent state machine, graph execution | Latest stable. The 1.x series is a production-ready rewrite with clean `interrupt()`/`Command` API, `Send` for parallel fan-out, and first-class multi-agent support via `Command.PARENT` handoffs. |
| langgraph-checkpoint | 4.0.2 | Checkpoint base classes (`InMemorySaver`) | Auto-installed by langgraph 1.1.9. |
| langgraph-prebuilt | 1.0.10 | `ToolNode`, `create_react_agent` | Auto-installed. Not central to debate graph but available if an agent needs tool-calling. |
| langchain-core | 1.3.1 | Message types, Runnable protocol | Required by langgraph. Provides `HumanMessage`, `AIMessage`, `BaseMessage`, and the `add_messages` reducer used by `MessagesState`. |
### LLM Integration
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| langchain-anthropic | 1.4.1 | `ChatAnthropic` wrapper for LangGraph nodes | Bridges Anthropic SDK to LangChain Runnable protocol. Required for `llm.with_structured_output(Model)` and `llm.bind_tools(tools)` patterns inside graph nodes. The 1.x line (released Oct 2025) stabilized the API; 1.4.1 is current as of Apr 2026. |
| anthropic | 0.97.0 | Direct Claude SDK; pulled in by langchain-anthropic | Latest as of Apr 23 2026. Contains native Structured Outputs API (`output_config` with `json_schema` type, added v0.77.0), Claude Managed Agents (v0.92.0), and CMA Memory public beta (v0.97.0). |
### Structured Outputs
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pydantic | 2.13.3 | Schema definitions for agent outputs | langchain-anthropic 1.4.1 requires `pydantic>=2.7.4`. Use `BaseModel` + `Field` to define per-round schemas (e.g., `AgentPosition`, `DebateRound`, `FinalVerdict`). Pass to `llm.with_structured_output(AgentPosition)`. |
### Semantic Divergence Detection
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| sentence-transformers | 5.4.1 | Compute embeddings for agent positions; cosine similarity to detect real vs surface disagreements | Still the standard library for local embedding computation. 210M+ downloads/month on HuggingFace as of Apr 2026. Version 5.4.1 released Apr 14, 2026. |
| Alternative | Why Not |
|-------------|---------|
| OpenAI `text-embedding-3-small` | Requires API call per divergence check — adds latency and cost to every debate round; not suitable for tight inner loops |
| fastembed 0.8.0 | Lighter runtime, good for RAG retrieval tasks, but sentence-transformers has wider model support and is the direct integration path for SBERT models |
| Claude API embeddings | Anthropic does not expose an embeddings endpoint; not available |
### Persistence
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| langgraph-checkpoint-sqlite | 3.0.3 | `SqliteSaver` for LangGraph state checkpointing (HITL resume, debate replay) | Zero-infrastructure SQLite persistence. Required for `interrupt()`-based human-in-the-loop and debate replay. Uses `aiosqlite>=0.20` internally. |
| aiosqlite | >=0.20 | Async SQLite driver | Pulled in by langgraph-checkpoint-sqlite. |
### Frontend
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| streamlit | 1.56.0 | Web UI: topic input, live debate progress, final report | Latest stable. `st.chat_message()` and `st.chat_input()` (available since 1.23) give a debate-feed UI out of the box. `st.session_state` is the correct mechanism for persisting `graph` and `thread_id` across Streamlit reruns. |
## LangGraph Patterns for This System
### Pattern 1: Parallel Fan-Out with `Send` (Round 1 Analysis)
### Pattern 2: `Command` for Debate Routing (Multi-Round Loop)
### Pattern 3: Shared State with Typed Reducers
### Pattern 4: `interrupt()` for Human Review (Optional)
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
## What NOT to Use and Why
## Installation
# Core LangGraph stack
# langgraph 1.1.9 auto-installs: langgraph-checkpoint>=4.0, langgraph-prebuilt>=1.0.9, langchain-core>=1.3.1
# Claude integration
# langchain-anthropic 1.4.1 auto-installs: anthropic>=0.96.0
# SQLite checkpointing (for graph interrupt/resume persistence + debate replay)
# auto-installs: aiosqlite>=0.20, sqlite-vec>=0.1.6
# UI
# Structured output (pydantic 2.13.3 already satisfied by langgraph requirements)
# No explicit install needed — langgraph pulls it in.
# Verify: python -c "import pydantic; print(pydantic.VERSION)"
# Semantic divergence detection
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
## Sources
- PyPI JSON API (live queries): langgraph 1.1.9, anthropic 0.97.0, streamlit 1.56.0, langchain-anthropic 1.4.1, langgraph-checkpoint-sqlite 3.0.3, pydantic 2.13.3, sentence-transformers 5.4.1, fastembed 0.8.0, langgraph-supervisor 0.0.31, langchain-core 1.3.1 — HIGH confidence
- Anthropic SDK CHANGELOG (github.com/anthropics/anthropic-sdk-python): structured outputs added v0.77.0, Claude Managed Agents v0.92.0, CMA Memory v0.97.0 — HIGH confidence
- LangGraph multi-agent concepts (docs.langchain.com): Send API, Command primitive, StateGraph pattern, shared state reducers — HIGH confidence
- HuggingFace model cards: all-MiniLM-L6-v2 (210M downloads/month), BAAI/bge-small-en-v1.5 (MTEB avg 62.17, 33.4M params) — MEDIUM confidence (no direct head-to-head benchmark for debate use case)
- SBERT pretrained models docs (sbert.net): all-MiniLM-L6-v2 750 sentences/sec CPU — MEDIUM confidence (benchmark from sbert.net docs, hardware-dependent)
- langgraph-supervisor PyPI / README: "We now recommend using the supervisor pattern directly via tools" — HIGH confidence
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
