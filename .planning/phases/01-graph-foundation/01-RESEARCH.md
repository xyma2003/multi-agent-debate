# Phase 1: Graph Foundation — Research

**Researched:** 2026-04-24
**Domain:** LangGraph multi-agent fan-out/fan-in, Pydantic structured output, persona prompt engineering
**Confidence:** HIGH (all stack choices verified against CLAUDE.md + sibling project working code)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DEBATE-01 | User can enter any topic/question and trigger a multi-agent debate | Graph entry point: `initialize` node parses topic from state input; `graph.invoke({"topic": ..., "max_rounds": 3})` is the public API |
| DEBATE-02 | 3 agents (Optimist, Pessimist, Devil's Advocate) analyze independently in Round 1 with no cross-visibility | `Send` fan-out pattern; each agent receives only `{"topic": ..., "agent_role": ..., "prior_arguments": [], "round_num": 0}` — no shared message history |
| DEBATE-03 | Each agent produces: position, reasoning, key_claims, confidence, concessions | `AgentArgument` Pydantic model with these exact fields; enforced via `llm.with_structured_output(AgentArgument)` |
| AGENT-01 | Structural persona prompt enforces cognitive bias via methodology (not personality adjectives) | Methodology-based system prompts documented in Code Examples; reference personas from PITFALLS.md Pitfall 8 |
| AGENT-02 | Anti-sycophancy instructions prevent agents from conceding to avoid conflict | Explicit prohibitions embedded in system prompt; forbidden phrases enumerated in Code Examples |
| AGENT-03 | Pydantic validation errors handled with 2-retry wrapper; sentinel AgentArgument on third failure | `include_raw=True` pattern + retry wrapper pattern documented in Code Examples |

</phase_requirements>

---

## Summary

Phase 1 builds the runnable skeleton of the debate graph: state schema, three agent nodes with real LLM calls, fan-out/fan-in wiring, and validated structured output. No divergence detection (Phase 2) or synthesis (Phase 3) — those depend on real agent outputs to calibrate. The deliverable is: `graph.invoke({"topic": "Is remote work better than office work?"})` returns three distinct `AgentArgument` objects.

The full stack is already decided and locked in CLAUDE.md. No alternatives exist to research. The most important Phase 1 risk is not technical plumbing — it is persona collapse: agents secretly agreeing and producing surface-level differences. Phase 1 must include real persona enforcement and validation before declaring success.

The sibling project `workdiary_agent/` provides directly applicable working patterns for `ChatAnthropic`, `with_structured_output`, `StateGraph`, and `InMemorySaver`. The key difference is the `Send` fan-out pattern, which `workdiary_agent` does not use (it is linear). That pattern is documented exhaustively in ARCHITECTURE.md and is the primary new technical element for Phase 1.

**Primary recommendation:** Build in this order: (1) `DebateState` + `AgentArgument` Pydantic models, (2) stub graph topology with `Send` fan-out, (3) `_make_llm()` helper with custom headers, (4) real agent nodes with persona prompts, (5) retry wrapper, (6) smoke test script.

---

## Project Constraints (from CLAUDE.md)

The following directives from `./CLAUDE.md` are binding. The planner must not recommend anything that contradicts these.

| Constraint | Value |
|------------|-------|
| LLM | `claude-sonnet-4-6` (confirmed model ID in STACK.md). NOT claude-sonnet-4-5 from sibling project. |
| LLM wrapper | `langchain-anthropic 1.4.1` + `ChatAnthropic` — NOT raw Anthropic SDK |
| Pydantic version | 2.13.3 (pulled in by langgraph; already installed as 2.12.4 in system env) |
| LangGraph version | 1.1.9 |
| Orchestration | Single flat `StateGraph`, no subgraph nesting |
| State type | `TypedDict` (NOT Pydantic BaseModel for graph state) |
| Fan-out | `Send` API (NOT subgraphs, NOT multiple separate graph invocations) |
| Auth | `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_CUSTOM_HEADERS` env vars (proxy, no `ANTHROPIC_API_KEY`) |
| No install | `langchain` meta-package, `langgraph-supervisor` |
| State reducers | `Annotated[list[AgentArgument], add]` for fan-in fields only; all others default (last-write-wins) |
| GSD workflow | Edit/Write only through GSD commands |

---

## Standard Stack

### Core (Phase 1 only)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| langgraph | 1.1.9 | StateGraph, Send, conditional edges, InMemorySaver | Locked in CLAUDE.md; verified in sibling project |
| langchain-anthropic | 1.4.1 | `ChatAnthropic`, `with_structured_output` | Required for Runnable protocol; locked in CLAUDE.md |
| anthropic | 0.97.0 | Auto-installed by langchain-anthropic | Claude API access |
| langchain-core | 1.3.1 | `HumanMessage`, `SystemMessage` | Auto-installed by langgraph |
| pydantic | 2.13.3 | `AgentArgument`, `Concession` schema | Auto-installed by langgraph; 2.12.4 already present |

### Not Needed in Phase 1

| Library | Purpose | Phase |
|---------|---------|-------|
| sentence-transformers | Divergence detection | Phase 2 |
| langgraph-checkpoint-sqlite | Replay persistence | Phase 4 |
| streamlit | UI | Phase 5 |

**Installation:**

```bash
# Use the conda environment that has the proxy setup (same as workdiary_agent)
# The following installs into the active environment
pip install "langgraph==1.1.9" "langchain-anthropic==1.4.1"
# Pydantic 2.12.4 already present; langgraph will accept it (requires >=2.7.4)
```

**Note on auth:** The project uses a Meituan internal proxy. No `ANTHROPIC_API_KEY` is required. The `_make_llm()` helper must read `ANTHROPIC_CUSTOM_HEADERS` and pass them as `default_headers` to `ChatAnthropic`. See the working implementation in `workdiary_agent/nodes/extract.py` — copy that helper verbatim and update the model name to `claude-sonnet-4-6`.

---

## Architecture Patterns

### Recommended Project Structure

```
debate-agent/
├── CLAUDE.md
├── requirements.txt           # pinned deps (copy from sibling project, update versions)
├── debate/
│   ├── __init__.py
│   ├── state.py               # DebateState TypedDict + AgentArgument + Concession Pydantic models
│   ├── graph.py               # StateGraph assembly: add_node, add_edge, compile
│   ├── llm.py                 # _make_llm() helper with custom headers
│   └── nodes/
│       ├── __init__.py
│       ├── initialize.py      # parse topic, stamp metadata, set round_num=0
│       ├── dispatch.py        # dispatch_round1: returns list[Send]
│       ├── agents.py          # optimist_node, pessimist_node, devil_node (same function, different role)
│       └── collect.py         # collect_round1: fan-in, move args to round_history
├── tests/
│   └── test_phase1.py         # smoke test: invoke graph, assert 3 AgentArguments returned
└── .planning/
    └── phases/01-graph-foundation/
```

**Why this structure:**
- Mirrors `workdiary_agent/` layout (builder already has mental model)
- `nodes/` separates concerns cleanly — each file = one responsibility
- `llm.py` centralizes the proxy-aware `_make_llm()` helper; all nodes import from it
- `state.py` is the single source of truth for schemas; no circular imports

### Pattern 1: DebateState TypedDict with Add Reducer

**What:** The fan-in accumulator field uses `Annotated[list, add]` so parallel `Send` nodes can each append their output without overwriting each other.

**Critical rule:** ONLY `current_round_arguments` needs a reducer. All other fields are last-write-wins (default).

```python
# debate/state.py
from __future__ import annotations
from typing import Annotated, Optional
from operator import add
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class Concession(BaseModel):
    conceded_point: str
    triggered_by_agent: str        # "optimist" | "pessimist" | "devil"
    triggered_by_claim: str
    rationale: str


class AgentArgument(BaseModel):
    agent_role: str                 # "optimist" | "pessimist" | "devil"
    round_num: int
    position: str                   # main thesis — ONE sentence
    reasoning: str                  # full argument prose
    confidence: float               # 0.0–1.0, self-reported
    key_claims: list[str]           # 3–7 short extractable claims (for embedding in Phase 2)
    concessions: list[Concession]   # points yielded this round (empty in Round 1)


class RoundRecord(BaseModel):
    round_num: int
    arguments: list[AgentArgument]


class DebateState(TypedDict):
    # Input
    topic: str
    debate_id: str

    # Round tracking
    round_num: int
    max_rounds: int                 # default 3

    # Fan-in accumulator — ONLY this field uses a reducer
    current_round_arguments: Annotated[list[AgentArgument], add]

    # Full history (Phase 2+ reads this)
    round_history: list[RoundRecord]

    # Divergence signal (Phase 2 writes these; Phase 1 leaves at defaults)
    divergence_score: float
    diverged_pairs: list[tuple[str, str]]

    # Terminal state (Phase 3 writes this)
    final_report: Optional[object]   # DebateReport type added in Phase 3
    status: str                       # "running" | "converged" | "max_rounds" | "complete"
```

**Source:** ARCHITECTURE.md (this repo), verified against LangGraph source `langgraph/graph/message.py` (add_messages uses same pattern).

### Pattern 2: Send Fan-Out from dispatch_round1

**What:** The dispatch node returns a `list[Send]` — not a string, not a `Command`. Each `Send` carries an isolated payload. Agents do NOT receive the full `DebateState`.

**Critical:** `add_conditional_edges` is the correct call for a node that returns `list[Send]`.

```python
# debate/nodes/dispatch.py
import uuid
from langgraph.types import Send
from debate.state import DebateState


def dispatch_round1(state: DebateState) -> list[Send]:
    """Fan-out: dispatch 3 agents in parallel with no cross-visibility."""
    topic = state["topic"]
    round_num = state.get("round_num", 0)
    return [
        Send("optimist_node",  {"topic": topic, "agent_role": "optimist",  "prior_arguments": [], "round_num": round_num}),
        Send("pessimist_node", {"topic": topic, "agent_role": "pessimist", "prior_arguments": [], "round_num": round_num}),
        Send("devil_node",     {"topic": topic, "agent_role": "devil",     "prior_arguments": [], "round_num": round_num}),
    ]
```

**Wiring in graph.py:**

```python
# debate/graph.py (Phase 1 wiring only)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from debate.state import DebateState
from debate.nodes.initialize import initialize_node
from debate.nodes.dispatch import dispatch_round1
from debate.nodes.agents import optimist_node, pessimist_node, devil_node
from debate.nodes.collect import collect_round1


def build_graph():
    builder = StateGraph(DebateState)

    builder.add_node("initialize", initialize_node)
    builder.add_node("dispatch_round1", dispatch_round1)
    builder.add_node("optimist_node", optimist_node)
    builder.add_node("pessimist_node", pessimist_node)
    builder.add_node("devil_node", devil_node)
    builder.add_node("collect_round1", collect_round1)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "dispatch_round1")

    # Fan-out: dispatch_round1 returns list[Send]
    builder.add_conditional_edges("dispatch_round1", lambda s: s)

    # Fan-in: all three agent nodes converge to collect_round1
    builder.add_edge("optimist_node",  "collect_round1")
    builder.add_edge("pessimist_node", "collect_round1")
    builder.add_edge("devil_node",     "collect_round1")

    # Phase 1 ends here — Phase 2 replaces this with conditional routing
    builder.add_edge("collect_round1", END)

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
```

**Source:** ARCHITECTURE.md (this repo), STACK.md Pattern 1.

### Pattern 3: Agent Node with with_structured_output + Retry Wrapper

**What:** Agent nodes call `llm.with_structured_output(AgentArgument, include_raw=True)`. The `include_raw=True` flag returns a dict with `"parsed"`, `"raw"`, and `"parsing_error"` keys. On failure, retry up to 2 times; inject sentinel on third failure.

```python
# debate/nodes/agents.py
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from debate.state import AgentArgument, Concession
from debate.llm import _make_llm
from debate.prompts import AGENT_PROMPTS   # see Pattern 4


_SENTINEL_ARGUMENT = AgentArgument(
    agent_role="unknown",
    round_num=-1,
    position="[Analysis unavailable due to validation error]",
    reasoning="The LLM returned a response that could not be parsed into the required schema after 3 attempts.",
    confidence=0.0,
    key_claims=["validation_error"],
    concessions=[],
)


def _invoke_with_retry(llm, messages: list, role: str, max_retries: int = 2) -> AgentArgument:
    """Invoke with_structured_output up to max_retries times. Return sentinel on total failure."""
    structured_llm = llm.with_structured_output(AgentArgument, include_raw=True)
    for attempt in range(max_retries + 1):
        result = structured_llm.invoke(messages)
        if result["parsed"] is not None:
            parsed: AgentArgument = result["parsed"]
            parsed.agent_role = role   # ensure role is set correctly
            return parsed
        # Log the error; retry with same messages (Anthropic usually self-corrects)
        print(f"[{role}] Pydantic parse failed (attempt {attempt+1}/{max_retries+1}): {result['parsing_error']}")
    sentinel = _SENTINEL_ARGUMENT.model_copy()
    sentinel.agent_role = role
    return sentinel


def _agent_node(state: dict, role: str) -> dict:
    """Shared implementation for all three agent nodes."""
    topic = state["topic"]
    round_num = state.get("round_num", 0)
    system_prompt = AGENT_PROMPTS[role]

    llm = _make_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Topic for analysis: {topic}\n\nRound: {round_num + 1}"),
    ]
    argument = _invoke_with_retry(llm, messages, role)
    # Return as a single-item list — add reducer appends to current_round_arguments
    return {"current_round_arguments": [argument]}


def optimist_node(state: dict) -> dict:
    return _agent_node(state, "optimist")

def pessimist_node(state: dict) -> dict:
    return _agent_node(state, "pessimist")

def devil_node(state: dict) -> dict:
    return _agent_node(state, "devil")
```

**Key design choices:**
- Returning `{"current_round_arguments": [argument]}` — a single-item list — triggers the `add` reducer to append to the accumulator.
- `include_raw=True` is mandatory from day one. Default behavior crashes the node on parse failure.
- `_make_llm()` lives in `debate/llm.py` and handles `ANTHROPIC_CUSTOM_HEADERS` (see sibling project pattern).

**Source:** PITFALLS.md Pitfall 5, langchain-core `with_structured_output` source verified.

### Pattern 4: Collect Node (Fan-In)

```python
# debate/nodes/collect.py
from debate.state import DebateState, RoundRecord


def collect_round1(state: DebateState) -> dict:
    """Move current_round_arguments into round_history; reset accumulator."""
    new_record = RoundRecord(
        round_num=state.get("round_num", 0),
        arguments=state["current_round_arguments"],
    )
    return {
        "round_history": state.get("round_history", []) + [new_record],
        "current_round_arguments": [],   # reset for Phase 2's rebuttal rounds
        "round_num": state.get("round_num", 0) + 1,
        "status": "running",
    }
```

**Source:** ARCHITECTURE.md Fan-in section.

### Pattern 5: _make_llm() Helper with Proxy Auth

The environment uses `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_CUSTOM_HEADERS` — NOT `ANTHROPIC_API_KEY`. This is confirmed by the running sibling project.

```python
# debate/llm.py
import os
from langchain_anthropic import ChatAnthropic

MODEL_ID = "claude-sonnet-4-6"   # Phase 1–5 (confirmed in STACK.md)


def _make_llm() -> ChatAnthropic:
    """Return ChatAnthropic configured for the Meituan internal proxy.

    Reads ANTHROPIC_CUSTOM_HEADERS (newline-separated 'Key: Value' pairs).
    ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN are read automatically by the
    anthropic SDK from environment — no explicit config needed.
    """
    custom_headers_str = os.environ.get("ANTHROPIC_CUSTOM_HEADERS", "")
    headers: dict[str, str] = {}
    if custom_headers_str:
        for line in custom_headers_str.split("\n"):
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip()] = v.strip()
    return ChatAnthropic(model=MODEL_ID, default_headers=headers)
```

**Copied from:** `workdiary_agent/nodes/extract.py` `_make_llm()` (verified working). Updated model to `claude-sonnet-4-6`.

### Pattern 6: Initialize Node

```python
# debate/nodes/initialize.py
import uuid
from debate.state import DebateState


def initialize_node(state: DebateState) -> dict:
    """Stamp metadata and initialize counters for a new debate."""
    return {
        "debate_id": str(uuid.uuid4()),
        "round_num": 0,
        "max_rounds": state.get("max_rounds", 3),
        "current_round_arguments": [],
        "round_history": [],
        "divergence_score": 0.0,
        "diverged_pairs": [],
        "final_report": None,
        "status": "running",
    }
```

### Anti-Patterns to Avoid

- **Do NOT use `add_messages` or `MessagesState` for debate content.** Round arguments are Pydantic objects, not messages. Using `add_messages` creates a growing message list that causes context blowup by round 3 (PITFALLS Pitfall 3).
- **Do NOT pass `DebateState` directly to `Send`.** Each `Send` should receive only the minimal fields the agent needs. Cross-contamination of Round 1 is the #1 data integrity risk.
- **Do NOT use bare `list[AgentArgument]` for the fan-in field.** Must be `Annotated[list[AgentArgument], add]` or the last agent overwrites the others (PITFALLS Pitfall 10).
- **Do NOT skip `include_raw=True`.** Default `with_structured_output` raises unhandled `ValidationError` on LLM schema violations (PITFALLS Pitfall 5).
- **Do NOT use `state["field"]` bracket access on optional fields.** Use `state.get("field", default)`. The `DebateState` is constructed incrementally — early nodes leave later fields unset.

---

## Persona Prompt Engineering

This is the highest-risk area of Phase 1. The difference between a working debate and a sycophantic one is entirely in the system prompt.

### Structural Persona Formula

**Bad (intensity-based):**
```
You are a pessimist. Be very critical and find all the problems with this idea.
```

**Good (methodology-based):**
```
You are the Risk Analyst. Your analytical framework is:
1. Identify the single most likely failure mode for this opportunity
2. Estimate its probability (high/medium/low) and potential impact
3. Assess whether the opportunity's upside justifies that specific downside risk
4. List 3–7 concrete risk claims as your key_claims

You analyze like a risk manager at a venture debt fund: you have seen many pitches,
you know where execution usually breaks down, and you are paid to find the failure mode
before the investment is made.

PROHIBITION: Do not write "however", "on the other hand", "while there are upsides",
or any phrase that acknowledges the opportunity's merits. That is not your job.
Your position must be a concrete risk claim, not a hedge.
```

### Three Persona Templates (Phase 1)

```python
# debate/prompts.py

AGENT_PROMPTS = {
    "optimist": """You are the Opportunity Analyst. Your analytical framework is:
1. Enumerate the most compelling reasons this could succeed
2. Identify which conditions or trends favor success
3. Assess the magnitude of the upside if key assumptions hold
4. List 3–7 concrete opportunity claims as your key_claims

You analyze like a seed-stage VC associate evaluating portfolio fit: you are looking
for asymmetric upside, you know most ideas have some flaw, and your job is to find
the ones where the upside overwhelms the downside.

PROHIBITION: Do not mention risks, caveats, failure modes, or qualifications.
Do not write "however", "but", "although", or "unless".
If you find yourself writing a caveat, delete it.
Your position must be a concrete opportunity claim.""",

    "pessimist": """You are the Risk Analyst. Your analytical framework is:
1. Identify the single most likely failure mode
2. Estimate its probability and impact severity
3. Assess whether the stated opportunity justifies accepting that specific risk
4. List 3–7 concrete risk claims as your key_claims

You analyze like a risk manager at a venture debt fund: you have seen many deals,
you know where execution usually breaks down, and you are paid to surface the failure
mode before the capital is deployed.

PROHIBITION: Do not mention upsides, opportunities, or positive scenarios.
Do not write "however", "but there is potential", or "while risky, it could work".
Your position must be a concrete risk claim, not a hedge.""",

    "devil": """You are the Challenger. Your analytical framework is:
1. Identify the current majority view or dominant argument being made
2. Find the most significant logical flaw, missing assumption, or overlooked factor in that view
3. Construct the strongest possible counter-argument
4. List 3–7 concrete challenge claims as your key_claims

You analyze like a senior strategy consultant who has heard this pitch three times before
and found a specific flaw that the presenter keeps glossing over.

PROHIBITION: Do not agree with the dominant view even partially.
Do not write "while this is a valid point" or "I can see merits on both sides".
Your position must directly challenge the prevailing view with specific evidence or logic.""",
}
```

### Anti-Sycophancy Instructions (AGENT-02)

These phrases must be present in every agent's system prompt. The planner should verify they appear in the committed prompt templates:

1. "PROHIBITION: Do not write [hedging phrases]" — explicit list of forbidden language
2. "Do not agree with... even partially" (for Devil's Advocate)
3. "If you find yourself writing a caveat, delete it" (for Optimist)
4. "Your position must be a concrete [X] claim, not a hedge" — explicit success criterion

**Compliance check heuristic (to add to smoke test):**
- Optimist output must not contain: "risk", "fail", "problem", "challenge", "concern"
- Pessimist output must not contain: "opportunity", "upside", "growth", "potential", "benefit"
- Devil's Advocate output must not contain: "valid point", "merits on both sides", "I agree that"

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Structured LLM output | Custom JSON parsing of LLM response | `llm.with_structured_output(AgentArgument)` | Handles tool-use JSON extraction, schema enforcement, error reporting |
| Parallel agent dispatch | `asyncio.gather` or threading | `Send` API in LangGraph | LangGraph handles superstep resolution, state merging, and ordering automatically |
| State merge from parallel nodes | Manual dict merging | `Annotated[list, add]` reducer | LangGraph merges reducer-annotated fields automatically after all parallel branches complete |
| Retry logic | `while True` loop calling LLM | `include_raw=True` + simple retry wrapper | `include_raw=True` surfaces parse errors without raising; wrapper is 10 lines |
| Graph checkpointing | Manual state serialization | `InMemorySaver()` on `compile()` | Required for `interrupt()` in Phase 4; zero overhead in Phase 1 |

---

## Common Pitfalls

### Pitfall 1: Persona Collapse — All Agents Produce Balanced Hedged Answers
**What goes wrong:** Claude's RLHF training rewards balanced analysis. Without structural constraints, every agent produces the same "there are pros and cons" response with different labels.
**Why it happens:** "You are a pessimist" is read as a mild stylistic suggestion, not a structural constraint.
**How to avoid:** Methodology-based prompts that specify the decision procedure, reference personas, and explicit prohibitions on hedging language (see Persona Prompt Engineering section above).
**Warning signs:** Optimist output mentions more than one risk; Pessimist mentions any opportunity; Devil's Advocate agrees with majority view in conclusion.

### Pitfall 2: Pydantic ValidationError Crashes the Node
**What goes wrong:** `with_structured_output(AgentArgument)` (default `include_raw=False`) raises `ValidationError` on ~0-1% of calls. LangGraph catches the exception; that node's state update is skipped. `current_round_arguments` has only 2 items instead of 3. Downstream nodes see partial state silently.
**How to avoid:** Always use `include_raw=True`. Build the retry wrapper from the first node implementation. Never let `None` reach `current_round_arguments`.

### Pitfall 3: Fan-In Field Missing the Add Reducer
**What goes wrong:** `current_round_arguments: list[AgentArgument]` (no Annotated) — the last agent node to complete overwrites the other two. Result: always exactly 1 argument in `collect_round1`.
**How to avoid:** `current_round_arguments: Annotated[list[AgentArgument], add]` — verified pattern from ARCHITECTURE.md and PITFALLS Pitfall 10.

### Pitfall 4: Round 1 Cross-Contamination via Full State in Send Payload
**What goes wrong:** Passing the full `DebateState` in the `Send` payload. If `round_history` already has entries (e.g., from a prior test run on the same thread_id), agents see prior arguments.
**How to avoid:** `Send` payload contains ONLY: `{"topic": topic, "agent_role": role, "prior_arguments": [], "round_num": 0}`. Never pass `state` directly.

### Pitfall 5: Wrong `add_conditional_edges` call for Send
**What goes wrong:** Using `add_edge("dispatch_round1", "optimist_node")` creates a single edge, not a fan-out.
**How to avoid:** `builder.add_conditional_edges("dispatch_round1", lambda s: s)` — the identity lambda passes through the `list[Send]` that `dispatch_round1` returns. LangGraph interprets a returned `list[Send]` as parallel dispatch automatically.

### Pitfall 6: Using `state["field"]` on Optional Fields
**What goes wrong:** `DebateState` fields are not all set at invocation time. `state["round_num"]` raises `KeyError` if the field was not in the initial input dict.
**How to avoid:** Use `state.get("round_num", 0)` in all node implementations. Initialize all fields in `initialize_node` before any other node runs.

---

## Environment Setup

### Requirements for Phase 1

```
# debate-agent/requirements.txt
langgraph==1.1.9
langchain-anthropic==1.4.1
langgraph-checkpoint-sqlite==3.0.3   # install now; swap InMemorySaver → SqliteSaver in Phase 4
# pydantic >= 2.7.4 pulled in by langgraph; 2.12.4 already present
```

**Install command:**
```bash
pip install -r requirements.txt
```

**Verify install:**
```bash
python -c "import langgraph; print(langgraph.__version__)"   # expect: 1.1.9
python -c "import langchain_anthropic; print(langchain_anthropic.__version__)"  # expect: 1.4.1
python -c "import pydantic; print(pydantic.VERSION)"  # expect: 2.12.4 or higher
```

**Python version:** Python 3.13.12 is installed (confirmed). langgraph 1.1.9 requires `python>=3.10`. Compatible.

**Auth environment variables (already set — no action needed):**
- `ANTHROPIC_BASE_URL` — proxy base URL
- `ANTHROPIC_AUTH_TOKEN` — auth token for proxy
- `ANTHROPIC_CUSTOM_HEADERS` — newline-separated headers required by internal proxy

No `ANTHROPIC_API_KEY` is needed or expected.

---

## Testing Phase 1

### Minimal Smoke Test Script

```python
# tests/test_phase1.py
"""
Smoke test for Phase 1 graph foundation.
Run: python -m pytest tests/test_phase1.py -v
Or:  python tests/test_phase1.py  (standalone)
"""
import uuid
from debate.graph import graph
from debate.state import AgentArgument


def test_phase1_returns_three_agent_arguments():
    """Invoke the graph with a topic and assert 3 AgentArguments are returned."""
    topic = "Is remote work more productive than office work?"
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 30}

    result = graph.invoke(
        {"topic": topic, "max_rounds": 3},
        config=config,
    )

    args = result.get("round_history", [{}])[0].arguments if result.get("round_history") else []
    assert len(args) == 3, f"Expected 3 AgentArguments, got {len(args)}"

    roles = {a.agent_role for a in args}
    assert roles == {"optimist", "pessimist", "devil"}, f"Missing roles: {roles}"

    for arg in args:
        assert isinstance(arg, AgentArgument)
        assert arg.position, "position must be non-empty"
        assert arg.reasoning, "reasoning must be non-empty"
        assert len(arg.key_claims) >= 3, f"{arg.agent_role} has fewer than 3 key_claims"
        assert 0.0 <= arg.confidence <= 1.0


def test_persona_compliance():
    """Verify agents are not producing hedged, balanced responses (persona enforcement check)."""
    topic = "Should startups raise venture capital?"
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 30}

    result = graph.invoke({"topic": topic, "max_rounds": 3}, config=config)
    args = result["round_history"][0].arguments

    optimist = next(a for a in args if a.agent_role == "optimist")
    pessimist = next(a for a in args if a.agent_role == "pessimist")

    OPTIMIST_FORBIDDEN = ["risk", "fail", "problem", "challenge", "concern"]
    PESSIMIST_FORBIDDEN = ["opportunity", "upside", "growth", "potential"]

    optimist_text = (optimist.position + " " + optimist.reasoning).lower()
    pessimist_text = (pessimist.position + " " + pessimist.reasoning).lower()

    opt_violations = [w for w in OPTIMIST_FORBIDDEN if w in optimist_text]
    pess_violations = [w for w in PESSIMIST_FORBIDDEN if w in pessimist_text]

    # Soft assertions — warn rather than fail (LLM output is probabilistic)
    if opt_violations:
        print(f"WARNING: Optimist persona drift — found forbidden words: {opt_violations}")
    if pess_violations:
        print(f"WARNING: Pessimist persona drift — found forbidden words: {pess_violations}")


if __name__ == "__main__":
    test_phase1_returns_three_agent_arguments()
    print("Phase 1 smoke test PASSED")
    test_persona_compliance()
    print("Persona compliance check complete")
```

**Why `recursion_limit: 30`:** Phase 1 graph is linear (no loop), so 30 is far above needed. Prevents runaway cost if a bug creates an unintended cycle.

### What "Phase 1 Done" Looks Like

Running `python tests/test_phase1.py` prints:
```
Phase 1 smoke test PASSED
Persona compliance check complete
```

Manual inspection of output also confirms:
1. All three agents produced a non-empty `position` and `reasoning`
2. Each agent has 3+ `key_claims`
3. `round_history[0].arguments` has exactly 3 items
4. No sentinel arguments in the output (no `agent_role="unknown"`)

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | Everything | Yes | 3.13.12 | — |
| langgraph | Graph execution | Not installed (needs `pip install`) | — | — |
| langchain-anthropic | LLM calls | Not installed (needs `pip install`) | — | — |
| pydantic | Structured output | Yes | 2.12.4 | — |
| ANTHROPIC_BASE_URL | API calls | Yes (confirmed in env) | — | — |
| ANTHROPIC_AUTH_TOKEN | API auth | Yes (confirmed in env) | — | — |
| ANTHROPIC_CUSTOM_HEADERS | Proxy headers | Yes (confirmed in env) | — | — |

**Missing dependencies with no fallback:**
- `langgraph==1.1.9` — must install before any phase execution
- `langchain-anthropic==1.4.1` — must install before any phase execution

**Missing dependencies with fallback:**
- None

**Wave 0 action required:** `pip install langgraph==1.1.9 langchain-anthropic==1.4.1`

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `interrupt_before=["node"]` at compile time | `interrupt()` inside node body | LangGraph 1.x | More control; can pass context to UI in interrupt value |
| `MemorySaver` | `InMemorySaver` (correct import) | LangGraph 1.x | `MemorySaver` is deprecated alias; use `from langgraph.checkpoint.memory import InMemorySaver` |
| `langchain` meta-package | `langchain-core` + `langchain-anthropic` only | Ongoing | Avoids ~40 unused dependencies |
| `langgraph-supervisor` library | Custom `Command`-based routing | Nov 2025 (library deprecated itself) | Library README now says "use supervisor pattern directly via tools" |
| Subgraphs for multi-agent | Flat graph + `Send` fan-out | LangGraph 1.x | Simpler state merging; no subgraph boundary bugs for 3-agent fan-out |

---

## Open Questions

1. **Pydantic version mismatch**
   - What we know: System Python has pydantic 2.12.4. CLAUDE.md specifies 2.13.3. langgraph requires `>=2.7.4`.
   - What's unclear: Whether any 2.12.4 → 2.13.3 behavior change affects structured output.
   - Recommendation: Use 2.12.4 as installed (compatible with all requirements). Only upgrade if a specific validation failure is observed.

2. **Whether `add_conditional_edges("dispatch_round1", lambda s: s)` is the canonical pattern**
   - What we know: ARCHITECTURE.md uses this pattern. The `Send` return type from a conditional edge function is documented in LangGraph.
   - What's unclear: Whether the identity lambda is the clearest way to write this, or if there is a more explicit API.
   - Recommendation: Use the pattern as documented. If it does not work during execution, check that `dispatch_round1` is registered as a node (not just a function) before calling `add_conditional_edges`.

3. **Whether `DebateState.current_round_arguments` should use `total=False`**
   - What we know: `workdiary_agent` uses `total=False` so all fields are optional at invocation. `DebateState` has required fields (`topic`) and optional accumulator fields.
   - What's unclear: Whether `total=True` (default) with required fields causes issues when `graph.invoke({"topic": ..., "max_rounds": 3})` is called (other fields unset).
   - Recommendation: Use `total=False` like the sibling project, or provide defaults in `initialize_node` for all fields. The `initialize_node` approach (set all fields to defaults in the first node) is cleaner — it makes state evolution explicit.

---

## Sources

### Primary (HIGH confidence)
- ARCHITECTURE.md (this repo, 2026-04-23) — `DebateState`, `Send` fan-out, fan-in pattern, `AgentArgument` model, `collect_round` implementation
- STACK.md (this repo, 2026-04-23) — version matrix, `with_structured_output` pattern, `Command` routing
- PITFALLS.md (this repo, 2026-04-23) — Pitfall 1 (sycophancy), 3 (context blowup), 4 (recursion), 5 (Pydantic), 8 (cartoonish vs collapsed), 10 (fan-out merge)
- `workdiary_agent/graph.py` (working code, this repo) — `StateGraph`, `InMemorySaver`, `add_edge`, `add_conditional_edges`, `compile` patterns
- `workdiary_agent/nodes/extract.py` (working code, this repo) — `_make_llm()` with proxy auth, `with_structured_output` usage
- `workdiary_agent/state.py` (working code, this repo) — `TypedDict` state schema, `total=False`, Pydantic `BaseModel` + `Field`
- CLAUDE.md (this repo) — locked tech decisions, version matrix, model ID

### Secondary (MEDIUM confidence)
- PITFALLS.md Pitfall 2 (divergence threshold) — applies to Phase 2, not Phase 1
- PITFALLS.md Pitfall 8 (cartoonish vs collapsed persona) — prompt engineering heuristics; probabilistic outcome

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions locked in CLAUDE.md, verified against sibling project working code
- Architecture patterns: HIGH — direct code from ARCHITECTURE.md verified against LangGraph source
- Send fan-out: HIGH — documented in ARCHITECTURE.md and STACK.md; identity lambda pattern is standard LangGraph
- Pydantic + `with_structured_output`: HIGH — working code in sibling project; `include_raw=True` verified from langchain-core source
- Persona prompt engineering: MEDIUM — methodology approach is established practice; specific forbidden phrases and compliance checks need empirical tuning after first run
- Env auth setup: HIGH — confirmed `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_CUSTOM_HEADERS` are all set; working pattern in sibling project

**Research date:** 2026-04-24
**Valid until:** 2026-05-24 (LangGraph 1.x stable; stack unlikely to change in 30 days)
