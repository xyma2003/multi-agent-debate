---
plan: 01
phase: 1
title: "Project Setup, State Schema, and Pydantic Models"
wave: 1
depends_on: []
requirements_addressed: [DEBATE-01, DEBATE-02, DEBATE-03]
files_modified:
  - requirements.txt
  - .env.example
  - debate/__init__.py
  - debate/state.py
  - debate/nodes/__init__.py
  - tests/__init__.py
autonomous: true

must_haves:
  truths:
    - "importing `from debate.state import DebateState, AgentArgument, Concession, RoundRecord` succeeds with no errors"
    - "`DebateState` TypedDict has `current_round_arguments: Annotated[list[AgentArgument], add]` — the fan-in accumulator"
    - "`AgentArgument` Pydantic model has all 7 required fields: agent_role, round_num, position, reasoning, confidence, key_claims, concessions"
    - "`langgraph==1.1.9` and `langchain-anthropic==1.4.1` are installable from requirements.txt"
  artifacts:
    - path: "debate/state.py"
      provides: "DebateState TypedDict, AgentArgument, Concession, RoundRecord Pydantic models"
      exports: [DebateState, AgentArgument, Concession, RoundRecord]
    - path: "requirements.txt"
      provides: "pinned dependency list for pip install"
      contains: "langgraph==1.1.9"
    - path: ".env.example"
      provides: "documentation of required env vars"
      contains: "ANTHROPIC_BASE_URL"
  key_links:
    - from: "debate/nodes/agents.py (Plan 02)"
      to: "debate/state.py"
      via: "from debate.state import AgentArgument, Concession"
    - from: "debate/graph.py (Plan 03)"
      to: "debate/state.py"
      via: "from debate.state import DebateState"
---

# Plan 01: Project Setup, State Schema, and Pydantic Models

## Objective

Create the project scaffolding and define all data contracts for Phase 1. This plan produces
the single source of truth for all schemas (`debate/state.py`) that Plans 02 and 03 import
from. It also installs dependencies and documents required environment variables.

No LLM calls are made in this plan. All outputs are verifiable with `python -c` imports
and `grep` checks — no running environment required beyond a successful `pip install`.

## Tasks

### Task 1: Install Dependencies and Create Project Structure

<read_first>
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/CLAUDE.md — version matrix (langgraph 1.1.9, langchain-anthropic 1.4.1, pydantic constraint)
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Environment Setup section (requirements.txt contents, install command, verify commands)
</read_first>

<action>
1. Create `requirements.txt` at project root with exactly these pinned versions:

```
langgraph==1.1.9
langchain-anthropic==1.4.1
langgraph-checkpoint-sqlite==3.0.3
```

   Do NOT add pydantic (pulled in by langgraph at >=2.7.4; 2.12.4 already installed).
   Do NOT add the `langchain` meta-package (only langchain-core and langchain-anthropic are needed).

2. Create `.env.example` at project root:

```
# corporate proxy — all three vars required. No ANTHROPIC_API_KEY needed.
ANTHROPIC_BASE_URL=https://your-proxy-base-url
ANTHROPIC_AUTH_TOKEN=your-auth-token
# Newline-separated "Key: Value" pairs, e.g.:
# ANTHROPIC_CUSTOM_HEADERS=X-Custom-Header: value\nX-Another: value2
ANTHROPIC_CUSTOM_HEADERS=
```

3. Create these empty `__init__.py` files to establish the Python package structure:
   - `debate/__init__.py`  (empty)
   - `debate/nodes/__init__.py`  (empty)
   - `tests/__init__.py`  (empty)

4. Run the install:
```bash
pip install -r /Users/maxinyue09/Downloads/projects/项目/debate-agent/requirements.txt
```

5. Verify install by running each of:
```bash
python -c "import langgraph; print(langgraph.__version__)"
python -c "import langchain_anthropic; print(langchain_anthropic.__version__)"
python -c "import pydantic; print(pydantic.VERSION)"
```
   Expected outputs: `1.1.9`, `1.4.1`, `2.12.4` (or higher for pydantic).
</action>

<acceptance_criteria>
- [ ] `cat requirements.txt` contains line `langgraph==1.1.9`
- [ ] `cat requirements.txt` contains line `langchain-anthropic==1.4.1`
- [ ] `cat .env.example` contains `ANTHROPIC_BASE_URL`
- [ ] `cat .env.example` contains `ANTHROPIC_CUSTOM_HEADERS`
- [ ] `python -c "import langgraph; print(langgraph.__version__)"` exits 0 and prints `1.1.9`
- [ ] `python -c "import langchain_anthropic"` exits 0
- [ ] `ls debate/__init__.py debate/nodes/__init__.py tests/__init__.py` all exist
</acceptance_criteria>

---

### Task 2: Write State Schema and Pydantic Models

<read_first>
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Pattern 1: DebateState TypedDict with Add Reducer (full code block)
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Anti-Patterns section (banned: `add_messages`, bare `list[AgentArgument]` without Annotated)
</read_first>

<action>
Create `debate/state.py` with the following exact implementation. Copy every field name, type
annotation, and comment verbatim — Plan 02 and Plan 03 import directly from this file.

```python
# debate/state.py
"""
Single source of truth for all debate graph schemas.

DebateState: The LangGraph state TypedDict. Only `current_round_arguments` uses
             the `add` reducer — required for fan-in from parallel Send dispatch.
             All other fields are last-write-wins (default).

AgentArgument: Pydantic model for one agent's output in one round.
Concession:    Pydantic model for a point yielded to another agent's argument.
RoundRecord:   Pydantic model collecting all three arguments from one round.
"""
from __future__ import annotations

from operator import add
from typing import Annotated, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Concession(BaseModel):
    """A point that an agent concedes in response to another agent's argument."""

    conceded_point: str = Field(description="The specific point being conceded")
    triggered_by_agent: str = Field(
        description="Role of the agent whose argument triggered this concession: "
                    "'optimist' | 'pessimist' | 'devil'"
    )
    triggered_by_claim: str = Field(
        description="The specific claim from the other agent that forced this concession"
    )
    rationale: str = Field(
        description="One-sentence explanation of why this point is conceded"
    )


class AgentArgument(BaseModel):
    """Structured output from one agent node in one debate round."""

    agent_role: str = Field(
        description="Role identifier: 'optimist' | 'pessimist' | 'devil'"
    )
    round_num: int = Field(description="0-indexed debate round number")
    position: str = Field(
        description="Main thesis in ONE sentence — no hedging, no 'however'"
    )
    reasoning: str = Field(
        description="Full argument prose supporting the position"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Self-reported confidence in the position, 0.0–1.0",
    )
    key_claims: list[str] = Field(
        min_length=3,
        description="3–7 short extractable claims (used for embedding in Phase 2)",
    )
    concessions: list[Concession] = Field(
        default_factory=list,
        description="Points conceded this round; empty in Round 1",
    )
    is_sentinel: bool = Field(
        default=False,
        description="True if this is a fallback object injected after 3 parse failures",
    )


class RoundRecord(BaseModel):
    """Container for all three agent arguments from a single debate round."""

    round_num: int = Field(description="0-indexed round number")
    arguments: list[AgentArgument] = Field(
        description="Exactly 3 AgentArguments: optimist, pessimist, devil"
    )


class DebateState(TypedDict, total=False):
    """
    LangGraph StateGraph state for the multi-agent debate system.

    IMPORTANT — fan-in accumulator:
        `current_round_arguments` uses `Annotated[list[AgentArgument], add]`.
        This is the ONLY field with a reducer. All parallel Send nodes append
        their single-item list here; LangGraph merges them automatically.
        Every other field is last-write-wins.

    Field lifecycle:
        topic, max_rounds  — set by caller at graph.invoke time
        debate_id          — set by initialize_node
        round_num          — initialized to 0 by initialize_node; incremented by collect_round1
        current_round_arguments — accumulates during fan-out; reset to [] by collect_round1
        round_history      — extended by collect_round1 after each round
        divergence_score, diverged_pairs — written by Phase 2 divergence node
        final_report       — written by Phase 3 synthesizer
        status             — "running" | "converged" | "max_rounds" | "complete"
    """

    # --- Input fields (set at invoke time) ---
    topic: str
    max_rounds: int

    # --- Metadata (set by initialize_node) ---
    debate_id: str

    # --- Round tracking ---
    round_num: int

    # --- Fan-in accumulator — ONLY this field uses a reducer ---
    current_round_arguments: Annotated[list[AgentArgument], add]

    # --- Full debate history ---
    round_history: list[RoundRecord]

    # --- Phase 2 fields (written by divergence detector; Phase 1 leaves at defaults) ---
    divergence_score: float
    diverged_pairs: list[tuple[str, str]]

    # --- Phase 3 fields ---
    final_report: Optional[object]  # DebateReport type added in Phase 3
    status: str
```

Key constraints that MUST be present (the executor verifying this plan will grep for them):
- `Annotated[list[AgentArgument], add]` — the fan-in reducer annotation
- `total=False` on DebateState — makes all fields optional at invocation time
- `is_sentinel: bool` field on AgentArgument — needed by Plan 02's retry wrapper
- `min_length=3` on `key_claims` — enforces at least 3 claims
- `ge=0.0` and `le=1.0` on `confidence` — enforces float range
</action>

<acceptance_criteria>
- [ ] `python -c "from debate.state import DebateState, AgentArgument, Concession, RoundRecord; print('OK')"` exits 0 and prints `OK`
- [ ] `grep "Annotated\[list\[AgentArgument\], add\]" debate/state.py` matches
- [ ] `grep "total=False" debate/state.py` matches
- [ ] `grep "is_sentinel" debate/state.py` matches
- [ ] `grep "min_length=3" debate/state.py` matches
- [ ] `grep "ge=0.0" debate/state.py` matches (confidence lower bound)
- [ ] `python -c "from debate.state import AgentArgument; a = AgentArgument(agent_role='optimist', round_num=0, position='p', reasoning='r', confidence=0.8, key_claims=['a','b','c']); print(a.is_sentinel)"` prints `False`
</acceptance_criteria>

## Verification

```bash
cd /Users/maxinyue09/Downloads/projects/项目/debate-agent

# Dependency install check
python -c "import langgraph; print('langgraph', langgraph.__version__)"
python -c "import langchain_anthropic; print('langchain_anthropic OK')"
python -c "import pydantic; print('pydantic', pydantic.VERSION)"

# Schema import check
python -c "from debate.state import DebateState, AgentArgument, Concession, RoundRecord; print('All schemas imported OK')"

# Structural checks
grep "Annotated\[list\[AgentArgument\], add\]" debate/state.py
grep "total=False" debate/state.py
grep "is_sentinel" debate/state.py
```

## must_haves

- `from debate.state import AgentArgument` succeeds and the model instantiates with required fields
- `DebateState` has `total=False` so `graph.invoke({"topic": ..., "max_rounds": 3})` works without supplying all fields
- `current_round_arguments` uses the `add` reducer — grep-verifiable in source
- `requirements.txt` is present and `pip install -r requirements.txt` installs langgraph 1.1.9 and langchain-anthropic 1.4.1
