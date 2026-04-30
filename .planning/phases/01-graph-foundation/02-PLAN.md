---
plan: 02
phase: 1
title: "LLM Helper, Persona Prompts, and Agent Nodes"
wave: 2
depends_on: [01]
requirements_addressed: [AGENT-01, AGENT-02, AGENT-03]
files_modified:
  - debate/llm.py
  - debate/prompts.py
  - debate/nodes/initialize.py
  - debate/nodes/dispatch.py
  - debate/nodes/agents.py
  - debate/nodes/collect.py
autonomous: true

must_haves:
  truths:
    - "All three agent node functions (optimist_node, pessimist_node, devil_node) are importable and each returns `{'current_round_arguments': [AgentArgument]}`"
    - "Each persona prompt contains a PROHIBITION block with explicit forbidden phrases — grep-verifiable"
    - "The retry wrapper returns a sentinel AgentArgument (is_sentinel=True) on total parse failure — never raises"
    - "`_make_llm()` reads ANTHROPIC_CUSTOM_HEADERS as newline-separated Key: Value pairs"
  artifacts:
    - path: "debate/llm.py"
      provides: "_make_llm() helper with proxy auth"
      exports: [_make_llm]
    - path: "debate/prompts.py"
      provides: "AGENT_PROMPTS dict keyed by role"
      contains: "PROHIBITION"
    - path: "debate/nodes/agents.py"
      provides: "optimist_node, pessimist_node, devil_node, _invoke_with_retry"
      exports: [optimist_node, pessimist_node, devil_node]
    - path: "debate/nodes/initialize.py"
      provides: "initialize_node — stamps debate_id, sets all DebateState defaults"
      exports: [initialize_node]
    - path: "debate/nodes/dispatch.py"
      provides: "dispatch_round1 — returns list[Send] for parallel fan-out"
      exports: [dispatch_round1]
    - path: "debate/nodes/collect.py"
      provides: "collect_round1 — moves accumulator to round_history, resets accumulator"
      exports: [collect_round1]
  key_links:
    - from: "debate/nodes/agents.py"
      to: "debate/llm.py"
      via: "from debate.llm import _make_llm"
    - from: "debate/nodes/agents.py"
      to: "debate/prompts.py"
      via: "from debate.prompts import AGENT_PROMPTS"
    - from: "debate/nodes/agents.py"
      to: "debate/state.py"
      via: "from debate.state import AgentArgument"
    - from: "debate/graph.py (Plan 03)"
      to: "debate/nodes/agents.py"
      via: "from debate.nodes.agents import optimist_node, pessimist_node, devil_node"
---

# Plan 02: LLM Helper, Persona Prompts, and Agent Nodes

## Objective

Build all six node implementations that the graph in Plan 03 wires together.
The critical outputs are: (1) `_make_llm()` with correct proxy-auth handling,
(2) methodology-based persona prompts with explicit PROHIBITION blocks satisfying AGENT-01
and AGENT-02, and (3) the retry wrapper with sentinel injection satisfying AGENT-03.

This plan makes no LLM calls (no live API needed). All verification is import-only.
The first actual LLM call happens in Plan 03's smoke test.

## Tasks

### Task 1: LLM Helper and Persona Prompts

<read_first>
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Pattern 5: _make_llm() Helper with Proxy Auth (exact code, ANTHROPIC_CUSTOM_HEADERS newline parsing)
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Persona Prompt Engineering section (Structural Persona Formula, Three Persona Templates, Anti-Sycophancy Instructions)
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/CLAUDE.md — model ID is `claude-sonnet-4-6`, NOT claude-sonnet-4-5
</read_first>

<action>
**File 1: `debate/llm.py`**

Create `debate/llm.py` with the proxy-aware `_make_llm()` helper. Copy this verbatim:

```python
# debate/llm.py
"""
Centralized LLM factory for all debate agent nodes.

The corporate proxy requires three env vars:
  ANTHROPIC_BASE_URL     — proxy endpoint (read automatically by anthropic SDK)
  ANTHROPIC_AUTH_TOKEN   — auth token (read automatically by anthropic SDK as api_key)
  ANTHROPIC_CUSTOM_HEADERS — newline-separated "Key: Value" pairs passed as default_headers

No ANTHROPIC_API_KEY is needed or used.
"""
import os

from langchain_anthropic import ChatAnthropic

MODEL_ID = "claude-sonnet-4-6"  # Locked in CLAUDE.md. Do NOT change to claude-sonnet-4-5.


def _make_llm() -> ChatAnthropic:
    """Return ChatAnthropic configured for the corporate proxy.

    Reads ANTHROPIC_CUSTOM_HEADERS as newline-separated 'Key: Value' pairs.
    ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN are picked up automatically
    by the underlying anthropic SDK from the environment.
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

**File 2: `debate/prompts.py`**

Create `debate/prompts.py` with methodology-based prompts. Each prompt MUST contain:
1. An analytical framework (numbered steps defining the decision procedure)
2. A reference persona (VC associate, risk manager, strategy consultant)
3. A PROHIBITION block with an explicit list of forbidden phrases
4. A terminal instruction "Your position must be a concrete [X] claim, not a hedge"

```python
# debate/prompts.py
"""
Persona prompt templates for the three debate agents.

Design principles:
  - Methodology-based, not intensity-based (AGENT-01)
    BAD: "You are very pessimistic."
    GOOD: "Your analytical framework is: 1. Identify failure mode..."
  - Explicit PROHIBITION blocks with named forbidden phrases (AGENT-02)
  - Terminal instruction requiring a concrete claim, not a hedge (AGENT-02)

AGENT_PROMPTS maps role → system prompt string.
Valid keys: "optimist", "pessimist", "devil"
"""

AGENT_PROMPTS: dict[str, str] = {
    "optimist": """You are the Opportunity Analyst. Your analytical framework is:
1. Enumerate the most compelling reasons this could succeed
2. Identify which conditions or trends favor success
3. Assess the magnitude of the upside if key assumptions hold
4. List 3-7 concrete opportunity claims as your key_claims

You analyze like a seed-stage VC associate evaluating portfolio fit: you are looking
for asymmetric upside, you know most ideas have some flaw, and your job is to find
the ones where the upside overwhelms the downside.

PROHIBITION: Do not mention risks, caveats, failure modes, or qualifications.
Do not write "however", "but", "although", "unless", "on the other hand",
"while there are risks", "balanced view", or "it depends".
If you find yourself writing a caveat, delete it.
Your position must be a concrete opportunity claim, not a hedge.

Maintain your analytical position unless presented with a logically superior argument.
Do not concede to avoid conflict.""",

    "pessimist": """You are the Risk Analyst. Your analytical framework is:
1. Identify the single most likely failure mode
2. Estimate its probability (high/medium/low) and impact severity
3. Assess whether the stated opportunity justifies accepting that specific risk
4. List 3-7 concrete risk claims as your key_claims

You analyze like a risk manager at a venture debt fund: you have seen many deals,
you know where execution usually breaks down, and you are paid to surface the failure
mode before the capital is deployed.

PROHIBITION: Do not mention upsides, opportunities, growth potential, or positive scenarios.
Do not write "however", "but there is potential", "while risky it could work",
"on the other hand", "balanced view", or "it depends".
Your position must be a concrete risk claim, not a hedge.

Maintain your analytical position unless presented with a logically superior argument.
Do not concede to avoid conflict.""",

    "devil": """You are the Challenger. Your analytical framework is:
1. Identify the current majority view or dominant argument being made
2. Find the most significant logical flaw, missing assumption, or overlooked factor
3. Construct the strongest possible counter-argument against the prevailing view
4. List 3-7 concrete challenge claims as your key_claims

You analyze like a senior strategy consultant who has heard this pitch three times
and found a specific flaw the presenter keeps glossing over.

PROHIBITION: Do not agree with the dominant view even partially.
Do not write "while this is a valid point", "I can see merits on both sides",
"I agree that", "on the other hand", "balanced view", or "it depends".
Your position must directly challenge the prevailing view with specific evidence or logic.

Maintain your analytical position unless presented with a logically superior argument.
Do not concede to avoid conflict.""",
}
```

Verify `AGENT_PROMPTS` has exactly three keys after writing the file.
</action>

<acceptance_criteria>
- [ ] `python -c "from debate.llm import _make_llm; print('OK')"` exits 0 and prints `OK`
- [ ] `grep "MODEL_ID = \"claude-sonnet-4-6\"" debate/llm.py` matches
- [ ] `grep "ANTHROPIC_CUSTOM_HEADERS" debate/llm.py` matches
- [ ] `grep "split.*\\\\n" debate/llm.py` OR `grep 'split("\\\\n")' debate/llm.py` matches (newline parsing)
- [ ] `python -c "from debate.prompts import AGENT_PROMPTS; assert set(AGENT_PROMPTS) == {'optimist','pessimist','devil'}; print('OK')"` prints `OK`
- [ ] `grep "PROHIBITION" debate/prompts.py` returns at least 3 matches (one per agent)
- [ ] `grep "not a hedge" debate/prompts.py` returns at least 2 matches
- [ ] `grep "Do not concede to avoid conflict" debate/prompts.py` returns 3 matches
</acceptance_criteria>

---

### Task 2: Graph Nodes — Initialize, Dispatch, Agents, Collect

<read_first>
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Pattern 2: Send Fan-Out from dispatch_round1 (exact code for dispatch_round1)
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Pattern 3: Agent Node with with_structured_output + Retry Wrapper (exact code)
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Pattern 4: Collect Node (Fan-In) (exact code)
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Pattern 6: Initialize Node (exact code)
- /Users/maxinyue09/Downloads/projects/项目/debate-agent/.planning/phases/01-graph-foundation/01-RESEARCH.md — Anti-Patterns (banned: full DebateState in Send payload, bracket access on optional fields)
</read_first>

<action>
Create four files in `debate/nodes/`.

**File 1: `debate/nodes/initialize.py`**

```python
# debate/nodes/initialize.py
"""
Initialize node: stamps metadata and sets all DebateState defaults.

Must run before any other node. Sets every field that downstream nodes read
via state.get("field", default) to an explicit value, preventing KeyError.
"""
import uuid

from debate.state import DebateState


def initialize_node(state: DebateState) -> dict:
    """Stamp debate_id and initialize all counters and accumulators."""
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

**File 2: `debate/nodes/dispatch.py`**

```python
# debate/nodes/dispatch.py
"""
Dispatch node: returns list[Send] to fan out the three agent nodes in parallel.

CRITICAL: Each Send payload contains ONLY the fields the agent needs.
Do NOT pass the full DebateState — that would expose round_history from
prior rounds, violating Round 1 isolation (PITFALL 4 in RESEARCH.md).
"""
from langgraph.types import Send

from debate.state import DebateState


def dispatch_round1(state: DebateState) -> list[Send]:
    """Fan-out: dispatch 3 agents in parallel with no cross-visibility."""
    topic = state.get("topic", "")
    round_num = state.get("round_num", 0)
    return [
        Send(
            "optimist_node",
            {"topic": topic, "agent_role": "optimist", "prior_arguments": [], "round_num": round_num},
        ),
        Send(
            "pessimist_node",
            {"topic": topic, "agent_role": "pessimist", "prior_arguments": [], "round_num": round_num},
        ),
        Send(
            "devil_node",
            {"topic": topic, "agent_role": "devil", "prior_arguments": [], "round_num": round_num},
        ),
    ]
```

**File 3: `debate/nodes/agents.py`**

```python
# debate/nodes/agents.py
"""
Agent node implementations: optimist_node, pessimist_node, devil_node.

All three share the same _agent_node() implementation — only the role differs.
The retry wrapper (_invoke_with_retry) uses include_raw=True to avoid unhandled
ValidationError crashes. On third failure it injects a sentinel AgentArgument
with is_sentinel=True rather than raising.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from debate.llm import _make_llm
from debate.prompts import AGENT_PROMPTS
from debate.state import AgentArgument

_SENTINEL_TEMPLATE = AgentArgument(
    agent_role="unknown",
    round_num=-1,
    position="[Analysis unavailable due to validation error]",
    reasoning=(
        "The LLM returned a response that could not be parsed into the required "
        "schema after 3 attempts."
    ),
    confidence=0.0,
    key_claims=["validation_error", "sentinel_injected", "no_data"],
    concessions=[],
    is_sentinel=True,
)


def _invoke_with_retry(
    llm,
    messages: list,
    role: str,
    round_num: int,
    max_retries: int = 2,
) -> AgentArgument:
    """Call with_structured_output up to max_retries+1 times.

    Uses include_raw=True so parse failures surface as result["parsed"] is None
    rather than raising ValidationError. Returns a sentinel on total failure.
    """
    structured_llm = llm.with_structured_output(AgentArgument, include_raw=True)
    for attempt in range(max_retries + 1):
        result = structured_llm.invoke(messages)
        if result.get("parsed") is not None:
            parsed: AgentArgument = result["parsed"]
            # Ensure role is correct — LLM may hallucinate a different role string
            parsed.agent_role = role
            parsed.round_num = round_num
            return parsed
        print(
            f"[{role}] Pydantic parse failed "
            f"(attempt {attempt + 1}/{max_retries + 1}): "
            f"{result.get('parsing_error')}"
        )
    # All retries exhausted — inject sentinel
    sentinel = _SENTINEL_TEMPLATE.model_copy(deep=True)
    sentinel.agent_role = role
    sentinel.round_num = round_num
    return sentinel


def _agent_node(state: dict, role: str) -> dict:
    """Shared implementation for all three agent nodes.

    Receives a minimal Send payload (not the full DebateState):
        {"topic": str, "agent_role": str, "prior_arguments": [], "round_num": int}

    Returns a single-item list in current_round_arguments so the `add` reducer
    appends it to the accumulator without overwriting the other agents' results.
    """
    topic = state.get("topic", "")
    round_num = state.get("round_num", 0)
    system_prompt = AGENT_PROMPTS[role]

    llm = _make_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Topic for analysis: {topic}\n\nRound: {round_num + 1}"),
    ]
    argument = _invoke_with_retry(llm, messages, role, round_num)
    # Return as single-item list — add reducer appends, not overwrites
    return {"current_round_arguments": [argument]}


def optimist_node(state: dict) -> dict:
    return _agent_node(state, "optimist")


def pessimist_node(state: dict) -> dict:
    return _agent_node(state, "pessimist")


def devil_node(state: dict) -> dict:
    return _agent_node(state, "devil")
```

**File 4: `debate/nodes/collect.py`**

```python
# debate/nodes/collect.py
"""
Collect node: fan-in after all three agent nodes complete.

Moves current_round_arguments into round_history as a RoundRecord,
then resets the accumulator to [] so Phase 2's rebuttal rounds start clean.
"""
from debate.state import DebateState, RoundRecord


def collect_round1(state: DebateState) -> dict:
    """Aggregate all three agent arguments from the current round into history."""
    current_args = state.get("current_round_arguments", [])
    round_num = state.get("round_num", 0)

    new_record = RoundRecord(
        round_num=round_num,
        arguments=current_args,
    )
    return {
        "round_history": state.get("round_history", []) + [new_record],
        "current_round_arguments": [],  # reset accumulator for next round
        "round_num": round_num + 1,
        "status": "running",
    }
```
</action>

<acceptance_criteria>
- [ ] `python -c "from debate.nodes.initialize import initialize_node; print('OK')"` exits 0
- [ ] `python -c "from debate.nodes.dispatch import dispatch_round1; print('OK')"` exits 0
- [ ] `python -c "from debate.nodes.agents import optimist_node, pessimist_node, devil_node; print('OK')"` exits 0
- [ ] `python -c "from debate.nodes.collect import collect_round1; print('OK')"` exits 0
- [ ] `grep "include_raw=True" debate/nodes/agents.py` matches
- [ ] `grep "is_sentinel=True" debate/nodes/agents.py` matches (sentinel object)
- [ ] `grep "model_copy" debate/nodes/agents.py` matches (sentinel copy, not mutation of template)
- [ ] `grep "current_round_arguments.*\[argument\]" debate/nodes/agents.py` matches (single-item list)
- [ ] `grep "Send(" debate/nodes/dispatch.py` returns 3 matches
- [ ] `grep "prior_arguments.*\[\]" debate/nodes/dispatch.py` matches (no cross-contamination)
- [ ] `python -c "from debate.nodes.agents import _SENTINEL_TEMPLATE; assert _SENTINEL_TEMPLATE.is_sentinel is True; print('OK')"` prints `OK`
</acceptance_criteria>

## Verification

```bash
cd /Users/maxinyue09/Downloads/projects/项目/debate-agent

# All imports
python -c "from debate.llm import _make_llm; print('llm OK')"
python -c "from debate.prompts import AGENT_PROMPTS; print('prompts OK, keys:', list(AGENT_PROMPTS))"
python -c "from debate.nodes.initialize import initialize_node; from debate.nodes.dispatch import dispatch_round1; from debate.nodes.agents import optimist_node, pessimist_node, devil_node; from debate.nodes.collect import collect_round1; print('all nodes OK')"

# Structural integrity
grep "PROHIBITION" debate/prompts.py | wc -l   # expect: 3
grep "Do not concede to avoid conflict" debate/prompts.py | wc -l   # expect: 3
grep "include_raw=True" debate/nodes/agents.py
grep "is_sentinel=True" debate/nodes/agents.py
grep "Send(" debate/nodes/dispatch.py | wc -l   # expect: 3

# Sentinel test (no LLM call)
python -c "
from debate.nodes.agents import _SENTINEL_TEMPLATE
assert _SENTINEL_TEMPLATE.is_sentinel is True
assert _SENTINEL_TEMPLATE.confidence == 0.0
assert len(_SENTINEL_TEMPLATE.key_claims) >= 3
print('Sentinel object OK')
"
```

## must_haves

- All six node functions import without error from their respective modules
- `debate/prompts.py` has 3 PROHIBITION blocks and 3 "Do not concede" instructions — grep-verifiable
- `debate/nodes/agents.py` uses `include_raw=True` on `with_structured_output` — grep-verifiable
- Sentinel `AgentArgument` has `is_sentinel=True` and `confidence=0.0` — verifiable without LLM call
- `dispatch_round1` sends `prior_arguments=[]` in each Send payload — Round 1 isolation maintained
