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
