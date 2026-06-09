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
    rather than raising ValidationError. Handles rate limits with backoff.
    Returns a sentinel on total failure.
    """
    import time

    structured_llm = llm.with_structured_output(AgentArgument, include_raw=True)
    for attempt in range(max_retries + 1):
        try:
            result = structured_llm.invoke(messages)
        except Exception as e:
            err = str(e).lower()
            if "rate" in err or "429" in err or "limit" in err:
                wait = 60 * (attempt + 1)
                print(f"[{role}] rate limit (attempt {attempt + 1}), waiting {wait}s...")
                time.sleep(wait)
                continue
            raise  # re-raise non-rate-limit errors

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

    Round 1 (round_num == 0): receives minimal payload with no prior arguments.
    Rebuttal rounds (round_num > 0): receives compact opposing summaries and
    appends concession instructions to the human message (DEBATE-07).

    Returns a single-item list in current_round_arguments so the `add` reducer
    appends it to the accumulator without overwriting the other agents' results.
    """
    topic = state.get("topic", "")
    round_num = state.get("round_num", 0)
    prior_arguments = state.get("prior_arguments", [])
    system_prompt = AGENT_PROMPTS[role]

    # Base human message — same for all rounds
    human_content = f"Topic for analysis: {topic}\n\nRound: {round_num + 1}"

    # Rebuttal rounds: inject full debate history and concession instructions
    if prior_arguments and round_num > 0:
        # Group all summaries by round_num
        rounds: dict[int, dict[str, dict]] = {}
        for s in prior_arguments:
            rnd = s["round_num"]
            if rnd not in rounds:
                rounds[rnd] = {}
            rounds[rnd][s["agent_role"]] = s

        # Own position history — lets agent stay consistent across rounds
        own_rounds = sorted(r for r in rounds if role in rounds[r])
        if own_rounds:
            human_content += "\n\n--- Your position history ---"
            for rnd in own_rounds:
                own = rounds[rnd][role]
                human_content += f"\n\n[YOUR POSITION — Round {rnd + 1}]"
                human_content += f"\nPosition: {own['position']}"

        # Opponent arguments from all rounds, grouped by round
        human_content += "\n\n--- Full debate history (opponents) ---"
        for rnd in sorted(rounds.keys()):
            opponent_args = [
                rounds[rnd][r] for r in rounds[rnd] if r != role
            ]
            if not opponent_args:
                continue
            human_content += f"\n\n== Round {rnd + 1} =="
            for arg_summary in opponent_args:
                claims_text = "\n".join(
                    f"  - {c}" for c in arg_summary.get("key_claims", [])
                )
                human_content += (
                    f"\n\n[{arg_summary['agent_role'].upper()}]"
                    f"\nPosition: {arg_summary['position']}"
                    f"\nKey claims:\n{claims_text}"
                    f"\nConfidence: {arg_summary['confidence']:.0%}"
                )

        human_content += (
            "\n\n--- Rebuttal instructions ---"
            "\nRebut the opposing arguments above. Maintain your analytical stance."
            "\nYou can see the full debate history — if an opponent dodged one of"
            " your earlier arguments, call it out explicitly."
            "\nIf (and ONLY if) an opponent's specific claim is logically superior,"
            " record it in your concessions list with:"
            "\n  triggered_by_agent: the opponent's role (e.g., 'pessimist')"
            "\n  triggered_by_claim: copy the EXACT claim text shown above"
            "\n  conceded_point: what specific position you are yielding"
            "\n  rationale: one sentence explaining why this argument is superior"
            "\nDo NOT concede to avoid conflict or to appear balanced."
            " Only concede on logical grounds."
        )

    llm = _make_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]
    argument = _invoke_with_retry(llm, messages, role, round_num)
    return {"current_round_arguments": [argument]}


def optimist_node(state: dict) -> dict:
    return _agent_node(state, "optimist")


def pessimist_node(state: dict) -> dict:
    return _agent_node(state, "pessimist")


def devil_node(state: dict) -> dict:
    return _agent_node(state, "devil")
