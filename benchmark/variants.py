# benchmark/variants.py
"""
Ablation variant graph builders for experiment B.

Each function builds and compiles a modified LangGraph StateGraph that differs
from the full system in exactly ONE component. This lets us measure each design
decision's individual contribution.

Variants:
  full_system         — Baseline. Uses current prompts (fixed devil: challenges shared assumptions).
  original_devil      — Devil uses old "challenge dominant view" prompt (pre-fix baseline).
  no_prohibition      — PROHIBITION blocks removed from all agent prompts.
  sequential          — Round 1 agents run in series (optimist → pessimist → devil);
                        later agents see earlier agents' Round 1 output (anchoring test).
  fixed_rounds        — Semantic divergence termination disabled; always runs max_rounds.
  fulltext_embedding  — Divergence computed on full reasoning text, not key_claims.
  nli_detection       — Divergence detected via NLI contradiction instead of cosine similarity.

Usage:
    from benchmark.variants import build_variant_graph
    graph = build_variant_graph("no_prohibition")
    report = graph.invoke({"topic": "...", "max_rounds": 3}, config={"configurable": {"thread_id": "x"}})
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from itertools import combinations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from debate.divergence import _get_model, DIVERGE_THRESHOLD
from debate.llm import _make_llm
from debate.nodes.agents import _invoke_with_retry
from debate.nodes.collect import collect_round1
from debate.nodes.dispatch import _build_compact_summaries
from debate.nodes.divergence_check import divergence_check_node
from debate.nodes.initialize import initialize_node
from debate.nodes.save import save_node
from debate.nodes.synthesize import synthesize_stub
from debate.prompts import AGENT_PROMPTS
from debate.state import AgentArgument, DebateState


# ---------------------------------------------------------------------------
# Helper: strip PROHIBITION blocks from a prompt
# ---------------------------------------------------------------------------

def _strip_prohibition(prompt: str) -> str:
    """Remove the PROHIBITION paragraph and everything after 'PROHIBITION:' until
    the next blank line (or end of string). Used for the no_prohibition variant."""
    # Match "PROHIBITION: ..." block (multi-line until blank line or end)
    cleaned = re.sub(
        r"PROHIBITION:.*?(?=\n\n|\Z)",
        "",
        prompt,
        flags=re.DOTALL,
    )
    # Collapse multiple blank lines left behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


_PROMPTS_NO_PROHIBITION: dict[str, str] = {
    role: _strip_prohibition(prompt)
    for role, prompt in AGENT_PROMPTS.items()
}

# Original devil prompt (pre-fix): "challenge the dominant view"
# Kept here for ablation comparison vs the fixed "challenge shared assumptions" prompt.
_DEVIL_PROMPT_ORIGINAL = """You are the Challenger. Your analytical framework is:
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
Do not concede to avoid conflict."""


# ---------------------------------------------------------------------------
# Variant node factories
# ---------------------------------------------------------------------------

# --- no_prohibition agent node ---

def _agent_node_no_prohibition(state: dict, role: str) -> dict:
    """Agent node that uses PROHIBITION-free prompts."""
    topic = state.get("topic", "")
    round_num = state.get("round_num", 0)
    prior_arguments = state.get("prior_arguments", [])
    system_prompt = _PROMPTS_NO_PROHIBITION[role]

    human_content = f"Topic for analysis: {topic}\n\nRound: {round_num + 1}"

    if prior_arguments and round_num > 0:
        human_content += "\n\n--- Opposing arguments from the previous round ---"
        for arg_summary in prior_arguments:
            if arg_summary.get("agent_role") == role:
                continue
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
            "\nIf (and ONLY if) an opponent's specific claim is logically superior,"
            " record it in your concessions list."
            "\nDo NOT concede to avoid conflict. Only concede on logical grounds."
        )

    llm = _make_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]
    argument = _invoke_with_retry(llm, messages, role, round_num)
    return {"current_round_arguments": [argument]}


def optimist_node_no_prohibition(state: dict) -> dict:
    return _agent_node_no_prohibition(state, "optimist")


def pessimist_node_no_prohibition(state: dict) -> dict:
    return _agent_node_no_prohibition(state, "pessimist")


def devil_node_no_prohibition(state: dict) -> dict:
    return _agent_node_no_prohibition(state, "devil")


# --- original_devil: uses old "challenge dominant view" prompt ---

def _agent_node_original_devil(state: dict, role: str) -> dict:
    """Agent node that uses the original devil prompt for the devil role,
    current prompts for optimist/pessimist."""
    from debate.prompts import AGENT_PROMPTS as CURRENT_PROMPTS
    topic = state.get("topic", "")
    round_num = state.get("round_num", 0)
    prior_arguments = state.get("prior_arguments", [])

    if role == "devil":
        system_prompt = _DEVIL_PROMPT_ORIGINAL
    else:
        system_prompt = CURRENT_PROMPTS[role]

    human_content = f"Topic for analysis: {topic}\n\nRound: {round_num + 1}"

    if prior_arguments and round_num > 0:
        human_content += "\n\n--- Opposing arguments from the previous round ---"
        for arg_summary in prior_arguments:
            if arg_summary.get("agent_role") == role:
                continue
            claims_text = "\n".join(f"  - {c}" for c in arg_summary.get("key_claims", []))
            human_content += (
                f"\n\n[{arg_summary['agent_role'].upper()}]"
                f"\nPosition: {arg_summary['position']}"
                f"\nKey claims:\n{claims_text}"
                f"\nConfidence: {arg_summary['confidence']:.0%}"
            )
        human_content += (
            "\n\n--- Rebuttal instructions ---"
            "\nRebut the opposing arguments above. Maintain your analytical stance."
            "\nIf (and ONLY if) an opponent's specific claim is logically superior,"
            " record it in your concessions list."
            "\nDo NOT concede to avoid conflict. Only concede on logical grounds."
        )

    llm = _make_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]
    argument = _invoke_with_retry(llm, messages, role, round_num)
    return {"current_round_arguments": [argument]}


def optimist_node_original_devil(state: dict) -> dict:
    return _agent_node_original_devil(state, "optimist")


def pessimist_node_original_devil(state: dict) -> dict:
    return _agent_node_original_devil(state, "pessimist")


def devil_node_original_devil(state: dict) -> dict:
    return _agent_node_original_devil(state, "devil")


# --- sequential Round 1 agent node ---

def _agent_node_sequential(state: dict, role: str) -> dict:
    """Agent node for sequential Round 1.

    In Round 1: reads from current_round_arguments to see prior agents' outputs
    (same round), breaking the cognitive isolation that the full system enforces.
    This tests whether parallel fan-out actually prevents anchoring.

    In rebuttal rounds: same as the standard node (uses prior_arguments).
    """
    from debate.prompts import AGENT_PROMPTS as ORIG_PROMPTS
    topic = state.get("topic", "")
    round_num = state.get("round_num", 0)
    prior_arguments = state.get("prior_arguments", [])
    system_prompt = ORIG_PROMPTS[role]

    human_content = f"Topic for analysis: {topic}\n\nRound: {round_num + 1}"

    if round_num == 0:
        # Sequential Round 1: inject same-round prior agents' outputs
        already_argued: list[AgentArgument] = state.get("current_round_arguments", [])
        prior_same_round = [
            a for a in already_argued if not a.is_sentinel
        ]
        if prior_same_round:
            human_content += (
                "\n\n--- Earlier analyses from this round (other agents) ---"
            )
            for prior_arg in prior_same_round:
                claims_text = "\n".join(
                    f"  - {c}" for c in prior_arg.key_claims[:3]
                )
                human_content += (
                    f"\n\n[{prior_arg.agent_role.upper()}]"
                    f"\nPosition: {prior_arg.position}"
                    f"\nKey claims:\n{claims_text}"
                )
    elif prior_arguments:
        # Rebuttal rounds: standard logic
        human_content += "\n\n--- Opposing arguments from the previous round ---"
        for arg_summary in prior_arguments:
            if arg_summary.get("agent_role") == role:
                continue
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
            "\nIf (and ONLY if) an opponent's specific claim is logically superior,"
            " record it in your concessions list."
            "\nDo NOT concede to avoid conflict. Only concede on logical grounds."
        )

    llm = _make_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]
    argument = _invoke_with_retry(llm, messages, role, round_num)
    return {"current_round_arguments": [argument]}


def optimist_node_sequential(state: dict) -> dict:
    return _agent_node_sequential(state, "optimist")


def pessimist_node_sequential(state: dict) -> dict:
    return _agent_node_sequential(state, "pessimist")


def devil_node_sequential(state: dict) -> dict:
    return _agent_node_sequential(state, "devil")


def dispatch_round1_sequential(state: DebateState) -> list[Send]:
    """Sequential Round 1: dispatch optimist only in initial fan-out.

    Pessimist and devil run as chained edges, each reading from
    current_round_arguments for prior context (see the graph structure below).
    Only optimist uses Send; pessimist/devil are direct edges.
    """
    topic = state.get("topic", "")
    round_num = state.get("round_num", 0)
    return [
        Send(
            "optimist_seq",
            {"topic": topic, "agent_role": "optimist",
             "prior_arguments": [], "round_num": round_num,
             "current_round_arguments": []},
        ),
    ]


# --- fixed_rounds: override route_divergence to always rebuttal ---

def route_divergence_fixed(state: DebateState):
    """Fixed-rounds variant: always run max_rounds, ignore divergence score."""
    round_num = state.get("round_num", 0)
    max_rounds = state.get("max_rounds", 3)
    topic = state.get("topic", "")
    round_history = state.get("round_history", [])

    if round_num >= max_rounds:
        return "synthesize_stub"

    # Always rebuttal regardless of divergence_score
    compact_summaries = _build_compact_summaries(round_history)
    return [
        Send("optimist_node", {"topic": topic, "agent_role": "optimist",
                               "prior_arguments": compact_summaries, "round_num": round_num}),
        Send("pessimist_node", {"topic": topic, "agent_role": "pessimist",
                                "prior_arguments": compact_summaries, "round_num": round_num}),
        Send("devil_node", {"topic": topic, "agent_role": "devil",
                            "prior_arguments": compact_summaries, "round_num": round_num}),
    ]


# --- fulltext_embedding: compute divergence on reasoning text, not key_claims ---

def divergence_check_node_fulltext(state: DebateState) -> dict:
    """Divergence check using full reasoning text instead of key_claims.

    This tests the design decision in divergence.py to embed key_claims
    rather than full text. Full-text embedding compresses semantic distance
    because all agents discuss the same topic — this should produce more
    false convergences (lower divergence scores when agents still disagree).
    """
    round_history = state.get("round_history", [])
    if not round_history:
        return {"divergence_score": 1.0, "diverged_pairs": []}

    latest_round = round_history[-1]
    arguments = [a for a in latest_round.arguments if not a.is_sentinel]

    if len(arguments) < 2:
        return {"divergence_score": 0.0, "diverged_pairs": []}

    model = _get_model()
    pairwise_max_sims: list[float] = []
    diverged_pairs: list[tuple[str, str]] = []

    for arg_a, arg_b in combinations(arguments, 2):
        # Use reasoning text instead of key_claims
        texts_a = [arg_a.reasoning] if arg_a.reasoning else []
        texts_b = [arg_b.reasoning] if arg_b.reasoning else []

        if not texts_a or not texts_b:
            pairwise_max_sims.append(1.0)
            continue

        all_texts = texts_a + texts_b
        embeddings = model.encode(all_texts, normalize_embeddings=True)
        emb_a = embeddings[: len(texts_a)]
        emb_b = embeddings[len(texts_a):]
        sim_matrix = emb_a @ emb_b.T
        max_sim = float(sim_matrix.max())
        pairwise_max_sims.append(max_sim)

        if max_sim < DIVERGE_THRESHOLD:
            diverged_pairs.append((arg_a.agent_role, arg_b.agent_role))
        else:
            diverged_pairs.append((arg_a.agent_role, arg_b.agent_role))

    score = 1.0 - (sum(pairwise_max_sims) / len(pairwise_max_sims)) if pairwise_max_sims else 0.0

    updated_record = latest_round.model_copy(update={"divergence_score": score})
    updated_history = list(round_history[:-1]) + [updated_record]

    return {
        "divergence_score": round(score, 4),
        "diverged_pairs": diverged_pairs,
        "round_history": updated_history,
    }


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def _base_graph_nodes(builder: StateGraph, use_nodes: dict) -> None:
    """Register standard nodes with optional overrides."""
    from debate.nodes.agents import optimist_node, pessimist_node, devil_node

    builder.add_node("initialize", initialize_node)
    builder.add_node("optimist_node", use_nodes.get("optimist_node", optimist_node))
    builder.add_node("pessimist_node", use_nodes.get("pessimist_node", pessimist_node))
    builder.add_node("devil_node", use_nodes.get("devil_node", devil_node))
    builder.add_node("collect_round1", collect_round1)
    builder.add_node("divergence_check_node",
                     use_nodes.get("divergence_check_node", divergence_check_node))
    builder.add_node("synthesize_stub", synthesize_stub)
    builder.add_node("save_node", save_node)


def _standard_edges(builder: StateGraph, dispatch_fn, route_fn) -> None:
    """Wire standard edges for all non-sequential variants."""
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", dispatch_fn)
    builder.add_edge("optimist_node", "collect_round1")
    builder.add_edge("pessimist_node", "collect_round1")
    builder.add_edge("devil_node", "collect_round1")
    builder.add_edge("collect_round1", "divergence_check_node")
    builder.add_conditional_edges("divergence_check_node", route_fn)
    builder.add_edge("synthesize_stub", "save_node")
    builder.add_edge("save_node", END)


def build_full_system_graph():
    """Unmodified full debate graph."""
    from debate.nodes.agents import optimist_node, pessimist_node, devil_node
    from debate.nodes.dispatch import dispatch_round1, route_divergence

    builder = StateGraph(DebateState)
    _base_graph_nodes(builder, {})
    _standard_edges(builder, dispatch_round1, route_divergence)
    return builder.compile(checkpointer=InMemorySaver())


def build_no_prohibition_graph():
    """No-PROHIBITION variant: agents lack the forbidden-phrase constraints."""
    from debate.nodes.dispatch import dispatch_round1, route_divergence

    builder = StateGraph(DebateState)
    _base_graph_nodes(builder, {
        "optimist_node": optimist_node_no_prohibition,
        "pessimist_node": pessimist_node_no_prohibition,
        "devil_node": devil_node_no_prohibition,
    })
    _standard_edges(builder, dispatch_round1, route_divergence)
    return builder.compile(checkpointer=InMemorySaver())


def build_sequential_graph():
    """Sequential Round 1 variant: agents run in order, each seeing prior outputs."""
    from debate.nodes.dispatch import route_divergence

    builder = StateGraph(DebateState)
    builder.add_node("initialize", initialize_node)

    # Sequential Round 1 nodes (different names to avoid conflicts)
    builder.add_node("optimist_seq", optimist_node_sequential)
    builder.add_node("pessimist_seq", pessimist_node_sequential)
    builder.add_node("devil_seq", devil_node_sequential)

    # Rebuttal rounds reuse standard nodes
    from debate.nodes.agents import optimist_node, pessimist_node, devil_node
    builder.add_node("optimist_node", optimist_node)
    builder.add_node("pessimist_node", pessimist_node)
    builder.add_node("devil_node", devil_node)

    builder.add_node("collect_round1", collect_round1)
    builder.add_node("divergence_check_node", divergence_check_node)
    builder.add_node("synthesize_stub", synthesize_stub)
    builder.add_node("save_node", save_node)

    # Round 1: chain in sequence
    builder.add_edge(START, "initialize")
    builder.add_conditional_edges("initialize", dispatch_round1_sequential)
    builder.add_edge("optimist_seq", "pessimist_seq")
    builder.add_edge("pessimist_seq", "devil_seq")
    builder.add_edge("devil_seq", "collect_round1")

    # Rebuttal rounds: standard fan-out
    builder.add_edge("collect_round1", "divergence_check_node")
    builder.add_conditional_edges("divergence_check_node", route_divergence)
    builder.add_edge("optimist_node", "collect_round1")
    builder.add_edge("pessimist_node", "collect_round1")
    builder.add_edge("devil_node", "collect_round1")

    builder.add_edge("synthesize_stub", "save_node")
    builder.add_edge("save_node", END)
    return builder.compile(checkpointer=InMemorySaver())


def build_fixed_rounds_graph():
    """Fixed-rounds variant: semantic termination disabled, always runs max_rounds."""
    from debate.nodes.dispatch import dispatch_round1

    builder = StateGraph(DebateState)
    _base_graph_nodes(builder, {})
    _standard_edges(builder, dispatch_round1, route_divergence_fixed)
    return builder.compile(checkpointer=InMemorySaver())


def build_fulltext_graph():
    """Full-text embedding variant: divergence computed on reasoning, not key_claims."""
    from debate.nodes.dispatch import dispatch_round1, route_divergence

    builder = StateGraph(DebateState)
    _base_graph_nodes(builder, {
        "divergence_check_node": divergence_check_node_fulltext,
    })
    _standard_edges(builder, dispatch_round1, route_divergence)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# NLI-based divergence detection variant
# ---------------------------------------------------------------------------

def divergence_check_node_nli(state: DebateState) -> dict:
    """Divergence check using NLI cross-encoder instead of cosine similarity.

    Uses compute_divergence_nli() which classifies claim pairs as
    CONTRADICTION / ENTAILMENT / NEUTRAL. High CONTRADICTION probability
    indicates genuine stance opposition even when vocabulary overlaps.

    This fixes the core flaw of cosine similarity: 'VC is good' and
    'VC is bad' score as similar under cosine, but NLI correctly flags
    them as CONTRADICTION.
    """
    from debate.divergence import compute_divergence_nli

    round_history = state.get("round_history", [])
    if not round_history:
        return {"divergence_score": 1.0, "diverged_pairs": []}

    latest_round = round_history[-1]
    score, diverged_pairs = compute_divergence_nli(latest_round.arguments)

    updated_record = latest_round.model_copy(update={"divergence_score": score})
    updated_history = list(round_history[:-1]) + [updated_record]

    return {
        "divergence_score": score,
        "diverged_pairs": diverged_pairs,
        "round_history": updated_history,
    }


def route_divergence_nli(state: DebateState):
    """Routing function for NLI variant — adaptive convergence with 4 guards.

    Uses NLI_CONTRADICTION_THRESHOLD (0.5) for Guard 2 (genuine convergence)
    instead of cosine DIVERGE_THRESHOLD (0.75). Guards 1/3/4 are identical
    to the cosine route_divergence.
    """
    from debate.divergence import (
        ABSOLUTE_MAX_ROUNDS,
        NLI_CONTRADICTION_THRESHOLD,
        PLATEAU_DELTA,
        PLATEAU_MIN_ROUNDS,
    )

    round_num = state.get("round_num", 0)
    divergence_score = state.get("divergence_score", 0.0)
    topic = state.get("topic", "")
    round_history = state.get("round_history", [])

    # Guard 1: absolute safety cap
    if round_num >= ABSOLUTE_MAX_ROUNDS:
        return "synthesize_stub"

    # Guard 2: genuine convergence (NLI threshold)
    if divergence_score < NLI_CONTRADICTION_THRESHOLD:
        return "synthesize_stub"

    # Guard 3: score plateau
    if len(round_history) >= PLATEAU_MIN_ROUNDS:
        prev_score = round_history[-2].divergence_score
        curr_score = round_history[-1].divergence_score
        if abs(prev_score - curr_score) < PLATEAU_DELTA:
            return "synthesize_stub"

    # Guard 4: no concessions last round
    if len(round_history) >= 2:
        last_round = round_history[-1]
        if sum(len(arg.concessions) for arg in last_round.arguments) == 0:
            return "synthesize_stub"

    compact_summaries = _build_compact_summaries(round_history)
    return [
        Send("optimist_node", {"topic": topic, "agent_role": "optimist",
                               "prior_arguments": compact_summaries, "round_num": round_num}),
        Send("pessimist_node", {"topic": topic, "agent_role": "pessimist",
                                "prior_arguments": compact_summaries, "round_num": round_num}),
        Send("devil_node", {"topic": topic, "agent_role": "devil",
                            "prior_arguments": compact_summaries, "round_num": round_num}),
    ]


def build_original_devil_graph():
    """Original devil variant: devil uses old 'challenge dominant view' prompt.

    Pre-fix baseline for comparing PDS before/after the devil prompt redesign.
    The old prompt caused devil to align with pessimist (both opposing optimist),
    reducing position diversity. The fixed prompt in full_system challenges shared
    assumptions instead, creating a genuinely orthogonal third voice.
    """
    from debate.nodes.dispatch import dispatch_round1, route_divergence

    builder = StateGraph(DebateState)
    _base_graph_nodes(builder, {
        "optimist_node": optimist_node_original_devil,
        "pessimist_node": pessimist_node_original_devil,
        "devil_node": devil_node_original_devil,
    })
    _standard_edges(builder, dispatch_round1, route_divergence)
    return builder.compile(checkpointer=InMemorySaver())


def build_nli_detection_graph():
    """NLI-based divergence detection variant.

    Replaces cosine similarity with cross-encoder NLI (contradiction detection).
    Expected to trigger multi-round debates that the cosine variant falsely
    terminates after Round 1.
    """
    from debate.nodes.dispatch import dispatch_round1

    builder = StateGraph(DebateState)
    _base_graph_nodes(builder, {
        "divergence_check_node": divergence_check_node_nli,
    })
    _standard_edges(builder, dispatch_round1, route_divergence_nli)
    return builder.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Adaptive PROHIBITION variant
# ---------------------------------------------------------------------------

def _make_adaptive_agent_nodes(question_type: str):
    """Return (optimist_fn, pessimist_fn, devil_fn) using type-appropriate prompts."""
    from debate.prompts_adaptive import ADAPTIVE_PROMPTS
    prompts = ADAPTIVE_PROMPTS.get(question_type, ADAPTIVE_PROMPTS["binary"])

    def _agent(state: dict, role: str) -> dict:
        topic = state.get("topic", "")
        round_num = state.get("round_num", 0)
        prior_arguments = state.get("prior_arguments", [])
        system_prompt = prompts[role]

        human_content = f"Topic for analysis: {topic}\n\nRound: {round_num + 1}"

        if prior_arguments and round_num > 0:
            rounds: dict = {}
            for s in prior_arguments:
                rnd = s["round_num"]
                if rnd not in rounds:
                    rounds[rnd] = {}
                rounds[rnd][s["agent_role"]] = s

            own_rounds = sorted(r for r in rounds if role in rounds[r])
            if own_rounds:
                human_content += "\n\n--- Your position history ---"
                for rnd in own_rounds:
                    human_content += f"\n\n[YOUR POSITION — Round {rnd + 1}]"
                    human_content += f"\nPosition: {rounds[rnd][role]['position']}"

            human_content += "\n\n--- Full debate history (opponents) ---"
            for rnd in sorted(rounds.keys()):
                opponent_args = [rounds[rnd][r] for r in rounds[rnd] if r != role]
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
                "\nIf (and ONLY if) an opponent's specific claim is logically superior,"
                " record it in your concessions list."
                "\nDo NOT concede to avoid conflict. Only concede on logical grounds."
            )

        llm = _make_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_content),
        ]
        argument = _invoke_with_retry(llm, messages, role, round_num)
        return {"current_round_arguments": [argument]}

    def opt(s): return _agent(s, "optimist")
    def pes(s): return _agent(s, "pessimist")
    def dev(s): return _agent(s, "devil")
    return opt, pes, dev


def _build_adaptive_graph_for_type(question_type: str):
    """Build a compiled graph using prompts appropriate for question_type."""
    from debate.nodes.dispatch import dispatch_round1, route_divergence
    opt, pes, dev = _make_adaptive_agent_nodes(question_type)
    builder = StateGraph(DebateState)
    _base_graph_nodes(builder, {
        "optimist_node": opt,
        "pessimist_node": pes,
        "devil_node": dev,
    })
    _standard_edges(builder, dispatch_round1, route_divergence)
    return builder.compile(checkpointer=InMemorySaver())


class AdaptiveProhibitionGraph:
    """Wrapper that classifies the topic at invoke time, then delegates to
    the appropriate graph (values_based / binary / context_dependent)."""

    def __init__(self):
        self._graphs = {
            "values_based":      _build_adaptive_graph_for_type("values_based"),
            "binary":            _build_adaptive_graph_for_type("binary"),
            "context_dependent": _build_adaptive_graph_for_type("context_dependent"),
        }

    def invoke(self, inputs: dict, config: dict | None = None) -> dict:
        from debate.classify import classify_question
        topic = inputs.get("topic", "")
        question_type = classify_question(topic)
        print(f"  [adaptive] '{topic[:50]}...' → {question_type}")
        return self._graphs[question_type].invoke(inputs, config or {})


def build_adaptive_prohibition_graph():
    """Adaptive PROHIBITION: classify question type first, then select prompt level.

    values_based      → full PROHIBITION (advocates must commit to values)
    binary            → moderate (must recommend, can acknowledge nuance)
    context_dependent → off (agents map conditions, 'it depends' is correct)
    """
    return AdaptiveProhibitionGraph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VARIANT_BUILDERS = {
    "full_system": build_full_system_graph,         # fixed devil (current)
    "original_devil": build_original_devil_graph,   # old devil (pre-fix)
    "no_prohibition": build_no_prohibition_graph,
    "sequential": build_sequential_graph,
    "fixed_rounds": build_fixed_rounds_graph,
    "fulltext_embedding": build_fulltext_graph,
    "nli_detection": build_nli_detection_graph,
    "adaptive_prohibition": build_adaptive_prohibition_graph,
}


def build_variant_graph(variant: str):
    """Build and return a compiled graph for the given variant name."""
    if variant not in VARIANT_BUILDERS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Valid options: {list(VARIANT_BUILDERS.keys())}"
        )
    return VARIANT_BUILDERS[variant]()
