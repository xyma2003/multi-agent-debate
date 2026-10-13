# app.py
"""
Multi-Agent Debate System — Streamlit UI (Phase 5)

Single-file app wrapping the LangGraph debate graph.
Stream pattern: graph.stream(stream_mode="updates") — synchronous only.
State machine: idle -> running -> complete | error
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # load .env, overriding any stale shell vars (e.g. a different OPENAI_API_KEY)
import uuid

import streamlit as st

from debate.graph import graph  # module-level singleton — never store in session_state
from debate.divergence import DIVERGENCE_MODE
from debate.state import AgentArgument, DebateReport
from debate.store import list_debates, load_debate

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Debate System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation — MUST run before any widget that reads state
# ---------------------------------------------------------------------------
if "debate_status" not in st.session_state:
    st.session_state.debate_status = "idle"   # idle | running | complete | error
    st.session_state.thread_id = None
    st.session_state.final_report = None
    st.session_state.error_msg = None

# ---------------------------------------------------------------------------
# Credentials are required only when starting a new debate.
# Supports direct API key, internal proxy, and SiliconFlow/OpenAI-compatible.
# ---------------------------------------------------------------------------
_backend = os.environ.get("LLM_BACKEND", "groq").lower()
_credential_env = {
    "groq": "GROQ_API_KEY",
    "qwen": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "together": "TOGETHER_API_KEY",
    "sambanova": "SAMBANOVA_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}
if _backend == "anthropic":
    _has_api_key = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
else:
    _required_credential = _credential_env.get(_backend)
    _has_api_key = bool(_required_credential and os.environ.get(_required_credential))
if not _has_api_key:
    st.warning(
        f"⚠️ No credentials found for `LLM_BACKEND={_backend}`. "
        "You can still browse saved reports in Past Debates. "
        "To start a new debate, configure the credentials described in `.env.example` or README.md."
    )

# ---------------------------------------------------------------------------
# Sidebar — past debates (STORE-02 replay via Phase 4 API)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Past Debates")
    past = list_debates()
    if past:
        for row in past[:10]:
            label = row["topic"][:40] + ("..." if len(row["topic"]) > 40 else "")
            if st.button(label, key=row["debate_id"]):
                loaded = load_debate(row["debate_id"])
                if loaded:
                    st.session_state.final_report = loaded
                    st.session_state.debate_status = "complete"
                    st.session_state.error_msg = None
                    st.rerun()
    else:
        st.caption("No past debates yet.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Multi-Agent Debate System")
st.caption(
    "Enter any topic and watch three agents — Optimist, Pessimist, and Devil's Advocate — "
    f"debate it in real time using {DIVERGENCE_MODE.upper()} divergence detection."
)

# ---------------------------------------------------------------------------
# Input section (UI-01)
# ---------------------------------------------------------------------------
topic = st.text_input(
    "Debate topic",
    placeholder="e.g. 'Should remote work be the default for knowledge workers?'",
    disabled=(st.session_state.debate_status == "running"),
)

col_btn, col_slider = st.columns([1, 3])
with col_btn:
    start_clicked = st.button(
        "Start Debate",
        disabled=(not _has_api_key or st.session_state.debate_status == "running" or not (topic or "").strip()),
        type="primary",
    )
with col_slider:
    max_rounds = st.slider(
        "Max rounds",
        min_value=1, max_value=3, value=2,
        disabled=(st.session_state.debate_status == "running"),
        help="Maximum debate rounds before forcing synthesis",
    )

# ---------------------------------------------------------------------------
# Helper: render one agent's chunk as a status container (UI-02)
# ---------------------------------------------------------------------------
AGENT_LABELS = {
    "optimist_node": "Optimist",
    "pessimist_node": "Pessimist",
    "devil_node": "Devil's Advocate",
}
AGENT_EMOJIS = {
    "optimist_node": "🟢",
    "pessimist_node": "🔴",
    "devil_node": "😈",
}


def _render_agent_chunk(node_name: str, node_update: dict) -> None:
    """Render one agent's argument as a collapsible status container.

    Uses explicit status.update(state="complete") — NOT a `with` block.
    `with` auto-closes on exit, cutting the stream short.
    """
    args: list[AgentArgument] = node_update.get("current_round_arguments", [])
    if not args:
        return
    arg = args[-1]  # This node's argument for this round (last item = most recent)

    label = AGENT_LABELS.get(node_name, node_name)
    emoji = AGENT_EMOJIS.get(node_name, "")
    status = st.status(
        f"{emoji} {label} — Round {arg.round_num + 1}",
        expanded=True,
        state="running",
    )

    if arg.is_sentinel:
        status.warning("Agent failed to produce structured output (sentinel injected).")
        status.update(state="error", expanded=True)
        return

    status.markdown(f"**Position:** {arg.position}")
    status.markdown(f"**Self-reported confidence:** {arg.confidence:.0%}")
    status.caption("The agent's own estimate, not calibrated accuracy or an input to the report index.")

    if arg.key_claims:
        status.markdown("**Key Claims:**")
        for claim in arg.key_claims:
            status.markdown(f"- {claim}")

    if arg.concessions:
        status.markdown("**Concessions this round:**")
        for c in arg.concessions:
            status.markdown(
                f"- _{c.conceded_point}_ "
                f"*(triggered by {c.triggered_by_agent}: {c.triggered_by_claim})*"
            )

    status.update(state="complete", expanded=False)


# ---------------------------------------------------------------------------
# Debate execution — fires only on Start Debate click (UI-02)
# ---------------------------------------------------------------------------
if start_clicked and _has_api_key and (topic or "").strip():
    st.session_state.debate_status = "running"
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.final_report = None
    st.session_state.error_msg = None

    st.markdown("### Debate in Progress")

    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": 30,
    }

    try:
        for chunk in graph.stream(
            {"topic": topic.strip(), "max_rounds": max_rounds},
            config=config,
            stream_mode="updates",  # yields {node_name: state_delta} per node
        ):
            for node_name, node_update in chunk.items():
                if node_name in ("optimist_node", "pessimist_node", "devil_node"):
                    _render_agent_chunk(node_name, node_update)

                elif node_name == "divergence_check_node":
                    score = node_update.get("divergence_score", 0.0)
                    diverged_pairs = node_update.get("diverged_pairs", [])
                    st.info(
                        f"Divergence score: **{score:.2f}** — "
                        f"{'divergent pair detected' if diverged_pairs else 'no divergent pair detected'}"
                    )

                elif node_name == "synthesize_stub":
                    report_obj = node_update.get("final_report")
                    if report_obj is not None:
                        st.session_state.final_report = report_obj
                    st.success("Synthesis complete.")

                # initialize, collect_round1, save_node: no UI output needed

        st.session_state.debate_status = "complete"

    except Exception as exc:
        st.session_state.debate_status = "error"
        st.session_state.error_msg = str(exc)

    # st.rerun() intentionally omitted here — Streamlit will re-render on next
    # interaction. Calling rerun() inside the stream loop would kill the stream.

# ---------------------------------------------------------------------------
# Error state (UI-04)
# ---------------------------------------------------------------------------
if st.session_state.debate_status == "error" and st.session_state.error_msg:
    st.error(f"Debate failed: {st.session_state.error_msg}")
    if st.button("Reset", key="reset_btn"):
        st.session_state.debate_status = "idle"
        st.session_state.error_msg = None
        st.rerun()

# ---------------------------------------------------------------------------
# Final report rendering (UI-03)
# ---------------------------------------------------------------------------

def render_report(report: DebateReport) -> None:
    """Render the full DebateReport with structured visual layout."""
    st.divider()
    st.subheader("Debate Result")

    # Keep the persisted field for replay compatibility; clarify its meaning.
    col1, col2 = st.columns([2, 3])
    with col1:
        st.metric(
            "Debate Agreement Index", f"{report.confidence_score * 100:.1f} / 100",
            help="A heuristic from detected divergence and round count, not answer accuracy.",
        )
    with col2:
        status_labels = {
            "converged": "No divergent pair detected",
            "plateau": "Divergence stopped changing",
            "stalled": "No new concessions",
            "max_rounds": "Round limit reached",
            "partial": "Fallback result",
        }
        st.caption(f"Stop reason: **{status_labels.get(report.convergence_status, report.convergence_status)}**")
        st.caption(
            "This heuristic uses the highest detected divergence across all rounds and a round-count penalty. "
            "It is not a calibrated probability of correctness, a measure of evidence quality, or proof of consensus."
        )

    with st.expander("How to read this index", expanded=False):
        st.code("index = 100 × (1 − highest divergence across all rounds) × round factor", language=None)
        st.markdown(
            "The round factor is **1.0 for one round, 0.9 for two rounds, and 0.8 for three or more rounds**. "
            "These are fixed weights, not a learned estimate of answer quality. "
            "A higher index reflects lower peak detected divergence and/or fewer rounds."
        )
        st.markdown(
            "**Example:** divergences of `0.8 → 0.2` over two rounds give "
            "`100 × (1 − 0.8) × 0.9 = 18 / 100`, not 72. "
            "The earlier disagreement still counts even if the last round becomes more aligned. "
            "Zero measured divergence, or an empty history in the formula's fallback, can produce 100; "
            "that does not establish that the answer is correct."
        )
        st.markdown(
            "**Detector limits:** NLI measures model-estimated contradictions; low contradiction can also mean "
            "neutral or unrelated claims. Cosine measures semantic similarity, which can confuse shared topics "
            "with agreement. Scores from the two modes are not directly comparable. "
            "Saved reports do not record their detector mode; the current app setting cannot identify the mode "
            "used by a past report. Historical values are displayed as saved, without recomputation."
        )
        st.markdown(
            "**Agent self-reported confidence** is each agent's own estimate about its position. "
            "It is not calibrated accuracy and is not used in this index. "
            "The stop reason describes how the loop ended; it does not verify consensus or factual correctness."
        )
        if report.reasoning_trace:
            peak = max(rr.divergence_score for rr in report.reasoning_trace)
            st.caption(
                f"Recorded rounds: {len(report.reasoning_trace)} · "
                f"Highest recorded divergence: {peak:.4f}. See the full reasoning trace below."
            )
        else:
            st.caption("No round trace is available; this saved index cannot be checked against recorded rounds.")

    # --- Verdict ---
    st.info(report.verdict)

    # --- Consensus vs disputed ---
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Points of Consensus")
        if report.consensus_points:
            for point in report.consensus_points:
                st.success(point)
        else:
            st.caption("No consensus reached.")

    with col_b:
        st.markdown("### Disputed Points")
        if report.disputed_points:
            for dp in report.disputed_points:
                with st.expander(dp.topic, expanded=False):
                    # DisputedPoint.agent_positions is dict[str, str]
                    for role, stance in dp.agent_positions.items():
                        st.markdown(f"**{role.title()}:** {stance}")
        else:
            st.caption("No disputed points recorded.")

    # --- Reasoning trace (expandable, UI-03) ---
    with st.expander("Full Reasoning Trace", expanded=False):
        for rr in report.reasoning_trace:
            st.markdown(
                f"**Round {rr.round_num + 1}** — divergence: `{rr.divergence_score:.3f}`"
            )
            for arg in rr.arguments:
                st.markdown(f"- **{arg.agent_role.title()}:** {arg.position}")
                if arg.is_sentinel:
                    st.caption("No valid agent output: a fallback was recorded for this round.")
                else:
                    st.caption(f"Self-reported confidence: {arg.confidence:.0%} · Agent estimate, not used in the report index.")

    # --- Concession log ---
    if report.concession_log:
        with st.expander(
            f"Concession Log ({len(report.concession_log)} concessions)", expanded=False
        ):
            for c in report.concession_log:
                st.markdown(
                    f"- **{c.conceded_point}** "
                    f"*(conceded in response to {c.triggered_by_agent}: "
                    f"{c.triggered_by_claim})*"
                )


if st.session_state.debate_status == "complete" and st.session_state.final_report is not None:
    render_report(st.session_state.final_report)
