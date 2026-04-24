# app.py
"""
Multi-Agent Debate System — Streamlit UI (Phase 5)

Single-file app wrapping the LangGraph debate graph.
Stream pattern: graph.stream(stream_mode="updates") — synchronous only.
State machine: idle -> running -> complete | error
"""
import os
import uuid

import streamlit as st

from debate.graph import graph  # module-level singleton — never store in session_state
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
# API key guard (fail loudly on fresh run with no key — UI-04)
# Supports both direct API key and internal proxy (ANTHROPIC_AUTH_TOKEN)
# ---------------------------------------------------------------------------
_has_api_key = bool(
    os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("ANTHROPIC_AUTH_TOKEN")
)
if not _has_api_key:
    st.warning(
        "⚠️ No API credentials found. Set one of:\n\n"
        "- `ANTHROPIC_API_KEY=sk-ant-...` (direct Anthropic API)\n"
        "- `ANTHROPIC_AUTH_TOKEN=...` + `ANTHROPIC_BASE_URL=...` (internal proxy)\n\n"
        "See README.md for setup instructions."
    )
    st.stop()

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
    "debate it in real time using semantic divergence detection."
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
        disabled=(st.session_state.debate_status == "running" or not (topic or "").strip()),
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
    status.markdown(f"**Confidence:** {arg.confidence:.0%}")

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
if start_clicked and (topic or "").strip():
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
                    continuing = score > 0.75
                    st.info(
                        f"Divergence score: **{score:.2f}** — "
                        f"{'debate continues (high divergence)' if continuing else 'converging'}"
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

    # --- Confidence score + convergence status ---
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Confidence Score", f"{report.confidence_score:.0%}")
    with col2:
        st.progress(report.confidence_score)
        status_labels = {
            "converged": "Converged",
            "max_rounds": "Max Rounds Reached",
            "partial": "Partial Convergence",
        }
        st.caption(f"Status: **{status_labels.get(report.convergence_status, report.convergence_status)}**")

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
