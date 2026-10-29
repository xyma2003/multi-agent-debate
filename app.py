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
import sqlite3
from datetime import datetime

import streamlit as st

from debate.graph import graph  # module-level singleton — never store in session_state
from debate.divergence import DIVERGENCE_MODE
from debate.state import AgentArgument, DebateReport
from debate.store import count_debates, list_debates, load_debate, save_debate
from debate.sample import load_offline_example
from debate.export import build_report_exports
from debate.detector_display import detector_details, detector_summary

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
st.session_state.setdefault("report_source", None)
st.session_state.setdefault("sample_provenance", None)
st.session_state.setdefault("pending_save_report", None)
st.session_state.setdefault("pending_save_error", None)


def _pending_report_is_open() -> bool:
    pending = st.session_state.pending_save_report
    opened = st.session_state.final_report
    return (
        pending is not None and opened is not None
        and st.session_state.report_source == "live"
        and pending.debate_id == opened.debate_id
    )


def _render_save_recovery() -> None:
    """Keep a generated report recoverable while other reports are browsed."""
    pending = st.session_state.pending_save_report
    if pending is None:
        return
    busy = st.session_state.debate_status == "running"
    with st.container(border=True):
        st.warning("Report generated, but saving has not been confirmed.")
        st.markdown(f"**Unsaved report:** {pending.topic}")
        st.caption(
            "Download it or retry saving before starting another debate. "
            "This copy is kept only in this browser session; closing or reloading the page may lose it. "
            "Discard removes this session copy, not any saved history."
        )
        if st.session_state.pending_save_error:
            st.error(f"Save failed: {st.session_state.pending_save_error}")
        view, retry, discard = st.columns(3)
        # Bind each action to this report, so an old widget event cannot act on
        # a later pending report that happens to occupy the same UI position.
        if view.button(
            "View unsaved report", key=f"pending_view_{pending.debate_id}",
            disabled=busy or _pending_report_is_open(), use_container_width=True,
        ) and not busy:
            st.session_state.final_report = pending.model_copy(deep=True)
            st.session_state.report_source = "live"
            st.session_state.sample_provenance = None
            st.session_state.thread_id = None
            st.session_state.debate_status = "error"
            st.session_state.error_msg = st.session_state.pending_save_error
            st.rerun()
        if retry.button(
            "Retry save", key=f"pending_retry_{pending.debate_id}",
            disabled=busy, use_container_width=True,
        ) and not busy:
            try:
                # Retry only persistence with the same ID and complete payload.
                # Never resume/reinvoke the graph or ask the model again.
                save_debate(pending)
            except Exception as exc:
                st.session_state.pending_save_error = str(exc)
                if _pending_report_is_open():
                    st.session_state.debate_status = "error"
                    st.session_state.error_msg = str(exc)
            else:
                if _pending_report_is_open():
                    st.session_state.debate_status = "complete"
                    st.session_state.error_msg = None
                st.session_state.pending_save_report = None
                st.session_state.pending_save_error = None
                st.session_state.save_recovery_notice = "Report saved to Past Debates."
            st.rerun()
        if discard.button(
            "Discard unsaved report", key=f"pending_discard_{pending.debate_id}",
            disabled=busy, use_container_width=True,
        ) and not busy:
            if _pending_report_is_open():
                st.session_state.final_report = None
                st.session_state.report_source = None
                st.session_state.sample_provenance = None
                st.session_state.thread_id = None
                st.session_state.debate_status = "idle"
                st.session_state.error_msg = None
            st.session_state.pending_save_report = None
            st.session_state.pending_save_error = None
            st.rerun()

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
        "You can view the offline example below or browse saved reports in Past Debates. "
        "To start a new debate, configure the credentials described in `.env.example` or README.md."
    )

# ---------------------------------------------------------------------------
# Sidebar — past debates (STORE-02 replay via Phase 4 API)
# ---------------------------------------------------------------------------
HISTORY_PAGE_SIZE = 10


def _history_timestamp(value: str) -> str:
    """Keep the saved timezone visible without noisy fractional seconds."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat(
            sep=" ", timespec="seconds",
        )
    except (TypeError, ValueError, AttributeError):
        return str(value)


def _clear_history_search() -> None:
    if st.session_state.debate_status != "running":
        st.session_state.history_query = ""
        st.session_state.history_page = 0


def _render_history_sidebar() -> None:
    st.header("Past Debates")
    busy = st.session_state.debate_status == "running"
    query = st.text_input(
        "Search topics", key="history_query", disabled=busy,
        placeholder="Find a saved debate…",
        help="Search saved topic text. Matches are literal; report contents are not searched.",
    ).strip()
    if query:
        st.button(
            "Clear search", key="history_clear", on_click=_clear_history_search,
            disabled=busy,
        )
    if st.button("Refresh", key="history_refresh", disabled=busy) and not busy:
        st.rerun()
    # Retain the active list even if a disabled widget event is forged while
    # the graph is running. Browsing must not replace an in-flight debate.
    if busy:
        query = st.session_state.get("_history_applied_query", "")
    elif query != st.session_state.get("_history_applied_query", ""):
        st.session_state.history_page = 0
    st.session_state._history_applied_query = query

    try:
        total = count_debates(query=query)
        max_page = max((total - 1) // HISTORY_PAGE_SIZE, 0)
        page = max(0, min(st.session_state.get("history_page", 0), max_page))
        st.session_state.history_page = page
        past = list_debates(query=query, limit=HISTORY_PAGE_SIZE, offset=page * HISTORY_PAGE_SIZE)
    except sqlite3.Error:
        st.error("Could not read saved debates. Use Refresh to retry. Your open report has been kept.")
        return

    if st.session_state.final_report is not None:
        st.caption("Search and pages change this list only; your open report stays visible.")
    if not total:
        if query:
            st.caption("No saved debates match this topic. Try another search or clear it.")
        else:
            st.caption("No past debates yet.")
        return

    st.caption(f"{total} saved {'debate' if total == 1 else 'debates'} · Page {page + 1} of {max_page + 1}")
    previous, next_page = st.columns(2)
    if previous.button(
        "Previous", key="history_prev", disabled=busy or page == 0,
        use_container_width=True,
    ) and not busy and page > 0:
        st.session_state.history_page = page - 1
        st.rerun()
    if next_page.button(
        "Next", key="history_next", disabled=busy or page >= max_page,
        use_container_width=True,
    ) and not busy and page < max_page:
        st.session_state.history_page = page + 1
        st.rerun()
    if not past:
        st.info("History changed while loading. Use Refresh to reload this page.")
        return

    opened = st.session_state.final_report
    selected_id = opened.debate_id if opened and st.session_state.report_source == "history" else None
    for row in past:
        label = row["topic"][:52] + ("…" if len(row["topic"]) > 52 else "")
        clicked = st.button(
            label, key=row["debate_id"], help=row["topic"], disabled=busy,
            type="primary" if row["debate_id"] == selected_id else "secondary",
            use_container_width=True,
        )
        short_id = row["debate_id"] if len(row["debate_id"]) <= 12 else "…" + row["debate_id"][-8:]
        st.caption(f"{_history_timestamp(row['created_at'])} · ID {short_id}")
        if clicked and not busy:
            try:
                loaded = load_debate(row["debate_id"])
                if loaded is None:
                    st.warning("This saved debate is no longer available. Use Refresh to update the list. Your open report has been kept.")
                    continue
                if loaded.debate_id != row["debate_id"]:
                    raise ValueError("Stored report ID does not match the selected debate")
            except (sqlite3.Error, ValueError):
                st.error("Could not open this saved debate. Your open report has been kept. You can retry or select another report.")
                continue
            st.session_state.final_report = loaded
            st.session_state.report_source = "history"
            st.session_state.sample_provenance = None
            st.session_state.thread_id = None
            st.session_state.debate_status = "complete"
            st.session_state.error_msg = None
            st.rerun()


with st.sidebar:
    _render_history_sidebar()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Multi-Agent Debate System")
st.caption(
    "Enter any topic and watch three agents — Optimist, Pessimist, and Devil's Advocate — "
    f"debate it in real time using {DIVERGENCE_MODE.upper()} divergence detection."
)
if notice := st.session_state.pop("save_recovery_notice", None):
    st.success(notice)
_render_save_recovery()

with st.container(border=True):
    st.markdown("### Explore a complete example")
    st.caption(
        "A fictional team's four-day workweek: two rounds, three perspectives, "
        "and a final report. No API key or model download needed."
    )
    example_clicked = st.button(
        "View offline example", key="offline_example_btn",
        disabled=(st.session_state.debate_status == "running"),
    )
    if example_clicked and st.session_state.debate_status != "running":
        try:
            example = load_offline_example()
        except (OSError, ValueError):
            st.error(
                "Could not open the offline example. Restore examples/offline_debate.json "
                "from the repository and try again. Your current report has been kept."
            )
        else:
            st.session_state.final_report = example.report
            st.session_state.report_source = "sample"
            st.session_state.sample_provenance = example.provenance
            st.session_state.thread_id = None
            st.session_state.error_msg = None
            st.session_state.debate_status = "complete"
            # The sidebar was rendered before this action; refresh its selected
            # report marker now that the main view contains the example.
            st.rerun()

# ---------------------------------------------------------------------------
# Input section (UI-01)
# ---------------------------------------------------------------------------
topic = st.text_input(
    "Debate topic", key="debate_topic",
    placeholder="e.g. 'Should remote work be the default for knowledge workers?'",
    disabled=(st.session_state.debate_status == "running"),
)

col_btn, col_slider = st.columns([1, 3])
with col_btn:
    start_clicked = st.button(
        "Start Debate",
        disabled=(
            not _has_api_key or st.session_state.debate_status == "running"
            or st.session_state.pending_save_report is not None or not (topic or "").strip()
        ),
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
if (
    start_clicked and _has_api_key and (topic or "").strip()
    and st.session_state.debate_status != "running"
    and st.session_state.pending_save_report is None
):
    st.session_state.debate_status = "running"
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.final_report = None
    st.session_state.report_source = "live"
    st.session_state.sample_provenance = None
    st.session_state.error_msg = None

    st.markdown("### Debate in Progress")

    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": 30,
    }

    generated_report = None
    save_confirmed = False
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
                        generated_report = report_obj.model_copy(deep=True)
                    st.success("Synthesis complete.")

                elif node_name == "save_node":
                    save_confirmed = True

                # initialize, collect_round1: no UI output needed

        if generated_report is not None and not save_confirmed:
            raise RuntimeError("The debate ended without confirming that its report was saved.")
        st.session_state.debate_status = "complete"

    except Exception as exc:
        st.session_state.debate_status = "error"
        st.session_state.error_msg = str(exc)
        if generated_report is not None and not save_confirmed:
            st.session_state.pending_save_report = generated_report
            st.session_state.pending_save_error = str(exc)

    # Refresh navigation only after the stream finishes (or fails), never from
    # inside the loop. Keep the success/error state above so a persistence failure
    # cannot become a successful report merely because the page refreshes.
    st.rerun()

# ---------------------------------------------------------------------------
# Error state (UI-04)
# ---------------------------------------------------------------------------
if (
    st.session_state.debate_status == "error" and st.session_state.error_msg
    and not _pending_report_is_open()
):
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
    st.markdown(f"**Topic:** {report.topic}")
    pending_is_open = _pending_report_is_open()
    if pending_is_open:
        st.caption("Saving is unconfirmed. These downloads preserve the generated report.")
    if st.session_state.report_source == "history":
        st.caption(
            f"Saved report · {_history_timestamp(report.created_at.isoformat())} · ID {report.debate_id}"
        )
    if st.session_state.report_source == "sample":
        st.info("Fixed synthetic example · No model/API execution · Not saved to your history")
        st.caption(st.session_state.sample_provenance)

    try:
        exports = build_report_exports(
            report, source=st.session_state.report_source,
            provenance=st.session_state.sample_provenance,
            persistence="unconfirmed" if pending_is_open else None,
        )
    except ValueError:
        st.warning(
            "This report could not be exported. Its contents remain visible; "
            "please select another report or try again."
        )
    else:
        st.caption("Download this open report, including all recorded rounds and concessions.")
        markdown_download, json_download, _ = st.columns([1, 1, 2])
        markdown_download.download_button(
            "Download Markdown", data=exports.markdown.encode("utf-8"),
            file_name=exports.markdown_filename, mime="text/markdown; charset=utf-8",
            key="report_export_markdown", on_click="ignore", width="stretch",
            help="A readable report with its source, full reasoning trace, and score explanations.",
        )
        json_download.download_button(
            "Download JSON", data=exports.json.encode("utf-8"),
            file_name=exports.json_filename, mime="application/json",
            key="report_export_json", on_click="ignore", width="stretch",
            help="Structured report data with source metadata; original saved values are preserved.",
        )

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
            "Each round's saved detector metadata is shown when available. Missing metadata means the detector "
            "is unknown; the current app setting cannot identify the mode used by a past report. "
            "Historical values are displayed as saved, without recomputation."
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

    if report.reasoning_trace:
        st.caption("Saved detectors by round: " + " · ".join(
            f"Round {rr.round_num + 1}: {rr.detector_metadata.mode.upper() if rr.detector_metadata else 'unknown'}"
            for rr in report.reasoning_trace
        ))
        with st.expander("Saved detector settings", expanded=False):
            for rr in report.reasoning_trace:
                st.markdown(f"**Round {rr.round_num + 1}**")
                st.caption(detector_summary(rr.detector_metadata))
                st.text("\n".join(f"{label}: {value}" for label, value in detector_details(rr.detector_metadata)))

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
            st.caption(detector_summary(rr.detector_metadata))
            for arg in rr.arguments:
                st.markdown(f"**{arg.agent_role.title()}:** {arg.position}")
                if arg.is_sentinel:
                    st.caption("No valid agent output: a fallback was recorded for this round.")
                else:
                    st.caption(f"Self-reported confidence: {arg.confidence:.0%} · Agent estimate, not used in the report index.")
                st.markdown(arg.reasoning)
                if arg.key_claims:
                    st.markdown("**Key claims:**")
                    for claim in arg.key_claims:
                        st.markdown(f"- {claim}")
                if arg.concessions:
                    st.markdown("**Concessions this round:**")
                    for concession in arg.concessions:
                        st.markdown(f"- {concession.conceded_point}")
                        st.caption(
                            f"In response to {concession.triggered_by_agent}: "
                            f"{concession.triggered_by_claim} — {concession.rationale}"
                        )

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
                st.caption(c.rationale)


if (
    st.session_state.debate_status == "complete" or (
        st.session_state.debate_status == "error" and _pending_report_is_open()
    )
) and st.session_state.final_report is not None:
    render_report(st.session_state.final_report)
