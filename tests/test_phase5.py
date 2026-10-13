"""Deterministic Streamlit acceptance tests using synthetic, local reports.

These tests run the real app and compile its real graph, but never execute a
model, download NLI weights, load a developer's .env, or open debates.db.
Run with: python -m pytest tests/test_phase5.py -v
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parent.parent / "app.py"
pytestmark = pytest.mark.ui


@pytest.fixture
def sample_reports():
    """Two distinguishable, persisted reports with every report section filled."""
    from debate.state import (
        AgentArgument, Concession, DebateReport, DisputedPoint, RoundRecord,
    )

    reports = []
    for index, subject in enumerate(("Remote work", "Four-day week")):
        concession = Concession(
            conceded_point=f"{subject} needs a trial period",
            triggered_by_agent="pessimist",
            triggered_by_claim=f"{subject} has transition costs",
            rationale="A bounded trial can test the transition cost.",
        )
        argument = AgentArgument(
            agent_role="optimist",
            round_num=0,
            position=f"{subject} improves flexibility",
            reasoning=f"A trial of {subject} can measure team outcomes.",
            confidence=0.8,
            key_claims=[
                f"{subject} supports flexibility",
                "Outcomes can be measured",
                "A trial limits transition risk",
            ],
            concessions=[concession],
        )
        reports.append(DebateReport(
            debate_id=f"offline-report-{index}",
            topic=f"Should we adopt {subject.lower()}?",
            consensus_points=[f"Measure {subject} outcomes before expansion"],
            disputed_points=[DisputedPoint(
                topic=f"{subject} coordination cost",
                agent_positions={
                    "optimist": f"{subject} costs are manageable",
                    "pessimist": f"{subject} costs remain uncertain",
                },
            )],
            verdict=f"Run a limited {subject} trial and review the results.",
            confidence_score=(0.72, 0.64)[index],
            convergence_status=("converged", "max_rounds")[index],
            reasoning_trace=[RoundRecord(
                round_num=0, arguments=[argument], divergence_score=0.3,
            )],
            concession_log=[concession],
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
            + timedelta(days=index),
        ))
    return reports


@pytest.fixture
def offline_app(monkeypatch, tmp_path):
    """Isolate credentials, persistence, model calls, and outbound networking."""
    import dotenv
    from debate import divergence, llm, store

    for credential in (
        "GROQ_API_KEY", "OPENAI_API_KEY", "CEREBRAS_API_KEY",
        "TOGETHER_API_KEY", "SAMBANOVA_API_KEY", "GOOGLE_API_KEY",
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    ):
        monkeypatch.delenv(credential, raising=False)
    monkeypatch.setenv("LLM_BACKEND", "groq")
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    monkeypatch.setattr(dotenv, "load_dotenv", Mock(return_value=False))

    no_model = Mock(side_effect=AssertionError("UI tests must not create an LLM"))
    no_nli = Mock(side_effect=AssertionError("UI tests must not load an NLI model"))
    no_embeddings = Mock(side_effect=AssertionError("UI tests must not load embeddings"))
    no_network = Mock(side_effect=AssertionError("UI tests must stay offline"))
    monkeypatch.setattr(llm, "_make_llm", no_model)
    monkeypatch.setattr(divergence, "_get_nli_model", no_nli)
    monkeypatch.setattr(divergence, "_get_model", no_embeddings)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    # Import the real graph under the model/network guards. Patch the aliases
    # too, because earlier tests in the full suite may have imported the nodes.
    from debate.graph import graph
    from debate.nodes import agents, synthesize

    monkeypatch.setattr(agents, "_make_llm", no_model)
    monkeypatch.setattr(synthesize, "_make_llm", no_model)
    stream = Mock(side_effect=AssertionError("Browsing must not execute the graph"))
    monkeypatch.setattr(graph, "stream", stream)

    database_path = tmp_path / "reports.sqlite3"
    connection = store.get_connection(database_path)
    # The default argument in get_connection is bound at definition time;
    # changing DB_PATH alone would still open the developer's real database.
    monkeypatch.setattr(store, "get_connection", Mock(return_value=connection))

    def open_app():
        return AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    yield SimpleNamespace(
        open=open_app,
        stream=stream,
        connection=connection,
    )

    connection.close()
    store._connections.pop(str(database_path.resolve()), None)
    no_model.assert_not_called()
    no_nli.assert_not_called()
    no_embeddings.assert_not_called()
    no_network.assert_not_called()


def _start_button(app):
    return next(button for button in app.button if button.label == "Start Debate")


def _values(elements):
    return "\n".join(str(element.value) for element in elements)


def _assert_report(app, report):
    """Check visible report data, not just the absence of a Streamlit error."""
    assert not app.exception
    assert app.session_state["debate_status"] == "complete"
    assert app.session_state["final_report"].debate_id == report.debate_id
    assert "Debate Result" in _values(app.subheader)
    assert report.verdict in _values(app.info)
    assert app.metric[0].label == "Debate Agreement Index"
    assert app.metric[0].value == f"{report.confidence_score * 100:.1f} / 100"
    captions = _values(app.caption)
    assert "heuristic" in captions.lower()
    assert "not a calibrated probability of correctness" in captions.lower()
    assert report.consensus_points[0] in _values(app.success)
    assert report.disputed_points[0].topic in [item.label for item in app.expander]
    rendered = _values(app.markdown)
    assert report.disputed_points[0].agent_positions["pessimist"] in rendered
    assert report.reasoning_trace[0].arguments[0].position in rendered
    assert "Round 1" in rendered
    assert "0.300" in rendered
    assert "Self-reported confidence" in captions
    assert "80%" in captions
    assert report.concession_log[0].conceded_point in rendered
    assert report.concession_log[0].triggered_by_claim in rendered


def test_no_credentials_empty_history_is_usable(offline_app):
    app = offline_app.open()

    assert not app.exception
    assert app.session_state["debate_status"] == "idle"
    assert app.session_state["thread_id"] is None
    assert app.session_state["final_report"] is None
    assert app.session_state["error_msg"] is None
    assert "Multi-Agent Debate System" in _values(app.title)
    assert "Past Debates" in _values(app.sidebar.header)
    assert "No past debates yet." in _values(app.sidebar.caption)
    assert "No credentials" in _values(app.warning)
    assert app.text_input[0].label == "Debate topic"
    assert not app.text_input[0].disabled
    assert _start_button(app).disabled
    offline_app.stream.assert_not_called()


def test_no_credentials_can_switch_between_saved_reports(offline_app, sample_reports):
    from debate.store import save_debate

    for report in sample_reports:
        save_debate(report, conn=offline_app.connection)
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    # History replay must work even when the database refuses all writes.
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    assert not app.exception
    assert [button.key for button in app.sidebar.button] == [
        sample_reports[1].debate_id, sample_reports[0].debate_id,
    ]

    for report, previous in (
        (sample_reports[1], sample_reports[0]),
        (sample_reports[0], sample_reports[1]),
        (sample_reports[1], sample_reports[0]),
    ):
        app.sidebar.button(key=report.debate_id).click().run()
        _assert_report(app, report)
        assert previous.verdict not in _values(app.info)
        assert previous.consensus_points[0] not in _values(app.success)
        assert _start_button(app).disabled

    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


def test_topic_does_not_enable_start_without_credentials(offline_app):
    app = offline_app.open()
    app.text_input[0].input("Should we adopt remote work?").run()

    assert not app.exception
    assert _start_button(app).disabled
    assert app.session_state["debate_status"] == "idle"
    # AppTest can submit a disabled button event. The server-side guard must
    # reject it as well, rather than relying solely on the browser's disabled UI.
    _start_button(app).click().run()
    assert not app.exception
    assert app.session_state["debate_status"] == "idle"
    offline_app.stream.assert_not_called()


@pytest.mark.parametrize("backend,credential", [
    ("groq", "GROQ_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("anthropic", "ANTHROPIC_AUTH_TOKEN"),
])
def test_fake_credentials_allow_mocked_debate(
    offline_app, sample_reports, monkeypatch, backend, credential,
):
    monkeypatch.setenv("LLM_BACKEND", backend)
    monkeypatch.setenv(credential, "offline-test-placeholder")
    report = sample_reports[0]
    argument = report.reasoning_trace[0].arguments[0]
    offline_app.stream.side_effect = None
    offline_app.stream.return_value = iter([
        {"optimist_node": {"current_round_arguments": [argument]}},
        {"divergence_check_node": {"divergence_score": 0.3, "diverged_pairs": []}},
        {"synthesize_stub": {"final_report": report}},
    ])

    app = offline_app.open()
    assert not app.exception
    assert not app.warning
    assert _start_button(app).disabled  # A key alone does not supply a topic.
    app.text_input[0].input(f"  {report.topic}  ").run()
    assert not _start_button(app).disabled
    _start_button(app).click().run()

    _assert_report(app, report)
    assert argument.key_claims[0] in _values(app.markdown)
    # Both the streamed argument and saved trace expose the agent's own 80%
    # separately from the persisted final index (72.0 / 100).
    confidence_rows = [
        str(item.value) for item in [*app.markdown, *app.caption]
        if "Self-reported confidence" in str(item.value)
    ]
    assert len(confidence_rows) == 2
    assert all("80%" in row for row in confidence_rows)
    assert "Divergence score: **0.30**" in _values(app.info)
    assert "Synthesis complete." in _values(app.success)
    offline_app.stream.assert_called_once()
    call = offline_app.stream.call_args
    assert call.args == ({"topic": report.topic, "max_rounds": 2},)
    assert call.kwargs["stream_mode"] == "updates"
    assert call.kwargs["config"]["configurable"]["thread_id"] == app.session_state["thread_id"]


def _index_explanation(app):
    return next(
        expander for expander in app.expander
        if expander.label == "How to read this index"
    )


def test_empty_legacy_trace_preserves_stored_index(offline_app, sample_reports):
    from debate.store import save_debate

    report = sample_reports[0].model_copy(update={
        "confidence_score": 0.375,
        "reasoning_trace": [],
        "concession_log": [],
    })
    save_debate(report, conn=offline_app.connection)
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    offline_app.connection.execute("PRAGMA query_only = ON")

    app = offline_app.open()
    app.sidebar.button(key=report.debate_id).click().run()

    assert not app.exception
    assert app.metric[0].value == "37.5 / 100"
    assert "No round trace is available" in (
        _values(app.caption) + _values(app.markdown) + _values(app.info)
    )
    assert app.session_state["final_report"].confidence_score == 0.375
    assert app.session_state["final_report"].reasoning_trace == []
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


def test_history_index_explanation_is_independent_of_current_detector(
    offline_app, sample_reports, monkeypatch,
):
    from debate import divergence
    from debate.store import save_debate

    # Existing report JSON has no detector/mode metadata. The UI cannot infer
    # its historical detector from whichever mode happens to be active today.
    report = sample_reports[0]
    assert not any("mode" in key or "detector" in key for key in report.model_dump())
    save_debate(report, conn=offline_app.connection)
    offline_app.connection.execute("PRAGMA query_only = ON")

    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "nli")
    app = offline_app.open()
    app.sidebar.button(key=report.debate_id).click().run()
    explanation = _index_explanation(app)
    original = _values(explanation.markdown) + _values(explanation.caption)
    assert original
    assert app.metric[0].value == "72.0 / 100"

    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "cosine")
    app.run()
    assert not app.exception
    explanation = _index_explanation(app)
    assert _values(explanation.markdown) + _values(explanation.caption) == original
    assert app.metric[0].value == "72.0 / 100"
    assert app.session_state["final_report"].model_dump() == report.model_dump()
    offline_app.stream.assert_not_called()


def test_index_explains_peak_divergence_without_recomputing_saved_value(
    offline_app, sample_reports,
):
    from debate.store import save_debate

    first = sample_reports[0].reasoning_trace[0]
    report = sample_reports[0].model_copy(update={
        "confidence_score": 0.437,
        "reasoning_trace": [
            first.model_copy(update={"divergence_score": 0.8}),
            first.model_copy(update={"round_num": 1, "divergence_score": 0.2}),
        ],
    })
    save_debate(report, conn=offline_app.connection)
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    app.sidebar.button(key=report.debate_id).click().run()

    assert not app.exception
    # An older report may not reproduce under today's formula. Its persisted
    # value remains authoritative; explaining the current formula is separate.
    assert app.metric[0].value == "43.7 / 100"
    explanation = _index_explanation(app)
    assert "highest divergence across all rounds" in _values(explanation.code)
    detail = _values(explanation.markdown)
    for weight in ("1.0", "0.9", "0.8"):
        assert weight in detail
    assert "18 / 100" in detail  # .8 then .2: peak-based, not final-round 72.
    trace = next(item for item in app.expander if item.label == "Full Reasoning Trace")
    assert "0.800" in _values(trace.markdown)
    assert "0.200" in _values(trace.markdown)
    assert app.session_state["final_report"].confidence_score == 0.437
    offline_app.stream.assert_not_called()


@pytest.mark.parametrize("status,label", [
    ("converged", "No divergent pair detected"),
    ("plateau", "Divergence stopped changing"),
    ("stalled", "No new concessions"),
    ("max_rounds", "Round limit reached"),
    ("partial", "Fallback result"),
])
def test_stop_reason_is_distinct_from_index(
    offline_app, sample_reports, status, label,
):
    from debate.store import save_debate

    # An identical stored index must not imply the same stopping condition.
    report = sample_reports[0].model_copy(update={"convergence_status": status})
    save_debate(report, conn=offline_app.connection)
    app = offline_app.open()
    app.sidebar.button(key=report.debate_id).click().run()

    assert not app.exception
    assert app.metric[0].value == "72.0 / 100"
    assert f"Stop reason: **{label}**" in _values(app.caption)
    offline_app.stream.assert_not_called()
