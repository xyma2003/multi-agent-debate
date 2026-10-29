"""Deterministic Streamlit acceptance tests using synthetic, local reports.

These tests run the real app and compile its real graph, but never execute a
model, download NLI weights, load a developer's .env, or open debates.db.
Run with: python -m pytest tests/test_phase5.py -v
"""
from datetime import datetime, timedelta, timezone
import html
import importlib
import json
from pathlib import Path
import re
import socket
import sqlite3
from string import punctuation
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


def _history_keys(app, reports):
    ids = {report.debate_id for report in reports}
    return [button.key for button in app.sidebar.button if button.key in ids]


def _selected_history_keys(app):
    return [button.key for button in app.sidebar.button if button.proto.type == "primary"]


@pytest.fixture
def export_media(monkeypatch):
    """Retain AppTest's actual in-memory downloads after its runtime closes.

    AppTest does not yet provide a download-button click API. Each run creates
    a fresh media store, so retain that real store and inspect files referenced
    by the final button protos instead of mocking the serializer or widget.
    """
    app_test_module = importlib.import_module("streamlit.testing.v1.app_test")
    storage_class = app_test_module.MemoryMediaFileStorage
    stores = []

    def capture_storage(*args, **kwargs):
        storage = storage_class(*args, **kwargs)
        stores.append(storage)
        return storage

    monkeypatch.setattr(app_test_module, "MemoryMediaFileStorage", capture_storage)
    return stores


def _markdown_export_text(content):
    """Resolve CommonMark literal escapes for content-preservation assertions."""
    return html.unescape(re.sub(
        rf"\\([{re.escape(punctuation)}])", r"\1", content.decode("utf-8"),
    ))


def _report_exports(app, export_media, report, source, *, provenance=None):
    """Check the files served by the two current download buttons."""
    assert not app.exception
    buttons = {button.proto.label: button.proto for button in app.get("download_button")}
    assert set(buttons) == {"Download Markdown", "Download JSON"}
    files = {}
    for label, key, extension, mime in (
        ("Download Markdown", "report_export_markdown", ".md", "text/markdown; charset=utf-8"),
        ("Download JSON", "report_export_json", ".json", "application/json"),
    ):
        button = buttons[label]
        assert button.id.endswith(f"-{key}")
        assert button.ignore_rerun, "Downloading must not rerun an active app session"
        assert not button.disabled
        assert button.url.startswith("/mock/media/")
        stored_file = export_media[-1].get_file(button.url.rsplit("/", 1)[-1])
        assert isinstance(stored_file.content, bytes)
        assert stored_file.mimetype == mime
        assert stored_file.filename.endswith(extension)
        assert Path(stored_file.filename).name == stored_file.filename
        files[extension] = stored_file.content

    payload = json.loads(files[".json"].decode("utf-8"))
    assert payload["schema_version"] == 1
    assert payload["source"]["kind"] == source
    assert payload["source"]["description"]
    assert payload["interpretation"]
    assert payload["report"] == report.model_dump(mode="json")
    markdown = _markdown_export_text(files[".md"])
    assert report.topic in markdown
    assert report.verdict in markdown
    assert report.created_at.strftime("%Y-%m-%d") in markdown
    if provenance is not None:
        assert provenance in markdown
        assert provenance in json.dumps(payload["source"], ensure_ascii=False)
    return files


def _assert_report(app, report, *, expected_status="complete"):
    """Check visible report data, not just the absence of a Streamlit error."""
    assert not app.exception
    assert app.session_state["debate_status"] == expected_status
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
    assert app.text_input(key="debate_topic").label == "Debate topic"
    assert not app.text_input(key="debate_topic").disabled
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
    assert _history_keys(app, sample_reports) == [
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
    app.text_input(key="debate_topic").input("Should we adopt remote work?").run()

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
        {"save_node": {}},
    ])

    app = offline_app.open()
    assert not app.exception
    assert not app.warning
    assert _start_button(app).disabled  # A key alone does not supply a topic.
    app.text_input(key="debate_topic").input(f"  {report.topic}  ").run()
    assert not _start_button(app).disabled
    _start_button(app).click().run()

    _assert_report(app, report)
    assert argument.key_claims[0] in _values(app.markdown)
    # Completion refreshes the sidebar and replaces transient streaming output
    # with the full persisted trace; the agent's own 80% remains visible there.
    confidence_rows = [
        str(item.value) for item in [*app.markdown, *app.caption]
        if "Self-reported confidence" in str(item.value)
    ]
    assert len(confidence_rows) == 1
    assert all("80%" in row for row in confidence_rows)
    assert "0.300" in _values(app.markdown)
    assert "### Debate in Progress" not in _values(app.markdown)
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


@pytest.mark.parametrize("source", ["history", "live"])
def test_saved_detector_metadata_stays_per_round_after_settings_change(
    offline_app, sample_reports, monkeypatch, export_media, source,
):
    from debate import divergence
    from debate.state import DetectorMetadata, RoundRecord
    from debate.store import save_debate

    report = sample_reports[0].model_copy(deep=True)
    first = report.reasoning_trace[0]
    first.detector_metadata = DetectorMetadata(
        mode="nli", model_id="saved/nli-id", input_field="reasoning",
        divergence_threshold=0.625, nli_contradiction_label_index=2,
        evaluated_pair_count=3, sentinels_excluded=True,
    )
    report.reasoning_trace.extend([
        RoundRecord(round_num=1, arguments=[], divergence_score=0.21,
            detector_metadata=DetectorMetadata(
                mode="cosine", model_id="saved/cosine-id", input_field="key_claims",
                divergence_threshold=0.72, cosine_fast_path_threshold=0.98,
                evaluated_pair_count=0, sentinels_excluded=False,
            )),
        RoundRecord(round_num=2, arguments=[], divergence_score=0.19),
    ])
    save_debate(report, conn=offline_app.connection)
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    offline_app.connection.execute("PRAGMA query_only = ON")
    monkeypatch.setenv("DIVERGENCE_MODE", "nli")
    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "nli")
    app = offline_app.open()
    if source == "history":
        app.sidebar.button(key=report.debate_id).click().run()
    else:
        app.session_state["debate_status"] = "complete"
        app.session_state["final_report"] = report
        app.session_state["report_source"] = "live"
        app.run()

    def detector_view():
        return next(item for item in app.expander if item.label == "Saved detector settings")

    view = detector_view()
    original = _values(view.caption) + _values(view.text)
    assert "Round 1: NLI · Round 2: COSINE · Round 3: unknown" in _values(app.caption)
    assert "Saved detector: NLI · 3 evaluated pairs." in original
    assert "Saved detector: COSINE · 0 evaluated pairs." in original
    assert "Saved detector: unknown" in original
    assert "No pairs were evaluated; no pairwise model inference was run." in original
    assert len(view.text) == 3
    assert "Model ID: saved/nli-id" in view.text[0].value
    assert "Input field: reasoning" in view.text[0].value
    assert "Divergence threshold: 0.625" in view.text[0].value
    assert "NLI contradiction label index: 2" in view.text[0].value
    assert "Sentinel filtering: Enabled" in view.text[0].value
    assert "Model ID: saved/cosine-id" in view.text[1].value
    assert "Input field: key_claims" in view.text[1].value
    assert "Cosine fast-path threshold: 0.98" in view.text[1].value
    assert "Sentinel filtering: Disabled" in view.text[1].value
    assert "Model ID:" not in view.text[2].value
    files_before = _report_exports(app, export_media, report, source)
    markdown = _markdown_export_text(files_before[".md"])
    assert "saved/nli-id" in markdown and "saved/cosine-id" in markdown
    assert "Saved detector: unknown" in markdown

    monkeypatch.setenv("DIVERGENCE_MODE", "cosine")
    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "cosine")
    monkeypatch.setattr(divergence, "NLI_CONTRADICTION_THRESHOLD", 0.91)
    monkeypatch.setattr(divergence, "DIVERGE_THRESHOLD", 0.89)
    monkeypatch.setattr(divergence, "CONVERGE_FAST_PATH", 0.99)
    monkeypatch.setattr(divergence, "_NLI_CONTRADICTION", 1)
    app.run()
    assert not app.exception
    assert _values(detector_view().caption) + _values(detector_view().text) == original
    assert _report_exports(app, export_media, report, source) == files_before
    assert app.metric[0].value == "72.0 / 100"
    assert app.session_state["final_report"].model_dump() == report.model_dump()
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


def test_sample_detector_settings_remain_unknown_and_synthetic(
    offline_app, export_media,
):
    from debate.sample import load_offline_example

    example = load_offline_example()
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    _sample_button(app).click().run()
    view = next(item for item in app.expander if item.label == "Saved detector settings")
    assert all(record.detector_metadata is None for record in example.report.reasoning_trace)
    assert _values(view.caption).count("Saved detector: unknown") == len(example.report.reasoning_trace)
    assert "Model ID:" not in _values(view.text)
    files = _report_exports(app, export_media, example.report, "sample", provenance=example.provenance)
    payload = json.loads(files[".json"])
    assert "no model execution" in payload["source"]["description"]
    assert all(record["detector_metadata"] is None for record in payload["report"]["reasoning_trace"])
    assert "Saved detector: unknown" in _markdown_export_text(files[".md"])
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


def _sample_button(app):
    button = app.button(key="offline_example_btn")
    assert button.label == "View offline example"
    return button


def _assert_offline_example(app, example):
    """Verify the actual sample's complete content and its visible provenance."""
    report = example.report
    assert not app.exception
    assert app.session_state["debate_status"] == "complete"
    assert app.session_state["report_source"] == "sample"
    assert app.session_state["final_report"].model_dump() == report.model_dump()
    assert app.session_state["sample_provenance"] == example.provenance
    assert "Fixed synthetic example" in _values(app.info)
    visible = "\n".join(_values(elements) for elements in (
        app.markdown, app.caption, app.info, app.success,
    ))
    assert example.provenance in visible
    assert report.topic in visible
    assert report.verdict in _values(app.info)
    assert app.metric[0].value == "18.0 / 100"
    for point in report.consensus_points:
        assert point in _values(app.success)
    for point in report.disputed_points:
        assert point.topic in [item.label for item in app.expander]
        for position in point.agent_positions.values():
            assert position in visible
    trace = next(item for item in app.expander if item.label == "Full Reasoning Trace")
    trace_text = _values(trace.markdown) + "\n" + _values(trace.caption)
    arguments = [argument for record in report.reasoning_trace for argument in record.arguments]
    assert len(arguments) == 6
    for argument in arguments:
        assert argument.position in trace_text
        assert argument.reasoning in trace_text
        for claim in argument.key_claims:
            assert claim in trace_text
        for concession in argument.concessions:
            assert concession.conceded_point in trace_text
            assert concession.triggered_by_claim in trace_text
            assert concession.rationale in trace_text
    for concession in report.concession_log:
        assert concession.conceded_point in visible
        assert concession.triggered_by_claim in visible
        assert concession.rationale in visible


def test_empty_install_can_read_the_full_offline_example_without_writes(offline_app):
    from debate.sample import load_offline_example

    example = load_offline_example()
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    assert rows_before == []
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    assert not _sample_button(app).disabled
    _sample_button(app).click().run()

    _assert_offline_example(app, example)
    assert "No past debates yet." in _values(app.sidebar.caption)
    assert _start_button(app).disabled
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


def test_sample_history_sample_switches_clear_provenance_and_content(
    offline_app, sample_reports,
):
    from debate.sample import load_offline_example
    from debate.store import save_debate

    example = load_offline_example()
    history_report = sample_reports[0]
    save_debate(history_report, conn=offline_app.connection)
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()

    _sample_button(app).click().run()
    _assert_offline_example(app, example)
    assert _selected_history_keys(app) == []
    assert history_report.verdict not in _values(app.info)

    app.sidebar.button(key=history_report.debate_id).click().run()
    _assert_report(app, history_report)
    assert _selected_history_keys(app) == [history_report.debate_id]
    assert app.session_state["report_source"] == "history"
    assert not app.session_state["sample_provenance"]
    assert "Fixed synthetic example" not in _values(app.info)
    assert example.provenance not in _values(app.caption) + _values(app.info)
    assert example.report.verdict not in _values(app.info)

    _sample_button(app).click().run()
    _assert_offline_example(app, example)
    assert _selected_history_keys(app) == []
    assert history_report.verdict not in _values(app.info)
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


def test_sample_does_not_enable_debate_without_credentials(offline_app):
    from debate.sample import load_offline_example

    example = load_offline_example()
    app = offline_app.open()
    _sample_button(app).click().run()
    app.text_input(key="debate_topic").input("A new topic must still require credentials").run()
    assert _start_button(app).disabled
    # A forged event must also retain the sample without starting the graph.
    _start_button(app).click().run()
    _assert_offline_example(app, example)
    offline_app.stream.assert_not_called()


@pytest.mark.parametrize("initial_source", ["sample", "history"])
def test_mocked_live_debate_refreshes_history_and_clears_prior_selection(
    offline_app, sample_reports, monkeypatch, initial_source, export_media,
):
    from debate.sample import load_offline_example
    from debate.store import save_debate

    example = load_offline_example()
    previous, report = sample_reports
    save_debate(previous, conn=offline_app.connection)
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-placeholder")

    def completed_stream(*args, **kwargs):
        yield {"synthesize_stub": {"final_report": report}}
        save_debate(report, conn=offline_app.connection)
        yield {"save_node": {}}

    offline_app.stream.side_effect = completed_stream
    app = offline_app.open()
    if initial_source == "sample":
        _sample_button(app).click().run()
        _assert_offline_example(app, example)
    else:
        app.sidebar.button(key=previous.debate_id).click().run()
        _assert_report(app, previous)
        assert _selected_history_keys(app) == [previous.debate_id]
    app.text_input(key="debate_topic").input(report.topic).run()
    _start_button(app).click().run()

    _assert_report(app, report)
    assert app.session_state["report_source"] == "live"
    assert not app.session_state["sample_provenance"]
    assert "Fixed synthetic example" not in _values(app.info)
    assert example.provenance not in _values(app.caption) + _values(app.info)
    assert example.report.verdict not in _values(app.info)
    assert _selected_history_keys(app) == []
    assert _history_keys(app, sample_reports) == [report.debate_id, previous.debate_id]
    _report_exports(app, export_media, report, "live")
    offline_app.stream.assert_called_once()


def _open_unsaved_report(offline_app, report, monkeypatch):
    """Finish synthesis, then fail the real save node against a read-only DB."""
    from debate.nodes.save import save_node

    offline_app.connection.execute("PRAGMA query_only = ON")
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-placeholder")

    def unsaved_stream(*args, **kwargs):
        yield {"synthesize_stub": {"final_report": report}}
        # The real persistence node must fail against the isolated read-only DB.
        save_node({"final_report": report})
        yield {"save_node": {}}

    offline_app.stream.side_effect = unsaved_stream
    app = offline_app.open()
    app.text_input(key="debate_topic").input(report.topic).run()
    _start_button(app).click().run()
    return app


def test_persistence_failure_after_synthesis_keeps_error_report_and_downloads(
    offline_app, sample_reports, monkeypatch, export_media,
):
    report = sample_reports[0]
    app = _open_unsaved_report(offline_app, report, monkeypatch)

    _assert_report(app, report, expected_status="error")
    assert "readonly" in app.session_state["error_msg"].lower()
    assert "Save failed" in _values(app.error)
    assert "saving has not been confirmed" in _values(app.warning)
    assert "only in this browser session" in _values(app.caption)
    assert not any(button.key == "reset_btn" for button in app.button)
    assert app.session_state["pending_save_report"].model_dump() == report.model_dump()
    files = _report_exports(app, export_media, report, "live")
    assert json.loads(files[".json"])["source"]["persistence"] == "unconfirmed"
    assert "saving to local history has not been confirmed" in _markdown_export_text(files[".md"])
    assert offline_app.connection.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 0
    offline_app.stream.assert_called_once()


def test_retry_save_keeps_same_payload_needs_no_key_and_does_not_rerun_debate(
    offline_app, sample_reports, monkeypatch, export_media,
):
    from debate.store import load_debate

    report = sample_reports[0]
    original = report.model_dump(mode="json")
    app = _open_unsaved_report(offline_app, report, monkeypatch)
    pending_key = f"pending_retry_{report.debate_id}"
    # Repeated failure keeps the same complete report and downloadable bytes.
    first_files = _report_exports(app, export_media, report, "live")
    app.button(key=pending_key).click().run()
    _assert_report(app, report, expected_status="error")
    assert _report_exports(app, export_media, report, "live") == first_files
    assert offline_app.connection.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 0

    monkeypatch.delenv("GROQ_API_KEY")
    app.run()
    assert _start_button(app).disabled
    assert not app.button(key=pending_key).disabled
    offline_app.connection.execute("PRAGMA query_only = OFF")
    app.button(key=pending_key).click().run()

    _assert_report(app, report)
    assert app.session_state["pending_save_report"] is None
    assert app.session_state["pending_save_error"] is None
    assert app.session_state["error_msg"] is None
    assert "Report saved to Past Debates." in _values(app.success)
    assert not any(button.key == pending_key for button in app.button)
    assert _history_keys(app, [report]) == [report.debate_id]
    assert offline_app.connection.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 1
    assert load_debate(report.debate_id).model_dump(mode="json") == original
    saved_files = _report_exports(app, export_media, report, "live")
    assert "persistence" not in json.loads(saved_files[".json"])["source"]
    app.run()
    assert offline_app.connection.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 1
    offline_app.stream.assert_called_once()


@pytest.mark.parametrize("browse", ["history", "sample"])
def test_browsing_keeps_pending_report_and_blocks_even_forged_new_debate(
    offline_app, sample_reports, monkeypatch, export_media, browse,
):
    from debate.sample import load_offline_example
    from debate.store import save_debate

    pending, history = sample_reports
    save_debate(history, conn=offline_app.connection)
    app = _open_unsaved_report(offline_app, pending, monkeypatch)
    if browse == "history":
        app.sidebar.button(key=history.debate_id).click().run()
        opened = history
        _report_exports(app, export_media, opened, "history")
    else:
        _sample_button(app).click().run()
        example = load_offline_example()
        opened = example.report
        _report_exports(app, export_media, opened, "sample", provenance=example.provenance)
    assert app.session_state["pending_save_report"].model_dump() == pending.model_dump()
    assert app.session_state["final_report"].debate_id == opened.debate_id
    assert _start_button(app).disabled
    _start_button(app).click().run()
    assert app.session_state["final_report"].debate_id == opened.debate_id
    assert app.session_state["pending_save_report"].model_dump() == pending.model_dump()
    offline_app.stream.assert_called_once()

    app.button(key=f"pending_view_{pending.debate_id}").click().run()
    _assert_report(app, pending, expected_status="error")
    files = _report_exports(app, export_media, pending, "live")
    assert json.loads(files[".json"])["source"]["persistence"] == "unconfirmed"
    assert offline_app.connection.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 1


def test_retry_while_browsing_saves_pending_without_replacing_open_report(
    offline_app, sample_reports, monkeypatch, export_media,
):
    from debate.store import load_debate, save_debate

    pending, history = sample_reports
    save_debate(history, conn=offline_app.connection)
    app = _open_unsaved_report(offline_app, pending, monkeypatch)
    app.sidebar.button(key=history.debate_id).click().run()
    before = _report_exports(app, export_media, history, "history")
    offline_app.connection.execute("PRAGMA query_only = OFF")
    app.button(key=f"pending_retry_{pending.debate_id}").click().run()
    _assert_report(app, history)
    assert _report_exports(app, export_media, history, "history") == before
    assert app.session_state["pending_save_report"] is None
    assert load_debate(pending.debate_id).model_dump() == pending.model_dump()
    assert load_debate(history.debate_id).model_dump() == history.model_dump()
    assert offline_app.connection.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 2
    offline_app.stream.assert_called_once()


@pytest.mark.parametrize("browse", [False, True])
def test_discard_is_explicit_and_keeps_saved_history(
    offline_app, sample_reports, monkeypatch, browse,
):
    from debate.store import save_debate

    pending, history = sample_reports
    save_debate(history, conn=offline_app.connection)
    before = list(map(tuple, offline_app.connection.execute("SELECT * FROM debates")))
    app = _open_unsaved_report(offline_app, pending, monkeypatch)
    if browse:
        app.sidebar.button(key=history.debate_id).click().run()
    app.button(key=f"pending_discard_{pending.debate_id}").click().run()
    assert app.session_state["pending_save_report"] is None
    assert app.session_state["pending_save_error"] is None
    app.text_input(key="debate_topic").input("Another synthetic debate").run()
    assert not _start_button(app).disabled
    if browse:
        _assert_report(app, history)
    else:
        assert app.session_state["debate_status"] == "idle"
        assert app.session_state["final_report"] is None
        assert not app.get("download_button")
    assert list(map(tuple, offline_app.connection.execute("SELECT * FROM debates"))) == before
    offline_app.stream.assert_called_once()


@pytest.mark.parametrize("confirmation", [False, True])
def test_lost_stream_confirmation_is_retryable_without_duplicate_history(
    offline_app, sample_reports, monkeypatch, confirmation,
):
    from debate.nodes.save import save_node
    from debate.store import load_debate

    report = sample_reports[0]
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-placeholder")

    def stream(*args, **kwargs):
        yield {"synthesize_stub": {"final_report": report}}
        save_node({"final_report": report})
        if confirmation:
            yield {"save_node": {}}
        raise RuntimeError("Synthetic stream transport failure")

    offline_app.stream.side_effect = stream
    app = offline_app.open()
    app.text_input(key="debate_topic").input(report.topic).run()
    _start_button(app).click().run()
    assert app.session_state["debate_status"] == "error"
    if confirmation:
        assert app.session_state["pending_save_report"] is None
        assert not app.get("download_button")
        assert "Debate failed" in _values(app.error)
    else:
        assert app.session_state["pending_save_report"].model_dump() == report.model_dump()
        app.button(key=f"pending_retry_{report.debate_id}").click().run()
        _assert_report(app, report)
    assert offline_app.connection.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 1
    assert load_debate(report.debate_id).model_dump() == report.model_dump()
    offline_app.stream.assert_called_once()


def test_stream_ending_before_save_confirmation_retains_generated_report(
    offline_app, sample_reports, monkeypatch,
):
    report = sample_reports[0]
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-placeholder")
    offline_app.stream.side_effect = lambda *a, **kw: iter([
        {"synthesize_stub": {"final_report": report}},
    ])
    app = offline_app.open()
    app.text_input(key="debate_topic").input(report.topic).run()
    _start_button(app).click().run()
    _assert_report(app, report, expected_status="error")
    assert "without confirming" in app.session_state["pending_save_error"]
    assert app.session_state["pending_save_report"].model_dump() == report.model_dump()
    assert offline_app.connection.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 0
    offline_app.stream.assert_called_once()


def test_failure_before_this_runs_synthesis_does_not_capture_old_report(
    offline_app, sample_reports, monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "offline-test-placeholder")
    app = offline_app.open()
    _sample_button(app).click().run()
    offline_app.stream.side_effect = RuntimeError("Synthetic agent failure")
    app.text_input(key="debate_topic").input("A new synthetic debate").run()
    _start_button(app).click().run()
    assert app.session_state["debate_status"] == "error"
    assert app.session_state["final_report"] is None
    assert app.session_state["pending_save_report"] is None
    assert not app.get("download_button")
    assert "Debate failed" in _values(app.error)
    offline_app.stream.assert_called_once()


@pytest.mark.parametrize("action", ["view", "retry", "discard"])
def test_running_state_rejects_forged_pending_report_actions(
    offline_app, sample_reports, monkeypatch, action,
):
    report = sample_reports[0]
    app = _open_unsaved_report(offline_app, report, monkeypatch)
    app.session_state["debate_status"] = "running"
    app.run()
    button = app.button(key=f"pending_{action}_{report.debate_id}")
    assert button.disabled
    button.click().run()
    assert not app.exception
    assert app.session_state["debate_status"] == "running"
    assert app.session_state["pending_save_report"].model_dump() == report.model_dump()
    assert not app.get("download_button")
    assert offline_app.connection.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 0
    offline_app.stream.assert_called_once()


def test_old_retry_event_cannot_save_a_different_pending_report(
    offline_app, sample_reports, monkeypatch,
):
    from debate.nodes.save import save_node

    first, second = sample_reports
    app = _open_unsaved_report(offline_app, first, monkeypatch)
    old_retry = app.button(key=f"pending_retry_{first.debate_id}")
    app.button(key=f"pending_discard_{first.debate_id}").click().run()

    def second_stream(*args, **kwargs):
        yield {"synthesize_stub": {"final_report": second}}
        save_node({"final_report": second})

    offline_app.stream.side_effect = second_stream
    app.text_input(key="debate_topic").input(second.topic).run()
    _start_button(app).click().run()
    assert app.session_state["pending_save_report"].debate_id == second.debate_id
    offline_app.connection.execute("PRAGMA query_only = OFF")
    old_retry.click().run()
    assert not app.exception
    assert app.session_state["pending_save_report"].model_dump() == second.model_dump()
    assert offline_app.connection.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 0
    assert offline_app.stream.call_count == 2


@pytest.mark.parametrize("failure", ["missing", "invalid_json"])
def test_sample_read_failure_preserves_report_and_retry_recovers(
    offline_app, sample_reports, monkeypatch, tmp_path, failure,
):
    from debate import sample
    from debate.store import save_debate

    example = sample.load_offline_example()
    valid_path = sample.EXAMPLE_PATH
    report = sample_reports[0]
    save_debate(report, conn=offline_app.connection)
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    app.sidebar.button(key=report.debate_id).click().run()
    broken_path = tmp_path / "unavailable-example.json"
    if failure == "invalid_json":
        broken_path.write_text("{broken JSON", encoding="utf-8")
    monkeypatch.setattr(sample, "EXAMPLE_PATH", broken_path)

    _sample_button(app).click().run()
    _assert_report(app, report)
    assert "Could not open the offline example" in _values(app.error)
    assert app.session_state["report_source"] == "history"
    assert not app.session_state["sample_provenance"]
    assert "Fixed synthetic example" not in _values(app.info)

    monkeypatch.setattr(sample, "EXAMPLE_PATH", valid_path)
    _sample_button(app).click().run()
    _assert_offline_example(app, example)
    assert "Could not open the offline example" not in _values(app.error)
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


def test_running_debate_rejects_even_a_forged_sample_button_event(
    offline_app, sample_reports, monkeypatch,
):
    from debate import sample

    app = offline_app.open()
    report = sample_reports[0]
    app.session_state["debate_status"] = "running"
    app.session_state["final_report"] = report
    app.session_state["report_source"] = "live"
    app.session_state["sample_provenance"] = None
    app.session_state["thread_id"] = "in-progress-thread"
    loader = Mock(side_effect=AssertionError("Running debates must not load the sample"))
    monkeypatch.setattr(sample, "load_offline_example", loader)
    app.run()
    assert _sample_button(app).disabled
    _sample_button(app).click().run()

    assert not app.exception
    assert app.session_state["debate_status"] == "running"
    assert app.session_state["final_report"].model_dump() == report.model_dump()
    assert app.session_state["report_source"] == "live"
    assert app.session_state["thread_id"] == "in-progress-thread"
    loader.assert_not_called()
    offline_app.stream.assert_not_called()


@pytest.fixture
def paged_reports(offline_app, sample_reports):
    """More than two pages, including distinct reports with identical titles."""
    from debate.store import save_debate

    reports = [sample_reports[0].model_copy(update={
        "debate_id": f"synthetic-report-page-{index:02d}",
        "topic": "Remote work policy" if index % 2 == 0 else "Office collaboration",
        "verdict": f"Synthetic decision {index}; review the trial outcomes.",
        "created_at": datetime(2026, 2, 1, tzinfo=timezone.utc) + timedelta(days=index),
    }) for index in range(23)]
    for report in reports:
        save_debate(report, conn=offline_app.connection)
    return reports


def test_history_pages_cover_all_reports_without_writes(offline_app, paged_reports):
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    expected = [report.debate_id for report in reversed(paged_reports)]
    assert not app.exception
    assert app.sidebar.text_input(key="history_query").label == "Search topics"
    assert _history_keys(app, paged_reports) == expected[:10]
    assert app.sidebar.button(key="history_prev").disabled
    assert not app.sidebar.button(key="history_next").disabled

    app.sidebar.button(key="history_next").click().run()
    assert not app.exception
    assert _history_keys(app, paged_reports) == expected[10:20]
    assert not app.sidebar.button(key="history_prev").disabled
    assert not app.sidebar.button(key="history_next").disabled

    app.sidebar.button(key="history_next").click().run()
    assert not app.exception
    assert _history_keys(app, paged_reports) == expected[20:]
    assert app.sidebar.button(key="history_next").disabled
    app.sidebar.button(key="history_prev").click().run()
    assert _history_keys(app, paged_reports) == expected[10:20]
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


def test_search_resets_page_clear_restores_history_and_preserves_open_report(
    offline_app, paged_reports,
):
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    opened = paged_reports[21]
    app.sidebar.button(key=opened.debate_id).click().run()
    _assert_report(app, opened)
    app.sidebar.button(key="history_next").click().run()
    _assert_report(app, opened)

    app.sidebar.text_input(key="history_query").input("  rEmOtE  ").run()
    matching = [r.debate_id for r in reversed(paged_reports) if "Remote" in r.topic]
    assert _history_keys(app, paged_reports) == matching[:10]
    assert app.sidebar.button(key="history_prev").disabled
    assert app.sidebar.button(key="history_clear").label == "Clear search"
    _assert_report(app, opened)
    assert app.session_state["report_source"] == "history"
    assert not app.session_state["sample_provenance"]

    app.sidebar.button(key="history_next").click().run()
    assert _history_keys(app, paged_reports) == matching[10:]
    _assert_report(app, opened)
    app.sidebar.text_input(key="history_query").input("Office").run()
    matching = [r.debate_id for r in reversed(paged_reports) if "Office" in r.topic]
    assert _history_keys(app, paged_reports) == matching[:10]
    assert app.sidebar.button(key="history_prev").disabled

    app.sidebar.text_input(key="history_query").input("a topic with no matches").run()
    assert not app.exception
    assert _history_keys(app, paged_reports) == []
    assert "No past debates yet." not in _values(app.sidebar.caption)
    assert "No saved debates match" in _values(app.sidebar.caption)
    _assert_report(app, opened)
    assert app.session_state["report_source"] == "history"
    app.sidebar.button(key="history_clear").click().run()
    assert app.sidebar.text_input(key="history_query").value == ""
    assert _history_keys(app, paged_reports) == [
        r.debate_id for r in reversed(paged_reports)
    ][:10]
    _assert_report(app, opened)
    offline_app.stream.assert_not_called()


def test_search_and_pagination_preserve_offline_example_provenance(
    offline_app, paged_reports,
):
    from debate.sample import load_offline_example

    example = load_offline_example()
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    _sample_button(app).click().run()
    _assert_offline_example(app, example)
    app.sidebar.button(key="history_next").click().run()
    _assert_offline_example(app, example)
    app.sidebar.text_input(key="history_query").input("no-saved-topic-like-this").run()
    _assert_offline_example(app, example)
    app.sidebar.button(key="history_clear").click().run()
    _assert_offline_example(app, example)
    offline_app.stream.assert_not_called()


def test_same_topic_reports_show_identity_and_open_the_correct_record(
    offline_app, paged_reports,
):
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    first, second = paged_reports[22], paged_reports[20]
    assert first.topic == second.topic
    assert app.sidebar.button(key=first.debate_id).label == app.sidebar.button(key=second.debate_id).label
    captions = _values(app.sidebar.caption)
    for report in (first, second):
        assert report.debate_id[-8:] in captions
        assert report.created_at.strftime("%Y-%m-%d") in captions
        app.sidebar.button(key=report.debate_id).click().run()
        _assert_report(app, report)
        # The open report remains identifiable even after it is filtered out.
        assert report.debate_id in _values(app.main.caption)
        assert report.created_at.strftime("%Y-%m-%d") in _values(app.main.caption)
    offline_app.stream.assert_not_called()


def test_history_refresh_clamps_last_page_after_external_row_removal(
    offline_app, paged_reports,
):
    app = offline_app.open()
    app.sidebar.button(key="history_next").click().run()
    app.sidebar.button(key="history_next").click().run()
    assert len(_history_keys(app, paged_reports)) == 3
    opened = paged_reports[2]
    app.sidebar.button(key=opened.debate_id).click().run()

    # Simulate a separate process removing the final page between UI reruns.
    offline_app.connection.executemany(
        "DELETE FROM debates WHERE debate_id = ?",
        [(report.debate_id,) for report in paged_reports[:3]],
    )
    offline_app.connection.commit()
    offline_app.connection.execute("PRAGMA query_only = ON")
    app.sidebar.button(key="history_refresh").click().run()
    assert not app.exception
    assert _history_keys(app, paged_reports) == [
        report.debate_id for report in reversed(paged_reports[3:13])
    ]
    assert app.sidebar.button(key="history_next").disabled
    assert not app.sidebar.button(key="history_prev").disabled
    _assert_report(app, opened)
    assert app.session_state["report_source"] == "history"
    offline_app.stream.assert_not_called()


@pytest.mark.parametrize("failure", ["missing", "corrupt", "database", "wrong_id"])
def test_history_load_failure_keeps_current_report_and_can_retry(
    offline_app, sample_reports, monkeypatch, failure,
):
    from debate import store

    for report in sample_reports:
        store.save_debate(report, conn=offline_app.connection)
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    current, target = sample_reports
    app.sidebar.button(key=current.debate_id).click().run()
    real_loader = store.load_debate
    errors = {
        "corrupt": ValueError("Synthetic invalid report JSON"),
        "database": sqlite3.OperationalError("Synthetic database lock"),
    }
    broken_loader = Mock(
        return_value=current if failure == "wrong_id" else None,
        side_effect=errors.get(failure),
    )
    monkeypatch.setattr(store, "load_debate", broken_loader)
    app.sidebar.button(key=target.debate_id).click().run()
    _assert_report(app, current)
    assert app.session_state["report_source"] == "history"
    assert not app.session_state["sample_provenance"]
    notice = _values(app.sidebar.error) + _values(app.sidebar.warning)
    assert "Your open report has been kept" in notice
    broken_loader.assert_called_once_with(target.debate_id)

    monkeypatch.setattr(store, "load_debate", real_loader)
    app.sidebar.button(key=target.debate_id).click().run()
    _assert_report(app, target)
    assert not app.sidebar.error
    offline_app.stream.assert_not_called()


@pytest.mark.parametrize("operation", ["count_debates", "list_debates"])
def test_history_database_read_error_does_not_break_main_view_and_retry_recovers(
    offline_app, sample_reports, monkeypatch, operation,
):
    from debate import store
    from debate.sample import load_offline_example

    report = sample_reports[0]
    store.save_debate(report, conn=offline_app.connection)
    offline_app.connection.execute("PRAGMA query_only = ON")
    example = load_offline_example()
    app = offline_app.open()
    _sample_button(app).click().run()
    original = getattr(store, operation)
    monkeypatch.setattr(store, operation, Mock(side_effect=sqlite3.OperationalError("synthetic locked database")))
    app.run()
    _assert_offline_example(app, example)
    assert _values(app.sidebar.error)
    assert "No past debates yet." not in _values(app.sidebar.caption)
    retry = app.sidebar.button(key="history_refresh")
    assert not retry.disabled
    monkeypatch.setattr(store, operation, original)
    retry.click().run()
    _assert_offline_example(app, example)
    assert _history_keys(app, sample_reports) == [report.debate_id]
    assert not app.sidebar.error
    offline_app.stream.assert_not_called()


@pytest.mark.parametrize("event", ["next", "previous", "clear", "report", "query"])
def test_running_debate_disables_history_and_rejects_forged_button_events(
    offline_app, paged_reports, monkeypatch, event,
):
    from debate import store

    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    app.sidebar.text_input(key="history_query").input("Remote").run()
    if event == "previous":
        app.sidebar.button(key="history_next").click().run()
    old_keys = _history_keys(app, paged_reports)
    report = paged_reports[0]
    app.session_state["debate_status"] = "running"
    app.session_state["final_report"] = report
    app.session_state["report_source"] = "live"
    app.session_state["sample_provenance"] = None
    app.session_state["thread_id"] = "synthetic-running-thread"
    loader = Mock(side_effect=AssertionError("Running debates must not load history"))
    monkeypatch.setattr(store, "load_debate", loader)
    app.run()
    assert app.sidebar.text_input(key="history_query").disabled
    for key in ("history_prev", "history_next", "history_clear", "history_refresh", *old_keys):
        assert app.sidebar.button(key=key).disabled
    if event == "query":
        app.sidebar.text_input(key="history_query").input("Office").run()
    else:
        event_key = {
            "next": "history_next", "previous": "history_prev",
            "clear": "history_clear", "report": old_keys[0],
        }[event]
        app.sidebar.button(key=event_key).click().run()
    assert not app.exception
    assert app.session_state["_history_applied_query"] == "Remote"
    assert _history_keys(app, paged_reports) == old_keys
    assert app.session_state["debate_status"] == "running"
    assert app.session_state["final_report"].model_dump() == report.model_dump()
    assert app.session_state["report_source"] == "live"
    assert app.session_state["thread_id"] == "synthetic-running-thread"
    loader.assert_not_called()
    offline_app.stream.assert_not_called()


def test_history_downloads_serve_complete_chinese_report_without_writes(
    offline_app, sample_reports, export_media,
):
    from debate.state import DebateReport
    from debate.store import save_debate

    data = sample_reports[0].model_dump()
    data.update(
        debate_id="export-chinese-report",
        topic="中文议题：是否实行弹性办公",
        verdict="先开展小规模试点，再根据结果调整。",
        consensus_points=["共同测量团队协作效果"],
        disputed_points=[{
            "topic": "远程协作的沟通成本",
            "agent_positions": {
                "optimist": "异步沟通能够降低干扰",
                "pessimist": "新成员需要更多同步指导",
            },
        }],
    )
    concession = {
        "conceded_point": "新成员入职期需要额外支持",
        "triggered_by_agent": "pessimist",
        "triggered_by_claim": "隐性知识难以通过文档传递",
        "rationale": "为入职期增加固定辅导时间可以降低风险。",
    }
    data["reasoning_trace"] = [{
        "round_num": index,
        "divergence_score": (0.8, 0.2)[index],
        "arguments": [{
            "agent_role": "optimist",
            "round_num": index,
            "position": f"第{index + 1}轮主张：逐步验证弹性办公",
            "reasoning": f"第{index + 1}轮论证：通过真实协作指标检查影响。",
            "confidence": 0.8,
            "key_claims": ["试点规模应可控", "协作效果应测量", "入职支持应保留"],
            "concessions": [concession] if index else [],
            "is_sentinel": False,
        }],
    } for index in range(2)]
    data["concession_log"] = [concession]
    report = DebateReport.model_validate(data)
    save_debate(report, conn=offline_app.connection)
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    offline_app.connection.execute("PRAGMA query_only = ON")

    app = offline_app.open()
    app.sidebar.button(key=report.debate_id).click().run()
    exports = _report_exports(app, export_media, report, "history")
    markdown = _markdown_export_text(exports[".md"])
    assert "Round 1" in markdown
    assert "Round 2" in markdown
    for record in report.reasoning_trace:
        for argument in record.arguments:
            assert argument.position in markdown
            assert argument.reasoning in markdown
            for claim in argument.key_claims:
                assert claim in markdown
    for text in (
        *report.consensus_points,
        report.disputed_points[0].topic,
        *report.disputed_points[0].agent_positions.values(),
        *concession.values(),
    ):
        assert text in markdown
    # The saved value is deliberately inconsistent with the current formula:
    # exporting must preserve historical facts instead of recomputing scores.
    assert json.loads(exports[".json"])["report"]["confidence_score"] == 0.72
    assert "中文议题".encode("utf-8") in exports[".json"]
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


def test_history_search_and_pagination_keep_downloads_bound_to_open_report(
    offline_app, paged_reports, export_media,
):
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    offline_app.connection.execute("PRAGMA query_only = ON")
    report = paged_reports[21]
    app = offline_app.open()
    app.sidebar.button(key=report.debate_id).click().run()
    original = _report_exports(app, export_media, report, "history")

    app.sidebar.button(key="history_next").click().run()
    assert _report_exports(app, export_media, report, "history") == original
    app.sidebar.text_input(key="history_query").input("Remote").run()
    assert report.debate_id not in _history_keys(app, paged_reports)
    assert _report_exports(app, export_media, report, "history") == original
    app.sidebar.text_input(key="history_query").input("No matching topic").run()
    assert not _history_keys(app, paged_reports)
    assert _report_exports(app, export_media, report, "history") == original
    app.sidebar.button(key="history_clear").click().run()
    assert _report_exports(app, export_media, report, "history") == original
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


def test_selecting_another_same_topic_report_changes_download_content(
    offline_app, paged_reports, export_media,
):
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    offline_app.connection.execute("PRAGMA query_only = ON")
    first, second = paged_reports[22], paged_reports[20]
    assert first.topic == second.topic
    app = offline_app.open()
    app.sidebar.button(key=first.debate_id).click().run()
    first_exports = _report_exports(app, export_media, first, "history")
    app.sidebar.button(key=second.debate_id).click().run()
    second_exports = _report_exports(app, export_media, second, "history")
    for extension in (".md", ".json"):
        assert first_exports[extension] != second_exports[extension]
    assert first.verdict not in _markdown_export_text(second_exports[".md"])
    assert first.verdict not in second_exports[".json"].decode("utf-8")
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


def test_sample_downloads_keep_synthetic_provenance_after_search_and_switching(
    offline_app, paged_reports, export_media,
):
    from debate.sample import load_offline_example

    example = load_offline_example()
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    _sample_button(app).click().run()
    original = _report_exports(
        app, export_media, example.report, "sample", provenance=example.provenance,
    )
    app.sidebar.button(key="history_next").click().run()
    assert _report_exports(
        app, export_media, example.report, "sample", provenance=example.provenance,
    ) == original
    app.sidebar.text_input(key="history_query").input("No matching topic").run()
    assert _report_exports(
        app, export_media, example.report, "sample", provenance=example.provenance,
    ) == original
    app.sidebar.button(key="history_clear").click().run()
    report = paged_reports[-1]
    app.sidebar.button(key=report.debate_id).click().run()
    history_files = _report_exports(app, export_media, report, "history")
    assert example.provenance not in _markdown_export_text(history_files[".md"])
    assert example.provenance not in history_files[".json"].decode("utf-8")
    _sample_button(app).click().run()
    assert _report_exports(
        app, export_media, example.report, "sample", provenance=example.provenance,
    ) == original
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()


@pytest.mark.parametrize("status", ["idle", "running", "error"])
def test_unfinished_states_offer_no_report_download_even_with_a_stale_report(
    offline_app, sample_reports, status,
):
    offline_app.connection.execute("PRAGMA query_only = ON")
    app = offline_app.open()
    assert not app.get("download_button")
    app.session_state["debate_status"] = status
    app.session_state["final_report"] = sample_reports[0]
    app.session_state["report_source"] = "live"
    app.session_state["error_msg"] = "Synthetic failure" if status == "error" else None
    app.run()

    assert not app.exception
    assert not app.get("download_button")
    assert "Debate Result" not in _values(app.subheader)
    assert app.session_state["debate_status"] == status
    offline_app.stream.assert_not_called()


def test_nonfinite_report_export_failure_keeps_content_visible_and_can_retry(
    offline_app, sample_reports, monkeypatch, export_media,
):
    from debate import store

    report = sample_reports[0]
    store.save_debate(report, conn=offline_app.connection)
    rows_before = offline_app.connection.execute("SELECT * FROM debates").fetchall()
    offline_app.connection.execute("PRAGMA query_only = ON")
    # Legacy/load-time data can contain nonfinite divergence because that field
    # is unconstrained. Retain a sound stored row and simulate this read result.
    malformed = report.model_copy(deep=True)
    malformed.reasoning_trace[0].divergence_score = float("nan")
    loader = Mock(return_value=malformed)
    monkeypatch.setattr(store, "load_debate", loader)
    app = offline_app.open()
    app.sidebar.button(key=report.debate_id).click().run()

    assert not app.exception
    assert app.session_state["debate_status"] == "complete"
    assert "Debate Result" in _values(app.subheader)
    assert report.topic in _values(app.markdown)
    assert report.verdict in _values(app.info)
    assert report.reasoning_trace[0].arguments[0].reasoning in _values(app.markdown)
    assert not app.get("download_button")
    assert (
        "This report could not be exported. Its contents remain visible; "
        "please select another report or try again."
    ) in _values(app.warning)

    loader.return_value = report
    app.sidebar.button(key=report.debate_id).click().run()
    _report_exports(app, export_media, report, "history")
    assert "This report could not be exported" not in _values(app.warning)
    assert offline_app.connection.execute("SELECT * FROM debates").fetchall() == rows_before
    offline_app.stream.assert_not_called()
