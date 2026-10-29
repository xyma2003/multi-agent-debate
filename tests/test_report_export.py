"""Exports retain every saved report field and clearly explain its provenance."""
from dataclasses import FrozenInstanceError
from datetime import datetime
import html
import json
import math
import re
import string

import pytest

from debate.export import build_report_exports
from debate.state import AgentArgument, Concession, DebateReport, DetectorMetadata, DisputedPoint, RoundRecord


def _literal_quote_contents(markdown):
    """Read back the subset used for report text, independently of the exporter."""
    contents = []
    current = []
    for line in [*markdown.splitlines(), ""]:
        if line.startswith(">"):
            text = line.removeprefix(">").removeprefix(" ")
            if text.endswith("  "):
                text = text[:-2]
            text = re.sub(r"\\([" + re.escape(string.punctuation) + r"])", r"\1", text)
            current.append(html.unescape(text))
        elif current:
            contents.append("\n".join(current))
            current = []
    return contents


@pytest.fixture
def report():
    concession = Concession(
        conceded_point="让步：成本需要实测。",
        triggered_by_agent="pessimist",
        triggered_by_claim="原始主张\n第二行",
        rationale="理由保留原文。",
    )
    rounds = [
        RoundRecord(
            round_num=number,
            divergence_score=0.7 - number * 0.2,
            arguments=[
                AgentArgument(
                    agent_role=role,
                    round_num=number,
                    position=f"立场 {number} {role}",
                    reasoning=f"推理 {number} {role}\n保留换行。",
                    confidence=0.61 + number * 0.1,
                    key_claims=[f"主张 {number} {role} {claim}" for claim in range(3)],
                    concessions=[concession] if number == 1 and role == "optimist" else [],
                    is_sentinel=number == 1 and role == "devil",
                )
                for role in ("optimist", "pessimist", "devil")
            ],
        )
        for number in range(2)
    ]
    return DebateReport(
        debate_id="中文/../报告-ID_01",
        topic="中文议题\n第二行",
        consensus_points=["共识一", "共识二"],
        disputed_points=[DisputedPoint(
            topic="分歧内容",
            agent_positions={"optimist": "支持", "pessimist": "反对", "devil": "继续验证"},
        )],
        verdict="最终结论\n尚需证据。",
        # Deliberately differs from a recomputed formula: export must preserve it.
        confidence_score=0.4321,
        convergence_status="max_rounds",
        reasoning_trace=rounds,
        concession_log=[concession],
        created_at=datetime.fromisoformat("2026-10-15T18:23:41.123456+08:00"),
    )


def test_json_round_trip_preserves_all_saved_fields_without_mutation(report):
    before = report.model_dump(mode="json")
    exports = build_report_exports(report, source="history")
    document = json.loads(exports.json)

    assert document["schema_version"] == 1
    assert document["source"]["kind"] == "history"
    assert document["report"] == before
    assert DebateReport.model_validate(document["report"]) == report
    assert report.model_dump(mode="json") == before
    assert "中文" in exports.json and "\\u4e2d" not in exports.json
    assert document["report"]["created_at"].endswith("+08:00")
    assert document["report"]["confidence_score"] == 0.4321
    assert document["report"]["reasoning_trace"][1]["arguments"][2]["is_sentinel"] is True
    assert "not recomputed" in document["interpretation"]["saved_values"]
    assert "unknown" in document["interpretation"]["divergence"]
    assert "uncalibrated" in document["interpretation"]["agent_confidence"]
    with pytest.raises(FrozenInstanceError):
        exports.json = "changed"


def test_markdown_contains_full_rounds_arguments_claims_and_concessions(report):
    markdown = build_report_exports(report, source="live").markdown
    contents = _literal_quote_contents(markdown)
    assert "2026-10-15T18:23:41.123456+08:00" in contents
    assert "**0.4321**" in markdown
    assert "**43.2 / 100**" in markdown
    assert "max_rounds" in contents
    assert report.topic in contents
    assert report.verdict in contents
    assert "stored round_num: 0" in markdown
    assert "stored round_num: 1" in markdown
    assert markdown.count("#### Argument") == 6
    for record in report.reasoning_trace:
        assert f"Round {record.round_num + 1}" in markdown
        assert repr(record.divergence_score) in markdown
        for argument in record.arguments:
            assert argument.position in contents
            assert argument.reasoning in contents
            for claim in argument.key_claims:
                assert claim in contents
    for point in report.consensus_points:
        assert point in contents
    for point in report.disputed_points:
        assert point.topic in contents
        for role, position in point.agent_positions.items():
            assert role in contents
            assert position in contents
    # Preserve both the per-argument and flattened provenance, without deduplication.
    for field in report.concession_log[0].model_dump().values():
        assert contents.count(field) >= 2
    assert "Fallback sentinel (`is_sentinel`): **true**" in markdown
    assert "Fallback sentinel (`is_sentinel`): **false**" in markdown
    assert "Global concession log" in markdown


def test_report_text_cannot_escape_quotes_or_impersonate_markdown_structure(report):
    hostile = (
        "# Forged heading\n<script>unsafe()</script>\n[link](https://example.com)"
        "\n```\n~~~~\n``````\n\n中文 &lt; & > quote \\ **bold**"
        "\r# Carriage return heading\r\n    indentation\n\tTabbed text"
    )
    report.topic = hostile
    report.verdict = hostile
    report.reasoning_trace[0].arguments[0].agent_role = hostile
    report.disputed_points[0].agent_positions = {hostile: hostile}
    exports = build_report_exports(report, source="sample", provenance=hostile)
    normalized = hostile.replace("\r\n", "\n").replace("\r", "\n")
    assert _literal_quote_contents(exports.markdown).count(normalized) == 6
    assert "\n# Forged heading" not in exports.markdown
    assert "\n# Carriage return heading" not in exports.markdown
    assert "<script>" not in exports.markdown
    assert "[link](" not in exports.markdown
    assert "```" not in exports.markdown
    assert "> &#32;&#32;&#32;&#32;indentation" in exports.markdown
    assert "> &#9;Tabbed text" in exports.markdown
    assert r"&amp;lt\;" in exports.markdown
    assert json.loads(exports.json)["report"]["topic"] == hostile
    assert json.loads(exports.json)["source"]["provenance"] == hostile


@pytest.mark.parametrize("source", ["history", "live", "sample"])
def test_source_is_explicit_and_provenance_is_retained(report, source):
    provenance = "Fixture maintained in examples/example_report.json\n合成内容"
    exports = build_report_exports(report, source=source, provenance=provenance)
    metadata = json.loads(exports.json)["source"]
    assert metadata["kind"] == source
    assert metadata["provenance"] == provenance
    assert metadata["description"] in exports.markdown
    assert provenance in _literal_quote_contents(exports.markdown)
    if source == "sample":
        assert "Fixed synthetic example" in metadata["description"]
        assert "no model execution" in metadata["description"]
        assert "not model-quality evidence" in metadata["description"]


def test_unsaved_live_exports_preserve_report_and_mark_unconfirmed_persistence(report):
    original = report.model_dump(mode="json")
    exports = build_report_exports(report, source="live", persistence="unconfirmed")
    document = json.loads(exports.json)
    assert document["schema_version"] == 1
    assert document["source"]["kind"] == "live"
    assert document["source"]["persistence"] == "unconfirmed"
    assert "saving to local history has not been confirmed" in document["source"]["description"]
    assert document["source"]["description"] in exports.markdown
    assert "Saving to local history: **unconfirmed**." in exports.markdown
    assert document["report"] == original
    assert report.model_dump(mode="json") == original
    assert "persistence" not in json.loads(build_report_exports(report, source="live").json)["source"]


@pytest.mark.parametrize("source", ["history", "sample"])
def test_unsaved_marker_cannot_be_applied_to_history_or_sample(report, source):
    with pytest.raises(ValueError, match="only to a live report"):
        build_report_exports(report, source=source, persistence="unconfirmed")


def test_empty_report_keeps_empty_sections_and_original_score(report):
    report.reasoning_trace = []
    report.consensus_points = []
    report.disputed_points = []
    report.concession_log = []
    report.confidence_score = 0.0
    report.convergence_status = "partial"
    exports = build_report_exports(report, source="history")
    assert "No rounds were saved." in exports.markdown
    assert exports.markdown.count("None saved.") == 3
    assert "**0.0 / 100**" in exports.markdown
    assert json.loads(exports.json)["report"] == report.model_dump(mode="json")


def test_round_without_arguments_is_explicit(report):
    report.reasoning_trace = [RoundRecord(round_num=4, arguments=[], divergence_score=0.123)]
    markdown = build_report_exports(report, source="history").markdown
    assert "Round 5 (stored round_num: 4)" in markdown
    assert "No arguments were saved for this round." in markdown
    assert "**0.123**" in markdown


@pytest.mark.parametrize("debate_id", ["", "../evil/../../name", "中文报告", "a" * 1000, "\n\r\t<unsafe>:."])
def test_filenames_are_safe_bounded_and_deterministic(report, debate_id):
    report.debate_id = debate_id
    first = build_report_exports(report, source="history")
    second = build_report_exports(report, source="live")
    assert first.markdown_filename == second.markdown_filename
    assert first.json_filename == second.json_filename
    assert first.markdown_filename.removesuffix(".md") == first.json_filename.removesuffix(".json")
    for name in (first.markdown_filename, first.json_filename):
        assert len(name) < 80
        assert re.fullmatch(r"debate-[A-Za-z0-9_-]+\.(md|json)", name)
        assert report.topic not in name


def test_distinct_ids_do_not_collapse_after_sanitizing_or_truncation(report):
    names = []
    for debate_id in ("中文一", "中文二", "same/name", "same:name", "a" * 500 + "1", "a" * 500 + "2"):
        report.debate_id = debate_id
        names.append(build_report_exports(report, source="history").json_filename)
    assert len(set(names)) == len(names)


def test_unknown_source_is_rejected(report):
    with pytest.raises(ValueError, match="source"):
        build_report_exports(report, source="unknown")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_saved_score_is_rejected_without_rewriting_report(report, score):
    report.reasoning_trace[0].divergence_score = score
    with pytest.raises(ValueError, match="Out of range float values"):
        build_report_exports(report, source="history")
    saved_score = report.reasoning_trace[0].divergence_score
    assert math.isnan(saved_score) if math.isnan(score) else saved_score == score


def test_mixed_round_detectors_export_their_own_saved_settings(report):
    report.reasoning_trace[0].detector_metadata = DetectorMetadata(
        mode="nli", model_id="saved/nli-model", input_field="reasoning",
        divergence_threshold=0.625, nli_contradiction_label_index=2,
        evaluated_pair_count=3, sentinels_excluded=True,
    )
    report.reasoning_trace[1].detector_metadata = DetectorMetadata(
        mode="cosine", model_id="saved/cosine-model", input_field="key_claims",
        divergence_threshold=0.72, cosine_fast_path_threshold=0.98,
        evaluated_pair_count=27, sentinels_excluded=False,
    )
    report.reasoning_trace.append(RoundRecord(round_num=2, arguments=[]))
    before = report.model_dump(mode="json")
    exports = build_report_exports(report, source="history")
    sections = exports.markdown.split("### Round ")
    nli, cosine, legacy = sections[1:]
    assert "Saved detector: NLI · 3 evaluated pairs." in nli
    assert "saved/nli-model" in _literal_quote_contents(nli)
    assert "reasoning" in _literal_quote_contents(nli)
    assert "0.625" in _literal_quote_contents(nli)
    assert "2" in _literal_quote_contents(nli)
    assert "Enabled" in _literal_quote_contents(nli)
    assert "Saved detector: COSINE · 27 evaluated pairs." in cosine
    assert "saved/cosine-model" in _literal_quote_contents(cosine)
    assert "key_claims" in _literal_quote_contents(cosine)
    assert "0.72" in _literal_quote_contents(cosine)
    assert "0.98" in _literal_quote_contents(cosine)
    assert "Disabled" in _literal_quote_contents(cosine)
    assert "Saved detector: unknown" in legacy
    assert "saved/nli-model" not in _literal_quote_contents(legacy)
    assert "saved/cosine-model" not in _literal_quote_contents(legacy)
    assert "pinned weight revision" in exports.markdown
    assert json.loads(exports.json)["report"] == before
    assert report.model_dump(mode="json") == before


@pytest.mark.parametrize("mode", ["nli", "cosine"])
def test_zero_evaluated_pairs_do_not_claim_inference_ran(report, mode):
    report.reasoning_trace = [report.reasoning_trace[0]]
    report.reasoning_trace[0].detector_metadata = DetectorMetadata(
        mode=mode, model_id="saved/unused-model", divergence_threshold=0.5,
        evaluated_pair_count=0,
    )
    exports = build_report_exports(report, source="history")
    assert f"Saved detector: {mode.upper()} · 0 evaluated pairs." in exports.markdown
    assert "No pairs were evaluated; no pairwise model inference was run." in _literal_quote_contents(exports.markdown)
    assert json.loads(exports.json)["report"]["reasoning_trace"][0]["detector_metadata"]["evaluated_pair_count"] == 0


def test_legacy_missing_metadata_is_unknown_and_not_filled_from_current_settings(report, monkeypatch):
    from debate import divergence

    legacy = report.model_dump(mode="json")
    for record in legacy["reasoning_trace"]:
        record.pop("detector_metadata")
    loaded = DebateReport.model_validate(legacy)
    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "nli")
    original = build_report_exports(loaded, source="history")
    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "cosine")
    assert build_report_exports(loaded, source="history") == original
    assert original.markdown.count("Saved detector: unknown") == 2
    assert all(record["detector_metadata"] is None for record in json.loads(original.json)["report"]["reasoning_trace"])


def test_model_identifier_is_exported_as_literal_text(report):
    model_id = "saved/model\n# Forged heading\n<script>unsafe()</script>"
    report.reasoning_trace[0].detector_metadata = DetectorMetadata(
        mode="nli", model_id=model_id, divergence_threshold=0.5,
        nli_contradiction_label_index=0, evaluated_pair_count=1,
    )
    exports = build_report_exports(report, source="history")
    assert model_id in _literal_quote_contents(exports.markdown)
    assert "\n# Forged heading" not in exports.markdown
    assert "<script>" not in exports.markdown
    assert "Saved detector: NLI · 1 evaluated pair." in exports.markdown
    assert json.loads(exports.json)["report"]["reasoning_trace"][0]["detector_metadata"]["model_id"] == model_id


@pytest.mark.parametrize("mode,input_field,fast_path,nli_label", [
    ("nli", "key_claims", "Not used", "Not recorded"),
    ("nli", "reasoning", "Not used", "Not recorded"),
    ("cosine", "key_claims", "Not recorded", "Not applicable"),
    ("cosine", "reasoning", "Not used", "Not applicable"),
])
def test_detector_details_distinguish_unused_from_missing_settings(mode, input_field, fast_path, nli_label):
    from debate.detector_display import detector_details

    metadata = DetectorMetadata(
        mode=mode, model_id="saved/model", input_field=input_field,
        divergence_threshold=0.5,
    )
    details = dict(detector_details(metadata))
    assert details["Cosine fast-path threshold"] == fast_path
    assert details["NLI contradiction label index"] == nli_label


def test_detector_details_preserve_explicit_values_even_for_an_unusual_configuration():
    from debate.detector_display import detector_details

    metadata = DetectorMetadata(
        mode="cosine", model_id="saved/model", input_field="reasoning",
        divergence_threshold=0.5, cosine_fast_path_threshold=0.99,
        nli_contradiction_label_index=0,
    )
    details = dict(detector_details(metadata))
    assert details["Cosine fast-path threshold"] == "0.99"
    assert details["NLI contradiction label index"] == "0"


@pytest.mark.parametrize("synthesis_status", ["complete", "failed", "unknown"])
def test_exports_record_synthesis_status_without_changing_report_values(report, synthesis_status):
    report.synthesis_status = synthesis_status
    original = report.model_dump(mode="json")
    exports = build_report_exports(report, source="history")
    payload = json.loads(exports.json)
    assert payload["schema_version"] == 1
    assert payload["report"] == original
    assert payload["report"]["synthesis_status"] == synthesis_status
    assert "independent of the stop reason" in payload["interpretation"]["synthesis_status"]
    assert f"Recorded `synthesis_status`: **{synthesis_status}**." in exports.markdown
    assert report.model_dump(mode="json") == original
    if synthesis_status == "failed":
        assert "Synthesis failed: the final summary is unavailable" in exports.markdown
        assert "recorded discussion remains available below" in exports.markdown
    elif synthesis_status == "unknown":
        assert "cannot be inferred from the verdict text" in exports.markdown
    else:
        assert "claims have not been fact-checked" in exports.markdown


def test_failed_synthesis_empty_sections_are_unavailable_not_absent(report):
    report.synthesis_status = "failed"
    report.consensus_points = []
    report.disputed_points = []
    exports = build_report_exports(report, source="live", persistence="unconfirmed")
    assert "Consensus summary unavailable because synthesis failed." in exports.markdown
    assert "Dispute summary unavailable because synthesis failed." in exports.markdown
    consensus = exports.markdown.split("### Consensus points", 1)[1].split("## Reasoning trace", 1)[0]
    assert "None saved." not in consensus
    payload = json.loads(exports.json)
    assert payload["source"]["persistence"] == "unconfirmed"
    assert payload["report"]["synthesis_status"] == "failed"
    assert payload["report"]["consensus_points"] == []
    assert payload["report"]["disputed_points"] == []
    assert payload["report"]["reasoning_trace"] == report.model_dump(mode="json")["reasoning_trace"]


def test_legacy_export_does_not_infer_synthesis_status_from_verdict(report):
    legacy = report.model_dump(mode="json")
    legacy.pop("synthesis_status")
    legacy["verdict"] = "[Synthesis unavailable due to repeated parse failures]"
    loaded = DebateReport.model_validate(legacy)
    exports = build_report_exports(loaded, source="history")
    assert json.loads(exports.json)["report"]["synthesis_status"] == "unknown"
    assert "Recorded `synthesis_status`: **unknown**." in exports.markdown
    assert "Synthesis failed: the final summary is unavailable" not in exports.markdown
    assert legacy["verdict"] in _literal_quote_contents(exports.markdown)
