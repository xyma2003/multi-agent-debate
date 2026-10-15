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
from debate.state import AgentArgument, Concession, DebateReport, DisputedPoint, RoundRecord


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
