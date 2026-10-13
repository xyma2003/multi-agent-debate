"""Bundled sample checks; no credentials, database, model, or network needed."""
import json

import pytest

from debate import sample
from debate.state import DebateReport


def test_bundled_example_has_complete_consistent_report_and_honest_provenance():
    example = sample.load_offline_example()
    raw = json.loads(sample.EXAMPLE_PATH.read_text(encoding="utf-8"))
    report = example.report
    assert isinstance(report, DebateReport)
    assert report.model_dump() == DebateReport.model_validate(raw["report"]).model_dump()
    assert example.title
    assert "fixed synthetic example" in example.provenance.lower()
    assert "not evidence of model performance" in example.provenance
    assert report.topic and report.verdict and report.consensus_points and report.disputed_points
    assert report.convergence_status == "max_rounds"
    assert report.created_at.utcoffset().total_seconds() == 0
    assert [record.round_num for record in report.reasoning_trace] == [0, 1]
    assert [record.divergence_score for record in report.reasoning_trace] == [0.8, 0.2]
    assert report.confidence_score == pytest.approx((1 - 0.8) * 0.9)
    assert report.confidence_score == pytest.approx(0.18)

    earlier_claims = {}
    flattened_concessions = []
    for record in report.reasoning_trace:
        assert {argument.agent_role for argument in record.arguments} == {
            "optimist", "pessimist", "devil",
        }
        assert len(record.arguments) == 3
        for argument in record.arguments:
            assert argument.round_num == record.round_num
            assert argument.position and argument.reasoning
            assert len(argument.key_claims) >= 3
            assert not argument.is_sentinel
            for concession in argument.concessions:
                assert concession.rationale
                assert concession.triggered_by_agent != argument.agent_role
                assert concession.triggered_by_claim in earlier_claims[concession.triggered_by_agent]
                flattened_concessions.append(concession.model_dump())
        # A concession must refer to an earlier round, not a later claim.
        for argument in record.arguments:
            earlier_claims.setdefault(argument.agent_role, []).extend(argument.key_claims)
    assert len(flattened_concessions) == 2
    assert flattened_concessions == [item.model_dump() for item in report.concession_log]


def test_load_is_independent_of_working_directory_and_does_not_write(monkeypatch, tmp_path):
    original_bytes = sample.EXAMPLE_PATH.read_bytes()
    expected = sample.load_offline_example().model_dump()
    monkeypatch.chdir(tmp_path)

    assert sample.load_offline_example().model_dump() == expected
    assert list(tmp_path.iterdir()) == []
    assert sample.EXAMPLE_PATH.read_bytes() == original_bytes


def test_repeated_loads_return_independent_nested_objects():
    original = sample.load_offline_example()
    expected = original.model_dump()
    original.report.consensus_points.clear()
    original.report.reasoning_trace[0].arguments[0].key_claims.append("caller-only mutation")
    original.report.concession_log[0].rationale = "caller-only mutation"

    reloaded = sample.load_offline_example()
    assert reloaded is not original
    assert reloaded.report is not original.report
    assert reloaded.model_dump() == expected


def test_missing_example_propagates_file_error(monkeypatch, tmp_path):
    monkeypatch.setattr(sample, "EXAMPLE_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        sample.load_offline_example()


@pytest.mark.parametrize("payload", [
    b"{broken JSON",
    b"\xff\xfe",
    b'{"title": "Incomplete", "provenance": "Synthetic", "report": {}}',
])
def test_invalid_example_propagates_value_error(monkeypatch, tmp_path, payload):
    path = tmp_path / "invalid.json"
    path.write_bytes(payload)
    monkeypatch.setattr(sample, "EXAMPLE_PATH", path)

    with pytest.raises(ValueError):
        sample.load_offline_example()
    assert path.read_bytes() == payload
