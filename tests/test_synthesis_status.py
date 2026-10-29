"""Run the real synthesis node against fixed offline SDK responses."""

from copy import deepcopy
import json
import sqlite3
import time

import pytest

from debate.nodes import synthesize
from debate.state import AgentArgument, Concession, DebateReport, RoundRecord
from debate.store import _init_schema, load_debate, save_debate


def _state():
    concession = Concession(
        conceded_point="The pilot needs a clear rollback condition.",
        triggered_by_agent="pessimist", triggered_by_claim="A pilot can still fail.",
        rationale="A small test still needs an exit condition.",
    )
    rounds = [
        RoundRecord(
            round_num=number, divergence_score=score,
            arguments=[
                AgentArgument(
                    agent_role=role, round_num=number, position=f"{role} position",
                    reasoning=f"{role} reasoning", confidence=0.7,
                    key_claims=[f"{role} claim {index}" for index in range(3)],
                    concessions=[concession] if number == 0 and role == "optimist" else [],
                )
                for role in ("optimist", "pessimist", "devil")
            ],
        )
        for number, score in enumerate((0.8, 0.2))
    ]
    return {
        "debate_id": "offline-synthesis-status",
        "topic": "Synthetic pilot discussion",
        "round_num": 2,
        "max_rounds": 3,
        "round_history": rounds,
        "diverged_pairs": [("optimist", "pessimist")],
    }


def _output(verdict="Synthetic final assessment."):
    return synthesize.SynthesizerOutput(
        consensus_points=["A pilot needs a rollback condition."],
        disputed_points=[], verdict=verdict,
    )


class _FixedSDK:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.schema = None
        self.include_raw = None

    def with_structured_output(self, schema, *, include_raw):
        self.schema = schema
        self.include_raw = include_raw
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("Unexpected extra synthesis attempt")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _run(monkeypatch, responses, *, state=None):
    sdk = _FixedSDK(responses)
    monkeypatch.setattr(synthesize, "_make_llm", lambda: sdk)
    state = _state() if state is None else state
    before = deepcopy(state)
    result = synthesize.synthesize_stub(state)
    assert state == before
    report = result["final_report"]
    assert report.confidence_score == 0.18
    assert report.reasoning_trace == before["round_history"]
    assert report.concession_log == [before["round_history"][0].arguments[0].concessions[0]]
    assert report.convergence_status == result["status"]
    assert report.convergence_status == ("stalled" if state["diverged_pairs"] else "converged")
    return report, sdk


def test_success_is_assigned_by_runtime_without_changing_model_schema(monkeypatch):
    report, sdk = _run(monkeypatch, [{"parsed": _output()}])
    assert report.synthesis_status == "complete"
    assert report.verdict == "Synthetic final assessment."
    assert len(sdk.calls) == 1
    assert sdk.schema is synthesize.SynthesizerOutput
    assert sdk.include_raw is True
    assert set(sdk.schema.model_json_schema()["properties"]) == {
        "consensus_points", "disputed_points", "verdict",
    }
    prompt = "\n".join(message.content for message in sdk.calls[0])
    assert "synthesis_status" not in prompt
    assert "confidence_score" not in prompt


@pytest.mark.parametrize("diverged_pairs", [[("optimist", "pessimist")], []])
def test_three_parse_failures_produce_failed_report_with_unchanged_trace_and_scores(
    monkeypatch, diverged_pairs,
):
    state = _state()
    state["diverged_pairs"] = diverged_pairs
    report, sdk = _run(monkeypatch, [
        {"parsed": None, "parsing_error": "fixed invalid structured output"},
    ] * 3, state=state)
    assert report.synthesis_status == "failed"
    assert len(sdk.calls) == 3
    assert not sdk.responses
    assert report.consensus_points == []
    assert report.disputed_points == []
    assert report.verdict == "[Synthesis unavailable due to repeated parse failures]"


@pytest.mark.parametrize("failed_attempts", [1, 2])
def test_success_after_parse_failures_is_complete(monkeypatch, failed_attempts):
    report, sdk = _run(monkeypatch, [
        {"parsed": None, "parsing_error": "fixed parse failure"},
    ] * failed_attempts + [{"parsed": _output()}])
    assert report.synthesis_status == "complete"
    assert len(sdk.calls) == failed_attempts + 1
    assert report.consensus_points == _output().consensus_points


def test_rate_limit_exhaustion_keeps_three_attempts_and_existing_backoff(monkeypatch):
    waits = []
    monkeypatch.setattr(time, "sleep", waits.append)
    report, sdk = _run(monkeypatch, [RuntimeError("429 rate limit")] * 3)
    assert report.synthesis_status == "failed"
    assert len(sdk.calls) == 3
    assert waits == [60, 120, 180]


def test_non_retryable_exception_still_propagates_without_another_call(monkeypatch):
    failure = RuntimeError("synthetic authentication failure")
    sdk = _FixedSDK([failure])
    monkeypatch.setattr(synthesize, "_make_llm", lambda: sdk)
    with pytest.raises(RuntimeError) as raised:
        synthesize.synthesize_stub(_state())
    assert raised.value is failure
    assert len(sdk.calls) == 1


def test_status_is_not_inferred_from_verdict_text(monkeypatch):
    report, _ = _run(monkeypatch, [{
        "parsed": _output("[Synthesis unavailable due to repeated parse failures]"),
    }])
    # This checks successful structured parsing, not the quality of its content.
    assert report.synthesis_status == "complete"


@pytest.mark.parametrize("status", ["complete", "failed", "unknown"])
def test_sqlite_and_json_roundtrip_keep_synthesis_status(monkeypatch, status):
    report, _ = _run(monkeypatch, [{"parsed": _output()}])
    report = report.model_copy(update={"synthesis_status": status})
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        _init_schema(conn)
        save_debate(report, conn=conn)
        row = conn.execute("SELECT status, report_json FROM debates").fetchone()
        assert row["status"] == report.convergence_status
        assert json.loads(row["report_json"])["synthesis_status"] == status
        loaded = load_debate(report.debate_id, conn=conn)
        assert loaded.model_dump() == report.model_dump()
        assert DebateReport.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize("verdict", [
    "An older saved conclusion.", "[Synthesis unavailable due to repeated parse failures]",
])
def test_legacy_report_is_unknown_and_reading_does_not_rewrite_saved_json(monkeypatch, verdict):
    report, _ = _run(monkeypatch, [{"parsed": _output(verdict)}])
    original = report.model_dump(mode="json")
    original.pop("synthesis_status")
    legacy_json = json.dumps(original)
    assert DebateReport.model_validate_json(legacy_json).synthesis_status == "unknown"
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        _init_schema(conn)
        conn.execute(
            "INSERT INTO debates (debate_id, topic, created_at, status, report_json) VALUES (?, ?, ?, ?, ?)",
            (report.debate_id, report.topic, report.created_at.isoformat(), report.convergence_status, legacy_json),
        )
        conn.commit()
        loaded = load_debate(report.debate_id, conn=conn)
        assert loaded.synthesis_status == "unknown"
        assert loaded.model_dump(mode="json", exclude={"synthesis_status"}) == original
        assert conn.execute("SELECT report_json FROM debates").fetchone()[0] == legacy_json
