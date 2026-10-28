"""Offline checks that detector provenance describes the computation that ran."""
from __future__ import annotations

import json
import math
import socket
import sqlite3
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from debate import divergence, store
from debate.nodes import divergence_check, synthesize
from debate.state import AgentArgument, DebateReport, DetectorMetadata, RoundRecord


def _argument(role: str, *, empty: bool = False, sentinel: bool = False):
    argument = AgentArgument(
        agent_role=role,
        round_num=0,
        position=f"{role} position",
        reasoning=f"{role} reasoning",
        confidence=0.7,
        key_claims=[f"{role} claim {index}" for index in range(3)],
        is_sentinel=sentinel,
    )
    # The detector already supports legacy/sentinel objects with no claims,
    # although the current LLM-facing schema requires at least three.
    return argument.model_copy(update={"key_claims": []}) if empty else argument


@pytest.fixture(autouse=True)
def offline_detector(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Detector metadata tests must not load models or use the network")

    monkeypatch.setattr(divergence, "_get_model", forbidden)
    monkeypatch.setattr(divergence, "_get_nli_model", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "nli")
    monkeypatch.setattr(divergence, "NLI_CONTRADICTION_THRESHOLD", 0.5)
    monkeypatch.setattr(divergence, "_NLI_CONTRADICTION", 0)
    monkeypatch.setattr(divergence, "DIVERGE_THRESHOLD", 0.75)
    monkeypatch.setattr(divergence, "CONVERGE_FAST_PATH", 0.97)


class _NLIModel:
    def __init__(self, probabilities, *, during_predict=None):
        self.probabilities = iter(probabilities)
        self.during_predict = during_predict
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        if self.during_predict:
            self.during_predict()
        contradiction = next(self.probabilities)
        row = [math.log(contradiction), math.log((1 - contradiction) / 2),
               math.log((1 - contradiction) / 2)]
        return [row[:] for _ in pairs]


class _EmbeddingMatrix:
    """Only the array operations used by the cosine detector, without NumPy."""
    def __init__(self, rows, similarity):
        self.rows = rows
        self.similarity = similarity

    def __getitem__(self, selection):
        return _EmbeddingMatrix(self.rows[selection], self.similarity)

    @property
    def T(self):
        return self

    def __matmul__(self, other):
        assert self.rows and other.rows
        return self

    def max(self):
        return self.similarity


class _CosineModel:
    def __init__(self, similarities, *, during_encode=None):
        self.similarities = iter(similarities)
        self.during_encode = during_encode
        self.calls = []

    def encode(self, claims, *, normalize_embeddings):
        assert normalize_embeddings is True
        self.calls.append(claims)
        if self.during_encode:
            self.during_encode()
        return _EmbeddingMatrix(claims, next(self.similarities))


def test_nli_result_records_actual_model_and_preserves_legacy_scores(monkeypatch):
    arguments = [_argument(role) for role in ("optimist", "pessimist", "devil")]
    models = []

    def load_model():
        model = _NLIModel([0.8, 0.2, 0.5])
        models.append(model)
        return model

    monkeypatch.setattr(divergence, "_get_nli_model", load_model)
    result = divergence.detect_divergence(arguments, mode="nli")
    expected = (0.5, [("optimist", "pessimist")])
    assert (result.score, result.diverged_pairs) == expected
    assert divergence.compute_divergence_nli(arguments) == expected
    assert divergence.compute_divergence_dispatch(arguments) == expected
    metadata = result.metadata
    assert metadata.mode == "nli"
    assert metadata.model_id == divergence.NLI_MODEL_ID
    assert metadata.model_id == "cross-encoder/nli-deberta-v3-small"
    assert metadata.divergence_threshold == 0.5
    assert metadata.nli_contradiction_label_index == 0
    assert metadata.cosine_fast_path_threshold is None
    assert metadata.evaluated_pair_count == 3
    assert metadata.input_field == "key_claims"
    assert metadata.sentinels_excluded is False
    assert all(len(model.calls) == 3 for model in models)
    assert all(len(pairs) == 9 for model in models for pairs in model.calls)


def test_cosine_result_records_actual_model_and_preserves_legacy_scores(monkeypatch):
    arguments = [_argument(role) for role in ("optimist", "pessimist", "devil")]
    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "cosine")
    models = []

    def load_model():
        model = _CosineModel([0.5, 0.8, 0.99])
        models.append(model)
        return model

    monkeypatch.setattr(divergence, "_get_model", load_model)
    result = divergence.detect_divergence(arguments, mode="cosine")
    expected = (1 - (0.5 + 0.8 + 0.99) / 3, [("optimist", "pessimist")])
    assert (result.score, result.diverged_pairs) == expected
    assert divergence.compute_divergence(arguments) == expected
    assert divergence.compute_divergence_dispatch(arguments) == expected
    metadata = result.metadata
    assert metadata.mode == "cosine"
    assert metadata.model_id == divergence.COSINE_MODEL_ID
    assert metadata.model_id == "BAAI/bge-small-en-v1.5"
    assert metadata.divergence_threshold == 0.75
    assert metadata.cosine_fast_path_threshold == 0.97
    assert metadata.nli_contradiction_label_index is None
    assert metadata.evaluated_pair_count == 3
    assert all(len(model.calls) == 3 for model in models)
    assert models[0].calls[0] == arguments[0].key_claims + arguments[1].key_claims


def test_nli_uses_threshold_and_label_captured_before_predict(monkeypatch):
    def alter_globals():
        monkeypatch.setattr(divergence, "NLI_CONTRADICTION_THRESHOLD", 0.99)
        monkeypatch.setattr(divergence, "_NLI_CONTRADICTION", 1)
        monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "cosine")

    model = _NLIModel([0.8, 0.8, 0.8], during_predict=alter_globals)
    monkeypatch.setattr(divergence, "_get_nli_model", lambda: model)
    arguments = [_argument(role) for role in ("optimist", "pessimist", "devil")]
    result = divergence.detect_divergence(arguments)
    assert result.score == 0.8
    assert result.diverged_pairs == [
        ("optimist", "pessimist"), ("optimist", "devil"), ("pessimist", "devil"),
    ]
    assert result.metadata.mode == "nli"
    assert result.metadata.divergence_threshold == 0.5
    assert result.metadata.nli_contradiction_label_index == 0


def test_cosine_uses_both_thresholds_captured_before_encode(monkeypatch):
    def alter_globals():
        monkeypatch.setattr(divergence, "DIVERGE_THRESHOLD", 0.1)
        monkeypatch.setattr(divergence, "CONVERGE_FAST_PATH", 0.1)
        monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "nli")

    model = _CosineModel([0.5, 0.5, 0.5], during_encode=alter_globals)
    monkeypatch.setattr(divergence, "_get_model", lambda: model)
    arguments = [_argument(role) for role in ("optimist", "pessimist", "devil")]
    result = divergence.detect_divergence(arguments, mode="cosine")
    assert result.score == 0.5
    assert len(result.diverged_pairs) == 3
    assert result.metadata.mode == "cosine"
    assert result.metadata.divergence_threshold == 0.75
    assert result.metadata.cosine_fast_path_threshold == 0.97


@pytest.mark.parametrize("mode", ["nli", "cosine"])
@pytest.mark.parametrize("shape", ["empty", "single"])
def test_no_comparisons_do_not_load_a_model_or_claim_evaluated_pairs(mode, shape):
    arguments = {
        "empty": [],
        "single": [_argument("optimist")],
    }[shape]
    result = divergence.detect_divergence(arguments, mode=mode)
    assert result.score == 0.0
    assert result.diverged_pairs == []
    assert result.metadata.mode == mode
    assert result.metadata.evaluated_pair_count == 0


@pytest.mark.parametrize("mode", ["nli", "cosine"])
def test_missing_claims_load_model_but_do_not_claim_pairwise_inference(monkeypatch, mode):
    arguments = [_argument("optimist", empty=True), _argument("pessimist")]
    model = _NLIModel([]) if mode == "nli" else _CosineModel([])
    loads = []

    def load_model():
        loads.append(mode)
        return model

    getter = "_get_nli_model" if mode == "nli" else "_get_model"
    monkeypatch.setattr(divergence, getter, load_model)
    result = divergence.detect_divergence(arguments, mode=mode)
    assert loads == [mode]  # Preserve the existing >=2-arguments loader behavior.
    assert model.calls == []
    assert result.score == 0.0
    assert result.diverged_pairs == []
    assert result.metadata.evaluated_pair_count == 0


@pytest.mark.parametrize("mode", ["nli", "cosine"])
def test_evaluated_pair_count_excludes_missing_claims_but_keeps_existing_score_math(
    monkeypatch, mode,
):
    arguments = [_argument("optimist"), _argument("pessimist"), _argument("devil", empty=True)]
    if mode == "nli":
        model = _NLIModel([0.8])
        monkeypatch.setattr(divergence, "_get_nli_model", lambda: model)
        expected_score = round(0.8 / 3, 4)
    else:
        model = _CosineModel([0.5])
        monkeypatch.setattr(divergence, "_get_model", lambda: model)
        expected_score = 1 - (0.5 + 1 + 1) / 3
    result = divergence.detect_divergence(arguments, mode=mode)
    assert result.score == expected_score
    assert result.diverged_pairs == [("optimist", "pessimist")]
    assert result.metadata.evaluated_pair_count == len(model.calls) == 1


@pytest.mark.parametrize("mode", ["nli", "cosine"])
def test_metadata_does_not_claim_sentinels_were_filtered(monkeypatch, mode):
    arguments = [_argument("optimist", sentinel=True), _argument("pessimist")]
    model = _NLIModel([0.8]) if mode == "nli" else _CosineModel([0.5])
    getter = "_get_nli_model" if mode == "nli" else "_get_model"
    monkeypatch.setattr(divergence, getter, lambda: model)
    result = divergence.detect_divergence(arguments, mode=mode)
    assert result.metadata.sentinels_excluded is False
    assert result.metadata.evaluated_pair_count == 1
    assert result.diverged_pairs == [("optimist", "pessimist")]
    assert "optimist claim 0" in str(model.calls[0])


def test_result_and_metadata_cannot_be_reassigned_after_evaluation():
    result = divergence.detect_divergence([], mode="nli")
    with pytest.raises(FrozenInstanceError):
        result.score = 1.0
    with pytest.raises(ValidationError, match="frozen"):
        result.metadata.mode = "cosine"


def test_node_backfills_only_a_copy_of_latest_round(monkeypatch):
    earlier = RoundRecord(
        round_num=0,
        arguments=[_argument("optimist")],
        divergence_score=0.25,
        detector_metadata=DetectorMetadata(
            mode="cosine", model_id="BAAI/bge-small-en-v1.5",
            divergence_threshold=0.75, cosine_fast_path_threshold=0.97,
            evaluated_pair_count=0,
        ),
    )
    latest = RoundRecord(
        round_num=1, arguments=[_argument("optimist"), _argument("pessimist")],
    )
    history = [earlier, latest]
    before = [record.model_dump() for record in history]
    monkeypatch.setattr(divergence, "_get_nli_model", lambda: _NLIModel([0.8]))
    update = divergence_check.divergence_check_node({"round_history": history})
    assert [record.model_dump() for record in history] == before
    assert update["round_history"] is not history
    assert update["round_history"][0].model_dump() == before[0]
    copied = update["round_history"][1]
    assert copied is not latest
    assert copied.divergence_score == update["divergence_score"] == 0.8
    assert copied.detector_metadata.mode == "nli"
    assert copied.detector_metadata.evaluated_pair_count == 1
    assert latest.detector_metadata is None


@pytest.fixture
def database(tmp_path):
    connection = sqlite3.connect(tmp_path / "detector-metadata.db")
    connection.row_factory = sqlite3.Row
    store._init_schema(connection)
    try:
        yield connection
    finally:
        connection.close()


def test_synthesis_and_sqlite_keep_metadata_without_capturing_environment(
    monkeypatch, database,
):
    secrets = ["synthetic-secret-api-key", "synthetic-private-gateway", "synthetic-custom-setting"]
    monkeypatch.setenv("OPENAI_API_KEY", secrets[0])
    monkeypatch.setenv("OPENAI_BASE_URL", secrets[1])
    monkeypatch.setenv("UNRELATED_PRIVATE_SETTING", secrets[2])
    monkeypatch.setattr(divergence, "_get_nli_model", lambda: _NLIModel([0.8]))
    original_round = RoundRecord(
        round_num=0, arguments=[_argument("optimist"), _argument("pessimist")],
    )
    state = {
        "debate_id": "detector-metadata-offline-test",
        "topic": "Synthetic detector provenance",
        "round_num": 1,
        "max_rounds": 1,
        "round_history": [original_round],
    }
    state.update(divergence_check.divergence_check_node(state))
    monkeypatch.setattr(synthesize, "_invoke_synthesizer", lambda context: synthesize.SynthesizerOutput(
        consensus_points=[], disputed_points=[], verdict="Synthetic offline synthesis.",
    ))
    report = synthesize.synthesize_stub(state)["final_report"]
    store.save_debate(report, conn=database)
    raw = database.execute("SELECT report_json FROM debates").fetchone()[0]
    metadata = json.loads(raw)["reasoning_trace"][0]["detector_metadata"]
    assert set(metadata) == {
        "schema_version", "mode", "model_id", "input_field", "divergence_threshold",
        "cosine_fast_path_threshold", "nli_contradiction_label_index",
        "evaluated_pair_count", "sentinels_excluded",
    }
    assert not any(secret in raw for secret in secrets)
    assert metadata["mode"] == "nli"
    assert metadata["evaluated_pair_count"] == 1
    # Replay must use the saved settings even after the current process changes.
    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "cosine")
    monkeypatch.setattr(divergence, "NLI_CONTRADICTION_THRESHOLD", 0.9)
    database.execute("PRAGMA query_only = ON")
    loaded = store.load_debate(report.debate_id, conn=database)
    assert loaded.model_dump() == report.model_dump()
    assert loaded.reasoning_trace[0].detector_metadata.divergence_threshold == 0.5
    assert database.execute("SELECT report_json FROM debates").fetchone()[0] == raw


def test_legacy_json_remains_unknown_and_is_not_rewritten_at_read_time(monkeypatch, database):
    report = DebateReport(
        debate_id="legacy-without-detector-metadata", topic="Legacy report",
        consensus_points=[], disputed_points=[], verdict="Historical conclusion",
        confidence_score=0.37, convergence_status="converged",
        reasoning_trace=[RoundRecord(round_num=0, arguments=[_argument("optimist")], divergence_score=0.63)],
        concession_log=[], created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
    )
    data = report.model_dump(mode="json")
    del data["reasoning_trace"][0]["detector_metadata"]
    legacy_json = json.dumps(data, ensure_ascii=False, indent=2)
    database.execute(
        "INSERT INTO debates VALUES (?, ?, ?, ?, ?)",
        (report.debate_id, report.topic, report.created_at.isoformat(), report.convergence_status, legacy_json),
    )
    database.commit()
    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "cosine")
    monkeypatch.setattr(divergence, "DIVERGE_THRESHOLD", 0.12)
    database.execute("PRAGMA query_only = ON")
    before = database.total_changes
    loaded = store.load_debate(report.debate_id, conn=database)
    assert loaded.reasoning_trace[0].detector_metadata is None
    assert loaded.reasoning_trace[0].divergence_score == 0.63
    assert loaded.confidence_score == 0.37
    assert database.total_changes == before
    assert database.execute("SELECT report_json FROM debates").fetchone()[0] == legacy_json
