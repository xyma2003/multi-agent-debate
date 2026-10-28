"""Benchmark detector provenance without loading models or using the network."""
import socket

import pytest

from benchmark import variants
from debate import divergence
from debate.state import AgentArgument, RoundRecord


class FixedEmbeddings:
    """Only the matrix operations used by the detector; no NumPy dependency."""

    def __init__(self, similarity):
        self.similarity = similarity

    def __getitem__(self, key):
        return self

    @property
    def T(self):
        return self

    def __matmul__(self, other):
        return self

    def max(self):
        return self.similarity


class FixedCosineModel:
    def __init__(self, similarity):
        self.similarity = similarity
        self.calls = []

    def encode(self, texts, *, normalize_embeddings):
        self.calls.append((list(texts), normalize_embeddings))
        return FixedEmbeddings(self.similarity)


class FixedNLIModel:
    def __init__(self):
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [[3.0, 1.0, 0.0] for _ in pairs]


def _argument(role, *, reasoning=None, sentinel=False):
    return AgentArgument(
        agent_role=role,
        round_num=1,
        position=f"{role} position",
        reasoning=f"{role} reasoning" if reasoning is None else reasoning,
        confidence=0.5,
        key_claims=[f"{role} claim {index}" for index in range(3)],
        is_sentinel=sentinel,
    )


def _state(arguments, *, saved_score=0.432187):
    return {"round_history": [
        RoundRecord(round_num=0, arguments=[], divergence_score=0.876543),
        RoundRecord(round_num=1, arguments=arguments, divergence_score=saved_score),
    ]}


@pytest.fixture(autouse=True)
def no_models_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmark metadata tests must remain offline")

    monkeypatch.setattr(divergence, "_get_model", forbidden)
    monkeypatch.setattr(divergence, "_get_nli_model", forbidden)
    monkeypatch.setattr(variants, "_get_model", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.mark.parametrize("mode", ["cosine", "nli"])
def test_pinned_benchmark_detector_ignores_current_default(mode, monkeypatch):
    monkeypatch.setattr(divergence, "DIVERGENCE_MODE", "nli" if mode == "cosine" else "cosine")
    model = FixedCosineModel(0.4) if mode == "cosine" else FixedNLIModel()
    loader = "_get_model" if mode == "cosine" else "_get_nli_model"
    monkeypatch.setattr(divergence, loader, lambda: model)
    state = _state([_argument("optimist"), _argument("pessimist")])
    before = [record.model_dump() for record in state["round_history"]]

    node = getattr(variants, f"divergence_check_node_{mode}")
    result = node(state)

    assert len(model.calls) == 1
    assert result["diverged_pairs"] == [("optimist", "pessimist")]
    updated = result["round_history"][-1]
    metadata = updated.detector_metadata
    assert metadata.mode == mode
    assert metadata.input_field == "key_claims"
    assert metadata.model_id == (
        "BAAI/bge-small-en-v1.5" if mode == "cosine" else "cross-encoder/nli-deberta-v3-small"
    )
    assert metadata.evaluated_pair_count == 1
    assert metadata.sentinels_excluded is False
    assert updated.divergence_score == result["divergence_score"]
    assert [record.model_dump() for record in state["round_history"]] == before
    assert result["round_history"] is not state["round_history"]
    assert result["round_history"][0] is state["round_history"][0]
    assert updated is not state["round_history"][-1]


def test_fulltext_records_reasoning_input_and_keeps_raw_round_score(monkeypatch):
    model = FixedCosineModel(0.1234567)
    monkeypatch.setattr(variants, "_get_model", lambda: model)
    state = _state([
        _argument("optimist"),
        _argument("pessimist"),
        _argument("empty", reasoning=""),
        _argument("devil", sentinel=True),
    ])
    before = [record.model_dump() for record in state["round_history"]]

    result = variants.divergence_check_node_fulltext(state)

    assert model.calls == [(["optimist reasoning", "pessimist reasoning"], True)]
    expected_raw = 1.0 - (0.1234567 + 1.0 + 1.0) / 3
    updated = result["round_history"][-1]
    assert updated.divergence_score == expected_raw
    assert result["divergence_score"] == round(expected_raw, 4)
    assert updated.divergence_score != result["divergence_score"]
    assert result["diverged_pairs"] == [("optimist", "pessimist")]
    assert updated.detector_metadata.model_dump() == {
        "schema_version": 1,
        "mode": "cosine",
        "model_id": "BAAI/bge-small-en-v1.5",
        "input_field": "reasoning",
        "divergence_threshold": 0.75,
        "cosine_fast_path_threshold": None,
        "nli_contradiction_label_index": None,
        "evaluated_pair_count": 1,
        "sentinels_excluded": True,
    }
    assert [record.model_dump() for record in state["round_history"]] == before
    assert result["round_history"][0] is state["round_history"][0]
    assert updated is not state["round_history"][-1]


@pytest.mark.parametrize("arguments", [
    [],
    [_argument("optimist")],
    [_argument("optimist"), _argument("devil", sentinel=True)],
])
def test_fulltext_insufficient_input_records_no_evaluation_without_changing_saved_score(arguments):
    state = _state(arguments)
    before = [record.model_dump() for record in state["round_history"]]

    result = variants.divergence_check_node_fulltext(state)

    assert result["divergence_score"] == 0.0
    assert result["diverged_pairs"] == []
    updated = result["round_history"][-1]
    assert updated.divergence_score == 0.432187
    assert updated.detector_metadata.mode == "cosine"
    assert updated.detector_metadata.input_field == "reasoning"
    assert updated.detector_metadata.sentinels_excluded is True
    assert updated.detector_metadata.evaluated_pair_count == 0
    assert [record.model_dump() for record in state["round_history"]] == before
    assert result["round_history"][0] is state["round_history"][0]
    assert updated is not state["round_history"][-1]


def test_fulltext_empty_reasoning_does_not_count_unevaluated_pairs(monkeypatch):
    model = FixedCosineModel(0.4)
    loader_calls = []

    def load_model():
        loader_calls.append(True)
        return model

    monkeypatch.setattr(variants, "_get_model", load_model)
    result = variants.divergence_check_node_fulltext(_state([
        _argument("optimist", reasoning=""), _argument("pessimist", reasoning=""),
    ]))

    # Preserve the existing eager loader timing for >=2 non-sentinel arguments.
    assert loader_calls == [True]
    assert model.calls == []
    assert result["divergence_score"] == 0.0
    assert result["round_history"][-1].divergence_score == 0.0
    assert result["round_history"][-1].detector_metadata.evaluated_pair_count == 0


def test_fulltext_uses_captured_threshold_and_has_no_cosine_fast_path(monkeypatch):
    monkeypatch.setattr(variants, "DIVERGE_THRESHOLD", 1.0)

    class SettingsChangingModel(FixedCosineModel):
        def encode(self, texts, *, normalize_embeddings):
            monkeypatch.setattr(variants, "DIVERGE_THRESHOLD", 0.2)
            monkeypatch.setattr(variants, "COSINE_MODEL_ID", "changed-after-start")
            return super().encode(texts, normalize_embeddings=normalize_embeddings)

    model = SettingsChangingModel(0.99)
    monkeypatch.setattr(variants, "_get_model", lambda: model)
    result = variants.divergence_check_node_fulltext(_state([
        _argument("optimist"), _argument("pessimist"),
    ]))

    # Fulltext has no 0.97 shortcut. Its original threshold comparison still applies.
    assert result["diverged_pairs"] == [("optimist", "pessimist")]
    metadata = result["round_history"][-1].detector_metadata
    assert metadata.divergence_threshold == 1.0
    assert metadata.model_id == "BAAI/bge-small-en-v1.5"
    assert metadata.cosine_fast_path_threshold is None


@pytest.mark.parametrize("mode", ["cosine", "nli", "fulltext"])
def test_benchmark_detectors_still_require_a_completed_round(mode):
    node = getattr(variants, f"divergence_check_node_{mode}")
    with pytest.raises(RuntimeError, match="requires a completed round"):
        node({"round_history": []})
