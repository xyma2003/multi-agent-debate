# debate/divergence.py
"""
Semantic divergence detector for the multi-agent debate system.

Two detection strategies, selectable via DIVERGENCE_MODE env var:

  cosine (default):
    Two-layer detection using bi-encoder embeddings:
      Layer 1: max cosine similarity on key_claims.
        > CONVERGE_FAST_PATH (0.97) → definitely converged.
        Otherwise → treated as diverged.
    Known limitation: cosine similarity measures topic overlap, not stance
    opposition. "VC is good" and "VC is bad" score as similar because they
    share the same vocabulary. This causes premature convergence detection.

  nli (research variant):
    Two-layer detection using NLI cross-encoder:
      Layer 1: cosine fast-path (>0.97 → skip NLI, definitely converged).
      Layer 2: CrossEncoder NLI on key_claims pairs.
        High CONTRADICTION probability → genuine stance divergence.
        ENTAILMENT / NEUTRAL → semantic agreement or unrelated.
    Fixes the cosine limitation: "VC is good" vs "VC is bad" correctly
    classified as CONTRADICTION regardless of vocabulary overlap.

Select via: DIVERGENCE_MODE=nli python ...
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from debate.state import AgentArgument

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIVERGE_THRESHOLD: float = 0.75      # cosine: max_sim below this → diverged
CONVERGE_FAST_PATH: float = 0.97     # both modes: cosine fast-path threshold

# NLI: contradiction probability above this → pair is diverged
NLI_CONTRADICTION_THRESHOLD: float = 0.5

DIVERGENCE_MODE: str = os.environ.get("DIVERGENCE_MODE", "cosine").lower()

# NLI label indices for cross-encoder/nli-deberta-v3-small
_NLI_CONTRADICTION = 0
_NLI_ENTAILMENT    = 1
_NLI_NEUTRAL       = 2


# ---------------------------------------------------------------------------
# Model singletons
# ---------------------------------------------------------------------------

_BIENCODER: SentenceTransformer | None = None
_CROSSENCODER: CrossEncoder | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load BAAI/bge-small-en-v1.5 bi-encoder."""
    global _BIENCODER
    if _BIENCODER is None:
        _BIENCODER = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _BIENCODER


def _get_nli_model() -> CrossEncoder:
    """Lazy-load cross-encoder/nli-deberta-v3-small NLI model (~180MB)."""
    global _CROSSENCODER
    if _CROSSENCODER is None:
        _CROSSENCODER = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _CROSSENCODER


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# NLI-based divergence
# ---------------------------------------------------------------------------

def compute_divergence_nli(
    arguments: list[AgentArgument],
) -> tuple[float, list[tuple[str, str]]]:
    """NLI-based divergence: detects genuine stance contradiction between agents.

    For each agent pair, runs cross-encoder NLI on all cross-claim combinations
    and takes the maximum CONTRADICTION probability as the divergence signal.

    This fixes the core limitation of cosine similarity: "VC is good" and
    "VC is bad" share vocabulary and score as similar under cosine, but NLI
    correctly classifies them as CONTRADICTION.

    Returns:
        divergence_score: mean(max_contradiction_prob per agent pair). [0, 1].
            0.0 = no contradictions detected (converged).
            1.0 = all pairs show maximum contradiction.
        diverged_pairs: (role_a, role_b) pairs where max contradiction > threshold.
    """
    if len(arguments) < 2:
        return 0.0, []

    biencoder = _get_model()
    nli_model = _get_nli_model()
    diverged_pairs: list[tuple[str, str]] = []
    pairwise_max_contradictions: list[float] = []

    for arg_a, arg_b in combinations(arguments, 2):
        claims_a = arg_a.key_claims
        claims_b = arg_b.key_claims

        if not claims_a or not claims_b:
            pairwise_max_contradictions.append(0.0)
            continue

        # --- Layer 1: cosine fast-path (skip NLI if clearly converged) ---
        all_claims = claims_a + claims_b
        embeddings = biencoder.encode(all_claims, normalize_embeddings=True)
        emb_a = embeddings[: len(claims_a)]
        emb_b = embeddings[len(claims_a):]
        cosine_max_sim = float((emb_a @ emb_b.T).max())

        if cosine_max_sim > CONVERGE_FAST_PATH:
            # Definitely converged — NLI not needed
            pairwise_max_contradictions.append(0.0)
            continue

        # --- Layer 2: NLI cross-encoder on all cross-claim pairs ---
        cross_pairs = [(c_a, c_b) for c_a in claims_a for c_b in claims_b]
        logits = nli_model.predict(cross_pairs)           # shape: (n_pairs, 3)
        probs = _softmax(np.array(logits))                # convert logits → probs
        contradiction_probs = probs[:, _NLI_CONTRADICTION] # shape: (n_pairs,)
        max_contradiction = float(contradiction_probs.max())

        pairwise_max_contradictions.append(max_contradiction)

        if max_contradiction > NLI_CONTRADICTION_THRESHOLD:
            diverged_pairs.append((arg_a.agent_role, arg_b.agent_role))

    if not pairwise_max_contradictions:
        return 0.0, []

    divergence_score = sum(pairwise_max_contradictions) / len(pairwise_max_contradictions)
    return round(divergence_score, 4), diverged_pairs


# ---------------------------------------------------------------------------
# Cosine-based divergence (original)
# ---------------------------------------------------------------------------

def compute_divergence(
    arguments: list[AgentArgument],
) -> tuple[float, list[tuple[str, str]]]:
    """Compute pairwise semantic divergence across all agent argument pairs.

    Embeds the key_claims of each AgentArgument (joined into one batch for
    efficiency), then computes per-pair max cosine similarity. Diverged pairs
    are those whose max cross-claim similarity falls below DIVERGE_THRESHOLD.

    Args:
        arguments: AgentArguments from a single round. Typically 3 (one per agent).

    Returns:
        divergence_score: float in [0.0, 1.0].
            0.0 = fully converged (identical claims).
            1.0 = completely diverged.
            Formula: 1.0 - mean(pairwise max_similarities).
        diverged_pairs: list of (role_a, role_b) tuples where divergence was detected.
    """
    if len(arguments) < 2:
        return 0.0, []

    model = _get_model()
    diverged_pairs: list[tuple[str, str]] = []
    pairwise_max_sims: list[float] = []

    for arg_a, arg_b in combinations(arguments, 2):
        claims_a = arg_a.key_claims
        claims_b = arg_b.key_claims

        if not claims_a or not claims_b:
            # No claims to compare — treat as converged for this pair
            pairwise_max_sims.append(1.0)
            continue

        # Encode all claims in one batch for efficiency (single model.encode call)
        all_claims = claims_a + claims_b
        # CRITICAL: normalize_embeddings=True makes dot product == cosine similarity
        embeddings = model.encode(all_claims, normalize_embeddings=True)
        emb_a = embeddings[: len(claims_a)]
        emb_b = embeddings[len(claims_a) :]

        # Cross-claim cosine similarity matrix: shape (len_a, len_b)
        # max_sim = the most similar claim pair between the two agents
        sim_matrix = emb_a @ emb_b.T
        max_sim = float(sim_matrix.max())
        pairwise_max_sims.append(max_sim)

        # Fast path: clearly converged — do not mark as diverged pair
        if max_sim > CONVERGE_FAST_PATH:
            continue

        # Below DIVERGE_THRESHOLD (and not fast-path): record as diverged
        # 0.75–0.97 zone is also treated as diverged until Claude judge is added.
        if max_sim < DIVERGE_THRESHOLD:
            diverged_pairs.append((arg_a.agent_role, arg_b.agent_role))
        else:
            # Borderline zone (0.75–0.97): treated conservatively as diverged
            diverged_pairs.append((arg_a.agent_role, arg_b.agent_role))

    if not pairwise_max_sims:
        return 0.0, []

    divergence_score = 1.0 - (sum(pairwise_max_sims) / len(pairwise_max_sims))
    return divergence_score, diverged_pairs


# ---------------------------------------------------------------------------
# Dispatcher — routes to cosine or NLI based on DIVERGENCE_MODE
# ---------------------------------------------------------------------------

def compute_divergence_dispatch(
    arguments: list[AgentArgument],
) -> tuple[float, list[tuple[str, str]]]:
    """Dispatch to NLI or cosine divergence based on DIVERGENCE_MODE env var."""
    if DIVERGENCE_MODE == "nli":
        return compute_divergence_nli(arguments)
    return compute_divergence(arguments)
