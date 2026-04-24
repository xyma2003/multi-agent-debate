# debate/divergence.py
"""
Semantic divergence detector for the multi-agent debate system.

Two-layer detection:
  Layer 1 (fast): pairwise cosine similarity on key_claims embeddings.
    - max_sim > CONVERGE_FAST_PATH (0.97): definitely converged, skip further checks.
    - max_sim < DIVERGE_THRESHOLD (0.75): diverged pair recorded.
    - 0.75–0.97 zone: treated as diverged for Phase 2. Claude judge can be added
      in Phase 3 if false positive rate is high on real debate topics.

Key implementation constraints (from RESEARCH.md Pitfalls):
  - normalize_embeddings=True is REQUIRED. Without it, dot product != cosine similarity
    and scores outside [0, 1] are possible.
  - Embed key_claims (NOT reasoning). Reasoning text collapses semantic distance
    because all agents discuss the same topic — claims are more discriminative.
  - _get_model() is a lazy singleton. First call downloads ~130MB to HF cache.
    Subsequent calls in the same process are instant.
"""
from __future__ import annotations

from itertools import combinations

from sentence_transformers import SentenceTransformer

from debate.state import AgentArgument

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIVERGE_THRESHOLD: float = 0.75   # max_sim below this → agents are diverged on key_claims
CONVERGE_FAST_PATH: float = 0.97  # max_sim above this → definitely converged, skip judge


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load BAAI/bge-small-en-v1.5. Downloads ~130MB on first call, then cached."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _MODEL


# ---------------------------------------------------------------------------
# Public API
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
