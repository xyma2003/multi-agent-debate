# benchmark/evaluator.py
"""
Evaluation metrics for comparing multi-agent debate vs. single-LLM baseline.

Four metrics:
  PDS  — Position Diversity Score
         Average pairwise semantic distance between agents' final key_claims.
         Higher = more diverse viewpoints maintained.
         Range: [0.0, 1.0]

  SSS  — Stance Stability Score
         Cosine similarity between each agent's Round 1 position and final position.
         Higher = agent held its ground = less sycophantic.
         Range: [0.0, 1.0]. None for single-LLM (no rounds).

  HR   — Hedge Ratio
         Hedge words / total words across all output text.
         Lower = less "on the other hand" hedging.
         Range: [0.0, 1.0]

  RTC  — Rounds to Convergence
         Number of debate rounds completed. Always 1 for single-LLM.

Usage:
    from benchmark.evaluator import evaluate_debate, evaluate_single_llm, EvaluationResult
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from debate.divergence import _get_model
from debate.state import AgentArgument, DebateReport


# ---------------------------------------------------------------------------
# Hedge word list
# ---------------------------------------------------------------------------

HEDGE_WORDS: list[str] = [
    "however", "but", "although", "though", "while", "unless",
    "on the other hand", "on the other side", "balanced", "it depends",
    "nevertheless", "nonetheless", "that said", "conversely",
    "in contrast", "yet", "despite", "regardless", "alternatively",
    "to be fair", "admittedly", "granted", "even so", "at the same time",
    "both sides", "nuanced", "complex", "complicated", "trade-off",
    "trade off", "pros and cons",
]

# Pre-compiled patterns for efficiency
_HEDGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)
    for w in HEDGE_WORDS
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Metrics for one (question, system_variant) pair."""

    question_id: int
    question: str
    system: str           # "full_system" | "no_prohibition" | "sequential" |
                          # "fixed_rounds" | "fulltext_embedding" | "single_llm"
    pds: float            # Position Diversity Score
    sss: Optional[float]  # Stance Stability Score (None for single_llm)
    hedge_ratio: float    # HR
    rounds: int           # Rounds to convergence
    convergence_status: str  # "converged" | "max_rounds" | "partial" | "single_llm"

    # Raw text snapshot for manual inspection
    agent_positions: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "system": self.system,
            "pds": round(self.pds, 4),
            "sss": round(self.sss, 4) if self.sss is not None else None,
            "hedge_ratio": round(self.hedge_ratio, 4),
            "rounds": self.rounds,
            "convergence_status": self.convergence_status,
            "agent_positions": self.agent_positions,
        }


# ---------------------------------------------------------------------------
# Single-LLM input schema
# ---------------------------------------------------------------------------

@dataclass
class SingleLLMPerspective:
    """One perspective extracted from a single-LLM multi-perspective response."""
    role: str           # "optimist" | "pessimist" | "devil"
    position: str       # One-sentence stance
    key_claims: list[str]
    full_text: str      # Full argument text


@dataclass
class SingleLLMResult:
    """Container for single-LLM output on one question."""
    question_id: int
    question: str
    perspectives: list[SingleLLMPerspective]  # Always 3: optimist, pessimist, devil


# ---------------------------------------------------------------------------
# Core metric implementations
# ---------------------------------------------------------------------------

def _compute_pds(key_claims_per_agent: list[list[str]]) -> float:
    """Position Diversity Score.

    Reuses the same logic as debate/divergence.py:
      For each pair of agents, find max cosine similarity across their claims.
      PDS = 1.0 - mean(pairwise max similarities).

    Returns 0.0 if fewer than 2 agents have claims.
    """
    valid = [claims for claims in key_claims_per_agent if claims]
    if len(valid) < 2:
        return 0.0

    model = _get_model()
    pairwise_max_sims: list[float] = []

    from itertools import combinations
    for claims_a, claims_b in combinations(valid, 2):
        all_claims = claims_a + claims_b
        embeddings = model.encode(all_claims, normalize_embeddings=True)
        emb_a = embeddings[: len(claims_a)]
        emb_b = embeddings[len(claims_a):]
        sim_matrix = emb_a @ emb_b.T
        pairwise_max_sims.append(float(sim_matrix.max()))

    return round(1.0 - sum(pairwise_max_sims) / len(pairwise_max_sims), 4)


def _compute_sss(
    round1_positions: list[str],
    final_positions: list[str],
) -> float:
    """Stance Stability Score.

    Embeds Round 1 position and final position for each agent.
    Returns mean cosine similarity across all (non-sentinel) agent pairs.
    Higher = agent maintained its stance = less sycophantic.

    Returns 1.0 if only one round (nothing could have changed).
    """
    if not round1_positions or not final_positions:
        return 1.0
    if len(round1_positions) != len(final_positions):
        # Mismatch — degenerate case
        return 1.0

    model = _get_model()
    all_texts = round1_positions + final_positions
    embeddings = model.encode(all_texts, normalize_embeddings=True)

    n = len(round1_positions)
    emb_r1 = embeddings[:n]
    emb_final = embeddings[n:]

    sims = [float(emb_r1[i] @ emb_final[i]) for i in range(n)]
    return round(sum(sims) / len(sims), 4)


def _compute_hedge_ratio(texts: list[str]) -> float:
    """Hedge Ratio.

    Counts hedge word occurrences across all texts.
    Returns hedge_count / total_word_count.
    Returns 0.0 if no text.
    """
    combined = " ".join(texts)
    total_words = len(combined.split())
    if total_words == 0:
        return 0.0

    hedge_count = sum(
        len(pattern.findall(combined))
        for pattern in _HEDGE_PATTERNS
    )
    return round(hedge_count / total_words, 4)


# ---------------------------------------------------------------------------
# Public API: evaluate a full DebateReport
# ---------------------------------------------------------------------------

def evaluate_debate(
    report: DebateReport,
    question_id: int,
    question: str,
    system: str = "full_system",
) -> EvaluationResult:
    """Compute all 4 metrics from a completed DebateReport.

    Args:
        report:      Completed DebateReport from the graph.
        question_id: Question index (from questions.json).
        question:    Original question text.
        system:      Variant label (e.g., "full_system", "no_prohibition").

    Returns:
        EvaluationResult with all metrics filled in.
    """
    trace = report.reasoning_trace
    if not trace:
        return EvaluationResult(
            question_id=question_id, question=question, system=system,
            pds=0.0, sss=None, hedge_ratio=0.0, rounds=0,
            convergence_status=report.convergence_status,
        )

    # --- PDS: final round key_claims per agent (skip sentinels) ---
    final_round = trace[-1]
    final_args: list[AgentArgument] = [
        a for a in final_round.arguments if not a.is_sentinel
    ]
    pds = _compute_pds([a.key_claims for a in final_args])

    # --- SSS: Round 1 position vs. final position per agent ---
    round1_args: list[AgentArgument] = [
        a for a in trace[0].arguments if not a.is_sentinel
    ]
    if len(trace) == 1:
        sss = 1.0  # Only one round — stance trivially stable
    else:
        # Match agents by role
        r1_map = {a.agent_role: a.position for a in round1_args}
        rf_map = {a.agent_role: a.position for a in final_args}
        shared_roles = [r for r in r1_map if r in rf_map]
        if shared_roles:
            r1_positions = [r1_map[r] for r in shared_roles]
            rf_positions = [rf_map[r] for r in shared_roles]
            sss = _compute_sss(r1_positions, rf_positions)
        else:
            sss = None

    # --- Hedge Ratio: all positions + reasoning across all rounds ---
    all_texts: list[str] = []
    for record in trace:
        for arg in record.arguments:
            if not arg.is_sentinel:
                all_texts.append(arg.position)
                all_texts.append(arg.reasoning)
    # Also include verdict
    all_texts.append(report.verdict)
    hr = _compute_hedge_ratio(all_texts)

    # --- Rounds ---
    rounds = len(trace)

    # --- Agent positions snapshot ---
    agent_positions = {a.agent_role: a.position for a in final_args}

    return EvaluationResult(
        question_id=question_id,
        question=question,
        system=system,
        pds=pds,
        sss=sss,
        hedge_ratio=hr,
        rounds=rounds,
        convergence_status=report.convergence_status,
        agent_positions=agent_positions,
    )


# ---------------------------------------------------------------------------
# Public API: evaluate a SingleLLMResult
# ---------------------------------------------------------------------------

def evaluate_single_llm(result: SingleLLMResult) -> EvaluationResult:
    """Compute applicable metrics for a single-LLM multi-perspective response.

    PDS and HR apply. SSS is None (no rounds). Rounds = 1.
    """
    perspectives = [p for p in result.perspectives if p.key_claims]

    # PDS: diversity across the three perspectives
    pds = _compute_pds([p.key_claims for p in perspectives])

    # Hedge Ratio: all perspective text
    all_texts = [p.full_text for p in result.perspectives]
    hr = _compute_hedge_ratio(all_texts)

    agent_positions = {p.role: p.position for p in result.perspectives}

    return EvaluationResult(
        question_id=result.question_id,
        question=result.question,
        system="single_llm",
        pds=pds,
        sss=None,
        hedge_ratio=hr,
        rounds=1,
        convergence_status="single_llm",
        agent_positions=agent_positions,
    )
