# Phase 3: Synthesis & Report - Research

**Researched:** 2026-04-24
**Domain:** LangGraph node replacement, Pydantic model design, LLM structured output, confidence scoring
**Confidence:** HIGH

## Summary

Phase 3 replaces `synthesize_stub` in `debate/nodes/synthesize.py` with a real Synthesizer agent. The stub currently writes only `{"status": "converged" | "max_rounds"}` to state. The real synthesizer must: (1) call the LLM with a compact summary of `round_history` to get structured `consensus_points`, `disputed_points`, and `verdict`; (2) compute `confidence_score` in Python code — never from the LLM; (3) assemble a `DebateReport` Pydantic model; and (4) write it to `state["final_report"]`.

The node name `"synthesize_stub"` stays unchanged in `debate/graph.py` — no graph rewiring needed. The only files that change are `debate/state.py` (add `DebateReport` and `DisputedPoint` models, type `final_report` properly) and `debate/nodes/synthesize.py` (replace stub with real implementation). A new `tests/test_phase3.py` covers the confidence formula, non-convergence path, and end-to-end DebateReport field completeness.

**Primary recommendation:** Add `DebateReport` to `debate/state.py`, implement the synthesizer as a drop-in replacement following the `_invoke_with_retry` pattern already established in `agents.py`, and compute confidence entirely in Python after the LLM call.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SYNTH-01 | Synthesizer agent produces a final verdict after debate completes | LLM call with `with_structured_output(SynthesizerOutput)` in `synthesize_stub` replacement; `verdict` field in `DebateReport` |
| SYNTH-02 | Final report contains: consensus points, disputed points, confidence score (formula-derived), verdict | `DebateReport` Pydantic model with all four fields; LLM provides consensus/disputed/verdict; code provides confidence_score |
| SYNTH-03 | Confidence score is formula-derived: `(1 - max_divergence_score) * round_adjustment` — never LLM-invented | Pure Python function `compute_confidence_score()` using `round_history` divergence scores and `round_num`; formula verified working |
| SYNTH-04 | Synthesizer has honest-uncertainty path: if debate did not converge, report says so explicitly | Prompt instructs LLM to include "agents did not reach consensus" language when `convergence_status == "max_rounds"`; injected as conditional text in the human message before the LLM call |
| SYNTH-05 | Full reasoning trace stored: all rounds, all arguments, all concessions with attribution | `reasoning_trace: list[RoundRecord]` copies `state["round_history"]` verbatim; `concession_log: list[Concession]` flattened from all `AgentArgument.concessions` across all rounds |
</phase_requirements>

---

## Standard Stack

### Core (all already installed — no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pydantic | 2.12.4 | `DebateReport`, `DisputedPoint`, `SynthesizerOutput` model definitions | Already project dependency; `BaseModel` + `Field` is the established pattern in `state.py` |
| langchain-anthropic | 1.4.1 | `ChatAnthropic.with_structured_output(SynthesizerOutput)` | Already used in `agents.py` with `include_raw=True` retry pattern |
| langchain-core | 1.3.1 | `HumanMessage`, `SystemMessage` | Already used in `agents.py` |
| datetime (stdlib) | — | `created_at: datetime` field on `DebateReport` | `datetime.now(timezone.utc)` produces UTC-aware timestamps; serializes cleanly with `model_dump_json()` |

**No new pip installs needed.** All dependencies are already in the project.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `with_structured_output(SynthesizerOutput)` | Plain LLM call + manual JSON parse | Don't hand-roll JSON parsing — existing retry wrapper handles ValidationError cleanly |
| Formula in Python code | Ask LLM to compute confidence | LLM-invented scores are non-deterministic and non-auditable; formula guarantees reproducibility (SYNTH-03) |
| `list[DisputedPoint]` for disputed_points | `list[dict]` | Pydantic model gives typed `topic` and `agent_positions` fields; better for Phase 4 JSON serialization and Phase 5 UI rendering |

---

## Architecture Patterns

### Recommended Project Structure (Phase 3 changes only)

```
debate/
├── state.py              # ADD: DebateReport, DisputedPoint models; type final_report
├── nodes/
│   └── synthesize.py     # REPLACE: stub -> real LLM + confidence formula + DebateReport
tests/
└── test_phase3.py        # NEW: unit + integration tests for SYNTH-01..05
```

No other files change. `debate/graph.py` node registration stays as-is.

### Pattern 1: DebateReport Model in state.py

**What:** Add two new Pydantic models alongside `Concession`, `AgentArgument`, `RoundRecord`. Update `final_report` type annotation.

**When to use:** Centralizing all schemas in `state.py` is the existing project convention (file docstring says "Single source of truth for all debate graph schemas").

```python
# Source: verified against existing state.py pattern + pydantic 2.12.4
from typing import Literal
from datetime import datetime

class DisputedPoint(BaseModel):
    """One topic where agents took opposing positions."""
    topic: str = Field(description="The specific point of disagreement")
    agent_positions: dict[str, str] = Field(
        description="Map of agent_role -> their position on this point"
    )

class DebateReport(BaseModel):
    """Final output of the debate system. Written to state['final_report']."""
    debate_id: str
    topic: str
    consensus_points: list[str] = Field(
        description="Points all agents agreed on (can be empty)"
    )
    disputed_points: list[DisputedPoint] = Field(
        description="Points where agents held opposing positions"
    )
    verdict: str = Field(
        description="Synthesizer's final assessment of the debate outcome"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Formula-derived score: (1 - max_divergence_score) * round_adjustment"
    )
    convergence_status: Literal["converged", "max_rounds", "partial"] = Field(
        description="How the debate loop terminated"
    )
    reasoning_trace: list[RoundRecord] = Field(
        description="All rounds from round_history — full argument provenance"
    )
    concession_log: list[Concession] = Field(
        description="All concessions across all rounds, flattened"
    )
    created_at: datetime
```

**DebateState update** — change `final_report` annotation:

```python
# Before (current):
final_report: Optional[object]  # DebateReport type added in Phase 3

# After:
final_report: Optional["DebateReport"]  # Forward ref; resolves at runtime
```

### Pattern 2: SynthesizerOutput (LLM-only fields)

**What:** Separate Pydantic model for just the fields the LLM produces. `confidence_score` is NOT in this model — it is computed in code after the LLM call.

```python
# Source: derived from agents.py AgentArgument pattern
class SynthesizerOutput(BaseModel):
    """Structured output from the Synthesizer LLM call.
    confidence_score is intentionally absent -- computed in code (SYNTH-03)."""
    consensus_points: list[str] = Field(
        description="3-7 claims all agents agreed on, or empty list if none"
    )
    disputed_points: list[DisputedPoint] = Field(
        description="2-5 key points of disagreement with per-agent positions"
    )
    verdict: str = Field(
        description="2-4 sentence synthesis of the debate outcome"
    )
```

### Pattern 3: Confidence Score Formula (pure Python)

**What:** A pure Python function that takes state values and returns a float. Lives in `debate/nodes/synthesize.py`. No LLM call involved.

```python
# Source: REQUIREMENTS.md SYNTH-03 + verified formula behavior
from debate.divergence import DIVERGE_THRESHOLD  # 0.75

_ROUND_ADJUSTMENTS: dict[int, float] = {1: 1.0, 2: 0.9, 3: 0.8}

def _compute_confidence_score(round_history: list, round_num: int) -> float:
    """Compute confidence score from debate outcome.

    Formula: (1 - max_divergence_score) * round_adjustment
      - max_divergence_score: highest per-round divergence score across all rounds
      - round_adjustment: 1.0 for 1 round, 0.9 for 2 rounds, 0.8 for 3+ rounds
        (penalizes debates that needed more rounds to resolve)

    Edge cases:
      - round_history empty or all divergence_score == 0.0: max_divergence = 0.0 -> score near 1.0
      - round_num 0 (should not occur in normal flow): treated as round 1 (adjustment=1.0)
    """
    if round_history:
        max_divergence = max(r.divergence_score for r in round_history)
    else:
        max_divergence = 0.0
    round_adjustment = _ROUND_ADJUSTMENTS.get(round_num, 0.8)
    return round((1.0 - max_divergence) * round_adjustment, 4)
```

**Verified formula outputs:**

| round_num | divergence_score | confidence_score |
|-----------|-----------------|-----------------|
| 1 | 0.0 | 1.0 (perfect convergence, one round) |
| 1 | 0.5 | 0.5 |
| 2 | 0.3 | 0.63 |
| 3 | 0.7 | 0.24 |
| 3 | 0.0 | 0.8 (converged but took 3 rounds) |
| 3 | 1.0 | 0.0 (max divergence) |

### Pattern 4: Convergence Status Determination (pure Python)

```python
# Source: verified against route_divergence logic in dispatch.py
from debate.divergence import DIVERGE_THRESHOLD  # 0.75

def _determine_convergence_status(
    divergence_score: float,
    round_num: int,
    max_rounds: int,
) -> str:
    if divergence_score < DIVERGE_THRESHOLD:
        return "converged"
    elif round_num >= max_rounds:
        return "max_rounds"
    else:
        return "partial"  # defensive fallback
```

**Critical fix:** The existing stub hardcodes `0.25` for the convergence threshold check instead of importing `DIVERGE_THRESHOLD` (0.75). The real synthesizer MUST import from `debate.divergence`.

### Pattern 5: LLM Prompt for Synthesizer

**What:** The synthesizer prompt takes a compact summary of `round_history` — NOT the raw `reasoning` prose (which is verbose and not needed for synthesis). This keeps the prompt under ~1500 tokens for a 3-round debate.

```python
def _build_synthesis_context(
    topic: str,
    round_history: list,
    convergence_status: str,
) -> str:
    """Build a compact human message for the synthesizer.

    Includes: per-round per-agent position + key_claims + concessions.
    Excludes: raw reasoning prose (too verbose, not needed for synthesis).
    """
    lines = [f"Topic: {topic}\n"]
    for record in round_history:
        lines.append(f"--- Round {record.round_num + 1} (divergence: {record.divergence_score:.2f}) ---")
        for arg in record.arguments:
            claims = "\n".join(f"  - {c}" for c in arg.key_claims)
            lines.append(
                f"[{arg.agent_role.upper()}] confidence={arg.confidence:.0%}\n"
                f"Position: {arg.position}\n"
                f"Claims:\n{claims}"
            )
            if arg.concessions:
                for c in arg.concessions:
                    lines.append(
                        f"  CONCESSION: '{c.conceded_point}' "
                        f"(triggered by {c.triggered_by_agent}: {c.triggered_by_claim})"
                    )
        lines.append("")
    
    # Honest non-convergence instruction (SYNTH-04)
    if convergence_status == "max_rounds":
        lines.append(
            "IMPORTANT: The agents did NOT reach consensus — the debate ended because "
            "the maximum number of rounds was reached. Your verdict MUST explicitly "
            "state that agents did not reach consensus. Do not fabricate agreement."
        )
    
    return "\n".join(lines)
```

### Pattern 6: synthesize_stub Replacement (full flow)

**What:** The function signature `synthesize_stub(state: DebateState) -> dict` stays identical. The graph node name `"synthesize_stub"` stays identical. Only the implementation changes.

```python
def synthesize_stub(state: DebateState) -> dict:
    """Phase 3 synthesizer. Replaces stub with real LLM call + DebateReport assembly."""
    topic = state.get("topic", "")
    debate_id = state.get("debate_id", "")
    round_history = state.get("round_history", [])
    round_num = state.get("round_num", 0)
    max_rounds = state.get("max_rounds", 3)
    divergence_score = state.get("divergence_score", 0.0)
    diverged_pairs = state.get("diverged_pairs", [])

    # 1. Determine convergence status (Python, not LLM)
    convergence_status = _determine_convergence_status(divergence_score, round_num, max_rounds)

    # 2. Build compact synthesis context
    context = _build_synthesis_context(topic, round_history, convergence_status)

    # 3. Call LLM for consensus/disputed/verdict fields
    synthesis = _invoke_synthesizer(context)

    # 4. Compute confidence score (Python formula, not LLM)
    confidence_score = _compute_confidence_score(round_history, round_num)

    # 5. Extract concession log from round_history
    concession_log = [
        c for record in round_history for arg in record.arguments for c in arg.concessions
    ]

    # 6. Assemble DebateReport
    report = DebateReport(
        debate_id=debate_id,
        topic=topic,
        consensus_points=synthesis.consensus_points,
        disputed_points=synthesis.disputed_points,
        verdict=synthesis.verdict,
        confidence_score=confidence_score,
        convergence_status=convergence_status,
        reasoning_trace=round_history,
        concession_log=concession_log,
        created_at=datetime.now(timezone.utc),
    )

    return {"final_report": report, "status": convergence_status}
```

### Anti-Patterns to Avoid

- **Compute confidence_score in the LLM prompt:** LLM-invented numbers are non-deterministic and non-auditable. Formula in code is required by SYNTH-03.
- **Dump raw `reasoning` prose into the synthesis prompt:** Verbatim reasoning for 3 rounds is ~3500 tokens of word salad. Use position + key_claims + concessions for ~1000 tokens.
- **Change the node name from `synthesize_stub`:** Would break `graph.py` and require rewiring. The name stays, the implementation changes.
- **Hardcode the convergence threshold (0.25 bug in stub):** Always import `DIVERGE_THRESHOLD` from `debate.divergence`.
- **Fabricate consensus when `convergence_status == "max_rounds"`:** Honest uncertainty path is a hard requirement (SYNTH-04). Use conditional prompt instruction.
- **Use `list[dict]` for `disputed_points`:** Use `DisputedPoint` Pydantic model for type safety and Phase 4 JSON serialization compatibility.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LLM structured output | Custom JSON schema + regex parse | `llm.with_structured_output(SynthesizerOutput, include_raw=True)` | Existing retry pattern in `agents.py`; handles ValidationError cleanly |
| JSON serialization of DebateReport | Custom `to_json()` method | `report.model_dump_json()` | Pydantic 2 handles `datetime`, nested models, `list[RoundRecord]` natively |
| Concession extraction | Complex history walker | One-liner list comprehension | `[c for r in round_history for a in r.arguments for c in a.concessions]` |

**Key insight:** All infrastructure is already built. Phase 3 is purely additive — new models in `state.py`, replacement implementation in `synthesize.py`, new tests.

---

## Common Pitfalls

### Pitfall 1: Convergence Threshold Inconsistency

**What goes wrong:** Using a hardcoded threshold (like `0.25` in the current stub) instead of `DIVERGE_THRESHOLD` (0.75) from `debate.divergence`. A debate that terminates with `divergence_score=0.4` would be misclassified as `"converged"` instead of `"max_rounds"`.

**Why it happens:** The stub was written as a placeholder and hardcoded `0.25` for simplicity.

**How to avoid:** Always `from debate.divergence import DIVERGE_THRESHOLD` and use it in `_determine_convergence_status`.

**Warning signs:** `convergence_status` disagrees with what `route_divergence` decided; tests with `divergence_score=0.5` return `"converged"`.

### Pitfall 2: round_num Off-by-One

**What goes wrong:** `round_num` in `DebateState` is the count of completed rounds (incremented by `collect_round1`), NOT the current round index. After 1 round: `round_num=1`. After 3 rounds: `round_num=3`. If you use `round_num` as an index (e.g., `round_history[round_num]`), you get an IndexError or wrong round.

**Why it happens:** `collect_round1` does `round_num = round_num + 1` at the end of each round collection.

**How to avoid:** Use `len(round_history)` for indexing; use `round_num` only for the round count in `_ROUND_ADJUSTMENTS` lookup.

**Warning signs:** `_ROUND_ADJUSTMENTS.get(round_num, 0.8)` returns `0.8` when only 1 round ran.

### Pitfall 3: SynthesizerOutput Includes confidence_score

**What goes wrong:** Including `confidence_score` in `SynthesizerOutput` causes the LLM to invent a number, violating SYNTH-03. Worse, Pydantic validates it as a float in [0,1] so it passes silently.

**How to avoid:** `SynthesizerOutput` has exactly three fields: `consensus_points`, `disputed_points`, `verdict`. Confidence is computed in `_compute_confidence_score()` after the LLM call.

### Pitfall 4: max_divergence_score vs. Final divergence_score

**What goes wrong:** Using only `state["divergence_score"]` (the last round's score) instead of `max(r.divergence_score for r in round_history)`. If divergence was 0.9 in round 1 but converged to 0.2 in round 2, the formula would yield `(1-0.2)*0.9 = 0.72` (inaccurate — the debate was highly contentious).

**Why it happens:** `divergence_score` in state is last-write-wins (only the latest round).

**How to avoid:** In `_compute_confidence_score`, iterate `round_history` to get the maximum divergence across all rounds.

**Note:** Edge case — if `round_history` is empty or all `divergence_score` values are 0.0 (e.g., Phase 1 test scenarios), fall back to `0.0`. The `default=0.0` on `RoundRecord.divergence_score` means this is safe.

### Pitfall 5: final_report Type in DebateState

**What goes wrong:** Keeping `final_report: Optional[object]` prevents Phase 4 from calling `.model_dump_json()` without a cast. Phase 5 UI accessing `state["final_report"].consensus_points` raises `AttributeError` without IDE hints.

**How to avoid:** Update to `final_report: Optional["DebateReport"]` in `DebateState`. The forward reference resolves correctly since `DebateReport` is defined in the same module before `DebateState`.

---

## Code Examples

### Full synthesize_stub Replacement (verified pattern)

```python
# debate/nodes/synthesize.py
# Source: verified against agents.py pattern + state.py schema inspection
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from debate.divergence import DIVERGE_THRESHOLD
from debate.llm import _make_llm
from debate.state import Concession, DebateReport, DebateState, DisputedPoint


class SynthesizerOutput(BaseModel):
    """LLM-only structured output. confidence_score intentionally absent (SYNTH-03)."""
    consensus_points: list[str] = Field(
        description="Claims all 3 agents agreed on (empty list if no consensus)"
    )
    disputed_points: list[DisputedPoint] = Field(
        description="Key points where agents disagreed, with each agent's position"
    )
    verdict: str = Field(
        description="2-4 sentence synthesis of the debate outcome"
    )


_ROUND_ADJUSTMENTS: dict[int, float] = {1: 1.0, 2: 0.9, 3: 0.8}

_SYNTHESIZER_SYSTEM = """You are the Debate Synthesizer. Your role is to produce an honest,
structured summary of a multi-agent debate.

Rules:
1. Extract only claims explicitly made by agents — do not add new claims
2. If agents disagreed on a point, list it as disputed with each agent's position
3. Only list consensus points where agents genuinely agreed (keyword overlap or explicit concession)
4. Your verdict summarizes the overall outcome in 2-4 sentences
5. Do NOT invent confidence scores or numerical assessments"""


def _compute_confidence_score(round_history: list, round_num: int) -> float:
    max_divergence = max((r.divergence_score for r in round_history), default=0.0)
    round_adjustment = _ROUND_ADJUSTMENTS.get(round_num, 0.8)
    return round((1.0 - max_divergence) * round_adjustment, 4)


def _determine_convergence_status(
    divergence_score: float, round_num: int, max_rounds: int
) -> str:
    if divergence_score < DIVERGE_THRESHOLD:
        return "converged"
    elif round_num >= max_rounds:
        return "max_rounds"
    return "partial"


def synthesize_stub(state: DebateState) -> dict:
    # ... (see Pattern 6 above for full implementation)
    ...
```

### DebateReport + DisputedPoint Models (verified instantiation)

```python
# Source: verified via python -c "..." — all fields instantiate and serialize cleanly
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Literal
from debate.state import RoundRecord, Concession

class DisputedPoint(BaseModel):
    topic: str
    agent_positions: dict[str, str]  # {'optimist': '...', 'pessimist': '...'}

class DebateReport(BaseModel):
    debate_id: str
    topic: str
    consensus_points: list[str]
    disputed_points: list[DisputedPoint]
    verdict: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    convergence_status: Literal["converged", "max_rounds", "partial"]
    reasoning_trace: list[RoundRecord]
    concession_log: list[Concession]
    created_at: datetime
# model_dump_json() verified working with nested RoundRecord and datetime
```

### Concession Log Extraction (one-liner, verified)

```python
# Source: verified against Concession and AgentArgument schemas in state.py
concession_log = [
    c
    for record in round_history
    for arg in record.arguments
    for c in arg.concessions
]
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `final_report: Optional[object]` | `final_report: Optional["DebateReport"]` | Phase 3 | Type-safe access in Phase 4/5 |
| `synthesize_stub` writes only `{"status": ...}` | Full `DebateReport` written to `state["final_report"]` | Phase 3 | Enables Phase 4 persistence |
| Hardcoded `0.25` convergence check in stub | Import `DIVERGE_THRESHOLD` from `debate.divergence` | Phase 3 (fix) | Consistent threshold across the system |

---

## Open Questions

1. **`convergence_status: "partial"` — when does it occur?**
   - What we know: `route_divergence` only produces `"synthesize_stub"` on `max_rounds` or `divergence < DIVERGE_THRESHOLD`. There is no current path that produces "partial" as a routing decision.
   - What's unclear: Is "partial" ever reachable in normal flow, or purely a defensive fallback in the synthesizer?
   - Recommendation: Keep `"partial"` in the type as a defensive fallback for the synthesizer's own classification. The synthesizer determines `convergence_status` independently of how `route_divergence` decided to terminate — it reads `divergence_score` and `round_num` from state. If `divergence_score >= DIVERGE_THRESHOLD` and `round_num < max_rounds` somehow (should not happen), "partial" fires. Document as edge case in code comment.

2. **Sentinel AgentArgument in reasoning_trace**
   - What we know: `_invoke_with_retry` can inject `is_sentinel=True` arguments on 3 parse failures.
   - What's unclear: Should the synthesizer skip sentinel arguments when building the synthesis context?
   - Recommendation: Filter `arg.is_sentinel == False` when building `_build_synthesis_context`. Sentinel arguments carry no real content. Document in code comment.

---

## Environment Availability

Step 2.6: SKIPPED — Phase 3 uses only already-installed project dependencies. No new external tools, services, or CLI utilities are required. All packages verified present: pydantic 2.12.4, langchain-anthropic, langchain-core, anthropic SDK.

---

## Validation Architecture

`workflow.nyquist_validation` is explicitly `false` in `.planning/config.json` — this section is skipped per configuration.

---

## Sources

### Primary (HIGH confidence)

- Direct code inspection: `debate/state.py` (all models, DebateState fields), `debate/nodes/synthesize.py` (stub implementation), `debate/nodes/agents.py` (`_invoke_with_retry` pattern), `debate/nodes/divergence_check.py` (RoundRecord back-fill), `debate/divergence.py` (`DIVERGE_THRESHOLD=0.75`, `compute_divergence` formula), `debate/graph.py` (node names, wiring)
- Live Python verification: `DebateReport` prototype instantiation and `model_dump_json()` serialization confirmed working; confidence formula edge cases computed; concession extraction one-liner verified; convergence threshold bug in stub confirmed (0.25 vs 0.75)
- `debate/nodes/collect.py`: confirmed `round_num` increments at end of each collect — `round_num` is count of completed rounds, not current index

### Secondary (MEDIUM confidence)

- `.planning/phases/02-debate-engine/02-VERIFICATION.md`: confirmed Phase 2 state at handoff — all 4 success criteria verified, `synthesize_stub` is Phase 3 placeholder
- `tests/test_phase2.py` test naming conventions: `test_{behavior}_{condition}` pattern used throughout

### Tertiary (LOW confidence)

- Token count estimate for synthesis prompt (~1000-3500 tokens for 3-round debate): rough calculation based on typical LLM response lengths, not measured from real runs

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; existing packages verified installed
- Architecture (DebateReport model, synthesizer flow): HIGH — verified by prototype instantiation and schema inspection
- Pitfalls (threshold bug, round_num semantics, max_divergence vs final): HIGH — verified programmatically
- Confidence formula: HIGH — formula specified in REQUIREMENTS.md SYNTH-03, edge cases verified with Python

**Research date:** 2026-04-24
**Valid until:** 2026-05-24 (stable libraries; LangGraph/Pydantic APIs unlikely to break in 30 days)
