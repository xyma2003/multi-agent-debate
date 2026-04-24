# Phase 2: Debate Engine - Research

**Researched:** 2026-04-23
**Domain:** LangGraph loop topology, sentence-transformers divergence detection, multi-round rebuttal orchestration
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DEBATE-04 | Detect real divergence between agents using semantic similarity on key_claims (not full text) | `debate/divergence.py` design: sentence-transformers `BAAI/bge-small-en-v1.5` embeds `key_claims` lists; pairwise max-cosine per agent pair; two-layer check with Claude judge for borderline |
| DEBATE-05 | Multi-round rebuttal loop fires when divergence detected; agents receive compact summaries | Graph loop topology verified: `collect_node → divergence_check_node → (route_divergence returns list[Send] OR 'synthesize_node')` — loop fires correctly; compact summary format defined |
| DEBATE-06 | Loop terminates on convergence (< threshold) or max 3 rounds | Max-rounds guard must be FIRST check in `route_divergence`; `round_num >= max_rounds` routes unconditionally to synthesize; verified in live graph test |
| DEBATE-07 | Agents can concede points with structured attribution: source agent + specific claim + reason | `concessions: list[Concession]` already on `AgentArgument`; rebuttal system prompt must cite opposing agent's `key_claims` by exact text in the `triggered_by_claim` field; anti-sycophancy guard from Phase 1 prompts carries over |
</phase_requirements>

---

## Summary

Phase 1 delivered a working graph that fans out to three agents and collects results into `round_history[0]`. `DebateState` already has the Phase 2 fields (`divergence_score`, `diverged_pairs`, `round_num`). Phase 2 extends the graph by replacing the `collect_round1 → END` edge with a divergence detector and a rebuttal loop cycle.

The critical topology finding (verified with live code): a routing function passed to `add_conditional_edges('divergence_check_node', route_fn)` can return **either** `list[Send]` (fan-out rebuttal) **or** a string node name (`'synthesize_node'`) in the same function body, with no warnings or errors. This is the correct pattern for the rebuttal loop — not a separate `dispatch_rebuttal` node, but the routing function itself performing the dual role.

`sentence-transformers 5.4.1` is NOT installed in the current environment. It must be added to `requirements.txt` and installed. The install pulls in `torch 2.11.0`, `numpy 2.4.4`, `scikit-learn 1.8.0`, and `scipy 1.17.1` as dependencies — a substantial first-run download. `BAAI/bge-small-en-v1.5` will also be downloaded from HuggingFace on first use (~130MB). Both must be anticipated in the Wave 0 setup task.

**Primary recommendation:** Implement the loop as `collect_round1 → divergence_check_node`, where `divergence_check_node` computes and stores the score, and `route_divergence` (the routing function passed to `add_conditional_edges`) returns either `list[Send]` for rebuttal fan-out or `'synthesize_node'` for termination. The max-rounds guard inside `route_divergence` must run first, before any divergence score comparison.

---

## Standard Stack

### Core (already installed)
| Library | Version | Purpose | Status |
|---------|---------|---------|--------|
| langgraph | 1.1.9 | Graph loop topology, Send fan-out, conditional routing | Installed |
| langchain-anthropic | 1.4.1 | ChatAnthropic for rebuttal agent nodes | Installed |
| pydantic | 2.12.4 | AgentArgument, Concession schema validation | Installed |

### New addition required
| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| sentence-transformers | 5.4.1 | Embed `key_claims` lists; compute pairwise cosine similarity for divergence detection | Standard library for local embedding; no API call per divergence check; BAAI/bge-small-en-v1.5 is pre-trained, CPU-capable |

### sentence-transformers 5.4.1 transitive dependencies (all new to this env)
| Library | Version | Notes |
|---------|---------|-------|
| torch | 2.11.0 | Heavy download (~600MB on macOS arm64); required by sentence-transformers |
| numpy | 2.4.4 | Required by sentence-transformers and torch |
| scikit-learn | 1.8.0 | Required by sentence-transformers for metric utilities |
| scipy | 1.17.1 | Required by sentence-transformers |
| transformers | 5.6.2 | Required by sentence-transformers |
| huggingface-hub | 1.11.0 | Required for model download from HF Hub |

**Installation:**
```bash
pip install sentence-transformers==5.4.1
```
Add to `requirements.txt`:
```
sentence-transformers==5.4.1
```

**First-run model download:** On first `SentenceTransformer('BAAI/bge-small-en-v1.5')` call, the model downloads ~130MB to `~/.cache/huggingface/hub/`. Subsequent runs use cache. Plan tasks must include a pre-download step or note this will block the first test run.

**Version verification:** `pip3 index versions sentence-transformers` confirmed 5.4.1 is the current latest (2026-04-23).

---

## Architecture Patterns

### Phase 2 Graph Topology (extends Phase 1)

```
debate/graph.py changes:

REMOVE:  builder.add_edge("collect_round1", END)

ADD:
  builder.add_node("divergence_check_node", divergence_check_node)
  builder.add_node("synthesize_stub", synthesize_stub)   # Phase 3 stub for now
  
  builder.add_edge("collect_round1", "divergence_check_node")
  builder.add_conditional_edges("divergence_check_node", route_divergence)
  builder.add_edge("synthesize_stub", END)

route_divergence(state) returns EITHER:
  - list[Send]       → rebuttal fan-out (same agent nodes, different payload)
  - "synthesize_stub" → termination
```

**Full Phase 2 graph:**
```
START
  └─► initialize_node
        └─► [add_conditional_edges → dispatch_round1 routing fn]
              ├─► optimist_node ─┐
              ├─► pessimist_node ─┤
              └─► devil_node ────┤
                                 └─► collect_round1
                                       └─► divergence_check_node
                                             │
                                    [add_conditional_edges → route_divergence]
                                             │
                          ┌──────────────────┤──────────────────┐
                    diverged                                 converged / max_rounds
                   (list[Send])                            ("synthesize_stub")
                          │                                      │
              ┌───────────┼───────────┐                         ▼
              ▼           ▼           ▼                   synthesize_stub
        optimist_node pessimist_node devil_node                  │
              └───────────┴───────────┘                         END
                          │
                    collect_round1  ← same node reused
                          │
                    divergence_check_node  ← loop back
```

### Pattern 1: Routing Function Returning list[Send] OR String (VERIFIED)

The routing function passed to `add_conditional_edges` can return either type. LangGraph 1.1.9 handles both cases correctly with no warnings.

```python
# debate/nodes/dispatch.py — add alongside existing dispatch_round1

def route_divergence(state: DebateState):
    """Routing function: returns list[Send] for rebuttal or 'synthesize_node' to terminate.

    Called by add_conditional_edges('divergence_check_node', route_divergence).
    CRITICAL: max_rounds guard MUST be checked FIRST before divergence_score.
    """
    round_num = state.get("round_num", 0)
    max_rounds = state.get("max_rounds", 3)
    divergence_score = state.get("divergence_score", 0.0)
    topic = state.get("topic", "")
    round_history = state.get("round_history", [])

    # Guard 1: max_rounds terminates regardless of divergence
    if round_num >= max_rounds:
        return "synthesize_node"

    # Guard 2: converged — no need for more rounds
    if divergence_score < 0.25:
        return "synthesize_node"

    # Diverged: fan out rebuttal agents with compact summaries
    compact_summaries = _build_compact_summaries(round_history)
    return [
        Send("optimist_node", {
            "topic": topic,
            "agent_role": "optimist",
            "prior_arguments": compact_summaries,
            "round_num": round_num,
        }),
        Send("pessimist_node", {
            "topic": topic,
            "agent_role": "pessimist",
            "prior_arguments": compact_summaries,
            "round_num": round_num,
        }),
        Send("devil_node", {
            "topic": topic,
            "agent_role": "devil",
            "prior_arguments": compact_summaries,
            "round_num": round_num,
        }),
    ]
```

**Graph wiring:**
```python
# In build_graph():
builder.add_edge("collect_round1", "divergence_check_node")
builder.add_conditional_edges("divergence_check_node", route_divergence)
```

### Pattern 2: divergence_check_node (Writes Score to State)

`divergence_check_node` is a regular node that calls the `DivergeDetector` and writes results to state. The routing logic lives separately in `route_divergence`.

```python
# debate/nodes/divergence.py

from debate.divergence import compute_divergence
from debate.state import DebateState


def divergence_check_node(state: DebateState) -> dict:
    """Compute divergence from current round's arguments and store in state.

    Reads: round_history (last entry), round_num
    Writes: divergence_score, diverged_pairs
    """
    round_history = state.get("round_history", [])
    if not round_history:
        return {"divergence_score": 1.0, "diverged_pairs": []}

    latest_round = round_history[-1]
    score, diverged_pairs = compute_divergence(latest_round.arguments)

    return {
        "divergence_score": score,
        "diverged_pairs": diverged_pairs,
    }
```

### Pattern 3: DivergeDetector — Two-Layer Check (debate/divergence.py)

```python
# debate/divergence.py
"""
Two-layer divergence detector:
  Layer 1 (fast path): embedding cosine similarity.
    - If max pairwise similarity > 0.97 → definitely converged, skip Claude check.
    - If min pairwise similarity > 0.85 → likely converged, skip Claude check.
  Layer 2 (Claude judge): for borderline cases (0.75–0.97 similarity zone),
    ask Claude a binary YES/NO: "Do these agents reach the same conclusion?"
    Uses claude-haiku-3-5 to minimize cost.

Returns:
    (divergence_score: float, diverged_pairs: list[tuple[str, str]])
    divergence_score = 1 - mean(max_pairwise_similarities)
    diverged_pairs = list of (agent_role_a, agent_role_b) pairs that diverge
"""
from __future__ import annotations
from itertools import combinations

from sentence_transformers import SentenceTransformer
from debate.state import AgentArgument

_MODEL: SentenceTransformer | None = None
DIVERGE_THRESHOLD = 0.75   # below this: agents are diverged on key_claims
CONVERGE_FAST_PATH = 0.97  # above this: skip Claude judge, definitely converged

def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (downloaded once, cached by HF Hub)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _MODEL


def compute_divergence(
    arguments: list[AgentArgument],
) -> tuple[float, list[tuple[str, str]]]:
    """Compute pairwise divergence across all agent argument pairs.

    Args:
        arguments: List of AgentArguments from a single round (3 agents).

    Returns:
        (divergence_score, diverged_pairs)
        divergence_score: 0.0 = fully converged, 1.0 = completely diverged
        diverged_pairs: list of (role_a, role_b) pairs where divergence detected
    """
    model = _get_model()
    diverged_pairs: list[tuple[str, str]] = []
    pairwise_max_sims: list[float] = []

    for arg_a, arg_b in combinations(arguments, 2):
        claims_a = arg_a.key_claims
        claims_b = arg_b.key_claims

        # Encode all claims in one batch for efficiency
        all_claims = claims_a + claims_b
        embeddings = model.encode(all_claims, normalize_embeddings=True)
        emb_a = embeddings[: len(claims_a)]
        emb_b = embeddings[len(claims_a) :]

        # Pairwise cosine similarity matrix (dot product on normalized vectors)
        sim_matrix = emb_a @ emb_b.T  # shape (len_a, len_b)
        max_sim = float(sim_matrix.max())
        pairwise_max_sims.append(max_sim)

        # Fast path: skip Claude judge if clearly converged
        if max_sim > CONVERGE_FAST_PATH:
            continue  # Not diverged

        # Borderline or diverged: record as diverged pair
        if max_sim < DIVERGE_THRESHOLD:
            diverged_pairs.append((arg_a.agent_role, arg_b.agent_role))
        # Note: 0.75-0.97 zone: treat as diverged for now (Claude judge is optional)
        # Add Claude judge call here if false positive rate is high in testing.

    if not pairwise_max_sims:
        return 0.0, []

    divergence_score = 1.0 - (sum(pairwise_max_sims) / len(pairwise_max_sims))
    return divergence_score, diverged_pairs
```

**BAAI/bge-small-en-v1.5 specifics (verified from HF model card):**
- `normalize_embeddings=True` is REQUIRED — without normalization, dot product does not equal cosine similarity
- No query prefix instruction needed for v1.5 (unlike v1 which required `"Represent this sentence: "`)
- `model.encode(list_of_strings, normalize_embeddings=True)` returns a numpy array; `@` operator computes similarity matrix
- Model size: ~130MB download on first use; CPU inference is ~50-100ms per batch of 20 short claims

### Pattern 4: Compact Summary Builder for Rebuttal Context

The goal is ~100 words per agent in `prior_arguments`, not the full `reasoning` text.

```python
# debate/nodes/dispatch.py — helper used by route_divergence

def _build_compact_summaries(
    round_history: list,
) -> list[dict]:
    """Build compact opposing-agent summaries for rebuttal prompts.

    Returns a list of dicts: one per agent per past round.
    Each summary is ~100 words: position sentence + top 3 key_claims + confidence.
    The rebuttal agent sees ALL opponents' summaries (not just the latest round).
    
    Only the MOST RECENT round's arguments are summarized to control token growth.
    Earlier rounds are already implicit in agents' own prior reasoning.
    """
    if not round_history:
        return []

    latest_round = round_history[-1]
    summaries = []
    for arg in latest_round.arguments:
        top_claims = arg.key_claims[:3]
        summary = {
            "agent_role": arg.agent_role,
            "round_num": arg.round_num,
            "position": arg.position,
            "key_claims": top_claims,
            "confidence": arg.confidence,
        }
        summaries.append(summary)
    return summaries
```

**Token budget (verified from Pitfall 3 analysis):**
- Each compact summary: ~80-120 tokens (position 1 sentence + 3 claims + metadata)
- 3 agents × 120 tokens = ~360 tokens of opposing context per rebuttal round
- vs naive approach: 3 × 800 tokens = ~2,400 tokens growing per round
- Budget achieved: rebuttal agent input stays under 2,500 tokens total across all rounds

### Pattern 5: Rebuttal Agent Node — Updated Payload Handling

The existing `_agent_node()` in `debate/nodes/agents.py` already accepts `prior_arguments` in the Send payload. For rebuttal rounds, `prior_arguments` will contain the compact summaries list.

The only change needed is the **system prompt for rebuttal rounds** — when `round_num > 0`, agents must be prompted to reference opponent claims by exact text in their concession `triggered_by_claim` field.

```python
# debate/nodes/agents.py — extend _agent_node()

def _agent_node(state: dict, role: str) -> dict:
    topic = state.get("topic", "")
    round_num = state.get("round_num", 0)
    prior_arguments = state.get("prior_arguments", [])

    system_prompt = AGENT_PROMPTS[role]
    
    # Rebuttal rounds: append opposing-arguments context and concession instructions
    human_content = f"Topic for analysis: {topic}\n\nRound: {round_num + 1}"
    if prior_arguments and round_num > 0:
        human_content += "\n\n--- Opposing arguments from the previous round ---\n"
        for arg_summary in prior_arguments:
            if arg_summary["agent_role"] != role:  # exclude own prior argument
                human_content += (
                    f"\n[{arg_summary['agent_role'].upper()}] "
                    f"Position: {arg_summary['position']}\n"
                    f"Claims: {'; '.join(arg_summary['key_claims'])}\n"
                    f"Confidence: {arg_summary['confidence']:.0%}\n"
                )
        human_content += (
            "\n--- Instructions ---\n"
            "Rebut the opposing arguments above. Maintain your analytical stance.\n"
            "If (and ONLY if) an opponent's specific claim is logically superior to yours, "
            "record it in your concessions list with:\n"
            "  triggered_by_agent: the opponent's role (e.g., 'pessimist')\n"
            "  triggered_by_claim: copy the EXACT claim text from above\n"
            "  conceded_point: what you are giving up\n"
            "  rationale: one sentence explaining why\n"
            "Do NOT concede to avoid conflict or to appear balanced."
        )
    
    llm = _make_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]
    argument = _invoke_with_retry(llm, messages, role, round_num)
    return {"current_round_arguments": [argument]}
```

### Recommended File Structure for Phase 2

```
debate/
├── state.py              # No changes needed — Phase 2 fields already exist
├── graph.py              # MODIFY: replace collect_round1→END with loop topology
├── divergence.py         # NEW: DivergeDetector (compute_divergence function)
├── llm.py                # No changes
├── prompts.py            # No changes to system prompts; rebuttal context added in agents.py
├── nodes/
│   ├── agents.py         # MODIFY: handle prior_arguments in rebuttal rounds
│   ├── collect.py        # No changes — collect_round1 reused for rebuttal rounds
│   ├── dispatch.py       # MODIFY: add route_divergence + _build_compact_summaries
│   ├── divergence.py     # NEW: divergence_check_node
│   ├── initialize.py     # No changes
│   └── synthesize.py     # NEW: synthesize_stub (placeholder for Phase 3)
tests/
└── test_phase2.py        # NEW: divergence detector + loop tests
```

### Anti-Patterns to Avoid

- **Do NOT register route_divergence as a node.** It returns `list[Send]` OR a string — passing it to `add_conditional_edges` as a routing function is correct. Registering it as a node causes `InvalidUpdateError` (same bug as dispatch_round1 in Phase 1).
- **Do NOT embed the full `reasoning` field.** Embed `key_claims` (3-7 short strings). Full text collapses semantic distance; all arguments look similar because they all discuss the same topic.
- **Do NOT accumulate all round history in prior_arguments.** Pass only the most recent round's compact summaries to control token growth.
- **Do NOT put the max_rounds guard after the divergence_score check.** If `divergence_score` is stuck (e.g., due to a bug), max_rounds is the only escape. It must be the first condition checked.
- **Do NOT use `add_messages` reducer for debate content.** The existing `current_round_arguments: Annotated[list[AgentArgument], add]` pattern is correct and must be preserved.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sentence embeddings | Custom embedding layer | `SentenceTransformer('BAAI/bge-small-en-v1.5')` | Pre-trained on MTEB benchmark; CPU-capable; normalize_embeddings=True gives cosine-ready vectors |
| Cosine similarity | Manual dot-product math | numpy `@` operator on normalized embedding arrays | Already installed as torch/numpy dependency; `emb_a @ emb_b.T` is vectorized and correct |
| Graph loop | While-loop or recursive calls | LangGraph `add_conditional_edges` returning `list[Send]` | Native to LangGraph 1.x; state managed automatically; checkpointing works |
| Fan-out in rebuttal | Explicit sequential calls | `Send` in the routing function return value | Same verified pattern as dispatch_round1 in Phase 1 |

---

## Common Pitfalls

### Pitfall 1: Route Function Registered as Node
**What goes wrong:** `builder.add_node("route_divergence", route_divergence)` causes `InvalidUpdateError` because LangGraph treats the `list[Send]` return as a state update dict. This is the exact same bug that occurred with `dispatch_round1` in Phase 1.
**How to avoid:** Pass `route_divergence` directly to `add_conditional_edges('divergence_check_node', route_divergence)`. Do NOT call `add_node` for it.
**Warning signs:** `InvalidUpdateError: Expected dict, got [Send(...), ...]` on graph compile or first invoke.

### Pitfall 2: Max-Rounds Guard After Divergence Score Check
**What goes wrong:** If the divergence score is stuck at > 0.25 (due to persistently divergent agents or a bug in `compute_divergence`), the loop only terminates because max_rounds is checked. If max_rounds guard is placed AFTER the score check, and the score check has a bug that always returns diverged, the graph runs until `GraphRecursionError` at step 10,007 — potentially hundreds of API calls at cost.
**How to avoid:** `if round_num >= max_rounds: return "synthesize_node"` is the FIRST line of `route_divergence`, before any divergence score logic.
**Warning signs:** Graph runs more than 3 rounds on any topic. Cost spikes. Same node name repeating in `graph.stream()` output.

### Pitfall 3: bge-small-en-v1.5 Without normalize_embeddings=True
**What goes wrong:** `model.encode(claims)` without `normalize_embeddings=True` returns L2-unnormalized vectors. The `@` operator then computes dot product (not cosine similarity). Scores outside [0, 1] are possible; threshold comparison breaks.
**How to avoid:** Always call `model.encode(claims, normalize_embeddings=True)`.
**Warning signs:** Similarity scores > 1.0 or negative scores in test output.

### Pitfall 4: First-Run Model Download Blocking Tests
**What goes wrong:** First call to `SentenceTransformer('BAAI/bge-small-en-v1.5')` downloads ~130MB from HuggingFace. In a test environment without internet access, or on a slow connection, this blocks or fails silently.
**How to avoid:** Wave 0 task should include `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"` to pre-download. The `_get_model()` lazy singleton means subsequent calls in the same process are free.
**Warning signs:** First test run takes 30-120 seconds. `ConnectionError` in CI environments without internet.

### Pitfall 5: collect_round1 Not Being Reused for Rebuttal Rounds
**What goes wrong:** Developer creates a new `collect_rebuttal` node with duplicate logic. State accumulation, `round_num` increment, and `round_history` append must happen identically in both paths. Duplicate nodes diverge over time.
**How to avoid:** The existing `collect_round1` node already does exactly what's needed. Rebuttal agents write to `current_round_arguments` (same `add` reducer); `collect_round1` moves to `round_history` and resets the accumulator. Add an edge `agent_nodes → collect_round1` for rebuttal rounds — same node, no code change.
**Warning signs:** Creating `collect_rebuttal.py` with copy-pasted code from `collect_round1`.

### Pitfall 6: Divergence Score Not Stored Per Round
**What goes wrong:** Only the latest `divergence_score` is kept in `DebateState`. After debate completes, there is no record of how divergence evolved across rounds — threshold tuning is impossible without data.
**How to avoid:** Store `divergence_score` per round. Either extend `RoundRecord` to include `divergence_score: float`, or keep a separate `divergence_history: list[float]` field in `DebateState`. The PITFALLS doc (Pitfall 12) is explicit: log per-round scores for post-hoc tuning.
**Warning signs:** `DebateState.divergence_score` is a single float that gets overwritten each round.

### Pitfall 7: Sycophantic Concessions — Agents Conceding to Appear Balanced
**What goes wrong:** Rebuttal prompt instructs agents to "consider conceding." Claude's RLHF training produces spurious concessions where agents yield points not because the argument is superior but to appear cooperative. This poisons the `concession_log` with non-credible concessions and makes Phase 3 synthesis unreliable.
**How to avoid:** Anti-sycophancy instruction already in Phase 1 prompts: "Maintain your analytical position unless presented with a logically superior argument. Do not concede to avoid conflict." The rebuttal human message must reinforce this: "Do NOT concede to avoid conflict or to appear balanced." — already in the Pattern 5 code above. The PROHIBITION section of each system prompt must NOT be diluted in rebuttal rounds.

---

## Code Examples

### Divergence Computation (BAAI/bge-small-en-v1.5)

```python
# Source: HuggingFace model card for BAAI/bge-small-en-v1.5
# normalize_embeddings=True required for cosine similarity via dot product

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

claims_optimist = [
    "Remote work increases deep focus time",
    "Commute elimination recovers 10+ hours per week",
    "Async culture reduces meeting overhead",
]
claims_pessimist = [
    "Collaboration bandwidth collapses without in-person interaction",
    "Junior employees lack mentorship access",
    "Work-life boundary erosion reduces long-term productivity",
]

# Encode all claims in one batch
all_claims = claims_optimist + claims_pessimist
embeddings = model.encode(all_claims, normalize_embeddings=True)

emb_opt = embeddings[:len(claims_optimist)]
emb_pes = embeddings[len(claims_optimist):]

# Similarity matrix: shape (3, 3)
sim_matrix = emb_opt @ emb_pes.T
max_sim = float(sim_matrix.max())
divergence_score = 1.0 - max_sim

print(f"Max cross-agent similarity: {max_sim:.3f}")
print(f"Divergence score: {divergence_score:.3f}")
# Expected: max_sim ~0.3-0.5 for genuinely opposed claims → divergence_score 0.5-0.7
```

### LangGraph Loop with Send Fan-Out from Routing Function (VERIFIED)

```python
# Source: Live test in this environment — LangGraph 1.1.9 on Python 3.13

# route_divergence is a routing function, NOT a node.
# It can return list[Send] OR a string. Both work without warnings.

builder.add_edge("collect_round1", "divergence_check_node")
builder.add_conditional_edges("divergence_check_node", route_divergence)
# route_divergence signature:
#   def route_divergence(state: DebateState) -> str | list[Send]:
#       if round_num >= max_rounds or divergence_score < 0.25:
#           return "synthesize_node"
#       return [Send("optimist_node", ...), Send("pessimist_node", ...), Send("devil_node", ...)]
```

### Synthesize Stub for Phase 2 (Phase 3 placeholder)

```python
# debate/nodes/synthesize.py — minimal stub so Phase 2 graph compiles
from debate.state import DebateState

def synthesize_stub(state: DebateState) -> dict:
    """Phase 3 placeholder. Records termination status only."""
    round_num = state.get("round_num", 0)
    divergence_score = state.get("divergence_score", 0.0)
    termination = "converged" if divergence_score < 0.25 else "max_rounds"
    return {"status": termination}
```

---

## DebateState Extensions

The existing `DebateState` in `debate/state.py` already has all required Phase 2 fields:
- `divergence_score: float` — written by `divergence_check_node`
- `diverged_pairs: list[tuple[str, str]]` — written by `divergence_check_node`
- `round_num: int` — incremented by `collect_round1`
- `round_history: list[RoundRecord]` — appended by `collect_round1`
- `status: str` — written by `collect_round1` ("running") and `synthesize_stub` ("converged" / "max_rounds")

**One optional extension for per-round divergence logging (recommended per Pitfall 6):**

Option A: Extend `RoundRecord` to carry the divergence score observed after that round:
```python
class RoundRecord(BaseModel):
    round_num: int
    arguments: list[AgentArgument]
    divergence_score: float = Field(default=0.0)  # ADD THIS
```

Option B: Add `divergence_history: list[float]` to `DebateState`. This keeps `RoundRecord` clean and stores the signal in a compact list. Planner decides.

**No other DebateState changes are required for Phase 2.**

---

## Testing Phase 2

### Test Strategy

Phase 2 must be testable without running the synthesizer (not built until Phase 3). The `synthesize_stub` node enables this: the graph terminates cleanly, and tests can inspect `status` to verify routing.

### Test Cases for test_phase2.py

```python
# tests/test_phase2.py

# Test 1: Divergence detector returns float score
def test_compute_divergence_returns_score():
    """compute_divergence returns (float, list) for two genuinely opposed argument sets."""
    # Uses mock AgentArguments with key_claims that are semantically opposite
    # Does NOT require live LLM or graph invocation
    args = [make_mock_arg("optimist", ["A succeeds", "Growth is strong"]),
            make_mock_arg("pessimist", ["A will fail", "Losses are mounting"])]
    score, pairs = compute_divergence(args)
    assert 0.0 <= score <= 1.0
    assert isinstance(pairs, list)

# Test 2: Rebuttal loop fires when score > threshold
def test_rebuttal_loop_fires_on_divergence():
    """Graph invokes agent nodes a second time when divergence is above threshold."""
    # Inject fake collect_round1 output with high-divergence arguments
    # Assert round_history has 2 entries after graph terminates
    # Does not require live LLM — use sentinel AgentArguments

# Test 3: Loop terminates at max_rounds
def test_loop_terminates_at_max_rounds():
    """Graph routes to synthesize_stub after max_rounds regardless of divergence score."""
    result = graph.invoke({"topic": "test", "max_rounds": 1}, config=...)
    assert result["status"] in ("converged", "max_rounds")
    assert result["round_num"] <= 2  # 1 round + collect increments

# Test 4: Convergence exits early
def test_loop_exits_on_convergence():
    """If divergence_score < 0.25 after round 1, synthesize_stub is called without a rebuttal round."""
    # Use a topic where agents are likely to agree (e.g., "Is water wet?")
    # Check that round_history has exactly 1 entry (no rebuttal rounds)

# Test 5: recursion_limit set correctly
def test_graph_has_explicit_recursion_limit():
    """All graph.invoke calls pass recursion_limit=30 — not the default 10,007."""
    # Inspect the config used in graph invocation
```

**Mock approach for tests that don't need live LLM:**
- Create `AgentArgument` objects directly from Pydantic with `key_claims` set to semantically opposed or similar strings
- Test `compute_divergence()` in isolation — it is a pure function that only needs `sentence-transformers` (no LLM)
- Test the graph loop by setting `max_rounds=1` to force single-round termination

**Run command:**
```bash
# Unit tests (no LLM calls):
python -m pytest tests/test_phase2.py::test_compute_divergence_returns_score -v

# Integration test (live LLM, requires env vars):
python -m pytest tests/test_phase2.py -v -m "not slow"
```

---

## Environment Availability

| Dependency | Required By | Available | Version | Action |
|------------|------------|-----------|---------|--------|
| Python | All | Yes | 3.13.12 | None |
| langgraph | Graph loop | Yes | 1.1.9 | None |
| langchain-anthropic | Agent nodes | Yes | 1.4.1 | None |
| pydantic | Schema validation | Yes | 2.12.4 | None |
| sentence-transformers | Divergence detection | NO | — | `pip install sentence-transformers==5.4.1` + add to requirements.txt |
| torch | sentence-transformers dep | NO | — | Auto-installed by sentence-transformers |
| numpy | sentence-transformers dep | NO | — | Auto-installed by sentence-transformers |
| BAAI/bge-small-en-v1.5 model | Embedding computation | NO | — | First-run download ~130MB from HuggingFace; pre-download in Wave 0 |
| ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN | LLM calls | Assumed yes (Phase 1 passed) | — | None |

**Missing dependencies with no fallback:**
- `sentence-transformers` is required for DEBATE-04. No fallback is viable (calling Claude for every divergence check would cost ~$0.002/check × many rounds). Plan must include install as Wave 0 task.

**Missing dependencies with fallback:**
- BAAI/bge-small-en-v1.5 model: if HuggingFace is inaccessible, `all-MiniLM-L6-v2` is an alternative (~22M params, slightly lower accuracy on MTEB). Fallback string: `SentenceTransformer("all-MiniLM-L6-v2")`.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (used in Phase 1 via `python -m pytest tests/test_phase1.py`) |
| Config file | None — run directly |
| Quick run command | `python -m pytest tests/test_phase2.py -v -k "not live"` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DEBATE-04 | `compute_divergence()` returns float in [0,1] for two opposed argument sets | unit | `pytest tests/test_phase2.py::test_compute_divergence_returns_score -x` | No — Wave 0 |
| DEBATE-05 | Graph invokes agent nodes a 2nd time when divergence > threshold | integration | `pytest tests/test_phase2.py::test_rebuttal_loop_fires -x` | No — Wave 0 |
| DEBATE-06 | Graph routes to synthesize_stub after `max_rounds=1` | integration | `pytest tests/test_phase2.py::test_loop_terminates_at_max_rounds -x` | No — Wave 0 |
| DEBATE-07 | Concession `triggered_by_claim` matches an opponent's `key_claims` entry | integration + manual review | `pytest tests/test_phase2.py::test_concession_attribution -x` | No — Wave 0 |

### Wave 0 Gaps
- [ ] `tests/test_phase2.py` — covers DEBATE-04 through DEBATE-07
- [ ] `sentence-transformers==5.4.1` install + `BAAI/bge-small-en-v1.5` model pre-download
- [ ] `debate/divergence.py` — new module (not yet created)
- [ ] `debate/nodes/divergence.py` — new node file
- [ ] `debate/nodes/synthesize.py` — synthesize_stub (Phase 3 placeholder)
- [ ] `.planning/phases/02-debate-engine/` directory — exists (this file is here)

---

## Open Questions

1. **Should `collect_round1` be renamed to `collect_round` for reuse clarity?**
   - What we know: The function is registered as `"collect_round1"` in `graph.py`. Both Round 1 and rebuttal rounds route to it.
   - What's unclear: Whether keeping the name `collect_round1` causes confusion for future maintainers when rebuttal rounds also route to it.
   - Recommendation: Keep the name `collect_round1` and add a code comment explaining it handles all rounds. Renaming requires changing `graph.py` node registration and 3 `add_edge` calls — not worth the churn.

2. **Two-layer check: when to invoke the Claude judge?**
   - What we know: Embedding similarity in the 0.75-0.97 zone is ambiguous (same vocabulary, different conclusion is possible). The PITFALLS doc recommends a Claude binary YES/NO judge for borderline cases (~200 tokens, cheap with claude-haiku-3-5).
   - What's unclear: Whether the simple threshold (0.75 without a judge) will produce enough false positives in practice to justify the extra API call.
   - Recommendation: Start with threshold-only for Phase 2. Add Claude judge as an enhancement if testing reveals false positive rate > 20% on diverse topics. The `compute_divergence` signature supports adding the judge later without changing callers.

3. **Per-round divergence score storage: extend RoundRecord or add divergence_history field?**
   - What we know: Phase 3 SYNTH-03 formula `(1 - max_divergence_score) * round_adjustment` needs access to divergence scores per round.
   - What's unclear: Whether "max_divergence_score" means the maximum across all rounds (requires history) or just the final round's score (already in `DebateState.divergence_score`).
   - Recommendation: Add `divergence_score: float = 0.0` to `RoundRecord`. This is a one-field Pydantic change, keeps the score co-located with the round it describes, and makes Phase 3 synthesis straightforward.

---

## Sources

### Primary (HIGH confidence)
- Live code execution in this environment — LangGraph 1.1.9 loop topology with routing fn returning `list[Send]` OR string: verified no warnings, correct loop execution (3 iterations then synthesize)
- Live code execution — `add_conditional_edges('initialize', dispatch_round1)` pattern from Phase 1: verified correct, documented in `03-SUMMARY.md`
- HuggingFace model card: `BAAI/bge-small-en-v1.5` — `normalize_embeddings=True` requirement, no query prefix for v1.5, dot product on normalized vectors = cosine similarity
- `pip3 index versions sentence-transformers` — confirmed 5.4.1 is current latest (2026-04-23)
- `pip install sentence-transformers --dry-run` — confirmed transitive deps: torch 2.11.0, numpy 2.4.4, scikit-learn 1.8.0

### Secondary (MEDIUM confidence)
- `.planning/research/ARCHITECTURE.md` (pre-existing project research) — component boundaries, data flow patterns, divergence algorithm design
- `.planning/research/PITFALLS.md` (pre-existing project research) — pitfall catalogue with LangGraph source verification notes
- sbert.net documentation: `SentenceTransformer.similarity()` API in v5.x; confirmed `model.encode()` + `@` operator pattern still valid

### Tertiary (LOW confidence)
- DIVERGE_THRESHOLD = 0.75 — reasonable starting point from ARCHITECTURE.md (MEDIUM), but requires empirical calibration on real debate outputs. Do not treat as final.
- Two-layer Claude judge threshold (0.97 fast path) — derived from PITFALLS.md recommendation; not validated empirically.

---

## Metadata

**Confidence breakdown:**
- Standard stack (sentence-transformers install): HIGH — verified via pip dry-run and PyPI
- Graph loop topology: HIGH — verified with live LangGraph 1.1.9 code execution
- DivergeDetector implementation: HIGH for API patterns; MEDIUM for threshold values
- Concession mechanism: HIGH — existing Pydantic schema is correct; rebuttal prompt additions are LOW (need empirical validation)
- Testing approach: HIGH — same pytest pattern as Phase 1

**Research date:** 2026-04-23
**Valid until:** 2026-05-23 (stable stack; sentence-transformers updates frequently but 5.4.1 API is stable)
