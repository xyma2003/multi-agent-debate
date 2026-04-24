# Feature Landscape: Multi-Agent Debate System

**Domain:** Multi-agent LLM orchestration with structured debate loop
**Researched:** 2026-04-23
**Overall confidence:** HIGH for categories and rationale; MEDIUM for specific implementation details in literature

---

## Summary of Evidence Base

Research covered:
- Foundational academic papers: Du et al. (2305.14325), Liang et al. (MAD, 2305.19118), ChatEval (2308.07201)
- Current 2025-2026 papers: PROClaim (2603.28488), MAD-M2 (2603.20215), deliberation dynamics (2510.10002), conformal social choice (2604.07667), RCS (2604.12196), sycophancy in debate (2604.21564), small-world debate topology (2512.18094), DiscoUQ divergence analysis
- Existing parallel research artifact: ARCHITECTURE.md (already defines data models, graph topology, divergence algorithm, concession data structures)

The ARCHITECTURE.md already encodes several features as concrete design decisions. This file augments that by mapping the full feature landscape and clearly categorizing what must exist vs. what differentiates vs. what to skip.

---

## Table Stakes

Features without which the system is not credible as a "multi-agent debate system." Missing any of these means the demo fails to demonstrate the core claim.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Independent Round 1 analysis | Without cross-isolation, agents contaminate each other; no independent perspective exists | Low | Already designed in ARCHITECTURE.md via Send payload isolation |
| Multi-round rebuttal loop | Single-round systems are just parallel querying, not debate | Low–Med | Already in graph topology; LangGraph loop handles it |
| Distinct agent personas / cognitive biases | Homogeneous agents produce trivially similar output; debate is meaningless | Low | System prompt differentiation; already in PROJECT.md roster |
| Semantic divergence detection | Without detecting *real* disagreements (vs surface phrasing differences), the debate loop is arbitrary | Med | Core differentiator vs naive majority-vote systems; already designed (sentence-transformers on key_claims) |
| Concession tracking | Without this, there is no auditability of *why* consensus was reached vs why a point remains disputed | Med | `Concession` model with `triggered_by_agent` + `triggered_by_claim` — already in ARCHITECTURE.md |
| Synthesizer / arbiter agent | Without a final integrating step, debate ends without verdict; output is just a list of arguments | Low | Already in agent roster (Synthesizer); standard in MAD literature (judge node) |
| Confidence score on final output | Without a confidence score, the output has no epistemic calibration — the audience cannot know how contested the verdict is | Low | `confidence_score: float` in `DebateReport`; self-reported per agent, aggregated by synthesizer |
| Consensus / disputed point split in final report | The output must separate "what all agents agreed on" from "what remains genuinely contested" | Low | `consensus_points: list[str]` + `disputed_points: list[str]` in `DebateReport` |
| Full reasoning trace stored | Without a trace, the output is unauditable — the demo cannot show *how* the verdict was reached | Med | `reasoning_trace: list[RoundRecord]` + `concession_log` in `DebateReport`; SQLite persistence |
| Live UI progress feedback | Static loading spinner with delayed final output kills demo credibility; audience cannot see "agents debating" | Low–Med | `graph.stream()` + `st.status()` per agent; Streamlit streaming pattern |

**Feature dependencies:**
```
Independent Round 1 → Round 2 Rebuttals (rebuttals only make sense if Round 1 was isolated)
Semantic Divergence Detection → Rebuttal Loop (detection gates whether another round is needed)
Concession Tracking → Reasoning Trace → Consensus/Disputed Split (the split is derived from concession history + final synthesis)
Full Reasoning Trace → SQLite Persistence (trace must be stored to be viewable)
```

---

## Differentiators

Features that distinguish this system from a naive "run 3 LLMs in parallel and average." Not expected by default, but they make the system demonstrably more sophisticated.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Claim-level divergence (not argument-level) | Embedding full arguments produces semantic middling (everything sounds like it's about the same topic). Embedding extracted `key_claims` at claim granularity produces real signal. | Med | Agents output `key_claims: list[str]`; divergence computed on claim pairs, not full text. This is architecturally embedded in ARCHITECTURE.md already. |
| Explicit concession attribution | Logging *which agent's argument* triggered each concession creates an auditable causal chain. "Pessimist conceded point X because Devil's Advocate argued Y." Most systems just show agreement/disagreement. | Med | `Concession.triggered_by_agent` + `triggered_by_claim` fields; structured output from agent itself |
| Diverged-pair reporting | Knowing *which pair* of agents diverges (optimist vs devil, etc.) is more informative than a single divergence score. Shows the fault lines. | Low | `diverged_pairs: list[tuple[str,str]]` in DebateState |
| Max-rounds guard with explicit reason | When the loop exits due to max rounds (not convergence), the output should label this explicitly — "debate was still diverged but reached round limit." This is honesty the audience respects. | Low | `status: "max_rounds"` in DebateState triggers a different verdict tone from the synthesizer |
| Confidence-modulated debate protocol | Research (2601.19921) shows vanilla debate without confidence weighting underperforms majority vote. Agents expressing calibrated self-confidence AND conditioning updates on others' confidence materially improves output quality. | Med | Self-reported `confidence: float` per AgentArgument. Synthesizer should weight concessions by confidence. Needs prompt engineering. |
| Memory masking for erroneous priors | MAD-M2 (2603.20215) shows agents become anchored to bad arguments in their context window. Allowing agents to "mask" prior claims they now consider erroneous (by conceding them) prevents compounding errors over rounds. | Med | Concession mechanism naturally handles this: conceded claims are flagged and the synthesizer can de-weight them. Needs explicit prompt instruction to agents to actively update position. |
| Sycophancy resistance via independent prompting | Research (2604.21564) shows debate triggers 2-3x more sycophancy than direct questioning. Agents in rebuttals must be explicitly instructed to maintain their position unless the counterargument is genuinely compelling. | Low | Prompt engineering: "maintain your position unless you encounter a logically superior argument; do not concede merely to agree" |
| Per-round divergence score in trace | Showing how divergence changes across rounds (Round 1: 0.8, Round 2: 0.5, Round 3: 0.2) gives the audience a quantitative narrative of debate resolution. | Low | Already in DebateState (`divergence_score` updated each round); needs to be stored in RoundRecord |

**Feature dependencies:**
```
Claim-level divergence ← key_claims in AgentArgument (agents must output structured claims)
Confidence-modulated protocol ← confidence: float in AgentArgument + synthesizer weighting logic
Memory masking ← Concession mechanism (conceded = masked from subsequent weighting)
Per-round divergence score ← RoundRecord must store divergence_score at that round
```

---

## Anti-Features

Things to deliberately NOT build in v1. Each has a concrete reason.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Real-time financial data integration | Adds API dependencies, rate limits, data freshness complexity; shifts demo focus from debate mechanics to data pipeline | Use domain-agnostic text input; the debate engine is the demo, not the data |
| More than 4 agents | Research (2601.19921) shows adding agents without diversity just amplifies noise. 4 agents (optimist, pessimist, devil, synthesizer) is the minimum to demonstrate adversarial dynamics with an arbiter. 5+ is harder to explain in a demo. | Keep 4 agents; differentiate via system prompt, not headcount |
| Voting / majority consensus | Majority vote ignores minority views that may be correct. Research (2601.19921, 2604.12196) shows vanilla voting underperforms when initial majority is wrong. The synthesizer with full trace is superior. | Synthesizer-with-trace replaces voting |
| User accounts / authentication | Portfolio demo has one user: the demo audience. Auth adds 1-2 days of work with zero demo value. | No auth; all debates are anonymous in demo mode |
| Production scaling / deployment | Horizontal scaling, load balancers, async workers — none of this is demonstrable or portfolio-differentiating in a debate demo. Premature optimization. | Deploy as single Streamlit process; demo-first |
| Automated external fact-checking / RAG | Progressive RAG (PROClaim, 2603.28488) is a differentiator in claim verification systems. For a general-purpose debate demo, adding RAG shifts complexity budget to retrieval rather than debate mechanics. | Agents reason from their internal knowledge; RAG can be Phase 2 if needed |
| Debate export to PDF/Word | Adding file export complexity before core debate mechanics work is gold-plating. | Show report in UI; markdown rendering is sufficient for portfolio demo |
| Real-time streaming tokens (per-token) | Token-level streaming requires async LangGraph setup and Streamlit async hacks. Node-level streaming (per-agent completion) is sufficient to show "live debate." | `stream_mode="updates"` per node completion is the right granularity |
| Automatic topic detection / classification | Classifying whether a topic is "investment" vs "policy" vs "technical" to adjust agent prompts adds complexity without visible demo value. | Use generic agent prompts; domain-agnostic is a feature, not a limitation |
| Agent memory across debates | Persistent per-agent memory (what each agent "learned" from prior debates) adds stateful complexity with no clear demo payoff. SQLite stores debate history, not agent learning. | Each debate starts fresh; no cross-debate agent state |
| Formal logic / argument graph | Turning debate arguments into formal logical structures (premises → conclusions in a graph) is academically interesting but over-engineers v1. | Structured Pydantic output + key_claims is sufficient formalism |

---

## Feature Dependencies (full map)

```
topic input (UI)
  └─► initialize_node
        └─► dispatch_round1 (fan-out)
              └─► [optimist, pessimist, devil] in parallel (Round 1)
                    └─► key_claims extracted                    ← required for divergence detection
                    └─► confidence: float                       ← required for confidence-modulated protocol
                    └─► concessions: list                       ← required for concession attribution
                          └─► collect_round1 (fan-in)
                                └─► divergence_check
                                      ├─► [divergence_score > threshold] → dispatch_rebuttal
                                      │       └─► [agents see prior_arguments] → [collect_rebuttal] → loop
                                      └─► [converged or max_rounds] → synthesize_node
                                                └─► DebateReport
                                                      ├─► consensus_points          ← requires concession_log
                                                      ├─► disputed_points           ← requires diverged_pairs
                                                      ├─► confidence_score          ← requires per-agent confidence
                                                      ├─► verdict
                                                      └─► reasoning_trace           ← requires RoundRecord history
                                                              └─► save_node (SQLite)
                                                                    └─► UI: final report view
```

Hard dependencies (cannot build B before A):
- Divergence detection requires `key_claims` output from agents
- Concession attribution requires `Concession` model with structured fields
- `disputed_points` in final report requires `diverged_pairs` from divergence_check
- `consensus_points` in final report requires `concession_log` (what was conceded and why)
- SQLite persistence requires `DebateReport` to be fully populated (save last)

---

## MVP Recommendation

Prioritize (Phase 1–3):
1. Round 1 independence + parallel analysis (table stakes, foundational)
2. Semantic divergence detection on key_claims (core differentiator, gates the loop)
3. Multi-round rebuttal loop with max-rounds guard (table stakes)
4. Concession tracking with attribution (differentiator, audit quality)
5. Synthesizer verdict + confidence score + consensus/disputed split (table stakes output)
6. Streamlit live progress feed with `st.status()` per agent (table stakes UX)
7. SQLite persistence of full reasoning trace (table stakes auditability)

Defer (Phase 2+):
- Confidence-modulated debate protocol (medium complexity; needs empirical tuning after v1 debates run)
- Per-round divergence score in trace (easy to add, but only useful if UI shows the trend chart — defer UI feature)
- RAG / external fact-checking (different complexity budget)

Never build in v1:
- Voting/majority consensus (inferior to synthesizer approach)
- More than 4 agents
- Auth, scaling, PDF export, token-streaming, formal logic graphs

---

## Key Research Findings Informing Categories

**On divergence detection:**
Research consistently shows that "agents are converged" ≠ "agents are correct." The conformal social choice paper (2604.07667) found 81.9% of wrong-consensus cases could be intercepted by treating confident agreement as a risk signal, not a quality signal. This confirms: divergence detection must operate on *semantic content of claims*, not surface agreement/disagreement, and the output must explicitly distinguish "converged with high confidence" from "converged at round limit."

**On concession mechanisms:**
The MAD literature (Liang et al., MAD) establishes that agents rarely concede voluntarily without structured prompting. The ARCHITECTURE.md's approach of having agents self-report `concessions` as structured output (with `triggered_by_agent` and `triggered_by_claim`) is the right mechanism: it makes concessions explicit, attributable, and auditable. Research shows agents in unstructured debate drift toward agreement (sycophancy, 2604.21564); structured concession tracking counteracts this by requiring the agent to name *what it is giving up and why*.

**On sycophancy risk:**
Debate triggers 2-3x more sycophancy than direct prompting (2604.21564). This is a build-time risk: if rebuttal prompts don't explicitly instruct agents to resist social pressure, agents will flip positions not because of better arguments but because of tone and persistence. The mitigation is prompt-level: "maintain your position unless the argument is logically superior; do not concede to avoid conflict."

**On output format:**
No academic consensus on a single "standard" output format exists. However, from surveying existing systems (ChatEval, MAD, PROClaim, MediHive, AgentVerse), the consistent output pattern is:
1. A final verdict / answer (the synthesized conclusion)
2. A confidence or agreement score (how settled the debate is)
3. A reasoning trace (what was said, in what order)
4. A divergence or dispute summary (what remained unresolved)

The `DebateReport` in ARCHITECTURE.md directly maps to this: `verdict` + `confidence_score` + `reasoning_trace` + `disputed_points`. This is the right format.

**On UI patterns:**
Streamlit's `st.status()` (state: "running" → "complete") combined with `graph.stream(stream_mode="updates")` per-node is the correct pattern for a portfolio debate demo. Per-token streaming adds async complexity without meaningful demo improvement — the audience cares about "what did Agent X conclude" not watching tokens appear one by one.

**On confidence scoring:**
Self-reported confidence from agents (`confidence: float` in AgentArgument) is the practical approach for v1. Research (2601.19921) on confidence-modulated debate shows that agents expressing *calibrated* confidence (not just "I am 90% confident" as a generic phrase) and conditioning position updates on others' confidence materially improves output. This requires prompt engineering — agents must be instructed to give a numeric confidence and explain why. The synthesizer aggregates these into a final score.

**On what makes debate trustworthy / auditable (confidence: HIGH):**
The consistent pattern across all reviewed literature and systems is: trustworthiness comes from traceability, not just from accuracy. A debate system is trusted when:
1. Every claim can be attributed to a specific agent at a specific round
2. Every concession can be traced to a specific triggering argument
3. The final verdict states what evidence supported it and what evidence was dismissed
4. Residual uncertainty is explicitly surfaced (disputed_points, divergence score)

The `DebateReport` structure in ARCHITECTURE.md satisfies all four. The anti-pattern is a "black box verdict" — a final answer with no path back to the reasoning.

---

## Sources

- Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (arXiv:2305.14325) — MEDIUM confidence (abstract + secondary summary only)
- Liang et al., "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate" (arXiv:2305.19118) — HIGH confidence (full HTML accessed; tit-for-tat mechanism, judge modes, adaptive break documented)
- Chan et al., "ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate" (arXiv:2308.07201) — MEDIUM confidence (abstract verified; referee team structure, multi-judge consensus documented)
- "Demystifying Multi-Agent Debate: The Role of Confidence and Diversity" (arXiv:2601.19921) — HIGH confidence (abstract + key findings: vanilla debate underperforms majority vote; confidence + diversity interventions fix this)
- "Courtroom-Style Multi-Agent Debate with Progressive RAG" / PROClaim (arXiv:2603.28488) — HIGH confidence (plaintiff/defense/judge roles, progressive RAG, multi-judge aggregation documented)
- "Multi-Agent Debate with Memory Masking" / MAD-M2 (arXiv:2603.20215) — HIGH confidence (memory masking mechanism, erroneous prior filtering documented)
- "From Debate to Decision: Conformal Social Choice" (arXiv:2604.07667) — HIGH confidence (81.9% wrong-consensus interception, probability aggregation, safety layer mechanism documented)
- "Measuring Opinion Bias and Sycophancy via LLM-based Coercion" (arXiv:2604.21564) — HIGH confidence (2-3x sycophancy in debate vs direct questioning finding documented; implications for trustworthy systems explicit)
- "Network Effects and Agreement Drift in LLM Debates" (arXiv:2604.11312) — MEDIUM confidence (directional bias / agreement drift finding; structural vs model bias distinction)
- "Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score" (arXiv:2604.12196) — MEDIUM confidence (RCS as replacement for majority voting; geometric semantic center mechanism documented)
- "Rethinking Multi-Agent Intelligence Through Small-World Networks" (arXiv:2512.18094) — MEDIUM confidence (semantic entropy as divergence signal; uncertainty-guided rewiring)
- DiscoUQ (divergence analysis paper, from arXiv search) — MEDIUM confidence (embedding geometry for divergence: cluster distance, dispersion, cohesion)
- "Deliberative Dynamics and Value Alignment in LLM Debates" (arXiv:2510.10002) — HIGH confidence (verdict revision rates: GPT-4.1 0.6-3.1% vs Claude/Gemini 28-41%; deliberation format effect documented)
- Streamlit st.status() official docs — HIGH confidence (live-accessed; running/complete/error states, update() method, nesting warnings)
- ARCHITECTURE.md in this repo — HIGH confidence (data models, graph topology, divergence algorithm, concession models already designed; this file builds on that work)
