# Forcing LLMs to Disagree: An Empirical Study on Multi-Agent Debate as a Defense Against Sycophancy

> This post documents the design and evaluation of a multi-agent debate system built to address a specific failure mode in LLMs: the tendency to produce hedged, non-committal analysis when asked to reason from multiple perspectives. Three findings emerged from benchmarking — each counter-intuitive, each requiring a redesign.

> **Evidence note:** this is an engineering retrospective, not a peer-reviewed study. Results are small-sample, mostly single-run, and partly LLM-judged. Validation-clean calculations and raw-data caveats are maintained in [PAPER.md](PAPER.md) and [results/README.md](results/README.md).

---

## The Problem: Sycophancy in LLMs

Large language models trained via RLHF are optimized to maximize human approval. This produces capable, fluent systems — but also introduces a systematic bias: **models learn to appease, not just to be correct**.

This phenomenon is known as **sycophancy**, and it manifests in two related ways:

**Capitulation under pressure.** When a user challenges a model's response — even when the model is correct — the model tends to concede. Perez et al. (2022) demonstrated that a simple "I don't think that's right" is sufficient to make models abandon accurate positions at rates far above chance.

**Preemptive position avoidance.** Without any external pressure, models proactively hedge to avoid taking a definitive stance:

> "Option A has its advantages in certain contexts, but Option B also offers unique value depending on your specific needs and constraints."

This is not nuance — it is evasion. For decisions that require genuine risk-benefit analysis (technical architecture choices, investment evaluation, strategic assessment), answers of this form are worthless.

This project targets the second manifestation: using structural mechanisms to force distinct, committed positions across multiple LLM agents. As it turns out, the first manifestation also surfaces during agent-to-agent debate — agents capitulate to each other's rebuttals rather than defending their positions. Both problems ultimately required explicit engineering countermeasures.

---

## Why "Just Use Multiple Agents" Doesn't Work

The naive approach is to prompt three agents separately, collect their responses, and synthesize. Two problems make this insufficient:

**Problem 1: Positions collapse under debate pressure.** Once an agent reads a rebuttal from another agent, a personality-based prompt ("you are a pessimistic analyst") provides no anchor for why the position should be maintained. The agent begins hedging: *"You raise a valid point. While I still see risks, I acknowledge the opportunity you described…"* The debate devolves into polite agreement within two rounds.

**Problem 2: Apparent divergence isn't real divergence.** Two agents can produce responses using different vocabulary while expressing essentially the same view. Without a mechanism to detect genuine semantic opposition — not just surface-level topic overlap — a system has no way to determine whether debate is still productive.

These two problems map directly to the two core design decisions in this system.

---

## Design Decision 1: Methodology-Based Personas with PROHIBITION Constraints

### Methodology, Not Personality

The first prompt design assigned personality labels:

```
You are an extremely pessimistic analyst. Evaluate this topic from the most negative possible angle.
```

This fails under pressure. Personality has no internal justification — when confronted with a compelling counter-argument, the model has no structural reason to remain pessimistic. "Because I'm pessimistic" is not an argument.

Replacing personality with **analytical methodology** gives the agent a principled anchor:

```
You are a risk analyst. Apply the following framework:
1. Identify the single most likely failure mode
2. Estimate its probability and impact magnitude
3. Assess whether the projected upside justifies the downside risk
```

Now, when challenged, the agent has a reason to hold its position: *because my methodology requires me to find failure modes, and I haven't found a counter-argument that invalidates this one.* Position maintenance becomes analytically grounded, not merely stubborn.

### PROHIBITION Constraints

Soft guidance fails consistently. Instructions like "avoid hedging language" are observed at the sentence-structure level but routinely violated at the semantic level — the model produces sentences that technically avoid the word "however" while conveying the same concessive meaning.

Hard lexical constraints are substantially more effective:

```
You are PROHIBITED from using the following words or any semantically equivalent expression:
"however", "but", "on the other hand", "it depends", "while X, also Y", "both sides"

Violating this constraint means your analysis has failed.
```

This forces the model to commit at the expression level, not just the intention level.

### The Three Agents

| Agent | Analytical Framework | Reference Role | Core Prohibition |
|-------|---------------------|---------------|-----------------|
| **Optimist** | Map asymmetric upside; enumerate success conditions; assess magnitude of opportunity | Seed-stage VC associate | No risks, caveats, or conditional qualifications |
| **Pessimist** | Identify the most probable failure mode; estimate probability × impact; assess whether upside justifies exposure | Venture debt risk manager | No opportunities, positive signals, or silver linings |
| **Devil's Advocate** | Find the shared assumption both sides take for granted; challenge the frame, not the positions | Philosopher-economist | Do not simply oppose the Optimist; do not align with the Pessimist; attack the premise |

The Devil's Advocate role deserves elaboration. The initial prompt defined it as "challenge the dominant view." In practice, this caused the Devil to side with the Pessimist in nearly every debate — because in most discussions, the optimistic position is the conventional one. The result was a 2-vs-1 dynamic rather than a genuine triangle of perspectives. The fix was to redefine the role as a **frame challenger**: an agent whose job is to identify what both other agents assume to be true and question whether that assumption holds. This seemingly minor prompt change had a measurable effect on position diversity, as discussed in the experiments section.

---

## Design Decision 2: NLI-Based Divergence Detection

### Why Cosine Similarity Fails

The system needs to determine whether agents have genuinely converged — i.e., whether another round of debate is likely to produce new information. The natural approach is to embed each agent's key claims and compute pairwise cosine similarity: high similarity implies convergence.

The first checked-in benchmark exposed this limitation: **10 out of 10 test questions terminated after Round 1**, with divergence scores between 0.097 and 0.258 against the then-used threshold of 0.75.

The root cause is that cosine similarity measures **topical overlap**, not **stance opposition**. The sentences "venture capital is an attractive financing vehicle" and "venture capital is a dangerous financing vehicle" have high cosine similarity — they share the same topic vocabulary. But they represent opposite positions.

A debate detector built on cosine similarity cannot distinguish "we are discussing the same topic" from "we agree on this topic."

### NLI Cross-Encoder as a Drop-in Replacement

Natural Language Inference (NLI) models are trained to classify the semantic relationship between two sentences: **Entailment**, **Neutral**, or **Contradiction**. This is precisely the signal needed.

The implementation uses `cross-encoder/nli-deberta-v3-small` to classify all cross-agent claim pairs, taking the maximum contradiction probability per agent pair as the divergence signal:

```python
from sentence_transformers import CrossEncoder
from scipy.special import softmax

nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

def compute_nli_divergence(claims_a: list[str], claims_b: list[str]) -> float:
    pairs = [(a, b) for a in claims_a for b in claims_b]
    logits = nli_model.predict(pairs)  # columns: contradiction, entailment, neutral
    probabilities = softmax(logits, axis=1)
    return max(row[0] for row in probabilities)
```

The default path evaluates every cross-agent claim pair with the NLI model. It deliberately avoids a cosine fast-path: one nearly identical claim can coexist with another contradictory claim, so maximum cosine similarity is not sufficient evidence of convergence.

In the canonical seven-question NLI run, four debates reached multiple rounds, with mean RTC 2.14 and mean SSS 0.977. The original 0.83–0.86 trajectory remains an illustrative early run, not an aggregate result.

---

## System Architecture

The full debate flow is implemented as a LangGraph `StateGraph` with explicit parallel fan-out via the `Send` API:

```
START
  └─→ initialize_node
        └─→ dispatch_round1          [routing function]
              ├─→ optimist_node ──┐
              ├─→ pessimist_node ─┤  parallel execution
              └─→ devil_node ─────┘
                        └─→ collect_round1    [fan-in, merge via `add` reducer]
                                └─→ divergence_check_node
                                        └─→ route_divergence  [routing function]
                                              ├─→ [diverged] rebuttal round (loop)
                                              └─→ [converged] synthesize_node
                                                              └─→ save_node → END
```

Round 1 enforces **complete cognitive isolation**: each agent receives only the debate topic, not the other agents' positions. This ensures initial stances are independently derived rather than reactions to pre-existing views.

Subsequent rebuttal rounds inject full debate history and concession instructions via compact round summaries.

### Convergence Routing: Four-Guard Logic

The routing function terminates debate when any of four conditions fires, evaluated in order:

```python
def route_divergence(state: DebateState) -> list[Send] | str:
    # Guard 1: Caller cap, followed by the absolute safety cap
    if state["round_num"] >= state["max_rounds"] or state["round_num"] >= 10:
        return "synthesize_stub"

    # Guard 2: detector-owned convergence decision
    if not state["diverged_pairs"]:
        return "synthesize_stub"

    # Guard 3: Score plateau (agents repeating themselves)
    if len(state["round_history"]) >= 2:
        prev = state["round_history"][-2].divergence_score
        curr = state["divergence_score"]
        if abs(prev - curr) < 0.05:
            return "synthesize_stub"

    # Guard 4: No concessions (no substantive position change)
    if len(state["round_history"]) >= 2:
        last_round = state["round_history"][-1]
        total_concessions = sum(len(a.concessions) for a in last_round.arguments)
        if total_concessions == 0:
            return "synthesize_stub"

    # Continue debate
    return build_rebuttal_sends(state)
```

Guards 3 and 4 address a failure mode that divergence-threshold-only systems miss: agents can maintain high divergence scores while making no actual argumentative progress — simply restating their positions in different words. If neither the score is moving nor any agent is conceding ground, continued debate is unproductive.

---

## Experiments and Results

### Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **PDS** (Position Diversity Score) | Mean pairwise vector distance across all agent claim embeddings — higher means more genuinely distinct positions |
| **HR** (Hedge Ratio) | Frequency of hedge markers ("however", "it depends", "both", "while") per 100 tokens |
| **SSS** (Stance Stability Score) | Cosine consistency of each agent's position embedding across rounds — higher means the agent maintained its core stance under rebuttal pressure |
| **RTC** (Rounds to Convergence) | Number of debate rounds before a termination guard fires |

Test set: 10 questions spanning business, technology, and policy domains. Each system variant run once per question.

---

### Finding A: A 28% Lower Mean Hedge Ratio in This Sample

| System | HR | Interpretation |
|--------|----|---------------|
| Multi-agent (this system) | **0.0093** | Committed, non-hedged positions |
| Single-LLM baseline | 0.0129 | Characteristic hedging — "both sides have merit" |

Across these 10 questions, the observed mean was about 28% lower. The checked-in results do not isolate PROHIBITION from every other system difference, so this is descriptive rather than a causal claim.

---

### Finding B: The PDS Paradox and the Devil's Advocate Fix

Initial multi-agent PDS was *lower* than the single-LLM baseline — the opposite of the design goal:

| System | PDS |
|--------|-----|
| Multi-agent (original Devil prompt) | 0.1707 |
| Single-LLM baseline | 0.2160 |
| **Multi-agent (revised Devil prompt)** | **0.2242** |

The root cause was the Devil's Advocate prompt. "Challenge the dominant view" reliably caused the Devil to align with the Pessimist — in most debates, the optimistic position is the conventional one, making the Pessimist's critique already the contrarian stance. The Devil then reinforced that critique rather than introducing a third perspective. The result: a 2-vs-1 configuration with lower aggregate diversity than a single model generating balanced output.

Redefining the Devil as a **frame challenger** — whose job is to identify the shared assumptions both other agents accept and question them — restored the triangular structure. PDS exceeded the baseline only after this revision.

This finding illustrates a broader design principle for multi-agent systems: **agent role definitions must be specified relative to the system's equilibrium dynamics, not in isolation**. "Be contrarian" is not a stable role definition when the system's natural attractor already contains a contrarian agent.

---

### Finding C: Cosine Detection vs. NLI Detection

| System | Round 1 Divergence | Mean RTC | SSS |
|--------|--------------------|----------|-----|
| Cosine detection | 0.097 – 0.258 | 1.0 | 1.000 (trivial) |
| **NLI detection (canonical n=7)** | detector-specific | **2.14** | **0.977** |

With cosine-based detection, 100% of debates terminated after Round 1 — the system never ran a multi-round debate. The SSS of 1.000 reflects this: stance stability is trivially perfect when there is only one round.

In the canonical seven-question NLI run, 4/7 debates reached multiple rounds, mean RTC was 2.14, and mean SSS was 0.977. One illustrative three-round question had SSS 0.883.

The practical implication: a debate system's behavior is fundamentally shaped by how it measures divergence. Cosine similarity produces a system that always converges immediately; NLI produces one that debates substantively.

---

## Part 2: The Limits of Uniform PROHIBITION

The three findings above validated the base system. But a follow-up question emerged during ablation testing: should PROHIBITION apply uniformly to every question?

Consider two questions:

- *"Should AI development be halted for safety reasons?"*
- *"Should a startup hire specialists or generalists?"*

The first is a values conflict. Reasonable people with different values reach different conclusions, and neither is objectively wrong. Committed advocacy from opposed value stances is the correct analytical mode — forcing agents to say "the other side has valid points" defeats the purpose.

The second has no universal answer. Whether to hire specialists or generalists genuinely depends on team size, product stage, and growth phase. Forcing agents into unconditional positions produces analytically inferior output — the correct answer *is* "it depends on which specific conditions apply."

PROHIBITION that works for the first question damages output quality on the second. The system was applying a single constraint level to questions with fundamentally different epistemic requirements.

---

## Design Decision 3: Adaptive PROHIBITION

### A Three-Class Question Taxonomy

Questions split into three types based on what kind of disagreement they require:

**Values-based:** The disagreement is fundamentally about values or ethics. Neither side is objectively wrong. Examples: *"Should AI development be halted?"*, *"Is capitalism compatible with climate action?"* → Full PROHIBITION appropriate.

**Binary:** One answer is likely better based on evidence or widely accepted principles, though reasonable people can disagree. Examples: *"Should startups prioritize growth over profitability?"*, *"Is 'move fast and break things' sound?"* → Moderate constraint: require a directional recommendation, but permit conditional framing.

**Context-dependent:** The correct answer genuinely depends on specific circumstances. "It depends on your situation" is a legitimately correct response. Examples: *"Should a startup hire specialists or generalists?"*, *"Microservices or monolith?"* → No PROHIBITION: agents map the conditions under which each approach works, rather than taking unconditional sides.

### The LLM Classifier

Question type is determined at inference time by a structured-output LLM classifier:

```python
class QuestionClassification(BaseModel):
    question_type: str  # "values_based" | "binary" | "context_dependent"
    confidence: str     # "high" | "medium" | "low"
    reasoning: str      # one-sentence explanation

_CLASSIFIER_SYSTEM = """
Classify the question into exactly one of three types.

  values_based:       Fundamental VALUES or ETHICS disagreement. Neither side
                      is objectively wrong.
  binary:             One answer is LIKELY BETTER based on evidence or analysis.
  context_dependent:  Correct answer GENUINELY DEPENDS on specific circumstances.

When in doubt between binary and context_dependent: if the question has a
recognizable 'default' right answer in most cases, it's binary.
"""
```

The classifier uses `with_structured_output` to enforce valid output, with three retries and a fallback to `"binary"` on failure.

### Three-Level PROHIBITION

Each question type receives calibrated agent prompts:

**Level 1 — Full (values-based):** Existing prompts unchanged. Hard lexical bans. Agents cannot acknowledge the opposing view has merit.

**Level 2 — Moderate (binary):** Word bans removed. Agents must end with a committed directional recommendation: *"RECOMMENDATION: Yes — because [one specific, falsifiable reason]"*. No open questions, no hedges.

**Level 3 — Off (context-dependent):** Agents become scenario analysts rather than advocates. Their mandate is condition mapping, not position taking:

```
# Optimist → "Scenario A Analyst"
Your position MUST take the form:
"This approach is optimal WHEN [specific condition set] because [causal mechanism]."

# Pessimist → "Scenario B Analyst"
"This approach fails WHEN [specific condition set] because [causal mechanism]."

# Devil's Advocate → "Variable Identifier"
"The decision hinges on [specific variable] — here is how to measure it: [method]"
```

PROHIBITION is not a binary on/off switch. It is a continuous spectrum mapped to the question's epistemic requirements.

---

### Finding D: One Judge Assigned Equal False-Certainty Rates

A natural concern: does forcing agents to commit cause them to make overconfident claims while ignoring obvious counterevidence — "false certainty"?

Every agent position across 7 test questions was scored by an independent judge on a 1–5 false certainty scale. The result:

| System | false_certainty | appropriate_hedge | role_appropriate_commitment |
|--------|----------------|-------------------|-----------------------------|
| `full_system` | 3/21 (14.3%) | 2/21 (9.5%) | 16/21 (76.2%) |
| `single_llm` | 3/21 (14.3%) | 6/21 (28.6%) | 12/21 (57.1%) |

One Qwen judge assigned identical false-certainty rates (14.3%) and more `appropriate_hedge` labels to `single_llm`. This suggests a useful hypothesis, but it does not establish that forced commitment is calibration-safe.

---

### Finding E: Historical Key-Factor Mention Exercise

A 10-case historical exercise used an LLM judge to score whether each output mentioned a predefined key factor.

| System | Partial-or-better key-factor mention rate (n=10) |
|--------|------------------------------|
| `full_system` | 0.40 |
| `single_llm` | 0.60 |
| `adaptive_prohibition` | **0.60** |

No response earned the rubric's highest score for clearly identifying the factor; all counted cases were partial or tangential mentions. The 40%/60% pattern is exploratory, not decision accuracy.

---

### Finding F: Question Type Determines How Much Adaptive Gains

Matched, validation-clean subsets exclude any question where either system emitted a sentinel:

| Question type | full_system focus | adaptive focus | Δ | n |
|---------------|------------------|----------------|---|---|
| binary | 2.750 | **3.125** | +13.6% | 8 |
| values-based | **3.333** | **3.333** | 0.0% | 9 |
| **context-dependent** | 2.000 | **3.719** | **+85.9%** | 16 |

Focus score = mean of type-specific focus dimensions (binary: analytical_depth + claim_specificity; values: perspective_diversity + analytical_depth; context: claim_specificity + practical_utility).

The values-based subset tied, while the adaptive sample had higher mean focus scores in the other two subsets. Because each question was generated once and evaluated by one judge, these differences motivate replication rather than validate the design.

The classifier routed many human-labeled binary questions to `context_dependent`. Routing and prompt choice changed together, so the experiment cannot identify which factor caused the observed difference.

---

## Auditability: The Confidence Score and Concession Log

### Formula-Derived Confidence Score

The final report includes a confidence score computed entirely in Python — it never appears in any LLM prompt or structured output schema:

```python
def compute_confidence(round_history: list[RoundRecord], round_num: int) -> float:
    max_divergence = max(r.divergence_score for r in round_history)
    round_adjustment = {1: 1.0, 2: 0.9, 3: 0.8}.get(round_num, 0.8)
    return (1.0 - max_divergence) * round_adjustment
```

Two-layer semantics: `(1 - max_divergence)` penalizes topics where agents never genuinely agreed; `round_adjustment` penalizes debates that required more rounds to converge (more rounds implies more contested ground). Early versions delegated confidence scoring to the LLM, which consistently produced scores in the 0.7–0.9 range regardless of actual debate dynamics. The formula replaced model-generated scores with a quantity that is directly computed from observable debate state.

### Attribution-Aware Concession Tracking

Every position change is recorded with full provenance:

```python
class Concession(BaseModel):
    conceded_point: str           # The position being yielded
    triggered_by_agent: str       # Which opponent's role caused the concession
    triggered_by_claim: str       # The specific claim text that prompted it
    rationale: str                # One-sentence explanation
```

This enables inspection of the full reasoning chain: not just what the final consensus is, but which argument moved which agent, and why.

---

## Summary

Six findings shaped this system's final design:

1. **Hard constraints changed hedging behavior in the checked-in sample.** The current data does not isolate a publication-grade causal effect.

2. **Role definitions interact with system dynamics.** The Devil's Advocate prompt produced a 2-vs-1 configuration rather than a triangle — not because the prompt was poorly written in isolation, but because it interacted with the system's existing equilibrium in an unintended way. Role definitions in multi-agent systems must be validated against actual system behavior, not evaluated in isolation.

3. **The divergence metric determines whether the system debates at all.** Cosine similarity and NLI cross-encoders measure fundamentally different things. For stance detection, the choice is not a parameter to tune — it determines whether the system ever runs more than one round.

4. **Context-dependent questions are a promising target for adaptive prompts.** Validation-clean single-run samples favored condition mapping, but require replication.

5. **Routing and constraint level must be ablated separately.** The current experiment changes both together and cannot isolate their effects.

6. **Forced commitment may reduce legitimate uncertainty signaling.** A single judge assigned equal false-certainty rates in this sample, which is not enough to establish calibration safety.

Each finding followed the same structure: the system produced unexpected output, diagnosis revealed a specific design assumption that failed under realistic conditions, and the fix required a principled redesign rather than a parameter adjustment. This pattern — surprising behavior → root-cause diagnosis → principled fix — is the most transferable lesson from this project.

---

*Full results and methodology: [PAPER.md](PAPER.md) · Code: [github.com/xyma2003/multi-agent-debate](https://github.com/xyma2003/multi-agent-debate)*

---

## Resources

- Tech stack: LangGraph · multi-backend LLM APIs · BAAI/bge-small-en-v1.5 · DeBERTa NLI cross-encoder · SQLite · Streamlit
- Experiment data and validation notes: [`results/README.md`](results/README.md)
- Reference: Perez et al., "Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models," 2022

---

*Questions and discussion welcome.*
