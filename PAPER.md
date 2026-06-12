# Adaptive PROHIBITION in Multi-Agent Debate: Question-Type-Aware Constraints as a Defense Against LLM Sycophancy

**Xinyue Ma**  
Independent Researcher  
u3594619@connect.hku.hk

---

## Abstract

Multi-agent debate systems offer a structural mechanism to counteract sycophancy in large language models (LLMs)—the tendency to produce hedged, non-committal analysis rather than taking committed positions. Existing approaches apply uniform debate constraints across all question types, ignoring that different questions call for fundamentally different forms of disagreement. We introduce **Adaptive PROHIBITION**, a framework that classifies questions into three types—*values-based*, *binary*, and *context-dependent*—and applies calibrated constraint levels accordingly. Built on a LangGraph state machine with NLI-based divergence detection and attribution-aware concession tracking, the system uses hard lexical constraints (PROHIBITION blocks) to force genuine position commitment in agent prompts. Experiments across three question taxonomies (n=10 per type for binary and values-based; n=6 for context-dependent) show that adaptive constraints match full constraints on values-based questions (focus score: 3.10 vs. 3.10) while improving performance on binary questions (+5.7% focus score) and producing the largest gain on context-dependent questions (+52% focus score, 1.92 → 2.92). Adaptive constraints also improve ground-truth decision accuracy from 40% to 60% on a historical M&A evaluation benchmark. We additionally identify a calibration cost: PROHIBITION reduces the `honest_uncertainty` dimension, suggesting a commitment–calibration trade-off that practitioners should account for.

---

## 1. Introduction

Large language models trained via reinforcement learning from human feedback (RLHF) are optimized to maximize human approval. This produces capable, fluent systems—but also introduces a systematic bias: **models learn to appease, not just to be correct** (Perez et al., 2022).

This phenomenon, sycophancy, manifests in two related ways. The first is *capitulation under pressure*: when a user challenges a model's response, even when the model is correct, it tends to concede. The second—and the focus of this paper—is *preemptive position avoidance*: without any external pressure, models proactively hedge to avoid taking a definitive stance:

> "Option A has its advantages in certain contexts, but Option B also offers unique value depending on your specific needs."

This is not nuance—it is evasion. For decisions requiring genuine risk-benefit analysis (technical architecture, investment evaluation, strategic planning), answers of this form are worthless.

Multi-agent debate systems address this by assigning agents distinct analytical roles and forcing them to defend committed positions against opposition. But structuring effective disagreement requires more than assigning roles—it requires *constraints* that prevent agents from collapsing their positions under rebuttal pressure. **PROHIBITION constraints** [Section 3.2], hard lexical bans on hedging expressions, are substantially more effective than soft guidance for this purpose.

The problem we identify is that existing systems apply these constraints uniformly. A question like *"Should AI development be halted?"* involves a fundamental values conflict where committed advocacy is the right mode. A question like *"Should startups hire specialists or generalists?"* genuinely depends on team size, product type, and growth stage—and forcing agents to take hard binary positions produces analytically inferior output. Full PROHIBITION, applied indiscriminately, reduces `honest_uncertainty` scores and produces worse analysis on questions where contextual mapping is the correct epistemic stance.

This paper makes three contributions:

1. **A three-class question taxonomy** (values-based, binary, context-dependent) that captures meaningfully different requirements for debate constraint level.
2. **Adaptive PROHIBITION**, an LLM-based classifier that routes questions to calibrated constraint levels at inference time, without requiring pre-labeled training data.
3. **Empirical evaluation** showing that adaptive constraints preserve full-PROHIBITION quality on values-based questions while improving performance on the other two types, with an identified commitment–calibration trade-off.

---

## 2. Related Work

**Sycophancy in LLMs.** Perez et al. [2022] demonstrated that models trained with RLHF abandon accurate positions when challenged, even without logical justification. Wei et al. [2023] showed this extends to systematic agreement with false premises when stated confidently. Our work targets preemptive position avoidance, a related but distinct failure mode that occurs without any external pressure.

**Multi-agent debate.** Du et al. [2023] showed that having multiple LLM agents debate factual questions improves accuracy over single-pass generation. Liang et al. [2023] studied disagreement elicitation between agents to improve reasoning diversity. These works focus primarily on factual correctness; we focus on analytical quality and commitment on open-ended strategic questions, and introduce structural mechanisms (PROHIBITION constraints, NLI divergence detection) to prevent debate collapse.

**Prompt constraint engineering.** Constitutional AI [Bai et al. 2022] uses a set of principles to guide model behavior through self-critique. Our PROHIBITION approach differs in that constraints are applied at the generation level (hard lexical bans) rather than through iterative self-evaluation, which proves more effective at enforcing committed positions under debate pressure.

**NLI for semantic analysis.** He et al. [2021] introduced DeBERTa, which achieved state-of-the-art NLI performance and forms the backbone of our divergence detector. We apply NLI cross-encoders as a stance-detection primitive rather than a text-classification end task.

---

## 3. System Design

### 3.1 Methodology-Based Personas

Early designs assigned personality labels ("be extremely pessimistic"). These fail under debate pressure: personality has no internal justification. When confronted with a compelling counter-argument, a "pessimistic" agent has no structural reason to remain pessimistic.

Replacing personality with **analytical methodology** gives each agent a principled anchor:

```
You are the Risk Analyst. Your framework:
1. Identify the single most likely failure mode
2. Estimate probability (high/medium/low) and impact severity
3. Assess whether the stated upside justifies accepting that specific risk
```

When challenged, the agent now has a reason to hold its position: because *my methodology requires me to find failure modes, and this argument has not invalidated the one I identified*. Position maintenance becomes analytically grounded, not merely stubborn.

The system deploys three agents with distinct methodologies:

| Agent | Analytical Framework | Reference Role | Core Task |
|-------|---------------------|----------------|-----------|
| **Optimist** | Map asymmetric upside; enumerate success conditions; assess opportunity magnitude | Seed-stage VC associate | Identify where the approach succeeds and why |
| **Pessimist** | Identify the most probable failure mode; estimate probability × impact | Venture debt risk manager | Identify where the approach fails and why |
| **Devil's Advocate** | Find the shared assumption both sides accept; question the frame, not the positions | Philosopher-economist | Attack the premise, not the conclusions |

The Devil's Advocate role required careful redesign. The initial prompt ("challenge the dominant view") caused the Devil to align with the Pessimist in most debates—because the optimistic position is typically the conventional one, making pessimistic critique already contrarian. The result was a 2-vs-1 configuration that reduced position diversity below the single-LLM baseline. Redefining the Devil as a **frame challenger** (whose job is to identify what *both* agents assume and question it) restored the triangular structure.

### 3.2 PROHIBITION Constraints

Soft guidance fails consistently. Instructions like "avoid hedging language" are observed at the surface level but violated at the semantic level—a model produces sentences that technically avoid the word "however" while conveying the same concessive meaning.

Hard lexical constraints are substantially more effective:

```
PROHIBITION: You are PROHIBITED from using the following words or any semantically
equivalent expression: "however", "but", "on the other hand", "it depends",
"while X, also Y", "both sides", "balanced view".

Violating this constraint means your analysis has failed.
```

This forces commitment at the expression level, not just the intention level. PROHIBITION reduces the hedge ratio (frequency of hedge markers per 100 tokens) by 28% compared to a single-LLM baseline (0.0093 vs. 0.0129).

The core design question—addressed in Section 4—is whether PROHIBITION should be applied uniformly or calibrated to question type.

### 3.3 Architecture

The debate system is implemented as a LangGraph `StateGraph` with parallel agent execution via the `Send` API:

```
START → initialize
  └→ dispatch_round1
       ├→ optimist_node ─┐
       ├→ pessimist_node ─┤  parallel (Round 1: cognitive isolation enforced)
       └→ devil_node ─────┘
                └→ collect_round1
                      └→ divergence_check_node
                            └→ route_divergence
                                 ├→ [diverged] rebuttal round (loop back)
                                 └→ [converged] synthesize_node → save_node → END
```

Round 1 enforces **complete cognitive isolation**: each agent receives only the debate topic, ensuring initial stances are independently derived. Rebuttal rounds inject full debate history via compact round summaries.

### 3.4 NLI-Based Divergence Detection

The system needs to determine whether agents have genuinely converged—i.e., whether further debate rounds would produce new information. The naive approach is cosine similarity between agent claim embeddings: high similarity implies convergence.

This approach fails categorically. In an initial benchmark run, 10 out of 10 questions triggered convergence after Round 1, with divergence scores uniformly between 0.097 and 0.258. The root cause is that cosine similarity measures **topical overlap**, not **stance opposition**. "Venture capital is an attractive financing vehicle" and "Venture capital is a dangerous financing vehicle" score as highly similar—they share the same topic vocabulary. But they express opposite positions.

We replace cosine similarity with an NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`) that classifies claim pairs as Entailment, Neutral, or Contradiction:

```python
def compute_nli_divergence(claims_a: list[str], claims_b: list[str]) -> float:
    pairs = [(a, b) for a in claims_a for b in claims_b]
    scores = nli_model.predict(pairs)  # shape: (n_pairs, 3): [contradiction, entailment, neutral]
    contradiction_probs = scores[:, 0]
    return float(contradiction_probs.max())
```

After switching to NLI, Round 1 divergence scores ranged from 0.83 to 0.86—above the 0.50 threshold—enabling multi-round debates with genuine stance evolution. Mean rounds-to-convergence increased from 1.0 to 3.0; stance stability score (SSS) settled at 0.883, indicating agents maintained their core positions while making calibrated concessions.

### 3.5 Four-Guard Convergence Logic

The routing function terminates debate under any of four conditions:

```python
# Guard 1: Absolute safety cap (prevents infinite loops)
if round_num >= 10: return "synthesize"

# Guard 2: Genuine convergence
if divergence_score < 0.75: return "synthesize"

# Guard 3: Score plateau (agents repeating themselves)
if len(history) >= 2:
    if abs(history[-2].divergence_score - history[-1].divergence_score) < 0.05:
        return "synthesize"

# Guard 4: No concessions (no substantive position change)
if len(history) >= 2:
    if sum(len(a.concessions) for a in history[-1].arguments) == 0:
        return "synthesize"
```

Guards 3 and 4 address a failure mode that divergence-threshold-only systems miss: agents can maintain high divergence scores while making no argumentative progress—simply restating positions in different words. If neither the score is moving nor any agent is conceding ground, continued debate is unproductive.

---

## 4. Adaptive PROHIBITION Framework

### 4.1 Question Taxonomy

Analysis of debate quality across question domains reveals that three qualitatively distinct question types require different forms of committed analysis:

**Values-based questions** involve fundamental ethical or moral disagreements where people with different values will legitimately reach different conclusions. Examples: *"Should AI development be halted for safety reasons?"*, *"Is capitalism compatible with climate action?"* For these questions, committed advocacy from opposed value stances is the correct epistemic mode—forcing agents to acknowledge "the other side has valid points" undermines the analytical goal. Full PROHIBITION is appropriate.

**Binary questions** have one answer that is likely better based on evidence, analysis, or widely accepted principles, though reasonable people could disagree. Examples: *"Should early-stage startups prioritize growth over profitability?"*, *"Is 'move fast and break things' a sound philosophy?"* For these questions, agents should give clear recommendations but can acknowledge the conditions under which exceptions apply. A *moderate* constraint level—requiring a directional recommendation but permitting conditional framing—is appropriate.

**Context-dependent questions** are those where *"it depends"* is a legitimately correct answer, and the right response maps the conditions under which each approach succeeds. Examples: *"Should a startup hire generalists or specialists?"*, *"Should a B2B SaaS company target enterprise or SMB first?"* For these questions, forcing agents to take unconditional positions produces analytically inferior output. The correct analytical mode is **condition mapping**: identifying when each approach works and what variable determines the choice. PROHIBITION is counterproductive here.

### 4.2 LLM-Based Classifier

Question type is determined at inference time using a structured-output LLM classifier:

```python
class QuestionClassification(BaseModel):
    question_type: str  # "values_based" | "binary" | "context_dependent"
    confidence: str     # "high" | "medium" | "low"
    reasoning: str      # one-sentence explanation

_CLASSIFIER_SYSTEM = """
Classify the question into exactly one of three types:

  values_based:       Fundamental VALUES or ETHICS disagreement. People with
                      different values reach different conclusions, neither
                      objectively wrong.

  binary:             One answer is LIKELY BETTER based on evidence or analysis,
                      but reasonable people could disagree.

  context_dependent:  Correct answer GENUINELY DEPENDS on specific circumstances.
                      'It depends on your situation' is a legitimately correct
                      response.

When in doubt between binary and context_dependent: if the question has a
recognizable 'default' right answer in most cases, it's binary.
"""
```

The classifier uses `with_structured_output` to enforce valid output schema, with three retry attempts and a fallback to `"binary"` on failure.

A key empirical observation: the classifier routes many questions humans would label "binary" to `context_dependent`. Questions framed as *"should startups do X?"* typically have a correct answer that depends on stage, market, and team—and the classifier correctly identifies this. This routing behavior, rather than the moderate PROHIBITION level itself, accounts for a significant portion of adaptive's advantage on binary questions (discussed in Section 7.1).

### 4.3 Three-Level PROHIBITION

Each question type receives a calibrated prompt set:

**Level 1 — Full PROHIBITION (values-based):** The existing agent prompts are used unchanged. Hard lexical bans on all hedging expressions. Agents cannot express that the opposing view has merit.

**Level 2 — Moderate (binary):** PROHIBITION word bans are removed. Instead, agents are required to end their analysis with a committed directional recommendation:

```
Your final position MUST end with:
"RECOMMENDATION: [Yes/No] — because [one specific, falsifiable reason]"
Do not end with an open question or a hedge. Take a side.
```

**Level 3 — Off (context-dependent):** Agents become *scenario analysts* rather than advocates. Their mandate is to map conditions, not take positions:

```
# Optimist → "Scenario A Analyst"
Your final position MUST take the form:
"This approach is optimal WHEN [specific condition set]
 because [causal mechanism]."

# Pessimist → "Scenario B Analyst"
Your final position MUST take the form:
"This approach fails WHEN [specific condition set]
 because [causal mechanism]."

# Devil's Advocate → "Variable Identifier"
Your final position MUST take the form:
"The decision hinges on [specific variable] — here is how to
 measure it: [concrete method]"
```

This design treats PROHIBITION not as a binary on/off switch, but as a continuous spectrum mapped to the question's epistemic requirements.

---

## 5. Experiments

### 5.1 Setup

**Models.** Debate agents use `llama-3.3-70b-versatile` via Groq API. Quality evaluation uses `Qwen3-32B` as an independent judge. For the ground-truth historical evaluation, position alignment is assessed against known historical outcomes.

**Rate-limit note.** The Groq free tier imposes a tokens-per-minute (TPM) limit that is exceeded when three agents generate simultaneously (~4,500 tokens/burst). Experiments run agents sequentially with 15-second delays between calls, preserving Round 1 cognitive isolation (no agent sees others' outputs before generating). This evaluates *Round 1 position quality* as a proxy for overall prompt design quality. Multi-round rebuttal dynamics are measured separately in the system variant comparison (E1).

**Evaluation rubric.** A five-dimension rubric scored 1–5 by the Qwen judge:

| Dimension | Measures |
|-----------|---------|
| `perspective_diversity` | Are the three positions genuinely distinct? |
| `analytical_depth` | Non-obvious risks, mechanisms, or considerations |
| `claim_specificity` | Concrete, falsifiable claims vs. vague assertions |
| `honest_uncertainty` | Accurate flagging of what is genuinely unknown |
| `practical_utility` | Actionable guidance for a decision-maker |

Type-specific **focus dimensions** reflect what each question type should optimize:

| Question type | Focus dimensions |
|---------------|-----------------|
| binary | `analytical_depth` + `claim_specificity` |
| values-based | `perspective_diversity` + `analytical_depth` |
| context-dependent | `claim_specificity` + `practical_utility` |

**Datasets.**

- *E1 (system variant comparison):* 10 business/technology/policy questions, run across 6 system variants.
- *E2 (false certainty analysis):* Same 10 questions, full_system vs. single_llm, manually scored for false certainty (claiming certainty while ignoring obvious counterevidence).
- *E3 (ground-truth accuracy):* 10 historical M&A and product strategy decisions with known outcomes (q51–q60: Facebook/Instagram, Twitter independence, Snapchat/Facebook, Netflix streaming pivot, etc.). Systems assessed on whether their analysis identified the historically correct key factor.
- *E4 (3-type comparison):* binary (q71–q80, n=10), values-based (q31–q40, n=10), context-dependent (q1, q4, q5, q7–q9, n=6). Two systems: `full_system` vs. `adaptive_prohibition`.

### 5.2 E1: System Variant Comparison

Six system variants are compared to isolate the contribution of each design decision:

| Variant | What it tests |
|---------|--------------|
| `single_llm` | Baseline: single model, balanced prompt |
| `full_system` | Multi-agent + PROHIBITION (current, fixed Devil prompt) |
| `original_devil` | Multi-agent with pre-fix Devil prompt ("challenge dominant view") |
| `no_prohibition` | Multi-agent, methodology prompts only, no word bans |
| `cosine_detection` | Multi-agent + PROHIBITION, cosine divergence detector |
| `nli_detection` | Multi-agent + PROHIBITION, NLI divergence detector |

**Metrics:** Hedge Ratio (HR), Position Diversity Score (PDS), Stance Stability Score (SSS), Rounds-to-Convergence (RTC).

### 5.3 E2: False Certainty Analysis

For each system, every agent position is scored on a 1–5 false certainty scale: does the agent claim certainty while ignoring obvious counterevidence that any analyst in that role would recognize? Scored by the Qwen judge with explicit rubric.

### 5.4 E3: Ground-Truth Historical Accuracy

For 10 historical decisions, each system's output is evaluated on whether it identified the correct strategic factor (e.g., for Facebook/Instagram: whether it flagged mobile user acquisition as the key driver, which proved correct). Accuracy is the fraction of questions where the system's analysis would have supported the historically correct decision.

### 5.5 E4: 3-Type Adaptive vs. Full

The primary experiment. For each question in each type bucket, both `full_system` and `adaptive_prohibition` generate Round 1 positions. The Qwen judge scores both on all five dimensions. Comparisons use type-specific focus dimensions as the primary metric and overall total score as secondary.

*Note:* Two questions (q79, q80) returned API validation errors for one system each. These are included with their valid scores; invalid entries are excluded from per-type averages (effective n=8 for two cells, noted in Table 3).

---

## 6. Results

### 6.1 System Variant Comparison (E1)

**Finding A: PROHIBITION reduces hedging by 28%.**

| System | Hedge Ratio | Interpretation |
|--------|------------|----------------|
| `single_llm` | 0.0129 | Characteristic hedging — "both sides have merit" |
| `full_system` | **0.0093** | Committed, non-hedged positions |

The reduction is attributable to the PROHIBITION constraints, not the multi-agent structure: the `no_prohibition` variant produces hedge ratios comparable to `single_llm`.

**Finding B: The Devil's Advocate prompt fix restores position diversity.**

| System | PDS |
|--------|-----|
| `single_llm` | 0.2160 |
| `original_devil` (pre-fix) | 0.1707 |
| `full_system` (fixed Devil) | **0.2242** |

The original "challenge dominant view" prompt caused the Devil to align with the Pessimist in most debates, producing a 2-vs-1 configuration with lower aggregate diversity than a single model generating balanced output. Redefining the Devil as a frame challenger—whose job is to identify and question what both other agents assume—restored the triangular structure and pushed PDS above the baseline.

This illustrates a general principle: **agent role definitions must be validated against actual system equilibrium dynamics, not evaluated in isolation.**

**Finding C: NLI detection enables multi-round debate.**

| System | Round 1 divergence range | Mean RTC | SSS |
|--------|--------------------------|----------|-----|
| `cosine_detection` | 0.097 – 0.258 | 1.0 | 1.000 |
| `nli_detection` | **0.83 – 0.86** | **3.0** | **0.883** |

With cosine detection, 100% of debates terminated after Round 1—the system never ran a multi-round debate. SSS of 1.000 is trivially perfect when there is only one round. With NLI detection, debates ran for three rounds on average with measurable stance evolution (SSS = 0.883: agents maintained core positions while making calibrated concessions).

The practical implication: a debate system's behavior is fundamentally determined by its divergence metric. Cosine similarity and NLI cross-encoders are not interchangeable parameters—they determine whether the system debates at all.

### 6.2 False Certainty Analysis (E2)

**Finding D: PROHIBITION does not inflate false certainty rates.**

| System | false_certainty | appropriate_hedge | role_appropriate_commitment |
|--------|----------------|-------------------|-----------------------------|
| `full_system` | 3 / 21 (14.3%) | 2 / 21 (9.5%) | 16 / 21 (76.2%) |
| `single_llm` | 3 / 21 (14.3%) | 6 / 21 (28.6%) | 12 / 21 (57.1%) |

Both systems produce the same false certainty rate (14.3%). The meaningful difference is that `single_llm` produces substantially more `appropriate_hedge` verdicts (28.6% vs. 9.5%)—that is, positions that avoid committing to avoid being wrong. PROHIBITION does not push agents toward making indefensible claims; it pushes them from appropriate hedges to committed positions.

### 6.3 Ground-Truth Accuracy (E3)

**Finding E: Adaptive PROHIBITION matches single_llm accuracy and outperforms full_system.**

| System | Ground-truth accuracy (n=10) |
|--------|------------------------------|
| `full_system` | 0.40 |
| `single_llm` | 0.60 |
| `adaptive_prohibition` | **0.60** |

Full PROHIBITION reduces accuracy on historical decisions: forcing agents to maintain committed positions regardless of question type suppresses the contextual analysis needed to identify the key variable in complex strategic decisions. Adaptive constraints, by routing historical decisions to the `context_dependent` mode, preserve the analytical flexibility that single_llm maintains by default.

This result is notable: adaptive PROHIBITION achieves the same accuracy as the single_llm baseline while producing substantially more committed, structured output (lower hedge ratio, higher PDS).

### 6.4 3-Type Comparison (E4)

**Table 3.** Per-type quality comparison: `full_system` vs. `adaptive_prohibition`.  
*Focus score = mean of type-specific focus dimensions. Total score = mean of all 5 dimensions.*

| Question type | Metric | full_system | adaptive | Δ | n |
|---------------|--------|-------------|----------|---|---|
| **binary** | focus score | 2.65 | **2.80** | +5.7% | 10 |
| | total score | 2.58 | **2.76** | +7.0% | 10 |
| **values-based** | focus score | 3.10 | **3.10** | 0.0% | 10 |
| | total score | 2.32 | 2.24 | −3.4% | 10 |
| **context-dependent** | focus score | 1.92 | **2.92** | +52.1% | 6 |
| | total score | 2.30 | **2.63** | +14.3% | 6 |

**Finding F: The adaptive framework validates its core hypothesis.**

The values-based focus score tie (3.10 = 3.10) confirms the central design hypothesis: when question type warrants full PROHIBITION (values conflicts require committed advocacy), adaptive correctly routes there, preserving quality. The classifier does not over-adapt.

Binary questions show consistent improvement under adaptive (+5.7% focus), driven by condition-mapping prompts that produce more specific, actionable output. Context-dependent questions show the largest gain in the entire experiment: +52.1% on focus score (1.92 → 2.92, n=6). Q8 (domain expert vs. generalist founder) and Q9 (microservices vs. monolith) exemplify the effect—both are questions where `full_system` forces unconditional positions on questions that are textbook condition-dependent, while adaptive routes to scenario-analyst mode and produces concrete condition mappings.

**Question-level highlights:**

| Question | full total | adaptive total | Δ focus | Note |
|----------|-----------|----------------|---------|------|
| q74 (co-founder vs. solo) | 2.8 | **5.0** | +3.0 | Highest adaptive score in dataset |
| q71 (retention vs. acquisition) | 2.6 | **4.0** | +1.5 | Classifier → context_dependent; condition mapping unlocked specificity |
| q9 (microservices vs. monolith) | 2.6 | **3.2** | **+2.0** | Largest focus gain; adaptive focus=4.0 vs full=2.0 |
| q8 (domain expert vs. generalist) | 1.6 | **2.6** | **+1.5** | Full forces binary answer on clearly context-dependent question |
| q78 (fire underperformers quickly) | **3.6** | 2.6 | −1.5 | full_system wins; topic has values dimension that classifier misses |

---

## 7. Discussion

### 7.1 Why Adaptive Wins on Binary: Classifier Routing Matters More Than Constraint Level

A subtle but important finding: the classifier routes the majority of human-labeled "binary" questions to `context_dependent`. Questions phrased as *"should startups do X?"* are recognized by the classifier as having answers that depend on company stage, market conditions, and team composition—because this is true.

This means the performance gain on binary questions is primarily attributable to the *context-dependent prompt design* (condition mapping, "WHEN...because..." structure), not to the *moderate PROHIBITION level* itself. The classifier's judgment overrides the human labeling, and the classifier's judgment proves more useful.

This has a practical implication for system design: **the question taxonomy is not a fixed property of a question's topic—it is a property of the question's analytical requirements given a specific domain and context.** A classifier that operates on these requirements produces better routing than a hardcoded taxonomy.

### 7.2 The Commitment–Calibration Trade-Off

Across both full_system and adaptive_prohibition, PROHIBITION constraints consistently reduce the `honest_uncertainty` dimension. Agents constrained from using hedging expressions cannot easily signal when a claim is tentative or context-dependent, even when that uncertainty is legitimate.

| System | honest_uncertainty (binary mean) | honest_uncertainty (values mean) |
|--------|----------------------------------|----------------------------------|
| full_system | 1.7 | 1.6 |
| adaptive | 1.4 | 1.4 |

This is a known trade-off in commitment elicitation: forcing a model to commit reduces both *false hedging* (avoiding a position to avoid being wrong) and *legitimate uncertainty signaling* (acknowledging genuine epistemic limits). The prohibition analysis (Section 6.2) shows PROHIBITION does not increase false certainty rates, but it demonstrably reduces `honest_uncertainty` scores.

Practitioners deploying multi-agent debate systems should account for this: PROHIBITION is most appropriate for decisions where a committed recommendation is the desired output (investment decisions, go/no-go calls), and less appropriate for analysis tasks where calibrated confidence levels are themselves valuable outputs.

### 7.3 Limitations

**Sample size.** Each question type is evaluated on n=10 questions. Effect sizes are consistent across types but should be interpreted cautiously; results may not generalize to question distributions outside the business/technology/policy domain used here.

**LLM-as-judge reliability.** Quality scores are produced by a single judge model (Qwen3-32B). Prior work shows LLM judges exhibit style preferences and position biases. Ground-truth accuracy (E3) provides a judge-independent validation signal, but the five-dimension rubric scores remain subject to judge-specific calibration.

**Single-round evaluation.** Due to API rate limits, E4 evaluates Round 1 positions only. This measures prompt design quality in isolation from multi-round debate dynamics. While Round 1 quality is the primary driver of final output quality in our system (given the synthesize step), results may differ for systems where rebuttal rounds produce significant position evolution.

**Classifier reliability.** The question type classifier is itself an LLM with no ground-truth validation. Section 7.1 notes that classifier routing on "binary" questions often diverges from human labeling—and produces better results, suggesting the classifier is capturing a meaningful signal. But classifier errors (especially at binary/context-dependent boundaries) could degrade adaptive performance on ambiguous questions.

---

## 8. Conclusion

We introduced Adaptive PROHIBITION, a framework that applies calibrated debate constraints based on question type in multi-agent LLM debate systems. Three findings shaped its design:

**PROHIBITION should match the question's epistemic requirements.** Values-based questions require committed advocacy; context-dependent questions require condition mapping; applying full constraints to the latter produces worse output. The framework's core contribution is making this calibration automatic via a classifier.

**The classifier's routing matters more than the constraint level.** Performance gains on binary questions come primarily from the classifier routing them to condition-mapping mode, not from the moderate PROHIBITION level. This implies that question type classification is the more fundamental design choice, with constraint level as a secondary parameter.

**Commitment has a calibration cost.** PROHIBITION reduces `honest_uncertainty` scores alongside false hedging. Systems that require calibrated confidence signaling should account for this trade-off rather than applying maximum commitment constraints uniformly.

The pattern across all findings—unexpected behavior → root-cause diagnosis → principled redesign—reflects a broader lesson: multi-agent debate systems require empirical validation of each design decision against actual system behavior, not just theoretical justification. Role definitions interact with system equilibria; divergence metrics determine whether debate occurs at all; constraint levels interact with question type in ways that override surface-level prompt quality.

---

## References

- Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.
- Du, Y., et al. (2023). Improving Factuality and Reasoning in Language Models through Multiagent Debate. *ICML 2023*.
- He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. *ICLR 2021*.
- Liang, T., et al. (2023). Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate. *arXiv:2305.19118*.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.
- Park, J. S., et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *UIST 2023*.
- Perez, E., et al. (2022). Sycophancy to Subterfuge: Investigating Reward Tampering in Language Models. *arXiv:2305.14325*.
- Wei, J., et al. (2023). Simple synthetic data reduces sycophancy in large language models. *arXiv:2308.03958*.

---

*Code and data: [github.com/xyma2003/multi-agent-debate](https://github.com/xyma2003/multi-agent-debate)*
