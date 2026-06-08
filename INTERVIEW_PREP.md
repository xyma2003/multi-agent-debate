# Debate-Agent Interview Prep

Structured answers for technical interviews. Each answer is written to be spoken naturally — trim or expand based on interviewer pacing.

---

## Elevator Pitches

### 30-second version
> "I built a multi-agent debate system where three LLM agents with distinct analytical frameworks — an opportunity analyst, a risk analyst, and an assumption challenger — analyze a topic independently, then argue against each other across multiple rounds. The system detects when agents genuinely disagree using NLI contradiction scoring, tracks concessions with attribution, and produces an auditable confidence-scored report. The goal is to produce more reliable multi-perspective analysis than a single LLM can, by making disagreements structural rather than simulated."

### 2-minute version (with technical depth)
> "The core problem I wanted to solve: when you ask a single LLM to analyze a topic from multiple perspectives, it typically hedges — it says 'on the one hand... but on the other hand...' without committing to any position. That's sycophancy at the system level.
>
> My approach: give each agent a *methodology*, not a personality. The Opportunity Analyst applies seed-stage VC thinking — look for asymmetric upside. The Risk Analyst applies venture debt thinking — find the most likely failure mode. The Assumption Challenger asks: what shared premise are both sides taking for granted?
>
> The agents run in parallel via LangGraph's Send API, then their key claims are compared using NLI cross-encoder to detect genuine stance contradiction. If they still disagree after comparing positions, they do a rebuttal round. Concessions are tracked with full attribution — who changed, why, triggered by which opponent's claim.
>
> I ran a proper ablation study: compared multi-agent vs single-LLM on hedge ratio and position diversity score across 10 questions. Found a 28% reduction in hedging, and — more interestingly — discovered a PDS paradox in my first implementation where multi-agent was *less* diverse than single-LLM. I diagnosed the root cause (old devil prompt caused 2-vs-1 alignment), fixed it, and re-validated. Also discovered cosine similarity is fundamentally broken for divergence detection in debate contexts — and implemented NLI as the fix."

---

## System Design Questions

### Q: Why LangGraph instead of AutoGen or CrewAI?

**Answer:**
Three reasons. First, **explicit state control**: debate has a precise execution topology — parallel fan-out for Round 1 (3 agents simultaneously), then collect, then conditional routing based on divergence score. LangGraph's `StateGraph` with `Send` for parallel branches and `Command` for conditional routing maps exactly to that structure. AutoGen's conversation-centric model would make me fight the framework to express it.

Second, **checkpointing**: `SqliteSaver` gives me interrupt/resume for free. Debates are long multi-step workflows — if I want to add human-in-the-loop review between rounds later, `interrupt()` already supports that. Third, **portfolio readability**: the graph topology is explicit and inspectable. A reviewer can read `graph.py` and understand the entire execution flow in one file.

I explicitly decided *against* `langgraph-supervisor` — its own docs now recommend using the supervisor pattern directly via tools rather than the library.

### Q: How does the parallel fan-out work?

**Answer:**
LangGraph's `Send` API. In `dispatch_round1`, instead of returning a state dict, I return a list of `Send` objects — one per agent — each containing the topic and initial context. LangGraph executes those three nodes concurrently, then routes all outputs to `collect_round1`, which uses a list reducer on `DebateState.agent_arguments` to merge the results. The key subtlety is that routing functions using `Send` must *not* be registered as graph nodes — only as conditional edge functions — otherwise LangGraph throws `InvalidUpdateError` because list[Send] is not a valid state update.

### Q: Why TypedDict for state instead of Pydantic BaseModel?

**Answer:**
LangGraph's reducer pattern — `Annotated[T, reducer_fn]` — works natively on TypedDict. If I used Pydantic for graph state, every node return would trigger Pydantic validation overhead on what's fundamentally just a dict update. The agent *output* models (AgentPosition, DebateRound) are Pydantic — that's where structured output validation matters. The graph state is TypedDict. Right tool, right layer.

### Q: How does concession tracking work?

**Answer:**
When an agent produces a rebuttal, its structured output includes `concessions: list[ConcessionRecord]`. Each record has: the original claim being conceded, the updated position, the `triggered_by_agent` (which opponent's argument caused it), and `triggered_by_claim` (which specific claim). This is stored in `DebateState.all_concessions` across rounds. The final report includes a full concession audit trail — you can see exactly what persuaded who. This was a deliberate design choice: if the synthesis conclusion is "the risk of founder dilution outweighs the growth acceleration," I want to trace that back to the specific Pessimist claim that persuaded the Optimist to concede on dilution timing.

---

## Technical Deep-Dive Questions

### Q: What is the PROHIBITION constraint and why did you implement it that way?

**Answer:**
PROHIBITION is a list of explicitly forbidden phrases in each agent's system prompt: "however", "but", "although", "on the other hand", "balanced view", "it depends". The Optimist is forbidden from mentioning risks; the Pessimist is forbidden from mentioning upsides.

The reason is behavioral, not cosmetic. Without it, agents — even with distinct personas — tend to produce balanced, hedge-everything responses because that's what the pretraining distribution rewards. The PROHIBITION blocks the linguistic *escape hatch*. Combined with a terminal instruction ("your position must be a concrete claim, not a hedge"), it forces each agent to commit.

The empirical result: 28% reduction in hedge ratio vs. single-LLM (HR: 0.0093 vs 0.0129, n=10). This is a medium-to-large effect size.

### Q: Why "methodology-based" personas instead of "personality-based"?

**Answer:**
"You are very pessimistic" produces theatrical pessimism — the model performs being grumpy. "Your analytical framework is: 1. Identify the most likely failure mode, 2. Estimate its probability and impact, 3. Assess whether the opportunity justifies that specific risk" produces *structured* risk analysis.

The practical difference: personality-based prompts cause sycophancy collapse in multi-round debate. If an agent is just "feeling pessimistic," it can abandon that feeling when faced with counterarguments. If an agent is applying a specific methodology, it has to find a reason *within that methodology* to change its position — which is what you actually want from a debate.

### Q: Explain the PDS metric. Why pairwise semantic distance?

**Answer:**
PDS (Position Diversity Score) is the average cosine distance between each pair of agents' final `key_claims` embeddings. For 3 agents, that's 3 pairs: Optimist-Pessimist, Optimist-Devil, Pessimist-Devil.

Why pairwise? Because I want to capture whether *all three* agents hold distinct positions, not just whether the most extreme two differ. If Optimist and Devil both converge on "VC is good" while only Pessimist disagrees, pairwise average captures that two-thirds of the agent space has collapsed — whereas a max-distance metric would miss it.

Higher PDS = more genuinely diverse viewpoints generated by the system. This is a core design goal: the system should produce real triangulation, not simulate it.

### Q: Why does cosine similarity fail for divergence detection?

**Answer:**
Cosine similarity is a *topic similarity* measure, not a *stance opposition* measure. "VC accelerates startup growth and increases success probability" and "VC creates unsustainable growth pressure and founder dilution that kills startups" are semantically *similar* by cosine — they share the vocabulary of VC, startup, growth. The embedding model encodes them as nearby vectors because they're about the same topic.

What I actually need to detect is stance *contradiction*. That requires NLI (Natural Language Inference). A cross-encoder NLI model classifies claim pairs as CONTRADICTION / ENTAILMENT / NEUTRAL using the *logical relationship* between claims, not vocabulary overlap. "VC is good → high contradiction probability with ← VC is bad" even though they share all the same nouns.

The empirical consequence: 100% of cosine-based debates terminated after Round 1, with divergence scores of 0.097–0.258, all far below the 0.75 threshold. The NLI variant correctly detected contradiction (scores 0.86–0.83 in Round 1) and ran 3 full rounds with actual stance evolution (SSS=0.883 vs SSS=1.000 for cosine).

---

## Research & Experiment Design Questions

### Q: How did you design your evaluation metrics?

**Answer:**
I wanted metrics that measured *what the system was designed to do*, not proxy measures. Three behavioral dimensions:

1. **Hedge Ratio (HR)**: Does the system actually reduce sycophantic hedging? Operationalized as: count of hedge words ("however", "but", "although", "it depends", etc.) divided by total word count. Simple, interpretable, directly tied to the PROHIBITION design goal.

2. **Position Diversity Score (PDS)**: Does the system generate genuinely diverse viewpoints? Operationalized as: avg pairwise cosine distance between agents' `key_claims` embeddings. Higher = more distinct positions.

3. **Stance Stability Score (SSS)**: Do agents maintain their positions under argumentative pressure without collapsing to consensus? Cosine similarity between Round-1 position embedding and final position embedding. High SSS = stance maintained (agents debated but weren't randomly swayed). Only meaningful for multi-round debates.

I deliberately did *not* use LLM-as-judge quality scoring, because for a research-oriented portfolio project I wanted metrics I could audit and compute deterministically.

### Q: What was the PDS paradox and how did you diagnose it?

**Answer:**
Initial results showed multi-agent PDS of 0.1707 vs single-LLM PDS of 0.2160 — the multi-agent system was generating *less* diverse viewpoints than a single LLM asked for multiple perspectives. That's the opposite of the design goal.

I diagnosed it by looking at per-question agent positions. The pattern was consistent: Optimist would argue pro, Pessimist would argue con, and Devil would also argue con — just with different framing. So instead of a triangle of perspectives, I was getting a line: Optimist on one end, Pessimist+Devil on the other. Two agents were pulling in the same direction.

Root cause: the Devil's original prompt was "challenge the dominant view." In most debates, the Optimist presents the dominant/mainstream view. So the Devil automatically aligned with the Pessimist to oppose it. Net effect: 2-vs-1 configuration, lower average pairwise distance.

Fix: change the Devil's mission from "oppose the dominant view" to "challenge the shared assumption both sides are taking for granted." Now the Devil targets the *frame* that both Optimist and Pessimist share, taking a genuinely third-dimensional position. Post-fix PDS: 0.2242 > 0.2160, paradox resolved.

### Q: How would you extend this to production?

**Answer:**
Several gaps between current state and production-ready:

1. **Scale the benchmark**: n=10 gives us directional findings but weak statistical power. Need 30 questions across all 4 categories to clear α=0.05 for all metrics.

2. **Replace Groq rate-limit dependency**: current backend (Groq) hits rate limits at >10 questions, requiring 2-minute delays. For production, swap to Anthropic direct or add multi-key rotation.

3. **Streaming UI**: the current Streamlit UI polls for a completed debate report. A production system would stream each agent's position as it's computed — LangGraph's `astream_events` API supports this with event filtering by node name.

4. **Human-in-the-loop**: LangGraph's `interrupt()` is already architecturally wired in — adding a "human review" checkpoint before synthesis would let a user redirect the debate ("push harder on the regulatory risk angle") before final report generation.

5. **Stronger synthesis**: current `synthesize_stub` just assembles agent outputs. A proper synthesis node would use weighted confidence scoring, identify which concessions changed the trajectory, and produce a structured conclusion with stated confidence and minority dissent.

---

## Failure Modes & Lessons Learned

### Q: What would you do differently if you rebuilt this?

**Answer:**
Three things:

1. **Start with NLI divergence detection.** I built cosine first because it was simpler, discovered the fundamental failure in the benchmark, then retrofitted NLI. I should have reasoned from first principles: stance opposition is a logical relationship, not a semantic similarity problem. Cosine was the wrong tool from the start.

2. **Instrument the debate state during development, not after.** I discovered the 2-vs-1 configuration (PDS paradox) by manually reading agent outputs after seeing bad metrics. I should have built a per-round "agent alignment heatmap" display that visualizes pairwise positions as I'm developing, so I can see configuration problems immediately.

3. **Define the Devil's role by what it *cannot* do, not just what it should do.** The final Devil prompt is effective partly because of its PROHIBITION block ("do not simply oppose the optimist, do not simply align with the pessimist"). Negative constraints were more robust than positive instructions for this role.

### Q: What's the hardest technical challenge you faced?

**Answer:**
Getting LangGraph's `Send` fan-out to work correctly with state merging. When 3 agents run in parallel and all return `AgentArgument` objects, those need to be merged into a single list in `DebateState.agent_arguments`. LangGraph merges them using a reducer — I had to annotate the state field with `Annotated[list[AgentArgument], add_messages_reducer]` (adapted for non-message types). The subtle issue: if you accidentally register the routing function (which returns `list[Send]`) as a graph node, LangGraph throws `InvalidUpdateError` because it tries to treat the Send list as a state dict. The error message is cryptic. The fix is to only pass routing functions to `add_conditional_edges`, never to `add_node`.

---

## Behavioral / Meta Questions

### Q: This is a portfolio project. Why should I take the results seriously?

**Answer:**
Fair challenge. I'd separate three claims:

1. **The design claims** (methodology-based personas, PROHIBITION, NLI over cosine) are principled and I can defend the reasoning from first principles independent of results.
2. **The directional findings** (HR reduction, PDS paradox identification and fix) are robust at n=10 — the effect sizes are large enough that they're unlikely to reverse at n=30.
3. **The NLI finding** (that cosine is fundamentally broken) is categorical, not statistical — it's a *qualitative* failure (100% premature termination) visible at n=2. n=10 wouldn't change it.

Where I'd be cautious: claiming specific numbers (28% HR reduction) as production-grade measurements. What I'm confident in: the direction of all findings and the architectural reasoning behind them.

### Q: How did you approach building something you'd never built before?

**Answer:**
I started from the core failure mode I wanted to solve (sycophancy in single-LLM analysis), worked backwards to the minimal architecture that addresses it, then instrumented it to measure whether it actually worked. The benchmark was designed *before* the system was complete — I wrote `evaluator.py` defining PDS, HR, and SSS as a spec for what "working" looks like, then built until the system met it.

When I got bad results (PDS paradox), I treated it as a hypothesis to diagnose rather than a failure to hide. That's the interesting part — discovering that the devil prompt was causing 2-vs-1 alignment was more valuable than if everything had worked first try.
