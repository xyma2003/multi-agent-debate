# Domain Pitfalls: Multi-Agent LLM Debate System

**Domain:** Multi-agent LLM orchestration with structured debate loop
**Researched:** 2026-04-23
**Stack context:** LangGraph 1.x + Claude sonnet-4-6 + Pydantic v2 + sentence-transformers + Streamlit 1.56 + SQLite

---

## Critical Pitfalls

Mistakes that cause rewrites, silent correctness failures, or runaway API costs.

---

### Pitfall 1: Agent Sycophancy — Agents That Fake Disagreement

**What goes wrong:**
All three debate agents (Optimist, Pessimist, Devil's Advocate) produce substantively identical analyses, then use surface-level hedge language ("while there are risks…", "to play devil's advocate…") to appear as though they disagree. The Debate Engine sees low embedding similarity between these surface reframings, declares divergence, and triggers unnecessary debate rounds. The system looks like it is working, but is producing zero additional insight.

This is the core correctness failure for this project. A debate system where agents secretly agree is worse than a single LLM pass — it costs 3-4x more and adds false confidence.

**Why it happens:**
- Claude's RLHF training strongly rewards balanced, hedged, "consider-multiple-perspectives" answers. Without strong persona enforcement, every agent converges to the same balanced response with different label sentences prepended.
- The system prompt pattern "You are an Optimist. Find the upside." is too weak. The model reads it as a mild suggestion, not a structural constraint.
- Role is specified once at session start. By round 2, the model has seen counterarguments, which its training causes it to partially accommodate — weakening the role bias.

**Warning signs:**
- Optimist output contains more than one risk mention.
- Pessimist output contains more than one opportunity mention.
- Devil's Advocate agrees with the majority view in its conclusion.
- Cosine similarity between agent outputs is high (> 0.85) but the system still flags "divergence."
- The rebuttal round changes very few concrete claims — only hedging language changes.

**Prevention strategy:**
1. **Structural persona injection, not a label.** The system prompt must not say "You are an Optimist." It must force a decision procedure:
   ```
   You are the Opportunity Analyst. Your job is to enumerate ONLY the reasons this
   could succeed. You are prohibited from mentioning risks, caveats, or downsides.
   If you find yourself writing a caveat, stop and delete it. Your score is based
   purely on how many credible upsides you surface.
   ```
2. **Re-inject the role directive at the start of each rebuttal round.** Do not rely on the system prompt persisting across rounds. Pass it again as the first human message in the rebuttal call.
3. **Add a compliance check.** After each agent call, before accepting the output, run a quick heuristic: if the Pessimist's output contains the words "opportunity", "upside", "growth" more than N times, flag a persona drift warning.
4. **Score outputs against expected skew.** Each agent should have a measurable sentiment bias. Use the sentiment score of the output as a health metric. Neutral outputs from a biased role = persona drift.

**Phase mapping:** Phase 1 (agent implementation). Persona enforcement is the foundation — if this is wrong, every subsequent phase is built on broken data.

---

### Pitfall 2: Divergence Detection Miscalibration — Wrong Threshold Kills the Debate

**What goes wrong:**
The sentence-transformers semantic similarity score is used as the sole divergence signal. Two failure modes exist:

- **Threshold too high (e.g., 0.95):** Agents that are genuinely disagreeing on substance but using different vocabulary get classified as "converged." The debate terminates prematurely. The Synthesizer receives inputs that are not actually in agreement, and produces a false consensus.
- **Threshold too low (e.g., 0.70):** Agents that rephrased the same answer with different words get classified as "diverged." Extra debate rounds fire. Costs 2-3x the expected API spend. The debate never terminates naturally — only the max-rounds guard saves it.

Cosine similarity between raw embeddings is a poor proxy for argumentative disagreement. Two passages can have identical conclusions phrased differently (low similarity) or use the same words to reach opposite conclusions (high similarity).

**Why it happens:**
- Semantic similarity measures vocabulary/topic overlap, not logical stance.
- A sentence like "This venture will likely fail because capital is insufficient" and "Capital constraints mean this is unlikely to succeed" have very high cosine similarity (~0.92) yet are the same claim.
- "Option A is superior" and "Option B has merit" have low cosine similarity but do not actually conflict.

**Warning signs:**
- Debate always hits max_rounds regardless of topic.
- Debate always converges in round 1 regardless of how controversial the topic is.
- Threshold calibration was done on a single test topic.

**Prevention strategy:**
1. **Do not use raw embedding cosine similarity as the divergence signal.** Use a two-layer check:
   - Layer 1: Ask Claude directly — "Do agent A and agent B reach the same conclusion on the key question? Answer YES or NO with a one-sentence reason." This is cheap (~200 tokens) and much more accurate.
   - Layer 2: Use embedding similarity only as a fast pre-filter to skip the Claude check when similarity is extremely high (> 0.97 → definitely converged).
2. **Define divergence at the claim level, not the passage level.** Extract the final verdict sentence from each agent output (e.g., "the structured field `verdict`" from the Pydantic schema). Compare verdicts, not full arguments. Verdict-level similarity is far more meaningful.
3. **Calibrate on at least 5 diverse test topics** before choosing a threshold. Use one unambiguously good topic, one unambiguously bad topic, and three genuinely ambiguous topics.
4. **Log similarity scores for every debate.** Without logging, threshold calibration is guesswork.

**Phase mapping:** Phase 2 (debate engine). Do not ship divergence detection without logged similarity scores and at least 3 test topics used for manual calibration.

---

### Pitfall 3: Context Window Blowup Across Debate Rounds

**What goes wrong:**
The debate state accumulates all previous agent arguments in each round. By round 3, each agent receives its own Round 1 output + Round 2 rebuttal + all three other agents' Round 1 and Round 2 outputs as context. For 3 agents at ~800 tokens per output, Round 3 input is already ~6,000 tokens of prior context before the system prompt and new question.

On a 4-round debate, this is: 3 agents × 4 rounds × ~800 tokens = ~9,600 tokens of debate history per call, multiplied by 3 agents = ~29,000 tokens just for round 4 rebuttals. Add the Synthesizer call on the full history: total cost easily reaches 50,000+ tokens per debate.

For a portfolio demo, this appears as a slow, expensive system. For production, it is a blocker.

**Why it happens:**
- The natural LangGraph pattern uses `add_messages` reducer, which appends all messages to a growing list. If agent outputs are stored as messages, the list grows indefinitely.
- Developers assume "just pass the full state" and do not think about what each agent actually needs to see.

**Warning signs:**
- Per-debate cost exceeds $0.50 at claude-sonnet-4-6 pricing (roughly 50K tokens).
- Response latency grows each round (more tokens = more TTFT and generation time).
- The `messages` field in the state graph exceeds 5,000 tokens after round 2.

**Prevention strategy:**
1. **Do not use `add_messages` for debate content.** Store agent outputs in structured fields (`round_1_outputs: list[AgentOutput]`, `round_2_rebuttals: list[AgentOutput]`) with explicit replacement semantics, not append semantics.
2. **Pass each agent only what it needs.** In the rebuttal prompt, do not pass the full previous round for all agents. Pass: (1) the agent's own previous output, (2) a compact summary of opposing arguments (~100 words each), not the full text.
3. **Summarize previous rounds before passing to later rounds.** Before dispatching Round 3, summarize Rounds 1-2 into ~300 tokens: "Agent X claims Y. Agent Z claims W. Disputed points: P, Q." The Synthesizer uses this summary, not the full transcript.
4. **Set a hard token budget per node.** Measure `tiktoken` estimate before each LLM call. If over budget, truncate to the most recent N arguments.
5. **claude-sonnet-4-6 pricing baseline:** ~$3/M input tokens, ~$15/M output tokens (April 2026). A well-designed debate should cost under $0.10/run. Budget accordingly.

**Phase mapping:** Phase 2 (debate engine architecture). Design the state schema with explicit token control from the start — retrofitting this in Phase 3 requires rewriting the state and all node inputs.

---

### Pitfall 4: LangGraph Graph Cycles With No Termination Guard

**What goes wrong:**
The architecture has an explicit cycle: `divergence_check → dispatch_rebuttal → collect_rebuttal → divergence_check`. If the `divergence_check` node never routes to `synthesize` (due to a logic bug, a stuck divergence score, or a Pydantic validation error that silently leaves the state unchanged), the graph runs until it hits `GraphRecursionError`.

`DEFAULT_RECURSION_LIMIT` in LangGraph 1.x is **10,007 steps** (verified from source: `int(getenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "10007"))`). The cycle in this system makes one API call per step (3 agents per round). A stuck debate will make ~3,000+ Claude API calls before the error fires. At claude-sonnet-4-6 pricing, this is hundreds of dollars before being caught.

**Why it happens:**
- The max-rounds guard lives inside `divergence_check`. If a Pydantic validation error in `collect_rebuttal` causes `round_num` to not increment (because the node raised an exception before updating state), the guard condition `round_num >= max_rounds` is never true.
- Missing edge: if an edge from a collection node to `divergence_check` is accidentally omitted, LangGraph falls through to END without calling `synthesize` — the error is silent.

**Warning signs:**
- The graph runs for more than 5 rounds on a simple topic.
- `round_num` in state is not incrementing as expected.
- API cost spikes unexpectedly during development testing.

**Prevention strategy:**
1. **Always set an explicit `recursion_limit`** far below the default when invoking the graph during development:
   ```python
   graph.invoke(input, config={"recursion_limit": 30})  # 30 = 10 rounds × 3 agents with margin
   ```
2. **The max-rounds guard must be the FIRST check in `divergence_check`**, before any divergence scoring logic. If `round_num >= MAX_ROUNDS`, route to `synthesize` unconditionally. This guard must be immune to state corruption.
3. **Increment `round_num` at the START of `dispatch_rebuttal`**, not at the end of `collect_rebuttal`. This ensures the increment happens even if collection partially fails.
4. **Add a graph-level cost watchdog** in development: count total node executions from a LangSmith callback or a simple global counter. Alert if > 15 node calls for a single invoke.
5. **Use `stream_mode="updates"` during testing** to see which nodes are being called. An infinite loop is immediately visible as the same node name repeating.

**Phase mapping:** Phase 2 (debate engine). The max-rounds guard and the recursion_limit config are non-negotiable from the first working loop implementation.

---

### Pitfall 5: Pydantic Validation Errors Mid-Debate — Silent State Corruption

**What goes wrong:**
Claude occasionally returns tool-use JSON that does not match the Pydantic schema. With `with_structured_output(MyModel)` (default `include_raw=False`), a `ValidationError` propagates as an unhandled exception, crashing the LangGraph node. The node raises, LangGraph catches the exception per its error handling, and the state update for that node is skipped. The next node sees a partially-updated state (one or two agents updated, one missing), which propagates silently.

Worse: with `include_raw=True`, the `parsed` field is `None` on failure and `parsing_error` contains the exception — but if the node does not check for this, it passes `None` into the state and downstream nodes process it as if it were a valid output.

**Why it happens:**
- Claude reliably follows the schema in >99% of calls, but 0-1% failures compound across 12+ agent calls per debate.
- Complex schemas (nested Pydantic models, union types, optional fields with defaults) are more likely to trigger subtle validation mismatches.
- The Anthropic API issue tracker confirmed that "model responses may violate input schema" (Issue #619, closed August 2024) — this is a known, accepted limitation.

**Warning signs:**
- Occasional `ValidationError` tracebacks in logs during longer test sessions.
- The number of items in a round's output list is sometimes 2 instead of 3.
- The Synthesizer receives a state where one agent's rebuttal is `None`.

**Prevention strategy:**
1. **Use `include_raw=True` on every agent call:**
   ```python
   llm.with_structured_output(AgentOutput, include_raw=True)
   ```
   Check `result["parsing_error"]` before using `result["parsed"]`.
2. **Build a retry wrapper** (max 2 retries) that re-invokes the LLM with the original prompt if `parsed` is None. On third failure, inject a sentinel `AgentOutput` with `error=True` and a fallback message. Never propagate `None` into state.
3. **Keep schemas flat.** No nested Pydantic models, no union types, no `Optional` fields with complex defaults. Every field should be a `str`, `int`, `float`, `bool`, or `list[str]`. Complex nesting is the leading cause of schema violations.
4. **Add a state validation step** in `collect_round1` and `collect_rebuttal` nodes: count non-error outputs before proceeding. If count < expected_agents, log a warning and either retry or proceed with a clearly-marked incomplete round.

**Phase mapping:** Phase 1 (individual agent nodes). Build the retry wrapper before the debate loop — it is much harder to retrofit after the state schema is set.

---

## Moderate Pitfalls

---

### Pitfall 6: Streamlit + LangGraph Sync vs. Async Confusion

**What goes wrong:**
Developers try to use LangGraph's `astream()` (async) inside a Streamlit script callback, then call `asyncio.run()` or `loop.run_until_complete()` — and hit `RuntimeError: This event loop is already running` because Streamlit's Tornado server runs its own loop on the main thread.

**Why it happens:**
Streamlit 1.56 runs user scripts in a **separate thread** (not the Tornado main thread), and `script_runner.py` has no asyncio imports. The script thread has no running event loop. However, if the developer calls `asyncio.get_event_loop()` and then `loop.run_until_complete()`, in Python 3.12 this can conflict with Tornado's loop if thread boundaries are crossed.

**Warning signs:**
- `RuntimeError: This event loop is already running` on first run.
- `RuntimeError: no running event loop` during async calls in a callback.
- LangGraph streaming works in unit tests but hangs in Streamlit.

**Prevention strategy:**
1. **Use the synchronous `graph.stream()` everywhere in Streamlit, not `graph.astream()`.** Verified from source: `stream()` uses `SyncPregelLoop` — fully synchronous, no asyncio dependency. It works correctly in Streamlit's script thread.
2. **If streaming token-by-token is needed**, use `stream_mode="messages"` with the synchronous `stream()`. This yields `(token_chunk, metadata)` tuples without requiring async.
3. **Never call `asyncio.run()` inside a Streamlit widget callback.** Move any async work to a background thread via `threading.Thread` if truly needed, but synchronous `stream()` eliminates the need.
4. **Test streaming behavior in a standalone Python script first**, then integrate into Streamlit. Isolating the integration issue is much easier this way.

**Phase mapping:** Phase 3 (Streamlit UI integration). Establish the sync streaming pattern in a spike before building the full UI.

---

### Pitfall 7: "Debate That Never Converges" — Stop Condition Design

**What goes wrong:**
The max-rounds guard fires, but the Synthesizer receives 3 rounds of agents arguing past each other without ever reaching genuine agreement. The final report says "consensus was reached" but the disputed_points list is full of unresolved claims. The confidence score is arbitrarily high because the Synthesizer was asked to produce one.

This is a correctness problem more than an infinite loop problem. The debate terminates, but produces low-quality output.

**Why it happens:**
- Some topics are genuinely irresolvable with 3 rounds. The agents are not converging because the underlying question has no clear answer.
- The Synthesizer is prompted to "reach a verdict" rather than "characterize the level of agreement."
- Confidence score is produced as a required field, so the LLM generates a plausible-sounding number regardless of actual evidential basis.

**Warning signs:**
- `disputed_points` list has 5+ items in the final report.
- Confidence score is the same (e.g., 72%) across wildly different topics.
- The Synthesizer verdict says "mixed evidence" or "uncertain" — which means it could not actually synthesize.

**Prevention strategy:**
1. **Design two synthesis modes:** (a) true consensus mode when similarity is high, (b) "unresolved debate" mode that explicitly reports the nature of the disagreement rather than forcing a verdict.
2. **The confidence score must be derived from a formula**, not free-formed by the LLM: `confidence = (1 - max_divergence_score) * round_adjustment`. Do not ask the LLM to invent a confidence score.
3. **Cap max_rounds at 3 for the MVP.** More rounds rarely produce more convergence — they just produce more tokens. Academic research on multi-agent debate (Du et al., 2023) found that most benefit appears in rounds 1-2; rounds 3+ have diminishing returns.
4. **The Synthesizer prompt must have an "honest uncertainty" path:** "If the agents did not reach agreement, list the key disputed claims and report confidence below 50%. Do not force a verdict."

**Phase mapping:** Phase 2 (debate engine) for the max-rounds logic; Phase 2/3 boundary for the Synthesizer prompt design.

---

### Pitfall 8: Prompt Engineering Failures for Biased Roles — Cartoonish vs. Collapsed

**What goes wrong:**
Two failure modes exist for role-playing agents:

- **Cartoonish:** The Pessimist uses hyperbolic doom language ("this will certainly fail", "catastrophic risk") that reads as obviously non-credible. The Optimist says "incredible opportunity!" repeatedly. The debate feels like a parody, not a credible analysis.
- **Collapsed:** As described in Pitfall 1 — the model's training collapses all roles toward a balanced, hedge-everything answer.

**Why it happens:**
- Cartoonish mode: prompts that say "BE EXTREMELY PESSIMISTIC" activate over-the-top behavior.
- Collapsed mode: prompts that are too mild.
- Neither mode produces the desired output: a credible, expert-quality analysis from a specific epistemic stance.

**Prevention strategy:**
1. **Define roles by methodology, not intensity.** Don't say "be very pessimistic." Say "your analytical framework is: identify the single most likely failure mode, estimate its probability and impact, and assess whether the opportunity justifies that specific downside risk. You ignore upside scenarios."
2. **Give each role a reference persona:** "You analyze like a risk manager at a venture debt fund" (Pessimist), "You analyze like a seed-stage VC associate looking for portfolio fit" (Optimist), "You analyze like a senior strategy consultant who has seen the pitch before and found a specific flaw" (Devil's Advocate).
3. **Prohibit specific language patterns** in the system prompt: "Do not use the words 'however', 'but', 'on the other hand', 'while there are risks'."
4. **Test persona drift explicitly:** On the same topic, compare outputs across 5 runs. If the standard deviation of the sentiment score is high, the persona is unstable.

**Phase mapping:** Phase 1 (agent prompts). Invest 2-3 iterations in prompt refinement before wiring up the debate loop. Bad prompts make all downstream work invalid.

---

### Pitfall 9: API Cost Blowup — Multi-Round Multi-Agent Compounding

**What goes wrong:**
A seemingly simple 3-agent, 3-round debate at claude-sonnet-4-6 pricing costs far more than expected if context is not managed. Rough cost model without mitigation:

```
Round 1: 3 agents × (2,000 token input + 600 token output) = ~7,800 tokens
Round 2: 3 agents × (2,000 + prev_context 2,400 + 600 output) = ~15,000 tokens
Round 3: 3 agents × (2,000 + prev_context 4,800 + 600 output) = ~21,600 tokens
Synthesizer: 1 call × (2,000 + full_history 9,600 + 1,200 output) = ~12,800 tokens

Total: ~57,200 tokens × ~$3/M input + output at ~$15/M
≈ $0.17/debate at minimum, easily $0.50+ with naive context handling
```

For a portfolio demo doing 20 runs per review session, naive implementation costs $3-10 per review session.

**Why it happens:**
- Context accumulation (Pitfall 3) is the primary driver.
- Developers do not track token usage during development, so cost surprises appear at demo time.

**Warning signs:**
- No `usage` field logging on any LLM call.
- Debate state contains full message history without summarization.
- Claude Sonnet used for divergence detection (expensive for a binary decision).

**Prevention strategy:**
1. **Log `response.usage.input_tokens` and `response.usage.output_tokens`** for every LLM call from day one. Sum per-debate. Budget: target < $0.10/debate.
2. **Use claude-haiku-3-5 for cheap classification tasks:** divergence detection (is A vs B the same?), persona compliance checks, debate round summaries. Reserve Sonnet for the substantive analysis calls.
3. **Implement context summarization** between rounds as described in Pitfall 3.
4. **Set a hard per-debate cost cap** in code. After round completion, check total tokens used. If > 40K tokens, route to Synthesizer immediately regardless of round count.
5. **Cache the topic preprocessing step.** If the same topic is submitted twice (common during demo testing), avoid re-running Round 1.

**Phase mapping:** Phase 1 (set up token logging infrastructure alongside the first agent node). Cost awareness from day one prevents surprises at demo time.

---

## Minor Pitfalls

---

### Pitfall 10: LangGraph `Send` Fan-Out State Merge Collisions

**What goes wrong:**
When using `Send` to dispatch to 3 agents in parallel, all three nodes write to the same state channel (e.g., `round_1_outputs`). If the state field uses a simple overwrite reducer (not a list-append reducer), the last agent to complete overwrites the others.

**Prevention:**
Use `Annotated[list[AgentOutput], operator.add]` as the reducer for any state field that receives outputs from parallel `Send` dispatches. This is a list-append reducer and is safe for fan-in. Never use bare `list[AgentOutput]` for a fan-in target — that uses the default overwrite reducer.

**Phase mapping:** Phase 1 (state schema design).

---

### Pitfall 11: SQLite Debate History — Blocking the Streamlit Thread

**What goes wrong:**
Synchronous SQLite writes inside a LangGraph node block the Streamlit script thread. For debate saves (which can be several KB of JSON), this adds 50-200ms latency on each write and can cause Streamlit to appear unresponsive.

**Prevention:**
Write debate history asynchronously — use a background thread for SQLite writes, or batch all writes to a single post-debate save rather than saving per-round. The LangGraph `SqliteSaver` checkpointer handles its own persistence; the custom debate history save is a separate concern and should happen only at debate completion.

**Phase mapping:** Phase 4 (persistence).

---

### Pitfall 12: Divergence Score Not Logged — Cannot Tune Later

**What goes wrong:**
Divergence scores are computed during the debate but not stored in the debate history. After 20 debates, there is no data to tune the threshold with. Threshold tuning becomes guesswork forever.

**Prevention:**
Store the per-round divergence matrix (agent A vs B, A vs C, B vs C similarity scores) in the `DebateResult` SQLite record from day one. This costs ~50 bytes per round and enables data-driven threshold tuning after 5-10 real debates.

**Phase mapping:** Phase 2 (divergence detection node).

---

## Phase-Specific Warning Map

| Phase | Topic | Highest Risk Pitfall | Mitigation |
|-------|-------|----------------------|------------|
| Phase 1 | Agent nodes + personas | Sycophancy / collapsed roles | Structural persona prompts + compliance check |
| Phase 1 | State schema | Fan-out merge collision | Use `operator.add` reducer for all fan-in fields |
| Phase 1 | Structured output | Pydantic validation errors | `include_raw=True` + retry wrapper from day 1 |
| Phase 1 | Cost tracking | Invisible spend | Log `usage` on every call |
| Phase 2 | Debate loop | Infinite loop / cost runaway | Hard `recursion_limit=30` + max-rounds guard first |
| Phase 2 | Divergence detection | Wrong threshold, false signals | Two-layer check (Claude + embeddings), log all scores |
| Phase 2 | Context management | Token blowup round 3+ | Structured state fields, not `add_messages`, + summarization |
| Phase 2 | Synthesis | Never-converging debate | Honest uncertainty path + formula-derived confidence |
| Phase 3 | Streamlit streaming | Async/event loop conflicts | Sync `graph.stream()` only, no `asyncio.run()` |
| Phase 3 | Agent prompts | Cartoonish personas | Methodology-based roles, not intensity-based |
| Phase 4 | SQLite persistence | Blocking writes | Post-debate batch save only |

---

## Confidence Assessment

| Area | Confidence | Source |
|------|------------|--------|
| LangGraph recursion limit value (10,007) | HIGH | Verified from installed source: `langgraph._internal._config.DEFAULT_RECURSION_LIMIT` |
| LangGraph sync `stream()` is asyncio-free | HIGH | Verified from installed `pregel/main.py` — uses `SyncPregelLoop` |
| `add_messages` append-only reducer behavior | HIGH | Verified from installed `langgraph/graph/message.py` |
| `InvalidUpdateError` on concurrent state writes | HIGH | Verified from installed `pregel/_write.py` and `pregel/main.py` |
| Pydantic schema violation rate with Claude | MEDIUM | Anthropic SDK issue #619 (closed Aug 2024); anecdotal engineering reports |
| Sycophancy / persona collapse behavior | MEDIUM | Well-documented in RLHF literature; specific prompt patterns from engineering experience |
| Divergence detection threshold failure modes | MEDIUM | Known NLP limitation of cosine similarity; no specific LangGraph source |
| Streamlit script thread has no running event loop | HIGH | Verified from installed `streamlit/runtime/scriptrunner/script_runner.py` — no asyncio imports |
| Cost estimates (claude-sonnet-4-6) | MEDIUM | Based on Anthropic published pricing; exact token counts are estimates |

---

## Sources

- LangGraph source (installed, verified): `langgraph._internal._config` — `DEFAULT_RECURSION_LIMIT = 10007`
- LangGraph source (installed, verified): `langgraph.errors` — `GraphRecursionError` docstring and error codes
- LangGraph source (installed, verified): `langgraph.pregel._loop` — `self.stop = self.step + self.config["recursion_limit"] + 1`
- LangGraph source (installed, verified): `langgraph.pregel.main` — sync `stream()` uses `SyncPregelLoop`
- LangGraph source (installed, verified): `langgraph.graph.message` — `add_messages` is append-only reducer
- LangGraph source (installed, verified): `langgraph.pregel._write` — `InvalidUpdateError` on write conflicts
- Streamlit source (installed, verified): `streamlit.runtime.scriptrunner.script_runner` — no asyncio, user scripts run in separate thread
- Streamlit source (installed, verified): `streamlit.runtime.app_session` — Tornado event loop on main thread
- langchain-core source (installed, verified): `with_structured_output` with `include_raw=True` pattern
- Anthropic SDK GitHub issues: Issue #619 (model responses violating schema — closed Aug 2024)
- Du et al. 2023 (arXiv:2305.14325): Multi-agent debate improves reasoning, primary gains in rounds 1-2
