# Multi-Agent Debate System

[![CI](https://github.com/xyma2003/multi-agent-debate/actions/workflows/ci.yml/badge.svg)](https://github.com/xyma2003/multi-agent-debate/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文说明](README_zh.md)

**See where three perspectives disagree—and which claims change their positions.**

A LangGraph application that follows a question through independent arguments, bounded rebuttal rounds, attributed concessions, and a report you can inspect and export. The three roles are an Optimist, a Pessimist, and a Frame Challenger, shown as Devil's Advocate in the interface.

[Read the case](docs/CASE_STUDY.md) · [Open the complete report](docs/example-report.md) · [Run without an API key](#quick-start) · [Evidence and limits](#experiments-and-evidence-level)

![Current Streamlit interface displaying the fixed synthetic four-day-workweek example](docs/images/offline-debate.png)

The current interface showing the bundled **fixed synthetic example**. Its content and scores are prewritten for demonstration; this is not a live model result or evidence of model quality.

## One question, three perspectives

**Should a fictional 20-person software team trial a 32-hour, four-day week at unchanged pay?**

| Perspective | Starting point | Position after rebuttal |
|---|---|---|
| Optimist | Begin a reversible eight-week pilot | Assign weekday support owners and review thresholds first |
| Pessimist | Keep the current schedule until preparation is complete | Accept a pilot after a representative baseline and rota rehearsal |
| Frame Challenger | Establish a fair allocation of time off and coverage | Require role-level agreement and workload checks |

The example records two attributed concessions, a conditional recommendation, and remaining disagreement about readiness and wider adoption. The [case walkthrough](docs/CASE_STUDY.md) connects each change to its triggering claim; the [complete report](docs/example-report.md) includes all six arguments and can be read directly on GitHub, without installing anything.

## Quick start

Requirements: Python 3.10+. An LLM API key is required only to start a new debate; the bundled example and saved reports can be viewed without credentials.

```bash
git clone https://github.com/xyma2003/multi-agent-debate.git
cd multi-agent-debate

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Open <http://localhost:8501> and click **View offline example**. Even on a fresh checkout, you can explore two rounds of a fictional team's four-day-workweek discussion, the full arguments and key claims, consensus, remaining disagreements, and attributed concessions. Expand **Full Reasoning Trace** and **Concession Log** to follow the discussion.

The [bundled example](examples/offline_debate.json) is a prewritten, fixed synthetic illustration, not the output of a live debate run or evidence of model quality. Its divergence values, confidence estimates, and timestamp are illustrative; the report index follows the documented formula. Viewing it makes no API calls or local model downloads and never saves the example to **Past Debates**. Saved reports remain available separately in that sidebar. **Start Debate** stays disabled until credentials are configured.

### Find a saved debate

In **Past Debates**, use **Search topics** to find a literal substring in a saved
topic, or **Previous / Next** to browse ten reports per page, newest first.
Dates and report IDs distinguish repeated topics. Changing the search returns to
the first page; **Clear search** restores the full list and **Refresh** reloads it.
Search covers topic text, not report contents; `%` and `_` are ordinary characters.

Searching or paging only changes the sidebar. The open report stays visible with
its own saved date and ID until you select another report. Missing or unreadable
entries show a message and preserve the open report. Browsing history needs no
API credentials and does not rerun models or change stored reports; history
controls are unavailable while a debate is running.

### Export the open report

Under **Debate Result**, choose **Download Markdown** for a readable report or
**Download JSON** for structured data. Both contain the report ID, original
timestamp, verdict, consensus and disputes, every recorded argument and claim,
and the full concession log. Files use UTF-8 and preserve Chinese text.
Search and pagination do not change the download until you open another report.

JSON uses a versioned envelope: `schema_version`, `source`, `interpretation`, and
`report`. The `report` object contains the complete `DebateReport` data; its
legacy `confidence_score` field stays on the saved 0–1 scale, without
recalculation. This agreement index and agent confidence are not calibrated
accuracy. Exports do not infer the detector mode from current app settings.
The offline example keeps its synthetic source label and provenance in both
formats. Downloading requires no API credentials, runs no models, and does not
write to history. A generated report remains viewable and downloadable if its
database save fails; those exports explicitly mark persistence as unconfirmed.

Use **Retry save** to store the same report without running the debate again or
requiring API credentials. Browsing history or the example keeps this pending
report; **View unsaved report** brings it back. Finish saving or explicitly
**Discard unsaved report** before starting another debate. Recovery is limited
to the current browser session: download a copy before refreshing, closing it, or restarting
the server. A download alone does not add the report to history.

Starting a new debate uses the configured LLM API. On its first divergence check, the default NLI mode downloads `cross-encoder/nli-deberta-v3-small` (~180 MB). The optional cosine comparison mode downloads `BAAI/bge-small-en-v1.5` (~130 MB).

### Configuration

To start new debates, copy `.env.example` to `.env` (or edit your existing `.env`), replace the placeholder with your own credentials, and restart the app. The checked-in `.env.example` documents all options. A minimal SiliconFlow/OpenAI-compatible setup is:

```dotenv
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.siliconflow.cn/v1
OPENAI_MODEL=Qwen/Qwen3-32B
DIVERGENCE_MODE=nli
```

Supported backends are `openai`, `anthropic`, `groq`, `qwen`, `cerebras`, `together`, `sambanova`, and `gemini`.

## How it works

```text
topic
  └─► initialize
       └─► Optimist ─┐
           Pessimist ├─► collect ─► divergence check
           Challenger─┘                    │
                     ┌─ no contradiction / limit ─► synthesize ─► save
                     └─ contradiction ─► rebuttal fan-out ─┘
```

The default application uses NLI contradiction detection for multi-round routing. A lightweight cosine mode remains available for comparison, but it is not a reliable stance detector.

The final `DebateReport` contains consensus points, disputed points, a verdict, the Debate Agreement Index, the complete round trace, and every concession with its triggering agent and claim. The index is a heuristic about detected disagreement, not the probability that the verdict is correct.

- Parallel LangGraph fan-out/fan-in with bounded rebuttal loops
- Pydantic structured outputs and validation fallbacks
- NLI-based contradiction detection with caller-controlled round limits
- A formula-derived Debate Agreement Index and attributed concession logs
- SQLite persistence, searchable Streamlit replay, and Markdown/JSON export
- Reproducible benchmark runners for prompt and routing ablations

## Design choices

| Decision | Rationale |
|---|---|
| Methodology-based roles | Gives each agent an analytical procedure, not just a personality label |
| NLI routing by default | Detects contradiction rather than topical vocabulary overlap |
| Detector-owned pair decisions | Avoids comparing cosine distance and NLI probability against one shared numeric threshold |
| Caller round cap plus hard safety cap | Honors `max_rounds` while preventing unbounded loops |
| Debate Agreement Index | Makes the heuristic calculation inspectable without asking an LLM to invent a score |
| Concession attribution | Records which claim changed which position and why |

## Reading the Debate Agreement Index

The report displays this heuristic as **18.0 / 100**, not as a percentage:

```text
Debate Agreement Index = 100 × (1 − highest divergence across all rounds) × round adjustment
Round adjustment: 1 round → 1.0; 2 rounds → 0.9; 3 or more rounds → 0.8
```

For two rounds with divergence scores `[0.8, 0.2]`, the index is `100 × (1 − 0.8) × 0.9 = 18.0 / 100`. It uses the highest observed divergence, so the lower final-round value does not change the result to 72.0. The round adjustment is a fixed design choice, not a measured relationship between debate length and answer quality.

**A high index does not establish correctness, sufficient evidence, or genuine consensus.** It is not a calibrated probability of correctness and does not measure how much the agents improved or finally agreed. Even zero detected divergence can occur with inadequate evidence. The reported stop reason is separate: `converged` means the active detector found no divergent agent pair, not that the verdict was fact-checked.

In NLI mode, each round's divergence is the mean of the highest claim-pair contradiction probabilities for each agent pair; low contradiction can also mean neutral or unrelated claims. Cosine mode measures semantic proximity, which can be high even for opposing claims. Do not compare index values across these modes. New reports record the detector settings for each round. Older reports without this metadata remain **unknown**; current settings cannot establish which detector produced a historical score.

Each agent's **self-reported confidence** describes its own position. It is also uncalibrated, remains separate from this index, and does not enter the formula. For compatibility, the stored report field is still named `confidence_score` and keeps its existing 0–1 value and calculation; the display multiplies it by 100.

## Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the deterministic suite used by CI:

```bash
python -m pytest -m "not integration and not model" -q
```

Optional suites:

```bash
# Downloads local embedding/NLI models
python -m pytest -m model -v

# Requires configured API credentials
python -m pytest -m integration -v

# Offline Streamlit application tests (also included in CI)
python -m pytest -m ui -v
```

## Experiments and evidence level

The repository includes raw outputs and runners for engineering exploration, not publication-grade evidence. Results are small-sample, mostly single-run, and some quality evaluations use one LLM judge.

The clean base comparison contains **10 matching questions per system**. The historical multi-agent runs in this table used **cosine routing**, not the current application's default NLI routing:

| System | n | PDS | Hedge ratio | Mean rounds |
|---|---:|---:|---:|---:|
| Fixed frame-challenger prompt | 10 | 0.2242 | 0.0093 | 1.00 |
| Single-LLM baseline | 10 | 0.2160 | 0.0129 | 1.00 |
| Original challenger prompt | 10 | 0.1707 | 0.0077 | 1.00 |

In this sample, the fixed prompt produced higher PDS than the original prompt, and the multi-agent output's mean hedge ratio was about 28% lower than the single-LLM baseline. These are observed sample differences, not causal or generalizable performance claims.

PDS summarizes embedding-based diversity between agents' key claims; hedge ratio counts matched hedge expressions per word. Stance stability (SSS) compares an agent's first and final positions using embedding similarity. These descriptive metrics do not establish answer correctness, evidence quality, or resistance to sycophancy. Fewer hedge expressions can also mean less appropriate uncertainty.

A separate canonical NLI run contains **7 questions**: **4/7** reached multiple rounds, mean rounds were **2.14**, and mean stance stability was **0.9772**. These are separate runs, not a paired estimate of a causal routing improvement. See [`results/README.md`](results/README.md) for canonical files, validation exclusions, judge limitations, and provenance gaps. The fixed offline example above is excluded from these experiments.

![Historical cosine and NLI divergence comparison](analysis/fig_C_divergence_detection.png)

Historical exploratory visualization; interpret it alongside the sample sizes and provenance notes above. Recompute the checked-in summaries without model calls:

```bash
python benchmark/summarize_results.py
```

Run a new experiment with:

```bash
python benchmark/run_experiment.py \
  --variants full_system single_llm nli_detection \
  --limit 10 --max-rounds 3 --delay 30
```

Model-backed runs are intentionally not part of CI because they require credentials, downloads, time, and cost.
Benchmark variants pin their detector explicitly so the cosine baseline and NLI ablation remain distinct even though the production application defaults to NLI.

## Known limitations

- Small, non-random benchmark sets; most variants have one run per question
- LLM-as-judge results are sensitive to the selected judge and rubric
- The cosine detector measures semantic proximity, not logical opposition
- NLI adds local model download and inference cost
- The Debate Agreement Index is a heuristic, not a calibrated probability of correctness or proof of consensus
- Prompt constraints can suppress legitimate uncertainty as well as empty hedging
- Older reports without detector metadata remain unknown; saved model IDs identify the configured loader model, not a pinned revision of its weights

## Project structure

```text
app.py                         Streamlit UI
debate/                        graph, schemas, prompts, routing, persistence
examples/offline_debate.json    fixed synthetic report and provenance
docs/                          case walkthrough, exported example, UI screenshot
benchmark/                     experiment runners, variants, evaluators
results/                       raw outputs plus provenance notes
analysis/                      notebooks and generated figures
tests/                         deterministic, model, UI, and integration tests
.github/workflows/ci.yml       deterministic CI
PAPER.md                       exploratory research-style write-up
BLOG_EN.md / BLOG_ZH.md        long-form engineering notes
```

## Observability

LangSmith tracing is optional. Add the following to `.env`:

```dotenv
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=multi-agent-debate
```

Traces may contain full prompts and outputs; do not enable them for sensitive topics without reviewing your data-handling requirements.

## Maintenance and contributions

The app currently includes a bundled offline example, searchable history, and Markdown/JSON export. Maintenance focuses on reliable inspection and replay, backward compatibility, and deterministic regression coverage. Model-backed experiments remain separate from CI; the [known limitations](#known-limitations) and [result provenance](results/README.md) apply to existing reports and benchmarks.

### Saved detector settings

Each new round records its detector mode, model ID, input field, pair-decision threshold, applicable fast-path threshold or NLI label index, sentinel filtering policy, and number of evaluated agent pairs. Settings are captured at evaluation and retained in `reasoning_trace[].detector_metadata` through report assembly, SQLite storage, history, and Markdown/JSON export. The settings section shows rounds individually, including runs with different detectors across rounds.

A zero evaluated-pair count means no pairwise model inference took place. Explicit benchmark variants retain their selected NLI/cosine detector independently of the app default; the full-text variant records `reasoning` input and its sentinel filter. Scoring formulas and existing saved values are unchanged. Model IDs do not pin weight revisions or establish model quality.

Older JSON with no metadata still loads as **unknown**. Current environment settings never fill those gaps, browsing does not rewrite history, and the bundled synthetic example remains unknown with its synthetic source label. This is an additive report field; no database migration or historical backfill is needed.

Regression checks and contribution entry points:

| Task | Starting points | Acceptance |
|---|---|---|
| Capture and persistence | [Schema](debate/state.py), [detector/storage tests](tests/test_detector_metadata.py), [benchmark tests](tests/test_detector_benchmark_metadata.py) | Fixed local model outputs verify unchanged decisions and precise settings, including zero evaluations; new and legacy reports survive SQLite save/reload. No model downloads or live inference. |
| Replay and exports | [UI](app.py), [serializer](debate/export.py), [export tests](tests/test_report_export.py), [offline UI tests](tests/test_phase5.py) | Screen, Markdown, and JSON preserve saved settings and scores after current settings change; unknown rounds and synthetic provenance remain explicit, with no model calls or history writes. |

### Report an issue or send a pull request

Include the OS, Python/dependency versions, commit, minimal reproduction, expected versus actual behavior, and the checks you actually ran. Use a small synthetic report when data is needed. For a PR, include the relevant offline tests and [deterministic suite](#tests) results, and state whether any model/API calls or downloads occurred. Do not attach API keys, `.env` files, or private report contents.

[Open an issue](https://github.com/xyma2003/multi-agent-debate/issues) or [submit a pull request](https://github.com/xyma2003/multi-agent-debate/pulls).

MIT License.
