# Multi-Agent Debate System

[![CI](https://github.com/xyma2003/multi-agent-debate/actions/workflows/ci.yml/badge.svg)](https://github.com/xyma2003/multi-agent-debate/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文说明](README_zh.md)

A LangGraph portfolio project in which three methodology-driven LLM agents—an Optimist, a Pessimist, and a Frame Challenger—analyze a question independently, rebut one another, record attributed concessions, and produce an auditable report.

The default application uses NLI contradiction detection for multi-round routing. A lightweight cosine mode remains available for comparison, but it is not a reliable stance detector.

## What this project demonstrates

- Parallel LangGraph fan-out/fan-in with bounded rebuttal loops
- Pydantic structured outputs and validation fallbacks
- NLI-based contradiction detection with caller-controlled round limits
- A formula-derived Debate Agreement Index and attributed concession logs
- SQLite persistence, Streamlit replay, and optional LangSmith traces
- Reproducible benchmark runners for prompt and routing ablations

## Illustrative flow

```text
topic
  └─► initialize
       └─► Optimist ─┐
           Pessimist ├─► collect ─► divergence check
           Challenger─┘                    │
                     ┌─ no contradiction / limit ─► synthesize ─► save
                     └─ contradiction ─► rebuttal fan-out ─┘
```

The final `DebateReport` contains consensus points, disputed points, a verdict, the Debate Agreement Index, the complete round trace, and every concession with its triggering agent and claim. The index is a heuristic about detected disagreement, not the probability that the verdict is correct.

![Cosine and NLI divergence comparison](analysis/fig_C_divergence_detection.png)

Historical exploratory visualization; see the evidence and provenance notes below before interpreting it.

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

In NLI mode, each round's divergence is the mean of the highest claim-pair contradiction probabilities for each agent pair; low contradiction can also mean neutral or unrelated claims. Cosine mode measures semantic proximity, which can be high even for opposing claims. Do not compare index values across these modes. Saved reports do not record the detector mode, so the current application setting cannot establish which mode produced a historical report.

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

The clean base comparison contains 10 questions per system:

| System | n | PDS ↑ | Hedge ratio ↓ | Mean rounds |
|---|---:|---:|---:|---:|
| Fixed frame-challenger prompt | 10 | 0.2242 | 0.0093 | 1.00 |
| Single-LLM baseline | 10 | 0.2160 | 0.0129 | 1.00 |
| Original challenger prompt | 10 | 0.1707 | 0.0077 | 1.00 |

In this sample, the fixed prompt produced higher PDS than the original prompt, and the multi-agent output's mean hedge ratio was about 28% lower than the single-LLM baseline. These are observed sample differences, not causal or generalizable performance claims.

A separate canonical NLI run contains 7 questions: 4/7 reached multiple rounds, mean rounds were 2.14, and mean stance stability was 0.977. See [`results/README.md`](results/README.md) for dataset provenance, validation failures, and interpretation rules.

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

## Project structure

```text
app.py                         Streamlit UI
debate/                        graph, schemas, prompts, routing, persistence
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

## Resume description

> Built a LangGraph multi-agent debate system with parallel role orchestration, structured outputs, NLI-based divergence routing, attributed concessions, SQLite replay, and an ablation-oriented evaluation pipeline for prompt and termination strategies.

## Status

Core development and experiments: **April–June 2026**. Later commits are maintenance, backend compatibility, and observability updates. The project is feature-complete and maintained in stabilization mode.

MIT License.
