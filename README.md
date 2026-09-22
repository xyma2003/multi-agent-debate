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
- Formula-derived confidence scores and attributed concession logs
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

The final `DebateReport` contains consensus points, disputed points, a verdict, a formula-derived confidence score, the complete round trace, and every concession with its triggering agent and claim.

![Cosine and NLI divergence comparison](analysis/fig_C_divergence_detection.png)

Historical exploratory visualization; see the evidence and provenance notes below before interpreting it.

## Quick start

Requirements: Python 3.10+ and one supported LLM API key.

```bash
git clone https://github.com/xyma2003/multi-agent-debate.git
cd multi-agent-debate

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
streamlit run app.py
```

Open <http://localhost:8501>.

The default NLI mode downloads `cross-encoder/nli-deberta-v3-small` (~180 MB) on first use. The optional cosine comparison mode downloads `BAAI/bge-small-en-v1.5` (~130 MB).

### Configuration

The checked-in `.env.example` documents all options. A minimal SiliconFlow/OpenAI-compatible setup is:

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
| Formula-derived confidence | Keeps the displayed score inspectable rather than LLM-invented |
| Concession attribution | Records which claim changed which position and why |

## Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the deterministic suite used by CI:

```bash
python -m pytest -m "not integration and not model and not ui" -q
```

Optional suites:

```bash
# Downloads local embedding/NLI models
python -m pytest -m model -v

# Requires configured API credentials
python -m pytest -m integration -v

# Streamlit application tests
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
- Confidence is a transparent heuristic, not a calibrated probability of correctness
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
