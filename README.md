# Multi-Agent Debate System

A LangGraph-based multi-agent system where three LLM agents with distinct cognitive biases — Optimist, Pessimist, and Devil's Advocate — debate any topic through multiple rounds of structured argumentation. Agents detect real semantic divergence, track concessions with attribution, and produce an auditable consensus report with a formula-derived confidence score.

Built as a portfolio project demonstrating: multi-agent LangGraph graphs, semantic divergence detection, Pydantic structured outputs, SQLite persistence, and Streamlit streaming UI.

## Prerequisites

- Python 3.10+
- Anthropic API key: `export ANTHROPIC_API_KEY=sk-ant-...`

## Installation

```bash
git clone <repo-url>
cd debate-agent
pip install -r requirements.txt
```

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # if not already set
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. Enter any topic or question in the input field (e.g. "Is remote work net positive for companies?")
2. Set the max debate rounds (1-3) — more rounds means more rebuttal cycles
3. Click **Start Debate** and watch the three agents argue in real time
4. Read the final report: confidence score, consensus points, disputed points, full reasoning trace
5. Past debates appear in the sidebar for instant replay without re-running agents

## How It Works

```
User topic
    │
    ▼
initialize ──► [Optimist | Pessimist | Devil's Advocate] (Round 1, parallel)
                    │
                    ▼
            collect_round1
                    │
                    ▼
         divergence_check_node (semantic similarity on key_claims)
                    │
          ┌─────────┴─────────┐
       diverged           converged / max_rounds
          │                    │
   [rebuttal round]      synthesize_stub
          │                    │
    (loop back)           save_node → SQLite
                               │
                           DebateReport
```

- **Divergence detection**: `sentence-transformers` + cosine similarity on `key_claims` embeddings. Score > 0.75 triggers another round.
- **Confidence score**: Formula-derived — `(1 - max_divergence_score) * round_adjustment`. Never LLM-invented.
- **Persistence**: Completed debates saved to `debates.db` (SQLite). Replayable from sidebar.

## Project Structure

```
debate-agent/
├── app.py                   # Streamlit UI (Phase 5)
├── requirements.txt         # All Python dependencies
├── debates.db               # Auto-created on first debate run
├── debate/
│   ├── graph.py             # StateGraph assembly + module-level `graph` singleton
│   ├── state.py             # DebateState TypedDict + all Pydantic models
│   ├── store.py             # SQLite save/load/list API
│   ├── divergence.py        # compute_divergence() with sentence-transformers
│   └── nodes/
│       ├── agents.py        # optimist_node, pessimist_node, devil_node
│       ├── collect.py       # collect_round1 fan-in node
│       ├── dispatch.py      # dispatch_round1, route_divergence routing functions
│       ├── divergence_check.py
│       ├── initialize.py
│       ├── save.py          # save_node (SQLite side-effect, no state mutation)
│       └── synthesize.py    # synthesize_stub → DebateReport assembly
└── tests/
    ├── test_phase1.py
    ├── test_phase2.py
    ├── test_phase3.py
    ├── test_phase4.py
    └── test_phase5.py       # UI tests (AppTest + unit)
```

## Tech Stack

| Component | Library |
|-----------|---------|
| Agent orchestration | LangGraph 1.1.9 |
| LLM | Claude (via langchain-anthropic) |
| Structured outputs | Pydantic 2.x + `llm.with_structured_output()` |
| Divergence detection | sentence-transformers 5.4.1 + bge-small-en-v1.5 |
| Persistence | SQLite (stdlib) |
| UI | Streamlit 1.56.0 |
