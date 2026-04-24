# Multi-Agent Debate System

A LangGraph-based multi-agent system where three LLM agents with distinct cognitive biases — **Optimist**, **Pessimist**, and **Devil's Advocate** — debate any topic through multiple rounds of structured argumentation. Agents detect real semantic divergence, track concessions with attribution, and produce an auditable consensus report with a formula-derived confidence score.

Built as a portfolio project demonstrating: multi-agent LangGraph graphs, semantic divergence detection, Pydantic structured outputs, SQLite persistence, and Streamlit streaming UI.

---

## Demo

```
User: "Is remote work net positive for companies?"

Round 1 (parallel):
  🟢 Optimist    → "Remote work increases productivity by 15-20%..."
  🔴 Pessimist   → "Collaboration and culture suffer irreparably..."
  😈 Devil's Adv → "The productivity gains are selection bias..."

Divergence score: 0.82 → Round 2 triggered

Round 2 (rebuttal):
  🟢 Optimist    → Concedes: "Culture risks are real for junior employees"
  🔴 Pessimist   → Maintains position
  😈 Devil's Adv → Shifts: "Hybrid is the actual optimum"

Final Report:
  Confidence: 71% | Status: Converged
  Consensus: ["Async communication tools are essential", ...]
  Disputed:  [{"topic": "Culture impact", "optimist": "...", "pessimist": "..."}]
```

---

## Prerequisites

- **Python 3.10+**
- **Anthropic API access** — either a direct API key or a proxy (see below)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/debate-agent.git
cd debate-agent
```

### 2. Create a virtual environment

```bash
# Option A: venv (built-in)
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Option B: conda
conda create -n debate-agent python=3.10
conda activate debate-agent
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run will download the `BAAI/bge-small-en-v1.5` embedding model (~130MB) from HuggingFace. This happens automatically on first debate start.

### 4. Configure API credentials

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

**Option A — Direct Anthropic API key (standard):**

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

**Option B — Internal proxy (e.g. corporate/Meituan proxy):**

```bash
# .env
ANTHROPIC_BASE_URL=https://your-proxy-base-url
ANTHROPIC_AUTH_TOKEN=your-auth-token
ANTHROPIC_CUSTOM_HEADERS=X-Custom-Header: value
```

> The app auto-detects which auth method to use based on which env vars are set. No code changes needed.

Then load the env file:

```bash
# macOS/Linux — add to your shell or run before streamlit:
export $(grep -v '^#' .env | xargs)

# Or use python-dotenv (already loaded by the app if .env exists):
pip install python-dotenv   # one-time, optional
```

### 5. Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Usage

1. Enter any topic or question (e.g. *"Is AI regulation good for innovation?"*)
2. Set **Max Rounds** (1–3) — more rounds = more rebuttal cycles
3. Click **Start Debate** — watch agents argue in real time
4. Read the final report: confidence score, verdict, consensus/disputed split, reasoning trace
5. Past debates appear in the **sidebar** for instant replay without re-running agents

---

## How It Works

```
User topic
    │
    ▼
initialize ──► [Optimist | Pessimist | Devil's Advocate]  (Round 1, parallel)
                    │
                    ▼
            collect_round1
                    │
                    ▼
     divergence_check_node  ← semantic similarity on key_claims embeddings
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

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| Methodology-based personas | "You apply bear-case scenario analysis" beats "be pessimistic" — prevents sycophancy collapse |
| Divergence on `key_claims` (not full text) | Full argument embeddings cluster by topic; claim-level embeddings preserve disagreement signal |
| Confidence formula in code | `(1 - max_divergence) * round_adjustment` — never LLM-invented, always auditable |
| Concession attribution | Each concession records `triggered_by_agent` + `triggered_by_claim` — full reasoning chain |
| Single flat StateGraph | No subgraph nesting — explicit state control and checkpointing for auditable trace |

---

## Project Structure

```
debate-agent/
├── app.py                    # Streamlit UI — single-file app
├── requirements.txt          # Pinned dependencies
├── .env.example              # API credential template
├── debates.db                # Auto-created SQLite DB on first run
├── debate/
│   ├── graph.py              # StateGraph assembly + compiled graph singleton
│   ├── state.py              # DebateState TypedDict + all Pydantic models
│   ├── store.py              # SQLite save / load / list API
│   ├── divergence.py         # compute_divergence() with sentence-transformers
│   ├── llm.py                # Auth-aware ChatAnthropic factory + retry wrapper
│   ├── prompts.py            # Methodology-based system prompts (PROHIBITION blocks)
│   └── nodes/
│       ├── initialize.py     # Sets debate_id, round_num=0
│       ├── agents.py         # optimist_node, pessimist_node, devil_node
│       ├── dispatch.py       # dispatch_round1 + route_divergence routing functions
│       ├── collect.py        # collect_round1 fan-in (reused for all rounds)
│       ├── divergence_check.py
│       ├── synthesize.py     # Synthesizer → DebateReport assembly
│       └── save.py           # save_node (SQLite side-effect, returns {})
└── tests/
    ├── test_phase1.py        # Graph foundation + smoke test
    ├── test_phase2.py        # Debate loop + divergence detection
    ├── test_phase3.py        # Synthesis + confidence formula
    ├── test_phase4.py        # SQLite persistence + replay
    └── test_phase5.py        # UI tests
```

---

## Running Tests

```bash
# Fast unit tests only (no API calls, ~5 seconds)
python -m pytest tests/ -m "not integration" -v

# Full suite including live LLM calls (~5 minutes)
python -m pytest tests/ -v
```

---

## Tech Stack

| Component | Library | Version |
|-----------|---------|---------|
| Agent orchestration | LangGraph | 1.1.9 |
| LLM | Claude via langchain-anthropic | 1.4.1 |
| Structured outputs | Pydantic | 2.x |
| Divergence detection | sentence-transformers + bge-small-en-v1.5 | 5.4.1 |
| Persistence | SQLite (stdlib) | — |
| UI | Streamlit | 1.56.0 |

---

## Publishing to GitHub

```bash
# 1. Create a new repo on github.com (do NOT initialize with README)

# 2. Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/debate-agent.git
git branch -M main
git push -u origin main
```

> Make sure `.env` is in `.gitignore` (it is by default) — never commit API keys.

---

## Resume

Built as a portfolio project to demonstrate multi-agent LLM system design.

**Resume bullet:**
> *Built a multi-agent debate system where specialized LLM agents with distinct cognitive biases analyze topics independently, then engage in structured argumentation with divergence detection and concession tracking, producing auditable consensus reports with confidence scoring. (LangGraph · Claude API · Pydantic · Streamlit · SQLite)*
