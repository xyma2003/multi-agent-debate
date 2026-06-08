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

**Option B — Internal proxy (e.g. corporate proxy):**

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
├── benchmark/
│   ├── questions.json        # 30 benchmark questions (business/tech/policy/prediction)
│   ├── evaluator.py          # PDS / HR / SSS / RTC metric definitions
│   ├── baseline.py           # Single-LLM runner
│   ├── variants.py           # 6 ablation variants
│   └── run_experiment.py     # CLI entry point
├── results/
│   ├── full_system.json      # Multi-agent fixed devil (n=10)
│   ├── original_devil.json   # Multi-agent old devil (n=10)
│   ├── single_llm.json       # Single-LLM baseline (n=10)
│   └── nli_detection.json    # NLI divergence (n=2)
├── analysis/
│   ├── analysis.ipynb        # 7-section analysis notebook
│   └── fig_*.png             # Experiment figures
└── tests/
    ├── test_phase1.py        # Graph foundation + smoke test
    ├── test_phase2.py        # Debate loop + divergence detection
    ├── test_phase3.py        # Synthesis + confidence formula
    ├── test_phase4.py        # SQLite persistence + replay
    └── test_phase5.py        # UI tests
```

---

## Experimental Findings

Ablation study across 4 system variants, 10 questions each (business + technology topics).

| Variant | n | PDS ↑ | HR ↓ | SSS | Rounds |
|---------|---|-------|------|-----|--------|
| **Multi-agent (fixed devil)** | 10 | **0.2242** | **0.0093** | 1.000 | 1.00 |
| Single-LLM baseline | 10 | 0.2160 | 0.0129 | N/A | 1.00 |
| Multi-agent (old devil prompt) | 10 | 0.1707 | 0.0077 | 1.000 | 1.00 |
| Multi-agent + NLI detection | 2 | 0.1439 | 0.0050 | **0.883** | **3.00** |

- **PDS** (Position Diversity Score): avg pairwise semantic distance between agents' final positions. Higher = more genuinely distinct viewpoints.
- **HR** (Hedge Ratio): hedge words / total words. Lower = less "on-the-other-hand" hedging.
- **SSS** (Stance Stability Score): similarity between Round-1 and final position embedding. Only meaningful in multi-round debates.

### Key findings

**Finding A — PROHIBITION reduces sycophantic hedging by 28%**
Multi-agent HR (0.0093) vs single-LLM HR (0.0129). The PROHIBITION constraints successfully prevent agents from retreating to balanced, non-committal language.

**Finding B — PDS Paradox: wrong devil prompt inverts diversity**
Old devil prompt ("challenge the dominant view") caused 2-vs-1 alignment — devil auto-sided with pessimist against optimist, producing *lower* PDS than single-LLM (0.1707 < 0.2160). Fixed by redefining devil's role as "Assumption Challenger" who targets the shared premise both sides take for granted. Post-fix PDS (0.2242) exceeds single-LLM baseline.

**Finding C — Cosine similarity is broken for stance detection**
100% of cosine-based debates terminated after Round 1 (divergence scores 0.097–0.258, all below 0.75 threshold). Cosine measures *topic overlap*, not *stance opposition* — "VC accelerates growth" and "VC destroys growth" score as *similar* because they share vocabulary. NLI cross-encoder correctly detects CONTRADICTION regardless of vocabulary overlap, enabling genuine multi-round debate (SSS = 0.883 vs 1.000).

See `analysis/analysis.ipynb` for full analysis with figures.

### Running the benchmark

```bash
# Requires VPN if using Groq backend
cd debate-agent

# Run all variants (n=10 each, 2-min delay between questions for rate limits)
python benchmark/run_experiment.py --variants full_system single_llm --limit 10 --delay 5
python benchmark/run_experiment.py --variants nli_detection --limit 10 --delay 120

# View results summary
python -c "
import json, statistics
for v in ['full_system', 'single_llm', 'original_devil', 'nli_detection']:
    with open(f'results/{v}.json') as f: d = json.load(f)
    pds = [r['pds'] for r in d['results']]
    hr  = [r['hedge_ratio'] for r in d['results']]
    print(f'{v}: n={len(pds)}  PDS={statistics.mean(pds):.4f}  HR={statistics.mean(hr):.4f}')
"
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
| LLM (default) | Groq `llama-3.3-70b-versatile` via `LLM_BACKEND=groq` | — |
| LLM (alt) | Claude / OpenAI via `LLM_BACKEND=anthropic\|openai` | — |
| Structured outputs | Pydantic | 2.x |
| Divergence (cosine) | sentence-transformers + bge-small-en-v1.5 | 5.4.1 |
| Divergence (NLI) | sentence-transformers + cross-encoder/nli-deberta-v3-small | 5.4.1 |
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
