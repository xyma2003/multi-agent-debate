# Project: Multi-Agent Debate System

## What This Is

A multi-agent debate system where specialized LLM agents with distinct cognitive biases analyze a topic independently, then engage in structured argumentation with divergence detection and concession tracking, producing auditable consensus reports with confidence scoring.

**Core value:** Given any topic or question, produce a more reliable, multi-perspective analysis than a single LLM can — by having agents with different "personalities" challenge each other, detect real disagreements, and reach a traceable consensus.

## Why It Exists

Single LLMs tend to "self-agree" — they rarely challenge their own conclusions. This system forces adversarial reasoning by design. The output is not just an answer but an auditable reasoning trace showing *why* the consensus was reached and *what* remains disputed.

**Primary use case:** Portfolio project demonstrating multi-agent coordination, structured reasoning, and production-thinking (verification, confidence scoring, audit trails). Target application domain: investment/financial analysis reports, but architecture is domain-agnostic.

## Who It's For

- Demo audience: recruiters and engineers reviewing the portfolio
- End user (for demo): anyone who wants a critical multi-perspective analysis of a topic (e.g., "Should I invest in X?", "Is this business idea viable?")

## What "Done" Looks Like

A working Streamlit web app where:
1. User enters a topic/question
2. System shows agents analyzing in parallel (visual feedback)
3. Agents debate — rounds visible in UI
4. Final report shows: consensus points, disputed points, confidence score, full reasoning trace
5. Code is clean, documented, and demo-ready for a portfolio

## Agent Roster

| Agent | Role | Cognitive Bias |
|-------|------|---------------|
| Optimist | Opportunity analyst | Finds upside, underweights risk |
| Pessimist | Risk analyst | Finds downside, challenges assumptions |
| Devil's Advocate | Challenger | Actively attacks the current majority view |
| Synthesizer | Final arbiter | No bias, weighs evidence to reach verdict |

## Key Technical Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| LangGraph for orchestration | User has prior experience, handles stateful multi-step flows natively | Chosen |
| Claude API (claude-sonnet-4-6) | Strong reasoning, reliable structured output | Chosen |
| Pydantic for structured outputs | Ensures each agent round is parseable | Chosen |
| Streamlit for frontend | Fast demo UI, user has experience | Chosen |
| sentence-transformers for divergence detection | Semantic similarity for detecting real vs surface disagreements | Chosen |
| SQLite for debate history | Lightweight persistence, supports replay | Chosen |

## Requirements

### Validated

- ✓ 3 agents (Optimist, Pessimist, Devil's Advocate) independently analyze a topic with no cross-visibility — validated in Phase 1
- ✓ Each agent produces structured output: position, reasoning, key_claims, confidence, concessions — validated in Phase 1
- ✓ Methodology-based persona prompts with anti-sycophancy PROHIBITION blocks — validated in Phase 1
- ✓ Pydantic retry wrapper (2 retries + sentinel injection) prevents graph crashes on parse failure — validated in Phase 1

### Active

- [ ] User can enter any topic/question and trigger a debate
- [ ] 3+ agents analyze independently in Round 1 (no cross-visibility)
- [ ] Debate Engine detects real disagreements (semantic, not surface-level)
- [ ] Multi-round debate loop: agents see each other's arguments and rebut
- [ ] Agents can "concede" points with logged reasoning
- [ ] Synthesizer produces final verdict with confidence score
- [ ] Output: consensus points, disputed points, confidence score
- [ ] Full reasoning trace is stored and viewable
- [ ] Streamlit UI shows live debate progress
- [ ] Demo-ready: clean code, README, example outputs

### Out of Scope

- Authentication / user accounts — not needed for portfolio demo
- Real-time financial data integration — Phase 1 is domain-agnostic text input
- Production deployment / scaling — demo-first
- More than 4 agents in v1 — keep it simple and explainable

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?

---
*Last updated: 2026-04-24 after Phase 1 (Graph Foundation) completion*
