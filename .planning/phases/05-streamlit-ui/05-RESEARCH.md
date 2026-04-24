# Phase 5: Streamlit UI — Research

**Researched:** 2026-04-23
**Domain:** Streamlit 1.56.0 + LangGraph 1.1.9 streaming integration
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| UI-01 | Streamlit app with topic input field and "Start Debate" button | `st.text_input` + `st.button` pattern with session state guard; verified in installed source |
| UI-02 | Live debate progress shown as agents complete each round (streaming via graph.stream) | `graph.stream(stream_mode="updates")` yields `{node_name: state_dict}` per node completion; dispatch by key |
| UI-03 | Final report with consensus/disputed split, confidence score, expandable reasoning trace | `st.metric`, `st.progress`, `st.expander`, columnar layout via `st.columns` — all in 1.56.0 |
| UI-04 | Demo-ready: clean layout, no broken states, works end-to-end on first try | Session state machine (idle/running/complete/error); thread_id reset per debate; error boundary with st.error + reset button |
</phase_requirements>

---

## Summary

Phase 5 builds a single-file Streamlit app (`app.py` at project root) that wraps the completed LangGraph debate graph. The app streams agent outputs round by round using the synchronous `graph.stream(stream_mode="updates")` API, then renders the final `DebateReport` with structured visual blocks for consensus points, disputed points, confidence score, and an expandable reasoning trace.

The critical integration decision is **synchronous streaming only**. PITFALLS.md confirms this with source-level verification: `graph.stream()` uses `SyncPregelLoop` with zero asyncio dependency. Streamlit's script thread has no running event loop. Never use `astream()` or `asyncio.run()` — they cause `RuntimeError` due to Tornado loop conflicts.

State management follows a simple four-state machine in `st.session_state`: `idle → running → complete / error`. The graph result and thread_id are stored in session state so they survive Streamlit reruns triggered by button clicks. The SQLite `list_debates()` function provides the past debates sidebar with zero extra work.

**Primary recommendation:** Single `app.py` at project root; `st.status()` containers per agent per round updated as stream chunks arrive; `st.metric` + `st.progress` for confidence score; `st.expander` for reasoning trace. No async code anywhere.

---

## Project Constraints (from CLAUDE.md)

The project CLAUDE.md is at `/Users/maxinyue09/Downloads/projects/项目/debate-agent/CLAUDE.md`.

| Constraint | Directive |
|------------|-----------|
| Tech stack locked | Python + LangGraph + Claude API + Streamlit + SQLite — no alternatives |
| UI framework | Streamlit only — fast prototype, no complex frontend |
| LLM | Anthropic Claude API (langchain-anthropic wrapper) |
| GSD workflow | Use `/gsd:execute-phase` for planned phase work; no direct edits outside GSD |

---

## Standard Stack

### Core (already in requirements.txt or installed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| streamlit | 1.56.0 | Web UI framework | Project-locked; latest stable; confirmed installed in llm-data-pipeline conda env |
| langgraph | 1.1.9 | Graph execution + `stream()` | Project-locked; already in requirements.txt |
| debate.graph | — | `from debate.graph import graph` | Module-level compiled graph; import once, reuse across sessions |
| debate.store | — | `list_debates()`, `load_debate()` | Already implemented in Phase 4; public API confirmed |
| debate.state | — | `DebateReport`, `AgentArgument`, `RoundRecord` | All Pydantic models needed for rendering |

### No New Dependencies Required

Phase 5 adds zero new packages. All rendering uses Streamlit built-ins. The debate execution path uses `debate.graph.graph` which was already wired in Phase 4.

**Installation:** `pip install streamlit==1.56.0` (only if not already present)

**Version verification (confirmed):**
```
streamlit 1.56.0  — confirmed via pip3 index versions, installed in conda env
langgraph 1.1.9   — confirmed in requirements.txt
```

---

## Architecture Patterns

### Recommended Project Structure

```
debate-agent/
├── app.py                    # Single-file Streamlit app (Phase 5 deliverable)
├── .streamlit/
│   └── config.toml           # Optional: server port, theme
├── debate/
│   ├── graph.py              # graph = build_graph() (already exists)
│   ├── state.py              # DebateReport, AgentArgument, RoundRecord (already exists)
│   └── store.py              # save_debate, load_debate, list_debates (already exists)
└── debates.db                # Auto-created by store.py
```

### Pattern 1: session_state as Four-State Machine

**What:** `st.session_state` holds the UI state machine. Never store mutable state outside session_state.

**States:** `idle` → `running` → `complete` | `error`

**What goes in session_state:**

| Key | Type | Purpose |
|-----|------|---------|
| `debate_status` | `str` | `"idle"` / `"running"` / `"complete"` / `"error"` |
| `thread_id` | `str` | UUID for the current LangGraph run; new UUID per debate |
| `final_report` | `DebateReport \| None` | Populated after graph completes |
| `stream_log` | `list[dict]` | Per-node update log for replay in "complete" state |
| `error_msg` | `str \| None` | Error message if `debate_status == "error"` |

**Initialization pattern:**
```python
# Source: verified session_state pattern from installed streamlit source
if "debate_status" not in st.session_state:
    st.session_state.debate_status = "idle"
    st.session_state.thread_id = None
    st.session_state.final_report = None
    st.session_state.stream_log = []
    st.session_state.error_msg = None
```

**Why this is correct:** Streamlit reruns the entire script on every user interaction. Session state persists across reruns within the same browser session. Using a status field prevents re-triggering the graph on reruns caused by unrelated widget interactions.

### Pattern 2: graph.stream() Dispatch by Node Name

**What:** `stream_mode="updates"` yields one dict per node completion. Each dict has the node name as key and the node's state update as value.

**Verified chunk format** (from `map_output_updates` in `langgraph/pregel/_io.py`):
```python
# Each chunk is: {node_name: state_update_dict}
# Example chunks from the debate graph:
{"initialize": {"debate_id": "abc123", "round_num": 0, "status": "running"}}
{"optimist_node": {"current_round_arguments": [AgentArgument(...)]}}
{"pessimist_node": {"current_round_arguments": [AgentArgument(...)]}}
{"devil_node": {"current_round_arguments": [AgentArgument(...)]}}
{"collect_round1": {"round_history": [...], "round_num": 1}}
{"divergence_check_node": {"divergence_score": 0.42, "diverged_pairs": [...]}}
{"synthesize_stub": {"final_report": DebateReport(...), "status": "converged"}}
{"save_node": {}}  # save_node writes to SQLite, returns no state update
```

**Dispatch pattern:**
```python
# Source: verified from LangGraph _io.py map_output_updates + PITFALLS.md Pitfall 6
config = {"configurable": {"thread_id": st.session_state.thread_id}}
for chunk in graph.stream(
    {"topic": topic, "max_rounds": 3},
    config=config,
    stream_mode="updates",
):
    for node_name, node_update in chunk.items():
        if node_name in ("optimist_node", "pessimist_node", "devil_node"):
            _render_agent_chunk(node_name, node_update)
        elif node_name == "collect_round1":
            _render_round_complete(node_update)
        elif node_name == "divergence_check_node":
            _render_divergence(node_update)
        elif node_name == "synthesize_stub":
            st.session_state.final_report = node_update.get("final_report")
        # initialize, save_node: no UI action needed
```

**CRITICAL:** Call `graph.stream()` in a plain `for` loop — synchronous. Never wrap with `asyncio.run()`, `loop.run_until_complete()`, or `await`. See PITFALLS.md Pitfall 6.

### Pattern 3: st.status() Containers for Live Agent Feed

**What:** `st.status()` creates a collapsible container with a spinner (running), checkmark (complete), or error icon. Auto-transitions to complete on `with` block exit.

**Verified API** (from `streamlit/elements/layouts.py:1677` and `mutable_status_container.py`):
```python
# st.status(label, expanded=False, state="running") → StatusContainer
# StatusContainer.update(label=None, expanded=None, state=None)
# States: "running" | "complete" | "error"
# Auto-completes when used as context manager (with block exits normally)
# Auto-errors when used as context manager and exception is raised
```

**Live feed pattern** (do NOT use `with` notation during streaming — `with` auto-closes on exit):
```python
def _render_agent_chunk(node_name: str, update: dict) -> None:
    role_labels = {
        "optimist_node": "Optimist",
        "pessimist_node": "Pessimist",
        "devil_node": "Devil's Advocate",
    }
    label = role_labels[node_name]
    args: list = update.get("current_round_arguments", [])
    if not args:
        return
    arg = args[-1]  # The agent's AgentArgument for this round

    # Create status container (starts in "running" state)
    status = st.status(f"{label} — Round {arg.round_num + 1}", expanded=True)
    status.markdown(f"**Position:** {arg.position}")
    status.markdown(f"**Confidence:** {arg.confidence:.0%}")
    if arg.key_claims:
        status.markdown("**Key Claims:**")
        for claim in arg.key_claims:
            status.markdown(f"- {claim}")
    if arg.concessions:
        status.markdown("**Concessions this round:**")
        for c in arg.concessions:
            status.markdown(f"- {c.conceded_point} *(triggered by {c.triggered_by_agent})*")
    status.update(state="complete", expanded=False)
```

**Note:** The `time.sleep(0.05)` in StatusContainer._create and `__exit__` is built into the Streamlit source to prevent race conditions on fast updates. No manual sleep needed.

### Pattern 4: Final Report Layout

**DebateReport fields to render:**

| Field | Widget | Notes |
|-------|--------|-------|
| `confidence_score` | `st.metric` + `st.progress` | `st.metric("Confidence", f"{score:.0%}")`; `st.progress(score)` |
| `verdict` | `st.info()` or `st.markdown` in a callout | Prominent placement after score |
| `consensus_points` | `st.success()` block per point or bulleted list | Green background for agreement |
| `disputed_points` | `st.warning()` block per point with agent positions | Yellow/orange for unresolved |
| `convergence_status` | `st.badge()` or inline text | `"converged"` / `"max_rounds"` / `"partial"` |
| `reasoning_trace` | `st.expander("Full Reasoning Trace")` | Contains per-round RoundRecord display |
| `concession_log` | Inside expander or separate `st.expander` | Flattened Concession list |

**Report rendering pattern:**
```python
def render_report(report: DebateReport) -> None:
    st.subheader("Debate Result")

    # Confidence score — prominent
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Confidence Score", f"{report.confidence_score:.0%}")
    with col2:
        st.progress(report.confidence_score)
        status_color = {"converged": "green", "partial": "orange", "max_rounds": "red"}
        st.caption(f"Status: **{report.convergence_status}**")

    # Verdict
    st.info(report.verdict)

    # Consensus vs disputed
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Points of Consensus")
        if report.consensus_points:
            for point in report.consensus_points:
                st.success(point)
        else:
            st.caption("No consensus reached.")
    with col_b:
        st.markdown("### Disputed Points")
        for dp in report.disputed_points:
            with st.warning(dp.topic):
                pass  # Use st.warning as block; add positions below
            for role, pos in dp.agent_positions.items():
                st.caption(f"**{role.title()}:** {pos}")

    # Expandable trace
    with st.expander("Full Reasoning Trace", expanded=False):
        for rr in report.reasoning_trace:
            st.markdown(f"**Round {rr.round_num + 1}** — divergence: `{rr.divergence_score:.3f}`")
            for arg in rr.arguments:
                st.markdown(f"- **{arg.agent_role.title()}:** {arg.position}")

    if report.concession_log:
        with st.expander("Concession Log", expanded=False):
            for c in report.concession_log:
                st.markdown(
                    f"- **{c.conceded_point}** "
                    f"*(conceded in response to {c.triggered_by_agent}: {c.triggered_by_claim})*"
                )
```

### Pattern 5: app.py Top-Level Structure

```python
# app.py — complete structure
import uuid
import streamlit as st
from debate.graph import graph
from debate.state import DebateReport
from debate.store import list_debates, load_debate

st.set_page_config(page_title="Multi-Agent Debate", layout="wide")

# --- Session state init (runs on every fresh browser session) ---
if "debate_status" not in st.session_state:
    st.session_state.debate_status = "idle"
    st.session_state.thread_id = None
    st.session_state.final_report = None
    st.session_state.error_msg = None

# --- Sidebar: past debates ---
with st.sidebar:
    st.header("Past Debates")
    past = list_debates()  # returns list[dict] newest-first
    for row in past[:10]:  # show last 10
        if st.button(row["topic"][:40], key=row["debate_id"]):
            loaded = load_debate(row["debate_id"])
            if loaded:
                st.session_state.final_report = loaded
                st.session_state.debate_status = "complete"

# --- Main area ---
st.title("Multi-Agent Debate System")
st.caption("Enter any topic and watch three agents debate it in real time.")

# --- Topic input (only shown when idle or complete) ---
topic = st.text_input(
    "Debate topic",
    placeholder="e.g. 'Should remote work be the default?'",
    disabled=(st.session_state.debate_status == "running"),
)

start_clicked = st.button(
    "Start Debate",
    disabled=(st.session_state.debate_status == "running" or not topic.strip()),
)

# --- Debate execution (fires when button clicked) ---
if start_clicked and topic.strip():
    st.session_state.debate_status = "running"
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.final_report = None
    st.session_state.error_msg = None

    try:
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        st.markdown("### Debate in Progress")
        for chunk in graph.stream(
            {"topic": topic.strip(), "max_rounds": 3},
            config=config,
            stream_mode="updates",
        ):
            for node_name, node_update in chunk.items():
                # dispatch by node name (see Pattern 2)
                ...
        st.session_state.debate_status = "complete"
    except Exception as exc:
        st.session_state.debate_status = "error"
        st.session_state.error_msg = str(exc)

# --- Render final report ---
if st.session_state.debate_status == "complete" and st.session_state.final_report:
    render_report(st.session_state.final_report)

# --- Error state ---
if st.session_state.debate_status == "error":
    st.error(f"Debate failed: {st.session_state.error_msg}")
    if st.button("Reset"):
        st.session_state.debate_status = "idle"
        st.rerun()
```

### Anti-Patterns to Avoid

- **Using `astream()` or `asyncio.run()`:** Causes `RuntimeError` due to Streamlit/Tornado event loop conflict. Verified from installed source. Use synchronous `graph.stream()` only.
- **Calling `graph.stream()` outside the button-click branch:** Every Streamlit rerun re-executes the whole script. If `graph.stream()` is not gated behind a state check, it fires on every widget interaction, re-running the debate.
- **Storing `graph` object in session_state:** The compiled graph is a module-level singleton (`from debate.graph import graph`). It is thread-safe and does not need to be stored in session state.
- **Using `st.rerun()` inside the streaming loop:** Calling `st.rerun()` terminates the current script run immediately, killing the stream mid-execution. Only call `st.rerun()` after the stream completes or in an error reset.
- **Not initializing session state before reading it:** If `"debate_status"` is read before the init block, Streamlit raises `KeyError` on fresh sessions.
- **`st.status()` with `with` block during a streaming loop:** The `with` block auto-closes on exit. If used inside the stream loop, the container closes after each agent, not at the end of the round. Use explicit `.update(state="complete")` instead.
- **Nesting `st.status()` containers:** Streamlit docs explicitly warn against nesting status containers — layout breaks.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Spinning "in progress" UI | Custom CSS spinner | `st.status(state="running")` | Built in, auto-transitions, verified in 1.56.0 |
| Collapsible sections | Manual `st.checkbox` + conditional render | `st.expander` | Dedicated API, preserves scroll position |
| Numeric confidence display | Custom progress bar HTML | `st.progress(float)` + `st.metric` | Both accept 0.0–1.0 float directly |
| Past debates list | Custom pagination widget | `list_debates()` from Phase 4 + sidebar buttons | Already implemented, zero extra work |
| Thread ID management | UUID in URL params | `str(uuid.uuid4())` in session state | LangGraph checkpointer just needs a unique string; no URL routing needed for demo |
| Error display | Custom error modal | `st.error(msg)` | Sufficient for demo; one-liner |

---

## Common Pitfalls

### Pitfall 1: Debate Re-fires on Rerun
**What goes wrong:** `graph.stream()` is called unconditionally at module level. Every widget interaction (e.g., clicking sidebar) triggers a Streamlit rerun, which re-fires the entire debate.
**Why it happens:** Streamlit reruns the entire script on every interaction.
**How to avoid:** Gate the streaming block behind `if start_clicked and topic.strip():` AND check `st.session_state.debate_status != "running"`. The button `start_clicked` is True only on the exact rerun triggered by the button click.
**Warning signs:** Debate starts multiple times for one topic entry.

### Pitfall 2: Blank Screen on Fresh Session (UI-04)
**What goes wrong:** App raises `KeyError` because `st.session_state.final_report` is read before the init block runs.
**How to avoid:** Always place the session state initialization block at the very top of `app.py`, before any widget that reads from session state.
**Warning signs:** `KeyError: 'debate_status'` in Streamlit logs on first load.

### Pitfall 3: stream_mode="values" Instead of "updates"
**What goes wrong:** `stream_mode="values"` yields the full state after every step — including all fields. For this graph, that means `current_round_arguments` contains all agents accumulated so far after each agent completes, causing duplicate renders.
**How to avoid:** Always use `stream_mode="updates"` for per-node dispatch. Each chunk is `{node_name: only_that_node's_update}`.
**Warning signs:** Same agent argument appears multiple times in the live feed.

### Pitfall 4: SQLite Thread Safety
**What goes wrong:** `store.get_connection()` uses `check_same_thread=False`. This is safe for single-threaded use, but `list_debates()` called from the sidebar and `save_debate()` called from `save_node` inside the stream loop must not overlap concurrently.
**How to avoid:** Since Streamlit's script runner is single-threaded per session and the graph runs synchronously, there is no actual concurrency risk. The concern only arises if Streamlit's threading is changed or background threads are added.
**Warning signs:** `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread` (only if threading is introduced).

### Pitfall 5: AgentArgument in current_round_arguments is Pydantic, Not Dict
**What goes wrong:** `node_update["current_round_arguments"]` contains `list[AgentArgument]` (Pydantic objects), not plain dicts. Calling `.get("position")` fails; must use `.position` attribute access.
**How to avoid:** Always treat chunk values as already-deserialized Pydantic models. Use `arg.position`, `arg.key_claims`, `arg.confidence`, `arg.concessions`.
**Warning signs:** `AttributeError: 'dict' object has no attribute 'position'`.

### Pitfall 6: st.status Flicker on Fast Updates
**What goes wrong:** StatusContainer source has a `time.sleep(0.05)` in `_create()` to prevent race conditions. Calling `.update()` immediately after creation can drop the first state.
**How to avoid:** This is already handled internally by Streamlit. Do not add extra sleeps. Do not call `.update()` within the same millisecond as construction — the pattern of writing content first, then calling `.update(state="complete")` at the end is safe.

### Pitfall 7: final_report is DebateReport Object, Not Dict
**What goes wrong:** `node_update` from `synthesize_stub` chunk contains `{"final_report": DebateReport(...), "status": "converged"}`. Accessing `node_update["final_report"]["consensus_points"]` fails with `TypeError`.
**How to avoid:** `st.session_state.final_report = node_update.get("final_report")` stores the Pydantic object. Use `report.consensus_points`, `report.confidence_score`, etc.

---

## Code Examples

### graph.stream() with updates mode (verified from LangGraph source)
```python
# Source: verified from langgraph/pregel/_io.py map_output_updates
# Each chunk: {node_name: state_fields_written_by_that_node}
config = {"configurable": {"thread_id": "some-uuid"}}
for chunk in graph.stream(
    {"topic": "Is AI beneficial?", "max_rounds": 3},
    config=config,
    stream_mode="updates",
):
    # chunk is a dict with exactly one key (the node that just completed)
    # Exception: parallel fan-out may yield multiple keys in one chunk
    for node_name, update in chunk.items():
        print(node_name, list(update.keys()) if isinstance(update, dict) else update)
```

### session_state initialization (verified pattern)
```python
# Source: Streamlit session_state pattern; verified from streamlit source
# Place at TOP of app.py before any widget call
if "debate_status" not in st.session_state:
    st.session_state.debate_status = "idle"
    st.session_state.thread_id = None
    st.session_state.final_report = None
    st.session_state.error_msg = None
```

### st.status() with explicit update (verified from layouts.py:1677)
```python
# Source: verified from streamlit/elements/layouts.py + mutable_status_container.py
# Do NOT use `with` block when you need to control close timing
status = st.status("Optimist — Round 1", expanded=True, state="running")
status.markdown("**Position:** AI is net positive for humanity")
status.markdown("**Confidence:** 85%")
# ... write more content ...
status.update(state="complete", expanded=False)
```

### st.metric + st.progress for confidence score
```python
# Source: verified from installed streamlit — both accept float 0.0–1.0
col1, col2 = st.columns([1, 3])
with col1:
    st.metric(label="Confidence", value=f"{report.confidence_score:.0%}")
with col2:
    st.progress(report.confidence_score)
```

### graph.invoke config with thread_id (from test_phase4.py)
```python
# Source: tests/test_phase4.py line 165 — confirmed working pattern
import uuid
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = graph.invoke({"topic": topic, "max_rounds": 1}, config=config)
final_report = result.get("final_report")  # DebateReport object
```

### list_debates() sidebar pattern
```python
# Source: debate/store.py — public API confirmed in Phase 4
from debate.store import list_debates, load_debate
past = list_debates()  # list[dict] with keys: debate_id, topic, created_at, status
for row in past[:10]:
    if st.button(row["topic"][:40], key=row["debate_id"]):
        loaded = load_debate(row["debate_id"])
        if loaded:
            st.session_state.final_report = loaded
            st.session_state.debate_status = "complete"
            st.rerun()
```

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| streamlit | UI rendering | Partial | 1.56.0 (installable) | None — required |
| langgraph | graph.stream() | Partial | 1.1.9 (in requirements.txt) | None — required |
| debate.graph | Core graph | Must be importable | — | Fix import errors before UI |
| debate.store | Past debates | Must be importable | — | Skip sidebar section if absent |
| debates.db | Past debates list | Auto-created | — | Created on first save_debate call |
| ANTHROPIC_API_KEY | LLM calls | Must be set | — | `st.error` + exit if not set |

**Note:** `streamlit` is not yet installed in the project's Python environment (pip3 check confirmed it's available on PyPI as 1.56.0 but not yet installed). Must be added to `requirements.txt` and installed before running `app.py`.

**Missing dependencies with no fallback:**
- `streamlit==1.56.0` — must be added to `requirements.txt` and installed
- `ANTHROPIC_API_KEY` env var — must be set; add a check at app startup with `st.error` if missing

**Missing dependencies with fallback:**
- `debates.db` — auto-created by `store.get_connection()`; no action needed

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (confirmed from existing tests/test_phase1.py through test_phase4.py) |
| Config file | none found — uses pytest defaults |
| Quick run command | `pytest tests/test_phase5.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| UI-01 | Topic input + Start button renders; session state initializes clean | unit (streamlit testing API) | `pytest tests/test_phase5.py::test_session_state_init -x` | No — Wave 0 |
| UI-02 | graph.stream() chunks dispatch correctly to UI handlers | unit (mock graph.stream) | `pytest tests/test_phase5.py::test_stream_dispatch -x` | No — Wave 0 |
| UI-03 | render_report() renders DebateReport without error | unit (mock DebateReport) | `pytest tests/test_phase5.py::test_render_report -x` | No — Wave 0 |
| UI-04 | Fresh session has no KeyError; error state shows reset button | unit (streamlit testing API) | `pytest tests/test_phase5.py::test_fresh_session_no_error -x` | No — Wave 0 |

**Note:** Full end-to-end UI test (actual browser) is manual-only for the portfolio demo context. The automated tests focus on: session state initialization logic, stream chunk dispatch logic (with mocked graph), and report rendering logic (with a DebateReport fixture from Phase 4 test helpers).

### Streamlit Testing API
Streamlit 1.56.0 ships with `streamlit.testing.v1.AppTest` — a headless testing framework that can run Streamlit scripts programmatically. Confirmed from installed source: `/streamlit/testing/v1/app_test.py`.

```python
# Verified: AppTest is in streamlit 1.56.0
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py")
at.run()
# assert no exceptions, check widget state
```

### Sampling Rate
- **Per task commit:** `pytest tests/test_phase5.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_phase5.py` — covers UI-01 through UI-04; needs `AppTest` import + DebateReport fixture from `test_phase4.py`
- [ ] `requirements.txt` — add `streamlit==1.56.0`
- [ ] `.streamlit/config.toml` — optional but recommended for demo stability

---

## Running the App

```bash
# Install streamlit (if not present)
pip install streamlit==1.56.0

# Run the app
streamlit run app.py
```

**Optional `.streamlit/config.toml`:**
```toml
[server]
port = 8501
headless = false

[theme]
base = "light"
```

No special config is required. The defaults work for a demo. Adding `config.toml` prevents port conflicts when restarting during demo sessions.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `loop.run_until_complete(graph.astream())` | `graph.stream(stream_mode="updates")` | LangGraph 1.x | Eliminates asyncio/Streamlit event loop conflict entirely |
| Manual streaming with callbacks | `stream_mode="updates"` iterator | LangGraph 0.x → 1.x | Clean per-node dispatch without callback registration |
| `st.spinner()` for progress | `st.status()` with `state="running"/"complete"/"error"` | Streamlit ~1.27 | Collapsible, stateful, auto-transitions — far better for multi-step processes |
| `session_state` as flat bag | `session_state` as explicit state machine | Always best practice | Prevents rerun side effects; required for streaming integration |

**Note:** `st.status()` was added in Streamlit 1.27. It is confirmed present in 1.56.0 (verified from installed source `layouts.py`).

---

## Open Questions

1. **Python environment for the project**
   - What we know: `requirements.txt` has langgraph==1.1.9 but not streamlit. The conda env `llm-data-pipeline` has streamlit 1.56.0 but not langgraph.
   - What's unclear: Does the project have a virtualenv that should have both? Or should the developer install both in `llm-data-pipeline`?
   - Recommendation: Add `streamlit==1.56.0` to `requirements.txt`. The plan's Wave 0 task should be `pip install -r requirements.txt`. This resolves the environment ambiguity without creating a new env.

2. **Parallel fan-out chunk format for agent nodes**
   - What we know: `stream_mode="updates"` with parallel `Send` fan-out: each agent node yields its own chunk separately (one chunk per node completion, not all three in one chunk). Verified from `map_output_updates` — it groups by task name and yields one dict per task.
   - What's unclear: In practice with the current graph's `Send` fan-out, the timing of when chunks arrive is non-deterministic (parallel execution). The UI handler must be robust to receiving agent chunks in any order.
   - Recommendation: The dispatch handler should not assume agent ordering. Use `node_name` key to identify which agent just completed.

3. **st.warning() block content**
   - What we know: `st.warning(body)` renders a warning block. It does not return a container that can be written into.
   - What's unclear: Rendering `DisputedPoint.agent_positions` (multi-line content) inside a warning block.
   - Recommendation: Use `st.markdown` with a custom HTML/CSS callout for disputed points, or use `st.expander` per disputed point. Avoid trying to nest content inside `st.warning`.

---

## Sources

### Primary (HIGH confidence)
- `langgraph/pregel/_io.py` (installed, verified) — `map_output_updates` function: confirms `stream_mode="updates"` chunk format as `{node_name: state_update_dict}`
- `langgraph/pregel/_loop.py` (installed, verified) — `SyncPregelLoop` class: confirms sync `graph.stream()` has zero asyncio dependency
- `streamlit/elements/layouts.py:1677` (installed in llm-data-pipeline env, verified) — `st.status()` signature: `label, expanded=False, state="running"/"complete"/"error"`
- `streamlit/elements/lib/mutable_status_container.py` (installed, verified) — `StatusContainer.update()` signature and auto-close behavior
- `debate/graph.py` (project source, verified) — node names, graph topology, `InMemorySaver` checkpointer
- `debate/state.py` (project source, verified) — `DebateReport`, `AgentArgument`, `RoundRecord`, `Concession` field names
- `debate/store.py` (project source, verified) — `list_debates()`, `load_debate()` public API
- `tests/test_phase4.py:165` (project source, verified) — confirmed `{"configurable": {"thread_id": str(uuid.uuid4())}}` config pattern

### Secondary (MEDIUM confidence)
- PITFALLS.md (project research, cross-verified with source) — Pitfall 6 (sync vs async), Pitfall 11 (SQLite blocking)
- Streamlit docs pattern for session_state state machine — standard Streamlit idiom, consistent with source

### Tertiary (LOW confidence)
- `.streamlit/config.toml` recommendations — standard community practice, not verified against 1.56.0 changelog specifically

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages verified from installed source or PyPI
- Architecture patterns: HIGH — verified from LangGraph pregel source and Streamlit source
- stream_mode="updates" chunk format: HIGH — verified from `map_output_updates` source
- st.status() API: HIGH — verified from installed streamlit 1.56.0 source
- Session state patterns: HIGH — verified from streamlit runtime source
- Pitfalls: HIGH — sourced from PITFALLS.md which was verified against installed source
- Validation architecture: HIGH — AppTest confirmed in installed streamlit testing module

**Research date:** 2026-04-23
**Valid until:** 2026-05-23 (stable libraries; 30-day validity)
