# tests/test_phase4.py
"""
Phase 4: Persistence — test suite.

STORE-01: test_save_and_load_roundtrip, test_load_nonexistent_returns_none,
          test_upsert_replaces_existing_row
STORE-02: test_list_debates_structure, test_list_debates_empty_db

Integration tests (require ANTHROPIC_API_KEY):
    test_full_graph_saves_to_db
    test_replay_by_debate_id

Unit tests use in-memory SQLite via mem_conn fixture — no file I/O, no LLM.
Integration tests use max_rounds=1 for speed and read from the default debates.db.
"""
import sqlite3
import uuid
from datetime import datetime, timezone

import pytest

from debate.state import DebateReport
from debate.store import _init_schema, load_debate, list_debates, save_debate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(
    debate_id: str = "test-01",
    topic: str = "Test topic",
) -> DebateReport:
    """Minimal DebateReport for unit tests — no LLM required."""
    return DebateReport(
        debate_id=debate_id,
        topic=topic,
        consensus_points=["Agreed point A"],
        disputed_points=[],
        verdict="Test verdict sentence.",
        confidence_score=0.85,
        convergence_status="converged",
        reasoning_trace=[],
        concession_log=[],
        created_at=datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_conn():
    """Fresh in-memory SQLite connection with schema initialised.

    Scope is 'function' (default) — each test gets an isolated DB.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# STORE-01 unit tests: save / load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(mem_conn):
    """STORE-01: save_debate then load_debate returns an identical DebateReport."""
    report = _make_report()
    save_debate(report, mem_conn)

    loaded = load_debate(report.debate_id, mem_conn)

    assert loaded is not None, "load_debate returned None for a saved report"
    assert loaded.debate_id == report.debate_id
    assert loaded.topic == report.topic
    assert loaded.confidence_score == report.confidence_score
    assert loaded.convergence_status == report.convergence_status
    assert loaded.consensus_points == report.consensus_points


def test_load_nonexistent_returns_none(mem_conn):
    """STORE-01: load_debate returns None for an unknown debate_id."""
    result = load_debate("no-such-id", mem_conn)
    assert result is None, f"Expected None, got {result!r}"


def test_upsert_replaces_existing_row(mem_conn):
    """STORE-01: saving twice with the same debate_id replaces the first row.

    INSERT OR REPLACE semantics: only 1 row exists, and it reflects
    the second save's data.
    """
    original = _make_report(debate_id="upsert-01", topic="Original topic")
    updated = _make_report(debate_id="upsert-01", topic="Updated topic")

    save_debate(original, mem_conn)
    save_debate(updated, mem_conn)

    loaded = load_debate("upsert-01", mem_conn)
    assert loaded is not None
    assert loaded.topic == "Updated topic", (
        f"Expected 'Updated topic', got {loaded.topic!r}. "
        "INSERT OR REPLACE may not be working."
    )

    rows = list_debates(mem_conn)
    assert len(rows) == 1, (
        f"Expected exactly 1 row after upsert, got {len(rows)}"
    )


# ---------------------------------------------------------------------------
# STORE-02 unit tests: list_debates
# ---------------------------------------------------------------------------

def test_list_debates_empty_db(mem_conn):
    """STORE-02: list_debates returns [] when the table has no rows."""
    rows = list_debates(mem_conn)
    assert rows == [], f"Expected [], got {rows!r}"


def test_list_debates_structure(mem_conn):
    """STORE-02: list_debates returns summary rows with correct keys, no report_json."""
    report_a = _make_report(debate_id="list-01", topic="Topic A")
    report_b = _make_report(debate_id="list-02", topic="Topic B")

    save_debate(report_a, mem_conn)
    save_debate(report_b, mem_conn)

    rows = list_debates(mem_conn)

    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"

    required_keys = {"debate_id", "topic", "created_at", "status"}
    for row in rows:
        assert required_keys.issubset(row.keys()), (
            f"Row missing required keys. Got: {set(row.keys())}"
        )
        assert "report_json" not in row, (
            "list_debates must NOT include report_json column"
        )


# ---------------------------------------------------------------------------
# Integration tests: full graph end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_graph_saves_to_db():
    """STORE-01 + STORE-02 integration: graph.invoke writes a row to debates.db.

    After a full debate run, open a separate read-only connection to the
    default DB path and verify the row exists with correct topic and status.
    """
    import sqlite3 as _sqlite3

    from debate.graph import graph
    from debate.store import DB_PATH

    topic = "Is remote work beneficial?"
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke(
        {"topic": topic, "max_rounds": 1},
        config=config,
    )

    final_report = result.get("final_report")
    assert final_report is not None, "graph.invoke() did not return final_report"

    debate_id = final_report.debate_id

    # Open a separate read-only connection to avoid interfering with the singleton
    verify_conn = _sqlite3.connect(str(DB_PATH))
    verify_conn.row_factory = _sqlite3.Row
    try:
        row = verify_conn.execute(
            "SELECT * FROM debates WHERE debate_id = ?", (debate_id,)
        ).fetchone()
    finally:
        verify_conn.close()

    assert row is not None, (
        f"No row found in debates.db for debate_id={debate_id!r}. "
        "save_node may not have written to the DB."
    )
    assert row["topic"] == topic, (
        f"Row topic={row['topic']!r} does not match expected {topic!r}"
    )
    assert row["status"] in ("converged", "max_rounds", "partial"), (
        f"Unexpected status value: {row['status']!r}"
    )


@pytest.mark.integration
def test_replay_by_debate_id():
    """STORE-01 integration: load_debate retrieves an exact copy of the saved DebateReport.

    Runs a full debate, then loads the stored report by debate_id and verifies
    that key fields survive the JSON round-trip.
    """
    from debate.graph import graph
    from debate.store import load_debate as _load_debate

    topic = "Should AI development be regulated by governments?"
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke(
        {"topic": topic, "max_rounds": 1},
        config=config,
    )

    final_report = result.get("final_report")
    assert final_report is not None

    debate_id = final_report.debate_id
    loaded = _load_debate(debate_id)

    assert loaded is not None, (
        f"load_debate({debate_id!r}) returned None. Row not persisted."
    )
    assert loaded.debate_id == debate_id
    assert loaded.topic == final_report.topic
    assert loaded.confidence_score == final_report.confidence_score
    assert loaded.convergence_status == final_report.convergence_status
    assert len(loaded.reasoning_trace) == len(final_report.reasoning_trace), (
        f"reasoning_trace length mismatch: "
        f"loaded={len(loaded.reasoning_trace)}, "
        f"original={len(final_report.reasoning_trace)}"
    )
