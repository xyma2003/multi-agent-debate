# debate/store.py
"""SQLite persistence layer for completed debates.

Public API:
    get_connection(db_path) -> sqlite3.Connection   # singleton per path
    save_debate(report, conn=None) -> None
    load_debate(debate_id, conn=None) -> DebateReport | None
    list_debates(conn=None) -> list[dict]
"""
import sqlite3
from pathlib import Path
from typing import Optional

from debate.state import DebateReport

# Default DB location: project root (next to debate/ package)
DB_PATH = Path(__file__).parent.parent / "debates.db"

# Module-level singleton connection (lazy init, keyed by resolved path)
_connections: dict[str, sqlite3.Connection] = {}


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Return (or create) a singleton sqlite3 connection for db_path.

    Sets row_factory = sqlite3.Row so callers can access columns by name.
    Initialises the schema on first connect.
    check_same_thread=False is safe here: the graph runs single-threaded.
    """
    key = str(Path(db_path).resolve())
    if key not in _connections:
        conn = sqlite3.connect(key, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _init_schema(conn)
        _connections[key] = conn
    return _connections[key]


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS debates (
            debate_id   TEXT PRIMARY KEY,
            topic       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            status      TEXT NOT NULL,
            report_json TEXT NOT NULL
        )
    """)
    conn.commit()


def save_debate(report: DebateReport, conn: Optional[sqlite3.Connection] = None) -> None:
    """Persist a DebateReport. Replaces existing row if debate_id already present."""
    if conn is None:
        conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO debates "
        "(debate_id, topic, created_at, status, report_json) VALUES (?,?,?,?,?)",
        (
            report.debate_id,
            report.topic,
            report.created_at.isoformat(),
            report.convergence_status,
            report.model_dump_json(),
        ),
    )
    conn.commit()


def load_debate(debate_id: str, conn: Optional[sqlite3.Connection] = None) -> Optional[DebateReport]:
    """Load a DebateReport by debate_id. Returns None if not found."""
    if conn is None:
        conn = get_connection()
    row = conn.execute(
        "SELECT report_json FROM debates WHERE debate_id = ?", (debate_id,)
    ).fetchone()
    if row is None:
        return None
    return DebateReport.model_validate_json(row["report_json"])


def list_debates(conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """Return summary rows (no report_json) ordered newest-first."""
    if conn is None:
        conn = get_connection()
    rows = conn.execute(
        "SELECT debate_id, topic, created_at, status FROM debates ORDER BY created_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]
