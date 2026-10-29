# debate/store.py
"""SQLite persistence layer for completed debates.

Public API:
    get_connection(db_path) -> sqlite3.Connection   # singleton per path
    save_debate(report, conn=None) -> None
    load_debate(debate_id, conn=None) -> DebateReport | None
    list_debates(conn=None, *, query=None, limit=None, offset=0) -> list[dict]
    count_debates(conn=None, *, query=None) -> int
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
    try:
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
    except sqlite3.Error:
        # A failed COMMIT can leave the inserted/replaced row visible through
        # this reused connection. Roll it back before history reads or retries.
        conn.rollback()
        raise


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


def _topic_filter(query: str | None) -> tuple[str, tuple[str, ...]]:
    """Build the shared literal-substring filter for history reads."""
    if query is not None and not isinstance(query, str):
        raise TypeError("query must be a string or None")
    query = query.strip() if query is not None else ""
    if not query:
        return "", ()
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return " WHERE topic LIKE ? ESCAPE '\\'", (f"%{escaped}%",)


def _validate_page_value(name: str, value: int, minimum: int) -> None:
    # bool is an int subclass, but is not a meaningful pagination value.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be between {minimum} and {2**63 - 1}")


def count_debates(
    conn: Optional[sqlite3.Connection] = None, *, query: str | None = None,
) -> int:
    """Count topics matching a trimmed literal substring; blank means all."""
    where, params = _topic_filter(query)
    if conn is None:
        conn = get_connection()
    return int(conn.execute("SELECT COUNT(*) FROM debates" + where, params).fetchone()[0])


def list_debates(
    conn: Optional[sqlite3.Connection] = None, *, query: str | None = None,
    limit: int | None = None, offset: int = 0,
) -> list[dict]:
    """Return summary rows, ordered by created_at then debate_id descending.

    Search is a trimmed literal substring of topic only, using SQLite's LIKE
    case handling (ASCII case-insensitive by default). An omitted limit keeps
    the original all-rows API. A supplied limit must be positive; offset must
    be nonnegative. Both must be integers within SQLite's signed 64-bit range.
    """
    where, filter_params = _topic_filter(query)
    params: tuple[str | int, ...] = filter_params
    if limit is not None:
        _validate_page_value("limit", limit, 1)
    _validate_page_value("offset", offset, 0)
    sql = (
        "SELECT debate_id, topic, created_at, status FROM debates" + where
        + " ORDER BY created_at DESC, debate_id DESC"
    )
    if limit is not None or offset:
        sql += " LIMIT ? OFFSET ?"
        params += (limit if limit is not None else -1, offset)
    if conn is None:
        conn = get_connection()
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]
