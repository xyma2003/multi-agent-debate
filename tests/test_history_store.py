"""Offline acceptance of topic search and stable, bounded history pages."""

import sqlite3

import pytest

from debate import store


@pytest.fixture
def history():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    store._init_schema(conn)
    yield conn
    conn.close()


def _insert(conn, debate_id, topic, created_at="2026-10-15T09:00:00+00:00"):
    conn.execute(
        "INSERT INTO debates VALUES (?, ?, ?, ?, ?)",
        (debate_id, topic, created_at, "converged", '{"private_body": "body-only-needle"}'),
    )
    conn.commit()


def _ids(rows):
    return [row["debate_id"] for row in rows]


def test_pages_exhaust_history_with_timestamp_ties_and_legacy_api(history):
    # Insert out of sort order. The first two pages cross equal-time groups.
    for number in list(range(0, 27, 2)) + list(range(1, 27, 2)):
        _insert(
            history, f"debate-{number:02}", f"辩题 {number}",
            "2026-10-15T09:00:00+00:00" if number < 14 else "2026-10-16T09:00:00+00:00",
        )

    pages = [store.list_debates(history, limit=10, offset=start) for start in (0, 10, 20)]
    expected = [f"debate-{number:02}" for number in range(26, -1, -1)]
    assert [len(page) for page in pages] == [10, 10, 7]
    assert _ids([row for page in pages for row in page]) == expected
    assert len(set(_ids([row for page in pages for row in page]))) == 27
    assert _ids(store.list_debates(history)) == expected
    assert store.count_debates(history) == 27
    assert store.list_debates(history, limit=10, offset=30) == []
    assert _ids(store.list_debates(history, offset=20)) == expected[20:]
    assert all(set(row) == {"debate_id", "topic", "created_at", "status"} for row in pages[0])


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (" 中文 ", ["cn"]),
        ("rEmOtE", ["ascii"]),
        ("100%", ["percent"]),
        ("_", ["underscore"]),
        ("\\", ["slash"]),
        ("%_\\", ["mixed"]),
        ("' OR 1=1 --", ["injection"]),
        ("body-only-needle", []),
        ("missing topic", []),
    ],
)
def test_literal_topic_search_and_count_agree(history, query, expected):
    topics = {
        "cn": "是否支持中文搜索？",
        "ascii": "Remote work improves focus",
        "percent": "Completion is 100%",
        "percent-decoy": "Completion is 1000",
        "underscore": "version_2",
        "underscore-decoy": "versionA2",
        "slash": "C:\\workspace",
        "slash-decoy": "C:workspace",
        "mixed": "literal %_\\ combination",
        "injection": "literal ' OR 1=1 -- topic",
    }
    for debate_id, topic in topics.items():
        _insert(history, debate_id, topic)

    # The combined metacharacter topic also contains each individual marker.
    if query in ("_", "\\"):
        expected = sorted([*expected, "mixed"], reverse=True)
    assert _ids(store.list_debates(history, query=query, limit=10)) == expected
    assert store.count_debates(history, query=query) == len(expected)
    assert store.count_debates(history) == len(topics)


@pytest.mark.parametrize("query", [None, "", " \t\n", "　"])
def test_empty_and_blank_queries_keep_all_rows(history, query):
    assert store.count_debates(history, query=query) == 0
    assert store.list_debates(history, query=query, limit=10) == []
    _insert(history, "one", "First topic")
    _insert(history, "two", "Second topic")
    assert store.count_debates(history, query=query) == 2
    assert _ids(store.list_debates(history, query=query, limit=1)) == ["two"]


def test_filtered_pages_use_the_same_population_as_count(history):
    for number in range(31):
        _insert(history, f"item-{number:02}", "匹配 辩题" if number % 3 else "其他辩题")
    expected = [f"item-{number:02}" for number in range(30, -1, -1) if number % 3]
    count = store.count_debates(history, query="匹配")
    pages = [
        store.list_debates(history, query="匹配", limit=7, offset=start)
        for start in range(0, count, 7)
    ]
    assert count == 20
    assert _ids([row for page in pages for row in page]) == expected


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"limit": 0}, ValueError),
        ({"limit": -1}, ValueError),
        ({"limit": 2**63}, ValueError),
        ({"limit": True}, TypeError),
        ({"limit": 1.5}, TypeError),
        ({"limit": "10"}, TypeError),
        ({"offset": -1}, ValueError),
        ({"offset": 2**63}, ValueError),
        ({"offset": False}, TypeError),
        ({"offset": 1.5}, TypeError),
        ({"offset": None}, TypeError),
    ],
)
def test_invalid_pagination_rejected_before_opening_database(monkeypatch, kwargs, error):
    def unexpected_connection():
        pytest.fail("Invalid arguments must not open the default database")

    monkeypatch.setattr(store, "get_connection", unexpected_connection)
    with pytest.raises(error):
        store.list_debates(**kwargs)


@pytest.mark.parametrize("reader", [store.list_debates, store.count_debates])
def test_invalid_query_rejected_before_opening_database(monkeypatch, reader):
    def unexpected_connection():
        pytest.fail("Invalid query must not open the default database")

    monkeypatch.setattr(store, "get_connection", unexpected_connection)
    with pytest.raises(TypeError, match="query"):
        reader(query=123)


def test_default_connection_api_and_read_only_queries_leave_database_unchanged(history, monkeypatch):
    _insert(history, "one", "可只读搜索 100%")
    _insert(history, "two", "Another topic")
    monkeypatch.setattr(store, "get_connection", lambda: history)
    original_rows = history.execute("SELECT * FROM debates ORDER BY debate_id").fetchall()
    original_changes = history.total_changes
    history.execute("PRAGMA query_only = ON")

    assert store.count_debates(query="100%") == 1
    assert _ids(store.list_debates(query="100%", limit=1)) == ["one"]
    assert store.count_debates() == 2
    assert len(store.list_debates()) == 2
    assert history.execute("SELECT * FROM debates ORDER BY debate_id").fetchall() == original_rows
    assert history.total_changes == original_changes
