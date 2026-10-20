"""The API cache (db/api_cache.sqlite), one table per API.

    s2_cache        keyed on semantic_scholar.normalise(title), so near-duplicate
                    titles share an entry
    together_cache  keyed on together_cache_key(model, prompt)

The Together key is frozen:

    sha256(f"{model}\\x1f{prompt}".encode("utf-8")).hexdigest()

Only the hash is stored, so changing the recipe makes every cached row
unreachable; tests/test_cache.py pins one digest. Prompts are not normalised,
since a prompt is the exact string sent, and the separator is the ASCII unit
separator because prompt text can contain any printable character.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from hallutrace.semantic_scholar import UNVERIFIABLE, normalise

DEFAULT_CACHE_PATH = "db/api_cache.sqlite"

# a second connection waits for a writer instead of raising "database is locked"
BUSY_TIMEOUT_MS = 5000

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS s2_cache (
    normalised_title TEXT PRIMARY KEY,
    verdict          TEXT NOT NULL,
    cached_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS together_cache (
    cache_key TEXT PRIMARY KEY,
    model     TEXT NOT NULL,
    response  TEXT NOT NULL,
    cached_at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def together_cache_key(model: str, prompt: str) -> str:
    """The frozen key recipe (see the module docstring)."""
    joined = "%s\x1f%s" % (model, prompt)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def connect_cache(path: str = DEFAULT_CACHE_PATH) -> sqlite3.Connection:
    """Open the cache in WAL mode, with a busy timeout for concurrent writers."""
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = %d" % BUSY_TIMEOUT_MS)
    conn.row_factory = sqlite3.Row
    return conn


def init_cache_db(conn: sqlite3.Connection) -> None:
    """Create both cache tables. Safe to call repeatedly."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()


class SemanticScholarCache:
    """s2_cache behind SemanticScholarClient's get(title) / put(title, verdict)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self, title):
        row = self.conn.execute(
            "SELECT verdict FROM s2_cache WHERE normalised_title = ?",
            (normalise(title),),
        ).fetchone()
        return row["verdict"] if row is not None else None

    def put(self, title, verdict):
        # a failed lookup must not become a permanent verdict
        if verdict == UNVERIFIABLE:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO s2_cache (normalised_title, verdict, cached_at) "
            "VALUES (?, ?, ?)",
            (normalise(title), verdict, _now_iso()),
        )
        self.conn.commit()


class TogetherCache:
    """get/put over together_cache, keyed by together_cache_key(model, prompt)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get(self, model, prompt):
        row = self.conn.execute(
            "SELECT response FROM together_cache WHERE cache_key = ?",
            (together_cache_key(model, prompt),),
        ).fetchone()
        return row["response"] if row is not None else None

    def put(self, model, prompt, response):
        self.conn.execute(
            "INSERT OR REPLACE INTO together_cache (cache_key, model, response, cached_at) "
            "VALUES (?, ?, ?, ?)",
            (together_cache_key(model, prompt), model, response, _now_iso()),
        )
        self.conn.commit()


class ResilientCache:
    """A get/put cache that degrades instead of raising when its connection fails.

    Wraps the cache built by inner_factory(conn) (SemanticScholarCache or
    TogetherCache). On a failure it reconnects once and retries, up to
    MAX_RECONNECTS reconnects in all; after that it keeps trying the operation
    without reconnecting. If a reconnect or its retry fails, the cache is
    disabled for good. get() and put() never raise. Failures are counted in
    `.failures` and the 1st, 10th, 100th and 1000th are logged to stderr.
    """

    MAX_RECONNECTS = 5
    _LOG_AT_FAILURE = (1, 10, 100, 1000)

    def __init__(self, path, inner_factory, conn=None):
        self.path = path
        self._inner_factory = inner_factory
        self.failures = 0
        self.reconnects = 0
        self.disabled = False
        self._conn = conn if conn is not None else connect_cache(path)
        init_cache_db(self._conn)
        self._inner = inner_factory(self._conn)

    @property
    def reconnect_budget_exhausted(self) -> bool:
        return self.reconnects >= self.MAX_RECONNECTS

    def _maybe_log(self, exc: BaseException) -> None:
        if self.failures not in self._LOG_AT_FAILURE:
            return
        print(
            "hallutrace.cache.ResilientCache: failure #%d on %s: %s: %s"
            % (self.failures, self.path, type(exc).__name__, exc),
            file=sys.stderr,
        )

    def _reconnect(self) -> bool:
        """One attempt: close the old connection, open and wire a new one."""
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._conn = connect_cache(self.path)
            init_cache_db(self._conn)
            self._inner = self._inner_factory(self._conn)
            self.reconnects += 1
            return True
        except Exception:  # noqa: BLE001
            return False

    def _guarded(self, op):
        """Run op(), counting and recovering from any failure.

        op looks up self._inner when called, so a retry uses the new connection.
        """
        if self.disabled:
            return None
        try:
            return op()
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            self._maybe_log(e)
            if self.reconnect_budget_exhausted:
                # no more reconnects, but later calls still try op()
                return None
            if not self._reconnect():
                self.disabled = True  # the old connection is closed
                return None
            try:
                return op()
            except Exception as e2:  # noqa: BLE001
                self.failures += 1
                self._maybe_log(e2)
                self.disabled = True  # a fresh connection fails too
                return None

    def get(self, *args):
        return self._guarded(lambda: self._inner.get(*args))

    def put(self, *args) -> None:
        self._guarded(lambda: self._inner.put(*args))

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
