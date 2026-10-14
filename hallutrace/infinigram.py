"""hallutrace/infinigram.py  |  rate-limited, resumable infini-gram client and its cache

Fetch half of the n_c counting method: find
every corpus mention of a subject, download a window of text around each, keep
them in a SQLite cache. Matching and counting read from the cache later; they
are not here.

Hard rules, from infini-gram's stated limit of 1 request per second:

- strictly sequential: no threads, no asyncio, no multiprocessing for HTTP;
- at least MIN_GAP_S between request STARTS, on an injectable monotonic clock;
- the wall-clock time of the last request start is persisted in the cache, so a
  crash-and-restart waits out the gap instead of bursting;
- one fetcher per cache file AND one per user, enforced by OS-level locks
  (Cache.lock; the per-user lock also holds the last-request time, so a second
  cache path cannot double the rate or burst after a crash);
- a block (HTTP 403/429) is persisted in the cache, so a restart cannot skip
  the wait or reset the "3 in a row" count;
- a request that already succeeded is never repeated: cache first, network second.

Pieces:

    Transport / HttpTransport   post(body) -> (status, parsed JSON or text); no redirects
    Throttle                    wait(), persist, mark_started() around every request
    Cache                       the SQLite schema, locks, run_meta, last-request time;
                                the `counts` table only via ensure_counts_table()
    InfinigramClient            find / get_doc_by_rank / tokenize / count, error policy

Tests use a fake Transport and a fake clock; nothing here is tested against the
live API except by hand (Stage 5 smoke test).
"""

from __future__ import annotations

import http.client
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from hallutrace.cache import connect_cache

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

API_URL = "https://api.infini-gram.io/"

# One index per subject model; OLMo 2 first,
# because `plan` and the first `full` pass run on it alone.
INDEXES = (
    "v4_olmo-mix-1124_llama",  # OLMo 2
    "v4_dclm-baseline_llama",  # DCLM
    "v4_dolma-v1_7_llama",  # OLMo 1.7
    "v4_rpj_llama_s4",  # RedPajama
    "v4_piletrain_llama",  # Pythia
)
# All five indexes share the Llama-2 tokenizer, so tokenizing once is enough.
TOKENIZE_INDEX = INDEXES[0]

MIN_GAP_S = 1.05  # between request STARTS; infini-gram's rule is 1 request/second
DISP_LEN = 800  # max_disp_len: fetch wide, count narrow (a window can be narrowed, never widened)
TIMEOUT_S = 180  # single requests have taken over 10 s
MAX_ATTEMPTS = 6  # timeouts, connection errors, 5xx: backoff 2, 4, 8 ... capped at BACKOFF_CAP_S
BACKOFF_CAP_S = 300
BLOCKED_SLEEP_S = 600  # after an HTTP 403 or 429
MAX_CONSECUTIVE_403 = 3  # the third block response in a row raises Blocked
BLOCK_STATUSES = (403, 429)  # 429 is a throttle answer: treated exactly like 403
# Consecutive requests that failed for good (after retries) before the client
# gives up on the whole run: a systematic 400, a changed API, a dead endpoint.
MAX_CONSECUTIVE_FAILURES = 25

FORMS = ("plain", "glued")

SNIPPET_CHARS = 500  # how much of a failed response body errors keeps

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    prompt_name        TEXT NOT NULL,
    query_string       TEXT NOT NULL,
    source             TEXT NOT NULL,           -- name | stripped | folded | alias
    qid                TEXT,
    ambiguous_name     INTEGER NOT NULL DEFAULT 0,
    ambiguous_stripped INTEGER NOT NULL DEFAULT 0,
    active             INTEGER NOT NULL DEFAULT 1,  -- 0: dropped by a later `plan`; rows are never deleted
    PRIMARY KEY (prompt_name, query_string)
);

-- Keyed by the exact text tokenized ("(" + query string), not the query string:
-- the key is what was sent, so it cannot drift from what the ids came from.
CREATE TABLE IF NOT EXISTS glued_ids (
    text          TEXT PRIMARY KEY,
    token_ids_json TEXT NOT NULL,
    tokens_json    TEXT NOT NULL
);

-- `form` is in every key below: ranks are per query token sequence, so (s, rank)
-- under plain and under glued are different occurrences.
CREATE TABLE IF NOT EXISTS finds (
    index_name           TEXT NOT NULL,
    query_string         TEXT NOT NULL,
    form                 TEXT NOT NULL,          -- plain | glued
    cnt                  INTEGER NOT NULL,
    segment_by_shard_json TEXT NOT NULL,
    tokens_json          TEXT NOT NULL,
    api_latency_ms       REAL,
    fetched_at           TEXT NOT NULL,
    PRIMARY KEY (index_name, query_string, form)
);

CREATE TABLE IF NOT EXISTS plan (
    index_name   TEXT NOT NULL,
    query_string TEXT NOT NULL,
    form         TEXT NOT NULL,
    s            INTEGER NOT NULL,
    rank         INTEGER NOT NULL,
    selected     TEXT NOT NULL,                  -- all | sample
    PRIMARY KEY (index_name, query_string, form, s, rank)
);

CREATE TABLE IF NOT EXISTS windows (
    index_name     TEXT NOT NULL,
    query_string   TEXT NOT NULL,
    form           TEXT NOT NULL,
    s              INTEGER NOT NULL,
    rank           INTEGER NOT NULL,
    doc_ix         INTEGER,
    doc_len        INTEGER,
    disp_len       INTEGER,
    needle_offset  INTEGER,
    spans_json     TEXT NOT NULL,
    metadata_json  TEXT,                         -- the string the API returned, never parsed here
    api_latency_ms REAL,
    fetched_at     TEXT NOT NULL,
    PRIMARY KEY (index_name, query_string, form, s, rank)
);

-- Failures only; a failed request is never a cached result, so it is retried
-- on the next run.
CREATE TABLE IF NOT EXISTS errors (
    ts           TEXT NOT NULL,
    index_name   TEXT,
    query_type   TEXT,
    request_json TEXT,
    http_status  INTEGER,                        -- NULL for timeouts and connection errors
    body_snippet TEXT
);

CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Plain `count` answers (the exact-phrase check in c_exec_untyped), keyed by
# what was sent. Not in SCHEMA: only Cache.ensure_counts_table() creates it, so
# opening a window cache never gains this table. `approx` is 1 / 0, NULL when the
# response had no `approx` key.
COUNTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS counts (
    index_name     TEXT NOT NULL,
    query          TEXT NOT NULL,
    cnt            INTEGER NOT NULL,
    approx         INTEGER,
    api_latency_ms REAL,
    fetched_at     TEXT NOT NULL,
    PRIMARY KEY (index_name, query)
);
"""

LAST_REQUEST_KEY = "last_request_wall"
BLOCKED_UNTIL_KEY = "blocked_until"  # run_meta: wall time until which we must not send
CONSECUTIVE_403_KEY = "consecutive_403"  # run_meta: the running block-response count

# The per-user guard: one lock file and one last-request file, whatever cache
# path a fetcher uses. Directory: the Cache keyword, else this environment
# variable, else the temp dir (per user on Windows).
LOCK_DIR_ENV = "HALLUTRACE_INFINIGRAM_LOCK_DIR"
GLOBAL_LOCK_NAME = "hallutrace_infinigram.lock"
GLOBAL_LAST_NAME = "hallutrace_infinigram.last"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value) -> float | None:
    """`value` as a finite float, or None (None, garbage, nan, inf)."""
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


class InfinigramError(Exception):
    """Base class for errors this module raises on purpose."""


class LockHeld(InfinigramError):
    """Another fetcher already holds the lock on this cache file, or the
    per-user lock (a fetcher on any cache path)."""


class Blocked(InfinigramError):
    """Three HTTP 403/429 in a row: infini-gram is blocking us. Wait and rerun."""


class TooManyFailures(InfinigramError):
    """MAX_CONSECUTIVE_FAILURES requests in a row failed for good. The run is
    not making progress (systematic 400, changed API, dead endpoint): stop."""


class MaxRequestsReached(InfinigramError):
    """The --max-requests budget is spent. A clean stop, not a failure."""


class Transport(Protocol):
    def post(self, body: dict) -> tuple[int, dict | str]:
        """POST `body` as JSON. Return (HTTP status, parsed JSON or raw text).

        Timeouts and connection errors are raised as OSError (or a subclass),
        not returned: the client owns the retry policy.
        """


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects. urllib would re-send a POST as a GET inside
    one post() call: a second request that is neither throttled nor counted.
    Returning None makes the 3xx surface as an HTTPError, i.e. as its status."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpTransport:
    """The real transport: urllib, JSON in and out, TIMEOUT_S timeout, and no
    redirects (a 3xx is returned as a status and handled as a failure).

    `opener` is injectable for tests (anything with .open(req, timeout=...)).
    """

    def __init__(self, url: str = API_URL, timeout: float = TIMEOUT_S, opener=None):
        self.url = url
        self.timeout = timeout
        self._opener = opener if opener is not None else urllib.request.build_opener(_NoRedirect)

    def post(self, body: dict) -> tuple[int, dict | str]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as e:
            # A non-2xx status is an answer, not a transport failure: the client
            # decides what 400, 403, 429, 3xx and 5xx each mean.
            status = e.code
            try:
                raw = e.read()
            except (http.client.HTTPException, OSError) as read_err:
                # e.g. IncompleteRead on a broken error body: retry path.
                raise OSError("reading HTTP %d body: %s" % (status, read_err)) from read_err
        except http.client.HTTPException as e:
            # e.g. IncompleteRead: a broken connection that is not an OSError.
            raise OSError("http.client: %s" % e) from e
        text = raw.decode("utf-8", errors="replace")
        try:
            return status, json.loads(text)
        except ValueError:
            return status, text


class Throttle:
    """Enforces MIN_GAP_S between request starts.

    `clock` is a monotonic clock, `sleep` the matching sleeper, both injectable.
    `last_start_wall` is the persisted wall-clock time (time.time()) of the
    previous run's last request start, or None. The first wait() sleeps until
    `wall()` has passed last_start_wall + min_gap_s, so a restart cannot burst.

    The caller's sequence for every request is:

        wall = throttle.wait()             # block until a start is allowed
        cache.set_last_request_wall(wall)  # persist (a commit of unknown duration)
        throttle.mark_started()            # re-check the gap, stamp the start
        transport.post(...)                # sent immediately after the stamp

    wait() does NOT stamp: stamping before the persist would let a slow commit
    eat into the gap between two real sends. mark_started() stamps, so the gap
    is measured between the moments the requests are actually sent.
    """

    def __init__(
        self,
        min_gap_s: float,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
        last_start_wall: float | None = None,
        wall: Callable[[], float] | None = None,
    ):
        self.min_gap_s = min_gap_s
        self.clock = clock
        self.sleep = sleep
        self.wall = wall if wall is not None else time.time
        self.last_start_wall = last_start_wall
        self._last_start_mono: float | None = None

    def _sleep_for(self, seconds: float) -> None:
        # Loop rather than sleep once: time.sleep may return marginally early,
        # and the rule is a floor on the gap, not a target. Measured on the
        # monotonic clock, so a jumping wall clock cannot stretch it.
        deadline = self.clock() + seconds
        while True:
            left = deadline - self.clock()
            if left <= 0:
                break
            self.sleep(left)

    def _wait_mono_gap(self) -> None:
        remaining = self._last_start_mono + self.min_gap_s - self.clock()
        if remaining > 0:
            self._sleep_for(remaining)

    def wait(self) -> float:
        """Block until a request may start; return the wall time to persist.
        Does not stamp the start (see mark_started)."""
        if self._last_start_mono is not None:
            self._wait_mono_gap()
        else:
            # First request after a restart: only the persisted wall clock knows
            # when the previous process last sent. The TOTAL wait is capped at
            # one gap: a timestamp from a skewed, backwards or future clock
            # cannot stall the run, and nan/inf/None are ignored.
            last = _finite(self.last_start_wall)
            if last is not None:
                remaining = min(last + self.min_gap_s - self.wall(), self.min_gap_s)
                if remaining > 0:
                    self._sleep_for(remaining)
        return self.wall()

    def mark_started(self) -> float:
        """Stamp the start of a request, right before it is sent. Re-checks the
        gap against the monotonic clock first, so nothing that happened between
        wait() and now (the persist) can shorten it. Returns the wall time."""
        if self._last_start_mono is not None:
            self._wait_mono_gap()
        self._last_start_mono = self.clock()
        self.last_start_wall = self.wall()
        return self.last_start_wall

    def pause(self, seconds: float) -> None:
        """Sleep `seconds` (backoff, block pause). Counts toward the gap: the next
        wait() measures from the last request START, so a pause never adds a
        second gap on top, and never lets a request start early."""
        if seconds > 0:
            self.sleep(seconds)


def _lock_dir(explicit=None) -> Path:
    """Directory of the per-user lock: the argument, else the environment
    variable, else the temp dir."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get(LOCK_DIR_ENV)
    if env:
        return Path(env)
    return Path(tempfile.gettempdir())


# The lock is one byte at offset 0 of the lock file; the pid and start time
# are written after it, because on Windows a locked byte range cannot be
# read by other handles and the loser needs to read the holder's details.
def _take_lock(lock_path: str, what: str):
    """Open `lock_path` and take an exclusive non-blocking lock on it: the open
    file on success, LockHeld (`what` % lock_path, plus the holder) otherwise."""
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    # Append mode: opening must not truncate the winner's message before we
    # know whether we are the winner.
    f = open(lock_path, "a+b")
    try:
        if sys.platform == "win32":
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = ""
        try:
            f.seek(1 if sys.platform == "win32" else 0)
            holder = f.read().decode("utf-8", errors="replace").strip()
        except OSError:
            pass
        f.close()
        raise LockHeld(
            "%s (%s). Wait for it to finish. The OS releases the lock when that "
            "process dies; if this message persists with no fetcher running, "
            "delete the .lock file." % (what % lock_path, holder or "holder unknown")
        ) from None
    info = "pid %d, started %s" % (os.getpid(), _now_iso())
    f.seek(0)
    f.truncate()
    f.write(b"L" + info.encode("utf-8"))  # byte 0 is the locked byte
    f.flush()
    return f


def _drop_lock(f) -> None:
    if f is None:
        return
    try:
        if sys.platform == "win32":
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass  # closing the handle releases it anyway
    f.close()


class Cache:
    """SQLite cache of every successful response, plus the fetch plan.

    Tables (all keys explicit; INSERT only after success; `form` is in every
    key because ranks are per query token sequence, so (s, rank) under plain and
    glued are different occurrences):

        entities   (prompt_name, query_string, source, qid, ambiguous_name,
                    ambiguous_stripped, active)          PK (prompt_name, query_string)
        glued_ids  (text, token_ids_json, tokens_json)   PK text; index-independent;
                    `text` is exactly what was tokenized, i.e. "(" + query string
        finds      (index_name, query_string, form, cnt, segment_by_shard_json,
                    tokens_json, api_latency_ms, fetched_at)
                                                         PK (index_name, query_string, form)
        plan       (index_name, query_string, form, s, rank, selected)
                                                         PK (index_name, query_string, form, s, rank)
        windows    (index_name, query_string, form, s, rank, doc_ix, doc_len,
                    disp_len, needle_offset, spans_json, metadata_json,
                    api_latency_ms, fetched_at)          PK as plan
        errors     (ts, index_name, query_type, request_json, http_status, body_snippet)
        run_meta   (key, value)

    The driver (scripts/fetch_windows.py) reads and writes entities, plan and
    run_meta through `conn` and the small helpers below; the client only touches
    finds, windows, glued_ids, counts, errors and the last-request time.

    `lock=True` (default) takes TWO OS-level exclusive locks and raises LockHeld,
    naming the holder, if either is held by another process:

      1. a per-user lock, `hallutrace_infinigram.lock` in the lock directory
         (`global_lock_dir`, else $HALLUTRACE_INFINIGRAM_LOCK_DIR, else the temp
         dir). It is the same file whatever cache path is used, so a second
         fetcher on ANY cache (a smoke-test cache, a scratch copy) is refused:
         two fetchers would double the request rate;
      2. `<path>.lock` next to this cache file.

    The last-request wall time is persisted in run_meta AND in a tiny sibling of
    the per-user lock, `hallutrace_infinigram.last` (written atomically), so a
    restart on a different cache path still waits out the gap.
    last_request_wall() returns the larger of the two; a shared file that is
    missing, unreadable, nan or inf is ignored.

    The OS drops both locks when the process dies, so a crash cannot leave a
    stale lock; if the message ever persists with no fetcher running, delete
    the named .lock file. `lock=False` (read-only callers: `status`,
    `--dry-run`) takes neither lock and does not use the shared file.

    Every write commits (a database commit, not a git commit), so a kill loses
    at most one request.
    """

    def __init__(self, path: str, lock: bool = True, global_lock_dir=None):
        self.path = str(path)
        self.lock_path = self.path + ".lock"
        self.global_lock_path: str | None = None
        self.global_last_path: str | None = None
        self._lock_file = None
        self._global_lock_file = None
        self.conn = None
        if lock:
            gdir = _lock_dir(global_lock_dir)
            self.global_lock_path = str(gdir / GLOBAL_LOCK_NAME)
            self.global_last_path = str(gdir / GLOBAL_LAST_NAME)
            self._acquire_lock()
        try:
            self.conn = connect_cache(self.path)
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        except BaseException:
            self.close()
            raise

    # -- locks ---------------------------------------------------------------
    def _acquire_lock(self) -> None:
        # Per-user lock first, then the cache's own; a failure of the second
        # releases the first.
        self._global_lock_file = _take_lock(
            self.global_lock_path,
            "another fetcher (on any cache file) already holds the per-user lock %s",
        )
        try:
            self._lock_file = _take_lock(
                self.lock_path, "another fetcher already holds the cache lock %s")
        except BaseException:
            _drop_lock(self._global_lock_file)
            self._global_lock_file = None
            raise

    def _release_lock(self) -> None:
        f, self._lock_file = self._lock_file, None
        g, self._global_lock_file = self._global_lock_file, None
        _drop_lock(f)
        _drop_lock(g)
        # The files themselves stay: deleting one races with a process about to open it.

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        self._release_lock()

    def __enter__(self) -> "Cache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- run_meta, last-request time ---------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM run_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def update_meta(self, values: dict, delete: tuple = ()) -> None:
        """Set several run_meta keys and delete others in ONE commit."""
        for key, value in values.items():
            self.conn.execute("INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)", (key, value))
        for key in delete:
            self.conn.execute("DELETE FROM run_meta WHERE key = ?", (key,))
        self.conn.commit()

    def last_request_wall(self) -> float | None:
        """Wall-clock start of the most recent request, or None: the larger of
        the cache's own run_meta value and the per-user shared file's value."""
        values = [
            v for v in (_finite(self.get_meta(LAST_REQUEST_KEY)), self._read_shared_last())
            if v is not None
        ]
        return max(values) if values else None

    def set_last_request_wall(self, wall: float) -> None:
        """Persist the last request start in run_meta and the shared file."""
        self.set_meta(LAST_REQUEST_KEY, repr(float(wall)))
        self._write_shared_last(float(wall))

    def _read_shared_last(self) -> float | None:
        if self.global_last_path is None:
            return None
        try:
            with open(self.global_last_path, "r", encoding="utf-8") as f:
                return _finite(f.read().strip())
        except OSError:
            return None

    def _write_shared_last(self, wall: float) -> None:
        if self.global_last_path is None:
            return
        tmp = "%s.%d.tmp" % (self.global_last_path, os.getpid())
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(repr(wall))
            for attempt in range(3):
                try:
                    os.replace(tmp, self.global_last_path)  # atomic
                    return
                except PermissionError:
                    # Windows: an antivirus scan or a reader can hold the target
                    # for a moment.
                    if attempt == 2:
                        raise
                    time.sleep(0.01)
        except OSError:
            # The cache's own value was already committed; the shared file is
            # the second line of defence, not worth aborting the run for.
            try:
                os.remove(tmp)
            except OSError:
                pass

    # -- the tables the client owns ----------------------------------------
    def get_glued_ids(self, text: str) -> tuple[list[int], list[str]] | None:
        row = self.conn.execute(
            "SELECT token_ids_json, tokens_json FROM glued_ids WHERE text = ?", (text,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["token_ids_json"]), json.loads(row["tokens_json"])

    def put_glued_ids(self, text: str, ids: list[int], tokens: list[str]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO glued_ids (text, token_ids_json, tokens_json) VALUES (?, ?, ?)",
            (text, json.dumps(ids), json.dumps(tokens, ensure_ascii=False)),
        )
        self.conn.commit()

    def get_find(self, index: str, query_string: str, form: str) -> dict | None:
        """Row as {"cnt", "segment_by_shard", "tokens", "api_latency_ms"}, or None."""
        row = self.conn.execute(
            "SELECT cnt, segment_by_shard_json, tokens_json, api_latency_ms FROM finds "
            "WHERE index_name = ? AND query_string = ? AND form = ?",
            (index, query_string, form),
        ).fetchone()
        if row is None:
            return None
        return {
            "cnt": row["cnt"],
            "segment_by_shard": json.loads(row["segment_by_shard_json"]),
            "tokens": json.loads(row["tokens_json"]),
            "api_latency_ms": row["api_latency_ms"],
        }

    def put_find(
        self, index: str, query_string: str, form: str, cnt: int,
        segment_by_shard: list, tokens: list, api_latency_ms: float,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO finds (index_name, query_string, form, cnt, "
            "segment_by_shard_json, tokens_json, api_latency_ms, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (index, query_string, form, cnt, json.dumps(segment_by_shard),
             json.dumps(tokens, ensure_ascii=False), api_latency_ms, _now_iso()),
        )
        self.conn.commit()

    def get_window(self, index: str, query_string: str, form: str, s: int, rank: int) -> dict | None:
        """Row as the window dict (see InfinigramClient.get_doc_by_rank), or None."""
        row = self.conn.execute(
            "SELECT doc_ix, doc_len, disp_len, needle_offset, spans_json, metadata_json, "
            "api_latency_ms, fetched_at FROM windows "
            "WHERE index_name = ? AND query_string = ? AND form = ? AND s = ? AND rank = ?",
            (index, query_string, form, s, rank),
        ).fetchone()
        if row is None:
            return None
        return {
            "doc_ix": row["doc_ix"],
            "doc_len": row["doc_len"],
            "disp_len": row["disp_len"],
            "needle_offset": row["needle_offset"],
            "spans": json.loads(row["spans_json"]),
            "metadata": row["metadata_json"],
            "api_latency_ms": row["api_latency_ms"],
            "fetched_at": row["fetched_at"],
        }

    def put_window(
        self, index: str, query_string: str, form: str, s: int, rank: int, doc: dict,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO windows (index_name, query_string, form, s, rank, doc_ix, "
            "doc_len, disp_len, needle_offset, spans_json, metadata_json, api_latency_ms, "
            "fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (index, query_string, form, s, rank, doc.get("doc_ix"), doc.get("doc_len"),
             doc.get("disp_len"), doc.get("needle_offset"),
             json.dumps(doc["spans"], ensure_ascii=False), doc.get("metadata"),
             doc.get("api_latency_ms"), _now_iso()),
        )
        self.conn.commit()

    def log_error(
        self, index: str, query_type: str, request: dict,
        http_status: int | None, body_snippet: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO errors (ts, index_name, query_type, request_json, http_status, "
            "body_snippet) VALUES (?, ?, ?, ?, ?, ?)",
            (_now_iso(), index, query_type, json.dumps(request, ensure_ascii=False, sort_keys=True),
             http_status, body_snippet[:SNIPPET_CHARS]),
        )
        self.conn.commit()

    def error_counts(self) -> dict[str, int]:
        """{"400": n, ...}: errors by HTTP status ("none" for timeouts), for `status`."""
        rows = self.conn.execute(
            "SELECT COALESCE(CAST(http_status AS TEXT), 'none') AS status, COUNT(*) AS n "
            "FROM errors GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def last_error(self) -> dict | None:
        row = self.conn.execute(
            "SELECT ts, index_name, query_type, request_json, http_status, body_snippet "
            "FROM errors ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None

    # -- plain counts (COUNTS_SCHEMA; not part of SCHEMA) ----------------------
    def ensure_counts_table(self) -> None:
        """Create the `counts` table if it is missing. The only place it is created."""
        self.conn.executescript(COUNTS_SCHEMA)
        self.conn.commit()

    def get_count(self, index: str, query: str) -> dict | None:
        """Row as {"count", "approx" (bool, or None when the response had no
        `approx`), "api_latency_ms"}, or None. Needs ensure_counts_table()."""
        row = self.conn.execute(
            "SELECT cnt, approx, api_latency_ms FROM counts WHERE index_name = ? AND query = ?",
            (index, query),
        ).fetchone()
        if row is None:
            return None
        return {
            "count": row["cnt"],
            "approx": None if row["approx"] is None else bool(row["approx"]),
            "api_latency_ms": row["api_latency_ms"],
        }

    def put_count(
        self, index: str, query: str, count: int, approx: bool | None, api_latency_ms: float | None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO counts (index_name, query, cnt, approx, api_latency_ms, "
            "fetched_at) VALUES (?, ?, ?, ?, ?, ?)",
            (index, query, count, None if approx is None else int(bool(approx)),
             api_latency_ms, _now_iso()),
        )
        self.conn.commit()


class InfinigramClient:
    """find / get_doc_by_rank / tokenize over a Transport, with the Cache first.

    `(query_string, form)` is the cache key; exactly one of `query` / `query_ids`
    is sent (ValueError if neither or both). For form "plain" the caller passes
    query=query_string; for "glued", query_ids=<glued ids>.

    Every method returns None when the request failed for good (logged in
    errors; the item stays unfetched and is retried on the next run) and the
    result otherwise. Results carry "from_cache": bool. `requests_sent` counts
    network requests started (retries included); once it reaches `max_requests`
    the next network call raises MaxRequestsReached instead of sending.

    Error policy:
      timeout / connection error / 5xx  retry, backoff 2, 4, 8 ... capped at
                                        BACKOFF_CAP_S, MAX_ATTEMPTS attempts,
                                        then log and return None
      403 or 429 (a block)              stop sending, sleep BLOCKED_SLEEP_S; the
                                        MAX_CONSECUTIVE_403-th in a row raises
                                        Blocked; any 2xx resets the count
      400 and other 4xx, 3xx            log and return None (our bug), no retry
      bad 200 body                      log and return None, never cached
    Backoff and block sleeps go through Throttle.pause, so nothing ever bursts.

    The block is persisted (run_meta `blocked_until` = wall time + BLOCKED_SLEEP_S,
    and the running count) BEFORE the sleep, so Ctrl+C or a crash followed by a
    rerun cannot skip the wait or reset the "3 in a row" rule. A new client
    loads both: before its first request it waits out the remainder (through
    Throttle.pause, at most BLOCKED_SLEEP_S in total, so a corrupt or
    future-dated value cannot stall for hours; nan, inf and negative values are
    ignored). A 2xx clears both.

    Circuit breaker: every request that fails for good (a give-up: retries
    exhausted, 400 and other non-2xx, or a bad 200 body) counts; any success
    resets the count. MAX_CONSECUTIVE_FAILURES in a row raise TooManyFailures,
    so a systematic error cannot burn a 100k-request run. The counter is per
    client instance (`consecutive_failures`). 403/429 are not give-ups: they
    have their own rule above.
    """

    def __init__(
        self,
        transport: Transport,
        throttle: Throttle,
        cache: Cache,
        max_requests: int | None = None,
    ):
        self.transport = transport
        self.throttle = throttle
        self.cache = cache
        self.max_requests = max_requests
        self._requests_sent = 0
        # Across calls on purpose: a block does not lift between two items.
        # Loaded from the cache, so it also survives a restart.
        count = _finite(cache.get_meta(CONSECUTIVE_403_KEY))
        self.consecutive_403 = min(int(count), MAX_CONSECUTIVE_403) if count is not None and count > 0 else 0
        until = _finite(cache.get_meta(BLOCKED_UNTIL_KEY))
        # Waited out before the next network request, then dropped from memory
        # (the persisted value stays until a success clears it).
        self._pending_block_until = until if until is not None and until > 0 else None
        self._block_persisted = self._pending_block_until is not None or self.consecutive_403 > 0
        self.consecutive_failures = 0

    @property
    def requests_sent(self) -> int:
        return self._requests_sent

    # -- block state ----------------------------------------------------------
    def _persist_block(self) -> float:
        """Write blocked_until and the running count (one commit); return blocked_until."""
        until = self.throttle.wall() + BLOCKED_SLEEP_S
        self.cache.update_meta({
            BLOCKED_UNTIL_KEY: repr(until),
            CONSECUTIVE_403_KEY: str(self.consecutive_403),
        })
        self._block_persisted = True
        return until

    def _clear_block(self) -> None:
        """A 2xx proves we are not blocked: reset the count, in memory and on disk."""
        self.consecutive_403 = 0
        self._pending_block_until = None
        if self._block_persisted:
            self.cache.update_meta({}, delete=(BLOCKED_UNTIL_KEY, CONSECUTIVE_403_KEY))
            self._block_persisted = False

    def _honour_persisted_block(self) -> None:
        until, self._pending_block_until = self._pending_block_until, None
        if until is None:
            return
        remaining = until - self.throttle.wall()
        if remaining > 0:
            self.throttle.pause(min(remaining, BLOCKED_SLEEP_S))

    def _give_up(self, index: str, query_type: str, body: dict, status, payload) -> None:
        """Log a request that failed for good and feed the circuit breaker."""
        self.cache.log_error(index, query_type, body, status, _snippet(payload))
        self.consecutive_failures += 1
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise TooManyFailures(
                "too many consecutive failures (%d in a row, last: %s); check the "
                "error log (status) and rerun, the cache resumes"
                % (self.consecutive_failures, "HTTP %s" % status if status is not None else "no response")
            )

    # -- the one place a request leaves the process --------------------------
    def _request(self, index: str, query_type: str, body: dict, required: tuple[str, ...]) -> dict | None:
        """POST with the error policy. The parsed body, or None after logging."""
        transient = 0
        while True:
            # The budget comes first: nothing below (block wait, throttle,
            # persist) may run for a request that will not be sent.
            if self.max_requests is not None and self._requests_sent >= self.max_requests:
                raise MaxRequestsReached("request budget spent (%d)" % self.max_requests)
            self._honour_persisted_block()
            wall = self.throttle.wait()
            # Persist before sending: a kill mid-request must still leave the
            # start time behind for the restart to wait out. Only then stamp
            # the start, so a slow commit cannot shorten the real gap.
            self.cache.set_last_request_wall(wall)
            self.throttle.mark_started()
            self._requests_sent += 1
            try:
                status, payload = self.transport.post(body)
            except OSError as e:
                status, payload = None, "%s: %s" % (type(e).__name__, e)

            if status is not None and 200 <= status < 300:
                self._clear_block()  # a 2xx proves we are not blocked
                if isinstance(payload, dict) and all(k in payload for k in required):
                    self.consecutive_failures = 0
                    return payload
                # Unexpected shape (an {"error": ...} body, a changed API): a
                # failure, never a cached result.
                self._give_up(index, query_type, body, status, payload)
                return None

            if status in BLOCK_STATUSES:
                self.consecutive_403 += 1
                self.cache.log_error(index, query_type, body, status, _snippet(payload))
                # Persisted before any sleep or raise: Ctrl+C and a rerun must
                # not skip the wait, and a restart must not reset the count.
                until = self._persist_block()
                if self.consecutive_403 >= MAX_CONSECUTIVE_403:
                    self._pending_block_until = until  # a reused client waits too
                    raise Blocked(
                        "blocked by infini-gram (HTTP 403/429, %d in a row); wait and rerun, "
                        "the cache resumes" % self.consecutive_403
                    )
                self.throttle.pause(BLOCKED_SLEEP_S)
                continue

            if status is None or status >= 500:
                transient += 1
                if transient >= MAX_ATTEMPTS:
                    self._give_up(index, query_type, body, status, payload)
                    return None
                self.throttle.pause(min(2 ** transient, BACKOFF_CAP_S))
                continue

            # 400 is our bug, other 4xx and 3xx are not retryable either: log
            # and move on (counted by the circuit breaker).
            self._give_up(index, query_type, body, status, payload)
            return None

    def tokenize(self, text: str, index: str = TOKENIZE_INDEX) -> tuple[list[int], list[str]] | None:
        """token ids and tokens of `text`, via a `count` call; cached by text,
        index-independent (all five indexes use the Llama-2 tokenizer)."""
        cached = self.cache.get_glued_ids(text)
        if cached is not None:
            return cached
        body = {"index": index, "query_type": "count", "query": text}
        payload = self._request(index, "count", body, ("token_ids", "tokens"))
        if payload is None:
            return None
        ids, tokens = list(payload["token_ids"]), list(payload["tokens"])
        self.cache.put_glued_ids(text, ids, tokens)
        return ids, tokens

    def find(
        self, index: str, query_string: str, form: str, *,
        query: str | None = None, query_ids: list[int] | None = None,
    ) -> dict | None:
        """{"cnt", "segment_by_shard": [[start, end), ...], "tokens", "api_latency_ms", "from_cache"}"""
        _check_query(query, query_ids)
        cached = self.cache.get_find(index, query_string, form)
        if cached is not None:
            cached["from_cache"] = True
            return cached
        body = {"index": index, "query_type": "find", **_query_part(query, query_ids)}
        payload = self._request(index, "find", body, ("cnt", "segment_by_shard"))
        if payload is None:
            return None
        result = {
            "cnt": payload["cnt"],
            "segment_by_shard": payload["segment_by_shard"],
            "tokens": payload.get("tokens", []),
            "api_latency_ms": payload.get("latency"),  # milliseconds
        }
        self.cache.put_find(
            index, query_string, form, result["cnt"], result["segment_by_shard"],
            result["tokens"], result["api_latency_ms"],
        )
        result["from_cache"] = False
        return result

    def get_doc_by_rank(
        self, index: str, query_string: str, form: str, s: int, rank: int, *,
        query: str | None = None, query_ids: list[int] | None = None,
        max_disp_len: int = DISP_LEN,
    ) -> dict | None:
        """{"doc_ix", "doc_len", "disp_len", "needle_offset", "spans", "metadata"
        (RAW string, never parsed or required), "api_latency_ms", "from_cache"}"""
        _check_query(query, query_ids)
        cached = self.cache.get_window(index, query_string, form, s, rank)
        if cached is not None:
            cached["from_cache"] = True
            return cached
        body = {
            "index": index, "query_type": "get_doc_by_rank", "s": s, "rank": rank,
            **_query_part(query, query_ids), "max_disp_len": max_disp_len,
        }
        payload = self._request(index, "get_doc_by_rank", body, ("doc_ix", "spans"))
        if payload is None:
            return None
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, str):
            metadata = json.dumps(metadata, ensure_ascii=False)  # keep it raw, as text
        doc = {
            "doc_ix": payload["doc_ix"],
            "doc_len": payload.get("doc_len"),
            "disp_len": payload.get("disp_len"),  # can be < max_disp_len near a document edge; accepted
            "needle_offset": payload.get("needle_offset"),
            "spans": payload["spans"],
            "metadata": metadata,
            "api_latency_ms": payload.get("latency"),  # milliseconds
        }
        self.cache.put_window(index, query_string, form, s, rank, doc)
        return {**doc, "from_cache": False}

    def count(self, index: str, query: str) -> dict | None:
        """{"count", "approx" (bool or None), "api_latency_ms", "from_cache"}: a plain
        infini-gram `count` of `query` on `index`, cached in `counts` by (index, query).
        Cache first; the request goes through _request, so the throttle, retries,
        block handling, circuit breaker and max_requests all apply. A failed request,
        or a 200 body whose `count` is not an integer >= 0, returns None and is never
        cached. The cache needs ensure_counts_table() first."""
        cached = self.cache.get_count(index, query)
        if cached is not None:
            cached["from_cache"] = True
            return cached
        body = {"index": index, "query_type": "count", "query": query}
        prev_failures = self.consecutive_failures
        payload = self._request(index, "count", body, ("count",))
        if payload is None:
            return None
        cnt = payload["count"]
        if isinstance(cnt, bool) or not isinstance(cnt, int) or cnt < 0:
            # _request took the body as a success and reset the breaker; undo that,
            # so a systematic bad body still reaches MAX_CONSECUTIVE_FAILURES.
            self.consecutive_failures = prev_failures
            self._give_up(index, "count", body, 200, payload)
            return None
        approx = payload.get("approx")
        approx = None if approx is None else bool(approx)
        result = {"count": cnt, "approx": approx, "api_latency_ms": payload.get("latency")}
        self.cache.put_count(index, query, cnt, approx, result["api_latency_ms"])
        result["from_cache"] = False
        return result


def _check_query(query, query_ids) -> None:
    # Before any cache or network access: sending neither or both is a 400.
    if (query is None) == (query_ids is None):
        raise ValueError("exactly one of query / query_ids must be given")


def _query_part(query, query_ids) -> dict:
    return {"query": query} if query is not None else {"query_ids": list(query_ids)}


def _snippet(payload) -> str:
    if isinstance(payload, str):
        return payload[:SNIPPET_CHARS]
    try:
        return json.dumps(payload, ensure_ascii=False)[:SNIPPET_CHARS]
    except (TypeError, ValueError):
        return repr(payload)[:SNIPPET_CHARS]
