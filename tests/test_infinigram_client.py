"""Tests for hallutrace/infinigram.py. No network, no real sleep, temp caches only.

Run from the repo root:  python -m unittest tests.test_infinigram_client -v
A fake Transport records every body it is sent; a fake clock's sleep advances
both the monotonic and the wall clock, so spacing between request starts can be
asserted exactly.
"""

import http.client
import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request

import hallutrace.infinigram as ig
from hallutrace.infinigram import (
    BLOCKED_SLEEP_S,
    BLOCKED_UNTIL_KEY,
    CONSECUTIVE_403_KEY,
    GLOBAL_LAST_NAME,
    GLOBAL_LOCK_NAME,
    LOCK_DIR_ENV,
    MAX_ATTEMPTS,
    MAX_CONSECUTIVE_FAILURES,
    MIN_GAP_S,
    TOKENIZE_INDEX,
    Blocked,
    Cache,
    HttpTransport,
    InfinigramClient,
    InfinigramError,
    LockHeld,
    MaxRequestsReached,
    Throttle,
    TooManyFailures,
)

INDEX = "v4_olmo-mix-1124_llama"
EPS = 1e-9


class FakeTime:
    """One timeline for monotonic and wall time; sleep just moves it forward.

    `interrupt_on`: the first sleep of at least this many seconds raises
    KeyboardInterrupt instead of advancing (Ctrl+C during a long pause).
    """

    def __init__(self, mono=100.0, wall=1_000_000.0):
        self.mono_now = mono
        self.wall_now = wall
        self.sleeps = []
        self.interrupt_on = None

    def mono(self):
        return self.mono_now

    def wall(self):
        return self.wall_now

    def sleep(self, seconds):
        if self.interrupt_on is not None and seconds >= self.interrupt_on:
            self.interrupt_on = None
            raise KeyboardInterrupt()
        if len(self.sleeps) > 10_000:  # a runaway wait loop must fail, not eat memory
            raise AssertionError("more than 10,000 sleeps")
        self.sleeps.append(seconds)
        self.mono_now += seconds
        self.wall_now += seconds


FIND_OK = {"cnt": 159, "segment_by_shard": [[0, 100], [0, 59]], "tokens": ["▁Ober", "to"], "latency": 12.5}
DOC_OK = {
    "doc_ix": 7, "doc_len": 5000, "disp_len": 461, "needle_offset": 200,
    "spans": [["before ", None], ["Oberto Pallavicino", "0"], [" after", None]],
    "metadata": '{"metadata": {"url": "http://x.example/a"}}',
    "token_ids": [1], "tokens": ["a"], "blocked": False, "latency": 3.0,
}
COUNT_OK = {"count": 12, "approx": False, "latency": 1.0, "token_ids": [29898, 5], "tokens": ["▁(", "Ober"]}


class FakeTransport:
    """Replays scripted responses in order; an entry may be an exception to raise.

    Records every body and the fake time it arrived at. When the script runs
    out, `default` is returned.
    """

    def __init__(self, ft, script=(), default=(200, FIND_OK), cache=None, duration=0.8):
        self.ft = ft
        self.script = list(script)
        self.default = default
        self.cache = cache
        self.duration = duration
        self.bodies = []
        self.starts = []
        self.persisted_at_send = []

    def post(self, body):
        self.bodies.append(body)
        self.starts.append(self.ft.mono_now)
        if self.cache is not None:
            self.persisted_at_send.append(self.cache.last_request_wall())
        self.ft.mono_now += self.duration  # the request takes time
        self.ft.wall_now += self.duration
        item = self.script.pop(0) if self.script else self.default
        if isinstance(item, BaseException):
            raise item
        return item


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "sub", "cache.sqlite")
        # The per-user lock and last-request file live in a per-test directory,
        # never in the real temp dir (which a real fetcher may be holding).
        self.lock_dir = os.path.join(self._tmp.name, "userlock")
        self._old_lock_env = os.environ.get(LOCK_DIR_ENV)
        os.environ[LOCK_DIR_ENV] = self.lock_dir
        self.ft = FakeTime()
        self._caches = []

    def tearDown(self):
        if self._old_lock_env is None:
            os.environ.pop(LOCK_DIR_ENV, None)
        else:
            os.environ[LOCK_DIR_ENV] = self._old_lock_env

    def cache(self, lock=True, path=None):
        c = Cache(path or self.path, lock=lock)
        self._caches.append(c)
        self.addCleanup(c.close)
        return c

    def make(self, script=(), default=(200, FIND_OK), max_requests=None, cache=None):
        cache = cache or self.cache()
        throttle = Throttle(
            MIN_GAP_S, self.ft.mono, self.ft.sleep,
            last_start_wall=cache.last_request_wall(), wall=self.ft.wall,
        )
        transport = FakeTransport(self.ft, script, default, cache=cache)
        client = InfinigramClient(transport, throttle, cache, max_requests=max_requests)
        return client, transport, cache

    def assert_spaced(self, starts, gap=MIN_GAP_S):
        for a, b in zip(starts, starts[1:]):
            self.assertGreaterEqual(b - a, gap - EPS, "starts %r too close" % ((a, b),))


class TestThrottle(Base):
    def test_five_requests_spaced(self):
        client, tr, _ = self.make()
        for i in range(5):
            client.find(INDEX, "q%d" % i, "plain", query="q%d" % i)
        self.assertEqual(len(tr.starts), 5)
        self.assert_spaced(tr.starts)

    def test_first_request_does_not_wait(self):
        client, tr, _ = self.make()
        client.find(INDEX, "q", "plain", query="q")
        self.assertEqual(self.ft.sleeps, [])

    def test_spacing_holds_after_backoff_sleeps(self):
        # Two failures then a success, then more requests: nothing starts early.
        client, tr, _ = self.make(script=[(500, "boom"), OSError("timeout")])
        client.find(INDEX, "a", "plain", query="a")
        for i in range(3):
            client.find(INDEX, "b%d" % i, "plain", query="b%d" % i)
        self.assertEqual(client.requests_sent, 6)
        self.assert_spaced(tr.starts)

    def test_spacing_holds_after_403_pause(self):
        client, tr, _ = self.make(script=[(403, {"message": "Forbidden"})])
        client.find(INDEX, "a", "plain", query="a")
        client.find(INDEX, "b", "plain", query="b")
        self.assert_spaced(tr.starts)
        self.assertGreaterEqual(tr.starts[1] - tr.starts[0], BLOCKED_SLEEP_S)

    def test_pause_does_not_add_a_second_gap(self):
        # The gap is measured from the previous START, so a 2 s backoff after a
        # 0.8 s request leaves the next start at start + 2.8, not + 2.8 + 1.05.
        client, tr, _ = self.make(script=[(500, "x")])
        client.find(INDEX, "a", "plain", query="a")
        self.assertAlmostEqual(tr.starts[1] - tr.starts[0], 0.8 + 2.0)

    def test_restart_waits_out_the_gap_from_persisted_wall_time(self):
        client, tr, cache = self.make()
        client.find(INDEX, "a", "plain", query="a")
        sent_wall = cache.last_request_wall()
        cache.close()

        # "New process": fresh Cache, fresh Throttle seeded from the file, and
        # only 0.2 s of wall time since the last start.
        self.ft.wall_now = sent_wall + 0.2
        self.ft.sleeps.clear()
        client2, tr2, _ = self.make(cache=self.cache())
        client2.find(INDEX, "b", "plain", query="b")
        self.assertEqual(len(self.ft.sleeps), 1)
        self.assertAlmostEqual(self.ft.sleeps[0], MIN_GAP_S - 0.2)
        self.assertGreaterEqual(self.ft.wall_now - sent_wall, MIN_GAP_S - EPS)

    def test_restart_after_long_gap_does_not_sleep(self):
        client, tr, cache = self.make()
        client.find(INDEX, "a", "plain", query="a")
        sent_wall = cache.last_request_wall()
        cache.close()
        self.ft.wall_now = sent_wall + 3600
        self.ft.sleeps.clear()
        client2, _, _ = self.make(cache=self.cache())
        client2.find(INDEX, "b", "plain", query="b")
        self.assertEqual(self.ft.sleeps, [])

    def test_wall_time_persisted_before_the_request_is_sent(self):
        client, tr, cache = self.make()
        client.find(INDEX, "a", "plain", query="a")
        self.assertEqual(len(tr.persisted_at_send), 1)
        self.assertIsNotNone(tr.persisted_at_send[0])
        self.assertLess(tr.persisted_at_send[0], self.ft.wall_now)

    def test_future_persisted_timestamp_cannot_stall_the_run(self):
        # A skewed clock: last start "in the future". The TOTAL wait is one gap.
        cache = self.cache()
        cache.set_last_request_wall(self.ft.wall_now + 10_000)
        client, _, _ = self.make(cache=cache)
        client.find(INDEX, "a", "plain", query="a")
        self.assertAlmostEqual(sum(self.ft.sleeps), MIN_GAP_S)


class TestLock(Base):
    def test_second_cache_on_same_path_refuses(self):
        first = self.cache()
        with self.assertRaises(LockHeld) as ctx:
            Cache(self.path)
        msg = str(ctx.exception)
        self.assertIn("delete the .lock file", msg)
        self.assertIn("pid %d" % os.getpid(), msg)  # says who holds it
        first.close()

    def test_lock_released_on_close(self):
        first = self.cache()
        first.close()
        second = self.cache()  # must not raise
        self.assertIsNotNone(second.conn)

    def test_context_manager_releases(self):
        with Cache(self.path):
            with self.assertRaises(LockHeld):
                Cache(self.path)
        with Cache(self.path):
            pass

    def test_lock_false_readers_coexist_with_a_writer(self):
        writer = self.cache()
        writer.set_meta("k", "v")
        reader = self.cache(lock=False)
        self.assertEqual(reader.get_meta("k"), "v")

    def test_failed_open_does_not_disturb_the_holder(self):
        first = self.cache()
        for _ in range(2):
            with self.assertRaises(LockHeld):
                Cache(self.path)
        first.set_meta("still", "works")
        self.assertEqual(first.get_meta("still"), "works")


class TestCacheHits(Base):
    def test_cached_find_is_never_rerequested(self):
        client, tr, _ = self.make()
        r1 = client.find(INDEX, "Oberto Pallavicino", "plain", query="Oberto Pallavicino")
        r2 = client.find(INDEX, "Oberto Pallavicino", "plain", query="Oberto Pallavicino")
        self.assertFalse(r1["from_cache"])
        self.assertTrue(r2["from_cache"])
        self.assertEqual(len(tr.bodies), 1)
        self.assertEqual(client.requests_sent, 1)
        self.assertEqual(r2["cnt"], 159)
        self.assertEqual(r2["segment_by_shard"], [[0, 100], [0, 59]])
        self.assertEqual(r2["api_latency_ms"], 12.5)

    def test_cached_window_is_never_rerequested(self):
        client, tr, _ = self.make(default=(200, DOC_OK))
        kw = dict(query="Oberto Pallavicino")
        r1 = client.get_doc_by_rank(INDEX, "Oberto Pallavicino", "plain", 3, 17, **kw)
        r2 = client.get_doc_by_rank(INDEX, "Oberto Pallavicino", "plain", 3, 17, **kw)
        self.assertFalse(r1["from_cache"])
        self.assertTrue(r2["from_cache"])
        self.assertEqual(len(tr.bodies), 1)
        self.assertEqual(r2["spans"], DOC_OK["spans"])
        self.assertEqual(r2["doc_ix"], 7)
        self.assertEqual(r2["disp_len"], 461)

    def test_cached_tokenize_is_never_rerequested(self):
        client, tr, _ = self.make(default=(200, COUNT_OK))
        a = client.tokenize("(Oberto Pallavicino")
        b = client.tokenize("(Oberto Pallavicino")
        self.assertEqual(a, b)
        self.assertEqual(a, ([29898, 5], ["▁(", "Ober"]))
        self.assertEqual(len(tr.bodies), 1)
        self.assertEqual(tr.bodies[0], {"index": TOKENIZE_INDEX, "query_type": "count", "query": "(Oberto Pallavicino"})

    def test_cache_survives_reopen(self):
        client, tr, cache = self.make()
        client.find(INDEX, "a", "plain", query="a")
        cache.close()
        client2, tr2, _ = self.make(cache=self.cache())
        r = client2.find(INDEX, "a", "plain", query="a")
        self.assertTrue(r["from_cache"])
        self.assertEqual(tr2.bodies, [])

    def test_form_and_index_are_part_of_the_key(self):
        client, tr, _ = self.make(default=(200, DOC_OK))
        client.get_doc_by_rank(INDEX, "n", "plain", 0, 0, query="n")
        client.get_doc_by_rank(INDEX, "n", "glued", 0, 0, query_ids=[1, 2])
        client.get_doc_by_rank("v4_dclm-baseline_llama", "n", "plain", 0, 0, query="n")
        self.assertEqual(len(tr.bodies), 3)

    def test_cache_hit_does_not_touch_throttle_or_budget(self):
        client, tr, _ = self.make(max_requests=1)
        client.find(INDEX, "a", "plain", query="a")
        sleeps = list(self.ft.sleeps)
        for _ in range(3):
            self.assertTrue(client.find(INDEX, "a", "plain", query="a")["from_cache"])
        self.assertEqual(self.ft.sleeps, sleeps)
        self.assertEqual(client.requests_sent, 1)

    def test_metadata_stored_raw(self):
        # Not valid JSON and not the DCLM shape: still stored byte for byte.
        raw = 'path=c4-filtered/x  {not json'
        doc = dict(DOC_OK, metadata=raw)
        client, _, cache = self.make(default=(200, doc))
        r = client.get_doc_by_rank(INDEX, "n", "plain", 0, 0, query="n")
        self.assertEqual(r["metadata"], raw)
        again = client.get_doc_by_rank(INDEX, "n", "plain", 0, 0, query="n")
        self.assertEqual(again["metadata"], raw)
        row = cache.conn.execute("SELECT metadata_json FROM windows").fetchone()
        self.assertEqual(row["metadata_json"], raw)

    def test_missing_metadata_is_accepted(self):
        doc = {k: v for k, v in DOC_OK.items() if k != "metadata"}
        client, _, _ = self.make(default=(200, doc))
        r = client.get_doc_by_rank(INDEX, "n", "plain", 0, 0, query="n")
        self.assertIsNone(r["metadata"])

    def test_non_ascii_round_trip(self):
        name = "Margareta Cederschiöld"
        doc = dict(DOC_OK, spans=[["x ", None], [name, "0"]])
        client, _, _ = self.make(default=(200, doc))
        client.get_doc_by_rank(INDEX, name, "plain", 0, 0, query=name)
        r = client.get_doc_by_rank(INDEX, name, "plain", 0, 0, query=name)
        self.assertEqual(r["spans"][1][0], name)

    def test_schema_has_all_tables_and_active_default(self):
        cache = self.cache()
        names = {r["name"] for r in cache.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"entities", "glued_ids", "finds", "plan", "windows", "errors", "run_meta"} <= names)
        cache.conn.execute("INSERT INTO entities (prompt_name, query_string, source) VALUES ('a', 'a', 'name')")
        self.assertEqual(cache.conn.execute("SELECT active FROM entities").fetchone()["active"], 1)
        mode = cache.conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode.lower(), "wal")

    def test_every_write_is_committed(self):
        client, _, _ = self.make()
        client.find(INDEX, "a", "plain", query="a")
        other = self.cache(lock=False)  # a second connection sees it at once
        self.assertIsNotNone(other.get_find(INDEX, "a", "plain"))


class TestErrors(Base):
    def test_403_pauses_600_and_retries(self):
        client, tr, cache = self.make(script=[(403, {"message": "Forbidden"})])
        r = client.find(INDEX, "a", "plain", query="a")
        self.assertFalse(r["from_cache"])
        self.assertIn(BLOCKED_SLEEP_S, self.ft.sleeps)
        self.assertEqual(client.requests_sent, 2)
        self.assertEqual(cache.error_counts(), {"403": 1})
        self.assertEqual(client.consecutive_403, 0)  # the success reset it

    def test_third_consecutive_403_raises_blocked_without_sleeping(self):
        client, tr, cache = self.make(default=(403, {"message": "Forbidden"}))
        with self.assertRaises(Blocked) as ctx:
            client.find(INDEX, "a", "plain", query="a")
        self.assertIn("wait and rerun", str(ctx.exception))
        self.assertEqual(client.requests_sent, 3)
        self.assertEqual(self.ft.sleeps.count(BLOCKED_SLEEP_S), 2)  # none before the third
        self.assertEqual(cache.error_counts(), {"403": 3})
        self.assertIsNone(cache.get_find(INDEX, "a", "plain"))

    def test_403_count_carries_across_calls_and_success_resets(self):
        script = [(403, "x"), (200, FIND_OK), (403, "x"), (403, "x"), (200, FIND_OK)]
        client, tr, _ = self.make(script=script)
        client.find(INDEX, "a", "plain", query="a")  # 403, ok
        self.assertEqual(client.consecutive_403, 0)
        client.find(INDEX, "b", "plain", query="b")  # 403, 403, ok: 2 in a row, not blocked
        self.assertEqual(client.consecutive_403, 0)
        self.assertEqual(client.requests_sent, 5)

    def test_403_streak_spans_calls(self):
        # The streak lives on the client instance, not the call: 403 that ends a
        # call by other means must still count. Two 403 then a 400, then 403.
        script = [(403, "x"), (403, "x"), (400, "bad"), (403, "x")]
        client, tr, _ = self.make(script=script)
        with self.assertRaises(Blocked):
            client.find(INDEX, "a", "plain", query="a")
            client.find(INDEX, "b", "plain", query="b")

    def test_400_logged_and_skipped_without_retry(self):
        client, tr, cache = self.make(script=[(400, "Exactly one of query and query_ids must be present!")])
        self.assertIsNone(client.find(INDEX, "a", "plain", query="a"))
        self.assertEqual(len(tr.bodies), 1)
        self.assertEqual(cache.error_counts(), {"400": 1})
        err = cache.last_error()
        self.assertEqual(err["http_status"], 400)
        self.assertEqual(err["query_type"], "find")
        self.assertIn("Exactly one", err["body_snippet"])
        self.assertEqual(json.loads(err["request_json"])["query"], "a")
        self.assertIsNone(cache.get_find(INDEX, "a", "plain"))
        # The next item still goes through, on schedule.
        self.assertIsNotNone(client.find(INDEX, "b", "plain", query="b"))

    def test_other_4xx_logged_and_skipped(self):
        client, tr, cache = self.make(script=[(404, "nope")])
        self.assertIsNone(client.find(INDEX, "a", "plain", query="a"))
        self.assertEqual(len(tr.bodies), 1)
        self.assertEqual(cache.error_counts(), {"404": 1})

    def test_5xx_retries_with_exponential_backoff(self):
        client, tr, cache = self.make(script=[(503, "x"), (502, "x"), (500, "x")])
        r = client.find(INDEX, "a", "plain", query="a")
        self.assertIsNotNone(r)
        self.assertEqual(self.ft.sleeps, [2, 4, 8])
        self.assertEqual(client.requests_sent, 4)
        self.assertEqual(cache.error_counts(), {})  # recovered: nothing to report

    def test_oserror_retries_then_gives_up_after_max_attempts(self):
        client, tr, cache = self.make(default=OSError("timed out"))
        self.assertIsNone(client.find(INDEX, "a", "plain", query="a"))
        self.assertEqual(len(tr.bodies), MAX_ATTEMPTS)
        self.assertEqual(client.requests_sent, MAX_ATTEMPTS)
        self.assertEqual(self.ft.sleeps, [2, 4, 8, 16, 32])  # no sleep after the last
        self.assertEqual(cache.error_counts(), {"none": 1})
        self.assertIn("timed out", cache.last_error()["body_snippet"])
        self.assert_spaced(tr.starts)
        self.assertIsNone(cache.get_find(INDEX, "a", "plain"))

    def test_5xx_gives_up_and_logs_status(self):
        client, tr, cache = self.make(default=(500, "internal"))
        self.assertIsNone(client.find(INDEX, "a", "plain", query="a"))
        self.assertEqual(len(tr.bodies), MAX_ATTEMPTS)
        self.assertEqual(cache.error_counts(), {"500": 1})

    def test_backoff_is_capped(self):
        import hallutrace.infinigram as ig

        old = ig.MAX_ATTEMPTS
        ig.MAX_ATTEMPTS = 12
        try:
            client, tr, _ = self.make(default=(500, "x"))
            client.find(INDEX, "a", "plain", query="a")
        finally:
            ig.MAX_ATTEMPTS = old
        self.assertEqual(max(self.ft.sleeps), 300)
        self.assertEqual(self.ft.sleeps[:9], [2, 4, 8, 16, 32, 64, 128, 256, 300])

    def test_failed_request_is_not_cached_and_retries_next_run(self):
        client, tr, cache = self.make(default=(500, "down"))
        self.assertIsNone(client.get_doc_by_rank(INDEX, "n", "plain", 1, 2, query="n"))
        self.assertIsNone(cache.get_window(INDEX, "n", "plain", 1, 2))
        cache.close()
        # Next run, API is back: the item is requested again and now cached.
        client2, tr2, cache2 = self.make(default=(200, DOC_OK), cache=self.cache())
        r = client2.get_doc_by_rank(INDEX, "n", "plain", 1, 2, query="n")
        self.assertFalse(r["from_cache"])
        self.assertEqual(len(tr2.bodies), 1)
        self.assertIsNotNone(cache2.get_window(INDEX, "n", "plain", 1, 2))

    def test_unexpected_200_body_is_logged_not_cached(self):
        for bad in ({"error": "something"}, "<html>oops</html>", {"cnt": 3}):
            client, tr, cache = self.make(script=[(200, bad)])
            self.assertIsNone(client.find(INDEX, "a", "plain", query="a"))
            self.assertEqual(len(tr.bodies), 1)
            self.assertEqual(cache.error_counts(), {"200": 1})
            self.assertIsNone(cache.get_find(INDEX, "a", "plain"))
            cache.close()
            os.remove(self.path)
            for ext in ("-wal", "-shm"):
                if os.path.exists(self.path + ext):
                    os.remove(self.path + ext)

    def test_unexpected_window_and_count_bodies(self):
        client, tr, cache = self.make(script=[(200, {"spans": []}), (200, {"count": 3})])
        self.assertIsNone(client.get_doc_by_rank(INDEX, "n", "plain", 0, 0, query="n"))
        self.assertIsNone(client.tokenize("(n"))
        self.assertEqual(cache.error_counts(), {"200": 2})


class TestQueryArguments(Base):
    def test_find_plain_sends_only_query(self):
        client, tr, _ = self.make()
        client.find(INDEX, "n", "plain", query="n")
        self.assertEqual(tr.bodies, [{"index": INDEX, "query_type": "find", "query": "n"}])

    def test_find_glued_sends_only_query_ids(self):
        client, tr, _ = self.make()
        client.find(INDEX, "n", "glued", query_ids=[5, 6])
        self.assertEqual(tr.bodies, [{"index": INDEX, "query_type": "find", "query_ids": [5, 6]}])

    def test_get_doc_by_rank_body(self):
        client, tr, _ = self.make(default=(200, DOC_OK))
        client.get_doc_by_rank(INDEX, "n", "plain", 4, 9, query="n")
        client.get_doc_by_rank(INDEX, "n", "glued", 4, 9, query_ids=[5, 6], max_disp_len=200)
        self.assertEqual(tr.bodies[0], {
            "index": INDEX, "query_type": "get_doc_by_rank", "s": 4, "rank": 9,
            "query": "n", "max_disp_len": 800,
        })
        self.assertEqual(tr.bodies[1], {
            "index": INDEX, "query_type": "get_doc_by_rank", "s": 4, "rank": 9,
            "query_ids": [5, 6], "max_disp_len": 200,
        })

    def test_neither_or_both_raises_before_any_cache_or_network_access(self):
        client, tr, cache = self.make()
        calls = [
            lambda: client.find(INDEX, "n", "plain"),
            lambda: client.find(INDEX, "n", "plain", query="n", query_ids=[1]),
            lambda: client.get_doc_by_rank(INDEX, "n", "plain", 0, 0),
            lambda: client.get_doc_by_rank(INDEX, "n", "plain", 0, 0, query="n", query_ids=[1]),
        ]
        for call in calls:
            with self.assertRaises(ValueError):
                call()
        self.assertEqual(tr.bodies, [])
        self.assertEqual(client.requests_sent, 0)
        self.assertIsNone(cache.last_request_wall())  # throttle never ran

    def test_empty_query_ids_list_still_counts_as_given(self):
        client, tr, _ = self.make()
        client.find(INDEX, "n", "glued", query_ids=[])
        self.assertEqual(tr.bodies[0]["query_ids"], [])
        self.assertNotIn("query", tr.bodies[0])


class TestMaxRequests(Base):
    def test_budget_stops_before_sending(self):
        client, tr, _ = self.make(max_requests=3)
        for i in range(3):
            client.find(INDEX, "q%d" % i, "plain", query="q%d" % i)
        with self.assertRaises(MaxRequestsReached):
            client.find(INDEX, "q3", "plain", query="q3")
        self.assertEqual(len(tr.bodies), 3)
        self.assertEqual(client.requests_sent, 3)

    def test_budget_counts_retries_and_stops_mid_retry(self):
        client, tr, cache = self.make(default=(500, "x"), max_requests=2)
        with self.assertRaises(MaxRequestsReached):
            client.find(INDEX, "a", "plain", query="a")
        self.assertEqual(len(tr.bodies), 2)
        self.assertIsNone(cache.get_find(INDEX, "a", "plain"))

    def test_raising_does_not_touch_the_persisted_wall_time(self):
        client, tr, cache = self.make(max_requests=1)
        client.find(INDEX, "a", "plain", query="a")
        before = cache.last_request_wall()
        with self.assertRaises(MaxRequestsReached):
            client.find(INDEX, "b", "plain", query="b")
        self.assertEqual(cache.last_request_wall(), before)

    def test_cache_hits_are_free_at_the_limit(self):
        client, tr, _ = self.make(max_requests=1)
        client.find(INDEX, "a", "plain", query="a")
        self.assertTrue(client.find(INDEX, "a", "plain", query="a")["from_cache"])

    def test_zero_budget_sends_nothing(self):
        client, tr, _ = self.make(max_requests=0)
        with self.assertRaises(MaxRequestsReached):
            client.find(INDEX, "a", "plain", query="a")
        self.assertEqual(tr.bodies, [])


class TestCacheHelpers(Base):
    def test_meta_round_trip_and_default(self):
        c = self.cache()
        self.assertIsNone(c.get_meta("nope"))
        self.assertEqual(c.get_meta("nope", "d"), "d")
        c.set_meta("seed", "20260916")
        c.set_meta("seed", "1")
        self.assertEqual(c.get_meta("seed"), "1")

    def test_last_request_wall_round_trip_keeps_precision(self):
        c = self.cache()
        self.assertIsNone(c.last_request_wall())
        c.set_last_request_wall(1_780_000_000.123456)
        self.assertEqual(c.last_request_wall(), 1_780_000_000.123456)

    def test_error_counts_and_last_error(self):
        c = self.cache()
        self.assertIsNone(c.last_error())
        c.log_error(INDEX, "find", {"query": "a"}, 400, "bad")
        c.log_error(INDEX, "find", {"query": "b"}, None, "timeout")
        c.log_error(INDEX, "find", {"query": "c"}, 400, "x" * 5000)
        self.assertEqual(c.error_counts(), {"400": 2, "none": 1})
        last = c.last_error()
        self.assertEqual(json.loads(last["request_json"]), {"query": "c"})
        self.assertLessEqual(len(last["body_snippet"]), 500)

    def test_glued_ids_keyed_by_text(self):
        c = self.cache()
        c.put_glued_ids("(Oberto", [1, 2], ["▁(", "Ober"])
        self.assertEqual(c.get_glued_ids("(Oberto"), ([1, 2], ["▁(", "Ober"]))
        self.assertIsNone(c.get_glued_ids("Oberto"))


class TestCircuitBreaker(Base):
    def test_constant(self):
        self.assertEqual(MAX_CONSECUTIVE_FAILURES, 25)
        self.assertTrue(issubclass(TooManyFailures, InfinigramError))

    def test_systematic_400_raises_on_the_25th_in_a_row(self):
        client, tr, cache = self.make(default=(400, "bad request"))
        for i in range(MAX_CONSECUTIVE_FAILURES - 1):
            self.assertIsNone(client.find(INDEX, "q%d" % i, "plain", query="q%d" % i))
        self.assertEqual(client.consecutive_failures, MAX_CONSECUTIVE_FAILURES - 1)
        with self.assertRaises(TooManyFailures) as ctx:
            client.find(INDEX, "last", "plain", query="last")
        self.assertIn("check the error log", str(ctx.exception))
        self.assertIn("the cache resumes", str(ctx.exception))
        self.assertEqual(client.requests_sent, MAX_CONSECUTIVE_FAILURES)
        # Every give-up was logged before the raise; nothing was cached.
        self.assertEqual(cache.error_counts(), {"400": MAX_CONSECUTIVE_FAILURES})
        self.assertIsNone(cache.get_find(INDEX, "last", "plain"))
        self.assert_spaced(tr.starts)

    def test_a_success_resets_the_count(self):
        script = (
            [(400, "x")] * (MAX_CONSECUTIVE_FAILURES - 1) + [(200, FIND_OK)]
            + [(400, "x")] * (MAX_CONSECUTIVE_FAILURES - 1)
        )
        client, _, _ = self.make(script=script)
        for i in range(len(script)):
            client.find(INDEX, "q%d" % i, "plain", query="q%d" % i)  # never raises
        self.assertEqual(client.consecutive_failures, MAX_CONSECUTIVE_FAILURES - 1)

    def test_counted_per_give_up_not_per_attempt(self):
        # A 5xx that exhausts its MAX_ATTEMPTS is ONE failure.
        client, _, cache = self.make(default=(500, "down"))
        for i in range(MAX_CONSECUTIVE_FAILURES - 1):
            self.assertIsNone(client.find(INDEX, "q%d" % i, "plain", query="q%d" % i))
        self.assertEqual(client.requests_sent, (MAX_CONSECUTIVE_FAILURES - 1) * MAX_ATTEMPTS)
        with self.assertRaises(TooManyFailures):
            client.find(INDEX, "last", "plain", query="last")
        self.assertEqual(client.requests_sent, MAX_CONSECUTIVE_FAILURES * MAX_ATTEMPTS)
        self.assertEqual(cache.error_counts(), {"500": MAX_CONSECUTIVE_FAILURES})

    def test_a_bad_200_body_counts(self):
        client, _, cache = self.make(default=(200, {"error": "something"}))
        with self.assertRaises(TooManyFailures):
            for i in range(MAX_CONSECUTIVE_FAILURES):
                client.find(INDEX, "q%d" % i, "plain", query="q%d" % i)
        self.assertEqual(cache.error_counts(), {"200": MAX_CONSECUTIVE_FAILURES})

    def test_3xx_and_exhausted_timeouts_count_too(self):
        for default in ((302, "moved"), OSError("timed out")):
            client, _, cache = self.make(default=default)
            with self.assertRaises(TooManyFailures):
                for i in range(MAX_CONSECUTIVE_FAILURES):
                    client.find(INDEX, "q%d" % i, "plain", query="q%d" % i)
            cache.close()

    def test_a_block_response_neither_counts_nor_resets(self):
        script = [(400, "x")] * (MAX_CONSECUTIVE_FAILURES - 1) + [(403, "x"), (400, "x")]
        client, _, _ = self.make(script=script)
        for i in range(MAX_CONSECUTIVE_FAILURES - 1):
            client.find(INDEX, "q%d" % i, "plain", query="q%d" % i)
        with self.assertRaises(TooManyFailures):  # 403, pause, then the 25th give-up
            client.find(INDEX, "last", "plain", query="last")

    def test_the_counter_is_per_client_instance(self):
        client, _, cache = self.make(default=(400, "x"))
        with self.assertRaises(TooManyFailures):
            for i in range(MAX_CONSECUTIVE_FAILURES):
                client.find(INDEX, "q%d" % i, "plain", query="q%d" % i)
        cache.close()
        client2, tr2, _ = self.make(default=(200, FIND_OK), cache=self.cache())
        self.assertEqual(client2.consecutive_failures, 0)
        self.assertIsNotNone(client2.find(INDEX, "again", "plain", query="again"))
        self.assertEqual(len(tr2.bodies), 1)


class TestRateLimitStatus429(Base):
    def test_429_is_handled_like_403(self):
        client, tr, cache = self.make(script=[(429, {"message": "Too Many Requests"})])
        r = client.find(INDEX, "a", "plain", query="a")
        self.assertFalse(r["from_cache"])
        self.assertIn(BLOCKED_SLEEP_S, self.ft.sleeps)
        self.assertEqual(client.requests_sent, 2)
        self.assertEqual(cache.error_counts(), {"429": 1})
        self.assertEqual(client.consecutive_403, 0)
        self.assert_spaced(tr.starts)

    def test_third_429_in_a_row_raises_blocked_naming_429(self):
        client, tr, cache = self.make(default=(429, "slow down"))
        with self.assertRaises(Blocked) as ctx:
            client.find(INDEX, "a", "plain", query="a")
        self.assertIn("429", str(ctx.exception))
        self.assertIn("403", str(ctx.exception))
        self.assertEqual(client.requests_sent, 3)
        self.assertEqual(self.ft.sleeps.count(BLOCKED_SLEEP_S), 2)
        self.assertEqual(cache.error_counts(), {"429": 3})

    def test_403_and_429_share_one_streak(self):
        client, _, cache = self.make(script=[(403, "x"), (429, "x"), (403, "x")])
        with self.assertRaises(Blocked):
            client.find(INDEX, "a", "plain", query="a")
        self.assertEqual(cache.error_counts(), {"403": 2, "429": 1})

    def test_block_statuses_constant(self):
        self.assertEqual(set(ig.BLOCK_STATUSES), {403, 429})


class TestPersistedBlock(Base):
    def test_block_is_persisted_before_the_sleep(self):
        client, tr, cache = self.make(script=[(403, "x")])
        w0 = self.ft.wall_now
        self.ft.interrupt_on = BLOCKED_SLEEP_S  # Ctrl+C in the 600 s sleep
        with self.assertRaises(KeyboardInterrupt):
            client.find(INDEX, "a", "plain", query="a")
        until = float(cache.get_meta(BLOCKED_UNTIL_KEY))
        self.assertAlmostEqual(until, w0 + 0.8 + BLOCKED_SLEEP_S)  # 0.8 s: the request itself
        self.assertEqual(cache.get_meta(CONSECUTIVE_403_KEY), "1")

    def test_ctrl_c_then_rerun_does_not_skip_the_wait(self):
        client, tr, cache = self.make(script=[(403, "x")])
        self.ft.interrupt_on = BLOCKED_SLEEP_S
        with self.assertRaises(KeyboardInterrupt):
            client.find(INDEX, "a", "plain", query="a")
        cache.close()
        # Rerun 100 s later: the remaining ~500 s must be waited before the send.
        self.ft.mono_now += 100
        self.ft.wall_now += 100
        self.ft.sleeps.clear()
        client2, tr2, cache2 = self.make(cache=self.cache())
        self.assertEqual(client2.consecutive_403, 1)  # the count survived too
        self.assertIsNotNone(client2.find(INDEX, "a", "plain", query="a"))
        self.assertAlmostEqual(sum(self.ft.sleeps), BLOCKED_SLEEP_S - 100)
        self.assertEqual(len(tr2.bodies), 1)
        self.assertGreaterEqual(tr2.starts[0] - tr.starts[0], BLOCKED_SLEEP_S)

    def test_success_clears_both_keys(self):
        client, _, cache = self.make(script=[(403, "x"), (200, FIND_OK)])
        client.find(INDEX, "a", "plain", query="a")
        self.assertIsNone(cache.get_meta(BLOCKED_UNTIL_KEY))
        self.assertIsNone(cache.get_meta(CONSECUTIVE_403_KEY))

    def test_blocked_persists_count_and_a_restart_cannot_reset_the_streak(self):
        client, tr, cache = self.make(default=(403, "x"))
        with self.assertRaises(Blocked):
            client.find(INDEX, "a", "plain", query="a")
        self.assertEqual(cache.get_meta(CONSECUTIVE_403_KEY), "3")
        self.assertAlmostEqual(
            float(cache.get_meta(BLOCKED_UNTIL_KEY)), self.ft.wall_now + BLOCKED_SLEEP_S)
        self.assertEqual(cache.error_counts(), {"403": 3})
        cache.close()

        # Immediate rerun: waits out the block, sends ONE probe, and that single
        # 403 is the third in a row again, so it raises at once.
        self.ft.sleeps.clear()
        client2, tr2, _ = self.make(default=(403, "x"), cache=self.cache())
        self.assertEqual(client2.consecutive_403, 3)
        with self.assertRaises(Blocked):
            client2.find(INDEX, "a", "plain", query="a")
        self.assertEqual(len(tr2.bodies), 1)
        self.assertAlmostEqual(sum(self.ft.sleeps), BLOCKED_SLEEP_S)

    def test_a_reused_client_waits_after_blocked(self):
        client, tr, cache = self.make(script=[(403, "x")] * 3)
        with self.assertRaises(Blocked):
            client.find(INDEX, "a", "plain", query="a")
        self.ft.sleeps.clear()
        self.assertIsNotNone(client.find(INDEX, "a", "plain", query="a"))  # default 200
        self.assertAlmostEqual(sum(self.ft.sleeps), BLOCKED_SLEEP_S)
        self.assertIsNone(cache.get_meta(BLOCKED_UNTIL_KEY))
        self.assertIsNone(cache.get_meta(CONSECUTIVE_403_KEY))

    def test_only_the_remainder_is_waited(self):
        cache = self.cache()
        cache.set_meta(BLOCKED_UNTIL_KEY, repr(self.ft.wall_now + 42.5))
        client, _, _ = self.make(cache=cache)
        client.find(INDEX, "a", "plain", query="a")
        self.assertAlmostEqual(sum(self.ft.sleeps), 42.5)

    def test_a_future_dated_value_waits_at_most_blocked_sleep_s_in_total(self):
        cache = self.cache()
        cache.set_meta(BLOCKED_UNTIL_KEY, repr(self.ft.wall_now + 10_000_000))
        client, _, _ = self.make(cache=cache)
        client.find(INDEX, "a", "plain", query="a")
        self.assertEqual(sum(self.ft.sleeps), BLOCKED_SLEEP_S)

    def test_corrupt_or_past_values_are_ignored(self):
        for i, bad in enumerate(("nan", "inf", "-inf", "-5", "0", "abc", "", repr(1_000_000.0 - 50))):
            cache = self.cache(path=self.path + str(i))
            cache.set_meta(BLOCKED_UNTIL_KEY, bad)
            cache.set_meta(CONSECUTIVE_403_KEY, "nan" if i % 2 else "-3")
            self.ft.mono_now += 100  # clear of the previous iteration's request gap
            self.ft.wall_now += 100
            client, _, _ = self.make(cache=cache)
            self.assertEqual(client.consecutive_403, 0, bad)
            self.ft.sleeps.clear()
            self.assertIsNotNone(client.find(INDEX, "a", "plain", query="a"), bad)
            self.assertEqual(self.ft.sleeps, [], bad)
            cache.close()

    def test_persisted_count_is_capped_at_the_limit(self):
        cache = self.cache()
        cache.set_meta(CONSECUTIVE_403_KEY, "999")
        client, _, _ = self.make(cache=cache)
        self.assertEqual(client.consecutive_403, ig.MAX_CONSECUTIVE_403)

    def test_budget_is_checked_before_any_block_wait(self):
        cache = self.cache()
        cache.set_meta(BLOCKED_UNTIL_KEY, repr(self.ft.wall_now + 500))
        client, tr, _ = self.make(cache=cache, max_requests=0)
        with self.assertRaises(MaxRequestsReached):
            client.find(INDEX, "a", "plain", query="a")
        self.assertEqual(self.ft.sleeps, [])
        self.assertEqual(tr.bodies, [])
        self.assertIsNone(cache.last_request_wall())
        # Not consumed: a later run with budget still waits.
        client2, _, _ = self.make(cache=cache)
        client2.find(INDEX, "a", "plain", query="a")
        self.assertAlmostEqual(sum(self.ft.sleeps), 500)

    def test_blocked_and_give_up_paths_leave_the_cache_consistent(self):
        client, _, cache = self.make(script=[(403, "x")] * 3)
        with self.assertRaises(Blocked):
            client.find(INDEX, "a", "plain", query="a")
        self.assertIsNone(cache.get_find(INDEX, "a", "plain"))
        self.assertIsNotNone(cache.last_request_wall())
        self.assertEqual(cache.conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        cache.close()

        client2, _, cache2 = self.make(default=(400, "x"), cache=self.cache(path=self.path + "b"))
        with self.assertRaises(TooManyFailures):
            for i in range(MAX_CONSECUTIVE_FAILURES):
                client2.find(INDEX, "q%d" % i, "plain", query="q%d" % i)
        self.assertEqual(cache2.conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        self.assertEqual(cache2.conn.execute("SELECT COUNT(*) FROM finds").fetchone()[0], 0)
        self.assertIsNone(cache2.get_meta(BLOCKED_UNTIL_KEY))  # a 400 is not a block


class TestGlobalLock(Base):
    def test_second_cache_on_another_path_is_refused(self):
        first = self.cache()
        other = os.path.join(self._tmp.name, "elsewhere", "other.sqlite")
        with self.assertRaises(LockHeld) as ctx:
            Cache(other)
        msg = str(ctx.exception)
        self.assertIn(GLOBAL_LOCK_NAME, msg)
        self.assertIn("pid %d" % os.getpid(), msg)  # names the holder
        # The refused attempt created no cache and did not disturb the holder.
        self.assertFalse(os.path.exists(other))
        first.set_meta("still", "works")
        self.assertEqual(first.get_meta("still"), "works")

    def test_released_on_close_so_another_path_then_works(self):
        first = self.cache()
        first.close()
        second = self.cache(path=os.path.join(self._tmp.name, "other.sqlite"))
        self.assertIsNotNone(second.conn)

    def test_a_failed_cache_lock_releases_the_user_lock(self):
        first = self.cache()
        other_dir = os.path.join(self._tmp.name, "otherlocks")
        taken = []
        real_take = ig._take_lock

        def recording_take(path, what):
            f = real_take(path, what)
            taken.append(f)  # keeps the handle alive, so only an explicit close counts
            return f

        ig._take_lock = recording_take
        try:
            with self.assertRaises(LockHeld) as ctx:
                Cache(self.path, global_lock_dir=other_dir)  # user lock free, cache lock held
        finally:
            ig._take_lock = real_take
        self.assertIn(".sqlite.lock", str(ctx.exception))
        self.assertEqual(len(taken), 1)  # the user lock, taken before the cache lock failed
        self.assertTrue(taken[0].closed)
        third = Cache(os.path.join(self._tmp.name, "third.sqlite"), global_lock_dir=other_dir)
        self.addCleanup(third.close)  # would have raised above if the user lock had leaked

    def test_lock_dir_keyword_overrides_the_environment(self):
        self.cache()
        other_dir = os.path.join(self._tmp.name, "otherlocks")
        second = Cache(os.path.join(self._tmp.name, "b.sqlite"), global_lock_dir=other_dir)
        self.addCleanup(second.close)
        self.assertTrue(os.path.exists(os.path.join(other_dir, GLOBAL_LOCK_NAME)))
        self.assertTrue(os.path.exists(os.path.join(self.lock_dir, GLOBAL_LOCK_NAME)))

    def test_default_dir_is_the_temp_dir_and_env_overrides_it(self):
        self.assertEqual(str(ig._lock_dir()), self.lock_dir)
        del os.environ[LOCK_DIR_ENV]
        self.assertEqual(ig._lock_dir(), ig.Path(tempfile.gettempdir()))
        self.assertEqual(str(ig._lock_dir("x")), "x")

    def test_lock_false_takes_neither_lock_nor_the_shared_file(self):
        writer = self.cache()
        writer.set_last_request_wall(123.0)
        reader = self.cache(lock=False, path=os.path.join(self._tmp.name, "r.sqlite"))
        self.assertIsNone(reader.global_lock_path)
        self.assertIsNone(reader.last_request_wall())  # the shared value is not read
        reader.set_last_request_wall(5.0)  # and not written
        self.assertEqual(writer.last_request_wall(), 123.0)

    def test_last_request_time_in_the_shared_file_is_honoured_on_another_path(self):
        first = self.cache()
        client, tr, _ = self.make(cache=first)
        client.find(INDEX, "a", "plain", query="a")
        sent_wall = first.last_request_wall()
        with open(os.path.join(self.lock_dir, GLOBAL_LAST_NAME), encoding="utf-8") as f:
            self.assertEqual(float(f.read()), sent_wall)
        first.close()

        # A fresh cache on a DIFFERENT path has no run_meta value of its own.
        second = self.cache(path=os.path.join(self._tmp.name, "second.sqlite"))
        self.assertIsNone(second.get_meta(ig.LAST_REQUEST_KEY))
        self.assertEqual(second.last_request_wall(), sent_wall)
        self.ft.wall_now = sent_wall + 0.2
        self.ft.sleeps.clear()
        client2, _, _ = self.make(cache=second)
        client2.find(INDEX, "b", "plain", query="b")
        self.assertAlmostEqual(sum(self.ft.sleeps), MIN_GAP_S - 0.2)

    def test_last_request_wall_is_the_max_of_both(self):
        c = self.cache()
        c.set_last_request_wall(500.0)
        c.set_meta(ig.LAST_REQUEST_KEY, repr(900.0))  # own value newer
        self.assertEqual(c.last_request_wall(), 900.0)
        c.set_meta(ig.LAST_REQUEST_KEY, repr(100.0))  # shared value newer
        self.assertEqual(c.last_request_wall(), 500.0)
        c.set_last_request_wall(700.0)  # writes both
        self.assertEqual(c.get_meta(ig.LAST_REQUEST_KEY), repr(700.0))
        self.assertEqual(c.last_request_wall(), 700.0)

    def test_unusable_shared_file_is_ignored(self):
        c = self.cache()
        shared = os.path.join(self.lock_dir, GLOBAL_LAST_NAME)
        c.set_meta(ig.LAST_REQUEST_KEY, repr(10.0))
        for junk in ("nan", "inf", "-inf", "garbage", ""):
            with open(shared, "w", encoding="utf-8") as f:
                f.write(junk)
            self.assertEqual(c.last_request_wall(), 10.0, junk)
        os.remove(shared)
        self.assertEqual(c.last_request_wall(), 10.0)

    def test_shared_write_is_atomic_and_leaves_no_temp_file(self):
        c = self.cache()
        for w in (1.5, 2.5, 3.5):
            c.set_last_request_wall(w)
        self.assertEqual(
            sorted(os.listdir(self.lock_dir)), sorted([GLOBAL_LOCK_NAME, GLOBAL_LAST_NAME]))

    def test_shared_write_failure_does_not_abort_the_run(self):
        c = self.cache()
        c.global_last_path = os.path.join(self.lock_dir, "no", "such", "dir", "x.last")
        c.set_last_request_wall(42.0)  # must not raise
        self.assertEqual(c.get_meta(ig.LAST_REQUEST_KEY), repr(42.0))


class TestThrottleOrdering(Base):
    def test_wait_does_not_stamp_and_mark_started_does(self):
        th = Throttle(MIN_GAP_S, self.ft.mono, self.ft.sleep, wall=self.ft.wall)
        th.wait()
        th.wait()  # no request started in between: nothing to wait for
        self.assertEqual(self.ft.sleeps, [])
        th.mark_started()
        th.wait()
        self.assertAlmostEqual(sum(self.ft.sleeps), MIN_GAP_S)

    def test_mark_started_rechecks_the_gap(self):
        th = Throttle(MIN_GAP_S, self.ft.mono, self.ft.sleep, wall=self.ft.wall)
        th.mark_started()
        th.mark_started()  # called without wait(): still never closer than the gap
        self.assertAlmostEqual(sum(self.ft.sleeps), MIN_GAP_S)

    def test_slow_commit_cannot_shrink_the_gap_between_real_sends(self):
        client, tr, cache = self.make()
        real = cache.set_last_request_wall
        calls = []

        def slow_commit(wall):
            calls.append(wall)
            real(wall)
            if len(calls) == 1:  # only the first commit stalls (an fsync)
                self.ft.mono_now += 0.3
                self.ft.wall_now += 0.3

        cache.set_last_request_wall = slow_commit
        for i in range(4):
            client.find(INDEX, "q%d" % i, "plain", query="q%d" % i)
        self.assertEqual(len(tr.starts), 4)
        self.assert_spaced(tr.starts)  # the sends themselves, not the stamps

    def test_wall_time_is_persisted_before_the_start_is_stamped(self):
        order = []
        client, tr, cache = self.make()
        real_set, real_mark = cache.set_last_request_wall, client.throttle.mark_started
        cache.set_last_request_wall = lambda w: (order.append("persist"), real_set(w))[1]
        client.throttle.mark_started = lambda: (order.append("stamp"), real_mark())[1]
        client.find(INDEX, "a", "plain", query="a")
        self.assertEqual(order, ["persist", "stamp"])


class TestFirstWaitAfterRestart(Base):
    def first_wait(self, last_start_wall):
        th = Throttle(MIN_GAP_S, self.ft.mono, self.ft.sleep,
                      last_start_wall=last_start_wall, wall=self.ft.wall)
        th.wait()
        return sum(self.ft.sleeps)

    def test_total_wait_is_the_remainder_of_the_gap(self):
        self.assertAlmostEqual(self.first_wait(self.ft.wall_now - 0.25), MIN_GAP_S - 0.25)

    def test_no_wait_when_the_gap_has_passed(self):
        self.assertEqual(self.first_wait(self.ft.wall_now - 60), 0)

    def test_future_dated_timestamp_waits_one_gap_in_total(self):
        self.assertAlmostEqual(self.first_wait(self.ft.wall_now + 3600), MIN_GAP_S)

    def test_backwards_clock_waits_one_gap_in_total(self):
        # The persisted stamp is an hour ahead of the (jumped-back) wall clock.
        self.assertAlmostEqual(self.first_wait(self.ft.wall_now + 3599.9), MIN_GAP_S)

    def test_wall_clock_that_never_advances_still_terminates(self):
        ft = self.ft

        def sleep_mono_only(s):
            ft.sleeps.append(s)
            ft.mono_now += s
            if len(ft.sleeps) > 10_000:
                raise AssertionError("runaway wait loop")

        th = Throttle(MIN_GAP_S, ft.mono, sleep_mono_only,
                      last_start_wall=ft.wall_now + 500, wall=ft.wall)
        th.wait()
        self.assertAlmostEqual(sum(ft.sleeps), MIN_GAP_S)

    def test_short_sleeps_are_repeated_until_the_gap_has_elapsed(self):
        ft = self.ft
        t0 = ft.mono_now

        def half_sleep(s):  # returns early: only 60% of the request (whole below 10 ms)
            ft.sleep(s if s < 0.01 else s * 0.6)

        th = Throttle(MIN_GAP_S, ft.mono, half_sleep, last_start_wall=ft.wall_now, wall=ft.wall)
        th.wait()
        self.assertGreaterEqual(ft.mono_now - t0, MIN_GAP_S - EPS)

    def test_nan_inf_none_and_garbage_are_ignored(self):
        for bad in (float("nan"), float("inf"), float("-inf"), None, "abc"):
            self.ft.sleeps.clear()
            self.assertEqual(self.first_wait(bad), 0, bad)

    def test_client_restart_with_a_future_shared_stamp_waits_one_gap(self):
        cache = self.cache()
        cache.set_last_request_wall(self.ft.wall_now + 10_000)
        client, _, _ = self.make(cache=cache)
        client.find(INDEX, "a", "plain", query="a")
        self.assertAlmostEqual(sum(self.ft.sleeps), MIN_GAP_S)


class FakeResponse:
    def __init__(self, status=200, body=b"{}", read_error=None):
        self.status = status
        self._body = body
        self._read_error = read_error

    def read(self):
        if self._read_error is not None:
            raise self._read_error
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    """Stands in for urllib's opener: open(req, timeout=...) -> response or raises."""

    def __init__(self, *items):
        self.items = list(items)
        self.requests = []

    def open(self, req, timeout=None):
        self.requests.append((req, timeout))
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class BrokenBody:
    """An error-response body whose read() dies part-way."""

    def __init__(self, error):
        self.error = error

    def read(self, *a):
        raise self.error

    def close(self):
        pass


def http_error(code, body=None):
    fp = body if body is not None else io.BytesIO(b'{"message": "x"}')
    return urllib.error.HTTPError("https://example.invalid/", code, "msg", {}, fp)


class TestHttpTransport(unittest.TestCase):
    def test_success_is_parsed_json_and_the_request_is_a_json_post(self):
        op = FakeOpener(FakeResponse(200, b'{"cnt": 3, "s": "\xc3\xa9"}'))
        tr = HttpTransport("https://example.invalid/", timeout=7, opener=op)
        self.assertEqual(tr.post({"a": "é"}), (200, {"cnt": 3, "s": "é"}))
        req, timeout = op.requests[0]
        self.assertEqual(timeout, 7)
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(json.loads(req.data.decode("utf-8")), {"a": "é"})
        self.assertEqual(req.get_header("Content-type"), "application/json")

    def test_non_json_body_comes_back_as_text(self):
        tr = HttpTransport(opener=FakeOpener(FakeResponse(200, b"<html>hi</html>")))
        self.assertEqual(tr.post({}), (200, "<html>hi</html>"))

    def test_http_errors_are_returned_as_statuses(self):
        for code in (400, 403, 429, 500, 302):
            tr = HttpTransport(opener=FakeOpener(http_error(code)))
            self.assertEqual(tr.post({}), (code, {"message": "x"}))

    def test_default_opener_does_not_follow_redirects(self):
        tr = HttpTransport()
        kinds = [type(h) for h in tr._opener.handlers]
        self.assertIn(ig._NoRedirect, kinds)
        self.assertNotIn(urllib.request.HTTPRedirectHandler, kinds)
        self.assertIsNone(
            ig._NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://elsewhere/"))

    def test_a_3xx_surfaces_as_a_status_and_the_target_is_never_requested(self):
        seen = []

        class FakeHttps(urllib.request.BaseHandler):
            handler_order = 100  # before the real HTTPSHandler (500): no socket, ever

            def https_open(self, req):
                seen.append((req.full_url, req.get_method()))
                resp = urllib.response.addinfourl(
                    io.BytesIO(b"moved"), {"Location": "https://elsewhere.invalid/x"},
                    req.full_url, 302)
                resp.msg = "Found"
                return resp

        opener = urllib.request.build_opener(ig._NoRedirect, FakeHttps)
        tr = HttpTransport("https://example.invalid/", opener=opener)
        self.assertEqual(tr.post({"q": 1}), (302, "moved"))
        self.assertEqual(seen, [("https://example.invalid/", "POST")])  # exactly one request

    def test_incomplete_read_in_an_error_body_becomes_oserror(self):
        err = http_error(503, BrokenBody(http.client.IncompleteRead(b"par", 10)))
        tr = HttpTransport(opener=FakeOpener(err))
        with self.assertRaises(OSError) as ctx:
            tr.post({})
        self.assertIn("503", str(ctx.exception))

    def test_incomplete_read_in_a_success_body_becomes_oserror(self):
        resp = FakeResponse(200, read_error=http.client.IncompleteRead(b"par", 10))
        with self.assertRaises(OSError):
            HttpTransport(opener=FakeOpener(resp)).post({})

    def test_other_http_client_errors_become_oserror(self):
        for exc in (http.client.BadStatusLine("x"), http.client.ResponseNotReady()):
            with self.assertRaises(OSError):
                HttpTransport(opener=FakeOpener(exc)).post({})

    def test_connection_errors_stay_oserror(self):
        with self.assertRaises(OSError):
            HttpTransport(opener=FakeOpener(urllib.error.URLError("no route"))).post({})
        with self.assertRaises(OSError):
            HttpTransport(opener=FakeOpener(TimeoutError("slow"))).post({})


class TestHttpTransportThroughClient(Base):
    def test_a_broken_error_body_is_retried_not_fatal(self):
        broken = http_error(503, BrokenBody(http.client.IncompleteRead(b"", 5)))
        ok = FakeResponse(200, json.dumps(FIND_OK).encode("utf-8"))
        client, _, cache = self.make()
        client.transport = HttpTransport(opener=FakeOpener(broken, ok))
        r = client.find(INDEX, "a", "plain", query="a")
        self.assertEqual(r["cnt"], 159)
        self.assertEqual(self.ft.sleeps, [2])  # one backoff, then the retry
        self.assertEqual(client.requests_sent, 2)

    def test_a_redirect_is_one_counted_failed_request(self):
        client, _, cache = self.make()
        client.transport = HttpTransport(opener=FakeOpener(http_error(301)))
        self.assertIsNone(client.find(INDEX, "a", "plain", query="a"))
        self.assertEqual(client.requests_sent, 1)  # no hidden second request
        self.assertEqual(cache.error_counts(), {"301": 1})


class TestLatencyUnits(Base):
    def test_window_latency_is_stored_in_ms_unchanged(self):
        client, _, cache = self.make(default=(200, dict(DOC_OK, latency=3.0)))
        r1 = client.get_doc_by_rank(INDEX, "n", "plain", 0, 0, query="n")
        self.assertEqual(r1["api_latency_ms"], 3.0)
        row = cache.conn.execute("SELECT api_latency_ms FROM windows").fetchone()
        self.assertEqual(row["api_latency_ms"], 3.0)
        r2 = client.get_doc_by_rank(INDEX, "n", "plain", 0, 0, query="n")
        self.assertTrue(r2["from_cache"])
        self.assertEqual(r2["api_latency_ms"], 3.0)

    def test_find_latency_is_stored_in_ms_unchanged(self):
        client, _, cache = self.make(default=(200, FIND_OK))
        client.find(INDEX, "n", "plain", query="n")
        row = cache.conn.execute("SELECT api_latency_ms FROM finds").fetchone()
        self.assertEqual(row["api_latency_ms"], 12.5)

    def test_missing_latency_is_null_not_zero(self):
        doc = {k: v for k, v in DOC_OK.items() if k != "latency"}
        client, _, cache = self.make(default=(200, doc))
        r = client.get_doc_by_rank(INDEX, "n", "plain", 0, 0, query="n")
        self.assertIsNone(r["api_latency_ms"])


if __name__ == "__main__":
    unittest.main()
