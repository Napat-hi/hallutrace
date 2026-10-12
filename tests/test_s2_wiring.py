"""Tests for wiring hallutrace/cache.py into HALoGEN's Semantic Scholar lookups
(verifier_runner._build_s2, cache.ResilientCache).

No key, no network, no HALoGEN import. Run from the repo root:

    python -m tests.test_s2_wiring
"""

import os
import shutil
import sqlite3
import sys
import tempfile

import pandas as pd

from hallutrace import cache
from hallutrace import verifier_runner as vr
from hallutrace.semantic_scholar import NOT_FOUND

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


os.environ.setdefault("S2_API_KEY", "test-not-a-real-key")
os.environ.setdefault("TOGETHER_API_KEY", "test-not-a-real-key")


class Reply:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = {}

    def json(self):
        return self._payload


class QueueTransport:
    """One reply per title, consumed once; an unlisted title raises
    AssertionError, so an unexpected network call fails the test."""

    def __init__(self, replies_by_title):
        self.replies = dict(replies_by_title)
        self.calls = []

    def __call__(self, url, headers=None, params=None):
        title = params["query"]
        self.calls.append(title)
        if title not in self.replies:
            raise AssertionError("unexpected network call for %r" % title)
        return self.replies.pop(title)


class _FakeReferencesEvaluator:
    """Stand-in for HALoGEN's ReferencesEvaluator, without together. Its own
    _query_semantic_scholar raises, so a call _build_s2 fails to intercept
    fails the test."""

    def __init__(self, s2_api_key, together_api_key, model):
        self.s2_api_key = s2_api_key
        self.together_api_key = together_api_key
        self.model = model

    def _query_semantic_scholar(self, title):
        raise AssertionError(
            "the released _query_semantic_scholar ran; _build_s2's override did not "
            "intercept the call"
        )


tmp = tempfile.mkdtemp(prefix="hallutrace_s2wiring_test_")
try:
    print("1. _build_s2 subclasses instead of patching an instance")
    # always under tmp; unset, _build_s2 would use the real db/api_cache.sqlite
    os.environ["HALLUTRACE_CACHE_PATH"] = os.path.join(tmp, "section1.sqlite")
    ev1 = vr._build_s2(_FakeReferencesEvaluator, "some-model", transport=QueueTransport({}))
    check(isinstance(ev1, _FakeReferencesEvaluator),
          "the wired evaluator is still an instance of the released class")
    check(
        type(ev1)._query_semantic_scholar is not _FakeReferencesEvaluator._query_semantic_scholar,
        "but _query_semantic_scholar is overridden, not the released method",
    )
    check(hasattr(ev1, "_hallutrace_s2_cache"),
          "the cache is reachable from the evaluator, for _run_once to close")
    ev1._hallutrace_s2_cache.close()

    print()
    print("2. the seam: _query_semantic_scholar routes through the real client")
    cache_path = os.path.join(tmp, "seam.sqlite")
    os.environ["HALLUTRACE_CACHE_PATH"] = cache_path
    transport = QueueTransport({
        "Attention Is All You Need": Reply(200, {"data": [{"title": "Attention Is All You Need"}]}),
        "A paper that does not exist": Reply(200, {"data": []}),
    })
    ev = vr._build_s2(_FakeReferencesEvaluator, "some-model", transport=transport)
    eq(ev._query_semantic_scholar("Attention Is All You Need"), "Attention Is All You Need",
       "a found title resolves through SemanticScholarClient")
    eq(ev._query_semantic_scholar("A paper that does not exist"), NOT_FOUND,
       "an empty result is NOT_FOUND, the real client's tri-state verdict")
    eq(len(transport.calls), 2, "exactly the two expected network calls were made")

    print(
        "   the cache actually wired in is the real, already-tested class "
        "(UNVERIFIABLE-never-cached and NOT_FOUND-is-cached are cache.py's own "
        "invariants, covered in tests/test_cache.py; not re-derived here)"
    )
    check(isinstance(ev._hallutrace_s2_cache, cache.ResilientCache),
          "the client's cache is a ResilientCache")
    check(isinstance(ev._hallutrace_s2_cache._inner, cache.SemanticScholarCache),
          "wrapping the real SemanticScholarCache, not a stand-in")
    ev._hallutrace_s2_cache.close()
    del os.environ["HALLUTRACE_CACHE_PATH"]

    print()
    print("3. persistence: a second, independent evaluator makes zero calls for a cached title")
    cache_path2 = os.path.join(tmp, "seam2.sqlite")
    os.environ["HALLUTRACE_CACHE_PATH"] = cache_path2
    ev_a = vr._build_s2(
        _FakeReferencesEvaluator, "m",
        transport=QueueTransport({"Attention Is All You Need":
                                 Reply(200, {"data": [{"title": "Attention Is All You Need"}]})}),
    )
    ev_a._query_semantic_scholar("Attention Is All You Need")
    ev_a._hallutrace_s2_cache.close()

    empty_transport = QueueTransport({})  # any call raises AssertionError
    ev_b = vr._build_s2(_FakeReferencesEvaluator, "m", transport=empty_transport)
    eq(ev_b._query_semantic_scholar("Attention Is All You Need"), "Attention Is All You Need",
       "a second, independent evaluator resolves the cached title")
    eq(len(empty_transport.calls), 0, "and makes zero network calls to do it")
    ev_b._hallutrace_s2_cache.close()
    del os.environ["HALLUTRACE_CACHE_PATH"]

    print()
    print("4. the chdir trap: run_verifier, a RELATIVE cache path, and the temp dir _run_once deletes")
    fakerepo = os.path.join(tmp, "fakerepo")
    fakesrc = os.path.join(fakerepo, "src")
    os.makedirs(fakesrc)
    with open(os.path.join(fakesrc, "scientific_attribution.py"), "w", encoding="utf-8") as fh:
        fh.write(
            "class ReferencesEvaluator:\n"
            "    def __init__(self, s2_api_key, together_api_key, model):\n"
            "        pass\n"
            "    def _query_semantic_scholar(self, title):\n"
            "        raise AssertionError('released _query_semantic_scholar ran')\n"
            "    def evaluate_references(self, csv_file, output_directory):\n"
            "        import os, pandas as pd\n"
            "        df = pd.read_csv(csv_file)\n"
            "        df['atomic_units'] = df['response'].apply(lambda r: [r])\n"
            "        df['s2_titles'] = df['atomic_units'].apply(\n"
            "            lambda ts: [self._query_semantic_scholar(t) for t in ts])\n"
            "        out = os.path.join(output_directory, os.path.basename(csv_file))\n"
            "        df.to_csv(out, index=False)\n"
            "        return out\n"
        )

    workdir = os.path.join(tmp, "workdir")
    os.makedirs(workdir)
    outer_cwd = os.getcwd()
    os.chdir(workdir)
    os.environ["HALLUTRACE_CACHE_PATH"] = "relative_cache.sqlite"  # the trap: no leading path
    real_build_s2 = vr._build_s2
    trap_transport = QueueTransport({
        "Tell me about attention": Reply(200, {"data": [{"title": "Attention Is All You Need"}]}),
    })
    vr._build_s2 = lambda klass, model: real_build_s2(klass, model, transport=trap_transport)
    try:
        sci_csv = os.path.join(workdir, "sci.csv")
        pd.DataFrame({"response": ["Tell me about attention"]}).to_csv(sci_csv, index=False)
        dest = os.path.join(workdir, "sci_out.csv")
        vr.run_verifier("scientific_attribution", sci_csv, dest, repo_root=fakerepo)
    finally:
        vr._build_s2 = real_build_s2
        del os.environ["HALLUTRACE_CACHE_PATH"]
        os.chdir(outer_cwd)

    expected_abs_path = os.path.join(workdir, "relative_cache.sqlite")
    check(
        os.path.isfile(expected_abs_path),
        "the cache file exists at the path resolved from run_verifier's caller's cwd, "
        "not lost inside the temp dir _run_once deleted",
    )
    conn = sqlite3.connect(expected_abs_path)
    try:
        rows = conn.execute("SELECT normalised_title, verdict FROM s2_cache").fetchall()
    finally:
        conn.close()
    eq(len(rows), 1, "one row, for the one title looked up during the run")
    check(rows and rows[0][1] == "Attention Is All You Need",
          "and it holds the real resolved verdict")
    eq(len(trap_transport.calls), 1, "exactly one network call was made for it")

finally:
    shutil.rmtree(tmp, ignore_errors=True)
    os.environ.pop("HALLUTRACE_CACHE_PATH", None)


# ---------------------------------------------------------------------------
# ResilientCache: degrade and count, reconnect once, never raise.
#
# A fake inner cache that fails on command, with connect_cache and
# init_cache_db replaced by no-ops, so only the control flow is tested.
# ---------------------------------------------------------------------------

print()
print("5. ResilientCache: degrade and count, reconnect once, never raise")


class FlakyInner:
    """A get/put target that raises sqlite3.OperationalError on command."""

    def __init__(self, data=None):
        self.data = dict(data or {})
        self.fail_next = 0  # number of upcoming calls that raise, then behave

    def get(self, title):
        if self.fail_next > 0:
            self.fail_next -= 1
            raise sqlite3.OperationalError("simulated: database is locked")
        return self.data.get(title)

    def put(self, title, verdict):
        if self.fail_next > 0:
            self.fail_next -= 1
            raise sqlite3.OperationalError("simulated: database is locked")
        self.data[title] = verdict


class Capture:
    """Stands in for sys.stderr; records every write()."""

    def __init__(self):
        self.lines = []

    def write(self, s):
        self.lines.append(s)

    def flush(self):
        pass


real_connect_cache, real_init_cache_db = cache.connect_cache, cache.init_cache_db
cache.connect_cache = lambda path: object()  # ResilientCache never touches this directly
cache.init_cache_db = lambda conn: None
try:
    print("   a. a clean cache: no failures, nothing counted")
    inner = FlakyInner()
    rc = cache.ResilientCache("irrelevant-path", lambda conn: inner)
    eq(rc.get("x"), None, "a miss on an empty cache returns None")
    rc.put("a title", "A Title")
    eq(rc.get("a title"), "A Title", "a clean put/get round trips")
    eq(rc.failures, 0, "and nothing was counted")
    eq(rc.disabled, False, "nor disabled")

    print("   b. a failure a reconnect FIXES: stays live, exactly one failure counted, logged once")
    captured = Capture()
    real_stderr = sys.stderr
    sys.stderr = captured
    try:
        recovered = FlakyInner({"b title": "recovered value"})
        inner.fail_next = 1
        rc._inner_factory = lambda conn: recovered  # the reconnect swaps to this
        got = rc.get("b title")
    finally:
        sys.stderr = real_stderr
    eq(got, "recovered value", "the retried get() ran against the NEW inner cache")
    eq(rc.failures, 1, "one failure counted for the one exception raised")
    eq(rc.disabled, False, "a fixed failure does not disable the cache")
    check(
        any("failure #1" in ln and "OperationalError" in ln for ln in captured.lines),
        "the first failure was logged immediately, with the exception text",
    )

    print("   c. a second, later, independent failure gets its own fresh reconnect attempt")
    captured2 = Capture()
    sys.stderr = captured2
    try:
        recovered.fail_next = 1
        recovered_again = FlakyInner({"c title": "second recovery"})
        rc._inner_factory = lambda conn: recovered_again
        got2 = rc.get("c title")
    finally:
        sys.stderr = real_stderr
    eq(got2, "second recovery", "the second incident also recovered, via its own reconnect")
    eq(rc.failures, 2, "its failure adds to the running total (now 2)")
    eq(rc.disabled, False, "still not disabled: the second reconnect also succeeded")
    eq(captured2.lines, [], "the second failure is NOT re-logged: only the very first one, ever, is")

    print("   d. reconnect itself fails (a fresh connection cannot be opened): disables, never raises")
    inner2 = FlakyInner()
    inner2.fail_next = 1

    def boom_factory(conn):
        raise RuntimeError("simulated: cannot rebuild the inner cache")

    rc_d = cache.ResilientCache("irrelevant-path", lambda conn: inner2)
    rc_d._inner_factory = boom_factory
    result = rc_d.get("anything")
    eq(result, None, "get() degrades to a miss, does not raise, when reconnect fails")
    eq(rc_d.disabled, True, "and the cache is disabled")
    eq(rc_d.failures, 1, "the triggering failure was counted")

    print("      once disabled, every further call is a silent, instant no-op")
    rc_d.put("x", "y")  # must not raise, must not attempt anything
    eq(rc_d.get("x"), None, "get() keeps returning None: a permanent, safe miss")
    eq(rc_d.failures, 1, "no further failures are counted: nothing is attempted once disabled")

    print("   e. reconnect succeeds but the RETRY also fails: disables too, counts both failures")
    always_fails = FlakyInner()
    always_fails.fail_next = 99  # every call fails, including the retry
    rc_e = cache.ResilientCache("irrelevant-path", lambda conn: always_fails)
    rc_e.get("t")
    eq(rc_e.disabled, True, "disabled: the retry after a successful reconnect still failed")
    eq(rc_e.failures, 2, "both the original failure and the failed retry are counted")

    print("   f. a non-sqlite exception degrades too, not only sqlite3.Error")

    class AlwaysRaisesValueError:
        def get(self, *a):
            raise ValueError("not a sqlite error at all")

        def put(self, *a):
            raise ValueError("not a sqlite error at all")

    rc_f = cache.ResilientCache("irrelevant-path", lambda conn: AlwaysRaisesValueError())
    eq(rc_f.get("x"), None, "a ValueError from get() still degrades, does not raise")
    rc_f.put("x", "y")
    check(True, "and put() does not raise either")

    print("   g. the reconnect budget: after MAX_RECONNECTS successes, no more reconnects, "
          "but op() keeps being tried")
    eq(cache.ResilientCache.MAX_RECONNECTS, 5, "the documented budget is 5 successful reconnects")
    shared = FlakyInner()
    rc_g = cache.ResilientCache("irrelevant-path", lambda conn: shared)
    for i in range(cache.ResilientCache.MAX_RECONNECTS):
        shared.fail_next = 1  # this call's first attempt fails, forcing a reconnect
        rc_g.get("call %d" % i)  # ... which then succeeds on the very next attempt
    eq(rc_g.reconnects, cache.ResilientCache.MAX_RECONNECTS,
       "five independent failures each earned their own successful reconnect")
    eq(rc_g.failures, cache.ResilientCache.MAX_RECONNECTS, "one failure counted per incident so far")
    eq(rc_g.disabled, False, "every one of those reconnects, and its retry, succeeded")
    check(rc_g.reconnect_budget_exhausted,
          "the budget is exhausted the instant the 5th reconnect succeeds, not only after a 6th attempt")

    shared.fail_next = 1
    result_6th = rc_g.get("call 5")
    eq(result_6th, None, "the 6th failure degrades to a miss")
    eq(rc_g.reconnects, cache.ResilientCache.MAX_RECONNECTS,
       "no 6th reconnect was attempted: the budget was already spent")
    eq(rc_g.disabled, False,
       "NOT fully disabled: budget-exhausted is a milder, different state (see the class docstring)")
    check(rc_g.reconnect_budget_exhausted, "the budget-exhausted flag now reflects this")

    print("      after the budget: op() is still tried on every future call, and can still succeed")
    shared.data["recovered without reconnecting"] = "still works"
    # fail_next was left at 0 by the 6th call's single (non-retried) attempt;
    # the connection from the 5th, successful reconnect is untouched.
    got = rc_g.get("recovered without reconnecting")
    eq(got, "still works",
       "a later call succeeds directly, with no reconnect, proving op() is still being attempted")
    eq(rc_g.reconnects, cache.ResilientCache.MAX_RECONNECTS, "and no new reconnect was spent getting there")

    print("   h. logging happens at failures 1, 10, 100 and 1000, and at no other count")
    shared2 = FlakyInner()
    rc_h = cache.ResilientCache("irrelevant-path", lambda conn: shared2)
    captured_h = Capture()
    real_stderr = sys.stderr
    sys.stderr = captured_h
    try:
        for _ in range(cache.ResilientCache.MAX_RECONNECTS):
            shared2.fail_next = 1
            rc_h.get("x")
        while rc_h.failures < 1000:
            shared2.fail_next = 1
            rc_h.get("x")
    finally:
        sys.stderr = real_stderr
    eq(rc_h.failures, 1000, "failures reached exactly 1000")
    eq(rc_h.reconnects, cache.ResilientCache.MAX_RECONNECTS,
       "not one additional reconnect was spent getting from 5 failures to 1000")
    eq(rc_h.disabled, False, "never disabled: budget-exhausted keeps trying, it does not give up")
    logged_numbers = sorted(
        int(ln.split("failure #", 1)[1].split(" ", 1)[0])
        for ln in captured_h.lines if "failure #" in ln
    )
    eq(logged_numbers, [1, 10, 100, 1000], "logged at exactly the four milestones, nowhere else")
finally:
    cache.connect_cache, cache.init_cache_db = real_connect_cache, real_init_cache_db


print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
