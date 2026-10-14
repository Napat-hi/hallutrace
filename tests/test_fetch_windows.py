"""Tests for scripts/fetch_windows.py, the CLI driver. End to end on a fake transport.

No network, no real sleep, temp caches and small fixture files only. Run from the
repo root:

    python -m unittest tests.test_fetch_windows -v

The fake transport emulates the infini-gram API on a tiny synthetic corpus: a
dict of (index, query string, form) -> mention count. `count` tokenizes one
character per token behind a '▁(' token, `find` returns three uneven shards,
`get_doc_by_rank` refuses a rank outside the shard's segment with a 400 (so a
driver that asks for a rank it was never given fails loudly), and every body is
recorded. A fake clock advances on sleep and per request, so nothing waits.
"""

import contextlib
import importlib.util
import io
import json
import os
import sqlite3
import tempfile
import unittest
import zlib
from pathlib import Path

from hallutrace import infinigram
from hallutrace.infinigram import INDEXES, MIN_GAP_S, Cache
from hallutrace.fetch_sampling import CAP_FULL, SAMPLE_N, SEED, choose_pilot

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("fetch_windows", ROOT / "scripts" / "fetch_windows.py")
fw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fw)

OLMO2, DCLM, DOLMA, RPJ, PILE = INDEXES
LEAD = "▁("


class FakeTime:
    """One timeline for the monotonic and the wall clock; sleep moves it forward."""

    def __init__(self, mono=100.0, wall=1_000_000.0):
        self.mono_now = mono
        self.wall_now = wall
        self.sleeps = []

    def mono(self):
        return self.mono_now

    def wall(self):
        return self.wall_now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.mono_now += seconds
        self.wall_now += seconds

    def advance(self, seconds):
        self.mono_now += seconds
        self.wall_now += seconds


class Corpus:
    """(index, string, form) -> mention count; index None means every index."""

    def __init__(self):
        self.counts = {}

    def set(self, string, plain, glued=0, index=None):
        self.counts[(index, string, "plain")] = plain
        self.counts[(index, string, "glued")] = glued

    def count(self, index, string, form):
        for key in ((index, string, form), (None, string, form)):
            if key in self.counts:
                return self.counts[key]
        return 0


def shards(n):
    """Three uneven shards. Every shard but the first starts at a nonzero rank,
    and shard 2 starts at 14 for both forms, so (s, rank) collides across forms."""
    a, b = n // 2, n // 3
    c = n - a - b
    return [[0, a], [7, 7 + b], [14, 14 + c]]


class FakeApi:
    """Transport for the driver. Records every body and its status."""

    def __init__(self, ft, corpus, duration=0.8):
        self.ft = ft
        self.corpus = corpus
        self.duration = duration
        self.calls = 0
        self.ok = []  # bodies answered with 2xx, in order
        self.fail_after = None  # raise fail_exc on every call after this many
        self.fail_exc = None
        self.always = None  # (status, payload) for every call
        self.reject = set()  # find / get_doc_by_rank of these strings answer HTTP 400
        self.reject_count = set()  # `count` (tokenize) of "(" + these strings answers HTTP 400
        self.count_attempts = []  # every `count` text received, answered or not

    def post(self, body):
        self.calls += 1
        self.ft.advance(self.duration)
        if self.fail_after is not None and self.calls > self.fail_after:
            raise self.fail_exc
        if self.always is not None:
            return self.always
        status, payload = self._answer(body)
        if 200 <= status < 300:
            self.ok.append(body)
        return status, payload

    def _string_and_form(self, body):
        if ("query" in body) == ("query_ids" in body):
            return None
        if "query" in body:
            return body["query"], "plain"
        return "".join(chr(i) for i in body["query_ids"]), "glued"

    def _answer(self, body):
        index, qt = body.get("index"), body.get("query_type")
        if index not in INDEXES:
            return 400, {"message": "unknown index"}
        if qt == "count":
            text = body["query"]
            self.count_attempts.append(text)
            if text.startswith("(") and text[1:] in self.reject_count:
                return 400, {"message": "rejected by the fixture"}
            if text.startswith("(Odd Tokens"):
                # A tokenization the driver cannot glue: no '▁(' prefix token.
                return 200, {"count": 0, "approx": False, "latency": 1.0,
                             "token_ids": [ord(c) for c in text], "tokens": list(text)}
            return 200, {"count": 0, "approx": False, "latency": 1.0,
                         "token_ids": [1] + [ord(c) for c in text[1:]],
                         "tokens": [LEAD] + list(text[1:])}
        sf = self._string_and_form(body)
        if sf is None:
            return 400, {"message": "Exactly one of query and query_ids must be present!"}
        string, form = sf
        if string in self.reject:
            return 400, {"message": "rejected by the fixture"}
        n = self.corpus.count(index, string, form)
        segs = shards(n)
        if qt == "find":
            return 200, {"cnt": n, "segment_by_shard": segs, "tokens": [string], "latency": 5.0}
        if qt == "get_doc_by_rank":
            s, rank = body["s"], body["rank"]
            if not (0 <= s < 3 and segs[s][0] <= rank < segs[s][1]):
                return 400, {"message": "rank out of range"}
            key = "%s|%s|%s|%d|%d" % (index, string, form, s, rank)
            return 200, {
                "doc_ix": zlib.crc32(key.encode("utf-8")), "doc_len": 5000, "disp_len": 300,
                "needle_offset": 100,
                "spans": [["before ", None], [string, "0"], [" after", None]],
                "metadata": json.dumps({"metadata": {"url": "http://x.example/" + key}}),
                "token_ids": [1], "tokens": ["a"], "blocked": False, "latency": 3.0,
            }
        return 400, {"message": "unknown query_type"}

    def bodies_json(self):
        return [json.dumps(b, sort_keys=True) for b in self.ok]


# Names and the strings the fixture alias table + snapshot give them:
#   Alice Low (poet)  name; "Alice Low" stripped; "A. Low" alias (label duplicates the stripped form)
#   Bob High          name; "Robert High" alias
#   Céline Mid        name; "Celine Mid" folded
#   Eve Namesake      two QIDs: main string only
#   Odd Tokens        the fake tokenizer gives no '(' prefix: plain form only
#   Dana Zero, Small 1..6: main string only
NAMES = ["Alice Low (poet)", "Bob High", "Céline Mid", "Dana Zero", "Eve Namesake",
         "Odd Tokens"] + ["Small %d" % i for i in range(1, 7)]
N_STRINGS = 16  # 12 names + Alice Low + A. Low + Robert High + Celine Mid


def make_corpus():
    c = Corpus()
    # Alice: low everywhere except DCLM, where the main name is over the cap.
    c.set("Alice Low (poet)", 4, 2)
    c.set("Alice Low (poet)", 400, 100, index=DCLM)
    c.set("Alice Low", 5, 1)
    c.set("A. Low", 2, 0)
    # Bob: over the cap everywhere except DCLM.
    c.set("Bob High", 250, 10)
    c.set("Bob High", 300, 60, index=OLMO2)
    c.set("Bob High", 15, 5, index=DCLM)
    c.set("Robert High", 3, 0)
    c.set("Céline Mid", 30, 10)
    c.set("Celine Mid", 6, 0)
    c.set("Dana Zero", 0, 0)
    c.set("Eve Namesake", 12, 3)
    c.set("Odd Tokens", 2, 0)
    for i in range(1, 7):
        c.set("Small %d" % i, 3, 1)
    c.set("Ally Low", 1, 0)
    c.set("Small 7", 3, 1)
    return c


def alias_rows():
    return {
        "alice low poet": ["Q1"], "alice low": ["Q1"], "a low": ["Q1"],
        "bob high": ["Q2"], "robert high": ["Q2"],
        "celine mid": ["Q3"],
        "eve namesake": ["Q5", "Q6"],
    }


def snapshot_rows():
    return {"entities": {
        "Q1": {"label": "Alice Low", "aliases": ["A. Low"]},
        "Q2": {"label": "Bob High", "aliases": ["Robert High"]},
    }}


class Env:
    """Fixture files, a temp cache path, a fake clock and a fake API."""

    def __init__(self, names=NAMES, alias=None, snapshot=None, corpus=None):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        d = Path(self.tmp.name)
        self.prompts = d / "prompts.csv"
        self.alias = d / "alias_table.json"
        self.snapshot = d / "snapshot.json"
        self.cache = d / "corpus_cache" / "infinigram.sqlite"
        self.ft = FakeTime()
        self.corpus = corpus if corpus is not None else make_corpus()
        self.api = FakeApi(self.ft, self.corpus)
        self.write_prompts(names)
        self.alias.write_text(json.dumps(alias if alias is not None else alias_rows()), encoding="utf-8")
        self.snapshot.write_text(json.dumps(snapshot if snapshot is not None else snapshot_rows()), encoding="utf-8")

    def cleanup(self):
        self.tmp.cleanup()

    def write_prompts(self, names):
        import csv
        with open(self.prompts, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt_id", "task", "source_file", "source_row", "prompt", "task_meta"])
            pairs = [names[i:i + 2] for i in range(0, len(names), 2)]
            for i, pair in enumerate(pairs):
                meta = {"name1": pair[0], "name2": pair[1] if len(pair) > 1 else ""}
                w.writerow(["p%d" % i, "historical_events", "x.csv", i, "Did they meet?", json.dumps(meta, ensure_ascii=False)])
            # Another task's row must not contribute names.
            w.writerow(["b0", "biographies", "x.csv", 0, "Tell me about X.", json.dumps({"name1": "Not A Subject"})])

    def argv(self, command, *extra):
        return [command, "--prompts", str(self.prompts), "--alias-table", str(self.alias),
                "--snapshot", str(self.snapshot), "--cache", str(self.cache), *extra]

    def run(self, command, *extra, transport="api"):
        """(exit code, stdout, stderr). transport=None runs with none injected."""
        return self.run_argv(self.argv(command, *extra), transport=transport)

    def run_argv(self, argv, transport="api"):
        out, err = io.StringIO(), io.StringIO()
        tr = self.api if transport == "api" else transport
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = fw.main(argv, transport=tr, clock=self.ft.mono, sleep=self.ft.sleep, wall=self.ft.wall)
        return code, out.getvalue(), err.getvalue()

    def run_default_inputs(self, command, *extra, transport="api"):
        """Only --cache is given: the three input files are NOT named on the command line."""
        return self.run_argv([command, "--cache", str(self.cache), *extra], transport=transport)

    def rows(self, sql, *params):
        """Read the cache directly (a plain sqlite3 connection; the fetcher is not running)."""
        conn = sqlite3.connect(str(self.cache))
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def meta(self, key):
        r = self.rows("SELECT value FROM run_meta WHERE key = ?", key)
        return r[0][0] if r else None

    def snapshot_state(self):
        """Everything in the cache that a read-only command must not change."""
        conn = sqlite3.connect(str(self.cache))
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            return {t: conn.execute("SELECT * FROM %s ORDER BY 1, 2" % t).fetchall() for t in tables}
        finally:
            conn.close()


def assert_no_repeated_success(tc, api):
    bodies = api.bodies_json()
    tc.assertEqual(len(bodies), len(set(bodies)), "a request that already succeeded was sent again")


@contextlib.contextmanager
def private_lock_dir():
    """Point the per-user fetcher lock at a temp dir, so the tests neither touch the
    real %TEMP% lock nor fail while a real fetch is running."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        old = os.environ.get(infinigram.LOCK_DIR_ENV)
        os.environ[infinigram.LOCK_DIR_ENV] = d
        try:
            yield d
        finally:
            if old is None:
                os.environ.pop(infinigram.LOCK_DIR_ENV, None)
            else:
                os.environ[infinigram.LOCK_DIR_ENV] = old


class EnvCase(unittest.TestCase):
    def setUp(self):
        cm = private_lock_dir()
        self.lock_dir = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def env(self, **kw):
        e = Env(**kw)
        self.addCleanup(e.cleanup)
        return e


class FullRunFixture(EnvCase):
    """One clean plan + full run shared by the read-only assertions."""

    @classmethod
    def setUpClass(cls):
        cls._lock = private_lock_dir()
        cls._lock.__enter__()
        cls.e = Env()
        cls.plan = cls.e.run("plan")
        cls.calls_after_plan = cls.e.api.calls
        cls.full = cls.e.run("full")
        cls.reference = set(cls.e.api.bodies_json())

    @classmethod
    def tearDownClass(cls):
        cls.e.cleanup()
        cls._lock.__exit__(None, None, None)


class TestCleanRun(FullRunFixture):
    def test_exit_codes_and_no_repeats(self):
        self.assertEqual(self.plan[0], 0, self.plan[2])
        self.assertEqual(self.full[0], 0, self.full[2])
        assert_no_repeated_success(self, self.e.api)

    def test_every_request_respects_the_gap_and_the_api_accepted_all(self):
        # A rank the driver was never given, or a body with neither/both query
        # forms, would have come back as a 400 and landed in errors.
        self.assertEqual(self.e.rows("SELECT COUNT(*) FROM errors")[0][0], 0)
        self.assertEqual(len(self.e.api.ok), self.e.api.calls)

    def test_tokenized_once_per_string_not_per_index(self):
        counts = [b for b in self.e.api.ok if b["query_type"] == "count"]
        self.assertEqual(len(counts), N_STRINGS)
        self.assertEqual({b["index"] for b in counts}, {OLMO2})

    def test_plan_only_touches_olmo2_and_fetches_no_windows(self):
        indexes = {b["index"] for b in self.e.api.ok[: self.calls_after_plan]}
        self.assertEqual(indexes, {OLMO2})
        kinds = {b["query_type"] for b in self.e.api.ok[: self.calls_after_plan]}
        self.assertEqual(kinds, {"count", "find"})

    def find_rows(self, index, string):
        return self.e.rows("SELECT form FROM finds WHERE index_name = ? AND query_string = ? ORDER BY form", index, string)

    def test_extra_forms_only_where_main_count_is_low_per_index(self):
        # Bob: 360 on OLMo 2 (over the cap), 20 on DCLM (under), 260 elsewhere.
        self.assertGreater(300 + 60, CAP_FULL)
        self.assertEqual(self.find_rows(OLMO2, "Robert High"), [])
        self.assertEqual(self.find_rows(DCLM, "Robert High"), [("glued",), ("plain",)])
        for index in (DOLMA, RPJ, PILE):
            self.assertEqual(self.find_rows(index, "Robert High"), [])
        # Alice: 6 nearly everywhere, 500 on DCLM: the opposite pattern.
        for index in (OLMO2, DOLMA, RPJ, PILE):
            self.assertEqual(self.find_rows(index, "Alice Low"), [("glued",), ("plain",)], index)
            self.assertEqual(self.find_rows(index, "A. Low"), [("glued",), ("plain",)], index)
        self.assertEqual(self.find_rows(DCLM, "Alice Low"), [])
        self.assertEqual(self.find_rows(DCLM, "A. Low"), [])
        # The main names are always queried, on every index.
        for index in INDEXES:
            self.assertEqual(self.find_rows(index, "Bob High"), [("glued",), ("plain",)])
            self.assertEqual(self.find_rows(index, "Alice Low (poet)"), [("glued",), ("plain",)])
        # A window of a gated-out string was never planned either.
        self.assertEqual(self.e.rows("SELECT COUNT(*) FROM plan WHERE index_name = ? AND query_string = ?",
                                     OLMO2, "Robert High")[0][0], 0)

    def windows(self, index, string):
        return self.e.rows("SELECT form, s, rank, doc_ix FROM windows WHERE index_name = ? AND query_string = ? "
                           "ORDER BY form, s, rank", index, string)

    def test_over_the_cap_gets_exactly_sample_n_windows_across_both_forms(self):
        w = self.windows(OLMO2, "Bob High")
        self.assertEqual(len(w), SAMPLE_N)
        self.assertEqual(len(set(w)), SAMPLE_N)
        forms = {f for f, _, _, _ in w}
        self.assertEqual(forms, {"plain", "glued"})  # one pool: the sample spans both forms
        for form, s, rank, _ in w:
            lo, hi = shards(300 if form == "plain" else 60)[s]
            self.assertTrue(lo <= rank < hi)
        planned = self.e.rows("SELECT DISTINCT selected FROM plan WHERE index_name = ? AND query_string = ?", OLMO2, "Bob High")
        self.assertEqual(planned, [("sample",)])
        # DCLM: the main name is over the cap there too (500 for Alice).
        self.assertEqual(len(self.windows(DCLM, "Alice Low (poet)")), SAMPLE_N)

    def test_under_the_cap_gets_every_occurrence_of_both_forms(self):
        w = self.windows(OLMO2, "Small 1")
        self.assertEqual(len(w), 3 + 1)
        self.assertEqual(sum(1 for r in w if r[0] == "plain"), 3)
        self.assertEqual(sum(1 for r in w if r[0] == "glued"), 1)
        self.assertEqual(self.e.rows("SELECT DISTINCT selected FROM plan WHERE index_name = ? AND query_string = ?",
                                     OLMO2, "Small 1"), [("all",)])
        # Bob on DCLM is 20 under the cap: all of them, and its alias too.
        self.assertEqual(len(self.windows(DCLM, "Bob High")), 20)
        self.assertEqual(len(self.windows(DCLM, "Robert High")), 3)

    def test_windows_are_keyed_by_form(self):
        w = self.windows(OLMO2, "Small 1")
        plain = {(s, r): d for f, s, r, d in w if f == "plain"}
        glued = {(s, r): d for f, s, r, d in w if f == "glued"}
        common = set(plain) & set(glued)
        self.assertTrue(common, "fixture must make one (shard, rank) appear under both forms")
        for key in common:
            self.assertNotEqual(plain[key], glued[key])  # two rows, two different documents

    def test_zero_count_name_has_no_windows_and_is_not_an_error(self):
        self.assertEqual(self.windows(OLMO2, "Dana Zero"), [])
        self.assertEqual(len(self.find_rows(OLMO2, "Dana Zero")), 2)

    def test_glued_tokenization_failure_is_recorded_and_not_fatal(self):
        self.assertEqual(self.find_rows(OLMO2, "Odd Tokens"), [("plain",)])
        self.assertEqual(len(self.windows(OLMO2, "Odd Tokens")), 2)
        code, out, _ = self.e.run("status", transport=None)
        self.assertEqual(code, 0)
        self.assertIn("glued-tokenization failures: 1", out)
        self.assertIn("Odd Tokens", out)

    def test_run_meta_provenance(self):
        e = self.e
        self.assertEqual(json.loads(e.meta("constants")),
                         dict(CAP_FULL=fw.CAP_FULL, SAMPLE_N=fw.SAMPLE_N, DISP_LEN=fw.DISP_LEN,
                              MIN_GAP_S=fw.MIN_GAP_S, SEED=fw.SEED))
        self.assertEqual(e.meta("script_version"), fw.SCRIPT_VERSION)
        self.assertIsNotNone(e.meta("git_head"))
        self.assertIn(e.meta("git_dirty"), ("true", "false", "unknown"))
        self.assertEqual(e.meta("input_sha256_prompts"), fw.sha256_file(e.prompts))
        self.assertEqual(e.meta("input_sha256_alias_table"), fw.sha256_file(e.alias))
        self.assertEqual(e.meta("input_sha256_snapshot"), fw.sha256_file(e.snapshot))
        self.assertEqual(e.meta("input_path_prompts"), os.path.abspath(str(e.prompts)))
        self.assertEqual(e.meta("input_path_alias_table"), os.path.abspath(str(e.alias)))
        self.assertEqual(e.meta("input_path_snapshot"), os.path.abspath(str(e.snapshot)))
        self.assertIsNotNone(e.meta("plan_time"))
        names = json.loads(e.meta("task_names"))
        self.assertEqual(names, sorted(NAMES))  # the biographies row contributed nothing

    def test_plan_reports_extra_form_requests(self):
        self.assertIn("extra forms", self.plan[1])
        self.assertIn("pilot list chosen", self.plan[1])

    def test_pilot_stored_once_and_stratified(self):
        pilot = json.loads(self.e.meta("pilot_names"))
        self.assertEqual(len(pilot), len(set(pilot)))
        self.assertTrue(set(pilot) <= set(NAMES))
        # Every non-empty bucket is represented: 0 / 1-10 / 11-100 / 101-1000.
        for must in ("Dana Zero", "Bob High", "Céline Mid", "Eve Namesake"):
            self.assertIn(must, pilot)
        self.assertEqual(len(pilot), 8)  # 4 of the 8 names in the 1-10 bucket + the 4 above


_REFERENCE = None


def reference_bodies():
    """The set of successful requests of one clean plan + full run (deterministic,
    so computed once for every test that compares a resumed run against it)."""
    global _REFERENCE
    if _REFERENCE is None:
        e = Env()
        try:
            assert e.run("plan")[0] == 0 and e.run("full")[0] == 0
            _REFERENCE = set(e.api.bodies_json())
        finally:
            e.cleanup()
    return _REFERENCE


class TestResume(EnvCase):
    def reference(self):
        return reference_bodies()

    def killed_then_resumed(self, exc, expect_code):
        ref = self.reference()
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        e.api.fail_after = e.api.calls + 300
        e.api.fail_exc = exc
        code, out, err = e.run("full")
        self.assertEqual(code, expect_code, err)
        killed_at = len(e.api.ok)
        self.assertGreater(killed_at, 0)
        # The lock was released on the way out.
        Cache(str(e.cache)).close()
        # Resume: the same command finishes the job.
        e.api.fail_after = None
        code, out, err = e.run("full")
        self.assertEqual(code, 0, err)
        # Total distinct requests equal a clean run's; nothing cached was re-sent.
        self.assertEqual(set(e.api.bodies_json()), ref)
        assert_no_repeated_success(self, e.api)

    def test_killed_by_ctrl_c_resumes_where_it_stopped(self):
        self.killed_then_resumed(KeyboardInterrupt(), 130)

    def test_killed_by_a_crash_resumes_where_it_stopped(self):
        self.killed_then_resumed(RuntimeError("boom"), 1)

    def test_max_requests_stops_cleanly_and_the_next_run_continues(self):
        ref = self.reference()
        e = self.env()
        code, out, _ = e.run("plan", "--max-requests", "5")
        self.assertEqual(code, 0)
        self.assertEqual(e.api.calls, 5)
        self.assertIn("stopped at --max-requests 5", out)
        self.assertIn("cache resumes", out)
        # Not all main finds are in yet, so no pilot has been chosen.
        self.assertIsNone(e.meta("pilot_names"))
        self.assertEqual(e.run("plan")[0], 0)
        self.assertIsNotNone(e.meta("pilot_names"))
        before = e.api.calls
        code, out, _ = e.run("full", "--max-requests", "40")
        self.assertEqual(code, 0)
        self.assertEqual(e.api.calls - before, 40)
        self.assertEqual(e.run("full")[0], 0)
        self.assertEqual(set(e.api.bodies_json()), ref)
        assert_no_repeated_success(self, e.api)
        # A run with nothing left to do sends nothing.
        calls = e.api.calls
        self.assertEqual(e.run("full")[0], 0)
        self.assertEqual(e.api.calls, calls)


class TestReplan(EnvCase):
    def test_replan_adds_strings_deactivates_removed_and_keeps_the_pilot(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        pilot = e.meta("pilot_names")
        self.assertIsNotNone(pilot)
        self.assertEqual(e.run("plan")[0], 0)  # unchanged inputs: nothing to do
        calls = e.api.calls

        # A. Low disappears from the alias table, Ally Low appears, and one name is new.
        table = alias_rows()
        del table["a low"]
        table["ally low"] = ["Q1"]
        snap = snapshot_rows()
        snap["entities"]["Q1"]["aliases"] = ["Ally Low"]
        e.alias.write_text(json.dumps(table), encoding="utf-8")
        e.snapshot.write_text(json.dumps(snap), encoding="utf-8")
        e.write_prompts(NAMES + ["Small 7"])
        finds_before = e.rows("SELECT COUNT(*) FROM finds WHERE query_string = ?", "A. Low")[0][0]
        self.assertEqual(finds_before, 2)

        code, out, _ = e.run("plan")
        self.assertEqual(code, 0)
        # 1 tokenize + 2 finds for Ally Low, 1 tokenize + 2 finds for Small 7.
        self.assertEqual(e.api.calls - calls, 6)
        active = dict(e.rows("SELECT query_string, active FROM entities WHERE prompt_name = ?", "Alice Low (poet)"))
        self.assertEqual(active, {"Alice Low (poet)": 1, "Alice Low": 1, "A. Low": 0, "Ally Low": 1})
        self.assertEqual(e.rows("SELECT active FROM entities WHERE query_string = ?", "Small 7"), [(1,)])
        # Rows are kept, and so are the removed string's cached finds.
        self.assertEqual(e.rows("SELECT COUNT(*) FROM finds WHERE query_string = ?", "A. Low")[0][0], 2)
        # The pilot list is never reselected, even though the name set changed.
        self.assertEqual(e.meta("pilot_names"), pilot)
        self.assertNotIn("Small 7", json.loads(pilot))
        self.assertIn("kept unchanged", out)
        self.assertIn(json.dumps(sorted(NAMES + ["Small 7"]), ensure_ascii=False), e.meta("task_names"))

        # The inactive string is not fetched by later phases...
        self.assertEqual(e.run("full")[0], 0)
        self.assertEqual(e.rows("SELECT COUNT(*) FROM windows WHERE query_string = ?", "A. Low")[0][0], 0)

        # ...and coming back reactivates it with no new requests for it.
        e.alias.write_text(json.dumps(alias_rows()), encoding="utf-8")
        snap["entities"]["Q1"]["aliases"] = ["A. Low"]
        e.snapshot.write_text(json.dumps(snap), encoding="utf-8")
        calls = e.api.calls
        code, out, _ = e.run("plan")
        self.assertEqual(code, 0)
        self.assertEqual(e.api.calls, calls)
        self.assertEqual(e.rows("SELECT active FROM entities WHERE prompt_name = ? AND query_string = ?",
                                "Alice Low (poet)", "A. Low"), [(1,)])
        self.assertEqual(e.rows("SELECT active FROM entities WHERE query_string = ?", "Ally Low"), [(0,)])
        assert_no_repeated_success(self, e.api)

    def test_a_removed_name_goes_inactive_with_all_its_strings(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        e.write_prompts([n for n in NAMES if n != "Bob High"] + ["Small 7"])
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.rows("SELECT DISTINCT active FROM entities WHERE prompt_name = ?", "Bob High"), [(0,)])
        self.assertGreater(e.rows("SELECT COUNT(*) FROM finds WHERE query_string = ?", "Bob High")[0][0], 0)


class TestDryRun(EnvCase):
    def test_dry_run_sends_nothing_writes_nothing_and_builds_no_transport(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.run("pilot", "--max-requests", "30")[0], 0)
        calls = e.api.calls
        state = e.snapshot_state()

        class NoTransport:
            def __init__(self, *a, **k):
                raise AssertionError("--dry-run must not construct a transport")

        real = fw.HttpTransport
        fw.HttpTransport = NoTransport
        self.addCleanup(setattr, fw, "HttpTransport", real)
        for command in ("plan", "pilot", "full"):
            code, out, err = e.run(command, "--dry-run", transport=None)
            self.assertEqual(code, 0, err)
            self.assertIn("DRY RUN", out)
            self.assertIn("finds", out)
        self.assertIn("EXTRAPOLATION", out)
        self.assertIn("windows", out)
        self.assertEqual(e.api.calls, calls)
        self.assertEqual(e.snapshot_state(), state)

    def test_dry_run_counts_match_what_a_real_run_then_sends(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        code, out, _ = e.run("pilot", "--dry-run", transport=None)
        self.assertEqual(code, 0)
        line = [l for l in out.splitlines() if l.startswith("requests still to send")][0]
        predicted = int(line.split(":")[1].split()[0])
        calls = e.api.calls
        self.assertEqual(e.run("pilot")[0], 0)
        # OLMo 2 finds are cached, so the derivable count is exact for the finds
        # and windows it knows about; extras of the other indexes are only known
        # once their main finds are in, hence a lower bound.
        sent = e.api.calls - calls
        self.assertLessEqual(predicted, sent)
        self.assertGreater(predicted, 0)

    def test_dry_run_on_a_missing_cache_creates_nothing(self):
        e = self.env()
        for command in ("plan", "pilot", "full"):
            code, out, _ = e.run(command, "--dry-run", transport=None)
            self.assertEqual(code, 0)
            self.assertIn("no cache at", out)
            self.assertFalse(e.cache.exists())
            self.assertFalse(e.cache.parent.exists())
        self.assertEqual(e.api.calls, 0)

    def test_dry_run_of_plan_without_a_cache_still_reports_the_query_strings(self):
        e = self.env()
        code, out, _ = e.run("plan", "--dry-run", transport=None)
        self.assertEqual(code, 0)
        self.assertIn("%d query strings" % N_STRINGS, out)
        self.assertIn("tokenizations: %d of %d" % (N_STRINGS, N_STRINGS), out)

    def test_dry_run_works_while_another_fetcher_holds_the_lock(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        with Cache(str(e.cache)):
            self.assertEqual(e.run("full", "--dry-run", transport=None)[0], 0)
            self.assertEqual(e.run("status", transport=None)[0], 0)


class TestStatus(EnvCase):
    def test_status_on_a_missing_cache_creates_nothing(self):
        e = self.env()
        code, out, _ = e.run("status", transport=None)
        self.assertEqual(code, 0)
        self.assertIn("no cache at", out)
        self.assertFalse(e.cache.exists())
        self.assertFalse(e.cache.parent.exists())

    def test_status_sends_nothing_and_reports(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.run("pilot", "--max-requests", "50")[0], 0)
        calls = e.api.calls
        state = e.snapshot_state()
        code, out, _ = e.run("status", transport=None)
        self.assertEqual(code, 0)
        self.assertEqual(e.api.calls, calls)
        self.assertEqual(e.snapshot_state(), state)
        for needle in (str(e.cache), "free disk", "phase plan", "phase pilot", "phase full", OLMO2, PILE,
                       "entities: %d active, 0 inactive" % N_STRINGS, "pilot list: 8 names",
                       "errors by HTTP status: none", "ETA"):
            self.assertIn(needle, out)
        self.assertNotIn("WARNING", out)
        self.assertIn("sha256 match", out)

    def test_status_warns_when_an_input_changed_after_plan(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        e.write_prompts(NAMES + ["Small 7"])
        code, out, _ = e.run("status", transport=None)
        self.assertEqual(code, 0)
        warnings = [l for l in out.splitlines() if l.startswith("WARNING")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("input prompts changed", warnings[0])
        e.alias.write_text(json.dumps({}), encoding="utf-8")
        code, out, _ = e.run("status", transport=None)
        self.assertEqual(len([l for l in out.splitlines() if l.startswith("WARNING")]), 2)
        # The live phases warn too (on stderr), then carry on.
        code, out, err = e.run("pilot", "--max-requests", "1")
        self.assertEqual(code, 0)
        self.assertIn("WARNING", err)

    def test_status_before_any_plan_says_so(self):
        e = self.env()
        Cache(str(e.cache)).close()  # an empty cache
        code, out, _ = e.run("status", transport=None)
        self.assertEqual(code, 0)
        self.assertIn("no plan has been run", out)

    def test_status_calls_out_400s_as_our_bug_and_shows_the_last_error(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        with Cache(str(e.cache)) as c:
            c.log_error(OLMO2, "find", {"index": OLMO2}, 400, "Exactly one of query and query_ids")
            c.log_error(OLMO2, "find", {"index": OLMO2}, 500, "oops")
        code, out, _ = e.run("status", transport=None)
        self.assertIn("1 HTTP 400 errors: our bug", out)
        self.assertIn("'400': 1", out)
        self.assertIn("status=500", out)

    def test_eta_falls_back_to_3200_per_hour_with_no_history(self):
        e = self.env()
        self.assertEqual(e.run("plan", "--max-requests", "5")[0], 0)  # tokenizations only
        code, out, _ = e.run("status", transport=None)
        self.assertIn("assumed 3200 requests/hour", out)

    def test_eta_uses_the_observed_rate_when_there_is_enough_history(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        conn = sqlite3.connect(str(e.cache))
        try:
            # 60 results one second apart, stamped by hand: the fake API runs faster than real time.
            conn.execute("DELETE FROM finds")
            for i in range(60):
                conn.execute(
                    "INSERT INTO finds VALUES (?, ?, 'plain', 1, '[[0, 1]]', '[]', 1.0, ?)",
                    (OLMO2, "q%d" % i, "2026-09-29T10:00:%02d+00:00" % i))
            conn.commit()
        finally:
            conn.close()
        code, out, _ = e.run("status", transport=None)
        self.assertIn("observed 3600 requests/hour", out)


class TestExitCodes(EnvCase):
    def test_blocked_exits_3_after_three_403_and_pauses_ten_minutes_each_time(self):
        e = self.env()
        e.api.always = (403, {"message": "Forbidden"})
        code, out, err = e.run("plan")
        self.assertEqual(code, 3)
        self.assertIn("blocked by infini-gram; wait and rerun, the cache resumes", err)
        self.assertEqual(e.api.calls, 3)
        self.assertEqual(sum(1 for s in e.ft.sleeps if s >= 600), 2)
        Cache(str(e.cache)).close()  # the lock was released
        # The block lifts: the same command carries on.
        e.api.always = None
        self.assertEqual(e.run("plan")[0], 0)

    def test_a_second_instance_exits_2_and_sends_nothing(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        calls = e.api.calls
        with Cache(str(e.cache)):
            for command in ("plan", "pilot", "full"):
                code, out, err = e.run(command)
                self.assertEqual(code, 2)
                # The per-cache or the per-user lock message, whichever is taken first.
                self.assertIn("already holds", err)
                self.assertTrue("per-user lock" in err or "cache lock" in err or "another fetcher already holds" in err)
        self.assertEqual(e.api.calls, calls)
        self.assertEqual(e.run("full", "--max-requests", "1")[0], 0)  # released afterwards

    def test_max_requests_zero_sends_nothing(self):
        e = self.env()
        code, out, _ = e.run("plan", "--max-requests", "0")
        self.assertEqual(code, 0)
        self.assertEqual(e.api.calls, 0)

    def test_bad_inputs_exit_1_with_a_message(self):
        e = self.env()
        e.prompts.unlink()
        code, out, err = e.run("plan")
        self.assertEqual(code, 1)
        self.assertIn("error:", err)
        Cache(str(e.cache)).close()  # and the lock is free


class TestPilot(EnvCase):
    def test_pilot_before_plan_errors_clearly(self):
        e = self.env()
        code, out, err = e.run("pilot")
        self.assertEqual(code, 1)
        self.assertIn("run `plan` first", err)
        self.assertFalse(e.cache.exists())  # no empty cache left behind
        self.assertEqual(e.api.calls, 0)

    def test_a_failed_main_find_leaves_the_pilot_unset_and_is_retried(self):
        e = self.env()
        e.api.reject = {"Dana Zero"}
        code, out, err = e.run("plan")
        self.assertEqual(code, 5)  # a 400 is logged and skipped, not fatal, but never "done"
        self.assertIn("items unfetched, rerun to retry", err)
        self.assertNotIn("done:", out)
        self.assertIn("pilot list NOT chosen", out)
        self.assertIn("1 main names still lack an OLMo 2 find", out)
        self.assertIsNone(e.meta("pilot_names"))
        self.assertEqual(e.rows("SELECT http_status FROM errors")[0], (400,))
        e.api.reject = set()
        self.assertEqual(e.run("plan")[0], 0)
        self.assertIsNotNone(e.meta("pilot_names"))
        assert_no_repeated_success(self, e.api)

    def test_a_failed_main_find_skips_the_extras_this_run_only(self):
        e = self.env()
        e.api.reject = {"Céline Mid"}  # its folded extra is "Celine Mid"
        self.assertEqual(e.run("plan")[0], 5)
        self.assertEqual(e.rows("SELECT COUNT(*) FROM finds WHERE query_string = ?", "Celine Mid")[0][0], 0)
        e.api.reject = set()
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.rows("SELECT COUNT(*) FROM finds WHERE query_string = ?", "Celine Mid")[0][0], 2)

    def test_pilot_before_the_list_is_chosen_errors_clearly(self):
        e = self.env()
        self.assertEqual(e.run("plan", "--max-requests", "5")[0], 0)
        code, out, err = e.run("pilot")
        self.assertEqual(code, 1)
        self.assertIn("pilot list is not set", err)

    def test_pilot_runs_its_names_on_all_five_indexes_and_no_others(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        pilot = set(json.loads(e.meta("pilot_names")))
        self.assertEqual(e.run("pilot")[0], 0)
        for index in INDEXES:
            done = {r[0] for r in e.rows(
                "SELECT DISTINCT e.prompt_name FROM finds f JOIN entities e ON e.query_string = f.query_string "
                "AND e.source = 'name' WHERE f.index_name = ?", index)}
            if index == OLMO2:
                self.assertEqual(done, set(NAMES))  # from plan
            else:
                self.assertEqual(done, pilot, index)
        # Windows too, on a non-OLMo index.
        self.assertGreater(e.rows("SELECT COUNT(*) FROM windows WHERE index_name = ?", PILE)[0][0], 0)
        # Order: the OLMo 2 pass is complete before the next index starts.
        order = [b["index"] for b in e.api.ok if b["query_type"] != "count"]
        first_other = next(i for i, ix in enumerate(order) if ix != OLMO2)
        self.assertNotIn(OLMO2, order[first_other:])
        # Finds come before windows for a name on an index.
        assert_no_repeated_success(self, e.api)

    def test_full_after_pilot_only_adds_the_rest(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.run("pilot")[0], 0)
        self.assertEqual(e.run("full")[0], 0)
        assert_no_repeated_success(self, e.api)
        self.assertEqual(e.run("full")[0], 0)

    def test_finds_come_before_windows_within_a_name(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.run("pilot", "--max-requests", "250")[0], 0)
        # For the first name on the first non-OLMo index: its main finds precede its windows.
        seq = [b for b in e.api.ok if b["index"] != OLMO2 and b["query_type"] in ("find", "get_doc_by_rank")]
        first_window = next(i for i, b in enumerate(seq) if b["query_type"] == "get_doc_by_rank")
        self.assertGreaterEqual(first_window, 2)


class TestThrottleIsNeverBypassed(EnvCase):
    def test_request_starts_are_at_least_a_gap_apart_across_a_whole_run(self):
        e = self.env()
        starts = []
        real_post = e.api.post

        def recording_post(body):
            starts.append(e.ft.mono_now)
            return real_post(body)

        e.api.post = recording_post
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.run("pilot", "--max-requests", "60")[0], 0)
        self.assertGreater(len(starts), 60)
        for a, b in zip(starts, starts[1:]):
            self.assertGreaterEqual(b - a, MIN_GAP_S - 1e-9)


class Harness:
    """A live Ctx on the cache of an Env, to step the pipeline without the CLI
    (fake transport, fake clock: nothing is real). Needs `plan` to have run."""

    def __init__(self, e):
        self.e = e
        self.cache = Cache(str(e.cache))
        throttle = infinigram.Throttle(
            MIN_GAP_S, e.ft.mono, e.ft.sleep, last_start_wall=self.cache.last_request_wall(), wall=e.ft.wall)
        self.client = infinigram.InfinigramClient(e.api, throttle, self.cache, None)
        self.ctx = fw.Ctx(self.client, self.cache)
        self.names = json.loads(self.cache.get_meta("task_names"))
        self.universe = fw.load_universe(self.cache, self.names)

    def close(self):
        self.cache.close()

    @property
    def sent(self):
        return self.client.requests_sent

    def main_finds(self, index):
        for n in self.names:
            fw.find_pool(self.ctx, index, fw.main_of(self.universe[n]).string)

    def analyse(self, index, with_windows):
        return fw.analyse_index(self.cache, self.universe, self.names, index, with_windows)

    def pipeline(self, indexes, with_windows):
        with contextlib.redirect_stdout(io.StringIO()):  # the progress lines
            fw.run_pipeline(self.ctx, self.universe, self.names, indexes, with_windows)


def predicted_requests(e, command):
    """The number a `--dry-run` of `command` reports as still to send."""
    code, out, err = e.run(command, "--dry-run", transport=None)
    assert code == 0, err
    line = [l for l in out.splitlines() if l.startswith("requests still to send")][0]
    return int(line.split(":")[1].split()[0]), out


class TestFailingTokenization(EnvCase):
    """R2-1: one string whose tokenization keeps failing must not block its pool."""

    TARGET = "(Small 1"

    def test_a_string_that_keeps_failing_is_given_up_after_three_runs(self):
        e = self.env()
        e.api.reject_count = {"Small 1"}
        for run in (1, 2):
            code, out, err = e.run("plan")
            self.assertEqual(code, 5, err)
            self.assertEqual(e.meta("tokenize_fail:Small 1"), str(run))
            self.assertEqual(e.api.count_attempts.count(self.TARGET), run)
            self.assertIsNone(e.meta("pilot_names"))
            self.assertIn("pilot list NOT chosen", out)
            # The plain find exists: the message must not claim the main name lacks a find.
            self.assertIn("plain find but not yet the glued one", out)
            self.assertNotIn("still lack an OLMo 2 find", out)
            status = e.run("status", transport=None)[1]
            self.assertIn("tokenization failing, still retried: 1 strings ['Small 1']", status)
        # Third run: given up in this run, plain only, and the pilot list can be chosen.
        code, out, err = e.run("plan")
        self.assertEqual(code, 5, err)  # this run's tokenize did fail
        self.assertEqual(e.meta("tokenize_fail:Small 1"), "3")
        self.assertIn("note: tokenizing 'Small 1' failed 3 runs in a row", out)
        self.assertIn("pilot list chosen", out)
        self.assertIsNotNone(e.meta("pilot_names"))
        # Fourth run: the failing count request is not sent again, and the run is clean.
        code, out, err = e.run("plan")
        self.assertEqual(code, 0, err)
        self.assertEqual(e.api.count_attempts.count(self.TARGET), 3)
        # Later phases do not resend it either; the string is fetched plain only.
        self.assertEqual(predicted_requests(e, "full")[1].count("tokenizations: 0 of"), 1)
        self.assertEqual(e.run("full")[0], 0)
        self.assertEqual(e.api.count_attempts.count(self.TARGET), 3)
        self.assertEqual(e.rows("SELECT DISTINCT form FROM finds WHERE query_string = ?", "Small 1"), [("plain",)])
        forms = e.rows("SELECT form, COUNT(*) FROM windows WHERE index_name = ? AND query_string = ? GROUP BY form",
                       OLMO2, "Small 1")
        self.assertEqual(forms, [("plain", 3)])
        status = e.run("status", transport=None)[1]
        self.assertIn("tokenization given up after 3 failed runs (plain form only): 1 strings ['Small 1']", status)
        self.assertIn("tokenization failing, still retried: 0 strings", status)

    def test_the_pilot_can_be_chosen_when_only_glued_halves_are_permanently_missing(self):
        e = self.env()
        e.api.reject_count = {"Bob High", "Eve Namesake"}
        for _ in range(3):
            e.run("plan")
        self.assertIsNotNone(e.meta("pilot_names"))
        pilot = json.loads(e.meta("pilot_names"))
        self.assertIn("Bob High", pilot)
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.run("pilot")[0], 0)


class TestFailureExitCodes(EnvCase):
    """R2-2 (driver half): no exit 0 or "done" over failures; a stuck run stops."""

    def test_too_many_consecutive_failures_exits_4_and_the_cache_resumes(self):
        e = self.env()
        e.api.always = (400, {"message": "nope"})
        code, out, err = e.run("plan")
        self.assertEqual(code, 4)
        self.assertEqual(e.api.calls, infinigram.MAX_CONSECUTIVE_FAILURES)
        self.assertIn("too many consecutive failures; check the error log (status) and rerun, the cache resumes", err)
        self.assertNotIn("done:", out)
        Cache(str(e.cache)).close()  # the lock was released
        e.api.always = None
        self.assertEqual(e.run("plan")[0], 0)

    def test_a_run_with_give_ups_exits_5_and_says_how_many_items_are_unfetched(self):
        e = self.env()
        e.api.reject = {"Dana Zero", "Small 2"}
        code, out, err = e.run("plan")
        self.assertEqual(code, 5)
        self.assertIn("2 items unfetched, rerun to retry", err)
        self.assertNotIn("done:", out)
        e.api.reject = set()
        code, out, err = e.run("plan")
        self.assertEqual(code, 0, err)
        self.assertIn("done:", out)

    def test_a_budget_stop_over_give_ups_is_not_a_clean_exit_0(self):
        e = self.env()
        e.api.reject = {"Dana Zero"}
        code, out, err = e.run("plan", "--max-requests", "35")
        self.assertIn("stopped at --max-requests 35", out)
        self.assertEqual(code, 5)
        self.assertIn("1 items unfetched, rerun to retry", err)

    def test_a_clean_run_still_exits_0_with_done(self):
        e = self.env()
        code, out, err = e.run("plan")
        self.assertEqual(code, 0, err)
        self.assertIn("done:", out)
        self.assertNotIn("unfetched", err)


class TestConstantsGuard(EnvCase):
    """R2-4: a changed constant must not silently mix two settings in one cache."""

    CHANGES = (("CAP_FULL", 100), ("SAMPLE_N", 40), ("DISP_LEN", 100), ("SEED", 7), ("MIN_GAP_S", 1.5))

    def test_plan_pilot_and_full_refuse_when_a_stored_constant_differs(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        stored, calls = e.meta("constants"), e.api.calls
        for name, value in self.CHANGES:
            with self.subTest(name):
                old = getattr(fw, name)
                setattr(fw, name, value)
                try:
                    for command in ("plan", "pilot", "full"):
                        code, out, err = e.run(command)
                        self.assertEqual(code, 1, (command, err))
                        self.assertIn(name, err)
                        self.assertIn("--allow-constants-change", err)
                finally:
                    setattr(fw, name, old)
                self.assertEqual(e.api.calls, calls)
                self.assertEqual(e.meta("constants"), stored)

    def test_status_and_dry_run_warn_about_a_changed_constant(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        old = fw.CAP_FULL
        fw.CAP_FULL = 100
        self.addCleanup(setattr, fw, "CAP_FULL", old)
        self.assertIn("WARNING: constants differ", e.run("status", transport=None)[1])
        self.assertIn("WARNING: constants differ", e.run("full", "--dry-run", transport=None)[1])

    def test_allow_constants_change_updates_run_meta_and_carries_on(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        old = fw.CAP_FULL
        fw.CAP_FULL = 100
        self.addCleanup(setattr, fw, "CAP_FULL", old)
        code, out, err = e.run("full", "--allow-constants-change")
        self.assertEqual(code, 0, err)
        self.assertIn("--allow-constants-change: run_meta constants updated (CAP_FULL: stored 200, now 100)", out)
        self.assertEqual(json.loads(e.meta("constants"))["CAP_FULL"], 100)
        # Accepted once: the next plain command is not refused, and nothing is resent.
        self.assertEqual(e.run("full")[0], 0)
        self.assertNotIn("WARNING: constants differ", e.run("status", transport=None)[1])
        assert_no_repeated_success(self, e.api)


class TestInputPaths(EnvCase):
    """R2-10: status, pilot and full compare the files `plan` recorded."""

    def test_default_flags_use_the_recorded_paths_not_the_default_files(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)  # non-default input paths, all given explicitly
        code, out, err = e.run_default_inputs("status", transport=None)
        self.assertEqual(code, 0, err)
        self.assertNotIn("WARNING", out)
        self.assertIn("sha256 match", out)
        code, out, err = e.run_default_inputs("pilot", "--max-requests", "1")
        self.assertEqual(code, 0, err)
        self.assertNotIn("WARNING", err)
        # An edit of the recorded file is still noticed.
        e.write_prompts(NAMES + ["Small 7"])
        out = e.run_default_inputs("status", transport=None)[1]
        warnings = [l for l in out.splitlines() if l.startswith("WARNING")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("input prompts changed", warnings[0])
        self.assertIn(os.path.abspath(str(e.prompts)), warnings[0])

    def test_an_explicit_path_is_compared_instead_of_the_recorded_one(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        other = Path(e.tmp.name) / "other_prompts.csv"
        other.write_bytes(e.prompts.read_bytes() + b"\n")
        out = e.run_argv(["status", "--cache", str(e.cache), "--prompts", str(other)], transport=None)[1]
        warnings = [l for l in out.splitlines() if l.startswith("WARNING")]
        self.assertEqual(len(warnings), 1)
        self.assertIn("input prompts changed", warnings[0])
        self.assertIn(str(other), warnings[0])

    def test_replan_refreshes_the_recorded_hashes(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        table = alias_rows()
        table["zzz yyy"] = ["Q99"]
        e.alias.write_text(json.dumps(table), encoding="utf-8")
        out = e.run("status", transport=None)[1]
        self.assertIn("input alias_table changed", out)
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.meta("input_sha256_alias_table"), fw.sha256_file(e.alias))
        for out in (e.run("status", transport=None)[1], e.run_default_inputs("status", transport=None)[1]):
            self.assertNotIn("WARNING", out)
            self.assertIn("sha256 match", out)


class TestDryRunEqualsSent(EnvCase):
    """R2-3: a string shared by two names is fetched once, and a dry run counts what a real run sends."""

    SHARED = ["Bill Hutchinson (baseball)", "Bill Hutchinson (politician)"]

    def shared_env(self):
        c = Corpus()
        c.set("Bill Hutchinson (baseball)", 3, 1)
        c.set("Bill Hutchinson (politician)", 4, 0)
        c.set("Bill Hutchinson", 5, 2)
        return self.env(names=self.SHARED, alias={}, snapshot={"entities": {}}, corpus=c)

    def fill_all_finds(self, e):
        h = Harness(e)
        try:
            h.pipeline(INDEXES, False)
        finally:
            h.close()

    def test_a_shared_string_is_tokenized_counted_and_sent_once(self):
        e = self.shared_env()
        out = e.run("plan", "--dry-run", transport=None)[1]
        self.assertIn("tokenizations: 3 of 3 strings", out)  # 2 names + the one stripped string they share
        self.assertEqual(e.run("plan")[0], 0)
        # 3 tokenizations, 2 mains x 2 forms, the shared string x 2 forms.
        self.assertEqual(e.api.calls, 3 + 4 + 2)
        self.assertEqual(len(e.rows("SELECT * FROM entities WHERE query_string = ?", "Bill Hutchinson")), 2)
        # Another index, mains cached: the shared extra still needs 2 finds, not 2 per name.
        h = Harness(e)
        try:
            h.main_finds(DCLM)
            r = h.analyse(DCLM, False)
            self.assertEqual(r["finds_missing"], 2)
            before = h.sent
            h.pipeline((DCLM,), False)
            self.assertEqual(h.sent - before, 2)
        finally:
            h.close()

    def test_dry_run_windows_equal_the_windows_sent_with_a_shared_string(self):
        e = self.shared_env()
        self.assertEqual(e.run("plan")[0], 0)
        self.fill_all_finds(e)
        predicted, out = predicted_requests(e, "full")
        self.assertNotIn("lower bound", out)
        calls = e.api.calls
        self.assertEqual(e.run("full")[0], 0)
        sent = e.api.calls - calls
        # Per index: 4 + 4 for the two mains, 7 once for the shared string.
        self.assertEqual(sent, 5 * (4 + 4 + 7))
        self.assertEqual(predicted, sent)

    def test_dry_run_windows_equal_the_windows_sent_on_the_main_fixture(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        self.fill_all_finds(e)
        predicted, out = predicted_requests(e, "full")
        self.assertNotIn("lower bound", out)
        calls = e.api.calls
        self.assertEqual(e.run("full")[0], 0)
        sent = e.api.calls - calls
        self.assertGreater(sent, 100)
        self.assertEqual(predicted, sent)

    def test_dry_run_counts_the_extra_forms_finds_and_then_the_windows_exactly(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        h = Harness(e)
        try:
            for index in INDEXES[1:]:
                h.main_finds(index)  # only the main names' finds are cached on this index
                r = h.analyse(index, False)
                self.assertGreater(r["finds_missing"], 0, index)  # the extras of the low-count names
                before = h.sent
                h.pipeline((index,), False)
                self.assertEqual(h.sent - before, r["finds_missing"], index)
                r = h.analyse(index, True)
                self.assertEqual(r["finds_missing"], 0)
                self.assertGreater(r["windows_missing"], 0, index)
                before = h.sent
                h.pipeline((index,), True)
                self.assertEqual(h.sent - before, r["windows_missing"], index)
        finally:
            h.close()


def edge_fixture():
    """Four names whose main count sits at the rule-5 gate (CAP_FULL = 200), each
    with one alias string, which is only queried while the main count is <= 200."""
    rows = [("Edge Plain", 200, 0, "Q11", "Edge Plain Alias"),        # 200 plain
            ("Edge Split", 150, 50, "Q12", "Edge Split Alias"),       # 150 + 50 = 200
            ("Edge Over", 150, 51, "Q13", "Edge Over Alias"),         # 150 + 51 = 201
            ("Edge Plain Over", 201, 0, "Q14", "Edge PO Alias")]      # 201 plain
    c, alias, snap = Corpus(), {}, {"entities": {}}
    for name, plain, glued, qid, extra in rows:
        c.set(name, plain, glued)
        c.set(extra, 1, 0)
        alias[name.lower()] = [qid]
        alias[extra.lower()] = [qid]
        snap["entities"][qid] = {"label": name, "aliases": [extra]}
    return [r[0] for r in rows], alias, snap, c


class TestRule5Boundary(EnvCase):
    """The extras gate reads the main name's COMBINED plain + glued count, at exactly CAP_FULL."""

    def edge_env(self):
        names, alias, snap, c = edge_fixture()
        return self.env(names=names, alias=alias, snapshot=snap, corpus=c)

    def n_finds(self, e, string, index=OLMO2):
        return e.rows("SELECT COUNT(*) FROM finds WHERE index_name = ? AND query_string = ?", index, string)[0][0]

    def test_extras_only_when_the_combined_main_count_is_at_most_the_cap(self):
        e = self.edge_env()
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(self.n_finds(e, "Edge Plain Alias"), 2)   # 200: allowed
        self.assertEqual(self.n_finds(e, "Edge Split Alias"), 2)   # 150 + 50 = 200: allowed
        self.assertEqual(self.n_finds(e, "Edge Over Alias"), 0)    # 150 + 51 = 201: gated out
        self.assertEqual(self.n_finds(e, "Edge PO Alias"), 0)      # 201 plain: gated out
        for name in ("Edge Plain", "Edge Split", "Edge Over", "Edge Plain Over"):
            self.assertEqual(self.n_finds(e, name), 2, name)       # mains are always queried

    def test_the_dry_run_applies_the_same_gate(self):
        e = self.edge_env()
        self.assertEqual(e.run("plan", "--max-requests", "0")[0], 0)  # entities only
        h = Harness(e)
        try:
            h.main_finds(OLMO2)
            r = h.analyse(OLMO2, False)
        finally:
            h.close()
        self.assertEqual(r["undecided_names"], 0)
        # Two allowed names x one alias x two forms.
        self.assertEqual(r["finds_missing"], 4)

    def test_plan_selected_label_and_form_at_the_boundary(self):
        e = self.edge_env()
        self.assertEqual(e.run("plan")[0], 0)

        def plan_rows(string):
            return e.rows("SELECT form, selected, COUNT(*) FROM plan WHERE index_name = ? AND query_string = ? "
                          "GROUP BY form, selected ORDER BY form", OLMO2, string)

        self.assertEqual(plan_rows("Edge Plain"), [("plain", "all", 200)])
        self.assertEqual(plan_rows("Edge Split"), [("glued", "all", 50), ("plain", "all", 150)])
        over = plan_rows("Edge Over")
        self.assertEqual({r[1] for r in over}, {"sample"})
        self.assertEqual(sum(r[2] for r in over), SAMPLE_N)
        self.assertEqual({r[1] for r in plan_rows("Edge Plain Over")}, {"sample"})
        # Glued positions of a sample keep their own form, and stay inside the glued shards.
        glued = e.rows("SELECT s, rank FROM plan WHERE index_name = ? AND query_string = ? AND form = 'glued'",
                       OLMO2, "Edge Over")
        segs = shards(51)
        for s, rank in glued:
            self.assertTrue(segs[s][0] <= rank < segs[s][1], (s, rank))
        # Every planned position of the fixture was accepted by the fake API when fetched.
        self.assertEqual(e.run("pilot", "--max-requests", "400")[0], 0)
        self.assertEqual(e.rows("SELECT COUNT(*) FROM errors")[0][0], 0)

    def test_the_pilot_is_bucketed_on_the_combined_count(self):
        c = Corpus()
        names = ["Combo %d" % i for i in range(1, 9)]
        for n in names:
            c.set(n, 8, 5)  # 8 plain alone: bucket 1-10; 13 combined: bucket 11-100
        e = self.env(names=names, corpus=c)
        self.assertEqual(e.run("plan")[0], 0)
        combined = choose_pilot({n: 13 for n in names}, SEED)
        plain_only = choose_pilot({n: 8 for n in names}, SEED)
        self.assertNotEqual(combined, plain_only, "fixture must tell the two bucketings apart")
        self.assertEqual(json.loads(e.meta("pilot_names")), combined)


class TestSeedsDifferPerIndex(EnvCase):
    def test_the_same_string_is_sampled_at_different_positions_on_each_index(self):
        from hallutrace.fetch_sampling import derive_seed, select_positions
        c = Corpus()
        c.set("Big Name", 300, 0)
        e = self.env(names=["Big Name"], corpus=c)
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.run("pilot")[0], 0)
        seen = {}
        for index in INDEXES:
            rows = e.rows("SELECT form, s, rank FROM plan WHERE index_name = ? AND query_string = ? "
                          "ORDER BY form, s, rank", index, "Big Name")
            self.assertEqual(len(rows), SAMPLE_N, index)
            expected = select_positions(shards(300), shards(0), fw.CAP_FULL, fw.SAMPLE_N,
                                        derive_seed(fw.SEED, index, "Big Name"))
            self.assertEqual([tuple(r) for r in rows], expected, index)
            seen[index] = tuple(rows)
        self.assertEqual(len(set(seen.values())), len(INDEXES), "two indexes drew the same sample")


if __name__ == "__main__":
    unittest.main()


class TestShards(EnvCase):
    """--shard i/n splits a task's names between fetchers, one cache per shard."""

    def test_shard_names_partition_the_sorted_list(self):
        names = ["d", "a", "c", "b", "e"]
        shards = [fw.shard_names(names, (i, 3)) for i in (1, 2, 3)]
        self.assertEqual(shards, [["a", "d"], ["b", "e"], ["c"]])
        self.assertEqual(sorted(sum(shards, [])), sorted(names))
        self.assertEqual(fw.shard_names(names, None), sorted(names))
        self.assertEqual(fw.shard_names(list(reversed(names)), (2, 3)), shards[1], "input order does not matter")

    def test_bad_specs_are_refused(self):
        self.assertEqual(fw.parse_shard("2/3"), (2, 3))
        self.assertIsNone(fw.parse_shard(None))
        for bad in ("0/3", "4/3", "1/0", "x", "1-3"):
            with self.assertRaises(fw.UsageError, msg=bad):
                fw.parse_shard(bad)

    def test_plan_records_its_shard_and_a_cache_holds_one_shard(self):
        e = self.env()
        code, out, err = e.run("plan", "--shard", "2/3")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(e.meta("task_names")), fw.shard_names(NAMES, (2, 3)))
        self.assertEqual(e.meta("shard"), "2/3")
        self.assertIn("shard 2/3", out)
        self.assertEqual(e.run("plan", "--shard", "1/3")[0], 1, "another shard on the same cache is refused")
        self.assertEqual(e.run("plan")[0], 1, "a whole-task plan on a shard's cache is refused")
        self.assertEqual(e.run("plan", "--shard", "2/3")[0], 0, "the same shard resumes")
        code, out, err = e.run("full", "--max-requests", "5")
        self.assertEqual(code, 0, err)
        self.assertIn("%d names" % len(fw.shard_names(NAMES, (2, 3))), out, "full runs the shard's names only")
        self.assertEqual(e.run("full", "--shard", "1/3", "--max-requests", "5")[0], 1)
        self.assertIn("shard 2/3", e.run("status")[1])

    def test_a_cache_planned_whole_is_not_sharded_later(self):
        e = self.env()
        self.assertEqual(e.run("plan")[0], 0)
        self.assertEqual(e.meta("shard"), "all")
        code, out, err = e.run("plan", "--shard", "1/2")
        self.assertEqual(code, 1)
        self.assertIn("separate --cache", err)

    def test_shard_dry_run_counts_only_its_names(self):
        e = self.env()
        code, out, err = e.run("plan", "--shard", "1/3", "--dry-run", transport=None)
        self.assertEqual(code, 0, err)
        self.assertIn("%d names (shard 1/3)" % len(fw.shard_names(NAMES, (1, 3))), out)


class TestBiographiesTask(EnvCase):
    """The biographies task reads `name` from task_meta and keeps its own run_meta keys."""

    def write_bio_prompts(self, e, names):
        import csv
        with open(e.prompts, "a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for i, n in enumerate(names):
                w.writerow(["bio%d" % i, "biographies", "x.csv", i, "Tell me a bio of %s" % n,
                            json.dumps({"name": n}, ensure_ascii=False)])

    def test_plan_biographies_shard(self):
        e = self.env()
        self.write_bio_prompts(e, NAMES[:6])
        code, out, err = e.run("plan", "--task", "biographies", "--shard", "1/2")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(e.meta("task_names:biographies")), fw.shard_names(NAMES[:6], (1, 2)))
        self.assertEqual(e.meta("shard:biographies"), "1/2")
        self.assertIsNone(e.meta("task_names"), "historical_events keys are untouched")
