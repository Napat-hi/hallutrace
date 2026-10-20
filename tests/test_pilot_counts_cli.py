"""Tests for scripts/pilot_counts.py, the offline CLI over the window cache.

Plain unittest. Fixture caches are built in a temp dir from hallutrace.infinigram.SCHEMA
(the only thing imported from it, so the fixture cannot drift from the real schema);
text is invented. Nothing touches the real cache or the network, and nothing is
written into the repo. Run:
python -m tests.test_pilot_counts_cli   or   python -m unittest tests.test_pilot_counts_cli
"""

import contextlib
import csv
import importlib.util
import io
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hallutrace import pilot_counts as pc
from hallutrace.infinigram import SCHEMA

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "pilot_counts.py"
_spec = importlib.util.spec_from_file_location("pilot_counts_cli", SCRIPT)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

IDX, IDX2 = list(pc.INDEX_LABELS)[:2]
EM_DASH = chr(0x2014)
SENTINEL = "zorblax"  # invented corpus word: must never reach results, README or stdout
NOW = "2026-09-30T00:00:00+00:00"


class CacheBuilder:
    """Writes a small window cache with the real schema."""

    def __init__(self, path):
        self.con = sqlite3.connect(str(path))
        self.con.executescript(SCHEMA)

    def entity(self, name, q, source="name", active=1, amb_name=0, amb_stripped=0):
        self.con.execute("INSERT INTO entities VALUES (?,?,?,?,?,?,?)",
                         (name, q, source, "Q1", amb_name, amb_stripped, active))

    def find(self, index, q, form, cnt):
        self.con.execute("INSERT INTO finds VALUES (?,?,?,?,?,?,?,?)",
                         (index, q, form, cnt, json.dumps([[0, cnt]]), "[]", 1.0, NOW))

    def plan(self, index, q, form, s, rank, selected):
        self.con.execute("INSERT INTO plan VALUES (?,?,?,?,?,?)",
                         (index, q, form, s, rank, selected))

    def window(self, index, q, form, s, rank, before, after, doc_ix, meta=None, needle=None):
        spans = [[before, None], [needle or q, "0"], [after, None]]
        self.con.execute("INSERT INTO windows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (index, q, form, s, rank, doc_ix, 1000, 100, 10,
                          json.dumps(spans), meta, 1.0, NOW))

    def meta(self, key, value):
        self.con.execute("INSERT OR REPLACE INTO run_meta VALUES (?,?)", (key, value))

    def close(self):
        self.con.commit()
        self.con.close()


def build_fixture(d, pilot=("Alpha Ames", "Bea Bright")):
    """Returns (cache path, prompts path).

    Alpha Ames: an `all` name (pool 4), a duplicate document (rank 0 and 2 share doc 0),
    a glued window, a window outside the plan, metadata with and without a URL, and two
    death years across prompts (year conflict). Bea Bright: a sampled name with one
    planned window never fetched, and no data at all on the second index. Cy Coe, Jr.:
    a comma in the name, a mixed name (an `all` string and a sampled stripped string).
    Ed Eld (poet) / Ed Eld (chef): the stripped string "Ed Eld" is shared by both.
    Gone Gil: only an inactive entity."""
    d = Path(d)
    b = CacheBuilder(d / "cache.sqlite")
    # Alpha Ames
    b.entity("Alpha Ames", "Alpha Ames")
    b.entity("Alpha Ames", "Old Alpha", source="alias", active=0)
    b.find(IDX, "Alpha Ames", "plain", 3)
    b.find(IDX, "Alpha Ames", "glued", 1)
    for form, rank in (("plain", 0), ("plain", 1), ("plain", 2), ("glued", 0)):
        b.plan(IDX, "Alpha Ames", form, 0, rank, "all")
    b.window(IDX, "Alpha Ames", "plain", 0, 0, "Notes on the quiltmaker %s. " % SENTINEL,
             " was born in 1850 near the river.", 0,
             meta=json.dumps({"url": "https://www.Example.org/a"}))
    b.window(IDX, "Alpha Ames", "plain", 0, 1, "A letter about ",
             " arrived in 1901 for the mayor.", 1, meta=json.dumps({"path": "x/y"}))
    b.window(IDX, "Alpha Ames", "plain", 0, 2, "Again, ",
             " was born in 1850, as %s says." % SENTINEL, 0)
    b.window(IDX, "Alpha Ames", "glued", 0, 0, "", " (1850-1901) painter", 5)
    b.window(IDX, "Alpha Ames", "plain", 1, 9, "", " extra 1850", 6)  # not in plan
    b.find(IDX2, "Alpha Ames", "plain", 0)
    b.find(IDX2, "Alpha Ames", "glued", 0)
    # Bea Bright
    b.entity("Bea Bright", "Bea Bright")
    b.find(IDX, "Bea Bright", "plain", 300)
    b.find(IDX, "Bea Bright", "glued", 0)
    b.plan(IDX, "Bea Bright", "plain", 0, 0, "sample")
    b.plan(IDX, "Bea Bright", "plain", 0, 1, "sample")  # planned, never fetched
    b.window(IDX, "Bea Bright", "plain", 0, 0, "The %s hall of " % SENTINEL,
             " died in 1890 in the capital.", 10)
    # Cy Coe, Jr.
    b.entity("Cy Coe, Jr.", "Cy Coe, Jr.")
    b.entity("Cy Coe, Jr.", "Cy Coe", source="stripped", amb_stripped=1)
    b.find(IDX, "Cy Coe, Jr.", "plain", 1)
    b.find(IDX, "Cy Coe, Jr.", "glued", 0)
    b.plan(IDX, "Cy Coe, Jr.", "plain", 0, 0, "all")
    b.window(IDX, "Cy Coe, Jr.", "plain", 0, 0, "", " skated well; born in 1901 says a page.", 7)
    b.find(IDX, "Cy Coe", "plain", 400)
    b.find(IDX, "Cy Coe", "glued", 0)
    b.plan(IDX, "Cy Coe", "plain", 0, 0, "sample")
    b.plan(IDX, "Cy Coe", "plain", 0, 1, "sample")
    b.window(IDX, "Cy Coe", "plain", 0, 0, "The skater ", " (b. 1901) won a cup.", 8)
    # Ed Eld (poet) and Ed Eld (chef) share the stripped string "Ed Eld"
    for who in ("poet", "chef"):
        b.entity("Ed Eld (%s)" % who, "Ed Eld (%s)" % who)
        b.entity("Ed Eld (%s)" % who, "Ed Eld", source="stripped", amb_stripped=1)
        b.find(IDX, "Ed Eld (%s)" % who, "plain", 0)
        b.find(IDX, "Ed Eld (%s)" % who, "glued", 0)
    b.find(IDX, "Ed Eld", "plain", 2)
    b.find(IDX, "Ed Eld", "glued", 0)
    b.plan(IDX, "Ed Eld", "plain", 0, 0, "all")
    b.plan(IDX, "Ed Eld", "plain", 0, 1, "all")
    b.window(IDX, "Ed Eld", "plain", 0, 0, "", " died in 1760 in a small town.", 20)
    b.window(IDX, "Ed Eld", "plain", 0, 1, "Born in 1710, ", " grew up nearby.", 21)
    # Gone Gil: inactive only
    b.entity("Gone Gil", "Gone Gil", active=0)
    if pilot is not None:
        b.meta("pilot_names", json.dumps(list(pilot)))
    b.close()

    prompts = d / "prompts.csv"
    with open(prompts, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "task", "source_file", "source_row", "prompt", "task_meta"])

        def add(pid, task, meta):
            w.writerow([pid, task, "x.csv", 1, "invented prompt", json.dumps(meta)])

        add("h1", "historical_events", {
            "name1": "Alpha Ames", "birth1": 1850, "death1": 1901,
            "name2": "Bea Bright", "birth2": 1820, "death2": 1890, "kind": "x"})
        add("h2", "historical_events", {
            "name1": "Cy Coe, Jr.", "birth1": 1901, "death1": 1960,
            "name2": "Ed Eld (poet)", "birth2": 1700, "death2": 1760})
        add("h3", "historical_events", {
            "name1": "Ed Eld (chef)", "birth1": 1710, "death1": 1770,
            "name2": "Alpha Ames", "birth2": 1850, "death2": 1902})
        add("b1", "biographies", {"name": "Somebody Else"})
    return d / "cache.sqlite", prompts


def run(*argv):
    """main(argv) with stdout and stderr captured: (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def read_csv(path, encoding="utf-8"):
    with open(path, encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


def write_packet(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pc.PACKET_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = Path(self._tmp.name)
        self.cache, self.prompts = build_fixture(self.d)
        self.out = self.d / "out"
        self.packet = self.d / "tmp" / "counting_rule_packet_with_text.csv"

    def args(self, cmd, *extra, packet=None):
        return [cmd, "--cache", self.cache, "--prompts", self.prompts, "--out", self.out,
                "--packet", packet or self.packet, *extra]

    def go(self, cmd, *extra, packet=None):
        return run(*self.args(cmd, *extra, packet=packet))

    def names_file(self, names):
        p = self.d / "names.json"
        p.write_text(json.dumps(names), encoding="utf-8")
        return p

    def all_files_text(self, folder):
        return "\n".join(p.read_text(encoding="utf-8", errors="replace")
                         for p in Path(folder).rglob("*") if p.is_file())


class ReadOnlyTests(Base):
    def test_connection_is_read_only(self):
        conn = cli.open_readonly(self.cache)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()["n"], 10)
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("INSERT INTO run_meta VALUES ('k', 'v')")
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM entities")
        finally:
            conn.close()

    def test_uri_is_mode_ro_never_immutable(self):
        seen = []
        real = sqlite3.connect

        def spy(target, *a, **k):
            seen.append((target, k))
            return real(target, *a, **k)

        with mock.patch.object(cli.sqlite3, "connect", spy):
            cli.open_readonly(self.cache).close()
        target, kwargs = seen[0]
        self.assertTrue(target.startswith("file:") and target.endswith("?mode=ro"))
        self.assertNotIn("immutable", target)
        self.assertNotIn("\\", target)
        self.assertTrue(kwargs["uri"])
        self.assertEqual(kwargs["timeout"], 60)
        self.assertIsNone(kwargs["isolation_level"])

    def test_one_read_transaction_and_no_writes(self):
        stmts = []
        real = cli.open_readonly

        def traced(path):
            conn = real(path)
            conn.set_trace_callback(stmts.append)
            return conn

        with mock.patch.object(cli, "open_readonly", traced):
            data = cli.load_cache(self.cache, cli.ALL_NAMES)
        self.assertEqual(stmts[0], "BEGIN")
        self.assertEqual(stmts[-1], "COMMIT")
        self.assertEqual(stmts.count("BEGIN"), 1)
        self.assertEqual(stmts.count("COMMIT"), 1)
        for s in stmts[1:-1]:
            self.assertTrue(s.lstrip().upper().startswith("SELECT"), s)
        self.assertEqual(len(data["names"]), 5)

    def test_run_leaves_the_cache_file_unchanged(self):
        before = self.cache.read_bytes()
        self.assertEqual(self.go("count")[0], 0)
        self.assertEqual(self.go("packet")[0], 0)
        self.assertEqual(self.cache.read_bytes(), before)
        self.assertEqual([p.name for p in self.d.glob("cache.sqlite*")], ["cache.sqlite"])

    def test_missing_cache_is_a_clean_error_and_creates_nothing(self):
        missing = self.d / "nope.sqlite"
        code, _, err = run("count", "--cache", missing, "--prompts", self.prompts,
                           "--out", self.out)
        self.assertEqual(code, 1)
        self.assertIn("cannot read the cache", err)
        self.assertNotIn("Traceback", err)
        self.assertFalse(missing.exists())

    def test_locked_database_retries_then_fails(self):
        def locked(path):
            raise sqlite3.OperationalError("database is locked")

        with mock.patch.object(cli, "open_readonly", locked), \
                mock.patch.object(cli.time, "sleep") as sleep:
            code, _, err = run("count", "--cache", self.cache, "--prompts", self.prompts,
                               "--out", self.out, "--names", "all")
        self.assertEqual(code, 1)
        self.assertEqual(sleep.call_args_list, [mock.call(2)] * 5)
        self.assertIn("locked", err)

    def test_locked_database_recovers(self):
        real = cli.open_readonly
        calls = []

        def flaky(path):
            calls.append(path)
            if len(calls) <= 2:
                raise sqlite3.OperationalError("database is locked")
            return real(path)

        with mock.patch.object(cli, "open_readonly", flaky), \
                mock.patch.object(cli.time, "sleep") as sleep:
            code, _, _ = self.go("count")
        self.assertEqual(code, 0)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(len(calls), 3)

    def test_other_operational_errors_do_not_retry(self):
        def broken(path):
            raise sqlite3.OperationalError("unable to open database file")

        with mock.patch.object(cli, "open_readonly", broken), \
                mock.patch.object(cli.time, "sleep") as sleep:
            code, _, err = self.go("count")
        self.assertEqual(code, 1)
        sleep.assert_not_called()
        self.assertIn("unable to open", err)


class CountTests(Base):
    def test_count_end_to_end(self):
        code, out, err = self.go("count")
        self.assertEqual(code, 0, err)
        counts, summary = self.out / "pilot_counts.csv", self.out / "pilot_summary.md"
        self.assertTrue(counts.exists() and summary.exists())
        with open(counts, encoding="utf-8", newline="") as f:
            self.assertEqual(next(csv.reader(f)), pc.CSV_COLUMNS)
        rows = read_csv(counts)
        # Default names = stored pilot_names (Alpha: 1 birth + 2 death years, Bea: 2 facts).
        self.assertEqual({r["prompt_name"] for r in rows}, {"Alpha Ames", "Bea Bright"})
        facts = 3 + 2
        self.assertEqual(len(rows), facts * 2 * len(pc.HALF_WIDTHS) * len(pc.MATCHERS) * 2)  # 2 indexes
        self.assertIn(str(counts), out)
        self.assertNotIn(SENTINEL, out)

        def cell(name, index, fact, year, width="full", matcher="bare", strings="all"):
            hits = [r for r in rows if (r["prompt_name"], r["index"], r["fact"], r["year"],
                                        r["half_width"], r["matcher"], r["strings"])
                    == (name, index, fact, str(year), width, matcher, strings)]
            self.assertEqual(len(hits), 1)
            return hits[0]

        a = cell("Alpha Ames", IDX, "birth", 1850)
        self.assertEqual((a["status"], a["selected"], a["n_c_unit"]), ("counted", "all", "docs"))
        self.assertEqual((a["mentions_matched"], a["n_docs"], a["n_c"]), ("3", "2", "2"))
        self.assertEqual(a["matched_glued"], "1")
        self.assertEqual(a["matched_with_url"], "1")  # only the URL metadata window
        self.assertEqual((a["windows_planned"], a["windows_fetched"]), ("4", "4"))
        self.assertEqual(a["year_conflict"], "0")
        self.assertEqual(cell("Alpha Ames", IDX, "death", 1902)["year_conflict"], "1")
        self.assertEqual(cell("Alpha Ames", IDX2, "birth", 1850)["status"], "zero")
        b = cell("Bea Bright", IDX, "death", 1890)
        self.assertEqual((b["selected"], b["n_c_unit"], b["incomplete"]),
                         ("sample", "est_mentions", "1"))
        self.assertEqual((b["windows_planned"], b["windows_fetched"]), ("2", "1"))
        self.assertEqual(cell("Bea Bright", IDX, "birth", 1820)["status"], "unknown")
        self.assertEqual(cell("Bea Bright", IDX2, "birth", 1820)["not_fetched_strings"], "1")

        text = summary.read_text(encoding="utf-8")
        self.assertIn("NOT tau", text)
        self.assertIn("Alpha Ames, death: 1901, 1902", text)
        self.assertIn("extra_windows: 1", text)  # the window outside the plan
        self.assertNotIn(SENTINEL, self.all_files_text(self.out))

    def test_names_file_all_and_default(self):
        code, _, err = self.go("count", "--names", self.names_file(["Cy Coe, Jr."]))
        self.assertEqual(code, 0, err)
        rows = read_csv(self.out / "pilot_counts.csv")
        self.assertEqual({r["prompt_name"] for r in rows}, {"Cy Coe, Jr."})
        cy = [r for r in rows if r["strings"] == "no_stripped" and r["half_width"] == "full"
              and r["matcher"] == "bare" and r["index"] == IDX and r["fact"] == "birth"]
        self.assertEqual(cy[0]["n_strings"], "1")
        cy_all = [r for r in rows if r["strings"] == "all" and r["half_width"] == "full"
                  and r["matcher"] == "bare" and r["index"] == IDX and r["fact"] == "birth"]
        self.assertEqual((cy_all[0]["n_strings"], cy_all[0]["selected"]), ("2", "sample"))

        code, out, err = self.go("count", "--names", "all")
        self.assertEqual(code, 0, err)
        names = {r["prompt_name"] for r in read_csv(self.out / "pilot_counts.csv")}
        self.assertEqual(names, {"Alpha Ames", "Bea Bright", "Cy Coe, Jr.", "Ed Eld (poet)",
                                 "Ed Eld (chef)"})  # not Gone Gil: inactive only
        rows = read_csv(self.out / "pilot_counts.csv")
        shared = {r["prompt_name"] for r in rows if r["shared_string"] == "1"}
        self.assertEqual(shared, {"Ed Eld (poet)", "Ed Eld (chef)"})
        self.assertTrue(any(r["prompt_name"] == "Alpha Ames" and r["n_inactive_strings"] == "1"
                            for r in rows))
        self.assertIn("5 names", out)

    def test_names_errors(self):
        code, _, err = self.go("count", "--names", self.names_file(["Nobody Known"]))
        self.assertEqual(code, 1)
        self.assertIn("Nobody Known", err)
        code, _, err = self.go("count", "--names", self.names_file(["Gone Gil"]))
        self.assertEqual(code, 1)  # inactive only counts as absent
        self.assertIn("Gone Gil", err)
        bad = self.d / "bad.json"
        bad.write_text('{"a": 1}', encoding="utf-8")
        code, _, err = self.go("count", "--names", bad)
        self.assertEqual(code, 1)
        self.assertIn("JSON list", err)
        code, _, err = self.go("count", "--names", self.d / "missing.json")
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", err)

    def test_no_stored_pilot_names_is_a_clear_error(self):
        (self.d / "second").mkdir()
        cache2, _ = build_fixture(self.d / "second", pilot=None)
        code, _, err = run("count", "--cache", cache2, "--prompts", self.prompts,
                           "--out", self.out)
        self.assertEqual(code, 1)
        self.assertIn("pilot_names", err)
        self.assertFalse((self.out / "pilot_counts.csv").exists())

    def test_name_without_years_is_an_error(self):
        prompts = self.d / "few.csv"
        with open(prompts, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt_id", "task", "source_file", "source_row", "prompt", "task_meta"])
            w.writerow(["h1", "historical_events", "x", 1, "p",
                        json.dumps({"name1": "Alpha Ames", "birth1": 1850, "death1": 1901})])
        code, _, err = run("count", "--cache", self.cache, "--prompts", prompts,
                           "--out", self.out)  # pilot names include Bea Bright
        self.assertEqual(code, 1)
        self.assertIn("Bea Bright", err)

    def test_load_years_collects_both_slots(self):
        years = cli.load_years(self.prompts)
        self.assertEqual(years["Alpha Ames"], {"birth": [1850], "death": [1901, 1902]})
        self.assertEqual(years["Ed Eld (chef)"], {"birth": [1710], "death": [1770]})
        self.assertNotIn("Somebody Else", years)


class PacketTests(Base):
    def run_packet(self, *extra):
        return self.go("packet", "--names", "all", *extra)

    def test_packet_end_to_end(self):
        code, out, err = self.run_packet()
        self.assertEqual(code, 0, err)
        self.assertTrue(self.packet.exists())
        raw = self.packet.read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))  # utf-8-sig
        rows = read_csv(self.packet, "utf-8-sig")
        self.assertGreater(len(rows), 4)
        self.assertLess(len(rows), pc.MIN_LABELLED)
        self.assertEqual(list(rows[0]), pc.PACKET_COLUMNS)
        self.assertEqual([r["packet_id"] for r in rows],
                         ["P%03d" % n for n in range(1, len(rows) + 1)])
        self.assertTrue(all(r["label"] == "" for r in rows))
        self.assertTrue(any(SENTINEL in r["snippet"] or "<<" in r["snippet"] for r in rows))
        self.assertIn(b'"packet_id"', raw.split(b"\n")[0])  # QUOTE_ALL
        strata = read_csv(self.out / "pilot_packet_strata.csv")
        self.assertEqual(list(strata[0]), pc.STRATA_COLUMNS)
        self.assertEqual(sum(int(s["drawn"]) for s in strata), len(rows))
        readme = self.packet.with_name("counting_rule_packet_with_text_README.md")
        self.assertTrue(readme.exists())
        head = readme.read_text(encoding="utf-8").splitlines()[:4]
        self.assertEqual(head, [
            "`correct`: the snippet states THIS person's birth year (fact = birth) or death "
            "year (fact = death) as that fact;",
            "`other_fact`: this person, but the year is about something else (an event, a "
            "publication, the other life date);",
            "`other_person`: the year, or the whole passage, is about someone else;",
            "`unclear`: the snippet does not let you decide. Judge from the snippet only.",
        ])
        text = readme.read_text(encoding="utf-8")
        self.assertIn("CSV UTF-8", text)
        self.assertIn("Only the `label` column may change", text)
        self.assertIn("quota unmet", text)  # the tiny fixture cannot meet the quotas
        # Copyright: corpus text only in the packet file, never in results, README or stdout.
        self.assertNotIn(SENTINEL, text)
        self.assertNotIn(SENTINEL, self.all_files_text(self.out))
        self.assertNotIn(SENTINEL, out)
        self.assertEqual(list(self.out.iterdir()), [self.out / "pilot_packet_strata.csv"])

    def test_same_seed_same_packet(self):
        self.assertEqual(self.run_packet("--seed", 5)[0], 0)
        first = self.packet.read_bytes()
        self.assertEqual(self.run_packet("--seed", 5)[0], 0)  # no labels: overwrite is fine
        self.assertEqual(self.packet.read_bytes(), first)

    def test_readme_name_follows_the_packet_name(self):
        self.packet = self.d / "tmp" / "other.csv"
        self.assertEqual(self.run_packet()[0], 0)
        self.assertTrue((self.d / "tmp" / "other_README.md").exists())

    def test_refuses_a_packet_under_out(self):
        for where in (self.out / "packet.csv", self.out / "sub" / "packet.csv"):
            code, _, err = self.go("packet", "--names", "all", packet=where)
            self.assertEqual(code, 1)
            self.assertIn("refusing --packet", err)
            self.assertFalse(where.exists())
        self.assertFalse(self.out.exists())  # refused before anything was written

    def test_refuses_a_packet_under_repo_results(self):
        where = ROOT / "results" / "zz_test_refused_packet.csv"
        code, _, err = self.go("packet", "--names", "all", packet=where)
        self.assertEqual(code, 1)
        self.assertIn("results", err)
        self.assertFalse(where.exists())
        code, _, err = run("score", "--out", self.out, "--packet", where)
        self.assertEqual(code, 1)
        self.assertIn("refusing --packet", err)

    def label_first_row(self, label="correct"):
        rows = read_csv(self.packet, "utf-8-sig")
        rows[0]["label"] = label
        write_packet(self.packet, rows)
        return self.packet.read_bytes()

    def test_refuses_to_overwrite_labels_without_force(self):
        self.assertEqual(self.run_packet()[0], 0)
        labelled = self.label_first_row()
        code, _, err = self.run_packet()
        self.assertEqual(code, 1)
        self.assertIn("labels", err)
        self.assertIn("--force", err)
        self.assertEqual(self.packet.read_bytes(), labelled)  # untouched
        self.assertEqual(list(self.packet.parent.glob("*.bak.csv")), [])

    def test_force_backs_up_then_overwrites(self):
        self.assertEqual(self.run_packet()[0], 0)
        labelled = self.label_first_row("other_fact")
        code, out, err = self.run_packet("--force")
        self.assertEqual(code, 0, err)
        baks = list(self.packet.parent.glob("counting_rule_packet_with_text.*.bak.csv"))
        self.assertEqual(len(baks), 1)
        self.assertRegex(baks[0].name, r"^counting_rule_packet_with_text\.\d{8}-\d{6}\.bak\.csv$")
        self.assertEqual(baks[0].read_bytes(), labelled)  # the labels survive in the backup
        rows = read_csv(self.packet, "utf-8-sig")
        self.assertTrue(all(r["label"] == "" for r in rows))
        self.assertIn(baks[0].name, out)
        # A second forced run inside the same second must not clobber the first backup.
        self.label_first_row("correct")
        self.assertEqual(self.run_packet("--force")[0], 0)
        self.assertEqual(len(list(self.packet.parent.glob("*.bak.csv"))), 2)
        self.assertEqual(baks[0].read_bytes(), labelled)

    def test_unreadable_existing_packet_counts_as_labelled(self):
        self.packet.parent.mkdir(parents=True)
        self.packet.write_bytes(b"\xff\xfe\x00 not a csv \x80")
        code, _, err = self.run_packet()
        self.assertEqual(code, 1)
        self.assertIn("--force", err)

    def test_no_candidates_writes_nothing(self):
        # Years that occur in no window: no candidates, so no packet and no overwrite.
        prompts = self.d / "other_years.csv"
        with open(prompts, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt_id", "task", "source_file", "source_row", "prompt", "task_meta"])
            w.writerow(["h1", "historical_events", "x", 1, "p", json.dumps(
                {"name1": "Alpha Ames", "birth1": 1111, "death1": 1222})])
        code, _, err = run("packet", "--cache", self.cache, "--prompts", prompts,
                           "--out", self.out, "--packet", self.packet,
                           "--names", self.names_file(["Alpha Ames"]))
        self.assertEqual(code, 1)
        self.assertIn("no packet candidates", err)
        self.assertFalse(self.packet.exists())


class ScoreTests(Base):
    def synthetic(self, n=50, labels=None):
        """Writes a labelled packet (n rows) and its strata file straight from the columns."""
        cycle = labels or ["correct", "correct", "other_fact", "other_person", "unclear"]
        rows = []
        for i in range(n):
            r = {c: "" for c in pc.PACKET_COLUMNS}
            r.update(packet_id="P%03d" % (i + 1), index=IDX, prompt_name="Alpha Ames",
                     query_string="Alpha Ames", query_source="name" if i % 3 else "stripped",
                     form="plain", s=0, rank=i, fact="birth" if i % 2 else "death", year=1850,
                     distance_chars=10, band=pc.BAND_LABELS[i % 6], matched_bare=1,
                     matched_cue=i % 2, snippet="invented snippet", label=cycle[i % len(cycle)])
            rows.append(r)
        self.packet.parent.mkdir(parents=True, exist_ok=True)
        write_packet(self.packet, rows)
        strata = [{"band": b, "matched_cue": cue, "query_source": "name", "population": 20,
                   "drawn": 4} for b in pc.BAND_LABELS for cue in (0, 1)]
        cli.write_rows(self.out / "pilot_packet_strata.csv", pc.STRATA_COLUMNS, strata)
        return rows

    def test_score_end_to_end(self):
        self.synthetic()
        code, out, err = run("score", "--out", self.out, "--packet", self.packet)
        self.assertEqual(code, 0, err)  # no --cache: score never opens it
        text = (self.out / "pilot_precision.md").read_text(encoding="utf-8")
        self.assertIn("Pilot precision", text)
        self.assertIn("Labelled rows: 50", text)
        self.assertNotIn("invented snippet", text)
        self.assertIn("pilot_precision.md", out)

    def test_score_does_not_need_the_cache(self):
        self.synthetic()
        with mock.patch.object(cli, "open_readonly", side_effect=AssertionError("opened")):
            code, _, err = run("score", "--cache", self.d / "absent.sqlite", "--out", self.out,
                               "--packet", self.packet)
        self.assertEqual(code, 0, err)

    def test_excel_style_labels_are_accepted(self):
        self.synthetic(labels=["Correct ", " OTHER_FACT", "other_person", "unclear"])
        code, _, err = run("score", "--out", self.out, "--packet", self.packet)
        self.assertEqual(code, 0, err)

    def test_too_few_labels_is_a_clean_error(self):
        rows = self.synthetic()
        for r in rows[10:]:
            r["label"] = ""
        write_packet(self.packet, rows)
        code, _, err = run("score", "--out", self.out, "--packet", self.packet)
        self.assertEqual(code, 1)
        self.assertIn("10 labelled rows", err)
        self.assertIn("at least %d" % pc.MIN_LABELLED, err)
        self.assertNotIn("Traceback", err)
        self.assertFalse((self.out / "pilot_precision.md").exists())

    def test_bad_label_lists_packet_ids(self):
        rows = self.synthetic()
        rows[3]["label"] = "maybe"
        rows[7]["label"] = "yes"
        write_packet(self.packet, rows)
        code, _, err = run("score", "--out", self.out, "--packet", self.packet)
        self.assertEqual(code, 1)
        self.assertIn("P004", err)
        self.assertIn("P008", err)
        self.assertFalse((self.out / "pilot_precision.md").exists())

    def test_packet_made_by_the_cli_scores_to_too_few(self):
        # A real packet from the tiny fixture has fewer than 40 rows: even fully labelled
        # it is refused, through the CLI, with the count in the message.
        self.assertEqual(self.go("packet", "--names", "all")[0], 0)
        rows = read_csv(self.packet, "utf-8-sig")
        self.assertLess(len(rows), pc.MIN_LABELLED)
        for r in rows:
            r["label"] = "correct"
        write_packet(self.packet, rows)
        code, _, err = self.go("score")
        self.assertEqual(code, 1)
        self.assertIn("%d labelled rows" % len(rows), err)

    def test_missing_files_and_columns(self):
        code, _, err = run("score", "--out", self.out, "--packet", self.packet)
        self.assertEqual(code, 1)
        self.assertIn("cannot read", err)
        self.synthetic()
        (self.out / "pilot_packet_strata.csv").unlink()
        code, _, err = run("score", "--out", self.out, "--packet", self.packet)
        self.assertEqual(code, 1)
        self.assertIn("pilot_packet_strata.csv", err)
        self.packet.write_text("a,b\n1,2\n", encoding="utf-8")
        code, _, err = run("score", "--out", self.out, "--packet", self.packet)
        self.assertEqual(code, 1)
        self.assertIn("lacks columns", err)

    def test_score_refuses_a_packet_under_out(self):
        self.out.mkdir()
        inside = self.out / "packet.csv"
        write_packet(inside, [])
        code, _, err = run("score", "--out", self.out, "--packet", inside)
        self.assertEqual(code, 1)
        self.assertIn("refusing --packet", err)


class ProductionTests(Base):
    def test_no_infinigram_import_anywhere(self):
        # The script is loaded (and a full count run) in a fresh interpreter that never
        # imported the test module, so sys.modules shows only what the script pulled in.
        code = (
            "import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('pc_cli', sys.argv[1])\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "assert 'hallutrace.infinigram' not in sys.modules, 'imported on load'\n"
            "rc = m.main(['count', '--cache', sys.argv[2], '--prompts', sys.argv[3],\n"
            "             '--out', sys.argv[4], '--names', 'all'])\n"
            "assert rc == 0, rc\n"
            "rc = m.main(['packet', '--cache', sys.argv[2], '--prompts', sys.argv[3],\n"
            "             '--out', sys.argv[4], '--packet', sys.argv[5], '--names', 'all'])\n"
            "assert rc == 0, rc\n"
            "bad = [k for k in sys.modules if k == 'hallutrace.infinigram'\n"
            "       or k.startswith(('urllib.request', 'http.client', 'requests'))]\n"
            "assert not bad, bad\n"
            "print('clean')\n"
        )
        res = subprocess.run(
            [sys.executable, "-c", code, str(SCRIPT), str(self.cache), str(self.prompts),
             str(self.out), str(self.packet)],
            capture_output=True, text=True, cwd=str(self.d), encoding="utf-8")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("clean", res.stdout)

    def test_no_infinigram_import_statement_in_the_script(self):
        for line in SCRIPT.read_text(encoding="utf-8").splitlines():
            if re.match(r"\s*(import|from)\s", line):
                self.assertNotIn("infinigram", line)

    def test_no_em_dash_in_the_files(self):
        for path in (SCRIPT, Path(__file__)):
            self.assertNotIn(EM_DASH, path.read_text(encoding="utf-8"), path.name)

    def test_help_and_bad_command_line(self):
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            cli.main(["nonsense"])
        self.assertEqual(cm.exception.code, 2)
        self.assertEqual(cli.DEFAULT_SEED, 20260930)
        args = cli.build_parser().parse_args(["count"])
        self.assertEqual(args.seed, 20260930)
        self.assertEqual(args.out, ROOT / "results" / "counting_pilot")
        self.assertEqual(args.packet, ROOT / "outputs" / "counting_rule_packet_with_text.csv")
        self.assertEqual(args.cache, ROOT / "data" / "corpus_cache" / "infinigram.sqlite")
        self.assertIsNone(args.names)


if __name__ == "__main__":
    unittest.main()
