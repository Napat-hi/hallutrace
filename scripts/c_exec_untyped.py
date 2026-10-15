r"""Untyped Biographies slots: merged type, phrase check, F5 range.

    .\.venv\Scripts\python.exe scripts\c_exec_untyped.py --dry-run
    .\.venv\Scripts\python.exe scripts\c_exec_untyped.py --cache <phrase cache> --max-requests <n>
        [--min-gap <seconds, default 1.5>] [--max-errors <n>] [--max-consecutive-errors <n>]
    .\.venv\Scripts\python.exe scripts\c_exec_untyped.py --cache <phrase cache> --cache-only

Reads results/bio_year_claims.csv, results/bio_counts/merged/counts.csv (the counting cell, roles c
and h; every row must match bio_counts.RULE) and historical_events' tau (results/c_exec/he_tau.csv,
variant headline_406, 10th percentile). Keeps the claims whose generation's sidecar
outcome is `ok` (`stability.ok_claims` on `stability.bio_samples(--bexec-root, "outputs_filtered")`,
a local copy of the verifier `inputs/` and `outputs_filtered/`), builds the slots with
`stability.slot_items` (a sample stating the right year is right; role h years: the single dominant
wrong year) through `c_exec_untyped.from_slot_items`, classifies them (rule C), sends infini-gram
`count` queries for the birth phrases of every `unknown` count of an S2 or S3 slot (rule E,
`--scope all-birth` for every regime), stopping at a target's first hit, classifies again with
the hits and writes the F5 range (rule F) over the slots with at least one wrong sample
(`c_exec2.f5_range`; never-wrong S1 slots are counted on their own line). Logic in hallutrace/c_exec_untyped.py; the count call and its cache in
hallutrace/infinigram.py. Tuple mode and the earlier sample rule (`c_exec_untyped.slots`) are not used here.

Outputs in --out (default results/c_exec/): untyped_phrase.csv, untyped_slots.csv,
untyped_f5.csv, untyped_summary.md, untyped_meta.json. Names, years, ids and numbers only. Only
these names are checked for --force; scripts/c_exec_tau.py's he_* files and meta.json are never touched.

Modes:
  --dry-run     no request, no lock, no file written, never creates the cache. Prints the
                targets and phrases, how many are cached (read-only; a missing cache or
                `counts` table is 0) and the query list.
  --cache-only  no network: rebuilds the outputs from the cache (read-only); refuses, listing
                the (index, phrase) pairs, when a target's walk reaches a phrase not cached.
  (neither)     live: the only network use, through InfinigramClient (throttle, retries, block
                handling, circuit breaker, --max-requests). Takes the per-user infini-gram lock.
                Rerun the same command to resume: cached answers send nothing.

Guards, before anything opens the cache for writing: a cache named infinigram.sqlite is
refused; a cache file whose windows, plan or finds table has a row (a window cache) is refused;
in live mode a cache file holding any table other than infinigram SCHEMA's, `counts` and
SQLite's own (an API cache, another project database) is refused; existing outputs are refused
without --force. Every read of the cache is read-only and creates no file beside it (no -wal,
-shm or .lock: `immutable=1` when the database has no -wal file).

Outputs are all or nothing: the five files go to temporary names in --out and replace the
outputs only once all five are written.

Exit 0 done; 1 refusal or error (message on stderr, no traceback; a stopped live run or Ctrl+C
writes no outputs); 2 bad command line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace import bio_counts as bc  # noqa: E402
from hallutrace import c_exec_tau as ct  # noqa: E402
from hallutrace import c_exec2 as c2  # noqa: E402
from hallutrace import c_exec_untyped as cu  # noqa: E402
from hallutrace import infinigram  # noqa: E402
from hallutrace import stability  # noqa: E402
from hallutrace.infinigram import (  # noqa: E402
    Blocked, Cache, HttpTransport, InfinigramClient, LockHeld, MaxRequestsReached, Throttle,
    TooManyFailures,
)
from scripts import pilot_counts as pcli  # noqa: E402
from scripts.bio_counts import git_head  # noqa: E402

DEFAULT_COUNTS = ROOT / "results" / "bio_counts" / "merged" / "counts.csv"
DEFAULT_CLAIMS = ROOT / "results" / "bio_year_claims.csv"
DEFAULT_TAU = ROOT / "results" / "c_exec" / "he_tau.csv"
DEFAULT_OUT = ROOT / "results" / "c_exec"
DEFAULT_BEXEC_ROOT = ROOT / "outputs" / "bexec_generations"   # a local copy of the verifier inputs/ and outputs_filtered/
BEXEC_OUTPUTS = "outputs_filtered"
DEFAULT_EXPECTED_TAU = ("v4_olmo-mix-1124_llama=1,v4_dclm-baseline_llama=1,v4_dolma-v1_7_llama=1,"
                        "v4_rpj_llama_s4=2,v4_piletrain_llama=3")
SCRIPT_VERSION = "2"
PHRASE_NAME = "untyped_phrase.csv"
SLOTS_NAME = "untyped_slots.csv"
F5_NAME = "untyped_f5.csv"
SUMMARY_NAME = "untyped_summary.md"
META_NAME = "untyped_meta.json"
OUTPUTS = (PHRASE_NAME, SLOTS_NAME, F5_NAME, SUMMARY_NAME, META_NAME)
WINDOW_CACHE_NAME = "infinigram.sqlite"
WINDOW_TABLES = ("windows", "plan", "finds")
# Tables a phrase cache may hold: infinigram.SCHEMA's, plus `counts` (COUNTS_SCHEMA).
CACHE_TABLES = frozenset(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", infinigram.SCHEMA)) | {"counts"}
SCOPES = ("s2s3", "all-birth")
TITLE = "# Untyped Biographies slots"
OPEN_S3 = "open: knowledge boundary or undetermined"
OPEN_S2 = "open: S2, any cell"
# F5 rows in a fixed order.
F5_ORDER = (cu.INFERENCE, cu.DATA, cu.PROCESS, cu.TOGETHER, cu.BOUNDARY, cu.UNDETERMINED)
DECISIONS = ["slot regimes S1, S2, S3, single_mention",
             "rules C (merged type), E (phrase check), F (F5 range with the open band)",
             "a sample stating the right year is right",
             "F5 over the slots with at least one wrong sample; never-wrong slots on their own line",
             "counting: near_or_cue matcher, full window, no stripped short forms",
             "types without Group B", "tau from historical_events",
             "an unknown count (a sampled pool with no match) is never 0"]
ROLE_H = "the single dominant wrong year (`stability.slot_items`; tuple mode is a check only)"

COUNTS_REQUIRED = ("role", "prompt_name", "index", "fact", "year", "status", "n_c",
                   "per_string_json") + tuple(bc.RULE)
CLAIMS_REQUIRED = ("generation_id", "model", "sample_index", "name", "kind", "stated_years",
                   "wikidata_years")
TAU_REQUIRED = ("variant", "index", "percentile", "tau")

PHRASE_COLUMNS = ["index", "model", "prompt_name", "name_string", "fact", "year", "roles", "regimes",
                  "phrase_no", "phrase", "count", "approx", "from_cache", "error", "hit", "year_present",
                  "target_status"]
SLOT_COLUMNS = ["model", "prompt_id", "name", "kind", "regime", "own_years", "dominant_wrong", "h_mode",
                "c_status", "c_max", "h_status", "h_max", "class", "type", "merged_type", "reachable",
                "f5_point", "t5_column", "h_present_on_merged", "c_zero_h_unknown", "untyped_before",
                "present", "never_wrong"]
F5_COLUMNS = ["point", "lower", "upper", "share_lower", "share_upper", "n"]

MISSING = object()   # --cache-only: the phrase is not in the cache


class Stopped(Exception):
    """A live run stopped before the end (budget, block, failures, Ctrl+C): no outputs."""


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def tau_arg(text):
    """argparse type: "INDEX=TAU,..." naming every index of MODEL_INDEX once -> {index: float}."""
    out = {}
    for part in text.split(","):
        k, sep, v = part.strip().partition("=")
        try:
            if not sep or not k:
                raise ValueError
            val = float(v)
        except ValueError:
            raise argparse.ArgumentTypeError("expected INDEX=TAU,..., got %r" % text)
        if k in out:
            raise argparse.ArgumentTypeError("index %s given twice" % k)
        out[k] = val
    want = set(cu.MODEL_INDEX.values())
    if set(out) != want:
        raise argparse.ArgumentTypeError(
            "--expected-tau must name each of %s once, got %s" % (", ".join(sorted(want)), ", ".join(sorted(out))))
    return out


MIN_GAP_DEFAULT = 1.5   # a 403 after 157 requests at 1.05 s; the API asks for 1 request per second


def min_gap_arg(text):
    """argparse type: seconds between request starts, never below infinigram.MIN_GAP_S."""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a number of seconds, got %r" % text)
    if not v >= infinigram.MIN_GAP_S:
        raise argparse.ArgumentTypeError("must be at least %s (infinigram.MIN_GAP_S), got %r"
                                         % (infinigram.MIN_GAP_S, text))
    return v


def nonneg_int(text):
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("expected an integer >= 0, got %r" % text)
    if v < 0:
        raise argparse.ArgumentTypeError("expected an integer >= 0, got %r" % text)
    return v


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--counts", type=Path, default=DEFAULT_COUNTS,
                    help="merged counts (the counting cell, roles c and h)")
    ap.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    ap.add_argument("--tau", type=Path, default=DEFAULT_TAU,
                    help="historical_events tau (variant %s, percentile %d)" % (ct.HEADLINE, bc.TAU_P))
    ap.add_argument("--expected-tau", type=tau_arg, default=tau_arg(DEFAULT_EXPECTED_TAU),
                    metavar="INDEX=TAU,...", help="refuse unless --tau gives these values")
    ap.add_argument("--cache", type=Path, default=None,
                    help="the phrase cache (required unless --dry-run); never a window cache")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scope", choices=SCOPES, default="s2s3")
    ap.add_argument("--bexec-root", type=Path, default=DEFAULT_BEXEC_ROOT,
                    help="local copy of the verifier inputs/ and %s/ (stability.bio_samples)" % BEXEC_OUTPUTS)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="no request, no lock, no file written")
    mode.add_argument("--cache-only", action="store_true", help="no network: rebuild the outputs from the cache")
    ap.add_argument("--max-requests", type=nonneg_int, default=None, help="stop after this many requests")
    ap.add_argument("--min-gap", type=min_gap_arg, default=MIN_GAP_DEFAULT,
                    help="live: seconds between request starts (default %(default)s, at least "
                         "infinigram.MIN_GAP_S): infini-gram allows 1 request per second, and network jitter "
                         "can bring two starts 1.05 s apart closer than 1 s at the server")
    ap.add_argument("--max-errors", type=nonneg_int, default=None,
                    help="live: stop once this many requests have failed for good (count returned nothing)")
    ap.add_argument("--max-consecutive-errors", type=nonneg_int, default=None,
                    help="live: stop once this many requests in a row have failed for good")
    ap.add_argument("--force", action="store_true", help="overwrite this script's outputs in --out")
    return ap


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def read_csv(path, required, what):
    header, rows = pcli.read_rows(path)   # utf-8, csv.field_size_limit raised for per_string_json
    missing = [c for c in required if c not in header]
    if missing:
        raise pcli.CliError("%s %s lacks column(s) %s" % (what, path, ", ".join(missing)))
    return rows


def check_rule(rows):
    bad = [r for r in rows if any(r[k] != v for k, v in bc.RULE.items())]
    if bad:
        r = bad[0]
        raise pcli.CliError("%d of %d count rows are not the counting cell %s (first: %s)" % (
            len(bad), len(rows), "/".join(bc.RULE.values()), "/".join(r[k] for k in bc.RULE)))
    if not rows:
        raise pcli.CliError("the counts file has no rows")


def read_tau(path, expected):
    """{index: tau float} from the headline rows at TAU_P; refuses on a missing, duplicate, empty
    or unexpected value."""
    rows = read_csv(path, TAU_REQUIRED, "tau file")
    got, used = {}, []
    for r in rows:
        if r["variant"] != ct.HEADLINE or r["percentile"] != str(bc.TAU_P):
            continue
        if r["index"] in got:
            raise pcli.CliError("%s has two %s rows at percentile %d for %s" % (path, ct.HEADLINE, bc.TAU_P, r["index"]))
        try:
            got[r["index"]] = float(r["tau"])
        except ValueError:
            raise pcli.CliError("%s: tau for %s is not a number (%r)" % (path, r["index"], r["tau"]))
        used.append({"index": r["index"], "model": r.get("model", ""), "tau": r["tau"]})
    absent = sorted(set(cu.MODEL_INDEX.values()) - set(got))
    if absent:
        raise pcli.CliError("%s has no %s row at percentile %d for %s" % (path, ct.HEADLINE, bc.TAU_P, ", ".join(absent)))
    diff = ["%s %s (expected %s)" % (i, _num(got[i]), _num(expected[i]))
            for i in sorted(expected) if got[i] != expected[i]]
    if diff:
        raise pcli.CliError("tau differs from --expected-tau: %s" % "; ".join(diff))
    tau = {i: got[i] for i in cu.MODEL_INDEX.values()}
    used = [u for u in used if u["index"] in tau]
    return tau, sorted(used, key=lambda u: list(cu.MODEL_INDEX.values()).index(u["index"]))


def load_base(args):
    """Claims (all of them), counts and tau, checked; no verifier output copy needed."""
    claims = read_csv(args.claims, CLAIMS_REQUIRED, "claims file")
    counts = read_csv(args.counts, COUNTS_REQUIRED, "counts file")
    check_rule(counts)
    tau, tau_rows = read_tau(args.tau, args.expected_tau)
    unknown_models = sorted({c["model"] for c in claims if c["wikidata_years"].strip()} - set(cu.MODEL_INDEX))
    if unknown_models:
        raise pcli.CliError("claims name model(s) with no index: %s" % ", ".join(unknown_models))
    return {"claims": claims, "counts": counts, "cidx": cu.counts_index(counts), "tau": tau, "tau_rows": tau_rows}


def load_inputs(args):
    """load_base, with the claims kept to the generations whose sidecar outcome is `ok`
    (`stability.ok_claims` on the verifier output copy at --bexec-root). A claim with no sidecar row
    means the copy and the claims file do not belong together: an error."""
    inputs = load_base(args)
    root = Path(args.bexec_root)
    if not (root / "inputs").is_dir() or not (root / BEXEC_OUTPUTS).is_dir():
        raise pcli.CliError("no verifier output copy at %s (needs inputs/ and %s/); pass --bexec-root" % (root, BEXEC_OUTPUTS))
    skipped = []
    samples = stability.bio_samples(str(root), BEXEC_OUTPUTS, skipped=skipped)
    kept, dropped = stability.ok_claims(inputs["claims"], samples)
    inputs["bexec"] = {"root": str(root.resolve()), "outputs": BEXEC_OUTPUTS, "samples": len(samples),
                       "skipped_units": skipped, "claims_read": len(inputs["claims"]),
                       "claims_kept": len(kept), "claims_dropped": dropped}
    inputs["claims"] = kept
    return inputs


# ---------------------------------------------------------------------------
# Cache guards and read-only access
# ---------------------------------------------------------------------------


def check_cache_name(path):
    if path is not None and Path(path).name.lower() == WINDOW_CACHE_NAME:
        raise pcli.CliError("refusing --cache %s: %s is the window cache; pass the phrase cache" % (path, WINDOW_CACHE_NAME))


def _tables(conn):
    return {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def open_ro(path):
    """A read-only connection that leaves no file beside `path`. A `mode=ro` connection to a
    WAL database creates `-wal` and `-shm` siblings when they are missing, so without a
    `-wal` file the database is opened `immutable=1` (nothing is created; with no WAL file
    there is no WAL content to ignore). With a `-wal` file (a writer has it open) it is
    `pcli.open_readonly` (`mode=ro`), which reads the WAL. Rows as dicts."""
    if Path(str(path) + "-wal").exists():
        return pcli.open_readonly(path)
    posix = Path(path).resolve().as_posix()
    conn = sqlite3.connect("file:%s?mode=ro&immutable=1" % quote(posix, safe="/:"), uri=True,
                           isolation_level=None)
    conn.row_factory = pcli._dict_row
    return conn


def check_not_window_cache(path):
    """Refuse an existing file whose windows, plan or finds table has a row. Read-only."""
    if path is None or not Path(path).exists():
        return
    conn = open_ro(path)
    try:
        tables = _tables(conn)
        filled = [t for t in WINDOW_TABLES
                  if t in tables and conn.execute('SELECT 1 FROM "%s" LIMIT 1' % t).fetchone() is not None]
    finally:
        conn.close()
    if filled:
        raise pcli.CliError("refusing --cache %s: its %s table(s) hold rows, so it is a window cache"
                            % (path, ", ".join(filled)))


def check_cache_tables(path):
    """Live mode: refuse an existing file holding a table outside infinigram's SCHEMA tables,
    `counts` and SQLite's own `sqlite_*` tables (an API cache, another project database), before
    Cache(...) would add SCHEMA's tables to it. Read-only."""
    if path is None or not Path(path).exists():
        return
    conn = open_ro(path)
    try:
        tables = _tables(conn)
    finally:
        conn.close()
    foreign = sorted(t for t in tables if t not in CACHE_TABLES and not t.startswith("sqlite_"))
    if foreign:
        raise pcli.CliError("refusing --cache %s: it holds table(s) %s, which are not the phrase cache's; "
                            "pass a new path or a phrase cache" % (path, ", ".join(foreign)))


class ReadOnlyCounts:
    """The counts table of an existing cache on a read-only connection; nothing when the file or
    the table is missing. Never constructs Cache, never creates a file."""

    def __init__(self, path):
        self.conn = None
        self.has_table = False
        if path is not None and Path(path).exists():
            self.conn = open_ro(path)
            self.has_table = "counts" in _tables(self.conn)

    def get(self, index, query):
        if not self.has_table:
            return None
        r = self.conn.execute("SELECT cnt, approx, api_latency_ms FROM counts WHERE index_name = ? AND query = ?",
                              (index, query)).fetchone()
        if r is None:
            return None
        return {"count": r["cnt"], "approx": None if r["approx"] is None else bool(r["approx"]),
                "api_latency_ms": r["api_latency_ms"], "from_cache": True}

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


# ---------------------------------------------------------------------------
# Slots, targets and the walk
# ---------------------------------------------------------------------------


def classify_all(slots, cidx, tau, present_by_key=None):
    out = []
    for s in slots:
        c_rows, h_rows = cu.rows_for(s, cidx)
        present = (present_by_key or {}).get(cu.slot_key(s), frozenset())
        out.append(cu.classify(s, c_rows, h_rows, tau[cu.MODEL_INDEX[s["model"]]], present=present))
    return out


def prepare(inputs, scope):
    slots = cu.from_slot_items(stability.slot_items(inputs["claims"]))
    before = classify_all(slots, inputs["cidx"], inputs["tau"])
    pt = cu.phrase_targets(before, inputs["cidx"], scope=scope)
    return {"slots": slots, "before": before, "targets": pt["targets"], "skipped": pt["skipped"]}


def tkey(t):
    return (t["index"], t["name_string"], int(t["year"]))


def target_queries(t):
    return [(t["index"], p) for p in cu.phrases(t["name_string"], t["fact"], t["year"])]


def served_pairs(t, by_key, cidx):
    """The (regime, role) pairs a target serves: its slots' unknown rows at the target's year."""
    pairs = set()
    for sk in t["slots"]:
        s = by_key[tuple(sk)]
        c_rows, h_rows = cu.rows_for(s, cidx)
        for role, years, rows in (("c", s["own_years"], c_rows), ("h", s["h_years"], h_rows)):
            for y, r in zip(years, rows):
                if int(y) == int(t["year"]) and r is not None and r["status"] == cu.UNKNOWN:
                    pairs.add((s["regime"], role))
    return pairs


def walk(targets, fetch):
    """Each target's birth phrases in PHRASES order through fetch(index, phrase), stopping at the
    first hit (count above 0 and approx not True). fetch returns a count dict, None (a failed
    request: recorded as an error, the walk goes on) or MISSING (--cache-only: the walk stops
    and the target is missing). Returns (phrase rows, {target key: status}, missing pairs)."""
    rows, status, missing = [], {}, []
    for t in targets:
        base = {"index": t["index"], "model": t["model"], "prompt_name": t["prompt_name"],
                "name_string": t["name_string"], "fact": t["fact"], "year": int(t["year"]),
                "roles": " ".join(t["roles"]), "regimes": " ".join(t["regimes"])}
        mine, hit, err, gone = [], False, False, False
        for no, (index, phrase) in enumerate(target_queries(t), 1):
            res = fetch(index, phrase)
            if res is MISSING:
                missing.append((index, phrase))
                gone = True
                break
            if res is None:
                err = True
                mine.append(dict(base, phrase_no=no, phrase=phrase, count="", approx="", from_cache=0,
                                 error=1, hit=0))
                continue
            approx = res["approx"]
            h = cu.is_hit(res["count"]) and approx is not True
            mine.append(dict(base, phrase_no=no, phrase=phrase, count=res["count"],
                             approx="" if approx is None else int(bool(approx)),
                             from_cache=int(bool(res["from_cache"])), error=0, hit=int(h)))
            if h:
                hit = True
                break
        st = "missing" if gone else ("hit" if hit else ("incomplete" if err else "no_hit"))
        status[tkey(t)] = st
        for r in mine:
            r.update(year_present=int(hit), target_status=st)
        rows.extend(mine)
    return rows, status, missing


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _num(v):
    """"" for None; an integer-valued number without a decimal; otherwise repr(float)."""
    if v is None or v == "":
        return ""
    v = float(v)
    return str(int(v)) if v.is_integer() else repr(v)


def _share4(x):
    return "" if x is None else "%.4f" % x


def _pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def _int(n):
    return "{:,}".format(n)


def _present_text(present):
    return " ".join("%s:%d" % (role, y) for role, y in sorted(present))


def slot_rows(after, present_by_key):
    out = []
    for s in after:
        out.append({
            "model": s["model"], "prompt_id": s["prompt_id"], "name": s["name"], "kind": s["kind"],
            "regime": s["regime"], "own_years": " ".join(str(y) for y in s["own_years"]),
            "dominant_wrong": s["dominant_wrong"], "h_mode": s["h_mode"], "c_status": s["c_status"],
            "c_max": _num(s["c_max"]), "h_status": s["h_status"], "h_max": _num(s["h_max"]),
            "class": s["class"], "type": s["type"], "merged_type": s["merged_type"],
            "reachable": "; ".join(sorted(s["reachable"])), "f5_point": s["f5_point"],
            "t5_column": s["t5_column"], "h_present_on_merged": int(s["h_present_on_merged"]),
            "c_zero_h_unknown": int(s["c_zero_h_unknown"]), "untyped_before": int(s["untyped_before"]),
            "present": _present_text(present_by_key.get(cu.slot_key(s), frozenset())),
            "never_wrong": int(c2.never_wrong(s)),
        })
    return out


def open_band(f5):
    n = f5["n"]
    by_regime = defaultdict(int)
    for g in f5["open_groups"]:
        by_regime[g["regime"]] += g["slots"]
    out = []
    for label, regime in ((OPEN_S3, "S3"), (OPEN_S2, "S2")):
        k = by_regime.get(regime, 0)
        out.append({"point": label, "lower": k, "upper": k, "share_lower": cu._share(k, n),
                    "share_upper": cu._share(k, n), "n": n})
    return out


def f5_rows(f5):
    rows = [dict(r, n=f5["n"]) for r in f5["rows"]] + open_band(f5)
    return [dict(r, share_lower=_share4(r["share_lower"]), share_upper=_share4(r["share_upper"])) for r in rows]


def _range(lo, hi, f=str):
    return f(lo) if lo == hi else "%s to %s" % (f(lo), f(hi))


def summary_markdown(run, args):
    t = {r["regime"]: r for r in run["table"]}
    f5 = run["f5"]
    st = run["stats"]
    lines = [TITLE, ""]
    lines += ["Scope `%s`, role h years: %s. Slots from `stability.slot_items` (a sample stating the "
              "right year is right). Rules C (merged type), E (phrase check) and F (F5 range); F5 over the "
              "slots with at least one wrong sample. The types are the tracker conditions with Group B left "
              "out; the labels are in `results/c_exec2/`." % (args.scope, ROLE_H), ""]

    lines += ["## (a) Slots", ""]
    total = t["all"]
    lines.append("Slots with a Wikidata year: %s (%s). Before C and E, %s were untyped (%s)." % (
        _int(total["all"]), ", ".join("%s %s" % (_reg(g), _int(t[g]["all"])) for g in cu.REGIMES),
        _int(total["untyped_before"]),
        ", ".join("%s %s" % (_reg(g), _int(t[g]["untyped_before"])) for g in cu.REGIMES)))
    lines += ["", "| Regime | All | Typed | Merged (C) | Phrase hit (E) | No merged type, no phrase hit | No count row |",
              "|---|---|---|---|---|---|---|"]
    for g in cu.REGIMES:
        r = t[g]
        if g not in ("S1", cu.SINGLE_MENTION):
            ph = _int(r["phrase_hit"])
        elif args.scope == "s2s3":
            ph = "not run"
        else:   # all-birth: phrase_hit is S2 / S3 only; here a hit changes no class
            ph = "%s (no class change)" % _int(st["present_by_regime"].get(g, 0))
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            _reg(g), _int(r["all"]), _int(r["typed"]), _int(r["merged"]), ph, _int(r["untyped"]),
            _int(r["no_count_row"])))
    lines.append("")
    nw = run["never_wrong"]
    lines.append("Merged (C) in total: %s. S1 to S3 slots with a count row and at least one wrong sample "
                 "(F5's slots): %s." % (_int(total["merged"]), _int(f5["n"])))
    lines.append("")
    lines += ["Never-wrong S1 slots (no wrong sample: not a hallucination, no label; counted in the regimes "
              "above, left out of T5's type columns and of F5): %s." % _int(sum(nw.values())),
              "", "| Model | S1, never wrong |", "|---|---|"]
    lines += ["| %s | %s |" % (m, _int(k)) for m, k in nw.items()]
    lines.append("")

    lines += ["## (b) Phrase check", ""]
    lines.append("Birth years only (`CHECKED_FACTS`), the plain `name` string, phrases in order, "
                 "stopping at a target's first hit. A target is one (index, name string, year).")
    lines.append("")
    lines.append("- Targets: %s (role c %s, role h %s)." % (
        _int(st["targets"]), _int(st["targets_role"].get("c", 0)), _int(st["targets_role"].get("h", 0))))
    lines += ["", "| Regime | Targets, role c | Targets, role h | Hits, role c | Hits, role h |", "|---|---|---|---|---|"]
    for g in cu.REGIMES:
        if not any(st["targets_by"].get((g, r)) for r in ("c", "h")):
            continue
        lines.append("| %s | %s | %s | %s | %s |" % (
            _reg(g), _int(st["targets_by"].get((g, "c"), 0)), _int(st["targets_by"].get((g, "h"), 0)),
            _int(st["hits_by"].get((g, "c"), 0)), _int(st["hits_by"].get((g, "h"), 0))))
    lines.append("")
    lines.append("A target serving several (regime, role) pairs is counted once in each.")
    lines.append("")
    lines += [
        "- Requests sent: %s. Answers from the cache: %s. Phrases walked: %s." % (
            _int(st["requests"]), _int(st["cached"]), _int(st["phrases"])),
        "- Errors (failed requests): %s. Incomplete targets (no hit, an error; a rerun retries them): %s." % (
            _int(st["errors"]), _int(st["incomplete"])),
        "- `approx` responses (not hits): %s." % _int(st["approx"]),
        "- Targets with a hit: %s of %s." % (_int(st["hits"]), _int(st["targets"])),
        "- Phrase-hit and untyped slots with a hit on wrong years only (every present row is role h): %s."
        % _int(len(st["wrong_only"])),
    ]
    lines += ["  - %s" % " ".join(k) for k in st["wrong_only"]]
    lines.append("- Merged slots with a wrong-year hit (h_present_on_merged, not in the line above): %s. "
                 "The merged type and point are kept." % _int(len(st["h_present_on_merged"])))
    lines += ["  - %s" % " ".join(k) for k in st["h_present_on_merged"]]
    lines.append("- Slots skipped (no name string, or two different ones): %s." % _int(len(run["skipped"])))
    for s in run["skipped"]:
        lines.append("  - skipped %s: %s" % (" ".join(s["slot"]), s["reason"]))
    for r in run["phrase_rows"]:
        if r["approx"] == 1:
            lines.append("  - approx: %s, \"%s\", count %s" % (r["index"], r["phrase"], r["count"]))
    lines.append("")

    lines += ["## (c) F5 range", "", "Over the %s S1 to S3 slots with a count row and at least one wrong sample "
              "(never-wrong S1 slots left out)." % _int(f5["n"]), "",
              "| Intervention point | Share | Slots |", "|---|---|---|"]
    rows = {r["point"]: r for r in f5["rows"]}
    for p in F5_ORDER:
        r = rows[p]
        lines.append("| %s | %s | %s |" % (p, _range(r["share_lower"], r["share_upper"], _pct),
                                         _range(r["lower"], r["upper"], _int)))
    lines.append("")
    lines.append("The %s open slots are a band of their own, not a cell: their cell depends on an `unknown` count."
                 % _int(f5["open"]))
    lines += ["", "| Open band | Slots | Share |", "|---|---|---|"]
    for r in open_band(f5):
        lines.append("| %s | %s | %s |" % (r["point"], _int(r["lower"]), _pct(r["share_lower"])))
    lines.append("")

    lines += ["## (d) Open slots by regime and reachable set", ""]
    if f5["open_groups"]:
        lines += ["| Regime | Reachable points | Slots |", "|---|---|---|"]
        for g in f5["open_groups"]:
            lines.append("| %s | %s | %s |" % (_reg(g["regime"]), "; ".join(g["reachable"]), _int(g["slots"])))
    else:
        lines.append("None.")
    lines.append("")

    p = run["provenance"]
    lines += ["## (e) Provenance", "",
              "- git head: `%s`, script version %s." % (p["git_head"], SCRIPT_VERSION)]
    for name, item in p["inputs"].items():
        lines.append("- %s: `%s`, git blob `%s`." % (name, item["path"], item["blob"]))
    b = p["bexec"]
    lines.append("- verifier output copy: `%s` (`%s`), %s samples, skipped units: %s; claims kept %s of %s (%s dropped, "
                 "`stability.ok_claims`)." % (b["root"], b["outputs"], _int(b["samples"]),
                                             ", ".join(b["skipped_units"]) or "none", _int(b["claims_kept"]),
                                             _int(b["claims_read"]), _int(b["claims_dropped"])))
    lines.append("- tau (%s, percentile %d): %s." % (
        ct.HEADLINE, bc.TAU_P, ", ".join("%s %s" % (r["model"] or r["index"], r["tau"]) for r in p["tau_rows"])))
    lines.append("- cache: `%s` (%s)." % (p["cache_path"], p["mode"]))
    lines.append("- run start %s, %s s." % (p["run_start"], p["run_seconds"]))
    lines.append("- decisions: %s." % "; ".join(p["decisions"]))
    lines.append("")
    return "\n".join(lines)


def _reg(g):
    return "`%s`" % g if g == cu.SINGLE_MENTION else g


def blob_hash(path):
    """`git hash-object` of the file, or the same sha1 computed here if git fails."""
    try:
        p = subprocess.run(["git", "--no-optional-locks", "hash-object", str(path)], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=30)
        h = p.stdout.strip()
        if p.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", h):
            return h
    except (OSError, subprocess.SubprocessError):
        pass
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _shown_path(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def dry_run(args, inputs):
    prep = prepare(inputs, args.scope)
    ro = ReadOnlyCounts(args.cache)
    try:
        qs = [q for t in prep["targets"] for q in target_queries(t)]
        cached = sum(1 for q in qs if ro.get(*q) is not None)
        roles = defaultdict(int)
        for t in prep["targets"]:
            for r in t["roles"]:
                roles[r] += 1
        pcli.say("dry run: no request, no lock, no file written (scope %s)" % args.scope)
        if args.cache is None:
            pcli.say("cache: none given (0 cached)")
        elif not Path(args.cache).exists():
            pcli.say("cache: no file at %s (nothing created, 0 cached)" % args.cache)
        else:
            pcli.say("cache: %s (read-only%s)" % (args.cache, "" if ro.has_table else ", no counts table: 0 cached"))
        pcli.say("targets %d (serving role c %d, role h %d), phrases at most %d, already cached %d, "
                 "requests at most %d" % (len(prep["targets"]), roles["c"], roles["h"], len(qs), cached,
                                          len(qs) - cached))
        if prep["skipped"]:
            pcli.say("%d slot(s) skipped (no name string or two different ones)" % len(prep["skipped"]))
        pcli.say("queries (index, phrase), sorted, %d:" % len(qs))
        for index, phrase in sorted(set(qs)):
            pcli.say("%s\t%s" % (index, phrase))
    finally:
        ro.close()
    return 0


def _write_file(path, payload):
    """One output: a str is written as text, a (columns, rows) pair as CSV."""
    if isinstance(payload, str):
        pcli.write_text(path, payload)
    else:
        pcli.write_rows(path, *payload)


def write_outputs(out, items):
    """All or nothing: every (name, payload) goes to a temporary name in `out` first
    (`.<name>.<pid>.tmp`, never one of OUTPUTS); only when all are written are they moved
    over the outputs with os.replace. Any failure or Ctrl+C while writing removes the
    temporary files and leaves the existing outputs as they were."""
    out.mkdir(parents=True, exist_ok=True)
    temps = []
    try:
        for name, payload in items:
            tmp = out / (".%s.%d.tmp" % (name, os.getpid()))
            temps.append((tmp, out / name))
            _write_file(tmp, payload)
    except BaseException:
        for tmp, _ in temps:
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
    for tmp, final in temps:
        os.replace(tmp, final)


def finish(args, inputs, prep, phrase_rows, status, requests, mode, t0, run_start):
    hit_keys = {k for k, s in status.items() if s == "hit"}
    present_by_key = {}
    for s in prep["slots"]:
        c_rows, h_rows = cu.rows_for(s, inputs["cidx"])
        p = cu.present_for(s, c_rows, h_rows, hit_keys)
        if p:
            present_by_key[cu.slot_key(s)] = p
    after = classify_all(prep["slots"], inputs["cidx"], inputs["tau"], present_by_key)
    f5 = c2.f5_range(after)
    table = cu.slot_table(after)
    never_wrong = c2.never_wrong_counts(after)

    by_key = {cu.slot_key(s): s for s in prep["before"]}
    targets_by, hits_by, roles = defaultdict(int), defaultdict(int), defaultdict(int)
    for t in prep["targets"]:
        for r in t["roles"]:
            roles[r] += 1
        for pair in served_pairs(t, by_key, inputs["cidx"]):
            targets_by[pair] += 1
            if status[tkey(t)] == "hit":
                hits_by[pair] += 1
    # Wrong years only, on phrase_hit and untyped slots; a merged slot whose
    # wrong year hit is h_present_on_merged, counted there and not here.
    wrong_only = sorted(cu.slot_key(s) for s in after if s["class"] in ("phrase_hit", "untyped")
                        and present_by_key.get(cu.slot_key(s))
                        and all(role == "h" for role, _ in present_by_key[cu.slot_key(s)]))
    present_by_regime = {g: sum(1 for s in after if s["regime"] == g and present_by_key.get(cu.slot_key(s)))
                         for g in cu.REGIMES}
    stats = {
        "targets": len(prep["targets"]), "targets_role": dict(roles), "targets_by": dict(targets_by),
        "hits_by": dict(hits_by), "requests": requests, "phrases": len(phrase_rows),
        "cached": sum(r["from_cache"] for r in phrase_rows), "errors": sum(r["error"] for r in phrase_rows),
        "approx": sum(1 for r in phrase_rows if r["approx"] == 1),
        "hits": sum(1 for s in status.values() if s == "hit"),
        "incomplete": sum(1 for s in status.values() if s == "incomplete"),
        "wrong_only": wrong_only,
        "h_present_on_merged": sorted(cu.slot_key(s) for s in after if s["h_present_on_merged"]),
        "present_by_regime": present_by_regime,
    }
    provenance = {
        "git_head": git_head(),
        "inputs": {name: {"path": _shown_path(p), "blob": blob_hash(p)}
                   for name, p in (("claims", args.claims), ("counts", args.counts), ("tau", args.tau))},
        "tau_rows": inputs["tau_rows"], "cache_path": str(Path(args.cache).resolve()), "mode": mode,
        "run_start": run_start, "run_seconds": round(time.time() - t0, 1),
        "decisions": DECISIONS + ["role h years: %s" % ROLE_H],
        "bexec": inputs["bexec"],
    }
    run = {"table": table, "f5": f5, "stats": stats, "skipped": prep["skipped"], "phrase_rows": phrase_rows,
           "provenance": provenance, "never_wrong": never_wrong}
    written_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {
        "provenance": provenance, "phrases": cu.PHRASES, "checked_facts": list(cu.CHECKED_FACTS),
        "scope": args.scope, "h_years": "dominant", "rule": bc.RULE, "never_wrong": never_wrong,
        "counts": {"requests": requests, "cached": stats["cached"], "errors": stats["errors"],
                   "approx": stats["approx"], "hits": stats["hits"], "targets": stats["targets"],
                   "incomplete": stats["incomplete"], "phrases": stats["phrases"],
                   "wrong_years_only": len(wrong_only), "h_present_on_merged": len(stats["h_present_on_merged"]),
                   "skipped": len(prep["skipped"]), "slots_with_present_by_regime": present_by_regime},
        "slot_table": table, "f5": {"n": f5["n"], "open": f5["open"], "rows": f5_rows(f5),
                                    "open_groups": f5["open_groups"]},
        "skipped": prep["skipped"], "script_version": SCRIPT_VERSION, "written_at": written_at,
    }
    summary = summary_markdown(run, args)   # everything built before any file is written
    meta_text = json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out = Path(args.out)
    write_outputs(out, [(PHRASE_NAME, (PHRASE_COLUMNS, phrase_rows)),
                        (SLOTS_NAME, (SLOT_COLUMNS, slot_rows(after, present_by_key))),
                        (F5_NAME, (F5_COLUMNS, f5_rows(f5))),
                        (SUMMARY_NAME, summary), (META_NAME, meta_text)])
    pcli.say("%s: %d targets, %d requests sent, %d answers from the cache, %d errors, %d approx, %d hits -> %s"
             % (mode, stats["targets"], requests, stats["cached"], stats["errors"], stats["approx"],
                stats["hits"], out))
    pcli.say("F5 over %d slots: %d open (S3 %d, S2 %d)" % (
        f5["n"], f5["open"], open_band(f5)[0]["lower"], open_band(f5)[1]["lower"]))
    if stats["incomplete"]:
        pcli.say("%d target(s) incomplete (a failed request, no hit); rerun the same command to retry them"
                 % stats["incomplete"], err=True)
    return 0


def cache_only(args, inputs, t0, run_start):
    prep = prepare(inputs, args.scope)
    ro = ReadOnlyCounts(args.cache)
    try:
        rows, status, missing = walk(prep["targets"], lambda i, q: ro.get(i, q) or MISSING)
    finally:
        ro.close()
    if missing:
        raise pcli.CliError("--cache-only: %d target(s) reach a phrase not in the cache:\n%s" % (
            len(missing), "\n".join("  %s\t%s" % m for m in missing)))
    return finish(args, inputs, prep, rows, status, 0, "cache-only", t0, run_start)


def live(args, inputs, t0, run_start, transport, clock, sleep, wall):
    prep = prepare(inputs, args.scope)
    try:
        cache = Cache(str(args.cache), lock=True)
    except LockHeld as e:
        raise pcli.CliError("%s (a window fetch or another fetcher is running; wait for it and rerun)" % e)
    try:
        cache.ensure_counts_table()
        throttle = Throttle(args.min_gap, clock or time.monotonic, sleep or time.sleep,
                            last_start_wall=cache.last_request_wall(), wall=wall)
        client = InfinigramClient(transport if transport is not None else HttpTransport(), throttle, cache,
                                  args.max_requests)
        errors = {"total": 0, "run": 0}

        def fetch(index, query):
            # A request that failed for good (None, after the client's own retries) counts
            # toward --max-errors / --max-consecutive-errors; any answer resets the run.
            r = client.count(index, query)
            if r is not None:
                errors["run"] = 0
                return r
            errors["total"] += 1
            errors["run"] += 1
            for limit, got, what in ((args.max_errors, errors["total"], "--max-errors"),
                                     (args.max_consecutive_errors, errors["run"], "--max-consecutive-errors")):
                if limit is not None and got >= limit:
                    raise Stopped("stopped at %s %d (%d failed requests, %d in a row); check the cache's "
                                  "errors table" % (what, limit, errors["total"], errors["run"]),
                                  client.requests_sent)
            return r

        try:
            rows, status, _ = walk(prep["targets"], fetch)
        except MaxRequestsReached:
            raise Stopped("stopped at --max-requests %d" % args.max_requests, client.requests_sent)
        except Blocked:
            raise Stopped("blocked by infini-gram (HTTP 403/429); wait before rerunning", client.requests_sent)
        except TooManyFailures:
            raise Stopped("too many consecutive failures; check the cache's errors table", client.requests_sent)
        except KeyboardInterrupt:
            raise Stopped("interrupted (Ctrl+C)", client.requests_sent)
        requests = client.requests_sent
    finally:
        cache.close()
    return finish(args, inputs, prep, rows, status, requests, "live", t0, run_start)


def run(args, transport=None, clock=None, sleep=None, wall=None):
    t0 = time.time()
    run_start = datetime.now(timezone.utc).isoformat(timespec="seconds")
    check_cache_name(args.cache)
    check_not_window_cache(args.cache)
    if not args.dry_run and not args.cache_only:
        check_cache_tables(args.cache)
    if not args.dry_run:
        out = Path(args.out)
        present = [n for n in OUTPUTS if (out / n).exists()]
        if present and not args.force:
            raise pcli.CliError("%s already has %s; pass --force to overwrite" % (out, ", ".join(present)))
    inputs = load_inputs(args)
    if args.dry_run:
        return dry_run(args, inputs)
    if args.cache_only:
        return cache_only(args, inputs, t0, run_start)
    return live(args, inputs, t0, run_start, transport, clock, sleep, wall)


def main(argv=None, transport=None, clock=None, sleep=None, wall=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cache is None and not args.dry_run:
        ap.error("--cache is required unless --dry-run")
    try:
        return run(args, transport, clock, sleep, wall)
    except Stopped as e:
        why, sent = e.args
        pcli.say("%s after %d requests; no outputs written. The cache resumes: rerun the same command." % (why, sent),
                 err=True)
        return 1
    except (pcli.CliError, sqlite3.Error, OSError, ValueError, KeyError) as e:
        pcli.say("error: %s" % e, err=True)
        return 1
    except KeyboardInterrupt:
        pcli.say("interrupted (Ctrl+C); no outputs written", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
