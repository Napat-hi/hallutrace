"""Tests for hallutrace/bio_counts.py and scripts/bio_counts.py. No network."""

import csv
import io
import json
import sqlite3
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from hallutrace import bio_counts as bc
from hallutrace import pilot_counts as pc
from hallutrace.infinigram import SCHEMA
from scripts import bio_counts as cli

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


NOW = "2026-10-03T00:00:00+00:00"
INDEXES = list(pc.INDEX_LABELS)
SNAP = {"entities": {"Q1": {"p569": [1869], "p570": [1948]}, "Q2": {"p569": [], "p570": []}}}
OWN = {"Ada Test": "Q1", "Bo Noyear": "Q2"}


def build_cache(path, shard="1/2", names=("Ada Test", "Bo Noyear"), indexes=INDEXES):
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    meta = {"shard:biographies": shard, "task_names:biographies": json.dumps(list(names)),
            "constants": json.dumps({"CAP_FULL": 200}), "script_version": "1", "git_head": "abc"}
    for k, v in meta.items():
        con.execute("INSERT OR REPLACE INTO run_meta VALUES (?,?)", (k, v))
    for n in names:
        con.execute("INSERT INTO entities VALUES (?,?,?,?,?,?,?)", (n, n, "name", None, 0, 0, 1))
        for idx in indexes:
            cnt = 2 if (idx == INDEXES[0] and n == "Ada Test") else 0
            for form, c in (("plain", cnt), ("glued", 0)):
                con.execute("INSERT INTO finds VALUES (?,?,?,?,?,?,?,?)",
                            (idx, n, form, c, json.dumps([[0, c]]), "[]", 1.0, NOW))
    idx = INDEXES[0]
    for rank, (doc, after) in enumerate(((0, " was born in 1869 in Porbandar."),
                                         (1, " was born in 1901, a page says."))):
        con.execute("INSERT INTO plan VALUES (?,?,?,?,?,?)", (idx, "Ada Test", "plain", 0, rank, "all"))
        spans = [["About ", None], ["Ada Test", "0"], [after, None]]
        con.execute("INSERT INTO windows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (idx, "Ada Test", "plain", 0, rank, doc, 1000, 100, 10, json.dumps(spans), None, 1.0, NOW))
    con.commit()
    con.close()


def write_claims(path):
    cols = ["name", "kind", "label", "stated_years", "wikidata_years", "wikidata_match"]
    rows = [{"name": "Ada Test", "kind": "birth", "label": "hallucinated", "stated_years": "1901",
             "wikidata_years": "1869", "wikidata_match": "False"},
            {"name": "Ada Test", "kind": "birth", "label": "hallucinated", "stated_years": "1869",
             "wikidata_years": "1869", "wikidata_match": "True"},     # a verifier error: no n_h
            {"name": "Ada Test", "kind": "death", "label": "supported", "stated_years": "1948",
             "wikidata_years": "1948", "wikidata_match": "True"},
            {"name": "Bo Noyear", "kind": "death", "label": "hallucinated", "stated_years": "1950",
             "wikidata_years": "", "wikidata_match": ""},              # no Wikidata year: no n_h
            {"name": "Elsewhere", "kind": "birth", "label": "hallucinated", "stated_years": "1700",
             "wikidata_years": "", "wikidata_match": ""}]              # another shard's name
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


print("1. the years each grid counts")
eq(bc.years_correct(["Ada Test", "Bo Noyear", "Nobody"], OWN, SNAP),
   {"Ada Test": {"birth": [1869], "death": [1948]}}, "own QID's years; no year or no QID: no n_c")
d = Path(tempfile.mkdtemp())
write_claims(d / "claims.csv")
eq(bc.years_hallucinated(bc.read_claims(d / "claims.csv"), ["Ada Test", "Bo Noyear"]),
   {"Ada Test": {"birth": [1901], "death": []}},
   "wrong years of this shard's names only; a hallucinated claim with the own year is left out")
check("Bo Noyear" not in bc.years_hallucinated(bc.read_claims(d / "claims.csv"), ["Ada Test", "Bo Noyear"]),
      "an in-shard name whose claim has no Wikidata year: no role h")
eq(bc.no_reference_year(bc.read_claims(d / "claims.csv"), ["Ada Test", "Bo Noyear"]),
   {"claims": 1, "names": 1, "facts": 1}, "the left-out claim is counted; another shard's is not")

print("2. export on a shard cache")
build_cache(d / "cache.sqlite")
out = d / "out"
sd = out / "shard_1_of_2"
sd.mkdir(parents=True)
(sd / "counts.csv").write_text("stale plain export", encoding="utf-8")
buf = io.StringIO()
with redirect_stdout(buf), redirect_stderr(buf):
    code = cli.export(d / "cache.sqlite", d / "claims.csv", out, OWN, SNAP)
eq(code, 0, "exit 0 on a complete shard")
meta, rows = bc.read_shard(str(sd))
check((sd / "counts.csv.gz").exists() and not (sd / "counts.csv").exists(),
      "the grid is written gzipped, and an earlier plain counts.csv is removed")
first = (sd / "counts.csv.gz").read_bytes()
with redirect_stdout(buf), redirect_stderr(buf):
    cli.export(d / "cache.sqlite", d / "claims.csv", out, OWN, SNAP)
check((sd / "counts.csv.gz").read_bytes() == first, "a re-export gives the same bytes (no gzip timestamp)")
eq((meta["shard"], meta["n_shards"], meta["names"]), ("1/2", 2, ["Ada Test", "Bo Noyear"]), "shard and names")
check(meta["completeness"]["complete"], "complete: nothing unfetched, all five indexes")
eq(sorted({r["role"] for r in rows}), ["c", "h"], "both grids")
cell = lambda role, year: [r for r in rows if r["role"] == role and r["year"] == year and r["index"] == INDEXES[0]
                           and r["half_width"] == "full" and r["matcher"] == "bare" and r["strings"] == "all"]
eq([r["n_c"] for r in cell("c", "1869")], ["1"], "n_c: one document with the correct birth year")
eq([r["n_c"] for r in cell("h", "1901")], ["1"], "n_h: one document with the stated wrong year")
check(len({r["half_width"] for r in rows}) == len(pc.HALF_WIDTHS) and {r["matcher"] for r in rows} == set(pc.MATCHERS),
      "the whole grid is kept, the counting rule's matcher included")
check("Elsewhere" not in {r["prompt_name"] for r in rows}, "another shard's claims are not counted here")
check(not [r for r in rows if r["role"] == "h" and r["prompt_name"] == "Bo Noyear"], "no role h rows for the no-year claim")
eq(meta["role_h_no_wikidata_year"], {"claims": 1, "names": 1, "facts": 1}, "meta.json records what role h left out")
d024 = [r for r in bc.select_cell(rows) if r["index"] == INDEXES[0]]
eq(sorted((r["role"], r["year"], r["n_c"]) for r in d024 if r["fact"] == "birth"), [("c", "1869", "1"), ("h", "1901", "1")],
   "select_cell finds the rule's cell in a real export: the near year counts for n_c and n_h")

print("3. an incomplete shard")
build_cache(d / "partial.sqlite", shard="2/2", names=("Ada Test",), indexes=INDEXES[:3])
with redirect_stdout(buf), redirect_stderr(buf):
    code = cli.export(d / "partial.sqlite", d / "claims.csv", out, OWN, SNAP)
eq(code, 1, "missing indexes: exit 1, nothing to merge yet")

print("4. the merge's completeness check")
ok = [{"shard": "1/2", "n_shards": 2, "names": ["a"], "constants": "x", "fetch_script_version": "1",
       "counting_code_sha256": "h", "claims_sha256": "c", "export_version": "1", "completeness": {"complete": True}},
      {"shard": "2/2", "n_shards": 2, "names": ["b"], "constants": "x", "fetch_script_version": "1",
       "counting_code_sha256": "h", "claims_sha256": "c", "export_version": "1", "completeness": {"complete": True}}]
eq(bc.check_mergeable(ok), [], "two complete, consistent shards merge")
check(bc.check_mergeable(ok[:1]), "a missing shard blocks the merge")
bad = [dict(ok[0]), dict(ok[1], counting_code_sha256="other")]
check(any("counting_code" in p for p in bc.check_mergeable(bad)), "a different counting rule version blocks it")
dup = [dict(ok[0]), dict(ok[1], names=["a"])]
check(any("is in shard" in p for p in bc.check_mergeable(dup)), "a name in two shards blocks it")
inc = [dict(ok[0]), dict(ok[1], completeness={"complete": False})]
check(any("not complete" in p for p in bc.check_mergeable(inc)), "an incomplete shard blocks it")


def grid_row(role, name, index, n_c, unit="docs", status="counted", matcher="near_or_cue", year="1900"):
    r = {c: "" for c in pc.CSV_COLUMNS}
    r.update(role=role, prompt_name=name, index=index, fact="birth", year=year, year_conflict="0",
             half_width="full", matcher=matcher, strings="no_stripped", status=status, n_c=n_c,
             n_c_unit=unit, ambiguous_name="0", shared_string="0", short_year="0")
    return r


def write_shard(root, i, n, names, rows, gz=False):
    sd = root / ("shard_%d_of_%d" % (i, n))
    sd.mkdir(parents=True)
    meta = dict(ok[0], shard="%d/%d" % (i, n), n_shards=n, names=names,
                role_h_no_wikidata_year={"claims": 2, "names": 1, "facts": 2})
    (sd / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    cols = [bc.ROLE_COLUMN] + pc.CSV_COLUMNS
    if gz:
        bc.write_counts(sd / "counts.csv.gz", cols, rows)
        return sd
    with open(sd / "counts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return sd


def shard_rows(names, docs, matcher="near_or_cue"):
    rows = []
    for name, v in zip(names, docs):
        rows.append(grid_row("c", name, INDEXES[0], v, matcher=matcher))
        rows.append(grid_row("c", name, INDEXES[0], v + 100, matcher="bare"))   # another cell
        rows.append(grid_row("h", name, INDEXES[0], 1, year="1700", matcher=matcher))
    return rows


print("5. the merge: the rule's cell from each shard, then the second tau")
root = Path(tempfile.mkdtemp())
n1 = ["n%02d" % k for k in range(1, 11)]
n2 = ["n%02d" % k for k in range(11, 21)] + ["Big", "Zed", "Unk"]
write_shard(root, 1, 2, n1, shard_rows(n1, range(1, 11)), gz=True)   # one gzipped, one plain
write_shard(root, 2, 2, n2, shard_rows(n2[:10], range(11, 21)) + [
    grid_row("c", "Big", INDEXES[0], 500, unit="est_mentions"),
    grid_row("c", "Zed", INDEXES[0], 0, status="zero"),
    grid_row("c", "Unk", INDEXES[0], "", status="unknown")])
dirs = sorted(str(p) for p in root.glob("shard_*"))
res = bc.merge(dirs)
eq(res["problems"], [], "two complete shards with the rule's cell merge, gzipped and plain")
check({(r["matcher"], r["half_width"], r["strings"]) for r in res["rows"]} == {("near_or_cue", "full", "no_stripped")},
      "only the rule's cell is kept")
eq((res["meta"]["rows"]["c"], res["meta"]["rows"]["h"]), (23, 20), "n_c rows from both shards, and the n_h rows")
t = {(x["index"], x["percentile"]): x for x in res["tau"]}
eq([t[(INDEXES[0], p)]["tau"] for p in (5, 10, 20)], [1, 2, 4],
   "nearest rank over n_c 1..20: 5% -> 1, 10% -> 2, 20% -> 4")
eq(t[(INDEXES[0], 10)]["tau_with_est"], 3, "the sampled pool's estimate only enters the second column")
eq({k: t[(INDEXES[0], 10)][k] for k in ("facts", "docs_above_0", "est_mentions", "zero", "unknown")},
   {"facts": 23, "docs_above_0": 20, "est_mentions": 1, "zero": 1, "unknown": 1},
   "zero and unknown facts stay out of tau")
eq(sorted({x["percentile"] for x in res["tau"]}), list(range(5, 21)), "every cutoff from 5% to 20%")
eq((t[(INDEXES[1], 10)]["tau"], t[(INDEXES[1], 10)]["facts"]), (None, 0), "an index with no counts has no tau")
eq(res["meta"]["tau_10"][pc.INDEX_LABELS[INDEXES[0]]], 2, "tau 10% per model in meta")
check(res["meta"]["n_h_precision"].startswith("unmeasured"), "meta says the rule's precision on n_h is unmeasured")
eq(res["meta"]["role_h_no_wikidata_year"], {"claims": 4, "names": 2, "facts": 4}, "the left-out counts summed over shards")

print("6. the merge refuses what it can't trust")
old = Path(tempfile.mkdtemp())
write_shard(old, 1, 2, n1, shard_rows(n1, range(1, 11), matcher="cue"))
write_shard(old, 2, 2, n2[:10], shard_rows(n2[:10], range(11, 21), matcher="cue"))
probs = bc.merge(sorted(str(p) for p in old.glob("shard_*")))["problems"]
check(probs and all("near_or_cue" in p for p in probs), "exports from before the rule's matcher: refused, not read as zeros")
eq(bc.merge(sorted(str(p) for p in old.glob("shard_*")), rule=dict(bc.RULE, matcher="cue"))["problems"], [],
   "another cell can be asked for by name (a sensitivity run)")
check(bc.merge(dirs[:1])["problems"], "one shard of two: refused")

print("7. merge on the command line")
outm = root / "merged"
with redirect_stdout(buf), redirect_stderr(buf):
    code = cli.main(["merge", "--in", str(root), "--out", str(outm)])
eq(code, 0, "exit 0")
tau_csv = list(csv.DictReader(open(outm / "tau.csv", encoding="utf-8")))
eq([r["tau"] for r in tau_csv if r["index"] == INDEXES[0] and r["percentile"] == "10"], ["2"], "tau.csv")
check((outm / "counts.csv").exists() and json.loads((outm / "meta.json").read_text(encoding="utf-8"))["rule"] == bc.RULE,
      "counts.csv and meta.json with the rule")
with redirect_stdout(buf), redirect_stderr(buf):
    eq(cli.main(["merge", "--in", str(old), "--out", str(old / "merged")]), 1, "exit 1 when the shards can't merge")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
