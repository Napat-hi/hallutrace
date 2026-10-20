"""Tests for hallutrace/he_tau.py and scripts/he_tau.py (historical_events tau).

Synthetic caches only (built from hallutrace.infinigram.SCHEMA, invented text), built in temp dirs.
Nothing touches the real cache, the repo's results/ or data/, or the network. Run:
python -m tests.test_he_tau
"""

import csv
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from hallutrace import bio_counts as bc
from hallutrace import he_tau as ct
from hallutrace import pilot_counts as pc
from hallutrace.infinigram import SCHEMA
from scripts import he_tau as cli
from scripts import pilot_counts as pcli

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r, want %r)" % (label, got, want))


NOW = "2026-10-07T00:00:00+00:00"
MARKER = "ZQXMARKER"
EM_DASH = chr(0x2014)
INDEXES = list(pc.INDEX_LABELS)
IDX0 = INDEXES[0]
MODEL0 = pc.INDEX_LABELS[IDX0]
TMP = Path(tempfile.mkdtemp())

# ---------------------------------------------------------------------------------------
# The synthetic cache. Ten names, all active, one birth year each (all on index 0):
#   short (a `stripped` string): Ann Test (MP), Bob Test (MP)      n_c 1, 2
#   ambiguous (ambiguous_name):  Cy Amb, Di Amb                    n_c 3, 4
#   plain:  Eve Plain (an INACTIVE stripped string), Fay Plain (an active folded string), Gus Plain
#           n_c 5, 6, 7; Zed Zero: a true zero; Unk Unknown: sampled pool, no match (unknown);
#           Est Estimated: sampled pool, a match (counted in est_mentions units)
# Every other index has pool 0 for every string, so every fact there is a true zero.
# ---------------------------------------------------------------------------------------
SHORT = ["Ann Test (MP)", "Bob Test (MP)"]
AMB = ["Cy Amb", "Di Amb"]
PLAIN = ["Eve Plain", "Fay Plain", "Gus Plain", "Zed Zero", "Unk Unknown", "Est Estimated"]
ALL = sorted(SHORT + AMB + PLAIN)
HEADLINE = sorted(AMB + PLAIN)
WITH_SHORT = ALL
NO_AMB = sorted(PLAIN)
BIRTH = {"Ann Test (MP)": 1850, "Bob Test (MP)": 1860, "Cy Amb": 1870, "Di Amb": 1880, "Eve Plain": 1890,
         "Fay Plain": 1900, "Gus Plain": 1910, "Zed Zero": 1920, "Unk Unknown": 1930, "Est Estimated": 1940}
DOCS = {"Ann Test (MP)": 1, "Bob Test (MP)": 2, "Cy Amb": 3, "Di Amb": 4, "Eve Plain": 5, "Fay Plain": 6,
        "Gus Plain": 7}
EXPECT_SIZES = "10,2,2"


def build_cache(path, ann_ambiguous=False):
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    ents = []   # (prompt_name, query_string, source, amb_name, amb_stripped, active)
    for n in SHORT:
        ents.append((n, n, "name", int(ann_ambiguous and n == "Ann Test (MP)"), 0, 1))
        ents.append((n, n.split(" (")[0], "stripped", 0, 1 if n.startswith("Ann") else 0, 1))
    for n in AMB:
        ents.append((n, n, "name", 1, 0, 1))
    for n in PLAIN:
        ents.append((n, n, "name", 0, 0, 1))
    ents.append(("Eve Plain", "Eve", "stripped", 0, 0, 0))             # inactive: does not make it short
    ents.append(("Fay Plain", "Fay Plaen", "folded", 0, 0, 1))         # active, not stripped: not short
    for n, q, src, an, as_, act in ents:
        con.execute("INSERT INTO entities VALUES (?,?,?,?,?,?,?)", (n, q, src, "Q1", an, as_, act))
    doc = [0]

    def find(idx, q, plain, glued=0):
        con.execute("INSERT INTO finds VALUES (?,?,?,?,?,?,?,?)",
                    (idx, q, "plain", plain, json.dumps([[0, plain]]), "[]", 1.0, NOW))
        con.execute("INSERT INTO finds VALUES (?,?,?,?,?,?,?,?)",
                    (idx, q, "glued", glued, json.dumps([[0, glued]]), "[]", 1.0, NOW))

    def window(idx, q, rank, text):
        doc[0] += 1
        spans = [["About %s " % MARKER, None], [q, "0"], [text, None]]
        meta = json.dumps({"url": "https://example.org/%s" % MARKER})
        con.execute("INSERT INTO windows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (idx, q, "plain", 0, rank, doc[0], 1000, 100, 10, json.dumps(spans), meta, 1.0, NOW))

    for idx in INDEXES:
        for n, q, src, an, as_, act in ents:
            if not act or (idx == IDX0 and src == "name" and n in BIRTH):
                continue
            find(idx, q, 0)
    for n in BIRTH:
        year = BIRTH[n]
        hit = " was born in %d near %s town." % (year, MARKER)
        miss = " was seen in the %s town square." % MARKER
        if n in DOCS:                                  # pool == k, all fetched, k matching documents
            k = DOCS[n]
            find(IDX0, n, k)
            for r in range(k):
                con.execute("INSERT INTO plan VALUES (?,?,?,?,?,?)", (IDX0, n, "plain", 0, r, "all"))
                window(IDX0, n, r, hit)
        elif n == "Zed Zero":                          # pool 0: a true zero, nothing planned
            find(IDX0, n, 0)
        else:                                          # sampled pool of 300, two windows fetched
            find(IDX0, n, 300)
            for r, text in enumerate((hit if n == "Est Estimated" else miss, miss)):
                con.execute("INSERT INTO plan VALUES (?,?,?,?,?,?)", (IDX0, n, "plain", 0, r, "sample"))
                window(IDX0, n, r, text)
    con.execute("INSERT OR REPLACE INTO run_meta VALUES (?,?)", ("script_version", "1"))
    con.commit()
    con.close()


def write_prompts(path, drop=()):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_id", "task", "source_file", "source_row", "prompt", "task_meta"])
        names = [n for n in ALL if n not in drop]
        names += [""] if len(names) % 2 else []
        for i in range(0, len(names), 2):
            meta = {"name1": names[i], "birth1": BIRTH.get(names[i]), "death1": None}
            if names[i + 1]:
                meta.update(name2=names[i + 1], birth2=BIRTH[names[i + 1]], death2=None)
            w.writerow(["h%d" % i, "historical_events", "x.csv", i, "invented prompt", json.dumps(meta)])
        w.writerow(["b1", "biographies", "x.csv", 99, "invented prompt", json.dumps({"name": "Somebody Else"})])


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def header_of(path):
    with open(path, encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def entity(name, q, source="name", amb=0, active=1):
    return {"prompt_name": name, "query_string": q, "source": source, "qid": "Q1", "ambiguous_name": amb,
            "ambiguous_stripped": 0, "active": active}


print("1. name_sets: short, ambiguous, all (from active rows)")
rows = [
    entity("Ann (MP)", "Ann (MP)"), entity("Ann (MP)", "Ann", "stripped"),
    entity("Inact (MP)", "Inact (MP)"), entity("Inact (MP)", "Inact", "stripped", active=0),
    entity("Inactstr (MP)", "Inactstr (MP)"), entity("Inactstr (MP)", "Inactstr", "stripped", active="0"),
    entity("Fold", "Fold"), entity("Fold", "Fold x", "folded"), entity("Fold", "Fold alias", "alias"),
    entity("Amb int", "Amb int", amb=1), entity("Amb str", "Amb str", amb="1"), entity("Amb bool", "Amb bool", amb=True),
    entity("No 0", "No 0", amb=0), entity("No str0", "No str0", amb="0"), entity("No empty", "No empty", amb=""),
    entity("No none", "No none", amb=None),
    entity("Amb inactive only", "Amb inactive only", amb=0), entity("Amb inactive only", "Old", amb=1, active=0),
    entity("Gone", "Gone", active=0),
]
s = ct.name_sets(rows)
check(set(s) >= {"all", "short", "ambiguous"}, "keys all, short, ambiguous")
eq(s["short"], ["Ann (MP)"], "a name with an active stripped string is short; an inactive one (0 or '0') is not")
eq(s["ambiguous"], ["Amb bool", "Amb int", "Amb str"],
   "ambiguous_name read from the rows: 1, '1', True count; 0, '0', '', None and an inactive row do not")
eq(s["all"], sorted(["Ann (MP)", "Inact (MP)", "Inactstr (MP)", "Fold", "Amb int", "Amb str", "Amb bool", "No 0",
                     "No str0", "No empty", "No none", "Amb inactive only"]),
   "all = every name with an active row, sorted; a name with only inactive rows is out")
check(all(s[k] == sorted(s[k]) for k in ("all", "short", "ambiguous")), "each list is sorted")
check("Fold" not in s["short"], "an active folded or alias string does not make a name short")
eq(ct.name_sets([entity("Both (MP)", "Both (MP)", amb=1), entity("Both (MP)", "Both", "stripped")])["short"],
   ["Both (MP)"], "the stripped row's own ambiguous flag is not read; the name row's ambiguous_name is")

print("2. set_sizes, check_sizes, variant_names")
sz = ct.set_sizes(s)
eq((sz["all"], sz["short"], sz["ambiguous"], sz["overlap"]), (12, 1, 3, 0), "sizes and overlap")
eq((sz["headline_406"], sz["with_short_444"], sz["no_ambiguous_265"]), (11, 12, 8),
   "each variant's size under its VARIANTS name")
eq(ct.check_sizes(s, {"all": 12, "short": 1, "ambiguous": 3, "overlap": 0}), [], "no problems when sizes match")
bad = ct.check_sizes(s, {"all": 12, "short": 2, "ambiguous": 3, "overlap": 0})
eq(len(bad), 1, "one line per differing key")
check(bad and bad[0].startswith("short:") and "1" in bad[0] and "2" in bad[0], "the line names the key and both sizes")
both = [entity("Both (MP)", "Both (MP)", amb=1), entity("Both (MP)", "Both", "stripped"), entity("Other", "Other")]
sb = ct.set_sizes(ct.name_sets(both))
eq(sb["overlap"], 1, "a name both short and ambiguous makes overlap 1")
check(any(p.startswith("overlap") for p in ct.check_sizes(ct.name_sets(both), {"all": 2, "short": 1, "ambiguous": 1,
                                                                             "overlap": 0})),
      "check_sizes reports a non-zero overlap")
eq(ct.EXPECTED_SIZES, {"all": 444, "short": 38, "ambiguous": 141, "overlap": 0}, "EXPECTED_SIZES on the 1 Oct cache")
eq(tuple(ct.VARIANTS), ("headline_406", "with_short_444", "no_ambiguous_265"), "VARIANTS in order")
eq(ct.HEADLINE, "headline_406", "HEADLINE")
vn = ct.variant_names(s)
eq(list(vn), list(ct.VARIANTS), "variant_names in VARIANTS order")
eq(vn["headline_406"], set(s["all"]) - {"Ann (MP)"}, "headline = all minus short")
eq(vn["with_short_444"], set(s["all"]), "with_short = all")
eq(vn["no_ambiguous_265"], set(s["all"]) - {"Ann (MP)"} - {"Amb int", "Amb str", "Amb bool"},
   "no_ambiguous = all minus short minus ambiguous")
check({"Amb int", "Amb str", "Amb bool"} <= vn["headline_406"], "headline keeps the ambiguous names")
check("Ann (MP)" not in vn["headline_406"] and "Ann (MP)" not in vn["no_ambiguous_265"], "short names are out of both")
eq(ct.TAU_P, 10, "TAU_P")
eq(ct.TAU_COLUMNS, ["variant", "index", "model", "percentile", "tau", "tau_with_est", "facts", "counted", "zero",
                    "unknown", "docs_above_0", "est_mentions", "year_conflict"], "TAU_COLUMNS")
eq(ct.COUNTS_COLUMNS, pc.CSV_COLUMNS + ["short_form", "in_headline"], "COUNTS_COLUMNS = CSV_COLUMNS + two flags")

print("3. as_csv_row")
r0 = {"a": None, "b": 5, "c": 1.5, "d": "x", "e": 0, "f": ""}
r1 = ct.as_csv_row(r0)
eq(r1, {"a": "", "b": "5", "c": "1.5", "d": "x", "e": "0", "f": ""}, "None -> '', everything else str(v)")
eq(r0["b"], 5, "the input row is not changed")
eq(ct.as_csv_row(r1), r1, "idempotent on rows already all str")


def grid_row(name, index, status, n_c, unit="docs", conflict=0):
    r = {c: "" for c in pc.CSV_COLUMNS}
    r.update(prompt_name=name, index=index, fact="birth", year=1900, year_conflict=conflict, half_width="full",
             matcher="near_or_cue", strings="no_stripped", status=status, n_c=n_c, n_c_unit=unit,
             ambiguous_name=int(name in ("A1", "A2")), shared_string=0, short_year=0)
    return r


print("4. tau_table equals bio_counts.tau_rows on the hand-filtered rows")
H_SHORT, H_AMB, H_PLAIN = ["S1", "S2"], ["A1", "A2"], ["P1", "P2", "P3", "PZ", "PU", "PE"]
H_ALL = H_SHORT + H_AMB + H_PLAIN
usets = {"all": sorted(H_ALL), "short": sorted(H_SHORT), "ambiguous": sorted(H_AMB)}
SPEC_ROWS = [("S1", "counted", 1, "docs"), ("S2", "counted", 2, "docs"), ("A1", "counted", 3, "docs"),
             ("A2", "counted", 4, "docs"), ("P1", "counted", 5, "docs"), ("P2", "counted", 6, "docs"),
             ("P3", "counted", 7, "docs"), ("PZ", "zero", 0, "docs"), ("PU", "unknown", 0.5, "est_mentions"),
             ("PE", "counted", 150.0, "est_mentions")]
cell = [grid_row(n, IDX0, st, v, u, conflict=1 if n == "P1" else 0) for n, st, v, u in SPEC_ROWS]
cell += [grid_row(n, INDEXES[1], "counted", 10 + i) for i, n in enumerate(H_ALL)]   # a second index; others empty
snapshot = [dict(r) for r in cell]
tt = ct.tau_table(cell, usets)
eq(cell, snapshot, "tau_table does not change its input rows (year_conflict stays an int)")
HAND = {"headline_406": sorted(H_AMB + H_PLAIN), "with_short_444": sorted(H_ALL),
        "no_ambiguous_265": sorted(H_PLAIN)}
want = []
for v in ct.VARIANTS:
    hand = [r for r in cell if r["prompt_name"] in HAND[v]]
    want += [dict(variant=v, **t) for t in bc.tau_rows([ct.as_csv_row(r) for r in hand])]
eq(tt, want, "every variant: the rows bio_counts.tau_rows gives on its hand-filtered cell rows, plus `variant`")
eq(len(tt), 3 * len(pc.INDEX_LABELS) * len(bc.PERCENTS), "3 variants x 5 indexes x 16 percentiles")
eq([r["variant"] for r in tt][::len(pc.INDEX_LABELS) * len(bc.PERCENTS)], list(ct.VARIANTS), "variants in order")
check(all(list(r) == ct.TAU_COLUMNS for r in tt), "keys in TAU_COLUMNS order")
check(all("tau" in r and "tau_with_est" in r for r in tt), "both tau and tau_with_est are kept")
pick = lambda v, idx, p, rows=tt: [r for r in rows if (r["variant"], r["index"], r["percentile"]) == (v, idx, p)][0]
eq(sorted(pick(v, IDX0, 10)["tau"] for v in ct.VARIANTS), [1, 3, 5], "tau at 10%: 1 (with short), 3 (headline), 5 (no ambiguous)")
eq((pick("headline_406", IDX0, 20)["tau"], pick("headline_406", IDX0, 20)["tau_with_est"]), (3, 4),
   "headline at 20%: tau over [3..7] is 3, tau_with_est over [3..7, 150.0] is 4 (they differ)")
h = pick("headline_406", IDX0, 10)
eq({k: h[k] for k in ("facts", "counted", "zero", "unknown", "docs_above_0", "est_mentions", "year_conflict")},
   {"facts": 8, "counted": 6, "zero": 1, "unknown": 1, "docs_above_0": 5, "est_mentions": 1, "year_conflict": 1},
   "headline populations; year_conflict counted from an int 1 (count_grid rows go through as_csv_row)")
eq((pick("headline_406", INDEXES[2], 10)["tau"], pick("headline_406", INDEXES[2], 10)["tau_with_est"],
    pick("headline_406", INDEXES[2], 10)["facts"]), (None, None, 0), "an index with no rows: both taus None")
csv_cell = [ct.as_csv_row(r) for r in cell]
eq(ct.tau_table(csv_cell, usets), tt, "count_grid-style rows (ints) and CSV-style rows (str) give the same table")
sub = ct.tau_table(cell, usets, percents=(10,))
eq(len(sub), 3 * len(pc.INDEX_LABELS), "percents can be passed")

print("5. unknown and zero facts stay out of tau")
no_unk = [r for r in cell if not (r["prompt_name"] == "PU" and r["index"] == IDX0)]
no_zero = [r for r in cell if not (r["prompt_name"] == "PZ" and r["index"] == IDX0)]
a, b = ct.tau_table(cell, usets), ct.tau_table(no_unk, usets)
same = lambda x, y: [(r["tau"], r["tau_with_est"], r["docs_above_0"], r["est_mentions"]) for r in x] == \
                    [(r["tau"], r["tau_with_est"], r["docs_above_0"], r["est_mentions"]) for r in y]
check(same(a, b), "dropping the unknown fact changes neither tau, nor the populations (its n_c of 0.5 is not used)")
eq((pick("headline_406", IDX0, 10, a)["unknown"], pick("headline_406", IDX0, 10, b)["unknown"]), (1, 0),
   "but the unknown count shows it")
check(all(r["tau_with_est"] != 0.5 for r in a if r["index"] == IDX0 and r["variant"] != "no_ambiguous_265"),
      "the unknown fact's n_c never becomes tau_with_est")
c = ct.tau_table(no_zero, usets)
check(same(a, c), "dropping the zero fact changes neither tau nor tau_with_est")
eq((pick("headline_406", IDX0, 10, a)["zero"], pick("headline_406", IDX0, 10, c)["zero"]), (1, 0),
   "but the zero count shows it")
only = ct.tau_table([grid_row("P1", IDX0, "zero", 0), grid_row("P2", IDX0, "unknown", 0, "est_mentions")],
                    {"all": ["P1", "P2"], "short": [], "ambiguous": []})
x = pick("headline_406", IDX0, 10, only)
eq((x["tau"], x["tau_with_est"], x["docs_above_0"], x["est_mentions"], x["zero"], x["unknown"]),
   (None, None, 0, 0, 1, 1), "only zero and unknown facts: no tau at all")

print("6. annotate")
ann = ct.annotate(cell, usets)
eq(len(ann), len(cell), "one row per cell row")
eq([r["prompt_name"] for r in ann], [r["prompt_name"] for r in cell], "order kept")
check(all(str(r["short_form"]) == ("1" if r["prompt_name"] in H_SHORT else "0") for r in ann), "short_form 1 for short names")
check(all(str(r["in_headline"]) == ("0" if r["prompt_name"] in H_SHORT else "1") for r in ann),
      "in_headline 0 for short names, 1 for the rest (ambiguous ones included)")
check(all(list(r) == ct.COUNTS_COLUMNS for r in ann), "columns COUNTS_COLUMNS")
eq(cell, snapshot, "the input rows are not changed")
check(all(str(r["ambiguous_name"]) == ("1" if r["prompt_name"] in H_AMB else "0") for r in ann),
      "ambiguous_name carried over from the cell rows")

print("7. summary_markdown")
prov = {"cache_path": "C:/x/cache.sqlite", "cache_size_bytes": 12345, "cache_sha256": "ab" * 32, "git_head": "feed123",
        "script_version": "1", "written_at": "2026-10-07T00:00:00+00:00", "run_seconds": 1.5, "cell_rows": 50}
md = ct.summary_markdown(tt, usets, prov)
check(md.startswith("# historical_events tau\n"), "headed 'historical_events tau'")
check(EM_DASH not in md, "no em dash")
check("n/a" in md, "tau None shows n/a")
check("neither" in md.lower(), "states that neither tau is chosen as the headline definition yet")
check(all(n in md for n in H_SHORT), "the short-form names are listed")
check("near_or_cue" in md and "no_stripped" in md, "bio_counts.RULE is stated")
check("feed123" in md and "C:/x/cache.sqlite" in md and ("ab" * 32) in md, "provenance values are printed")
check(all(m in md for m in pc.INDEX_LABELS.values()), "every model appears")
for v in ct.VARIANTS:
    check(v in md, "variant %s appears" % v)

print("8. the synthetic cache through count_grid: what each fact is")
d = TMP / "main"
d.mkdir()
cache, prompts = d / "cache.sqlite", d / "prompts.csv"
build_cache(cache)
write_prompts(prompts)
data = pcli.load_cache(cache, pcli.ALL_NAMES)
years = pcli.load_years(prompts)
grid = pc.count_grid(data["entities"], data["finds"], data["plan"], data["windows"], years, data["names"], data["indexes"])
cellrows = bc.select_cell(grid, bc.RULE)
eq(len(cellrows), 10 * 5, "the counting cell: 10 names x 1 fact x 5 indexes")
c0 = {r["prompt_name"]: r for r in cellrows if r["index"] == IDX0}
eq({n: (r["status"], r["n_c_unit"]) for n, r in c0.items() if n in DOCS}, {n: ("counted", "docs") for n in DOCS},
   "documents with a match: counted, docs units")
eq({n: float(c0[n]["n_c"]) for n in DOCS}, {n: float(v) for n, v in DOCS.items()}, "n_c = the planned matching documents")
eq((c0["Zed Zero"]["status"], c0["Zed Zero"]["n_c_unit"]), ("zero", "docs"), "a fully fetched pool with no match: zero")
eq(c0["Unk Unknown"]["status"], "unknown", "a sampled pool with no match: unknown")
eq((c0["Est Estimated"]["status"], c0["Est Estimated"]["n_c_unit"], float(c0["Est Estimated"]["n_c"])),
   ("counted", "est_mentions", 150.0), "a sampled pool with a match: counted in est_mentions units (1 of 2 x pool 300)")
check(all(r["status"] == "zero" for r in cellrows if r["index"] != IDX0), "every other index: zero")
sets = ct.name_sets(data["entities"])
eq(sets, {"all": ALL, "short": sorted(SHORT), "ambiguous": sorted(AMB)}, "name_sets on the cache's entities")
eq((ct.set_sizes(sets)["headline_406"], ct.set_sizes(sets)["with_short_444"], ct.set_sizes(sets)["no_ambiguous_265"]),
   (8, 10, 6), "variant sizes on the synthetic cache")
eq(ct.variant_names(sets), {"headline_406": set(HEADLINE), "with_short_444": set(WITH_SHORT),
                            "no_ambiguous_265": set(NO_AMB)}, "variant membership on the synthetic cache, by hand")
live = ct.tau_table(cellrows, sets)
wantl = []
for v, members in (("headline_406", HEADLINE), ("with_short_444", WITH_SHORT), ("no_ambiguous_265", NO_AMB)):
    wantl += [dict(variant=v, **t) for t in bc.tau_rows([ct.as_csv_row(r) for r in cellrows if r["prompt_name"] in members])]
eq(live, wantl, "tau_table on real count_grid rows equals tau_rows on the hand-filtered rows")
eq(live, ct.tau_table([ct.as_csv_row(r) for r in cellrows], sets), "ints and strs agree on real rows")
hh = pick("headline_406", IDX0, 10, live)
eq((hh["tau"], hh["tau_with_est"], hh["facts"], hh["counted"], hh["zero"], hh["unknown"], hh["docs_above_0"],
    hh["est_mentions"]), (3, 3, 8, 6, 1, 1, 5, 1), "headline at 10%: tau 3, populations")
eq(pick("with_short_444", IDX0, 10, live)["tau"], 1, "with short names: tau 1")
eq(pick("no_ambiguous_265", IDX0, 10, live)["tau"], 5, "without ambiguous names: tau 5")
eq((pick("headline_406", IDX0, 20, live)["tau"], pick("headline_406", IDX0, 20, live)["tau_with_est"]), (3, 4),
   "headline at 20%: tau 3, tau_with_est 4")
anl = ct.annotate(cellrows, sets)
check({r["prompt_name"] for r in anl if str(r["short_form"]) == "1"} == set(SHORT), "annotate: short_form on the two short names")
check({r["prompt_name"] for r in anl if str(r["in_headline"]) == "0"} == set(SHORT), "annotate: in_headline 0 on the same")

print("9. the command line end to end")
before_bytes = cache.read_bytes()
before_stat = os.stat(cache)
before_files = sorted(p.name for p in d.glob("cache.sqlite*"))
out = TMP / "out"
code, so, se = run("--cache", cache, "--prompts", prompts, "--out", out, "--expect-sizes", EXPECT_SIZES)
eq(code, 0, "exit 0 (stderr: %s)" % se.strip()[:200])
files = sorted(p.name for p in out.iterdir())
eq(files, sorted(cli.OUTPUTS), "writes the four files and nothing else")
eq(sorted(cli.OUTPUTS), sorted(["he_counts.csv", "he_tau.csv", "he_tau_summary.md", "meta.json"]), "OUTPUTS names")
eq(cli.SCRIPT_VERSION, "1", "SCRIPT_VERSION")
eq(cli.DEFAULT_OUT.parts[-2:], ("results", "slot_types"), "DEFAULT_OUT is results/slot_types")
eq(cache.read_bytes(), before_bytes, "the cache file's bytes are unchanged")
eq(os.stat(cache).st_mtime_ns, before_stat.st_mtime_ns, "the cache file's mtime is unchanged")
eq(os.stat(cache).st_size, before_stat.st_size, "the cache file's size is unchanged")
eq(sorted(p.name for p in d.glob("cache.sqlite*")), before_files, "no -wal, -shm or journal file left beside the cache")

counts = read_csv(out / "he_counts.csv")
eq(header_of(out / "he_counts.csv"), pc.CSV_COLUMNS + ["short_form", "in_headline"], "he_counts.csv columns")
eq(len(counts), 50, "he_counts.csv: the counting cell only, 50 rows")
eq({(r["matcher"], r["half_width"], r["strings"]) for r in counts}, {("near_or_cue", "full", "no_stripped")},
   "only the rule's cell")
eq({r["prompt_name"] for r in counts if r["short_form"] == "1"}, set(SHORT), "short_form 1 on the short names only")
eq({r["prompt_name"] for r in counts if r["in_headline"] == "0"}, set(SHORT), "in_headline 0 on the short names only")
eq({r["prompt_name"] for r in counts if r["ambiguous_name"] == "1"}, set(AMB), "ambiguous_name carried over")
ci = {r["prompt_name"]: r for r in counts if r["index"] == IDX0}
eq({n: (r["status"], r["n_c_unit"]) for n, r in ci.items()},
   {**{n: ("counted", "docs") for n in DOCS}, "Zed Zero": ("zero", "docs"), "Unk Unknown": ("unknown", "est_mentions"),
    "Est Estimated": ("counted", "est_mentions")}, "statuses and units per name")
eq({n: float(ci[n]["n_c"]) for n in DOCS}, {n: float(v) for n, v in DOCS.items()}, "n_c per name")

tau = read_csv(out / "he_tau.csv")
eq(header_of(out / "he_tau.csv"), ct.TAU_COLUMNS, "he_tau.csv columns")
eq(len(tau), 3 * 5 * 16, "he_tau.csv: 3 variants x 5 indexes x 16 percentiles")
eq([r["variant"] for r in tau][::80], list(ct.VARIANTS), "variants in order")
fmt = lambda v: "" if v is None else "%g" % v
expected = [{k: (fmt(r[k]) if k in ("tau", "tau_with_est") else str(r[k])) for k in ct.TAU_COLUMNS} for r in live]
eq(tau, expected, "he_tau.csv equals tau_table: tau and tau_with_est '' if None else %g, the rest str")
tp = lambda v, idx, p: [r for r in tau if (r["variant"], r["index"], r["percentile"]) == (v, idx, str(p))][0]
eq((tp("headline_406", IDX0, 10)["tau"], tp("headline_406", IDX0, 10)["tau_with_est"]), ("3", "3"), "headline 10%: 3 and 3")
eq((tp("headline_406", IDX0, 20)["tau"], tp("headline_406", IDX0, 20)["tau_with_est"]), ("3", "4"), "headline 20%: 3 and 4")
eq((tp("with_short_444", IDX0, 10)["tau"], tp("no_ambiguous_265", IDX0, 10)["tau"]), ("1", "5"), "other variants 10%: 1 and 5")
eq((tp("headline_406", INDEXES[1], 10)["tau"], tp("headline_406", INDEXES[1], 10)["tau_with_est"],
    tp("headline_406", INDEXES[1], 10)["zero"]), ("", "", "8"), "an index with only zero facts: tau empty")
eq(tp("headline_406", IDX0, 10)["model"], MODEL0, "model label")

summ = (out / "he_tau_summary.md").read_text(encoding="utf-8")
check(summ.startswith("# historical_events tau\n"), "he_tau_summary.md heading")
check(all(n in summ for n in SHORT), "the short-form names are in the summary")
check(EM_DASH not in summ, "no em dash in the summary")
check("n/a" in summ, "n/a for the models with no tau")

meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
keys = ["cache_path", "cache_size_bytes", "cache_sha256", "cache_mtime", "git_head", "script_version", "rule", "percents",
        "tau_percentile", "near", "cap_full", "set_sizes", "short_names", "variants", "headline_variant",
        "headline_definition", "cell_rows", "grid_rows", "names", "indexes", "diagnostics", "decisions", "run_seconds",
        "written_at"]
eq([k for k in keys if k not in meta], [], "meta.json has every key")
eq(meta["cache_size_bytes"], len(before_bytes), "meta: cache_size_bytes")
eq(meta["cache_sha256"], hashlib.sha256(before_bytes).hexdigest(), "meta: cache_sha256")
eq(meta["rule"], bc.RULE, "meta: rule")
eq(meta["percents"], list(range(5, 21)), "meta: percents 5..20")
eq((meta["tau_percentile"], meta["near"], meta["cap_full"], meta["script_version"], meta["headline_variant"]),
   (10, pc.NEAR, pc.CAP_FULL, "1", "headline_406"), "meta: tau_percentile, near, cap_full, script_version, headline_variant")
eq((meta["cell_rows"], meta["grid_rows"]), (50, 10 * 5 * len(pc.HALF_WIDTHS) * len(pc.MATCHERS) * 2),
   "meta: cell_rows and grid_rows")
eq(meta["short_names"], sorted(SHORT), "meta: short_names")
eq({k: meta["set_sizes"][k] for k in ("all", "short", "ambiguous", "overlap")},
   {"all": 10, "short": 2, "ambiguous": 2, "overlap": 0}, "meta: set_sizes")
eq(meta["indexes"], INDEXES, "meta: indexes")
check(meta["names"] in (10, ALL), "meta: names (a count or the list)")
check(all(v in json.dumps(meta["variants"]) for v in ct.VARIANTS), "meta: variants names the three variants")

blob = b"".join(p.read_bytes() for p in out.iterdir())
check(MARKER.encode() not in blob and MARKER.lower().encode() not in blob.lower(),
      "no output file contains the marker planted in every window and in metadata_json")
check(MARKER not in so and MARKER not in se, "nor does stdout or stderr")
check(so.strip() != "", "stdout prints the sizes and headline tau")
check(EM_DASH not in so, "no em dash on stdout")

print("10. refusals")
first = {n: (out / n).read_bytes() for n in ("he_counts.csv", "he_tau.csv", "meta.json")}
code, so2, se2 = run("--cache", cache, "--prompts", prompts, "--out", out, "--expect-sizes", EXPECT_SIZES)
eq(code, 1, "an out dir that already has outputs: exit 1 without --force")
check(se2.startswith("error:") or "error:" in se2, "message 'error: ...' on stderr")
check("Traceback" not in se2, "no traceback")
eq({n: (out / n).read_bytes() for n in first}, first, "nothing was overwritten")
partial = TMP / "partial"
partial.mkdir()
(partial / "meta.json").write_text("{}", encoding="utf-8")
code, _, se3 = run("--cache", TMP / "no_such_cache.sqlite", "--prompts", prompts, "--out", partial)
eq(code, 1, "one existing output is enough to refuse")
check("--force" in se3 or "exist" in se3, "refused for the outputs, before the cache is opened (message names the cause)")
check(not (TMP / "no_such_cache.sqlite").exists(), "a missing --cache is not created")
code, _, se4 = run("--cache", cache, "--prompts", prompts, "--out", out, "--expect-sizes", EXPECT_SIZES, "--force")
eq(code, 0, "--force runs again (stderr: %s)" % se4.strip()[:200])
eq({n: (out / n).read_bytes() for n in ("he_counts.csv", "he_tau.csv")},
   {n: first[n] for n in ("he_counts.csv", "he_tau.csv")}, "a forced re-run gives the same counts and tau files")
eq(cache.read_bytes(), before_bytes, "the cache is still unchanged")

other = TMP / "unrelated"
other.mkdir()
(other / "notes.txt").write_text("keep", encoding="utf-8")
code, _, se5 = run("--cache", cache, "--prompts", prompts, "--out", other, "--expect-sizes", EXPECT_SIZES)
eq(code, 0, "an out dir with only unrelated files is fine (stderr: %s)" % se5.strip()[:200])
check((other / "notes.txt").read_text(encoding="utf-8") == "keep", "unrelated files are left alone")
nested = TMP / "a" / "b" / "slot_types"
code, _, _ = run("--cache", cache, "--prompts", prompts, "--out", nested, "--expect-sizes", EXPECT_SIZES)
eq((code, sorted(p.name for p in nested.iterdir()) if nested.exists() else None), (0, sorted(cli.OUTPUTS)),
   "--out is created if missing")


def refused(label, *argv, mention=()):
    o = TMP / ("refuse_" + str(len(failures)) + "_" + str(abs(hash(label)) % 10 ** 6))
    code, so_, se_ = run(*argv, "--out", o)
    eq(code, 1, label + ": exit 1")
    check("error:" in se_ and "Traceback" not in se_, label + ": 'error:' on stderr, no traceback")
    check(all(m in se_ for m in mention), label + ": message mentions %s" % (list(mention),))
    check(not [p for p in (o.iterdir() if o.exists() else []) if p.name in cli.OUTPUTS], label + ": nothing written")


refused("set sizes off by one (all)", "--cache", cache, "--prompts", prompts, "--expect-sizes", "11,2,2", mention=["11", "10"])
refused("set sizes off by one (short)", "--cache", cache, "--prompts", prompts, "--expect-sizes", "10,3,2", mention=["short"])
refused("set sizes off by one (ambiguous)", "--cache", cache, "--prompts", prompts, "--expect-sizes", "10,2,3", mention=["ambiguous"])
refused("the default sizes (444,38,141) on a 10-name cache", "--cache", cache, "--prompts", prompts)
cache2 = d / "overlap.sqlite"
build_cache(cache2, ann_ambiguous=True)
refused("a short name that is also ambiguous (overlap 1)", "--cache", cache2, "--prompts", prompts, "--expect-sizes", "10,2,3",
        mention=["overlap"])
prompts2 = d / "prompts_missing.csv"
write_prompts(prompts2, drop=("Zed Zero",))
refused("a name with no year in the prompts", "--cache", cache, "--prompts", prompts2, "--expect-sizes", EXPECT_SIZES,
        mention=["Zed Zero"])
refused("a missing cache file", "--cache", d / "missing.sqlite", "--prompts", prompts, "--expect-sizes", EXPECT_SIZES)
check(not (d / "missing.sqlite").exists(), "a missing cache is not created")
eq(cache.read_bytes(), before_bytes, "the cache is unchanged after the refusals")
try:
    with redirect_stderr(io.StringIO()):
        cli.main(["--no-such-flag"])
    bad_code = None
except SystemExit as e:
    bad_code = e.code
eq(bad_code, 2, "an unknown flag: exit 2")

print()
shutil.rmtree(TMP, ignore_errors=True)
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
