"""Tests for hallutrace/c_exec_untyped.py (rules C, E and F on the untyped Biographies
slots). Sections 1 to 10: the pure module. Section 11: present_for; section 12: the cached
plain count in hallutrace/infinigram.py; section 13: the CLI, scripts/c_exec_untyped.py.

Synthetic claims and count rows built in memory; no network. Sections 12 and 13 write SQLite
caches, inputs, outputs and the per-user lock only in per-test temp dirs, through fake
transports and a fake clock; section 13 replaces HttpTransport with a stand-in that fails if
it is ever constructed, gives every CLI run a synthetic verifier output copy (`--bexec-root`) and points
the CLI's default `--bexec-root` at a folder that does not exist.
Expected values are derived by hand from the module's definitions (the candidate-value
rule for reachable points). Run: python -m tests.test_c_exec_untyped
"""

import json
import os
import sqlite3
import sys
import tempfile

from hallutrace import c_exec_untyped as cu
from hallutrace import infinigram
from hallutrace import stability

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r, want %r)" % (label, got, want))


def raises(fn, exc, label):
    try:
        fn()
    except exc:
        check(True, label)
        return
    except Exception as e:  # noqa: BLE001
        check(False, "%s  (raised %r)" % (label, e))
        return
    check(False, "%s  (no exception)" % label)


CLAIM_COLUMNS = ("generation_id", "model", "sample_index", "unit_id", "name", "kind", "label", "atom",
                 "stated_years", "own_qid", "wikidata_years", "wikidata_match", "namesake_qids",
                 "namesake_year_qids", "conflation_candidate")


def claim(model, pid, sample, stated, own="1869", kind="birth", name="Ada Test"):
    """One row of results/bio_year_claims.csv, all strings."""
    r = {c: "" for c in CLAIM_COLUMNS}
    right = set(stated.split()) <= set(own.split()) if own else False
    r.update(generation_id="%s|%s|%d" % (model, pid, sample), model=model, sample_index=str(sample),
             unit_id="%s-b%05d" % (model, sample), name=name, kind=kind,
             label="supported" if right else "hallucinated", atom="%s was born in %s." % (name, stated),
             stated_years=stated, own_qid="Q1", wikidata_years=own, wikidata_match=str(right),
             conflation_candidate="False")
    return r


def wrong_samples(model, pid, stated_list, own="1869", kind="birth", name="Ada Test", start=1):
    """One claim per sample, samples numbered from `start`."""
    return [claim(model, pid, start + i, s, own=own, kind=kind, name=name) for i, s in enumerate(stated_list)]


OLMO = "v4_olmo-mix-1124_llama"
DCLM = "v4_dclm-baseline_llama"


def row(role, year, status, n_c="", name="Ada Test", index=OLMO, fact="birth", q="same", unit="docs"):
    """One merged/counts.csv row (strings). `q` is the per_string_json name string ("same" = name,
    None = no `name` entry)."""
    ps = [{"q": name + " alias", "src": "alias", "pool": 10, "sel": "all", "planned": 10, "fetched": 10,
           "matched": 0}]
    if q is not None:
        ps.append({"q": name if q == "same" else q, "src": "name", "pool": 500, "sel": "sample",
                   "planned": 50, "fetched": 50, "matched": 0})
    return {"role": role, "prompt_name": name, "index": index, "fact": fact, "year": str(year),
            "status": status, "n_c": n_c, "n_c_unit": unit, "per_string_json": json.dumps(ps),
            "half_width": "full", "matcher": "near_or_cue", "strings": "no_stripped"}


def mk_slot(regime, own=(1869,), h=(1901,), model="olmo2-7b", pid="biographies-00001", name="Ada Test",
            kind="birth"):
    """A slot dict with every field `slots` returns (dominant mode)."""
    n_wrong = {"S1": 4, "S2": 5, "S3": 5, "single_mention": 1}[regime]
    return {"model": model, "prompt_id": pid, "name": name, "kind": kind, "own_years": tuple(own),
            "n_right": 1 if regime == "S1" else 0, "n_wrong": n_wrong, "n_absent": 5 - n_wrong - (regime == "S1"),
            "c": None if regime == "single_mention" else (1.0 if regime == "S2" else 0.4), "regime": regime,
            "dominant_wrong": " ".join(str(y) for y in h), "h_years": tuple(h), "h_mode": "dominant"}


print("1. constants")
eq(cu.POINTS, ("at inference", "at training data", "at training process", "at the knowledge boundary",
               "undetermined"), "POINTS, in order")
INF, TD, TP, KB, UND = cu.POINTS
eq(cu.REGIMES, ("S1", "S2", "S3", "single_mention"), "REGIMES")
eq(cu.H_YEARS_MODES, ("dominant", "tuple"), "H_YEARS_MODES")
eq(cu.CHECKED_FACTS, ("birth",), "CHECKED_FACTS: births only")
eq(cu.TOGETHER, "training data and process together", "TOGETHER")
eq(cu.OPEN, "open", "OPEN")
eq(cu.UNTYPED_LABEL, "untyped: count below sampling resolution", "UNTYPED_LABEL (T5's column name)")
eq(cu.MODEL_INDEX, {"olmo2-7b": "v4_olmo-mix-1124_llama", "dclm-7b": "v4_dclm-baseline_llama",
                    "olmo17-7b": "v4_dolma-v1_7_llama", "redpajama-7b": "v4_rpj_llama_s4",
                    "pythia-69b": "v4_piletrain_llama"}, "MODEL_INDEX, model by model")
eq(sorted(cu.MODEL_INDEX.values()), sorted(infinigram.INDEXES), "MODEL_INDEX covers infinigram.INDEXES exactly")
BH, BL, WR, FA, LT, RF = ("BOTH_IN_DATA (rho >= 0.5)", "BOTH_IN_DATA (rho < 0.5)", "WRONG_IN_DATA",
                          "FABRICATION", "LONG_TAIL", "RECALL_FAILURE")
eq(cu.CELL, {"S1": {BH: INF, BL: INF, WR: INF, FA: INF, LT: INF, RF: INF},
             "S2": {BH: TD, BL: UND, WR: TD, FA: KB, LT: TP, RF: UND},
             "S3": {BH: UND, BL: UND, WR: UND, FA: KB, LT: UND, RF: UND}},
   "CELL: the decision table for the six types")
check("single_mention" not in cu.CELL, "single_mention has no cell")

print("2. phrases and is_hit")
eq(cu.phrases("Ada Test", "birth", 1869),
   ["Ada Test (born 1869", "Ada Test (1869", "Ada Test was born in 1869", "Ada Test, born 1869"],
   "the four birth phrases, in order")
eq(cu.phrases("Ada Test", "death", 1948),
   ["Ada Test died in 1948", "Ada Test (died 1948", "Ada Test, who died in 1948"],
   "the three death phrases, in order (kept for the record)")
eq(cu.PHRASES, {"birth": ["{name} (born {year}", "{name} ({year}", "{name} was born in {year}",
                          "{name}, born {year}"],
                "death": ["{name} died in {year}", "{name} (died {year}", "{name}, who died in {year}"]},
   "PHRASES templates")
check("death" not in cu.CHECKED_FACTS, "death is never a target fact")
eq([cu.is_hit(x) for x in (0, 0.0, None, {}.get("count"))], [False] * 4, "0, None and a missing value are not hits")
eq([cu.is_hit(x) for x in (1, 7)], [True, True], "1 and above are hits")

print("3. phase2_type and merged_type")
eq(cu.phase2_type(2, 0, 2), LT, "n_c == tau, n_h = 0: LONG_TAIL")
eq(cu.phase2_type(3, 0, 2), RF, "n_c just above tau: RECALL_FAILURE")
eq(cu.phase2_type(2.1, 0, 2), RF, "an est_mentions value with a decimal just above tau: RECALL_FAILURE")
eq(cu.phase2_type(1, 0, 1), LT, "n_c = 1, n_h = 0 at tau = 1: LONG_TAIL")
eq(cu.phase2_type(3, 3, 2), BH, "rho exactly 0.5 is the high side")
eq(cu.phase2_type(4, 3, 2), BL, "rho 3/7 < 0.5")
eq(cu.phase2_type(0, 4, 2), WR, "n_h > 0, n_c = 0: WRONG_IN_DATA")
eq(cu.phase2_type(0, 0, 2), FA, "n_h = 0, n_c = 0: FABRICATION")
eq(cu.merged_type(3, 3), "BOTH_IN_DATA (rho < 0.5) or LONG_TAIL", "merged, Pile tau 3: n_c 3 <= tau")
eq(cu.merged_type(3, 2), "BOTH_IN_DATA (rho < 0.5) or RECALL_FAILURE", "merged, RedPajama tau 2: n_c 3 > tau")
eq(cu.merged_type(1, 1), "BOTH_IN_DATA (rho < 0.5) or LONG_TAIL", "merged, OLMo 2 tau 1: n_c 1 <= tau")
eq(cu.merged_type(60.2, 1), "BOTH_IN_DATA (rho < 0.5) or RECALL_FAILURE", "merged, an estimate 60.2 > tau 1")

print("4. slots, dominant mode (default)")
M = "olmo2-7b"
claims = []
# p01 S1: one right sample beats four wrong
claims += [claim(M, "p01", 1, "1869")] + wrong_samples(M, "p01", ["1901"] * 4, start=2)
# p02 S2: per year 1901 = 4 of 5 wrong samples (c 0.8); the tuple (1901 1905) x3 would give
# dominant_wrong "1901 1905" in tuple mode and a tuple c of 3/5 (S3)
claims += wrong_samples(M, "p02", ["1901 1905", "1901 1905", "1901 1905", "1901", "1902"])
# p03 S3: per year 1901, 1902, 1905 each 2 of 4 (tie: earliest year 1901), c 0.5; tuples (1905,) x2 and
# (1901 1902) x2 tie (earliest tuple (1901, 1902))
claims += wrong_samples(M, "p03", ["1905", "1905", "1901 1902", "1901 1902"])
# p04 single_mention: one wrong sample, four absent
claims += wrong_samples(M, "p04", ["1950"], start=3)
# p05 greedy ignored: a right greedy pass on an otherwise S2 slot
claims += [claim(M, "p05", 0, "1869")] + wrong_samples(M, "p05", ["1901", "1901"])
# p06 greedy only: no stated sample, dropped
claims += [claim(M, "p06", 0, "1901")]
# p07 empty wikidata_years: dropped
claims += wrong_samples(M, "p07", ["1901", "1901"], own="")
# p08 S2 from claims of a sample taken together: sample 1 states the own year in one claim and
# 1901 in another, so it is wrong (1901); sample 2 states 1901
claims += [claim(M, "p08", 1, "1869"), claim(M, "p08", 1, "1901"), claim(M, "p08", 2, "1901")]
# p09 two own years, given unsorted: own_years (1869, 1870); 1870 is right, so S1
claims += [claim(M, "p09", 1, "1870", own="1870 1869"), claim(M, "p09", 2, "1901", own="1870 1869")]
# p10 per year differs from per tuple: years 1902 x3 (c 3/5), tuple (1904,) x2 is the top tuple
claims += wrong_samples(M, "p10", ["1901 1902", "1902 1903", "1902", "1904", "1904"])
# p11 death slot of the p01 prompt: S3 (c 2/4)
claims += wrong_samples(M, "p01", ["1950", "1950", "1951", "1952"], own="1948", kind="death")
# another model, same prompt id: its own slot
claims += wrong_samples("dclm-7b", "p01", ["1901", "1901"])
claims = claims[::-1]   # input order must not matter
dom = cu.slots(claims)
by = {cu.slot_key(s): s for s in dom}
eq([cu.slot_key(s) for s in dom], sorted(cu.slot_key(s) for s in dom), "sorted by (model, prompt_id, kind)")
eq(sorted(by), sorted([(M, p, "birth") for p in ("p01", "p02", "p03", "p04", "p05", "p08", "p09", "p10")]
                      + [(M, "p01", "death"), ("dclm-7b", "p01", "birth")]),
   "greedy-only and empty-wikidata_years slots are dropped")
s = by[(M, "p01", "birth")]
eq((s["regime"], s["n_right"], s["n_wrong"], s["n_absent"]), ("S1", 1, 4, 0), "S1: one right sample beats four wrong")
eq((s["name"], s["own_years"], s["dominant_wrong"], s["h_years"], s["h_mode"]),
   ("Ada Test", (1869,), "1901", (1901,), "dominant"), "S1 slot fields")
s = by[(M, "p02", "birth")]
eq((s["regime"], s["n_wrong"]), ("S2", 5), "S2: a sample stating two wrong years counts for each (1901: 4 of 5)")
check(s["c"] is not None and abs(s["c"] - 0.8) < 1e-4, "c = 4/5, per year (got %r)" % (s["c"],))
eq((s["dominant_wrong"], s["h_years"]), ("1901", (1901,)), "dominant: the single most frequent wrong year")
s = by[(M, "p03", "birth")]
eq(s["regime"], "S3", "S3 (c 2/4 < 0.8)")
check(s["c"] is not None and abs(s["c"] - 0.5) < 1e-4, "S3 c = 0.5 (got %r)" % (s["c"],))
eq((s["dominant_wrong"], s["h_years"]), ("1901", (1901,)), "a three-way year tie goes to the earliest year")
s = by[(M, "p04", "birth")]
eq((s["regime"], s["n_wrong"], s["n_absent"]), ("single_mention", 1, 4), "exactly one wrong sample: single_mention")
check(s["c"] is None, "single_mention has no c (c only from 2 or more wrong samples)")
eq((s["dominant_wrong"], s["h_years"]), ("1950", (1950,)), "single_mention keeps its wrong year for role h")
s = by[(M, "p05", "birth")]
eq((s["regime"], s["n_right"], s["n_wrong"], s["n_absent"]), ("S2", 0, 2, 3), "the greedy pass is ignored")
s = by[(M, "p08", "birth")]
eq((s["regime"], s["n_right"], s["n_wrong"]), ("S2", 0, 2),
   "a sample's wrong state uses all its claims together (own year plus 1901 is wrong)")
s = by[(M, "p09", "birth")]
eq((s["own_years"], s["regime"]), ((1869, 1870), "S1"), "own_years: sorted tuple of int; either own year is right")
s = by[(M, "p10", "birth")]
eq((s["regime"], s["dominant_wrong"], s["h_years"]), ("S3", "1902", (1902,)),
   "dominant counts per year (1902 in 3 samples), not per tuple ((1904,) in 2)")
check(s["c"] is not None and abs(s["c"] - 0.6) < 1e-4, "c = 3/5 (got %r)" % (s["c"],))
s = by[(M, "p01", "death")]
eq((s["regime"], s["own_years"], s["h_years"]), ("S3", (1948,), (1950,)), "a death slot of its own")
eq(by[("dclm-7b", "p01", "birth")]["regime"], "S2", "another model: its own slot")
eq(cu.slots(claims, h_years="dominant"), dom, "h_years='dominant' is the default")

print("5. slots, tuple mode")
tup = cu.slots(claims, h_years="tuple")
tb = {cu.slot_key(s): s for s in tup}
eq(sorted(tb), sorted(by), "same slots in both modes")
fields = ("regime", "n_right", "n_wrong", "n_absent", "c", "own_years", "name")
eq([tuple(tb[k][f] for f in fields) for k in sorted(tb)], [tuple(by[k][f] for f in fields) for k in sorted(by)],
   "regime, n_right, n_wrong, n_absent and c are the same in both modes (c always per year)")
s = tb[(M, "p02", "birth")]
eq((s["regime"], s["dominant_wrong"], s["h_years"], s["h_mode"]), ("S2", "1901 1905", (1901, 1905), "tuple"),
   "tuple mode: h years are every year of the top tuple; regime still per year (S2, not the tuple's 3/5)")
s = tb[(M, "p03", "birth")]
eq((s["dominant_wrong"], s["h_years"]), ("1901 1902", (1901, 1902)),
   "tuple tie ((1905,) and (1901, 1902), 2 each) goes to the earliest tuple")
eq(tb[(M, "p10", "birth")]["h_years"], (1904,), "tuple mode: the top tuple (1904,), not the top year")
eq(tb[(M, "p01", "birth")]["h_years"], (1901,), "a one-year tuple gives the same h year as dominant")
raises(lambda: cu.slots(claims, h_years="year"), ValueError, "an unknown h_years mode raises ValueError")
# single_mention with one sample stating two wrong years: the one real-data slot whose class
# depends on the mode (redpajama-7b biographies-00332 birth)
sm2 = wrong_samples(M, "p12", ["1902 1901"])
s, t = cu.slots(sm2)[0], cu.slots(sm2, h_years="tuple")[0]
eq((s["regime"], s["c"], s["dominant_wrong"], s["h_years"]), ("single_mention", None, "1901", (1901,)),
   "single_mention, two wrong years, dominant: the earliest year")
eq((t["regime"], t["c"], t["dominant_wrong"], t["h_years"]), ("single_mention", None, "1901 1902", (1901, 1902)),
   "single_mention, two wrong years, tuple: both years")
# an all-right slot: empty h in both modes
ar = [claim(M, "p13", 1, "1869"), claim(M, "p13", 2, "1869")]
for mode in cu.H_YEARS_MODES:
    s = cu.slots(ar, h_years=mode)[0]
    eq((s["regime"], s["n_wrong"], s["dominant_wrong"], s["h_years"], s["c"]), ("S1", 0, "", (), None),
       "all-right slot, %s mode: S1, no h years, no c" % mode)

print("5b. from_slot_items (stability.slot_items rows)")
# section 4's claims: only p08 has a sample stating the right year and a wrong one
ad = cu.from_slot_items(stability.slot_items(claims))
ab = {cu.slot_key(s): s for s in ad}
eq([cu.slot_key(s) for s in ad], sorted(ab), "adapter: sorted by (model, prompt_id, kind)")
eq(sorted(ab), sorted(by), "adapter: the same slots as the stand-in (greedy-only p06 dropped, no regime)")
check(all(set(s) == set(by[k]) | {"low_n"} for k, s in ab.items()), "adapter: the stand-in's keys plus low_n")
same = [k for k in sorted(by) if k != (M, "p08", "birth")]
eq([{f: ab[k][f] for f in by[k]} for k in same], [by[k] for k in same],
   "adapter: every slot but p08 equals the stand-in's, field by field (dominant mode)")
s, t = ab[(M, "p08", "birth")], by[(M, "p08", "birth")]
eq((t["regime"], t["n_right"], t["n_wrong"]), ("S2", 0, 2), "stand-in p08: the right-and-wrong sample is wrong, S2")
eq((s["regime"], s["n_right"], s["n_wrong"], s["dominant_wrong"], s["h_years"]), ("S1", 1, 1, "1901", (1901,)),
   "adapter p08: that sample is right, S1")
eq(ab[(M, "p09", "birth")]["own_years"], (1869, 1870), "adapter: own_years '1870 1869' becomes the sorted int tuple")
eq((ab[(M, "p05", "birth")]["regime"], ab[(M, "p05", "birth")]["low_n"]), ("S2", 1), "adapter: low_n carried (S2 on 2)")
eq(ab[(M, "p02", "birth")]["low_n"], 0, "adapter: low_n 0 on 5 samples")
check(all(s["h_mode"] == "dominant" for s in ad), "adapter: h_mode is always dominant")
eq(cu.from_slot_items([dict(stability.slot_items(ar)[0], regime="")]), [], "adapter: a row with no regime is dropped")

print("6. counts_index and rows_for")
grid = [row("c", 1869, "counted", "5"), row("c", 1870, "zero", "0"), row("h", 1901, "unknown"),
        row("c", 1869, "unknown", index=DCLM), row("h", 1901, "counted", "2", name="Other Name")]
idx = cu.counts_index(grid)
eq(sorted(idx), sorted([("c", OLMO, "Ada Test", "birth", 1869), ("c", OLMO, "Ada Test", "birth", 1870),
                        ("h", OLMO, "Ada Test", "birth", 1901), ("c", DCLM, "Ada Test", "birth", 1869),
                        ("h", OLMO, "Other Name", "birth", 1901)]),
   "key (role, index, prompt_name, fact, int(year))")
raises(lambda: cu.counts_index(grid + [row("c", "1869", "zero", "0")]), ValueError, "a duplicate key raises ValueError")
cr, hr = cu.rows_for(mk_slot("S2", own=(1869, 1870, 1871), h=(1901, 1902)), idx)
eq([r["status"] if r else None for r in cr], ["counted", "zero", None], "c rows aligned with own_years, None if missing")
eq([r["status"] if r else None for r in hr], ["unknown", None], "h rows aligned with h_years, None if missing")
cr, hr = cu.rows_for(mk_slot("S2", model="dclm-7b"), idx)
eq(([r["status"] if r else None for r in cr], hr), (["unknown"], [None]), "rows looked up on the slot model's index")

print("7. classify and the F5 point")
TAU = 2


def cls(regime, c, h, present=frozenset(), own=None, hy=None, tau=TAU, **kw):
    """classify a slot with c rows `c` and h rows `h` given as (status, n_c) pairs (None = missing)."""
    own = own or tuple(1869 + i for i in range(len(c)))
    hy = hy if hy is not None else tuple(1901 + i for i in range(len(h)))
    slot = mk_slot(regime, own=own, h=hy, **kw)
    cr = [None if x is None else row("c", y, x[0], x[1]) for x, y in zip(c, own)]
    hr = [None if x is None else row("h", y, x[0], x[1]) for x, y in zip(h, hy)]
    return cu.classify(slot, cr, hr, tau, present=present)


def point(r, cls_, p, label, reach=None):
    eq(r["class"], cls_, label + ": class")
    eq(r["f5_point"], p if not isinstance(p, frozenset) else cu.OPEN, label + ": f5_point")
    want = p if isinstance(p, frozenset) else (frozenset() if p == "" else frozenset({p}))
    eq(frozenset(r["f5_set"]), want, label + ": f5_set")
    if reach is not None:
        eq(frozenset(r["reachable"]), reach, label + ": reachable")


r = cls("S2", [("counted", "5"), ("zero", "0")], [("counted", "10")])
point(r, "typed", TD, "typed S2, BOTH rho 10/15 >= 0.5", reach=frozenset({TD}))
eq((r["type"], r["merged_type"], r["t5_column"]), (BH, "", BH), "typed: its type is the T5 column")
eq((r["c_status"], r["h_status"], r["c_max"], r["h_max"]), ("counted zero", "counted", 5.0, 10.0),
   "statuses space-joined in row order; max of the known values")
eq((r["untyped_before"], r["h_present_on_merged"], r["c_zero_h_unknown"]), (False, False, False), "typed: no flags")
eq((r["model"], r["prompt_id"], r["regime"]), ("olmo2-7b", "biographies-00001", "S2"), "the slot's fields are merged in")
r = cls("S3", [("zero", "0")], [("zero", "0")])
point(r, "typed", KB, "typed S3 FABRICATION", reach=frozenset({KB}))
r = cls("S1", [("counted", "5")], [], hy=())
point(r, "typed", INF, "typed S1 with no wrong year: n_h taken as 0 (RECALL_FAILURE)")
eq((r["type"], r["h_max"]), (RF, None), "S1 no h rows: type from n_h 0, h_max None")
r = cls("S2", [("counted", "60.2")], [("counted", "1")], tau=1)
eq(r["type"], BL, "an est_mentions n_c 60.2 used as is: rho 1/61.2 < 0.5")

r = cls("S1", [("counted", "5")], [("unknown", "")])
point(r, "merged", INF, "merged S1")
eq((r["type"], r["merged_type"], r["t5_column"]), ("", "BOTH_IN_DATA (rho < 0.5) or RECALL_FAILURE",
                                                   "BOTH_IN_DATA (rho < 0.5) or RECALL_FAILURE"),
   "merged: the merged type is the T5 column")
eq((r["untyped_before"], r["h_present_on_merged"], r["h_max"]), (True, False, None),
   "merged is untyped before C and E; h_max None with no known h row")
r = cls("S2", [("counted", "5")], [("unknown", "")])
point(r, "merged", UND, "merged S2, X RECALL_FAILURE (5 > tau 2)")
r = cls("S2", [("counted", "2"), ("zero", "0")], [("unknown", "")])
point(r, "merged", frozenset({TP, UND}), "merged S2, X LONG_TAIL (2 <= tau 2): open", reach=frozenset({TP, UND}))
eq(r["merged_type"], "BOTH_IN_DATA (rho < 0.5) or LONG_TAIL", "merged S2 LONG_TAIL type")
r = cls("S3", [("counted", "1")], [("unknown", "")])
point(r, "merged", UND, "merged S3")
r = cls("S2", [("counted", "5")], [("unknown", "")], present=frozenset({("h", 1901)}))
point(r, "merged", UND, "h_present_on_merged: the merged type and its point stay")
eq((r["h_present_on_merged"], r["h_status"], r["merged_type"]),
   (True, "present", "BOTH_IN_DATA (rho < 0.5) or RECALL_FAILURE"), "h_present_on_merged flag, h status present")

# c {0, 1, 3, 10**6} (lo 0, tau 2), h {3}: WRONG_IN_DATA or BOTH, all undetermined in S3
r = cls("S3", [("unknown", "")], [("counted", "3")])
point(r, "untyped", UND, "S3 c unknown, a wrong year counted above 0: fixed undetermined", reach=frozenset({UND}))
eq((r["t5_column"], r["untyped_before"], r["c_max"], r["h_max"]), (cu.UNTYPED_LABEL, True, None, 3.0),
   "untyped: T5 column UNTYPED_LABEL; c_max None with no known c row")
# c {0, 1, 3, 10**6}, h {0}: FABRICATION (KB), LONG_TAIL / RECALL_FAILURE (undetermined)
r = cls("S3", [("unknown", "")], [("zero", "0")])
point(r, "untyped", frozenset({KB, UND}), "S3 c unknown, h known 0: open", reach=frozenset({KB, UND}))
# c present: lo 1, {1, 4, 10**6}; h {0}: LONG_TAIL, RECALL_FAILURE, both undetermined in S3
r = cls("S3", [("unknown", "")], [("zero", "0")], present=frozenset({("c", 1869)}))
point(r, "phrase_hit", UND, "S3 c present: fixed undetermined", reach=frozenset({UND}))
eq((r["c_status"], r["c_max"], r["t5_column"]), ("present", None, cu.UNTYPED_LABEL),
   "present is not a known value; phrase_hit keeps UNTYPED_LABEL")
# c {0, 1, 3, 10**6}, h {0, 1, 3, 10**6}: FAB (KB), WRONG (TD), LONG_TAIL (TP), RECALL (UND),
# BOTH (1, 1) rho 0.5 (TD), BOTH (10**6, 1) rho < 0.5 (UND)
r = cls("S2", [("unknown", "")], [("unknown", "")])
point(r, "untyped", frozenset({KB, TD, TP, UND}), "S2 c and h unknown: open", reach=frozenset({KB, TD, TP, UND}))
# c present {1, 4, 10**6}, h {0, 1, 3, 10**6}: LONG_TAIL (1, 0) TP, RECALL (4, 0) UND, BOTH (1, 1) TD;
# no c = 0, so no FABRICATION and no WRONG_IN_DATA
r = cls("S2", [("unknown", "")], [("unknown", "")], present=frozenset({("c", 1869)}))
point(r, "phrase_hit", frozenset({TD, TP, UND}), "S2 c present, h unknown: open",
      reach=frozenset({TD, TP, UND}))
# c {0, 1, 3, 10**6}, h present {1, 4, 10**6}: WRONG (0, h) TD, BOTH (1, 1) TD, BOTH (10**6, 1) UND;
# h = 0 ruled out, so FABRICATION, LONG_TAIL and RECALL_FAILURE are gone
r = cls("S2", [("unknown", "")], [("unknown", "")], present=frozenset({("h", 1901)}))
point(r, "phrase_hit", frozenset({TD, UND}), "S2 h present only: open, n_h = 0 types ruled out",
      reach=frozenset({TD, UND}))
eq((r["c_status"], r["h_status"], r["h_present_on_merged"]), ("unknown", "present", False),
   "h present on a non-merged slot is not h_present_on_merged")
# c [counted 1, unknown]: lo 1, {1, 1, 4, 10**6}; h {0}: LONG_TAIL (TP), RECALL (UND); no KB since lo > 0
r = cls("S2", [("counted", "1"), ("unknown", "")], [("zero", "0")])
point(r, "untyped", frozenset({TP, UND}), "S2 one c year counted 1, the other unknown: lo from the known value",
      reach=frozenset({TP, UND}))
eq((r["c_status"], r["c_max"]), ("counted unknown", 1.0), "c_max over the known rows only")
# c [counted 5, unknown]: {5, 5, 8, 10**6}; h {0}: RECALL only, undetermined
r = cls("S3", [("counted", "5"), ("unknown", "")], [("zero", "0")])
point(r, "untyped", UND, "S3 one own year counted above 0, the other unknown: fixed undetermined",
      reach=frozenset({UND}))
r = cls("S1", [("unknown", "")], [("zero", "0")])
point(r, "untyped", INF, "S1 untyped: at inference")
r = cls("S1", [("unknown", "")], [("zero", "0")], present=frozenset({("c", 1869)}))
point(r, "untyped", INF, "S1 with a phrase hit stays untyped (phrase_hit is S2 / S3 only)")
eq(r["c_status"], "present", "S1 c present status")
r = cls("single_mention", [("counted", "5")], [("unknown", "")])
point(r, "merged", "", "single_mention: no point")
r = cls("single_mention", [("unknown", "")], [("zero", "0")])
point(r, "untyped", "", "single_mention untyped: no point")
r = cls("S2", [None], [("counted", "3")])
point(r, "no_count_row", "", "a missing c row")
eq(r["untyped_before"], False, "no_count_row is not untyped before C and E")
r = cls("S3", [("counted", "5")], [None])
point(r, "no_count_row", "", "a missing h row")
# c {0}, h {0, 1, 3, 10**6}: FABRICATION (KB) and WRONG_IN_DATA (undetermined in S3)
r = cls("S3", [("zero", "0")], [("unknown", "")])
point(r, "untyped", frozenset({KB, UND}), "c zero, h unknown: untyped, open", reach=frozenset({KB, UND}))
eq((r["c_zero_h_unknown"], r["h_present_on_merged"]), (True, False), "c_zero_h_unknown flag")
r = cls("S2", [("counted", "5")], [("zero", "0")], present=frozenset({("c", 1869)}))
eq((r["class"], r["c_status"], r["type"]), ("typed", "counted", RF), "present on a counted row changes nothing")
r = cls("S2", [("zero", "0")], [("unknown", "")], present=frozenset({("h", 1901)}))
eq((r["class"], r["c_zero_h_unknown"], r["h_status"]), ("untyped", True, "present"),
   "c_zero_h_unknown stays untyped even when its h row hit")
# real unknown rows carry n_c "0.0" (est_mentions): never read as a known 0
r = cls("S3", [("unknown", "0.0")], [("zero", "0")])
point(r, "untyped", frozenset({KB, UND}), "an unknown row with n_c 0.0 is not a known 0")
eq((r["c_max"], r["type"]), (None, ""), "unknown 0.0: no c_max, not typed FABRICATION")
# a known value above 10**6 on one role: the other role's unknown count can still exceed it
r = cls("S2", [("unknown", "0.0")], [("counted", "2000000")])
point(r, "untyped", frozenset({TD, UND}), "h counted 2,000,000, c unknown: BOTH rho < 0.5 still reachable")
r = cls("S3", [("unknown", "")], [("zero", "0")], present=frozenset({("c", "1869")}))
eq(r["class"], "phrase_hit", "present given with a string year still matches")
slot = mk_slot("S2")
raises(lambda: cu.classify(slot, [], [None], 2), ValueError, "rows not aligned with own_years raise ValueError")
r = cls("S1", [("unknown", "")], [], hy=())
point(r, "untyped", INF, "all-right S1 slot with no h rows, c unknown: at inference")

print("8. f5_range and slot_table")
hand = [
    cls("S1", [("counted", "5")], [("zero", "0")], pid="A"),             # typed, inference
    cls("S1", [("counted", "5")], [("unknown", "")], pid="B"),           # merged, inference
    cls("S2", [("counted", "5")], [("counted", "10")], pid="C"),         # typed, training data
    cls("S2", [("unknown", "")], [("unknown", "")], pid="F"),            # open {KB, TD, TP, UND}
    cls("S2", [("counted", "2")], [("unknown", "")], pid="G"),           # merged, open {TP, UND}
    cls("S2", [None], [("unknown", "")], pid="J"),                       # no_count_row
    cls("S3", [("zero", "0")], [("zero", "0")], pid="D"),                # typed, knowledge boundary
    cls("S3", [("counted", "5")], [("unknown", "")], pid="E"),           # merged, undetermined
    cls("S3", [("unknown", "")], [("zero", "0")], pid="H"),              # open {KB, UND}
    cls("S3", [("unknown", "")], [("zero", "0")], pid="H2"),             # open {KB, UND}
    cls("S3", [("unknown", "")], [("zero", "0")], present=frozenset({("c", 1869)}), pid="K"),  # phrase_hit, UND
    cls("single_mention", [("counted", "5")], [("zero", "0")], pid="I"),  # typed, no point
]
f5 = cu.f5_range(hand)
eq(f5["n"], 10, "n: S1 to S3 slots with a count row (single_mention and no_count_row out)")
eq(f5["open"], 4, "open slots: F, G, H, H2")
eq([x["point"] for x in f5["rows"]], list(cu.POINTS) + [cu.TOGETHER], "rows: POINTS in order, then together")
fr = {x["point"]: x for x in f5["rows"]}
want = {INF: (2, 2), TD: (1, 2), TP: (0, 2), KB: (1, 4), UND: (2, 6), cu.TOGETHER: (1, 3)}
eq({p: (fr[p]["lower"], fr[p]["upper"]) for p in want}, want,
   "lower = fixed at the point, upper adds the open slots reaching it")
eq((fr[cu.TOGETHER]["lower"], fr[cu.TOGETHER]["upper"]), (1, 3),
   "together: F reaches both training points and counts once (1 + F + G = 3, not 4)")
eq({p: (fr[p]["share_lower"], fr[p]["share_upper"]) for p in want},
   {p: (lo / 10, up / 10) for p, (lo, up) in want.items()}, "shares: lower / n and upper / n")
og = f5["open_groups"]
eq([g["regime"] for g in og], ["S2", "S2", "S3"], "open groups sorted by regime")
HK = lambda pid: ("olmo2-7b", pid, "birth")   # noqa: E731
eq([(g["regime"], frozenset(g["reachable"]), g["slots"], g["slot_keys"]) for g in og],
   [("S2", frozenset({KB, TD, TP, UND}), 1, [HK("F")]), ("S2", frozenset({TP, UND}), 1, [HK("G")]),
    ("S3", frozenset({KB, UND}), 2, [HK("H"), HK("H2")])],
   "open groups by regime and reachable set, with slot counts and keys")
check(all(isinstance(g["reachable"], list) for g in og), "open group reachable is a list")
tab = cu.slot_table(hand)
cols = ("regime", "all", "typed", "merged", "phrase_hit", "untyped", "no_count_row", "untyped_before")
eq([tuple(t[c] for c in cols) for t in tab],
   [("S1", 2, 1, 1, 0, 0, 0, 1),
    ("S2", 4, 1, 1, 0, 1, 1, 2),
    ("S3", 5, 1, 1, 1, 2, 0, 4),
    ("single_mention", 1, 1, 0, 0, 0, 0, 0),
    ("all", 12, 4, 3, 1, 3, 1, 7)],
   "slot_table: per regime in REGIMES order, then all")

print("9. name_string and phrase_targets")
eq(cu.name_string(row("c", 1869, "unknown")), "Ada Test", "name_string: the q of the src == name entry")
eq(cu.name_string(row("c", 1869, "unknown", q="Ada T. Test")), "Ada T. Test", "name_string can differ from prompt_name")
eq(cu.name_string(row("c", 1869, "unknown", q=None)), None, "no name entry: None")
g = {}


def add(slot, c, h):
    """Rows for slot: c and h given as (status, n_c, q) per year; returns the slot. Two slots of one
    person on one model share their rows, as in merged/counts.csv (the first one given is kept)."""
    ix = cu.MODEL_INDEX[slot["model"]]
    for role, ys, spec in (("c", slot["own_years"], c), ("h", slot["h_years"], h)):
        for y, x in zip(ys, spec):
            if x is not None:
                g.setdefault((role, ix, slot["name"], slot["kind"], y),
                             row(role, y, x[0], x[1], name=slot["name"], index=ix, fact=slot["kind"],
                                 q=x[2] if len(x) > 2 else "same"))
    return slot


U = ("unknown", "")
targets_slots = [
    add(mk_slot("S2", pid="P1"), [U], [U]),                                        # Ada 1869 c, 1901 h
    add(mk_slot("S3", pid="P2", h=(1905,)), [U], [("counted", "4")]),             # Ada 1869 c again; 1905 counted
    add(mk_slot("S3", pid="P3", name="Cy Dup", own=(1950,), h=(1960,)), [U], [U]),
    add(mk_slot("S3", pid="P4", name="Cy Dup II", own=(1960,), h=(1950,)), [U + ("Cy Dup",)], [U + ("Cy Dup",)]),
    add(mk_slot("S2", pid="P5", name="Dee Death", own=(1948,), h=(1950,), kind="death"), [U], [U]),
    add(mk_slot("S1", pid="P6", name="Dee One", own=(1820,), h=(1821,)), [U], [("counted", "2")]),
    add(mk_slot("single_mention", pid="P7", name="Eve Solo", own=(1830,), h=(1831,)), [("counted", "2")], [U]),
    add(mk_slot("S2", pid="P8", name="Fay Gap", own=(1840,), h=(1841,)), [U], [None]),      # no_count_row
    add(mk_slot("S2", pid="P9", name="No Name", own=(1700,), h=(1701,)), [U + (None,)], [U + (None,)]),
    add(mk_slot("S3", pid="P10", name="Two Names", own=(1710,), h=(1711,)), [U + ("Two A",)], [U + ("Two B",)]),
    add(mk_slot("S2", pid="P11", model="dclm-7b"), [U], [("counted", "3")]),      # Ada 1869 on DCLM
    add(mk_slot("S2", pid="P12", name="Gus Merged", own=(1880,), h=(1881,)), [("counted", "9")], [U]),  # merged
    add(mk_slot("S3", pid="P13", name="Hal Typed", own=(1890,), h=(1891,)), [("zero", "0")], [("zero", "0")]),
]
cidx = cu.counts_index(list(g.values()))
classified = [cu.classify(s, *cu.rows_for(s, cidx), TAU) for s in targets_slots]
eq(classified[7]["class"], "no_count_row", "P8 has no h row")
pt = cu.phrase_targets(classified, cidx)
K = lambda pid, model="olmo2-7b": (model, pid, "birth")   # noqa: E731
got = [(t["index"], t["name_string"], int(t["year"]), list(t["roles"]), list(t["regimes"]), list(t["slots"]))
       for t in pt["targets"]]
want_s2s3 = [
    (DCLM, "Ada Test", 1869, ["c"], ["S2"], [K("P11", "dclm-7b")]),
    (OLMO, "Ada Test", 1869, ["c"], ["S2", "S3"], sorted([K("P1"), K("P2")])),
    (OLMO, "Ada Test", 1901, ["h"], ["S2"], [K("P1")]),
    (OLMO, "Cy Dup", 1950, ["c", "h"], ["S3"], sorted([K("P3"), K("P4")])),
    (OLMO, "Cy Dup", 1960, ["c", "h"], ["S3"], sorted([K("P3"), K("P4")])),
    (OLMO, "Gus Merged", 1881, ["h"], ["S2"], [K("P12")]),
]
eq(got, want_s2s3, "s2s3: unknown birth rows of S2 / S3 slots, deduplicated on (index, name string, year), "
                   "sorted; slots and roles merged")
check(all(t["fact"] == "birth" for t in pt["targets"]), "birth only (the death slot P5 is never a target)")
eq([(t["model"], t["prompt_name"]) for t in pt["targets"] if t["name_string"] == "Ada Test"],
   [("dclm-7b", "Ada Test"), ("olmo2-7b", "Ada Test"), ("olmo2-7b", "Ada Test")], "model and prompt_name kept")
eq(sorted((tuple(x["slot"]), x["reason"]) for x in pt["skipped"]),
   [(K("P10"), "different name strings: Two A | Two B"), (K("P9"), "no name string")],
   "slots with no name string or two different ones are skipped with a reason")
check(not [t for t in pt["targets"] if t["name_string"] in (None, "No Name", "Two A", "Two B")],
      "skipped slots send no target")
pa = cu.phrase_targets(classified, cidx, scope="all-birth")
got = [(t["index"], t["name_string"], int(t["year"]), list(t["roles"]), list(t["regimes"]), list(t["slots"]))
       for t in pa["targets"]]
want_all = sorted(want_s2s3 + [(OLMO, "Dee One", 1820, ["c"], ["S1"], [K("P6")]),
                               (OLMO, "Eve Solo", 1831, ["h"], ["single_mention"], [K("P7")])],
                  key=lambda t: (t[0], t[1], t[2]))
eq(got, want_all, "all-birth adds S1 and single_mention; still no death, no no_count_row, no known row")
check(not [t for t in pa["targets"] if t["name_string"] == "Fay Gap"], "a no_count_row slot is never a target")
raises(lambda: cu.phrase_targets(classified, cidx, scope="all"), ValueError, "an unknown scope raises ValueError")

print("10. slots to classify, end to end")
e2e = cu.slots(wrong_samples("pythia-69b", "q1", ["1901 1902", "1901 1902", "1903"]))
eq([(s["regime"], s["h_years"]) for s in e2e], [("S3", (1901,))], "S3 (c 2/3), dominant h year 1901 (tie, earliest)")
PILE = "v4_piletrain_llama"
eidx = cu.counts_index([row("c", 1869, "unknown", index=PILE), row("h", 1901, "zero", "0", index=PILE),
                        row("h", 1902, "counted", "7", index=PILE)])
r = cu.classify(e2e[0], *cu.rows_for(e2e[0], eidx), 3)
# h is 1901 only (zero): c {0, 1, 4, 10**6} at tau 3, h {0}: FABRICATION (KB) or LONG_TAIL / RECALL (UND)
point(r, "untyped", frozenset({KB, UND}), "dominant mode looks up the 1901 row only", reach=frozenset({KB, UND}))
e2t = cu.slots(wrong_samples("pythia-69b", "q1", ["1901 1902", "1901 1902", "1903"]), h_years="tuple")
r = cu.classify(e2t[0], *cu.rows_for(e2t[0], eidx), 3)
# tuple mode: h is 1901 and 1902, max 7 > 0: WRONG_IN_DATA or BOTH, undetermined in S3
point(r, "untyped", UND, "tuple mode also looks up 1902 (counted 7): fixed undetermined", reach=frozenset({UND}))
eq(r["h_status"], "zero counted", "h statuses in h_years order")

print("11. present_for")
# section 9's slots and rows: hit keys are (index, name string, int year)
by9 = {s["prompt_id"]: s for s in classified}
HITS = {(OLMO, "Ada Test", 1869), (OLMO, "Ada Test", 1905), (OLMO, "Cy Dup", 1960), (OLMO, "Dee Death", 1948),
        (OLMO, "Fay Gap", 1840), (OLMO, "Two A", 1710), (OLMO, "Gus Merged", 1881), (OLMO, "Hal Typed", 1890)}


def pf(pid, hits=HITS):
    s = by9[pid]
    return cu.present_for(s, *cu.rows_for(s, cidx), hits)


check(isinstance(pf("P1"), frozenset), "present_for returns a frozenset")
eq(pf("P1"), frozenset({("c", 1869)}), "P1: the unknown c row whose key hit; h 1901 did not hit")
eq(pf("P2"), frozenset({("c", 1869)}), "P2: a hit on a counted row (h 1905) is not present")
eq(pf("P3"), frozenset({("h", 1960)}), "P3: role h")
eq(pf("P4"), frozenset({("c", 1960)}), "P4: looked up by the slot's name string (Cy Dup), not prompt_name (Cy Dup II)")
eq(pf("P5"), frozenset(), "P5: a death slot never gets present (a hit key carries no fact)")
eq(pf("P8"), frozenset({("c", 1840)}), "P8: a missing h row is skipped, the c row still matches")
eq(pf("P10"), frozenset(),
   "P10: two different name strings (Two A, Two B): nothing, even though (Two A, 1710) is in the hits")
eq(pf("P9"), frozenset(), "P9: no name string at all: nothing")
# a row with no `name` entry of its own (q=None) beside a row carrying the slot's string: the slot
# string is the one phrase_targets queries, so the row still becomes present when that string hit
zslot = mk_slot("S2", pid="P14", name="Zed Prompt", own=(1720,), h=(1721,))
zc, zh = row("c", 1720, "unknown", name="Zed Prompt", q=None), row("h", 1721, "unknown", name="Zed Prompt", q="Zed")
zidx = cu.counts_index([zc, zh])
eq(cu.slot_name_strings([zc], [zh]), ["Zed"], "slot_name_strings: one string over the slot's c and h rows")
eq([t["name_string"] for t in cu.phrase_targets([cu.classify(zslot, [zc], [zh], TAU)], zidx)["targets"]],
   ["Zed", "Zed"], "phrase_targets queries the slot's string for both rows")
eq(cu.present_for(zslot, [zc], [zh], {(OLMO, "Zed", 1720)}), frozenset({("c", 1720)}),
   "a row with no name entry of its own is present when the slot's string hit")
eq(cu.present_for(zslot, [zc], [zh], {(OLMO, "Zed Prompt", 1720), (OLMO, None, 1720)}), frozenset(),
   "prompt_name and a None string never match")
eq(cu.slot_name_strings([None], [row("h", 1, "zero", "0", q="A"), row("h", 2, "unknown", q="B")]), ["A", "B"],
   "slot_name_strings: None rows skipped, rows of any status counted, sorted")
eq(pf("P11"), frozenset(), "P11: the hit is on OLMo 2's index, the slot is DCLM's")
eq(pf("P12"), frozenset({("h", 1881)}), "P12: a merged slot's unknown h row")
eq(pf("P13"), frozenset(), "P13: a zero row is not present even when its key is in the hits")
eq([pf(p, set()) for p in sorted(by9)], [frozenset()] * len(by9), "no hits: nothing present")
eq(pf("P1", [(OLMO, "Ada Test", 1869)]), frozenset({("c", 1869)}), "hit_keys may be any iterable")
raises(lambda: cu.present_for(by9["P1"], [], [None], HITS), ValueError, "rows not aligned raise ValueError")
r = cu.classify(by9["P1"], *cu.rows_for(by9["P1"], cidx), TAU, present=pf("P1"))
eq((r["class"], r["c_status"], r["h_status"]), ("phrase_hit", "present", "unknown"),
   "classify with present_for: P1 becomes phrase_hit")
r = cu.classify(by9["P12"], *cu.rows_for(by9["P12"], cidx), TAU, present=pf("P12"))
eq((r["class"], r["h_present_on_merged"]), ("merged", True), "classify with present_for: P12 h_present_on_merged")

print("12. cached plain count (infinigram.Cache counts table, InfinigramClient.count)")
# FakeTime, FakeTransport and COUNT_OK copied from tests/test_infinigram_client.py (:45, :85, :82),
# not imported, so this module stands alone. FakeTransport's default answer is COUNT_OK here.
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


COUNT_OK = {"count": 12, "approx": False, "latency": 1.0, "token_ids": [29898, 5], "tokens": ["▁(", "Ober"]}


class FakeTransport:
    """Replays scripted responses in order; an entry may be an exception to raise.

    Records every body and the fake time it arrived at. When the script runs
    out, `default` is returned.
    """

    def __init__(self, ft, script=(), default=(200, COUNT_OK), cache=None, duration=0.8):
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


class Env:
    """A per-test temp dir with the per-user lock directory inside it (LOCK_DIR_ENV set for
    the test and restored afterwards, as test_infinigram_client's Base does), and every
    Cache opened through it closed before the dir is removed."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = os.path.join(self.tmp.name, "sub", "phrase_counts.sqlite")
        self.ft = FakeTime()
        self.caches = []

    def __enter__(self):
        self.old = os.environ.get(infinigram.LOCK_DIR_ENV)
        os.environ[infinigram.LOCK_DIR_ENV] = os.path.join(self.tmp.name, "userlock")
        return self

    def __exit__(self, *exc):
        try:
            for c in self.caches:
                c.close()
        finally:
            if self.old is None:
                os.environ.pop(infinigram.LOCK_DIR_ENV, None)
            else:
                os.environ[infinigram.LOCK_DIR_ENV] = self.old
            self.tmp.cleanup()
        return False

    def cache(self, ensure=True):
        c = infinigram.Cache(self.path, lock=True)
        self.caches.append(c)
        if ensure:
            c.ensure_counts_table()
        return c

    def client(self, cache, script=(), default=(200, COUNT_OK), max_requests=None):
        throttle = infinigram.Throttle(infinigram.MIN_GAP_S, self.ft.mono, self.ft.sleep,
                                       last_start_wall=cache.last_request_wall(), wall=self.ft.wall)
        tr = FakeTransport(self.ft, script, default, cache=cache)
        return infinigram.InfinigramClient(tr, throttle, cache, max_requests=max_requests), tr


def run_case(label, fn):
    """Run one case in its own Env; an unexpected exception is a failure, not a crash, so the
    environment variable is restored before the next case."""
    try:
        with Env() as env:
            fn(env)
    except Exception as e:  # noqa: BLE001
        check(False, "%s  (raised %r)" % (label, e))


def tables(cache):
    return {r[0] for r in cache.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def n_rows(cache, table):
    return cache.conn.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]


Q = "Ada Test (born 1869"
BODY = {"index": OLMO, "query_type": "count", "query": Q}


def case_table(env):
    check("counts" not in infinigram.SCHEMA, "SCHEMA does not mention the counts table")
    check("DEFAULT" not in infinigram.COUNTS_SCHEMA.upper(), "COUNTS_SCHEMA has no column DEFAULT")
    c = env.cache(ensure=False)
    check("counts" not in tables(c) and "finds" in tables(c), "a fresh Cache has SCHEMA's tables but no counts table")
    raises(lambda: c.get_count(OLMO, Q), sqlite3.OperationalError, "get_count before ensure_counts_table raises")
    c.ensure_counts_table()
    c.ensure_counts_table()
    check("counts" in tables(c), "ensure_counts_table creates it (twice is fine)")
    info = c.conn.execute("PRAGMA table_info(counts)").fetchall()
    eq([(r["name"], r["type"], r["notnull"], r["pk"]) for r in info],
       [("index_name", "TEXT", 1, 1), ("query", "TEXT", 1, 2), ("cnt", "INTEGER", 1, 0),
        ("approx", "INTEGER", 0, 0), ("api_latency_ms", "REAL", 0, 0), ("fetched_at", "TEXT", 1, 0)],
       "counts columns, NOT NULL and PRIMARY KEY (index_name, query)")


def case_count_and_cache(env):
    c = env.cache()
    cl, tr = env.client(c)
    got = cl.count(OLMO, Q)
    eq(tr.bodies, [BODY], "one request, the exact count body")
    eq(got, {"count": 12, "approx": False, "api_latency_ms": 1.0, "from_cache": False}, "count result")
    eq(c.get_count(OLMO, Q), {"count": 12, "approx": False, "api_latency_ms": 1.0}, "cached by (index, query)")
    check(bool(c.conn.execute("SELECT fetched_at FROM counts").fetchone()[0]), "fetched_at stamped")
    eq((n_rows(c, "glued_ids"), n_rows(c, "finds")), (0, 0), "count writes no glued_ids or finds row")
    got = cl.count(OLMO, Q)
    eq((len(tr.bodies), cl.requests_sent), (1, 1), "a second call sends nothing")
    eq(got, {"count": 12, "approx": False, "api_latency_ms": 1.0, "from_cache": True}, "second call from_cache True")
    cl.count(DCLM, Q)
    eq(tr.bodies[-1], {"index": DCLM, "query_type": "count", "query": Q}, "another index is another request")
    eq(n_rows(c, "counts"), 2, "two rows, one per index")
    cl.tokenize("(Ada Test")
    eq((n_rows(c, "counts"), n_rows(c, "glued_ids")), (2, 1), "tokenize still caches token ids only, not counts")
    c.close()
    c2 = env.cache()
    cl2, tr2 = env.client(c2)
    got = cl2.count(OLMO, Q)
    eq((got["from_cache"], got["count"], tr2.bodies), (True, 12, []), "a reopened cache (a new process) answers it")


def case_approx(env):
    c = env.cache()
    cl, tr = env.client(c, script=[(200, dict(COUNT_OK, approx=True)), (200, {"count": 5, "latency": 2.0}),
                                   (200, COUNT_OK)])
    got = [cl.count(OLMO, q)["approx"] for q in ("q1", "q2", "q3")]
    eq(got, [True, None, False], "approx True, missing -> None, False")
    raw = dict(c.conn.execute("SELECT query, approx FROM counts").fetchall())
    eq(raw, {"q1": 1, "q2": None, "q3": 0}, "approx stored as 1, NULL, 0")
    eq([c.get_count(OLMO, q)["approx"] for q in ("q1", "q2", "q3")], [True, None, False], "approx round-trips")
    eq(cl.count(OLMO, "q2"), {"count": 5, "approx": None, "api_latency_ms": 2.0, "from_cache": True},
       "a missing approx comes back None from the cache")


def case_failures(env):
    c = env.cache()
    cl, tr = env.client(c, script=[(400, {"error": "bad query"})])
    eq(cl.count(OLMO, Q), None, "HTTP 400: None")
    eq((len(tr.bodies), c.get_count(OLMO, Q), n_rows(c, "counts")), (1, None, 0), "400: no retry, nothing cached")
    err = c.last_error()
    eq((err["query_type"], err["http_status"]), ("count", 400), "400 logged in errors as a count")
    cl, tr = env.client(c, script=[(500, "boom")] * infinigram.MAX_ATTEMPTS)
    eq(cl.count(OLMO, Q), None, "5xx until MAX_ATTEMPTS: None")
    eq((cl.requests_sent, n_rows(c, "counts")), (infinigram.MAX_ATTEMPTS, 0), "5xx: MAX_ATTEMPTS sent, nothing cached")
    got = cl.count(OLMO, Q)
    eq((got["count"], got["from_cache"], n_rows(c, "counts")), (12, False, 1), "a failed count is retried next time")
    for i, bad in enumerate(({"error": "x"}, {"count": None}, {"count": True}, {"count": -1}, {"count": "3"})):
        cl, tr = env.client(c, script=[(200, bad)])
        eq((cl.count(OLMO, "bad%d" % i), len(tr.bodies)), (None, 1), "bad 200 body %r: None, no retry" % (bad,))
    eq(n_rows(c, "counts"), 1, "no bad body was cached")
    eq(cl.consecutive_failures, 1, "a bad body feeds the circuit breaker")
    k = infinigram.MAX_CONSECUTIVE_FAILURES
    cl, tr = env.client(c, script=[(200, {"count": None})] * k)
    for i in range(k - 1):
        cl.count(OLMO, "null%d" % i)
    eq(cl.consecutive_failures, k - 1, "bad bodies in a row accumulate (the 200 does not reset the breaker)")
    raises(lambda: cl.count(OLMO, "null_last"), infinigram.TooManyFailures,
           "MAX_CONSECUTIVE_FAILURES bad count bodies in a row raise TooManyFailures")
    eq((len(tr.bodies), n_rows(c, "counts")), (k, 1), "every bad body sent once, none cached")


def case_max_requests(env):
    c = env.cache()
    cl, tr = env.client(c, max_requests=1)
    cl.count(OLMO, Q)
    raises(lambda: cl.count(OLMO, "other"), infinigram.MaxRequestsReached, "max_requests raises MaxRequestsReached")
    eq((len(tr.bodies), c.get_count(OLMO, "other")), (1, None), "nothing sent past the budget")
    eq(cl.count(OLMO, Q)["from_cache"], True, "a cached count still answers after the budget is spent")


def case_throttle(env):
    c = env.cache()
    cl, tr = env.client(c)
    cl.count(OLMO, "a")
    cl.count(OLMO, "b")
    eq(len(tr.starts), 2, "two count requests")
    check(tr.starts[1] - tr.starts[0] >= infinigram.MIN_GAP_S - EPS,
          "starts at least MIN_GAP_S apart (%r)" % (tr.starts[1] - tr.starts[0],))
    check(tr.persisted_at_send[0] is not None, "the start time is persisted before the first count is sent")
    eq(len(env.ft.sleeps) > 0, True, "the second request waited")


old_lock_env = os.environ.get(infinigram.LOCK_DIR_ENV)
for label, fn in (("table", case_table), ("count and cache", case_count_and_cache), ("approx", case_approx),
                  ("failures", case_failures), ("max_requests", case_max_requests), ("throttle", case_throttle)):
    run_case(label, fn)
eq(os.environ.get(infinigram.LOCK_DIR_ENV), old_lock_env, "LOCK_DIR_ENV restored after the cases")

print("13. the CLI (scripts/c_exec_untyped.py), fake transports only")
import copy  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import shutil  # noqa: E402
from collections import defaultdict  # noqa: E402
from contextlib import redirect_stderr, redirect_stdout  # noqa: E402
from pathlib import Path  # noqa: E402

from hallutrace import bio_counts as bc  # noqa: E402
from scripts import c_exec_nh_sensitivity as nh  # noqa: E402
from scripts import c_exec_untyped as cli  # noqa: E402


class NoNetwork:
    """Stands in for HttpTransport during this section: constructing it fails the test loudly,
    so a forgotten `transport=` can never send a real request."""

    def __init__(self, *a, **k):
        raise AssertionError("HttpTransport constructed in a test")


def ans(n, approx=False):
    return (200, {"count": n, "approx": approx, "latency": 1.0})


# Dominant-mode targets, in walk order: Ada 1869 (c), Bo 1850 (c), Bo 1860 (h), Cy 1870 (c).
# Ada 1869 hits on its 2nd phrase; Bo 1850 never hits; Bo 1860 is approx on its 1st phrase (not a
# hit) and hits on its 4th; Cy 1870's 1st phrase fails (400) the first time and is 0 afterwards.
ANSWERS = {"Ada Test (1869": [ans(3)], "Bo Test (born 1860": [ans(5, approx=True)],
           "Bo Test, born 1860": [ans(2)], "Cy Test (born 1870": [(400, {"error": "bad"}), ans(0)]}
FIRST_RUN = ["Ada Test (born 1869", "Ada Test (1869",
             "Bo Test (born 1850", "Bo Test (1850", "Bo Test was born in 1850", "Bo Test, born 1850",
             "Bo Test (born 1860", "Bo Test (1860", "Bo Test was born in 1860", "Bo Test, born 1860",
             "Cy Test (born 1870", "Cy Test (1870", "Cy Test was born in 1870", "Cy Test, born 1870"]


class PhraseTransport:
    """Answers a count body by its query: the answers listed for it in order (the last one
    repeats), (200, count 0) for any other query. Records every body and its fake start time."""

    def __init__(self, ft, answers=None):
        self.ft = ft
        self.answers = copy.deepcopy(ANSWERS if answers is None else answers)
        self.bodies, self.starts = [], []

    def post(self, body):
        self.bodies.append(body)
        self.starts.append(self.ft.mono_now)
        self.ft.mono_now += 0.5
        self.ft.wall_now += 0.5
        q = self.answers.get(body["query"])
        if q:
            return q.pop(0) if len(q) > 1 else q[0]
        return ans(0)

    @property
    def queries(self):
        return [b["query"] for b in self.bodies]


class Refuse:
    """A transport that must never be used."""

    def __init__(self):
        self.bodies = []

    def post(self, body):
        self.bodies.append(body)
        raise AssertionError("a request was sent")


M13 = "olmo2-7b"
CLAIMS13 = (wrong_samples(M13, "biographies-00001", ["1901 1902", "1901 1902", "1903"], own="1869", name="Ada Test")
            + wrong_samples(M13, "biographies-00002", ["1860"] * 5, own="1850", name="Bo Test")
            + wrong_samples(M13, "biographies-00003", ["1880", "1880", "1881", "1882"], own="1870", name="Cy Test")
            + [claim(M13, "biographies-00004", 1, "1890", own="1890", name="Di Test"),
               claim(M13, "biographies-00004", 2, "1891", own="1890", name="Di Test")]
            # Ed: never wrong (sample 2 states the own year and 1841, so it
            # is right): S1, n_wrong 0, no role h year. Under the stand-in cu.slots sample 2 would
            # be wrong (dominant 1841, no h row for it: no_count_row), so Ed also shows the CLI's slots
            # come from stability.slot_items.
            + [claim(M13, "biographies-00005", 1, "1840", own="1840", name="Ed Test"),
               claim(M13, "biographies-00005", 2, "1840 1841", own="1840", name="Ed Test"),
               claim(M13, "biographies-00005", 3, "1840", own="1840", name="Ed Test")])
COUNTS13 = [row("c", 1869, "unknown", name="Ada Test"), row("h", 1901, "zero", "0", name="Ada Test"),
            row("h", 1902, "unknown", name="Ada Test"),
            row("c", 1850, "unknown", name="Bo Test"), row("h", 1860, "unknown", name="Bo Test"),
            row("c", 1870, "unknown", name="Cy Test"), row("h", 1880, "zero", "0", name="Cy Test"),
            row("c", 1890, "counted", "5", name="Di Test"), row("h", 1891, "zero", "0", name="Di Test"),
            row("c", 1840, "counted", "4", name="Ed Test")]
# generation ids (model|prompt_id|sample) the verifier-output cases name
ADA1, BO5, DI1 = ("%s|biographies-0000%d|%d" % (M13, p, k) for p, k in ((1, 1), (2, 5), (4, 1)))
TAU_COLUMNS13 = ["variant", "index", "model", "percentile", "tau"]
TAU_DEFAULT = {"v4_olmo-mix-1124_llama": "1", "v4_dclm-baseline_llama": "1", "v4_dolma-v1_7_llama": "1",
               "v4_rpj_llama_s4": "2", "v4_piletrain_llama": "3"}


def tau_rows(values=None, drop=()):
    values = dict(TAU_DEFAULT, **(values or {}))
    rows = [{"variant": "headline_406", "index": i, "model": i, "percentile": "10", "tau": t}
            for i, t in values.items() if i not in drop]
    # decoys the CLI must skip: another variant, another percentile
    rows += [{"variant": "with_short_444", "index": i, "model": i, "percentile": "10", "tau": "9"} for i in values]
    rows += [{"variant": "headline_406", "index": i, "model": i, "percentile": "5", "tau": "9"} for i in values]
    return rows


def write_csv(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


BEXEC_META = ["generation_id", "prompt_id", "model", "sample_index", "task", "batch_id", "outcome",
              "decomposer_empty_reason", "sent", "input_row", "stripped_tail"]


def write_bexec(root, claims, outcomes=None, stray=()):
    """A synthetic local copy of the verifier files for `claims`, laid out as tests/test_stability.py
    section 5 builds one: per model, unit <model>-b00000 with inputs/<model>/<unit>.meta.csv (one
    sidecar row per generation the claims name, outcome "ok" unless `outcomes` {generation_id:
    outcome} says otherwise; an ok row is sent, others are not and have input_row -1),
    inputs/<model>/<unit>.csv (one name per sent generation: its id) and
    outputs_filtered/<model>/<unit>.csv (atoms per sent generation). `stray`: generation ids a
    second unit <model>-biographies-s0 also lists, with no output (bio_samples skips that unit)."""
    outcomes = outcomes or {}
    gens = {c["generation_id"]: c["model"] for c in claims}
    by_model = defaultdict(list)
    for gid in sorted(gens):
        by_model[gens[gid]].append(gid)
    for model, gids in by_model.items():
        (root / "inputs" / model).mkdir(parents=True, exist_ok=True)
        (root / "outputs_filtered" / model).mkdir(parents=True, exist_ok=True)
        metas, names = [], []
        for gid in gids:
            _, pid, sample = gid.split("|")
            outcome = outcomes.get(gid, "ok")
            ok = outcome == "ok"
            metas.append({"generation_id": gid, "prompt_id": pid, "model": model, "sample_index": sample,
                          "task": "biographies", "batch_id": "b", "outcome": outcome,
                          "decomposer_empty_reason": "", "sent": int(ok), "input_row": len(names) if ok else -1,
                          "stripped_tail": ""})
            if ok:
                names.append(gid)
        unit = "%s-b00000" % model
        write_csv(root / "inputs" / model / (unit + ".meta.csv"), BEXEC_META, metas)
        write_csv(root / "inputs" / model / (unit + ".csv"), ["name", "response", "generation_id"],
                  [{"name": n, "response": "r", "generation_id": n} for n in names])
        write_csv(root / "outputs_filtered" / model / (unit + ".csv"), ["name", "atoms"],
                  [{"name": n, "atoms": "| A.\tTrue |\n"} for n in names])
        strays = [dict(m, input_row=i) for i, m in enumerate(m for m in metas if m["generation_id"] in stray)]
        if strays:
            su = "%s-biographies-s0" % model
            write_csv(root / "inputs" / model / (su + ".meta.csv"), BEXEC_META, strays)
            write_csv(root / "inputs" / model / (su + ".csv"), ["name", "response", "generation_id"],
                      [{"name": m["generation_id"], "response": "r", "generation_id": m["generation_id"]}
                       for m in strays])


class CliEnv(Env):
    """Env plus the three synthetic inputs, a synthetic verifier output copy for the claims (`bexec`), an
    out dir holding scripts/c_exec_tau.py's files (which the CLI must never touch) and a phrase cache
    path that does not exist yet."""

    def __enter__(self):
        super().__enter__()
        d = Path(self.tmp.name)
        self.d = d
        self.claims, self.counts, self.tau = d / "claims.csv", d / "counts.csv", d / "tau.csv"
        self.bexec = d / "bexec"
        self.set_claims(CLAIMS13)
        write_csv(self.counts, list(COUNTS13[0]), COUNTS13)
        write_csv(self.tau, TAU_COLUMNS13, tau_rows())
        self.out = d / "out"
        self.out.mkdir()
        self.sentinels = {"he_tau.csv": b"variant,index\nkeep,me\n", "meta.json": b'{"keep": "me"}\n'}
        for n, b in self.sentinels.items():
            (self.out / n).write_bytes(b)
        self.pc = d / "phrase_counts.sqlite"
        return self

    def set_claims(self, rows, **bexec):
        """The claims file and a fresh verifier output copy at self.bexec for `rows` (write_bexec's keywords)."""
        write_csv(self.claims, list(CLAIM_COLUMNS), rows)
        if self.bexec.exists():
            shutil.rmtree(self.bexec)
        write_bexec(self.bexec, rows, **bexec)

    def args(self, *extra, cache="default", bexec="default"):
        """`bexec` "default": --bexec-root self.bexec; None: no --bexec-root (the CLI's default,
        patched in this section to a folder that does not exist); else that path."""
        a = ["--claims", str(self.claims), "--counts", str(self.counts), "--tau", str(self.tau),
             "--out", str(self.out)]
        if cache == "default":
            a += ["--cache", str(self.pc)]
        elif cache is not None:
            a += ["--cache", str(cache)]
        if bexec == "default":
            a += ["--bexec-root", str(self.bexec)]
        elif bexec is not None:
            a += ["--bexec-root", str(bexec)]
        return a + list(extra)

    def run(self, argv, transport=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = cli.main(argv, transport=transport, clock=self.ft.mono, sleep=self.ft.sleep,
                                wall=self.ft.wall)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def outputs(self):
        return sorted(n for n in cli.OUTPUTS if (self.out / n).exists())

    def sentinels_ok(self):
        return all((self.out / n).read_bytes() == b for n, b in self.sentinels.items())

    def phrase(self):
        return read_csv(self.out / cli.PHRASE_NAME)

    def slots(self):
        return {r["prompt_id"]: r for r in read_csv(self.out / cli.SLOTS_NAME)}

    def meta(self):
        return json.loads((self.out / cli.META_NAME).read_text(encoding="utf-8"))


def case_cli_error_limits(env):
    bad = (400, {"error": "bad"})
    # error, hit, error: the hit resets the run of errors but not the total
    pattern = {"Ada Test (born 1869": [bad], "Ada Test (1869": [ans(3)], "Bo Test (born 1850": [bad]}
    tr = PhraseTransport(env.ft, pattern)
    code, out, err = env.run(env.args("--max-consecutive-errors", "2"), tr)
    eq(code, 0, "--max-consecutive-errors 2: an answer between two errors resets the run, the run finishes")
    tr = PhraseTransport(env.ft, pattern)
    code, out, err = env.run(env.args("--max-errors", "2", "--force"), tr)
    eq((code, tr.queries), (1, ["Ada Test (born 1869", "Bo Test (born 1850"]),
       "--max-errors 2: failures are never cached, so both are resent; stops at the 2nd failure")
    check("--max-errors 2" in err and "rerun the same command" in err and "Traceback" not in err,
          "the stop names the limit and says the cache resumes")
    all_bad = {q: [bad] for q in FIRST_RUN + cu.phrases("Ada Test", "birth", 1869)}
    for f in cli.OUTPUTS:
        (env.out / f).unlink()
    tr = PhraseTransport(env.ft, all_bad)
    code, out, err = env.run(env.args("--max-consecutive-errors", "3", cache=env.pc.parent / "fresh.sqlite"), tr)
    eq((code, len(tr.bodies)), (1, 3), "--max-consecutive-errors 3: exit 1 after 3 failed requests in a row")
    eq(env.outputs(), [], "a run stopped on errors writes no output")
    check(env.sentinels_ok(), "he_tau.csv and meta.json left alone")
    # --min-gap: 1.5 s between starts by default; never below infinigram.MIN_GAP_S
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args(cache=env.pc.parent / "gap.sqlite"), tr)
    gaps = [b - a for a, b in zip(tr.starts, tr.starts[1:])]
    check(code == 0 and len(gaps) > 5 and min(gaps) >= cli.MIN_GAP_DEFAULT - EPS,
          "default --min-gap: starts at least %s s apart (min %r)" % (cli.MIN_GAP_DEFAULT, min(gaps or [0])))
    for f in cli.OUTPUTS:
        (env.out / f).unlink()
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args("--min-gap", "2.5", cache=env.pc.parent / "gap2.sqlite"), tr)
    gaps = [b - a for a, b in zip(tr.starts, tr.starts[1:])]
    check(code == 0 and min(gaps) >= 2.5 - EPS, "--min-gap 2.5: starts at least 2.5 s apart")
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args("--min-gap", "1.0", "--force"), tr)
    eq((code, tr.bodies), (2, []), "--min-gap below infinigram.MIN_GAP_S: exit 2, nothing sent")


def fill_cache(path, queries, answers):
    """Every (index, phrase) of `queries` straight into a phrase cache: the last answer `answers`
    lists for the phrase (a failed first request ends at its retry's answer), else a count of 0."""
    c = infinigram.Cache(str(path), lock=False)
    try:
        c.ensure_counts_table()
        for index, phrase in sorted(set(queries)):
            _, body = (answers.get(phrase) or [ans(0)])[-1]
            c.put_count(index, phrase, body["count"], body["approx"], body["latency"])
    finally:
        c.close()


def case_nh_sensitivity(env):
    # scripts/c_exec_nh_sensitivity.py rebuilds this CLI's baseline through the CLI's own
    # parser, so a change to the CLI's inputs or prepare() fails here, not silently on real data
    args = cli.build_parser().parse_args(env.args("--cache-only"))
    fill_cache(env.pc, [q for t in cli.prepare(cli.load_inputs(args), "s2s3")["targets"]
                        for q in cli.target_queries(t)], ANSWERS)
    eq(env.run(env.args("--cache-only"))[0], 0, "nh sensitivity: the CLI writes untyped_slots.csv first")
    base = ["--cache", str(env.pc), "--claims", str(env.claims), "--counts", str(env.counts), "--tau", str(env.tau),
            "--out", str(env.out)]

    def run_nh(argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = nh.main(argv)
            except SystemExit as e:   # argparse: an option the script does not take
                code = e.code
        return code, out.getvalue(), err.getvalue()

    code, out, err = run_nh(base + ["--bexec-root", str(env.bexec)])
    eq(code, 0, "nh sensitivity: exit 0, its baseline equal to untyped_slots.csv slot by slot")
    md = env.out / "nh_sensitivity.md"
    check(md.exists() and (env.out / "nh_sensitivity.csv").exists(), "nh sensitivity: both outputs written to --out")
    text = md.read_text(encoding="utf-8")
    check("stability.slot_items" in text and "over the slots with a wrong sample" in text,
          "nh sensitivity: the report names slot_items and the wrong-sample F5 pool")
    eq(run_nh(base + ["--bexec-root", str(env.bexec)])[0], 1, "nh sensitivity: refuses existing outputs without --force")
    code, out, err = run_nh(base + ["--force"])
    check(code == 1 and "no verifier output copy" in err, "nh sensitivity: no --bexec-root falls back to the CLI's default, exit 1")
    rows = read_csv(env.out / cli.SLOTS_NAME)
    rows[0]["t5_column"] = "changed"
    write_csv(env.out / cli.SLOTS_NAME, list(rows[0]), rows)
    code, out, err = run_nh(base + ["--bexec-root", str(env.bexec), "--force"])
    check(code == 1 and "baseline differs" in err, "nh sensitivity: a slot that differs from untyped_slots.csv, exit 1")


def run_cli_case(label, fn):
    try:
        with CliEnv() as env:
            fn(env)
    except Exception as e:  # noqa: BLE001
        check(False, "%s  (raised %r)" % (label, e))


def case_cli_live(env):
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args(), tr)
    eq(code, 0, "live run: exit 0")
    eq(tr.queries, FIRST_RUN, "the walk: phrases in PHRASES order per target, targets in order, stop at the hit")
    check("Ada Test was born in 1869" not in tr.queries and "Ada Test, born 1869" not in tr.queries,
          "no request after a target's hit phrase")
    check(not [q for q in tr.queries if "1902" in q or "1901" in q], "dominant mode: no target for 1902 (or the zero 1901)")
    check(all(b == {"index": OLMO, "query_type": "count", "query": b["query"]} for b in tr.bodies),
          "every body is a plain count on the slot model's index")
    check(all(b - a >= infinigram.MIN_GAP_S - EPS for a, b in zip(tr.starts, tr.starts[1:])),
          "request starts at least MIN_GAP_S apart on the fake clock")
    eq(env.outputs(), sorted(cli.OUTPUTS), "all five outputs written")
    check(env.sentinels_ok(), "he_tau.csv and meta.json in --out left byte-identical")
    ph = env.phrase()
    eq(list(ph[0]), cli.PHRASE_COLUMNS, "untyped_phrase.csv columns")
    eq(len(ph), 14, "one phrase row per phrase sent")
    by = {(r["year"], r["phrase_no"]): r for r in ph}
    eq([(by[("1869", n)]["hit"], by[("1869", n)]["count"]) for n in ("1", "2")], [("0", "0"), ("1", "3")],
       "Ada 1869: phrase 1 count 0, phrase 2 count 3 is the hit")
    eq({r["target_status"] for r in ph if r["year"] == "1869"} | {r["year_present"] for r in ph if r["year"] == "1869"},
       {"hit", "1"}, "Ada 1869: target hit, year_present 1")
    eq((by[("1860", "1")]["approx"], by[("1860", "1")]["hit"], by[("1860", "1")]["count"]), ("1", "0", "5"),
       "approx true with count 5: recorded, not a hit")
    eq((by[("1860", "4")]["hit"], by[("1860", "4")]["target_status"]), ("1", "hit"),
       "the walk continued past the approx answer and hit on phrase 4")
    eq((by[("1870", "1")]["error"], by[("1870", "1")]["count"], by[("1870", "1")]["approx"]), ("1", "", ""),
       "a failed request: error 1, empty count and approx")
    eq({r["target_status"] for r in ph if r["year"] == "1870"}, {"incomplete"}, "no hit and an error: incomplete")
    eq({r["target_status"] for r in ph if r["year"] == "1850"}, {"no_hit"}, "no hit, no error: no_hit")
    eq({r["from_cache"] for r in ph}, {"0"}, "first run: nothing from the cache")
    eq((by[("1869", "1")]["roles"], by[("1869", "1")]["regimes"], by[("1860", "1")]["roles"]), ("c", "S3", "h"),
       "roles and regimes")
    sl = env.slots()
    eq(list(next(iter(sl.values()))), cli.SLOT_COLUMNS, "untyped_slots.csv columns")
    with open(env.out / cli.SLOTS_NAME, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    eq((len(header), header[17], header[22]), (23, "t5_column", "never_wrong"),
       "untyped_slots.csv: 23 columns, t5_column still the 18th, never_wrong the 23rd (last)")
    eq({p: r["never_wrong"] for p, r in sl.items()},
       {"biographies-00001": "0", "biographies-00002": "0", "biographies-00003": "0", "biographies-00004": "0",
        "biographies-00005": "1"}, "never_wrong: 1 on Ed (no wrong sample) only; Di's single wrong sample makes 0")
    e_ = sl["biographies-00005"]
    eq((e_["regime"], e_["dominant_wrong"], e_["h_status"], e_["h_max"], e_["c_status"], e_["c_max"]),
       ("S1", "", "", "", "counted", "4"),
       "Ed: slot_items' rule (the own year beside 1841 is right): S1, no dominant wrong year, no h row looked up")
    eq((e_["class"], e_["type"], e_["t5_column"], e_["f5_point"]),
       ("typed", "RECALL_FAILURE", "RECALL_FAILURE", "at inference"),
       "Ed's row keeps classify's type and point (n_c 4 > tau 1, n_h 0); F5's count leaves it out")
    a, b, c_, d_ = (sl["biographies-0000%d" % i] for i in (1, 2, 3, 4))
    eq((a["class"], a["f5_point"], a["present"], a["c_status"], a["untyped_before"], a["t5_column"]),
       ("phrase_hit", "undetermined", "c:1869", "present", "1", cu.UNTYPED_LABEL),
       "a c-row hit turns the S3 untyped slot into phrase_hit, undetermined")
    eq((b["class"], b["f5_point"], b["present"], b["reachable"]),
       ("phrase_hit", "open", "h:1860", "at training data; undetermined"),
       "S2 with a wrong-year hit only: phrase_hit, open, the n_h = 0 types ruled out")
    eq((c_["class"], c_["f5_point"], c_["reachable"], c_["present"]),
       ("untyped", "open", "at the knowledge boundary; undetermined", ""), "the incomplete S3 slot stays untyped, open")
    eq((d_["class"], d_["f5_point"], d_["type"], d_["c_max"], d_["h_max"]), ("typed", "at inference", "RECALL_FAILURE", "5", "0"),
       "the S1 slot: typed (n_c 5 > tau 1), at inference, numbers without a decimal")
    eq((a["own_years"], a["dominant_wrong"], a["h_mode"], a["h_max"]), ("1869", "1901", "dominant", "0"),
       "own years, dominant wrong year, h mode")
    f5 = {r["point"]: r for r in read_csv(env.out / cli.F5_NAME)}
    eq(list(f5), list(cu.POINTS) + [cu.TOGETHER, cli.OPEN_S3, cli.OPEN_S2], "untyped_f5.csv rows")
    eq({k: (f5[k]["lower"], f5[k]["upper"], f5[k]["n"]) for k in (cu.INFERENCE, cu.UNDETERMINED, cli.OPEN_S3, cli.OPEN_S2)},
       {cu.INFERENCE: ("1", "1", "4"), cu.UNDETERMINED: ("1", "3", "4"), cli.OPEN_S3: ("1", "1", "4"),
        cli.OPEN_S2: ("1", "1", "4")},
       "F5: lower, upper and n; the open band split S3 / S2; never-wrong Ed left out (n 4 not 5, at inference "
       "1 (Di) not 2)")
    eq((f5[cu.INFERENCE]["share_lower"], f5[cli.OPEN_S2]["share_upper"]), ("0.2500", "0.2500"), "shares, 4 decimals")
    m = env.meta()
    eq({k: m["counts"][k] for k in ("requests", "errors", "approx", "hits", "incomplete", "cached", "targets")},
       {"requests": 14, "errors": 1, "approx": 1, "hits": 2, "incomplete": 1, "cached": 0, "targets": 4},
       "meta counts")
    eq((m["scope"], m["h_years"], m["rule"], m["script_version"], m["checked_facts"]),
       ("s2s3", "dominant", bc.RULE, cli.SCRIPT_VERSION, ["birth"]), "meta: scope, h_years, RULE, version, facts")
    eq(cli.SCRIPT_VERSION, "2", "SCRIPT_VERSION 2 (slot_items, never_wrong, F5 over wrong samples)")
    eq(m["never_wrong"], {"olmo2-7b": 1, "dclm-7b": 0, "olmo17-7b": 0, "redpajama-7b": 0, "pythia-69b": 0},
       "meta never_wrong: every model, zeros included")
    eq((m["f5"]["n"], m["slot_table"][0]["regime"], m["slot_table"][0]["all"]), (4, "S1", 2),
       "meta: F5 n 4, while the slot table still counts Ed among S1's 2 slots")
    eq(m["provenance"]["bexec"], {"root": str(env.bexec.resolve()), "outputs": "outputs_filtered", "samples": 17,
                                  "skipped_units": [], "claims_read": 17, "claims_kept": 17, "claims_dropped": 0},
       "meta provenance bexec: 17 generations, one claim each, all ok")
    eq(m["phrases"], cu.PHRASES, "meta: PHRASES")
    raw = (env.out / cli.META_NAME).read_text(encoding="utf-8")
    eq(raw, json.dumps(m, indent=2, sort_keys=True, ensure_ascii=False) + "\n", "meta.json: sorted keys, indent 2")
    check(len(m["provenance"]["inputs"]["claims"]["blob"]) == 40, "meta: a blob hash per input")
    eq([r["tau"] for r in m["provenance"]["tau_rows"]], ["1", "1", "1", "2", "3"], "meta: the tau rows used")
    s = (env.out / cli.SUMMARY_NAME).read_text(encoding="utf-8")
    check(s.startswith(cli.TITLE + "\n"), "summary title")
    for part in ("## (a)", "## (b)", "## (c)", "## (d)", "## (e)", "| S1 | 2 | 2 | 0 | not run | 0 | 0 |",
                 "| `single_mention` | 0 | 0 | 0 | not run | 0 | 0 |", "| at inference | 25.0% | 1 |",
                 "| undetermined | 25.0% to 75.0% | 1 to 3 |", "| %s | 1 | 25.0%% |" % cli.OPEN_S3,
                 "| %s | 1 | 25.0%% |" % cli.OPEN_S2, "Requests sent: 14", "Incomplete targets", "tau from historical_events",
                 "Slots with a Wikidata year: 5 (S1 2, S2 1, S3 2, `single_mention` 0). Before C and E, 3 were "
                 "untyped (S1 0, S2 1, S3 2, `single_mention` 0).",
                 "Merged (C) in total: 0. S1 to S3 slots with a count row and at least one wrong sample (F5's "
                 "slots): 4.",
                 "Over the 4 S1 to S3 slots with a count row and at least one wrong sample (never-wrong S1 slots "
                 "left out).",
                 "- verifier output copy: `%s` (`outputs_filtered`), 17 samples, skipped units: none; claims kept 17 of 17 "
                 "(0 dropped, `stability.ok_claims`)." % env.bexec.resolve()):
        check(part in s, "summary has %r" % part)
    check("of F5): 1.\n\n| Model | S1, never wrong |\n|---|---|\n| olmo2-7b | 1 |\n"
          "| dclm-7b | 0 |\n| olmo17-7b | 0 |\n| redpajama-7b | 0 |\n| pythia-69b | 0 |\n\n## (b)" in s,
          "summary (a): never-wrong S1 slots 1, then the table, all five models in MODEL_INDEX order, zeros included")
    check(chr(0x2014) not in s and chr(0x2013) not in s, "summary: no em or en dashes")
    check("incomplete" in err, "stderr says a target is incomplete")

    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args(), tr)
    eq((code, tr.bodies), (1, []), "existing outputs without --force: exit 1, nothing sent")
    check("--force" in err and "Traceback" not in err, "the refusal names --force, no traceback")
    check(env.sentinels_ok(), "the refusal leaves he_tau.csv and meta.json alone")

    tr = PhraseTransport(env.ft, dict(ANSWERS, **{"Cy Test (born 1870": [ans(0)]}))
    code, out, err = env.run(env.args("--force"), tr)
    eq((code, tr.queries), (0, ["Cy Test (born 1870"]), "--force rerun resumes: only the failed phrase is sent")
    ph = env.phrase()
    eq([(r["phrase"], r["from_cache"]) for r in ph if r["from_cache"] == "0"], [("Cy Test (born 1870", "0")],
       "every other answer from the cache")
    eq({r["target_status"] for r in ph if r["year"] == "1870"}, {"no_hit"}, "the incomplete target is now no_hit")
    check(env.sentinels_ok(), "--force leaves he_tau.csv and meta.json alone")
    slots_bytes = (env.out / cli.SLOTS_NAME).read_bytes()

    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args("--force"), tr)
    eq((code, tr.bodies), (0, []), "a full rerun sends nothing")
    eq({r["from_cache"] for r in env.phrase()}, {"1"}, "rerun: from_cache 1 everywhere")
    eq(env.meta()["counts"]["requests"], 0, "rerun: requests 0")

    rf = Refuse()
    code, out, err = env.run(env.args("--cache-only", "--force"), rf)
    eq((code, rf.bodies), (0, []), "--cache-only on a complete cache: exit 0, nothing sent")
    eq(({r["from_cache"] for r in env.phrase()}, env.meta()["counts"]["requests"]), ({"1"}, 0),
       "--cache-only: from_cache 1, requests 0")
    eq((env.out / cli.SLOTS_NAME).read_bytes(), slots_bytes, "--cache-only rebuilds the same slot file")
    eq(env.meta()["provenance"]["mode"], "cache-only", "meta records the mode")

    rf = Refuse()
    code, out, err = env.run(env.args("--dry-run"), rf)
    eq(code, 0, "dry run on the filled cache")
    check("targets 4 (serving role c 3, role h 1), phrases at most 16, already cached 14, requests at most 2"
          in out, "dry run counts what is cached (the 2 phrases after Ada's hit are not; Bo 1860 hit on its last)")
    check(not any(os.path.exists(str(env.pc) + s) for s in ("-wal", "-shm")),
          "--cache-only and --dry-run read the closed WAL phrase cache without creating -wal or -shm")
    held = infinigram.Cache(str(env.pc), lock=True)
    env.caches.append(held)
    code, out, err = env.run(env.args("--dry-run"), Refuse())
    eq(code, 0, "dry run works while another process holds the cache lock (it takes no lock)")
    code, out, err = env.run(env.args("--cache-only", "--force"), Refuse())
    eq(code, 0, "--cache-only takes no lock either")
    code, out, err = env.run(env.args("--force"), PhraseTransport(env.ft))
    eq(code, 1, "live run while the lock is held: exit 1")
    check("another fetcher" in err and "window fetch" in err, "LockHeld: says another fetcher holds the lock")


def case_cli_dry_run(env):
    rf = Refuse()
    code, out, err = env.run(env.args("--dry-run"), rf)
    eq((code, rf.bodies), (0, []), "--dry-run: exit 0, nothing sent")
    check(not env.pc.exists(), "--dry-run creates no cache file")
    eq(env.outputs(), [], "--dry-run writes no output")
    check(not os.path.exists(os.path.join(env.tmp.name, "userlock")), "--dry-run takes no per-user lock")
    # one mode (role h: the dominant wrong year): Ada 1869 c, Bo 1850 c, Bo 1860 h, Cy 1870 c; Ada's
    # 1901 and Cy's 1880 are zero rows, Di and Ed are S1 (out of s2s3)
    head, sep, tail = out.partition("queries (index, phrase), sorted, 16:\n")
    eq(head.splitlines(), ["dry run: no request, no lock, no file written (scope s2s3)",
                           "cache: no file at %s (nothing created, 0 cached)" % env.pc,
                           "targets 4 (serving role c 3, role h 1), phrases at most 16, already cached 0, "
                           "requests at most 16"],
       "dry run: one targets line (one mode), nothing about h-years modes, before the query list")
    check(sep and not any(x in out for x in ("h-years", "union of both", "subset")),
          "no h-years, union or subset line")
    lines = tail.splitlines()
    want = sorted("%s\t%s" % (OLMO, p) for name, y in (("Ada Test", 1869), ("Bo Test", 1850), ("Bo Test", 1860),
                                                      ("Cy Test", 1870)) for p in cu.phrases(name, "birth", y))
    eq(lines, want, "the query list: the 4 birth phrases of each of the 4 targets, sorted, one per line")
    check("%s\tAda Test (born 1902" % OLMO not in lines, "no target for the tuple's 1902")
    code, out, err = env.run(env.args("--dry-run", "--h-years", "tuple"), Refuse())
    check(code == 2 and "--h-years" in err and out == "", "--dry-run --h-years tuple: a usage error, exit 2")
    code, out, err = env.run(env.args("--dry-run", cache=None), Refuse())
    check(code == 0 and "none given" in out, "--dry-run without --cache")


def case_cli_cache_only_missing(env):
    code, out, err = env.run(env.args("--cache-only"), Refuse())
    eq(code, 1, "--cache-only with no cache file: exit 1")
    check("Ada Test (born 1869" in err and "Traceback" not in err, "it lists the missing (index, phrase)")
    check(not env.pc.exists(), "--cache-only creates no cache file")
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args("--max-requests", "3"), tr)
    eq((code, len(tr.bodies)), (1, 3), "--max-requests 3: exit 1 after 3 requests")
    check("after 3 requests" in err and "rerun the same command" in err, "the stop says the cache resumes")
    eq(env.outputs(), [], "a stopped run writes no output")
    code, out, err = env.run(env.args("--cache-only"), Refuse())
    eq(code, 1, "--cache-only on a partial cache: exit 1")
    missing = [ln.strip() for ln in err.splitlines() if ln.startswith("  ")]
    eq(missing, ["%s\t%s" % (OLMO, q) for q in ("Bo Test (1850", "Bo Test (born 1860", "Cy Test (born 1870")],
       "missing: the first uncached phrase of each target that did not hit (Ada's hit is cached)")
    eq(env.outputs(), [], "the refusal writes no output")
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args(), tr)
    eq((code, tr.queries), (0, FIRST_RUN[3:]), "the next live run sends only the rest")


WINDOW_ROW = ("INSERT INTO windows (index_name, query_string, form, s, rank, spans_json, fetched_at) "
              "VALUES ('i', 'q', 'plain', 0, 0, '[]', 'now')")
PLAN_ROW = "INSERT INTO plan (index_name, query_string, form, s, rank, selected) VALUES ('i', 'q', 'plain', 0, 0, 'all')"


def window_cache(path, insert=WINDOW_ROW):
    """A real window cache: infinigram.Cache (WAL mode, as the fetcher writes it) with one row,
    closed, so the file stands alone with no -wal or -shm. Called inside a CliEnv, so
    LOCK_DIR_ENV points at the per-test temp dir (lock=False takes no lock anyway)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = infinigram.Cache(str(path), lock=False)
    try:
        c.conn.execute(insert)
        c.conn.commit()
    finally:
        c.close()


def listing(d):
    return sorted(os.listdir(str(d)))


def case_cli_window_cache(env):
    wdir = env.d / "wc"
    wc = wdir / "shard_cache.sqlite"
    window_cache(wc)
    before, names = wc.read_bytes(), listing(wdir)
    eq(before[18:20], b"\x02\x02", "the fixture is a WAL database (header bytes 18 and 19 are 2)")
    eq(names, ["shard_cache.sqlite"], "the fixture: the database file alone")
    for extra in ((), ("--dry-run",), ("--cache-only",)):
        tr = PhraseTransport(env.ft)
        label = " ".join(extra) or "(live)"
        code, out, err = env.run(env.args(*extra, cache=wc), tr)
        eq((code, tr.bodies), (1, []), "window cache refused %s" % label)
        check("window cache" in err, "the refusal says it is a window cache")
        eq(listing(wdir), names, "%s: no -wal, -shm or .lock beside the window cache" % label)
        eq(wc.read_bytes(), before, "%s: the window cache file is byte-identical" % label)
    check(not os.path.exists(os.path.join(env.tmp.name, "userlock")), "no per-user lock taken")
    eq(env.outputs(), [], "no output")
    pdir = env.d / "pc"
    emptyish = pdir / "plan_only.sqlite"
    window_cache(emptyish, insert=PLAN_ROW)
    pb = emptyish.read_bytes()
    code, out, err = env.run(env.args("--dry-run", cache=emptyish), Refuse())
    check(code == 1 and "plan" in err, "a cache with a plan row is refused too")
    eq((listing(pdir), emptyish.read_bytes() == pb), (["plan_only.sqlite"], True),
       "the plan-row cache: nothing beside it, byte-identical")
    for name in ("infinigram.sqlite", "Infinigram.SQLite"):
        p = env.d / "sub2" / name
        code, out, err = env.run(env.args(cache=p), Refuse())
        check(code == 1 and "infinigram.sqlite" in err and not p.exists(), "a cache named %s is refused, not created" % name)


def case_cli_inputs(env):
    bad = [dict(r) for r in COUNTS13]
    bad[3]["matcher"] = "bare"
    write_csv(env.counts, list(COUNTS13[0]), bad)
    code, out, err = env.run(env.args("--dry-run"), Refuse())
    check(code == 1 and "the counting cell" in err and "1 of 10" in err, "a count row outside bio_counts.RULE is refused")
    write_csv(env.counts, list(COUNTS13[0]), COUNTS13)
    write_csv(env.tau, TAU_COLUMNS13, tau_rows({"v4_piletrain_llama": "4"}))
    code, out, err = env.run(env.args("--dry-run"), Refuse())
    check(code == 1 and "differs" in err and "v4_piletrain_llama 4 (expected 3)" in err, "a tau mismatch is refused")
    exp = cli.DEFAULT_EXPECTED_TAU.replace("v4_piletrain_llama=3", "v4_piletrain_llama=4.0")
    code, out, err = env.run(env.args("--dry-run", "--expected-tau", exp), Refuse())
    eq(code, 0, "--expected-tau overrides it (compared as floats)")
    write_csv(env.tau, TAU_COLUMNS13, tau_rows(drop=("v4_rpj_llama_s4",)))
    code, out, err = env.run(env.args("--dry-run"), Refuse())
    check(code == 1 and "no headline_406 row" in err and "v4_rpj_llama_s4" in err, "a missing tau row is refused")
    write_csv(env.tau, TAU_COLUMNS13, tau_rows())
    for argv, label in ((["--dry-run", "--cache-only"], "--dry-run with --cache-only"),
                        ([], "no --cache outside --dry-run"),
                        (["--max-requests", "-1"], "--max-requests -1"),
                        (["--dry-run", "--expected-tau", "v4_piletrain_llama=3"], "--expected-tau missing indexes"),
                        (["--dry-run", "--scope", "all"], "an unknown --scope"),
                        (["--dry-run", "--h-years", "tuple"], "--h-years tuple (the option left the CLI)"),
                        (["--dry-run", "--h-years", "dominant"], "--h-years dominant (the option left the CLI)")):
        code, out, err = env.run(env.args(*argv, cache=None), Refuse())
        eq(code, 2, "bad command line, exit 2: %s" % label)


def case_cli_h_years_and_scope(env):
    """--h-years left the CLI (tuple mode is not a CLI option): role h
    is the single dominant wrong year on every run; --scope all-birth."""
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args("--h-years", "tuple"), tr)
    eq((code, tr.bodies, env.outputs()), (2, [], []), "--h-years tuple: exit 2, nothing sent, no output")
    check("--h-years" in err and not env.pc.exists(), "the usage error names --h-years; no cache file created")
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args(), tr)
    eq(code, 0, "a live run: exit 0")
    eq([q for q in tr.queries if "1902" in q], [], "Ada's tuple (1901 1902) gives no 1902 target")
    a = env.slots()["biographies-00001"]
    eq((a["h_mode"], a["dominant_wrong"], a["h_status"]), ("dominant", "1901", "zero"),
       "Ada: role h is 1901 alone (tie with 1902 at 2 of 3 samples, the earliest), its row zero")
    eq(env.meta()["h_years"], "dominant", "meta h_years: always dominant")
    code, out, err = env.run(env.args("--force", "--scope", "all-birth"), PhraseTransport(env.ft))
    eq(code, 0, "--scope all-birth runs")
    s = (env.out / cli.SUMMARY_NAME).read_text(encoding="utf-8")
    check("not run" not in s, "all-birth: the phrase-hit column shows counts, not 'not run'")


def case_cli_bexec(env):
    """--bexec-root: stability.bio_samples on the copy, stability.ok_claims on the claims."""
    def refused(argv, label, needle):
        rf = Refuse()
        code, out, err = env.run(argv, rf)
        eq((code, rf.bodies, env.outputs()), (1, [], []), "%s: exit 1, nothing sent, no output" % label)
        check(needle in err and "Traceback" not in err, "%s: the error says %r (%r)" % (label, needle, err.strip()))
        check(not env.pc.exists() and env.sentinels_ok(), "%s: no cache file, he_tau.csv and meta.json untouched" % label)

    # no --bexec-root: the CLI's default, patched in this section to a folder that does not exist
    refused(env.args(bexec=None), "no --bexec-root", "no verifier output copy at %s" % cli.DEFAULT_BEXEC_ROOT)
    for extra in ((), ("--dry-run",)):
        refused(env.args(*extra, bexec=env.d / "nowhere"), "a missing --bexec-root folder%s" % "".join(" " + e for e in extra),
                "no verifier output copy at %s" % (env.d / "nowhere"))
    half = env.d / "half"
    (half / "inputs").mkdir(parents=True)
    refused(env.args(bexec=half), "a copy with inputs/ but no outputs_filtered/", "needs inputs/ and outputs_filtered/")
    short = env.d / "short"
    write_bexec(short, [c for c in CLAIMS13 if c["generation_id"] != BO5])
    refused(env.args(bexec=short), "a claim whose generation has no sidecar row",
            "%s: no sidecar row for this claim's generation" % BO5)

    # Di's right sample (1) abstained: its claim is dropped, so Di has one wrong sample and no right
    # one (single_mention, out of F5). The stray unit lists Ada's sample 1 again, with no output.
    env.set_claims(CLAIMS13, outcomes={DI1: "abstention"}, stray=[ADA1])
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args(), tr)
    eq((code, tr.queries), (0, FIRST_RUN), "dropped claim and stray unit: exit 0, the same walk (Di is never a target)")
    m = env.meta()
    eq(m["provenance"]["bexec"], {"root": str(env.bexec.resolve()), "outputs": "outputs_filtered", "samples": 17,
                                  "skipped_units": ["olmo2-7b-biographies-s0"], "claims_read": 17, "claims_kept": 16,
                                  "claims_dropped": 1},
       "meta provenance bexec: 17 generations (the stray's one counted once), the stray unit skipped, 1 claim dropped")
    s = (env.out / cli.SUMMARY_NAME).read_text(encoding="utf-8")
    check("- verifier output copy: `%s` (`outputs_filtered`), 17 samples, skipped units: olmo2-7b-biographies-s0; claims kept "
          "16 of 17 (1 dropped, `stability.ok_claims`)." % env.bexec.resolve() in s,
          "summary provenance: the stray unit and the dropped claim")
    d_ = env.slots()["biographies-00004"]
    eq((d_["regime"], d_["f5_point"], d_["never_wrong"]), ("single_mention", "", "0"),
       "Di without its abstained right sample: single_mention, no point")
    f5 = {r["point"]: r for r in read_csv(env.out / cli.F5_NAME)}
    eq((f5[cu.INFERENCE]["lower"], f5[cu.INFERENCE]["upper"], f5[cu.INFERENCE]["n"]), ("0", "0", "3"),
       "F5 over Ada, Bo and Cy: n 3, nothing at inference (Di single_mention, Ed never wrong)")


def case_cli_foreign_tables(env):
    fdir = env.d / "foreign"
    fdir.mkdir()
    fp = fdir / "api_cache.sqlite"
    conn = sqlite3.connect(str(fp))
    conn.executescript(infinigram.SCHEMA + "CREATE TABLE responses (k TEXT PRIMARY KEY, v TEXT);"
                       "INSERT INTO responses VALUES ('a', 'b');")
    conn.commit()
    conn.close()
    before, names = fp.read_bytes(), listing(fdir)
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args(cache=fp), tr)
    eq((code, tr.bodies), (1, []), "live run on a file with an unrelated table: exit 1, nothing sent")
    check("responses" in err and "Traceback" not in err, "the refusal names the unexpected table")
    check("table(s) responses, which" in err, "only the unrelated table is named, not SCHEMA's")
    eq((fp.read_bytes() == before, listing(fdir)), (True, names), "the file is byte-identical, nothing beside it")
    check(not os.path.exists(os.path.join(env.tmp.name, "userlock")), "refused before Cache(...): no per-user lock")
    eq(env.outputs(), [], "no output")
    code, out, err = env.run(env.args("--dry-run", cache=fp), Refuse())
    eq(code, 0, "--dry-run reads it (live mode only refuses): exit 0")
    eq((fp.read_bytes() == before, listing(fdir)), (True, names), "--dry-run leaves it byte-identical")
    # a phrase cache Cache built (SCHEMA's tables plus counts) is accepted
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args(), tr)
    eq(code, 0, "a fresh phrase cache: exit 0")
    tr = PhraseTransport(env.ft)
    code, out, err = env.run(env.args("--force"), tr)
    eq(code, 0, "an existing phrase cache (SCHEMA's tables and counts) is accepted in live mode")


def case_cli_atomic(env):
    tr = PhraseTransport(env.ft)
    eq(env.run(env.args(), tr)[0], 0, "atomic: the first run writes the outputs")
    prior = {n: (env.out / n).read_bytes() for n in cli.OUTPUTS}
    names = listing(env.out)
    real = cli._write_file

    def failing(exc, at=3):
        calls = []

        def w(path, payload):
            calls.append(path)
            if len(calls) == at:
                Path(path).write_bytes(b"partial")   # the 3rd temp file exists, half written
                raise exc
            real(path, payload)
        return w

    for exc, label, needle in ((OSError("disk full (test)"), "an OSError", "disk full (test)"),
                               (KeyboardInterrupt(), "Ctrl+C", "interrupted (Ctrl+C)")):
        cli._write_file = failing(exc)
        try:
            code, out, err = env.run(env.args("--force"), PhraseTransport(env.ft))
        finally:
            cli._write_file = real
        eq(code, 1, "%s on the 3rd file: exit 1" % label)
        check(needle in err and "Traceback" not in err and len(err.strip().splitlines()) == 1,
              "%s: one line on stderr, no traceback (%r)" % (label, err.strip()))
        eq(listing(env.out), names, "%s: no temp file left, no new file in --out" % label)
        eq({n: (env.out / n).read_bytes() for n in cli.OUTPUTS}, prior, "%s: the prior outputs are untouched" % label)
        check(env.sentinels_ok(), "%s: he_tau.csv and meta.json untouched" % label)
    fresh = env.d / "out2"
    cli._write_file = failing(OSError("disk full (test)"))
    try:
        argv = env.args("--force")
        argv[argv.index("--out") + 1] = str(fresh)
        code, out, err = env.run(argv, PhraseTransport(env.ft))
    finally:
        cli._write_file = real
    eq((code, listing(fresh)), (1, []), "no prior outputs: a failure on the 3rd file leaves --out empty")


SUM_CLAIMS = (wrong_samples(M13, "biographies-00011", ["1801"] * 5, own="1800", name="Mo Merged")
              + wrong_samples(M13, "biographies-00012", ["1811"] * 5, own="1810", name="Hy Wrong")
              + [claim(M13, "biographies-00013", 1, "1820", own="1820", name="Es One"),
                 claim(M13, "biographies-00013", 2, "1821", own="1820", name="Es One")]
              + wrong_samples(M13, "biographies-00014", ["1831"], own="1830", name="Si Mention"))
SUM_COUNTS = [row("c", 1800, "counted", "9", name="Mo Merged"), row("h", 1801, "unknown", name="Mo Merged"),
              row("c", 1810, "unknown", name="Hy Wrong"), row("h", 1811, "unknown", name="Hy Wrong"),
              row("c", 1820, "unknown", name="Es One"), row("h", 1821, "zero", "0", name="Es One"),
              row("c", 1830, "counted", "5", name="Si Mention"), row("h", 1831, "unknown", name="Si Mention")]
SUM_ANSWERS = {"Mo Merged (born 1801": [ans(2)], "Hy Wrong (born 1811": [ans(3)], "Es One (born 1820": [ans(4)],
               "Si Mention (born 1831": [ans(1)]}
WRONG_LINE = "- Phrase-hit and untyped slots with a hit on wrong years only (every present row is role h): %d."
MERGED_LINE = "- Merged slots with a wrong-year hit (h_present_on_merged, not in the line above): %d."
ROLE_H_TEXT = ("role h years: the single dominant wrong year (`stability.slot_items`; tuple mode is a "
               "check only)")


def case_cli_summary_counts(env):
    """Findings 2, 5, 6: a merged slot whose wrong year hit (Mo, S2), an S2 phrase hit on its wrong
    year only (Hy), an S1 slot whose own year hit (Es) and a merged single_mention slot whose wrong
    year hit (Si)."""
    env.set_claims(SUM_CLAIMS)
    write_csv(env.counts, list(SUM_COUNTS[0]), SUM_COUNTS)
    code, out, err = env.run(env.args(), PhraseTransport(env.ft, SUM_ANSWERS))
    eq(code, 0, "summary case, s2s3: exit 0")
    sl, s, m = env.slots(), (env.out / cli.SUMMARY_NAME).read_text(encoding="utf-8"), env.meta()
    eq([(sl["biographies-000%d" % i]["class"], sl["biographies-000%d" % i]["present"]) for i in (11, 12, 13, 14)],
       [("merged", "h:1801"), ("phrase_hit", "h:1811"), ("untyped", ""), ("merged", "")],
       "s2s3: Mo merged with an h hit, Hy phrase_hit on h only; S1 and single_mention not queried")
    check(WRONG_LINE % 1 in s, "s2s3: wrong years only counts Hy, not the merged Mo")
    check(MERGED_LINE % 1 in s, "s2s3: Mo is on its own h_present_on_merged line")
    eq((m["counts"]["wrong_years_only"], m["counts"]["h_present_on_merged"]), (1, 1),
       "meta: wrong_years_only 1, h_present_on_merged 1 (no double count)")
    wrong_block = s.split(WRONG_LINE % 1)[1].split("- Merged slots")[0]
    merged_block = s.split(MERGED_LINE % 1 + " The merged type and point are kept.")[1].split("- Slots skipped")[0]
    eq((wrong_block.strip(), merged_block.strip()),
       ("- olmo2-7b biographies-00012 birth", "- olmo2-7b biographies-00011 birth"),
       "each slot list sits under its own count line: Hy under wrong years only, Mo under merged")
    check("| S1 | 1 | 0 | 0 | not run | 1 | 0 |" in s and "| `single_mention` | 1 | 0 | 1 | not run | 0 | 0 |" in s,
          "s2s3: S1 and single_mention phrase-hit cells say not run")
    check(("Scope `s2s3`, %s. Slots from `stability.slot_items` (a sample stating the right year is right). "
           "Rules C (merged type), E (phrase check) and F (F5 range); F5 over the slots with at least one wrong "
           "sample." % ROLE_H_TEXT) in s,
          "the summary header: role h the dominant wrong year, slot_items' sample rule, F5 over wrong samples")
    check(ROLE_H_TEXT in m["provenance"]["decisions"] and ROLE_H_TEXT in s.split("## (e)")[1],
          "provenance (meta and summary): role h years, tuple mode a check only")
    check(all(d in m["provenance"]["decisions"] for d in ("a sample stating the right year is right",
                                                          "F5 over the slots with at least one wrong sample; never-wrong slots on their own line")),
          "provenance decisions name the sample rule and F5's slots")
    check(not any("9 Oct" in d for d in m["provenance"]["decisions"]) and "9 Oct" not in s,
          "no h-years mode note left (dominant is now the only mode)")

    code, out, err = env.run(env.args("--force", "--scope", "all-birth"), PhraseTransport(env.ft, SUM_ANSWERS))
    eq(code, 0, "summary case, all-birth: exit 0")
    sl, s, m = env.slots(), (env.out / cli.SUMMARY_NAME).read_text(encoding="utf-8"), env.meta()
    eq([(sl["biographies-000%d" % i]["class"], sl["biographies-000%d" % i]["present"]) for i in (13, 14)],
       [("untyped", "c:1820"), ("merged", "h:1831")], "all-birth: Es and Si hit, their classes do not change")
    check("| S1 | 1 | 0 | 0 | 1 (no class change) | 1 | 0 |" in s,
          "all-birth: S1's phrase-hit cell is the slots with a present row, no class change")
    check("| `single_mention` | 1 | 0 | 1 | 1 (no class change) | 0 | 0 |" in s,
          "all-birth: single_mention's phrase-hit cell likewise")
    check("| S2 | 2 | 0 | 1 | 1 | 0 | 0 |" in s, "all-birth: S2 still counts its phrase_hit class")
    check(WRONG_LINE % 1 in s and MERGED_LINE % 2 in s,
          "all-birth: wrong years only 1 (Hy); merged with a wrong-year hit 2 (Mo, Si)")
    eq((m["counts"]["wrong_years_only"], m["counts"]["h_present_on_merged"]), (1, 2), "meta, all-birth: 1 and 2")
    eq(m["counts"]["slots_with_present_by_regime"], {"S1": 1, "S2": 2, "S3": 0, "single_mention": 1},
       "meta: slots with a present row per regime")
    eq(m["never_wrong"], dict.fromkeys(cu.MODEL_INDEX, 0), "no never-wrong slot here (Es has one wrong sample)")
    check(chr(0x2014) not in s and chr(0x2013) not in s, "summary: no em or en dashes")


old_http = (cli.HttpTransport, infinigram.HttpTransport)
old_bexec_default = cli.DEFAULT_BEXEC_ROOT
no_default = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
cli.HttpTransport = infinigram.HttpTransport = NoNetwork
# a run that forgets --bexec-root must never read a real copy: build_parser reads this on every main()
cli.DEFAULT_BEXEC_ROOT = Path(no_default.name) / "no_bexec_copy"
try:
    for label, fn in (("cli live", case_cli_live), ("cli dry run", case_cli_dry_run),
                      ("cli cache-only missing", case_cli_cache_only_missing),
                      ("cli window cache", case_cli_window_cache), ("cli inputs", case_cli_inputs),
                      ("cli h-years and scope", case_cli_h_years_and_scope), ("cli bexec root", case_cli_bexec),
                      ("cli foreign tables", case_cli_foreign_tables), ("cli atomic outputs", case_cli_atomic),
                      ("cli summary counts", case_cli_summary_counts),
                      ("cli error limits", case_cli_error_limits),
                      ("nh sensitivity", case_nh_sensitivity)):
        run_cli_case(label, fn)
finally:
    cli.HttpTransport, infinigram.HttpTransport = old_http
    cli.DEFAULT_BEXEC_ROOT = old_bexec_default
    no_default.cleanup()
eq(os.environ.get(infinigram.LOCK_DIR_ENV), old_lock_env, "LOCK_DIR_ENV restored after the CLI cases")
eq(cli.DEFAULT_BEXEC_ROOT, old_bexec_default, "the CLI's default --bexec-root restored after the CLI cases")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
