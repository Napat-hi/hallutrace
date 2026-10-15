"""Tests for hallutrace/c_exec2.py (label records for the Biographies birth/death-year
slots, pure, no I/O), written against the interface described in hallutrace/c_exec2.py's
docstrings and CONFIDENCE_RULE.

A slot with no wrong sample (n_wrong == 0) labels no claim (each hallucination gets one label,
and labels are per claim), so label() raises on it and F5 leaves it out; it is counted on its
own per model. Confidence v2 dropped incomplete_presence and added no_count_above_0 (High
needs something found in the corpus) and dominant_in_one_sample; v3 drops
dominant_in_one_sample again (see CONFIDENCE_RULE). LABEL_COLUMNS follows the labels table's
column order, then kind and the extras.

Sections: 1 constants; 2 decision-rule cells through label(); 3 single_mention; 4 the
FABRICATION S1 flag; 5 residual and no_count_row; 6 coverage_downgrade; 7 confidence; 8 the
label record; 9 tau_rows and percentiles (synthetic rows and the two real tau files, read
only); 10 sweep_sets; 11 never-wrong slots (never_wrong, never_wrong_counts, f5_range, label
raises); 12 confidence v2 (found, dominant_samples, the two new reasons, all records).

Classified dicts come from cu.classify on hand-made slots and count rows (the fixtures were
checked against cu.classify alone before the module existed); a few boundary probes override
one or two keys of a real classify output. Every S1 fixture passed to label() has at least
one wrong sample. No network, no writes.
Run: python -m tests.test_c_exec2
"""

import copy
import csv
import os
import sys

from hallutrace import c_exec_untyped as cu
from hallutrace import c_exec2 as ce
from scripts import c_exec_untyped as scu

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


class _Raised:
    """What api() returns when the call raised: equal to nothing, so the check fails and
    shows the error instead of stopping the script."""

    def __init__(self, e):
        self.e = e

    def __eq__(self, other):
        return False

    __hash__ = None

    def __repr__(self):
        return "<raised %r>" % (self.e,)


def api(name, *args, **kw):
    """ce.<name>(*args, **kw) for the functions v2 adds; a missing name or any exception gives
    a _Raised value. Not for raises() checks: those call ce.<name> directly."""
    try:
        return getattr(ce, name)(*args, **kw)
    except Exception as e:  # noqa: BLE001
        return _Raised(e)


ROOT =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAU = 2
INF, TD, TP, KB, UND = cu.INFERENCE, cu.DATA, cu.PROCESS, cu.BOUNDARY, cu.UNDETERMINED
BH, BL, WR, FA, LT, RF = cu.BOTH_HIGH, cu.BOTH_LOW, cu.WRONG_IN_DATA, cu.FABRICATION, cu.LONG_TAIL, cu.RECALL_FAILURE
MR, ML = cu.MERGED_RECALL, cu.MERGED_LONG_TAIL
OLMO, DCLM, DOLMA, RPJ, PILE = (cu.MODEL_INDEX[m] for m in
                                ("olmo2-7b", "dclm-7b", "olmo17-7b", "redpajama-7b", "pythia-69b"))


def crow(status, n_c=""):
    """A count row as cu.classify reads it (status, n_c as a string)."""
    return {"status": status, "n_c": "" if n_c == "" else str(n_c)}


def mk(regime, n_right=None, n_wrong=None, c="auto", model="olmo2-7b", pid="biographies-00001",
       kind="birth", own=(1869,), h=(1901,)):
    """A slot dict with every field cu.slots returns (dominant mode). Defaults: S1 1 right and
    4 wrong (c 1.0), S2 5 wrong (c 1.0), S3 5 wrong (c 0.4), single_mention 1 wrong. A
    never-wrong slot (n_wrong 0, h=()) is built only for sections 8, 11 and 12, never labelled."""
    if n_wrong is None:
        n_wrong = {"S1": 4, "S2": 5, "S3": 5, "single_mention": 1}[regime]
    if n_right is None:
        n_right = 1 if regime == "S1" else 0
    if c == "auto":
        c = None if n_wrong < 2 else {"S1": 1.0, "S2": 1.0, "S3": 0.4}[regime]
    return {"model": model, "prompt_id": pid, "name": "Ada Test", "kind": kind,
            "own_years": tuple(own), "n_right": n_right, "n_wrong": n_wrong,
            "n_absent": 5 - n_right - n_wrong, "c": c, "regime": regime,
            "dominant_wrong": " ".join(str(y) for y in h), "h_years": tuple(h), "h_mode": "dominant"}


# Count rows per type at tau = 2 (n_c, n_h): checked against cu.classify alone.
ROWS = {
    BH: ([crow("counted", 3)], [crow("counted", 3)]),    # rho 0.5
    BL: ([crow("counted", 10)], [crow("counted", 1)]),   # rho 0.0909
    WR: ([crow("zero", 0)], [crow("counted", 5)]),
    FA: ([crow("zero", 0)], [crow("zero", 0)]),
    LT: ([crow("counted", 2)], [crow("zero", 0)]),       # n_c == tau
    RF: ([crow("counted", 5)], [crow("zero", 0)]),
    MR: ([crow("counted", 5)], [crow("unknown")]),       # merged, X = RECALL_FAILURE
    ML: ([crow("counted", 2)], [crow("unknown")]),       # merged, X = LONG_TAIL
    "untyped": ([crow("unknown")], [crow("unknown")]),
}
ALL_TYPES = cu.TYPES + (MR, ML)


def cl(regime, key, tau=TAU, present=frozenset(), **kw):
    c_rows, h_rows = ROWS[key]
    return cu.classify(mk(regime, **kw), c_rows, h_rows, tau, present)


def cl_rows(regime, c_rows, h_rows, tau=TAU, present=frozenset(), **kw):
    return cu.classify(mk(regime, **kw), c_rows, h_rows, tau, present)


# The labels table's columns in its order, then kind and the extras (v2)
EXPECTED_COLUMNS = ("prompt_id", "model", "claim_id", "phase2_type", "regime", "claim_rate",
                    "intervention_point", "confidence", "coverage_downgrade", "rule_version", "kind", "tau",
                    "class", "t5_column", "merged", "untyped_reason", "open_band", "flag",
                    "no_count_row", "residual", "low_n", "n_right", "n_wrong", "c", "dominant_wrong",
                    "n_c", "n_h", "rho", "f5_set", "confidence_reasons")
SCHEMA_LABEL_COLUMNS = ("prompt_id", "model", "claim_id", "phase2_type", "regime", "claim_rate", "intervention_point",
              "confidence", "coverage_downgrade", "rule_version")
made = []   # every classified dict built below, for the all-records checks of section 8


def lab(d, tau=TAU, sweep=None):
    made.append(d)
    return ce.label(d, tau, sweep)


# ---------------------------------------------------------------------------
print("1. constants")
# the module's constants and the names reused from cu
eq((cu.INFERENCE, cu.DATA, cu.PROCESS, cu.BOUNDARY, cu.UNDETERMINED),
   ("at inference", "at training data", "at training process", "at the knowledge boundary",
    "undetermined"), "the five point constants exist in cu with their values")
eq(ce.RULE_VERSION, "c_exec2-v3 (confidence rule v3, 12 Oct 2026)", "RULE_VERSION bumped to v3")
check(all(t in ce.CONFIDENCE_RULE for t in ("v3", "1 of 5 usable samples", "already per slot", "not phrase-checked")),
      "CONFIDENCE_RULE states v3 and why the 1-of-5 rule is not applied")
check(isinstance(ce.CONFIDENCE_RULE, str) and "\n" in ce.CONFIDENCE_RULE.strip(),
      "CONFIDENCE_RULE is a multi-line str")
eq(ce.MIN_USABLE, 3, "MIN_USABLE 3")
eq(ce.DOWNGRADE, {"olmo2-7b": 1, "dclm-7b": 2}, "DOWNGRADE: OLMo 2 one level, DCLM two")
eq(tuple(ce.ABSENCE_POINTS), (KB, TP), "ABSENCE_POINTS: knowledge boundary, training process")
eq(ce.OPEN_BAND, {"S3": "open: knowledge boundary or undetermined", "S2": "open: S2, any cell"}, "OPEN_BAND")
eq(ce.OPEN_BAND, {"S3": scu.OPEN_S3, "S2": scu.OPEN_S2}, "OPEN_BAND equals scripts/c_exec_untyped.py OPEN_S3 / OPEN_S2")
eq(ce.FLAG_FABRICATION_S1, "fabrication_s1", "FLAG_FABRICATION_S1")
eq((ce.HIGH, ce.MEDIUM, ce.LOW), ("High", "Medium", "Low"), "HIGH, MEDIUM, LOW")
eq((ce.LOW_ABSENCE, ce.LOW_N, ce.MED_SWEEP, ce.MED_RHO, ce.MED_C, ce.MED_ONE_RIGHT),
   ("absence_point", "low_n", "sweep_point", "s2_rho_near_half", "s3_c_near_cut", "s1_one_right"),
   "reason names kept from v1")
eq(getattr(ce, "MED_NO_COUNT", "<missing>"), "no_count_above_0", "v2 reason name: MED_NO_COUNT")
check(not hasattr(ce, "MED_ONE_SAMPLE"), "v3: no attribute MED_ONE_SAMPLE")
check(not hasattr(ce, "MED_INCOMPLETE"), "v2: no attribute MED_INCOMPLETE")
REASONS_V3 = ("absence_point", "low_n", "sweep_point", "s2_rho_near_half", "s3_c_near_cut", "s1_one_right",
              "no_count_above_0")
eq(tuple(ce.REASONS), REASONS_V3, "REASONS order (v3)")
eq(tuple(ce.LOW_REASONS), ("absence_point", "low_n"), "LOW_REASONS")
eq(tuple(ce.MEDIUM_REASONS), REASONS_V3[2:], "MEDIUM_REASONS: the other five, in order (v3)")
for name in ("REASONS", "LOW_REASONS", "MEDIUM_REASONS"):
    check("incomplete_presence" not in tuple(getattr(ce, name)), "v2: 'incomplete_presence' not in %s" % name)
    check("dominant_in_one_sample" not in tuple(getattr(ce, name)), "v3: 'dominant_in_one_sample' not in %s" % name)
eq(tuple(ce.LABEL_COLUMNS), EXPECTED_COLUMNS, "LABEL_COLUMNS: v2 order")
eq(tuple(ce.LABEL_COLUMNS)[:10], SCHEMA_LABEL_COLUMNS, "LABEL_COLUMNS begins with the labels table's columns, in its order")
eq(tuple(ce.LABEL_COLUMNS)[10], "kind", "LABEL_COLUMNS: kind right after the labels table's columns")

# ---------------------------------------------------------------------------
print("2. decision-rule cells through label()")
# the decision table and the merged types
# Hand-written from the spec table, not from cu.CELL. MYTH and Group B never occur.
SPEC = {
    "S1": {BH: INF, BL: INF, WR: INF, FA: INF, LT: INF, RF: INF, MR: INF, ML: INF},
    "S2": {BH: TD, BL: UND, WR: TD, FA: KB, LT: TP, RF: UND, MR: UND, ML: cu.OPEN},
    "S3": {BH: UND, BL: UND, WR: UND, FA: KB, LT: UND, RF: UND, MR: UND, ML: UND},
}
for regime in ("S1", "S2", "S3"):
    for t in ALL_TYPES:
        d = cl(regime, t)
        want_cls = "merged" if t in (MR, ML) else "typed"
        eq((d["class"], d["t5_column"]), (want_cls, t), "fixture %s %s: class %s" % (regime, t, want_cls))
        r = lab(d)
        eq((r["phase2_type"], r["intervention_point"]), (t, SPEC[regime][t]),
           "%s x %s -> %s" % (t, regime, SPEC[regime][t]))
r = lab(cl("S2", ML))
eq(r["f5_set"], tuple(sorted([TP, UND])), "merged LONG_TAIL in S2: f5_set (training process, undetermined)")
eq(r["open_band"], scu.OPEN_S2, "merged LONG_TAIL in S2: open band 'S2, any cell'")

# ---------------------------------------------------------------------------
print("3. single_mention")
# no point, counted per Phase 2 type
for t in ALL_TYPES:
    for model in ("olmo2-7b", "dclm-7b"):
        d = cl("single_mention", t, model=model)
        eq(d["f5_point"], "", "fixture single_mention %s: no point" % t)
        r = lab(d)
        eq((r["regime"], r["phase2_type"], r["intervention_point"], r["confidence"], r["coverage_downgrade"],
            r["confidence_reasons"], r["open_band"], r["flag"], r["residual"]),
           ("single_mention", t, "", "", 0, (), "", "", 0),
           "single_mention %s %s: type kept, no point, blank confidence, downgrade 0" % (t, model))
d = cl("single_mention", FA)
eq(ce.confidence_reasons(d, [KB, UND]), (), "single_mention: a sweep set cannot give a reason")
eq(ce.low_n(d), 0, "single_mention: low_n 0")

# ---------------------------------------------------------------------------
print("4. FABRICATION with S1: flagged")
# the flag; untyped S1 unflagged
r = lab(cl("S1", FA))
eq((r["intervention_point"], r["flag"]), (INF, "fabrication_s1"), "typed FABRICATION S1: at inference, flagged")
eq(ce.flag(cl("S1", FA)), ce.FLAG_FABRICATION_S1, "flag(): typed FABRICATION S1")
for t in (BH, BL, WR, LT, RF):
    eq(lab(cl("S1", t))["flag"], "", "S1 %s: no flag" % t)
for t in (MR, ML, "untyped"):
    d = cl("S1", t)
    eq(d["class"], "merged" if t != "untyped" else "untyped", "fixture S1 %s" % t)
    r = lab(d)
    eq((r["intervention_point"], r["flag"]), (INF, ""), "S1 %s: at inference, no flag" % t)
for regime in ("S2", "S3", "single_mention"):
    eq(lab(cl(regime, FA))["flag"], "", "FABRICATION %s: no flag (S1 only)" % regime)

# ---------------------------------------------------------------------------
print("5. residual and no_count_row")
# the residual bucket is never RECALL_FAILURE
for regime in ("S1", "S2", "S3", "single_mention"):
    d = cl_rows(regime, [None], [crow("zero", 0)])
    eq(d["class"], "no_count_row", "fixture %s: missing c row is no_count_row" % regime)
    r = lab(d)
    eq((r["phase2_type"], r["no_count_row"], r["residual"], r["intervention_point"], r["confidence"],
        r["coverage_downgrade"], r["t5_column"]), ("", 1, 0, "", "", 0, ""),
       "%s no count row: phase2_type '', no_count_row 1, residual 0, no point" % regime)
    check(r["phase2_type"] != "RECALL_FAILURE", "%s no count row: never RECALL_FAILURE" % regime)
d = cl_rows("S2", [crow("counted", 5)], [None])
eq(lab(d)["no_count_row"], 1, "missing h row: no_count_row 1")
d = cu.classify(mk("S2", own=()), [], [crow("zero", 0)], TAU)
eq((d["class"], lab(d)["no_count_row"], lab(d)["residual"]), ("no_count_row", 1, 0),
   "no own year: no_count_row 1, residual 0")
probe = dict(cl("S2", RF), t5_column="", type="")   # boundary probe: a class with no rule fired
eq(ce.label(probe, TAU)["residual"], 1, "probe: class typed with t5_column '' is residual 1")
eq(ce.label(probe, TAU)["phase2_type"], "", "probe: residual phase2_type stays '', not RECALL_FAILURE")
# every class classify can produce gives residual 0 (checked again over all records in section 8)
d_ph = cl_rows("S3", [crow("unknown")], [crow("zero", 0)], present={("c", 1869)})
eq((d_ph["class"], d_ph["f5_point"]), ("phrase_hit", UND), "fixture S3 phrase_hit: undetermined")
for name, d in (("typed", cl("S2", RF)), ("merged", cl("S2", MR)), ("phrase_hit", d_ph),
                ("untyped", cl("S3", "untyped"))):
    eq(lab(d)["residual"], 0, "class %s: residual 0" % name)

# ---------------------------------------------------------------------------
print("6. coverage_downgrade")
# coverage
for model in cu.MODEL_INDEX:
    for point in cu.POINTS + (cu.OPEN, ""):
        want = {"olmo2-7b": 1, "dclm-7b": 2}.get(model, 0) if point in (KB, TP) else 0
        eq(ce.coverage_downgrade(model, point), want, "coverage_downgrade(%s, %r) = %d" % (model, point, want))
raises(lambda: ce.coverage_downgrade("gpt-4", KB), ValueError, "unknown model: ValueError")
raises(lambda: ce.coverage_downgrade("gpt-4", ""), ValueError, "unknown model, no point: ValueError")
eq(lab(cl("S2", FA, model="dclm-7b"))["coverage_downgrade"], 2, "label: DCLM S2 FABRICATION (boundary) 2")
eq(lab(cl("S2", LT, model="olmo2-7b"))["coverage_downgrade"], 1, "label: OLMo 2 S2 LONG_TAIL (process) 1")
eq(lab(cl("S2", WR, model="dclm-7b"))["coverage_downgrade"], 0, "label: DCLM S2 WRONG_IN_DATA (data) 0")
eq(lab(cl("S2", ML, model="dclm-7b"))["coverage_downgrade"], 0, "label: DCLM open slot 0")
eq(lab(cl("S1", FA, model="dclm-7b"))["coverage_downgrade"], 0, "label: DCLM S1 FABRICATION (inference) 0")
for model in ("olmo17-7b", "redpajama-7b", "pythia-69b"):
    eq(lab(cl("S3", FA, model=model))["coverage_downgrade"], 0, "label: %s S3 FABRICATION 0" % model)
raises(lambda: ce.label(dict(cl("S2", FA), model="gpt-4"), TAU), ValueError, "label: unknown model ValueError")

# ---------------------------------------------------------------------------
print("7. confidence")
# confidence rule v2 of 12 Oct 2026
# (ce.CONFIDENCE_RULE). FABRICATION and untyped fixtures have no count above 0, so
# no_count_above_0 joins their reasons; merged and phrase_hit slots with a count above 0 do not
# get it (v1's incomplete_presence is gone). The new reasons get their own checks in section 12.
H, M, L = "High", "Medium", "Low"


def conf(d, sweep=None):
    r = lab(d, TAU, sweep)
    return r["confidence"], r["confidence_reasons"]


# High baseline: positive presence, S2, enough samples
eq(conf(cl("S2", WR)), (H, ()), "S2 WRONG_IN_DATA, 5 wrong: High, no reason")
eq(conf(cl("S2", BL)), (H, ()), "S2 BOTH_IN_DATA rho 0.09: High")
# LOW_ABSENCE
eq(conf(cl("S2", FA)), (L, ("absence_point", "no_count_above_0")),
   "S2 FABRICATION (boundary): Low, absence_point (nothing found: no_count_above_0 too)")
eq(conf(cl("S3", FA)), (L, ("absence_point", "no_count_above_0")),
   "S3 FABRICATION (boundary): Low, absence_point (nothing found: no_count_above_0 too)")
eq(conf(cl("S2", LT)), (L, ("absence_point",)), "S2 LONG_TAIL (training process): Low, absence_point")
eq(conf(cl("S2", BH)), (M, ("s2_rho_near_half",)), "S2 BOTH_IN_DATA rho 0.5 (training data): no absence_point")
# LOW_N through the key, and derived (S2/S3 only)
eq(conf(dict(cl("S2", WR, n_wrong=2), low_n=1)), (L, ("low_n",)), "low_n key 1 agreeing with S2 on 2: Low, low_n")
eq(ce.low_n(dict(cl("S2", WR), low_n=0)), 0, "low_n(): key 0 agreeing with S2 on 5: 0")
# review of 7a21a43: low_n is recomputed from the regime and n_right + n_wrong, a key must agree
raises(lambda: ce.low_n(dict(cl("S2", WR), low_n=1)), ValueError, "low_n(): key 1 on S2 with 5 samples (derived 0): ValueError")
raises(lambda: ce.low_n(dict(cl("S2", WR, n_wrong=2), low_n=0)), ValueError,
       "low_n(): key 0 on S2 with 2 samples (derived 1): ValueError")
raises(lambda: ce.label(dict(cl("S2", WR), low_n=1), TAU), ValueError, "label(): a low_n key that disagrees: ValueError")
d = cl("S2", WR, n_wrong=2)
eq(ce.low_n(d), 1, "low_n(): S2, n_right + n_wrong = 2, derived 1")
eq(conf(d), (L, ("low_n",)), "S2 WRONG_IN_DATA on 2 samples: Low")
eq(ce.low_n(cl("S2", WR, n_wrong=3)), 0, "low_n(): S2 on 3 samples, 0")
eq(conf(cl("S2", WR, n_wrong=3)), (H, ()), "S2 WRONG_IN_DATA on 3 samples: High")
d = cl("S3", WR, n_wrong=2, c=0.5)
eq(ce.low_n(d), 1, "low_n(): S3 on 2 samples, 1")
eq(conf(d), (L, ("low_n",)), "S3 WRONG_IN_DATA (undetermined) on 2 samples, c 0.5: Low (low_n)")
eq(ce.low_n(cl("S1", RF, n_right=1, n_wrong=1, c=None)), 0, "low_n(): S1 on 2 samples, 0")
eq(ce.low_n(cl("single_mention", RF)), 0, "low_n(): single_mention, 0")
# MED_SWEEP
d = cl("S2", WR)
eq(conf(d, None), (H, ()), "sweep None: sweep_point cannot fire")
eq(conf(d, {TD}), (H, ()), "sweep set {point} alone: no sweep_point")
eq(conf(d, frozenset([TD])), (H, ()), "sweep frozenset {point} alone: no sweep_point")
eq(conf(d, [TD, TD]), (H, ()), "sweep list of the point repeated: no sweep_point")
eq(conf(d, {TD, UND}), (M, ("sweep_point",)), "sweep set with a second point: Medium, sweep_point")
eq(conf(d, {UND}), (M, ("sweep_point",)), "sweep set lacking the point (union with point is 2): Medium")
eq(conf(d, {TD, cu.OPEN}), (M, ("sweep_point",)), "sweep set with 'open': Medium")
eq(conf(d, {TD, ""}), (M, ("sweep_point",)), "sweep set with '': Medium")
eq(conf(d, set()), (H, ()), "empty sweep set: union is {point}, no sweep_point")
# MED_RHO: S2, typed, BOTH_IN_DATA, 0.4 <= rho < 0.6
d = cl_rows("S2", [crow("counted", 3)], [crow("counted", 2)])
eq((d["type"], d["f5_point"]), (BL, UND), "fixture rho 0.4: BOTH_IN_DATA (rho < 0.5), undetermined")
eq(ce.rho(d), 0.4, "rho() 0.4")
eq(conf(d), (M, ("s2_rho_near_half",)), "S2 rho 0.4 (in): Medium, s2_rho_near_half")
d = cl_rows("S2", [crow("counted", 4)], [crow("counted", 2)])
eq(ce.rho(d), 0.3333, "rho() 2/6 rounded to 0.3333")
eq(conf(d), (H, ()), "S2 rho 0.3333 (below 0.4): High")
d = cl_rows("S2", [crow("counted", 41)], [crow("counted", 59)])
eq((d["type"], ce.rho(d)), (BH, 0.59), "fixture rho 0.59: high side")
eq(conf(d), (M, ("s2_rho_near_half",)), "S2 rho 0.59 (in): Medium")
d = cl_rows("S2", [crow("counted", 2)], [crow("counted", 3)])
eq((d["type"], d["f5_point"], ce.rho(d)), (BH, TD, 0.6), "fixture rho 0.6: high side, training data")
eq(conf(d), (H, ()), "S2 rho 0.6 (out): High")
d = cl_rows("S3", [crow("counted", 3)], [crow("counted", 2)])
eq(conf(d), (H, ()), "S3 rho 0.4: no s2_rho_near_half (S2 only), c 0.4: High")
d = cl_rows("S1", [crow("counted", 3)], [crow("counted", 2)], n_right=2, n_wrong=3)
eq(conf(d), (H, ()), "S1 rho 0.4: no s2_rho_near_half: High")
eq(ce.rho(cl("S2", WR)), 1.0, "rho() WRONG_IN_DATA (n_c 0, n_h 5): 1.0")
eq(ce.rho(cl("S2", FA)), None, "rho() n_c + n_h = 0: None")
eq(ce.rho(cl("S2", MR)), None, "rho() merged (h unknown): None")
eq(ce.rho(cl("S3", "untyped")), None, "rho() untyped: None")
eq(conf(cl("S2", WR)), (H, ()), "S2 WRONG_IN_DATA rho 1.0: not BOTH_IN_DATA, no s2_rho_near_half")
# MED_C: S3, 0.6 <= c < 0.8
eq(conf(cl("S3", WR, c=0.6)), (M, ("s3_c_near_cut",)), "S3 c 0.6 (in): Medium, s3_c_near_cut")
eq(conf(cl("S3", WR, c=0.75)), (M, ("s3_c_near_cut",)), "S3 c 0.75 (in): Medium")
eq(conf(cl("S3", WR, c=0.5999)), (H, ()), "S3 c 0.5999 (out): High")
eq(conf(cl("S3", WR, c=0.8)), (H, ()), "probe: S3 with c 0.8 (out): High")
eq(conf(cl("S1", RF, n_right=2, n_wrong=3, c=0.6667)), (H, ()), "S1 c 0.6667: no s3_c_near_cut (S3 only)")
# MED_ONE_RIGHT: S1, n_right 1, n_wrong >= 2, c >= 0.8
eq(conf(cl("S1", RF, n_right=1, n_wrong=4, c=1.0)), (M, ("s1_one_right",)), "S1 1 right, 4 wrong, c 1.0: Medium")
eq(conf(cl("S1", RF, n_right=1, n_wrong=2, c=1.0)), (M, ("s1_one_right",)), "S1 1 right, 2 wrong, c 1.0: Medium")
eq(conf(cl("S1", RF, n_right=1, n_wrong=4, c=0.8)), (M, ("s1_one_right",)), "probe: S1 c 0.8 (in): Medium")
eq(conf(cl("S1", RF, n_right=1, n_wrong=4, c=0.75)), (H, ()), "S1 c 0.75 (out): High")
eq(conf(cl("S1", RF, n_right=2, n_wrong=3, c=1.0)), (H, ()), "S1 n_right 2 (out): High")
eq(conf(cl("S1", RF, n_right=1, n_wrong=1, c=None)), (H, ()),
   "S1 n_wrong 1, c None: s1_one_right out; High in v3 (k 1 no longer lowers it)")
eq(conf(cl("S1", FA, n_right=1, n_wrong=4, c=1.0)), (M, ("s1_one_right", "no_count_above_0")),
   "S1 FABRICATION (at inference, flagged): no absence_point, s1_one_right and no_count_above_0")
# v1's incomplete_presence is gone: merged and phrase_hit slots with a count above 0 can be High
eq(conf(cl("S1", MR, n_right=2, n_wrong=3)), (H, ()), "merged S1 (n_c 5 counted), no other reason: High")
eq(conf(cl("S3", ML)), (H, ()), "merged S3 (undetermined, n_c 2 counted): High")
eq(conf(cl("S2", MR)), (H, ()), "merged S2 RECALL (undetermined, n_c 5 counted): High")
eq(conf(cl("S1", "untyped", n_right=2, n_wrong=3)), (M, ("no_count_above_0",)),
   "untyped S1, nothing above 0: Medium, no_count_above_0")
eq(conf(d_ph), (H, ()), "phrase_hit S3 (undetermined, c row present): High")
# open band and blank
for regime, key, pres in (("S2", ML, frozenset()), ("S3", "untyped", frozenset()), ("S2", "untyped", frozenset())):
    d = cl(regime, key, present=pres)
    eq(d["f5_point"], cu.OPEN, "fixture %s %s: open" % (regime, key))
    eq(conf(d, {UND, TP}), ("", ()), "%s %s open: blank confidence, no reasons, even with a sweep" % (regime, key))
    eq(ce.confidence_reasons(dict(d, low_n=1)), (), "%s %s open: low_n gives no reason" % (regime, key))
d = cl_rows("S2", [crow("unknown")], [crow("zero", 0)], present={("c", 1869)})
eq((d["class"], d["f5_point"]), ("phrase_hit", cu.OPEN), "fixture S2 phrase_hit: open")
eq(conf(d), ("", ()), "S2 phrase_hit open: blank")
# worst wins and several reasons in REASONS order
eq(conf(cl("S3", FA, c=0.6)), (L, ("absence_point", "s3_c_near_cut", "no_count_above_0")),
   "Low + Medium: Low (absence, c near cut, nothing found)")
eq(conf(cl("S2", WR, n_wrong=2), {TD, UND}), (L, ("low_n", "sweep_point")), "low_n + sweep: Low")
eq(conf(cl("S2", FA, n_wrong=2), {KB, UND}), (L, ("absence_point", "low_n", "sweep_point", "no_count_above_0")),
   "absence + low_n + sweep + nothing found: Low, reasons in REASONS order")
d = cl_rows("S2", [crow("counted", 3)], [crow("counted", 2)])
eq(conf(d, {UND, TP}), (M, ("sweep_point", "s2_rho_near_half")), "two Medium reasons: Medium, in order")
eq(conf(cl("S3", ML, c=0.6), {UND, KB}), (M, ("sweep_point", "s3_c_near_cut")),
   "merged S3 with sweep and c near cut: two Medium reasons in order, no incomplete_presence")
eq(conf(cl("S1", "untyped", n_right=1, n_wrong=2, c=1.0), {INF, UND}),
   (M, ("sweep_point", "s1_one_right", "no_count_above_0")), "three Medium reasons in order (v2)")
# coverage downgrade not applied on top
for model, dg in (("olmo2-7b", 1), ("dclm-7b", 2)):
    r = lab(cl("S2", FA, model=model))
    eq((r["confidence"], r["coverage_downgrade"]), (L, dg), "%s S2 FABRICATION: Low, downgrade %d recorded only" % (model, dg))
    r = lab(cl("S2", WR, model=model))
    eq((r["confidence"], r["coverage_downgrade"]), (H, 0), "%s S2 WRONG_IN_DATA: High, downgrade 0" % model)
# confidence() directly
eq(ce.confidence((), ""), "", "confidence((), ''): blank")
eq(ce.confidence(("absence_point",), cu.OPEN), "", "confidence on 'open': blank")
eq(ce.confidence((), INF), H, "confidence((), point): High")
eq(ce.confidence(("s1_one_right",), INF), M, "one Medium reason: Medium")
eq(ce.confidence(("sweep_point", "low_n"), TD), L, "Medium + Low reason: Low")
eq(ce.confidence(("absence_point",), KB), L, "absence_point: Low")
eq(ce.confidence(("no_count_above_0",), INF), M, "no_count_above_0 alone: Medium")
eq(ce.confidence(("low_n", "no_count_above_0"), UND), L, "low_n + no_count_above_0: Low")
eq(ce.confidence(("no_count_above_0",), ""), "", "no_count_above_0 on no point: blank")
eq(ce.confidence_reasons(dict(cl("S2", WR), f5_point="")), (), "probe: point '' gives no reasons")

# ---------------------------------------------------------------------------
print("8. the label record")
# label(): keys, values, no mutation; the open band names
d = cl("S2", BL)
before = copy.deepcopy(d)
r = ce.label(d, 2)
eq(list(r), list(EXPECTED_COLUMNS), "keys in LABEL_COLUMNS order")
eq(d, before, "input not mutated")
check(isinstance(d["f5_set"], frozenset), "input f5_set still a frozenset")
eq(r["model"], "olmo2-7b", "model")
eq(r["prompt_id"], "biographies-00001", "prompt_id")
eq(r["kind"], "birth", "kind")
eq(r["claim_id"], "biographies-00001|birth|1901", "claim_id prompt|kind|dominant wrong year")
eq(lab(cl("S2", BL, kind="death", h=(1950,)))["claim_id"], "biographies-00001|death|1950", "claim_id, death slot")
eq(r["rule_version"], ce.RULE_VERSION, "rule_version")
check(r["tau"] == 2.0 and isinstance(r["tau"], float), "tau recorded as float(tau)")
eq(ce.label(d, "3")["tau"], 3.0, "tau given as a string is recorded as a float, not recomputed")
eq((r["n_right"], r["n_wrong"], r["c"], r["dominant_wrong"]), (0, 5, 1.0, "1901"), "n_right, n_wrong, c, dominant_wrong")
eq((r["n_c"], r["n_h"], r["rho"]), (10.0, 1.0, 0.0909), "n_c = c_max, n_h = h_max, rho")
eq(r["f5_set"], (UND,), "f5_set as a sorted tuple")
check(isinstance(r["confidence_reasons"], tuple), "confidence_reasons a tuple")
eq((r["class"], r["merged"], r["untyped_reason"], r["low_n"]), ("typed", 0, "", 0), "class, merged, untyped_reason, low_n")
d = cl("S2", MR)
r = lab(d)
eq((r["class"], r["merged"], r["untyped_reason"], r["n_h"], r["rho"]), ("merged", 1, "", None, None),
   "merged slot: merged 1, n_h None, rho None")
for name, d in (("untyped", cl("S3", "untyped")), ("phrase_hit", d_ph)):
    r = lab(d)
    eq((r["merged"], r["untyped_reason"], r["phase2_type"]), (0, cu.UNTYPED_LABEL, cu.UNTYPED_LABEL),
       "%s: untyped_reason and phase2_type UNTYPED_LABEL" % name)
# open_band strings equal the script's
r = lab(cl("S3", "untyped"))
eq((r["intervention_point"], r["open_band"]), (cu.OPEN, scu.OPEN_S3), "S3 open slot: OPEN_S3")
r = lab(cl("S2", "untyped"))
eq((r["intervention_point"], r["open_band"]), (cu.OPEN, scu.OPEN_S2), "S2 open slot: OPEN_S2")
eq(lab(cl("S3", FA))["open_band"], "", "fixed point: open_band ''")
raises(lambda: ce.label(dict(cl("S1", RF), f5_point=cu.OPEN), TAU), ValueError,
       "probe: 'open' in a regime with no open band: ValueError")
# t5_column passes through for every class
for d in (cl("S2", RF), cl("S2", ML), cl("S3", "untyped"), d_ph, cl_rows("S2", [None], [crow("zero", 0)]),
          cl("single_mention", WR)):
    r = lab(d)
    eq((r["t5_column"], r["phase2_type"]), (d["t5_column"], d["t5_column"]),
       "t5_column and phase2_type pass through (%s, %r)" % (d["class"], d["t5_column"]))
# claim_rate
eq(ce.claim_rate(cl("S2", WR)), 1.0, "claim_rate S2 5 of 5")
eq(ce.claim_rate(cl("S3", WR, c=0.4)), 0.4, "claim_rate S3 c 0.4 of 5 wrong: 2 / 5")
eq(ce.claim_rate(cl("S1", RF, n_right=1, n_wrong=4, c=0.75)), 0.6, "claim_rate S1 1 right, c 0.75 of 4 wrong: 3 / 5")
eq(ce.claim_rate(cl("S1", RF, n_right=2, n_wrong=1, c=None)), 0.3333, "claim_rate S1 2 right, 1 wrong: 1 / 3")
eq(ce.claim_rate(cl("S3", WR, n_wrong=3, c=0.6667)), 0.6667, "claim_rate c 0.6667 of 3 wrong: round(2.0001) = 2 / 3")
eq(ce.claim_rate(cl("single_mention", WR)), 1.0, "claim_rate single_mention 1 / 1")
d = cu.classify(mk("S1", n_right=3, n_wrong=0, c=None, h=()), [crow("counted", 5)], [], TAU)
eq((d["dominant_wrong"], d["n_wrong"]), ("", 0), "fixture S1 all right (never wrong): no dominant wrong year")
eq(ce.claim_rate(d), None, "claim_rate: no dominant wrong year, None")
raises(lambda: ce.label(d, TAU), ValueError, "label: never-wrong slot labels no claim, ValueError (v2)")
eq(ce.claim_rate(dict(cl("S2", WR), n_right=0, n_wrong=0)), None, "probe: claim_rate with no stated sample, None")
eq(lab(cl("S2", WR))["claim_rate"], 1.0, "label: claim_rate")
# all records: keys, residual, value ranges
ok = all(list(ce.label(d, TAU)) == list(EXPECTED_COLUMNS) for d in made)
check(ok, "all %d records: every LABEL_COLUMNS key, in order" % len(made))
check(all(ce.label(d, TAU)["residual"] == 0 for d in made), "all records from classify: residual 0")
check(all(ce.label(d, TAU)["confidence"] in ("", H, M, L) for d in made), "all records: confidence in '', High, Medium, Low")
check(all(ce.label(d, TAU)["coverage_downgrade"] in (0, 1, 2) for d in made), "all records: coverage_downgrade 0, 1 or 2")
check(all((ce.label(d, TAU)["confidence"] == "") == (d["f5_point"] in ("", cu.OPEN)) for d in made),
      "all records: blank confidence exactly when there is no point or the point is open")

# ---------------------------------------------------------------------------
print("9. tau_rows and percentiles")
# tau_rows and percentiles; real files results/c_exec/he_tau.csv and results/bio_counts/merged/tau.csv
IDX = list(cu.MODEL_INDEX.values())


def trow(index, pct, tau, variant=None):
    r = {"index": index, "model": "m", "percentile": str(pct), "tau": str(tau)}
    if variant is not None:
        r = dict({"variant": variant}, **r)
    return r


V = [trow(i, p, t, "headline_406") for p in (5, 10) for i, t in zip(IDX, (1, 1, 1, 2, 3))]
V += [trow(i, 10, 9, "with_short_444") for i in IDX]            # same index, another variant
V += [trow("v4_other_index", 10, 7, "headline_406")]          # index not in MODEL_INDEX, ignored
N = [trow(i, p, t) for p in (5, 10) for i, t in zip(IDX, (1, 2, 1, 1, 3))]
N += [trow("v4_other_index", 10, 7)]
got = ce.tau_rows(V, 10, variant="headline_406")
eq(got, dict(zip(IDX, (1.0, 1.0, 1.0, 2.0, 3.0))), "variant file: headline p10")
eq(list(got), IDX, "keys in MODEL_INDEX order")
check(all(isinstance(v, float) for v in got.values()), "values are floats")
eq(ce.tau_rows(V, 10, variant="with_short_444"), dict.fromkeys(IDX, 9.0), "variant file: other variant, no duplicate")
eq(ce.tau_rows(N, 10), dict(zip(IDX, (1.0, 2.0, 1.0, 1.0, 3.0))), "no-variant file: p10")
eq(list(ce.tau_rows(N, 10)), IDX, "no-variant file: MODEL_INDEX order")
eq(ce.tau_rows([dict(r, percentile=" 10 ") if r["percentile"] == "10" else r for r in N], 10), ce.tau_rows(N, 10),
   "percentile stripped")
eq(ce.tau_rows(N + [trow(OLMO, 6, "abc")], 10), ce.tau_rows(N, 10), "non-numeric tau at another percentile: ignored")
eq(ce.tau_rows([dict(r, tau="1.5") if r["index"] == RPJ else r for r in N], 10)[RPJ], 1.5, "tau 1.5 kept")
eq(ce.tau_rows(N, 10, expected=dict(zip(IDX, (1.0, 2.0, 1.0, 1.0, 3.0)))), dict(zip(IDX, (1.0, 2.0, 1.0, 1.0, 3.0))),
   "expected that matches: no error")
raises(lambda: ce.tau_rows([r for r in N if r["index"] != PILE], 10), ValueError, "missing index: ValueError")
raises(lambda: ce.tau_rows(N, 11), ValueError, "no row at that percentile: ValueError (missing)")
raises(lambda: ce.tau_rows(N + [trow(DCLM, 10, 2)], 10), ValueError, "duplicate index: ValueError")
raises(lambda: ce.tau_rows(V + [trow(DCLM, 10, 1, "headline_406")], 10, variant="headline_406"), ValueError,
       "duplicate index within the variant: ValueError")
raises(lambda: ce.tau_rows([dict(r, tau="") if r["index"] == DOLMA and r["percentile"] == "10" else r for r in N], 10),
       ValueError, "empty tau: ValueError")
raises(lambda: ce.tau_rows([dict(r, tau="abc") if r["index"] == DOLMA and r["percentile"] == "10" else r for r in N], 10),
       ValueError, "text tau: ValueError")
# review of 7a21a43: float() accepts these strings, the reader must still refuse them
for bad in ("nan", "NaN", "inf", "-inf", "Infinity"):
    raises(lambda bad=bad: ce.tau_rows([dict(r, tau=bad) if r["index"] == DOLMA and r["percentile"] == "10" else r
                                        for r in N], 10), ValueError, "tau %r: ValueError (not a finite number)" % bad)
raises(lambda: ce.tau_rows(N, 10, expected=dict(zip(IDX, (1.0, 1.0, 1.0, 1.0, 3.0)))), ValueError,
       "unexpected value vs expected: ValueError")
raises(lambda: ce.tau_rows(N, 10, variant="headline_406"), ValueError, "variant given, file has no variant column: ValueError")
raises(lambda: ce.tau_rows(V, 10), ValueError, "variant None, file has a variant column: ValueError")
raises(lambda: ce.tau_rows(V + [trow(OLMO, 20, 1)], 10, variant="headline_406"), ValueError,
       "variant given, one row lacks a variant key: ValueError")
raises(lambda: ce.tau_rows(N + [trow(OLMO, 20, 1, "headline_406")], 10), ValueError,
       "variant None, one row has a variant key: ValueError")
eq(ce.percentiles(N), [5, 10], "percentiles: sorted distinct ints")
eq(ce.percentiles(V, "headline_406"), [5, 10], "percentiles filtered by variant")
eq(ce.percentiles(V, "with_short_444"), [10], "percentiles of another variant")


def read_csv(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


he = read_csv(os.path.join("results", "c_exec", "he_tau.csv"))
mg = read_csv(os.path.join("results", "bio_counts", "merged", "tau.csv"))
HE10 = {OLMO: 1.0, DCLM: 1.0, DOLMA: 1.0, RPJ: 2.0, PILE: 3.0}
MG10 = {OLMO: 1.0, DCLM: 2.0, DOLMA: 1.0, RPJ: 1.0, PILE: 3.0}
eq(ce.tau_rows(he, 10, variant="headline_406"), HE10, "he_tau.csv headline_406 p10: 1, 1, 1, 2, 3")
eq(ce.tau_rows(he, 10, variant="headline_406", expected=HE10), HE10, "he_tau.csv p10 with expected")
eq(ce.tau_rows(mg, 10), MG10, "merged/tau.csv p10: 1, 2, 1, 1, 3")
eq(ce.tau_rows(mg, 10, expected=MG10), MG10, "merged/tau.csv p10 with expected")
raises(lambda: ce.tau_rows(he, 10), ValueError, "he_tau.csv with variant None: ValueError")
raises(lambda: ce.tau_rows(mg, 10, variant="headline_406"), ValueError, "merged/tau.csv with a variant: ValueError")
eq(ce.percentiles(he, "headline_406"), list(range(5, 21)), "he_tau.csv headline percentiles 5..20")
eq(ce.percentiles(he), list(range(5, 21)), "he_tau.csv percentiles 5..20 (all variants)")
eq(ce.percentiles(mg), list(range(5, 21)), "merged/tau.csv percentiles 5..20")
check(all(len(ce.tau_rows(he, p, variant="headline_406")) == 5 for p in range(5, 21)),
      "he_tau.csv headline: five indexes at every percentile 5..20")
check(all(len(ce.tau_rows(mg, p)) == 5 for p in range(5, 21)), "merged/tau.csv: five indexes at every percentile")

# ---------------------------------------------------------------------------
print("10. sweep_sets")
# sweep_points in confidence_reasons; sweep_sets
a1 = cl_rows("S2", [crow("counted", 3)], [crow("zero", 0)], tau=1)     # RECALL_FAILURE: undetermined
a5 = cl_rows("S2", [crow("counted", 3)], [crow("zero", 0)], tau=5)     # LONG_TAIL: training process
eq((a1["f5_point"], a5["f5_point"]), (UND, TP), "fixture: n_c 3 is undetermined at tau 1, training process at tau 5")
b1, b5 = cl("S2", WR, pid="biographies-00002", tau=1), cl("S2", WR, pid="biographies-00002", tau=5)
c1 = cl("S3", "untyped", pid="biographies-00003", tau=1)
c5 = cl("S3", "untyped", pid="biographies-00003", tau=5)
by_p = {5: [a1, b1, c1], 10: [a1, b1, c1], 20: [c5, b5, a5]}
before = copy.deepcopy(by_p)
ss = ce.sweep_sets(by_p)
eq(ss, {cu.slot_key(a1): frozenset([UND, TP]), cu.slot_key(b1): frozenset([TD]),
        cu.slot_key(c1): frozenset([cu.OPEN])}, "sweep_sets: union of f5_point over percentiles")
check(all(isinstance(v, frozenset) for v in ss.values()), "values are frozensets")
eq(by_p, before, "input not mutated")
eq(conf(a5, ss[cu.slot_key(a5)]), (L, ("absence_point", "sweep_point")), "slot whose point moves: sweep_point fires")
eq(conf(b5, ss[cu.slot_key(b5)]), (H, ()), "slot whose point holds: no sweep_point")
eq(ce.sweep_sets({10: [a1, b1]}), {cu.slot_key(a1): frozenset([UND]), cu.slot_key(b1): frozenset([TD])},
   "one percentile")
raises(lambda: ce.sweep_sets({5: [a1, b1], 10: [a1]}), ValueError, "a slot missing at one percentile: ValueError")
raises(lambda: ce.sweep_sets({5: [a1], 10: [a1, b1]}), ValueError, "an extra slot at one percentile: ValueError")
raises(lambda: ce.sweep_sets({5: [a1, b1], 10: [a1, c1]}), ValueError, "different slots, same count: ValueError")
raises(lambda: ce.sweep_sets({5: [a1, a5], 10: [a1, a5]}), ValueError, "a slot key repeated in one percentile: ValueError")

# ---------------------------------------------------------------------------
print("11. never-wrong slots")
# Each hallucination gets one point and labels.csv is per claim: a slot with no wrong sample (n_wrong 0, dominant_wrong "") labels no claim. label()
# raises on it, F5 leaves it out, and it is counted on its own line per model.
NW = dict(n_wrong=0, c=None, h=())
nw_rf = cu.classify(mk("S1", n_right=3, pid="biographies-00101", **NW), [crow("counted", 5)], [], TAU)
nw_un = cu.classify(mk("S1", n_right=2, model="pythia-69b", pid="biographies-00102", **NW), [crow("unknown")], [], TAU)
nw_nc = cu.classify(mk("S1", n_right=1, model="dclm-7b", pid="biographies-00108", **NW), [None], [], TAU)
eq((nw_rf["class"], nw_rf["type"], nw_rf["f5_point"], nw_rf["dominant_wrong"]), ("typed", RF, INF, ""),
   "fixture never-wrong S1 typed RECALL_FAILURE: at inference, no dominant wrong year")
eq((nw_un["class"], nw_un["f5_point"]), ("untyped", INF), "fixture never-wrong S1 untyped: at inference")
eq((nw_nc["class"], nw_nc["f5_point"]), ("no_count_row", ""), "fixture never-wrong S1 with no count row: no point")
# never_wrong()
for name, d in (("typed", nw_rf), ("untyped", nw_un), ("no_count_row", nw_nc)):
    eq(api("never_wrong", d), True, "never_wrong(): n_wrong 0 (%s), True" % name)
for name, d in (("S1 4 right, 1 wrong", cl("S1", RF, n_right=4, n_wrong=1, c=None)), ("S1 1 right, 4 wrong", cl("S1", RF)),
                ("S2", cl("S2", WR)), ("S3 open", cl("S3", "untyped")), ("single_mention", cl("single_mention", FA))):
    eq(api("never_wrong", d), False, "never_wrong(): %s, False" % name)
eq(api("never_wrong", dict(cl("S1", RF), n_wrong=0)), True, "probe: never_wrong() is n_wrong == 0 (dominant_wrong left set)")
# label() raises on a never-wrong slot; one wrong sample is enough to be labelled
for name, d in (("typed", nw_rf), ("untyped", nw_un), ("no_count_row", nw_nc)):
    raises(lambda d=d: ce.label(d, TAU), ValueError, "label(): never-wrong %s slot, ValueError" % name)
raises(lambda: ce.label(dict(cl("S1", RF), n_wrong=0), TAU), ValueError,
       "probe: label() raises on n_wrong 0 even with a dominant_wrong left set")
r = lab(cl("S1", RF, n_right=4, n_wrong=1, c=None))
eq((r["n_wrong"], r["claim_id"], r["intervention_point"], r["claim_rate"]), (1, "biographies-00001|birth|1901", INF, 0.2),
   "label(): S1 with 4 right and 1 wrong is labelled (claim_rate 1 / 5)")
# never_wrong_counts(): every model in MODEL_INDEX order, zeros included
w_s1 = cl("S1", RF, pid="biographies-00103")
w_s2 = cl("S2", WR, pid="biographies-00104")
w_open = cl("S3", "untyped", pid="biographies-00105")
w_sm = cl("single_mention", RF, pid="biographies-00106")
w_ncr = cl_rows("S2", [None], [crow("zero", 0)], pid="biographies-00107")
POP = [nw_rf, nw_un, w_s1, w_s2, w_open, w_sm, w_ncr, nw_nc]
before = copy.deepcopy(POP)
got = api("never_wrong_counts", POP)
eq(got, {"olmo2-7b": 1, "dclm-7b": 1, "olmo17-7b": 0, "redpajama-7b": 0, "pythia-69b": 1},
   "never_wrong_counts(): per model, never-wrong slots of every class (no_count_row too)")
eq(list(got) if isinstance(got, dict) else got, list(cu.MODEL_INDEX), "never_wrong_counts(): keys in MODEL_INDEX order")
eq(api("never_wrong_counts", []), dict.fromkeys(cu.MODEL_INDEX, 0), "never_wrong_counts([]): every model, 0")
two = [nw_rf, dict(nw_rf, prompt_id="biographies-00109"), w_s1]
eq(api("never_wrong_counts", two), dict(dict.fromkeys(cu.MODEL_INDEX, 0), **{"olmo2-7b": 2}),
   "never_wrong_counts(): two never-wrong olmo2 slots and one wrong one: olmo2 2, the rest 0")
raises(lambda: ce.never_wrong_counts([nw_rf, dict(nw_un, model="gpt-4")]), ValueError,
       "never_wrong_counts(): a never-wrong slot of an unknown model, ValueError")
raises(lambda: ce.never_wrong_counts([nw_rf, dict(w_s2, model="gpt-4")]), ValueError,
       "never_wrong_counts(): any slot whose model is not in MODEL_INDEX (one with wrong samples), ValueError")
# f5_range(): cu.f5_range over the slots with at least one wrong sample, same return shape


def n_and_inference(fr):
    """(n, slots fixed at inference) of an f5_range result, or _Raised."""
    try:
        return fr["n"], next(row["lower"] for row in fr["rows"] if row["point"] == INF)
    except Exception as e:  # noqa: BLE001
        return _Raised(e)


wrong_only = [s for s in POP if s["n_wrong"] > 0]
eq(n_and_inference(cu.f5_range(POP)), (5, 3), "fixture: cu.f5_range over every slot: n 5, at inference 3")
eq(n_and_inference(cu.f5_range(wrong_only)), (3, 1), "fixture: cu.f5_range over slots with a wrong sample: n 3, at inference 1")
got = api("f5_range", POP)
eq(n_and_inference(got), (3, 1), "f5_range(): n 3 and at inference 1 (never-wrong left out; cu gives 5 and 3)")
eq(got, cu.f5_range(wrong_only), "f5_range(): equals cu.f5_range over the slots with at least one wrong sample")
eq(sorted(got) if isinstance(got, dict) else got, sorted(cu.f5_range(POP)), "f5_range(): same keys as cu.f5_range")
eq(api("f5_range", [w_s1, w_s2, w_open]), cu.f5_range([w_s1, w_s2, w_open]), "f5_range(): no never-wrong slot, same as cu.f5_range")
eq(POP, before, "never_wrong_counts and f5_range: input not mutated")

# ---------------------------------------------------------------------------
print("12. confidence v3: found, dominant_samples, no_count_above_0; dominant_in_one_sample dropped")
# High needs a positive presence signal (something was found in the corpus); dominant_samples
# stays for claim_rate; v3 does not apply the 1-of-5 rule
d_c = cl_rows("S2", [crow("counted", 5)], [crow("zero", 0)])
d_h = cl("S2", WR)
d_hp = cl_rows("S3", [crow("unknown")], [crow("unknown")], present={("h", 1901)})
d_tok = cl_rows("S3", [crow("unknown"), crow("unknown")], [crow("zero", 0)], own=(1869, 1870), present={("c", 1870)})
d_two = cl_rows("S3", [crow("unknown"), crow("counted", 3)], [crow("unknown")], own=(1869, 1870))
d_uz = cl_rows("S3", [crow("unknown")], [crow("zero", 0)])
d_zu = cl_rows("S1", [crow("zero", 0)], [crow("unknown")], n_right=2, n_wrong=3)
eq((d_c["c_max"], d_c["h_max"], d_h["c_max"], d_h["h_max"]), (5.0, 0.0, 0.0, 5.0), "fixtures: c_max 5 / h_max 0, c_max 0 / h_max 5")
eq((d_ph["c_status"], d_ph["c_max"], d_ph["h_max"]), ("present", None, 0.0), "fixture phrase_hit: c row present, no count above 0")
eq((d_hp["class"], d_hp["c_status"], d_hp["h_status"], d_hp["f5_point"]), ("phrase_hit", "unknown", "present", UND),
   "fixture S3 phrase_hit, h row present: undetermined")
eq((d_tok["class"], d_tok["c_status"], d_tok["c_max"], d_tok["f5_point"]), ("phrase_hit", "unknown present", None, UND),
   "fixture S3 phrase_hit, c status 'unknown present': undetermined")
eq((d_two["class"], d_two["c_status"], d_two["c_max"], d_two["f5_point"]), ("untyped", "unknown counted", 3.0, UND),
   "fixture S3 untyped, one own year unknown, one counted 3: undetermined")
eq((d_uz["class"], d_uz["c_max"], d_uz["h_max"], d_uz["f5_point"]), ("untyped", None, 0.0, cu.OPEN),
   "fixture S3 untyped, c unknown, h zero: open")
eq((d_zu["class"], d_zu["c_max"], d_zu["h_max"], d_zu["f5_point"]), ("untyped", 0.0, None, INF),
   "fixture S1 untyped, c zero, h unknown: at inference")
# found(): some count known to be above 0; class does not matter
for name, d, want in (("c_max 5 > 0", d_c, True), ("h_max 5 > 0, c_max 0", d_h, True),
                      ("'present' in c_status", d_ph, True), ("'present' in h_status", d_hp, True),
                      ("'present' a token of 'unknown present'", d_tok, True),
                      ("untyped, an own year counted 3 next to an unknown one", d_two, True),
                      ("merged, n_c counted 5, h unknown", cl("S2", MR), True),
                      ("typed BOTH_IN_DATA", cl("S2", BH), True),
                      ("all zero (typed FABRICATION)", cl("S2", FA), False),
                      ("all unknown (untyped)", cl("S3", "untyped"), False),
                      ("c unknown, h zero", d_uz, False), ("c zero, h unknown", d_zu, False)):
    eq(api("found", d), want, "found(): %s -> %s" % (name, want))
# MED_NO_COUNT: fires exactly when nothing is found
r = lab(cl("S1", FA, n_right=2, n_wrong=3))
eq((r["intervention_point"], r["confidence"], r["confidence_reasons"], r["flag"]),
   (INF, M, ("no_count_above_0",), "fabrication_s1"),
   "typed FABRICATION S1 (High in v1): Medium, no_count_above_0, still flagged fabrication_s1")
eq(conf(cl("S1", "untyped", n_right=2, n_wrong=3)), (M, ("no_count_above_0",)), "untyped S1, nothing above 0: Medium")
eq(conf(d_zu), (M, ("no_count_above_0",)), "untyped S1, c zero and h unknown: Medium, no_count_above_0")
eq(conf(cl("S1", MR, n_right=2, n_wrong=3)), (H, ()), "merged S1 (n_c counted 5), no other reason: High")
eq(conf(cl("S1", ML, n_right=2, n_wrong=3)), (H, ()), "merged S1 (n_c counted 2), no other reason: High")
eq(conf(cl_rows("S1", [crow("unknown"), crow("counted", 3)], [crow("unknown")], own=(1869, 1870), n_right=2, n_wrong=3)),
   (H, ()), "untyped S1, an own year counted 3 next to an unknown one: found, High")
eq(conf(d_two), (H, ()), "untyped S3 (undetermined), an own year counted 3 next to an unknown one: High")
eq(conf(d_hp), (H, ()), "phrase_hit S3, h row present: High")
eq(conf(d_tok), (H, ()), "phrase_hit S3, one of two c rows present: High")
eq(conf(cl("S2", WR)), (H, ()), "typed WRONG_IN_DATA (h counted 5, c zero): found, High")
# dominant_samples(): k = round(c * n_wrong), 1 when n_wrong 1, else 0; claim_rate = k / stated
for name, d, k in (("c None, n_wrong 1 (S1)", cl("S1", RF, n_right=1, n_wrong=1, c=None), 1),
                   ("c None, n_wrong 1 (single_mention)", cl("single_mention", RF), 1),
                   ("c 0.5, n_wrong 2", cl("S3", WR, n_wrong=2, c=0.5), 1),
                   ("c 0.3333, n_wrong 3", cl("S3", WR, n_wrong=3, c=0.3333), 1),
                   ("c 1.0, n_wrong 2", cl("S2", WR, n_wrong=2), 2),
                   ("c 0.4, n_wrong 5", cl("S3", WR), 2),
                   ("c 0.6667, n_wrong 3", cl("S3", WR, n_wrong=3, c=0.6667), 2),
                   ("c 1.0, n_wrong 5", cl("S2", WR), 5),
                   ("S1 1 right, c 0.75 of 4 wrong", cl("S1", RF, n_right=1, n_wrong=4, c=0.75), 3),
                   ("never wrong: c None, n_wrong 0", nw_rf, 0)):
    eq(api("dominant_samples", d), k, "dominant_samples(): %s -> %d" % (name, k))
    n = d["n_right"] + d["n_wrong"]
    want = round(k / n, 4) if d["dominant_wrong"] and n else None
    eq(ce.claim_rate(d), want, "claim_rate() consistent with k %d: %s -> %r" % (k, name, want))
# review of 7a21a43: c is computed from 2 wrong samples on, so a missing c there is an error, not k 0
d = dict(cl("S3", WR, n_wrong=2, c=0.5), c=None)
raises(lambda: ce.dominant_samples(d), ValueError, "dominant_samples(): c None on 2 wrong samples: ValueError")
raises(lambda: ce.claim_rate(d), ValueError, "claim_rate(): c None on 2 wrong samples: ValueError")
raises(lambda: ce.dominant_samples(dict(cl("S1", RF, n_right=1, n_wrong=4, c=1.0), c=None)), ValueError,
       "dominant_samples(): c None on 4 wrong samples (S1): ValueError")
# v3: k 1 alone no longer lowers confidence (dominant_in_one_sample dropped)
eq(conf(cl("S3", WR, n_wrong=3, c=0.3333)), (H, ()), "S3 on 3 samples, k 1: High in v3")
eq(conf(cl("S3", WR, n_wrong=4, c=0.5)), (H, ()), "S3 on 4 samples, k 2: High")
eq(conf(cl("S1", RF, n_right=4, n_wrong=1, c=None)), (H, ()), "S1 4 right, 1 wrong (k 1): High in v3")
eq(conf(cl("S1", RF, n_right=3, n_wrong=2, c=0.5)), (H, ()), "S1 3 right, 2 wrong years apart (k 1): High in v3")
eq(conf(cl("S1", RF, n_right=3, n_wrong=2, c=1.0)), (H, ()), "S1 3 right, 2 wrong on one year (k 2): High")
# worst wins, REASONS order, blank on no point
eq(conf(cl("S3", WR, n_wrong=2, c=0.5)), (L, ("low_n",)), "worst wins: low_n S3 slot with k 1 is Low")
eq(conf(cl("S3", FA, n_wrong=3, c=0.3333), {KB, UND}),
   (L, ("absence_point", "sweep_point", "no_count_above_0")),
   "absence + sweep + nothing found: Low, reasons in REASONS order")
eq(conf(cl("S1", "untyped", n_right=1, n_wrong=1, c=None)), (M, ("no_count_above_0",)),
   "untyped S1, 1 right and 1 wrong: Medium via no_count_above_0 only")
for name, d in (("single_mention untyped (k 1, nothing found)", cl("single_mention", "untyped")),
                ("single_mention FABRICATION", cl("single_mention", FA)),
                ("S3 untyped open, k 1", cl("S3", "untyped", n_wrong=3, c=0.3333)),
                ("S2 no count row, low_n", cl_rows("S2", [None], [crow("zero", 0)], n_wrong=2))):
    eq(conf(d), ("", ()), "no point (%s): blank confidence, no reasons" % name)
# all records built in sections 2 to 12
recs = [ce.label(d, TAU) for d in made]
check(all(d["n_wrong"] >= 1 for d in made), "all %d labelled fixtures have at least one wrong sample" % len(made))
check(all(list(r) == list(EXPECTED_COLUMNS) for r in recs), "all records: keys in v2 LABEL_COLUMNS order")
check(all("incomplete_presence" not in r["confidence_reasons"] for r in recs),
      "all records: 'incomplete_presence' in no confidence_reasons")
check(all(list(r["confidence_reasons"]) == [x for x in REASONS_V3 if x in r["confidence_reasons"]] for r in recs),
      "all records: reasons are v3 REASONS names, in REASONS order, no repeats")
check(all((r["confidence"] == "") == (d["f5_point"] in ("", cu.OPEN)) for d, r in zip(made, recs)),
      "all records: blank confidence exactly when there is no point or the point is open")
check(all(r["confidence"] != H or api("found", d) is True for d, r in zip(made, recs)),
      "all records: High only with something found")

# ---------------------------------------------------------------------------
print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("all passed")
