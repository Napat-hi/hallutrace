"""Tests for hallutrace/d3_figures.py, the counting behind the supplementary figures."""

import csv
import json
import os
import sys
import tempfile

from hallutrace import d3_figures as d3

OK = "   OK  "
failures = []


def check(cond, label):
    print((OK if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


META = ["generation_id", "prompt_id", "model", "sample_index", "task", "batch_id", "outcome",
        "decomposer_empty_reason", "sent", "input_row", "stripped_tail"]


def meta(path, model, task, outcomes):
    write(path, META, [["%s|%s-%05d|1" % (model, task, i), "%s-%05d" % (task, i), model, "1", task,
                        "b", o, "", "1", str(i), ""] for i, o in enumerate(outcomes)])


def drops(*reasons):
    return json.dumps([{"unit": "u%d" % i, "reason": r} for i, r in enumerate(reasons)])


tmp = tempfile.mkdtemp()
outs, ins = os.path.join(tmp, "outputs_filtered"), os.path.join(tmp, "inputs")

print("the task comes from the sidecar")
meta(os.path.join(ins, "m", "m-b1.meta.csv"), "m", "summarization", ["ok", "ok"])
check(d3.sidecar_task(os.path.join(ins, "m", "m-b1.meta.csv")) == "summarization", "one task read")
mixed = os.path.join(tmp, "mixed.meta.csv")
write(mixed, META, [["g1", "p1", "m", "1", "summarization", "b", "ok", "", "1", "0", ""],
                    ["g2", "p2", "m", "1", "simplification", "b", "ok", "", "1", "1", ""]])
try:
    d3.sidecar_task(mixed)
    check(False, "a unit with two tasks raises")
except ValueError:
    check(True, "a unit with two tasks raises")

print("kept units, counted as posthoc_filter.py counts them")
check(d3.kept_units("summarization", {"atomic_units": "A fact | False\nAnother | True\n | False"}) == 2,
      "decomposer lines: two atoms, the blank line is not one")
check(d3.kept_units("rationalization_binary", {"atomic_units": "['yes', '7']"}) == 1,
      "a bare yes or no is not a unit")
check(d3.kept_units("false_presuppositions", {"atomic_units": "[]"}) == 0, "an empty list is 0")

print("filter_drops")
DEC = ["passage", "response", "atomic_units", "hallucinated_atomic_units", "dropped_units",
       "all_units_dropped", "zero_unit_rule"]
write(os.path.join(outs, "m", "m-b1.csv"), DEC, [
    ["p", "r", "A | False\nB | True", "", drops("framing", "blank"), "False", ""],
    ["p", "r", "C | False", "", drops(), "False", ""]])
meta(os.path.join(ins, "m", "m-b2.meta.csv"), "m", "simplification", ["ok"])
write(os.path.join(outs, "m", "m-b2.csv"), DEC, [["p", "r", "", "", drops("framing", "framing"), "True",
                                                  "out_of_n_usable"]])
FP = ["response", "list", "list_condition", "generation_id", "parsed_response", "atomic_units",
      "hallucinated_atomic_units", "dropped_units", "all_units_dropped", "zero_unit_rule"]
meta(os.path.join(ins, "n", "n-b3.meta.csv"), "n", "false_presuppositions", ["ok", "abstention"])
write(os.path.join(outs, "n", "n-b3.csv"), FP, [
    ["r", "[]", "[]", "g", "x", "['zinc']", "[]", drops("not_in_response", "none_token"), "False", ""],
    ["r", "[]", "[]", "g", "x", "[]", "[]", drops(), "False", "not_hallucinated"]])
HE = ["response", "name1", "name2", "generation_id"]
meta(os.path.join(ins, "n", "n-b4.meta.csv"), "n", "historical_events", ["ok"])
write(os.path.join(outs, "n", "n-b4.csv"), HE, [["r", "a", "b", "g"]])
t = d3.filter_drops(outs, ins)
s = t[("summarization", "m")]
check(s["rows"] == 2 and s["units_kept"] == 3, "summarization: 2 rows, 3 units kept")
check(s["framing"] == 1 and s["blank"] == 1, "each drop counted under its reason, blank included")
check(s["units_raw"] == 4, "units_raw is kept plus dropped, without the blank line")
check(t[("simplification", "m")]["units_raw"] == 2 and t[("simplification", "m")]["units_kept"] == 0,
      "simplification told apart from summarization by its sidecar")
f = t[("false_presuppositions", "n")]
check(f["units_raw"] == 3 and f["not_in_response"] == 1 and f["none_token"] == 1, "FP counts")
check(("historical_events", "n") not in t, "a task outside the six is skipped")
bt = d3.by_task(t)
check(bt["summarization"]["units_raw"] == 4 and set(bt) == {"summarization", "simplification",
                                                             "false_presuppositions"}, "by_task sums")
check(all(c["units_raw"] - c["units_kept"] == sum(c[r] for r in d3.REASONS if r != "blank")
          for c in t.values()), "the reasons other than blank sum to raw minus kept")

write(os.path.join(outs, "m", "m-b9.csv"), DEC, [["p", "r", "", "", drops(), "False", ""]])
try:
    d3.filter_drops(outs, ins)
    check(False, "an output without a sidecar raises")
except FileNotFoundError:
    check(True, "an output without a sidecar raises")
os.remove(os.path.join(outs, "m", "m-b9.csv"))
write(os.path.join(outs, "m", "m-b2.csv"), DEC, [["p", "r", "", "", drops("misc"), "False", ""]])
try:
    d3.filter_drops(outs, ins)
    check(False, "an unknown drop reason raises")
except ValueError:
    check(True, "an unknown drop reason raises")

print("outcomes")
o = d3.outcomes(ins)
check(o[("summarization", "m")]["ok"] == 2 and o[("false_presuppositions", "n")]["abstention"] == 1,
      "outcomes counted per task and model")
check(("historical_events", "n") not in o, "a task outside the six is skipped")
meta(os.path.join(ins, "n", "n-b5.meta.csv"), "n", "rationalization_binary", ["mystery"])
try:
    d3.outcomes(ins)
    check(False, "an unknown outcome raises")
except ValueError:
    check(True, "an unknown outcome raises")

print("fixed order")
keys = [("simplification", "olmo2-7b"), ("false_presuppositions", "zzz"),
        ("false_presuppositions", "pythia-69b"), ("false_presuppositions", "olmo2-7b")]
check(d3.ordered(keys) == [("false_presuppositions", "olmo2-7b"), ("false_presuppositions", "pythia-69b"),
                           ("false_presuppositions", "zzz"), ("simplification", "olmo2-7b")],
      "task order first, then model order, unknown models last")

print("supplementary: F1, F3, F4")


def t1row(task, model, h, s1, aw, sm, noh):
    return {"task": task, "model": model, "h": str(h), "items": str(s1 + aw + sm + noh),
            "S1_items": str(s1), "always_wrong_items": str(aw), "single_mention_items": str(sm), "no_h_items": str(noh)}


t1 = [t1row(t, m, 0.5, 2, 1, 1, 0) for t in d3.ALL_TASKS for m in d3.MODELS]
g = d3.h_grid(t1)
check(len(g) == 40 and g[("biographies", "olmo2-7b")] == 0.5, "h_grid: one cell per task and model")
for bad, label in ((t1 + [t1[0]], "a cell given twice raises"),
                   (t1 + [t1row("gpt", "olmo2-7b", 0.1, 1, 0, 0, 0)], "an unknown task raises")):
    try:
        d3.h_grid(bad)
        check(False, label)
    except ValueError:
        check(True, label)
kap = {"task": "x", "all_rows": {"verifier": {"n": 5, "kappa": 0.41, "ci95": [0.1, 0.7], "gate": "quarantine",
       "matrix": {"row_labels": ["hallucinated", "supported"], "col_labels": ["hallucinated", "supported", "unverifiable"],
                  "cells": [[2, 0, 1], [0, 2, 0]]}}}}
acc = {"task": "y", "all_rows": {"extractor": {"n": 50, "accuracy": 0.66, "gate": "quarantine"}}}
check(d3.gate(kap) == ("kappa", 0.41, "quarantine") and d3.gate(acc) == ("accuracy", 0.66, "quarantine"),
      "gate: kappa or accuracy, as T3 reports it")
check(d3.gate_label("kappa", 0.41) == "\u03ba 0.41" and d3.gate_label("accuracy", 0.66) == "acc. 0.66", "gate labels")
c = d3.confusion(kap)
check(c["cells"] == [[2, 0, 1], [0, 2, 0]] and c["rows"] == ["hallucinated", "supported"], "confusion: verifier rows, gold columns")
for bad, label in ((acc, "an extractor task has no matrix"),):
    try:
        d3.confusion(bad)
        check(False, label)
    except ValueError:
        check(True, label)
kap["all_rows"]["verifier"]["n"] = 6
try:
    d3.confusion(kap)
    check(False, "a matrix that does not add up to n raises")
except ValueError:
    check(True, "a matrix that does not add up to n raises")
r = d3.regime_shares(t1)
check(r[("summarization", "pythia-69b")] == {"S1_items": 2, "always_wrong_items": 1, "single_mention_items": 1, "no_h_items": 0},
      "regime_shares per task and model")
bad = dict(t1[0]); bad["items"] = "9"
try:
    d3.regime_shares([bad])
    check(False, "regimes that do not add up to items raise")
except ValueError:
    check(True, "regimes that do not add up to items raise")
real = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "annotation", "e1")
if os.path.exists(os.path.join(real, "biographies_years", "agreement.json")):
    import json as _json
    yj = d3.confusion(_json.load(open(os.path.join(real, "biographies_years", "agreement.json"))))
    check(round(yj["kappa"], 3) == 0.786 and yj["gate"] == "tentative", "on annotation/e1: year judge kappa 0.786, tentative")
    gates = [d3.gate(_json.load(open(os.path.join(real, t, "agreement.json"))))[2] for t in d3.ALL_TASKS]
    check(set(gates) == {"quarantine"}, "on annotation/e1: all eight tasks quarantined")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
