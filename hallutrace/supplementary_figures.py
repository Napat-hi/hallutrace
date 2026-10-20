"""Supplementary figures and the tables behind them.

Every figure is drawn from a table this module computes, and the table is
written next to the figure as CSV, so the paper quotes the numbers the figure
shows. The counting lives here and is tested; `scripts/supplementary_figures.py` only
reads folders, calls these functions and writes files.

Two read the verifier files:

- F2, `filter_drops`: units the unit filter dropped, by reason, per task and
  model (docs/DEVIATIONS.md, 17 to 19). Read from `outputs_filtered/` (written by
  `scripts/posthoc_filter.py`) and the unit sidecars in `inputs/`.
- T2, `outcomes`: generations per outcome (ok, abstention, empty, emptied),
  per task and model, from the sidecars alone. These are the counts the rate's
  denominators are built from; h itself comes from `scripts/stability.py`.

The others read files already in this repository:

- F1, `h_grid`: h by task and model from T1
  (`results/supplementary/t1_h_by_task_model.csv`), each task with its gate
  number beside it, because every rate in it comes from a quarantined verifier.
- F3, `confusion`: each kappa-gated judge's verdicts against the adjudicated
  gold, from `annotation/verifier_check/<task>/agreement.json` (verifier rows, gold columns).
  The three extractor tasks are gated on accuracy and have no matrix; their
  accuracy is carried in the CSV and the figure's note.
- F4, `regime_shares`: items per stability regime by task and model,
  from the same T1 (S1, always wrong, single mention, no h).

A unit is one batch file, and every unit holds one task, so the task of an
output file is read from its sidecar (`{unit}.meta.csv`). The summarization
and simplification outputs carry no `generation_id`, and `posthoc.py` names
both "decompose", so the sidecar is the only place that tells them apart.
"""

import collections
import csv
import glob
import json
import os
from typing import Dict, Iterable, List, Tuple

from hallutrace import posthoc as ph

csv.field_size_limit(10 ** 9)

# Fixed orders, so a figure never reorders when a task or model is missing.
TASKS = ("false_presuppositions", "rationalization_numerical", "rationalization_binary",
         "scientific_attribution", "summarization", "simplification")
MODELS = ("olmo2-7b", "olmo17-7b", "dclm-7b", "redpajama-7b", "pythia-69b")
REASONS = (ph.NOT_IN_RESPONSE, ph.NONE_TOKEN, ph.FRAMING, ph.BLANK)
OUTCOMES = ("ok", "abstention", "empty", "emptied")


ALL_TASKS = ("biographies", "historical_events", "false_presuppositions", "rationalization_numerical",
             "rationalization_binary", "scientific_attribution", "summarization", "simplification")
# The judges F3 draws: the kappa-gated verifiers, then the Biographies year judge.
KAPPA_JUDGES = ("biographies", "historical_events", "scientific_attribution", "summarization",
                "simplification", "biographies_years")
REGIMES = ("S1_items", "always_wrong_items", "single_mention_items", "no_h_items")


def _rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sidecar_task(meta_path: str) -> str:
    """The one task a unit holds, from its sidecar. Raises if it holds more than one."""
    tasks = {r["task"] for r in _rows(meta_path)}
    if len(tasks) != 1:
        raise ValueError("%s holds %d tasks, expected 1: %s" % (meta_path, len(tasks), sorted(tasks)))
    return tasks.pop()


def kept_units(task: str, row: dict) -> int:
    """Units left in a filtered row, counted as `scripts/posthoc_filter.py` counts them."""
    if task in ("summarization", "simplification"):
        return len(ph._atom_lines(row["atomic_units"]))
    return len([u for u in ph.parse_list(row["atomic_units"]) if u not in ("yes", "no")])


def filter_drops(outputs_filtered: str, inputs: str) -> Dict[Tuple[str, str], collections.Counter]:
    """{(task, model): Counter} with rows, units_kept, units_raw and one key per drop reason.

    units_raw is kept plus dropped, leaving out `blank`: a decomposer line with
    no atom text is not a unit, and `scripts/posthoc_filter.py` does not count it
    either, so units_raw here equals its summary.csv. `blank` is still counted in
    its own column, and the other reasons sum to units_raw minus units_kept.
    """
    out: Dict[Tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    for path in sorted(glob.glob(os.path.join(outputs_filtered, "*", "*.csv"))):
        model = os.path.basename(os.path.dirname(path))
        unit = os.path.splitext(os.path.basename(path))[0]
        meta = os.path.join(inputs, model, unit + ".meta.csv")
        if not os.path.exists(meta):
            raise FileNotFoundError("no sidecar for %s (looked for %s)" % (path, meta))
        task = sidecar_task(meta)
        if task not in TASKS:
            continue
        c = out[(task, model)]
        for row in _rows(path):
            dropped = json.loads(row["dropped_units"] or "[]")
            kept = kept_units(task, row)
            c["rows"] += 1
            c["units_kept"] += kept
            c["units_raw"] += kept + sum(d["reason"] != ph.BLANK for d in dropped)
            for d in dropped:
                if d["reason"] not in REASONS:
                    raise ValueError("%s: unknown drop reason %r" % (path, d["reason"]))
                c[d["reason"]] += 1
    return dict(out)


def outcomes(inputs: str) -> Dict[Tuple[str, str], collections.Counter]:
    """{(task, model): Counter of outcomes} over every generation in the sidecars."""
    out: Dict[Tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    for meta in sorted(glob.glob(os.path.join(inputs, "*", "*.meta.csv"))):
        for r in _rows(meta):
            if r["task"] not in TASKS:
                continue
            if r["outcome"] not in OUTCOMES:
                raise ValueError("%s: unknown outcome %r" % (meta, r["outcome"]))
            out[(r["task"], r["model"])][r["outcome"]] += 1
    return dict(out)


def by_task(table: Dict[Tuple[str, str], collections.Counter]) -> Dict[str, collections.Counter]:
    """Sum a (task, model) table over models."""
    out: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for (task, _), c in table.items():
        out[task].update(c)
    return dict(out)


def ordered(keys: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """(task, model) keys in the fixed task and model order; unknown models last, by name."""
    rank = {m: i for i, m in enumerate(MODELS)}
    return sorted(keys, key=lambda k: (TASKS.index(k[0]), rank.get(k[1], len(MODELS)), k[1]))


def write_csv(path: str, header: List[str], rows: Iterable[list]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def h_grid(t1: List[dict]) -> Dict[Tuple[str, str], float]:
    """{(task, model): h} from T1. Raises on a task or model outside the fixed orders,
    or on a cell given twice."""
    out: Dict[Tuple[str, str], float] = {}
    for r in t1:
        key = (r["task"], r["model"])
        if key[0] not in ALL_TASKS or key[1] not in MODELS:
            raise ValueError("T1 row %s is outside the fixed task and model orders" % (key,))
        if key in out:
            raise ValueError("T1 gives %s twice" % (key,))
        out[key] = float(r["h"])
    return out


def gate(agreement: dict) -> Tuple[str, float, str]:
    """(measure, value, gate) of a task's judge on all rows, as T3 reports it:
    ("kappa", 0.1485, "quarantine") or ("accuracy", 0.66, "quarantine")."""
    rows = agreement["all_rows"]
    if "extractor" in rows:
        e = rows["extractor"]
        return "accuracy", float(e["accuracy"]), e["gate"]
    v = rows["verifier"]
    return "kappa", float(v["kappa"]), v["gate"]


def gate_label(measure: str, value: float) -> str:
    return ("\u03ba %.2f" if measure == "kappa" else "acc. %.2f") % value


def confusion(agreement: dict) -> dict:
    """The verifier-by-gold matrix of a kappa-gated task, with its kappa, interval and gate.
    Raises for an extractor task, which has no matrix."""
    v = agreement["all_rows"].get("verifier")
    if v is None or "matrix" not in v:
        raise ValueError("%s has no verifier matrix" % agreement.get("task"))
    m = v["matrix"]
    cells = [[int(x) for x in row] for row in m["cells"]]
    if len(cells) != len(m["row_labels"]) or any(len(r) != len(m["col_labels"]) for r in cells):
        raise ValueError("%s: matrix shape does not match its labels" % agreement.get("task"))
    if sum(map(sum, cells)) != int(v["n"]):
        raise ValueError("%s: matrix holds %d rows, n is %s" % (agreement.get("task"), sum(map(sum, cells)), v["n"]))
    return {"rows": list(m["row_labels"]), "cols": list(m["col_labels"]), "cells": cells,
            "kappa": float(v["kappa"]), "ci95": [float(x) for x in v["ci95"]], "gate": v["gate"], "n": int(v["n"])}


def regime_shares(t1: List[dict]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """{(task, model): {regime column: items}} from T1. Raises if the regimes do not add
    up to the task's items."""
    out = {}
    for r in t1:
        c = {k: int(r[k]) for k in REGIMES}
        if sum(c.values()) != int(r["items"]):
            raise ValueError("%s %s: regimes add up to %d, items is %s" % (r["task"], r["model"], sum(c.values()), r["items"]))
        out[(r["task"], r["model"])] = c
    return out
