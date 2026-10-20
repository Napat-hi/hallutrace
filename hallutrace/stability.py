"""The stability signal, the h side: how often each item is hallucinated over five samples.

Per item (task, model, prompt) over the five sampled generations, sample_index 1 to 5. The
greedy pass (sample_index 0) stays out of h and is kept as `greedy_hallucinated`.

    n_usable        samples that count. Out: abstained (false_presuppositions' "no response"
                    included, which is tagged as an abstention), empty, emptied by the tail rule,
                    and a zero-unit sample on the four tasks where that is no answer
    n_hallucinated  usable samples with at least one hallucinated unit
    h               n_hallucinated / n_usable; none when n_usable = 0 (no h, no regime)
    n_zero_unit     zero-unit samples left out, so HALoGEN's
                    n_hallucinated / (n_usable + n_zero_unit) can be recomputed

Zero-unit samples, per task: out on summarization, simplification
and scientific_attribution. On rationalization_numerical too, except a sample whose first
non-empty line holds the number 0, which is scored as the answer 0 (HALoGEN's
`parse_zero_response`): not hallucinated when the gold count is 0, hallucinated otherwise.
Elsewhere a zero-unit sample counts as not hallucinated and stays in.

The regime from h alone, for items on every task: S1 when h < 1. When
h = 1 the item is always wrong. With exactly one usable sample it is `single_mention`: one
sample gives c = 1 by construction, so consistency can't be judged. With 2 usable samples it
is flagged `low_n` (Low confidence, regime kept). Off Biographies slots `cluster_method` is
`none`, so items get no c and no S2/S3 split; Biographies' years get theirs per slot below.

A sample is a dict: task, model, prompt_id, sample_index, outcome, n_units (None if never
decomposed), hallucinated, and either `zero_unit_rule` (posthoc's label, from the loader below)
or, for rationalization_numerical, first_line and gold_count.
"""
from __future__ import annotations

import csv
import glob
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

OUT_OUTCOMES = ("abstention", "empty", "emptied", "missing")   # missing: an ok generation whose unit has no output
ZERO_UNIT_OUT = ("summarization", "simplification", "scientific_attribution", "rationalization_numerical")
RN = "rationalization_numerical"
MIN_USABLE = 3   # S2 and S3 below this many usable samples are Low
MIN_WRONG = 2    # c needs this many wrong samples; one alone is `single_mention`
SINGLE_MENTION = "single_mention"
SAMPLED = (1, 2, 3, 4, 5)
GREEDY = 0

HALLUCINATED, NOT_HALLUCINATED, ZERO_UNIT = "hallucinated", "not_hallucinated", "zero_unit"

# posthoc.zero_unit_rule's labels (the same rule as below) as statuses
ZERO_UNIT_LABELS = {"not_hallucinated": NOT_HALLUCINATED, "out_of_n_usable": ZERO_UNIT,
                    "answer_0_right": NOT_HALLUCINATED, "answer_0_wrong": HALLUCINATED}

ITEM_COLUMNS = ("task", "model", "prompt_id", "n_samples", "n_usable", "n_hallucinated", "h",
                "n_zero_unit", "n_abstention", "n_empty", "n_emptied", "n_missing", "regime_h", "low_n",
                "greedy_hallucinated", "cluster_method")


def first_line(text: Optional[str]) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def says_zero(line: str) -> bool:
    """HALoGEN's check for a 0 answer, on the first non-empty line."""
    return "0" in re.findall(r"\d+", line or "")


def status(s: dict) -> str:
    """One sample: an outcome that leaves h, ZERO_UNIT, HALLUCINATED or NOT_HALLUCINATED."""
    if s["outcome"] in OUT_OUTCOMES:
        return s["outcome"]
    if s["outcome"] != "ok":
        raise ValueError("unknown outcome %r for %s" % (s["outcome"], s.get("prompt_id")))
    if s.get("n_units") is None:
        raise ValueError("%s sample %s is ok but has no verifier result" % (s["prompt_id"], s["sample_index"]))
    if s["n_units"] == 0:
        if s.get("zero_unit_rule"):   # posthoc already labelled the row with the zero-unit rule
            return ZERO_UNIT_LABELS[s["zero_unit_rule"]]
        if s["task"] == RN and says_zero(s.get("first_line", "")):
            return NOT_HALLUCINATED if int(s["gold_count"]) == 0 else HALLUCINATED
        return ZERO_UNIT if s["task"] in ZERO_UNIT_OUT else NOT_HALLUCINATED
    return HALLUCINATED if s["hallucinated"] else NOT_HALLUCINATED


def items(samples: Iterable[dict]) -> List[dict]:
    """One row per (task, model, prompt), ITEM_COLUMNS."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for s in samples:
        groups[(s["task"], s["model"], s["prompt_id"])].append(s)
    out = []
    for (task, model, pid), ss in sorted(groups.items()):
        by_index = {int(s["sample_index"]): s for s in ss}
        st = [status(by_index[i]) for i in SAMPLED if i in by_index]
        n_h, n_ok = st.count(HALLUCINATED), st.count(NOT_HALLUCINATED)
        n_usable = n_h + n_ok
        h = n_h / n_usable if n_usable else None
        if h is None:
            regime = ""
        elif h < 1:
            regime = "S1"
        else:
            regime = SINGLE_MENTION if n_usable < MIN_WRONG else "always_wrong"
        greedy = by_index.get(GREEDY)
        g = status(greedy) if greedy else None
        out.append({
            "task": task, "model": model, "prompt_id": pid, "n_samples": len(st),
            "n_usable": n_usable, "n_hallucinated": n_h, "h": None if h is None else round(h, 4),
            "n_zero_unit": st.count(ZERO_UNIT), "n_abstention": st.count("abstention"),
            "n_empty": st.count("empty"), "n_emptied": st.count("emptied"), "n_missing": st.count("missing"),
            "regime_h": regime, "low_n": int(regime == "always_wrong" and n_usable < MIN_USABLE),
            "greedy_hallucinated": "" if g not in (HALLUCINATED, NOT_HALLUCINATED) else int(g == HALLUCINATED),
            "cluster_method": "none",
        })
    return out


def summary(rows: Iterable[dict]) -> Dict[str, dict]:
    """Per task and model: items by regime, the response ratio, the zero-unit count and
    HALoGEN's h beside ours (both pooled over samples). `items_missing_samples` counts items
    with fewer than five sampled generations, whose h comes from the ones present."""
    acc: Dict[str, dict] = {}
    for r in rows:
        a = acc.setdefault("%s|%s" % (r["task"], r["model"]), defaultdict(int))
        a["items"] += 1
        a["no_h"] += r["h"] is None
        a["S1"] += r["regime_h"] == "S1"
        a["always_wrong"] += r["regime_h"] == "always_wrong"
        a["always_wrong_low_n"] += r["low_n"]
        a[SINGLE_MENTION] += r["regime_h"] == SINGLE_MENTION
        a["items_missing_samples"] += r["n_samples"] < len(SAMPLED)   # a partial unit
        for k in ("n_samples", "n_usable", "n_hallucinated", "n_zero_unit", "n_abstention", "n_empty",
                  "n_emptied", "n_missing"):
            a[k] += r[k]
    out = {}
    for key, a in sorted(acc.items()):
        task, model = key.split("|")
        d = dict(a, task=task, model=model)
        d["response_ratio"] = round(a["n_usable"] / a["n_samples"], 4) if a["n_samples"] else None
        d["h_pooled"] = round(a["n_hallucinated"] / a["n_usable"], 4) if a["n_usable"] else None
        denom = a["n_usable"] + a["n_zero_unit"]
        d["h_pooled_halogen_zero_units"] = round(a["n_hallucinated"] / denom, 4) if denom else None
        out[key] = d
    return out


# ---------------------------------------------------------------------------
# Per slot: Biographies birth and death years
# ---------------------------------------------------------------------------
#
# Item-level h saturates on Biographies: a biography holds about 20 atoms, so nearly every
# sample has one hallucinated atom and nearly every item reads h = 1 (S1 on 0.25% of items,
# RedPajama 1 of 400, to 22%, OLMo 1.7 89 of 400). So the regime is scored per (prompt,
# model, relation) slot: each sample is right, wrong or absent for the slot, absent out of
# the denominator, S1 = at least one right. A sample that states the slot's right year is
# right even if it also states a wrong one: most such wrong years belong to
# someone else ("The child was born on May 14, 1939"), and the rest are a sample
# contradicting itself, which has still produced the right year. Biographies' birth and
# death years are checked against Wikidata, so those two slots have a right
# value, and a wrong value is the year itself, so c needs no clustering (`cluster_method`
# `year`): the share of wrong samples that state the most frequent wrong year, a sample
# stating two wrong years counting for each. c only from 2 or more
# wrong samples, S2 at c >= 0.8, S3 below; a slot with one wrong sample and no right one is
# `single_mention`; S2/S3 on 2 stated samples are Low. `greedy=True` adds the greedy pass
# as a sixth sample, the robustness run.
#
# `dominant_wrong` is the most frequent wrong year on every slot with a wrong sample, S1
# and `single_mention` included: the corpus counts take n_h for it on those slots too.
# c is filled from 2 wrong samples, but only S2 and S3 use it.

SLOT_COLUMNS = ("model", "prompt_id", "name", "kind", "own_years", "n_right", "n_wrong", "n_absent",
                "h", "dominant_wrong", "c", "regime", "low_n", "greedy", "cluster_method", "c_cut")
C_CUT = 0.8


def _slot_state(claims: List[dict]) -> tuple:
    """(state, wrong years) for one sample's claims on one slot. Stating the right year
    makes the sample right, with no wrong years, whatever else it states."""
    if not claims:
        return "absent", ()
    own = {int(y) for y in claims[0]["wikidata_years"].split()}
    stated = {int(y) for c in claims for y in c["stated_years"].split()}
    if stated & own:
        return "right", ()
    return "wrong", tuple(sorted(stated - own))


def ok_claims(year_claims: Iterable[dict], samples: Iterable[dict]) -> tuple:
    """(kept, dropped): the year claims whose generation's sidecar outcome is `ok`.
    `samples` are bio_samples' rows. A claim from a generation the sidecars don't list is an
    error: the claims and the verification output don't belong together."""
    outcome = {(s["model"], s["prompt_id"], int(s["sample_index"])): s["outcome"] for s in samples}
    kept, dropped = [], 0
    for c in year_claims:
        key = (c["model"], c["generation_id"].split("|")[1], int(c["sample_index"]))
        if key not in outcome:
            raise ValueError("%s: no sidecar row for this claim's generation" % c["generation_id"])
        if outcome[key] == "ok":
            kept.append(c)
        else:
            dropped += 1
    return kept, dropped


def slot_items(year_claims: Iterable[dict], greedy: bool = False) -> List[dict]:
    """One row per (model, prompt, birth/death) slot that has Wikidata years, from
    results/bio_year_claims.csv. Slots whose person has no Wikidata year can't be judged and
    are left out. `greedy` counts the greedy pass as a sixth sample (supplementary run)."""
    indexes = SAMPLED + ((GREEDY,) if greedy else ())
    groups: Dict[tuple, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in year_claims:
        if not c["wikidata_years"].strip():
            continue
        pid = c["generation_id"].split("|")[1]
        groups[(c["model"], pid, c["kind"])][int(c["sample_index"])].append(c)
    out = []
    for (model, pid, kind), by_index in sorted(groups.items()):
        any_claim = next(iter(by_index.values()))[0]
        states = [_slot_state(by_index.get(i, [])) for i in indexes]
        n_right = sum(s == "right" for s, _ in states)
        wrongs = [w for s, w in states if s == "wrong"]
        n_wrong = len(wrongs)
        n = n_right + n_wrong
        per_year: Dict[int, int] = defaultdict(int)   # wrong samples stating each wrong year
        for w in wrongs:
            for y in w:
                per_year[y] += 1
        top = min(per_year, key=lambda y: (-per_year[y], y)) if per_year else None   # ties: the earliest year
        dom = "" if top is None else str(top)
        c = round(per_year[top] / n_wrong, 4) if n_wrong >= MIN_WRONG else None
        if not n:
            regime = ""
        elif n_right:
            regime = "S1"
        elif n_wrong < MIN_WRONG:
            regime = SINGLE_MENTION
        else:
            regime = "S2" if c >= C_CUT else "S3"
        g = _slot_state(by_index.get(GREEDY, []))[0]
        out.append({"model": model, "prompt_id": pid, "name": any_claim["name"], "kind": kind,
                    "own_years": any_claim["wikidata_years"], "n_right": n_right, "n_wrong": n_wrong,
                    "n_absent": len(indexes) - n, "h": round(n_wrong / n, 4) if n else None,
                    "dominant_wrong": dom, "c": c, "regime": regime,
                    "low_n": int(regime in ("S2", "S3") and n < MIN_USABLE), "greedy": g,
                    "cluster_method": "year", "c_cut": C_CUT})
    return out


def slot_summary(rows: Iterable[dict]) -> Dict[str, dict]:
    """Per model: slots with a regime, by regime (`single_mention` included), and the low-n count."""
    acc: Dict[str, dict] = {}
    for r in rows:
        a = acc.setdefault(r["model"], defaultdict(int))
        if not r["regime"]:
            continue
        a["slots"] += 1
        a[r["regime"]] += 1
        a["low_n"] += r["low_n"]
    return {m: dict(a) for m, a in sorted(acc.items())}


# ---------------------------------------------------------------------------
# Biographies: the sidecar plus OpenFActScore's output
# ---------------------------------------------------------------------------

def _read(path: str) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def bio_samples(verifier_root: str, outputs: str = "outputs_filtered", skipped: Optional[list] = None) -> List[dict]:
    """Every Biographies generation from the verification files: the sidecar's outcome, and for a
    sent one the atoms of its output row (any False atom: hallucinated). `outputs` is
    `outputs_filtered` (after the unit filter) or `outputs` (as released).

    A generation listed by two units (a stray unit from an earlier layout) is taken from
    the unit that has an output file; the other unit is added to `skipped`. Two units with
    output for one generation, or an `ok` generation with no output row, is an error."""
    from hallutrace import bio_filter
    csv.field_size_limit(1 << 30)
    by_gen: Dict[str, List[tuple]] = defaultdict(list)
    units = {}
    for meta_path in sorted(glob.glob(os.path.join(verifier_root, "inputs", "*", "*.meta.csv"))):
        metas = [m for m in _read(meta_path) if m["task"] == "biographies"]
        if not metas:
            continue  # another task's unit
        model = os.path.basename(os.path.dirname(meta_path))
        unit = os.path.basename(meta_path)[: -len(".meta.csv")]
        out_path = os.path.join(verifier_root, outputs, model, unit + ".csv")
        units[unit] = (model, out_path, os.path.exists(out_path) and os.path.getsize(out_path) > 0)
        for m in metas:
            by_gen[m["generation_id"]].append((unit, m))

    loaded: Dict[str, tuple] = {}

    def unit_rows(unit):
        if unit not in loaded:
            model, out_path, has_out = units[unit]
            inputs = _read(os.path.join(verifier_root, "inputs", model, unit + ".csv"))
            rows = {r["name"]: r for r in _read(out_path)} if has_out else {}
            loaded[unit] = (inputs, rows)
        return loaded[unit]

    out, stray = [], set()
    for gid in sorted(by_gen):
        cands = by_gen[gid]
        with_out = [c for c in cands if units[c[0]][2]]
        if len(with_out) > 1:
            raise ValueError("%s: in two units with output: %s" % (gid, [c[0] for c in with_out]))
        unit, m = (with_out or cands)[0]
        stray.update(c[0] for c in cands if c[0] != unit)
        s = {"task": m["task"], "model": m["model"], "prompt_id": m["prompt_id"],
             "sample_index": int(m["sample_index"]), "outcome": m["outcome"],
             "n_units": None, "hallucinated": None}
        if m["outcome"] == "ok":
            if m["sent"] != "1":
                raise ValueError("%s: ok but not sent" % gid)
            inputs, rows = unit_rows(unit)
            name = inputs[int(m["input_row"])]["name"]
            if name not in rows:
                raise ValueError("%s: no %s row for %r in %s" % (gid, outputs, name, unit))
            atoms = bio_filter.parse_atoms(rows[name].get("atoms"))
            s["n_units"] = len(atoms)
            s["hallucinated"] = any(not v for _, v in atoms)
        out.append(s)
    if skipped is not None:
        skipped.extend(sorted(stray))
    return out


# ---------------------------------------------------------------------------
# The other seven tasks: the sidecar plus the verifier's output
# ---------------------------------------------------------------------------
#
# Historical Events comes from the first verification run (`verifier/generations`, the
# Biographies root); the six extraction tasks from the v2 run (`verifier/generations-revised-prompts`):
#
#   - the sidecar's `input_row` is the output row: rows keep input order, and summarization and
#     simplification outputs carry no generation_id. Where an output row has a generation_id
#     or a response, it must match the sidecar and the input row, or it is an error;
#   - `outcome` comes from the sidecar, not the output: the first run sent Historical Events'
#     615 abstentions to the verifier, so they have output rows, and they stay out of h;
#   - `filtered` applies the unit filter (posthoc.filter_row, summarization and simplification as
#     "decompose") and takes h from the units left; `filtered=False` is the run as released.
#     Historical Events extracts nothing, so it is the same both ways;
#   - a row with no units enters h by posthoc's `zero_unit_rule` label, not a second
#     copy of the rule. Historical Events' yes or no is its unit, which posthoc's count leaves
#     out, so there a row with no units (the model did not answer) stays in, not hallucinated;
#   - an ok generation whose unit has no output file is `missing` (still failing after the
#     reruns, in `failed/`), out of h and counted.

HE = "historical_events"
DECOMPOSE = ("summarization", "simplification")
V2_TASKS = ("false_presuppositions", "rationalization_numerical", "rationalization_binary",
            "scientific_attribution") + DECOMPOSE
OTHER_TASKS = (HE,) + V2_TASKS


def task_samples(verifier_root: str, task: str, filtered: bool = True) -> List[dict]:
    """Every generation of `task` (one of OTHER_TASKS) under `verifier_root`, as samples for
    items(): the sidecar's outcome, and for an ok one the units and verdict of its output row."""
    from hallutrace import posthoc
    if task not in OTHER_TASKS:
        raise ValueError("task must be one of %s, got %r" % (OTHER_TASKS, task))
    key = "decompose" if task in DECOMPOSE else task
    csv.field_size_limit(1 << 30)
    out, seen = [], set()
    for meta_path in sorted(glob.glob(os.path.join(verifier_root, "inputs", "*", "*.meta.csv"))):
        metas = [m for m in _read(meta_path) if m["task"] == task]
        if not metas:
            continue
        model = os.path.basename(os.path.dirname(meta_path))
        unit = os.path.basename(meta_path)[: -len(".meta.csv")]
        out_path = os.path.join(verifier_root, "outputs", model, unit + ".csv")
        rows = _read(out_path) if os.path.exists(out_path) else None
        inputs = _read(os.path.join(verifier_root, "inputs", model, unit + ".csv")) if rows is not None else None
        for m in metas:
            if m["generation_id"] in seen:
                raise ValueError("%s: in two units" % m["generation_id"])
            seen.add(m["generation_id"])
            s = {"task": task, "model": m["model"], "prompt_id": m["prompt_id"],
                 "sample_index": int(m["sample_index"]), "outcome": m["outcome"],
                 "n_units": None, "hallucinated": None}
            if m["outcome"] == "ok":
                if m["sent"] != "1":
                    raise ValueError("%s: ok but not sent" % m["generation_id"])
                if rows is None:
                    s["outcome"] = "missing"
                else:
                    s.update(_score_row(task, key, _output_row(m, unit, rows, inputs), filtered, posthoc))
            out.append(s)
    return out


def _output_row(m: dict, unit: str, rows: List[dict], inputs: List[dict]) -> dict:
    i = int(m["input_row"])
    if not 0 <= i < len(rows) or len(rows) != len(inputs):
        raise ValueError("%s: input_row %d, but %s has %d output and %d input rows"
                         % (m["generation_id"], i, unit, len(rows), len(inputs)))
    row, given = rows[i], inputs[i]
    for col in ("generation_id", "response"):
        if col in row and col in given and row[col] != given[col]:
            raise ValueError("%s: output row %d of %s doesn't match its input (%s)" % (m["generation_id"], i, unit, col))
    if given.get("generation_id", m["generation_id"]) != m["generation_id"]:
        raise ValueError("%s: input row %d of %s is %s" % (m["generation_id"], i, unit, given["generation_id"]))
    return row


def _score_row(task: str, key: str, row: dict, filtered: bool, posthoc) -> dict:
    if task == HE:
        return {"n_units": len(posthoc.parse_list(row.get("atomic_units"))),
                "hallucinated": bool(posthoc.parse_list(row.get("hallucinated_atomic_units")))}
    if filtered:
        row = posthoc.filter_row(key, row)
    if key == "decompose":
        # the verifier's own column as released; filter_row rewrites it from the atoms it keeps
        hallucinated = bool(str(row.get("hallucinated_atomic_units") or "").strip())
    else:
        hallucinated = bool(posthoc.parse_list(row.get("hallucinated_atomic_units")))
    return {"n_units": posthoc.unit_count(key, row), "hallucinated": hallucinated,
            "zero_unit_rule": posthoc.zero_unit_rule(key, row)}
