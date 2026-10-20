"""hallutrace/intervention_points.py  |  Biographies year-slot labels: point, confidence, coverage, flags (pure, no I/O)

The label record on the Biographies birth/death-year slots, one record per slot as
`slot_types.classify` returns it (type, class and F5 point taken unchanged):

  - the slot regimes (`single_mention` kept as a regime value; merged and untyped slots, the
    open band; a sample stating the right year is right, from `stability.slot_items`
    through `cu.from_slot_items`); `low_n` for S2 / S3 on fewer than MIN_USABLE stated
    samples
  - Group B left out of the types; tau from historical_events (the headline file), with
    Biographies' own tau as a check: `tau_rows` reads either file shape
  - the decision rule: the point is the cell; slots with no count row get their own line,
    out of F5; the residual bucket is for slots no rule fired on
  - the labelled population: only slots with at least one wrong sample. A never-wrong slot
    (n_wrong 0, all S1) is not a hallucination and labels no claim, so it gets no label,
    stays out of T5's type columns and F5 (`f5_range`), and is counted on its own line,
    "S1, never wrong" (`never_wrong_counts`); the regime counts keep it
  - confidence by rule v3 (`CONFIDENCE_RULE`): worst condition wins, no point means no
    confidence
  - coverage: `coverage_downgrade` is recorded as its own column and not applied on top
    (the two absence points are already Low)

The CLI reads the files, runs the tau sweep and writes the CSV; this module only turns
plain dicts into plain dicts. No database: `regime` keeps `single_mention`, which the
`labels` table of hallutrace/store.py would reject. The record starts with the
`labels.csv` columns in the `labels` table's order, then `kind` and the extras.

See also: hallutrace/slot_types.py (slots, classify, F5 range), hallutrace/store.py
(the `labels` table).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from hallutrace import slot_types as cu

RULE_VERSION = "intervention-points-v3 (confidence rule v3, 12 Oct 2026)"
CONFIDENCE_RULE = """\
Confidence v3 (12 Oct 2026). Worst condition wins: Low over Medium over High.
Only slots with at least one wrong sample are labelled.
- No point (single_mention, open band, no count row): confidence is blank.
- Low: the point is at the knowledge boundary or at training process (it rests on
  absence), or the slot is low_n (S2 or S3 on fewer than 3 stated samples).
- Medium: the point changes somewhere in the 5th to 20th percentile tau sweep; or S2
  BOTH_IN_DATA with rho in [0.4, 0.6); or S3 with c in [0.6, 0.8); or S1 with n_right = 1,
  n_wrong >= 2 and c >= 0.8; or no count is known to be above 0 (no positive presence
  signal; an unknown count is not above 0; on S1 it means not phrase-checked and nothing
  found, since S1 slots are never phrase-checked).
- High: otherwise.
The schema's rule that a claim in 1 of 5 usable samples does not get High is not applied:
it exists so that a claim seen once does not inherit an item-level regime, and Biographies
regimes are already per slot, so nothing is inherited. With 5
samples, one sample stating the dominant wrong year follows from the S1 and S3
definitions (57 of 60 S1 and 161 of 166 S3 slots with a point on the stand-in), so the
rule would remove High by construction. claim_rate stays in labels.csv.
The coverage downgrade (1 OLMo 2, 2 DCLM, on the two absence points) is written as its own
column and not applied on top: those points are already Low."""

MIN_USABLE = 3   # stability.MIN_USABLE
DOWNGRADE = {"olmo2-7b": 1, "dclm-7b": 2}
ABSENCE_POINTS = (cu.BOUNDARY, cu.PROCESS)   # the points that rest on absence
# The same strings as scripts/slot_types.py OPEN_S3 / OPEN_S2.
OPEN_BAND = {"S3": "open: knowledge boundary or undetermined", "S2": "open: S2, any cell"}
FLAG_FABRICATION_S1 = "fabrication_s1"
HIGH, MEDIUM, LOW = "High", "Medium", "Low"

LOW_ABSENCE = "absence_point"
LOW_N = "low_n"
MED_SWEEP = "sweep_point"
MED_RHO = "s2_rho_near_half"
MED_C = "s3_c_near_cut"
MED_ONE_RIGHT = "s1_one_right"
MED_NO_COUNT = "no_count_above_0"
REASONS = (LOW_ABSENCE, LOW_N, MED_SWEEP, MED_RHO, MED_C, MED_ONE_RIGHT, MED_NO_COUNT)
LOW_REASONS = (LOW_ABSENCE, LOW_N)
MEDIUM_REASONS = (MED_SWEEP, MED_RHO, MED_C, MED_ONE_RIGHT, MED_NO_COUNT)

# The `labels` table's columns, then `kind` and the extras.
LABEL_COLUMNS = (
    "prompt_id", "model", "claim_id", "phase2_type", "regime", "claim_rate",
    "intervention_point", "confidence", "coverage_downgrade", "rule_version", "kind", "tau",
    "class", "t5_column", "merged", "untyped_reason", "open_band", "flag", "no_count_row",
    "residual", "low_n", "n_right", "n_wrong", "c", "dominant_wrong", "n_c", "n_h", "rho",
    "f5_set", "confidence_reasons",
)


# ---------------------------------------------------------------------------
# Slot measures
# ---------------------------------------------------------------------------


def low_n(classified: dict) -> int:
    """1 for S2 / S3 with fewer than MIN_USABLE stated samples (n_right + n_wrong), 0
    otherwise. A `low_n` key (`slot_items`, through `cu.from_slot_items`) must agree:
    ValueError if it does not."""
    n = classified["n_right"] + classified["n_wrong"]
    derived = 1 if classified["regime"] in ("S2", "S3") and n < MIN_USABLE else 0
    if "low_n" in classified and int(classified["low_n"]) != derived:
        raise ValueError("low_n %r disagrees with regime %s on %d stated samples for %r" % (
            classified["low_n"], classified["regime"], n, cu.slot_key(classified)))
    return derived


def rho(classified: dict) -> Optional[float]:
    """n_h / (n_h + n_c) from c_max and h_max, rounded to 4 places, for a typed slot with
    both known and a sum above 0; None otherwise."""
    n_c, n_h = classified["c_max"], classified["h_max"]
    if classified["class"] != "typed" or n_c is None or n_h is None or n_c + n_h <= 0:
        return None
    return round(n_h / (n_h + n_c), 4)


def never_wrong(classified: dict) -> bool:
    """No wrong sample (n_wrong 0): not a hallucination, no label."""
    return classified["n_wrong"] == 0


def dominant_samples(classified: dict) -> int:
    """k, the wrong samples stating the dominant wrong year: round(c * n_wrong) when c is
    known, 1 for a single wrong sample, 0 with none. ValueError when c is missing on 2 or
    more wrong samples (c is computed from 2 wrong samples on)."""
    c, n_wrong = classified["c"], classified["n_wrong"]
    if c is not None:
        return round(c * n_wrong)
    if n_wrong >= 2:
        raise ValueError("c missing on %d wrong samples for %r" % (n_wrong, cu.slot_key(classified)))
    return n_wrong


def found(classified: dict) -> bool:
    """Some count is known to be above 0: a known c or h count above 0, or a `present` row
    (the phrase check hit). An `unknown` count is not; the class does not matter."""
    if any(v is not None and v > 0 for v in (classified["c_max"], classified["h_max"])):
        return True
    return cu.PRESENT in classified["c_status"].split() + classified["h_status"].split()


def claim_rate(classified: dict) -> Optional[float]:
    """The dominant wrong year's share of the stated samples, rounded to 4 places:
    dominant_samples / (n_right + n_wrong). None with no dominant wrong
    year or no stated sample."""
    n = classified["n_right"] + classified["n_wrong"]
    if not classified["dominant_wrong"] or n == 0:
        return None
    return round(dominant_samples(classified) / n, 4)


def flag(classified: dict) -> str:
    """FLAG_FABRICATION_S1 for a typed S1 FABRICATION slot, else "" (untyped and merged S1
    slots carry no flag)."""
    if (classified["regime"] == "S1" and classified["class"] == "typed"
            and classified["type"] == cu.FABRICATION):
        return FLAG_FABRICATION_S1
    return ""


# ---------------------------------------------------------------------------
# Confidence and coverage
# ---------------------------------------------------------------------------


def coverage_downgrade(model: str, point: str) -> int:
    """DOWNGRADE[model] on an absence point, else 0 (recorded, not applied). ValueError on
    a model that is not a key of cu.MODEL_INDEX."""
    if model not in cu.MODEL_INDEX:
        raise ValueError("model must be one of %s, got %r" % (sorted(cu.MODEL_INDEX), model))
    return DOWNGRADE.get(model, 0) if point in ABSENCE_POINTS else 0


def confidence_reasons(classified: dict, sweep_points: Optional[Iterable[str]] = None) -> tuple:
    """Every reason that fires, in REASONS order; () for no point ("" or open).
    `sweep_points` is the slot's f5_point at each percentile of the sweep (None: not run)."""
    point = classified["f5_point"]
    if point in ("", cu.OPEN):
        return ()
    regime, cls = classified["regime"], classified["class"]
    c = classified["c"]
    fired = set()
    if point in ABSENCE_POINTS:
        fired.add(LOW_ABSENCE)
    if low_n(classified) == 1:
        fired.add(LOW_N)
    if sweep_points is not None and len(set(sweep_points) | {point}) > 1:
        fired.add(MED_SWEEP)
    if regime == "S2" and cls == "typed" and classified["type"] in (cu.BOTH_HIGH, cu.BOTH_LOW):
        r = rho(classified)
        if r is not None and 0.4 <= r < 0.6:
            fired.add(MED_RHO)
    if regime == "S3" and c is not None and 0.6 <= c < 0.8:
        fired.add(MED_C)
    if (regime == "S1" and classified["n_right"] == 1 and classified["n_wrong"] >= 2
            and c is not None and c >= 0.8):
        fired.add(MED_ONE_RIGHT)
    if not found(classified):
        fired.add(MED_NO_COUNT)
    return tuple(r for r in REASONS if r in fired)


def confidence(reasons: Iterable[str], point: str) -> str:
    """Worst condition wins: LOW on any LOW_REASONS, else MEDIUM on any MEDIUM_REASONS,
    else HIGH. "" when there is no point ("" or open)."""
    if point in ("", cu.OPEN):
        return ""
    reasons = set(reasons)
    if reasons & set(LOW_REASONS):
        return LOW
    if reasons & set(MEDIUM_REASONS):
        return MEDIUM
    return HIGH


# ---------------------------------------------------------------------------
# Label record
# ---------------------------------------------------------------------------


def label(classified: dict, tau, sweep_points: Optional[Iterable[str]] = None) -> dict:
    """The slot's label record, keys in LABEL_COLUMNS order, Python values. `tau` is the
    model's tau as used (recorded, not recomputed). The input is not changed. ValueError
    on a never-wrong slot: it labels no claim."""
    s = classified
    if never_wrong(s):
        raise ValueError("never-wrong slot %r has no label" % (cu.slot_key(s),))
    model, point, regime, cls = s["model"], s["f5_point"], s["regime"], s["class"]
    if point == cu.OPEN:
        if regime not in OPEN_BAND:
            raise ValueError("open point on regime %r with no open band for %r"
                             % (regime, cu.slot_key(s)))
        band = OPEN_BAND[regime]
    else:
        band = ""
    reasons = confidence_reasons(s, sweep_points)
    rec = {
        "model": model, "prompt_id": s["prompt_id"], "kind": s["kind"],
        "claim_id": "%s|%s|%s" % (s["prompt_id"], s["kind"], s["dominant_wrong"]),
        "phase2_type": s["t5_column"], "regime": regime, "claim_rate": claim_rate(s),
        "intervention_point": point, "confidence": confidence(reasons, point),
        "coverage_downgrade": coverage_downgrade(model, point),
        "rule_version": RULE_VERSION, "tau": float(tau),
        "class": cls, "t5_column": s["t5_column"],
        "merged": 1 if cls == "merged" else 0,
        "untyped_reason": cu.UNTYPED_LABEL if cls in ("phrase_hit", "untyped") else "",
        "open_band": band, "flag": flag(s),
        "no_count_row": 1 if cls == "no_count_row" else 0,
        "residual": 1 if cls != "no_count_row" and s["t5_column"] == "" else 0,
        "low_n": low_n(s), "n_right": int(s["n_right"]), "n_wrong": int(s["n_wrong"]),
        "c": s["c"], "dominant_wrong": s["dominant_wrong"],
        "n_c": s["c_max"], "n_h": s["h_max"], "rho": rho(s),
        "f5_set": tuple(sorted(s["f5_set"])), "confidence_reasons": reasons,
    }
    return {k: rec[k] for k in LABEL_COLUMNS}


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def never_wrong_counts(classified: Iterable[dict]) -> Dict[str, int]:
    """{model: never-wrong slots} for every model of cu.MODEL_INDEX, in its order, zeros
    included: the "S1, never wrong" line. ValueError on a model not in MODEL_INDEX."""
    out = {m: 0 for m in cu.MODEL_INDEX}
    for s in classified:
        if s["model"] not in out:
            raise ValueError("model must be one of %s, got %r" % (sorted(cu.MODEL_INDEX), s["model"]))
        out[s["model"]] += int(never_wrong(s))
    return out


def f5_range(classified: Iterable[dict]) -> dict:
    """cu.f5_range over the slots with at least one wrong sample (its own pool rule, S1 to
    S3 with a count row, still applies)."""
    return cu.f5_range(s for s in classified if not never_wrong(s))


# ---------------------------------------------------------------------------
# Tau sweep
# ---------------------------------------------------------------------------


def sweep_sets(classified_by_percentile: Dict[int, List[dict]]) -> Dict[tuple, frozenset]:
    """{cu.slot_key: frozenset of the slot's f5_point over every percentile}. ValueError if
    the percentiles do not hold the same slot keys, or a key repeats within one."""
    out: Dict[tuple, set] = {}
    first_keys, first_p = None, None
    for p in sorted(classified_by_percentile):
        keys = set()
        for s in classified_by_percentile[p]:
            k = cu.slot_key(s)
            if k in keys:
                raise ValueError("slot %r repeats at percentile %r" % (k, p))
            keys.add(k)
            out.setdefault(k, set()).add(s["f5_point"])
        if first_keys is None:
            first_keys, first_p = keys, p
        elif keys != first_keys:
            raise ValueError("percentile %r holds different slot keys from percentile %r "
                             "(%d only here, %d only there)"
                             % (p, first_p, len(keys - first_keys), len(first_keys - keys)))
    return {k: frozenset(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# Tau rows
# ---------------------------------------------------------------------------


def _check_variant_key(rows: List[dict], variant: Optional[str]) -> None:
    for r in rows:
        if variant is not None and "variant" not in r:
            raise ValueError("variant %r asked but a tau row has no variant column" % (variant,))
        if variant is None and "variant" in r:
            raise ValueError("variant None but a tau row has a variant column (%r): "
                             "name the variant" % (r["variant"],))


def tau_rows(rows: Iterable[dict], percentile, variant: Optional[str] = None,
             expected: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """{index: tau} for every index of cu.MODEL_INDEX, in its order, from tau CSV rows
    (results/slot_types/he_tau.csv with `variant`, or results/bio_counts/merged/tau.csv with
    `variant=None`) at `percentile`. Rows of other indexes are ignored. ValueError on a
    variant column mismatch, a duplicate, a tau that is not a number (NaN and infinities
    included), a missing index, or a value differing from `expected`."""
    rows = list(rows)
    _check_variant_key(rows, variant)
    want = str(int(percentile))
    indexes = set(cu.MODEL_INDEX.values())
    found: Dict[str, float] = {}
    for r in rows:
        if str(r["percentile"]).strip() != want or r["index"] not in indexes:
            continue
        if variant is not None and r["variant"] != variant:
            continue
        idx = r["index"]
        if idx in found:
            raise ValueError("duplicate tau row for index %r at percentile %s" % (idx, want))
        try:
            v = float(r["tau"])
        except (TypeError, ValueError):
            v = None
        if v is None or not math.isfinite(v):
            raise ValueError("tau for index %r at percentile %s is not a number: %r"
                             % (idx, want, r["tau"]))
        found[idx] = v
    missing = [i for i in cu.MODEL_INDEX.values() if i not in found]
    if missing:
        raise ValueError("missing tau row at percentile %s (variant %r) for %s"
                         % (want, variant, missing))
    out = {i: found[i] for i in cu.MODEL_INDEX.values()}
    if expected is not None:
        bad = {i: (out.get(i), v) for i, v in expected.items() if out.get(i) != v}
        if bad:
            raise ValueError("unexpected value of tau at percentile %s (got, expected): %r"
                             % (want, bad))
    return out


def percentiles(rows: Iterable[dict], variant: Optional[str] = None) -> List[int]:
    """Sorted distinct int percentiles of the rows (only `variant`'s rows when given)."""
    return sorted({int(str(r["percentile"]).strip()) for r in rows
                   if variant is None or r.get("variant") == variant})
