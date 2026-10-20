"""hallutrace/slot_types.py  |  untyped Biographies slots: merged type, phrase check, F5 range (pure, no I/O)

Untyped Biographies birth/death-year slots:

  C  the merged type "BOTH_IN_DATA (rho < 0.5) or X" when n_c is counted and n_h is
     `unknown`; X is the type n_h = 0 would give (RECALL_FAILURE if n_c > tau, LONG_TAIL
     if n_c <= tau)
  E  the exact-phrase check: fixed phrases built from the slot's plain `name` string and
     the year (`PHRASES`, birth years only, `CHECKED_FACTS`); a hit moves an `unknown`
     count to `present` (above 0, size not known), no hit leaves it `unknown`
  F  the intervention-point shares (F5) as a range, with the slots whose cell still
     depends on an `unknown` count shown as their own open band

The CLI (scripts/slot_types.py) reads results/bio_year_claims.csv, the merged
counts (results/bio_counts/merged/counts.csv, the counting rule's cell, roles c and h) and the tau
rows, sends the phrase queries, and hands plain dicts to this module.

The Phase 2 types here are approximate: the tracker conditions of the cause taxonomy,
with Group B left out. They are the types the decision was tested with, not intervention_points's
labels. n_c and n_h are both read from a count
row's `n_c` column (role c: the person's own Wikidata year; role h: a wrong year), in
`docs` or `est_mentions` units as the row says, used as is.

The outputs' slots come from `stability.slot_items` through `from_slot_items`: a sample
stating the right year is right even if it also states a wrong one. `slots` is an earlier
stand-in, kept for comparison in the tests; under it such a
sample is wrong. Both otherwise count each stated wrong year once per wrong sample, c
only from 2 or more wrong samples, `single_mention` for exactly one wrong sample and no right
one, S2 if c >= 0.8.

Role h years default to the single dominant wrong year: the most frequent wrong year,
ties to the earliest (n_h is taken at the dominant wrong year, and c from the most frequent
wrong year, as `slot_items` does). `h_years="tuple"` takes every year of the most frequent
wrong tuple and only reproduces the decision's test.
Regime, n_right, n_wrong and c are the same in both modes.

See also hallutrace/pilot_counts.py (count rows, statuses, per_string_json).
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# Model to infini-gram index (order and comments: infinigram.INDEXES; labels:
# pilot_counts.INDEX_LABELS).
MODEL_INDEX = {
    "olmo2-7b": "v4_olmo-mix-1124_llama",     # OLMo 2
    "dclm-7b": "v4_dclm-baseline_llama",      # DCLM
    "olmo17-7b": "v4_dolma-v1_7_llama",       # Dolma (OLMo 1.7)
    "redpajama-7b": "v4_rpj_llama_s4",        # RedPajama
    "pythia-69b": "v4_piletrain_llama",       # Pile (Pythia)
}

# Rule E's phrases, in query order. `{name}` is the slot's plain `name` string (no alias,
# no glued form), `{year}` the year. Every phrase puts the year within the rule's 100
# characters of the name. The death list is kept for the record and never queried
# (calibration: 1 of 10 known death years hit).
PHRASES = {
    "birth": ["{name} (born {year}", "{name} ({year}", "{name} was born in {year}",
              "{name}, born {year}"],
    "death": ["{name} died in {year}", "{name} (died {year}", "{name}, who died in {year}"],
}
CHECKED_FACTS = ("birth",)

POINTS = ("at inference", "at training data", "at training process",
          "at the knowledge boundary", "undetermined")
INFERENCE, DATA, PROCESS, BOUNDARY, UNDETERMINED = POINTS
TOGETHER = "training data and process together"
OPEN = "open"
UNTYPED_LABEL = "untyped: count below sampling resolution"   # T5's column name

REGIMES = ("S1", "S2", "S3", "single_mention")
SINGLE_MENTION = "single_mention"
H_YEARS_MODES = ("dominant", "tuple")

BOTH_HIGH = "BOTH_IN_DATA (rho >= 0.5)"
BOTH_LOW = "BOTH_IN_DATA (rho < 0.5)"
WRONG_IN_DATA = "WRONG_IN_DATA"
FABRICATION = "FABRICATION"
LONG_TAIL = "LONG_TAIL"
RECALL_FAILURE = "RECALL_FAILURE"
TYPES = (BOTH_HIGH, BOTH_LOW, WRONG_IN_DATA, FABRICATION, LONG_TAIL, RECALL_FAILURE)
MERGED_RECALL = BOTH_LOW + " or " + RECALL_FAILURE
MERGED_LONG_TAIL = BOTH_LOW + " or " + LONG_TAIL

# The decision rule's table for the types used here.
# `single_mention` has no intervention point and no entry.
CELL = {
    "S1": {t: INFERENCE for t in TYPES},
    "S2": {BOTH_HIGH: DATA, BOTH_LOW: UNDETERMINED, WRONG_IN_DATA: DATA,
           LONG_TAIL: PROCESS, RECALL_FAILURE: UNDETERMINED, FABRICATION: BOUNDARY},
    "S3": {BOTH_HIGH: UNDETERMINED, BOTH_LOW: UNDETERMINED, WRONG_IN_DATA: UNDETERMINED,
           LONG_TAIL: UNDETERMINED, RECALL_FAILURE: UNDETERMINED, FABRICATION: BOUNDARY},
}

# Slots, as stability.py builds them.
SAMPLED = (1, 2, 3, 4, 5)   # greedy (sample_index 0) stays out
C_CUT = 0.8
MIN_WRONG = 2   # c needs this many wrong samples; one alone is `single_mention`

# Row statuses: pilot_counts' three, plus `present` (rule E) and `missing` (no count row).
KNOWN = ("counted", "zero")
UNKNOWN, PRESENT, MISSING = "unknown", "present", "missing"
CLASSES = ("no_count_row", "typed", "merged", "phrase_hit", "untyped")
BIG = 10 ** 6   # a candidate value far above any tau and any other candidate


# ---------------------------------------------------------------------------
# Phrases and hits
# ---------------------------------------------------------------------------


def phrases(name: str, fact: str, year) -> List[str]:
    """The fact's phrases (PHRASES order) with `name` and `year` filled in. ValueError on a
    fact other than "birth" or "death"."""
    if fact not in PHRASES:
        raise ValueError("fact must be one of %s, got %r" % (sorted(PHRASES), fact))
    return [t.format(name=name, year=year) for t in PHRASES[fact]]


def is_hit(count) -> bool:
    """True only for a count above 0. None, "", 0, a non-number and a bool are not hits."""
    if count is None or count == "" or isinstance(count, bool):
        return False
    try:
        return float(count) > 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def _check_value(v, what: str) -> float:
    if v is None or v < 0:
        raise ValueError("%s must be a number >= 0, got %r" % (what, v))
    return v


def phase2_type(n_c, n_h, tau) -> str:
    """The approximate Phase 2 type from the tracker conditions (Group B left out).
    n_h > 0 and n_c > 0: BOTH_IN_DATA, rho = n_h / (n_h + n_c), rho >= 0.5 the high side;
    n_h > 0, n_c = 0: WRONG_IN_DATA; both 0: FABRICATION; n_h = 0 and 0 < n_c <= tau:
    LONG_TAIL; n_h = 0 and n_c > tau: RECALL_FAILURE. n_c may be an est_mentions value."""
    n_c, n_h = _check_value(n_c, "n_c"), _check_value(n_h, "n_h")
    if n_h > 0:
        if n_c > 0:
            return BOTH_HIGH if n_h / (n_h + n_c) >= 0.5 else BOTH_LOW
        return WRONG_IN_DATA
    if n_c == 0:
        return FABRICATION
    return LONG_TAIL if n_c <= tau else RECALL_FAILURE


def merged_type(n_c, tau) -> str:
    """Rule C: "BOTH_IN_DATA (rho < 0.5) or X", X the type n_h = 0 would give."""
    return MERGED_RECALL if _check_value(n_c, "n_c") > tau else MERGED_LONG_TAIL


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------


def _slot_state(claims: List[dict]) -> tuple:
    """(state, wrong years) for one sample's claims on one slot, as stability's `_slot_state`:
    the wrong years are the stated years of ALL the sample's claims together minus the
    person's own years, as a sorted tuple."""
    if not claims:
        return "absent", ()
    own = {int(y) for y in claims[0]["wikidata_years"].split()}
    stated = {int(y) for c in claims for y in c["stated_years"].split()}
    wrong = tuple(sorted(stated - own))
    return ("wrong" if wrong else "right"), wrong


def slot_key(slot: dict) -> tuple:
    return (slot["model"], slot["prompt_id"], slot["kind"])


def slots(claims: Iterable[dict], h_years: str = "dominant") -> List[dict]:
    """An earlier stand-in for `stability.slot_items`, kept for comparison in the tests:
    a sample stating the right year and a wrong one is
    wrong here, right in `slot_items`. The outputs use `from_slot_items`.

    One dict per (model, prompt, birth/death) slot from the rows of
    results/bio_year_claims.csv, sorted by (model, prompt_id, kind).

    Claims with an empty `wikidata_years` are left out; samples 1 to 5 only. Per sample:
    absent, wrong or right (`_slot_state`). Slots with no stated sample are dropped.
    Regime: S1 if n_right > 0; `single_mention` if n_wrong == 1;
    else c = (wrong samples stating the most frequent wrong year) / n_wrong, rounded to 4
    places, S2 if c >= C_CUT else S3. c is None when n_wrong < 2.

    `h_years` picks role h's years: "dominant" (default) the single most frequent wrong
    year, ties to the earliest; "tuple" every year of the most frequent wrong tuple, ties
    to the earliest tuple (the decision's test only).
    `name` and `own_years` come from the slot's first claim."""
    if h_years not in H_YEARS_MODES:
        raise ValueError("h_years must be one of %s, got %r" % (H_YEARS_MODES, h_years))
    groups: Dict[tuple, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in claims:
        if not c["wikidata_years"].strip():
            continue
        pid = c["generation_id"].split("|")[1]
        groups[(c["model"], pid, c["kind"])][int(c["sample_index"])].append(c)
    out = []
    for (model, pid, kind), by_index in sorted(groups.items()):
        first = next(iter(by_index.values()))[0]
        states = [_slot_state(by_index.get(i, [])) for i in SAMPLED]
        n_right = sum(s == "right" for s, _ in states)
        wrongs = [w for s, w in states if s == "wrong"]
        n_wrong = len(wrongs)
        n = n_right + n_wrong
        if not n:
            continue
        per_year: Dict[int, int] = defaultdict(int)   # wrong samples stating each wrong year
        for w in wrongs:
            for y in w:
                per_year[y] += 1
        top = min(per_year, key=lambda y: (-per_year[y], y)) if per_year else None
        c_share = round(per_year[top] / n_wrong, 4) if n_wrong >= MIN_WRONG else None
        if n_right:
            regime = "S1"
        elif n_wrong < MIN_WRONG:
            regime = SINGLE_MENTION
        else:
            regime = "S2" if c_share >= C_CUT else "S3"
        if top is None:
            hy: Tuple[int, ...] = ()
        elif h_years == "dominant":
            hy = (top,)
        else:
            hy = max(sorted(set(wrongs)), key=wrongs.count)   # ties: the earliest tuple
        out.append({
            "model": model, "prompt_id": pid, "name": first["name"], "kind": kind,
            "own_years": tuple(sorted({int(y) for y in first["wikidata_years"].split()})),
            "n_right": n_right, "n_wrong": n_wrong, "n_absent": len(SAMPLED) - n,
            "c": c_share, "regime": regime,
            "dominant_wrong": " ".join(str(y) for y in hy), "h_years": tuple(hy),
            "h_mode": h_years,
        })
    return out


def from_slot_items(rows: Iterable[dict]) -> List[dict]:
    """The slots `classify` and `rows_for` take, from `stability.slot_items` rows, sorted by
    (model, prompt_id, kind). Rows with no regime (only the greedy pass stated a year) are
    dropped, as `slots` drops slots with no stated sample. `own_years` (a string there)
    becomes a sorted tuple of int; role h is the single dominant wrong year (`h_mode`
    "dominant", the only mode `slot_items` gives); `low_n` is carried through. Same keys as
    `slots` plus `low_n`."""
    out = []
    for r in rows:
        if not r["regime"]:
            continue
        dom = str(r["dominant_wrong"] or "").strip()
        out.append({
            "model": r["model"], "prompt_id": r["prompt_id"], "name": r["name"], "kind": r["kind"],
            "own_years": tuple(sorted({int(y) for y in str(r["own_years"]).split()})),
            "n_right": int(r["n_right"]), "n_wrong": int(r["n_wrong"]), "n_absent": int(r["n_absent"]),
            "c": r["c"], "regime": r["regime"],
            "dominant_wrong": dom, "h_years": (int(dom),) if dom else (), "h_mode": "dominant",
            "low_n": int(r["low_n"]),
        })
    return sorted(out, key=slot_key)


# ---------------------------------------------------------------------------
# Count rows
# ---------------------------------------------------------------------------


def _row_key(role: str, index: str, name: str, fact: str, year) -> tuple:
    return (role, index, name, fact, int(year))


def counts_index(rows: Iterable[dict]) -> Dict[tuple, dict]:
    """{(role, index, prompt_name, fact, int(year)): row}. ValueError on a duplicate key."""
    out: Dict[tuple, dict] = {}
    for r in rows:
        k = _row_key(r["role"], r["index"], r["prompt_name"], r["fact"], r["year"])
        if k in out:
            raise ValueError("duplicate count row %r" % (k,))
        out[k] = r
    return out


def rows_for(slot: dict, counts_idx: Dict[tuple, dict]) -> Tuple[list, list]:
    """(c_rows, h_rows): the count rows of the slot's own years (role c) and of its h years
    (role h), aligned with `own_years` / `h_years`, None for a missing row."""
    index = MODEL_INDEX[slot["model"]]
    c_rows = [counts_idx.get(_row_key("c", index, slot["name"], slot["kind"], y))
              for y in slot["own_years"]]
    h_rows = [counts_idx.get(_row_key("h", index, slot["name"], slot["kind"], y))
              for y in slot["h_years"]]
    return c_rows, h_rows


def name_string(row: dict) -> Optional[str]:
    """The `q` of the row's first per_string_json entry with src == "name", or None."""
    try:
        entries = json.loads(row.get("per_string_json") or "[]")
    except (TypeError, ValueError):
        return None
    for e in entries:
        if isinstance(e, dict) and e.get("src") == "name" and e.get("q"):
            return e["q"]
    return None


# ---------------------------------------------------------------------------
# Reachable points
# ---------------------------------------------------------------------------


def candidates(pairs: Iterable[tuple], tau, big=BIG) -> List[float]:
    """Candidate values of one role from its rows' (status, value) pairs (statuses after
    the phrase check; value used only for `counted` / `zero`). All rows known: {max known}
    (0 when there are no rows). Some row `unknown`, none `present`: lo = max known or 0,
    {lo, max(lo, 1), lo + tau + 1, big}. Some row `present`: lo = max(lo, 1),
    {lo, lo + tau + 1, big} (0 is ruled out, nothing else is)."""
    pairs = list(pairs)
    known = [v for s, v in pairs if s in KNOWN]
    lo = max(known) if known else 0.0
    statuses = {s for s, _ in pairs}
    if PRESENT in statuses:
        lo = max(lo, 1)
        vals = {lo, lo + tau + 1, big}
    elif UNKNOWN in statuses:
        vals = {lo, max(lo, 1), lo + tau + 1, big}
    else:
        vals = {lo}
    return sorted(vals)


def reachable_points(regime: str, c_pairs: Iterable[tuple], h_pairs: Iterable[tuple],
                     tau) -> frozenset:
    """The CELL points over every type the counts allow: `phase2_type` over every pair of
    role c and role h candidates (`candidates`). Empty for a regime with no CELL entry.
    The top candidate is BIG, raised above twice the largest known value of either role
    when one exceeds it, so an `unknown` count can always outweigh a known one."""
    if regime not in CELL:
        return frozenset()
    c_pairs, h_pairs = list(c_pairs), list(h_pairs)
    known = [v for s, v in c_pairs + h_pairs if s in KNOWN]
    big = max(BIG, 2 * max(known, default=0) + tau + 1)
    cs, hs = candidates(c_pairs, tau, big), candidates(h_pairs, tau, big)
    return frozenset(CELL[regime][phase2_type(c, h, tau)] for c in cs for h in hs)


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------


def _value(row: dict) -> Optional[float]:
    """float(n_c) of a known row (`counted` / `zero`); None otherwise (an `unknown` row
    with no strings has an empty n_c, and an unknown row's n_c is never used)."""
    if row is None or row["status"] not in KNOWN:
        return None
    return float(row["n_c"])


def _statuses(rows: list, years: tuple, role: str, present) -> List[str]:
    out = []
    for row, y in zip(rows, years):
        if row is None:
            out.append(MISSING)
        elif row["status"] == UNKNOWN and (role, int(y)) in present:
            out.append(PRESENT)
        else:
            out.append(row["status"])
    return out


def _known_max(rows: list) -> Optional[float]:
    vals = [v for v in (_value(r) for r in rows) if v is not None]
    return max(vals) if vals else None


def classify(slot: dict, c_rows: list, h_rows: list, tau, present=frozenset()) -> dict:
    """The slot's class, type and F5 point, merged into a copy of the slot dict.

    `c_rows` / `h_rows` are aligned with `own_years` / `h_years` (None: no row); `present`
    holds the (role, year) pairs whose phrase check hit (role "c" or "h", year int). A
    row's status is its `status`, or `present` when it is `unknown` and in `present`.

    The class is decided on the statuses before the phrase check; hits only refine it:
      no_count_row  a c or h row is missing (or the slot has no own year)
      typed         no row `unknown`: phase2_type(max n_c, max n_h or 0, tau)
      merged        every c row known with max n_c > 0, an h row `unknown`:
                    merged_type(max n_c, tau); an h row `present` keeps the merged type and
                    its point and sets `h_present_on_merged`
      phrase_hit    S2 or S3, neither of the above, at least one `present` row
      untyped       the rest
    `c_zero_h_unknown` flags every c row `zero` with an h row `unknown` (before the check;
    c and h of one name share strings and pool, so expected 0). Such a slot is `untyped`
    even when a phrase hit.

    F5: `single_mention` and `no_count_row` have none (`f5_point` "", `f5_set` empty). S1
    is at inference. typed: its cell. merged: the cells of BOTH_IN_DATA (rho < 0.5) and X.
    phrase_hit / untyped in S2 or S3: `reachable_points`. One reachable point fixes
    `f5_point`; several give `open`. `f5_set` is the reachable set (one member when fixed).
    `t5_column`: the type, the merged type, UNTYPED_LABEL (phrase_hit, untyped), or "" for
    no_count_row."""
    regime = slot["regime"]
    own, hy = tuple(slot["own_years"]), tuple(slot["h_years"])
    c_rows, h_rows = list(c_rows), list(h_rows)
    if len(c_rows) != len(own) or len(h_rows) != len(hy):
        raise ValueError("rows not aligned with own_years / h_years for %r" % (slot_key(slot),))
    present = {(role, int(y)) for role, y in present}   # a year given as a string still matches
    c_st = _statuses(c_rows, own, "c", present)
    h_st = _statuses(h_rows, hy, "h", present)
    c_max, h_max = _known_max(c_rows), _known_max(h_rows)
    has_rows = bool(c_rows) and MISSING not in c_st and MISSING not in h_st
    c_before = [r["status"] for r in c_rows if r is not None]
    h_before = [r["status"] for r in h_rows if r is not None]
    c_all_known = has_rows and all(s in KNOWN for s in c_before)
    h_unknown_before = UNKNOWN in h_before
    c_zero_h_unknown = (c_all_known and all(s == "zero" for s in c_before)
                        and h_unknown_before)

    typ, mtyp, flag_merged = "", "", False
    if not has_rows:
        cls = "no_count_row"
    elif UNKNOWN not in c_before and UNKNOWN not in h_before:
        cls = "typed"
        typ = phase2_type(c_max, h_max or 0.0, tau)
    elif c_all_known and c_max > 0 and h_unknown_before:
        cls = "merged"
        mtyp = merged_type(c_max, tau)
        flag_merged = PRESENT in h_st
    elif regime in ("S2", "S3") and PRESENT in c_st + h_st and not c_zero_h_unknown:
        cls = "phrase_hit"
    else:
        cls = "untyped"   # the impossible c_zero_h_unknown case stays here, hit or not

    if cls == "no_count_row" or regime not in CELL:
        reach = frozenset()
    elif regime == "S1":
        reach = frozenset([INFERENCE])
    elif cls == "typed":
        reach = frozenset([CELL[regime][typ]])
    elif cls == "merged":
        x = RECALL_FAILURE if mtyp == MERGED_RECALL else LONG_TAIL
        reach = frozenset([CELL[regime][BOTH_LOW], CELL[regime][x]])
    else:
        reach = reachable_points(
            regime, [(s, _value(r)) for s, r in zip(c_st, c_rows)],
            [(s, _value(r)) for s, r in zip(h_st, h_rows)], tau)
    if not reach:
        f5_point = ""
    elif len(reach) == 1:
        f5_point = next(iter(reach))
    else:
        f5_point = OPEN

    if cls == "typed":
        t5 = typ
    elif cls == "merged":
        t5 = mtyp
    elif cls in ("phrase_hit", "untyped"):
        t5 = UNTYPED_LABEL
    else:
        t5 = ""
    return dict(slot, **{
        "class": cls, "c_status": " ".join(c_st), "h_status": " ".join(h_st),
        "c_max": c_max, "h_max": h_max, "type": typ, "merged_type": mtyp,
        "reachable": reach, "f5_point": f5_point, "f5_set": reach, "t5_column": t5,
        "h_present_on_merged": flag_merged, "c_zero_h_unknown": c_zero_h_unknown,
        "untyped_before": cls not in ("no_count_row", "typed"),
    })


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _share(k: int, n: int) -> Optional[float]:
    return k / n if n else None


def f5_range(classified: Iterable[dict]) -> dict:
    """F5 as a range over the S1 to S3 slots with a count row (n). Per point in POINTS
    order, then TOGETHER: `lower` = slots fixed at it, `upper` = lower + open slots whose
    set contains it (TOGETHER: fixed at either training point, plus each open slot reaching
    either once); shares lower / n and upper / n (None when n is 0). `open`: the open slot
    count; `open_groups`: {regime, reachable (sorted list), slots (count), slot_keys
    (sorted list of slot_key)} sorted by regime then reachable."""
    pool = [s for s in classified
            if s["regime"] in ("S1", "S2", "S3") and s["class"] != "no_count_row"]
    n = len(pool)
    fixed = [s for s in pool if s["f5_point"] != OPEN]
    opened = [s for s in pool if s["f5_point"] == OPEN]

    def row(point, members):
        lower = sum(1 for s in fixed if s["f5_point"] in members)
        upper = lower + sum(1 for s in opened if s["f5_set"] & members)
        return {"point": point, "lower": lower, "upper": upper,
                "share_lower": _share(lower, n), "share_upper": _share(upper, n)}

    rows = [row(p, frozenset([p])) for p in POINTS]
    rows.append(row(TOGETHER, frozenset([DATA, PROCESS])))
    groups: Dict[tuple, list] = defaultdict(list)
    for s in opened:
        groups[(s["regime"], tuple(sorted(s["f5_set"])))].append(slot_key(s))
    open_groups = [{"regime": r, "reachable": list(reach), "slots": len(keys),
                    "slot_keys": sorted(keys)}
                   for (r, reach), keys in sorted(groups.items())]
    return {"n": n, "rows": rows, "open": len(opened), "open_groups": open_groups}


def slot_table(classified: Iterable[dict]) -> List[dict]:
    """One row per regime in REGIMES order, then "all": {regime, all, typed, merged,
    phrase_hit, untyped, no_count_row, untyped_before}."""
    classified = list(classified)

    def count(part, regime):
        r = {"regime": regime, "all": len(part)}
        for cls in ("typed", "merged", "phrase_hit", "untyped", "no_count_row"):
            r[cls] = sum(1 for s in part if s["class"] == cls)
        r["untyped_before"] = sum(1 for s in part if s["untyped_before"])
        return r

    out = [count([s for s in classified if s["regime"] == g], g) for g in REGIMES]
    out.append(count(classified, "all"))
    return out


# ---------------------------------------------------------------------------
# Phrase targets
# ---------------------------------------------------------------------------


def phrase_targets(classified: Iterable[dict], counts_idx: Dict[tuple, dict],
                   scope: str = "s2s3") -> dict:
    """The rows rule E queries: every `unknown` count row (roles c and h) of a birth slot
    (fact in CHECKED_FACTS) in S2 or S3 (scope "s2s3") or in any regime ("all-birth"),
    never of a no_count_row slot, deduplicated on (index, name string, year).

    {"targets": [{index, model, prompt_name, name_string, fact, year, roles, regimes,
    slots}, ...] sorted by (index, name_string, year), roles / regimes / slots (slot_key)
    as sorted lists; "skipped": [{slot (slot_key), reason}, ...]} for slots with at least
    one `unknown` row whose rows carry no `name` string ("no name string") or two
    different ones ("different name strings: a | b"); those are not queried."""
    if scope not in ("s2s3", "all-birth"):
        raise ValueError("scope must be 's2s3' or 'all-birth', got %r" % (scope,))
    regimes = ("S2", "S3") if scope == "s2s3" else REGIMES
    targets: Dict[tuple, dict] = {}
    skipped = []
    for s in classified:
        if (s["kind"] not in CHECKED_FACTS or s["regime"] not in regimes
                or s["class"] == "no_count_row"):
            continue
        c_rows, h_rows = rows_for(s, counts_idx)
        tagged = ([("c", y, r) for y, r in zip(s["own_years"], c_rows)]
                  + [("h", y, r) for y, r in zip(s["h_years"], h_rows)])
        todo = [(role, y, r) for role, y, r in tagged if r is not None and r["status"] == UNKNOWN]
        if not todo:
            continue
        strings = slot_name_strings(c_rows, h_rows)
        if not strings:
            skipped.append({"slot": slot_key(s), "reason": "no name string"})
            continue
        if len(strings) > 1:
            skipped.append({"slot": slot_key(s),
                            "reason": "different name strings: " + " | ".join(strings)})
            continue
        index = MODEL_INDEX[s["model"]]
        for role, y, _ in todo:
            k = (index, strings[0], int(y))
            t = targets.setdefault(k, {
                "index": index, "model": s["model"], "prompt_name": s["name"],
                "name_string": strings[0], "fact": s["kind"], "year": int(y),
                "roles": set(), "regimes": set(), "slots": set()})
            t["roles"].add(role)
            t["regimes"].add(s["regime"])
            t["slots"].add(slot_key(s))
    out = []
    for k in sorted(targets):
        t = targets[k]
        out.append(dict(t, roles=sorted(t["roles"]), regimes=sorted(t["regimes"]),
                        slots=sorted(t["slots"])))
    return {"targets": out, "skipped": skipped}


def slot_name_strings(c_rows: list, h_rows: list) -> List[str]:
    """The distinct non-empty `name` strings (`name_string`) over a slot's c and h rows
    that exist (None rows skipped, any status), sorted. One string is the slot's name
    string; none or several and the slot is not queried (`phrase_targets`) and gets no
    present pair (`present_for`)."""
    return sorted({q for q in (name_string(r) for r in list(c_rows) + list(h_rows) if r is not None) if q})


def present_for(slot: dict, c_rows: list, h_rows: list, hit_keys) -> frozenset:
    """The (role, year) pairs of `slot` for `classify(..., present=...)`: every row that is
    `unknown` and whose (MODEL_INDEX[slot model], slot name string, year) is in `hit_keys`
    (the (index, name string, int year) of the phrase targets that hit). The slot name
    string is the one `phrase_targets` queries (`slot_name_strings`: the single distinct
    string over the slot's c and h rows), so a row with no `name` entry of its own still
    matches; a slot with no string or two different ones gets nothing. Rows are aligned
    with `own_years` / `h_years` (None: no row); the year comes from the slot, as an int.
    Only slots whose kind is in CHECKED_FACTS: a hit key carries no fact, and a birth
    phrase says nothing about a death count. ValueError on rows not aligned."""
    own, hy = tuple(slot["own_years"]), tuple(slot["h_years"])
    c_rows, h_rows = list(c_rows), list(h_rows)
    if len(c_rows) != len(own) or len(h_rows) != len(hy):
        raise ValueError("rows not aligned with own_years / h_years for %r" % (slot_key(slot),))
    if slot["kind"] not in CHECKED_FACTS:
        return frozenset()
    strings = slot_name_strings(c_rows, h_rows)
    if len(strings) != 1:
        return frozenset()
    index = MODEL_INDEX[slot["model"]]
    hits = set(hit_keys)
    out = set()
    for role, years, rows in (("c", own, c_rows), ("h", hy, h_rows)):
        for y, r in zip(years, rows):
            if r is not None and r["status"] == UNKNOWN and (index, strings[0], int(y)) in hits:
                out.add((role, int(y)))
    return frozenset(out)
