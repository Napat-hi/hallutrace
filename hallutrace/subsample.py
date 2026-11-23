"""Subsample check: how much the Biographies year-slot regimes depend on having
five samples, whether the regimes predict a generation left out of them, and how far an exact year
match stands above chance.

Everything here reads `results/bio_year_claims.csv` and assigns regimes with
`stability.slot_items`, the function behind the paper's 1,027 slots, so a subset of samples
is judged by exactly the paper's rule. No generation, verification or annotation.

Four parts:

1. Subsets. Every 3-of-5 and 4-of-5 subset of the sampled generations (indexes 1 to 5), the
   five samples themselves, and the five plus the greedy pass (a larger slot set, 1,269 slots
   on the paper's claims, since some slots state a year only in the greedy pass). Per subset:
   the regimes; the
   share at inference (S1 slots with a wrong sample, over those plus S2 and S3); the S1 slots
   found; and the overturn rate (of the slots that are S2 or S3 on the subset, the share that
   are S1 on all five samples; blank on five and on five plus greedy, where it is 0 by
   construction).
2. Held-out sample. For each 4-of-5 subset, the left-out sample in the slots that are S2 or
   S3 on the other four: silent, right or wrong.
3. Held-out greedy pass. The regimes never use the greedy answer. Where it states a year:
   right or wrong, by the slot's five-sample class (S1 never wrong, S1 with a wrong sample,
   S2 or S3).
4. Exact-year baseline. Every year a sampled generation states for a slot, against the
   Wikidata year: equal, one year below, one year above; and the matches expected if each
   stated year were compared with the Wikidata year of another slot of the same model and
   kind with a sampled year (the 1,027 slots), chosen at random. Over all slots, and over the slots stated twice or more and wrong
   at least once; counted per year claim and per sampled generation.
"""
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

from hallutrace import stability as st

SAMPLED = st.SAMPLED
GREEDY = st.GREEDY


def slot_key(row: dict) -> Tuple[str, str, str]:
    return (row["model"], row["prompt_id"], row["kind"])


def _claim_key(c: dict) -> Tuple[str, str, str]:
    return (c["model"], c["generation_id"].split("|")[1], c["kind"])


def regimes(claims: Sequence[dict], indexes: Iterable[int], greedy: bool = False) -> Dict[tuple, dict]:
    """slot_items on the claims of the given sample indexes only: the paper's rule, with the
    other samples treated as silent. Slots with no regime (no stated year) are left out."""
    keep = set(indexes)
    sub = [c for c in claims if int(c["sample_index"]) in keep or (greedy and int(c["sample_index"]) == GREEDY)]
    return {slot_key(r): r for r in st.slot_items(sub, greedy=greedy) if r["regime"]}


def _counts(rows: Dict[tuple, dict]) -> dict:
    reg = Counter(r["regime"] for r in rows.values())
    inference = sum(1 for r in rows.values() if r["regime"] == "S1" and r["n_wrong"] > 0)
    s23 = reg["S2"] + reg["S3"]
    return {"slots": len(rows), "S1": reg["S1"], "S1_with_wrong": inference, "S2": reg["S2"],
            "S3": reg["S3"], st.SINGLE_MENTION: reg[st.SINGLE_MENTION], "S2_S3": s23,
            "share_inference": round(inference / (inference + s23), 4) if inference + s23 else None}


def subset_table(claims: Sequence[dict]) -> List[dict]:
    """One row per subset: 3-of-5 (10), 4-of-5 (5), all five, and five plus greedy."""
    full = regimes(claims, SAMPLED)
    full_s1 = {k for k, r in full.items() if r["regime"] == "S1"}
    plan = [(k, idx, False) for k in (3, 4) for idx in combinations(SAMPLED, k)]
    plan += [(5, SAMPLED, False), (6, SAMPLED, True)]
    out = []
    for size, idx, greedy in plan:
        rows = regimes(claims, idx, greedy=greedy)
        c = _counts(rows)
        s23 = [k for k, r in rows.items() if r["regime"] in ("S2", "S3")]
        overturned = sum(1 for k in s23 if k in full_s1)
        s1 = {k for k, r in rows.items() if r["regime"] == "S1"}
        by_construction = size >= 5   # five samples are the reference; greedy adds to them
        out.append({"samples": size, "indexes": " ".join(str(i) for i in idx) + (" greedy" if greedy else ""),
                    **c, "S1_of_five_found": len(s1 & full_s1), "S1_not_S1_on_five": len(s1 - full_s1),
                    "S2_S3_overturned": None if by_construction else overturned,
                    "overturn_rate": None if by_construction or not s23 else round(overturned / len(s23), 4)})
    return out


def summarise(table: List[dict]) -> List[dict]:
    """Per sample size: totals over its subsets, and the range of the share at inference."""
    out = []
    for size in sorted({r["samples"] for r in table}):
        rows = [r for r in table if r["samples"] == size]
        shares = [r["share_inference"] for r in rows]
        s23 = sum(r["S2_S3"] for r in rows)
        over = None if size >= 5 else sum(r["S2_S3_overturned"] for r in rows)
        out.append({"samples": size, "subsets": len(rows),
                    "share_inference_min": min(shares), "share_inference_max": max(shares),
                    "share_inference_mean": round(sum(shares) / len(shares), 4),
                    "S1_min": min(r["S1"] for r in rows), "S1_max": max(r["S1"] for r in rows),
                    "S2_S3_slot_subsets": s23, "S2_S3_overturned": over,
                    "overturn_rate": round(over / s23, 4) if over is not None and s23 else None})
    return out


def _states(claims: Sequence[dict]) -> Dict[tuple, Dict[int, str]]:
    """Per slot and sample index: right, wrong or absent, by stability's rule."""
    by: Dict[tuple, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for c in claims:
        if c["wikidata_years"].strip():
            by[_claim_key(c)][int(c["sample_index"])].append(c)
    return {k: {i: st._slot_state(v.get(i, []))[0] for i in (GREEDY,) + SAMPLED} for k, v in by.items()}


def held_out_sample(claims: Sequence[dict]) -> dict:
    """For each 4-of-5 subset, the left-out sample in the slots S2 or S3 on the other four."""
    states = _states(claims)
    acc = Counter()
    for idx in combinations(SAMPLED, 4):
        left = (set(SAMPLED) - set(idx)).pop()
        for k, r in regimes(claims, idx).items():
            if r["regime"] in ("S2", "S3"):
                acc["slot_subsets"] += 1
                acc[states[k][left]] += 1
    stated = acc["right"] + acc["wrong"]
    return {"slot_subsets": acc["slot_subsets"], "silent": acc["absent"], "stated": stated,
            "right": acc["right"], "wrong": acc["wrong"],
            "right_share_of_stated": round(acc["right"] / stated, 4) if stated else None}


def held_out_greedy(claims: Sequence[dict]) -> List[dict]:
    """The greedy pass against the five-sample class of each slot. slot_items reads the greedy
    pass into `greedy` without using it for the regime, so all claims go in."""
    full = {slot_key(r): r for r in st.slot_items(claims) if r["regime"]}
    classes = {"S1 never wrong": lambda r: r["regime"] == "S1" and r["n_wrong"] == 0,
               "S1 with a wrong sample (at inference)": lambda r: r["regime"] == "S1" and r["n_wrong"] > 0,
               "S2 or S3": lambda r: r["regime"] in ("S2", "S3"),
               "single mention": lambda r: r["regime"] == st.SINGLE_MENTION}
    out = []
    for name, test in classes.items():
        g = Counter(r["greedy"] for r in full.values() if test(r))
        stated = g["right"] + g["wrong"]
        out.append({"class": name, "slots": sum(g.values()), "greedy_silent": g["absent"],
                    "greedy_right": g["right"], "greedy_wrong": g["wrong"],
                    "right_share_of_stated": round(g["right"] / stated, 4) if stated else None})
    return out


def _stated(claims: Sequence[dict], unit: str) -> Dict[tuple, List[list]]:
    """Per slot, the stated years of the sampled generations, grouped by `unit`: "claim" (one
    group per year claim, so a year two atoms of one generation state counts twice) or
    "sample" (one group per generation, each year once)."""
    by: Dict[tuple, Dict[object, list]] = defaultdict(lambda: defaultdict(list))
    for n, c in enumerate(claims):
        if c["wikidata_years"].strip() and int(c["sample_index"]) in SAMPLED:
            group = n if unit == "claim" else int(c["sample_index"])
            for y in c["stated_years"].split():
                if unit == "claim" or int(y) not in by[_claim_key(c)][group]:
                    by[_claim_key(c)][group].append(int(y))
    return {k: list(v.values()) for k, v in by.items()}


def exact_year_baseline(claims: Sequence[dict]) -> List[dict]:
    """Stated years against the Wikidata year: equal, one below, one above, and the expected
    number of exact matches against another slot of the same model and kind."""
    own: Dict[tuple, set] = {}
    for c in claims:
        if c["wikidata_years"].strip():
            own[_claim_key(c)] = {int(y) for y in c["wikidata_years"].split()}
    full = regimes(claims, SAMPLED)
    groups: Dict[tuple, List[tuple]] = defaultdict(list)   # slots with a sampled year, by model and kind
    for k in full:
        groups[(k[0], k[2])].append(k)
    scopes = {"all slots with a stated year": set(full),
              "slots stated twice or more and wrong at least once":
                  {k for k, r in full.items() if r["n_right"] + r["n_wrong"] >= 2 and r["n_wrong"] >= 1}}
    out = []
    for (name, keys), unit in [(s, u) for s in scopes.items() for u in ("claim", "sample")]:
        stated = _stated(claims, unit)
        acc = Counter()
        expected = 0.0
        for k in keys:
            mine = own[k]
            others = [own[o] for o in groups[(k[0], k[2])] if o != k]
            for years in stated.get(k, []):
                for y in years:
                    acc["stated"] += 1
                    if y in mine:
                        acc["exact"] += 1
                    elif y + 1 in mine:
                        acc["one_below"] += 1
                    elif y - 1 in mine:
                        acc["one_above"] += 1
                    if others:
                        expected += sum(y in o for o in others) / len(others)
        out.append({"scope": name, "unit": unit, "slots": len(keys), "stated_years": acc["stated"], "exact": acc["exact"],
                    "one_below": acc["one_below"], "one_above": acc["one_above"],
                    "expected_exact_other_slot": round(expected, 2)})
    return out
