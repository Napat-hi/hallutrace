"""The namesake CONFLATION check on Biographies year claims, against chance.

A wrong birth or death year is a CONFLATION candidate when it equals the same kind of year
of a namesake: another person the name can denote (the alias table's other QIDs),
with years from data/alias_build/namesake_years.json. A wrong year can equal a namesake's
year by coincidence, so the count is reported only as an excess over chance.

Population: the hallucinated year claims whose stated year is not the person's own
(`wikidata_match` not True: a stated own year is a verifier error), from names with at least
one namesake. Observed: how many of them hit a namesake's year of the same kind.

Two baselines, fixed before either was computed:

- Permutation: each ambiguous name gets another ambiguous name's namesakes (a
  random derangement of the namesake sets), the claims stay as they are, and the hits are
  counted again, N_PERM times. p = (1 + permutations with at least the observed hits) /
  (1 + N_PERM).
- Year offset: the hits counted with the namesakes' years shifted by d = -10..-1, 1..10.
  If namesakes' years pull the models' years, exact hits (d = 0) stand above the nearby
  offsets; if not, d = 0 is one more draw. p = (1 + offsets with at least the observed
  hits) / (1 + 20), so 0.048 is the smallest it can be.

Verdict: "excess over chance" only when both p values are at most ALPHA. Otherwise "not
distinguishable from chance". The excess over each baseline's
mean is reported either way, as a lower bound on CONFLATION among these claims; nothing
here sees the corpus.
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence

PROPERTY = {"birth": "p569", "death": "p570"}
N_PERM = 10000
SEED = 20261004
OFFSETS = tuple(d for d in range(-10, 11) if d)
ALPHA = 0.05


def population(claims: Iterable[dict]) -> List[dict]:
    """Hallucinated claims with a wrong year, from names that have namesakes."""
    return [c for c in claims
            if c["label"] == "hallucinated" and c["wikidata_match"] != "True" and c["namesake_qids"].strip()]


def namesake_sets(claims: Iterable[dict]) -> Dict[str, List[str]]:
    """{name: [namesake QIDs]} from the claims' `namesake_qids` column."""
    return {c["name"]: c["namesake_qids"].split() for c in claims if c["namesake_qids"].strip()}


def years_of(entities: dict, qids: Sequence[str], kind: str, shift: int = 0) -> set:
    prop = PROPERTY[kind]
    return {y + shift for q in qids for y in (entities.get(q) or {}).get(prop, [])}


def hit(claim: dict, qids: Sequence[str], entities: dict, shift: int = 0) -> bool:
    stated = {int(y) for y in claim["stated_years"].split()}
    return bool(stated & years_of(entities, qids, claim["kind"], shift))


def count_hits(pop: Sequence[dict], sets: Dict[str, List[str]], entities: dict,
               mapping: Dict[str, str] = None, shift: int = 0) -> int:
    """Claims that hit a namesake year; `mapping` gives each name another name's namesakes."""
    return sum(hit(c, sets[(mapping or {}).get(c["name"], c["name"])], entities, shift) for c in pop)


def derangement(names: Sequence[str], rng: random.Random) -> Dict[str, str]:
    """A random mapping of names onto names with no name mapped to itself."""
    if len(names) < 2:
        raise ValueError("a derangement needs at least two names")
    while True:
        shuffled = list(names)
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(names, shuffled)):
            return dict(zip(names, shuffled))


def _quantile(sorted_vals: List[int], q: float) -> int:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def test(claims: Iterable[dict], entities: dict, n_perm: int = N_PERM, seed: int = SEED) -> dict:
    claims = list(claims)
    pop = population(claims)
    sets = namesake_sets(claims)
    names = sorted(sets)
    observed = count_hits(pop, sets, entities)

    rng = random.Random(seed)
    perm = sorted(count_hits(pop, sets, entities, derangement(names, rng)) for _ in range(n_perm))
    perm_mean = sum(perm) / len(perm)
    p_perm = (1 + sum(v >= observed for v in perm)) / (1 + n_perm)

    by_offset = {d: count_hits(pop, sets, entities, shift=d) for d in OFFSETS}
    off_mean = sum(by_offset.values()) / len(by_offset)
    p_off = (1 + sum(v >= observed for v in by_offset.values())) / (1 + len(by_offset))

    excess = p_perm <= ALPHA and p_off <= ALPHA
    return {
        "population": {"claims": len(pop), "names": len({c["name"] for c in pop}),
                       "ambiguous_names": len(names)},
        "observed": {"claims": observed,
                     "names": len({c["name"] for c in pop if hit(c, sets[c["name"]], entities)})},
        "permutation": {"n": n_perm, "seed": seed, "mean": round(perm_mean, 3),
                        "q025": _quantile(perm, 0.025), "q975": _quantile(perm, 0.975),
                        "p": round(p_perm, 4), "excess": round(observed - perm_mean, 3)},
        "offset": {"offsets": list(OFFSETS), "hits": {str(d): v for d, v in by_offset.items()},
                   "mean": round(off_mean, 3), "p": round(p_off, 4),
                   "excess": round(observed - off_mean, 3)},
        "alpha": ALPHA,
        "verdict": "excess over chance" if excess else "not distinguishable from chance",
    }


def candidates(claims: Iterable[dict], entities: dict) -> List[dict]:
    """The observed hits, one row per claim, with the namesakes it matched."""
    out = []
    for c in population(claims):
        qids = [q for q in c["namesake_qids"].split() if hit(c, [q], entities)]
        if qids:
            out.append({"name": c["name"], "kind": c["kind"], "stated_years": c["stated_years"],
                        "own_qid": c["own_qid"], "own_years": c["wikidata_years"],
                        "namesake_qids": " ".join(qids),
                        "namesake_labels": " | ".join((entities.get(q) or {}).get("label") or "" for q in qids),
                        "namesake_years": " | ".join(" ".join(str(y) for y in sorted(years_of(entities, [q], c["kind"])))
                                                     for q in qids),
                        "model": c["model"], "generation_id": c["generation_id"], "atom": c["atom"]})
    return out
