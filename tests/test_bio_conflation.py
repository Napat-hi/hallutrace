"""Tests for hallutrace/bio_conflation.py and scripts/bio_conflation.py. Synthetic claims, no network."""

import csv
import io
import json
import os
import random
import sys
import tempfile
from contextlib import redirect_stdout

from hallutrace import bio_conflation as bc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import bio_conflation as cli  # noqa: E402

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


def claim(name, stated, kind="birth", label="hallucinated", match="False", namesakes="", own_years="1800"):
    return {"name": name, "kind": kind, "label": label, "stated_years": str(stated), "wikidata_match": match,
            "wikidata_years": own_years, "own_qid": "Q0", "namesake_qids": namesakes,
            "model": "m", "generation_id": "g-%s-%s" % (name, stated), "atom": "%s was born in %s." % (name, stated)}


print("1. who is in the population")
cs = [claim("A", 1901, namesakes="QA1"),
      claim("A", 1800, match="True", namesakes="QA1"),               # the person's own year: a verifier error
      claim("A", 1901, label="supported", match="True", namesakes="QA1"),
      claim("B", 1950, namesakes=""),                                # no namesakes
      claim("C", 1777, match="", namesakes="QC1 QC2")]                # own years unknown: stays in
eq([(c["name"], c["stated_years"]) for c in bc.population(cs)], [("A", "1901"), ("C", "1777")],
   "hallucinated, not the own year, and the name has namesakes")
ents = {"QA1": {"p569": [1901], "p570": [1960]}, "QC1": {"p569": [1700]}, "QC2": {"p570": [1777]}}
eq(bc.count_hits(bc.population(cs), bc.namesake_sets(cs), ents), 1,
   "1901 is QA1's birth year; 1777 is QC2's death year, the wrong kind")
eq(bc.count_hits(bc.population(cs), bc.namesake_sets(cs), ents, shift=1), 0, "shifted by a year: no hit")

print("2. the derangement")
rng = random.Random(1)
for _ in range(50):
    m = bc.derangement(["a", "b", "c", "d"], rng)
    if any(k == v for k, v in m.items()) or sorted(m.values()) != ["a", "b", "c", "d"]:
        check(False, "a derangement maps no name to itself and uses every name once")
        break
else:
    check(True, "a derangement maps no name to itself and uses every name once")

print("3. planted conflation is found; coincidence is not")
# 40 names, each with one namesake born in 1700 + i. Planted: every claim states its own
# namesake's year. Chance: claims state years unrelated to their namesake.
names = ["N%02d" % i for i in range(40)]
ents = {"Q%02d" % i: {"p569": [1700 + 5 * i]} for i in range(40)}
planted = [claim(n, 1700 + 5 * i, namesakes="Q%02d" % i) for i, n in enumerate(names)]
res = bc.test(planted, ents, n_perm=500, seed=7)
eq((res["observed"]["claims"], res["verdict"]), (40, "excess over chance"), "every claim on its namesake's year: excess")
check(res["permutation"]["p"] <= 0.05 and res["offset"]["p"] <= 0.05, "both baselines reject chance")
r2 = random.Random(3)
chance = [claim(n, 1700 + 5 * r2.randrange(40), namesakes="Q%02d" % i) for i, n in enumerate(names)]
res2 = bc.test(chance, ents, n_perm=500, seed=7)
eq(res2["verdict"], "not distinguishable from chance", "years unrelated to the name's namesakes: no excess")
eq(bc.test(chance, ents, n_perm=500, seed=7), res2, "seeded: the same inputs give the same result")
eq(len(res2["offset"]["hits"]), 20, "20 offsets, -10..10 without 0")

print("4. the candidates file rows")
rows = bc.candidates(planted[:2], ents)
eq([(r["name"], r["namesake_qids"], r["namesake_years"]) for r in rows], [("N00", "Q00", "1700"), ("N01", "Q01", "1705")],
   "one row per hit, with the namesake and its year")

print("5. the command line")
d = tempfile.mkdtemp()
with open(os.path.join(d, "claims.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(planted[0]))
    w.writeheader()
    w.writerows(planted)
with open(os.path.join(d, "ns.json"), "w", encoding="utf-8") as f:
    json.dump({"fetched_at": "2026-10-04T00:00:00+00:00", "entities": ents}, f)
buf = io.StringIO()
with redirect_stdout(buf):
    code = cli.main(["--claims", os.path.join(d, "claims.csv"), "--namesake-years", os.path.join(d, "ns.json"),
                     "--out", os.path.join(d, "out.json"), "--n-perm", "200"])
eq(code, 0, "exit 0")
out = json.load(open(os.path.join(d, "out.json"), encoding="utf-8"))
eq((out["observed"]["claims"], out["inputs"]["namesake_years_fetched_at"]), (40, "2026-10-04T00:00:00+00:00"),
   "results and the fetch date of the namesake years")
check(os.path.exists(os.path.join(d, "out_candidates.csv")), "the candidates file next to it")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
