r"""The namesake CONFLATION check on Biographies year claims, against chance.

    .\.venv\Scripts\python.exe scripts\bio_conflation.py

Reads results/bio_year_claims.csv (scripts\bio_year_claims.py, run with the namesake
years) and data/alias_build/namesake_years.json (scripts\fetch_namesake_years.py).
Writes results/bio_conflation.json (the observed hits, both baselines and the verdict)
and results/bio_conflation_candidates.csv (one row per hit, for a look by hand).
Offline and seeded. See hallutrace/bio_conflation.py for the rules.
"""

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hallutrace import bio_conflation as bc  # noqa: E402

CAND_COLUMNS = ["name", "kind", "stated_years", "own_qid", "own_years", "namesake_qids",
                "namesake_labels", "namesake_years", "model", "generation_id", "atom"]


def _rel(path):
    """Repo-relative with forward slashes; the absolute path for a file on another drive."""
    try:
        return os.path.relpath(path, ROOT).replace(os.sep, "/")
    except ValueError:
        return os.path.abspath(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--claims", default=os.path.join(ROOT, "results", "bio_year_claims.csv"))
    ap.add_argument("--namesake-years", default=os.path.join(ROOT, "data", "alias_build", "namesake_years.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bio_conflation.json"))
    ap.add_argument("--n-perm", type=int, default=bc.N_PERM)
    args = ap.parse_args(argv)
    with open(args.claims, encoding="utf-8", newline="") as f:
        claims = list(csv.DictReader(f))
    with open(args.namesake_years, encoding="utf-8") as f:
        ns = json.load(f)
    entities = ns["entities"]
    res = bc.test(claims, entities, n_perm=args.n_perm)
    res["inputs"] = {"claims": _rel(args.claims), "namesake_years": _rel(args.namesake_years),
                     "namesake_years_fetched_at": ns.get("fetched_at")}
    cands = bc.candidates(claims, entities)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    cand_path = os.path.splitext(args.out)[0] + "_candidates.csv"
    with open(cand_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAND_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(cands)
    p, perm, off = res["population"], res["permutation"], res["offset"]
    print("population: %d hallucinated wrong-year claims from %d names with namesakes"
          % (p["claims"], p["names"]))
    print("observed: %d claims (%d names) equal a namesake's year"
          % (res["observed"]["claims"], res["observed"]["names"]))
    print("permutation baseline (%d, seed %d): mean %.2f, 95%% range %d to %d, p %.4f, excess %.2f"
          % (perm["n"], perm["seed"], perm["mean"], perm["q025"], perm["q975"], perm["p"], perm["excess"]))
    print("offset baseline (d = -10..10, not 0): mean %.2f, p %.4f, excess %.2f; hits %s"
          % (off["mean"], off["p"], off["excess"], off["hits"]))
    print("verdict: %s (both p <= %.2f needed)" % (res["verdict"], res["alpha"]))
    print("-> %s, %s" % (args.out, cand_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
