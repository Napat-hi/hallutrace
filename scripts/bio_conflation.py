r"""The namesake CONFLATION check on Biographies year claims, against chance.

    .\.venv\Scripts\python.exe scripts\bio_conflation.py

Reads results/bio_year_claims.csv (scripts\bio_year_claims.py, run with the namesake
years) and data/alias_build/namesake_years.json (scripts\fetch_namesake_years.py).
Writes results/bio_conflation.json (the observed hits, both baselines and the verdict)
and results/bio_conflation_candidates.csv (one row per hit, for a look by hand).
Offline and seeded. See hallutrace/bio_conflation.py for the rules.

    .\.venv\Scripts\python.exe scripts\bio_conflation.py --select wikidata

The variant on claims wrong by Wikidata alone, no verifier label. Writes
results/bio_conflation_wikidata_only.json, _candidates.csv and _selection_diff.csv (the
claims that enter or leave against the default selection). The default files are untouched.
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
DIFF_COLUMNS = ["change", "why", "name", "kind", "label", "stated_years", "own_qid", "own_years",
                "wikidata_match", "namesake_hit", "model", "generation_id", "atom"]
DEFAULT_OUT = {"verifier": "bio_conflation.json", "wikidata": "bio_conflation_wikidata_only.json"}


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
    ap.add_argument("--select", choices=bc.SELECTIONS, default="verifier",
                    help="verifier: the hallucinated label (default); wikidata: wrong by Wikidata alone")
    ap.add_argument("--out", default=None, help="default: results/bio_conflation.json, or "
                    "results/bio_conflation_wikidata_only.json with --select wikidata")
    ap.add_argument("--n-perm", type=int, default=bc.N_PERM)
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(ROOT, "results", DEFAULT_OUT[args.select])
    wikidata = args.select == "wikidata"
    with open(args.claims, encoding="utf-8", newline="") as f:
        claims = list(csv.DictReader(f))
    with open(args.namesake_years, encoding="utf-8") as f:
        ns = json.load(f)
    entities = ns["entities"]
    res = bc.test(claims, entities, n_perm=args.n_perm, select=args.select)
    if wikidata:  # the default selection's file keeps its layout
        diff = bc.selection_diff(claims, entities)
        res["selection"] = {
            "select": "wikidata",
            "rule": "wikidata_match is False and the name has a namesake; no verifier label",
            "verifier_selection_claims": len(bc.population(claims, "verifier")),
            "enter": sum(r["change"] == "enters" for r in diff),
            "leave": sum(r["change"] == "leaves" for r in diff),
            "enter_hits": sum(r["change"] == "enters" and r["namesake_hit"] for r in diff),
            "leave_hits": sum(r["change"] == "leaves" and r["namesake_hit"] for r in diff)}
    res["inputs"] = {"claims": _rel(args.claims), "namesake_years": _rel(args.namesake_years),
                     "namesake_years_fetched_at": ns.get("fetched_at")}
    cands = bc.candidates(claims, entities, select=args.select)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    cand_path = os.path.splitext(args.out)[0] + "_candidates.csv"
    with open(cand_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAND_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(cands)
    if wikidata:
        diff_path = os.path.splitext(args.out)[0] + "_selection_diff.csv"
        with open(diff_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DIFF_COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(diff)
    p, perm, off = res["population"], res["permutation"], res["offset"]
    print("population: %d %s claims from %d names with namesakes"
          % (p["claims"], "wrong-by-Wikidata" if wikidata else "hallucinated wrong-year", p["names"]))
    if wikidata:
        s = res["selection"]
        print("against the verifier selection (%d): %d enter (%d hit), %d leave (%d hit) -> %s"
              % (s["verifier_selection_claims"], s["enter"], s["enter_hits"], s["leave"], s["leave_hits"], diff_path))
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
