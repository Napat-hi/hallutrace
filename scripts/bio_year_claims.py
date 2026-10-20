r"""Biographies birth/death year claims, with Wikidata and the namesake check.

Reads a local copy of the Biographies verification files:

    modal volume get hallutrace-data verifier/generations/outputs_filtered/<model> <root>\outputs_filtered
    modal volume get hallutrace-data verifier/generations/inputs/<model> <root>\inputs

    .\.venv\Scripts\python.exe scripts\bio_year_claims.py <root>

Writes results/bio_year_claims.csv (one row per year claim) and
results/bio_year_claims_summary.json. Each person's own QID is the one the alias table decided
(`scripts/build_alias_table.py`, `_decide` over the snapshot and the overrides); the
namesakes are the alias table's other QIDs for the name, with their years from
data/alias_build/namesake_years.json when it exists (scripts/fetch_namesake_years.py).
Read-only, no network.
See hallutrace/bio_years.py for the rules.
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace import bio_years as by  # noqa: E402
from hallutrace.claim_key import AliasTable  # noqa: E402


NAMESAKE_YEARS = ROOT / "data" / "alias_build" / "namesake_years.json"


def own_qids(snapshot: dict) -> dict:
    """{prompt name: QID} as the alias table decided it; a name overridden to null is absent."""
    from scripts import build_alias_table as bat
    with open(ROOT / "data" / "alias_overrides.json", encoding="utf-8") as f:
        overrides = json.load(f)
    decided, _ = bat._decide(snapshot, overrides, bat.load_prompt_names())
    return decided


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="directory holding outputs_filtered/ and inputs/ (local copies)")
    ap.add_argument("--out", default=str(ROOT / "results" / "bio_year_claims.csv"))
    ap.add_argument("--namesake-years", default=str(NAMESAKE_YEARS),
                    help="namesakes' years (default: %(default)s, used if it exists)")
    args = ap.parse_args(argv)
    csv.field_size_limit(10 ** 9)

    with open(ROOT / "data" / "alias_build" / "wikidata_snapshot.json", encoding="utf-8") as f:
        snapshot = json.load(f)
    with open(ROOT / "data" / "alias_table.json", encoding="utf-8") as f:
        aliases = AliasTable(json.load(f))
    own = own_qids(snapshot)
    namesake_years = None
    if args.namesake_years and os.path.exists(args.namesake_years):
        with open(args.namesake_years, encoding="utf-8") as f:
            namesake_years = json.load(f)

    claims, responses = [], 0
    for path in sorted(glob.glob(os.path.join(args.root, "outputs_filtered", "*", "*.csv"))):
        if not os.path.getsize(path):
            continue
        model = os.path.basename(os.path.dirname(path))
        unit = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(args.root, "inputs", model, unit + ".csv"), encoding="utf-8", newline="") as f:
            inputs = {r["name"]: r for r in csv.DictReader(f)}
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                responses += 1
                inp = inputs[row["name"]]
                meta = {"generation_id": inp["generation_id"], "model": model,
                        "sample_index": inp.get("sample_index", ""), "unit_id": unit}
                claims += by.claims_in_row(row, meta, own.get(row["name"]),
                                           aliases.candidates(row["name"]), snapshot, namesake_years)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by.COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(claims)
    summary = dict(by.summarise(claims), responses_read=responses,
                   namesake_years=Path(args.namesake_years).name if namesake_years else None)
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", args.out, "and", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
