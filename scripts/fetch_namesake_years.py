r"""Wikidata birth/death years of the Biographies namesakes.

    .\.venv\Scripts\python.exe scripts\fetch_namesake_years.py            # fetch, write the file
    .\.venv\Scripts\python.exe scripts\fetch_namesake_years.py --dry-run  # list the QIDs and the call count

The namesakes are, for every name with a year claim in results/bio_year_claims.csv, the
alias table's other QIDs for that name, the person's own QID left out. Their years
come from batched `wbgetentities` calls, 50 QIDs each (13 calls for 618 QIDs), sent one at
a time with the User-Agent and pause of scripts/build_alias_table.py, whose
`fetch_entities` makes the call. Writes data/alias_build/namesake_years.json with the fetch
time, this command and the claims file's hash. wikidata_snapshot.json and the alias table
are not touched.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace.claim_key import AliasTable  # noqa: E402
from scripts import build_alias_table as bat  # noqa: E402
from scripts.bio_year_claims import own_qids  # noqa: E402

OUT = ROOT / "data" / "alias_build" / "namesake_years.json"
CLAIMS = ROOT / "results" / "bio_year_claims.csv"
COMMAND = r".\.venv\Scripts\python.exe scripts\fetch_namesake_years.py"


def namesake_qids(claim_names, aliases: AliasTable, own: dict) -> list:
    """Every QID a claim name can denote other than its own, sorted."""
    out = set()
    for name in claim_names:
        out.update(q for q in aliases.candidates(name) if q != own.get(name))
    return sorted(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--claims", default=str(CLAIMS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true", help="list the QIDs and the call count; no network")
    args = ap.parse_args(argv)

    with open(args.claims, encoding="utf-8", newline="") as f:
        names = sorted({r["name"] for r in csv.DictReader(f)})
    with open(ROOT / "data" / "alias_build" / "wikidata_snapshot.json", encoding="utf-8") as f:
        snapshot = json.load(f)
    with open(ROOT / "data" / "alias_table.json", encoding="utf-8") as f:
        aliases = AliasTable(json.load(f))
    qids = namesake_qids(names, aliases, own_qids(snapshot))
    calls = math.ceil(len(qids) / bat.BATCH_SIZE)
    print("%d names with a year claim, %d namesake QIDs, %d wbgetentities calls"
          % (len(names), len(qids), calls))
    if args.dry_run:
        return 0

    entities = bat.fetch_entities(qids)
    # The names, not the claims file: rerunning bio_year_claims.py with these years rewrites
    # the file's namesake columns but not the names, which are what picks the QIDs.
    names_sha = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    data = {
        "description": "Wikidata P569/P570 years of the Biographies namesakes: "
                       "the alias table's other QIDs for every name with a year claim. "
                       "Separate from wikidata_snapshot.json, which build_alias_table.py built.",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": COMMAND,
        "claims_file": "results/bio_year_claims.csv",
        "claim_names": len(names),
        "claim_names_sha256": names_sha,
        "calls": calls,
        "qids": qids,
        "missing": [q for q in qids if q not in entities],
        "entities": {q: {k: entities[q][k] for k in ("label", "p31", "p569", "p570")}
                     for q in qids if q in entities},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    with_years = sum(1 for e in data["entities"].values() if e["p569"] or e["p570"])
    print("wrote %s: %d entities, %d with a birth or death year, %d missing"
          % (args.out, len(data["entities"]), with_years, len(data["missing"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
