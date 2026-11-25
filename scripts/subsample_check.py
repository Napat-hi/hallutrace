"""Subsample check: regimes on 3, 4, 5 and 6 samples, the held-out sample and
greedy pass, and the exact-year baseline, from results/bio_year_claims.csv. Offline.

    python scripts/subsample_check.py [--claims results/bio_year_claims.csv]
                                      [--out results/subsample_check] [--force]

Writes subsets.csv (one row per subset), by_size.csv, held_out_sample.csv, held_out_greedy.csv
and exact_year.csv. See hallutrace/subsample.py for the definitions.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hallutrace import subsample as ss  # noqa: E402

PAPER_FIVE = {"S1": 233, "S2": 13, "S3": 207, "single_mention": 574}   # paper, Section IV
FILES = ("subsets.csv", "by_size.csv", "held_out_sample.csv", "held_out_greedy.csv", "exact_year.csv")


def write(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--claims", default=os.path.join(ROOT, "results", "bio_year_claims.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "subsample_check"))
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    a = ap.parse_args(argv)
    existing = [f for f in FILES if os.path.exists(os.path.join(a.out, f))]
    if existing and not a.force:
        print("refusing to overwrite %s in %s (use --force)" % (", ".join(existing), a.out), file=sys.stderr)
        return 1
    with open(a.claims, encoding="utf-8", newline="") as f:
        claims = list(csv.DictReader(f))
    os.makedirs(a.out, exist_ok=True)
    table = ss.subset_table(claims)
    five = [r for r in table if r["samples"] == 5][0]
    got = {k: five[k] for k in PAPER_FIVE}
    if a.claims == ap.get_default("claims") and got != PAPER_FIVE:
        print("five-sample regimes %s differ from the paper's %s" % (got, PAPER_FIVE), file=sys.stderr)
        return 2
    by_size = ss.summarise(table)
    outputs = {"subsets.csv": table, "by_size.csv": by_size,
               "held_out_sample.csv": [ss.held_out_sample(claims)],
               "held_out_greedy.csv": ss.held_out_greedy(claims),
               "exact_year.csv": ss.exact_year_baseline(claims)}
    for name, rows in outputs.items():
        write(os.path.join(a.out, name), rows)
    for r in by_size:
        print("%d samples: share at inference %.3f to %.3f, overturned %s of %s" % (
            r["samples"], r["share_inference_min"], r["share_inference_max"],
            r["S2_S3_overturned"], r["S2_S3_slot_subsets"]))
    print("wrote %d files to %s" % (len(outputs), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
