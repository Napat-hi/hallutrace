"""Apply the unit filters to downloaded verifier output, raw left untouched.

    modal volume get hallutrace-data bexec/generations-prompts-v2/outputs results/bexec_v2/
    python scripts/posthoc_filter.py results/bexec_v2/outputs --out results/bexec_v2/outputs_filtered

Writes outputs_filtered/{model}/{unit}.csv, each row as the verifier wrote it
but with filtered unit columns plus `dropped_units` and `all_units_dropped`
(hallutrace/posthoc.py says what is dropped and why) and `zero_unit_rule`,
and a summary CSV next to it with raw and filtered counts per task and model,
zero unit rows included, so HALoGEN's own h can be recomputed.
"""

import argparse
import collections
import csv
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hallutrace import posthoc as ph  # noqa: E402

csv.field_size_limit(10 ** 8)


def _count(task, row):
    if task == "decompose":
        atoms = ph._atom_lines(row["atomic_units"])
        return len(atoms), sum(not v for _, v in atoms)
    units = [u for u in ph.parse_list(row["atomic_units"]) if u not in ("yes", "no")]
    return len(units), len(ph.parse_list(row["hallucinated_atomic_units"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stats = collections.defaultdict(collections.Counter)
    for path in sorted(glob.glob(os.path.join(args.outputs, "*", "*.csv"))):
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        task = ph.task_of_columns(rows[0].keys())
        if task is None:
            continue
        model = os.path.basename(os.path.dirname(path))
        out_rows = []
        for r in rows:
            fr = ph.filter_row(task, r)
            out_rows.append(fr)
            s = stats[(task, model)]
            u, h = _count(task, r)
            fu, fh = _count(task, fr)
            s["rows"] += 1
            s["units_raw"] += u
            s["hallucinated_raw"] += h
            s["units_filtered"] += fu
            s["hallucinated_filtered"] += fh
            s["all_units_dropped"] += fr["all_units_dropped"] == "True"
            s["zero_units_raw"] += u == 0
            s["zero_units_filtered"] += fu == 0
            if fr["zero_unit_rule"]:
                s["zero_" + fr["zero_unit_rule"]] += 1
            for d in json.loads(fr["dropped_units"]):
                s["dropped_" + d["reason"]] += 1
        dst = os.path.join(args.out, model, os.path.basename(path))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(out_rows)
    keys = ["rows", "units_raw", "units_filtered", "hallucinated_raw", "hallucinated_filtered",
            "all_units_dropped", "dropped_not_in_response", "dropped_none_token", "dropped_framing",
            "dropped_blank", "zero_units_raw", "zero_units_filtered", "zero_" + ph.NOT_HALLUCINATED,
            "zero_" + ph.OUT_OF_USABLE, "zero_" + ph.ANSWER_0_RIGHT, "zero_" + ph.ANSWER_0_WRONG]
    summary = os.path.join(args.out, "summary.csv")
    os.makedirs(args.out, exist_ok=True)
    with open(summary, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["task", "model"] + keys)
        for (task, model), s in sorted(stats.items()):
            w.writerow([task, model] + [s[k] for k in keys])
    tot = collections.defaultdict(collections.Counter)
    for (task, _), s in stats.items():
        tot[task].update(s)
    for task, s in sorted(tot.items()):
        print("%-26s rows %5d  units %5d -> %5d  hallucinated %5d -> %5d  all dropped %d"
              "  zero units %d -> %d"
              % (task, s["rows"], s["units_raw"], s["units_filtered"], s["hallucinated_raw"],
                 s["hallucinated_filtered"], s["all_units_dropped"],
                 s["zero_units_raw"], s["zero_units_filtered"]))
    print("wrote", summary)


if __name__ == "__main__":
    main()
