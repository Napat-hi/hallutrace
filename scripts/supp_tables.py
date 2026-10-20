r"""Supplementary tables T1 and T2 from two runs of scripts/stability.py.

    .\.venv\Scripts\python.exe scripts\stability.py --verifier-root <root> --v2-root <v2 root> --out outputs\stability
    .\.venv\Scripts\python.exe scripts\stability.py --verifier-root <root> --v2-root <v2 root> --outputs outputs --out outputs\stability_outputs
    .\.venv\Scripts\python.exe scripts\supp_tables.py --stability outputs\stability --stability-outputs outputs\stability_outputs

t1_h_by_task_model.csv   h per task and model, pooled over the usable samples, with the regime
                         counts per item. `h` is the headline: the six v2 tasks after the unit
                         filter, Biographies after its fact filter, Historical Events as
                         released. `h_unfiltered` is the same before the filter (blank for
                         Historical Events, which has none).
t2_outcomes_by_task_model.csv
                         every sampled generation by outcome: usable, zero units, abstention,
                         empty, emptied by the tail rule, missing. They add up to `n_samples`.

Writes to results\supplementary\ by default. Offline; reads only the two summary.json files.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

T1_COLUMNS = ["task", "model", "items", "n_samples", "n_usable", "response_ratio", "h", "h_unfiltered",
              "S1_items", "always_wrong_items", "always_wrong_low_n_items", "single_mention_items",
              "no_h_items", "h_halogen_zero_units"]
T2_COLUMNS = ["task", "model", "n_samples", "usable", "zero_unit", "abstention", "empty", "emptied",
              "missing", "response_ratio"]
UNFILTERED_FROM_OUTPUTS = ("biographies",)   # the other tasks' unfiltered h is in items_raw


def load(path):
    with open(os.path.join(path, "summary.json"), encoding="utf-8") as f:
        return json.load(f)


def tables(stability, stability_outputs):
    t1, t2 = [], []
    for key, s in sorted(stability["items"].items()):
        task = s["task"]
        if task in UNFILTERED_FROM_OUTPUTS:
            unf = stability_outputs["items"][key]
        else:
            unf = stability["items_raw"].get(key)
        t1.append({"task": task, "model": s["model"], "items": s["items"], "n_samples": s["n_samples"],
                   "n_usable": s["n_usable"], "response_ratio": s["response_ratio"], "h": s["h_pooled"],
                   "h_unfiltered": "" if unf is None else unf["h_pooled"],
                   "S1_items": s["S1"], "always_wrong_items": s["always_wrong"],
                   "always_wrong_low_n_items": s["always_wrong_low_n"],
                   "single_mention_items": s["single_mention"], "no_h_items": s["no_h"],
                   "h_halogen_zero_units": s["h_pooled_halogen_zero_units"]})
        parts = [s["n_usable"], s["n_zero_unit"], s["n_abstention"], s["n_empty"], s["n_emptied"], s["n_missing"]]
        if sum(parts) != s["n_samples"]:
            raise ValueError("%s: outcomes add up to %d, not %d" % (key, sum(parts), s["n_samples"]))
        t2.append(dict(zip(T2_COLUMNS, [task, s["model"], s["n_samples"]] + parts + [s["response_ratio"]])))
    return t1, t2


def write(path, columns, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stability", required=True, help="stability.py output with the filtered Biographies facts")
    ap.add_argument("--stability-outputs", required=True, help="stability.py output run with --outputs outputs")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "supplementary"))
    args = ap.parse_args(argv)
    try:
        t1, t2 = tables(load(args.stability), load(args.stability_outputs))
    except (OSError, KeyError, ValueError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    write(os.path.join(args.out, "t1_h_by_task_model.csv"), T1_COLUMNS, t1)
    write(os.path.join(args.out, "t2_outcomes_by_task_model.csv"), T2_COLUMNS, t2)
    print("%d rows each -> %s" % (len(t1), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
