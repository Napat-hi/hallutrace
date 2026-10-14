"""Draw the gold cut and write the annotation packets.

Reads verifier output downloaded from the data volume, for example:

    modal volume get hallutrace-data bexec/generations/outputs results/bexec_outputs
    python scripts/gold_cut.py results/bexec_outputs/outputs --out results/gold_cut --seed 20261004

Writes, per task, results/gold_cut/{task}/annotator_A.csv, annotator_B.csv,
key.csv and manifest.json. The annotator files never contain the verifier's
answer; key.csv does, and stays with whoever computes kappa.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hallutrace import gold_cut as gc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", help="directory holding {model}/{unit}.csv verifier outputs")
    ap.add_argument("--out", required=True, help="where the packets go")
    ap.add_argument("--seed", required=True, help="recorded in every manifest")
    ap.add_argument("--inputs", default=None,
                    help="the sidecars' inputs/ directory: needed for summarization and simplification, "
                         "and for leaving abstentions out of every task")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tasks", default="", help="comma separated, empty for all found")
    args = ap.parse_args()

    skipped, excluded = [], []
    rows = gc.read_outputs(args.outputs, args.inputs, skipped, excluded)
    if not rows:
        raise SystemExit("no verifier output under %s" % args.outputs)
    wanted = {t for t in args.tasks.split(",") if t}
    rows = [(r, s) for r, s in rows if not wanted or gc.task_of_row(r) in wanted]
    units = gc.build_units(rows)
    sources = {}
    for r, s in rows:
        sources.setdefault(gc.task_of_row(r), set()).add(s)

    print("%-27s %5s %6s %7s %6s %7s %s" % ("task", "pool", "flag", "base", "drawn", "ratio", "note"))
    for task in sorted(units):
        d = gc.draw(task, units[task], args.seed, n=args.n)
        # None means it could not be checked: no --inputs, so no sidecars.
        extra = {"skipped_files": skipped, "abstentions_excluded":
                 sum(g.split("|")[1].rsplit("-", 1)[0] == task for g in excluded) if args.inputs else None}
        if task in ("summarization", "simplification"):
            extra["unit_guard_loaded"] = gc.unit_guard_loaded()
        gc.write_packets(d, args.out, sources[task], extra)
        s = d.stats
        note = []
        if s["short_of_n"]:
            note.append("pool smaller than %d" % args.n)
        if not s["in_flagged_range"]:
            note.append("flagged outside 40 to 50")
        print("%-27s %5d %6d %6.1f%% %6d %6.1f%% %s" % (
            task, s["pool"], s["pool_flagged"], 100 * (s["base_rate"] or 0), s["drawn"],
            100 * (s["sampling_ratio"] or 0), "; ".join(note)))
    if skipped:
        print("skipped %d positional files with no sidecar (pass --inputs): recorded in each manifest"
              % len(skipped))
    if args.inputs is None:
        print("no --inputs: abstentions could not be left out")
    elif excluded:
        print("left out %d abstentions: counted in each manifest" % len(excluded))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
