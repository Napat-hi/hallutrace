r"""Agreement numbers for one task's annotation packet: kappa, matrices and adjudication.

    .\.venv\Scripts\python.exe scripts\agreement.py compute results\gold_cut\biographies
    .\.venv\Scripts\python.exe scripts\agreement.py adjudicate results\gold_cut\biographies

Each annotator's file is `annotation/e1/<task>/annotator_<name>.csv`; point `--a`
and `--b` at those files (A and B as the packet's SEND.txt assigns them).

`compute` prints the numbers and writes agreement.json next to the packet. It works
on whatever both annotators have labelled so far, so a 10-item pilot gets its
inter-annotator kappa the same way the full 100 do. The verifier's kappa needs a gold
label on every labelled row, so it appears once adjudication is done.

`adjudicate` writes adjudication.csv: the rows where A and B differ, with both labels
and notes. The pair fills in `gold_label`, `tag` (`not a claim` takes the row out of
the verifier's "without" numbers) and `reason`; rerunning keeps what is
already written.

`cut <drawn task dir> --out <dir> --n 50 --packet <annotator.csv>` writes the first n
items' key.csv and manifest.json (stats recounted for n) for annotators who label only
those, after checking the packet lists the same first n item_ids.

Reads only the packet; needs no key or network. See `hallutrace/agreement.py`.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hallutrace import agreement as ag  # noqa: E402
from hallutrace import gold_cut  # noqa: E402


def _line(name, block):
    i = block["inter_annotator"]
    s = "  %-24s inter-annotator kappa %s, 95%% CI %s (agree %s, n=%d)" % (
        name, i["kappa"], i["ci95"], i["agreement"], i["n"])
    if "extractor" in block:
        e = block["extractor"]
        if "accuracy" in e:
            return s + " | extractor accuracy %s, 95%% CI %s (n=%d) -> %s%s" % (
                e["accuracy"], e["ci95"], e["n"], e["gate"],
                ", CI crosses %s" % e["edges_crossed"] if e["edge_crossing"] else "")
        return s + (" | extractor: gold missing on %d rows, run adjudicate" % e["gold_missing"]
                    if i["n"] else "")
    v = block["verifier"]
    if "kappa" in v:
        s += " | verifier kappa %s, 95%% CI %s (agree %s, n=%d) -> %s%s" % (
            v["kappa"], v["ci95"], v["agreement"], v["n"], v["gate"],
            ", CI crosses %s" % v["edges_crossed"] if v["edge_crossing"] else "")
    elif i["n"]:
        s += " | verifier: gold missing on %d rows, run adjudicate" % v["gold_missing"]
    return s


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=["compute", "adjudicate", "cut"])
    p.add_argument("task_dir")
    p.add_argument("--a", help="annotator A's file instead of <task_dir>/annotator_A.csv, "
                               "e.g. annotation/e1/biographies/annotator_purin.csv")
    p.add_argument("--b", help="annotator B's file instead of <task_dir>/annotator_B.csv")
    p.add_argument("--out", help="cut: where the first-n key and manifest go")
    p.add_argument("--n", type=int, default=50, help="cut: how many items (default 50)")
    p.add_argument("--packet", help="cut: the annotator packet; its item_ids must be the first n")
    args = p.parse_args(argv)
    if args.command == "cut":
        if not args.out:
            p.error("cut needs --out")
        try:
            paths = gold_cut.first_n(args.task_dir, args.out, args.n, args.packet)
        except ValueError as e:
            print("cut refused: %s" % e, file=sys.stderr)
            return 1
        with open(paths["manifest.json"], encoding="utf-8") as f:
            s = json.load(f)["stats"]
        print("%s: first %d, %d flagged (sampling ratio %s) -> %s"
              % (args.task_dir, s["drawn"], s["drawn_flagged"], s["sampling_ratio"], args.out))
        return 0
    packet = ag.load(args.task_dir, args.a, args.b)
    if args.command == "adjudicate":
        path = ag.write_adjudication(args.task_dir, packet)
        n = len(ag.disagreements(packet))
        print("%s: %d disagreements -> %s" % (packet["task"], n, path))
        return 0
    out = ag.compute(packet)
    path = os.path.join(args.task_dir, "agreement.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    s = out["sampling"]
    print("%s: %d of %d items labelled; natural base rate %s, drawn flagged %s"
          % (out["task"], out["labelled"], out["items"], s["natural_base_rate"], s["sampling_ratio"]))
    print(_line("all rows", out["all_rows"]))
    nac = out["not_a_claim"]
    if nac["n"] or nac["settled_not_a_claim"]:
        print(_line("without not a claim (%d)" % nac["n"], nac["without"]))
        print("  %d claim rows; %d rows marked by one annotator only stay in, for adjudication"
              % (nac["claim_rows"], nac["marked_by_one"]))
    if nac["settled_not_a_claim"]:
        print("  settled as not a claim at adjudication, out of the verifier's 'without' numbers"
              " only (%d rows left): %s"
              % (nac["verifier_rows"], ", ".join(nac["settled_not_a_claim"])))
    if nac["mentions_not_a_claim"]:
        print("  notes mention 'not a claim' but the tag isn't it, kept in both splits: %s"
              % ", ".join(nac["mentions_not_a_claim"]))
    tagged = ["%s A %d, B %d, both %d" % (t, c["A"], c["B"], c["both"])
              for t, c in out["tags"].items() if c["A"] or c["B"]]
    print("  tags (from %s): %s" % (out["tag_source"], "; ".join(tagged) or "none"))
    print("-> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
