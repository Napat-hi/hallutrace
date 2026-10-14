r"""Stability: h per item on all eight tasks, and the regime per slot for Biographies.

    .\.venv\Scripts\python.exe scripts\stability.py --bexec-root <root> [--v2-root <v2 root>]

<root> holds local copies of the first verification run (`bexec/generations`): `inputs/` with the
`.meta.csv` sidecars, `outputs_filtered/` for Biographies and `outputs/` for Historical Events.
<v2 root> holds the v2 run (`bexec/generations-prompts-v2`): `inputs/` and `outputs/` for the
six extraction tasks. --claims is results/bio_year_claims.csv. Writes to outputs\d1\
(gitignored) by default.

    items.csv          one row per (task, model, prompt): n_usable, h, regime from h (S1, always
                       wrong or single_mention). The six v2 tasks after the unit filter, Biographies
                       from --outputs, Historical Events as released
    items_raw.csv      the six v2 tasks as released, without the unit filter (with --v2-root)
    slots.csv          one row per (model, prompt, birth/death): right, wrong, absent, h, c,
                       regime (S1, S2, S3 or single_mention)
    slots_greedy.csv   the same with the greedy pass as a sixth sample (supplementary)
    summary.json

--outputs switches the Biographies items only. The slots come from --claims, which is built
from `outputs_filtered`; only claims whose generation's sidecar outcome is `ok` count, so a
verification run that sends abstentions can't leak into them.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hallutrace import stability as st  # noqa: E402


def write(path, cols, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cols), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def num(x):
    return "n/a" if x is None else "%.3f" % x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bexec-root", required=True)
    ap.add_argument("--v2-root")
    ap.add_argument("--outputs", default="outputs_filtered", choices=("outputs_filtered", "outputs"))
    ap.add_argument("--claims", default=os.path.join(ROOT, "results", "bio_year_claims.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "d1"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    skipped = []
    samples = st.bio_samples(args.bexec_root, args.outputs, skipped)
    items = st.items(samples) + st.items(st.task_samples(args.bexec_root, st.HE))
    raw = []
    for task in (st.V2_TASKS if args.v2_root else ()):
        filtered = st.task_samples(args.v2_root, task)
        if not filtered:
            ap.error("no %s generations under --v2-root %s" % (task, args.v2_root))
        items += st.items(filtered)
        raw += st.items(st.task_samples(args.v2_root, task, filtered=False))
    write(os.path.join(args.out, "items.csv"), st.ITEM_COLUMNS, items)
    if raw:
        write(os.path.join(args.out, "items_raw.csv"), st.ITEM_COLUMNS, raw)
    summary = {"items": st.summary(items), "items_raw": st.summary(raw), "skipped_units": skipped,
               "outputs": args.outputs, "v2_filter": "#23, posthoc.filter_row" if raw else None}
    for name, label in (("items", "items"), ("items_raw", "items as released, no unit filter")):
        if summary[name]:
            print("%s (any hallucinated unit per sample; Biographies from %s):" % (label, args.outputs))
        for key, s in summary[name].items():
            print("  %-26s %-13s %d items: S1 %d, always wrong %d (%d Low), single_mention %d, no h %d, "
                  "fewer than 5 samples %d, missing %d; response ratio %s, pooled h %s"
                  % (s["task"], s["model"], s["items"], s["S1"], s["always_wrong"], s["always_wrong_low_n"],
                     s[st.SINGLE_MENTION], s["no_h"], s["items_missing_samples"], s["n_missing"],
                     num(s["response_ratio"]), num(s["h_pooled"])))
    if not any(r["task"] == st.HE for r in items):
        print("  no historical_events under --bexec-root")
    if skipped:
        print("  skipped stray units: %s" % ", ".join(skipped))

    if os.path.exists(args.claims):
        with open(args.claims, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        with open(args.claims, encoding="utf-8", newline="") as f:
            claims, dropped = st.ok_claims(csv.DictReader(f), samples)
        summary["slots_from"] = {"claims": os.path.basename(args.claims), "sha256": sha,
                                 "built_from": "outputs_filtered", "claims_not_ok_dropped": dropped}
        print("slots from %s (sha256 %s, built from outputs_filtered); %d claims dropped, "
              "sidecar outcome not ok" % (os.path.basename(args.claims), sha[:8], dropped))
        summary["c_cut"] = st.C_CUT
        for name, greedy in (("slots", False), ("slots_greedy", True)):
            slots = st.slot_items(claims, greedy=greedy)
            write(os.path.join(args.out, name + ".csv"), st.SLOT_COLUMNS, slots)
            summary[name] = st.slot_summary(slots)
            print("%s (birth/death year; c from %d wrong samples, cut %.1f):"
                  % (name, st.MIN_WRONG, st.C_CUT))
            for model, s in summary[name].items():
                print("  %-13s %d slots: S1 %d, S2 %d, S3 %d, single_mention %d; %d S2/S3 Low"
                      % (model, s.get("slots", 0), s.get("S1", 0), s.get("S2", 0), s.get("S3", 0),
                         s.get(st.SINGLE_MENTION, 0), s.get("low_n", 0)))
    else:
        print("no %s: slots skipped" % args.claims)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("-> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
