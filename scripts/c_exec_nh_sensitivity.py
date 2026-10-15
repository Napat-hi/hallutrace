r"""Sensitivity of the Biographies slot types to false matches in n_h.

    .\.venv\Scripts\python.exe scripts\c_exec_nh_sensitivity.py --cache <phrase cache> [--bexec-root <verifier output copy>] [--force]

The counting rule was hand-checked on correct years only, so its precision on wrong years
(role h) is not measured (results/bio_counts/merged/meta.json, `n_h_precision`). This bounds what
false matches could do to the labels instead: it rebuilds scripts/c_exec_untyped.py's
--cache-only classification (slots from `stability.slot_items` on the claims `stability.ok_claims`
keeps; dominant wrong year; scope s2s3), checks it slot by slot against
the committed results/c_exec/untyped_slots.csv, then for each K treats the role h rows resting on
K or fewer matches as if every match were false, and classifies again:

    docs      `counted` in docs units with n_h <= K -> `zero`
    sampled   `counted` from a sampled pool (est_mentions) with mentions_matched <= K -> `unknown`
              (a sampled name with no match is unknown, never 0)
    combined  both

The phrase-check hits are reused. A row made `unknown` here has had no phrase check, so it can
only widen what the slot can reach: a worst case for each K, not an estimate, since real
precision is unknown.

The inputs (--claims, --counts, --tau, --bexec-root) go through scripts/c_exec_untyped.py's own
parser, with its defaults when left out, so the baseline is built exactly as that script builds
it. F5 is over the slots with a wrong sample (`c_exec2.f5_range`).

Writes nh_sensitivity.csv (one row per variant, K, regime and T5 change) and nh_sensitivity.md
to --out (default results/c_exec/, which holds untyped_slots.csv). Offline; the phrase cache is
opened read-only. Refuses to overwrite without --force. Exit 0 done, 1 refusal or error.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace import c_exec2 as c2  # noqa: E402
from hallutrace import c_exec_untyped as cu  # noqa: E402
from scripts import c_exec_untyped as sx  # noqa: E402

KS = (1, 3, 10)
VARIANTS = ("docs", "sampled", "combined")
OUT = ROOT / "results" / "c_exec"
INPUT_OPTIONS = ("--claims", "--counts", "--tau", "--bexec-root")   # passed on to c_exec_untyped.py's parser
NEEDS_NH = (cu.BOTH_HIGH, cu.BOTH_LOW, cu.WRONG_IN_DATA)


def perturb(counts, k, variant):
    """A copy of the counts rows with the role h rows of `variant` at K or fewer matches changed,
    and how many rows changed."""
    out, n = copy.deepcopy(counts), 0
    for r in out:
        if r["role"] != "h" or r["status"] != "counted":
            continue
        if variant in ("docs", "combined") and r["n_c_unit"] == "docs" and float(r["n_c"]) <= k:
            r["status"], r["n_c"] = "zero", "0"
            n += 1
        elif (variant in ("sampled", "combined") and r["n_c_unit"] == "est_mentions"
              and float(r["mentions_matched"]) <= k):
            r["status"] = cu.UNKNOWN
            n += 1
    return out, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True, help="phrase cache of c_exec_untyped.py (read-only)")
    for opt in INPUT_OPTIONS:
        ap.add_argument(opt, type=Path, default=None, help="as in c_exec_untyped.py (its default if left out)")
    ap.add_argument("--out", type=Path, default=OUT, help="the folder holding untyped_slots.csv; outputs go there")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = ap.parse_args(argv)
    slots_csv = args.out / sx.SLOTS_NAME
    csv_out, md_out = args.out / "nh_sensitivity.csv", args.out / "nh_sensitivity.md"
    if not args.force and (csv_out.exists() or md_out.exists()):
        print("error: outputs exist; --force to overwrite", file=sys.stderr)
        return 1

    # c_exec_untyped.py's own parser and defaults, so a change to its inputs can't leave this
    # script building a different baseline
    cli_argv = ["--cache", str(args.cache), "--cache-only"]
    for opt in INPUT_OPTIONS:
        v = getattr(args, opt[2:].replace("-", "_"))
        if v is not None:
            cli_argv += [opt, str(v)]
    cli = sx.build_parser().parse_args(cli_argv)
    try:
        inputs = sx.load_inputs(cli)
    except sx.pcli.CliError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    prep = sx.prepare(inputs, cli.scope)
    reader = sx.ReadOnlyCounts(args.cache)
    try:
        _, status, missing = sx.walk(prep["targets"], lambda i, p: reader.get(i, p) or sx.MISSING)
    finally:
        reader.close()
    if missing:
        print("error: %d phrases not in the cache" % len(missing), file=sys.stderr)
        return 1
    hit_keys = {k for k, s in status.items() if s == "hit"}

    def classify(cidx):
        present = {}
        for s in prep["slots"]:
            c_rows, h_rows = cu.rows_for(s, cidx)
            p = cu.present_for(s, c_rows, h_rows, hit_keys)
            if p:
                present[cu.slot_key(s)] = p
        return sx.classify_all(prep["slots"], cidx, inputs["tau"], present)

    base = classify(inputs["cidx"])
    if not slots_csv.exists():
        print("error: no %s to check the baseline against" % slots_csv, file=sys.stderr)
        return 1
    with open(slots_csv, encoding="utf-8", newline="") as f:
        committed = {(r["model"], r["prompt_id"], r["kind"]): r for r in csv.DictReader(f)}
    diff = [cu.slot_key(s) for s in base if cu.slot_key(s) not in committed
            or committed[cu.slot_key(s)]["t5_column"] != s["t5_column"]
            or committed[cu.slot_key(s)]["f5_point"] != s["f5_point"]]
    if diff or len(committed) != len(base):
        print("error: baseline differs from %s on %d slots" % (slots_csv.name, len(diff)), file=sys.stderr)
        return 1
    by_key = {cu.slot_key(s): s for s in base}
    exposed = Counter((s["regime"], s["t5_column"]) for s in base if s["t5_column"] in NEEDS_NH)
    f5_base = c2.f5_range(base)

    rows, summary = [], []
    for variant in VARIANTS:
        for k in KS:
            counts, changed = perturb(inputs["counts"], k, variant)
            after = classify(cu.counts_index(counts))
            trans, by_regime, f5_changed = Counter(), Counter(), Counter()
            for s in after:
                b = by_key[cu.slot_key(s)]
                if s["t5_column"] != b["t5_column"]:
                    trans[(s["regime"], b["t5_column"], s["t5_column"])] += 1
                    by_regime[s["regime"]] += 1
                if s["f5_point"] != b["f5_point"]:
                    f5_changed[(s["regime"], b["f5_point"], s["f5_point"])] += 1
            for (g, a, b2), n in sorted(trans.items()):
                rows.append({"variant": variant, "k": k, "h_rows_changed": changed, "what": "t5_column",
                             "regime": g, "from": a, "to": b2, "slots": n})
            for (g, a, b2), n in sorted(f5_changed.items()):
                rows.append({"variant": variant, "k": k, "h_rows_changed": changed, "what": "f5_point",
                             "regime": g, "from": a, "to": b2, "slots": n})
            f5_after = c2.f5_range(after)
            summary.append((variant, k, changed, sum(trans.values()), by_regime,
                            sum(f5_changed.values()), f5_after, f5_after == f5_base))

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "k", "h_rows_changed", "what", "regime", "from", "to", "slots"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    def f5_text(f5):
        return "; ".join("%s %d-%d" % (r["point"], r["lower"], r["upper"]) for r in f5["rows"]) + \
            "; open %d (n %d)" % (f5["open"], f5["n"])

    lines = [
        "# Biographies slot types if small n_h counts were false matches",
        "",
        "Written by `scripts/c_exec_nh_sensitivity.py`. Numbers only. The counting rule's precision on wrong years is",
        "not measured (`results/bio_counts/merged/meta.json`, `n_h_precision`), so this is a worst case per K:",
        "role h rows resting on K or fewer matches are treated as if every match were false. `docs`: a",
        "`counted` docs row with n_h <= K becomes `zero`. `sampled`: a `counted` row from a sampled pool",
        "with K or fewer matched windows becomes `unknown`, with no phrase check. `combined`:",
        "both. Baseline: `scripts/c_exec_untyped.py --cache-only` (slots from `stability.slot_items`;",
        "dominant wrong year, scope s2s3), equal slot by slot to `untyped_slots.csv`",
        "(T5 column and F5 point, %d slots, the slot table of" % len(base),
        "that file). historical_events' tau.",
        "",
        "Slots whose type needs n_h > 0, before: %d." % sum(exposed.values()),
        "",
        "| Regime | T5 column | Slots |",
        "|---|---|---|",
    ]
    lines += ["| %s | %s | %d |" % (g, t, n) for (g, t), n in sorted(exposed.items())]
    lines += ["", "| Variant | K | Role h rows changed | Slots changing T5 column | By regime | Slots changing F5 point | F5 range unchanged |",
              "|---|---|---|---|---|---|---|"]
    for variant, k, changed, n, by_reg, f5c, _, same in summary:
        lines.append("| %s | %d | %d | %d | %s | %d | %s |" % (
            variant, k, changed, n, ", ".join("%s %d" % (g, c) for g, c in sorted(by_reg.items())) or "none",
            f5c, "yes" if same else "no"))
    lines += ["", "F5 (lower-upper slots per point, over the slots with a wrong sample and an F5 point):", "",
              "- baseline: " + f5_text(f5_base)]
    lines += ["- %s, K %d: %s" % (v, k, f5_text(fa)) for v, k, _, _, _, _, fa, same in summary if not same]
    lines += ["", "Every change is in `nh_sensitivity.csv`.", ""]
    md_out.write_text("\n".join(lines), encoding="utf-8")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
