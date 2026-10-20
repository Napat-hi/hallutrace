"""Build the supplementary figures and their tables. Counting is in hallutrace/supplementary_figures.py.

    python scripts/posthoc_filter.py results/verifier/generations-revised-prompts/outputs --out outputs/v2_filtered
    python scripts/supplementary_figures.py drops outputs/v2_filtered results/verifier/generations-revised-prompts/inputs
    python scripts/supplementary_figures.py outcomes results/verifier/generations-revised-prompts/inputs
    python scripts/supplementary_figures.py h <t1_h_by_task_model.csv>
    python scripts/supplementary_figures.py confusion
    python scripts/supplementary_figures.py regimes <t1_h_by_task_model.csv>

Each figure is written as .pdf, .png and the .csv it plots, into --out (default
results/supplementary/figures):
f2_units_dropped (`drops`), f1_h_by_task_model (`h`), f3_confusion_matrices (`confusion`) and
f4_stability_regimes (`regimes`). `outcomes` writes T2's t2_outcomes.csv. T1 is
results/supplementary/t1_h_by_task_model.csv. Run posthoc_filter.py
before `drops`, so outputs_filtered/ exists. Labels are set in a Times face at 7 pt or larger
to match the paper.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hallutrace import supplementary_figures as figures  # noqa: E402

# Categorical slots 1 to 3 of the dataviz reference palette, in fixed order, one per
# drop reason. Checked with its validator against a white page: CVD and normal vision
# pass; aqua sits under 3:1 contrast, so the legend and the CSV carry it.
# `blank` lines are not units (hallutrace/supplementary_figures.py), so the figure leaves them out
# and only the CSV counts them.
REASON_COLORS = {"not_in_response": "#2a78d6", "none_token": "#eb6834", "framing": "#1baf7a"}
FIG_REASONS = tuple(REASON_COLORS)
INK, MUTED, GRID, PAGE = "#0b0b0b", "#52514e", "#e1e0d9", "#ffffff"
TASK_LABELS = {"biographies": "Biographies", "historical_events": "Historical events",
               "biographies_years": "Biographies years (year judge)",
               "false_presuppositions": "False presuppositions",
               "rationalization_numerical": "Rationalization (numerical)",
               "rationalization_binary": "Rationalization (binary)",
               "scientific_attribution": "Scientific attribution",
               "summarization": "Summarization", "simplification": "Simplification"}
MODEL_LABELS = {"olmo2-7b": "OLMo 2", "olmo17-7b": "OLMo 1.7", "dclm-7b": "DCLM",
                "redpajama-7b": "RedPajama", "pythia-69b": "Pythia"}
OUT = "results/supplementary/figures"
SERIF = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]   # TrueType Times faces: embed cleanly as Type 42


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "font.serif": SERIF,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42})
    return plt


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(out, name + ".{pdf,png,csv}"))


def cmd_drops(args):
    table = figures.filter_drops(args.outputs_filtered, args.inputs)
    cols = ["rows", "units_raw", "units_kept"] + list(figures.REASONS)
    figures.write_csv(os.path.join(args.out, "f2_units_dropped.csv"), ["task", "model"] + cols,
                 ([t, m] + [table[(t, m)][c] for c in cols] for t, m in figures.ordered(table)))
    tasks = figures.by_task(table)
    present = [t for t in figures.TASKS if t in tasks]
    figures.write_csv(os.path.join(args.out, "f2_units_dropped_by_task.csv"),
                 ["task"] + cols + ["dropped_share"],
                 ([t] + [tasks[t][c] for c in cols]
                  + ["%.4f" % (1 - tasks[t]["units_kept"] / tasks[t]["units_raw"])
                     if tasks[t]["units_raw"] else ""] for t in present))
    for t in present:
        c = tasks[t]
        print("%-26s units %6d -> %6d  dropped %5.1f%%" % (
            t, c["units_raw"], c["units_kept"],
            100 * (1 - c["units_kept"] / c["units_raw"]) if c["units_raw"] else 0))
    if args.no_pdf:
        return
    draw_drops(tasks, present, args.out, args.width)


def draw_drops(tasks, present, out, width):
    plt = _plt()
    rows = list(reversed(present))  # first task at the top
    fig, ax = plt.subplots(figsize=(width, 0.32 * len(rows) + 0.9))
    fig.patch.set_facecolor(PAGE)
    for y, t in enumerate(rows):
        raw = tasks[t]["units_raw"] or 1
        left = 0.0
        for r in FIG_REASONS:
            share = 100 * tasks[t][r] / raw
            if share:
                # A white edge leaves the 2px gap between stacked segments.
                ax.barh(y, share, left=left, height=0.55, color=REASON_COLORS[r],
                        edgecolor=PAGE, linewidth=1.0)
            left += share
        ax.text(left + 0.8, y, "%.0f%%" % left, va="center", ha="left", color=INK, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([TASK_LABELS[t] for t in rows], color=INK, fontsize=7)
    ax.set_xlabel("Units dropped (% of extracted units)", color=MUTED)
    top = max(100 * (1 - tasks[t]["units_kept"] / (tasks[t]["units_raw"] or 1)) for t in rows)
    ax.set_xlim(0, min(100, 10 * (int(top // 10) + 2)))
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0, labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=REASON_COLORS[r]) for r in FIG_REASONS]
    fig.legend(handles, [r.replace("_", " ") for r in FIG_REASONS], ncol=3, frameon=False,
               loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=7, handlelength=1, columnspacing=1)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out, "f2_units_dropped")
    plt.close(fig)


def cmd_outcomes(args):
    table = figures.outcomes(args.inputs)
    figures.write_csv(os.path.join(args.out, "t2_outcomes.csv"),
                 ["task", "model", "generations"] + list(figures.OUTCOMES),
                 ([t, m, sum(table[(t, m)].values())] + [table[(t, m)][o] for o in figures.OUTCOMES]
                  for t, m in figures.ordered(table)))
    print("wrote", os.path.join(args.out, "t2_outcomes.csv"))


def _t1(path):
    return figures._rows(path)


def cmd_h(args):
    """F1: h by task and model, each task with its gate number (all eight quarantined)."""
    grid = figures.h_grid(_t1(args.t1))
    gates = {t: figures.gate(json.load(open(os.path.join(args.verifier_check, t, "agreement.json")))) for t in figures.ALL_TASKS}
    figures.write_csv(os.path.join(args.out, "f1_h_by_task_model.csv"),
                 ["task", "model", "h", "gate_measure", "gate_value", "gate"],
                 ([t, m, "%.4f" % grid[(t, m)]] + [gates[t][0], "%.4f" % gates[t][1], gates[t][2]]
                  for t in figures.ALL_TASKS for m in figures.MODELS))
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#e4eefb", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
    data = [[grid[(t, m)] for m in figures.MODELS] for t in figures.ALL_TASKS]
    fig, ax = plt.subplots(figsize=(args.width, 0.27 * len(figures.ALL_TASKS) + 0.8))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=7, color="white" if v >= 0.55 else INK)
        ax.text(len(figures.MODELS) - 0.35, i, figures.gate_label(*gates[figures.ALL_TASKS[i]][:2]), ha="left", va="center",
                fontsize=7, color=MUTED)
    ax.text(len(figures.MODELS) - 0.35, -0.85, "gate", ha="left", va="center", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(figures.MODELS)), [MODEL_LABELS[m] for m in figures.MODELS], fontsize=7)
    ax.set_yticks(range(len(figures.ALL_TASKS)), [TASK_LABELS[t] for t in figures.ALL_TASKS], fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(figures.MODELS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(figures.ALL_TASKS))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.17)
    cb.set_label("h, share of usable samples with a hallucinated unit", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    _save(fig, args.out, "f1_h_by_task_model")
    plt.close(fig)


def cmd_confusion(args):
    """F3: each kappa-gated judge's verdicts against the adjudicated gold."""
    mats, extractors = [], []
    for t in figures.ALL_TASKS + ("biographies_years",):
        a = json.load(open(os.path.join(args.verifier_check, t, "agreement.json")))
        if t in figures.KAPPA_JUDGES:
            mats.append((t, figures.confusion(a)))
        else:
            extractors.append((t, figures.gate(a)))
    order = {t: i for i, t in enumerate(figures.KAPPA_JUDGES)}
    mats.sort(key=lambda x: order[x[0]])
    rows = []
    for t, m in mats:
        for i, r in enumerate(m["rows"]):
            for j, c in enumerate(m["cols"]):
                rows.append([t, r, c, m["cells"][i][j], "%.4f" % m["kappa"], "%.4f" % m["ci95"][0], "%.4f" % m["ci95"][1], m["gate"]])
    for t, (meas, val, g) in extractors:
        rows.append([t, "", "", "", "", "", "", g + " (accuracy %.4f)" % val])
    figures.write_csv(os.path.join(args.out, "f3_confusion_matrices.csv"),
                 ["task", "verifier", "gold", "items", "kappa", "ci95_low", "ci95_high", "gate"], rows)
    plt = _plt()
    ncol = 3
    nrow = (len(mats) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(args.width, 1.35 * nrow + 0.35))
    short = {"hallucinated": "halluc.", "supported": "supp.", "unverifiable": "unverif."}
    for ax, (t, m) in zip(axes.flat, mats):
        cells = m["cells"]
        top = max(map(max, cells)) or 1
        ax.imshow(cells, cmap="Greys", vmin=0, vmax=top * 1.6, aspect="auto")
        for i, row in enumerate(cells):
            for j, v in enumerate(row):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(m["cols"])), [short[c] for c in m["cols"]], fontsize=7)
        ax.set_yticks(range(len(m["rows"])), [short[r] for r in m["rows"]], fontsize=7)
        ax.tick_params(length=0)
        ax.set_title("%s\n\u03ba %.2f (%.2f to %.2f), %s" % (TASK_LABELS[t], m["kappa"], m["ci95"][0], m["ci95"][1], m["gate"]),
                     fontsize=7)
    for ax in list(axes.flat)[len(mats):]:
        ax.axis("off")
    fig.supylabel("judge", fontsize=7)
    fig.supxlabel("adjudicated gold (50 items per task)", fontsize=7)
    note = "Gated on extractor accuracy, no matrix: " + ", ".join(
        "%s %.2f" % (TASK_LABELS[t], v) for t, (_, v, _) in extractors)
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=7, color=MUTED)
    fig.tight_layout()
    _save(fig, args.out, "f3_confusion_matrices")
    plt.close(fig)


REGIME_PANELS = (("S1_items", "S1, sometimes right"), ("always_wrong_items", "always wrong"),
                 ("single_mention_items", "single mention"), ("no_h_items", "no h"))


def cmd_regimes(args):
    """F4: share of items per stability regime, by task and model.

    One small heatmap per regime, every cell printed as a percentage of the task's items,
    so each number can be read off the figure; a stacked bar per task and model hid the
    small segments and gave no values.
    """
    sh = figures.regime_shares(_t1(args.t1))
    figures.write_csv(os.path.join(args.out, "f4_stability_regimes.csv"),
                 ["task", "model", "items"] + [k for k, _ in REGIME_PANELS],
                 ([t, m, sum(sh[(t, m)].values())] + [sh[(t, m)][k] for k, _ in REGIME_PANELS]
                  for t in figures.ALL_TASKS for m in figures.MODELS))
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff", "#c9ddf7", "#6da7ec", "#256abf", "#0d366b"])
    fig, axes = plt.subplots(1, len(REGIME_PANELS), figsize=(args.width, 0.24 * len(figures.ALL_TASKS) + 0.9),
                             sharey=True)
    for ax, (k, title) in zip(axes, REGIME_PANELS):
        data = [[100.0 * sh[(t, m)][k] / sum(sh[(t, m)].values()) for m in figures.MODELS] for t in figures.ALL_TASKS]
        ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i, row in enumerate(data):
            for j, v in enumerate(row):
                ax.text(j, i, "%.0f" % v, ha="center", va="center", fontsize=7,
                        color="white" if v >= 55 else INK)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(len(figures.MODELS)), [MODEL_LABELS[m] for m in figures.MODELS], fontsize=7, rotation=45,
                      ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(figures.ALL_TASKS)), [TASK_LABELS[t] for t in figures.ALL_TASKS], fontsize=7)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([x - 0.5 for x in range(1, len(figures.MODELS))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(figures.ALL_TASKS))], minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
    fig.text(0.5, -0.01, "% of the 400 items per task and model; a cell read across the four panels adds up to 100 before rounding",
             ha="center", va="top", fontsize=7, color=MUTED)
    fig.tight_layout(w_pad=0.6)
    _save(fig, args.out, "f4_stability_regimes")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("drops", help="F2: units dropped by the unit filter")
    p.add_argument("outputs_filtered")
    p.add_argument("inputs")
    p.add_argument("--out", default=OUT)
    p.add_argument("--width", type=float, default=3.4, help="figure width in inches (one column)")
    p.add_argument("--no-pdf", action="store_true", help="tables only, no matplotlib needed")
    p.set_defaults(func=cmd_drops)
    p = sub.add_parser("outcomes", help="T2: generations per outcome")
    p.add_argument("inputs")
    p.add_argument("--out", default=OUT)
    p.set_defaults(func=cmd_outcomes)
    VERIFIER_CHECK = os.path.join(str(Path(__file__).resolve().parent.parent), "annotation", "verifier_check")
    for name, fn, helptext, needs_t1, width in (
            ("h", cmd_h, "F1: h by task and model, with each task's gate", True, 3.49),
            ("confusion", cmd_confusion, "F3: judge against gold, per kappa-gated task", False, 7.0),
            ("regimes", cmd_regimes, "F4: items per stability regime", True, 7.0)):
        p = sub.add_parser(name, help=helptext)
        if needs_t1:
            p.add_argument("t1", help="t1_h_by_task_model.csv from the public repo")
        p.add_argument("--verifier-check", default=VERIFIER_CHECK, help="annotation/verifier_check, for the gate numbers")
        p.add_argument("--out", default=OUT)
        p.add_argument("--width", type=float, default=width)
        p.set_defaults(func=fn)
    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
