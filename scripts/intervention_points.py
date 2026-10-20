r"""Labels for the Biographies year slots, the tau sweep and Biographies' own tau.

    .\.venv\Scripts\python.exe scripts\intervention_points.py --cache <phrase cache> [--verifier-root <verifier output copy>] [--force]

One process, offline. Reads the claims, counts and phrase presence once, as
scripts/slot_types.py builds them (`stability.ok_claims` on the verifier output copy at --verifier-root,
`stability.slot_items`, the phrase cache on a read-only connection), then classifies the
slots at every tau (`slot_types.classify`) and labels them (hallutrace/intervention_points.py):

  headline     historical_events' tau, results/slot_types/he_tau.csv, variant headline_406, 10th
               percentile
  sweep        the same file at every percentile from 5 to 20, one file with a `percentile` column
  sensitivity  Biographies' own tau, results/bio_counts/merged/tau.csv, 10th percentile: reported
               beside the headline, never in its place

Never-wrong S1 slots (no wrong sample) get no label and are counted on their own line. Confidence is intervention_points's rule v3; its sweep_point reason uses each slot's F5
point at every percentile of the headline file. The phrase targets do not depend on tau (a
slot's class before the phrase check does not); a run whose targets differ at some percentile is
refused, since it would need a request.

Outputs in --out (default results/intervention_points/): labels.csv (headline), labels_sweep.csv (SWEEP_COLUMNS
only; the others are in labels.csv), labels_bio_tau.csv, t5.csv, f5_by_model.csv (F5's input: each
model's points with their confidence), summary.md, meta.json. All or nothing (temporary names first);
existing outputs are refused without --force. No request is ever sent: a phrase a walk needs
that is not in the cache is refused, listing it (scripts/slot_types.py's live run fills the
cache). Never a window cache.

Exit 0 done; 1 refusal or error (message on stderr, no traceback); 2 bad command line.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace import intervention_points as c2  # noqa: E402
from hallutrace import he_tau as ct  # noqa: E402
from hallutrace import slot_types as cu  # noqa: E402
from scripts import slot_types as cli  # noqa: E402
from scripts import pilot_counts as pcli  # noqa: E402
from scripts.bio_counts import git_head  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "intervention_points"
DEFAULT_BIO_TAU = ROOT / "results" / "bio_counts" / "merged" / "tau.csv"
DEFAULT_EXPECTED_BIO_TAU = ("v4_olmo-mix-1124_llama=1,v4_dclm-baseline_llama=2,v4_dolma-v1_7_llama=1,"
                            "v4_rpj_llama_s4=1,v4_piletrain_llama=3")
SCRIPT_VERSION = "1"
HEADLINE_P = 10
SWEEP = tuple(range(5, 21))
BIO_TAU_REQUIRED = ("index", "percentile", "tau")
LABELS_NAME, SWEEP_NAME, BIO_NAME = "labels.csv", "labels_sweep.csv", "labels_bio_tau.csv"
T5_NAME, SUMMARY_NAME, META_NAME = "t5.csv", "summary.md", "meta.json"
F5_MODEL_NAME = "f5_by_model.csv"
OUTPUTS = (LABELS_NAME, SWEEP_NAME, BIO_NAME, T5_NAME, F5_MODEL_NAME, SUMMARY_NAME, META_NAME)
T5_COLUMNS = ["model", "regime", "t5_column", "slots"]
# labels_sweep.csv keeps the key, the tau and what can move with it
SWEEP_COLUMNS = ["prompt_id", "model", "kind", "percentile", "tau", "phase2_type", "t5_column",
                 "intervention_point", "confidence", "confidence_reasons"]
F5_MODEL_COLUMNS = ["model", "point", "lower", "upper", "share_lower", "share_upper", "n", "high", "medium", "low"]
ALL = "all"
NO_COUNT_ROW = "no count row"
NEVER_WRONG = "never wrong (no label)"
T5_ORDER = cu.TYPES + (cu.MERGED_RECALL, cu.MERGED_LONG_TAIL, cu.UNTYPED_LABEL)
MODELS = tuple(cu.MODEL_INDEX)
SHORT = {"olmo2-7b": "OLMo 2", "dclm-7b": "DCLM", "olmo17-7b": "OLMo 1.7", "redpajama-7b": "RedPajama",
         "pythia-69b": "Pythia"}
F5_POINTS = (cu.INFERENCE, cu.DATA, cu.PROCESS, cu.TOGETHER, cu.BOUNDARY, cu.UNDETERMINED)
# The slot that carried s2_rho_near_half under the earlier sample rule; the current rule moves it
# from S2 to S1.
RHO_SLOT = ("redpajama-7b", "biographies-00665", "death")
RUN_FILES = ("scripts/intervention_points.py", "scripts/slot_types.py", "hallutrace/intervention_points.py",
             "hallutrace/slot_types.py", "hallutrace/stability.py")   # the code a run's numbers come from
DECISIONS = ["slot regimes, merged and untyped slots, the open band; a sample stating the right year is right; "
             "never-wrong slots unlabelled",
             "counting: near_or_cue matcher, full window, no stripped short forms", "types without Group B",
             "tau from historical_events", "an unknown count (a sampled pool with no match) is never 0",
             "only Biographies year slots are labelled", "no LLM step for year slots", "confidence rule v3"]


# ---------------------------------------------------------------------------
# Command line and inputs
# ---------------------------------------------------------------------------


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--claims", type=Path, default=cli.DEFAULT_CLAIMS)
    ap.add_argument("--counts", type=Path, default=cli.DEFAULT_COUNTS,
                    help="merged counts (the counting cell, roles c and h)")
    ap.add_argument("--tau", type=Path, default=cli.DEFAULT_TAU,
                    help="historical_events tau: the headline (percentile %d) and the sweep" % HEADLINE_P)
    ap.add_argument("--expected-tau", type=cli.tau_arg, default=cli.tau_arg(cli.DEFAULT_EXPECTED_TAU),
                    metavar="INDEX=TAU,...", help="refuse unless --tau's headline at percentile %d is this" % HEADLINE_P)
    ap.add_argument("--bio-tau", type=Path, default=DEFAULT_BIO_TAU,
                    help="Biographies' own tau (no variant column), the sensitivity run")
    ap.add_argument("--expected-bio-tau", type=cli.tau_arg, default=cli.tau_arg(DEFAULT_EXPECTED_BIO_TAU),
                    metavar="INDEX=TAU,...", help="refuse unless --bio-tau at percentile %d is this" % HEADLINE_P)
    ap.add_argument("--cache", type=Path, required=True, help="the phrase cache, read-only; never a window cache")
    ap.add_argument("--verifier-root", type=Path, default=cli.DEFAULT_VERIFIER_ROOT,
                    help="local copy of the verifier inputs/ and %s/ (stability.bio_samples)" % cli.VERIFIER_OUTPUTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="overwrite this script's outputs in --out")
    return ap


def read_taus(args):
    """({percentile: {index: tau}} from --tau's headline rows, {index: tau} from --bio-tau at the
    headline percentile), through intervention_points.tau_rows and its refusals."""
    he_rows = cli.read_csv(args.tau, cli.TAU_REQUIRED, "tau file")
    bio_rows = cli.read_csv(args.bio_tau, BIO_TAU_REQUIRED, "Biographies tau file")
    pcts = tuple(c2.percentiles(he_rows, variant=ct.HEADLINE))
    if pcts != SWEEP:
        raise pcli.CliError("%s: %s percentiles %s, expected 5 to 20" % (args.tau, ct.HEADLINE, list(pcts)))
    try:
        he = {p: c2.tau_rows(he_rows, p, variant=ct.HEADLINE,
                             expected=args.expected_tau if p == HEADLINE_P else None) for p in SWEEP}
        bio = c2.tau_rows(bio_rows, HEADLINE_P, variant=None, expected=args.expected_bio_tau)
    except ValueError as e:
        raise pcli.CliError("tau: %s" % e)
    return he, bio


def presence(inputs, prep, cache):
    """({slot key: present pairs}, walk stats): the phrase check from the cache, read-only, as
    scripts/slot_types.py's cache-only run does it."""
    ro = cli.ReadOnlyCounts(cache)
    try:
        rows, status, missing = cli.walk(prep["targets"], lambda i, q: ro.get(i, q) or cli.MISSING)
    finally:
        ro.close()
    if missing:
        raise pcli.CliError("%d target(s) reach a phrase not in the cache (a request would be needed):\n%s" % (
            len(missing), "\n".join("  %s\t%s" % m for m in missing)))
    hit_keys = {k for k, s in status.items() if s == "hit"}
    present = {}
    for s in prep["slots"]:
        c_rows, h_rows = cu.rows_for(s, inputs["cidx"])
        p = cu.present_for(s, c_rows, h_rows, hit_keys)
        if p:
            present[cu.slot_key(s)] = p
    return present, {"targets": len(prep["targets"]), "hits": len(hit_keys), "phrases": len(rows)}


def check_targets(prep, inputs, taus):
    """Refuse if the phrase targets at any tau differ from the headline's (they should not)."""
    base = sorted(cli.tkey(t) for t in prep["targets"])
    for label, tau in taus:
        before = cli.classify_all(prep["slots"], inputs["cidx"], tau)
        got = sorted(cli.tkey(t) for t in cu.phrase_targets(before, inputs["cidx"], scope="s2s3")["targets"])
        if got != base:
            raise pcli.CliError("the phrase targets at %s differ from the headline's (%d against %d); a request "
                                "would be needed" % (label, len(got), len(base)))


# ---------------------------------------------------------------------------
# Labels, tables and comparisons
# ---------------------------------------------------------------------------


def changed_run_files():
    """RUN_FILES that differ from the git head at run time (`git status --porcelain`), or None
    if git cannot say: the provenance's git head names the code only when this is empty."""
    try:
        p = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain", "--"] + list(RUN_FILES),
                           cwd=str(ROOT), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return sorted(line[3:].strip() for line in p.stdout.splitlines() if line.strip())


def labels_for(classified, tau, sweep):
    """intervention_points.label for every slot with a wrong sample; never-wrong slots get none."""
    return [c2.label(s, tau[cu.MODEL_INDEX[s["model"]]], sweep[cu.slot_key(s)])
            for s in classified if not c2.never_wrong(s)]


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (tuple, list, set, frozenset)):
        return "; ".join(str(x) for x in v)
    if isinstance(v, float):
        return cli._num(v)
    return str(v)


def label_rows(labels, **extra):
    return [dict(extra, **{k: fmt(r[k]) for k in c2.LABEL_COLUMNS}) for r in labels]


def t5_column(r):
    return r["t5_column"] or NO_COUNT_ROW


def t5_rows(classified, labels):
    """t5.csv: labelled slots per (model, regime, T5 column, or `no count row`), then the
    never-wrong S1 slots per model; zero cells left out."""
    n = Counter((r["model"], r["regime"], t5_column(r)) for r in labels)
    nw = c2.never_wrong_counts(classified)
    out = []
    for m in MODELS:
        for g in cu.REGIMES:
            for col in T5_ORDER + (NO_COUNT_ROW,):
                if n[(m, g, col)]:
                    out.append({"model": m, "regime": g, "t5_column": col, "slots": n[(m, g, col)]})
        if nw[m]:
            out.append({"model": m, "regime": "S1", "t5_column": NEVER_WRONG, "slots": nw[m]})
    return out


def shares_by_model(classified, labels):
    """{quantity: {model: share}} for the ordering test: each T5 column's share of the model's
    labelled slots with a count row, and each F5 point's lower and upper share of the model's F5
    slots."""
    q = defaultdict(dict)
    for m in MODELS:
        mine = [r for r in labels if r["model"] == m and r["t5_column"]]
        cnt = Counter(r["t5_column"] for r in mine)
        for col in T5_ORDER:
            q["T5 %s" % col][m] = cnt[col] / len(mine) if mine else None
        f5 = c2.f5_range(s for s in classified if s["model"] == m)
        rows = {r["point"]: r for r in f5["rows"]}
        for p in F5_POINTS:
            q["F5 %s, lower" % p][m] = rows[p]["share_lower"]
            q["F5 %s, upper" % p][m] = rows[p]["share_upper"]
    return q


def reversals(by_pct):
    """{quantity: [(model a, model b, percentiles a above b, percentiles b above a)]}: a pair of
    models whose order swaps somewhere in the sweep (ties do not count as either order)."""
    out = {}
    quantities = by_pct[HEADLINE_P].keys()
    for qn in quantities:
        found = []
        for i, a in enumerate(MODELS):
            for b in MODELS[i + 1:]:
                up, down = [], []
                for p in SWEEP:
                    x, y = by_pct[p][qn][a], by_pct[p][qn][b]
                    if x is None or y is None:
                        continue
                    if x > y:
                        up.append(p)
                    elif x < y:
                        down.append(p)
                if up and down:
                    found.append((a, b, up, down))
        out[qn] = found
    return out


def changes(a, b, field):
    """Slots (keys) whose `field` differs between two classified lists of the same slots."""
    bk = {cu.slot_key(s): s for s in b}
    return [(cu.slot_key(s), s[field], bk[cu.slot_key(s)][field]) for s in a if s[field] != bk[cu.slot_key(s)][field]]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _pct(x):
    return cli._pct(x)


def _rng(lo, hi, f=str):
    return f(lo) if lo == hi else "%s to %s" % (f(lo), f(hi))


def _key(k):
    return " ".join(k)


def _reg(g):
    return "`%s`" % g if g == cu.SINGLE_MENTION else g


def _tau_line(tau):
    return ", ".join("%s %s" % (SHORT[m], cli._num(tau[cu.MODEL_INDEX[m]])) for m in MODELS)


def f5_model_rows(classified, labels):
    """f5_by_model.csv, F5's input: per model, then `all`, intervention_points.f5_range's rows (each point's
    lower and upper count and share over the model's F5 slots) with the confidence of the slots fixed
    at the point (training data and process together: those fixed at either), then the open band
    (no confidence: an open slot has no point)."""
    out = []
    for m in MODELS + (ALL,):
        f5 = c2.f5_range(s for s in classified if m == ALL or s["model"] == m)
        conf = Counter((r["intervention_point"], r["confidence"]) for r in labels
                       if (m == ALL or r["model"] == m) and r["confidence"])
        for r in f5["rows"]:
            pts = (cu.DATA, cu.PROCESS) if r["point"] == cu.TOGETHER else (r["point"],)
            k = {lvl: sum(conf[(pt, lvl)] for pt in pts) for lvl in (c2.HIGH, c2.MEDIUM, c2.LOW)}
            out.append({"model": m, "point": r["point"], "lower": r["lower"], "upper": r["upper"],
                        "share_lower": cli._share4(r["share_lower"]), "share_upper": cli._share4(r["share_upper"]),
                        "n": f5["n"], "high": k[c2.HIGH], "medium": k[c2.MEDIUM], "low": k[c2.LOW]})
        for r in cli.open_band(f5):
            out.append({"model": m, "point": r["point"], "lower": r["lower"], "upper": r["upper"],
                        "share_lower": cli._share4(r["share_lower"]), "share_upper": cli._share4(r["share_upper"]),
                        "n": f5["n"], "high": "", "medium": "", "low": ""})
    return out


def f5_table(f5):
    rows = {r["point"]: r for r in f5["rows"]}
    L = ["| Intervention point | Share | Slots |", "|---|---|---|"]
    for p in F5_POINTS:
        r = rows[p]
        L.append("| %s | %s | %s |" % (p, _rng(r["share_lower"], r["share_upper"], _pct), _rng(r["lower"], r["upper"])))
    for r in cli.open_band(f5):
        L.append("| %s | %s | %s |" % (r["point"], _pct(r["share_lower"]), r["lower"]))
    return L


def t5_table(labels, models):
    n = Counter((r["regime"], t5_column(r)) for r in labels if r["model"] in models)
    L = ["| T5 column | S1 | S2 | S3 | `single_mention` | All |", "|---|---|---|---|---|---|"]
    for col in T5_ORDER + (NO_COUNT_ROW,):
        vals = [n[(g, col)] for g in cu.REGIMES]
        if sum(vals):
            L.append("| %s | %s | %d |" % (col, " | ".join(str(v) for v in vals), sum(vals)))
    tot = [sum(n[(g, c)] for c in T5_ORDER + (NO_COUNT_ROW,)) for g in cu.REGIMES]
    L.append("| all labelled | %s | %d |" % (" | ".join(str(v) for v in tot), sum(tot)))
    return L


def summary_markdown(run):
    lab, cls, f5 = run["labels"], run["classified"], run["f5"]
    nw = run["never_wrong"]
    regs = Counter(s["regime"] for s in cls)
    L = ["# Labels: Biographies birth and death year slots", "",
         "Rule `%s`. Slots from `stability.slot_items` on the claims `stability.ok_claims` keeps (a sample stating "
         "the right year is right), types from `slot_types.classify` (the tracker conditions, Group B left out), "
         "the point from the decision rule, tau from historical_events "
         "(headline_406, 10th percentile): %s." % (c2.RULE_VERSION, _tau_line(run["tau"][HEADLINE_P])), "",
         "Slots with a Wikidata year: %s (%s). Never-wrong S1 slots (no wrong sample: not a hallucination, no "
         "label): %s. Labelled slots: %s. Labels: `labels.csv`; the sweep: `labels_sweep.csv`; "
         "Biographies' own tau: `labels_bio_tau.csv`; T5 counts: `t5.csv`." % (
             cli._int(len(cls)), ", ".join("%s %s" % (_reg(g), cli._int(regs[g])) for g in cu.REGIMES),
             cli._int(sum(nw.values())), cli._int(len(lab))), "",
         "Placement: Biographies' T5 columns and F5 stay in the body if the year judge's kappa is 0.667 or above, "
         "and go to the supplementary material unchanged below it. The kappa is in "
         "`annotation/verifier_check/biographies_years/agreement.json`; placement is not decided here.", ""]

    # (a) T5
    L += ["## (a) T5: Phase 2 type by regime", "",
          "Labelled slots only. A slot's T5 column is its type, its merged type (rule C) or "
          "`%s`." % cu.UNTYPED_LABEL, "", "All models:", ""] + t5_table(lab, MODELS) + [""]
    for m in MODELS:
        L += ["%s (`%s`):" % (SHORT[m], m), ""] + t5_table(lab, (m,)) + [""]
    ncr = Counter(r["model"] for r in lab if r["no_count_row"])
    ncr_reg = Counter(r["regime"] for r in lab if r["no_count_row"])
    L += ["Own lines, not T5 type columns:", "", "| Model | S1, never wrong | No count row |", "|---|---|---|"]
    L += ["| %s | %d | %d |" % (SHORT[m], nw[m], ncr[m]) for m in MODELS]
    L += ["| all | %d | %d |" % (sum(nw.values()), sum(ncr.values())), "",
          "- S1, never wrong: counted in the regimes, left out of T5's type columns and of F5.",
          "- No count row (%s: %s): a slot whose own or wrong year has no count row never reached the rules. Its own "
          "line, out of F5, not the residual bucket." % (
              sum(ncr_reg.values()), ", ".join("%s %d" % (_reg(g), ncr_reg[g]) for g in cu.REGIMES if ncr_reg[g])),
          "- Residual bucket (a slot with count rows that no rule typed): %d. Every slot with its count rows gets a type, a merged type or the "
          "untyped column, so the bucket is empty by construction for year slots." % sum(r["residual"] for r in lab),
          ""]

    # (b) F5
    L += ["## (b) F5 range and open band", "",
          "Over the %s S1 to S3 slots with a count row and at least one wrong sample (never-wrong S1 slots and slots "
          "with no count row left out). The open slots are a band of their own, not a cell: their cell depends on an "
          "`unknown` count." % cli._int(f5["n"]), ""] + f5_table(f5) + [""]

    # (b2) F5 by model with confidence (F5's input)
    L += ["## (b2) F5 by model, with confidence", "",
          "F5's input (`f5_by_model.csv`): each model's points over its own F5 slots, and how many of the slots "
          "fixed at each point are High, Medium and Low (rule v3). Training data and process together counts the "
          "slots fixed at either; the open band has no confidence. Rows with no slot at all are left out here and "
          "kept in the file.", "",
          "| Model | Point | Slots | Share | High | Medium | Low |", "|---|---|---|---|---|---|---|"]
    for r in run["f5_by_model"]:
        if r["upper"] == 0:
            continue
        share = cli._pct(float(r["share_lower"])) if r["share_lower"] == r["share_upper"] else "%s to %s" % (
            cli._pct(float(r["share_lower"])), cli._pct(float(r["share_upper"])))
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            SHORT.get(r["model"], r["model"]), r["point"], _rng(r["lower"], r["upper"]), share,
            r["high"], r["medium"], r["low"]))
    L.append("")

    # (c) confidence
    filled = [r for r in lab if r["confidence"]]
    L += ["## (c) Confidence (rule v3)", "", "```", c2.CONFIDENCE_RULE, "```", "",
          "Slots with a point: %s (= F5's %s less the %s open). Blank: %s (`single_mention`, the open band, no count "
          "row)." % (cli._int(len(filled)), cli._int(f5["n"]), cli._int(f5["open"]), cli._int(len(lab) - len(filled))),
          "", "| Regime | High | Medium | Low | Blank |", "|---|---|---|---|---|"]
    for g in cu.REGIMES:
        c = Counter(r["confidence"] for r in lab if r["regime"] == g)
        L.append("| %s | %d | %d | %d | %d |" % (_reg(g), c[c2.HIGH], c[c2.MEDIUM], c[c2.LOW], c[""]))
    c = Counter(r["confidence"] for r in lab)
    L += ["| all | %d | %d | %d | %d |" % (c[c2.HIGH], c[c2.MEDIUM], c[c2.LOW], c[""]), "",
          "| Model | High | Medium | Low |", "|---|---|---|---|"]
    for m in MODELS:
        c = Counter(r["confidence"] for r in filled if r["model"] == m)
        L.append("| %s | %d | %d | %d |" % (SHORT[m], c[c2.HIGH], c[c2.MEDIUM], c[c2.LOW]))
    rc, sole = defaultdict(Counter), Counter()
    for r in filled:
        for x in r["confidence_reasons"]:
            rc[x][r["model"]] += 1
        meds = [x for x in r["confidence_reasons"] if x in c2.MEDIUM_REASONS]
        if r["confidence"] == c2.MEDIUM and len(meds) == 1:
            sole[meds[0]] += 1
    L += ["", "Every reason that fires, on slots with a point (a slot can have several; the worst wins):", "",
          "| Reason | Level | Slots | %s | Sole reason of a Medium slot |" % " | ".join(SHORT[m] for m in MODELS),
          "|---|---|---|%s---|" % ("---|" * len(MODELS))]
    for x in c2.REASONS:
        lvl = "Low" if x in c2.LOW_REASONS else "Medium"
        L.append("| %s | %s | %d | %s | %s |" % (x, lvl, sum(rc[x].values()), " | ".join(str(rc[x][m]) for m in MODELS),
                                                 sole[x] if x in c2.MEDIUM_REASONS else "n/a"))
    with_pt = {cu.slot_key(s): s for s in cls if s["f5_point"] not in ("", cu.OPEN) and not c2.never_wrong(s)}
    k1 = {g: (sum(1 for s in with_pt.values() if s["regime"] == g and c2.dominant_samples(s) == 1),
              sum(1 for s in with_pt.values() if s["regime"] == g)) for g in ("S1", "S3")}
    s1_nc = sum(1 for r in filled if r["regime"] == "S1" and c2.MED_NO_COUNT in r["confidence_reasons"])
    fab_typed = sorted(cu.slot_key(s) for s in cls if s["regime"] == "S1" and s["class"] == "typed"
                       and s["type"] == cu.FABRICATION)
    fab_flag = sum(1 for r in lab if r["flag"] == c2.FLAG_FABRICATION_S1)
    rho_n = sum(1 for r in filled if c2.MED_RHO in r["confidence_reasons"])
    s2_both = sum(1 for r in lab if r["regime"] == "S2" and r["class"] == "typed"
                  and r["phase2_type"] in (cu.BOTH_HIGH, cu.BOTH_LOW))
    rho_slot = next((s for s in cls if cu.slot_key(s) == RHO_SLOT), None)
    dg = Counter(r["coverage_downgrade"] for r in lab)
    L += ["",
          "- The rule text's k = 1 figures are the stand-in's. In this run the dominant wrong year is stated in one "
          "sample on %d of %d S1 and %d of %d S3 slots with a point." % (k1["S1"] + k1["S3"]),
          "- On S1, `%s` (%d slots) means not phrase-checked, nothing found: the phrase check runs on S2 and S3 "
          "slots only (scope `s2s3`), so an S1 slot's `unknown` counts were never looked up." % (c2.MED_NO_COUNT, s1_nc),
          "- `%s` (FABRICATION with S1, at inference and flagged): %d labelled slots. The size of this cell measures "
          "how much the absence evidence misses. "
          "Typed FABRICATION S1 slots in this run: %d (%s), %s." % (
              c2.FLAG_FABRICATION_S1, fab_flag, len(fab_typed), "; ".join(_key(k) for k in fab_typed) or "none",
              "all never wrong, so none is labelled" if fab_typed and all(
                  c2.never_wrong(s) for s in cls if cu.slot_key(s) in set(fab_typed)) else "see labels.csv"),
          "- `%s`: %d slots (S2 slots typed BOTH_IN_DATA: %d). The slot that carried it under the earlier sample rule, "
          "%s, is %s here (a sample stating the right year is right moves it from S2 to S1)." % (
              c2.MED_RHO, rho_n, s2_both, _key(RHO_SLOT), rho_slot["regime"] if rho_slot else "absent"),
          "- Coverage downgrade, recorded and not applied (the two absence points are already Low; a gap in the "
          "spec): %s." % ", ".join("%s on %d slots" % (k, v) for k, v in sorted(dg.items())), ""]

    # (d) sweep
    sw = run["sweep"]
    L += ["## (d) The tau sweep (historical_events, 5% to 20%)", "",
          "| Percentile | %s | T5 column changes (of %s) | F5 point changes (of %s) | Confidence changes (of %s) "
          "| Type changes, all slots (of %s) |" % (
              " | ".join(SHORT[m] for m in MODELS), cli._int(len(lab)), cli._int(f5["n"]), cli._int(len(lab)),
              cli._int(len(cls))),
          "|---|%s---|---|---|---|" % ("---|" * len(MODELS))]
    for p in SWEEP:
        d = sw["deltas"][p]
        L.append("| %d%%%s | %s | %d | %d | %d | %d |" % (
            p, " (headline)" if p == HEADLINE_P else "",
            " | ".join(cli._num(run["tau"][p][cu.MODEL_INDEX[m]]) for m in MODELS), d["t5"], d["f5"], d["conf"],
            d["t5_all"]))
    rev = sw["reversals"]
    bad = {q: v for q, v in rev.items() if v}
    L += ["", "Changes are against the headline (10%%): T5 and confidence on the labelled slots, the F5 point on F5's "
          "slots; the last column counts every slot's type, the never-wrong slots by the type `classify` gives them "
          "before they are left unlabelled. The ordering of models: for each quantity, each model's share (a T5 column's share of the model's "
          "labelled slots with a count row; an F5 point's lower and upper share of the model's F5 slots), and a pair "
          "of models reverses if one is above the other at some percentile and below it at another (ties count as "
          "neither). Quantities: %d. With a reversal: %d." % (len(rev), len(bad)), ""]
    f5_bad = [q for q in bad if q.startswith("F5 ")]
    t5_bad = [q[len("T5 "):] for q in bad if q.startswith("T5 ")]
    L += ["%s; %s." % (
        "Every F5 share keeps the order of models across the whole sweep" if not f5_bad else
        "F5 shares whose order of models changes: %s" % ", ".join(f5_bad),
        "every T5 column too" if not t5_bad else
        "the order changes for %d T5 column(s), each tied to tau through n_c against it: %s" % (
            len(t5_bad), ", ".join(t5_bad))), ""]
    if bad:
        L += ["| Quantity | Models | First above at | Second above at |", "|---|---|---|---|"]
        for q, v in bad.items():
            for a, b, up, down in v:
                L.append("| %s | %s, %s | %s | %s |" % (q, SHORT[a], SHORT[b], ", ".join("%d%%" % p for p in up),
                                                       ", ".join("%d%%" % p for p in down)))
        L.append("")
    else:
        L += ["The ordering of models holds across the whole sweep for every quantity.", ""]

    # (e) sensitivity
    se = run["sensitivity"]
    L += ["## (e) Sensitivity: Biographies' own tau", "",
          "Reported beside the headline, never in its place. Biographies' tau (results/bio_counts/merged/tau.csv, 10th "
          "percentile): %s. Headline: %s." % (_tau_line(run["bio_tau"]), _tau_line(run["tau"][HEADLINE_P])), "",
          "- T5 column changes: %d of %s labelled slots (%d of all %s slots, counting the never-wrong slots by their "
          "pre-label type)." % (
              len(se["t5"]), cli._int(len(lab)), len(se["t5_all"]), cli._int(len(cls))),
          "- F5 point changes: %d of %s F5 slots." % (len(se["f5"]), cli._int(f5["n"])),
          "- Confidence changes: %d of %s labelled slots." % (len(se["conf"]), cli._int(len(lab))), ""]
    if se["t5_all"] or se["f5"]:
        L += ["A never-wrong slot has no label: its two type columns show the type `classify` gives it before it is "
              "left out (pre-label), and it has no F5 point.", "",
              "| Slot | Regime | Never wrong | T5 column, headline | T5 column, Biographies' tau | F5 point, headline | "
              "F5 point, Biographies' tau |", "|---|---|---|---|---|---|---|"]
        hk = {cu.slot_key(s): s for s in cls}
        bk = {cu.slot_key(s): s for s in run["bio_classified"]}
        for k in sorted({k for k, _, _ in se["t5_all"]} | {k for k, _, _ in se["f5"]}):
            h, b = hk[k], bk[k]
            if c2.never_wrong(h):
                L.append("| %s | %s | yes | %s (pre-label) | %s (pre-label) | none (never wrong) | none (never wrong) |"
                         % (_key(k), _reg(h["regime"]), h["t5_column"] or NO_COUNT_ROW, b["t5_column"] or NO_COUNT_ROW))
                continue
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (_key(k), _reg(h["regime"]), "no",
                                                              h["t5_column"] or NO_COUNT_ROW,
                                                         b["t5_column"] or NO_COUNT_ROW, h["f5_point"] or "none",
                                                         b["f5_point"] or "none"))
        L.append("")

    # (f) no LLM step, (g) provenance
    p = run["provenance"]
    L += ["## (f) No LLM step for year slots", "",
          "No LLM step is needed for Biographies year slots: the year is the claim (`cluster_method` "
          "`year` in `stability.slot_items`), extracted by a fixed rule (`hallutrace/bio_years.py`) and judged by an "
          "exact Wikidata match; Group B and its NER step are not run; the stance judge serves MYTH "
          "only, which needs a misconception set for years and has none.", "",
          "## (f2) Not labelled here", "",
          "- The other seven tasks (historical_events and the six v2 tasks): item-level regimes only. They "
          "have no Phase 2 type, so no intervention point, no label and no T5 or F5 row (only Biographies "
          "years are labelled). historical_events enters only through tau.", "",
          "## (g) Provenance", "", "- git head: `%s`, script version %s, rule `%s`. Code files changed against the "
          "head at run time: %s." % (p["git_head"], SCRIPT_VERSION, c2.RULE_VERSION,
                                     "unknown (git could not say)" if p["changed_run_files"] is None else
                                     (", ".join("`%s`" % f for f in p["changed_run_files"]) or "none"))]
    for name, item in p["inputs"].items():
        L.append("- %s: `%s`, git blob `%s`." % (name, item["path"], item["blob"]))
    b = p["verifier"]
    L += ["- verifier output copy: `%s` (`%s`), %s samples, skipped units: %s; claims kept %s of %s (%s dropped)." % (
              b["root"], b["outputs"], cli._int(b["samples"]), ", ".join(b["skipped_units"]) or "none",
              cli._int(b["claims_kept"]), cli._int(b["claims_read"]), cli._int(b["claims_dropped"])),
          "- phrase check: cache `%s` (read-only), %d targets, %d with a hit, %d phrases walked, 0 requests." % (
              p["cache_path"], p["walk"]["targets"], p["walk"]["hits"], p["walk"]["phrases"]),
          "- run start %s, %s s." % (p["run_start"], p["run_seconds"]),
          "- decisions: %s." % "; ".join(DECISIONS), ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(args):
    t0 = time.time()
    run_start = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cli.check_cache_name(args.cache)
    cli.check_not_window_cache(args.cache)
    if not Path(args.cache).exists():
        raise pcli.CliError("no phrase cache at %s" % args.cache)
    out = Path(args.out)
    present_out = [n for n in OUTPUTS if (out / n).exists()]
    if present_out and not args.force:
        raise pcli.CliError("%s already has %s; pass --force to overwrite" % (out, ", ".join(present_out)))
    he, bio = read_taus(args)
    inputs = cli.load_inputs(args)   # claims kept by ok_claims, counts, the headline tau (read_tau's refusals)
    prep = cli.prepare(inputs, "s2s3")
    check_targets(prep, inputs, [("percentile %d" % p, he[p]) for p in SWEEP] + [("Biographies' tau", bio)])
    present, walk = presence(inputs, prep, args.cache)
    by_pct = {p: cli.classify_all(prep["slots"], inputs["cidx"], he[p], present) for p in SWEEP}
    bio_cls = cli.classify_all(prep["slots"], inputs["cidx"], bio, present)
    sweep = c2.sweep_sets(by_pct)
    labels_by_pct = {p: labels_for(by_pct[p], he[p], sweep) for p in SWEEP}
    bio_labels = labels_for(bio_cls, bio, sweep)
    head, lab = by_pct[HEADLINE_P], labels_by_pct[HEADLINE_P]

    def conf_changes(other):
        ok = {(r["model"], r["prompt_id"], r["kind"]): r["confidence"] for r in other}
        return [((r["model"], r["prompt_id"], r["kind"]), r["confidence"], ok[(r["model"], r["prompt_id"], r["kind"])])
                for r in lab if r["confidence"] != ok[(r["model"], r["prompt_id"], r["kind"])]]

    labelled = {cu.slot_key(s) for s in head if not c2.never_wrong(s)}
    pool = {cu.slot_key(s) for s in head if s["regime"] in ("S1", "S2", "S3") and s["class"] != "no_count_row"
            and not c2.never_wrong(s)}   # intervention_points.f5_range's slots

    def delta(other):
        t5 = changes(head, other, "t5_column")
        return {"t5": [x for x in t5 if x[0] in labelled], "t5_all": t5,
                "f5": [x for x in changes(head, other, "f5_point") if x[0] in pool]}

    deltas = {}
    for p in SWEEP:
        d = delta(by_pct[p])
        deltas[p] = {"t5": len(d["t5"]), "t5_all": len(d["t5_all"]), "f5": len(d["f5"]),
                     "conf": len(conf_changes(labels_by_pct[p]))}
    sens = dict(delta(bio_cls), conf=conf_changes(bio_labels))
    shares = {p: shares_by_model(by_pct[p], labels_by_pct[p]) for p in SWEEP}
    rev = reversals(shares)
    provenance = {
        "git_head": git_head(), "changed_run_files": changed_run_files(),
        "inputs": {name: {"path": cli._shown_path(pth), "blob": cli.blob_hash(pth)}
                   for name, pth in (("claims", args.claims), ("counts", args.counts), ("tau", args.tau),
                                     ("Biographies tau", args.bio_tau))},
        "verifier": inputs["verifier"], "cache_path": str(Path(args.cache).resolve()), "walk": walk,
        "run_start": run_start, "run_seconds": round(time.time() - t0, 1), "decisions": DECISIONS,
    }
    f5 = c2.f5_range(head)
    nw = c2.never_wrong_counts(head)
    f5m = f5_model_rows(head, lab)
    runinfo = {"labels": lab, "classified": head, "f5": f5, "never_wrong": nw, "tau": he, "bio_tau": bio,
               "f5_by_model": f5m,
               "bio_classified": bio_cls, "sweep": {"deltas": deltas, "reversals": rev},
               "sensitivity": sens, "provenance": provenance}
    summary = summary_markdown(runinfo)
    if chr(0x2014) in summary or chr(0x2013) in summary:
        raise pcli.CliError("an em or en dash in the summary")
    meta = {
        "script_version": SCRIPT_VERSION, "rule_version": c2.RULE_VERSION, "confidence_rule": c2.CONFIDENCE_RULE,
        "provenance": provenance,
        "tau": {"headline_percentile": HEADLINE_P,
                "historical_events": {str(p): he[p] for p in SWEEP}, "biographies": bio},
        "slots": len(head), "labelled": len(lab), "never_wrong": nw,
        "f5": {"n": f5["n"], "open": f5["open"], "rows": cli.f5_rows(f5), "open_groups": f5["open_groups"]},
        "sweep": {str(p): deltas[p] for p in SWEEP},
        "sweep_columns": SWEEP_COLUMNS,
        "sweep_dropped_columns": [c for c in c2.LABEL_COLUMNS if c not in SWEEP_COLUMNS],
        "ordering_reversals": {q: [[a, b, up, down] for a, b, up, down in v] for q, v in rev.items() if v},
        "sensitivity": {k: [[list(x[0]), x[1], x[2]] for x in v] for k, v in sens.items()},
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cols = list(c2.LABEL_COLUMNS)
    cli.write_outputs(out, [
        (LABELS_NAME, (cols, label_rows(lab))),
        (SWEEP_NAME, (SWEEP_COLUMNS, [row for p in SWEEP for row in label_rows(labels_by_pct[p], percentile=p)])),
        (BIO_NAME, (cols, label_rows(bio_labels))),
        (T5_NAME, (T5_COLUMNS, t5_rows(head, lab))),
        (F5_MODEL_NAME, (F5_MODEL_COLUMNS, f5m)),
        (SUMMARY_NAME, summary),
        (META_NAME, json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n"),
    ])
    pcli.say("%d slots, %d labelled, %d never wrong; F5 over %d (%d open); sweep 5%% to 20%%: %d quantity(ies) "
             "with a reversal; Biographies' tau: %d T5 and %d F5 changes -> %s" % (
                 len(head), len(lab), sum(nw.values()), f5["n"], f5["open"], sum(1 for v in rev.values() if v),
                 len(sens["t5"]), len(sens["f5"]), out))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (pcli.CliError, sqlite3.Error, OSError, ValueError, KeyError) as e:
        pcli.say("error: %s" % e, err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
