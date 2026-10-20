r"""Score the machine's atoms against a filled hand-decomposition packet.

    .\.venv\Scripts\python.exe scripts\hand_decomposition_score.py score <task> --verifier-root <root>
    .\.venv\Scripts\python.exe scripts\hand_decomposition_score.py check --root biographies=<root> --root summarization=<root> ...

`score` reads annotation\hand_decomposition\<task>\{key.csv, annotator.csv} and prints precision, recall
and F1, overall and per model, for the `filtered` and `raw` variants, under three readings:
the containment match at 0.8 (the headline), the same at 0.6 (sensitivity) and the
one-to-one match (`score`). <root> holds local copies of the verification `inputs/` and, for
Biographies, `outputs_filtered/` (as for scripts\bio_year_claims.py), or for Summarization
and Simplification the v2 run's `outputs/`.

`check` draws the stress test for the threshold: pairs the containment matcher
scored, a machine atom with its best human atom, across the tasks given, half at or above
the threshold and half below, seeded. These are the matches behind precision; the human
atoms' best machine atoms, behind recall, are not drawn, so the check covers precision's
side of the matcher only. The packet shows only the two atoms, so the person
judging does not see the matcher's call; the score and the call go to `<out>_key.csv`.
The `you_say` column is left blank for a person to fill (`inside` or `not inside`);
nothing here fills it.

    .\.venv\Scripts\python.exe scripts\hand_decomposition_score.py t4 --root biographies=<root> --root summarization=<root> ...

`t4` writes results\hand_decomposition\t4.csv for supplementary table T4: one row per task, variant and
reading, with micro precision, recall and F1 and their 95% intervals from a bootstrap
over items (2,000 draws, seeded).

    .\.venv\Scripts\python.exe scripts\hand_decomposition_score.py rescore --root biographies=<root> --root summarization=<root> ...

`rescore` rebuilds what the matcher computes in results\hand_decomposition\: the `scores` of
scores.json (every task, variant and reading, as `score` gives them), the `score` and
`matcher_says_0_8` columns of stress_test.csv and stress_test_set2.csv, and the agreement
counts between the matcher and each person's column at thresholds 0.4 to 0.9. The pairs, the
people's judgments and the notes in scores.json are kept as they are; nothing here fills a
person's column. It needs --root for all three tasks, and it reads and checks both stress
files (every pair's item in its packet, its machine atom among that item's filtered machine
atoms, every person's call `inside` or `not inside`) before it writes anything, so a refusal
leaves every file as it was.

`score` and `check` write to outputs\hand_decomposition\ (gitignored) unless --out says otherwise. Every command
prints the join's notes, so a Summarization or Simplification row joined out of line shows.
"""
import argparse
import csv
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hallutrace import hand_decomposition_score as es  # noqa: E402

PAIR_COLUMNS = ["item_id", "model", "kind", "machine_atom", "human_atom", "score"]
CHECK_COLUMNS = ["check_id", "task", "machine_atom", "human_atom", "you_say", "note"]
CHECK_KEY_COLUMNS = ["check_id", "task", "item_id", "model", "score", "matcher_says"]
READINGS = (("containment_0.8", "containment", es.P0_10_1_THRESHOLD),
            ("containment_0.6", "containment", es.SENSITIVITY_THRESHOLD),
            ("one_to_one_0.6", "one_to_one", es.DEFAULT_THRESHOLD))


def _fmt(d):
    return "P %s  R %s  F1 %s" % tuple("n/a" if d[k] is None else "%.3f" % d[k]
                                       for k in ("precision", "recall", "f1"))


def pair_rows(items, result):
    """One-to-one: matched pairs and the unmatched atoms on each side."""
    rows = []
    for it, r in zip(items, result["items"]):
        got_m = {i for i, _, _ in r["pairs"]}
        got_h = {j for _, j, _ in r["pairs"]}
        for i, j, s in r["pairs"]:
            rows.append({"item_id": it["item_id"], "model": it["model"], "kind": "matched",
                         "machine_atom": it["machine"][i], "human_atom": it["human"][j], "score": s})
        rows += [{"item_id": it["item_id"], "model": it["model"], "kind": "machine_only",
                  "machine_atom": a, "human_atom": "", "score": ""}
                 for i, a in enumerate(it["machine"]) if i not in got_m]
        rows += [{"item_id": it["item_id"], "model": it["model"], "kind": "human_only",
                  "machine_atom": "", "human_atom": a, "score": ""}
                 for j, a in enumerate(it["human"]) if j not in got_h]
    return rows


def containment_rows(items, result):
    """Containment: each machine atom with its best human atom, and each human atom with
    its best machine atom, the score and whether it clears the threshold."""
    t = result["threshold"]
    rows = []
    for it, r in zip(items, result["items"]):
        for i, (j, s) in enumerate(r["machine_best"]):
            rows.append({"item_id": it["item_id"], "model": it["model"],
                         "kind": "machine_matched" if s >= t else "machine_unmatched",
                         "machine_atom": it["machine"][i],
                         "human_atom": it["human"][j] if j is not None else "", "score": s})
        for j, (i, s) in enumerate(r["human_best"]):
            rows.append({"item_id": it["item_id"], "model": it["model"],
                         "kind": "human_matched" if s >= t else "human_unmatched",
                         "machine_atom": it["machine"][i] if i is not None else "",
                         "human_atom": it["human"][j], "score": s})
    return rows


def _load(task, packet, root, variant):
    return es.load_items(task, os.path.join(packet, "key.csv"),
                         os.path.join(packet, "annotator.csv"), root, variant)


def _print_notes(task, variant, notes):
    for n in notes:
        print("  %s, %s, note: %s" % (task, variant, n))


def cmd_score(args):
    packet = args.packet or os.path.join(ROOT, "annotation", "hand_decomposition", args.task)
    out = args.out or os.path.join(ROOT, "outputs", "hand_decomposition", args.task)
    os.makedirs(out, exist_ok=True)
    summary = {}
    for variant in es.VARIANTS:
        try:
            items, notes = _load(args.task, packet, args.verifier_root, variant)
        except es.JoinError as e:
            print("join failed: %s" % e, file=sys.stderr)
            return 1
        if not any(it["human"] for it in items):
            print("no human atoms in %s: is the packet filled in?" % packet, file=sys.stderr)
            return 1
        summary[variant] = {"join_notes": notes}
        for name, kind, t in READINGS:
            if kind == "containment":
                res = es.score_containment(items, threshold=t)
                rows = containment_rows(items, res)
            else:
                res = es.score(items, threshold=t, matcher_name="one-to-one token overlap")
                rows = pair_rows(items, res)
            summary[variant][name] = {k: v for k, v in res.items() if k != "items"}
            with open(os.path.join(out, "pairs_%s_%s.csv" % (variant, name)), "w",
                      encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PAIR_COLUMNS)
                w.writeheader()
                w.writerows(rows)
            o = res["overall"]
            print("%s, %s, %s: %d items, %d machine atoms, %d human atoms"
                  % (args.task, variant, name, o["items"], o["machine_atoms"], o["human_atoms"]))
            print("  micro  %s" % _fmt(o["micro"]))
            print("  macro  %s" % _fmt(o["macro"]))
            if name == READINGS[0][0]:
                for model, s in res["by_model"].items():
                    print("  %-13s micro %s  (%d items)" % (model, _fmt(s["micro"]), s["items"]))
        for n in notes:
            print("  note: %s" % n)
    with open(os.path.join(out, "scores.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("-> %s" % out)
    return 0


def cmd_check(args):
    roots = dict(r.split("=", 1) for r in args.root)
    bad = set(roots) - set(es.TASKS)
    if bad:
        print("unknown task in --root: %s" % ", ".join(sorted(bad)), file=sys.stderr)
        return 1
    rng = random.Random(args.seed)
    per_task = args.n // len(roots)
    picked = []
    for task in sorted(roots):
        packet = os.path.join(args.packets, task)
        try:
            items, notes = _load(task, packet, roots[task], "filtered")
        except es.JoinError as e:
            print("join failed: %s" % e, file=sys.stderr)
            return 1
        _print_notes(task, "filtered", notes)
        res = es.score_containment(items, threshold=args.threshold)
        above, below = [], []
        for it, r in zip(items, res["items"]):
            for i, (j, s) in enumerate(r["machine_best"]):
                if j is None:
                    continue
                row = {"task": task, "item_id": it["item_id"], "model": it["model"],
                       "machine_atom": it["machine"][i], "human_atom": it["human"][j], "score": s}
                (above if s >= args.threshold else below).append(row)
        half = per_task // 2
        for pool, says, n in ((above, "inside", per_task - half), (below, "not inside", half)):
            for row in rng.sample(pool, min(n, len(pool))):
                picked.append(dict(row, matcher_says=says))
    rng.shuffle(picked)
    out = args.out or os.path.join(ROOT, "outputs", "hand_decomposition", "stress_test.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    key_out = os.path.splitext(out)[0] + "_key.csv"
    for path, cols in ((out, CHECK_COLUMNS), (key_out, CHECK_KEY_COLUMNS)):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for n, row in enumerate(picked, 1):
                w.writerow(dict(row, check_id=n, you_say="", note=""))
    print("%d pairs (seed %d, threshold %.2f) -> %s, the matcher's calls in %s"
          % (len(picked), args.seed, args.threshold, out, key_out))
    return 0


T4_COLUMNS = ["task", "variant", "reading", "threshold", "items", "machine_atoms", "human_atoms",
              "precision", "precision_lo", "precision_hi", "recall", "recall_lo", "recall_hi",
              "f1", "f1_lo", "f1_hi"]


def t4_rows(task, variant, items):
    """One T4 row per reading: micro P/R/F1 with 95% bootstrap intervals over items."""
    rows = []
    for name, kind, t in READINGS:
        if kind == "containment":
            res = es.score_containment(items, threshold=t)
        else:
            res = es.score(items, threshold=t, matcher_name="one-to-one token overlap")
        o, ci = res["overall"], es.bootstrap_micro(res["items"])
        row = {"task": task, "variant": variant, "reading": name, "threshold": t,
               "items": o["items"], "machine_atoms": o["machine_atoms"],
               "human_atoms": o["human_atoms"]}
        for key in ("precision", "recall", "f1"):
            row[key] = o["micro"][key]
            row[key + "_lo"], row[key + "_hi"] = ci[key] or ("", "")
        rows.append(row)
    return rows


def cmd_t4(args):
    roots = dict(r.split("=", 1) for r in args.root)
    bad = set(roots) - set(es.TASKS)
    if bad:
        print("unknown task in --root: %s" % ", ".join(sorted(bad)), file=sys.stderr)
        return 1
    rows = []
    for task in [t for t in es.TASKS if t in roots]:
        for variant in es.VARIANTS:
            try:
                items, notes = _load(task, os.path.join(args.packets, task), roots[task], variant)
            except es.JoinError as e:
                print("join failed: %s" % e, file=sys.stderr)
                return 1
            if not any(it["human"] for it in items):
                print("no human atoms in %s's packet: is it filled in?" % task, file=sys.stderr)
                return 1
            _print_notes(task, variant, notes)
            rows += t4_rows(task, variant, items)
    out = args.out or os.path.join(ROOT, "results", "hand_decomposition", "t4.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=T4_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["variant"] == "filtered":
            print("%-14s %-16s %s" % (r["task"], r["reading"], "  ".join(
                "%s %s [%s, %s]" % (label, "n/a" if r[k] is None else "%.3f" % r[k],
                                    r[k + "_lo"], r[k + "_hi"])
                for label, k in (("P", "precision"), ("R", "recall"), ("F1", "f1")))))
    print("-> %s (%d rows; %d bootstrap draws over items, seed %d)"
          % (out, len(rows), es.N_BOOT, es.BOOT_SEED))
    return 0


# results\hand_decomposition\'s stress-test files and the people's columns in each
STRESS_SETS = (("stress_test.csv", ("rater1_pass1", "rater1_pass2")), ("stress_test_set2.csv", ("rater2",)))
AGREEMENT_THRESHOLDS = ("0.4", "0.5", "0.6", "0.7", "0.8", "0.9")


def says_inside(row, col):
    """A person's call on a stress pair: True for `inside`, False for `not inside`.

    Anything else raises, so a blank or misspelt call is never counted as `not inside`.
    """
    call = row[col].strip()
    if call not in ("inside", "not inside"):
        raise ValueError("pair %s: %s says %r, not 'inside' or 'not inside'"
                         % (row.get("check_id", "?"), col, row[col]))
    return call == "inside"


def agreement(rows, col):
    """Matcher (score >= t) against a person's `inside` call, at each threshold."""
    out = {}
    for th in AGREEMENT_THRESHOLDS:
        t = float(th)
        counts = {"agree": 0, "n": len(rows), "matcher_inside_person_not": 0, "matcher_not_person_inside": 0}
        for r in rows:
            m, p = float(r["score"]) >= t, says_inside(r, col)
            counts["agree"] += m == p
            counts["matcher_inside_person_not"] += m and not p
            counts["matcher_not_person_inside"] += p and not m
        out[th] = counts
    return out


def agreement_by_task(rows, col, t=es.P0_10_1_THRESHOLD):
    return {task: sum(1 for r in rows if r["task"] == task and
                      (float(r["score"]) >= t) == says_inside(r, col)) for task in es.TASKS}


def cmd_rescore(args):
    roots = dict(r.split("=", 1) for r in args.root)
    bad = set(roots) - set(es.TASKS)
    if bad:
        print("unknown task in --root: %s" % ", ".join(sorted(bad)), file=sys.stderr)
        return 1
    absent = [t for t in es.TASKS if t not in roots]
    if absent:
        print("rescore rewrites the scores of every task: give --root for %s" % ", ".join(absent),
              file=sys.stderr)
        return 1
    scores, topics, machine = {}, {}, {}
    for task in es.TASKS:
        scores[task] = {}
        for variant in es.VARIANTS:
            try:
                items, notes = _load(task, os.path.join(args.packets, task), roots[task], variant)
            except es.JoinError as e:
                print("join failed: %s" % e, file=sys.stderr)
                return 1
            if not any(it["human"] for it in items):
                print("no human atoms in %s's packet: is it filled in?" % task, file=sys.stderr)
                return 1
            _print_notes(task, variant, notes)
            if variant == "filtered":
                topics[task] = {it["item_id"]: it.get("topic", "") for it in items}
                machine[task] = {it["item_id"]: set(it["machine"]) for it in items}
            scores[task][variant] = {"join_notes": notes}
            for name, kind, t in READINGS:
                if kind == "containment":
                    res = es.score_containment(items, threshold=t)
                else:
                    res = es.score(items, threshold=t, matcher_name="one-to-one token overlap")
                scores[task][variant][name] = {k: v for k, v in res.items() if k != "items"}

    # Read, check and compute everything before writing anything, so a refusal leaves
    # the stress files and scores.json as they were.
    stress = {}
    for fname, cols in STRESS_SETS:
        path = os.path.join(args.results, fname)
        try:
            with open(path, encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                fields, rows = reader.fieldnames, list(reader)
        except FileNotFoundError:
            print("%s is not in %s" % (fname, args.results), file=sys.stderr)
            return 1
        no_col = [c for c in cols if c not in (fields or [])]
        if no_col:
            print("%s has no column %s" % (fname, ", ".join(no_col)), file=sys.stderr)
            return 1
        changed = []
        for r in rows:
            where = "%s, pair %s" % (fname, r["check_id"])
            if r["task"] not in topics:
                print("%s: unknown task %r" % (where, r["task"]), file=sys.stderr)
                return 1
            if r["item_id"] not in topics[r["task"]]:
                print("%s: %s is not in %s's packet" % (where, r["item_id"], r["task"]), file=sys.stderr)
                return 1
            if r["machine_atom"] not in machine[r["task"]][r["item_id"]]:
                print("%s: the machine atom is not among %s's filtered atoms under this --root"
                      % (where, r["item_id"]), file=sys.stderr)
                return 1
            s = round(es.contained(r["machine_atom"], r["human_atom"], topics[r["task"]][r["item_id"]]), 4)
            says = "inside" if s >= es.P0_10_1_THRESHOLD else "not inside"
            if says != r["matcher_says_0_8"]:
                changed.append(r["check_id"])
            r["score"], r["matcher_says_0_8"] = s, says
        stress[fname] = (path, fields, rows, changed)

    path = os.path.join(args.results, "scores.json")
    try:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
    except FileNotFoundError:
        summary = {}
    summary["scores"] = scores
    set1, set2 = stress["stress_test.csv"][2], stress["stress_test_set2.csv"][2]
    try:
        st = summary.setdefault("stress_test", {})
        st.setdefault("pass1", {})["by_threshold"] = agreement(set1, "rater1_pass1")
        st.setdefault("pass2", {})["by_threshold"] = agreement(set1, "rater1_pass2")
        st["pass2"]["by_task_at_0.8"] = agreement_by_task(set1, "rater1_pass2")
        s2 = summary.setdefault("stress_test_set2", {})
        s2["by_threshold"] = agreement(set2, "rater2")
        s2["by_task_at_0.8"] = agreement_by_task(set2, "rater2")
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    summary["stress_test_combined"] = {
        th: {k: st["pass2"]["by_threshold"][th][k] + s2["by_threshold"][th][k]
             for k in st["pass2"]["by_threshold"][th]} for th in AGREEMENT_THRESHOLDS}

    for fname, (spath, fields, rows, changed) in stress.items():
        with open(spath, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print("%s: %d pairs rescored, %d calls at %.1f changed%s"
              % (fname, len(rows), len(changed), es.P0_10_1_THRESHOLD,
                 " (%s)" % ", ".join(changed) if changed else ""))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    c = summary["stress_test_combined"]["0.8"]
    print("stress tests at 0.8, both sets: %d of %d agree; false matches %d, misses %d"
          % (c["agree"], c["n"], c["matcher_inside_person_not"], c["matcher_not_person_inside"]))
    print("-> %s" % args.results)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="P/R/F1 for one task")
    s.add_argument("task", choices=es.TASKS)
    s.add_argument("--verifier-root", required=True, help="local copies of the verification inputs/ and outputs")
    s.add_argument("--packet", default=None, help="default annotation/hand_decomposition/<task>")
    s.add_argument("--out", default=None, help="default outputs/hand_decomposition/<task>")
    c = sub.add_parser("check", help="draw the threshold stress test")
    c.add_argument("--root", action="append", required=True, metavar="TASK=ROOT")
    c.add_argument("--packets", default=os.path.join(ROOT, "annotation", "hand_decomposition"),
                   help="folder holding one packet folder per task")
    c.add_argument("--n", type=int, default=30)
    c.add_argument("--seed", type=int, default=20261008)
    c.add_argument("--threshold", type=float, default=es.P0_10_1_THRESHOLD)
    c.add_argument("--out", default=None, help="default outputs/hand_decomposition/stress_test.csv")
    t = sub.add_parser("t4", help="T4: micro P/R/F1 with 95%% intervals, every task, variant and reading")
    t.add_argument("--root", action="append", required=True, metavar="TASK=ROOT")
    t.add_argument("--packets", default=os.path.join(ROOT, "annotation", "hand_decomposition"),
                   help="folder holding one packet folder per task")
    t.add_argument("--out", default=None, help="default results/hand_decomposition/t4.csv")
    r = sub.add_parser("rescore", help="rebuild the matcher's numbers in results/hand_decomposition/")
    r.add_argument("--root", action="append", required=True, metavar="TASK=ROOT")
    r.add_argument("--packets", default=os.path.join(ROOT, "annotation", "hand_decomposition"),
                   help="folder holding one packet folder per task")
    r.add_argument("--results", default=os.path.join(ROOT, "results", "hand_decomposition"),
                   help="folder holding scores.json and the stress-test files")
    args = ap.parse_args(argv)
    return {"score": cmd_score, "check": cmd_check, "t4": cmd_t4, "rescore": cmd_rescore}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
