"""Truncation and repetition in generated responses, per model or task.

A response cut by the token limit ends in a half-written clause, which the
decomposer turns into a unit that is unlikely to be entailed, inflating h. A
base model can also fall into a repetition loop that holds no claims.

Metrics.
    hit_cap      share of responses that stopped because the budget ran out
    cut          share that hit the cap AND whose text ends without terminal
                 punctuation or a line break, i.e. what bexec strips
    sentences    mean complete sentences per response
    tail_share   mean of 1 / (sentences + 1) over cut responses, the share of a
                 response's claims contributed by the cut clause
    h_inflate    cut * tail_share, expected inflation of h from truncation alone
    emptied      share whose text `strip_incomplete_tail` would remove entirely,
                 because no complete sentence precedes the cut
    repeat       share of responses where one sentence occurs three or more times
    distinct     mean distinct sentences divided by total sentences, so 1.00 is
                 no repetition and 0.10 is a loop

One sentence is assumed to be one atomic unit, so read `h_inflate` as an order
of magnitude. What counts as a cut is decided by hallutrace.outcomes, the code
bexec runs.

Usage:
    python scripts/generation_quality.py results/pilot
    python scripts/generation_quality.py results/pilot --out results/generation_quality.csv
"""

import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hallutrace.outcomes import ends_mid_clause, strip_incomplete_tail  # noqa: E402

SENTENCE_END = re.compile(r"[.!?]['\")\]]?(?:\s|$)")
SENTENCE_SPLIT = re.compile(r"[^.!?\n]+[.!?\n]?")

FIELDS = ["model", "n", "empty", "hit_cap", "mean_out_tokens", "cut",
          "sentences", "tail_share", "h_inflate", "emptied", "repeat", "distinct"]


def sentences_of(text):
    return [s.strip() for s in SENTENCE_SPLIT.findall(text) if len(s.strip()) > 3]


def task_of(rec):
    """The task name is the prompt_id stem, e.g. biographies-00042."""
    return rec["prompt_id"].rsplit("-", 1)[0]


def load(root, group="model"):
    """group is model, task, or both."""
    rows = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(root, "*", "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if group == "model":
                key = rec["model"]
            elif group == "task":
                key = task_of(rec)
            else:
                key = "%s / %s" % (rec["model"], task_of(rec))
            rows[key].append(rec)
    return rows


def analyse(records):
    n = len(records)
    empty = hit_cap = repeat = emptied = 0
    cut, kept, distinct, out_tokens = [], [], [], []

    for rec in records:
        raw = rec.get("response") or ""
        text = raw.strip()
        out_tokens.append(rec.get("n_output_tokens", 0))
        if rec.get("finish_reason") == "length":
            hit_cap += 1
        if not text:
            empty += 1
            distinct.append(0.0)
            continue

        sents = sentences_of(text)
        if sents:
            counts = collections.Counter(s.lower() for s in sents)
            distinct.append(len(counts) / len(sents))
            if counts.most_common(1)[0][1] >= 3:
                repeat += 1
        else:
            distinct.append(0.0)

        complete = len(SENTENCE_END.findall(text))
        kept.append(complete)
        # the raw text, since a trailing line break means the last line is
        # finished; only a response that hit the cap counts as cut
        if rec.get("finish_reason") == "length" and ends_mid_clause(raw):
            cut.append(1.0 / (complete + 1))
            if not strip_incomplete_tail(raw)[0].strip():
                emptied += 1

    scored = n - empty
    cut_rate = len(cut) / scored if scored else 0.0
    tail_share = statistics.fmean(cut) if cut else 0.0
    return {
        "n": n,
        "empty": empty,
        "hit_cap": hit_cap / n if n else 0.0,
        "mean_out_tokens": round(statistics.fmean(out_tokens), 1) if out_tokens else 0.0,
        "cut": cut_rate,
        "sentences": round(statistics.fmean(kept), 1) if kept else 0.0,
        "tail_share": tail_share,
        "h_inflate": cut_rate * tail_share,
        "emptied": emptied / scored if scored else 0.0,
        "repeat": repeat / n if n else 0.0,
        "distinct": round(statistics.fmean(distinct), 2) if distinct else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory holding <model>/<batch>.jsonl")
    ap.add_argument("--out", help="write the table as CSV here")
    ap.add_argument("--group", choices=("model", "task", "both"), default="model",
                    help="one row per model, per task, or per model and task")
    args = ap.parse_args()

    rows = load(args.root, args.group)
    if not rows:
        raise SystemExit("no jsonl found under %s" % args.root)

    table = []
    for model in sorted(rows):
        r = analyse(rows[model])
        r["model"] = model
        table.append(r)

    width = max(14, max(len(r["model"]) for r in table) + 1)
    print("%-*s %5s %6s %8s %8s %6s %10s %11s %10s %8s %8s %9s"
          % (width, "group", "n", "empty", "hit_cap", "out_tok", "cut",
             "sentences", "tail_share", "h_inflate", "emptied", "repeat", "distinct"))
    for r in table:
        print("%-*s %5d %6d %7.0f%% %8.1f %5.0f%% %10.1f %10.1f%% %9.1f%% %7.1f%% %7.0f%% %9.2f"
              % (width, r["model"], r["n"], r["empty"], 100 * r["hit_cap"],
                 r["mean_out_tokens"], 100 * r["cut"], r["sentences"],
                 100 * r["tail_share"], 100 * r["h_inflate"], 100 * r["emptied"],
                 100 * r["repeat"], r["distinct"]))

    infl = [r["h_inflate"] for r in table]
    lo = min(table, key=lambda r: r["h_inflate"])
    hi = max(table, key=lambda r: r["h_inflate"])
    print()
    print("h inflation from truncation alone: %.1f%% (%s) to %.1f%% (%s), a "
          "spread of %.1f points across %s."
          % (100 * lo["h_inflate"], lo["model"], 100 * hi["h_inflate"],
             hi["model"], 100 * (max(infl) - min(infl)), args.group))
    if args.group == "model":
        print("That spread does not cancel when models are compared.")

    worst = min(table, key=lambda r: r["distinct"])
    if worst["distinct"] < 0.5:
        print("%s repeats in %.0f%% of responses, distinct ratio %.2f. Those "
              "outputs are loops, not answers."
              % (worst["model"], 100 * worst["repeat"], worst["distinct"]))

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in table:
                w.writerow({k: r[k] for k in FIELDS})
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
