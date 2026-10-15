"""Pilot: check the n_c counting rule chosen from the hand-labelled packet.

Reads, never writes, the labelled packet (outputs/c_pilot_packet.csv), the strata counts
(results/c_pilot/pilot_packet_strata.csv) and the window cache (read-only, through
scripts/pilot_counts.py's loader), and writes results/c_pilot/pilot_rule_check.md:
numbers only, no corpus text. No network.

    .venv\\Scripts\\python.exe scripts\\pilot_rule_check.py

Two checks, both with the short forms of disambiguated names (source `stripped`) left out:
1. Precision of four rules on the packet, weighted by each stratum's share of the candidate
   population, with sensitivity variants (identical snippets once, without the name that
   supplies 15 of 60 rows, two disputed labels flipped) and 95% Wilson intervals on the raw
   counts. A stratum with fewer than 3 scored rows takes the pooled precision of its cue value.
   "Kept" = estimated correct matches the rule keeps, out of those the bare year at any
   distance finds; it is relative to the stored windows, not a true recall.
2. What each rule does to n_c on the 20 pilot names: per index, facts at 0 and the
   nearest-rank 10th percentile over n_c > 0 (tau's population), docs-unit facts only.
   Unlike the packet strata (one nearest occurrence per window), a window matches here if
   ANY occurrence of the year passes the rule.
"""
import csv
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace import pilot_counts as pc  # noqa: E402

_spec = importlib.util.spec_from_file_location("pilot_counts_cli", ROOT / "scripts" / "pilot_counts.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

PACKET = ROOT / "outputs" / "c_pilot_packet.csv"
STRATA = ROOT / "results" / "c_pilot" / "pilot_packet_strata.csv"
OUT = ROOT / "results" / "c_pilot" / "pilot_rule_check.md"
BANDS = ["0-100", "101-200", "201-400", "401-800", "801-1600"]
NEAR = 100  # characters: the bare-year limit of the chosen rule
MIN_STRATUM = 3

RULES = {  # (band of the nearest occurrence, matched_cue) -> counted?
    "cue, whole window": lambda b, c: c == 1,
    "bare, whole window": lambda b, c: True,
    "bare <= 100": lambda b, c: b == "0-100",
    "bare <= 100 OR cue (chosen)": lambda b, c: b == "0-100" or c == 1,
}


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def precision_table(rows, pop, label=lambda r: r["label"]):
    rows = [r for r in rows if r["query_source"] != "stripped" and label(r) != "unclear"]
    cell = defaultdict(lambda: [0, 0])
    for r in rows:
        k = (r["band"], int(r["matched_cue"]))
        cell[k][1] += 1
        cell[k][0] += label(r) == "correct"
    pooled = {c: [sum(cell[(b, c)][i] for b in BANDS) for i in (0, 1)] for c in (0, 1)}
    prec = {}
    for b in BANDS:
        for c in (0, 1):
            k = (b, c)
            prec[k] = cell[k][0] / cell[k][1] if cell[k][1] >= MIN_STRATUM else pooled[c][0] / max(pooled[c][1], 1)
    total = sum(pop[k] * prec[k] for k in prec)
    out = []
    for name, f in RULES.items():
        ks = [k for k in prec if f(*k)]
        n = sum(pop[k] for k in ks)
        cor = sum(pop[k] * prec[k] for k in ks)
        raw = [r for r in rows if f(r["band"], int(r["matched_cue"]))]
        rk = sum(label(r) == "correct" for r in raw)
        lo, hi = wilson(rk, len(raw))
        out.append([name, "%.2f" % (cor / n), "%.2f" % (cor / total), "%d of %d" % (rk, len(raw)),
                    "%.2f to %.2f" % (lo, hi)])
    return len(rows), cell, prec, out


def md_table(header, body):
    return ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"] + [
        "| " + " | ".join(str(x) for x in row) + " |" for row in body]


def nearest_rank(vals, p):
    v = sorted(vals)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)] if v else ""


def window_matches(entries, rule):
    if rule == "cue, whole window":
        return any(c for _, c in entries)
    if rule == "bare, whole window":
        return bool(entries)
    if rule == "bare <= 100":
        return any(o.distance <= NEAR for o, _ in entries)
    return any(o.distance <= NEAR or c for o, c in entries)


def main():
    packet = list(csv.DictReader(open(PACKET, encoding="utf-8-sig")))
    pop = defaultdict(int)
    for s in csv.DictReader(open(STRATA, encoding="utf-8")):
        if s["query_source"] != "stripped":
            pop[(s["band"], int(s["matched_cue"]))] += int(s["population"])
    labelled = [r for r in packet if r["label"].strip()]
    out = ["# Counting rule check", "",
           "Numbers only. Written by `scripts/pilot_rule_check.py` from the labelled packet "
           "(%d of %d rows labelled), `pilot_packet_strata.csv` and the window cache (read-only). "
           "Short forms of disambiguated names (source `stripped`) are left out throughout: "
           "they were correct in 1 of 9 scored packet rows (`pilot_precision.md`)."
           % (len(labelled), len(packet)), ""]

    n, cell, prec, base = precision_table(packet, pop)
    out += ["## 1. Precision on the packet", "",
            "Per stratum (nearest occurrence's distance band x relation cue), non-stripped rows. "
            "Strata with fewer than %d scored rows use the pooled precision of their cue value." % MIN_STRATUM, ""]
    out += md_table(["band", "cue", "correct of scored", "precision used", "candidates in population"],
                    [[b, c, "%d of %d" % tuple(cell[(b, c)]), "%.2f" % prec[(b, c)], pop[(b, c)]]
                     for b in BANDS for c in (0, 1)])
    head = ["rule", "weighted precision", "kept", "raw correct", "95% Wilson interval (raw)"]
    variants = [
        ("As labelled (%d scored rows)" % n, packet, lambda r: r["label"]),
    ]
    seen, once = set(), []
    for r in packet:
        if r["snippet"] not in seen:
            seen.add(r["snippet"])
            once.append(r)
    variants += [
        ("Identical snippets counted once (%d of %d rows kept)" % (len(once), len(packet)), once, lambda r: r["label"]),
        ("Without Aleksei Musin-Pushkin (15 of 60 rows)",
         [r for r in packet if not r["prompt_name"].startswith("Aleksei Musin-Pushkin")], lambda r: r["label"]),
        ("P019 and P044 relabelled correct", packet,
         lambda r: "correct" if r["packet_id"] in ("P019", "P044") else r["label"]),
    ]
    for title, rows, lab in variants:
        _, _, _, body = precision_table(rows, pop, lab)
        out += ["", "### " + title, ""] + md_table(head, body)

    data = cli.load_cache(cli.DEFAULT_CACHE, None)
    years = cli.load_years(cli.DEFAULT_PROMPTS)
    strings = defaultdict(list)
    for e in data["entities"]:
        if e["active"] and e["source"] != "stripped":
            strings[e["prompt_name"]].append(e["query_string"])
    sampled = defaultdict(bool)
    for p in data["plan"]:
        sampled[(p["index_name"], p["query_string"])] |= p["selected"] == "sample"
    wins = defaultdict(list)
    for w in data["windows"]:
        wins[(w["index_name"], w["query_string"])].append(w)
    cache = pc._WinCache()
    n_c = defaultdict(lambda: defaultdict(list))
    for name in data["names"]:
        for fact in pc.fact_match.FACTS:
            for year in years.get(name, {}).get(fact, []):
                py = {f: year if f == fact else (min(years[name][f]) if years[name][f] else None)
                      for f in pc.fact_match.FACTS}
                for idx in data["indexes"]:
                    if any(sampled[(idx, q)] for q in strings[name]):
                        continue  # est_mentions units, kept apart as in pilot_summary.md
                    for rule in RULES:
                        docs = {cache.get(w).doc_key for q in strings[name] for w in wins[(idx, q)]
                                if window_matches(pc._entries(cache.get(w), year, fact, py), rule)}
                        n_c[rule][idx].append(len(docs))
    body = []
    for rule in RULES:
        for idx in data["indexes"]:
            v = n_c[rule][idx]
            pos = [x for x in v if x > 0]
            body.append([rule, pc.INDEX_LABELS.get(idx, idx), len(v), v.count(0), len(pos),
                         nearest_rank(pos, 10), nearest_rank(v, 50)])
    disamb = sorted({e["prompt_name"] for e in data["entities"] if e["active"] and e["source"] == "stripped"})
    names_all = sorted({e["prompt_name"] for e in data["entities"] if e["active"]})
    out += ["", "## 2. n_c on the 20 pilot names", "",
            "Distinct documents per (name, fact, index) under each rule; facts whose strings are all fully "
            "fetched (docs units) only. p10 is nearest-rank over n_c > 0. A 20-name preview, not tau.", ""]
    out += md_table(["rule", "index", "facts", "n_c = 0", "n_c > 0", "p10 (n_c > 0)", "median (incl. 0)"], body)
    out += ["", "## 3. Names counted without their short form", "",
            "%d of the %d names in the cache have a short (stripped) form; with it left out, their n_c "
            "comes from the name as written and any aliases, so it is a lower bound." % (len(disamb), len(names_all))]
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
