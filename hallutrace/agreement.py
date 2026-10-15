"""Agreement numbers for one task's annotation packet.

    pilot        inter-annotator kappa on the rows both people have labelled
    adjudicate   the rows where A and B differ, written to adjudication.csv for the
                 pair to fill in `gold_label` and `reason`; the originals are never
                 touched
    compute      per task: inter-annotator kappa, verifier kappa against gold,
                 verifier raw agreement, the two confusion matrices with every cell,
                 and the stratification ratio beside the natural base rate

Input is one task directory as `gold_cut.write_packets` leaves it: annotator_A.csv,
annotator_B.csv, key.csv, manifest.json, plus adjudication.csv once adjudication has
run. Gold is the shared label where A and B agree, and the adjudicated one where they
don't.

The gate applies to the verifier's kappa against gold, not to the annotators' kappa:
accept at 0.800, tentative from 0.667, quarantine below.

Three extraction tasks (false_presuppositions, rationalization_binary,
rationalization_numerical) ask the annotator whether the extractor captured what the
model said, while key.csv's verifier label says whether the model's answer was wrong.
The two answer different questions, so kappa between them measures nothing. For these
tasks the second number is the extractor's accuracy instead:
the share of rows gold calls `supported` (a faithful extraction). A manifest may name
the measure itself (`"measure": "extractor_accuracy"` or `"kappa"`), as
`gold_cut.write_packets` does from `gold_cut.MEASURES`.

Tags go in the packet's `tag` column, one of `not a claim`, `exaggerated`, `no article` or
blank, with `notes` left as free text. Any other value in `tag` is
refused, so a typo stops the script instead of moving a row between splits unseen. A
packet drawn before the column existed falls back to the old rule, a `notes` cell that
is exactly a tag (`tag_source` says which applied).

Rows both annotators tagged `not a claim` are atoms the
verifier can only guess on. Every number is reported twice, with and without those
rows, so a single kappa does not mostly measure the guessing. A row only one annotator
tagged stays in both splits: it is mostly a disagreement, and adjudication settles it
(dropping it would push the inter-annotator kappa up). `marked_by_one`
counts them, and `claim_rows` is the n the "without" inter-annotator numbers rest on.
`tags` counts every tag per annotator, `exaggerated` included.

Adjudication can settle a row as `not a claim` in adjudication.csv's own `tag` column.
Such a row also leaves the "without" split for the verifier's
numbers, since gold decides what counts as a claim. The inter-annotator numbers keep
the both-tagged rule, which protects them. `settled_not_a_claim` lists those
rows and `verifier_rows` is the n of the "without" verifier numbers. The gate reads the
all-rows numbers either way, so this moves only the diagnostic.

Every gated number carries a 95% interval: Wilson for extractor accuracy, a bootstrap
over items for kappa, resampled within the flagged and unflagged strata as the draw
takes them. The band is set on the point estimate; `edge_crossing` flags a
number whose interval crosses a band edge, for the team to look at (no extra
labelling by default).

Kappa on the drawn sample is stratified (the draw takes 40 to 50 flagged units of
100). `natural` reweights each row by how over- or under-drawn its flagged side
was, as an estimate for the task's own base rate; it is labelled as such and
never replaces the stratified number.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from typing import Dict, List, Optional

from sklearn.metrics import cohen_kappa_score, confusion_matrix

HALLUCINATED = "hallucinated"
SUPPORTED = "supported"
UNVERIFIABLE = "unverifiable"
LABELS = [HALLUCINATED, SUPPORTED, UNVERIFIABLE]   # same order as gold_cut.LABELS
NOT_A_CLAIM, EXAGGERATED, NO_ARTICLE = "not a claim", "exaggerated", "no article"
TAGS = (NOT_A_CLAIM, EXAGGERATED, NO_ARTICLE)   # the `tag` column, besides blank
ACCEPT, TENTATIVE = 0.800, 0.667
# Extractor accuracy (FP, RN, RB), fixed before annotation started. The packets are drawn
# from the extraction after the unit filter.
ACC_ACCEPT, ACC_TENTATIVE = 0.90, 0.80
# Only scientific_attribution's verifier can say unverifiable; the other seven are
# binary, so their verifier-against-gold matrix drops that row.
THREE_WAY_VERIFIERS = {"scientific_attribution"}
EXTRACTOR_TASKS = {"false_presuppositions", "rationalization_binary", "rationalization_numerical"}
VERIFIER_KAPPA, EXTRACTOR_ACCURACY = "kappa", "extractor_accuracy"   # gold_cut.KAPPA, gold_cut.EXTRACTOR_ACCURACY

ADJUDICATION_COLUMNS = ("item_id", "label_A", "label_B", "tag_A", "tag_B", "notes_A",
                        "notes_B", "gold_label", "tag", "reason")


class PacketError(ValueError):
    pass


def _read(path: str) -> List[dict]:
    # utf-8-sig: the packets go out with a BOM for Excel, and Excel's "CSV UTF-8" saves one,
    # which plain utf-8 would read into the first column name ("﻿item_id").
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _label(value: str, where: str) -> str:
    v = (value or "").strip().lower()
    if v and v not in LABELS:
        raise PacketError("%s: %r is not one of %s" % (where, value, LABELS))
    return v


def _tag(value: str, where: str) -> str:
    v = (value or "").strip().lower()
    if v and v not in TAGS:
        raise PacketError("%s: tag %r is not blank or one of %s" % (where, value, list(TAGS)))
    return v


def tag_from_notes(notes: str) -> str:
    """A packet without a `tag` column: a `notes` cell that is exactly a tag, else blank."""
    n = (notes or "").strip().lower()
    return n if n in TAGS else ""


def mentions_not_a_claim(notes: str, tag: str) -> bool:
    """`notes` mentions `not a claim` but the row's tag isn't it ("not a claim, off topic").

    Such a row is not dropped from the "without" split, so it is listed for a look.
    """
    return NOT_A_CLAIM in (notes or "").lower() and tag != NOT_A_CLAIM


def load(task_dir: str, a_path: Optional[str] = None, b_path: Optional[str] = None) -> dict:
    """Join A, B, key and adjudication by item_id. Checks the packet is consistent.
    `a_path` / `b_path` read the annotators' files from elsewhere, e.g. the
    `annotator_<name>.csv` each annotator commits to `annotation/e1/<task>/`."""
    a = _read(a_path or os.path.join(task_dir, "annotator_A.csv"))
    b = _read(b_path or os.path.join(task_dir, "annotator_B.csv"))
    key = _read(os.path.join(task_dir, "key.csv"))
    with open(os.path.join(task_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    ids = [r["item_id"] for r in a]
    if [r["item_id"] for r in b] != ids or [r["item_id"] for r in key] != ids:
        raise PacketError("annotator_A, annotator_B and key must list the same item_ids "
                          "in the same order")
    adj = {}
    adj_path = os.path.join(task_dir, "adjudication.csv")
    if os.path.exists(adj_path):
        for r in _read(adj_path):
            adj[r["item_id"]] = r
    has_tag = [bool(rows) and "tag" in rows[0] for rows in (a, b)]
    if has_tag[0] != has_tag[1]:
        raise PacketError("only one of annotator_A and annotator_B has a tag column")
    rows = []
    for ra, rb, rk in zip(a, b, key):
        i = ra["item_id"]
        la = _label(ra.get("annotator_label"), "annotator_A " + i)
        lb = _label(rb.get("annotator_label"), "annotator_B " + i)
        if bool(la) != bool(lb):
            raise PacketError("%s is labelled by only one annotator" % i)
        r = {"item_id": i, "a": la, "b": lb,
             "notes_a": ra.get("notes", ""), "notes_b": rb.get("notes", ""),
             "verifier": _label(rk["verifier_label"], "key " + i),
             "flagged": rk.get("flagged", "") == "1"}
        if has_tag[0]:
            r["tag_a"] = _tag(ra.get("tag"), "annotator_A " + i)
            r["tag_b"] = _tag(rb.get("tag"), "annotator_B " + i)
        else:
            r["tag_a"], r["tag_b"] = tag_from_notes(r["notes_a"]), tag_from_notes(r["notes_b"])
        nac_a, nac_b = r["tag_a"] == NOT_A_CLAIM, r["tag_b"] == NOT_A_CLAIM
        r["not_a_claim"] = nac_a and nac_b
        r["marked_by_one"] = nac_a != nac_b
        r["mentions_not_a_claim"] = (mentions_not_a_claim(r["notes_a"], r["tag_a"])
                                     or mentions_not_a_claim(r["notes_b"], r["tag_b"]))
        if la and la == lb:
            r["gold"] = la
        else:
            r["gold"] = _label(adj.get(i, {}).get("gold_label", ""), "adjudication " + i)
        # adjudication.csv's own tag; files written before the column read as blank
        r["gold_tag"] = _tag(adj.get(i, {}).get("tag", ""), "adjudication " + i)
        r["settled_not_a_claim"] = r["gold_tag"] == NOT_A_CLAIM and not r["not_a_claim"]
        rows.append(r)
    return {"task": manifest["task"], "manifest": manifest, "rows": rows,
            "tag_source": "tag column" if has_tag[0] else "notes, exact (no tag column)",
            "adjudication": adj_path if adj else None}


def disagreements(packet: dict) -> List[dict]:
    """The labelled rows where A and B differ, keeping any gold already filled in."""
    out = []
    for r in packet["rows"]:
        if r["a"] and r["a"] != r["b"]:
            out.append({"item_id": r["item_id"], "label_A": r["a"], "label_B": r["b"],
                        "tag_A": r["tag_a"], "tag_B": r["tag_b"],
                        "notes_A": r["notes_a"], "notes_B": r["notes_b"],
                        "gold_label": r["gold"], "tag": r["gold_tag"], "reason": ""})
    return out


def write_adjudication(task_dir: str, packet: dict) -> str:
    """Write adjudication.csv, with a BOM so Excel opens it as UTF-8 (the tie-breaker fills
    it in by hand). Gold labels, tags and reasons already written there are kept."""
    path = os.path.join(task_dir, "adjudication.csv")
    old = {r["item_id"]: r for r in _read(path)} if os.path.exists(path) else {}
    rows = disagreements(packet)
    for r in rows:
        r["reason"] = old.get(r["item_id"], {}).get("reason", "")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ADJUDICATION_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def _kappa(x: List[str], y: List[str], weights=None) -> Optional[float]:
    if not x:
        return None
    k = cohen_kappa_score(x, y, labels=LABELS, sample_weight=weights)
    return None if math.isnan(k) else round(float(k), 4)


def _agree(x: List[str], y: List[str], weights=None) -> Optional[float]:
    if not x:
        return None
    w = weights or [1.0] * len(x)
    return round(sum(wi for xi, yi, wi in zip(x, y, w) if xi == yi) / sum(w), 4)


def band(kappa: Optional[float]) -> Optional[str]:
    if kappa is None:
        return None
    return "accept" if kappa >= ACCEPT else "tentative" if kappa >= TENTATIVE else "quarantine"


N_BOOT = 2000


def _kappa_fast(x: List[str], y: List[str]) -> Optional[float]:
    """Unweighted Cohen's kappa over LABELS, as sklearn computes it; for the bootstrap."""
    n = len(x)
    if not n:
        return None
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    pe = sum((x.count(l) / n) * (y.count(l) / n) for l in LABELS)
    return None if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_kappa(rows: List[dict], xk: str, yk: str, n_boot: int = N_BOOT,
                    seed: int = 20261002) -> Optional[List[float]]:
    """95% percentile interval for kappa, resampling items within the flagged strata."""
    strata = [s for s in ([r for r in rows if r["flagged"]], [r for r in rows if not r["flagged"]]) if s]
    if sum(len(s) for s in strata) < 2:
        return None
    rng = random.Random(seed)
    ks = []
    for _ in range(n_boot):
        sample = [rng.choice(s) for s in strata for _ in s]
        k = _kappa_fast([r[xk] for r in sample], [r[yk] for r in sample])
        if k is not None:
            ks.append(k)
    if len(ks) < n_boot // 2:
        return None
    ks.sort()
    return [round(ks[int(0.025 * (len(ks) - 1))], 4), round(ks[int(0.975 * (len(ks) - 1))], 4)]


def edges_crossed(ci: Optional[List[float]], edges) -> List[float]:
    """Band edges strictly inside the interval."""
    if not ci:
        return []
    return [e for e in edges if ci[0] < e < ci[1]]


def accuracy_band(acc: Optional[float]) -> Optional[str]:
    if acc is None:
        return None
    return "accept" if acc >= ACC_ACCEPT else "tentative" if acc >= ACC_TENTATIVE else "quarantine"


def wilson(k: float, n: int, z: float = 1.96) -> Optional[List[float]]:
    """95% Wilson interval for k successes in n; n = 50 at 0.90 gives about 0.79 to 0.96."""
    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def natural_weights(rows: List[dict], base_rate, sampling_ratio) -> Optional[List[float]]:
    """Weight flagged rows by base/ratio and the rest by (1-base)/(1-ratio)."""
    if base_rate is None or not sampling_ratio or sampling_ratio >= 1:
        return None
    wf = base_rate / sampling_ratio
    wu = (1 - base_rate) / (1 - sampling_ratio)
    return [wf if r["flagged"] else wu for r in rows]


def measure_of(task: str, manifest: dict) -> str:
    m = manifest.get("measure") or (EXTRACTOR_ACCURACY if task in EXTRACTOR_TASKS
                                    else VERIFIER_KAPPA)
    if m not in (VERIFIER_KAPPA, EXTRACTOR_ACCURACY):
        raise PacketError("manifest measure %r is neither %r nor %r"
                          % (m, VERIFIER_KAPPA, EXTRACTOR_ACCURACY))
    return m


def _extractor(golded: List[dict], missing: int, stats: dict) -> dict:
    """Share of rows gold calls a faithful extraction, and the label counts."""
    out = {"n": len(golded), "gold_missing": missing}
    if not golded or missing:
        return out
    ok = [1.0 if r["gold"] == SUPPORTED else 0.0 for r in golded]
    w = natural_weights(golded, stats.get("base_rate"), stats.get("sampling_ratio"))
    acc = round(sum(ok) / len(ok), 4)
    ci = wilson(sum(ok), len(ok))
    crossed = edges_crossed(ci, (ACC_TENTATIVE, ACC_ACCEPT))
    out.update({
        "accuracy": acc,
        "ci95": ci,
        "edge_crossing": bool(crossed), "edges_crossed": crossed,
        "gold_counts": {l: sum(r["gold"] == l for r in golded) for l in LABELS},
        "gate": accuracy_band(acc),
        "gate_note": "band %.2f / %.2f, fixed before annotation"
                     % (ACC_ACCEPT, ACC_TENTATIVE),
        "natural": None if w is None else {
            "accuracy": round(sum(o * wi for o, wi in zip(ok, w)) / sum(w), 4),
            "note": "reweighted to the verifier's natural flag rate; an estimate"},
    })
    return out


def _numbers(rows: List[dict], task: str, stats: dict, measure: str = VERIFIER_KAPPA,
             gold_rows: Optional[List[dict]] = None) -> dict:
    """Inter-annotator numbers on `rows`; the verifier's (or extractor's) on `gold_rows`,
    which defaults to `rows` and is smaller only in the "without" split."""
    lab = [r for r in rows if r["a"]]
    a, b = [r["a"] for r in lab], [r["b"] for r in lab]
    inter = {"n": len(lab), "kappa": _kappa(a, b), "agreement": _agree(a, b),
             "ci95": bootstrap_kappa(lab, "a", "b"),
             "matrix": {"rows": "annotator_A", "cols": "annotator_B", "labels": LABELS,
                        "cells": confusion_matrix(a, b, labels=LABELS).tolist() if lab else None}}
    glab = lab if gold_rows is None else [r for r in gold_rows if r["a"]]
    golded = [r for r in glab if r["gold"]]
    missing = len(glab) - len(golded)
    if measure == EXTRACTOR_ACCURACY:
        return {"inter_annotator": inter, "extractor": _extractor(golded, missing, stats)}
    verifier = {"n": len(golded), "gold_missing": missing}
    if golded and not missing:
        g, v = [r["gold"] for r in golded], [r["verifier"] for r in golded]
        cells = confusion_matrix(v, g, labels=LABELS).tolist()   # rows verifier, cols gold
        row_labels = list(LABELS)
        if task not in THREE_WAY_VERIFIERS:
            if any(cells[2]):
                raise PacketError("%s's verifier is binary but key.csv has unverifiable" % task)
            cells, row_labels = cells[:2], LABELS[:2]
        k = _kappa(g, v)
        ci = bootstrap_kappa(golded, "gold", "verifier")
        crossed = edges_crossed(ci, (TENTATIVE, ACCEPT))
        w = natural_weights(golded, stats.get("base_rate"), stats.get("sampling_ratio"))
        verifier.update({
            "kappa": k, "agreement": _agree(g, v), "gate": band(k),
            "ci95": ci, "edge_crossing": bool(crossed), "edges_crossed": crossed,
            "matrix": {"rows": "verifier", "cols": "gold", "row_labels": row_labels,
                       "col_labels": LABELS, "cells": cells},
            "natural": None if w is None else {
                "kappa": _kappa(g, v, w), "agreement": _agree(g, v, w),
                "note": "reweighted to the natural base rate; an estimate, not the gate"},
        })
    return {"inter_annotator": inter, "verifier": verifier}


def compute(packet: dict) -> dict:
    """All the numbers for one task, with and without the `not a claim` rows."""
    rows, task = packet["rows"], packet["task"]
    stats = packet["manifest"].get("stats", {})
    measure = measure_of(task, packet["manifest"])
    lab = [r for r in rows if r["a"]]
    nac = [r for r in lab if r["not_a_claim"]]
    settled = [r for r in lab if r["settled_not_a_claim"]]
    return {
        "task": task,
        "measure": measure,
        "items": len(rows), "labelled": len(lab),
        "sampling": {"natural_base_rate": stats.get("base_rate"),
                     "sampling_ratio": stats.get("sampling_ratio"),
                     "drawn_flagged": stats.get("drawn_flagged"), "drawn": stats.get("drawn")},
        "all_rows": _numbers(rows, task, stats, measure),
        "tag_source": packet.get("tag_source"),
        "tags": {t: {"A": sum(r["tag_a"] == t for r in lab), "B": sum(r["tag_b"] == t for r in lab),
                     "both": sum(r["tag_a"] == t == r["tag_b"] for r in lab)} for t in TAGS},
        "gold_tags": {t: sum(r["gold_tag"] == t for r in lab) for t in TAGS},
        "not_a_claim": {
            "n": len(nac),
            "rule": "dropped when both annotators tagged it; for the verifier's numbers "
                    "also when adjudication settled it as not a claim",
            "marked_by_one": sum(1 for r in lab if r["marked_by_one"]),
            "claim_rows": len(lab) - len(nac),
            "settled_not_a_claim": [r["item_id"] for r in settled],
            "verifier_rows": len(lab) - len(nac) - len(settled),
            "mentions_not_a_claim": [r["item_id"] for r in lab if r["mentions_not_a_claim"]],
            "without": _numbers([r for r in rows if not r["not_a_claim"]], task, stats, measure,
                                gold_rows=[r for r in rows if not (r["not_a_claim"]
                                                                   or r["settled_not_a_claim"])]),
        },
        "thresholds": {"accept": ACCEPT, "tentative": TENTATIVE},
        "accuracy_thresholds": {"accept": ACC_ACCEPT, "tentative": ACC_TENTATIVE},
    }
