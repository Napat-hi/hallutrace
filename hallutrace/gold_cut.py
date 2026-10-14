"""The gold cut: which verified units two people annotate, and the files they get.

    units   what one unit is, per task, and what the annotator is shown
    draw    100 units per task, about 40 to 50 the verifier flagged and 50 to 60
            it did not, shuffled, with the seed, the sampling ratio and the
            natural base rate recorded
    files   two identical files per task, one per annotator, with the verifier's
            answer deleted rather than hidden

What a unit is, per task:

    scientific_attribution     one extracted reference; shown the response
    false_presuppositions      one response's extraction, i.e. the extracted
    rationalization_binary     list or answer; shown the response, and asked
    rationalization_numerical  whether the extractor captured what it said
    historical_events          one whole response; shown it and the two names
    summarization              one atom of the decomposed response; shown the
    simplification             source passage, and asked whether it supports it

Summarization and Simplification are positional readers (verifier_runner), so
their output has no `generation_id`; `read_outputs` takes it from each unit's
sidecar (`inputs/{model}/{unit}.meta.csv`), in input row order. Draw them from
`outputs_filtered/` (`posthoc.py`), so the decomposer's framing lines never
reach an annotator. Biographies is in `gold_cut_bio.py`.

"The response" here is the text the verifier was given, after any unfinished
tail was dropped. Showing the tail would ask annotators to judge text the
verifier never saw.

The key file keeps what the annotators must not see: the verifier's label, the
unit's origin, and `would_drop_23`, whether the unit guard (drop an extracted
unit that is not in the response) would remove it. The sample is drawn from the
verifier's raw output either way, so both versions of every number can be
computed from the same annotation.
"""

from __future__ import annotations

import ast
import csv
import glob
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HALLUCINATED = "hallucinated"
SUPPORTED = "supported"
UNVERIFIABLE = "unverifiable"
LABELS = (HALLUCINATED, SUPPORTED, UNVERIFIABLE)

ANNOTATOR_COLUMNS = ("item_id", "context_shown", "unit_to_judge", "question",
                     "annotator_label", "tag", "notes")
# Values the `tag` column takes besides blank. The same tuple as
# agreement.TAGS, which reads them; kept here so drawing a packet does not
# import agreement's dependencies. A test checks the two agree.
TAGS = ("not a claim", "exaggerated", "no article")
KEY_COLUMNS = ("item_id", "unit_id", "generation_id", "model", "task",
               "verifier_label", "flagged", "would_drop_23", "source_file")

QUESTIONS = {
    "scientific_attribution":
        "Is this a real paper, and did the model actually cite it in this response?",
    "false_presuppositions":
        "Did the extractor capture what the model actually said?",
    "rationalization_binary":
        "Did the extractor capture what the model actually said?",
    "rationalization_numerical":
        "Did the extractor capture what the model actually said?",
    # a yes/no question, with the three-way mapping stated where the annotator
    # reads it, so two people cannot map it differently
    "historical_events":
        "Does this response assert that the meeting occurred? "
        "Yes: hallucinated. No: supported. Can't tell: unverifiable.",
    # the procedure's wording, word for word
    "summarization":
        "Is this fact entailed by the source passage?",
    "simplification":
        "Is this fact entailed by the source sentence?",
}

# What agreement reports per task. The extraction tasks judge whether the
# extractor captured what the model said, a question the verifier never answers,
# so they report extractor accuracy, not a kappa against the verifier.
# scientific_attribution's question covers both, so it stays on kappa.
# agreement.py reads this from the manifest.
EXTRACTOR_ACCURACY = "extractor_accuracy"
KAPPA = "kappa"
MEASURES = {
    "false_presuppositions": EXTRACTOR_ACCURACY,
    "rationalization_binary": EXTRACTOR_ACCURACY,
    "rationalization_numerical": EXTRACTOR_ACCURACY,
}


@dataclass
class Unit:
    unit_id: str
    generation_id: str
    task: str
    context: str
    unit_text: str
    verifier_label: str
    would_drop_23: bool = False
    source_file: str = ""

    @property
    def model(self) -> str:
        return self.generation_id.split("|", 1)[0]

    @property
    def flagged(self) -> bool:
        return self.verifier_label != SUPPORTED


# ------------------------------------------------------------------ parsing

def as_list(value) -> list:
    """A verifier cell as a list: "['a', 'b']", "set()", "", NaN all work."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        got = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(got, (list, tuple, set)):
        return list(got)
    return [got]


def _norm(text) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(text).lower())).strip()


def in_response(item, response: str, loose: bool = False) -> bool:
    """Whether an extracted item appears in the response.

    Exact means the normalised item occurs as a whole word sequence in the
    normalised response. Loose, for titles, means at least 80% of the item's
    words occur anywhere in it. An empty item is treated as present.
    """
    item_n = _norm(item)
    if not item_n:
        return True
    resp_n = _norm(response)
    if re.search(r"(?<!\w)" + re.escape(item_n) + r"(?!\w)", resp_n):
        return True
    if loose:
        words = set(item_n.split())
        return len(words & set(resp_n.split())) / len(words) >= 0.8
    return False


def _clean(items) -> List[str]:
    return [str(i).strip() for i in items if str(i).strip()]


# ------------------------------------------------------------------ units, per task

def _scientific_attribution(row: dict, source: str) -> List[Unit]:
    titles = _clean(as_list(row.get("atomic_units")))
    hal = {_norm(t) for t in as_list(row.get("hallucinated_atomic_units"))}
    unv = {_norm(t) for t in as_list(row.get("unverifiable_atomic_units"))}
    out = []
    for i, title in enumerate(titles):
        n = _norm(title)
        label = UNVERIFIABLE if n in unv else HALLUCINATED if n in hal else SUPPORTED
        out.append(Unit("%s:%d" % (row["generation_id"], i), row["generation_id"],
                        "scientific_attribution", row["response"], title, label,
                        not in_response(title, row["response"], loose=True), source))
    return out


def _extraction(task: str, fields: Sequence[str]) -> Callable[[dict, str], List[Unit]]:
    """One unit per response: the extraction, judged against the response."""
    def build(row: dict, source: str) -> List[Unit]:
        items = _clean(as_list(row.get("atomic_units")))
        parts = []
        for f in fields:
            v = row.get(f)
            if v not in (None, "") and str(v).strip() not in ("", "[]", "set()", "nan"):
                parts.append("%s: %s" % (f, ", ".join(_clean(as_list(v))) or str(v).strip()))
        shown = "; ".join(parts) if parts else "(nothing extracted)"
        flagged = bool(_clean(as_list(row.get("hallucinated_atomic_units"))))
        drop = any(not in_response(i, row["response"]) for i in items)
        return [Unit("%s:0" % row["generation_id"], row["generation_id"], task,
                     row["response"], shown, HALLUCINATED if flagged else SUPPORTED,
                     drop, source)]
    return build


def _historical_events(row: dict, source: str) -> List[Unit]:
    asserted = bool(_clean(as_list(row.get("hallucinated_atomic_units"))))
    return [Unit("%s:0" % row["generation_id"], row["generation_id"], "historical_events",
                 row["response"], "%s and %s" % (row["name1"], row["name2"]),
                 HALLUCINATED if asserted else SUPPORTED, False, source)]


def atom_lines(text) -> List[Tuple[str, bool]]:
    """(atom, verdict) from the decomposer cell: one `atom | True |` per line."""
    out = []
    for line in str(text or "").split("\n"):
        parts = line.split("|")
        if len(parts) >= 2 and parts[0].strip():
            out.append((parts[0].strip(), parts[1].strip() == "True"))
    return out


def unit_guard_loaded() -> bool:
    """Whether `would_drop_23` can be computed for summarization and simplification.

    Recorded in the manifest, so an all zero column is never mistaken for a
    measurement.
    """
    try:
        from hallutrace.unit_guard import framing_line  # noqa: F401
    except ImportError:
        return False
    return True


def _is_framing(atom: str, response: str) -> bool:
    if not unit_guard_loaded():
        return False
    from hallutrace.unit_guard import framing_line
    return framing_line(atom, response)


def _decompose(task: str) -> Callable[[dict, str], List[Unit]]:
    """One unit per atom, judged against the source passage."""
    def build(row: dict, source: str) -> List[Unit]:
        out = []
        for i, (atom, ok) in enumerate(atom_lines(row.get("atomic_units"))):
            out.append(Unit("%s:%d" % (row["generation_id"], i), row["generation_id"], task,
                            row["passage"], atom, SUPPORTED if ok else HALLUCINATED,
                            _is_framing(atom, row.get("response", "")), source))
        return out
    return build


UNIT_BUILDERS: Dict[str, Callable[[dict, str], List[Unit]]] = {
    "scientific_attribution": _scientific_attribution,
    "false_presuppositions": _extraction("false_presuppositions", ("atomic_units",)),
    "rationalization_binary": _extraction("rationalization_binary", ("yesno", "atomic_units")),
    "rationalization_numerical": _extraction(
        "rationalization_numerical", ("numerical_atomic_unit", "list_atomic_unit")),
    "historical_events": _historical_events,
    "summarization": _decompose("summarization"),
    "simplification": _decompose("simplification"),
}


def task_of_row(row: dict) -> str:
    return row["generation_id"].split("|")[1].rsplit("-", 1)[0]


def _sent_ids(meta_path: str) -> List[str]:
    """generation_ids a positional verifier was sent, in input row order."""
    with open(meta_path, encoding="utf-8", newline="") as f:
        sent = [r for r in csv.DictReader(f) if r.get("sent") == "1"]
    return [r["generation_id"] for r in sorted(sent, key=lambda r: int(r["input_row"]))]


# Outcomes a gold cut never draws: abstentions stay out of h, so kappa
# should measure only the verdicts we use.
EXCLUDED_OUTCOMES = ("abstention",)


def _outcomes(meta_path: str) -> Dict[str, str]:
    with open(meta_path, encoding="utf-8", newline="") as f:
        return {r["generation_id"]: r.get("outcome", "") for r in csv.DictReader(f)}


def read_outputs(outputs_dir: str, inputs_dir: Optional[str] = None,
                 skipped: Optional[List[str]] = None,
                 excluded: Optional[List[str]] = None) -> List[Tuple[dict, str]]:
    """Every row of every verifier output under outputs_dir/{model}/*.csv.

    With inputs_dir, each file's sidecar is read and rows whose outcome is in
    EXCLUDED_OUTCOMES are left out; pass a list as `excluded` to get their
    generation_ids back. Without inputs_dir nothing can be excluded.

    A file whose rows carry no generation_id (Summarization, Simplification)
    is joined to its sidecar under inputs_dir by row order, and skipped when
    inputs_dir or its sidecar is missing; pass a list as `skipped` to get those
    files back, so a smaller pool is never silent. A row count that disagrees with the sidecar raises,
    since a shifted join would pair every atom with the wrong response.
    """
    csv.field_size_limit(10 ** 9)
    rows = []
    for path in sorted(glob.glob(os.path.join(outputs_dir, "*", "*.csv"))):
        with open(path, encoding="utf-8", newline="") as f:
            got = list(csv.DictReader(f))
        if not got:
            continue
        rel = os.path.relpath(path, outputs_dir)
        meta = os.path.join(inputs_dir, os.path.splitext(rel)[0] + ".meta.csv") if inputs_dir else ""
        if "generation_id" not in got[0]:
            if "passage" not in got[0]:
                continue  # Biographies (name, atoms): gold_cut_bio.py builds those
            if not meta or not os.path.exists(meta):
                if skipped is not None:
                    skipped.append(rel)
                continue
            ids = _sent_ids(meta)
            if len(ids) != len(got):
                raise ValueError("%s has %d rows but its sidecar sent %d" % (rel, len(got), len(ids)))
            for row, gid in zip(got, ids):
                row["generation_id"] = gid
        if meta and os.path.exists(meta):
            outcome = _outcomes(meta)
            out = [r for r in got if outcome.get(r.get("generation_id")) in EXCLUDED_OUTCOMES]
            if excluded is not None:
                excluded.extend(r["generation_id"] for r in out)
            got = [r for r in got if outcome.get(r.get("generation_id")) not in EXCLUDED_OUTCOMES]
        rows.extend((row, rel) for row in got if row.get("generation_id"))
    return rows


def build_units(rows: List[Tuple[dict, str]]) -> Dict[str, List[Unit]]:
    by_task: Dict[str, List[Unit]] = {}
    for row, source in rows:
        task = task_of_row(row)
        if task not in UNIT_BUILDERS:
            raise KeyError("no gold cut adapter for %s (it is judged against a source; "
                           "write one once its verifier output exists)" % task)
        by_task.setdefault(task, []).extend(UNIT_BUILDERS[task](row, source))
    return by_task


# ------------------------------------------------------------------ the draw

@dataclass
class Draw:
    task: str
    seed: str
    units: List[Unit]
    pool: int
    pool_flagged: int
    stats: dict = field(default_factory=dict)


def draw(task: str, units: List[Unit], seed, n: int = 100,
         flagged_range: Tuple[int, int] = (40, 50)) -> Draw:
    """Aim for the middle of flagged_range, fill to n with unflagged.

    When one side is short, all of it is taken and the other side fills the
    rest, so a small pool still gives n units if it has n. The natural base
    rate and the sampling ratio are recorded either way, because the ratio is
    what agreement needs to reweight to the base rate.
    """
    rng = random.Random("%s:%s" % (seed, task))
    pos = sorted((u for u in units if u.flagged), key=lambda u: u.unit_id)
    neg = sorted((u for u in units if not u.flagged), key=lambda u: u.unit_id)
    lo, hi = flagged_range
    k_pos = min(len(pos), (lo + hi) // 2)
    k_neg = min(len(neg), n - k_pos)
    k_pos = min(len(pos), max(k_pos, n - k_neg))
    picked = rng.sample(pos, k_pos) + rng.sample(neg, k_neg)
    rng.shuffle(picked)
    total = len(units)
    stats = {
        "pool": total, "pool_flagged": len(pos),
        "base_rate": round(len(pos) / total, 4) if total else None,
        "drawn": len(picked), "drawn_flagged": k_pos,
        "sampling_ratio": round(k_pos / len(picked), 4) if picked else None,
        "in_flagged_range": lo <= k_pos <= hi,
        "short_of_n": len(picked) < n,
    }
    return Draw(task, str(seed), picked, total, len(pos), stats)


# ------------------------------------------------------------------ the files

def first_n(src_dir: str, out_dir: str, n: int = 50, packet_csv: Optional[str] = None,
            flagged_range: Tuple[int, int] = (40, 50)) -> Dict[str, str]:
    """The first n items of a drawn task, for annotators who label only those (50 per
    task). The draw is already shuffled, so the first n are a random subset; they
    are cut, not redrawn, so the item_ids and the seed stay those of `src_dir`.

    Writes key.csv (the first n rows) and manifest.json with the stats agreement reads
    recounted for n: drawn, drawn_flagged and sampling_ratio, the flagged range scaled
    to n, and the full draw kept as drawn_of_<N> and drawn_flagged_of_<N>, as
    biographies_first50 has them. With `packet_csv` (the annotator packet), its
    item_ids must be exactly the first n, in order. Never overwrites."""
    with open(os.path.join(src_dir, "key.csv"), encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols, key = reader.fieldnames, list(reader)
    with open(os.path.join(src_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if not 0 < n <= len(key):
        raise ValueError("n = %d, but %s has %d items" % (n, src_dir, len(key)))
    cut = key[:n]
    if packet_csv:
        with open(packet_csv, encoding="utf-8-sig", newline="") as f:
            ids = [r["item_id"] for r in csv.DictReader(f)]
        if ids != [r["item_id"] for r in cut]:
            raise ValueError("%s does not list the first %d item_ids of %s in order"
                             % (packet_csv, n, src_dir))
    if os.path.exists(os.path.join(out_dir, "key.csv")):
        raise ValueError("%s already has a key.csv; not overwriting" % out_dir)
    stats = dict(manifest.get("stats", {}))
    total, total_flagged = len(key), sum(r["flagged"] == "1" for r in key)
    k = sum(r["flagged"] == "1" for r in cut)
    lo, hi = flagged_range
    stats.update({"drawn": n, "drawn_flagged": k, "sampling_ratio": round(k / n, 4),
                  "in_flagged_range": lo * n / total <= k <= hi * n / total,
                  "drawn_of_%d" % total: total, "drawn_flagged_of_%d" % total: total_flagged})
    manifest = dict(manifest, stats=stats,
                    cut="first %d of the shuffled %d; same seed and order as %s"
                        % (n, total, os.path.basename(os.path.normpath(src_dir))))
    os.makedirs(out_dir, exist_ok=True)
    paths = {"key.csv": os.path.join(out_dir, "key.csv"),
             "manifest.json": os.path.join(out_dir, "manifest.json")}
    with open(paths["key.csv"], "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(cut)
    with open(paths["manifest.json"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return paths


def write_packets(d: Draw, out_dir: str, sources: Sequence[str] = (),
                  extra: Optional[dict] = None) -> Dict[str, str]:
    """annotator_A.csv and annotator_B.csv (identical), key.csv, manifest.json.

    `extra` goes into the manifest as is, for what the run knew about its
    inputs (skipped files, whether unit_guard was loaded).
    """
    task_dir = os.path.join(out_dir, d.task)
    os.makedirs(task_dir, exist_ok=True)
    rows, key = [], []
    for i, u in enumerate(d.units, 1):
        item_id = "%s-%03d" % (d.task, i)
        rows.append({"item_id": item_id, "context_shown": u.context,
                     "unit_to_judge": u.unit_text, "question": QUESTIONS[u.task],
                     "annotator_label": "", "tag": "", "notes": ""})
        key.append({"item_id": item_id, "unit_id": u.unit_id,
                    "generation_id": u.generation_id, "model": u.model, "task": u.task,
                    "verifier_label": u.verifier_label, "flagged": int(u.flagged),
                    "would_drop_23": int(u.would_drop_23), "source_file": u.source_file})
    paths = {}
    for name, cols, data in (("annotator_A.csv", ANNOTATOR_COLUMNS, rows),
                             ("annotator_B.csv", ANNOTATOR_COLUMNS, rows),
                             ("key.csv", KEY_COLUMNS, key)):
        p = os.path.join(task_dir, name)
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cols), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
        paths[name] = p
    manifest = {"task": d.task, "seed": d.seed, "labels": list(LABELS), "tags": list(TAGS),
                "question": QUESTIONS[d.task], "measure": MEASURES.get(d.task, KAPPA),
                "stats": d.stats,
                "sources": sorted(set(sources))}
    manifest.update(extra or {})
    p = os.path.join(task_dir, "manifest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    paths["manifest.json"] = p
    return paths
