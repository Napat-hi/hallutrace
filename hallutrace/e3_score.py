"""The hand decomposition check: the machine's atomic facts scored against a human's.

The machine's atoms are matched to the human's and scored by precision, recall and F1.
The match is many-to-one: the machine splits finer than a person does, so several machine
atoms may sit inside one human atom, and each of them counts as found.

The rule, `score_containment`: atom a is inside atom b when at least `threshold` of a's
content words (`tokens`) appear in b (`coverage`). A machine atom and a human atom match
when either is inside the other. Precision is the share of machine atoms that match some
human atom; recall is the share of human atoms that match some machine atom. The headline
threshold is 0.8 (`P0_10_1_THRESHOLD`); 0.6 is a sensitivity row. `score`, the one-to-one
matcher, stays as the second reading, which shows what the finer split costs.

The join: the packet's key.csv (`decomp_cut.packet_files`) gives each item's generation_id
and verification unit. For Biographies, the unit's input file maps the generation_id to the row
name, and the verifier output row with that name holds OpenFActScore's generator output
after `bio_filter` (`atoms`); the `raw` variant adds back what the filter dropped
(`dropped atoms`). Summarization and Simplification read their input by position, so the
unit's sidecar (`.meta.csv`) maps the generation_id to `input_row`, the row of the output
file; `filtered` is the decomposer's atoms after `posthoc.filter_row`, `raw` as written.

Counts, per item: machine atoms m, human atoms h. Micro sums over items; macro averages
the per-item values over the items where each is defined. An item with no human atoms (a
response that states nothing, e.g. `biographies-d21`) counts as h = 0, not as a
missing row: its machine atoms are all unmatched.

Intervals (`bootstrap_micro`, for the supplementary table): 95% percentile intervals for the micro values,
resampling whole items, seeded.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from hallutrace import bio_filter, posthoc

TASKS = ("biographies", "summarization", "simplification")
VARIANTS = ("filtered", "raw")
P0_10_1_THRESHOLD = 0.8      # the headline containment threshold
SENSITIVITY_THRESHOLD = 0.6  # reported beside it, in the supplementary material
# Words that only point at the subject: the annotator writes the name where Gemma writes
# "He", and "the speaker" where the response says "I", so neither side is credited for them.
# "speaker" alone is a content word ("He is a speaker."), so only the phrase "the speaker"
# and the possessive "speaker's" are dropped (`_SPEAKER`).
SUBJECT_WORDS = frozenset("he she they him her his hers their theirs them i me my".split())
STOPWORDS = frozenset("a an the is was are were be been of in on at to and as for by with "
                      "that this from".split())
# One side negated and the other not is never a match, however many words they share:
# "He was not British." does not sit inside "Ada Zed was British."
NEGATIONS = frozenset("not no never cannot".split())
_SPEAKER = re.compile(r"\bthe speaker's\b|\bthe speaker\b|\bspeaker's\b", re.I)
DEFAULT_THRESHOLD = 0.6

csv.field_size_limit(1 << 30)


class JoinError(ValueError):
    pass


def _read(path: str) -> List[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def human_atoms(cell: Optional[str]) -> List[str]:
    """One atom per line; blank lines and leading bullets or numbers dropped."""
    out = []
    for line in (cell or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line:
            out.append(line)
    return out


def machine_atoms_bio(row: dict, variant: str = "filtered") -> List[str]:
    """The generator's atoms for one Biographies output row."""
    if variant not in VARIANTS:
        raise ValueError("variant must be one of %s" % (VARIANTS,))
    atoms = [a for a, _ in bio_filter.parse_atoms(row.get("atoms"))]
    if variant == "raw":
        dropped = json.loads(row.get("dropped atoms") or "{}")
        for reason in ("header", "invented", "refusal"):
            atoms += dropped.get(reason, [])
    return atoms


def machine_atoms_decompose(row: dict, variant: str = "filtered") -> List[str]:
    """The decomposer's atoms for one Summarization or Simplification output row:
    `raw` as the verifier wrote them, `filtered` after the filter (framing lines out)."""
    if variant not in VARIANTS:
        raise ValueError("variant must be one of %s" % (VARIANTS,))
    if variant == "filtered":
        row = posthoc.filter_row("decompose", row)
    return [a for a, _ in posthoc._atom_lines(row.get("atomic_units"))]


def _stem(t: str) -> str:
    """A crude suffix strip, so "represents" and "represented" meet. "-es" comes off only
    after s, x, z, ch or sh ("passes", "boxes"); elsewhere only the "s" does, and a final
    "e" goes, so "roles" meets "role" and "released" meets "release". A stem that would
    read as a negation becomes "note" ("note", "notes", "noted" meet; none is "not")."""
    stem = t
    if len(t) > 4 and t.endswith("ies"):
        stem = t[:-3] + "y"
    else:
        for suffix in ("ing", "ed", "es", "s"):
            if not t.endswith(suffix) or len(t) - len(suffix) < 3:
                continue
            if suffix == "es" and not re.search(r"(?:s|x|z|ch|sh)$", t[:-2]):
                continue
            if suffix == "s" and t.endswith("ss"):
                break
            stem = t[: -len(suffix)]
            break
    if stem.endswith("e") and len(stem) >= 4:
        stem = stem[:-1]
    return stem + "e" if stem in NEGATIONS and t not in NEGATIONS else stem


def _clean(text: str) -> str:
    """Curly apostrophes made straight, contractions opened ("wasn't" -> "was not"), and
    possessives and the other clitics dropped ("Ada Zed's mother" -> "Ada Zed mother"), so
    no stray "s" or "t" is left as a word; "the speaker" taken out (see SUBJECT_WORDS)."""
    t = text.replace("’", "'").replace("‘", "'")
    t = re.sub(r"\bcan't\b", "can not", t, flags=re.I)
    t = re.sub(r"\bwon't\b", "will not", t, flags=re.I)
    t = re.sub(r"n't\b", " not", t, flags=re.I)
    t = _SPEAKER.sub(" ", t)
    t = re.sub(r"'(?:s|re|ve|ll|d|m)\b", "", t, flags=re.I)
    return t.replace("'", " ")


def negated(text: str) -> bool:
    """Whether the atom carries a negation (`NEGATIONS`, contractions opened)."""
    return any(w in NEGATIONS for w in re.findall(r"[a-z]+", _clean(text).lower()))


def tokens(text: str, topic: str = "") -> Counter:
    """Content words, stemmed, with pronouns and the subject's name left out. A word of the
    name counts as the name only where it is capitalised, so with the topic "Will Smith",
    "He will release an album." keeps "will"."""
    name = set(re.findall(r"[a-z0-9]+", topic.lower()))
    out = Counter()
    for w in re.findall(r"[A-Za-z0-9]+", _clean(text)):
        lw = w.lower()
        if lw in SUBJECT_WORDS or lw in STOPWORDS:
            continue
        if lw in name and not w.islower():
            continue
        out[_stem(lw)] += 1
    return out


def coverage(part: str, whole: str, topic: str = "") -> float:
    """Share of `part`'s content words found in `whole`: how far one atom is contained in
    another, for a many-to-one reading of the match (one human atom covering several finer
    machine atoms). The headline measure, through `contained`."""
    tp, tw = tokens(part, topic), tokens(whole, topic)
    return sum((tp & tw).values()) / sum(tp.values()) if tp else 0.0


def token_overlap(a: str, b: str, topic: str = "") -> float:
    """Provisional similarity: F1 of the two atoms' content words (multisets); 0 when only
    one of them is negated."""
    if negated(a) != negated(b):
        return 0.0
    ta, tb = tokens(a, topic), tokens(b, topic)
    if not ta or not tb:
        return 1.0 if not ta and not tb and a.strip().lower() == b.strip().lower() else 0.0
    common = sum((ta & tb).values())
    return 2 * common / (sum(ta.values()) + sum(tb.values()))


def match(machine: Sequence[str], human: Sequence[str], topic: str = "",
          sim: Callable[[str, str, str], float] = token_overlap,
          threshold: float = DEFAULT_THRESHOLD) -> List[Tuple[int, int, float]]:
    """One-to-one pairs (machine index, human index, score), greedy by score, ties by
    position, each pair at or above the threshold."""
    cand = [(sim(m, h, topic), i, j) for i, m in enumerate(machine) for j, h in enumerate(human)]
    cand = [c for c in cand if c[0] >= threshold]
    cand.sort(key=lambda c: (-c[0], c[1], c[2]))
    used_m, used_h, pairs = set(), set(), []
    for s, i, j in cand:
        if i not in used_m and j not in used_h:
            used_m.add(i)
            used_h.add(j)
            pairs.append((i, j, round(s, 4)))
    return sorted(pairs)


def _prf(k: int, m: int, h: int) -> Dict[str, Optional[float]]:
    p = k / m if m else None
    r = k / h if h else None
    f = 2 * p * r / (p + r) if p is not None and r is not None and p + r else (
        0.0 if p is not None and r is not None else None)
    return {"precision": None if p is None else round(p, 4),
            "recall": None if r is None else round(r, 4),
            "f1": None if f is None else round(f, 4)}


def score(items: List[dict], sim=token_overlap, threshold: float = DEFAULT_THRESHOLD,
          matcher_name: str = "token_overlap (provisional, not P0-10.1)") -> dict:
    """Per item and overall. Each item: item_id, model, topic, machine [...], human [...]."""
    per_item = []
    for it in items:
        pairs = match(it["machine"], it["human"], it.get("topic", ""), sim, threshold)
        k, m, h = len(pairs), len(it["machine"]), len(it["human"])
        per_item.append(dict({"item_id": it["item_id"], "model": it.get("model", ""),
                              "machine": m, "human": h, "matched": k}, **_prf(k, m, h),
                             pairs=pairs))

    def summary(rows):
        k = sum(r["matched"] for r in rows)
        m = sum(r["machine"] for r in rows)
        h = sum(r["human"] for r in rows)
        macro = {}
        for key in ("precision", "recall", "f1"):
            vals = [r[key] for r in rows if r[key] is not None]
            macro[key] = round(sum(vals) / len(vals), 4) if vals else None
            macro["n_" + key] = len(vals)
        return {"items": len(rows), "machine_atoms": m, "human_atoms": h, "matched": k,
                "micro": _prf(k, m, h), "macro": macro}

    models = sorted({r["model"] for r in per_item})
    return {"matcher": matcher_name, "threshold": threshold,
            "overall": summary(per_item),
            "by_model": {mo: summary([r for r in per_item if r["model"] == mo]) for mo in models},
            "items": per_item}


def contained(a: str, b: str, topic: str = "") -> float:
    """How far one atom sits inside the other, either way round; 0 when only one
    of them is negated, since the shorter could otherwise sit inside its own denial."""
    if negated(a) != negated(b):
        return 0.0
    return max(coverage(a, b, topic), coverage(b, a, topic))


def _best(atom: str, others: Sequence[str], topic: str) -> Tuple[Optional[int], float]:
    """(index, score) of the atom in `others` that `atom` best matches; ties by position."""
    best, best_s = None, -1.0
    for j, o in enumerate(others):
        s = contained(atom, o, topic)
        if s > best_s:
            best, best_s = j, s
    return best, max(best_s, 0.0)


def _rate(k: int, n: int) -> Optional[float]:
    return round(k / n, 4) if n else None


def _f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None:
        return None
    return round(2 * p * r / (p + r), 4) if p + r else 0.0


def score_containment(items: List[dict], threshold: float = P0_10_1_THRESHOLD) -> dict:
    """Many-to-one matching by containment. Each item gets, per machine atom and
    per human atom, its best partner on the other side and the score (`machine_best`,
    `human_best`), so a sample of the matcher's calls can be checked by hand."""
    per_item = []
    for it in items:
        M, H, top = it["machine"], it["human"], it.get("topic", "")
        mb = [_best(a, H, top) for a in M]
        hb = [_best(x, M, top) for x in H]
        km = sum(1 for _, s in mb if s >= threshold)
        kh = sum(1 for _, s in hb if s >= threshold)
        p, r = _rate(km, len(M)), _rate(kh, len(H))
        per_item.append({"item_id": it["item_id"], "model": it.get("model", ""),
                         "machine": len(M), "human": len(H),
                         "matched_machine": km, "matched_human": kh,
                         "precision": p, "recall": r, "f1": _f1(p, r),
                         "machine_best": [(j, round(s, 4)) for j, s in mb],
                         "human_best": [(j, round(s, 4)) for j, s in hb]})

    def summary(rows):
        m = sum(r["machine"] for r in rows)
        h = sum(r["human"] for r in rows)
        km = sum(r["matched_machine"] for r in rows)
        kh = sum(r["matched_human"] for r in rows)
        p, r_ = _rate(km, m), _rate(kh, h)
        macro = {}
        for key in ("precision", "recall", "f1"):
            vals = [r[key] for r in rows if r[key] is not None]
            macro[key] = round(sum(vals) / len(vals), 4) if vals else None
            macro["n_" + key] = len(vals)
        return {"items": len(rows), "machine_atoms": m, "human_atoms": h,
                "matched_machine": km, "matched_human": kh,
                "micro": {"precision": p, "recall": r_, "f1": _f1(p, r_)}, "macro": macro}

    models = sorted({r["model"] for r in per_item})
    return {"matcher": "containment (P0-10.1, many-to-one)", "threshold": threshold,
            "overall": summary(per_item),
            "by_model": {mo: summary([r for r in per_item if r["model"] == mo]) for mo in models},
            "items": per_item}


N_BOOT = 2000
BOOT_SEED = 20261009


def _micro(rows: Sequence[dict]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Micro P, R, F1 over per-item rows of either matcher: containment rows carry
    `matched_machine` and `matched_human`, one-to-one rows one `matched` for both."""
    m = sum(r["machine"] for r in rows)
    h = sum(r["human"] for r in rows)
    km = sum(r.get("matched_machine", r.get("matched", 0)) for r in rows)
    kh = sum(r.get("matched_human", r.get("matched", 0)) for r in rows)
    p = km / m if m else None
    r_ = kh / h if h else None
    f = (2 * p * r_ / (p + r_) if p + r_ else 0.0) if p is not None and r_ is not None else None
    return p, r_, f


def bootstrap_micro(per_item: Sequence[dict], n_boot: int = N_BOOT,
                    seed: int = BOOT_SEED) -> Dict[str, Optional[List[float]]]:
    """95% percentile intervals for micro precision, recall and F1, resampling items
    (a response and all its atoms together, since atoms within one response are not
    independent). Seeded, so the same packet gives the same interval."""
    rows = list(per_item)
    if len(rows) < 2:
        return {"precision": None, "recall": None, "f1": None}
    rng = random.Random(seed)
    draws = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        sample = [rng.choice(rows) for _ in rows]
        for key, v in zip(("precision", "recall", "f1"), _micro(sample)):
            if v is not None:
                draws[key].append(v)
    out = {}
    for key, vals in draws.items():
        if len(vals) < n_boot // 2:
            out[key] = None
            continue
        vals.sort()
        out[key] = [round(vals[int(0.025 * (len(vals) - 1))], 4),
                    round(vals[int(0.975 * (len(vals) - 1))], 4)]
    return out


def load_items(task: str, key_csv: str, filled_csv: str, bexec_root: str,
               variant: str = "filtered") -> Tuple[List[dict], List[str]]:
    """Join a filled hand-decomposition packet of `task` to the verifier output."""
    if task == "biographies":
        return load_bio_items(key_csv, filled_csv, bexec_root, variant)
    if task in ("summarization", "simplification"):
        return load_decompose_items(key_csv, filled_csv, bexec_root, variant)
    raise ValueError("task must be one of %s" % (TASKS,))


def load_decompose_items(key_csv: str, filled_csv: str, bexec_root: str,
                         variant: str = "filtered") -> Tuple[List[dict], List[str]]:
    """Join a filled Summarization or Simplification packet to the decomposer's output, by
    the sidecar's `input_row` (these tasks read their input by position, so the input file
    has no generation_id). `bexec_root` holds `inputs/` and `outputs/` of the v2 run."""
    key = {r["item_id"]: r for r in _read(key_csv)}
    filled = _read(filled_csv)
    if [r["item_id"] for r in filled] != list(key):
        raise JoinError("the filled packet and key.csv list different item_ids")
    items, notes, cache = [], [], {}
    for row in filled:
        k = key[row["item_id"]]
        model, unit = k["model"], k["unit_id"]
        if (model, unit) not in cache:
            meta = os.path.join(bexec_root, "inputs", model, unit + ".meta.csv")
            out = os.path.join(bexec_root, "outputs", model, unit + ".csv")
            for p in (meta, out):
                if not os.path.exists(p):
                    raise JoinError("%s: %s is missing" % (row["item_id"], p))
            rows = {r["generation_id"]: r for r in _read(meta)}
            cache[(model, unit)] = (rows, _read(out))
        rows, outs = cache[(model, unit)]
        m = rows.get(k["generation_id"])
        if m is None or str(m.get("sent", "")).lower() not in ("1", "true") or m.get("input_row", "") == "":
            raise JoinError("%s: %s was not sent to the decomposer in unit %s"
                            % (row["item_id"], k["generation_id"], unit))
        i = int(m["input_row"])
        if not 0 <= i < len(outs):
            raise JoinError("%s: input_row %d outside unit %s's %d output rows"
                            % (row["item_id"], i, unit, len(outs)))
        orow = outs[i]
        if (orow.get("response") or "").strip() != (row.get("response") or "").strip():
            notes.append("%s: the verifier's response text differs from the packet's" % row["item_id"])
        items.append({"item_id": row["item_id"], "generation_id": k["generation_id"],
                      "model": model, "topic": "",
                      "machine": machine_atoms_decompose(orow, variant),
                      "human": human_atoms(row.get("human_atoms"))})
    return items, notes


def load_bio_items(key_csv: str, filled_csv: str, bexec_root: str,
                   variant: str = "filtered") -> Tuple[List[dict], List[str]]:
    """Join a filled Biographies hand-decomposition packet to the verifier output. Returns (items, notes):
    notes list responses whose verifier text differs from the packet's (expected none).
    Raises JoinError when an item can't be found."""
    key = {r["item_id"]: r for r in _read(key_csv)}
    filled = _read(filled_csv)
    if [r["item_id"] for r in filled] != list(key):
        raise JoinError("the filled packet and key.csv list different item_ids")
    items, notes, cache = [], [], {}
    for row in filled:
        k = key[row["item_id"]]
        model, unit = k["model"], k["unit_id"]
        if (model, unit) not in cache:
            inp = os.path.join(bexec_root, "inputs", model, unit + ".csv")
            out = os.path.join(bexec_root, "outputs_filtered", model, unit + ".csv")
            for p in (inp, out):
                if not os.path.exists(p):
                    raise JoinError("%s: %s is missing" % (row["item_id"], p))
            names = {r["generation_id"]: r["name"] for r in _read(inp)}
            outs = {r["name"]: r for r in _read(out)}
            cache[(model, unit)] = (names, outs)
        names, outs = cache[(model, unit)]
        name = names.get(k["generation_id"])
        if name is None or name not in outs:
            raise JoinError("%s: %s not in unit %s's verifier output"
                            % (row["item_id"], k["generation_id"], unit))
        if name != row.get("topic"):
            raise JoinError("%s: packet topic %r, verifier row %r" % (row["item_id"], row.get("topic"), name))
        orow = outs[name]
        if (orow.get("response") or "").strip() != (row.get("response") or "").strip():
            notes.append("%s: the verifier's response text differs from the packet's" % row["item_id"])
        items.append({"item_id": row["item_id"], "generation_id": k["generation_id"],
                      "model": model, "topic": name,
                      "machine": machine_atoms_bio(orow, variant),
                      "human": human_atoms(row.get("human_atoms"))})
    return items, notes
