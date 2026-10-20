"""Filters applied after verification to what the verifier wrote.

The verifier's output stays as it was (`outputs/`). This module reads each row
and writes the same row with its unit columns filtered (`outputs_filtered/`),
the same layout as Biographies, so h can be reported raw and filtered from one
run, and undoing a filter is a recompute with no new call.

What is dropped:

    not_in_response   an extracted unit the model never wrote, by the rules in
                      `unit_guard` (number words count, titles at 80%)
    none_token        a unit that is just NONE, which the v2 prompts can tack
                      onto a list (`2, NONE`)
    framing           a decomposer line about its own output ("Here are the
                      facts...", "Note that..."), summarization and
                      simplification only
    blank             a decomposer line with no atom text (`  | False |`),
                      which the parser would otherwise lose silently

Verdicts are recomputed with HALoGEN's own rule for each task, on the units
that remain. Senator names are repaired for parser defect 9 first: the
released `strip(",. </s>")` cut characters off the ends of a name, so the
name is found again in the extractor's reply and cleaned the fixed way.
Without that, `Max Baucu` would not match `Max Baucus` in the response and
would be dropped as invented. (Defect 9 does not change the senator verdict
itself: rationalization_binary counts every named senator as hallucinated.)

`all_units_dropped` marks a row that had units before and has none after.
`zero_unit_rule` says how a row with no units enters h, per task: kept in as
not hallucinated for false_presuppositions and rationalization_binary, out of
`n_usable` otherwise, except a rationalization_numerical answer of 0, which is
scored against the gold count. `stability.py` applies it; this only labels the
row.

historical_events extracts nothing and Biographies has its own filter
(`bio_filter.py`), so both pass through untouched.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Dict, List, Optional, Tuple

from hallutrace import extraction as ex
from hallutrace import unit_guard as ug

NOT_IN_RESPONSE = "not_in_response"
NONE_TOKEN = "none_token"
FRAMING = "framing"
BLANK = "blank"  # a decomposer line `  | False |` with no atom text

# How a row left with no units enters h, per task. Written to `zero_unit_rule`;
# empty when the row has units.
NOT_HALLUCINATED = "not_hallucinated"  # FP, RB: no units can be the right answer
OUT_OF_USABLE = "out_of_n_usable"      # SA, summ/simp, RN: no units is no response
ANSWER_0_RIGHT = "answer_0_right"      # RN whose first line says 0, gold count 0
ANSWER_0_WRONG = "answer_0_wrong"      # RN whose first line says 0, gold count not 0


def parse_list(text) -> list:
    """A list or set as the verifier wrote it into its CSV, or []."""
    if text is None:
        return []
    s = str(text).strip()
    if not s or s in ("None", "nan", "set()"):
        return []
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return list(v) if isinstance(v, (list, tuple, set)) else [v]


def is_none(unit) -> bool:
    return str(unit).strip().strip(".").upper() == "NONE"


def task_of_columns(cols) -> Optional[str]:
    """Which task wrote a CSV, from its header alone."""
    c = set(cols)
    if "category" in c and "yesno" in c:
        return "rationalization_binary"
    if "count" in c and "list_condition" in c:
        return "rationalization_numerical"
    if "list_condition" in c:
        return "false_presuppositions"
    if "s2_titles" in c:
        return "scientific_attribution"
    if "passage" in c:
        return "decompose"
    return None  # historical_events, biographies, anything else: not ours to filter


_STRIPPED = ",. </s>"
# What the repair may grow a name over: the stripped characters except the
# space, so it never reaches into the next or previous word ("is Max Baucus"
# must not become "s Max Baucus").
_REGROW = _STRIPPED.replace(" ", "")


def repair_senator(name: str, reply: str) -> str:
    """Undo defect 9 on one name by finding it again in the extractor's reply."""
    best = name
    for m in re.finditer(re.escape(name), str(reply)):
        a, b = m.start(), m.end()
        while a > 0 and reply[a - 1] in _REGROW:
            a -= 1
        while b < len(reply) and reply[b] in _REGROW:
            b += 1
        fixed = ex.clean_senator_name(reply[a:b])
        if len(fixed) > len(best):
            best = fixed
    return best


def _keep(units: List[Tuple[object, str]], response: str):
    kept, dropped = [], []
    for u, kind in units:
        if is_none(u):
            dropped.append({"unit": str(u), "reason": NONE_TOKEN})
        elif ug.in_response(u, kind, response):
            kept.append(u)
        else:
            dropped.append({"unit": u if isinstance(u, (int, str)) else list(u),
                            "reason": NOT_IN_RESPONSE})
    return kept, dropped


def _atom_lines(text: str) -> List[Tuple[str, bool]]:
    """(atom, supported) per decomposer line, " atom | True |". The verdict is the last field:
    an atom can itself hold a "|" (code, wiki tables, "talk | contribs"), and reading the second
    field would take part of the atom as the verdict."""
    out = []
    for line in str(text or "").split("\n"):
        if len(line.split("|")) < 2 or not line.split("|")[0].strip():
            continue  # no verdict, or a blank atom (filter_row drops it as `blank`)
        body = line.strip()
        if body.endswith("|"):
            body = body[:-1]
        atom, _, verdict = body.rpartition("|")
        if not atom.strip():  # " atom |" with no verdict field: the atom, not supported
            atom, verdict = body, ""
        out.append((atom.strip(), verdict.strip() == "True"))
    return out


def _fmt_atoms(atoms: List[Tuple[str, bool]]) -> str:
    return "\n".join(" %s | %s |" % (a, v) for a, v in atoms)


def unit_count(task: str, row: Dict[str, str]) -> int:
    """Units in a row as the scorers count them: yes and no are not units."""
    if task == "decompose":
        return len(_atom_lines(row.get("atomic_units")))
    return len([u for u in parse_list(row.get("atomic_units")) if u not in ("yes", "no")])


def says_zero(response: str) -> bool:
    """HALoGEN's `parse_zero_response` (rationalization_scorer_numerical.py L94),
    on the first non-empty line rather than the first line, since our responses
    often open with a line break."""
    line = next((l for l in str(response or "").split("\n") if l.strip()), "")
    return "0" in re.findall(r"\d+", line)


def zero_unit_rule(task: str, row: Dict[str, str]) -> str:
    """How this row enters h if it has no units, or "" if it has some."""
    if unit_count(task, row):
        return ""
    if task in ("false_presuppositions", "rationalization_binary"):
        return NOT_HALLUCINATED
    if task == "rationalization_numerical" and says_zero(row.get("response")):
        gold = str(row.get("count", "")).strip()
        gold = gold[:-2] if gold.endswith(".0") else gold
        return ANSWER_0_RIGHT if gold == "0" else ANSWER_0_WRONG
    return OUT_OF_USABLE


def filter_row(task: str, row: Dict[str, str]) -> Dict[str, str]:
    """The row with filtered unit columns, plus dropped_units and all_units_dropped."""
    out = dict(row)
    resp = row.get("response", "")
    raw_units = parse_list(row.get("atomic_units"))
    dropped: list = []

    if task == "false_presuppositions":
        units = [] if "no response" in str(resp).lower() else \
            ex.parse_false_presuppositions(row.get("parsed_response") or "")
        kept, dropped = _keep([(u, "item") for u in units], resp)
        truth = {str(x).lower().strip() for x in parse_list(row.get("list_condition"))}
        out["atomic_units"] = repr(kept)
        out["hallucinated_atomic_units"] = repr(sorted({str(u).lower().strip() for u in kept} - truth))

    elif task == "rationalization_numerical":
        num, ents = ex.parse_rationalization_numerical(row.get("parsed_response") or "")
        cand = ([(num, "count")] if num is not None else []) + [(e, "item") for e in (ents or [])]
        kept, dropped = _keep(cand, resp)
        truth = {str(x).lower() for x in [row.get("count")] + parse_list(row.get("list_condition"))}
        truth = {t[:-2] if t.endswith(".0") else t for t in truth}
        out["atomic_units"] = repr(kept)
        out["hallucinated_atomic_units"] = repr(sorted({str(u).lower() for u in kept} - truth))

    elif task == "rationalization_binary":
        reply, cat = row.get("parsed_response") or "", row.get("category")
        yesno = [u for u in raw_units if u in ("yes", "no")]
        rest = [u for u in raw_units if u not in ("yes", "no")]
        if cat == "senator":
            cand = [(repair_senator(str(u), reply), "name") for u in rest]
        elif cat == "graph":
            cand = [(tuple(u), "flight") for u in rest]
        else:
            cand = [(u, "factor") for u in rest]
        kept, dropped = _keep(cand, resp)
        kept = [list(u) if isinstance(u, tuple) else u for u in kept]
        out["atomic_units"] = repr(yesno + kept)
        out["hallucinated_atomic_units"] = repr(kept)

    elif task == "scientific_attribution":
        kept, dropped = _keep([(t, "title") for t in raw_units], resp)
        keep = set(kept)
        out["atomic_units"] = repr(kept)
        for col in ("hallucinated_atomic_units", "unverifiable_atomic_units"):
            out[col] = repr([t for t in parse_list(row.get(col)) if t in keep])

    elif task == "decompose":
        atoms = _atom_lines(row.get("atomic_units"))
        raw_units = atoms
        kept = []
        for line in str(row.get("atomic_units") or "").split("\n"):
            parts = line.split("|")
            if len(parts) >= 2 and not parts[0].strip():
                dropped.append({"unit": line.strip(), "reason": BLANK})
        for a, v in atoms:
            if ug.framing_line(a, resp):
                dropped.append({"unit": a, "reason": FRAMING})
            else:
                kept.append((a, v))
        out["atomic_units"] = _fmt_atoms(kept)
        out["hallucinated_atomic_units"] = _fmt_atoms([(a, v) for a, v in kept if not v])

    else:
        raise ValueError("no filter for task %r" % task)

    out["dropped_units"] = json.dumps(dropped, ensure_ascii=False)
    out["all_units_dropped"] = str(bool(raw_units) and not kept)
    out["zero_unit_rule"] = zero_unit_rule(task, out)
    return out
