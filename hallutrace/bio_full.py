"""The full Biographies run: several work units per verify_biographies call.

Each call loads Gemma once, so units are grouped, scored in one call, and the
output is split back into the per-unit files verifier writes, unchanged from what
one call per unit would give. Inputs come from verifier_inputs.prepare.

A group holds one model and one sample_index, where every person appears once,
so the output splits back to units by `name`.

Files, under the same output root as verifier:

    groups/inputs/{model}/{group_id}.csv     what verify_biographies reads
    groups/outputs/{model}/{group_id}.csv    what it wrote, before the split
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence, Tuple

from hallutrace import verifier_inputs as bx

GROUP_DIR = "groups"


def group_units(units: Sequence[bx.WorkUnit], per_call: int) -> List[List[bx.WorkUnit]]:
    """Biographies units in groups of at most `per_call`, one (model, sample_index) each."""
    if per_call < 1:
        raise ValueError("per_call must be at least 1")
    by_key: Dict[Tuple[str, int], List[bx.WorkUnit]] = {}
    for u in units:
        if u.task != bx.BIOGRAPHIES:
            raise ValueError("%s is a %s unit, not biographies" % (u.unit_id, u.task))
        by_key.setdefault((u.model, u.sample_index), []).append(u)
    groups = []
    for key in sorted(by_key):
        us = sorted(by_key[key], key=lambda u: u.unit_id)
        groups += [us[i:i + per_call] for i in range(0, len(us), per_call)]
    return groups


def group_id(group: Sequence[bx.WorkUnit]) -> str:
    first, last = group[0].unit_id, group[-1].unit_id
    return first if first == last else "%s--%s" % (first, last)


def group_paths(out_root: str, group: Sequence[bx.WorkUnit]) -> Tuple[str, str]:
    model, gid = group[0].model, group_id(group)
    return (os.path.join(out_root, GROUP_DIR, "inputs", model, gid + ".csv"),
            os.path.join(out_root, GROUP_DIR, "outputs", model, gid + ".csv"))


def write_group_input(preps: Sequence[bx.Prepared], path: str) -> int:
    """Concatenate the units' verifier rows into one CSV. Returns the row count.

    Refuses a repeated name: the split back to units goes by name.
    """
    columns = bx.INPUT_COLUMNS[bx.BIOGRAPHIES]
    rows = [r for p in preps for r in p.rows]
    names = [r["name"] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError("a name repeats inside one group; units are not one (model, sample_index)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def split_output(group_out: str, preps: Sequence[bx.Prepared], out_root: str) -> Dict[str, int]:
    """Write each unit's output file from the group's output, by name.

    Columns are the group output's. A unit with nothing sent gets an empty file;
    `group_out=None` means nothing in the group was sent. Returns rows per unit.
    """
    columns, out_rows = [], []
    if group_out is None:
        if any(p.rows for p in preps):
            raise ValueError("group_out is None but some unit has rows to score")
    else:
        with open(group_out, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            out_rows = list(reader)
    owner = {r["name"]: p.unit.unit_id for p in preps for r in p.rows}
    unknown = sorted({r["name"] for r in out_rows} - set(owner))
    if unknown:
        raise ValueError("output names not in any unit's input: %s" % unknown[:5])
    written = {}
    for p in preps:
        path = bx.output_path(out_root, p.unit)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not p.rows:
            open(path, "w").close()
            written[p.unit.unit_id] = 0
            continue
        mine = [r for r in out_rows if owner[r["name"]] == p.unit.unit_id]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
            w.writeheader()
            w.writerows(mine)
        os.replace(tmp, path)
        written[p.unit.unit_id] = len(mine)
    return written
