"""Verify again only the Biographies responses a tail-rule change touched.

The first full run stripped the unfinished tail of every response; the rule now strips
only responses that hit the token limit. This module re-scores only the responses whose
sent text changed (663 in 115 units) and merges them back into each unit's output.

Per unit:

    changed     generation_ids whose sent text differs between the old sidecar and
                `verifier_inputs.prepare` under the new rule (sent flag or stripped tail)
    subset      the new Prepared, cut to those rows: what goes to Gemma
    merge       old output rows minus the changed names, plus the new rows, in the
                new input order

Abstentions are not sent (they are not counted in h). The old files are kept under
superseded/option_a_35/. Files for the redo, under the output root:

    redo_option_a_35/groups/inputs|outputs/{model}/{group}.csv   verify_biographies I/O
    redo_option_a_35/outputs/{model}/{unit}.csv                  the changed rows' verdicts
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence, Set

from hallutrace import verifier_inputs as bx

TAG = "option_a_35"
REDO_DIR = "redo_" + TAG


def read_csv(path: str) -> List[dict]:
    if not os.path.exists(path) or not os.path.getsize(path):
        return []
    csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sidecar_path(out_root: str, unit: bx.WorkUnit) -> str:
    return os.path.join(out_root, "inputs", unit.model, unit.unit_id + ".meta.csv")


def changed(new: bx.Prepared, old_meta: Sequence[dict]) -> Set[str]:
    """generation_ids sent now whose text differs from what the old run sent.

    A row sent before and not now should not happen (the new rule strips less) and
    raises.
    """
    old = {m["generation_id"]: m for m in old_meta}
    out = set()
    for m in new.meta:
        o = old.get(m["generation_id"])
        if o is None:
            raise ValueError("%s is not in the old sidecar of %s" % (m["generation_id"], new.unit.unit_id))
        was_sent, now_sent = str(o["sent"]) == "1", bool(int(m["sent"]))
        if was_sent and not now_sent:
            raise ValueError("%s was sent before and is not now" % m["generation_id"])
        if now_sent and (not was_sent or (o.get("stripped_tail") or "") != (m["stripped_tail"] or "")):
            out.add(m["generation_id"])
    return out


def subset(prep: bx.Prepared, gids: Set[str]) -> bx.Prepared:
    """The Prepared cut to these generations: only their verifier rows are sent."""
    rows = [r for r in prep.rows if r["generation_id"] in gids]
    meta = [m for m in prep.meta if m["generation_id"] in gids]
    return bx.Prepared(prep.unit, prep.columns, rows, meta)


def merge(old_rows: Sequence[dict], new_rows: Sequence[dict], prep: bx.Prepared,
          gids: Set[str]) -> List[dict]:
    """The unit's output after the redo, in the new input order, keyed by name.

    Every sent row of `prep` must end with exactly one output row: the redo's for a
    changed generation, the old run's otherwise.
    """
    redo_names = {r["name"] for r in prep.rows if r["generation_id"] in gids}
    by_name: Dict[str, dict] = {r["name"]: r for r in old_rows if r["name"] not in redo_names}
    for r in new_rows:
        if r["name"] not in redo_names:
            raise ValueError("redo output has %r, which was not re-sent" % r["name"])
        by_name[r["name"]] = r
    order = [r["name"] for r in prep.rows]
    missing = [n for n in order if n not in by_name]
    if missing:
        raise ValueError("%s: no output row for %s" % (prep.unit.unit_id, missing[:5]))
    extra = sorted(set(by_name) - set(order))
    if extra:
        raise ValueError("%s: output rows for names no longer sent: %s" % (prep.unit.unit_id, extra[:5]))
    return [by_name[n] for n in order]


def write_rows(path: str, rows: Sequence[dict], columns: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
