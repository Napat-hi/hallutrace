"""Turn generation batches into verifier inputs, one work unit at a time.

No GPU, key or network. A work unit is one task's part of one generation batch
(a batch of 64 holds one model and one sample_index, and sometimes the end of
one task and the start of the next). A unit is done when its output file
exists, so pending() never repeats work.

Before verification each response goes through outcomes.classify. Empty
responses are not sent. Abstentions are sent, since some carry claims before
the abstention phrase, and keep `outcome = abstention` in the sidecar. A
response that hit the token limit (`finish_reason == "length"`) loses its
unfinished tail (outcomes.strip_incomplete_tail); one stripped to nothing is
not sent and is recorded as `no_units_found`.

Summarization and Simplification read their input by position, so their CSV
cannot carry generation_id; the sidecar's input_row joins their output back.

Files, relative to one output root:

    inputs/{model}/{unit_id}.csv        what the verifier reads
    inputs/{model}/{unit_id}.meta.csv   one row per generation, sent or not
    outputs/{model}/{unit_id}.csv       what the verifier wrote
"""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from hallutrace import outcomes as oc

BIOGRAPHIES = "biographies"
EMPTIED = "emptied"  # outcome for a response the tail rule strips to nothing

# Columns of each verifier input; generation_id is ours, for the join back.
INPUT_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "biographies": ("name", "response", "model", "sample_index", "generation_id"),
    "false_presuppositions": ("response", "list", "list_condition", "generation_id"),
    "historical_events": ("response", "name1", "name2", "generation_id"),
    "rationalization_binary": ("response", "category", "prompt", "generation_id"),
    "rationalization_numerical": ("response", "list", "list_condition", "count", "generation_id"),
    "scientific_attribution": ("response", "generation_id"),
    # read by position: exactly two columns, source text first
    "simplification": ("prompt", "response"),
    "summarization": ("prompt", "response"),
}
POSITIONAL = ("simplification", "summarization")

META_COLUMNS = (
    "generation_id", "prompt_id", "model", "sample_index", "task", "batch_id",
    "outcome", "decomposer_empty_reason", "sent", "input_row", "stripped_tail",
)


@dataclass
class Prompt:
    prompt_id: str
    task: str
    prompt: str
    meta: dict


@dataclass
class WorkUnit:
    unit_id: str
    task: str
    model: str
    sample_index: int
    batch_paths: List[str] = field(default_factory=list)


@dataclass
class Prepared:
    unit: WorkUnit
    columns: Tuple[str, ...]
    rows: List[dict]
    meta: List[dict]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for m in self.meta:
            out[m["outcome"]] = out.get(m["outcome"], 0) + 1
        return out


def task_of(prompt_id: str) -> str:
    return prompt_id.rsplit("-", 1)[0]


def load_prompts(path: str) -> Dict[str, Prompt]:
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r["prompt_id"]] = Prompt(
                r["prompt_id"], r["task"], r["prompt"], json.loads(r["task_meta"] or "{}")
            )
    return out


def read_batch(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _batch_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _first_record(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError("%s is empty" % path)


def output_path(out_root: str, unit: WorkUnit) -> str:
    return os.path.join(out_root, "outputs", unit.model, unit.unit_id + ".csv")


def discover(gen_root: str, plan_dir: str) -> List[WorkUnit]:
    """Every work unit whose batch is on disk, done or not.

    A batch holding two tasks becomes two units. A batch whose records name
    another model than its directory raises. `plan_dir` is not read.
    """
    units: List[WorkUnit] = []
    for model_dir in sorted(glob.glob(os.path.join(gen_root, "*"))):
        if not os.path.isdir(model_dir):
            continue
        model = os.path.basename(model_dir)
        for path in sorted(glob.glob(os.path.join(model_dir, "*.jsonl"))):
            recs = read_batch(path)
            if not recs:
                continue
            models = {r["model"] for r in recs}
            if models != {model}:
                raise ValueError("%s holds model(s) %s but sits under %r"
                                 % (path, sorted(models), model))
            keys = sorted({(task_of(r["prompt_id"]), r["sample_index"]) for r in recs},
                          key=lambda k: (k[1], k[0]))
            sis = {si for _, si in keys}
            bid = _batch_id(path)
            for task, si in keys:
                uid = bid
                if len(keys) > 1:
                    uid = "%s-%s" % (bid, task)
                    if len(sis) > 1:
                        uid += "-s%d" % si
                units.append(WorkUnit(uid, task, model, si, [path]))
    return units


def supersede(out_root: str, units: Iterable[WorkUnit], tag: str) -> List[str]:
    """Move these units' output and inputs to superseded/{tag}/, so they run again.

    Nothing is deleted, so old and new verdicts can be compared. A unit with no
    output is left alone. Files already under the tag are not overwritten; the
    next copy goes to superseded/{tag}.2/, then .3. Returns the moved paths.
    """
    moved = []
    for u in units:
        out = output_path(out_root, u)
        if not os.path.exists(out):
            continue
        paths = [p for p in (out,
                             os.path.join(out_root, "inputs", u.model, u.unit_id + ".csv"),
                             os.path.join(out_root, "inputs", u.model, u.unit_id + ".meta.csv"))
                 if os.path.exists(p)]
        rels = [os.path.relpath(p, out_root) for p in paths]
        folder, n = tag, 1
        while any(os.path.exists(os.path.join(out_root, "superseded", folder, r)) for r in rels):
            n += 1
            folder = "%s.%d" % (tag, n)
        for path, rel in zip(paths, rels):
            dst = os.path.join(out_root, "superseded", folder, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(path, dst)
            moved.append(os.path.join("superseded", folder, rel))
    return moved


def pending(gen_root: str, out_root: str, plan_dir: str) -> List[WorkUnit]:
    """Units that are ready and have no output yet. Safe to call repeatedly."""
    return [u for u in discover(gen_root, plan_dir)
            if not os.path.exists(output_path(out_root, u))]


def done_per_task(gen_root: str, out_root: str, plan_dir: str,
                  by_model: bool = False) -> Dict:
    """How many units of each task already have an output.

    Keyed by task, or by (task, model) with by_model=True.
    """
    out: Dict = {}
    for u in discover(gen_root, plan_dir):
        if os.path.exists(output_path(out_root, u)):
            k = (u.task, u.model) if by_model else u.task
            out[k] = out.get(k, 0) + 1
    return out


def first_per_task(units: List[WorkUnit], tasks: Iterable[str] = (),
                   per_task: int = 0,
                   done: Optional[Dict[str, int]] = None,
                   done_by_model: Optional[Dict[Tuple[str, str], int]] = None) -> List[WorkUnit]:
    """Order units so every task gets a verdict before any task gets a second.

    Within a task, whole batches come before shared ones and models take turns.
    `tasks` limits which tasks run. `per_task` caps the verified units per task
    (0: no cap), counting those in `done`. `done_by_model` puts models that
    already have verdicts at the back of their turn.
    """
    wanted = set(tasks)
    done = done or {}
    queues: Dict[str, List[WorkUnit]] = {}
    shared = lambda u: u.unit_id.endswith("-" + u.task)  # noqa: E731
    turn: Dict[Tuple[str, str, bool], int] = {}
    ranked = []
    for i, u in enumerate(sorted(units, key=shared)):
        k = (u.task, u.model, shared(u))
        if k not in turn:
            turn[k] = (done_by_model or {}).get((u.task, u.model), 0)
        ranked.append(((shared(u), turn[k], u.model, i), u))
        turn[k] = turn.get(k, 0) + 1
    for _, u in sorted(ranked, key=lambda t: t[0]):
        if wanted and u.task not in wanted:
            continue
        q = queues.setdefault(u.task, [])
        if not per_task or len(q) + done.get(u.task, 0) < per_task:
            q.append(u)
    order: List[WorkUnit] = []
    depth = max((len(q) for q in queues.values()), default=0)
    for i in range(depth):
        for task in sorted(queues):
            if i < len(queues[task]):
                order.append(queues[task][i])
    return order


def _input_row(task: str, p: Prompt, rec: dict, text: str) -> dict:
    m = p.meta
    gid = rec["generation_id"]
    if task == "biographies":
        return {"name": m["name"], "response": text, "model": rec["model"],
                "sample_index": rec["sample_index"], "generation_id": gid}
    if task in ("false_presuppositions",):
        return {"response": text, "list": m["list"], "list_condition": m["list_condition"],
                "generation_id": gid}
    if task == "rationalization_numerical":
        return {"response": text, "list": m["list"], "list_condition": m["list_condition"],
                "count": m["count"], "generation_id": gid}
    if task == "historical_events":
        return {"response": text, "name1": m["name1"], "name2": m["name2"], "generation_id": gid}
    if task == "rationalization_binary":
        return {"response": text, "category": m["category"], "prompt": p.prompt,
                "generation_id": gid}
    if task == "scientific_attribution":
        return {"response": text, "generation_id": gid}
    if task in POSITIONAL:
        # column 0 is the source text, without the instruction around it
        return {"prompt": m["text"], "response": text}
    raise KeyError("no input adapter for task %r" % task)


def prepare(unit: WorkUnit, prompts: Dict[str, Prompt], detector=None,
            strip: bool = True, send_abstained: bool = True) -> Prepared:
    """Build the verifier input and the per-generation sidecar for one unit.

    `detector` is an AbstentionDetector; None skips the abstention check.
    `strip=False` keeps unfinished tails. `send_abstained=False` does not send
    abstentions.
    """
    task = unit.task
    rows: List[dict] = []
    meta: List[dict] = []
    for path in unit.batch_paths:
        bid = _batch_id(path)
        for rec in read_batch(path):
            # a batch can hold two tasks
            if task_of(rec["prompt_id"]) != task or rec["sample_index"] != unit.sample_index:
                continue
            p = prompts[rec["prompt_id"]]
            if p.task != task:
                raise ValueError("%s is a %s prompt in prompts.csv but its id says %s"
                                 % (rec["prompt_id"], p.task, task))
            response = rec.get("response")
            outcome, reason = oc.classify(response, detector)
            if (outcome == oc.OK and task == "false_presuppositions"
                    and oc.says_no_response(response)):
                outcome, reason = oc.ABSTENTION, oc.ABSTAINED
            if outcome == oc.ABSTENTION and send_abstained:
                reason = None  # sent; the decomposer decides if it has units
            sendable = outcome == oc.OK or (outcome == oc.ABSTENTION and send_abstained)
            kept, dropped = response, ""
            # only a response cut by the token limit is stripped
            if sendable and strip and rec.get("finish_reason") == "length":
                kept, dropped = oc.strip_incomplete_tail(response)
                if not kept.strip():
                    # an abstention keeps its outcome
                    if outcome != oc.ABSTENTION:
                        outcome = EMPTIED
                    reason = oc.NO_UNITS_FOUND
                    sendable = False
            sent = sendable
            meta.append({
                "generation_id": rec["generation_id"], "prompt_id": rec["prompt_id"],
                "model": rec["model"], "sample_index": rec["sample_index"], "task": task,
                "batch_id": bid, "outcome": outcome,
                "decomposer_empty_reason": reason or "",
                "sent": int(sent), "input_row": len(rows) if sent else -1,
                "stripped_tail": dropped,
            })
            if sent:
                rows.append(_input_row(task, p, rec, kept))
    if not meta:
        raise ValueError("unit %s holds no %s records at sample_index %d"
                         % (unit.unit_id, task, unit.sample_index))
    return Prepared(unit, INPUT_COLUMNS[task], rows, meta)


def write_inputs(prep: Prepared, out_root: str) -> Tuple[str, str]:
    """Write the verifier CSV and its sidecar. Returns both paths."""
    d = os.path.join(out_root, "inputs", prep.unit.model)
    os.makedirs(d, exist_ok=True)
    inp = os.path.join(d, prep.unit.unit_id + ".csv")
    side = os.path.join(d, prep.unit.unit_id + ".meta.csv")
    with open(inp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prep.columns), lineterminator="\n")
        w.writeheader()
        w.writerows(prep.rows)
    with open(side, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(META_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(prep.meta)
    return inp, side
