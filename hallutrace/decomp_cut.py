"""Draw the generations an annotator decomposes by hand, to check the decomposer.

An annotator breaks 30 generations per task into atomic facts without seeing the
machine's atoms; the two lists are then matched and scored by precision, recall and F1.
Tasks: Biographies, Summarization and Simplification.

The draw reads the generations directly and applies bexec's text rule, so the text is
what the verifier gets. Only `ok` responses are drawn (abstentions, empty and emptied
responses are not counted in h). The machine's atoms are joined to key.csv by
generation_id once the verifier output exists.

Files, under decomposition/{task}/:

    annotator.csv   item_id, topic (the person, for Biographies), response,
                    human_atoms (one per line), notes
    key.csv         item_id, generation_id, prompt_id, model, sample_index, unit_id
    manifest.json   task, seed, pool, counts per model, the instruction
"""
from __future__ import annotations

import csv
import io
import json
import random
from typing import Dict, List, Optional

from hallutrace import bexec as bx
from hallutrace import outcomes as oc

TASKS = ("biographies", "summarization", "simplification")
_COMMON = (
    "An atomic fact is a short sentence that states one piece of information. Cover every "
    "sentence of the response, including claims that look wrong or off topic; skip only text "
    "that states nothing (a question, a greeting, a fragment). Do not check whether a fact is "
    "true or supported: this is about what the response says, not whether it is right."
)
INSTRUCTIONS = {
    "biographies": "Break the response into atomic facts, one per line in human_atoms. Name the "
                   "person (topic) instead of writing 'he' or 'she'. " + _COMMON,
    "summarization": "Break the summary into atomic facts, one per line in human_atoms. " + _COMMON,
    "simplification": "Break the simplified text into atomic facts, one per line in human_atoms. "
                      + _COMMON,
}
ANNOTATOR_COLUMNS = ("item_id", "topic", "response", "human_atoms", "notes")
KEY_COLUMNS = ("item_id", "generation_id", "prompt_id", "model", "sample_index", "unit_id")


def verifier_text(rec: dict, detector=None) -> Optional[str]:
    """The text bexec sends the verifier, or None for a response not counted in h."""
    response = rec.get("response")
    outcome, _ = oc.classify(response, detector)
    if outcome != oc.OK:
        return None
    if rec.get("finish_reason") == "length":
        kept, _ = oc.strip_incomplete_tail(response)
        return kept if kept.strip() else None
    return response


def candidates(gen_root: str, prompts: Dict[str, bx.Prompt], task: str,
               detector=None) -> List[dict]:
    """Every generation of `task` that would reach the verifier, with its text."""
    if task not in TASKS:
        raise ValueError("Step 8 covers %s, not %r" % (", ".join(TASKS), task))
    out = []
    for unit in bx.discover(gen_root, ""):
        if unit.task != task:
            continue
        for path in unit.batch_paths:
            for rec in bx.read_batch(path):
                if bx.task_of(rec["prompt_id"]) != task or rec["sample_index"] != unit.sample_index:
                    continue
                text = verifier_text(rec, detector)
                if text is None:
                    continue
                meta = prompts[rec["prompt_id"]].meta
                out.append({"generation_id": rec["generation_id"], "prompt_id": rec["prompt_id"],
                            "model": rec["model"], "sample_index": rec["sample_index"],
                            "unit_id": unit.unit_id, "response": text,
                            "topic": meta.get("name", "") if task == "biographies" else ""})
    return out


def draw(gens: List[dict], seed, task: str, n: int = 30) -> List[dict]:
    """n generations split evenly across models (any remainder to random models), shuffled."""
    rng = random.Random("%s:%s:decomposition" % (seed, task))
    by_model: Dict[str, List[dict]] = {}
    for g in sorted(gens, key=lambda g: g["generation_id"]):
        by_model.setdefault(g["model"], []).append(g)
    models = sorted(by_model)
    quota = {m: n // len(models) for m in models}
    for m in rng.sample(models, n % len(models)):
        quota[m] += 1
    picked = []
    for m in models:
        if len(by_model[m]) < quota[m]:
            print("warning: %s has %d generations, fewer than its quota of %d"
                  % (m, len(by_model[m]), quota[m]))
        picked += rng.sample(by_model[m], min(quota[m], len(by_model[m])))
    rng.shuffle(picked)
    return picked


def _table(cols, rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(cols), lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def packet_files(picked: List[dict], seed, task: str, pool: int) -> Dict[str, str]:
    """The three files as text, keyed by name."""
    ann, key = [], []
    for i, g in enumerate(picked, 1):
        item_id = "%s-d%02d" % (task, i)
        ann.append({"item_id": item_id, "topic": g["topic"], "response": g["response"],
                    "human_atoms": "", "notes": ""})
        key.append(dict(g, item_id=item_id))
    by_model: Dict[str, int] = {}
    for g in picked:
        by_model[g["model"]] = by_model.get(g["model"], 0) + 1
    manifest = {"task": task, "step": "8 (decomposition, E3)", "seed": str(seed),
                "source": "A2 generations, B_Exec's text rule (option A on length only, #36); "
                          "ok responses only (D-023)",
                "pool": pool, "drawn": len(picked), "by_model": dict(sorted(by_model.items())),
                "words": sum(len(g["response"].split()) for g in picked),
                "instruction": INSTRUCTIONS[task]}
    return {"annotator.csv": _table(ANNOTATOR_COLUMNS, ann), "key.csv": _table(KEY_COLUMNS, key),
            "manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False)}
