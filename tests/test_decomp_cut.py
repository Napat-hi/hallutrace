"""Tests for hallutrace/decomp_cut.py. Synthetic generation batches, no network."""

import csv
import io
import json
import os
import sys
import tempfile

from hallutrace import bexec as bx
from hallutrace import decomp_cut as dc
from hallutrace.abstention import AbstentionDetector

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


DET = AbstentionDetector("fixed")

print("1. the text bexec sends (the tail rule only on a capped response)")
eq(dc.verifier_text({"response": "He was born in Rome.\n\nyes", "finish_reason": "stop"}, DET),
   "He was born in Rome.\n\nyes", "a response that stopped on its own goes through whole")
eq(dc.verifier_text({"response": "He was born in Rome. He studied at the Uni", "finish_reason": "length"}, DET),
   "He was born in Rome.", "a capped response loses its unfinished clause")
eq(dc.verifier_text({"response": "He studied at the Uni", "finish_reason": "length"}, DET), None,
   "a capped response emptied by the tail rule is left out")
eq(dc.verifier_text({"response": "", "finish_reason": "stop"}, DET), None, "an empty response is left out")
eq(dc.verifier_text({"response": "I'm sorry, I cannot answer that.", "finish_reason": "stop"}, DET), None,
   "an abstention is left out")
eq(dc.verifier_text({"response": "He was born in Rome. He studied", "finish_reason": None}, DET),
   "He was born in Rome. He studied", "no finish_reason is treated as finished, as in bexec")

print("2. candidates from A2 batches")
root = tempfile.mkdtemp()
MODELS = ["dclm-7b", "olmo17-7b", "olmo2-7b", "pythia-69b", "redpajama-7b"]
prompts = {}
for k in range(10):
    pid = "biographies-%05d" % k
    prompts[pid] = bx.Prompt(pid, "biographies", "Tell me a bio of P%d." % k, {"name": "Person %d" % k})
    sid = "summarization-%05d" % k
    prompts[sid] = bx.Prompt(sid, "summarization", "Summarize.", {"text": "passage"})
for m in MODELS:
    os.makedirs(os.path.join(root, m))
    with open(os.path.join(root, m, "%s-b00000.jsonl" % m), "w", encoding="utf-8") as f:
        for k in range(10):
            pid = "biographies-%05d" % k
            resp = "I'm sorry, I cannot answer that." if k == 9 else "Person %d was a poet. Born in" % k
            f.write(json.dumps({"generation_id": "%s|%s|0" % (m, pid), "prompt_id": pid, "model": m,
                                "sample_index": 0, "response": resp,
                                "finish_reason": "stop" if k % 2 else "length"}) + "\n")
    with open(os.path.join(root, m, "%s-b00001.jsonl" % m), "w", encoding="utf-8") as f:
        for k in range(10):
            sid = "summarization-%05d" % k
            f.write(json.dumps({"generation_id": "%s|%s|1" % (m, sid), "prompt_id": sid, "model": m,
                                "sample_index": 1, "response": "A summary %d." % k,
                                "finish_reason": "stop"}) + "\n")
gens = dc.candidates(root, prompts, "biographies", DET)
eq(len(gens), 45, "9 of 10 per model: the abstention is left out")
g = next(x for x in gens if x["generation_id"] == "pythia-69b|biographies-00003|0")
eq((g["topic"], g["response"]), ("Person 3", "Person 3 was a poet. Born in"),
   "a stop response keeps its last line; the topic is the person")
g = next(x for x in gens if x["generation_id"] == "pythia-69b|biographies-00004|0")
eq(g["response"], "Person 4 was a poet.", "a capped one loses it")
eq(g["unit_id"], "pythia-69b-b00000", "the bexec unit is recorded, for joining machine atoms later")
summ = dc.candidates(root, prompts, "summarization", DET)
eq((len(summ), summ[0]["topic"]), (50, ""), "summarization drawn from its own batches, no topic")
try:
    dc.candidates(root, prompts, "historical_events", DET)
    check(False, "historical_events is refused")
except ValueError:
    check(True, "historical_events is refused (its verifier never splits a response)")

print("3. the draw and the files")
picked = dc.draw(gens, "20261001", "biographies")
eq(len(picked), 30, "30 generations")
eq(sorted(sum(p["model"] == m for p in picked) for m in MODELS), [6] * 5, "6 per model")
eq([p["generation_id"] for p in dc.draw(gens, "20261001", "biographies")],
   [p["generation_id"] for p in picked], "same seed, same draw")
check([p["generation_id"] for p in dc.draw(gens, "20261001", "summarization")]
      != [p["generation_id"] for p in picked], "the task is part of the seed")
files = dc.packet_files(picked, "20261001", "biographies", len(gens))
ann = list(csv.DictReader(io.StringIO(files["annotator.csv"])))
key = list(csv.DictReader(io.StringIO(files["key.csv"])))
eq(tuple(ann[0]), dc.ANNOTATOR_COLUMNS, "annotator columns")
eq(tuple(key[0]), dc.KEY_COLUMNS, "key columns")
eq([r["item_id"] for r in ann], [r["item_id"] for r in key], "annotator and key rows align")
check(all(r["human_atoms"] == "" for r in ann), "human_atoms left empty")
m = json.loads(files["manifest.json"])
eq((m["pool"], m["drawn"]), (45, 30), "manifest counts")
check("Name the person" in m["instruction"], "the Biographies instruction travels with the packet")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
