"""Tests for hallutrace/bexec.py. No key, no network, no GPU, no pandas."""

import csv
import glob
import json
import os
import sys
import tempfile

from hallutrace import bexec as bx
from hallutrace import outcomes as oc
from hallutrace.abstention import AbstentionDetector

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


def raises(fn, exc, label):
    try:
        fn()
    except exc:
        check(True, label)
        return
    except Exception as e:  # noqa: BLE001
        check(False, "%s  (raised %r)" % (label, e))
        return
    check(False, "%s  (did not raise)" % label)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_CSV = os.path.join(ROOT, "data", "prompts.csv")
PLAN_DIR = os.path.join(ROOT, "data", "generation_plan")
PILOT2 = os.path.join(ROOT, "results", "pilot2")
prompts = bx.load_prompts(PROMPTS_CSV)
det = AbstentionDetector("fixed")


def first(task):
    return next(p for p in prompts.values() if p.task == task)


def rec(model, prompt_id, si, response, batch_id, finish_reason="length"):
    # "length" by default, so the tail-rule tests below see a response cut by the limit.
    return {"generation_id": "%s|%s|%d" % (model, prompt_id, si), "prompt_id": prompt_id,
            "model": model, "sample_index": si, "response": response, "batch_id": batch_id,
            "finish_reason": finish_reason}


def write_batch(root, model, batch_id, recs):
    d = os.path.join(root, model)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, batch_id + ".jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def write_plan(plan_dir, model, rows):
    os.makedirs(plan_dir, exist_ok=True)
    with open(os.path.join(plan_dir, model + ".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "sample_index", "batch_id"])
        w.writeheader()
        w.writerows(rows)


print("1. prompts and the task adapters")
eq(len(prompts), 3200, "prompts.csv holds 3,200 prompts")
eq(sorted({p.task for p in prompts.values()}), sorted(bx.INPUT_COLUMNS),
   "every task in prompts.csv has an input adapter, and no adapter is spare")
eq(bx.task_of("rationalization_binary-00042"), "rationalization_binary",
   "task_of keeps underscores in the task name")
for task in sorted(bx.INPUT_COLUMNS):
    p = first(task)
    row = bx._input_row(task, p, rec("m", p.prompt_id, 1, "Text.", "m-b00000"), "Text.")
    eq(tuple(row), bx.INPUT_COLUMNS[task], "%s: the row has exactly the declared columns, in order" % task)
for task in bx.POSITIONAL:
    p = first(task)
    row = bx._input_row(task, p, rec("m", p.prompt_id, 1, "Text.", "m-b00000"), "Text.")
    eq(len(row), 2, "%s: exactly two columns, because the verifier reads by position" % task)
    eq(row["prompt"], p.meta["text"], "%s: column 0 is the source text" % task)
    check(row["prompt"] != p.prompt, "%s: column 0 is not the instruction wrapped around it" % task)
p = first("rationalization_binary")
eq(bx._input_row("rationalization_binary", p, rec("m", p.prompt_id, 1, "x.", "b"), "x.")["prompt"],
   p.prompt, "rationalization_binary: prompt is the question itself, which its verifier reads")
p = first("false_presuppositions")
r = bx._input_row("false_presuppositions", p, rec("m", p.prompt_id, 1, "x.", "b"), "x.")
check(r["list"].startswith("[") and "'" in r["list"],
      "false_presuppositions: list stays a Python list literal, which ast.literal_eval needs")

print("2. what happens to each kind of response before verification")
p = first("historical_events")
cases = [
    ("", oc.EMPTY, oc.EMPTY_RESPONSE, 0, ""),
    ("I cannot answer that.", oc.ABSTENTION, oc.ABSTAINED, 0, ""),
    ("They met in Paris. They talked about", oc.OK, "", 1, "They talked about"),
    ("they talked about poetry and", bx.EMPTIED, oc.NO_UNITS_FOUND, 0, "they talked about poetry and"),
    ("They met in Paris.", oc.OK, "", 1, ""),
]
tmp = tempfile.mkdtemp()
write_batch(tmp, "m", "m-b00001", [rec("m", p.prompt_id, 2, c[0], "m-b00001") for c in cases])
unit = bx.WorkUnit("m-b00001", "historical_events", "m", 2, [os.path.join(tmp, "m", "m-b00001.jsonl")])
# abstentions not sent; the default (sent) is checked below
prep = bx.prepare(unit, prompts, det, send_abstained=False)
for (resp, outcome, reason, sent, tail), m in zip(cases, prep.meta):
    label = repr(resp[:24])
    eq(m["outcome"], outcome, "%s: outcome %s" % (label, outcome))
    eq(m["decomposer_empty_reason"], reason, "%s: reason %r" % (label, reason))
    eq(m["sent"], sent, "%s: sent=%d" % (label, sent))
    eq(m["stripped_tail"], tail, "%s: the dropped text is recorded exactly" % label)
eq(len(prep.rows), 2, "only the two responses that can yield a unit are sent")
eq([m["input_row"] for m in prep.meta], [-1, -1, 0, -1, 1],
   "input_row numbers the sent rows in order, and is -1 for the rest")
eq(prep.rows[0]["response"], "They met in Paris.", "the verifier sees the text with its tail removed")
eq(prep.counts(), {oc.EMPTY: 1, oc.ABSTENTION: 1, oc.OK: 2, bx.EMPTIED: 1},
   "counts() tallies every generation, sent or not")
check(bx.EMPTIED != oc.EMPTY, "an emptied response and an empty one are different outcomes")
prep_off = bx.prepare(unit, prompts, det, strip=False, send_abstained=False)
eq(prep_off.rows[0]["response"], cases[2][0], "strip=False sends the cut response unchanged")
eq(len(prep_off.rows), 3, "strip=False also sends the response the tail rule would empty")
eq(len(bx.prepare(unit, prompts, None).rows), 3, "detector=None sends the abstention too")
eq(prep.meta[2]["generation_id"], "m|%s|2" % p.prompt_id, "the sidecar keeps generation_id")
eq(prep.rows[0]["generation_id"], "m|%s|2" % p.prompt_id, "a named task carries generation_id")
wrong = bx.WorkUnit("m-b00001", "summarization", "m", 2, unit.batch_paths)
raises(lambda: bx.prepare(wrong, prompts, det), ValueError,
       "a unit whose batch holds none of its task raises instead of verifying nothing")

print("   abstained responses are sent too, by default")
p18 = bx.prepare(unit, prompts, det)
m18 = p18.meta[1]
eq((m18["outcome"], m18["decomposer_empty_reason"], m18["sent"]), (oc.ABSTENTION, "", 1),
   "an abstention is sent, keeps outcome abstention, and has no reason yet")
eq(len(p18.rows), 3, "so three of the five are sent")
eq(p18.rows[m18["input_row"]]["response"], "I cannot answer that.", "with its text whole")
eq(p18.counts(), prep.counts(), "counts() does not change: the outcome is still abstention")
print("   'no response' on false_presuppositions is an abstention")
check(oc.says_no_response("\n\nNo response.\n\nIt appears to me") and oc.says_no_response(" no response;"),
      "the first non-empty line, any case, final punctuation ignored")
check(not oc.says_no_response("For example, if you cannot answer, write: no response"),
      "a prompt echo that mentions it later in the line is not")
fpq = first("false_presuppositions")
g18 = tempfile.mkdtemp()
write_batch(g18, "m", "m-b00007", [rec("m", fpq.prompt_id, 0, " no response\n\ninput question: List", "m-b00007", "stop"),
                                   rec("m", fpq.prompt_id, 1, "I cannot answer that and", "m-b00007", "length")])
u18 = bx.WorkUnit("m-b00007", "false_presuppositions", "m", 0, [os.path.join(g18, "m", "m-b00007.jsonl")])
m18 = bx.prepare(u18, prompts, det).meta[0]
eq((m18["outcome"], m18["sent"]), (oc.ABSTENTION, 1), "tagged abstention, and sent like any abstention")
u18b = bx.WorkUnit("m-b00007", "false_presuppositions", "m", 1, u18.batch_paths)
m18b = bx.prepare(u18b, prompts, det).meta[0]
eq((m18b["outcome"], m18b["decomposer_empty_reason"], m18b["sent"]), (oc.ABSTENTION, oc.NO_UNITS_FOUND, 0),
   "an abstention the tail rule empties keeps outcome abstention, with no_units_found as the reason")
print("   the tail rule only for a response that hit the cap")
rb = first("rationalization_binary")
g35 = tempfile.mkdtemp()
write_batch(g35, "m", "m-b00009", [rec("m", rb.prompt_id, 0, "Is 7 prime?\n\nyes", "m-b00009", "stop"),
                                   rec("m", rb.prompt_id, 1, "Is 7 prime?\n\nyes", "m-b00009", "length")])
u35 = bx.WorkUnit("m-b00009", "rationalization_binary", "m", 0, [os.path.join(g35, "m", "m-b00009.jsonl")])
m0 = bx.prepare(u35, prompts, None).meta
eq((m0[0]["stripped_tail"], m0[0]["sent"]), ("", 1),
   "a stop response ending in a bare yes keeps it: the model finished, the yes is the answer")
u35b = bx.WorkUnit("m-b00009", "rationalization_binary", "m", 1, u35.batch_paths)
m1 = bx.prepare(u35b, prompts, None).meta
eq(m1[0]["stripped_tail"], "yes", "the same text cut by the cap loses its unfinished last line")
r_none = rec("m", rb.prompt_id, 2, "Is 7 prime?\n\nyes", "m-b00009")
del r_none["finish_reason"]
write_batch(g35, "m", "m-b00010", [r_none])
m2 = bx.prepare(bx.WorkUnit("m-b00010", "rationalization_binary", "m", 2,
                            [os.path.join(g35, "m", "m-b00010.jsonl")]), prompts, None).meta
eq(m2[0]["stripped_tail"], "", "a record with no finish_reason is treated as finished, never stripped")

print("3. write_inputs round trip")
out = tempfile.mkdtemp()
inp, side = bx.write_inputs(prep, out)
with open(inp, encoding="utf-8", newline="") as f:
    got = list(csv.reader(f))
eq(tuple(got[0]), bx.INPUT_COLUMNS["historical_events"], "input header is the declared columns")
eq(len(got) - 1, 2, "input has one line per sent row")
with open(side, encoding="utf-8", newline="") as f:
    side_rows = list(csv.DictReader(f))
eq(len(side_rows), 5, "the sidecar has one line per generation, sent or not")
eq(side_rows[2]["stripped_tail"], "They talked about", "the dropped tail survives CSV unchanged")
eq(side_rows[3]["stripped_tail"], "they talked about poetry and",
   "an emptied response keeps its whole text as the dropped tail")
with open(inp, "rb") as f:
    check(b"\r\n" not in f.read(), "line endings are LF")

print("4. discover and pending")
g = tempfile.mkdtemp()
plan = os.path.join(g, "_plan")
gen = os.path.join(g, "gen")
bio = [q for q in prompts.values() if q.task == "biographies"][:3]
he = [q for q in prompts.values() if q.task == "historical_events"][:2]
write_plan(plan, "m", [
    {"prompt_id": bio[0].prompt_id, "sample_index": 1, "batch_id": "m-b00010"},
    {"prompt_id": bio[1].prompt_id, "sample_index": 1, "batch_id": "m-b00010"},
    {"prompt_id": bio[2].prompt_id, "sample_index": 1, "batch_id": "m-b00011"},
    {"prompt_id": he[0].prompt_id, "sample_index": 1, "batch_id": "m-b00020"},
])
write_batch(gen, "m", "m-b00020", [rec("m", q.prompt_id, 1, "A.", "m-b00020") for q in he])
write_batch(gen, "m", "m-b00010", [rec("m", q.prompt_id, 1, "A.", "m-b00010") for q in bio[:2]])
units = bx.discover(gen, plan)
eq(sorted(u.unit_id for u in units), ["m-b00010", "m-b00020"],
   "a Biographies batch is a unit on its own, without waiting for the rest of its group")
write_batch(gen, "m", "m-b00011", [rec("m", bio[2].prompt_id, 1, "A.", "m-b00011")])
units = {u.unit_id: u for u in bx.discover(gen, plan)}
check("m-b00011" in units, "the next Biographies batch becomes the next unit")
eq(len(units["m-b00010"].batch_paths), 1, "a Biographies unit holds one batch")
eq(len(bx.prepare(units["m-b00010"], prompts, det).rows), 2,
   "and only that batch's names")
bio_row = bx.prepare(units["m-b00010"], prompts, det).rows[0]
eq((bio_row["model"], bio_row["sample_index"]), ("m", 1),
   "a Biographies row carries model and sample_index, which verify_biographies shards on")
eq(len(bx.pending(gen, out, plan)), 3, "all three units are pending before any output exists")
o = bx.output_path(out, units["m-b00020"])
os.makedirs(os.path.dirname(o), exist_ok=True)
open(o, "w").close()
eq([u.unit_id for u in bx.pending(gen, out, plan)], ["m-b00010", "m-b00011"],
   "a unit with an output file is no longer pending")
eq(bx.done_per_task(gen, out, plan), {"historical_events": 1},
   "done_per_task counts units with an output, by task")
eq(bx.done_per_task(gen, out, plan, by_model=True), {("historical_events", "m"): 1},
   "and by (task, model) when asked")
eq(len(bx.pending(gen, out, plan)), 2, "pending is safe to call again and repeats nothing")
print("   supersede moves a finished unit aside so it runs again")
meta = os.path.join(out, "inputs", "m", "m-b00020.meta.csv")
os.makedirs(os.path.dirname(meta), exist_ok=True)
open(meta, "w").close()
moved = bx.supersede(out, [units["m-b00020"], units["m-b00010"]], "t35")
eq(sorted(p.replace(os.sep, "/") for p in moved),
   ["superseded/t35/inputs/m/m-b00020.meta.csv", "superseded/t35/outputs/m/m-b00020.csv"],
   "only the finished unit moves, with its sidecar; a missing inputs csv is skipped")
check(os.path.exists(os.path.join(out, "superseded", "t35", "outputs", "m", "m-b00020.csv")),
      "the old output is kept under superseded/{tag}/, nothing is deleted")
eq(len(bx.pending(gen, out, plan)), 3, "and the unit is pending again")
eq(bx.supersede(out, [units["m-b00020"]], "t35"), [], "superseding it again moves nothing")
open(o, "w").close()
moved2 = bx.supersede(out, [units["m-b00020"]], "t35")
eq([p.replace(os.sep, "/") for p in moved2], ["superseded/t35.2/outputs/m/m-b00020.csv"],
   "a second redo under the same tag goes to t35.2, so the first kept copy survives")
check(os.path.exists(os.path.join(out, "superseded", "t35", "outputs", "m", "m-b00020.csv")),
      "and t35 still holds the first one")
open(o, "w").close()
print("   a batch that holds two tasks (a task is 6.25 batches)")
fp = [q for q in prompts.values() if q.task == "false_presuppositions"][:2]
write_batch(gen, "m", "m-b00031", [rec("m", he[0].prompt_id, 1, "A.", "m-b00031"),
                                   rec("m", fp[0].prompt_id, 1, "B.", "m-b00031"),
                                   rec("m", fp[1].prompt_id, 1, "C.", "m-b00031")])
units = {u.unit_id: u for u in bx.discover(gen, plan)}
check("m-b00031-historical_events" in units and "m-b00031-false_presuppositions" in units,
      "it becomes two units, named by batch and task")
check("m-b00020" in units, "a batch with one task keeps the batch's own name")
fp_prep = bx.prepare(units["m-b00031-false_presuppositions"], prompts, det)
eq([m["prompt_id"] for m in fp_prep.meta], [fp[0].prompt_id, fp[1].prompt_id],
   "each unit takes only its own task's records")
eq(len(bx.prepare(units["m-b00031-historical_events"], prompts, det).meta), 1,
   "and the other unit takes the rest")
os.remove(os.path.join(gen, "m", "m-b00031.jsonl"))
write_batch(gen, "m", "m-b00011", [rec("m", bio[2].prompt_id, 1, "A.", "m-b00011"),
                                   rec("m", fp[0].prompt_id, 1, "B.", "m-b00011")])
bio_unit = {u.unit_id: u for u in bx.discover(gen, plan)}["m-b00011-biographies"]
eq(len(bx.prepare(bio_unit, prompts, det).meta), 1,
   "a Biographies batch that also holds another task is split like any other, and takes only its name")
check("m-b00011-false_presuppositions" in {u.unit_id for u in bx.discover(gen, plan)},
      "and the other task's part of that batch is a unit of its own")
write_batch(gen, "m", "m-b00011", [rec("m", bio[2].prompt_id, 1, "A.", "m-b00011")])
write_batch(gen, "other", "m-b00040", [rec("m", he[0].prompt_id, 1, "A.", "m-b00040")])
raises(lambda: bx.discover(gen, plan), ValueError,
       "a batch filed under the wrong model directory raises")

print("5. first_per_task: every task gets a verdict before any gets a second")
U = [bx.WorkUnit("%s-%d" % (t, i), t, "m", 1) for t in ("b_task", "a_task", "c_task")
     for i in range(3)]
got = [u.unit_id for u in bx.first_per_task(U)]
eq(got[:3], ["a_task-0", "b_task-0", "c_task-0"], "the first three units cover three tasks")
eq(len(got), 9, "nothing is dropped without a cap")
eq([u.unit_id for u in bx.first_per_task(U) if u.task == "b_task"],
   ["b_task-0", "b_task-1", "b_task-2"], "within a task the order is kept")
eq([u.unit_id for u in bx.first_per_task(U, per_task=1)],
   ["a_task-0", "b_task-0", "c_task-0"], "per_task=1 gives exactly one unit per task")
eq([u.task for u in bx.first_per_task(U, tasks=["c_task"])], ["c_task"] * 3,
   "tasks= limits which tasks run")
eq(bx.first_per_task([]), [], "no units, no order")
eq([u.unit_id for u in bx.first_per_task(U, per_task=1, done={"a_task": 1})],
   ["b_task-0", "c_task-0"], "a task that already has its one verified unit gets no more")
eq([u.unit_id for u in bx.first_per_task(U, per_task=2, done={"a_task": 1})],
   ["a_task-0", "b_task-0", "c_task-0", "b_task-1", "c_task-1"],
   "per_task counts what is done, so a_task gets one more to reach two")

print("6. pilot 2, end to end without a verifier")
if not os.path.isdir(PILOT2):
    print("   SKIP results/pilot2 not present")
else:
    units = bx.discover(PILOT2, PLAN_DIR)
    files = glob.glob(os.path.join(PILOT2, "*", "*.jsonl"))
    eq(len(units), len(files),
       "every pilot 2 batch file became exactly one unit, Biographies included")
    firsts = [(f, json.loads(open(f, encoding="utf-8").readline())) for f in sorted(files)]
    every = [bx.WorkUnit(os.path.basename(f), bx.task_of(r["prompt_id"]), "", r["sample_index"], [f])
             for f, r in firsts]
    totals = {}
    for u in every:
        pr = bx.prepare(u, prompts, None)
        for k, v in pr.counts().items():
            totals[k] = totals.get(k, 0) + v
    n_nonempty = totals.get(oc.OK, 0) + totals.get(bx.EMPTIED, 0) + totals.get(oc.ABSTENTION, 0)
    eq(sum(totals.values()), 5120, "all 5,120 pilot 2 generations are accounted for")
    eq(totals.get(oc.EMPTY, 0), 5120 - 4993, "127 empty, recorded as empty_response and never sent")
    eq(n_nonempty, 4993, "4,993 non empty responses")
    eq(totals.get(oc.ABSTENTION, 0), 3,
       "with no detector, the 3 false_presuppositions answers that open with 'no response'")
    eq(totals.get(bx.EMPTIED, 0), 25,
       "25 of them emptied by the tail rule (0.5%): a line break is a boundary, and only "
       "a response that hit the cap is stripped (71 without that gate)")

print("7. the real batch layout, rebuilt from the generation plan (tasks straddle batches)")
import collections
plan_rows = list(csv.DictReader(open(os.path.join(PLAN_DIR, "dclm-7b.csv"), encoding="utf-8")))
a2 = tempfile.mkdtemp()
by_batch = collections.OrderedDict()
for r in plan_rows:
    by_batch.setdefault(r["batch_id"], []).append(r)
for bid, rs in by_batch.items():
    write_batch(a2, "dclm-7b", bid, [rec("dclm-7b", r["prompt_id"], int(r["sample_index"]), "A.", bid)
                                     for r in rs])
a2_units = bx.discover(a2, PLAN_DIR)
cover = collections.Counter()
for u in a2_units:
    cover[(u.task, u.sample_index)] += len(bx.prepare(u, prompts, None).meta)
eq(len(by_batch), 300, "the plan has 300 batches for one model")
eq(set(cover.values()), {400}, "every (task, sample_index) is covered by exactly 400 records")
eq(len(cover), 48, "8 tasks x 6 sample indices")
bio_batches = {r["batch_id"] for r in plan_rows if bx.task_of(r["prompt_id"]) == "biographies"}
eq(sum(u.task == "biographies" for u in a2_units), len(bio_batches),
   "Biographies is one unit per batch that holds any of it, shared batches included")
check(all(len(u.batch_paths) == 1 for u in a2_units), "and every unit holds exactly one batch")
check(any(u.unit_id.endswith("-false_presuppositions") for u in a2_units),
      "straddling batches are split and named by task")
first = {u.task: u for u in bx.first_per_task(a2_units, per_task=1)}
check(all(not u.unit_id.endswith("-" + u.task) for u in first.values()),
      "per_task=1 picks a whole batch for every task, not a shared fragment")
eq(len(first), 8, "and one unit for each of the eight tasks")
multi = [bx.WorkUnit("%s-b%05d" % (m, i), "summarization", m, 1)
         for m in ("dclm-7b", "olmo17-7b", "olmo2-7b") for i in range(3)]
eq([u.model for u in bx.first_per_task(multi, per_task=3)],
   ["dclm-7b", "olmo17-7b", "olmo2-7b"],
   "within a task, models take turns, so three units are one per model")
pending_multi = [u for u in multi if u.unit_id != "dclm-7b-b00000"]
eq([u.model for u in bx.first_per_task(pending_multi, per_task=3, done={"summarization": 1},
                                       done_by_model={("summarization", "dclm-7b"): 1})],
   ["olmo17-7b", "olmo2-7b"],
   "a model that already has a verdict waits, so a rerun reaches the others first")
eq(len({u.unit_id for u in a2_units}), len(a2_units), "unit ids are unique")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all tests passed")
