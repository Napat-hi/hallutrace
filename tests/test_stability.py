"""Tests for hallutrace/stability.py and scripts/stability.py. Synthetic, no network."""

import csv
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout

from hallutrace import stability as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import stability as cli  # noqa: E402

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


def raises(fn, label):
    try:
        fn()
    except ValueError:
        check(True, label)
        return
    check(False, label + "  (no ValueError)")


def s(task="biographies", i=1, outcome="ok", n=3, hall=True, pid="p1", model="m", **kw):
    return dict({"task": task, "model": model, "prompt_id": pid, "sample_index": i, "outcome": outcome,
                 "n_units": n if outcome == "ok" else None, "hallucinated": hall}, **kw)


print("1. one sample")
for o in ("abstention", "empty", "emptied"):
    eq(st.status(s(outcome=o)), o, "%s leaves h" % o)
eq(st.status(s(n=2, hall=True)), st.HALLUCINATED, "a hallucinated unit")
eq(st.status(s(n=2, hall=False)), st.NOT_HALLUCINATED, "no hallucinated unit")
for task in ("summarization", "simplification", "scientific_attribution", "rationalization_numerical"):
    eq(st.status(s(task=task, n=0, hall=False, first_line="", gold_count=3)), st.ZERO_UNIT,
       "%s: zero units is no answer" % task)
for task in ("false_presuppositions", "rationalization_binary", "biographies", "historical_events"):
    eq(st.status(s(task=task, n=0, hall=False)), st.NOT_HALLUCINATED, "%s: zero units stays in, not hallucinated" % task)
eq(st.status(s(task="rationalization_numerical", n=0, hall=False, first_line="0 such words.", gold_count=0)),
   st.NOT_HALLUCINATED, "RN: a first line saying 0 is the answer 0, right when the gold count is 0")
eq(st.status(s(task="rationalization_numerical", n=0, hall=False, first_line="There are 0.", gold_count=2)),
   st.HALLUCINATED, "RN: the answer 0 against a gold count of 2")
eq(st.first_line("\n\n  0, none\nmore"), "0, none", "first non-empty line")
raises(lambda: st.status(s(n=None)), "ok without a verifier result is an error")

print("2. one item")
rows = st.items([s(i=0, hall=False), s(i=1), s(i=2), s(i=3, outcome="abstention"), s(i=4, hall=False), s(i=5)])
r = rows[0]
eq((r["n_samples"], r["n_usable"], r["n_hallucinated"], r["h"], r["regime_h"]), (5, 4, 3, 0.75, "S1"),
   "4 usable, 3 hallucinated: h 0.75, S1 (the worked example)")
eq((r["n_abstention"], r["greedy_hallucinated"]), (1, 0), "the greedy pass is kept apart")
rows = st.items([s(i=1), s(i=2), s(i=3, outcome="empty"), s(i=4, outcome="emptied"), s(i=5, outcome="abstention")])
eq((rows[0]["h"], rows[0]["regime_h"], rows[0]["low_n"]), (1.0, "always_wrong", 1),
   "always wrong on 2 usable samples: flagged low_n, regime kept")
rows = st.items([s(i=1)] + [s(i=k, outcome="abstention") for k in range(2, 6)])
eq((rows[0]["h"], rows[0]["regime_h"], rows[0]["low_n"]), (1.0, "single_mention", 0),
   "always wrong on 1 usable sample: single_mention, not a Low S2")
eq(rows[0]["cluster_method"], "none", "items carry no c: cluster_method none")
rows = st.items([s(i=1, hall=False)] + [s(i=k, outcome="abstention") for k in range(2, 6)])
eq(rows[0]["regime_h"], "S1", "one usable right sample is S1: S1 needs no minimum")
rows = st.items([s(i=k, outcome="abstention") for k in range(1, 6)])
eq((rows[0]["h"], rows[0]["regime_h"]), (None, ""), "no usable sample: no h, no regime")
rows = st.items([s(task="summarization", i=k, n=0, hall=False, first_line="") for k in range(1, 4)] +
                [s(task="summarization", i=4), s(task="summarization", i=5)])
eq((rows[0]["n_usable"], rows[0]["n_zero_unit"], rows[0]["h"]), (2, 3, 1.0),
   "summarization: 3 zero-unit samples out, h from the 2 left")

print("3. per task and model")
sm = st.summary(rows + st.items([s(pid="p2", i=k, hall=False) for k in range(1, 6)]))
eq(set(sm), {"summarization|m", "biographies|m"}, "keys")
eq((sm["summarization|m"]["response_ratio"], sm["summarization|m"]["h_pooled"],
    sm["summarization|m"]["h_pooled_halogen_zero_units"]), (0.4, 1.0, 0.4),
   "response ratio 2/5; HALoGEN's h counts the zero-unit samples as right: 2/5")
eq((sm["summarization|m"]["items_missing_samples"], sm["biographies|m"]["items_missing_samples"]), (0, 0),
   "every item has its five samples")
eq(st.summary(st.items([s(i=1), s(i=2), s(i=3)]))["biographies|m"]["items_missing_samples"], 1,
   "an item with 3 of 5 samples is counted as missing samples")

print("4. per slot: Biographies years")


def yc(i, years, kind="birth", own="1869", name="Ada", model="m", pid="biographies-00001"):
    return {"generation_id": "%s|%s|%d" % (model, pid, i), "model": model, "sample_index": str(i), "name": name,
            "kind": kind, "stated_years": years, "wikidata_years": own}


slots = st.slot_items([yc(1, "1869"), yc(2, "1870"), yc(3, "1870"), yc(0, "1870")])
r = slots[0]
eq((r["n_right"], r["n_wrong"], r["n_absent"], r["regime"], r["greedy"]), (1, 2, 2, "S1", "wrong"),
   "one right sample makes S1; absent samples are out of the denominator")
r = st.slot_items([yc(k, "1870") for k in (1, 2, 3, 4)] + [yc(5, "1871")])[0]
eq((r["h"], r["dominant_wrong"], r["c"], r["regime"], r["low_n"]), (1.0, "1870", 0.8, "S2", 0),
   "4 of 5 on the same wrong year: c 0.8, S2")
r = st.slot_items([yc(1, "1870"), yc(2, "1871"), yc(3, "1872")])[0]
eq((r["c"], r["regime"], r["low_n"]), (0.3333, "S3", 0), "three different wrong years: S3")
r = st.slot_items([yc(1, "1870")])[0]
eq((r["regime"], r["c"], r["low_n"], r["dominant_wrong"]), ("single_mention", None, 0, "1870"),
   "one wrong sample, no right one: single_mention, no c")
r = st.slot_items([yc(1, "1870"), yc(2, "1870")])[0]
eq((r["regime"], r["c"], r["low_n"]), ("S2", 1.0, 1), "two wrong samples on one year: S2, Low below 3")
r = st.slot_items([yc(1, "1870"), yc(2, "1871")])[0]
eq((r["regime"], r["c"], r["low_n"]), ("S3", 0.5, 1), "two wrong samples on two years: S3, Low")
r = st.slot_items([yc(1, "1870 1871"), yc(2, "1871"), yc(3, "1872")])[0]
eq((r["dominant_wrong"], r["c"], r["regime"]), ("1871", 0.6667, "S3"),
   "a sample stating two wrong years counts for each: 1871 in 2 of 3")
r = st.slot_items([yc(k, "1870") for k in (1, 2, 3, 4)] + [yc(5, "1869")])[0]
eq((r["c"], r["regime"]), (1.0, "S1"), "a right sample wins over 4 agreeing wrong ones")
eq((r["cluster_method"], r["c_cut"]), ("year", 0.8), "cluster_method year, cut 0.8 on every slot")
eq(st.slot_items([yc(0, "1870")])[0]["regime"], "", "a year only the greedy pass states: no regime")
r = st.slot_items([yc(0, "1870"), yc(1, "1870")], greedy=True)[0]
eq((r["n_wrong"], r["n_absent"], r["regime"], r["c"]), (2, 4, "S2", 1.0),
   "greedy as a sixth sample: two wrong, S2 (supplementary run)")
eq(st.slot_items([yc(1, "1870", own="")]), [], "no Wikidata year: the slot can't be judged")
r = st.slot_items([yc(1, "1869 1870")])[0]
eq((r["n_right"], r["n_wrong"], r["dominant_wrong"], r["regime"]), (1, 0, "", "S1"),
   "a sample stating the right and a wrong year is right")
r = st.slot_items([yc(1, "1869 1870"), yc(2, "1870"), yc(3, "1870")])[0]
eq((r["n_right"], r["n_wrong"], r["dominant_wrong"], r["c"], r["regime"]), (1, 2, "1870", 1.0, "S1"),
   "the wrong years of a right sample don't count; the other samples' do, and S1 keeps dominant_wrong")
r = st.slot_items([yc(1, "1870"), yc(2, "1870"), yc(3, "1869 1871")])[0]
eq((r["n_wrong"], r["regime"]), (2, "S1"), "a sample contradicting itself makes S1 (S3 under the old rule)")
eq(st.slot_summary(st.slot_items([yc(1, "1870"), yc(1, "1869", kind="death", own="1869")])),
   {"m": {"slots": 2, "single_mention": 1, "low_n": 0, "S1": 1}}, "per model")

print("5. Biographies from the verification files")


def write(path, cols, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


META = ["generation_id", "prompt_id", "model", "sample_index", "task", "batch_id", "outcome",
        "decomposer_empty_reason", "sent", "input_row", "stripped_tail"]
root = tempfile.mkdtemp()


def meta(i, outcome="ok", row=-1, pid="biographies-00001"):
    return {"generation_id": "m|%s|%d" % (pid, i), "prompt_id": pid, "model": "m", "sample_index": i,
            "task": "biographies", "batch_id": "b", "outcome": outcome, "decomposer_empty_reason": "",
            "sent": int(outcome == "ok"), "input_row": row, "stripped_tail": ""}


write(os.path.join(root, "inputs", "m", "m-b00000.meta.csv"), META,
      [meta(0, row=0), meta(1, row=1), meta(2, "abstention"), meta(3, row=2), meta(4, "empty"), meta(5, row=3)])
write(os.path.join(root, "inputs", "m", "m-b00000.csv"), ["name", "response", "generation_id"],
      [{"name": "Ada %d" % k, "response": "r", "generation_id": "g"} for k in range(4)])
write(os.path.join(root, "outputs_filtered", "m", "m-b00000.csv"), ["name", "atoms"],
      [{"name": "Ada 0", "atoms": "| A.\tTrue |\n"}, {"name": "Ada 1", "atoms": "| A.\tFalse |\n| B.\tTrue |\n"},
       {"name": "Ada 2", "atoms": ""}, {"name": "Ada 3", "atoms": "| A.\tFalse |\n"}])
# a stray unit from an earlier layout, listing the same generations, with no output
write(os.path.join(root, "inputs", "m", "m-biographies-s0.meta.csv"), META, [meta(1, row=0)])
write(os.path.join(root, "inputs", "m", "m-biographies-s0.csv"), ["name", "response", "generation_id"],
      [{"name": "Ada 1", "response": "r", "generation_id": "g"}])
write(os.path.join(root, "inputs", "m", "m-b00001-historical_events.meta.csv"), META,
      [dict(meta(1), task="historical_events")])
skipped = []
ss = st.bio_samples(root, skipped=skipped)
eq(len(ss), 6, "six generations, the other task's unit ignored")
eq(skipped, ["m-biographies-s0"], "the stray unit is skipped and reported")
it = st.items(ss)[0]
eq((it["n_usable"], it["n_hallucinated"], it["h"], it["greedy_hallucinated"]), (3, 2, 0.6667, 0),
   "samples 1 and 5 hallucinated, 3 has no atoms (stays in), 2 and 4 out")
os.remove(os.path.join(root, "outputs_filtered", "m", "m-b00000.csv"))
raises(lambda: st.bio_samples(root), "an ok generation with no output row is an error")
write(os.path.join(root, "outputs_filtered", "m", "m-b00000.csv"), ["name", "atoms"],
      [{"name": "Ada %d" % k, "atoms": "| A.\tFalse |\n"} for k in range(4)])
ss = st.bio_samples(root)
kept, dropped = st.ok_claims([yc(1, "1870"), yc(2, "1869"), yc(3, "1869")], ss)
eq(([c["sample_index"] for c in kept], dropped), (["1", "3"], 1),
   "a claim from an abstained generation (sample 2) is dropped")
raises(lambda: st.ok_claims([yc(1, "1870", pid="biographies-00009")], ss),
       "a claim whose generation has no sidecar row is an error")

print("6. the command line")
# a second model whose samples all abstained: no usable sample, no pooled h
write(os.path.join(root, "inputs", "m2", "m2-b00000.meta.csv"), META,
      [dict(meta(k, "abstention"), model="m2", generation_id="m2|biographies-00001|%d" % k) for k in range(6)])
claims = os.path.join(root, "claims.csv")
write(claims, list(yc(1, "1870")), [yc(1, "1870"), yc(2, "1869"), yc(3, "1869")])
out = os.path.join(root, "out")
with redirect_stdout(io.StringIO()):
    eq(cli.main(["--verifier-root", root, "--claims", claims, "--out", out]), 0,
       "exit 0, with a model that has no usable sample")
sm = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
eq((sm["items"]["biographies|m"]["items"], sm["slots"]["m"]["S1"], sm["skipped_units"]),
   (1, 1, ["m-biographies-s0"]), "items, slots and the skipped unit in summary.json")
eq(sm["items"]["biographies|m2"]["h_pooled"], None, "no pooled h is written as null")
eq((sm["slots_from"]["claims"], sm["slots_from"]["built_from"], sm["slots_from"]["claims_not_ok_dropped"]),
   ("claims.csv", "outputs_filtered", 1), "summary.json records where the slots come from and the dropped claim")
eq(len(sm["slots_from"]["sha256"]), 64, "and the claims file's sha256")
check(all(os.path.exists(os.path.join(out, f)) for f in ("items.csv", "slots.csv", "slots_greedy.csv")),
      "items.csv, slots.csv and slots_greedy.csv")
eq(sm["slots_greedy"]["m"]["slots"], 1, "the greedy run in summary.json")

print("7. the other seven tasks from the verification files")
v2 = tempfile.mkdtemp()


def unit(root_, name, task, metas, inputs, outputs, model="m"):
    rows = [dict(meta(i, o, row=r, pid="%s-00001" % task), task=task, model=model,
                 generation_id="%s|%s-00001|%d" % (model, task, i)) for i, o, r in metas]
    write(os.path.join(root_, "inputs", model, name + ".meta.csv"), META, rows)
    write(os.path.join(root_, "inputs", model, name + ".csv"), list(inputs[0]), inputs)
    if outputs is not None:
        write(os.path.join(root_, "outputs", model, name + ".csv"), list(outputs[0]), outputs)


def gid(task, i, model="m"):
    return "%s|%s-00001|%d" % (model, task, i)


SA = "scientific_attribution"
sa_out = [
    {"response": "See 'Deep Learning for Cats' by X.", "generation_id": gid(SA, 1), "atomic_units": "['Deep Learning for Cats']",
     "s2_titles": "[]", "hallucinated_atomic_units": "['Deep Learning for Cats']", "unverifiable_atomic_units": "[]"},
    {"response": "See 'Real Paper'.", "generation_id": gid(SA, 2), "atomic_units": "['Real Paper', 'Invented Title Here']",
     "s2_titles": "['Real Paper']", "hallucinated_atomic_units": "['Invented Title Here']", "unverifiable_atomic_units": "[]"},
    {"response": "No titles.", "generation_id": gid(SA, 3), "atomic_units": "[]", "s2_titles": "[]",
     "hallucinated_atomic_units": "[]", "unverifiable_atomic_units": "[]"}]
unit(v2, "m-b00001-" + SA, SA, [(1, "ok", 0), (2, "ok", 1), (3, "ok", 2), (4, "abstention", -1), (5, "empty", -1)],
     [{"response": r["response"], "generation_id": r["generation_id"]} for r in sa_out], sa_out)
raw = st.items(st.task_samples(v2, SA, filtered=False))[0]
filt = st.items(st.task_samples(v2, SA))[0]
eq((raw["n_usable"], raw["n_hallucinated"], raw["n_zero_unit"], raw["n_abstention"], raw["n_empty"]), (2, 2, 1, 1, 1),
   "SA as released: 2 usable, both hallucinated; no titles is no answer; the abstention and the empty one out")
eq((filt["n_usable"], filt["n_hallucinated"], filt["h"]), (2, 1, 0.5),
   "SA filtered: the invented title is dropped, so sample 2 is not hallucinated")

SU = "summarization"
su_out = [
    {"passage": "p", "response": "Ada wrote code.",
     "atomic_units": "Here are the independent facts broken down from the passage: | False |\n Ada wrote code. | True |",
     "hallucinated_atomic_units": "Here are the independent facts broken down from the passage:"},
    {"passage": "p", "response": "Ada wrote code in 1900.",
     "atomic_units": " Ada wrote code. | True |\n Ada was born in 1900. | False |", "hallucinated_atomic_units": " Ada was born in 1900."}]
unit(v2, "m-b00002-" + SU, SU, [(1, "ok", 0), (2, "ok", 1)],
     [{"prompt": "x", "response": r["response"]} for r in su_out], su_out)
raw = st.items(st.task_samples(v2, SU, filtered=False))[0]
filt = st.items(st.task_samples(v2, SU))[0]
eq((raw["n_hallucinated"], filt["n_hallucinated"]), (2, 1),
   "summarization as decompose: the framing line is the only false atom of sample 1 and the filter drops it")

RN = "rationalization_numerical"
rn_cols = {"list": "[]", "list_condition": "[]", "parsed_response": "", "numerical_atomic_unit": "",
           "list_atomic_unit": "", "true_answer": "", "atomic_units": "[]", "hallucinated_atomic_units": "set()"}
rn_out = [dict(rn_cols, response="0 such words.", count="0", generation_id=gid(RN, 1)),
          dict(rn_cols, response="\n0", count="3.0", generation_id=gid(RN, 2))]
unit(v2, "m-b00003", RN, [(1, "ok", 0), (2, "ok", 1)],
     [{k: r[k] for k in ("response", "list", "list_condition", "count", "generation_id")} for r in rn_out], rn_out)
ss = st.task_samples(v2, RN)
eq([x["zero_unit_rule"] for x in ss], ["answer_0_right", "answer_0_wrong"],
   "RN: posthoc labels the answer 0 against the gold count (3.0 read as 3)")
eq((st.items(ss)[0]["n_usable"], st.items(ss)[0]["n_hallucinated"]), (2, 1), "RN: the labels decide h, not a second copy of the rule")

HE_ = "historical_events"
he_out = [{"response": "Yes.", "name1": "A", "name2": "B", "generation_id": gid(HE_, i), "responds": resp,
           "verification": "", "verified_response": "", "atomic_units": au, "hallucinated_atomic_units": hu}
          for i, resp, au, hu in ((1, "True", "['yes']", "['yes']"), (2, "True", "['no']", "[]"),
                                  (3, "False", "[]", "[]"), (4, "True", "['yes']", "['yes']"))]
unit(root, "m-b00004-" + HE_, HE_, [(1, "ok", 0), (2, "ok", 1), (3, "ok", 2), (4, "abstention", 3)],
     [{k: r[k] for k in ("response", "name1", "name2", "generation_id")} for r in he_out], he_out)
def he_item(filtered=True):   # section 5's HE sidecar has no unit files, so pick this prompt
    return [r for r in st.items(st.task_samples(root, HE_, filtered)) if r["prompt_id"] == HE_ + "-00001"][0]


it = he_item()
eq((it["n_usable"], it["n_hallucinated"], it["n_abstention"]), (3, 1, 1),
   "HE: yes or no is the unit, no answer stays in, the sent abstention (outcome from the sidecar) is out")
eq(he_item(False)["h"], it["h"], "HE is the same with and without the filter")

# false_presuppositions and rationalization_binary: no units is a right answer, it stays in
FP = "false_presuppositions"
fp_out = [{"response": "None of them.", "list": "[]", "list_condition": "[]", "generation_id": gid(FP, 1),
           "parsed_response": "", "atomic_units": "[]", "hallucinated_atomic_units": "[]"}]
unit(v2, "m-b00005-" + FP, FP, [(1, "ok", 0)], [{k: fp_out[0][k] for k in ("response", "list", "list_condition", "generation_id")}], fp_out)
RB = "rationalization_binary"
rb_out = [{"response": "No.", "category": "primality", "prompt": "q", "generation_id": gid(RB, 1), "extract_atomic_unit_prompt": "",
           "parsed_response": "", "atomic_units": "['no']", "hallucinated_atomic_units": "[]", "true_answer": "", "yesno": "['no']"}]
unit(v2, "m-b00006-" + RB, RB, [(1, "ok", 0)], [{k: rb_out[0][k] for k in ("response", "category", "prompt", "generation_id")}], rb_out)
for task in (FP, RB):
    r = st.items(st.task_samples(v2, task))[0]
    eq((r["n_usable"], r["n_hallucinated"], r["n_zero_unit"]), (1, 0, 0), "%s: no units stays in as not hallucinated" % task)
SI = "simplification"
unit(v2, "m-b00007-" + SI, SI, [(1, "ok", 0)], [{"prompt": "x", "response": su_out[1]["response"]}], su_out[1:])
eq(st.items(st.task_samples(v2, SI))[0]["n_hallucinated"], 1, "simplification goes through the decompose filter too")

bad = tempfile.mkdtemp()
wrong = [dict(sa_out[0], generation_id=gid(SA, 9))]
unit(bad, "m-b00001-" + SA, SA, [(1, "ok", 0)], [{"response": sa_out[0]["response"], "generation_id": gid(SA, 1)}], wrong)
raises(lambda: st.task_samples(bad, SA), "an output row with another generation_id is an error")
bad = tempfile.mkdtemp()
unit(bad, "m-b00002-" + SU, SU, [(1, "ok", 0)], [{"prompt": "x", "response": "something else"}], su_out[:1])
raises(lambda: st.task_samples(bad, SU), "summarization: an output row whose response isn't its input's is an error")
bad = tempfile.mkdtemp()
unit(bad, "m-b00001-" + SA, SA, [(1, "ok", 0), (2, "ok", 5)], [{"response": "r", "generation_id": gid(SA, 1)}] * 2,
     [dict(sa_out[0], response="r", generation_id=gid(SA, 1))] * 2)
raises(lambda: st.task_samples(bad, SA), "an input_row past the end of the output is an error")
bad = tempfile.mkdtemp()
unit(bad, "m-b00001-" + SA, SA, [(1, "ok", 0), (2, "abstention", -1)], [{"response": "r", "generation_id": gid(SA, 1)}], None)
it = st.items(st.task_samples(bad, SA))[0]
eq((it["n_missing"], it["n_abstention"], it["h"]), (1, 1, None), "a unit with no output: its ok generation is missing, out of h")
eq(st.summary([it])["%s|m" % SA]["n_missing"], 1, "missing is counted per task and model")
raises(lambda: st.task_samples(v2, "biographies"), "Biographies has its own loader")

out2 = os.path.join(root, "out2")
with redirect_stdout(io.StringIO()):
    eq(cli.main(["--verifier-root", root, "--v2-root", v2, "--claims", claims, "--out", out2]), 0, "exit 0 with --v2-root")
sm = json.load(open(os.path.join(out2, "summary.json"), encoding="utf-8"))
eq(sorted(k for k in sm["items"]),
   sorted(["biographies|m", "biographies|m2", HE_ + "|m"] + ["%s|m" % t for t in st.V2_TASKS]),
   "items: Biographies, HE and all six v2 tasks")
eq(sorted(sm["items_raw"]), sorted("%s|m" % t for t in st.V2_TASKS), "items_raw: the six v2 tasks")
eq((sm["items"][SA + "|m"]["n_hallucinated"], sm["items_raw"][SA + "|m"]["n_hallucinated"]), (1, 2),
   "items filtered, items_raw as released")
check(os.path.exists(os.path.join(out2, "items_raw.csv")), "items_raw.csv")
with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
    try:
        cli.main(["--verifier-root", root, "--v2-root", bad, "--claims", claims, "--out", out2])
        check(False, "a --v2-root without every task is refused  (no exit)")
    except SystemExit as e:
        check(e.code == 2, "a --v2-root without every task is refused")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
