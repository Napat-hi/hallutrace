"""Tests for hallutrace/gold_cut.py. No key, no network, no pandas."""

import csv
import json
import os
import sys
import tempfile

from hallutrace import gold_cut as gc

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("1. reading verifier cells")
eq(gc.as_list("[' Mercury', 'Mars']"), [" Mercury", "Mars"], "a list literal")
eq(gc.as_list("set()"), [], "set() is empty, as rationalization_numerical writes it")
eq(gc.as_list(""), [], "an empty cell")
eq(gc.as_list("nan"), [], "a pandas NaN written as text")
eq(gc.as_list("John Kerry"), ["John Kerry"], "a bare string is one item")

print("2. the unit check: is an extracted item in the response")
R = "The correct answer is:\n\nJanuary, June, November\n"
check(gc.in_response("June", R), "an item the model wrote is present")
check(not gc.in_response("Mimas", R), "the prompt's planet example is not")
check(not gc.in_response("Jun", R), "a word fragment does not count as present")
check(gc.in_response("", R), "an empty item is treated as present")
T = 'Smith J. "Deep Learning for Protein Folding", Nature 2020.'
check(gc.in_response("Deep learning for protein folding", T), "case and punctuation are ignored")
check(gc.in_response("Deep Learning Approaches for Protein Folding", T, loose=True),
      "loose: a reworded title with 80% of its words present counts")
check(not gc.in_response("Financial Accounting Theory", T, loose=True),
      "loose: an invented title does not")

print("3. units, per task")
FP = {"generation_id": "m|false_presuppositions-00001|1", "response": R,
      "atomic_units": "['January', 'Mimas']", "hallucinated_atomic_units": "['mimas']"}
u = gc.UNIT_BUILDERS["false_presuppositions"](FP, "m/b.csv")
eq(len(u), 1, "false_presuppositions: one unit per response, the extraction")
eq(u[0].unit_text, "atomic_units: January, Mimas", "the annotator sees what was extracted")
eq(u[0].context, R, "and the response it was extracted from")
check(u[0].flagged and u[0].unit_guard_drops, "flagged, and the unit guard would drop it")
SA = {"generation_id": "m|scientific_attribution-00002|1", "response": T,
      "atomic_units": "['Deep Learning for Protein Folding', 'Financial Accounting Theory', 'Odd Title']",
      "hallucinated_atomic_units": "['Financial Accounting Theory']",
      "unverifiable_atomic_units": "['Odd Title']"}
u = gc.UNIT_BUILDERS["scientific_attribution"](SA, "m/b.csv")
eq([x.unit_id for x in u], ["m|scientific_attribution-00002|1:%d" % i for i in range(3)],
   "scientific_attribution: one unit per reference")
eq([x.verifier_label for x in u], [gc.SUPPORTED, gc.HALLUCINATED, gc.UNVERIFIABLE],
   "the verifier's three answers are kept apart, UNVERIFIABLE included (DEVIATIONS 3)")
eq([x.unit_guard_drops for x in u], [False, True, True], "the guard would drop the two not in the response")
HE = {"generation_id": "m|historical_events-00003|0", "response": "They met in 1901.",
      "name1": "A", "name2": "B", "hallucinated_atomic_units": "['yes']"}
u = gc.UNIT_BUILDERS["historical_events"](HE, "m/b.csv")
eq((u[0].unit_text, u[0].verifier_label), ("A and B", gc.HALLUCINATED),
   "historical_events: the whole response, with the two names")
RN = {"generation_id": "m|rationalization_numerical-00004|1", "response": "2, mars, venus",
      "numerical_atomic_unit": "2", "list_atomic_unit": "['mars', 'venus']",
      "atomic_units": "['mars', 'venus']", "hallucinated_atomic_units": "set()"}
u = gc.UNIT_BUILDERS["rationalization_numerical"](RN, "m/b.csv")
eq(u[0].unit_text, "numerical_atomic_unit: 2; list_atomic_unit: mars, venus",
   "rationalization_numerical: the number and the list")
check(not u[0].flagged, "set() means nothing hallucinated")
eq(u[0].model, "m", "the model comes from generation_id")
try:
    gc.build_units([({"generation_id": "m|biographies-00001|1", "response": "x"}, "s")])
    check(False, "a task without an adapter raises")
except KeyError as e:
    check("biographies" in str(e), "a task without an adapter raises, naming it")

print("summarization and simplification: one atom per unit, against the passage")
SU = {"generation_id": "m|summarization-00003|0", "passage": "Cats sleep a lot.", "response": "Cats sleep.",
      "atomic_units": "Here are the facts: | True |\n Cats sleep. | True |\n Cats fly. | False |"}
u = gc.UNIT_BUILDERS["summarization"](SU, "m/b.csv")
eq([x.unit_text for x in u], ["Here are the facts:", "Cats sleep.", "Cats fly."], "every atom is a unit")
eq([x.verifier_label for x in u], [gc.SUPPORTED, gc.SUPPORTED, gc.HALLUCINATED], "True is supported, False hallucinated")
eq(u[1].context, "Cats sleep a lot.", "the annotator is shown the source passage, not the response")
eq(gc.atom_lines(""), [], "an empty cell has no atoms")
eq(gc.UNIT_BUILDERS["simplification"](dict(SU, generation_id="m|simplification-00001|0"), "s")[2].task,
   "simplification", "simplification is built the same way")

print("read_outputs joins a positional output to its sidecar")
root = tempfile.mkdtemp()
os.makedirs(os.path.join(root, "outputs", "m"))
os.makedirs(os.path.join(root, "inputs", "m"))
with open(os.path.join(root, "outputs", "m", "m-b00001.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["passage", "response", "atomic_units", "hallucinated_atomic_units"])
    w.writerow(["P1", "R1", "a | True |", ""])
    w.writerow(["P2", "R2", "b | False |", ""])
with open(os.path.join(root, "inputs", "m", "m-b00001.meta.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["generation_id", "sent", "input_row"])
    w.writerow(["m|summarization-00002|0", "1", "1"])
    w.writerow(["m|summarization-00009|0", "0", "-1"])
    w.writerow(["m|summarization-00001|0", "1", "0"])
got = gc.read_outputs(os.path.join(root, "outputs"), os.path.join(root, "inputs"))
eq([(r["passage"], r["generation_id"]) for r, _ in got],
   [("P1", "m|summarization-00001|0"), ("P2", "m|summarization-00002|0")],
   "rows get their generation_id by input_row order, unsent rows skipped")
eq(gc.read_outputs(os.path.join(root, "outputs")), [], "without inputs_dir a positional file is skipped, as before")
sk = []
gc.read_outputs(os.path.join(root, "outputs"), None, sk)
eq(sk, [os.path.join("m", "m-b00001.csv")], "and the skipped file is handed back, never silent")
with open(os.path.join(root, "inputs", "m", "m-b00001.meta.csv"), "a", newline="") as f:
    csv.writer(f).writerow(["m|summarization-00005|0", "1", "2"])
try:
    gc.read_outputs(os.path.join(root, "outputs"), os.path.join(root, "inputs"))
    check(False, "a row count that disagrees with the sidecar raises")
except ValueError:
    check(True, "a row count that disagrees with the sidecar raises")

print("abstentions are left out of the draw")
ab = tempfile.mkdtemp()
os.makedirs(os.path.join(ab, "outputs", "m"))
os.makedirs(os.path.join(ab, "inputs", "m"))
with open(os.path.join(ab, "outputs", "m", "m-b00003.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["response", "name1", "name2", "generation_id", "hallucinated_atomic_units"])
    w.writerow(["They met.", "A", "B", "m|historical_events-00001|0", "['yes']"])
    w.writerow(["I can't say. They met.", "C", "D", "m|historical_events-00002|0", "['yes']"])
with open(os.path.join(ab, "inputs", "m", "m-b00003.meta.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["generation_id", "outcome", "sent", "input_row"])
    w.writerow(["m|historical_events-00001|0", "ok", "1", "0"])
    w.writerow(["m|historical_events-00002|0", "abstention", "1", "1"])
ex = []
got = gc.read_outputs(os.path.join(ab, "outputs"), os.path.join(ab, "inputs"), excluded=ex)
eq([r["generation_id"] for r, _ in got], ["m|historical_events-00001|0"], "an abstention is not in the pool")
eq(ex, ["m|historical_events-00002|0"], "and it is handed back as excluded")
eq(len(gc.read_outputs(os.path.join(ab, "outputs"))), 2, "without inputs_dir nothing can be excluded")


def units(n_pos, n_neg, task="false_presuppositions"):
    return ([gc.Unit("p%03d" % i, "m|%s-%05d|1" % (task, i), task, "c", "u", gc.HALLUCINATED)
             for i in range(n_pos)] +
            [gc.Unit("n%03d" % i, "m|%s-%05d|2" % (task, i), task, "c", "u", gc.SUPPORTED)
             for i in range(n_neg)])


print("4. the draw")
big = units(300, 700)
d = gc.draw("false_presuppositions", big, 7)
eq(len(d.units), 100, "100 units from a large pool")
eq(sum(u.flagged for u in d.units), 45, "45 flagged, the middle of 40 to 50")
check(d.stats["in_flagged_range"] and not d.stats["short_of_n"], "and the stats say so")
eq(d.stats["base_rate"], 0.3, "the natural base rate is recorded")
eq(d.stats["sampling_ratio"], 0.45, "and the sampling ratio")
eq([u.unit_id for u in gc.draw("false_presuppositions", big, 7).units], [u.unit_id for u in d.units],
   "the same seed gives the same draw")
check([u.unit_id for u in gc.draw("false_presuppositions", big, 8).units] != [u.unit_id for u in d.units],
      "a different seed gives a different one")
check([u.unit_id for u in gc.draw("false_presuppositions", list(reversed(big)), 7).units] == [u.unit_id for u in d.units],
      "input order does not change the draw")
first = [u.flagged for u in d.units[:45]]
check(0 < sum(first) < 45, "shuffled: flagged units are not all at the top")
small = gc.draw("false_presuppositions", units(20, 200), 7)
eq(sum(u.flagged for u in small.units), 20, "too few flagged: all of them are taken")
eq(len(small.units), 100, "and unflagged fill the rest")
check(not small.stats["in_flagged_range"], "which the stats flag")
tiny = gc.draw("false_presuppositions", units(10, 5), 7)
eq(len(tiny.units), 15, "a pool smaller than 100 is taken whole")
check(tiny.stats["short_of_n"], "and marked short")

print("5. the files")
out = tempfile.mkdtemp()
paths = gc.write_packets(d, out, ["m/b.csv"])
with open(paths["annotator_A.csv"], encoding="utf-8") as f:
    a = list(csv.reader(f))
with open(paths["annotator_B.csv"], encoding="utf-8") as f:
    b = list(csv.reader(f))
eq(a, b, "the two annotators get identical files, in the same order")
eq(tuple(a[0]), gc.ANNOTATOR_COLUMNS, "annotator columns as the procedure lists them")
check(not any(c in a[0] for c in ("verifier_label", "flagged", "unit_guard_drops", "unit_id")),
      "the verifier's answer is deleted, not hidden")
check(a[0][4:] == ["annotator_label", "tag", "notes"],
      "tag sits between annotator_label and notes, as in the Biographies packet")
check(all(r[4] == "" and r[5] == "" and r[6] == "" for r in a[1:]), "label, tag and notes start empty")
with open(paths["key.csv"], encoding="utf-8") as f:
    key = list(csv.DictReader(f))
eq([k["item_id"] for k in key], [r[0] for r in a[1:]], "key.csv aligns with the packets by item_id")
m = json.load(open(paths["manifest.json"]))
eq((m["seed"], m["labels"]), ("7", ["hallucinated", "supported", "unverifiable"]),
   "the manifest records the seed and the three labels")
eq(m["tags"], ["not a claim", "exaggerated", "no article"], "the manifest lists the allowed tags")
try:
    from hallutrace import agreement as ag
    eq(tuple(gc.TAGS), tuple(ag.TAGS), "gold_cut.TAGS is the tuple agreement.py reads")
except ImportError:
    print("   (agreement.py not importable here, TAGS cross check skipped)")
eq(m["measure"], gc.EXTRACTOR_ACCURACY,
   "the extraction tasks report extractor accuracy, not kappa")
for t in ("false_presuppositions", "rationalization_binary", "rationalization_numerical"):
    eq(gc.MEASURES.get(t), gc.EXTRACTOR_ACCURACY, "%s is measured as extractor accuracy" % t)
for t in ("scientific_attribution", "historical_events", "summarization", "simplification"):
    eq(gc.MEASURES.get(t, gc.KAPPA), gc.KAPPA, "%s stays on kappa" % t)
m2 = json.load(open(gc.write_packets(d, tempfile.mkdtemp(), [], {"skipped_files": ["x.csv"],
                                                                "unit_guard_loaded": False})["manifest.json"]))
eq((m2["skipped_files"], m2["unit_guard_loaded"]), (["x.csv"], False),
   "what the run knew about its inputs goes into the manifest")
check("Yes: hallucinated. No: supported. Can't tell: unverifiable." in gc.QUESTIONS["historical_events"],
      "historical_events states its yes/no to label mapping in the question")
eq(gc.QUESTIONS["summarization"], "Is this fact entailed by the source passage?", "summarization asks the procedure's question")
eq(gc.QUESTIONS["simplification"], "Is this fact entailed by the source sentence?", "simplification asks the procedure's question")
check(isinstance(gc.unit_guard_loaded(), bool), "unit_guard_loaded says whether unit_guard_drops is real")

print("5b. the first n of a drawn task (annotators label 50 per task)")
src = os.path.join(out, "false_presuppositions")
cut_dir = os.path.join(tempfile.mkdtemp(), "fp50")
pk =os.path.join(tempfile.mkdtemp(), "packet50.csv")
with open(pk, "w", encoding="utf-8-sig", newline="") as f:   # a BOM, as Excel saves it
    w = csv.writer(f)
    w.writerow(a[0])
    w.writerows(a[1:51])
cp = gc.first_n(src, cut_dir, 50, packet_csv=pk)
with open(cp["key.csv"], encoding="utf-8") as f:
    key50 = list(csv.DictReader(f))
eq([k["item_id"] for k in key50], [k["item_id"] for k in key[:50]], "key.csv keeps the first 50 rows, in order")
m50 = json.load(open(cp["manifest.json"], encoding="utf-8"))
k50 = sum(k["flagged"] == "1" for k in key[:50])
eq((m50["stats"]["drawn"], m50["stats"]["drawn_flagged"], m50["stats"]["sampling_ratio"]),
   (50, k50, round(k50 / 50, 4)), "drawn, drawn_flagged and sampling_ratio recounted for 50")
eq((m50["stats"]["drawn_of_100"], m50["stats"]["drawn_flagged_of_100"], m50["stats"]["base_rate"]),
   (100, 45, 0.3), "the full draw and the natural base rate are kept")
eq(m50["stats"]["in_flagged_range"], 20 <= k50 <= 25, "the flagged range is scaled to 50 (20 to 25)")
eq((m50["seed"], m50["measure"]), (m["seed"], m["measure"]), "seed and measure unchanged")
check(m50["cut"].startswith("first 50 of the shuffled 100"), "the manifest says it is a cut")
try:
    gc.first_n(src, cut_dir, 50)
    check(False, "an existing key.csv is not overwritten  (no ValueError)")
except ValueError:
    check(True, "an existing key.csv is not overwritten")
with open(pk, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(a[0])
    w.writerows([a[2], a[1]] + a[3:51])
try:
    gc.first_n(src, os.path.join(tempfile.mkdtemp(), "x"), 50, packet_csv=pk)
    check(False, "a packet in another order is refused  (no ValueError)")
except ValueError:
    check(True, "a packet in another order is refused")
try:
    gc.first_n(src, os.path.join(tempfile.mkdtemp(), "x"), 101)
    check(False, "more items than drawn is refused  (no ValueError)")
except ValueError:
    check(True, "more items than drawn is refused")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import agreement as agreement_cli  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    eq(agreement_cli.main(["cut", src, "--out", os.path.join(tempfile.mkdtemp(), "c"), "--n", "10"]), 0,
       "scripts/agreement.py cut: exit 0")
    eq(agreement_cli.main(["cut", src, "--out", cut_dir]), 1, "scripts/agreement.py cut: a refusal exits 1")

print("6. pilot 2 verifier output, if downloaded")
P2 = os.path.join(ROOT, "results", "bexec_pilot2_outputs")
if not os.path.isdir(P2):
    print("   SKIP results/bexec_pilot2_outputs not present")
else:
    by_task = gc.build_units(gc.read_outputs(P2))
    check(set(by_task) >= {"false_presuppositions", "historical_events",
                           "rationalization_binary", "rationalization_numerical",
                           "scientific_attribution"}, "units for all five verified tasks")
    check(all(len({u.unit_id for u in v}) == len(v) for v in by_task.values()),
          "unit ids are unique within each task")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all tests passed")
