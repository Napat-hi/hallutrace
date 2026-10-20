"""Tests for hallutrace/hand_decomposition_score.py and scripts/hand_decomposition_score.py. Synthetic files, no network."""

import csv
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout

from hallutrace import hand_decomposition_score as es

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import hand_decomposition_score as cli  # noqa: E402

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
    except es.JoinError:
        check(True, label)
        return
    check(False, label + "  (no JoinError)")


print("1. reading the atoms")
eq(es.human_atoms("- Ada Zed was born in 1850.\n\n2) Ada Zed was a painter.\n  Ada Zed lived in Ely.  \n"),
   ["Ada Zed was born in 1850.", "Ada Zed was a painter.", "Ada Zed lived in Ely."],
   "one per line; bullets, numbers and blank lines dropped")
eq(es.human_atoms(""), [], "an empty cell is zero atoms, not a missing row")
row = {"atoms": "| He was born in 1850.\tTrue |\n| He was a painter.\tFalse |\n",
       "dropped atoms": json.dumps({"header": ["He is famous.\n\nPlease breakdown the following sentence"],
                                    "invented": ["The dog barked."], "refusal": []})}
eq(es.machine_atoms_bio(row), ["He was born in 1850.", "He was a painter."], "filtered: the kept atoms")
eq(len(es.machine_atoms_bio(row, "raw")), 4, "raw: the dropped atoms added back")

print("2. the provisional similarity")
eq(es.tokens("He represents the city of Ely.", "Ada Zed"), es.tokens("Ada Zed represented Ely city.", "Ada Zed"),
   "the subject's name, pronouns, stopwords and suffixes don't count")
eq(es.token_overlap("Ada Zed was born in 1850.", "She was born in 1850.", "Ada Zed"), 1.0,
   "a pronoun and the name are the same subject")
eq(es.token_overlap("He was a painter.", "He was born in 1850.", "Ada Zed"), 0.0, "disjoint content: 0")
eq(es.coverage("He was a painter.", "Ada Zed was a painter and a poet.", "Ada Zed"), 1.0,
   "coverage: a finer atom contained in a broader one")

print("3. one-to-one matching")
machine = ["He was a painter.", "He was a painter.", "He was born in 1850."]
human = ["Ada Zed was a painter.", "Ada Zed lived in Ely."]
eq(es.match(machine, human, "Ada Zed"), [(0, 0, 1.0)], "each human atom matches one machine atom at most")
eq(es.match(["He painted portraits of horses."], ["Ada Zed painted horses."], "Ada Zed", threshold=0.9), [],
   "below the threshold: no pair")
eq(es.match(["He painted horses in Ely.", "He painted horses."], ["Ada Zed painted horses."], "Ada Zed"),
   [(1, 0, 1.0)], "the best-scoring pair is taken first")

print("4. scores, against a hand count")
items = [{"item_id": "a", "model": "m1", "topic": "Ada Zed",
          "machine": ["He was a painter.", "He was born in 1850.", "He lived in Ely.", "He had a dog."],
          "human": ["Ada Zed was a painter.", "Ada Zed was born in 1850."]},
         {"item_id": "d21", "model": "m2", "topic": "Bo Lee", "machine": ["X.", "Y.", "Z."], "human": []}]
res = es.score(items)
a, d = res["items"]
eq((a["matched"], a["precision"], a["recall"], a["f1"]), (2, 0.5, 1.0, 0.6667), "item a: 2 of 4 machine, 2 of 2 human")
eq((d["precision"], d["recall"], d["f1"]), (0.0, None, None), "no human atoms: precision 0, recall undefined")
o = res["overall"]
eq((o["matched"], o["machine_atoms"], o["human_atoms"]), (2, 7, 2), "micro sums")
eq(o["micro"], {"precision": 0.2857, "recall": 1.0, "f1": 0.4444}, "micro P/R/F1")
eq((o["macro"]["precision"], o["macro"]["n_precision"], o["macro"]["n_recall"]), (0.25, 2, 1),
   "macro averages each value over the items where it is defined")
eq(sorted(res["by_model"]), ["m1", "m2"], "per model")

print("4b. containment, many-to-one")
eq(es.contained("Ada Zed was British.", "Ada Zed was a British actor.", "Ada Zed"), 1.0,
   "a finer machine atom inside a human atom")
eq(es.contained("Ada Zed was a British actor.", "Ada Zed was British.", "Ada Zed"), 1.0,
   "either way round: a human atom inside a broader machine atom")
eq(es.contained("Ada Zed was a person.", "Ada Zed was a British actor.", "Ada Zed"), 0.0,
   "a filler atom the human never wrote")
eq(es.contained("Ada Zed was not British.", "Ada Zed was a British actor.", "Ada Zed"), 0.0,
   "one side negated: no match")
eq(es.contained("He was not British.", "Ada Zed was British.", "Ada Zed"), 0.0,
   "one side negated: no match, though every word of the shorter is in the longer")
eq(es.contained("He wasn't British.", "Ada Zed was not British.", "Ada Zed"), 1.0,
   "both negated, one as a contraction: a match")

print("4b2. the matcher's word rules")
eq(es.contained("He played two roles.", "Ada Zed played a role.", "Ada Zed"), 1.0,
   "a plural in -es after a vowel meets its singular (roles, role)")
eq([es._stem(w) for w in ("passes", "boxes", "studies", "releases", "released", "release")],
   ["pass", "box", "study", "releas", "releas", "releas"], "-es after s, x, z, ch, sh; -ies; a final e")
eq({es._stem(w) for w in ("note", "notes", "noted")}, {"note"}, "note, notes and noted meet, and none is 'not'")
eq(es.contained("Ada Zed's mother was a painter and poet.", "Her mother was a painter and poet.", "Ada Zed"), 1.0,
   "a possessive leaves no stray 's'")
eq(dict(es.tokens("He wasn’t married.")), {"not": 1, "marri": 1}, "a curly contraction opens to 'not'")
eq(es.contained("He is a speaker.", "Ada Zed is a speaker.", "Ada Zed"), 1.0, "'speaker' alone is a content word")
eq(es.tokens("The speaker's eyes were blue."), es.tokens("My eyes were blue."), "'the speaker's' is the subject, as 'my' is")
eq(dict(es.tokens("He will release an album.", "Will Smith")), {"will": 1, "releas": 1, "album": 1},
   "a word of the name in lower case is not the name")
eq(dict(es.tokens("Will Smith released an album.", "Will Smith")), {"releas": 1, "album": 1},
   "the capitalised name is dropped")
split = [{"item_id": "s", "model": "m1", "topic": "Ada Zed",
          "machine": ["He was British.", "He was an actor.", "He was a person."],
          "human": ["Ada Zed was a British actor.", "Ada Zed lived in Ely."]},
         {"item_id": "d21", "model": "m2", "topic": "Bo Lee", "machine": ["X."], "human": []}]
cres = es.score_containment(split)
s0, s1 = cres["items"]
eq((s0["matched_machine"], s0["matched_human"], s0["precision"], s0["recall"]), (2, 1, 0.6667, 0.5),
   "two machine atoms inside one human atom both count; the filler and Ely don't")
eq((s1["precision"], s1["recall"]), (0.0, None), "no human atoms: precision 0, recall undefined")
eq(cres["overall"]["micro"], {"precision": 0.5, "recall": 0.5, "f1": 0.5}, "micro over both items")
eq((cres["threshold"], es.P0_10_1_THRESHOLD, es.SENSITIVITY_THRESHOLD), (0.8, 0.8, 0.6), "headline 0.8, sensitivity 0.6")
eq(s0["machine_best"][0], (0, 1.0), "each machine atom keeps its best human atom, for the hand check")
eq(es.score_containment(split, threshold=0.5)["items"][0]["matched_machine"], 2,
   "the threshold is a parameter")

print("4c. intervals for T4: bootstrap over items")
same = [{"machine": 4, "human": 2, "matched_machine": 2, "matched_human": 1}] * 5
eq(es.bootstrap_micro(same), {"precision": [0.5, 0.5], "recall": [0.5, 0.5], "f1": [0.5, 0.5]},
   "identical items: the interval is the point")
mixed = [{"machine": 4, "human": 2, "matched_machine": k, "matched_human": k // 2} for k in (0, 1, 2, 3, 4)]
ci = es.bootstrap_micro(mixed)
p, r, f = es._micro(mixed)
check(all(ci[k][0] <= v <= ci[k][1] for k, v in (("precision", p), ("recall", r), ("f1", f))),
      "each interval holds its point estimate")
eq(es.bootstrap_micro(mixed), ci, "seeded: same items, same interval")
one = [{"machine": 3, "human": 2, "matched": 1}, {"machine": 2, "human": 2, "matched": 2}]
eq(es._micro(one)[:2], (0.6, 0.75), "one-to-one rows: `matched` counts on both sides")
eq(es.bootstrap_micro(one[:1]), {"precision": None, "recall": None, "f1": None}, "one item: no interval")

print("5. the join to the verification output")


def write(path, cols, rows, bom=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


root = tempfile.mkdtemp()
verifier = os.path.join(root, "verifier")
write(os.path.join(verifier, "inputs", "m1", "m1-b00001.csv"), ["name", "response", "generation_id"],
      [{"name": "Ada Zed", "response": "x", "generation_id": "m1|biographies-00001|0"},
       {"name": "Bo Lee", "response": "y", "generation_id": "m1|biographies-00002|0"}])
write(os.path.join(verifier, "outputs_filtered", "m1", "m1-b00001.csv"),
      ["name", "response", "atoms", "hallucinated atoms", "dropped atoms"],
      [{"name": "Ada Zed", "response": "Ada Zed was a painter.", "atoms": "| He was a painter.\tTrue |\n",
        "hallucinated atoms": "", "dropped atoms": json.dumps({"header": [], "invented": ["The dog barked."], "refusal": []})},
       {"name": "Bo Lee", "response": "Bo Lee sang.", "atoms": "| He sang.\tTrue |\n",
        "hallucinated atoms": "", "dropped atoms": "{}"}])
packet = os.path.join(root, "packet")
write(os.path.join(packet, "key.csv"), ["item_id", "generation_id", "prompt_id", "model", "sample_index", "unit_id"],
      [{"item_id": "biographies-d01", "generation_id": "m1|biographies-00001|0", "prompt_id": "biographies-00001",
        "model": "m1", "sample_index": "0", "unit_id": "m1-b00001"},
       {"item_id": "biographies-d02", "generation_id": "m1|biographies-00002|0", "prompt_id": "biographies-00002",
        "model": "m1", "sample_index": "0", "unit_id": "m1-b00001"}])
filled = [{"item_id": "biographies-d01", "topic": "Ada Zed", "response": "Ada Zed was a painter.",
           "human_atoms": "Ada Zed was a painter.", "notes": ""},
          {"item_id": "biographies-d02", "topic": "Bo Lee", "response": "Bo Lee sang!",
           "human_atoms": "Bo Lee sang.", "notes": ""}]
write(os.path.join(packet, "annotator.csv"), ["item_id", "topic", "response", "human_atoms", "notes"], filled, bom=True)
items, notes = es.load_bio_items(os.path.join(packet, "key.csv"), os.path.join(packet, "annotator.csv"), verifier)
eq([(i["item_id"], i["machine"], i["human"]) for i in items],
   [("biographies-d01", ["He was a painter."], ["Ada Zed was a painter."]),
    ("biographies-d02", ["He sang."], ["Bo Lee sang."])],
   "joined by generation_id through the unit's input file; a BOM-saved packet reads")
eq(notes, ["biographies-d02: the verifier's response text differs from the packet's"],
   "a response the verifier saw differently is noted")
eq(es.load_bio_items(os.path.join(packet, "key.csv"), os.path.join(packet, "annotator.csv"), verifier, "raw")[0][0]["machine"],
   ["He was a painter.", "The dog barked."], "raw variant")
write(os.path.join(packet, "annotator.csv"), ["item_id", "topic", "response", "human_atoms", "notes"],
      [dict(filled[0], topic="Someone Else"), filled[1]])
raises(lambda: es.load_bio_items(os.path.join(packet, "key.csv"), os.path.join(packet, "annotator.csv"), verifier),
       "a topic that isn't the verifier row's name is refused")
write(os.path.join(packet, "annotator.csv"), ["item_id", "topic", "response", "human_atoms", "notes"], filled)
raises(lambda: es.load_bio_items(os.path.join(packet, "key.csv"), os.path.join(packet, "annotator.csv"),
                                 os.path.join(root, "nowhere")),
       "a missing unit file is refused")

print("5b. the join for Summarization and Simplification (by input_row)")
v2 = os.path.join(root, "v2")
write(os.path.join(v2, "inputs", "m1", "m1-b00002.meta.csv"),
      ["generation_id", "prompt_id", "model", "sample_index", "task", "sent", "input_row"],
      [{"generation_id": "m1|summarization-00001|0", "prompt_id": "summarization-00001", "model": "m1",
        "sample_index": "0", "task": "summarization", "sent": "1", "input_row": "1"},
       {"generation_id": "m1|summarization-00002|0", "prompt_id": "summarization-00002", "model": "m1",
        "sample_index": "0", "task": "summarization", "sent": "0", "input_row": ""},
       {"generation_id": "m1|summarization-00003|0", "prompt_id": "summarization-00003", "model": "m1",
        "sample_index": "0", "task": "summarization", "sent": "1", "input_row": "0"}])
write(os.path.join(v2, "outputs", "m1", "m1-b00002.csv"),
      ["passage", "response", "atomic_units", "hallucinated_atomic_units"],
      [{"passage": "p0", "response": "Cats sleep.", "atomic_units": " Cats sleep. | True |", "hallucinated_atomic_units": ""},
       {"passage": "p1", "response": "Dogs bark loudly.",
        "atomic_units": "Here are the independent facts broken down from the passage: | True |\n Dogs bark. | True |\n Dogs are loud. | False |",
        "hallucinated_atomic_units": ""}])
spacket = os.path.join(root, "spacket")
skey = [{"item_id": "summarization-d01", "generation_id": "m1|summarization-00001|0", "prompt_id": "summarization-00001",
         "model": "m1", "sample_index": "0", "unit_id": "m1-b00002"}]
write(os.path.join(spacket, "key.csv"), ["item_id", "generation_id", "prompt_id", "model", "sample_index", "unit_id"], skey)
write(os.path.join(spacket, "annotator.csv"), ["item_id", "topic", "response", "human_atoms", "notes"],
      [{"item_id": "summarization-d01", "topic": "", "response": "Dogs bark loudly.",
        "human_atoms": "Dogs bark loudly.", "notes": ""}], bom=True)
sitems, snotes = es.load_items("summarization", os.path.join(spacket, "key.csv"),
                               os.path.join(spacket, "annotator.csv"), v2)
eq((sitems[0]["machine"], sitems[0]["human"], snotes), (["Dogs bark.", "Dogs are loud."], ["Dogs bark loudly."], []),
   "the sidecar's input_row picks the output row; filtered drops the framing line")
eq(len(es.load_items("summarization", os.path.join(spacket, "key.csv"), os.path.join(spacket, "annotator.csv"),
                     v2, "raw")[0][0]["machine"]), 3, "raw keeps the framing line")
write(os.path.join(spacket, "key.csv"), ["item_id", "generation_id", "prompt_id", "model", "sample_index", "unit_id"],
      [dict(skey[0], generation_id="m1|summarization-00002|0")])
raises(lambda: es.load_items("summarization", os.path.join(spacket, "key.csv"), os.path.join(spacket, "annotator.csv"), v2),
       "a generation never sent to the decomposer is refused")
write(os.path.join(spacket, "key.csv"), ["item_id", "generation_id", "prompt_id", "model", "sample_index", "unit_id"], skey)

print("6. the command line")
out = os.path.join(root, "out")
buf = io.StringIO()
with redirect_stdout(buf), redirect_stderr(buf):
    code = cli.main(["score", "biographies", "--verifier-root", verifier, "--packet", packet, "--out", out])
eq(code, 0, "exit 0")
check(all(os.path.exists(os.path.join(out, "pairs_%s_%s.csv" % (v, r[0]))) for v in es.VARIANTS for r in cli.READINGS)
      and os.path.exists(os.path.join(out, "scores.json")), "scores.json and the pairs for every variant and reading")
s = json.load(open(os.path.join(out, "scores.json"), encoding="utf-8"))
eq(s["filtered"]["containment_0.8"]["overall"]["micro"]["recall"], 1.0, "both human atoms recalled")
eq(s["filtered"]["containment_0.8"]["matcher"], "containment (many-to-one)", "the headline reading is containment")
eq(len(s["filtered"]["join_notes"]), 1, "the join note is kept with the scores")
chk = os.path.join(root, "check.csv")
shutil.copytree(packet, os.path.join(root, "packets", "biographies"))
with redirect_stdout(buf), redirect_stderr(buf):
    eq(cli.main(["check", "--root", "biographies=" + verifier, "--packets", os.path.join(root, "packets"),
                 "--n", "2", "--out", chk]), 0, "check: exit 0")
crow = list(csv.DictReader(open(chk, encoding="utf-8-sig")))
check(len(crow) >= 1 and all(r["you_say"] == "" for r in crow), "check: pairs drawn, you_say left blank for a person")
check("matcher_says" not in crow[0] and "score" not in crow[0], "check: the packet hides the matcher's call")
ckey = list(csv.DictReader(open(os.path.join(root, "check_key.csv"), encoding="utf-8-sig")))
check([r["check_id"] for r in ckey] == [r["check_id"] for r in crow]
      and {r["matcher_says"] for r in ckey} <= {"inside", "not inside"}, "check: the call goes to the key file")
with redirect_stdout(buf), redirect_stderr(buf):
    eq(cli.main(["check", "--root", "nosuchtask=" + verifier, "--out", chk]), 1, "check: an unknown task is refused")
t4 = os.path.join(root, "t4.csv")
with redirect_stdout(buf), redirect_stderr(buf):
    eq(cli.main(["t4", "--root", "biographies=" + verifier, "--packets", os.path.join(root, "packets"),
                 "--out", t4]), 0, "t4: exit 0")
trows = list(csv.DictReader(open(t4, encoding="utf-8")))
eq([(r["variant"], r["reading"]) for r in trows], [(v, n) for v in es.VARIANTS for n, _, _ in cli.READINGS],
   "t4: one row per variant and reading")
eq(list(trows[0]), cli.T4_COLUMNS, "t4: the columns the table reads")
eq(trows[0]["recall"], str(s["filtered"]["containment_0.8"]["overall"]["micro"]["recall"]),
   "t4: the headline row matches `score`")
with redirect_stdout(buf), redirect_stderr(buf):
    eq(cli.main(["t4", "--root", "nosuchtask=" + verifier, "--out", t4]), 1, "t4: an unknown task is refused")
for cmd in ("check", "t4"):
    out_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(out_buf):
        cli.main([cmd, "--root", "biographies=" + verifier, "--packets", os.path.join(root, "packets"),
                  "--out", os.path.join(root, "notes_%s.csv" % cmd)] + (["--n", "2"] if cmd == "check" else []))
    check("biographies, filtered, note:" in out_buf.getvalue(), "%s prints the join notes, as `score` does" % cmd)

# rescore: the matcher's columns and counts rebuilt, the people's columns and the notes kept
shutil.copytree(spacket, os.path.join(root, "packets", "summarization"))
write(os.path.join(v2, "inputs", "m1", "m1-b00003.meta.csv"),
      ["generation_id", "prompt_id", "model", "sample_index", "task", "sent", "input_row"],
      [{"generation_id": "m1|simplification-00001|0", "prompt_id": "simplification-00001", "model": "m1",
        "sample_index": "0", "task": "simplification", "sent": "1", "input_row": "0"}])
write(os.path.join(v2, "outputs", "m1", "m1-b00003.csv"),
      ["passage", "response", "atomic_units", "hallucinated_atomic_units"],
      [{"passage": "p0", "response": "Birds fly.", "atomic_units": " Birds fly. | True |",
        "hallucinated_atomic_units": ""}])
write(os.path.join(root, "packets", "simplification", "key.csv"),
      ["item_id", "generation_id", "prompt_id", "model", "sample_index", "unit_id"],
      [{"item_id": "simplification-d01", "generation_id": "m1|simplification-00001|0",
        "prompt_id": "simplification-00001", "model": "m1", "sample_index": "0", "unit_id": "m1-b00003"}])
write(os.path.join(root, "packets", "simplification", "annotator.csv"),
      ["item_id", "topic", "response", "human_atoms", "notes"],
      [{"item_id": "simplification-d01", "topic": "", "response": "Birds fly.", "human_atoms": "Birds fly.",
        "notes": ""}])
rescore = ["rescore", "--root", "biographies=" + verifier, "--root", "summarization=" + v2,
           "--root", "simplification=" + v2, "--packets", os.path.join(root, "packets")]
res_dir = os.path.join(root, "results")
os.makedirs(res_dir)
pairs = [dict(k, machine_atom=c["machine_atom"], human_atom=c["human_atom"])
         for k, c in zip(ckey, crow)]
set_cols = ["check_id", "task", "item_id", "model", "machine_atom", "human_atom", "score", "matcher_says_0_8"]


def write_set1(**change):
    write(os.path.join(res_dir, "stress_test.csv"), set_cols + ["rater1_pass1", "rater1_pass2", "note"],
          [dict(dict({k: p[k] for k in set_cols[:6]}, score="0.123", matcher_says_0_8="stale",
                     rater1_pass1="inside", rater1_pass2="not inside", note="kept"), **change) for p in pairs])


def write_set2(**change):
    write(os.path.join(res_dir, "stress_test_set2.csv"), set_cols + ["rater2", "note"],
          [dict(dict({k: p[k] for k in set_cols[:6]}, score="0.123", matcher_says_0_8="stale",
                     rater2="inside", note=""), **change) for p in pairs])


write_set1()
write_set2()
json.dump({"stress_test": {"seed": 7, "pass2": {"note": "kept"}}, "matcher_note": "kept"},
          open(os.path.join(res_dir, "scores.json"), "w", encoding="utf-8"))
with redirect_stdout(buf), redirect_stderr(buf):
    eq(cli.main(rescore + ["--results", res_dir]), 0, "rescore: exit 0")
r1 = list(csv.DictReader(open(os.path.join(res_dir, "stress_test.csv"), encoding="utf-8-sig")))
check(len(r1) == len(ckey) and all(float(r["score"]) == float(k["score"]) for r, k in zip(r1, ckey)),
      "rescore: each score is the matcher's (as `check` scored the same pair)")
check(all(r["matcher_says_0_8"] == ("inside" if float(r["score"]) >= 0.8 else "not inside") for r in r1),
      "rescore: each call follows its score at 0.8")
eq({(r["rater1_pass1"], r["rater1_pass2"], r["note"]) for r in r1}, {("inside", "not inside", "kept")},
   "rescore: the people's columns and notes are untouched")
sj = json.load(open(os.path.join(res_dir, "scores.json"), encoding="utf-8"))
eq((sj["stress_test"]["seed"], sj["stress_test"]["pass2"]["note"], sj["matcher_note"]), (7, "kept", "kept"),
   "rescore: the notes in scores.json are kept")
eq(sj["scores"]["biographies"]["filtered"]["containment_0.8"]["overall"],
   s["filtered"]["containment_0.8"]["overall"], "rescore: scores match `score`")
inside = sum(float(r["score"]) >= 0.8 for r in r1)
eq(sj["stress_test"]["pass1"]["by_threshold"]["0.8"],
   {"agree": inside, "n": len(r1), "matcher_inside_person_not": 0, "matcher_not_person_inside": len(r1) - inside},
   "rescore: pass 1 counted against a person who said inside every time")
eq(sj["stress_test_combined"]["0.8"]["n"], 2 * len(r1), "rescore: the combined count adds pass 2 and set 2")
eq(sorted(sj["scores"]), sorted(es.TASKS), "rescore: scores for every task")

# Every refusal leaves all three files as they were. Set 1 goes back to its stale values
# first, so a run that wrote set 1 before refusing set 2 would show.
write_set1()
names = ("stress_test.csv", "stress_test_set2.csv", "scores.json")


def refused(argv, what):
    before = [open(os.path.join(res_dir, n), "rb").read() for n in names]
    with redirect_stdout(buf), redirect_stderr(buf):
        eq(cli.main(argv), 1, "rescore: %s is refused" % what)
    eq([open(os.path.join(res_dir, n), "rb").read() for n in names], before,
       "rescore: after %s, both stress files and scores.json are unchanged" % what)


refused(["rescore", "--root", "biographies=" + verifier, "--packets", os.path.join(root, "packets"),
         "--results", res_dir], "a task with no --root")
write_set2(item_id="biographies-d99")
refused(rescore + ["--results", res_dir], "a set 2 pair whose item is not in its packet")
write_set2(task="summarization")
refused(rescore + ["--results", res_dir], "a pair filed under another task")
write_set2(machine_atom="An atom from another run.")
refused(rescore + ["--results", res_dir], "a machine atom not among the item's filtered atoms")
write_set2(rater2="maybe")
refused(rescore + ["--results", res_dir], "a person's call other than inside or not inside")
write_set2(rater2="")
refused(rescore + ["--results", res_dir], "a blank call")
write_set2()
write_set1(rater1_pass2="Inside!")
refused(rescore + ["--results", res_dir], "a misspelt call in set 1")
write_set1()
with redirect_stdout(buf), redirect_stderr(buf):
    eq(cli.main(rescore + ["--results", res_dir]), 0, "rescore: the restored files rescore again")
write(os.path.join(packet, "annotator.csv"), ["item_id", "topic", "response", "human_atoms", "notes"],
      [dict(r, human_atoms="") for r in filled])
with redirect_stdout(buf), redirect_stderr(buf):
    eq(cli.main(["score", "biographies", "--verifier-root", verifier, "--packet", packet, "--out", out]), 1,
       "an unfilled packet: exit 1")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
