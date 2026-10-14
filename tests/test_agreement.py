"""Tests for hallutrace/agreement.py. Synthetic packets, no key, no network."""

import csv
import json
import os
import sys
import tempfile

from sklearn.metrics import cohen_kappa_score

from hallutrace import agreement as ag

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
    except ag.PacketError:
        check(True, label)
        return
    check(False, label + "  (no PacketError)")


H, S, U = "hallucinated", "supported", "unverifiable"


def packet(a, b, verifier, task="biographies", notes_a=None, notes_b=None,
           base_rate=0.5, ratio=0.5, b_ids=None, tag_a=None, tag_b=None):
    """Write a packet the way gold_cut.write_packets lays it out; returns its dir.
    With tag_a/tag_b, the annotator files get a `tag` column."""
    d = tempfile.mkdtemp()
    n = len(a)
    ids = ["%s-%03d" % (task, i) for i in range(1, n + 1)]
    notes_a = notes_a or [""] * n
    notes_b = notes_b or [""] * n

    def write(name, cols, rows):
        with open(os.path.join(d, name), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    def ann(name, item_ids, labels, notes, tags):
        cols = ["item_id", "context_shown", "unit_to_judge", "question", "annotator_label"]
        cols += (["tag"] if tags is not None else []) + ["notes"]
        rows = [{"item_id": i, "context_shown": "c", "unit_to_judge": "u", "question": "q",
                 "annotator_label": l, "notes": t} for i, l, t in zip(item_ids, labels, notes)]
        for r, g in zip(rows, tags or []):
            r["tag"] = g
        write(name, cols, rows)

    ann("annotator_A.csv", ids, a, notes_a, tag_a)
    ann("annotator_B.csv", b_ids or ids, b, notes_b, tag_b)
    write("key.csv", ["item_id", "verifier_label", "flagged", "task"],
          [{"item_id": i, "verifier_label": v, "flagged": int(v != S), "task": task}
           for i, v in zip(ids, verifier)])
    with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"task": task, "stats": {"base_rate": base_rate, "sampling_ratio": ratio,
                                           "drawn": n, "drawn_flagged": sum(v != S for v in verifier)}}, f)
    return d


def fill_gold(d, gold, tags=None, columns=ag.ADJUDICATION_COLUMNS):
    path = os.path.join(d, "adjudication.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    for r in rows:
        r["gold_label"] = gold[r["item_id"]]
        r["tag"] = (tags or {}).get(r["item_id"], "")
        r["reason"] = "agreed " + gold[r["item_id"]]
        for k in list(r):
            if k not in columns:
                del r[k]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


print("1. inter-annotator kappa, against a hand count")
# 10 items, 8 agree; each annotator says H five times and S five times:
# p_o = 0.8, p_e = 0.5*0.5 + 0.5*0.5 = 0.5, kappa = (0.8 - 0.5) / 0.5 = 0.6
A = [H, H, H, H, H, S, S, S, S, S]
B = [H, H, H, H, S, S, S, S, S, H]
V = [H, H, H, H, H, S, S, S, S, S]
d = packet(A, B, V)
out = ag.compute(ag.load(d))
i = out["all_rows"]["inter_annotator"]
eq(i["kappa"], 0.6, "kappa 0.6")
eq(i["agreement"], 0.8, "raw agreement 0.8")
eq(i["matrix"]["cells"], [[4, 1, 0], [1, 4, 0], [0, 0, 0]],
   "3x3 with the unused unverifiable row and column kept as zeros")
eq(out["all_rows"]["verifier"]["gold_missing"], 2, "verifier waits for gold on the 2 disagreements")
check("kappa" not in out["all_rows"]["verifier"], "no verifier kappa before adjudication")

print("2. adjudication.csv, then the verifier's numbers")
ag.write_adjudication(d, ag.load(d))
adj = list(csv.DictReader(open(os.path.join(d, "adjudication.csv"), encoding="utf-8-sig")))
eq([r["item_id"] for r in adj], ["biographies-005", "biographies-010"], "only the rows that differ")
eq((adj[0]["label_A"], adj[0]["label_B"]), (H, S), "both original labels kept")
with open(os.path.join(d, "adjudication.csv"), "rb") as f:
    check(f.read(3) == b"\xef\xbb\xbf", "adjudication.csv starts with a BOM, so Excel opens it as UTF-8")
with open(os.path.join(d, "adjudication.csv"), encoding="utf-8-sig", newline="") as f:
    header = next(csv.reader(f))
eq(header, ["item_id", "label_A", "label_B", "tag_A", "tag_B", "notes_A", "notes_B", "gold_label", "tag",
            "reason"],
   "exactly these ten columns, gold's own tag included: the tie-breaker never sees the "
   "verifier's label")
fill_gold(d, {"biographies-005": H, "biographies-010": S})
ag.write_adjudication(d, ag.load(d))
adj = list(csv.DictReader(open(os.path.join(d, "adjudication.csv"), encoding="utf-8-sig")))
eq([(r["gold_label"], r["reason"]) for r in adj], [(H, "agreed hallucinated"), (S, "agreed supported")],
   "rerunning keeps the gold labels and reasons already written")
v = ag.compute(ag.load(d))["all_rows"]["verifier"]
eq((v["kappa"], v["agreement"], v["gate"]), (1.0, 1.0, "accept"), "verifier matches gold: kappa 1, accept")
eq(v["matrix"]["row_labels"], [H, S], "binary verifier: matrix rows are hallucinated and supported only")
eq(v["matrix"]["cells"], [[5, 0, 0], [0, 5, 0]], "2x3, verifier by gold")
eq(v["natural"]["kappa"], 1.0, "natural estimate given beside, not instead")

print("3. verifier against gold, against a hand count")
# gold = A (both agree everywhere); verifier wrong on 2 of 5 H and 0 of 5 S:
# p_o = 0.8; gold 5H 5S, verifier 3H 7S: p_e = 0.5*0.3 + 0.5*0.7 = 0.5, kappa = 0.6
d = packet(A, A, [H, H, H, S, S, S, S, S, S, S])
v = ag.compute(ag.load(d))["all_rows"]["verifier"]
eq((v["kappa"], v["agreement"], v["gate"]), (0.6, 0.8, "quarantine"), "kappa 0.6 -> quarantine")
eq(v["matrix"]["cells"], [[3, 0, 0], [2, 5, 0]], "missed hallucinations sit in the supported row")
lo, hi = v["ci95"]
check(lo <= 0.6 <= hi, "the bootstrap interval holds the point estimate (%s, %s)" % (lo, hi))
check(v["edge_crossing"] and 0.667 in v["edges_crossed"], "n = 10 is wide: the interval crosses 0.667")
eq(ag.compute(ag.load(d))["all_rows"]["verifier"]["ci95"], v["ci95"], "the bootstrap is seeded: same packet, same interval")

print("3b. the fast kappa used by the bootstrap matches sklearn")
for xs, ys in (([H, H, S, S, U], [H, S, S, S, U]), ([H, S, H, S], [S, H, S, H]), ([H, H, H, S], [H, H, H, H])):
    want = cohen_kappa_score(xs, ys, labels=ag.LABELS)
    check(abs(ag._kappa_fast(xs, ys) - want) < 1e-12, "fast kappa = sklearn on %s / %s" % (xs, ys))
eq(ag.edges_crossed([0.786, 0.957], (0.80, 0.90)), [0.80, 0.90], "45/50's Wilson interval crosses both accuracy edges")
eq(ag.edges_crossed([0.81, 0.95], (0.667, 0.80)), [], "an interval above both edges crosses none")

print("4. natural base rate reweighting")
# drawn 30% flagged, natural 75% flagged: flagged rows weigh 2.5, the rest 5/14
d = packet(A, A, [H, H, H, S, S, S, S, S, S, S], base_rate=0.75, ratio=0.3)
v = ag.compute(ag.load(d))["all_rows"]["verifier"]
# all 3 flagged rows agree (weight 2.5 each); of 7 unflagged, 5 agree (gold S) and 2 do not
want = round((3 * 2.5 + 5 * 5 / 14) / (3 * 2.5 + 7 * 5 / 14), 4)
eq(v["natural"]["agreement"], want, "natural agreement uses the weights")
eq(v["agreement"], 0.8, "stratified agreement unchanged")

print("5. the not-a-claim rows, a packet without a tag column: exact notes")
A2 = [H, H, H, S, S, U, U, U, S, H]
B2 = [H, H, H, S, S, U, U, H, S, S]
V2 = [H, H, H, S, S, S, H, H, S, S]
na = ["", "", "", "", "", "not a claim", " Not a claim ", "", "", ""]
nb = ["", "", "", "", "", "not a claim", "", "not a claim, I think", "", ""]
d = packet(A2, B2, V2, notes_a=na, notes_b=nb)
out = ag.compute(ag.load(d))
eq(out["not_a_claim"]["n"], 1, "dropped only when both wrote it; case and outer spaces ignored")
eq(out["not_a_claim"]["marked_by_one"], 1, "' Not a claim ' from A alone stays in; 'not a claim, I think' is no mark")
eq(out["not_a_claim"]["claim_rows"], 9, "claim rows, the n beside the 'without' kappa")
eq(out["not_a_claim"]["without"]["inter_annotator"]["n"], 9, "9 rows left without them")
check(out["not_a_claim"]["without"]["inter_annotator"]["kappa"] != out["all_rows"]["inter_annotator"]["kappa"],
      "kappa reported both ways")
eq(out["not_a_claim"]["mentions_not_a_claim"], ["biographies-008"],
   "'not a claim, I think' is listed, since it stays in the 'without' split")
eq(out["tag_source"], "notes, exact (no tag column)", "the fallback is reported")

print("5b. the tag column: tags from `tag`, notes free text")
ta = ["", "", "", "", "exaggerated", "not a claim", " Not a Claim ", "", "", "no article"]
tb = ["", "", "", "", "exaggerated", "not a claim", "", "not a claim", "", "exaggerated"]
nt_a = ["", "", "", "", "", "a question, not about him", "", "", "", ""]
nt_b = ["not a claim", "", "", "", "", "", "", "off topic, I think", "", ""]
d = packet(A2, B2, V2, notes_a=nt_a, notes_b=nt_b, tag_a=ta, tag_b=tb)
out = ag.compute(ag.load(d))
eq(out["tag_source"], "tag column", "tags read from the tag column")
eq(out["not_a_claim"]["n"], 1, "dropped when both tagged it, whatever notes say")
eq(out["not_a_claim"]["marked_by_one"], 2, "tagged by one annotator only: rows 7 and 8")
eq(out["not_a_claim"]["claim_rows"], 9, "claim rows beside the 'without' numbers")
eq(out["not_a_claim"]["mentions_not_a_claim"], ["biographies-001"],
   "notes saying 'not a claim' with no tag are listed, not counted")
eq(out["tags"]["exaggerated"], {"A": 1, "B": 2, "both": 1}, "exaggerated counted per annotator")
eq(out["tags"]["no article"], {"A": 1, "B": 0, "both": 0}, "no article counted too")
ag.write_adjudication(d, ag.load(d))
adj = list(csv.DictReader(open(os.path.join(d, "adjudication.csv"), encoding="utf-8-sig")))
eq([(r["item_id"], r["tag_A"], r["tag_B"]) for r in adj],
   [("biographies-008", "", "not a claim"), ("biographies-010", "no article", "exaggerated")],
   "adjudication shows both tags beside both labels")
raises(lambda: ag.load(packet(A2, B2, V2, tag_a=["not a claim, off topic"] + [""] * 9, tag_b=[""] * 10)),
       "a tag outside the three values is refused, not silently kept in")
raises(lambda: ag.load(packet(A2, B2, V2, tag_a=[""] * 10)), "a tag column in A only is refused")

print("5d. a row adjudication settles as not a claim leaves the verifier's 'without' split")
# Same packet as 5b. Disagreements: row 8 (U / H, tagged by B only) and row 10 (H / S).
# Gold settles row 8 as unverifiable + not a claim, row 10 as supported.
d = packet(A2, B2, V2, tag_a=ta, tag_b=tb)
ag.write_adjudication(d, ag.load(d))
fill_gold(d, {"biographies-008": U, "biographies-010": S}, tags={"biographies-008": "not a claim"})
out = ag.compute(ag.load(d))
nac = out["not_a_claim"]
eq(nac["settled_not_a_claim"], ["biographies-008"], "row 8 listed as settled at adjudication")
eq((nac["n"], nac["claim_rows"], nac["verifier_rows"]), (1, 9, 8),
   "both-tagged row 6 out of both; row 8 out of the verifier's numbers only")
eq(nac["without"]["inter_annotator"]["n"], 9, "inter-annotator 'without' keeps row 8 (both-tagged rule)")
eq(nac["without"]["verifier"]["n"], 8, "verifier 'without' drops rows 6 and 8")
eq(out["all_rows"]["verifier"]["n"], 10, "all rows, the gate's numbers, keep every row")
gold = [H, H, H, S, S, U, U, U, S, S]
keep = [i for i in range(10) if i not in (5, 7)]
want = round(float(cohen_kappa_score([gold[i] for i in keep], [V2[i] for i in keep], labels=ag.LABELS)), 4)
eq(nac["without"]["verifier"]["kappa"], want, "verifier 'without' kappa by hand on the 8 rows (%s)" % want)
eq(out["gold_tags"], {"not a claim": 1, "exaggerated": 0, "no article": 0},
   "gold's tags counted apart from the annotators'")
ag.write_adjudication(d, ag.load(d))
adj = list(csv.DictReader(open(os.path.join(d, "adjudication.csv"), encoding="utf-8-sig")))
eq([(r["item_id"], r["gold_label"], r["tag"]) for r in adj],
   [("biographies-008", U, "not a claim"), ("biographies-010", S, "")], "rerunning keeps gold's tag")
d2 = packet(A2, B2, V2, tag_a=ta, tag_b=tb)
ag.write_adjudication(d2, ag.load(d2))
fill_gold(d2, {"biographies-008": U, "biographies-010": S}, tags={"biographies-008": "not claim"})
raises(lambda: ag.load(d2), "a gold tag outside the three values is refused")
old_cols = tuple(c for c in ag.ADJUDICATION_COLUMNS if c != "tag")
fill_gold(d2, {"biographies-008": U, "biographies-010": S}, columns=old_cols)
nac = ag.compute(ag.load(d2))["not_a_claim"]
eq((nac["settled_not_a_claim"], nac["verifier_rows"]), ([], 9),
   "an adjudication.csv from before the column reads as no gold tags")

print("5c. files saved by Excel as CSV UTF-8 (with a BOM) load")
d = packet(A2, B2, V2, tag_a=ta, tag_b=tb)
for name in ("annotator_A.csv", "annotator_B.csv", "key.csv"):
    p = os.path.join(d, name)
    raw = open(p, encoding="utf-8").read()
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        f.write(raw)
try:
    got = ag.compute(ag.load(d))
    eq((got["labelled"], got["tag_source"], got["not_a_claim"]["n"]), (10, "tag column", 1),
       "a BOM before item_id is not part of the column name")
except (KeyError, ag.PacketError) as e:
    check(False, "a BOM before item_id is not part of the column name  (%r)" % e)

print("6. scientific_attribution keeps the 3x3")
d = packet([H, S, U, U], [H, S, U, U], [H, S, U, H], task="scientific_attribution")
v = ag.compute(ag.load(d))["all_rows"]["verifier"]
eq(v["matrix"]["cells"], [[1, 0, 1], [0, 1, 0], [0, 0, 1]], "3x3, verifier unverifiable row kept")
d = packet([H, S], [H, S], [U, S], task="summarization")
raises(lambda: ag.compute(ag.load(d)), "a binary task's key with unverifiable is refused")

print("7. the pilot: only the rows both have labelled")
d = packet([H, S, H, "", ""], [H, S, S, "", ""], [H, S, H, S, H])
out = ag.compute(ag.load(d))
eq((out["labelled"], out["all_rows"]["inter_annotator"]["n"]), (3, 3), "3 of 5 labelled")

print("8. a broken packet is refused")
raises(lambda: ag.load(packet([H, S], [H, ""], [H, S])), "a row labelled by one annotator")
raises(lambda: ag.load(packet([H, "maybe"], [H, S], [H, S])), "a label outside the three")
raises(lambda: ag.load(packet([H, S], [S, H], [H, S], b_ids=["biographies-002", "biographies-001"])),
       "A and B in a different order")

print("9. the gate's bands (verifier_validation.md, Threshold)")
eq([ag.band(k) for k in (0.8, 0.7999, 0.667, 0.6669, None)],
   ["accept", "tentative", "tentative", "quarantine", None], "0.800 accept, 0.667 tentative")

print("10. the CLI writes agreement.json")
d = packet(A, A, V)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import agreement as cli  # noqa: E402
eq(cli.main(["compute", d]), 0, "compute exits 0")
check(os.path.exists(os.path.join(d, "agreement.json")), "agreement.json next to the packet")

print("10b. annotators' committed files read from elsewhere (--a / --b)")
d = packet(A, B, V)
own = tempfile.mkdtemp()
for src, name in (("annotator_A.csv", "annotator_purin.csv"), ("annotator_B.csv", "annotator_earth.csv")):
    os.replace(os.path.join(d, src), os.path.join(own, name))
pa, pb = os.path.join(own, "annotator_purin.csv"), os.path.join(own, "annotator_earth.csv")
eq(ag.compute(ag.load(d, pa, pb))["all_rows"]["inter_annotator"]["kappa"], 0.6,
   "same kappa as test 1 with the files under their own names")
eq(cli.main(["compute", d, "--a", pa, "--b", pb]), 0, "compute --a --b exits 0")

print("11. the extraction tasks: extractor accuracy, no verifier kappa")
# The model was wrong on rows 1-3 (verifier hallucinated), but the extractor captured rows
# 1, 2 and 4 faithfully: accuracy is 3 of 4, whatever the verifier said.
d = packet([S, S, H, S], [S, S, H, S], [H, H, H, S], task="false_presuppositions")
out = ag.compute(ag.load(d))
e = out["all_rows"]["extractor"]
eq(out["measure"], "extractor_accuracy", "false_presuppositions measures the extractor")
check("verifier" not in out["all_rows"], "no verifier kappa against a different question")
eq((e["accuracy"], e["gold_counts"]), (0.75, {H: 1, S: 3, U: 0}), "3 of 4 faithful")
eq(e["gate"], "quarantine", "0.75 is below 0.80")
check("provisional" not in e["gate_note"] and "provisional" not in out["accuracy_thresholds"],
      "the band is final: nothing says provisional")
eq([ag.accuracy_band(a) for a in (0.9, 0.8999, 0.8, 0.7999, None)],
   ["accept", "tentative", "tentative", "quarantine", None], "0.90 accept, 0.80 tentative")
check(e["edge_crossing"], "3 of 4 has a wide interval, flagged as crossing a band edge")
lo, hi = ag.wilson(45, 50)
check(0.78 < lo < 0.80 and 0.95 < hi < 0.97, "Wilson 95%% CI for 45/50 is about 0.79 to 0.96 (got %s, %s)" % (lo, hi))
d = packet([S, H], [S, H], [S, H], task="historical_events")
eq(ag.compute(ag.load(d))["measure"], "kappa", "historical_events keeps the verifier kappa")
with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
    m = json.load(f)
m["measure"] = "extractor_accuracy"
with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(m, f)
eq(ag.compute(ag.load(d))["measure"], "extractor_accuracy", "a manifest can name the measure")

m["measure"] = "verifier_kappa"
with open(os.path.join(d, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(m, f)
raises(lambda: ag.compute(ag.load(d)), "an unknown measure name is refused")

print("12. a packet written by gold_cut.write_packets reads end to end")
from hallutrace import gold_cut as gc  # noqa: E402
for task, want in (("false_presuppositions", "extractor_accuracy"), ("historical_events", "kappa")):
    units = [gc.Unit("%s|%s-%05d|0:0" % ("m", task, i), "m|%s-%05d|0" % (task, i), task, "resp",
                     "unit", gc.HALLUCINATED if i % 2 else gc.SUPPORTED) for i in range(10)]
    out_dir = tempfile.mkdtemp()
    gc.write_packets(gc.draw(task, units, "s", n=10), out_dir)
    tdir = os.path.join(out_dir, task)
    for name in ("annotator_A.csv", "annotator_B.csv"):
        p = os.path.join(tdir, name)
        rows = list(csv.DictReader(open(p, encoding="utf-8", newline="")))
        for r in rows:
            r["annotator_label"] = S
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    got = ag.compute(ag.load(tdir))
    eq((got["measure"], got["labelled"]), (want, 10), "%s: measure from gold_cut's manifest" % task)

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
