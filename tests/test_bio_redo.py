"""Tests for hallutrace/bio_redo.py (re-scoring changed Biographies rows). No network, no GPU."""

import os
import sys
import tempfile

from hallutrace import verifier_inputs as bx
from hallutrace import bio_redo as br

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


U = bx.WorkUnit("m-b00001", "biographies", "m", 0, [])
COLS = bx.INPUT_COLUMNS["biographies"]


def meta(gid, sent, tail="", outcome="ok"):
    return {"generation_id": gid, "sent": sent, "stripped_tail": tail, "outcome": outcome}


def row(gid, name, text):
    return {"name": name, "response": text, "model": "m", "sample_index": 0, "generation_id": gid}


# New rule: g1 capped, still stripped; g2 stopped, now whole; g3 stopped and was
# emptied before, now sent; g4 unchanged and whole; g5 an abstention, never sent.
new = bx.Prepared(U, COLS,
                  [row("g1", "A", "a."), row("g2", "B", "b. last"), row("g3", "C", "c"), row("g4", "D", "d.")],
                  [meta("g1", 1, " cut"), meta("g2", 1), meta("g3", 1), meta("g4", 1),
                   meta("g5", 0, outcome="abstention")])
old_meta = [meta("g1", "1", " cut"), meta("g2", "1", " last"), meta("g3", "0", "c", "emptied"),
            meta("g4", "1"), meta("g5", "0", outcome="abstention")]

print("1. which rows changed")
gids = br.changed(new, old_meta)
eq(gids, {"g2", "g3"}, "a stop row that lost its tail, and one that was emptied; capped and unchanged rows stay")
raises(lambda: br.changed(new, old_meta[:-1]), "a generation missing from the old sidecar is refused")
raises(lambda: br.changed(bx.Prepared(U, COLS, [], [meta("g1", 0)]), [meta("g1", "1")]),
       "a row sent before and not now is refused, not silently dropped")

print("2. only the changed rows go to Gemma")
sub = br.subset(new, gids)
eq([r["name"] for r in sub.rows], ["B", "C"], "subset rows")
eq([m["generation_id"] for m in sub.meta], ["g2", "g3"], "subset sidecar")

print("3. merging back into the unit's output")
old_rows = [{"name": "A", "atoms": "a-old"}, {"name": "B", "atoms": "b-stripped"}, {"name": "D", "atoms": "d-old"}]
new_rows = [{"name": "C", "atoms": "c-new"}, {"name": "B", "atoms": "b-whole"}]
merged = br.merge(old_rows, new_rows, new, gids)
eq([(r["name"], r["atoms"]) for r in merged],
   [("A", "a-old"), ("B", "b-whole"), ("C", "c-new"), ("D", "d-old")],
   "redo replaces B, adds C, keeps A and D, in the new input order")
raises(lambda: br.merge(old_rows, new_rows[:1], new, gids), "a changed row with no redo output is refused")
raises(lambda: br.merge(old_rows, new_rows + [{"name": "A", "atoms": "x"}], new, gids),
       "a redo row for a name that was not re-sent is refused")
raises(lambda: br.merge(old_rows + [{"name": "Z", "atoms": "z"}], new_rows, new, gids),
       "an old row for a name no longer sent is refused")

print("4. files")
d = tempfile.mkdtemp()
p = os.path.join(d, "x", "out.csv")
br.write_rows(p, merged, ["name", "atoms"])
eq([r["name"] for r in br.read_csv(p)], ["A", "B", "C", "D"], "written and read back")
open(os.path.join(d, "empty.csv"), "w").close()
eq(br.read_csv(os.path.join(d, "empty.csv")), [], "an empty unit file reads as no rows")
eq(br.read_csv(os.path.join(d, "missing.csv")), [], "a missing file reads as no rows")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
