"""Tests for the claim_key alias table: data/alias_table.json,
data/alias_overrides.json and data/alias_build/wikidata_snapshot.json,
built by scripts/build_alias_table.py. Offline: reads the three files and
runs the build script once as a subprocess for the determinism check, no
network.

Run from the repo root: python -m tests.test_alias_table
"""

import json
import os
import subprocess
import sys
import tempfile

from hallutrace import claim_key as ck
from scripts import build_alias_table as bat

failures = []

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HALOGEN = os.path.join(ROOT, "external", "HALoGEN")
TABLE_PATH = os.path.join(ROOT, "data", "alias_table.json")
OVERRIDES_PATH = os.path.join(ROOT, "data", "alias_overrides.json")
SNAPSHOT_PATH = os.path.join(ROOT, "data", "alias_build", "wikidata_snapshot.json")
BUILD_SCRIPT = os.path.join(ROOT, "scripts", "build_alias_table.py")

# Worked examples, each checked by hand against Wikidata on 2026-09-28.
# An exact enwiki title with no redirect, auto-accepted.
EXACT_TITLE_NAME = "Cris Cyborg"
EXACT_TITLE_QID = "Q264979"

# A repaired double-encoded name (DEVIATIONS 14), auto-accepted.
REPAIRED_NAME = "Alexandre Lachevardière"
REPAIRED_QID = "Q16802250"

# A name reviewed as null in data/alias_overrides.json.
NULL_REVIEWED_NAME = "Thomas Joseph Drury"

# A key with two QIDs from a collision between two prompt entities.
COLLISION_KEY = "father of the nation"

# A stripped key that is ambiguous, and the same entity's full prompt name,
# which must still resolve BY_ALIAS.
AMBIGUOUS_STRIPPED_KEY = "william williams"
AMBIGUOUS_STRIPPED_FULL_NAME = "William Williams (cricketer, born 1844)"

# DEVIATIONS 14: the 15 names historicalevents.csv stores double encoded,
# repaired spelling. Verified present 2026-09-28.
REPAIRED_SPELLINGS = [
    "Adolphe Dugléré",
    "Alexandre Lachevardière",
    "Altymyrat Orazdurdyýew",
    "Anthony de la Roché",
    "António Carneiro",
    "Axel Törneman",
    "Bernat Martínez",
    "Cécile Cerf",
    "Juan de la Abadía",
    "Julián Bourdeu",
    "Józef Gazy",
    "Leif Mæhle",
    "Louis Honoré Arnavon",
    "Margareta Cederschiöld",
    "Mohamed Iguerbouchène",
]


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


def skip(label):
    print("   SKIP " + label)


def _is_human(p31):
    return "Q5" in (p31 or [])


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


table = _load(TABLE_PATH)
overrides = _load(OVERRIDES_PATH)
snapshot = _load(SNAPSHOT_PATH)
aliases = ck.AliasTable(table)


def _through_claim_key(name):
    """The subject side of one claim_key() call, which is what the alias
    table actually drives."""
    return ck.claim_key(name, "mentioned in", "a test object", aliases=aliases).subject


if os.path.isdir(HALOGEN):
    try:
        prompt_names = bat.load_prompt_names()
    except Exception as e:
        print("SKIP: load_prompt_names() failed even though external/HALoGEN "
              "is checked out: %s" % e)
        prompt_names = None
else:
    prompt_names = None


print("1. file shape: data/alias_table.json")
check(isinstance(table, dict), "the table is a dict")
bad_keys = [k for k in table if not k or ck.normalise(k) != k]
check(not bad_keys,
      "every key is non-empty and already normalised (%d bad: %r)"
      % (len(bad_keys), bad_keys[:5]))
bad_values = []
for k, v in table.items():
    if not isinstance(v, list) or not v:
        bad_values.append(k)
        continue
    if len(v) != len(set(v)):
        bad_values.append(k)
        continue
    if list(v) != sorted(v):
        bad_values.append(k)
        continue
    if not all(ck.is_qid(q) for q in v):
        bad_values.append(k)
check(not bad_values,
      "every value is a non-empty, sorted list of unique valid QIDs "
      "(%d bad: %r)" % (len(bad_values), bad_values[:5]))


print("2. loads without loss")
eq(len(ck.AliasTable(table)), len(table),
   "AliasTable(json.load(f)) keeps every key, none collide on re-normalisation")


print("3. data/alias_overrides.json")
bad_reason = [k for k, v in overrides.items() if not v.get("reason")]
check(not bad_reason,
      "every entry has a non-empty reason (%d missing: %r)"
      % (len(bad_reason), bad_reason[:5]))
bad_qid = [k for k, v in overrides.items()
           if v.get("qid") is not None and not ck.is_qid(v.get("qid"))]
check(not bad_qid,
      "every qid is null or a valid QID (%d bad: %r)" % (len(bad_qid), bad_qid[:5]))

if prompt_names is None:
    skip("override-key check needs external/HALoGEN, see README Setup")
else:
    name_set = {n["name"] for n in prompt_names}
    bad_override_keys = [k for k in overrides if k not in name_set]
    check(not bad_override_keys,
          "every override key is a real prompt name (%d not found: %r)"
          % (len(bad_override_keys), bad_override_keys[:5]))


print("4. coverage: every prompt name is reachable")
if prompt_names is None:
    skip("coverage check needs external/HALoGEN, see README Setup")
else:
    # A name with no override was auto-accepted (build refuses to exit 0
    # otherwise), so its decided QID is the snapshot's own
    # resolved qid for that name.
    snap_names = snapshot.get("titles", {})
    not_covered = []
    for n in prompt_names:
        name = n["name"]
        ov = overrides.get(name)
        if ov is not None:
            qid = ov.get("qid")
            if qid is None:
                continue  # null override: stays on the string fallback
        else:
            snap_entry = snap_names.get(name) or {}
            qid = snap_entry.get("qid")
        if qid is None:
            not_covered.append(name)
            continue
        if qid not in table.get(ck.normalise(name), []):
            not_covered.append(name)
    check(not not_covered,
          "every prompt name's decided QID is in the table under its "
          "normalised form, or it is a null override (%d not covered: %r)"
          % (len(not_covered), not_covered[:5]))


print("5. behaviour through claim_key()")
r = _through_claim_key(EXACT_TITLE_NAME)
check(r.source == ck.BY_ALIAS and r.key == EXACT_TITLE_QID,
      "%r (exact title, auto-accepted) resolves BY_ALIAS to %s" % (EXACT_TITLE_NAME, EXACT_TITLE_QID))
r = _through_claim_key(REPAIRED_NAME)
check(r.source == ck.BY_ALIAS and r.key == REPAIRED_QID,
      "%r (repaired spelling) resolves BY_ALIAS to %s" % (REPAIRED_NAME, REPAIRED_QID))
r = _through_claim_key(NULL_REVIEWED_NAME)
eq(r.source, ck.BY_STRING, "%r (reviewed null) resolves BY_STRING" % NULL_REVIEWED_NAME)
r = _through_claim_key(COLLISION_KEY)
check(r.source == ck.BY_AMBIGUOUS, "%r resolves BY_AMBIGUOUS" % COLLISION_KEY)
check(ck.FLAG_AMBIGUOUS in ck.claim_key(COLLISION_KEY, "x", "y", aliases=aliases).flags,
      "%r carries ambiguous_alias" % COLLISION_KEY)


print("6. determinism: build --out is byte-identical to the committed table")
if prompt_names is None:
    # build reads the prompt names from external/HALoGEN, so it cannot run
    # without it; skipped like sections 3, 4 and 8.
    skip("determinism check needs external/HALoGEN, see README Setup")
else:
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "alias_table.json")
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, BUILD_SCRIPT, "build", "--out", out_path],
            cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
        )
        check(proc.returncode == 0,
              "build --out exits 0 (stderr: %s)" % proc.stderr.strip()[-500:])
        if proc.returncode == 0 and os.path.isfile(out_path):
            with open(out_path, "rb") as f1, open(TABLE_PATH, "rb") as f2:
                check(f1.read() == f2.read(),
                      "the rebuild is byte-identical to the committed data/alias_table.json")


print("7. namesake check on every key")
namesakes = snapshot.get("namesakes", {})
namesakes_more = snapshot.get("namesakes_more", {})
namesakes_label = snapshot.get("namesakes_label", {})
unsearched = [k for k in table if k not in namesakes or k not in namesakes_label
              or (namesakes[k].get("truncated") and k not in namesakes_more)]
check(not unsearched,
      "every key has a first search and a label search in the snapshot, paged on "
      "when its first page filled (%d unsearched: %r)"
      % (len(unsearched), unsearched[:5]))

# Owners are the prompt entities' QIDs; everything else in a key's list must
# be an exact-label item from its search.
decided_qids = {v["qid"] for v in overrides.values() if ck.is_qid(v.get("qid"))}
decided_qids |= {e["qid"] for n, e in snapshot.get("titles", {}).items()
                 if n not in overrides and ck.is_qid(e.get("qid"))}
missing_other = []
unexplained = []
for key, qids in table.items():
    table_qids = set(qids)
    ignored = set()
    for ov in overrides.values():
        if ov.get("qid") in table_qids:
            ignored.update(ov.get("ignore_namesakes") or [])
    hits = [h for section in (namesakes, namesakes_more, namesakes_label)
            for h in (section.get(key, {}).get("hits", []) or [])]
    for h in hits:
        if set(h.get("p31") or []) & bat.IGNORED_P31 or h["qid"] in ignored:
            continue
        if h["qid"] not in table_qids:
            missing_other.append((key, h["qid"]))
    hit_qids = {h["qid"] for h in hits}
    for q in table_qids - decided_qids - hit_qids:
        unexplained.append((key, q))
check(not missing_other,
      "every exact-label item that is not only a name, human or not, ends up "
      "in its key's QID list (%d missing: %r)" % (len(missing_other), missing_other[:5]))
check(not unexplained,
      "every QID in the table is a prompt entity or a search hit for that key "
      "(%d unexplained: %r)" % (len(unexplained), unexplained[:5]))

check(len(table.get("william williams", [])) >= 2, "william williams has at least 2 QIDs")
r = _through_claim_key(AMBIGUOUS_STRIPPED_KEY)
check(r.source == ck.BY_AMBIGUOUS,
      "%r (stripped key) resolves BY_AMBIGUOUS" % AMBIGUOUS_STRIPPED_KEY)
r = _through_claim_key(AMBIGUOUS_STRIPPED_FULL_NAME)
check(r.source == ck.BY_ALIAS,
      "%r (the same entity's full prompt name) still resolves BY_ALIAS" % AMBIGUOUS_STRIPPED_FULL_NAME)
for word in ("scientist", "victoria", "focus"):
    r = _through_claim_key(word)
    check(r.source == ck.BY_AMBIGUOUS,
          "%r, an ordinary word as well as a prompt entity's name, resolves BY_AMBIGUOUS" % word)
r = _through_claim_key("Scientist (musician)")
check(r.source == ck.BY_ALIAS and r.key == "Q898207",
      "'Scientist (musician)', the full prompt name, still resolves BY_ALIAS to Q898207")
r = _through_claim_key("Deyverson")
check(r.source == ck.BY_ALIAS,
      "'Deyverson', a one-word name with no other exact-label item, still resolves BY_ALIAS")
r = _through_claim_key("Giovanni Paolo")
check(r.source == ck.BY_AMBIGUOUS,
      "'Giovanni Paolo', an alias of both John Paul I and John Paul II found only past "
      "the first page of search results, resolves BY_AMBIGUOUS")
r = _through_claim_key("Ho Chi Minh")
check(r.source == ck.BY_AMBIGUOUS,
      "'Ho Chi Minh', also an alias of Ho Chi Minh City found only by the label "
      "search past the search cap, resolves BY_AMBIGUOUS")
# Namesakes the first search missed and only the label search finds: an
# item matched only through an alias (Medgar Evers College has the alias
# "Medgar Evers"), or through a spelling with different punctuation
# ("Ulysses S. Grant" for "Ulysses S Grant").
for key in ("ulysses s grant", "ella j baker", "medgar evers",
            "dwight david eisenhower", "humphrey de trafford"):
    r = _through_claim_key(key)
    check(r.source == ck.BY_AMBIGUOUS,
          "%r, whose namesake only the label search finds, resolves BY_AMBIGUOUS" % key)
for title_key in ("1927 charlestown shootings", "death of catherine cando"):
    check(title_key not in table,
          "%r, an enwiki title naming an event, is not a key (titles are not a source)" % title_key)


print("8. repair_double_encoding was applied to every prompt name")
if prompt_names is None:
    skip("repair check needs external/HALoGEN, see README Setup")
else:
    from scripts.draw_sample import repair_double_encoding
    names_now = {n["name"] for n in prompt_names}
    unrepaired = [n["name"] for n in prompt_names
                  if repair_double_encoding(n["name"]) != n["name"]]
    check(not unrepaired,
          "repair_double_encoding leaves every prompt name unchanged, so "
          "none is still double encoded (%d changed: %r)"
          % (len(unrepaired), unrepaired[:5]))
    missing_spellings = [s for s in REPAIRED_SPELLINGS if s not in names_now]
    check(not missing_spellings,
          "all 15 repaired spellings from DEVIATIONS 14 are present "
          "(%d missing: %r)" % (len(missing_spellings), missing_spellings))


print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
