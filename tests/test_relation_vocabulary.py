"""Tests for the shipped relation vocabulary, data/relation_vocabulary.json.

`tests/test_relations.py` covers the loader with neutral fixtures. This file
covers what only the real data can show: the file loads, every relation has its
space form, rule 6 of the claim extractor's prompt lists exactly these names, and real
relation names key as intended.
"""

import json
import os
import re
import sys

from hallutrace import claim_key as ck
from hallutrace.relations import RelationVocabulary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "relation_vocabulary.json")
# rule 6 of the claim extractor's prompt, as written in the cause taxonomy
PROMPT_RULE6 = ("[author_of, award_received, cause_of_death, child, cited_reference, "
                "country_of_citizenship, date_of_birth, date_of_death, date_of_event, educated_at, "
                "employer, father, field_of_work, has_parameter, instance_of, inventor_of, "
                "located_in, member_of, member_of_sports_team, mother, notable_work, occupation, "
                "place_of_birth, place_of_death, position_held, prevents, residence, sex_or_gender, "
                "sibling, spouse]")

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


with open(DATA, "r", encoding="utf-8") as f:
    raw_text = f.read()
raw = json.loads(raw_text)
vocab = RelationVocabulary.load(DATA)

RULE6 = ("place_of_birth", "award_received", "inventor_of", "occupation",
         "instance_of", "has_parameter", "prevents", "author_of", "located_in",
         "date_of_event")
NULL_PID = ("author_of", "cited_reference", "date_of_event", "has_parameter",
            "inventor_of", "located_in", "prevents")

print("1. the shipped file loads and is well formed")
eq(vocab.version, "2026-09-30.1", "version is the approved string")
eq(len(vocab), 30, "30 canonicals")
check(raw_text == json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
      "the file is deterministic: sorted keys, two space indent, trailing newline")
check(chr(0x2014) not in raw_text and chr(0x2013) not in raw_text,
      "no em or en dash in the data")
for c in vocab.canonicals:
    check(re.fullmatch(r"[a-z0-9_]+", c) is not None,
          "canonical %s matches [a-z0-9_]+ (the loader checks normal form only)" % c)

print("2. every canonical and its space form resolve to it (the re-key invariant)")
for c in vocab.canonicals:
    eq(vocab.resolve(c), (c, True), "canonical %s" % c)
    space = c.replace("_", " ")
    if space != c:
        check(space in raw["relations"][c]["synonyms"],
              "%s lists its space form %r as a synonym in the data" % (c, space))
    else:
        check(space not in raw["relations"][c]["synonyms"],
              "%s is one word, so it is its own space form and lists none" % c)
    eq(vocab.resolve(space), (c, True), "space form of %s" % c)
    eq(vocab.resolve(vocab.resolve(c)[0]), (c, True),
       "resolving the stored part gives the same part (%s)" % c)

print("3. every listed synonym resolves to its canonical")
n = 0
for c, e in raw["relations"].items():
    for s in e["synonyms"]:
        n += 1
        got = vocab.resolve(s)
        check(got == (c, True), "%r -> %s" % (s, c) if got == (c, True)
              else "%r -> %s  (got %r)" % (s, c, got))
eq(n, 126, "126 synonyms in all (21 space forms and 105 others; nine canonicals are one word)")

print("4. PIDs")
eq(tuple(sorted(c for c in vocab.canonicals if vocab.pid(c) is None)), NULL_PID,
   "exactly the seven expected relations have no PID")
eq(vocab.pid("place_of_birth"), "P19", "place_of_birth is P19")
eq(vocab.pid("date_of_birth"), "P569", "date_of_birth is P569")
eq(vocab.pid("date_of_death"), "P570", "date_of_death is P570")
pids = [vocab.pid(c) for c in vocab.canonicals if vocab.pid(c)]
eq(len(pids), len(set(pids)), "no PID is used by two canonicals")
for c in ("located_in", "date_of_event", "inventor_of", "author_of"):
    check("Nearest" in raw["relations"][c]["description"]
          or "Inverse of" in raw["relations"][c]["description"],
          "%s names its nearest property in the description" % c)

print("5. rule 6 of the extractor prompt holds exactly prompt_list()")
for c in RULE6:
    check(c in vocab.canonicals, "rule 6 name %s is kept" % c)
eq(PROMPT_RULE6, vocab.prompt_list(), "rule 6 lists exactly prompt_list()")

print("6. real relation names")
# dropped as possible over-merges
for s in ("field of study", "area of study", "played for", "plays for"):
    eq(vocab.resolve(s), (s, False), "%r is left unmapped" % s)
# ambiguous words the draft deliberately left out
for s in ("born in", "born", "died in", "is a", "job", "known for", "attended",
          "occurred on", "son", "brother", "partner", "written by"):
    eq(vocab.resolve(s)[1], False, "%r is not a synonym of anything" % s)
# the added synonyms are present
for s, c in (("birth_year", "date_of_birth"), ("death year", "date_of_death"),
             ("death_year", "date_of_death"), ("is located in", "located_in"),
             ("situated in", "located_in"), ("event date", "date_of_event"),
             ("invented", "inventor_of"), ("wrote", "author_of"),
             ("authored", "author_of"), ("writer of", "author_of"),
             ("date of the event", "date_of_event"), ("time of event", "date_of_event")):
    eq(vocab.resolve(s), (c, True), "%r -> %s" % (s, c))
eq(vocab.resolve("Nationality"), ("country_of_citizenship", True), "nationality is kept")
# no automatic OTHER matching
eq(vocab.resolve("OTHER:birth_place"), ("other birth_place", False),
   "OTHER:birth_place stays unknown")
eq(vocab.resolve("OTHER:place_of_birth"), ("other place_of_birth", False),
   "OTHER: around a canonical stays unknown")
eq(vocab.resolve("OTHER:"), ("other", False), "a bare OTHER: is just the word other")
# pairs that must never merge
PAIRS = (("place_of_birth", "place_of_death"), ("date_of_birth", "date_of_death"),
         ("father", "child"), ("father", "mother"), ("occupation", "position_held"),
         ("inventor_of", "author_of"), ("located_in", "residence"))
for a, b in PAIRS:
    check(a != b and vocab.resolve(a)[0] != vocab.resolve(b)[0],
          "%s and %s stay two relations" % (a, b))
eq(vocab.resolve("Place of Birth"), ("place_of_birth", True), "case is ignored")
eq(vocab.resolve("  Café  Place  "), ("cafe place", False), "accent and spacing fold, unknown stays unknown")
eq(vocab.resolve(""), (None, False), "empty")
eq(vocab.resolve(None), (None, False), "None")

print("7. end to end with claim_key(..., vocabulary=vocab)")
k_canon = ck.claim_key("Ada Lovelace", "date_of_birth", "1816", vocabulary=vocab)
k_year = ck.claim_key("Ada Lovelace", "birth_year", "1816", vocabulary=vocab)
k_spaced = ck.claim_key("Ada Lovelace", "Date of Birth", "1816", vocabulary=vocab)
eq(k_canon.key, "ada lovelace|date_of_birth|1816", "the key carries the canonical")
eq(k_year.claim_id, k_canon.claim_id, "birth_year and date_of_birth give one claim_id")
eq(k_spaced.claim_id, k_canon.claim_id, "the space form gives the same claim_id")
eq(k_canon.claim_id, "64c26ea63c501133",
   "the worked example's claim_id")
check(not k_canon.flagged, "a canonical claim carries no flags")
k_born_in = ck.claim_key("Ada Lovelace", "born in", "1816", vocabulary=vocab)
check(ck.FLAG_UNKNOWN_RELATION in k_born_in.flags, "'born in' is flagged unknown_relation")
k_other = ck.claim_key("Ada Lovelace", "OTHER:birth_place", "London", vocabulary=vocab)
check(ck.FLAG_UNKNOWN_RELATION in k_other.flags, "an OTHER relation is flagged unknown_relation")
eq(k_other.relation, "other birth_place", "and keeps the normalised OTHER string")
k_pob = ck.claim_key("Ada Lovelace", "place_of_birth", "London", vocabulary=vocab)
k_pod = ck.claim_key("Ada Lovelace", "place_of_death", "London", vocabulary=vocab)
check(k_pob.claim_id != k_pod.claim_id, "place_of_birth and place_of_death differ")
k_dod = ck.claim_key("Ada Lovelace", "date_of_death", "1816", vocabulary=vocab)
check(k_dod.claim_id != k_canon.claim_id, "date_of_birth and date_of_death differ")

print("8. the re-key recipe for a key stored under the old placeholder")
stored = "ada lovelace|birth_year|1816"
s_part, r_part, o_part = stored.split("|")
rekeyed = ck.claim_key(s_part, r_part, o_part, vocabulary=vocab)
eq(rekeyed.key, "ada lovelace|date_of_birth|1816", "the old key re-keys to the canonical")
eq(rekeyed.claim_id, k_canon.claim_id, "and to the same claim_id")
eq(vocab.resolve(k_canon.relation), (k_canon.relation, True),
   "resolving the stored relation part gives back the same part")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
