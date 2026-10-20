"""Tests for hallutrace/relations.py. No key, no network, no data file.

Every relation name here is a neutral fixture. None is a chosen relation.
"""

import copy
import json
import os
import sys
import tempfile

from hallutrace import claim_key as ck
from hallutrace.relations import RelationVocabulary

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


def raises(fn, needle, label):
    """fn must raise ValueError, and the message must contain `needle`."""
    try:
        fn()
    except ValueError as e:
        check(needle in str(e), label if needle in str(e)
              else "%s  (message %r lacks %r)" % (label, str(e), needle))
    except Exception as e:
        check(False, "%s  (raised %s, not ValueError)" % (label, type(e).__name__))
    else:
        check(False, "%s  (nothing raised)" % label)


FIXTURE = {
    "version": "test-1",
    "relations": {
        "alpha_rel": {"pid": "P19", "description": "first fixture",
                      "synonyms": ["Alpha Relation", "alpha-syn", "àlpha accent"]},
        "beta_rel": {"pid": None, "description": "second fixture",
                     "synonyms": ["beta syn"]},
        "gamma_rel": {"pid": "P569", "description": "third fixture",
                      "synonyms": []},
    },
}


def fixture(**edits):
    """A deep copy of FIXTURE with the given top level keys replaced or added."""
    m = copy.deepcopy(FIXTURE)
    for k, v in edits.items():
        m[k] = v
    return m


def with_entry(name, entry):
    m = copy.deepcopy(FIXTURE)
    m["relations"][name] = entry
    return m


def entry(**over):
    e = {"pid": None, "description": "d", "synonyms": []}
    e.update(over)
    return e


vocab = RelationVocabulary(FIXTURE)

print("1. each canonical resolves to itself, so re-keying a stored part is stable")
for c in ("alpha_rel", "beta_rel", "gamma_rel"):
    eq(vocab.resolve(c), (c, True), "canonical %s" % c)
    eq(vocab.resolve(vocab.resolve(c)[0]), (c, True),
       "resolve() of the stored part gives back the same part (%s)" % c)

print("2. each synonym resolves to its canonical")
eq(vocab.resolve("Alpha Relation"), ("alpha_rel", True), "synonym as written")
eq(vocab.resolve("ALPHA RELATION"), ("alpha_rel", True), "different case")
eq(vocab.resolve("alpha relation"), ("alpha_rel", True), "lower case")
eq(vocab.resolve("  alpha \t relation \n"), ("alpha_rel", True), "extra whitespace")
eq(vocab.resolve("alpha, relation!"), ("alpha_rel", True), "punctuation stripped")
eq(vocab.resolve("alpha:syn"), ("alpha_rel", True), "colon and hyphen forms agree")
eq(vocab.resolve("alpha syn"), ("alpha_rel", True), "the stored form is normalised")
eq(vocab.resolve("alpha accent"), ("alpha_rel", True),
   "a synonym stored with an accent matches the unaccented text")
eq(vocab.resolve("ÀLPHA ACCENT"), ("alpha_rel", True),
   "and the accented text in a different case")
eq(vocab.resolve("beta syn"), ("beta_rel", True), "a synonym of another canonical")

print("3. an unknown relation is returned normalised and unknown, never dropped")
eq(vocab.resolve("Delta Rel!"), ("delta rel", False), "unknown gives (normalise(r), False)")
eq(vocab.resolve("alpha_rel extra"), ("alpha_rel extra", False),
   "no fuzzy matching: a canonical plus a word is not the canonical")
eq(vocab.resolve("alpha"), ("alpha", False), "a prefix of a canonical is not a match")
eq(vocab.resolve("OTHER:alpha_rel"), ("other alpha_rel", False),
   "OTHER: is not special, even around a canonical")
eq(vocab.resolve("OTHER:"), ("other", False), "a bare OTHER: is just the word other")

print("4. empty input has no canonical")
for raw in ("", None, "!!", "  "):
    eq(vocab.resolve(raw), (None, False), "%r gives (None, False)" % (raw,))

print("5. validation, one case per rule, and the message names the culprit")
raises(lambda: RelationVocabulary([]), "vocabulary", "rule 1: mapping is not a dict")
raises(lambda: RelationVocabulary({"version": "v"}), "relations",
       "rule 1: top level key missing")
raises(lambda: RelationVocabulary(fixture(extra=1)), "extra",
       "rule 1: top level key extra")
raises(lambda: RelationVocabulary(fixture(relations={})), "relations",
       "relations must be non-empty")
raises(lambda: RelationVocabulary(fixture(relations=[])), "relations",
       "relations must be a dict")
raises(lambda: RelationVocabulary(with_entry("bad_rel", "text")), "bad_rel",
       "rule 1: an entry that is not a dict")
raises(lambda: RelationVocabulary(with_entry("bad_rel", {"pid": None})), "bad_rel",
       "rule 1: entry with missing keys")
raises(lambda: RelationVocabulary(with_entry("bad_rel", entry(more=1))), "bad_rel",
       "rule 1: entry with extra keys")
raises(lambda: RelationVocabulary(fixture(version="")), "version",
       "rule 2: empty version")
raises(lambda: RelationVocabulary(fixture(version=3)), "version",
       "rule 2: non-string version")
raises(lambda: RelationVocabulary(with_entry("bad_rel", entry(description=""))),
       "bad_rel", "rule 2: empty description")
raises(lambda: RelationVocabulary(with_entry("bad_rel", entry(description=None))),
       "bad_rel", "rule 2: non-string description")
for bad in ("19", "P0", "P01", "p19", "P19x", 19):
    raises(lambda bad=bad: RelationVocabulary(with_entry("bad_rel", entry(pid=bad))),
           "bad_rel", "rule 3: pid %r" % (bad,))
raises(lambda: RelationVocabulary(with_entry("Bad Rel", entry())), "Bad Rel",
       "rule 4: canonical not normal (case and space)")
raises(lambda: RelationVocabulary(with_entry("bad-rel", entry())), "bad-rel",
       "rule 4: canonical not normal (hyphen)")
raises(lambda: RelationVocabulary(with_entry("", entry())), "canonical",
       "rule 4: empty canonical")
raises(lambda: RelationVocabulary(with_entry(7, entry())), "7",
       "rule 4: non-string canonical")
raises(lambda: RelationVocabulary(with_entry("bad_rel", entry(synonyms="x"))),
       "bad_rel", "rule 5: synonyms is not a list")
raises(lambda: RelationVocabulary(with_entry("bad_rel", entry(synonyms=["ok", 3]))),
       "bad_rel", "rule 5: synonyms holds a non-string")
raises(lambda: RelationVocabulary(with_entry("bad_rel", entry(synonyms=["!!"]))),
       "bad_rel", "rule 6: synonym normalises to empty")
raises(lambda: RelationVocabulary(with_entry("bad_rel", entry(synonyms=["BETA_REL"]))),
       "BETA_REL", "rule 7: synonym equals another canonical once normalised")
raises(lambda: RelationVocabulary(with_entry("bad_rel", entry(synonyms=["BAD REL", "bad_rel"]))),
       "bad_rel", "rule 7: synonym equals its own canonical")
raises(lambda: RelationVocabulary(with_entry("bad_rel", entry(synonyms=["twin", "TWIN"]))),
       "TWIN", "rule 8: duplicate synonym under one canonical")
raises(lambda: RelationVocabulary(with_entry("bad_rel", entry(synonyms=["beta   syn"]))),
       "beta   syn", "rule 8: duplicate synonym under two canonicals")
raises(lambda: RelationVocabulary({"version": "v", "relations": {
    "aaa_rel": entry(synonyms=["zzz_rel"]), "zzz_rel": entry()}}),
    "zzz_rel", "rule 7: holds when the canonical is written after the synonym")

print("6. load() reads UTF-8 JSON and refuses duplicate keys at any depth")
tmpdir = tempfile.mkdtemp()


def write(name, text):
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


utf8 = fixture()
utf8["relations"]["alpha_rel"]["description"] = "café fixture"
loaded = RelationVocabulary.load(write("ok.json", json.dumps(utf8, ensure_ascii=False)))
eq(loaded.canonicals, vocab.canonicals, "a valid file round-trips the canonicals")
eq(loaded.resolve("alpha-syn"), ("alpha_rel", True), "and the synonyms")
eq(loaded.version, "test-1", "and the version")

dup_top = ('{"version": "a", "version": "b", "relations": '
           '{"alpha_rel": {"pid": null, "description": "d", "synonyms": []}}}')
raises(lambda: RelationVocabulary.load(write("dup_top.json", dup_top)), "version",
       "a duplicate top level key raises")
dup_canon = ('{"version": "a", "relations": '
             '{"alpha_rel": {"pid": null, "description": "d", "synonyms": []}, '
             '"alpha_rel": {"pid": null, "description": "e", "synonyms": []}}}')
raises(lambda: RelationVocabulary.load(write("dup_canon.json", dup_canon)),
       "alpha_rel", "a duplicate canonical key raises")
dup_inner = ('{"version": "a", "relations": {"alpha_rel": '
             '{"pid": null, "pid": "P1", "description": "d", "synonyms": []}}}')
raises(lambda: RelationVocabulary.load(write("dup_inner.json", dup_inner)), "pid",
       "a duplicate key inside an entry raises")
raises(lambda: RelationVocabulary.load(write("bad.json", '{"version": "a"}')),
       "relations", "a file that parses but fails validation raises ValueError")

print("7. accessors")
eq(vocab.version, "test-1", "version")
eq(vocab.canonicals, ("alpha_rel", "beta_rel", "gamma_rel"), "canonicals, sorted")
shuffled = RelationVocabulary({"version": "v", "relations": {
    "zulu_rel": entry(), "alpha_rel": entry(), "mike_rel": entry()}})
eq(shuffled.canonicals, ("alpha_rel", "mike_rel", "zulu_rel"),
   "canonicals are sorted whatever the input order")
check(isinstance(vocab.canonicals, tuple), "canonicals is a tuple")
eq(vocab.pid("alpha_rel"), "P19", "pid of a linked relation")
eq(vocab.pid("beta_rel"), None, "pid of an unlinked relation is None")
try:
    vocab.pid("nope_rel")
    check(False, "pid of an unknown canonical raises KeyError")
except KeyError:
    check(True, "pid of an unknown canonical raises KeyError")
try:
    vocab.pid("Alpha Relation")
    check(False, "pid of a synonym raises KeyError, it takes canonicals only")
except KeyError:
    check(True, "pid of a synonym raises KeyError, it takes canonicals only")
eq(len(vocab), 3, "len is the number of canonicals, not synonyms")
check("alpha_rel" in vocab, "a canonical is in the vocabulary")
check("ALPHA RELATION" in vocab, "a synonym is in the vocabulary")
check("delta rel" not in vocab, "an unknown relation is not")
check("" not in vocab and None not in vocab, "empty input is not")

print("8. prompt_list")
eq(vocab.prompt_list(), "[alpha_rel, beta_rel, gamma_rel]", "exact string, sorted")
eq(shuffled.prompt_list(), "[alpha_rel, mike_rel, zulu_rel]",
   "sorted whatever the input order")

print("9. end to end with claim_key(..., vocabulary=vocab)")
k_canon = ck.claim_key("Ada Lovelace", "alpha_rel", "1815", vocabulary=vocab)
k_syn = ck.claim_key("Ada Lovelace", "Alpha Relation", "1815", vocabulary=vocab)
eq(k_syn.claim_id, k_canon.claim_id, "a synonym and its canonical share a claim_id")
eq(k_syn.relation, "alpha_rel", "and the stored relation is the canonical")
k_beta = ck.claim_key("Ada Lovelace", "beta_rel", "1815", vocabulary=vocab)
check(k_beta.claim_id != k_canon.claim_id, "two canonicals give different claim_ids")
k_unk = ck.claim_key("Ada Lovelace", "Delta Rel", "1815", vocabulary=vocab)
check(ck.FLAG_UNKNOWN_RELATION in k_unk.flags, "an unknown relation is flagged")
eq(k_unk.relation, "delta rel", "and keeps its normalised relation")
eq(k_unk.key, "ada lovelace|delta rel|1815", "and still gets a key")
eq(len(k_unk.claim_id), 16, "and a claim_id")
check(ck.FLAG_UNKNOWN_RELATION not in k_canon.flags, "a canonical carries no relation flag")
check(ck.FLAG_UNKNOWN_RELATION not in k_syn.flags, "a synonym carries no relation flag")
k_empty = ck.claim_key("Ada Lovelace", "!!", "1815", vocabulary=vocab)
check(ck.FLAG_UNKNOWN_RELATION in k_empty.flags and k_empty.relation is None,
      "an empty relation is flagged and has no relation")
eq(ck.claim_key("Ada Lovelace", vocab.resolve("alpha-syn")[0], "1815",
                vocabulary=vocab).claim_id, k_canon.claim_id,
   "re-keying a stored relation part gives the same claim_id")

print("10. drop-in for AnyRelation: same known input, same key shape")
any_key = ck.claim_key("Ada Lovelace", "alpha_rel", "1815")
eq(k_canon.key, any_key.key, "for a canonical, the key is identical to the default's")
eq(k_canon.key.count("|"), 2, "subject|relation|object")
eq(len(k_canon.key.split("|")), 3, "splits into exactly three parts")
eq(sorted(k_canon.as_row()), sorted(any_key.as_row()), "the claim-table row has the same columns")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
