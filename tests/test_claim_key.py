"""Tests for hallutrace/claim_key.py. No key, no network, no fixture file."""

import sys

from hallutrace import claim_key as ck

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


print("1. normalisation, the three steps the rule names")
eq(ck.normalise("Ada Lovelace"), "ada lovelace", "lowercase")
eq(ck.normalise("Ada  Lovelace\n"), "ada lovelace", "whitespace collapsed")
eq(ck.normalise("Lovelace, Ada."), "lovelace ada", "punctuation stripped")
eq(ck.normalise("  O'Brien-Smith  "), "o brien smith",
   "apostrophe and hyphen both become separators, not deletions")
eq(ck.normalise("Beyonce"), ck.normalise("Beyonc\u00e9"),
   "accents folded, so an accented name is not two claims")
eq(ck.normalise(None), "", "a null field normalises to empty, it does not raise")
eq(ck.normalise(1815), "1815", "a number is accepted, since objects can be years")

print("2. QID recognition")
check(ck.is_qid("Q7259"), "a real QID is recognised")
check(not ck.is_qid("Q0"), "Q0 is not a QID")
check(not ck.is_qid("q7259"), "the prefix is case sensitive")
check(not ck.is_qid("7259"), "a bare number is not a QID")
check(not ck.is_qid(None) and not ck.is_qid(""), "nothing is not a QID")

print("3. the four branches of the rule, in order")
aliases = ck.AliasTable({
    "Ada Lovelace": ["Q7259"],
    "Michael Jordan": ["Q41421", "Q3308285"],   # the player and the professor
    "  charles   babbage ": ["Q46633"],
})
eq(len(aliases), 3, "the table normalises its own keys on construction")
check("ADA LOVELACE!" in aliases, "lookups are normalised, not literal")

r = ck.resolve_entity("whatever the extractor said", qid="Q42", aliases=aliases)
eq((r.key, r.source), ("Q42", ck.BY_QID),
   "a qid from the extractor wins, and the surface string is not consulted")

r = ck.resolve_entity("ada lovelace", aliases=aliases)
eq((r.key, r.source), ("Q7259", ck.BY_ALIAS), "exactly one alias match is adopted")

r = ck.resolve_entity("Charles Babbage", aliases=aliases)
eq(r.key, "Q46633", "the table's own key was normalised, so this still matches")

r = ck.resolve_entity("Someone With No Entry", aliases=aliases)
eq((r.key, r.source), ("someone with no entry", ck.BY_STRING),
   "no match keys on the normalised string")

print("4. the asymmetry: two candidates means refuse to merge, and flag")
r = ck.resolve_entity("Michael Jordan", aliases=aliases)
eq(r.source, ck.BY_AMBIGUOUS, "two candidates is its own outcome")
eq(r.key, "michael jordan",
   "and the key is the string, NOT either QID, so nothing is merged")
eq(r.candidates, ("Q3308285", "Q41421"),
   "both candidates are kept, in a fixed order, so a flagged row reads the "
   "same on every run")
check(r.is_ambiguous, "the row is marked for a human to look at")

qid_keyed = ck.resolve_entity("Michael Jordan", qid="Q41421", aliases=aliases)
check(qid_keyed.key != r.key,
      "an ambiguous key can never collide with a QID key, so the refusal to "
      "merge actually holds")

print("5. a claim key is the three parts, and only those")
k1 = ck.claim_key("Ada Lovelace", "born in", "1815", aliases=aliases)
k2 = ck.claim_key("ada  lovelace!", "BORN IN", "1815", aliases=aliases)
eq(k1.key, k2.key, "wording that normalises the same is one claim")
eq(k1.claim_id, k2.claim_id, "so it is one claim_id")
check(k1.key.startswith("Q7259|"), "the subject resolved through the alias table")

k3 = ck.claim_key("Ada Lovelace", "born in", "1816", aliases=aliases)
check(k3.key != k1.key, "a different object is a different claim")
k4 = ck.claim_key("Ada Lovelace", "died in", "1815", aliases=aliases)
check(k4.key != k1.key, "a different relation is a different claim")

eq(ck._claim_id("x|y|z"), ck._claim_id("x|y|z"),
   "claim_id is deterministic, so two runs agree without sharing state")
eq(len(k1.claim_id), 16, "claim_id is 16 hex characters")

print("6. the separator cannot appear inside a part")
odd = ck.claim_key("A|B", "r", "C|D", aliases=aliases)
eq(odd.key.count("|"), 2,
   "normalisation strips the separator from the parts, so the key always "
   "splits into exactly three")

print("7. flags")
amb = ck.claim_key("Michael Jordan", "won", "six NBA titles", aliases=aliases)
check(ck.FLAG_AMBIGUOUS in amb.flags, "an ambiguous subject flags the claim")
check(not amb.key.startswith("Q"), "and the claim is not merged onto a QID")

empty = ck.claim_key("", "born in", "1815", aliases=aliases)
check(ck.FLAG_EMPTY_FIELD in empty.flags, "an empty field is flagged")

clean = ck.claim_key("Ada Lovelace", "born in", "1815", aliases=aliases)
eq(clean.flags, (), "a clean claim carries no flags")
check(not clean.flagged, "and reports itself as unflagged")


class OnlyBornIn:
    """Stands in for the controlled vocabulary."""

    def resolve(self, relation):
        canonical = ck.normalise(relation)
        return (canonical, canonical == "born in")


k = ck.claim_key("Ada Lovelace", "was born in", "1815",
                 aliases=aliases, vocabulary=OnlyBornIn())
check(ck.FLAG_UNKNOWN_RELATION in k.flags,
      "a relation outside the vocabulary is flagged, not dropped")
eq(k.relation, "was born in",
   "and the claim still gets a key, so the row is never silently lost")

print("8. the default vocabulary invents nothing")
k = ck.claim_key("Ada Lovelace", "anything at all", "x", aliases=aliases)
eq(k.flags, (), "with no vocabulary passed, no relation is unknown")

print("9. the module works with no alias table at all")
bare = ck.claim_key("Ada Lovelace", "born in", "1815")
eq(bare.subject.source, ck.BY_STRING,
   "without a table every entity keys on its string, which is the old "
   "behaviour and is safe")
eq(bare.key, "ada lovelace|born in|1815", "and the key is readable")

print("10. the row written to the claim table")
row = amb.as_row()
eq(sorted(row), sorted([
    "claim_id", "claim_key", "subject_key", "subject_source", "relation",
    "object_key", "object_source", "flags", "subject_candidates",
    "object_candidates",
]), "one value per column, nothing nested")
eq(row["subject_candidates"], "Q3308285|Q41421",
   "the declined candidates are recorded, so the refusal is auditable")
check(all(isinstance(v, str) for v in row.values()),
      "every value is a string, so the row survives a CSV round trip")

print("11. run log counts")
s = ck.summarise([k1, k3, amb, bare])
eq(s["n"], 4, "four claims")
eq(s["flagged"], 1, "one flagged")
eq(s[ck.BY_AMBIGUOUS], 1, "one entity was ambiguous")
check(s["distinct"] == 4, "four distinct keys")
check(s[ck.BY_QID] + s[ck.BY_ALIAS] + s[ck.BY_STRING] + s[ck.BY_AMBIGUOUS] == 8,
      "every entity of every claim is counted exactly once, two per claim")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
