"""Tests for hallutrace/bio_years.py. No network."""

import sys

from hallutrace import bio_years as by

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


print("1. which atoms are year claims")
eq(by.kind_of("Gandhi was born in 1869."), "birth", "born + year")
eq(by.kind_of("Mahatma Gandhi (1869-1948) was a lawyer."), "birth", "a lifespan opens with the birth year")
eq(by.kind_of("He was assassinated in 1948."), "death", "assassinated + year")
eq(by.kind_of("He was born in Porbandar."), None, "no year")
eq(by.kind_of("He wrote a book in 1909."), None, "a year with no birth or death word")
eq(by.stated_years("born 1869, or maybe 1870, in 1869"), [1869, 1870], "years, sorted and unique")

SNAP = {"entities": {
    "Q1": {"p569": [1869], "p570": [1948]},   # the person
    "Q2": {"p569": [1901], "p570": [1960]},   # a namesake
    "Q3": {},                                 # a namesake with no years in the snapshot
}}
META = {"generation_id": "m|biographies-00001|1", "model": "m", "sample_index": "1", "unit_id": "m-b00001"}


def row(*atoms):
    return {"name": "Gandhi", "atoms": "".join("| %s\t%s |\n" % a for a in atoms)}


print("2. Wikidata and the namesake check")
cs = by.claims_in_row(row(("He was born in 1869.", True), ("He was born in 1901.", False),
                          ("He died in 1948.", False), ("He was born in 1777.", False),
                          ("He likes tea.", False)),
                      META, "Q1", ("Q1", "Q2", "Q3"), SNAP)
eq(len(cs), 4, "four year claims, the tea atom is not one")
eq([c["wikidata_match"] for c in cs], ["True", "False", "True", "False"], "matched against the own QID's years")
eq([c["conflation_candidate"] for c in cs], ["False", "True", "False", "False"],
   "1901 is the namesake's birth year: a candidate; 1948 is the person's own year, a verifier error, not one")
eq(cs[1]["namesake_year_qids"], "Q2", "which namesake")
eq(cs[0]["namesake_qids"], "Q2 Q3", "every other QID the name can denote")
eq((cs[0]["label"], cs[2]["label"]), ("supported", "hallucinated"), "labels from the verifier")
eq(cs[0]["generation_id"], META["generation_id"], "the row's ids travel with each claim")

print("3. no own QID (null in the alias table)")
cs = by.claims_in_row(row(("He was born in 1901.", False),), META, None, ("Q2",), SNAP)
eq((cs[0]["wikidata_match"], cs[0]["conflation_candidate"]), ("", "True"),
   "no Wikidata verdict; a namesake match stays a candidate, an upper bound")

print("4. atoms that span a line break are kept")
multi = {"name": "Gandhi", "atoms": "| He was born\nin 1869.\tTrue |\n"}
eq(len(by.claims_in_row(multi, META, "Q1", ("Q1",), SNAP)), 1, "parse_atoms reads across the line break")

print("4b. namesakes' years from namesake_years.json")
NS = {"entities": {"Q3": {"p569": [1777], "p570": []}, "Q2": {"p569": [], "p570": []}}}
cs = by.claims_in_row(row(("He was born in 1777.", False), ("He was born in 1901.", False)),
                      META, "Q1", ("Q1", "Q2", "Q3"), SNAP, NS)
eq([(c["conflation_candidate"], c["namesake_year_qids"]) for c in cs], [("True", "Q3"), ("False", "")],
   "namesakes' years come from the file; the person's own still from the snapshot")
eq([c["wikidata_match"] for c in cs], ["False", "False"], "own years unchanged")

print("5. summary")
s = by.summarise(by.claims_in_row(row(("He was born in 1869.", True), ("He was born in 1901.", False)),
                                  META, "Q1", ("Q1", "Q2"), SNAP))
eq((s["claims"], s["conflation_candidates"], s["wikidata"]["label_agrees"]), (2, 1, 2), "counts")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
