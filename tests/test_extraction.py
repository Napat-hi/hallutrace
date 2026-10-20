"""Tests for hallutrace/extraction.py. No key, no network.

Checks that each prompt is found in the HALoGEN checkout under
external/HALoGEN (skipped when absent), and pins the parsers' behaviour,
including the released bugs.
"""

import os
import sys

from hallutrace import extraction as ex

OK = "   OK  "
failures = []


def check(cond, label):
    print((OK if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, "%s  (got %r)" % (label, got) if got != want else label)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HALOGEN = os.path.join(ROOT, "external", "HALoGEN")

print("1. decode settings")
eq(sorted(ex.UNPINNED), ["decompose", "entailment", "false_presuppositions"],
   "the three unpinned calls are named")
check(ex.DECODING["historical_events"]["max_tokens"] == 20,
      "historical_events answers yes or no in 20 tokens")
check(len(ex.describe()) > 200, "describe() renders")

print("2. prompts are HALoGEN's, character for character")
if not os.path.isdir(HALOGEN):
    print("   SKIP  external/HALoGEN is not checked out")
else:
    def in_file(rel, needle, label):
        src = open(os.path.join(HALOGEN, rel), encoding="utf-8").read()
        check(needle in src, label)

    p = ex.prompt_false_presuppositions("XX")
    in_file("verifiers/false_presuppositions.py",
            "here is the list of plantes", "the plantes typo is preserved")
    check("entities’ list" in p, "the curly apostrophe is preserved")
    check("```### List with `and````" in p, "the stray backtick block is kept")

    in_file("verifiers/historical_events.py", "comfirm the occurrence of meeting",
            "comfirm typo is in the source we copied from")
    h = ex.prompt_historical_events("R", "A", "B")
    check("comfirm the occurrence of meeting between A and B" in h,
          "comfirm typo is preserved in our prompt")
    check(h.endswith("/n <bot>:"),
          "the literal slash n is preserved, not turned into a newline")

    rb = ex.prompt_rationalization_binary("R", "primality")
    in_file("verifiers/rationalization_binary.py",
            "7411 is not a prime number", "primality few shot matches source")
    check(rb.endswith("/n <bot>:"), "rationalization_binary keeps the slash n")

    in_file("src/scientific_attribution.py", "Essential oils cure coronavirus",
            "scientific_attribution few shot matches source")
    in_file("verifiers/summarization.py",
            "Please breakdown the following passage into independent facts",
            "decomposition instruction matches source")
    in_file("verifiers/summarization.py",
            "Given the premise, is the hypothesis correct?",
            "entailment instruction matches source")

print("3. rationalization_binary, an unknown category has no prompt")
eq(ex.prompt_rationalization_binary("R", "senator") is None, False,
   "senator is a known category")
eq(ex.prompt_rationalization_binary("R", "geography"), None,
   "an unknown category returns None, which upstream skips the row silently")

print("4. parsers, including the traps")
eq(ex.parse_false_presuppositions("Mars, Venus, Earth"),
   ["Mars", "Venus", "Earth"], "comma space splits")
eq(ex.parse_false_presuppositions("Mars,Venus"), [],
   "no space after the comma yields nothing")
eq(ex.parse_false_presuppositions("Mars"), [],
   "a single entity with no comma yields nothing")

eq(ex.parse_rationalization_numerical("4, earth, mars"), (4, ["earth", "mars"]),
   "number and entities")
eq(ex.parse_rationalization_numerical("no idea"), (None, None),
   "no match returns a pair of None")

eq(ex.parse_primality("3, 3, 7, 13"), [3, 3, 7, 13], "factors parse")
eq(ex.parse_primality("three, 7, x"), [7], "non integers are dropped")

eq(ex.parse_graph("There is a flight from city C to city H, and from city H to city F"),
   [("C", "H"), ("H", "F")], "flights parse as pairs")
eq(ex.parse_graph("from city AB to city CD"), [],
   "only single letter cities match, by the released regex")

eq(ex.parse_yes_no("Yes, there is."), ["yes"], "leading yes")
eq(ex.parse_yes_no("no."), ["no"], "leading no")
eq(ex.parse_yes_no("Certainly, yes"), [],
   "yes later in the sentence is not counted, which reads as a non answer")

eq(ex.parse_titles("Title: A; Title: B;"), ["A", "B"], "titles parse")
eq(ex.parse_titles("Title: A; Title: B"), ["A"],
   "a final title without a semicolon is dropped")
eq(ex.parse_titles(""), [], "empty reply yields nothing")

eq(ex.parse_decomposition("1. Ada was born in 1815.\n2. She wrote notes."),
   [" Ada was born in 1815.", " She wrote notes."],
   "TRAP, preserved: the number is stripped but the space after it is not, so "
   "every numbered atomic unit carries a leading space")
eq(ex.parse_decomposition("Sure! Here you go.\n1. A fact."), [" A fact."],
   "the Sure! preamble is dropped")
eq(ex.parse_decomposition("A fact with no number."), ["A fact with no number."],
   "unnumbered lines pass through, with no leading space, so units from the "
   "same run are not consistently shaped")
eq(ex.parse_decomposition("1."), [""],
   "TRAP, preserved: a bare number becomes an empty atomic unit, not a skip")

check(ex.parse_entailment("Yes") is True, "yes is supported")
check(ex.parse_entailment("No") is False, "no is unsupported")
check(ex.parse_entailment("No, yesterday it was not") is True,
      "TRAP, preserved: yesterday makes an unsupported claim look supported")

print("parser defect 9: parse_senator's strip")


class _Ent:
    def __init__(self, text, label="PERSON"):
        self.text, self.label_ = text, label


class _FakeNLP:
    """Stands in for spaCy: returns the entities it was built with."""

    def __init__(self, *ents):
        self.ents = list(ents)

    def __call__(self, text):
        return self


eq(ex.clean_senator_name("Susan Collins", released=True), "Susan Collin",
   "released: strip eats the final s of Collins")
eq(ex.clean_senator_name("Max Baucus", released=True), "Max Baucu",
   "released: and of Baucus")
eq(ex.clean_senator_name("Susan Collins"), "Susan Collins", "fixed: Collins keeps its s")
eq(ex.clean_senator_name("Max Baucus."), "Max Baucus", "fixed: a trailing period still goes")
eq(ex.clean_senator_name("Jacky Rosen</s>"), "Jacky Rosen", "fixed: a trailing </s> still goes")
eq(ex.clean_senator_name(" Chris Coons, "), "Chris Coons", "fixed: spaces and commas around it go")
eq(ex.clean_senator_name("Jacky Rosen</s>", released=True), "Jacky Rosen",
   "a name that does not end in s is the same either way")
nlp = _FakeNLP(_Ent("Susan Collins"), _Ent("Nevada", "GPE"), _Ent("Max Baucus</s>"))
eq(ex.parse_senator("any text", nlp=nlp), ["Susan Collins", "Max Baucus"],
   "parse_senator uses the fixed cleaning by default, PERSON entities only")
eq(ex.parse_senator("any text", nlp=nlp, released=True), ["Susan Collin", "Max Baucu"],
   "released=True reproduces HALoGEN's output")

print("prompts v2: a NONE example and a use-only-the-context sentence")
R = "ZZ_RESPONSE_ZZ"
v2 = {
    "false_presuppositions": ex.prompt_false_presuppositions_v2(R),
    "rationalization_numerical": ex.prompt_rationalization_numerical_v2(R),
    "scientific_attribution": ex.prompt_scientific_attribution_v2(R),
    "primality": ex.prompt_rationalization_binary_v2(R, "primality"),
    "graph": ex.prompt_rationalization_binary_v2(R, "graph"),
    "senator": ex.prompt_rationalization_binary_v2(R, "senator"),
}
for name, text in v2.items():
    check(text.count(R) == 1, "%s v2 holds the response exactly once" % name)
    check("NONE" in text.split(R)[0], "%s v2 shows a NONE answer before the response" % name)
    check(text.split(R)[1].count(ex._ONLY_CONTEXT.strip()) == 1,
          "%s v2 repeats the use-only-the-context sentence after the response" % name)
eq(ex.prompt_rationalization_binary_v2(R, "unknown"), None,
   "v2 still returns None for an unknown category, as released")
check(ex.prompt_decompose_v2(R).startswith(ex.prompt_decompose(R)),
      "decompose v2 is the released prompt with one sentence added")
check(ex.prompt_decompose_v2(R).endswith("with no introduction or notes."),
      "and the sentence asks for the facts only")
eq(ex.PROMPTS_V2["historical_events"], ex.PROMPTS["historical_events"],
   "historical_events extracts nothing, so v2 leaves it alone")
eq(ex.PROMPTS_V2["entailment"], ex.PROMPTS["entailment"], "and the entailment judge too")
eq(ex.prompt_false_presuppositions("x").startswith("<Context>: Sure, here is the list of plantes"),
   True, "the released prompt is untouched")
eq(sorted(ex.PROMPT_VERSIONS), ["released", "v2"], "two prompt versions, picked by name")
for reply in ("NONE", " NONE", "NONE\n", "NONE."):
    eq(ex.parse_false_presuppositions(reply), [], "%r parses to no presupposition units" % reply)
    eq(ex.parse_rationalization_numerical(reply), (None, None), "%r parses to no count" % reply)
    eq(ex.parse_primality(reply), [], "%r parses to no factors" % reply)
    eq(ex.parse_graph(reply), [], "%r parses to no flights" % reply)
    eq(ex.parse_titles(reply), [], "%r parses to no titles" % reply)
nlp_none = _FakeNLP(_Ent("NONE", "ORG"))
eq(ex.parse_senator("NONE", nlp=nlp_none), [],
   "NONE is tagged ORG by spaCy 3.8 (checked by hand), so it is no senator")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
