"""Tests for hallutrace/unit_guard.py. No key, no network."""

import sys

from hallutrace import unit_guard as ug

OK = "   OK  "
failures = []


def check(cond, label):
    print((OK if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


print("items and names: every word, in order, next to each other")
R = "The 9 months with a t are: August, September, October."
check(ug.in_response("September", "item", R), "a list item in the response")
check(not ug.in_response("January", "item", R), "an item the model never wrote")
check(ug.in_response("august", "item", R), "case does not matter")
check(not ug.in_response("", "item", R), "an empty unit is never present")
SUB = "The second line contains n integers a 1, a 2, \u2026, a n."
check(ug.in_response("a1", "item", SUB) and ug.in_response("an", "item", SUB),
      "a flattened subscript: a1 and an are in 'a 1, ..., a n' (v2 smoke test)")
check(not ug.in_response("b1", "item", SUB), "but b1 is not")
check(not ug.in_response("a1 a2", "item", "a 1 a 2"), "only a one word unit is joined, never a phrase")
check(ug.in_response("a10", "item", "integers a 1, ..., a 10"), "a letter and a number join: a10 is in 'a 10'")
check(not ug.in_response("Togo", "item", "I want to go home."), "real words never join: Togo is not in 'to go'")
check(not ug.in_response("tin", "item", "I wasn't in."), "nor tin in 't in'")
check(not ug.in_response("Augusta", "item", "In August a storm hit."), "nor Augusta in 'August a'")
check(ug.in_response("Susan Collins", "name", "I think Susan Collins, of Maine."), "a full name")
check(not ug.in_response("Jacky Rosen", "name", "If no, then provide the name of the senator for Nevada"),
      "the extractor supplied Nevada's real senator")
check(not ug.in_response("Collins Susan", "name", "Susan Collins"), "word order matters")
check(ug.in_response("Max Baucu", "name", "Max Baucus") is False,
      "a strip damaged name (defect 9) is not a token match, so fix defect 9 first")

print("titles: 80% of the words, anywhere")
check(ug.in_response("Human Compatible: AI and the Problem of Control", "title",
                     "Russell, Human Compatible: Artificial Intelligence and the Problem of Control"),
      "a slightly reworded title passes (7 of 8 words)")
check(not ug.in_response("Financial Accounting Theory", "title",
                         "Q: what is accounting? A: it is a field of study."),
      "an invented textbook title fails")

print("counts and factors")
check(ug.in_response(2, "count", "The input has two integers n and m."),
      "a number word counts")
check(ug.in_response("21", "count", "twenty one rows"), "twenty one is 21")
check(ug.in_response(7, "count", "there are 7 of them"), "digits count")
check(not ug.in_response(7, "count", "red, orange, yellow"), "7 was never written")
check(ug.in_response(13, "factor", "7411 = 3 x 3 x 7 x 13"), "a factor as a token")
check(not ug.in_response(1, "factor", "7411 = 3 x 3 x 7 x 13"), "1 is not a substring match of 13")
check(not ug.in_response("x", "count", "anything"), "a count that is not a number is absent")

print("flights")
check(ug.in_response(("C", "H"), "flight", "C -> H -> F"), "both letters present")
check(not ug.in_response(("C", "Z"), "flight", "C -> H -> F"), "a city never named")
check(not ug.in_response(("A", "B"), "flight", "a big airport"), "lowercase letters in words do not count")

print("absent and unit_key")
check(ug.absent(["August", "January"], "item", R) == ["January"], "absent keeps order")
check(ug.unit_key("Max  Baucus.") == ug.unit_key("max baucus"), "one key per spelling")
check(ug.unit_key(("C", "H")) == "C->H", "a flight key")

print("framing lines")
for line in ("Here are the independent facts broken down from the passage:",
             "Note that the passage is mostly repetitive.",
             "If you provide a real passage, I'll be happy to help.",
             "There is no factual content here."):
    check(ug.framing_line(line, "Some summary text."), "framing: %s" % line[:30])
check(not ug.framing_line("Ada was born in 1815.", "x"), "a fact is not framing")
check(ug.framing_line("The passage is about cats.", "Cats sleep a lot."),
      "The passage line when the response never says passage")
check(not ug.framing_line("The passage describes a storm.", "This passage describes a storm."),
      "kept when the response itself says passage")
check(not ug.framing_line("Notebooks were sold.", "x"), "Note must be a whole word")

try:
    ug.in_response("a", "planet", "a")
    check(False, "an unknown kind raises")
except ValueError:
    check(True, "an unknown kind raises")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
