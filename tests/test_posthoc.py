"""Tests for hallutrace/posthoc.py, the unit filters on verifier output."""

import json
import sys

from hallutrace import posthoc as ph
from hallutrace import unit_guard as ug

OK = "   OK  "
failures = []


def check(cond, label):
    print((OK if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def dropped(row):
    return [(d["unit"], d["reason"]) for d in json.loads(row["dropped_units"])]


print("task from the CSV header")
check(ph.task_of_columns(["response", "list", "list_condition", "parsed_response"]) == "false_presuppositions", "FP")
check(ph.task_of_columns(["list_condition", "count", "true_answer"]) == "rationalization_numerical", "RN")
check(ph.task_of_columns(["category", "prompt", "yesno"]) == "rationalization_binary", "RB")
check(ph.task_of_columns(["response", "s2_titles"]) == "scientific_attribution", "SA")
check(ph.task_of_columns(["passage", "response", "atomic_units"]) == "decompose", "summarization and simplification")
check(ph.task_of_columns(["response", "name1", "name2"]) is None, "historical_events passes through")
check(ph.task_of_columns(["name", "atoms"]) is None, "Biographies passes through (it has its own filter)")

print("false_presuppositions: the planet example")
row = {"response": "\n\nThe answer is:", "list_condition": "['zinc', 'zirconium']",
       "parsed_response": " Mercury, Mars, Mimas, Mab, Miranda, Metis, Melpomene \n\n ",
       "atomic_units": "[' Mercury', 'Mars']", "hallucinated_atomic_units": "['mars']"}
f = ph.filter_row("false_presuppositions", row)
check(f["atomic_units"] == "[]" and f["hallucinated_atomic_units"] == "[]", "every invented planet goes")
check(f["all_units_dropped"] == "True", "and the row is marked all_units_dropped")
check(len(dropped(f)) == 7 and all(r == ph.NOT_IN_RESPONSE for _, r in dropped(f)), "each listed as not_in_response")
row = {"response": "Zinc, zirconium and gold", "list_condition": "['zinc', 'zirconium']",
       "parsed_response": "Zinc, zirconium, gold, NONE", "atomic_units": "[]", "hallucinated_atomic_units": "[]"}
f = ph.filter_row("false_presuppositions", row)
check(f["hallucinated_atomic_units"] == "['gold']", "HALoGEN's rule on what is left: gold is not in list_condition")
check(("NONE", ph.NONE_TOKEN) in dropped(f), "a trailing NONE is dropped as none_token")
check(f["all_units_dropped"] == "False", "units remain, so not all dropped")
f = ph.filter_row("false_presuppositions", dict(row, response="no response. Zinc, gold"))
check(f["atomic_units"] == "[]", "HALoGEN's 'no response' rule still empties the row")

print("rationalization_numerical: the count and number words")
row = {"response": "The input has two integers n and m.", "count": "0", "list_condition": "[]",
       "parsed_response": " <Response>: 2, n, m, ai \n\n ", "atomic_units": "[2, 'n', 'm', 'ai']",
       "hallucinated_atomic_units": "{'2', 'n', 'm', 'ai'}"}
f = ph.filter_row("rationalization_numerical", row)
check(f["atomic_units"] == "[2, 'n', 'm']", "2 stays (two), n and m stay, ai goes")
check(f["hallucinated_atomic_units"] == "['2', 'm', 'n']", "and all three are still wrong against count 0")
row = dict(row, response="7 colours", parsed_response="7, red, orange", count="7", list_condition="['red']")
f = ph.filter_row("rationalization_numerical", row)
check(f["hallucinated_atomic_units"] == "[]", "a count equal to the truth and invented colours dropped leave nothing")

print("rationalization_binary: defect 9 repaired before the guard")
check(ph.repair_senator("Max Baucu", "Max Baucus.</s>") == "Max Baucus", "Max Baucu is found again as Max Baucus")
check(ph.repair_senator("Susan Collin", "I think Susan Collins, of Maine") == "Susan Collins", "and Susan Collins")
check(ph.repair_senator("Jacky Rosen", "Jacky Rosen") == "Jacky Rosen", "an undamaged name is unchanged")
# the repair must not grow across a space into the next word
check(ph.repair_senator("Max Baucu", "The senator is Max Baucus.") == "Max Baucus",
      "a name after 'is' does not take the s of 'is'")
check(ph.repair_senator("Chris Murphy", "Chris Murphy served") == "Chris Murphy",
      "a name before 'served' does not take its s")
check(ph.repair_senator("Susan Collin", "Susan Collins, Maine") == "Susan Collins",
      "the cut s is still found again when a comma follows")
check(ug.in_response(ph.repair_senator("Max Baucu", "The senator is Max Baucus."), "name",
                     "The senator is Max Baucus."), "and the repaired name passes the response check")
row = {"response": "Yes, Max Baucus went there.", "category": "senator", "yesno": "['yes']",
       "parsed_response": "Max Baucus</s>", "atomic_units": "['yes', 'Max Baucu']",
       "hallucinated_atomic_units": "['Max Baucu']"}
f = ph.filter_row("rationalization_binary", row)
check(f["atomic_units"] == "['yes', 'Max Baucus']", "without the repair the guard would drop him as invented")
check(f["hallucinated_atomic_units"] == "['Max Baucus']", "every named senator is hallucinated, as HALoGEN counts it")
row = {"response": "If no, then provide the name of the senator for Nevada", "category": "senator",
       "yesno": "[]", "parsed_response": "Jacky Rosen", "atomic_units": "['Jacky Rosen']",
       "hallucinated_atomic_units": "['Jacky Rosen']"}
f = ph.filter_row("rationalization_binary", row)
check(f["atomic_units"] == "[]" and f["all_units_dropped"] == "True", "an invented Jacky Rosen goes")
row = {"response": "No, 7411 = 3 x 3 x 7 x 13", "category": "primality", "yesno": "['no']",
       "parsed_response": "3, 3, 7, 13, 17", "atomic_units": "['no', 3, 3, 7, 13, 17]",
       "hallucinated_atomic_units": "[3, 3, 7, 13, 17]"}
f = ph.filter_row("rationalization_binary", row)
check(f["atomic_units"] == "['no', 3, 3, 7, 13]", "a factor never written goes, yes/no stays")
row = {"response": "Yes: C -> H -> E", "category": "graph", "yesno": "['yes']", "parsed_response": "",
       "atomic_units": "['yes', ('C', 'H'), ('H', 'Z')]", "hallucinated_atomic_units": "[]"}
f = ph.filter_row("rationalization_binary", row)
check(f["atomic_units"] == "['yes', ['C', 'H']]", "a flight to a city never named goes")

print("scientific_attribution")
row = {"response": "See Human Compatible: Artificial Intelligence and the Problem of Control.",
       "atomic_units": "['Human Compatible: AI and the Problem of Control', 'Financial Accounting Theory']",
       "hallucinated_atomic_units": "['Financial Accounting Theory']", "unverifiable_atomic_units": "[]"}
f = ph.filter_row("scientific_attribution", row)
check(f["atomic_units"] == "['Human Compatible: AI and the Problem of Control']", "a reworded real title stays")
check(f["hallucinated_atomic_units"] == "[]", "the invented textbook leaves the hallucinated list too")

print("summarization and simplification: framing lines")
row = {"passage": "p", "response": "Cats sleep a lot.",
       "atomic_units": "Here are the independent facts: | True |\n Cats sleep. | True |\n Cats fly. | False |\nNote that this is short. | False |",
       "hallucinated_atomic_units": ""}
f = ph.filter_row("decompose", row)
check(f["atomic_units"] == " Cats sleep. | True |\n Cats fly. | False |", "two framing lines go, facts stay")
check(f["hallucinated_atomic_units"] == " Cats fly. | False |", "hallucinated is the False facts that remain")
check([r for _, r in dropped(f)] == [ph.FRAMING, ph.FRAMING], "both listed as framing")

row = dict(row, atomic_units="Cats sleep. | True |\n  | False |")
check(("| False |", ph.BLANK) in dropped(ph.filter_row("decompose", row)),
      "a blank atom is logged as blank instead of vanishing")

print("zero unit rows: how each task enters h")
fp0 = ph.filter_row("false_presuppositions", {"response": "There are none.", "list_condition": "['zinc']",
                                              "parsed_response": "NONE", "atomic_units": "[]",
                                              "hallucinated_atomic_units": "[]"})
check(fp0["zero_unit_rule"] == ph.NOT_HALLUCINATED, "false_presuppositions: kept in, not hallucinated")
rb0 = {"response": "yes, 7 is prime", "category": "primality", "parsed_response": "",
       "atomic_units": "['yes']", "hallucinated_atomic_units": "[]"}
check(ph.filter_row("rationalization_binary", rb0)["zero_unit_rule"] == ph.NOT_HALLUCINATED,
      "rationalization_binary: a bare yes has no units and is kept in")
sa0 = {"response": "I don't know of any.", "atomic_units": "[]", "hallucinated_atomic_units": "[]",
       "unverifiable_atomic_units": "[]"}
check(ph.filter_row("scientific_attribution", sa0)["zero_unit_rule"] == ph.OUT_OF_USABLE,
      "scientific_attribution: out of n_usable")
check(ph.filter_row("decompose", {"passage": "p", "response": "Here is it.", "atomic_units": "Here are the facts: | True |",
                                  "hallucinated_atomic_units": ""})["zero_unit_rule"] == ph.OUT_OF_USABLE,
      "summarization and simplification: out of n_usable, also when the filter emptied it")
rn0 = {"response": "\n0\n\nThere are none.", "count": "0", "list_condition": "[]",
       "parsed_response": "", "atomic_units": "[]", "hallucinated_atomic_units": "[]"}
check(ph.filter_row("rationalization_numerical", rn0)["zero_unit_rule"] == ph.ANSWER_0_RIGHT,
      "rationalization_numerical: first non-empty line says 0 and gold is 0, right")
check(ph.filter_row("rationalization_numerical", dict(rn0, count="3.0"))["zero_unit_rule"] == ph.ANSWER_0_WRONG,
      "and the same answer against gold 3 is wrong")
check(ph.filter_row("rationalization_numerical", dict(rn0, response="I am not sure."))["zero_unit_rule"]
      == ph.OUT_OF_USABLE, "with no 0 on that line it is out of n_usable")
check(ph.says_zero("There are 10 of them") is False, "10 is not 0, as in HALoGEN's digit match")
check(ph.filter_row("false_presuppositions", dict(fp0, parsed_response="Zinc, gold",
                                                  response="Zinc and gold"))["zero_unit_rule"] == "",
      "a row with units has no rule")

# decomposer lines: the verdict is the last field, so an atom may hold "|"
check(ph._atom_lines(" A is B. | True |\n A is C. | False |") == [("A is B.", True), ("A is C.", False)],
      "one atom per line, verdict after the last |")
check(ph._atom_lines(" (laureano | born: january in kribi | x) | True |") == [("(laureano | born: january in kribi | x)", True)],
      "an atom holding | keeps its text and its True verdict")
check(ph._atom_lines(" talk | contribs | False |") == [("talk | contribs", False)], "and its False verdict")
check(ph._atom_lines("| | True |\n| Fact # | Description | | False |") == [], "a line with a blank first field is no atom")
check(ph._atom_lines(" A is B. |") == [("A is B.", False)], "an atom with no verdict is not supported")

try:
    ph.filter_row("historical_events", {})
    check(False, "an unfiltered task raises")
except ValueError:
    check(True, "an unfiltered task raises")
check(ph.parse_list("") == [] and ph.parse_list("set()") == [] and ph.parse_list("{'a'}") == ["a"],
      "parse_list reads lists, sets and empties as the verifiers write them")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
