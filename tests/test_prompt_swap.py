"""Tests for hallutrace/prompt_swap.py and LLMClient's rewrite hook."""

import sys

from hallutrace import extraction as ex
from hallutrace import prompt_swap as ps
from hallutrace.llm_client import LLMClient

OK = "   OK  "
failures = []


def check(cond, label):
    print((OK if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


R = "Sure! 1. Mercury; <Instruction>: odd text\n with a newline."

print("recognise: every released extraction prompt, response recovered exactly")
cases = [
    ("false_presuppositions", ex.prompt_false_presuppositions(R), ex.prompt_false_presuppositions_v2(R)),
    ("rationalization_numerical", ex.prompt_rationalization_numerical(R),
     ex.prompt_rationalization_numerical_v2(R)),
    ("scientific_attribution", ex.prompt_scientific_attribution(R), ex.prompt_scientific_attribution_v2(R)),
]
for cat in ("primality", "graph", "senator"):
    cases.append(("rationalization_binary_" + cat, ex.prompt_rationalization_binary(R, cat),
                  ex.prompt_rationalization_binary_v2(R, cat)))
for label, released, v2 in cases:
    check(ps.recognise(released) == (label, R), "%s is recognised, response intact" % label)
    check(ps.to_v2(released) == v2, "%s becomes its v2 prompt" % label)
    check(ps.recognise(v2) is None, "%s v2 is not recognised again, so no double swap" % label)
check(ps.recognise(ex.prompt_false_presuppositions("")) == ("false_presuppositions", ""),
      "an empty response is still recognised")

print("everything else goes out unchanged")
he = ex.prompt_historical_events(R, "Ada", "Babbage")
check(ps.recognise(he) is None, "historical_events is not an extraction prompt")
check(ps.recognise(ex.prompt_entailment("p", "c")) is None, "nor is the entailment judge")
check(ps.recognise(ex.prompt_decompose(R)) is None, "the decomposer keeps its released prompt")
check(ps.recognise("hello") is None, "a random prompt")

print("Swapper, the LLMClient hook")
sw = ps.Swapper()
check(sw("completion", cases[0][1]) == cases[0][2], "a released completion prompt is swapped")
check(sw("completion", he) == he, "an unknown completion prompt passes")
msgs = [{"role": "user", "content": cases[0][1]}]
check(sw("chat", msgs) is msgs, "chat calls are never touched")
check(sw.stats() == {"swapped": {"false_presuppositions": 1}, "passed": 2}, "and all of it is counted")


class Backend:
    def __init__(self):
        self.sent = []

    def complete(self, model, prompt, **decode):
        self.sent.append(prompt)
        return "reply"

    def chat(self, model, messages, **decode):
        return "chat reply"


class Cache:
    def __init__(self):
        self.d = {}

    def get(self, model, prompt):
        return self.d.get((model, prompt))

    def put(self, model, prompt, text):
        self.d[(model, prompt)] = text


print("LLMClient sends, and caches, what was actually sent")
b, c = Backend(), Cache()
client = LLMClient(b, c, rewrite=ps.Swapper())
client.completions.create(model="m", prompt=cases[1][1])
check(b.sent == [cases[1][2]], "the backend receives the v2 prompt")
check(("m", cases[1][2]) in c.d and ("m", cases[1][1]) not in c.d,
      "the cache key is the v2 prompt, so released and v2 replies never mix")
b2 = Backend()
plain = LLMClient(b2, None)
plain.completions.create(model="m", prompt=cases[1][1])
check(b2.sent == [cases[1][1]], "without rewrite nothing changes")

print("swap_missing: a v2 unit that swapped nothing is caught")
from hallutrace.prompt_swap import V2_TASKS, swap_missing  # noqa: E402
check(swap_missing("false_presuppositions", 12, 0), "FP used the LLM 12 times and swapped none: caught")
check(not swap_missing("false_presuppositions", 12, 12), "every call swapped: fine")
check(not swap_missing("rationalization_binary", 0, 0), "no LLM call at all, nothing to swap: fine")
check(not swap_missing("summarization", 30, 0), "summarization keeps the released decomposer: never caught")
check(swap_missing("rationalization_binary", 30, 15),
      "a mixed graph and senator unit that swapped 15 of 30: caught")
check(set(V2_TASKS) == {"false_presuppositions", "rationalization_numerical",
                        "rationalization_binary", "scientific_attribution"}, "V2_TASKS is the four extraction tasks")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
