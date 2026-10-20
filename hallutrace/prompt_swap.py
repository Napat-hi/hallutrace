"""Swap HALoGEN's released extraction prompts for v2 as they are sent.

HALoGEN builds each extraction prompt inside the verifier. Every call goes
through llm_client.LLMClient, whose rewrite hook (Swapper) recognises a released
prompt, recovers the response inside it, and sends the v2 prompt for the same
response. HALoGEN's code is unchanged.

A prompt matches when it starts and ends with the fixed text a released builder
in extraction.py puts around the response. Anything else (historical_events,
the entailment judge) goes out unchanged and is counted in `Swapper.passed`.
The decomposer keeps its released prompt.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, List, Optional, Tuple

from hallutrace import extraction as ex

_CUT = "\x00HALLUTRACE_RESPONSE\x00"

# (label, released builder, v2 builder), each taking only the response.
_PAIRS: List[Tuple[str, Callable[[str], str], Callable[[str], str]]] = [
    ("false_presuppositions", ex.prompt_false_presuppositions, ex.prompt_false_presuppositions_v2),
    ("rationalization_numerical", ex.prompt_rationalization_numerical,
     ex.prompt_rationalization_numerical_v2),
    ("scientific_attribution", ex.prompt_scientific_attribution,
     ex.prompt_scientific_attribution_v2),
]
for _cat in ex.RB_CATEGORY_PROMPTS:
    _PAIRS.append((
        "rationalization_binary_" + _cat,
        (lambda c: lambda r: ex.prompt_rationalization_binary(r, c))(_cat),
        (lambda c: lambda r: ex.prompt_rationalization_binary_v2(r, c))(_cat),
    ))


def _frames():
    out = []
    for label, released, v2 in _PAIRS:
        head, tail = released(_CUT).split(_CUT)
        out.append((label, head, tail, v2))
    # longest head first, so a shorter lookalike never claims a prompt
    return sorted(out, key=lambda f: -len(f[1]))


_FRAMES = _frames()


def recognise(prompt: str) -> Optional[Tuple[str, str]]:
    """(label, response) for a released extraction prompt, else None."""
    for label, head, tail, _ in _FRAMES:
        if prompt.startswith(head) and prompt.endswith(tail) and len(prompt) >= len(head) + len(tail):
            return label, prompt[len(head):len(prompt) - len(tail)]
    return None


def to_v2(prompt: str) -> Optional[str]:
    """The v2 prompt for the same response, or None when not recognised."""
    got = recognise(prompt)
    if got is None:
        return None
    label, response = got
    v2 = {f[0]: f[3] for f in _FRAMES}[label]
    return v2(response)


# tasks whose extraction prompt has a v2
V2_TASKS = ("false_presuppositions", "rationalization_numerical",
            "rationalization_binary", "scientific_attribution")


def swap_missing(task: str, llm_calls: int, swapped: int) -> bool:
    """True when a v2 unit made more LLM calls than it swapped prompts.

    Every LLM call of these tasks is an extraction prompt, so fewer swaps than
    calls means some prompt went out as released. `llm_calls` includes cache
    hits, since the swap happens before the cache lookup.
    """
    return task in V2_TASKS and swapped < llm_calls


class Swapper:
    """The `rewrite` hook for LLMClient: completion prompts only, counted."""

    def __init__(self):
        self.swapped: Counter = Counter()
        self.passed = 0

    def __call__(self, kind: str, prompt):
        if kind != "completion" or not isinstance(prompt, str):
            self.passed += 1
            return prompt
        got = recognise(prompt)
        if got is None:
            self.passed += 1
            return prompt
        self.swapped[got[0]] += 1
        return to_v2(prompt)

    def stats(self) -> dict:
        return {"swapped": dict(self.swapped), "passed": self.passed}
