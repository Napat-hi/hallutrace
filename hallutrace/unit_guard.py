"""Is an extracted unit actually in the response?

HALoGEN asks Llama to pull units out of a model's response, and Llama sometimes
answers the question itself instead; a unit the model never wrote is then scored
as its hallucination. in_response() checks a unit against the text the verifier
was given:

    item, name   the unit's words appear in the response, in order and adjacent,
                 after lowercasing and turning punctuation into spaces. A one-word
                 unit also matches a one-character word followed by a
                 one-character word or a number ("a 1" holds `a1`)
    title        at least 80% of the title's words appear in the response
    count        the number appears as digits or as a number word
    factor       the number appears as a token
    flight       both city letters appear as standalone capitals

framing_line() finds a line the decomposer adds about its own output ("Here are
the independent facts...", "Note that..."). "The passage ..." counts only when
the response never says "passage". Nothing here calls a model.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]

TITLE_SHARE = 0.8


def tokens(text: str) -> List[str]:
    return _TOKEN.findall(str(text).lower())


def _contains(seq: Sequence[str], sub: Sequence[str]) -> bool:
    if not sub:
        return False
    n = len(sub)
    return any(list(seq[i:i + n]) == list(sub) for i in range(len(seq) - n + 1))


def number_words(n: int) -> List[List[str]]:
    """Word forms of 0 to 99, as token lists: 21 gives twenty one."""
    if n < 0 or n > 99:
        return []
    if n < 20:
        return [[_ONES[n]]]
    tens, ones = divmod(n, 10)
    return [[_TENS[tens]]] if ones == 0 else [[_TENS[tens], _ONES[ones]]]


def in_response(unit, kind: str, response: str) -> bool:
    """True when `unit` of this kind can be found in `response`."""
    resp = tokens(response)
    if kind in ("item", "name"):
        words = tokens(unit)
        if _contains(resp, words):
            return True
        # one-character words only, so "to go" never matches Togo
        return len(words) == 1 and any(
            len(a) == 1 and (len(b) == 1 or b.isdigit()) and a + b == words[0]
            for a, b in zip(resp, resp[1:]))
    if kind == "title":
        words = tokens(unit)
        if not words:
            return False
        have = set(resp)
        return sum(w in have for w in words) >= TITLE_SHARE * len(words)
    if kind in ("count", "factor"):
        try:
            n = int(unit)
        except (TypeError, ValueError):
            return False
        if str(n) in resp:
            return True
        return kind == "count" and any(_contains(resp, w) for w in number_words(n))
    if kind == "flight":
        a, b = unit
        caps = set(re.findall(r"\b([A-Z])\b", str(response)))
        return a in caps and b in caps
    raise ValueError("unknown unit kind %r" % kind)


def absent(units: Iterable, kind: str, response: str) -> list:
    """The units that are not in the response, in order."""
    return [u for u in units if not in_response(u, kind, response)]


_FRAMING = re.compile(r"^\s*(here is|here are|note|if you|there is no|the passage)\b", re.I)


def framing_line(line: str, response: str) -> bool:
    """A decomposer line about its own output rather than a fact."""
    m = _FRAMING.match(str(line))
    if not m:
        return False
    if m.group(1).lower() == "the passage":
        return "passage" not in tokens(response)
    return True


def unit_key(unit) -> str:
    """One spelling per unit, for comparing two extractions of the same text."""
    if isinstance(unit, (tuple, list)):
        return "->".join(str(u) for u in unit)
    return " ".join(tokens(unit))
