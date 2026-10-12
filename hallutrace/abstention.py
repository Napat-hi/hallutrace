"""Abstention detector: HALoGEN's phrase list with its missing comma fixed.

The released scorers/abstention_detector.py has no comma after "I am sorry", so
Python joins it with "Sorry" into "I am sorrySorry" and neither phrase matches.
The released list also lacks "I cannot" and "I am unable".

mode="fixed" (the default) uses the corrected list. mode="released" reproduces
the released behaviour, so the effect of the fix can be measured with
disagreements().
"""

from typing import Iterable, List, Optional

# the released list, in order; its last entries are in the tails below
_BASE: List[str] = [
    "I could not find any information",
    "The search results do not provide",
    "There is no information",
    "There are no search results",
    "there are no provided search results",
    "not provided in the search results",
    "is not mentioned in the provided search results",
    "There seems to be a mistake in the question",
    "Not sources found",
    "No sources found",
    "Try a more general question",
    "Unfortunately,",
    "There doesn't seem to be",
    "There does not seem to be",
    "I do not",
    "I don't",
    "**No relevant",
    "I'm afraid",
    "I am afraid",
    "I apologize,",
    "I'm sorry",
]

_TAIL_FIXED: List[str] = [
    "I am sorry",
    "Sorry",
]

# what Python builds from the released source
_TAIL_RELEASED: List[str] = [
    "I am sorrySorry",
]

_COMMON_TAIL: List[str] = [
    "provide more",
    "I am not familiar with",
    "I'm not familiar with",
]

# not in the released list
_ADDITIONS: List[str] = [
    "I cannot",
    "I can't",
    "I am unable",
    "I'm unable",
]

PHRASES_RELEASED: List[str] = _BASE + _TAIL_RELEASED + _COMMON_TAIL
PHRASES_FIXED: List[str] = _BASE + _TAIL_FIXED + _COMMON_TAIL + _ADDITIONS


class AbstentionDetector:
    """Substring matcher over a phrase list, as in HALoGEN.

    A non-string generation (NaN, None) counts as an abstention, as in the
    released code.
    """

    def __init__(self, mode: str = "fixed", extra: Optional[Iterable[str]] = None):
        if mode not in ("fixed", "released"):
            raise ValueError("mode must be 'fixed' or 'released', got %r" % mode)
        self.mode = mode
        self.phrases: List[str] = list(
            PHRASES_FIXED if mode == "fixed" else PHRASES_RELEASED
        )
        if extra:
            self.phrases.extend(extra)

    def is_abstained(self, generation) -> bool:
        if not isinstance(generation, str):
            return True
        return any(p in generation for p in self.phrases)

    def matched(self, generation) -> List[str]:
        """Every phrase that matches the generation."""
        if not isinstance(generation, str):
            return ["<non-string>"]
        return [p for p in self.phrases if p in generation]


def disagreements(generations: Iterable) -> dict:
    """Abstentions under each mode, and the generations where the modes differ."""
    rel = AbstentionDetector("released")
    fix = AbstentionDetector("fixed")
    gens = list(generations)
    r = [rel.is_abstained(g) for g in gens]
    f = [fix.is_abstained(g) for g in gens]
    return {
        "n": len(gens),
        "released_abstained": sum(r),
        "fixed_abstained": sum(f),
        "newly_caught": sum(1 for a, b in zip(r, f) if b and not a),
        "lost": sum(1 for a, b in zip(r, f) if a and not b),
    }
