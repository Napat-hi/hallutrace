"""Drop the atoms Gemma's fact generator invents, after verification.

OpenFActScore asks Gemma to break one sentence into facts, few-shot. Gemma often
answers and then keeps going in the demo's format:

    - Rachel Bilson's height is 5'4".
    - Rachel Bilson's height is 163 cm.

    Please breakdown the following sentence into independent facts: The restaurant serves ...
    - The restaurant serves delicious pasta dishes.
    - The restaurant serves fresh seafood.

The invented sentence's facts are parsed as atoms of the person's response and mostly
come back False. A sentence that is only punctuation gets a refusal ("Please provide
the sentence ...") that is parsed the same way.

Each atom is verified on its own, so dropping these leaves the other verdicts valid.
The raw generator outputs come from the FActScore cache (`InstructGPT.pkl`, key
`{prompt}_0`, value `(text, scores)`). Dropped, by reason:

    header      an atom that contains the leaked "Please breakdown ..." line
    invented    an atom from the text after a leaked line, and not also in a kept part
    refusal     an atom from an output that refuses instead of answering
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set, Tuple

LEAK = "Please breakdown the following sentence"
REFUSAL_STARTS = ("please provide the sentence", "please provide a sentence")

_ATOM = re.compile(r"\| (.*?)\t(True|False) \|(?:\n|$)", re.S)


def parse_atoms(cell: str) -> List[Tuple[str, bool]]:
    """The `atoms` column of HALoGEN's Biographies output, as (atom, supported) pairs."""
    if not isinstance(cell, str):
        return []
    return [(a, v == "True") for a, v in _ATOM.findall(cell)]


def format_atoms(atoms: Iterable[Tuple[str, bool]], only_false: bool = False) -> str:
    """Back to HALoGEN's format: `| atom\\tTrue |` lines (only the False ones if asked)."""
    return "".join("| %s\t%s |\n" % (a, v) for a, v in atoms if not (only_false and v))


def _bullets(text: str) -> List[str]:
    return [b.strip() for b in re.split(r"(?:^|\n)\s*- ", text)[1:] if b.strip()]


def invented_from_cache(outputs: Iterable[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """From the generator's raw outputs: (invented atoms, refusal texts, kept atoms).

    Kept atoms are the bullets before any leaked line, so an invented atom that happens to
    match a real one is not dropped.
    """
    invented, refusals, kept = set(), set(), set()
    for out in outputs:
        text = out[0] if isinstance(out, tuple) else out
        if not isinstance(text, str):
            continue
        if text.strip().lower().startswith(REFUSAL_STARTS):
            refusals.add(text.strip())
            continue
        i = text.find(LEAK)
        head, tail = (text, "") if i < 0 else (text[:i], text[i:])
        kept.update(_bullets(head))
        invented.update(_bullets(tail))
    return invented - kept, refusals, kept


def filter_atoms(atoms: List[Tuple[str, bool]], invented: Set[str], refusals: Set[str]
                 ) -> Tuple[List[Tuple[str, bool]], Dict[str, List[str]]]:
    """Split one response's atoms into kept atoms and dropped atoms by reason."""
    kept, dropped = [], {"header": [], "invented": [], "refusal": []}
    for a, v in atoms:
        s = a.strip()
        if LEAK in s:
            dropped["header"].append(s)
        elif s in invented:
            dropped["invented"].append(s)
        elif any(s and s in r for r in refusals):
            dropped["refusal"].append(s)
        else:
            kept.append((a, v))
    return kept, dropped
