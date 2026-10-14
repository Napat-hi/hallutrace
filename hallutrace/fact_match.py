"""hallutrace/fact_match.py  |  find a year in an infini-gram window text (pure, no I/O)

A window is a list of [text, label] spans; the needle span (the person's name as
searched) has label "0", the others null. Work in `text = before + needle + after`;
the needle occupies [len(before), len(before) + len(needle)).

A year occurrence is a match of (?<!\\d)YEAR(?!\\d) in `before` or `after` ONLY, never
inside the needle (a needle like "Patrick O'Brien (footballer, born 1875)" would
match its own birth year). Its distance is the number of characters between the
occurrence and the nearest needle edge. A half-width W (characters each side of the
needle) contains an occurrence if its whole span lies within W characters of the
needle; half_width None means the whole stored window.

Matchers: `bare` (any occurrence) and `cue` (a relation cue for the fact type, see
CUES). Cue words end at most CUE_GAP characters before the year with no other 3 or 4
digit number in between; "b." and "d." count only directly before the year; a full
lifespan "B-D" counts for both facts only when both numbers are this person's years,
an open lifespan "(B- )" counts for birth. Exact year only: a source one year off is
not a match (known limitation, not fixed here).

The cue table lives in one place (CUES) so it is easy to change.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

CUE_GAP = 40  # a word cue must end at most this many characters before the year

# Word cues per fact. Whole-word, case-insensitive. "b." and "d." are handled by a
# stricter rule (directly followed by optional whitespace and the year).
CUES: dict[str, tuple[str, ...]] = {
    "birth": ("born", "birth", "b."),
    "death": ("died", "death", "d.", "passed away"),
}

FACTS = tuple(CUES)

# Hyphen-minus, en dash, em dash (the em dash only as an escape, never literal).
_DASH = r"[-\u2013\u2014]"
_APPROX = r"(?:circa|ca\.|c\.)?"  # "c. 1197-1269": an approximate marker is allowed
_BIG_NUMBER = re.compile(r"(?<!\d)\d{3,4}(?!\d)")  # blocks a word cue from the year
_SPACE = re.compile(r"\s+")

SNIPPET_NEAR = 600  # gap up to this: one stretch; beyond: two pieces
SNIPPET_PAD = 150
DEDUP_SPAN = 200


@dataclass(frozen=True)
class Occurrence:
    """One year occurrence. `start`/`end` are offsets into text = before + needle +
    after (end exclusive); `side` is "before" or "after"; `distance` is the number
    of characters between the occurrence and the nearest needle edge."""

    side: str
    start: int
    end: int
    distance: int


def window_text(spans, needle_offset=None, disp_len=None) -> tuple[str, str, str]:
    """(before, needle, after) from a window's spans ([[text, label], ...]).

    A window can hold several spans labelled "0" (the name occurs more than once;
    1,077 of 5,616 pilot windows) and needle_offset is a TOKEN offset into a window
    of disp_len tokens, not a character offset. The chosen needle is the only "0"
    span; with several, the one whose start, as a fraction of the whole text length,
    is closest to needle_offset / disp_len (the first if either is None). Every
    other "0" span stays in before/after as ordinary text, but its digits are
    replaced by "#" (same length), so a year inside the needle string itself
    ("... born 1875)") can never be found in a repeated needle. No "0" span raises
    ValueError."""
    starts, pos = [], 0
    for text, _label in spans:
        starts.append(pos)
        pos += len(text)
    total = pos
    # The label is the string "0"; int 0 is accepted, but None and False are not
    # needles (False == 0 in Python, hence the bool check).
    needles = [
        i for i, (_t, lab) in enumerate(spans)
        if lab == "0" or (isinstance(lab, int) and not isinstance(lab, bool) and lab == 0)
    ]
    if not needles:
        raise ValueError("window has no needle span (label \"0\")")
    chosen = needles[0]
    if len(needles) > 1 and needle_offset is not None and disp_len:
        # Token offset over window length, mapped to a character fraction of the text.
        target = needle_offset / disp_len
        chosen = min(
            needles, key=lambda i: abs((starts[i] / total if total else 0.0) - target)
        )  # min keeps the first on ties
    parts = []
    for i, (text, _lab) in enumerate(spans):
        if i in needles and i != chosen:
            text = re.sub(r"\d", "#", text)  # same length, so offsets are unchanged
        parts.append(text)
    return "".join(parts[:chosen]), spans[chosen][0], "".join(parts[chosen + 1:])


def find_year(before: str, needle: str, after: str, year: int) -> list[Occurrence]:
    """Every occurrence of `year` (whole number, (?<!\\d)YEAR(?!\\d)) in before or
    after, never inside the needle, in text order."""
    text = before + needle + after
    ns, ne = len(before), len(before) + len(needle)
    out = []
    # Search the whole text so that a digit touching the needle still blocks the
    # match, then drop anything that overlaps the needle.
    for m in re.finditer(r"(?<!\d)%d(?!\d)" % year, text):
        s, e = m.start(), m.end()
        if e <= ns:
            out.append(Occurrence("before", s, e, ns - e))
        elif s >= ne:
            out.append(Occurrence("after", s, e, s - ne))
    return out


def in_width(occurrence: Occurrence, half_width: int | None) -> bool:
    """True if the whole occurrence lies within `half_width` characters of the
    needle; None means the full stored window."""
    if half_width is None:
        return True
    # distance is to the near end of the occurrence; the far end must fit too.
    return occurrence.distance + (occurrence.end - occurrence.start) <= half_width


def cue_match(text: str, occurrence: Occurrence, year: int, fact: str, person_years: dict) -> bool:
    """True if the occurrence carries a relation cue for `fact` ("birth" or
    "death"). `text` = before + needle + after; `person_years` is
    {"birth": int | None, "death": int | None} for this person."""
    if fact not in CUES:
        raise ValueError("unknown fact %r" % (fact,))
    start, end = occurrence.start, occurrence.end
    # 1) word cues ending at most CUE_GAP characters before the year.
    for cue in CUES[fact]:
        if cue.endswith("."):
            # "b." / "d.": only directly before the year ("B." is often an initial).
            if re.search(r"(?<!\w)%s\s*$" % re.escape(cue), text[:start], re.I):
                return True
            continue
        pat = re.compile(r"\b%s\b" % r"\s+".join(map(re.escape, cue.split())), re.I)
        lo = max(0, start - CUE_GAP - 64)
        for m in pat.finditer(text, lo):
            if m.end() > start:
                break
            if m.end() >= start - CUE_GAP and not _BIG_NUMBER.search(text[m.end():start]):
                return True
    # 2) lifespans, which need this person's own years.
    b, d = person_years.get("birth"), person_years.get("death")
    if b is None:
        return False
    if d is not None:
        pat = re.compile(
            r"(?<!\d)(%d)\s*%s\s*%s\s*(%d)(?!\d)" % (b, _DASH, _APPROX, d)
        )
        for m in pat.finditer(text):
            if fact == "birth" and m.start(1) == start:
                return True
            if fact == "death" and m.end(2) == end:
                return True
    if fact == "birth":
        pat = re.compile(r"\(\s*%s\s*(?<!\d)(%d)\s*%s(?!\s*\d)" % (_APPROX, b, _DASH))
        for m in pat.finditer(text):
            if m.start(1) == start:
                return True
    return False


def snippet(before: str, needle: str, after: str, occurrence: Occurrence) -> str:
    """Whitespace-collapsed context with the needle as [[...]] and the year as
    <<...>>. Gap needle to year <= 600 characters: the whole stretch plus 150
    characters on the outer sides of both; else 150 around the needle and 150
    around the year joined by " [...] "."""
    text = before + needle + after
    ns, ne = len(before), len(before) + len(needle)
    os_, oe = occurrence.start, occurrence.end
    marks_n, marks_o = (ns, ne, "[[", "]]"), (os_, oe, "<<", ">>")
    gap = os_ - ne if occurrence.side == "after" else ns - oe
    if gap <= SNIPPET_NEAR:
        out = _render(text, min(ns, os_) - SNIPPET_PAD, max(ne, oe) + SNIPPET_PAD,
                      sorted([marks_n, marks_o]))
    else:
        piece_n = _render(text, ns - SNIPPET_PAD, ne + SNIPPET_PAD, [marks_n])
        piece_o = _render(text, os_ - SNIPPET_PAD, oe + SNIPPET_PAD, [marks_o])
        pieces = [piece_o, piece_n] if os_ < ns else [piece_n, piece_o]
        out = " [...] ".join(pieces)
    return _SPACE.sub(" ", out).strip()


def _render(text: str, lo: int, hi: int, marks: list) -> str:
    """text[lo:hi] (clipped) with each (start, end, open, close) mark inserted."""
    lo, hi = max(0, lo), min(len(text), hi)
    out, pos = [], lo
    for s, e, op, cl in marks:
        out += [text[pos:s], op, text[s:e], cl]
        pos = e
    out.append(text[pos:hi])
    return "".join(out)


def dedup_key(text: str, occurrence: Occurrence) -> str:
    """sha1 hex of a 200-character slice of `text` centred on the occurrence,
    lowercased, whitespace collapsed to single spaces, digits kept (texts with
    different years never merge)."""
    centre = (occurrence.start + occurrence.end) // 2
    piece = text[max(0, centre - DEDUP_SPAN // 2): centre + DEDUP_SPAN // 2]
    piece = _SPACE.sub(" ", piece.lower()).strip()
    return hashlib.sha1(piece.encode("utf-8")).hexdigest()
