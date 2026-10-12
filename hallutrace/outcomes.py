"""What happened to a generation, decided before it is decomposed.

HALoGEN's code sends every generation to the decomposer, so an empty response,
an abstention and an answer with nothing to check all come out as zero atomic
units. This module tells them apart and records why a generation has no units
(`decomposer_empty_reason`).

It also holds the rule that drops a trailing clause cut off by the token limit
(strip_incomplete_tail), applied before anything reaches a verifier.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Values for decomposer_empty_reason; None means the generation was decomposed.
# The first three are known before decomposition, the last two only to the
# caller that ran the decomposer.
EMPTY_RESPONSE = "empty_response"
ABSTAINED = "abstained"
NO_UNITS_FOUND = "no_units_found"
FAILED_PARSE = "failed_parse"
FAILED_API = "failed_api"

REASONS = (EMPTY_RESPONSE, ABSTAINED, NO_UNITS_FOUND, FAILED_PARSE, FAILED_API)

# outcomes
OK = "ok"
EMPTY = "empty"
ABSTENTION = "abstention"

_TERMINAL = re.compile(r"[.!?]['\")\]]?\s*$")
# base models answer list tasks one item per line with no full stop
_LINE_END = re.compile(r"\n[ \t\r]*$")
_SENTENCE = re.compile(r"[^.!?]*[.!?]['\")\]]?(?:\s|$)")


def classify(generation, detector=None) -> Tuple[str, Optional[str]]:
    """Return (outcome, decomposer_empty_reason) before decomposition.

    `detector` is an AbstentionDetector, or None to skip the abstention check.
    An empty string is checked first: it matches no abstention phrase and would
    otherwise pass as an answer with no claims.
    """
    if generation is None or not isinstance(generation, str) or not generation.strip():
        return EMPTY, EMPTY_RESPONSE
    if detector is not None and detector.is_abstained(generation):
        return ABSTENTION, ABSTAINED
    return OK, None


def says_no_response(text) -> bool:
    """A false-presuppositions answer that opts out as its prompt asks.

    True only when the first non-empty line is "no response", ignoring case,
    outer spaces and final punctuation. HALoGEN's scorer matches the phrase
    anywhere, which also catches prompt echoes.
    """
    line = next((l for l in str(text or "").split("\n") if l.strip()), "")
    return line.strip().rstrip(".;,:!").strip().lower() == "no response"


def reason_after_decomposition(outcome: str, units) -> Optional[str]:
    """The reason to store after a decomposition that did not fail.

    A usable generation that yields no units gets `no_units_found`. When the
    decomposer raised, use record_decomposition instead.
    """
    if outcome != OK:
        return EMPTY_RESPONSE if outcome == EMPTY else ABSTAINED
    return None if units else NO_UNITS_FOUND


def row_is_valid(n_units, reason) -> bool:
    """Whether (n_units, reason) satisfies the generations table's constraints.

        n_units IS NULL   requires reason IS NULL      not decomposed yet
        n_units = 0       requires reason IS NOT NULL
        n_units > 0       requires reason IS NULL

    and reason, when set, is one of REASONS.
    """
    if reason is not None and reason not in REASONS:
        return False
    if n_units is None:
        return reason is None
    if n_units == 0:
        return reason is not None
    if n_units > 0:
        return reason is None
    return False


def record_decomposition(outcome: str, units=None, error=None) -> Tuple[int, Optional[str]]:
    """Turn one decomposition attempt into the (n_units, reason) pair to store.

    `error` is what the decomposer raised, or None. It is classified by type:
    errors from the HTTP client or an HTTP status give `failed_api` (worth a
    retry), anything else `failed_parse`.
    """
    if error is not None:
        return 0, (FAILED_API if _looks_like_api_error(error) else FAILED_PARSE)

    reason = reason_after_decomposition(outcome, units)
    n = len(units) if units else 0
    if n > 0 and reason is not None:
        # the caller passed an outcome that disagrees with its units
        raise ValueError(
            "outcome %r says %r but %d units were produced; one of the two is "
            "wrong and the store would reject the row" % (outcome, reason, n)
        )
    return n, reason


_API_ERROR_NAMES = (
    "timeout",
    "connection",
    "httperror",
    "http_error",
    "apierror",
    "api_error",
    "ratelimit",
    "rate_limit",
    "serviceunavailable",
    "toomanyrequests",
    "requestexception",
    "ssl",
)


def _looks_like_api_error(error) -> bool:
    """Classify by the exception type's name and bases, not its message.

    This covers requests, httpx, together and openai without importing them.
    """
    if isinstance(error, type):
        klass = error
    else:
        klass = type(error)
    names = [c.__name__.lower() for c in getattr(klass, "__mro__", [klass])]
    for name in names:
        if any(marker in name for marker in _API_ERROR_NAMES):
            return True
    if isinstance(error, (OSError,)):
        return True
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    return isinstance(status, int) and status >= 400


def ends_mid_clause(text: str) -> bool:
    """True when the text stops inside a line that has no terminal punctuation."""
    return bool(text) and not _TERMINAL.search(text) and not _LINE_END.search(text)


def strip_incomplete_tail(text: str) -> Tuple[str, str]:
    """Drop the unfinished last line or clause, and nothing more.

    Returns (kept, dropped). The tail starts after the last complete sentence
    or the last line break, whichever comes later. A response with no boundary
    loses everything, and the caller should treat the empty `kept` as
    `no_units_found`.
    """
    if not isinstance(text, str) or not text:
        return "", ""
    if not ends_mid_clause(text):
        return text, ""

    end = 0
    matches = list(_SENTENCE.finditer(text))
    if matches:
        end = matches[-1].end()
    line = text.rfind("\n") + 1  # 0 when there is no line break
    end = max(end, line)
    return text[:end].rstrip(), text[end:]


def summarise(records, detector=None) -> dict:
    """Outcome counts for a batch of records."""
    counts = {OK: 0, EMPTY: 0, ABSTENTION: 0}
    for rec in records:
        text = rec.get("response") if isinstance(rec, dict) else rec
        outcome, _ = classify(text, detector)
        counts[outcome] += 1
    total = sum(counts.values())
    counts["n"] = total
    counts["scored_share"] = counts[OK] / total if total else 0.0
    return counts
