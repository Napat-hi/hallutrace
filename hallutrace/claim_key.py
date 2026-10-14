"""claim_key: turning an extracted triple into a stable identity.

Two claims are the same claim when they get the same `claim_key`. Everything
downstream counts on that: corpus queries are made per claim, `n_h` and `n_c`
are stored per claim, and Phase 2's type is assigned per claim. So this file
decides, for the whole project, when two differently worded claims are one.

The rule:

    QID first, with an alias table. Use the extractor's qid when it resolves.
    When it does not, normalise the string (lowercase, strip punctuation,
    collapse whitespace) and look it up in a qid-to-string map: adopt the QID
    if exactly one matches, and refuse to merge plus flag the row if two or
    more do.

The alias table is data this module does not own: it is passed in.
`EMPTY_ALIASES` is the default so the module runs standalone and so tests need
no fixture file.

**The asymmetry is deliberate and is the only thing here worth arguing about.**
Over-merging and under-merging are not equally bad:

    over-merging   two different claims share one key, so their corpus counts
                   are added together. The result looks like one well attested
                   claim. Nothing downstream can detect it and nothing can undo
                   it without re-running the corpus queries
    under-merging  one claim gets two keys, so it is queried twice. The counts
                   are still right for each key. It costs wall clock

So every ambiguous case resolves towards under-merging. An ambiguous entity
keeps its normalised string as its key, which never collides with a QID keyed
entity, and the row is flagged so a human can look later. Two rows carrying the
same ambiguous string still share a key: that is the same text making the same
claim, which is the behaviour we would have had with no alias table at all, and
it is not a QID level merge.

The relation vocabulary is passed in too, through `vocabulary` (`relations.py`). The default accepts any relation and normalises it, so this module
does not silently invent a controlled vocabulary of its own.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Why an entity ended up with the key it has. Stored per claim so that the
# share of claims resting on each path can be reported, which is the only way
# to know whether the alias table is pulling its weight.
BY_QID = "qid"
BY_ALIAS = "alias"
BY_STRING = "string"
BY_AMBIGUOUS = "ambiguous"

# Flags travel with the claim rather than stopping the run.
FLAG_AMBIGUOUS = "ambiguous_alias"
FLAG_UNKNOWN_RELATION = "unknown_relation"
FLAG_EMPTY_FIELD = "empty_field"

_QID = re.compile(r"^Q[1-9][0-9]*$")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalise(text) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Nothing else.

    Accents are folded first, so that "Beyonce" and "Beyonce" with an accent
    do not become two claims. That is not in the rule as written; it is the
    same kind of normalisation as the three steps that are, and leaving it out
    would under-merge every accented name in the benchmark. Recorded here
    because it is an addition, not an interpretation.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _PUNCT.sub(" ", s.lower())
    return _SPACE.sub(" ", s).strip()


def is_qid(value) -> bool:
    return bool(value) and bool(_QID.match(str(value).strip()))


class AliasTable:
    """Normalised surface string to the QIDs that string can denote.

    The table is built elsewhere. This class only answers lookups, and it
    answers them on normalised keys so that the caller cannot accidentally look
    up a raw string and miss.

    `candidates(text)` returns every QID the string could mean, in a fixed
    order, so that a flagged row reports the same candidates on every run.
    """

    def __init__(self, mapping: Optional[Dict[str, Iterable[str]]] = None):
        self._map: Dict[str, Tuple[str, ...]] = {}
        for surface, qids in (mapping or {}).items():
            key = normalise(surface)
            if not key:
                continue
            seen: List[str] = []
            for q in qids if not isinstance(qids, str) else [qids]:
                q = str(q).strip()
                if is_qid(q) and q not in seen:
                    seen.append(q)
            if seen:
                self._map[key] = tuple(sorted(seen))

    def candidates(self, text) -> Tuple[str, ...]:
        return self._map.get(normalise(text), ())

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, text) -> bool:
        return normalise(text) in self._map


EMPTY_ALIASES = AliasTable()


class AnyRelation:
    """The default vocabulary: accepts everything, normalises the surface form.

    The controlled vocabulary (`relations.py`) is an object exposing
    `resolve(relation) -> (canonical_or_None, known: bool)`; keeping the seam means
    the claim key does not change shape when it is passed in.
    """

    def resolve(self, relation) -> Tuple[Optional[str], bool]:
        canonical = normalise(relation)
        return (canonical or None), bool(canonical)


@dataclass
class Resolved:
    """One entity, after the rule has run."""

    key: str
    source: str
    candidates: Tuple[str, ...] = ()

    @property
    def is_ambiguous(self) -> bool:
        return self.source == BY_AMBIGUOUS


def resolve_entity(
    text,
    qid: Optional[str] = None,
    aliases: Optional[AliasTable] = None,
) -> Resolved:
    """Apply the QID first rule to one entity.

    Order, top to bottom, first match wins:

        1. the extractor gave a qid that resolves          -> BY_QID
        2. the normalised string matches exactly one QID   -> BY_ALIAS
        3. it matches two or more                          -> BY_AMBIGUOUS
        4. it matches none                                 -> BY_STRING

    Cases 3 and 4 both key on the normalised string, so neither can collide
    with a QID keyed entity. They are kept apart because only case 3 is
    reported: it is the one where we know a merge was available and declined
    it.
    """
    table = aliases if aliases is not None else EMPTY_ALIASES
    surface = normalise(text)

    if is_qid(qid):
        return Resolved(key=str(qid).strip(), source=BY_QID)

    hits = table.candidates(surface)
    if len(hits) == 1:
        return Resolved(key=hits[0], source=BY_ALIAS, candidates=hits)
    if len(hits) > 1:
        return Resolved(key=surface, source=BY_AMBIGUOUS, candidates=hits)
    return Resolved(key=surface, source=BY_STRING)


@dataclass
class ClaimKey:
    """The identity of one claim, and how it was arrived at."""

    key: str
    claim_id: str
    subject: Resolved
    relation: Optional[str]
    object: Resolved
    flags: Tuple[str, ...] = ()

    @property
    def flagged(self) -> bool:
        return bool(self.flags)

    def as_row(self) -> dict:
        """Flat form, for the claim table. One value per column, no nesting."""
        return {
            "claim_id": self.claim_id,
            "claim_key": self.key,
            "subject_key": self.subject.key,
            "subject_source": self.subject.source,
            "relation": self.relation or "",
            "object_key": self.object.key,
            "object_source": self.object.source,
            "flags": "|".join(self.flags),
            "subject_candidates": "|".join(self.subject.candidates),
            "object_candidates": "|".join(self.object.candidates),
        }


def _claim_id(key: str) -> str:
    """A short, stable id for a key. Deterministic across runs and machines."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def claim_key(
    subject,
    relation,
    obj,
    subject_qid: Optional[str] = None,
    object_qid: Optional[str] = None,
    aliases: Optional[AliasTable] = None,
    vocabulary=None,
) -> ClaimKey:
    """Build the key for one extracted triple `(s, r, o_h)`.

    The key is the three resolved parts joined by a separator that cannot occur
    in a normalised string, since normalisation strips punctuation. So the key
    is readable in a CSV and still unambiguous.
    """
    vocab = vocabulary if vocabulary is not None else AnyRelation()

    s = resolve_entity(subject, subject_qid, aliases)
    o = resolve_entity(obj, object_qid, aliases)
    canonical, known = vocab.resolve(relation)

    flags: List[str] = []
    if s.is_ambiguous or o.is_ambiguous:
        flags.append(FLAG_AMBIGUOUS)
    if not known:
        flags.append(FLAG_UNKNOWN_RELATION)
    if not s.key or not o.key:
        flags.append(FLAG_EMPTY_FIELD)

    key = "|".join([s.key, canonical or "", o.key])
    return ClaimKey(
        key=key,
        claim_id=_claim_id(key),
        subject=s,
        relation=canonical,
        object=o,
        flags=tuple(flags),
    )


def summarise(keys: Sequence[ClaimKey]) -> dict:
    """Counts for the run log.

    `ambiguous` is the number this rule refused to merge. It is the price of
    the asymmetry, and it should be reported rather than discovered later: if
    it is large, the alias table needs work, and if it is zero the table may
    not be loaded at all.
    """
    counts = {
        "n": len(keys),
        "distinct": len({k.key for k in keys}),
        BY_QID: 0,
        BY_ALIAS: 0,
        BY_STRING: 0,
        BY_AMBIGUOUS: 0,
        "flagged": 0,
    }
    for k in keys:
        for part in (k.subject, k.object):
            counts[part.source] += 1
        if k.flagged:
            counts["flagged"] += 1
    return counts
