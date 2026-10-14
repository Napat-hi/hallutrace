"""relations: the controlled relation vocabulary.

This replaces `AnyRelation` in `claim_key.py`. It exposes the same seam,
`resolve(relation) -> (canonical_or_None, known: bool)`, so `claim_key()` takes
it through `vocabulary=` and the key keeps its shape.

The vocabulary is data, not code. It lives in a separate JSON file (not yet
created) and this module only loads and validates it. The loader mirrors
`AliasTable`: the constructor takes an already-parsed mapping, and `load()` is
the one place that touches the disk.

**Synonyms are explicit only. There is no fuzzy matching.** The asymmetry from
`claim_key.py` applies to relations too:

    over-merging   two different relations share one canonical, so their claims
                   share one key and their corpus counts are added together.
                   Nothing downstream can detect it and nothing can undo it
                   without re-running the corpus queries
    under-merging  one relation gets two canonicals, so its claims are queried
                   twice. The counts are still right for each key

So a relation resolves only when its normalised text is a canonical or a listed
synonym. Anything else comes back as `(normalise(relation), False)`: flagged by
`claim_key()`, never dropped, and still keyed on its own text.

Canonicals appear verbatim in claim keys, so each must already be normal
(`normalise(c) == c`). A canonical that is not normal could never be looked up.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional, Tuple

from hallutrace.claim_key import normalise

_PID = re.compile(r"^P[1-9][0-9]*$")
_TOP_KEYS = {"version", "relations"}
_ENTRY_KEYS = {"pid", "description", "synonyms"}


def _no_duplicate_keys(pairs):
    """`object_pairs_hook` for json.load. Plain json.load keeps the last of two
    equal keys silently, which for a vocabulary would drop a relation without
    a trace."""
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key in vocabulary file: %r" % (key,))
        out[key] = value
    return out


def _check_keys(obj, want, where: str) -> None:
    if not isinstance(obj, dict):
        raise ValueError("%s must be a dict, got %s" % (where, type(obj).__name__))
    missing = sorted(want - set(obj))
    extra = sorted(set(obj) - want, key=str)
    if missing:
        raise ValueError("%s is missing keys %s" % (where, missing))
    if extra:
        raise ValueError("%s has unexpected keys %s" % (where, extra))


def _non_empty_str(value, where: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("%s must be a non-empty string, got %r" % (where, value))


class RelationVocabulary:
    """Canonical relations, each with the synonyms that resolve to it.

    `mapping` has exactly this shape, no more and no fewer keys at each level:

        {"version": "<non-empty str>",
         "relations": {"<canonical>": {"pid": "P19" or None,
                                       "description": "<non-empty str>",
                                       "synonyms": ["<str>", ...]}}}

    A bad mapping raises `ValueError` naming the canonical or synonym at fault,
    so a mistake in the data file is found when it loads, not when a claim is
    keyed.
    """

    def __init__(self, mapping):
        _check_keys(mapping, _TOP_KEYS, "vocabulary")
        _non_empty_str(mapping["version"], "version")
        relations = mapping["relations"]
        if not isinstance(relations, dict) or not relations:
            raise ValueError("relations must be a non-empty dict")

        self._version: str = mapping["version"]
        self._pids: Dict[str, Optional[str]] = {}
        # normalised synonym -> the canonical that owns it
        self._synonyms: Dict[str, str] = {}

        for canonical, entry in relations.items():
            if not isinstance(canonical, str) or not canonical:
                raise ValueError("canonical must be a non-empty string, got %r"
                                 % (canonical,))
            if normalise(canonical) != canonical:
                raise ValueError(
                    "canonical %r is not normal (normalise gives %r)"
                    % (canonical, normalise(canonical)))
            _check_keys(entry, _ENTRY_KEYS, "entry %r" % canonical)
            _non_empty_str(entry["description"], "description of %r" % canonical)
            pid = entry["pid"]
            if pid is not None and not (isinstance(pid, str) and _PID.match(pid)):
                raise ValueError("pid of %r must be None or like 'P19', got %r"
                                 % (canonical, pid))
            syns = entry["synonyms"]
            if not isinstance(syns, list) or not all(isinstance(s, str) for s in syns):
                raise ValueError("synonyms of %r must be a list of strings"
                                 % canonical)
            self._pids[canonical] = pid

        # Synonyms are checked after every canonical is known, so that rule 7
        # holds whatever order the relations were written in.
        for canonical, entry in relations.items():
            for syn in entry["synonyms"]:
                norm = normalise(syn)
                if not norm:
                    raise ValueError("synonym %r of %r normalises to nothing"
                                     % (syn, canonical))
                if norm in self._pids:
                    raise ValueError(
                        "synonym %r of %r normalises to the canonical %r"
                        % (syn, canonical, norm))
                if norm in self._synonyms:
                    raise ValueError(
                        "synonym %r of %r is already a synonym of %r"
                        % (syn, canonical, self._synonyms[norm]))
                self._synonyms[norm] = canonical

    @classmethod
    def load(cls, path) -> "RelationVocabulary":
        with open(path, "r", encoding="utf-8") as f:
            parsed = json.load(f, object_pairs_hook=_no_duplicate_keys)
        return cls(parsed)

    def resolve(self, relation) -> Tuple[Optional[str], bool]:
        s = normalise(relation)
        if not s:
            return (None, False)
        if s in self._pids:
            return (s, True)
        if s in self._synonyms:
            return (self._synonyms[s], True)
        return (s, False)

    @property
    def version(self) -> str:
        return self._version

    @property
    def canonicals(self) -> Tuple[str, ...]:
        return tuple(sorted(self._pids))

    def pid(self, canonical: str) -> Optional[str]:
        return self._pids[canonical]

    def prompt_list(self) -> str:
        return "[" + ", ".join(self.canonicals) + "]"

    def __len__(self) -> int:
        return len(self._pids)

    def __contains__(self, text) -> bool:
        return self.resolve(text)[1]
