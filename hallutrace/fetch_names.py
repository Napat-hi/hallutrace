"""hallutrace/fetch_names.py  |  query strings for the infini-gram fetcher (pure, no network)

Turns a prompt name into the strings worth searching for in a training corpus.
`find` is exact and case-sensitive, so every string must be in its original
casing and punctuation. NEVER send an alias_table key: those are normalised
(claim_key.normalise: lowercased, punctuation stripped, accents folded) and
would not match "M. K. Gandhi". Surface forms come from the prompt itself and
from the Wikidata snapshot; the alias table is only consulted to resolve a
string to QIDs (AliasTable.candidates).

Query-string rules (the extra-forms cnt gate, rule 5, lives in the driver):

    1. name       the prompt name as written; always queried
    2. stripped   without a Wikipedia disambiguator, "Bill Hutchinson (baseball)"
                  -> "Bill Hutchinson"; ambiguous_stripped=True (mixes namesakes)
    3. folded     accent-folded, "Margareta Cederschiöld" -> "Margareta Cederschiold"
    4. alias      snapshot label + aliases, only when the name resolves to exactly
                  one QID; a surface is kept only if it resolves to that same
                  single QID and has at least two words. Names with 0 or 2+ QIDs
                  get no aliases (ambiguous_name=True for 2+).
    6. dedupe     per name, first source wins (name > stripped > folded > alias)

The glued form (drop the leading "▁(" token of "(" + string) is index
independent; the driver tokenizes through the client and calls glued_ids.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass

SOURCES = ("name", "stripped", "folded", "alias")

# "Bill Hutchinson (baseball)", "Brian Palmes (died 1519)". Exactly one trailing
# parenthetical, set off by whitespace, after a name with at least one visible
# character. A parenthetical in the middle of a name is not a disambiguator.
_DISAMBIGUATOR = re.compile(r"^(?P<base>.*\S)\s+\([^()]*\)$")

# SentencePiece marks a word-initial piece with U+2581. "(" tokenizes as one
# piece "▁(" or, in the rarer variant, as "▁" followed by "(".
_LEAD_ONE = "▁("
_LEAD_TWO = ("▁", "(")


@dataclass(frozen=True)
class QueryString:
    prompt_name: str
    string: str
    source: str  # one of SOURCES
    qid: str | None  # the single QID this string resolves to, else None
    ambiguous_name: bool  # the prompt name resolves to 2+ QIDs (namesakes)
    ambiguous_stripped: bool  # source == "stripped": the stripped form mixes namesakes


def _raise_csv_limit() -> None:
    """Prompt text can exceed csv's 128 KiB default field limit. sys.maxsize
    overflows the C long on Windows, so back off until the setter accepts it."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


# Which task_meta fields hold a subject name, per task. historical_events asks about two
# people per prompt; biographies about one.
NAME_KEYS = {
    "historical_events": ("name1", "name2"),
    "biographies": ("name",),
}


def load_names(prompts_path: str, task: str = "historical_events") -> list[str]:
    """Unique subject names of `task` in data/prompts.csv (NAME_KEYS of task_meta), sorted.

    A task not in NAME_KEYS is read through name1/name2."""
    keys = NAME_KEYS.get(task, ("name1", "name2"))
    _raise_csv_limit()
    names: set[str] = set()
    with open(prompts_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("task") != task:
                continue
            meta = json.loads(row["task_meta"])
            for key in keys:
                value = (meta.get(key) or "").strip()
                if value:
                    names.add(value)
    return sorted(names)


def strip_disambiguator(name: str) -> str | None:
    """"Bill Hutchinson (baseball)" -> "Bill Hutchinson"; "Brian Palmes (died 1519)"
    -> "Brian Palmes". None if the name has no trailing parenthetical."""
    m = _DISAMBIGUATOR.match(name.strip())
    return m.group("base") if m else None


def fold_accents(name: str) -> str | None:
    """NFKD, drop combining marks, keep case and punctuation. None if unchanged.

    Only what NFKD decomposes is folded. Letters with no decomposition (o-slash,
    the ae ligature, d-stroke) are left alone rather than transliterated from a
    guessed table: a wrong guess would search for a string nobody wrote."""
    decomposed = unicodedata.normalize("NFKD", name)
    kept = "".join(c for c in decomposed if not unicodedata.combining(c))
    folded = unicodedata.normalize("NFC", kept)
    return None if folded == name else folded


def query_strings(name: str, alias_table, snapshot: dict) -> list[QueryString]:
    """Rules 1 to 4 and 6 for one prompt name.

    `alias_table` is a hallutrace.claim_key.AliasTable; `snapshot` is the parsed
    data/alias_build/wikidata_snapshot.json ({"entities": {qid: {label, aliases, ...}}}).
    Order: name, stripped, folded, aliases in snapshot order (label first).
    """
    name_qids = alias_table.candidates(name)
    ambiguous_name = len(name_qids) >= 2

    out: list[QueryString] = []
    seen: set[str] = set()

    def add(string, source, qid, ambiguous_stripped=False):
        string = (string or "").strip()
        # Dedupe by exact string: find is case-sensitive, so "Gandhi" and
        # "gandhi" are different searches. First source wins (rule 6).
        if not string or string in seen:
            return
        seen.add(string)
        out.append(QueryString(name, string, source, qid, ambiguous_name, ambiguous_stripped))

    def single_qid(surface):
        hits = alias_table.candidates(surface)
        return hits[0] if len(hits) == 1 else None

    add(name, "name", single_qid(name))

    stripped = strip_disambiguator(name)
    if stripped:
        # The bare name mixes every namesake, so it carries no QID.
        add(stripped, "stripped", None, ambiguous_stripped=True)

    folded = fold_accents(name)
    if folded:
        add(folded, "folded", single_qid(folded))
    # A name with both a disambiguator and accents also gets its stripped form
    # folded, so the gain of each rule does not depend on the other. It mixes
    # namesakes for the same reason the stripped form does.
    if stripped:
        folded_stripped = fold_accents(stripped)
        if folded_stripped:
            add(folded_stripped, "folded", None, ambiguous_stripped=True)

    # Aliases only for an unambiguous name. With namesakes we could not tell
    # whose aliases to search for, and every wrong one inflates n_c.
    if len(name_qids) == 1:
        qid = name_qids[0]
        entity = snapshot.get("entities", {}).get(qid)
        if entity:
            for surface in [entity.get("label"), *(entity.get("aliases") or [])]:
                surface = (surface or "").strip()
                if len(surface.split()) < 2:
                    continue  # one-word aliases ("Bapu", "A-7713") match far too much
                # Surfaces come from the snapshot (original casing); the table
                # only vets them: drop any that also denote another entity.
                if alias_table.candidates(surface) != (qid,):
                    continue
                add(surface, "alias", qid)
    return out


def glued_ids(plain_ids: list[int], plain_tokens: list[str]) -> list[int]:
    """Token ids of a string glued to a preceding "(": tokenize "(" + string, then
    drop the leading "▁(" token(s). The result is what `query_ids` must carry to
    find the mentions NOT preceded by a space (line starts, headings, tables).
    Raises ValueError if the tokens do not start with the expected prefix."""
    if len(plain_ids) != len(plain_tokens):
        raise ValueError(f"ids/tokens length mismatch: {len(plain_ids)} vs {len(plain_tokens)}")
    if plain_tokens[:1] == [_LEAD_ONE]:
        rest = list(plain_ids[1:])
    elif tuple(plain_tokens[:2]) == _LEAD_TWO:
        rest = list(plain_ids[2:])
    else:
        raise ValueError(f"tokens do not start with the '(' prefix: {plain_tokens[:3]!r}")
    if not rest:
        raise ValueError("nothing left after dropping the '(' prefix")
    return rest
