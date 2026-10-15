"""Biographies birth/death year claims and the offline namesake check.

Biographies enters Phase 2 for one kind of claim only: the year a person was born
or died, the one Biographies fact with a countable correct value (Wikidata P569/P570 in
`data/alias_build/wikidata_snapshot.json`). This module lists every such atom in the
verifier's filtered output, with what Wikidata says.

Which atoms count: the atom holds a year from 1000 to 2029 and says
born/birth or opens a lifespan "(1869-" (birth), or says died/death/passed away/killed/
executed/assassinated (death). Atoms are parsed with `bio_filter.parse_atoms`, which keeps
atoms that span a line break (a line-by-line read misses about 90 of them).

Wikidata years come from the person's own QID as the alias table decided it (auto-accepted
title or `data/alias_overrides.json`), passed in as `own`. A name left at `null` has no own
QID, so its claims get no Wikidata verdict.

The namesake check: a hallucinated year is a CONFLATION candidate when it
equals the same kind of year of another person the name can denote (the alias table's
other QIDs for the name) and not the person's own year. Knowing the own QID keeps the
verifier's mistakes out of it (a stated year that is the person's own is a verifier error,
not a conflation). Without corpus evidence this stays a lower bound on Group B and an
upper bound on these few claims, and it is reported as such.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

from hallutrace import bio_filter as bfl

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
BIRTH = re.compile(r"\b(born|birth)\b", re.I)
DEATH = re.compile(r"\b(died|death|passed away|killed|executed|assassinated)\b", re.I)
LIFESPAN = re.compile(r"\(\s*(1[0-9]{3}|20[0-2][0-9])\s*[-–—]")
PROPERTY = {"birth": "p569", "death": "p570"}

COLUMNS = ("generation_id", "model", "sample_index", "unit_id", "name", "kind", "label", "atom",
           "stated_years", "own_qid", "wikidata_years", "wikidata_match",
           "namesake_qids", "namesake_year_qids", "conflation_candidate")


def kind_of(text: str) -> Optional[str]:
    """'birth', 'death' or None for one atom."""
    if not YEAR.search(text):
        return None
    if BIRTH.search(text) or LIFESPAN.search(text):
        return "birth"
    if DEATH.search(text):
        return "death"
    return None


def stated_years(text: str) -> List[int]:
    return sorted({int(y) for y in YEAR.findall(text)})


def entity_years(snapshot: dict, qid: Optional[str], kind: str) -> List[int]:
    ent = (snapshot.get("entities") or {}).get(qid or "", {})
    return sorted(set(ent.get(PROPERTY[kind]) or []))


def claims_in_row(row: dict, meta: dict, own: Optional[str], candidates: Sequence[str],
                  snapshot: dict, namesake_years: Optional[dict] = None) -> List[dict]:
    """The year claims of one output row. `meta` carries generation_id, model,
    sample_index and unit_id; `candidates` are every QID the name can denote.
    `namesake_years` is data/alias_build/namesake_years.json: when given, the namesakes'
    years come from it (the snapshot holds years for few of them), the person's own from
    the snapshot."""
    out = []
    namesakes = [q for q in candidates if q != own]
    ns_source = snapshot if namesake_years is None else namesake_years
    for atom, supported in bfl.parse_atoms(row.get("atoms", "")):
        text = atom.strip()
        kind = kind_of(text)
        if kind is None:
            continue
        years = stated_years(text)
        own_years = entity_years(snapshot, own, kind)
        match = bool(set(years) & set(own_years)) if own_years else None
        hits = [q for q in namesakes
                if set(years) & set(entity_years(ns_source, q, kind))]
        label = "supported" if supported else "hallucinated"
        out.append(dict(meta, name=row["name"], kind=kind, label=label, atom=text,
                        stated_years=" ".join(str(y) for y in years), own_qid=own or "",
                        wikidata_years=" ".join(str(y) for y in own_years),
                        wikidata_match="" if match is None else str(match),
                        namesake_qids=" ".join(namesakes),
                        namesake_year_qids=" ".join(hits),
                        conflation_candidate=str(label == "hallucinated" and bool(hits)
                                                 and match is not True)))
    return out


def summarise(claims: Iterable[dict]) -> dict:
    """Counts by kind and label, by model, Wikidata agreement and conflation."""
    claims = list(claims)
    out = {"claims": len(claims), "by_kind_label": {}, "hallucinated_by_model": {},
           "wikidata": {"no_own_qid_or_years": 0, "label_agrees": 0, "label_disagrees": 0},
           "names": len({c["name"] for c in claims}),
           "responses": len({c["generation_id"] for c in claims}),
           "namesake_names": len({c["name"] for c in claims if c["namesake_qids"]}),
           "namesake_claims": sum(1 for c in claims if c["namesake_qids"]),
           "conflation_candidates": sum(1 for c in claims if c["conflation_candidate"] == "True"),
           "conflation_names": len({c["name"] for c in claims if c["conflation_candidate"] == "True"})}
    for c in claims:
        k = "%s_%s" % (c["kind"], c["label"])
        out["by_kind_label"][k] = out["by_kind_label"].get(k, 0) + 1
        if c["label"] == "hallucinated":
            out["hallucinated_by_model"][c["model"]] = out["hallucinated_by_model"].get(c["model"], 0) + 1
        if c["wikidata_match"] == "":
            out["wikidata"]["no_own_qid_or_years"] += 1
        else:
            agrees = (c["wikidata_match"] == "True") == (c["label"] == "supported")
            out["wikidata"]["label_agrees" if agrees else "label_disagrees"] += 1
    return out
