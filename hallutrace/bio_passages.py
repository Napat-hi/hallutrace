"""The passages OpenFActScore's validator saw, for the Biographies gold-cut atoms.

Annotators judge an atom against the whole Wikipedia article; the validator judged it
against the 5 passages retrieval returned. Keeping the passages separates retrieval
errors from judgement errors.

FActScore's retrieval cache, `retrieval-enwiki-20230401.json` in each
`factscore_cache/<input stem>/`, maps `"<topic>#<topic> <atom>"` to those passages
(`{"title", "text"}`). The caches are spread over several directories, so all are read.
Annotators never see the result.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def cache_keys(topic: str, atom: str) -> List[str]:
    """FActScore's key is topic + '#' + topic + ' ' + atom; try the atom as given and stripped."""
    out = []
    for a in (atom, atom.strip()):
        k = "%s#%s %s" % (topic, topic, a)
        if k not in out:
            out.append(k)
    return out


def lookup(caches: Iterable[Dict[str, list]], items: List[dict]) -> Dict[str, Optional[list]]:
    """item_id -> the cached passages, or None when no cache holds the atom.

    `items` carry item_id, topic and atom. Caches are read one at a time, so only the
    entries asked for are kept in memory.
    """
    want = {k: it["item_id"] for it in items for k in cache_keys(it["topic"], it["atom"])}
    found: Dict[str, Optional[list]] = {it["item_id"]: None for it in items}
    for cache in caches:
        for k in want.keys() & cache.keys():
            if found[want[k]] is None:
                found[want[k]] = cache[k]
    return found


def clean(passages: Optional[list]) -> Optional[List[dict]]:
    """Drop the <s>/</s> markers the DocDB keeps."""
    if passages is None:
        return None
    return [{"title": p.get("title", ""),
             "text": p.get("text", "").replace("<s>", "").replace("</s>", " ").strip()}
            for p in passages]
