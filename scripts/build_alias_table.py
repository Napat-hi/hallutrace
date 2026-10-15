"""Build the claim_key alias table from Wikidata/Wikipedia.

Usage (repo root, PYTHONPATH=. so `hallutrace` and `scripts` import):

    python scripts/build_alias_table.py fetch
    python scripts/build_alias_table.py fetch --missing-only
    python scripts/build_alias_table.py build
    python scripts/build_alias_table.py build --out <path>
    python scripts/build_alias_table.py build --allow-unreviewed

`fetch` talks to the network (enwiki + Wikidata APIs) and writes
`data/alias_build/wikidata_snapshot.json`. `build` is offline: it reads the
snapshot and `data/alias_overrides.json` and writes `data/alias_table.json`
(or `--out`) plus a report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from hallutrace.claim_key import is_qid, normalise
from scripts.draw_sample import repair_double_encoding

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "external" / "HALoGEN" / "prompts"
BIO_CSV = PROMPTS_DIR / "prompts_biographies.csv"
HIST_CSV = PROMPTS_DIR / "prompts_historicalevents.csv"

ALIAS_BUILD_DIR = ROOT / "data" / "alias_build"
SNAPSHOT_PATH = ALIAS_BUILD_DIR / "wikidata_snapshot.json"
OVERRIDES_PATH = ROOT / "data" / "alias_overrides.json"
TABLE_PATH = ROOT / "data" / "alias_table.json"
REPORT_PATH = ROOT / "outputs" / "alias_build_report.json"

USER_AGENT = "hallutrace-alias-build/0.1 (https://github.com/Napat-hi/hallutrace)"
ENWIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

BATCH_SIZE = 50
# Pause after every request. Requests are strictly one at a time, as
# Wikimedia's API etiquette asks, with a descriptive User-Agent.
SLEEP_SECONDS = 0.3
RETRY_BACKOFF_SECONDS = 10.0
# How far fetch pages a search that fills its first page of 50. In practice
# wbsearchentities stops offering pages at 100 results, well before this.
MAX_SEARCH_RESULTS = 500
# How far the label search pages. It reaches further than wbsearchentities
# but returns bare QIDs, so each result is fetched and checked.
MAX_LABEL_RESULTS = 1000
MAX_RETRIES = 3

# Wikidata classes that make an item only a name or a Wikimedia page, not a
# thing a claim could be about. The namesake check ignores exact matches of
# these kinds: "Victoria" the given name is no reason to refuse a merge.
NAME_ONLY_P31 = frozenset({
    "Q4167410",   # Wikimedia disambiguation page
    "Q22808320",  # Wikimedia human name disambiguation page
    "Q13406463",  # Wikimedia list article
    "Q4167836",   # Wikimedia category
    "Q17362920",  # Wikimedia duplicated page
    "Q101352",    # family name
    "Q202444",    # given name
    "Q12308941",  # male given name
    "Q11879590",  # female given name
    "Q3409032",   # unisex given name
    "Q66480858",  # affixed family name
    "Q1071027",   # full name
    "Q49614",     # nickname
    "Q61002",     # pseudonym
    "Q21286738",  # Wikimedia permanent duplicate item
    "Q11266439",  # Wikimedia template
    "Q11753321",  # Wikimedia navigational template
})

# Depictions of a person and records about them, titled with their name:
# a portrait called "Golda Meir", a statue called "Nelson Mandela", a journal
# article called "Bertrand Russell". A claim naming the person is not about
# these, so the namesake check ignores them too. Films,
# albums, songs, places, schools and fictional characters still count: a
# claim can be about those.
DEPICTION_P31 = frozenset({
    "Q3305213",   # painting
    "Q860861",    # sculpture
    "Q179700",    # statue
    "Q241045",    # bust
    "Q11060274",  # print
    "Q125191",    # photograph
    "Q93184",     # drawing
    "Q13442814",  # scholarly article
    "Q18918145",  # academic journal article
    "Q13433827",  # encyclopedia article
    "Q191067",    # article
    "Q3331189",   # version, edition or translation
    "Q19389637",  # biographical article
    "Q838948",    # work of art
    "Q4502142",   # visual artwork
    "Q18761202",  # watercolor painting
    "Q22569957",  # screen print
    "Q15123870",  # lithograph print
    "Q18396864",  # aquatint print
    "Q245117",    # relief sculpture
    "Q721747",    # commemorative plaque
    "Q5003624",   # memorial
})

# Everything the namesake check ignores.
IGNORED_P31 = NAME_ONLY_P31 | DEPICTION_P31

_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
_YEAR = re.compile(r"^([+-]\d+)-\d\d-\d\d")


def strip_parenthetical(text: str) -> str:
    """Remove one trailing parenthetical qualifier, for example
    "Daniel Dubois (boxer)" -> "Daniel Dubois". Text with no trailing
    parenthetical is returned unchanged.
    """
    return _TRAILING_PAREN.sub("", text).strip()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _http_get(base_url: str, params: dict) -> dict:
    """One API call, with the required User-Agent.

    Retries up to MAX_RETRIES times, backing off RETRY_BACKOFF_SECONDS, on
    a 429 (the design's requirement) and also on a transient network error
    (read timeout, dropped connection): a run this long, over hundreds of
    requests, hits ordinary network flakiness that has nothing to do with
    Wikimedia rate limiting, and treating only 429 as retryable meant one
    timed-out read could kill a 25-minute run.
    """
    url = base_url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES:
                attempt += 1
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            raise
        except (TimeoutError, ConnectionError, urllib.error.URLError) as e:
            if attempt < MAX_RETRIES:
                attempt += 1
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            raise


def _api_call(base_url: str, params: dict) -> dict:
    """A rate-limited API call: sleeps SLEEP_SECONDS after every request,
    success or failure, so the caller never has to remember to.

    An API error raises instead of returning: read as an empty result, it
    would store "no namesakes" and leave a key with a single QID, the
    unsafe direction.
    """
    try:
        data = _http_get(base_url, params)
    finally:
        time.sleep(SLEEP_SECONDS)
    if not isinstance(data, dict) or "error" in data:
        raise RuntimeError("API error for %s: %r" % (params.get("action"), data.get("error") if isinstance(data, dict) else data))
    return data


# ---------------------------------------------------------------------------
# load_prompt_names
# ---------------------------------------------------------------------------


def _year_or_none(value) -> Optional[int]:
    if pd.isna(value):
        return None
    return int(value)


def load_prompt_names() -> List[dict]:
    """One dict per unique name across both HALoGEN CSVs, repaired spelling.

    Each dict has:
    - name: the repaired name string.
    - source: "bio", "hist" or "both".
    - birth, death: int or None. Hist only. The first (sorted) year pair
      when a name appears with more than one pair across rows.
    - years: the full sorted list of [birth, death] pairs seen for this
      name in the historical-events CSV (empty for bio-only names). Kept
      alongside birth/death because a name can appear with inconsistent
      years across rows and no single pair would be complete.
    """
    bio = pd.read_csv(BIO_CSV).map(repair_double_encoding)
    hist = pd.read_csv(HIST_CSV).map(repair_double_encoding)

    entries: Dict[str, dict] = {}

    for name in bio["name"]:
        name = str(name).strip()
        e = entries.setdefault(name, {"name": name, "sources": set(), "years": set()})
        e["sources"].add("bio")

    for _, row in hist.iterrows():
        for i in (1, 2):
            name = str(row[f"name{i}"]).strip()
            b = _year_or_none(row[f"birth{i}"])
            d = _year_or_none(row[f"death{i}"])
            e = entries.setdefault(name, {"name": name, "sources": set(), "years": set()})
            e["sources"].add("hist")
            e["years"].add((b, d))

    result = []
    for name, e in entries.items():
        sources = e["sources"]
        source = "both" if len(sources) == 2 else next(iter(sources))
        years = sorted(e["years"], key=lambda p: (p[0] is None, p[0], p[1] is None, p[1]))
        birth, death = (years[0] if years else (None, None))
        result.append(
            {
                "name": name,
                "source": source,
                "birth": birth,
                "death": death,
                "years": [[b, d] for b, d in years],
            }
        )
    result.sort(key=lambda x: x["name"])
    return result


# ---------------------------------------------------------------------------
# fetch: title resolution
# ---------------------------------------------------------------------------


def _chunked(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def resolve_titles(titles: List[str]) -> Dict[str, dict]:
    """Resolve a batch (<=50) of enwiki titles via redirects to pageprops.

    Returns {original_title: {title, normalized_to, redirected_to, qid,
    disambiguation, missing}}.
    """
    results: Dict[str, dict] = {}
    for batch in _chunked(titles, BATCH_SIZE):
        data = _api_call(
            ENWIKI_API,
            {
                "action": "query",
                "titles": "|".join(batch),
                "redirects": 1,
                "prop": "pageprops",
                "ppprop": "wikibase_item|disambiguation",
                "format": "json",
                "formatversion": "2",
            },
        )
        query = data.get("query", {})
        normalized = {n["from"]: n["to"] for n in query.get("normalized", [])}
        redirects = {r["from"]: r["to"] for r in query.get("redirects", [])}
        pages = query.get("pages", [])
        page_by_title = {p["title"]: p for p in pages}

        for orig in batch:
            norm = normalized.get(orig, orig)
            node = norm
            seen = set()
            while node in redirects and node not in seen:
                seen.add(node)
                node = redirects[node]
            final_title = node
            redirected_to = final_title if final_title != norm else None

            page = page_by_title.get(final_title)
            missing = bool(page.get("missing")) if page else True
            qid = None
            disambiguation = False
            if page and not missing and "pageprops" in page:
                pp = page["pageprops"]
                qid = pp.get("wikibase_item")
                disambiguation = "disambiguation" in pp

            results[orig] = {
                "title": orig,
                "normalized_to": norm if norm != orig else None,
                "redirected_to": redirected_to,
                "qid": qid,
                "disambiguation": disambiguation,
                "missing": missing,
            }
    return results


# ---------------------------------------------------------------------------
# fetch: entity data (labels, aliases, enwiki title, P31/P569/P570)
# ---------------------------------------------------------------------------


def _parse_wd_year(time_str: Optional[str]) -> Optional[int]:
    if not time_str:
        return None
    m = _YEAR.match(time_str)
    if not m:
        return None
    return int(m.group(1))


def fetch_entities(qids: List[str]) -> Dict[str, dict]:
    """Batched wbgetentities for label, aliases, enwiki title, P31, P569, P570."""
    results: Dict[str, dict] = {}
    qids = sorted(set(q for q in qids if is_qid(q)))
    for batch in _chunked(qids, BATCH_SIZE):
        data = _api_call(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels|aliases|claims|sitelinks",
                "languages": "en",
                "sitefilter": "enwiki",
                "format": "json",
                "formatversion": "2",
            },
        )
        entities = data.get("entities", {})
        for qid, ent in entities.items():
            if ent.get("missing"):
                continue
            label = ent.get("labels", {}).get("en", {}).get("value")
            aliases = [a["value"] for a in ent.get("aliases", {}).get("en", [])]
            title = ent.get("sitelinks", {}).get("enwiki", {}).get("title")

            claims = ent.get("claims", {})
            p31 = []
            for c in claims.get("P31", []):
                dv = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                if isinstance(dv, dict) and dv.get("id"):
                    p31.append(dv["id"])
            p569 = []
            for c in claims.get("P569", []):
                dv = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                if isinstance(dv, dict):
                    y = _parse_wd_year(dv.get("time"))
                    if y is not None:
                        p569.append(y)
            p570 = []
            for c in claims.get("P570", []):
                dv = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                if isinstance(dv, dict):
                    y = _parse_wd_year(dv.get("time"))
                    if y is not None:
                        p570.append(y)

            results[qid] = {
                "label": label,
                "aliases": sorted(set(aliases)),
                "title": title,
                "p31": sorted(set(p31)),
                "p569": sorted(set(p569)),
                "p570": sorted(set(p570)),
            }
    return results


# ---------------------------------------------------------------------------
# fetch: namesake search
# ---------------------------------------------------------------------------


def search_namesakes(display: str, offset: int = 0) -> Tuple[List[dict], Optional[int]]:
    """One page (50 results) of wbsearchentities for a key's display string.

    Returns (raw_results, next_offset). next_offset is None when the search
    has no more results. Filtering to exact-normalised matches happens in
    the caller.
    """
    params = {
        "action": "wbsearchentities",
        "search": display,
        "language": "en",
        "uselang": "en",
        "type": "item",
        "limit": 50,
        "format": "json",
        "formatversion": "2",
    }
    if offset:
        params["continue"] = offset
    data = _api_call(WIKIDATA_API, params)
    if not data.get("success"):
        raise RuntimeError("wbsearchentities did not report success for %r: %r" % (display, data))
    return data.get("search", []), data.get("search-continue")


def label_search(display: str) -> Tuple[List[str], int]:
    """Wikidata full-text search restricted to labels and aliases
    (`inlabel:"..."`), paged up to MAX_LABEL_RESULTS. Returns (qids, total
    hits). Run on every key: wbsearchentities reports one match text per
    item, so an item whose exact match is only an alias can be dropped, and
    it is punctuation-sensitive ("Ulysses S Grant" misses "Ulysses S.
    Grant"). This search matches any label or alias, and every result is
    then checked against all its names by exact_label_items."""
    qids: List[str] = []
    offset: Optional[int] = 0
    total = 0
    while offset is not None and offset < MAX_LABEL_RESULTS:
        data = _api_call(
            WIKIDATA_API,
            {
                "action": "query",
                "list": "search",
                "srsearch": 'inlabel:"%s"' % display.replace('"', ""),
                "srnamespace": 0,
                "srlimit": 500,
                "sroffset": offset,
                "srinfo": "totalhits",
                "srprop": "",
                "format": "json",
                "formatversion": "2",
            },
        )
        q = data.get("query", {})
        total = q.get("searchinfo", {}).get("totalhits", total)
        qids.extend(r["title"] for r in q.get("search", []) if is_qid(r.get("title")))
        offset = data.get("continue", {}).get("sroffset")
    return qids, total


def fetch_names(qids: List[str]) -> Dict[str, dict]:
    """Batched wbgetentities for every label and alias, in any language, and
    P31. Returns {qid: {"forms": [...], "p31": [...]}}. Label searches for
    many keys share these batches, so one request checks results for
    several keys."""
    results: Dict[str, dict] = {}
    for batch in _chunked(sorted(set(q for q in qids if is_qid(q))), BATCH_SIZE):
        data = _api_call(
            WIKIDATA_API,
            {"action": "wbgetentities", "ids": "|".join(batch), "props": "labels|aliases|claims",
             "format": "json", "formatversion": "2"},
        )
        for qid, ent in data.get("entities", {}).items():
            if ent.get("missing"):
                continue
            forms = [v["value"] for v in ent.get("labels", {}).values()]
            forms += [a["value"] for al in ent.get("aliases", {}).values() for a in al]
            p31 = set()
            for c in ent.get("claims", {}).get("P31", []):
                dv = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                if isinstance(dv, dict) and dv.get("id"):
                    p31.add(dv["id"])
            results[qid] = {"forms": forms, "p31": sorted(p31)}
    return results


def exact_label_items(qids: List[str], key: str, names: Dict[str, dict]) -> List[dict]:
    """The items among `qids` with a label or alias, in any language, that
    normalises to `key`, with their P31. `names` is from fetch_names."""
    hits = []
    for qid in sorted(set(qids)):
        entry = names.get(qid)
        if entry is None:
            continue
        match = next((f for f in entry["forms"] if normalise(f) == key), None)
        if match is not None:
            hits.append({"qid": qid, "matched_text": match, "p31": entry["p31"]})
    return hits


def _exact_hits(raw: List[dict], key: str) -> List[dict]:
    hits = []
    for r in raw:
        label = r.get("label")
        match_text = r.get("match", {}).get("text")
        if normalise(label) == key or normalise(match_text) == key:
            hits.append({"qid": r["id"], "matched_text": match_text or label, "p31": []})
    return hits


def fetch_p31(qids: List[str]) -> Dict[str, List[str]]:
    """Batched wbgetentities for P31 only. Used for namesake hits, which need
    their kind (human, place, work, name) and nothing else, so the snapshot
    does not carry labels and aliases for thousands of unrelated items."""
    results: Dict[str, List[str]] = {}
    qids = sorted(set(q for q in qids if is_qid(q)))
    for batch in _chunked(qids, BATCH_SIZE):
        data = _api_call(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "claims",
                "format": "json",
                "formatversion": "2",
            },
        )
        for qid, ent in data.get("entities", {}).items():
            if ent.get("missing"):
                continue
            p31 = set()
            for c in ent.get("claims", {}).get("P31", []):
                dv = c.get("mainsnak", {}).get("datavalue", {}).get("value")
                if isinstance(dv, dict) and dv.get("id"):
                    p31.add(dv["id"])
            results[qid] = sorted(p31)
    return results


# ---------------------------------------------------------------------------
# fetch command
# ---------------------------------------------------------------------------


def _load_json(path: Path, default):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict):
    """Write to a temporary file, then replace, so a run stopped mid-write
    never leaves a truncated snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, sort_keys=True, indent=1, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def cmd_fetch(missing_only: bool) -> None:
    names = load_prompt_names()
    overrides = _load_json(OVERRIDES_PATH, {})
    override_qids = {
        name: entry["qid"] for name, entry in overrides.items() if is_qid(entry.get("qid"))
    }

    if missing_only:
        if not SNAPSHOT_PATH.exists():
            print("fetch --missing-only: no existing snapshot at %s" % SNAPSHOT_PATH)
            sys.exit(1)
        snapshot = _load_json(SNAPSHOT_PATH, {})
        snapshot.setdefault("titles", {})
        snapshot.setdefault("entities", {})
        snapshot.setdefault("namesakes", {})
        snapshot.setdefault("namesakes_more", {})
        snapshot.setdefault("namesakes_label", {})
        snapshot.setdefault("fetched_at_extra", [])
    else:
        snapshot = {
            "fetched_at": None,
            "fetched_at_extra": [],
            "titles": {},
            "entities": {},
            "namesakes": {},
            "namesakes_more": {},
            "namesakes_label": {},
        }

    # Checkpoint after every phase. A run this long (well over a thousand
    # rate-limited requests) can hit an ordinary network hiccup near the
    # end; without this, that one failure would have thrown away everything
    # fetched before it.
    def checkpoint():
        _write_json(SNAPSHOT_PATH, snapshot)

    all_names = [n["name"] for n in names]
    to_resolve = [n for n in all_names if n not in snapshot["titles"]]
    print("resolving %d titles (%d already in snapshot)" % (len(to_resolve), len(all_names) - len(to_resolve)))
    if to_resolve:
        for batch in _chunked(to_resolve, BATCH_SIZE):
            snapshot["titles"].update(resolve_titles(batch))
            checkpoint()

    needed_qids = set(override_qids.values())
    for entry in snapshot["titles"].values():
        if is_qid(entry.get("qid")):
            needed_qids.add(entry["qid"])
    to_fetch_qids = sorted(q for q in needed_qids if q not in snapshot["entities"])
    print("fetching %d entities (%d already in snapshot)" % (len(to_fetch_qids), len(needed_qids) - len(to_fetch_qids)))
    if to_fetch_qids:
        for batch in _chunked(to_fetch_qids, BATCH_SIZE):
            snapshot["entities"].update(fetch_entities(batch))
            checkpoint()

    # Every key the table would hold is searched, not a subset: the
    # keys are read off the table as it would be built now. Which keys exist
    # does not depend on search results, so this can run before the searches.
    _, surfaces, displays, _ = table_surfaces(snapshot, overrides, names)
    to_search = {k: sorted(displays[k])[0] for k in surfaces if k not in snapshot["namesakes"]}
    print("searching %d keys (%d already in snapshot)" % (len(to_search), len(surfaces) - len(to_search)))
    new_hit_qids: set = set()
    for i, (key, display) in enumerate(sorted(to_search.items()), 1):
        raw, next_offset = search_namesakes(display)
        hits = _exact_hits(raw, key)
        new_hit_qids.update(h["qid"] for h in hits)
        # Hits are stored with an empty p31 first; the P31 pass below fills
        # them in. A crash during that pass loses only that pass.
        snapshot["namesakes"][key] = {"searched": display, "hits": hits, "truncated": next_offset is not None}
        if i % 50 == 0:
            print("  searched %d/%d" % (i, len(to_search)), flush=True)
            checkpoint()
    checkpoint()

    # A search that filled its first page may have more exact matches past
    # it. Page on, up to MAX_SEARCH_RESULTS, and store the extra pages in
    # their own section, so the first page's entry is never edited.
    to_page = sorted(k for k in surfaces
                     if snapshot["namesakes"].get(k, {}).get("truncated") and k not in snapshot["namesakes_more"])
    print("paging %d searches past their first page" % len(to_page))
    paged_keys = []
    for key in to_page:
        entry = snapshot["namesakes"][key]
        hits, offset, pages, results, last = [], 50, 0, 50, []
        while offset is not None and offset < MAX_SEARCH_RESULTS:
            last, offset = search_namesakes(entry["searched"], offset)
            pages += 1
            results += len(last)
            hits.extend(_exact_hits(last, key))
        new_hit_qids.update(h["qid"] for h in hits)
        # The search API stops offering more pages at some point (100
        # results, in practice) even when its last page was full, so a full
        # last page also counts as truncated.
        snapshot["namesakes_more"][key] = {
            "pages": pages,
            "results": results,
            "hits": hits,
            "truncated": offset is not None or len(last) >= 50,
        }
        paged_keys.append(key)
    checkpoint()

    # P31 for the hits of keys searched in this run only, so an entry
    # searched in an earlier run is never edited. Kinds already known from
    # `entities` are reused; the rest are fetched P31-only.
    known = {q: e.get("p31", []) for q, e in snapshot["entities"].items()}
    p31 = fetch_p31(sorted(q for q in new_hit_qids if q not in known))
    p31.update(known)
    for key in to_search:
        for h in snapshot["namesakes"][key]["hits"]:
            h["p31"] = p31.get(h["qid"], [])
    for key in paged_keys:
        for h in snapshot["namesakes_more"][key]["hits"]:
            h["p31"] = p31.get(h["qid"], [])
    checkpoint()

    # Every key also gets the label search, in a third section (see
    # label_search for why the first search alone misses namesakes).
    to_label = sorted(k for k in surfaces if k not in snapshot["namesakes_label"])
    print("label-searching %d keys (%d already in snapshot)" % (len(to_label), len(surfaces) - len(to_label)))
    # Keys go in groups: search each key in the group, fetch the names of all
    # their results together (skipping any fetched earlier in this run), then
    # store the group's entries and checkpoint. A key's entry is written only
    # once its results are all checked.
    names_cache: Dict[str, dict] = {}
    done = 0
    for group in _chunked(to_label, 50):
        searched = {key: label_search(snapshot["namesakes"][key]["searched"]) for key in group}
        needed = {q for qids, _ in searched.values() for q in qids} - set(names_cache)
        names_cache.update(fetch_names(sorted(needed)))
        for key, (qids, total) in searched.items():
            snapshot["namesakes_label"][key] = {
                "total": total,
                "checked": len(qids),
                "hits": exact_label_items(qids, key, names_cache),
                "truncated": len(qids) < total,
            }
        done += len(group)
        print("  label-searched %d/%d" % (done, len(to_label)), flush=True)
        checkpoint()

    today = datetime.now(timezone.utc).date().isoformat()
    if missing_only:
        snapshot["fetched_at_extra"].append(today)
    else:
        snapshot["fetched_at"] = today

    _write_json(SNAPSHOT_PATH, snapshot)
    print("wrote %s" % SNAPSHOT_PATH)


# ---------------------------------------------------------------------------
# build_mapping (pure)
# ---------------------------------------------------------------------------


def _auto_accept(name_entry: dict, title_entry: Optional[dict], entities: Dict[str, dict]) -> Tuple[bool, Optional[str], Optional[str]]:
    """Returns (accepted, qid, failure_reason). failure_reason is one of
    "missing", "non_human", "disambiguation", "year_mismatch" or None.
    """
    if title_entry is None:
        return False, None, "missing"
    if title_entry.get("missing"):
        return False, None, "missing"
    if title_entry.get("redirected_to"):
        return False, None, "redirect"
    qid = title_entry.get("qid")
    if not is_qid(qid):
        return False, None, "missing"
    if title_entry.get("disambiguation"):
        return False, qid, "disambiguation"
    ent = entities.get(qid)
    if ent is None or "Q5" not in ent.get("p31", []):
        return False, qid, "non_human"

    if name_entry["source"] in ("hist", "both") and name_entry.get("years"):
        p569 = set(ent.get("p569", []))
        p570 = set(ent.get("p570", []))
        birth_ok = True
        death_ok = True
        any_birth = any(b is not None for b, d in name_entry["years"])
        any_death = any(d is not None for b, d in name_entry["years"])
        if any_birth:
            birth_ok = any(b is not None and b in p569 for b, d in name_entry["years"])
        if any_death:
            death_ok = any(d is not None and d in p570 for b, d in name_entry["years"])
        if not (birth_ok and death_ok):
            return False, qid, "year_mismatch"

    return True, qid, None


def _decide(snapshot: dict, overrides: dict, names: List[dict]) -> Tuple[Dict[str, str], dict]:
    """Auto-accept or override, for every prompt name.

    Returns (decided, outcome). decided is {name: qid} for every name that
    ends up with a QID. outcome holds the lists the report needs:
    auto_accepted, overridden_qid, overridden_null, needs_review (names
    with no override and no auto-accept) and validation_failures (exact
    titles that failed a check).
    """
    titles = snapshot.get("titles", {})
    entities = snapshot.get("entities", {})

    decided: Dict[str, str] = {}
    outcome = {
        "auto_accepted": [],
        "overridden_qid": [],
        "overridden_null": [],
        "needs_review": [],
        "validation_failures": [],
    }

    for n in names:
        name = n["name"]
        accepted, exact_qid, reason = _auto_accept(n, titles.get(name), entities)

        if reason in ("non_human", "disambiguation", "year_mismatch") and exact_qid:
            ent = entities.get(exact_qid, {})
            wd_years = "%s to %s" % (
                "/".join(str(y) for y in ent.get("p569", [])) or "none",
                "/".join(str(y) for y in ent.get("p570", [])) or "none",
            )
            csv_years = "/".join(
                "%s to %s" % (b if b is not None else "none", d if d is not None else "none")
                for b, d in n.get("years", [])
            ) or "n/a"
            outcome["validation_failures"].append(
                {
                    "name": name,
                    "qid": exact_qid,
                    "failed": reason,
                    "wikidata_years": wd_years,
                    "csv_years": csv_years,
                }
            )

        override = overrides.get(name)
        if override is not None:
            if is_qid(override.get("qid")):
                decided[name] = override["qid"]
                outcome["overridden_qid"].append(name)
            else:
                outcome["overridden_null"].append(name)
            continue

        if accepted:
            decided[name] = exact_qid
            outcome["auto_accepted"].append(name)
        else:
            outcome["needs_review"].append(name)

    return decided, outcome


def table_surfaces(snapshot: dict, overrides: dict, names: List[dict]):
    """The surface strings each decided entity contributes, before the
    namesake check.

    Returns (decided, surfaces, displays, single_token_dropped). surfaces is
    {key: set of QIDs}; displays is {key: set of the raw strings the key came
    from}, which `fetch` uses as the text to search.

    Sources, per entity: the prompt name, the prompt name without a trailing
    parenthetical, the Wikidata en label, and en aliases of two or more
    tokens. A one-token alias is kept only when it is already one of the
    others. The enwiki title is not a source: where it differed from
    all of these, it named an event or another page rather than the person,
    for example "1927 Charlestown shootings" for Stephanus Swart.
    """
    entities = snapshot.get("entities", {})
    decided, _ = _decide(snapshot, overrides, names)

    surfaces: Dict[str, set] = defaultdict(set)
    displays: Dict[str, set] = defaultdict(set)
    single_token_dropped = []

    def add(form: str, qid: str) -> str:
        key = normalise(form)
        if key:
            surfaces[key].add(qid)
            displays[key].add(form.strip())
        return key

    for name, qid in decided.items():
        ent = entities.get(qid, {})
        own = set()
        for form in (name, strip_parenthetical(name), ent.get("label")):
            if form:
                own.add(add(form, qid))
        for alias in ent.get("aliases", []):
            key = normalise(alias)
            if not key:
                continue
            if len(key.split()) >= 2 or key in own:
                add(alias, qid)
            else:
                single_token_dropped.append({"name": name, "qid": qid, "alias": alias})

    return decided, surfaces, displays, single_token_dropped


def build_mapping(snapshot: dict, overrides: dict, names: List[dict]) -> Tuple[dict, dict]:
    """Pure function: snapshot + overrides + prompt names -> (mapping, report).

    mapping: {normalised_surface_string: sorted list of QIDs}.
    report: counts, validation failures, the namesake check, and
    `needs_review` (names with no override and no auto-accept), which the
    CLI uses to decide the exit code.

    The namesake check runs on every key. Each key was searched on
    Wikidata; every other item whose label or alias normalises to the key
    counts, human or not, unless it is only a name or a Wikimedia page
    (NAME_ONLY_P31) or a depiction of or record about the person
    (DEPICTION_P31). Those items join the key's QID list, so resolve_entity
    returns BY_AMBIGUOUS and declines the merge. Items with no P31 count too:
    an unknown kind is not evidence that the string means only the prompt
    entity. A key with no search in the snapshot is listed in
    `namesake_check.unsearched`, and cmd_build refuses to write the table.
    """
    titles = snapshot.get("titles", {})
    namesakes = snapshot.get("namesakes", {})
    namesakes_more = snapshot.get("namesakes_more", {})
    namesakes_label = snapshot.get("namesakes_label", {})

    decided, outcome = _decide(snapshot, overrides, names)
    _, surfaces, _, single_token_dropped = table_surfaces(snapshot, overrides, names)

    ignore_by_qid: Dict[str, set] = defaultdict(set)
    for name, qid in decided.items():
        ignore_by_qid[qid] |= set(overrides.get(name, {}).get("ignore_namesakes", []) or [])

    mapping: Dict[str, set] = {k: set(v) for k, v in surfaces.items()}
    unsearched = []
    gained = []
    total_added = 0
    others_by_kind = {"human": 0, "other": 0, "no_p31": 0}
    truncated_zero = []
    ignore_applied = []

    for key in sorted(mapping):
        entry = namesakes.get(key)
        more = namesakes_more.get(key)
        label = namesakes_label.get(key)
        if entry is None or label is None or (entry.get("truncated") and more is None):
            # Every key needs its first search and its label search, and a
            # first search that filled up must have been paged on; otherwise
            # the key's other items are not known.
            unsearched.append(key)
            continue
        owners = set(mapping[key])
        ignore = set()
        for q in owners:
            ignore |= ignore_by_qid[q]
        others = set()
        for h in entry.get("hits", []) + (more or {}).get("hits", []) + (label or {}).get("hits", []):
            q = h["qid"]
            p31 = set(h.get("p31") or [])
            if q in owners or p31 & IGNORED_P31:
                continue
            if q in ignore:
                ignore_applied.append({"key": key, "ignored_qid": q})
                continue
            if q not in others:
                others.add(q)
                others_by_kind["human" if "Q5" in p31 else ("other" if p31 else "no_p31")] += 1
        if others:
            mapping[key] |= others
            total_added += len(others)
            gained.append({"key": key, "owners": sorted(owners), "added": len(others),
                           "sample": sorted(others)[:3]})
        elif (label or more or entry).get("truncated") and len(owners) == 1:
            truncated_zero.append(key)

    final_mapping = {}
    for key, qids in mapping.items():
        qids = sorted(q for q in qids if is_qid(q))
        if qids:
            final_mapping[key] = qids

    multi_qid_keys = sorted(k for k, v in final_mapping.items() if len(v) > 1)
    own_name_ambiguous = sorted(
        name for name in decided if len(final_mapping.get(normalise(name), [])) > 1
    )

    exact_title_total = sum(
        1
        for n in names
        if titles.get(n["name"])
        and not titles[n["name"]].get("missing")
        and not titles[n["name"]].get("redirected_to")
        and is_qid(titles[n["name"]].get("qid"))
    )
    failures = outcome["validation_failures"]

    report = {
        "counts": {
            "names_total": len(names),
            "names_bio_only": sum(1 for n in names if n["source"] == "bio"),
            "names_hist_only": sum(1 for n in names if n["source"] == "hist"),
            "names_both": sum(1 for n in names if n["source"] == "both"),
            "auto_accepted": len(outcome["auto_accepted"]),
            "overridden_with_qid": len(outcome["overridden_qid"]),
            "overridden_null": len(outcome["overridden_null"]),
            "surface_strings": len(final_mapping),
            "keys_with_multiple_qids": len(multi_qid_keys),
            "single_token_aliases_dropped": len(single_token_dropped),
        },
        "keys_with_multiple_qids": multi_qid_keys,
        "single_token_aliases_dropped": single_token_dropped,
        "prompt_names_ambiguous_under_own_name": own_name_ambiguous,
        "validation": {
            "exact_title_matches_total": exact_title_total,
            "failures_total": len(failures),
            "non_human": [f for f in failures if f["failed"] == "non_human"],
            "year_mismatch": [f for f in failures if f["failed"] == "year_mismatch"],
            "disambiguation": [f for f in failures if f["failed"] == "disambiguation"],
        },
        "namesake_check": {
            "keys_checked": len(mapping) - len(unsearched),
            "keys_gained_others": len(gained),
            "total_qids_added": total_added,
            "others_by_kind": others_by_kind,
            "keys_still_single_qid": sum(1 for v in final_mapping.values() if len(v) == 1),
            "truncated_zero_others": truncated_zero,
            "ignore_namesakes_applied": ignore_applied,
            "unsearched": unsearched,
            "gained": gained,
        },
        "needs_review": sorted(outcome["needs_review"]),
    }
    return final_mapping, report


# ---------------------------------------------------------------------------
# build command
# ---------------------------------------------------------------------------


def cmd_build(out_path: Path, allow_unreviewed: bool) -> None:
    if not SNAPSHOT_PATH.exists():
        print("build: no snapshot at %s, run `fetch` first" % SNAPSHOT_PATH)
        sys.exit(1)
    snapshot = _load_json(SNAPSHOT_PATH, {})
    overrides = _load_json(OVERRIDES_PATH, {})
    names = load_prompt_names()

    mapping, report = build_mapping(snapshot, overrides, names)
    needs_review = report["needs_review"]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, sort_keys=True, indent=1, ensure_ascii=False)
        f.write("\n")

    print(json.dumps(report["counts"], sort_keys=True, indent=1, ensure_ascii=False))
    print("validation failures: %d" % report["validation"]["failures_total"])
    nc = report["namesake_check"]
    print("namesake check: %d keys checked, %d gained other items (%d qids: %s), %d still single" % (
        nc["keys_checked"], nc["keys_gained_others"], nc["total_qids_added"],
        ", ".join("%s %d" % kv for kv in sorted(nc["others_by_kind"].items())), nc["keys_still_single_qid"],
    ))
    print("prompt names ambiguous under their own name: %d" % len(report["prompt_names_ambiguous_under_own_name"]))
    print("wrote %s" % REPORT_PATH)

    if nc["unsearched"]:
        print("\n%d keys have no search in the snapshot; run `fetch --missing-only`:" % len(nc["unsearched"]))
        for key in nc["unsearched"][:50]:
            print("  " + key)
        if len(nc["unsearched"]) > 50:
            print("  ... and %d more" % (len(nc["unsearched"]) - 50))
        sys.exit(1)

    if needs_review and not allow_unreviewed:
        print("\n%d names need a decision in %s:" % (len(needs_review), OVERRIDES_PATH))
        for name in needs_review:
            print("  " + name)
        sys.exit(1)

    if needs_review:
        print("\n--allow-unreviewed: writing table, excluding %d unreviewed names" % len(needs_review))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(mapping, f, sort_keys=True, indent=1, ensure_ascii=False)
        f.write("\n")
    print("wrote %s" % out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--missing-only", action="store_true")

    p_build = sub.add_parser("build")
    p_build.add_argument("--out", type=Path, default=TABLE_PATH)
    p_build.add_argument("--allow-unreviewed", action="store_true")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(missing_only=args.missing_only)
    elif args.command == "build":
        cmd_build(out_path=args.out, allow_unreviewed=args.allow_unreviewed)


if __name__ == "__main__":
    main()
