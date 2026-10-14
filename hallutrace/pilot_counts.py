"""hallutrace/pilot_counts.py  |  count correct birth/death years near a name (pure, no SQLite, no I/O)

Rows in, rows out. The CLI (scripts/pilot_counts.py) reads the window cache
read-only and hands this module plain dicts named like the cache columns:

  entities  prompt_name, query_string, source, qid, ambiguous_name, ambiguous_stripped, active
  finds     index_name, query_string, form, cnt, segment_by_shard_json
  plan      index_name, query_string, form, s, rank, selected
  windows   index_name, query_string, form, s, rank, doc_ix, doc_len, disp_len,
            needle_offset, spans_json, metadata_json
  years     {prompt_name: {"birth": [int, ...], "death": [int, ...]}}: every distinct
            year the name has across prompts (more than one means year_conflict)

n_c is the number of distinct documents where the correct fact appears in a window
around a mention of the person. Per (prompt_name, index, fact, year, half_width,
matcher, strings) grid cell:

- pool per (index, string) = summed segment lengths of the plain and glued finds
  (as scripts/fetch_windows.py pool_positions does); no finds row = not_fetched (not
  a zero); a find with pool 0 is a true zero.
- `selected` is per string (all if its pool <= 200, else sample); a cell is `all`
  only if every string is.
- n_docs is exact only when selected = all. A sampled pool with no match is UNKNOWN,
  never n_c = 0, and stays out of tau. est_mentions = sum over strings of matched
  (all) or matched / fetched x pool (sample); never one ratio for the whole name.
- n_c = n_docs (unit docs) if selected = all and not incomplete, else est_mentions
  rounded to 1 decimal (unit est_mentions).
- status: zero (every string all, complete, nothing matched), unknown (nothing
  matched and some string sampled or incomplete), else counted.
- Documents are keyed (index, doc_ix), or (index, s, doc_ix) if doc_ix is per shard
  (see DOC_KEY_WITH_SHARD).
- n_docs_dedup counts document GROUPS: two matched documents are in one group when any
  of their matched windows share a dedup key (transitively), so it is never above n_docs.

Half-widths are characters each side of the needle. Matchers: `bare` (any occurrence of
the year), `cue` (a birth/death cue, fact_match.cue_match) and `near_or_cue`, the
counting rule (the year within NEAR characters of the name, or a cue). The rule's count
is the `near_or_cue` / `full` / `no_stripped` cell; the other widths of that matcher are
grid filler. The pilot's precision table (score_labels) scores `bare` and `cue` only.

Interpretation choices (each is also marked where it is applied):
- A string with pool 0 (a find, no plan, no windows) is a true zero: selected all,
  planned 0, fetched 0, matched 0. A pool above 0 with no plan rows is incomplete.
- A cell with no strings (a name that only has `stripped` strings, under no_stripped)
  gets status unknown, incomplete 1, and an empty n_c: nothing was counted.
- `overlap` looks at the strings of the cell, so no_stripped cells drop it.
- Windows rows that are not in plan are ignored and counted by `diagnostics`.
- A window is a candidate for a fact once, at its nearest year occurrence; when a
  name has two years for one fact the nearer of the two wins (packet only).
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from urllib.parse import urlsplit

from . import fact_match

HALF_WIDTHS = (100, 200, 400, 800, 1600, None)  # None = full stored window
WIDTH_LABELS = ("100", "200", "400", "800", "1600", "full")
MATCHERS = ("bare", "cue", "near_or_cue")
NEAR = 100  # characters from the name to the year's near end (pilot_rule_check.NEAR)
SCORED_MATCHERS = ("bare", "cue")  # the ones the hand-check packet's strata can score
STRING_SETS = ("all", "no_stripped")
BANDS = ((0, 100), (101, 200), (201, 400), (401, 800), (801, 1600), (1601, None))
BAND_LABELS = ("0-100", "101-200", "201-400", "401-800", "801-1600", ">1600")
INDEX_LABELS = {
    "v4_olmo-mix-1124_llama": "OLMo 2",
    "v4_dclm-baseline_llama": "DCLM",
    "v4_dolma-v1_7_llama": "Dolma",
    "v4_rpj_llama_s4": "RedPajama",
    "v4_piletrain_llama": "Pile",
}
CAP_FULL = 200  # a string is `all` if its pool is at most this many windows
PER_STRATUM = 5  # packet target rows per (band, matched_cue) stratum
MIN_LABELLED = 40  # `score` refuses to run with fewer labelled rows
LABELS = ("correct", "other_fact", "other_person", "unclear")

# Checked on the real cache 30 Sep: no doc_ix appears under two shards, and (index,
# s, doc_ix) never has two doc_len, so (index, doc_ix) identifies a document.
DOC_KEY_WITH_SHARD = False

CSV_COLUMNS: list[str] = [
    "prompt_name", "index", "fact", "year", "year_conflict", "half_width", "matcher",
    "strings",
    "status", "n_c", "n_c_unit", "selected", "incomplete",
    "n_strings", "n_inactive_strings", "not_fetched_strings", "pool",
    "windows_planned", "windows_fetched", "coverage",
    "mentions_matched", "est_mentions", "n_docs", "n_docs_dedup", "n_domains",
    "url_share", "matched_with_url",
    "matched_plain", "matched_glued",
    "matched_name", "matched_stripped", "matched_folded", "matched_alias",
    "ambiguous_name", "ambiguous_stripped", "shared_string", "overlap", "short_year",
    "pool_mismatch", "glued_missing",
    "per_string_json",  # [{q, src, pool, sel, planned, fetched, matched}, ...]: names and numbers only
]
SOURCES = ("name", "stripped", "folded", "alias")

PACKET_COLUMNS = [
    "packet_id", "index", "prompt_name", "query_string", "query_source", "form", "s",
    "rank", "fact", "year", "distance_chars", "band", "matched_bare", "matched_cue",
    "snippet", "label",
]
STRATA_COLUMNS = ["band", "matched_cue", "query_source", "population", "drawn"]

# Soft quotas for the hand-check packet: name -> minimum.
QUOTAS = {"stripped_rows": 10, "indexes": 3, "facts": 2, "names": 8}


class LabelError(ValueError):
    """A label outside LABELS; the message lists the packet_ids."""


class TooFewLabels(ValueError):
    """Fewer than MIN_LABELLED labelled rows."""


# ---------------------------------------------------------------------------
# Reading the cache rows
# ---------------------------------------------------------------------------


def _active(row) -> bool:
    return str(row.get("active", 1)).strip().lower() not in ("0", "false", "none", "")


def _truthy(v) -> bool:
    return str(v).strip().lower() not in ("0", "false", "none", "", "nan")


def _tables(finds, plan, windows):
    """(finds_by, plan_by, win_by, extra): everything keyed by (index, query_string).
    win_by keeps only windows whose key is in plan; `extra` counts the others."""
    finds_by: dict = {}
    for r in finds:
        finds_by.setdefault((r["index_name"], r["query_string"]), {})[r["form"]] = r
    plan_by: dict = {}
    for r in plan:
        plan_by.setdefault((r["index_name"], r["query_string"]), {})[
            (r["form"], r["s"], r["rank"])
        ] = r["selected"]
    win_by: dict = {}
    extra = 0
    for r in windows:
        k = (r["index_name"], r["query_string"])
        pk = (r["form"], r["s"], r["rank"])
        if pk in plan_by.get(k, {}):
            win_by.setdefault(k, {})[pk] = r
        else:
            extra += 1
    return finds_by, plan_by, win_by, extra


def diagnostics(plan, windows) -> dict:
    """Counts the grid does not carry: `extra_windows` (windows rows whose key is not
    in plan, ignored by every count), `plan_rows`, `windows_rows`, `plan_unfetched`
    (plan rows without a windows row)."""
    _, plan_by, win_by, extra = _tables([], plan, windows)
    planned = sum(len(v) for v in plan_by.values())
    fetched = sum(len(v) for v in win_by.values())
    return {
        "extra_windows": extra,
        "plan_rows": planned,
        "windows_rows": fetched + extra,
        "plan_unfetched": planned - fetched,
    }


def year_conflict_list(years, names) -> list[dict]:
    """[{prompt_name, fact, years: [..]}] for every name with more than one distinct
    year for a fact (the `year_conflicts` argument of summary_markdown)."""
    out = []
    for name in names:
        for fact in fact_match.FACTS:
            ys = sorted({int(y) for y in (years.get(name, {}).get(fact) or [])})
            if len(ys) > 1:
                out.append({"prompt_name": name, "fact": fact, "years": ys})
    return out


def _pool_of(by_form) -> tuple[int, bool]:
    """(pool, pool_mismatch) from one string's find rows on one index. Pool is the
    sum of segment lengths over both forms (pool_positions in fetch_windows.py);
    mismatch = that differs from the summed cnt, or a segment list is unreadable
    (then cnt stands in for it)."""
    pool = cnt = 0
    bad = False
    for form in ("plain", "glued"):
        r = by_form.get(form)
        if r is None:
            continue
        c = int(r.get("cnt") or 0)
        cnt += c
        try:
            pool += sum(int(e) - int(s) for s, e in json.loads(r["segment_by_shard_json"]))
        except (TypeError, ValueError, KeyError):
            pool += c
            bad = True
    return pool, bad or pool != cnt


class _StringData:
    """What the cache holds for one (index, query string)."""

    __slots__ = ("pool", "selected", "planned", "fetched", "not_fetched", "unplanned",
                 "glued_missing", "pool_mismatch", "wins")


def _string_data(key, finds_by, plan_by, win_by) -> _StringData:
    sd = _StringData()
    by_form = finds_by.get(key)
    plan_sel = plan_by.get(key, {})
    sd.wins = list(win_by.get(key, {}).values())
    sd.planned = len(plan_sel)
    sd.fetched = len(sd.wins)
    sd.not_fetched = by_form is None
    if by_form is None:
        # No finds row: never fetched. Not a zero, so it makes the cell incomplete.
        sd.pool, sd.pool_mismatch, sd.glued_missing = 0, False, False
        sd.selected, sd.unplanned = "", False
        return sd
    sd.pool, sd.pool_mismatch = _pool_of(by_form)
    sd.glued_missing = "glued" not in by_form
    if plan_sel:
        # selected is decided per string in store_plan, so read it per string.
        sd.selected = "sample" if "sample" in plan_sel.values() else "all"
    else:
        # No plan rows: pool 0 is a true zero (all, nothing to fetch); a non-empty
        # pool without a plan was never fetched, so `unplanned` makes it incomplete.
        sd.selected = "all" if sd.pool <= CAP_FULL else "sample"
    sd.unplanned = sd.pool > 0 and not plan_sel
    return sd


# ---------------------------------------------------------------------------
# Windows: text, document, URL
# ---------------------------------------------------------------------------


def _find_url(node) -> str | None:
    """First string value starting with http:// or https://, depth first."""
    if isinstance(node, str):
        return node if node.startswith(("http://", "https://")) else None
    if isinstance(node, dict):
        node = node.values()
    elif not isinstance(node, (list, tuple)):
        return None
    for child in node:
        found = _find_url(child)
        if found:
            return found
    return None


def _domain(metadata_json) -> str | None:
    """Lowercased host without a leading www., or None (NULL, malformed, no URL)."""
    if metadata_json is None:
        return None
    try:
        meta = json.loads(metadata_json) if isinstance(metadata_json, (str, bytes)) else metadata_json
        url = _find_url(meta)
        host = urlsplit(url).hostname if url else None
    except (ValueError, TypeError):
        return None
    if not host:
        return None
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


class _Win:
    __slots__ = ("form", "doc_key", "domain", "parts", "text", "row")


def _load_win(row) -> _Win:
    w = _Win()
    w.row = row
    w.form = row["form"]
    w.doc_key = (
        (row["index_name"], row["s"], row["doc_ix"]) if DOC_KEY_WITH_SHARD
        else (row["index_name"], row["doc_ix"])
    )
    w.domain = _domain(row.get("metadata_json"))
    try:
        spans = json.loads(row["spans_json"])
        w.parts = fact_match.window_text(spans, row.get("needle_offset"), row.get("disp_len"))
        w.text = "".join(w.parts)
    except (TypeError, ValueError, KeyError):
        w.parts, w.text = None, ""  # unreadable window: fetched, but it can never match
    return w


class _WinCache:
    def __init__(self):
        self._d: dict = {}

    def get(self, row) -> _Win:
        k = (row["index_name"], row["query_string"], row["form"], row["s"], row["rank"])
        w = self._d.get(k)
        if w is None:
            w = self._d[k] = _load_win(row)
        return w


def _entries(win: _Win, year: int, fact: str, person_years: dict) -> list:
    """[(occurrence, cue_match)] for the year, nearest first."""
    if win.parts is None:
        return []
    occs = fact_match.find_year(*win.parts, year)
    out = [(o, fact_match.cue_match(win.text, o, year, fact, person_years)) for o in occs]
    out.sort(key=lambda oc: (oc[0].distance, oc[0].start))
    return out


def _passes(occ, cue, matcher) -> bool:
    if matcher == "bare":
        return True
    if matcher == "cue":
        return bool(cue)
    if matcher == "near_or_cue":
        # occ.distance, not fact_match.in_width(occ, NEAR): in_width adds the year's own
        # length, so a year ending 101 to 104 characters out would drop.
        return occ.distance <= NEAR or bool(cue)
    raise ValueError("unknown matcher %r" % matcher)


def _nearest(entries, half_width, matcher):
    """The occurrence with the smallest distance that is inside the width and passes
    the matcher, or None."""
    for occ, cue in entries:
        if fact_match.in_width(occ, half_width) and _passes(occ, cue, matcher):
            return occ
    return None


def _band(distance: int) -> str:
    for label, (lo, hi) in zip(BAND_LABELS, BANDS):
        if hi is None or distance <= hi:
            return label
    return BAND_LABELS[-1]


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


class _Res:
    """Matches of one string in one (width, matcher)."""

    __slots__ = ("matched", "plain", "glued", "docs", "dedup", "url_n", "domains")

    def __init__(self):
        self.matched = self.plain = self.glued = self.url_n = 0
        # dedup holds (document key, dedup key) pairs, so near-duplicate removal can group
        # DOCUMENTS: a document with two mentions near the year must not count twice.
        self.docs, self.dedup, self.domains = set(), set(), set()


def _match_string(items, half_width, matcher) -> _Res:
    res = _Res()
    for win, entries in items:
        occ = _nearest(entries, half_width, matcher)
        if occ is None:
            continue
        res.matched += 1
        if win.form == "plain":
            res.plain += 1
        else:
            res.glued += 1
        res.docs.add(win.doc_key)
        res.dedup.add((win.doc_key, fact_match.dedup_key(win.text, occ)))
        if win.domain:
            res.url_n += 1
            res.domains.add(win.domain)
    return res


def _person_years(ylists: dict, fact: str, year: int) -> dict:
    """For cue_match: the row's year for this fact; for the other fact the only year,
    or the smallest when the prompts disagree (arbitrary but deterministic)."""
    out = {}
    for f in fact_match.FACTS:
        out[f] = year if f == fact else (min(ylists[f]) if ylists[f] else None)
    return out


def count_grid(entities, finds, plan, windows, years, names, indexes) -> list[dict]:
    """One dict per grid cell (CSV_COLUMNS), sorted by prompt_name, index, fact,
    strings, matcher, width (full last). `names` = prompt names to count, `indexes` =
    index names. Includes `year`, `year_conflict`, the flags (ambiguous_name,
    ambiguous_stripped, shared_string, overlap, short_year, pool_mismatch,
    glued_missing), matched_plain/glued and matched_name/stripped/folded/alias."""
    finds_by, plan_by, win_by, _extra = _tables(finds, plan, windows)
    cache = _WinCache()
    active: dict = defaultdict(list)
    inactive: dict = defaultdict(int)
    owners: dict = defaultdict(set)  # active query string -> prompt names
    for e in entities:
        if _active(e):
            active[e["prompt_name"]].append(e)
            owners[e["query_string"]].add(e["prompt_name"])
        else:
            inactive[e["prompt_name"]] += 1
    sdata: dict = {}

    def sd_of(index, qs):
        k = (index, qs)
        if k not in sdata:
            sdata[k] = _string_data(k, finds_by, plan_by, win_by)
        return sdata[k]

    rows: list[dict] = []
    for name in names:
        ents = sorted(active.get(name, []), key=lambda e: e["query_string"])
        ylists = {f: sorted({int(y) for y in (years.get(name, {}).get(f) or [])})
                  for f in fact_match.FACTS}
        amb_name = int(any(_truthy(e.get("ambiguous_name")) for e in ents))
        amb_strip = int(any(_truthy(e.get("ambiguous_stripped")) for e in ents))
        for fact in fact_match.FACTS:
            for year in ylists[fact]:
                pyears = _person_years(ylists, fact, year)
                for index in indexes:
                    items = {}  # query string -> [(win, entries)]
                    for e in ents:
                        sd = sd_of(index, e["query_string"])
                        items[e["query_string"]] = [
                            (w, _entries(w, year, fact, pyears))
                            for w in (cache.get(r) for r in sd.wins)
                        ]
                    for wi, half_width in enumerate(HALF_WIDTHS):
                        for matcher in MATCHERS:
                            res = {q: _match_string(items[q], half_width, matcher)
                                   for q in items}
                            for sset in STRING_SETS:
                                strs = [e for e in ents
                                        if sset == "all" or e["source"] != "stripped"]
                                rows.append(_cell(
                                    name, index, fact, year, len(ylists[fact]) > 1,
                                    WIDTH_LABELS[wi], matcher, sset, strs, res, sd_of,
                                    owners, inactive[name], amb_name, amb_strip))
    rows.sort(key=lambda r: (r["prompt_name"], r["index"], r["fact"], r["year"],
                             r["strings"], r["matcher"], WIDTH_LABELS.index(r["half_width"])))
    return rows


def _dedup_groups(pairs) -> int:
    """Number of document groups in `pairs`, a set of (document key, dedup key): documents
    that share a dedup key are joined, transitively (A with B, B with C gives one group)."""
    parent: dict = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first: dict = {}  # dedup key -> the first document seen with it
    for doc, key in sorted(pairs, key=repr):  # sorted only so the result cannot depend on set order
        parent.setdefault(doc, doc)
        if key in first:
            a, b = find(first[key]), find(doc)
            if a != b:
                parent[b] = a
        else:
            first[key] = doc
    return len({find(d) for d in parent})


def _cell(name, index, fact, year, conflict, width, matcher, sset, strs, res, sd_of,
          owners, n_inactive, amb_name, amb_strip) -> dict:
    row = {c: "" for c in CSV_COLUMNS}
    row.update(
        prompt_name=name, index=index, fact=fact, year=year, year_conflict=int(conflict),
        half_width=width, matcher=matcher, strings=sset, n_strings=len(strs),
        n_inactive_strings=n_inactive, ambiguous_name=amb_name,
        ambiguous_stripped=amb_strip, short_year=int(year < 1000),
        shared_string=0, overlap=0, pool_mismatch=0, glued_missing=0,
        matched_plain=0, matched_glued=0, mentions_matched=0, matched_with_url=0,
        matched_name=0, matched_stripped=0, matched_folded=0, matched_alias=0,
        windows_planned=0, windows_fetched=0, not_fetched_strings=0, pool=0,
        est_mentions=0.0, n_docs=0, n_docs_dedup=0, per_string_json="[]",
    )
    if not strs:
        # A name with only stripped strings under no_stripped: nothing to count.
        row.update(status="unknown", incomplete=1, selected="none")
        return row
    planned = fetched = matched = est = 0
    docs, dedup, domains = set(), set(), set()
    incomplete = False
    all_sel = True
    detail = []
    for e in strs:
        q = e["query_string"]
        sd, r = sd_of(index, q), res[q]
        planned += sd.planned
        fetched += sd.fetched
        matched += r.matched
        est += r.matched if sd.selected == "all" else (
            r.matched / sd.fetched * sd.pool if sd.fetched else 0.0)
        docs |= r.docs
        dedup |= r.dedup
        domains |= r.domains
        row["matched_plain"] += r.plain
        row["matched_glued"] += r.glued
        row["matched_with_url"] += r.url_n
        if e["source"] in SOURCES:
            row["matched_" + e["source"]] += r.matched
        row["pool"] += sd.pool
        row["not_fetched_strings"] += int(sd.not_fetched)
        row["pool_mismatch"] |= int(sd.pool_mismatch)
        row["glued_missing"] |= int(sd.glued_missing and not sd.not_fetched)
        row["shared_string"] |= int(len(owners.get(q, ())) > 1)
        incomplete |= sd.not_fetched or sd.unplanned or sd.fetched < sd.planned
        all_sel &= sd.selected == "all"
        detail.append({"q": q, "src": e["source"], "pool": None if sd.not_fetched else sd.pool,
                       "sel": sd.selected, "planned": sd.planned, "fetched": sd.fetched,
                       "matched": r.matched})
    # overlap: a string of this cell is a substring of another string of it (case sensitive).
    qs = [e["query_string"] for e in strs]
    row["overlap"] = int(any(a != b and a in b for a in qs for b in qs))
    n_c_docs = len(docs)
    row.update(
        windows_planned=planned, windows_fetched=fetched,
        coverage=round(fetched / planned, 4) if planned else (1.0 if not incomplete else ""),
        incomplete=int(incomplete), selected="all" if all_sel else "sample",
        mentions_matched=matched, est_mentions=round(est, 3), n_docs=n_c_docs,
        n_docs_dedup=_dedup_groups(dedup), n_domains=len(domains) if domains else "",
        url_share=round(row["matched_with_url"] / matched, 4) if matched else "",
        per_string_json=json.dumps(detail, separators=(",", ":")),
    )
    if all_sel and not incomplete:
        row.update(n_c=n_c_docs, n_c_unit="docs")
    else:
        row.update(n_c=round(est, 1), n_c_unit="est_mentions")
    if matched:
        row["status"] = "counted"
    elif all_sel and not incomplete:
        row["status"] = "zero"  # a true 0: every string fully fetched and nothing matched
    else:
        row["status"] = "unknown"  # never n_c = 0; stays out of tau
    return row


def truncated_side_share(windows, indexes) -> dict[str, float]:
    """Per index: the share of (window, side) pairs whose stored text on that side
    of the needle is shorter than 1600 characters (the widest fixed width)."""
    widest = max(w for w in HALF_WIDTHS if w is not None)
    short: dict = defaultdict(int)
    total: dict = defaultdict(int)
    for r in windows:
        idx = r["index_name"]
        if idx not in indexes:
            continue
        try:
            before, _needle, after = fact_match.window_text(
                json.loads(r["spans_json"]), r.get("needle_offset"), r.get("disp_len"))
        except (TypeError, ValueError, KeyError):
            continue
        for side in (before, after):
            total[idx] += 1
            short[idx] += int(len(side) < widest)
    return {i: short[i] / total[i] for i in indexes if total[i]}


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _f(v):
    """A float from a row value (numbers or CSV strings), or None if empty."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _nearest_rank(values, p):
    """Nearest-rank percentile: sorted value at rank ceil(p / 100 x n)."""
    if not values:
        return None
    s = sorted(values)
    return s[max(1, math.ceil(p / 100 * len(s))) - 1]


def _g(x) -> str:
    return "n/a" if x is None else "%g" % round(x, 3)


def _pct(x) -> str:
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def _short(index) -> str:
    return INDEX_LABELS.get(index, index)


def _table(header, body) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return out + [""]


def _stats(rows):
    """(counted, zero, unknown, docs values, est_mentions facts) for a group of rows."""
    st = [r["status"] for r in rows]
    docs = [_f(r["n_c"]) for r in rows
            if r["status"] in ("counted", "zero") and r["n_c_unit"] == "docs"
            and _f(r["n_c"]) is not None]
    est = sum(1 for r in rows if r["status"] in ("counted", "zero")
              and r["n_c_unit"] == "est_mentions")
    return st.count("counted"), st.count("zero"), st.count("unknown"), docs, est


def _split_stats(rows):
    """(docs values above 0, est_mentions values of counted facts) for a group of rows.
    The first is tau's population (n_c > 0) restricted to `docs` units; the second
    is kept apart and never mixed with it."""
    pos = [v for v in (_f(r["n_c"]) for r in rows
                       if r["status"] in ("counted", "zero") and r["n_c_unit"] == "docs")
           if v is not None and v > 0]
    est = [v for v in (_f(r["n_c"]) for r in rows
                       if r["status"] == "counted" and r["n_c_unit"] == "est_mentions")
           if v is not None]
    return pos, est


def summary_markdown(rows, truncated_share, year_conflicts, extras=None) -> str:
    """The text of pilot_summary.md, headed "Pilot preview over 20 names, NOT tau".
    Every number comes from `rows`. Nearest-rank percentiles (say so). No em dashes.
    `truncated_share` = truncated_side_share output; `year_conflicts` =
    year_conflict_list output; `extras` = optional {label: number} printed as given
    (for example diagnostics)."""
    rows = list(rows)
    indexes = sorted({r["index"] for r in rows})
    by_rule: dict = defaultdict(list)
    for r in rows:
        by_rule[(r["index"], r["strings"], r["matcher"], r["half_width"])].append(r)
    facts_n = len({(r["prompt_name"], r["fact"], str(r["year"])) for r in rows})
    names_n = len({r["prompt_name"] for r in rows})
    out = [
        "# Pilot preview over 20 names, NOT tau",
        "",
        "Nothing here sets tau or picks the counting rule. Every number below is computed "
        "from the rows of pilot_counts.csv (%d rows, %d names, %d name-fact-year facts, "
        "%d indexes)." % (len(rows), names_n, facts_n, len(indexes)),
        "",
        "Reading guide: a fact is one (name, fact, year). `counted` = at least one window "
        "matched; `zero` = every string fully fetched and nothing matched (a true 0); "
        "`unknown` = nothing matched but a string was sampled or incomplete (never a 0, "
        "kept out of tau). Percentiles use the nearest-rank method (sorted value at rank "
        "ceil(p/100 x n); the median is the 50th percentile, the lower middle value for an "
        "even n) and cover facts with unit `docs` only. The `incl. 0` columns include true "
        "zeros. `facts n_c>0` and `p10 n_c>0` follow D-011's population (facts with "
        "n_c > 0) restricted to `docs` units: they are still a 20-name preview, NOT tau. "
        "Famous names have sampled pools, so their n_c is in `est_mentions` units and "
        "appears only in the `est_mentions facts` and `median est_mentions` columns, never "
        "mixed with `docs` values.",
        "",
        "## Per index and rule",
        "",
    ]
    body = []
    for key in sorted(by_rule, key=lambda k: (k[0], k[1], k[2], WIDTH_LABELS.index(k[3]))):
        c, z, u, docs, est = _stats(by_rule[key])
        pos, est_vals = _split_stats(by_rule[key])
        body.append([_short(key[0]), key[1], key[2], key[3], c, z, u, len(docs),
                     _g(_nearest_rank(docs, 50)), _g(_nearest_rank(docs, 10)),
                     len(pos), _g(_nearest_rank(pos, 10)), est,
                     _g(_nearest_rank(est_vals, 50))])
    out += _table(["index", "strings", "matcher", "width", "counted", "zero", "unknown",
                   "docs facts", "median n_c (incl. 0)", "p10 n_c (incl. 0)", "facts n_c>0",
                   "p10 n_c>0", "est_mentions facts", "median est_mentions"], body)

    out += ["## Width and matcher (strings all): median n_c (incl. 0) over docs facts, "
            "and the count", ""]
    body = []
    for idx in indexes:
        for m in MATCHERS:
            cells = []
            for w in WIDTH_LABELS:
                _, _, _, docs, _ = _stats(by_rule.get((idx, "all", m, w), []))
                cells.append("%s (n=%d)" % (_g(_nearest_rank(docs, 50)), len(docs)))
            body.append([_short(idx), m] + cells)
    out += _table(["index", "matcher"] + list(WIDTH_LABELS), body)

    out += ["## Width and matcher (strings all): p10 n_c>0 over docs facts with n_c > 0, "
            "and the count (20-name preview, NOT tau)", ""]
    body = []
    for idx in indexes:
        for m in MATCHERS:
            cells = []
            for w in WIDTH_LABELS:
                pos, _ = _split_stats(by_rule.get((idx, "all", m, w), []))
                cells.append("%s (n=%d)" % (_g(_nearest_rank(pos, 10)), len(pos)))
            body.append([_short(idx), m] + cells)
    out += _table(["index", "matcher"] + list(WIDTH_LABELS), body)

    out += ["## All strings vs no_stripped (full width)", ""]
    body = []
    for idx in indexes:
        for m in MATCHERS:
            for s in STRING_SETS:
                grp = by_rule.get((idx, s, m, "full"), [])
                c, z, u, docs, est = _stats(grp)
                pos, est_vals = _split_stats(grp)
                body.append([_short(idx), m, s, c, z, u, _g(_nearest_rank(docs, 50)),
                             _g(_nearest_rank(docs, 10)), len(docs), len(pos),
                             _g(_nearest_rank(pos, 10)), _g(_nearest_rank(est_vals, 50))])
    out += _table(["index", "matcher", "strings", "counted", "zero", "unknown",
                   "median n_c (incl. 0)", "p10 n_c (incl. 0)", "docs facts", "facts n_c>0",
                   "p10 n_c>0", "median est_mentions"], body)

    out += ["## Where matches come from (strings all, full width)", "",
            "Glued share = matched_glued / mentions_matched; stripped share = "
            "matched_stripped / mentions_matched, summed over the facts of that index.", ""]
    body = []
    for idx in indexes:
        for m in MATCHERS:
            rs = by_rule.get((idx, "all", m, "full"), [])
            tot = sum(_f(r["mentions_matched"]) or 0 for r in rs)
            gl = sum(_f(r["matched_glued"]) or 0 for r in rs)
            sp = sum(_f(r["matched_stripped"]) or 0 for r in rs)
            body.append([_short(idx), m, int(tot), int(gl), _pct(gl / tot if tot else None),
                         int(sp), _pct(sp / tot if tot else None)])
    out += _table(["index", "matcher", "mentions matched", "glued", "glued share",
                   "stripped", "stripped share"], body)

    out += ["## n_docs_dedup vs n_docs (strings all, full width, facts with a match)", ""]
    body = []
    for idx in indexes:
        for m in MATCHERS:
            rs = [r for r in by_rule.get((idx, "all", m, "full"), [])
                  if (_f(r["mentions_matched"]) or 0) > 0]
            nd = sum(_f(r["n_docs"]) or 0 for r in rs)
            ndd = sum(_f(r["n_docs_dedup"]) or 0 for r in rs)
            lower = sum(1 for r in rs if (_f(r["n_docs_dedup"]) or 0) < (_f(r["n_docs"]) or 0))
            body.append([_short(idx), m, len(rs), int(nd), int(ndd), lower])
    out += _table(["index", "matcher", "facts", "sum n_docs", "sum n_docs_dedup",
                   "facts with dedup < docs"], body)
    over = sum(1 for r in rows if (_f(r["n_docs_dedup"]) or 0) > (_f(r["n_docs"]) or 0))
    out += ["Rows over the whole table where n_docs_dedup > n_docs: %d (0 by construction: "
            "dedup counts document groups)." % over, ""]

    out += ["## URL share and truncated sides, per index", "",
            "url_share = matched windows with a URL / matched windows (strings all, full "
            "width, bare). truncated_side_share = share of (window, side) pairs whose stored "
            "text on that side of the needle is shorter than 1600 characters.", ""]
    body = []
    for idx in indexes:
        rs = by_rule.get((idx, "all", "bare", "full"), [])
        tot = sum(_f(r["mentions_matched"]) or 0 for r in rs)
        url = sum(_f(r["matched_with_url"]) or 0 for r in rs)
        ts = (truncated_share or {}).get(idx)
        body.append([_short(idx), int(tot), int(url), _pct(url / tot if tot else None),
                     _pct(ts)])
    out += _table(["index", "matched windows", "with URL", "url_share",
                   "truncated_side_share"], body)

    out += ["## Coverage (strings all)", "",
            "One line per (name, index): windows fetched / windows planned, summed over "
            "names; incomplete = names with a planned window not fetched or a string "
            "never fetched.", ""]
    seen = {}
    for r in rows:
        if r["strings"] == "all":
            seen.setdefault((r["prompt_name"], r["index"]), r)
    body = []
    for idx in indexes:
        rs = [r for k, r in seen.items() if k[1] == idx]
        pl = sum(_f(r["windows_planned"]) or 0 for r in rs)
        fe = sum(_f(r["windows_fetched"]) or 0 for r in rs)
        inc = sum(1 for r in rs if _f(r["incomplete"]) == 1)
        body.append([_short(idx), len(rs), int(pl), int(fe), _pct(fe / pl if pl else None), inc])
    out += _table(["index", "names", "planned", "fetched", "coverage", "incomplete names"], body)

    out += ["## Year conflicts", ""]
    yc = list(year_conflicts or [])
    n_rows = sum(1 for r in rows if _f(r.get("year_conflict")) == 1)
    out.append("%d name-fact pairs have more than one year across prompts; %d rows carry "
               "year_conflict = 1." % (len(yc), n_rows))
    for c in yc:
        if isinstance(c, dict):
            name, fact, ys = c["prompt_name"], c["fact"], c["years"]
        else:
            name, fact, ys = c
        out.append("- %s, %s: %s" % (name, fact, ", ".join(str(y) for y in ys)))
    out.append("")
    if extras:
        out += ["## Diagnostics", ""]
        out += ["- %s: %s" % (k, extras[k]) for k in extras]
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Hand-check packet
# ---------------------------------------------------------------------------


def packet_candidates(entities, plan, windows, years, names) -> list[dict]:
    """Every (window, fact) whose NEAREST occurrence of the fact's year passes bare
    within full width, strings all. One dict per candidate: index, prompt_name,
    query_string, query_source, form, s, rank, fact, year, distance_chars, band,
    matched_bare (1), matched_cue, snippet."""
    _, _, win_by, _ = _tables([], plan, windows)
    cache = _WinCache()
    out = []
    for name in names:
        ents = sorted((e for e in entities if e["prompt_name"] == name and _active(e)),
                      key=lambda e: e["query_string"])
        ylists = {f: sorted({int(y) for y in (years.get(name, {}).get(f) or [])})
                  for f in fact_match.FACTS}
        for e in ents:
            for k in sorted(k for k in win_by if k[1] == e["query_string"]):
                for row in win_by[k].values():
                    win = cache.get(row)
                    if win.parts is None:
                        continue
                    for fact in fact_match.FACTS:
                        best = None  # (distance, year, occ, cue); nearer year wins on a conflict
                        for year in ylists[fact]:
                            ent = _entries(win, year, fact, _person_years(ylists, fact, year))
                            if ent and (best is None or ent[0][0].distance < best[0]):
                                best = (ent[0][0].distance, year, ent[0][0], ent[0][1])
                        if best is None:
                            continue
                        dist, year, occ, cue = best
                        out.append({
                            "index": row["index_name"], "prompt_name": name,
                            "query_string": e["query_string"], "query_source": e["source"],
                            "form": row["form"], "s": row["s"], "rank": row["rank"],
                            "fact": fact, "year": year, "distance_chars": dist,
                            "band": _band(dist), "matched_bare": 1, "matched_cue": int(cue),
                            "snippet": fact_match.snippet(*win.parts, occ),
                        })
    return out


def _cand_key(c):
    return (c["index"], c["query_string"], c["form"], c["s"], c["rank"], c["fact"],
            c["prompt_name"])


def _quota_counts(rows) -> dict:
    return {
        "stripped_rows": sum(1 for r in rows if r["query_source"] == "stripped"),
        "indexes": len({r["index"] for r in rows}),
        "facts": len({r["fact"] for r in rows}),
        "names": len({r["prompt_name"] for r in rows}),
    }


def _helps(cand, quota, chosen_rows) -> bool:
    """Would adding `cand` raise the count for `quota`?"""
    if quota == "stripped_rows":
        return cand["query_source"] == "stripped"
    key = {"indexes": "index", "facts": "fact", "names": "prompt_name"}[quota]
    return cand[key] not in {r[key] for r in chosen_rows}


def _swap_for(quota, cands, chosen, stratum_of, order):
    """Replace one drawn row by an undrawn row that raises `quota`, without breaking
    a quota that is met now. Deterministic (no RNG). True if a swap was made."""
    in_set = set(chosen)
    rows = [cands[i] for i in chosen]
    before = _quota_counts(rows)
    size = defaultdict(int)
    for i in chosen:
        size[stratum_of[i]] += 1
    for u in range(len(cands)):
        if u in in_set or not _helps(cands[u], quota, rows):
            continue
        su = stratum_of[u]
        # Victims: same stratum first (keeps the strata coverage), then rows from the
        # strata that hold more than the per-stratum target, largest first.
        victims = [v for v in chosen if stratum_of[v] == su]
        for st in sorted(order, key=lambda s: -size[s]):
            if st != su and size[st] > PER_STRATUM:
                victims += [v for v in chosen if stratum_of[v] == st]
        for v in victims:
            after = _quota_counts([cands[i] for i in chosen if i != v] + [cands[u]])
            if after[quota] <= before[quota]:
                continue
            if any(before[q] >= QUOTAS[q] and after[q] < QUOTAS[q] for q in QUOTAS):
                continue
            chosen[chosen.index(v)] = u
            return True
    return False


def select_packet(candidates, seed) -> tuple[list[dict], list[dict], list[str]]:
    """(packet rows with packet_id P001..., strata rows, unmet-quota messages).
    5 per (band, matched_cue) stratum, short strata taken whole with the gap filled
    from the largest strata; quotas: >= 10 rows from stripped strings, >= 3 indexes,
    both facts, >= 8 names. Deterministic: candidates sorted by (index, query_string,
    form, s, rank, fact) before a random.Random(seed) draw. Strata rows: band,
    matched_cue, query_source, population, drawn."""
    cands = sorted(candidates, key=_cand_key)
    rng = random.Random(seed)
    order = [(b, cue) for b in BAND_LABELS for cue in (0, 1)]
    stratum_of = [(c["band"], int(c["matched_cue"])) for c in cands]
    members = {st: [i for i, s in enumerate(stratum_of) if s == st] for st in order}
    drawn: dict = {}
    rest: dict = {}
    for st in order:
        pool = members[st]
        picked = sorted(rng.sample(range(len(pool)), min(PER_STRATUM, len(pool))))
        drawn[st] = [pool[j] for j in picked]
        rest[st] = [pool[j] for j in range(len(pool)) if j not in set(picked)]
    # Fill the gap left by short strata from the strata with the most rows still undrawn.
    gap = PER_STRATUM * len(order) - sum(len(v) for v in drawn.values())
    while gap > 0:
        st = max(order, key=lambda s: (len(rest[s]), -order.index(s)))
        if not rest[st]:
            break
        drawn[st].append(rest[st].pop(rng.randrange(len(rest[st]))))
        gap -= 1
    chosen = [i for st in order for i in drawn[st]]

    for quota in QUOTAS:
        while _quota_counts([cands[i] for i in chosen])[quota] < QUOTAS[quota]:
            if not _swap_for(quota, cands, chosen, stratum_of, order):
                break
    counts = _quota_counts([cands[i] for i in chosen])
    labels = {"stripped_rows": "rows from stripped strings", "indexes": "indexes",
              "facts": "facts", "names": "different names"}
    messages = ["quota unmet: %d %s (need at least %d)" % (counts[q], labels[q], QUOTAS[q])
                for q in QUOTAS if counts[q] < QUOTAS[q]]
    if len(chosen) < PER_STRATUM * len(order):
        messages.append("packet has %d rows, target %d: candidates ran out"
                        % (len(chosen), PER_STRATUM * len(order)))

    chosen.sort(key=lambda i: (_cand_key(cands[i]), i))
    packet = []
    for n, i in enumerate(chosen, 1):
        row = {c: "" for c in PACKET_COLUMNS}
        row.update({k: v for k, v in cands[i].items() if k in row})
        row["packet_id"] = "P%03d" % n
        row["label"] = ""
        packet.append(row)
    pop: dict = defaultdict(int)
    got: dict = defaultdict(int)
    for i, c in enumerate(cands):
        pop[(stratum_of[i], c["query_source"])] += 1
    for i in chosen:
        got[(stratum_of[i], cands[i]["query_source"])] += 1
    strata = [
        {"band": st[0], "matched_cue": st[1], "query_source": src,
         "population": pop[(st, src)], "drawn": got[(st, src)]}
        for st in order for src in sorted({s for (t, s) in pop if t == st})
    ]
    return packet, strata, messages


# ---------------------------------------------------------------------------
# Scoring the labelled packet
# ---------------------------------------------------------------------------


def _prec(rows):
    """(labelled, unclear, scored, correct, precision or None). Precision is over the
    labelled rows minus `unclear`."""
    labelled = len(rows)
    unclear = sum(1 for r in rows if r["label"] == "unclear")
    scored = labelled - unclear
    correct = sum(1 for r in rows if r["label"] == "correct")
    return labelled, unclear, scored, correct, (correct / scored if scored else None)


def _fmt_prec(rows) -> str:
    labelled, unclear, scored, correct, p = _prec(rows)
    tail = "correct %d of %d scored; labelled %d, unclear %d" % (correct, scored, labelled, unclear)
    if scored < 5 or p is None:
        return "too few rows (%s)" % tail
    return "%.3f (%s)" % (p, tail)


def _matcher_rows(rows, matcher):
    if matcher not in SCORED_MATCHERS:
        raise ValueError("the precision table scores %s only, not %r" % (SCORED_MATCHERS, matcher))
    return rows if matcher == "bare" else [r for r in rows if r["cue"] == 1]


def score_labels(packet_rows, strata_rows) -> str:
    """The text of pilot_precision.md from labelled packet rows (dicts with a `label`
    column). Trims and lowercases labels; raises LabelError (listing packet_ids) on a
    label outside LABELS, TooFewLabels under MIN_LABELLED labelled rows. Precision =
    correct / (labelled - unclear), per band x matcher, per fact, per query source,
    and per rule (width x matcher) both unweighted and weighted by band population
    share from `strata_rows`; every figure carries its row count, "too few rows"
    under 5. Numbers only."""
    rows, bad = [], []
    for r in packet_rows:
        lab = str(r.get("label") or "").strip().lower()
        if not lab:
            continue  # empty = not labelled
        if lab not in LABELS:
            bad.append(str(r.get("packet_id")))
            continue
        rows.append({"label": lab, "band": str(r["band"]), "cue": int(float(r["matched_cue"])),
                     "fact": r["fact"], "source": r["query_source"]})
    if bad:
        raise LabelError("labels outside %s for packet_ids: %s" % (", ".join(LABELS), ", ".join(bad)))
    if len(rows) < MIN_LABELLED:
        raise TooFewLabels("%d labelled rows, need at least %d" % (len(rows), MIN_LABELLED))

    pop: dict = defaultdict(int)  # (band, matched_cue) -> candidates in the population
    for s in strata_rows:
        pop[(str(s["band"]), int(float(s["matched_cue"])))] += int(float(s["population"]))

    out = [
        "# Pilot precision (hand-labelled packet)",
        "",
        "Numbers only, no snippets. precision = correct / (labelled - unclear). `bare` uses "
        "every labelled row, `cue` only rows with matched_cue = 1. 'scored' = labelled minus "
        "unclear; a figure with fewer than 5 scored rows is shown as 'too few rows'. Labelled "
        "rows: %d. The packet over-samples far bands, so only the weighted rule figure "
        "estimates a rule's real precision." % len(rows),
        "",
        "## Per band and matcher", "",
    ]
    out += _table(["band", "bare", "cue"], [
        [b, _fmt_prec([r for r in rows if r["band"] == b]),
         _fmt_prec(_matcher_rows([r for r in rows if r["band"] == b], "cue"))]
        for b in BAND_LABELS])
    out += ["## Per fact", ""]
    out += _table(["fact", "bare", "cue"], [
        [f, _fmt_prec([r for r in rows if r["fact"] == f]),
         _fmt_prec(_matcher_rows([r for r in rows if r["fact"] == f], "cue"))]
        for f in sorted({r["fact"] for r in rows})])
    out += ["## Per query source", ""]
    out += _table(["query source", "bare", "cue"], [
        [q, _fmt_prec([r for r in rows if r["source"] == q]),
         _fmt_prec(_matcher_rows([r for r in rows if r["source"] == q], "cue"))]
        for q in sorted({r["source"] for r in rows})])

    out += [
        "## Per candidate rule (width x matcher)", "",
        "A band is inside width W if its upper bound is at most W (full: all bands). "
        "Unweighted = correct / scored over all labelled rows inside W. Weighted = sum over "
        "inside bands of (population share of the band among inside candidates for that "
        "matcher) x (precision within the band). A band with no scored rows is excluded and "
        "the weights of the remaining bands are renormalised to sum to 1; excluded bands are "
        "listed.", "",
    ]
    body = []
    for wi, W in enumerate(HALF_WIDTHS):
        inside = [b for b, (lo, hi) in zip(BAND_LABELS, BANDS)
                  if W is None or (hi is not None and hi <= W)]
        for m in SCORED_MATCHERS:
            inrows = _matcher_rows([r for r in rows if r["band"] in inside], m)
            parts, excluded = [], []
            for b in inside:
                brows = _matcher_rows([r for r in rows if r["band"] == b], m)
                p = _prec(brows)[4]
                weight = pop[(b, 0)] + pop[(b, 1)] if m == "bare" else pop[(b, 1)]
                if p is None or weight == 0:
                    excluded.append(b)
                else:
                    parts.append((weight, p))
            tot = sum(w for w, _ in parts)
            scored = _prec(inrows)[2]
            if scored < 5 or not tot:
                weighted = "too few rows"
            else:
                weighted = "%.3f" % (sum(w / tot * p for w, p in parts))
            body.append([WIDTH_LABELS[wi], m, len(inrows), scored, _fmt_prec(inrows), weighted,
                         ", ".join(b for b in inside if b not in excluded) or "none",
                         ", ".join(excluded) or "none"])
    out += _table(["width", "matcher", "labelled rows", "scored rows", "unweighted",
                   "weighted", "bands used", "bands excluded"], body)
    return "\n".join(out)
