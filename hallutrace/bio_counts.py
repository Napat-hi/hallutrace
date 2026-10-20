"""Biographies corpus counts for one fetch shard, and what the merge checks.

Each shard holder counts on their own window cache and commits numbers only. Two grids,
both from `pilot_counts.count_grid`, the same counter the pilot uses for historical_events:

    role c   n_c: the person's correct birth/death years (Wikidata, the alias table's QID)
    role h   n_h: the years the models stated wrongly (hallucinated year claims from
             results/bio_year_claims.csv whose year is not one of the person's own).
             A claim with no Wikidata year for its fact is left out and counted in `meta`,
             so role h uses only (name, fact) pairs that role c also has

The export keeps the whole grid (every half-width, matcher and string set): the merge
picks the counting rule's cell from each grid.
For role h a name with several wrong years shows `year_conflict`; there it only means
the models gave more than one wrong year, and every year has its own row.

`meta` records what the merge must check before combining shards: the shard and its
names, the fetcher's constants and version, whether every planned window was fetched,
the indexes present, and hashes of the counting code and the claims file, so every
shard is counted by the same rule on the same claims.

The merge checks the shards, keeps the counting rule's cell from each grid and sets the
second tau per model: the 10th percentile of n_c over the correct facts with n_c > 0, every
cutoff from 5% to 20%. The population is the pilot's (`pilot_counts._split_stats`): `docs`
counts only, with facts counted from a sampled pool (`est_mentions`) kept apart and shown
in a second column; `unknown` never enters.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

from . import pilot_counts as pc

ROLE_COLUMN = "role"
EXPORT_VERSION = "2"   # 2: role h without claims that have no Wikidata year
# The counting rule: a year within 100 characters of the name, or a birth/death cue anywhere
# in the window, without the disambiguator-stripped short forms (pilot_counts' `near_or_cue`
# matcher). The merge refuses an export without this cell rather than read
# its absence as zero counts.
RULE = {"matcher": "near_or_cue", "half_width": "full", "strings": "no_stripped"}
PERCENTS = tuple(range(5, 21))   # the tau sweep
# A shard's grid is about 50 MB as CSV and under 1 MB gzipped, so the export
# writes it gzipped. read_shard still reads a plain counts.csv; the numbers are the same.
COUNTS_GZ = "counts.csv.gz"
COUNTS_CSV = "counts.csv"
TAU_P = 10


def sha256_files(paths: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def years_correct(names: Iterable[str], own: Dict[str, str], snapshot: dict) -> Dict[str, dict]:
    """{name: {"birth": [...], "death": [...]}} from each person's own QID; a name with no
    own QID or no year is left out (it has no n_c)."""
    ents = snapshot.get("entities") or {}
    out = {}
    for n in names:
        ent = ents.get(own.get(n) or "", {})
        y = {"birth": sorted(set(ent.get("p569") or [])), "death": sorted(set(ent.get("p570") or []))}
        if y["birth"] or y["death"]:
            out[n] = y
    return out


def _hallucinated(claims: Iterable[dict], names: Iterable[str]) -> List[dict]:
    wanted = set(names)
    return [c for c in claims if c["name"] in wanted and c["label"] == "hallucinated"]


def years_hallucinated(claims: Iterable[dict], names: Iterable[str]) -> Dict[str, dict]:
    """{name: {"birth": [...], "death": [...]}}: stated years of hallucinated claims that are
    not the person's own year (a hallucinated claim with the right year is a verifier error).
    Only claims Wikidata can check (`wikidata_match` False); `no_reference_year` counts the
    ones with no Wikidata year."""
    out: Dict[str, dict] = {}
    for c in _hallucinated(claims, names):
        if c["wikidata_match"] != "False":
            continue
        own = {int(y) for y in c["wikidata_years"].split()}
        for y in c["stated_years"].split():
            if int(y) not in own:
                out.setdefault(c["name"], {"birth": set(), "death": set()})[c["kind"]].add(int(y))
    return {n: {k: sorted(v) for k, v in d.items()} for n, d in out.items()}


def no_reference_year(claims: Iterable[dict], names: Iterable[str]) -> dict:
    """What role h leaves out for want of a Wikidata year: the hallucinated claims, their
    names, and the (name, fact, year) rows they alone would have added."""
    claims = list(claims)
    left = [c for c in _hallucinated(claims, names) if c["wikidata_match"] == ""]
    kept = {(n, k, y) for n, d in years_hallucinated(claims, names).items()
            for k, ys in d.items() for y in ys}
    facts = {(c["name"], c["kind"], int(y)) for c in left for y in c["stated_years"].split()} - kept
    return {"claims": len(left), "names": len({c["name"] for c in left}), "facts": len(facts)}


def tag(rows: List[dict], role: str) -> List[dict]:
    return [dict(r, **{ROLE_COLUMN: role}) for r in rows]


def read_claims(path: str) -> List[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def completeness(diag: dict, indexes: List[str], expected_indexes: Iterable[str]) -> dict:
    """Whether a shard's fetch is complete enough to merge: every planned window fetched,
    and all expected indexes present."""
    missing_idx = [i for i in expected_indexes if i not in indexes]
    return {"plan_rows": diag.get("plan_rows"), "plan_unfetched": diag.get("plan_unfetched"),
            "extra_windows": diag.get("extra_windows"), "missing_indexes": missing_idx,
            "complete": not diag.get("plan_unfetched") and not missing_idx}


def check_mergeable(metas: List[dict], n_shards: Optional[int] = None) -> List[str]:
    """Problems that block a merge (empty list = mergeable): the completeness check."""
    problems = []
    if not metas:
        return ["no shard exports"]
    n = n_shards or metas[0].get("n_shards")
    shards = sorted(m.get("shard") for m in metas)
    want = ["%d/%d" % (i, n) for i in range(1, (n or 0) + 1)]
    if shards != want:
        problems.append("shards present %s, expected %s" % (shards, want))
    seen: Dict[str, str] = {}
    for m in metas:
        for name in m.get("names", []):
            if name in seen:
                problems.append("%s is in shard %s and %s" % (name, seen[name], m["shard"]))
            seen[name] = m["shard"]
    for key in ("constants", "fetch_script_version", "counting_code_sha256", "claims_sha256",
                "export_version"):
        values = {str(m.get(key)) for m in metas}
        if len(values) > 1:
            problems.append("%s differs between shards" % key)
    for m in metas:
        if not (m.get("completeness") or {}).get("complete"):
            problems.append("shard %s is not complete: %s" % (m.get("shard"), m.get("completeness")))
    return problems


def write_counts(path, columns: List[str], rows: List[dict]) -> None:
    """A shard's grid as gzipped CSV. mtime 0 and no file name in the gzip header, so the
    same rows give the same bytes on every machine and a re-export shows no git diff."""
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
            io.TextIOWrapper(gz, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_shard(shard_dir: str) -> Tuple[dict, List[dict]]:
    """(meta.json, counts rows) of one shard export: counts.csv.gz, or a plain counts.csv."""
    with open(os.path.join(shard_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    gz = os.path.join(shard_dir, COUNTS_GZ)
    if os.path.exists(gz):
        with gzip.open(gz, "rt", encoding="utf-8", newline="") as f:
            return meta, list(csv.DictReader(f))
    with open(os.path.join(shard_dir, COUNTS_CSV), encoding="utf-8", newline="") as f:
        return meta, list(csv.DictReader(f))


def select_cell(rows: List[dict], rule: Dict[str, str] = RULE) -> List[dict]:
    """The rows of the rule's cell. Raises ValueError when no row has the rule's matcher,
    half-width or string set, so an export from before the rule existed cannot pass for
    a grid with nothing counted."""
    for key, value in rule.items():
        if rows and not any(r[key] == value for r in rows):
            raise ValueError("no %s %r in the export; export again once pilot_counts has it"
                             % (key, value))
    return [r for r in rows if all(r[k] == v for k, v in rule.items())]


def tau_rows(rows_c: List[dict], percents: Iterable[int] = PERCENTS) -> List[dict]:
    """Per index and cutoff: tau over the role c rows of the rule's cell (nearest rank)."""
    out = []
    for index, model in pc.INDEX_LABELS.items():
        rs = [r for r in rows_c if r["index"] == index]
        pos, est = pc._split_stats(rs)
        status = [r["status"] for r in rs]
        for p in percents:
            out.append({
                "index": index, "model": model, "percentile": p,
                "tau": pc._nearest_rank(pos, p), "tau_with_est": pc._nearest_rank(pos + est, p),
                "facts": len(rs), "counted": status.count("counted"), "zero": status.count("zero"),
                "unknown": status.count("unknown"), "docs_above_0": len(pos), "est_mentions": len(est),
                "year_conflict": sum(r["year_conflict"] == "1" for r in rs),
            })
    return out


def merge(shard_dirs: List[str], rule: Dict[str, str] = RULE,
          n_shards: Optional[int] = None) -> dict:
    """{"problems": [...]} when the shards can't merge; otherwise also `rows` (the
    rule's cell from every shard, roles c and h), `tau` (tau_rows) and `meta`."""
    shards = [read_shard(d) for d in shard_dirs]
    metas = [m for m, _ in shards]
    problems = check_mergeable(metas, n_shards)
    rows: List[dict] = []
    for m, rs in shards:
        try:
            rows += select_cell(rs, rule)
        except ValueError as e:
            problems.append("shard %s: %s" % (m.get("shard"), e))
    if problems:
        return {"problems": problems}
    rows.sort(key=lambda r: (r[ROLE_COLUMN], r["prompt_name"], r["index"], r["fact"], int(r["year"])))
    tau = tau_rows([r for r in rows if r[ROLE_COLUMN] == "c"])
    meta = {
        "rule": rule, "shards": sorted(m["shard"] for m in metas),
        "counting_code_sha256": metas[0].get("counting_code_sha256"),
        "claims_sha256": metas[0].get("claims_sha256"), "export_version": metas[0].get("export_version"),
        "rows": {role: sum(r[ROLE_COLUMN] == role for r in rows) for role in ("c", "h")},
        "names": {role: len({r["prompt_name"] for r in rows if r[ROLE_COLUMN] == role}) for role in ("c", "h")},
        "role_h_no_wikidata_year": {k: sum((m.get("role_h_no_wikidata_year") or {}).get(k, 0) for m in metas)
                                    for k in ("claims", "names", "facts")},
        "flagged_rows": {f: sum(r[f] == "1" for r in rows)
                         for f in ("ambiguous_name", "shared_string", "year_conflict", "short_year")},
        "tau_%d" % TAU_P: {t["model"]: t["tau"] for t in tau if t["percentile"] == TAU_P},
        "tau_method": "nearest rank over role c facts with n_c > 0 in docs units "
                      "(pilot_counts._split_stats); tau_with_est adds the sampled-pool estimates",
        "n_h_precision": "unmeasured: the counting rule was hand-checked on correct years (n_c) "
                         "only; role h counts wrong years with it by default",
    }
    return {"problems": [], "rows": rows, "tau": tau, "meta": meta}
