r"""Biographies corpus counts for one fetch shard. Numbers only, from your own cache.

After your shard's `fetch_windows.py full --task biographies` shows 0 missing:

    .\.venv\Scripts\python.exe scripts\bio_counts.py export --cache <your shard cache>

Writes results/bio_counts/shard_<i>_of_<n>/counts.csv.gz (n_c and n_h grids, role c / h,
gzipped: about 50 MB as plain CSV) and meta.json (what the merge checks). Commit those two
files; never the cache, which holds corpus text. An older plain counts.csv in that folder is
removed, so commit its deletion too. Read-only on the cache, no network. See hallutrace/bio_counts.py.

Once every shard's export is committed:

    .\.venv\Scripts\python.exe scripts\bio_counts.py merge

checks the shards, keeps the counting rule's cell and writes results/bio_counts/merged/: counts.csv
(n_c and n_h per name, index, fact and year), tau.csv (the second tau per model, 5% to
20%) and meta.json. Exit 1, with the reasons, if the shards can't merge.
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace import bio_counts as bc  # noqa: E402
from hallutrace import pilot_counts as pc  # noqa: E402
from scripts import pilot_counts as pcli  # noqa: E402

TASK = "biographies"


def run_meta(cache_path):
    conn = pcli.open_readonly(cache_path)
    try:
        return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM run_meta")}
    finally:
        conn.close()


def git_head():
    try:
        return subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def cmd_export(args):
    with open(ROOT / "data" / "alias_build" / "wikidata_snapshot.json", encoding="utf-8") as f:
        snapshot = json.load(f)
    from scripts.bio_year_claims import own_qids
    return export(args.cache, args.claims, args.out, own_qids(snapshot), snapshot)


def export(cache, claims_path, out_dir, own, snapshot):
    """Count one shard's cache; write counts.csv and meta.json. Returns the exit code."""
    meta_rows = run_meta(cache)
    shard = meta_rows.get("shard:" + TASK)
    names = json.loads(meta_rows.get("task_names:" + TASK) or "null")
    if not names or not shard or shard == "all":
        raise pcli.CliError("the cache has no biographies shard plan (run_meta shard:%s and "
                            "task_names:%s); is this a shard cache?" % (TASK, TASK))
    claims = bc.read_claims(claims_path)

    data = pcli.load_cache(cache, names)
    indexes = data["indexes"]
    yc = bc.years_correct(names, own, snapshot)
    yh = bc.years_hallucinated(claims, names)
    rows = bc.tag(pc.count_grid(data["entities"], data["finds"], data["plan"], data["windows"],
                                yc, sorted(yc), indexes), "c")
    rows += bc.tag(pc.count_grid(data["entities"], data["finds"], data["plan"], data["windows"],
                                 yh, sorted(yh), indexes), "h")
    diag = pc.diagnostics(data["plan"], data["windows"])

    i, n = (int(x) for x in shard.split("/"))
    out = Path(out_dir) / ("shard_%d_of_%d" % (i, n))
    out.mkdir(parents=True, exist_ok=True)
    bc.write_counts(out / bc.COUNTS_GZ, [bc.ROLE_COLUMN] + pc.CSV_COLUMNS, rows)
    stale = out / bc.COUNTS_CSV
    if stale.exists():
        stale.unlink()
        pcli.say("removed the plain %s from an earlier export; commit its deletion" % stale)
    meta = {
        "task": TASK, "shard": shard, "n_shards": n, "names": names,
        "names_with_correct_year": len(yc), "names_with_hallucinated_year": len(yh),
        "role_h_no_wikidata_year": bc.no_reference_year(claims, names),
        "constants": meta_rows.get("constants"), "fetch_script_version": meta_rows.get("script_version"),
        "fetch_git_head": meta_rows.get("git_head"), "plan_time": meta_rows.get("plan_time"),
        "indexes": indexes,
        "completeness": bc.completeness(diag, indexes, list(pc.INDEX_LABELS)),
        "counting_code_sha256": bc.sha256_files([ROOT / "hallutrace" / "pilot_counts.py",
                                                 ROOT / "hallutrace" / "fact_match.py"]),
        "claims_sha256": bc.sha256_files([claims_path]),
        "export_version": bc.EXPORT_VERSION, "export_git_head": git_head(),
        "rows": {"c": sum(1 for r in rows if r[bc.ROLE_COLUMN] == "c"),
                 "h": sum(1 for r in rows if r[bc.ROLE_COLUMN] == "h")},
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    pcli.say("shard %s: %d names, n_c rows %d, n_h rows %d (%d claims with no Wikidata year left out), "
             "complete %s -> %s" % (shard, len(names), meta["rows"]["c"], meta["rows"]["h"],
                                    meta["role_h_no_wikidata_year"]["claims"],
                                    meta["completeness"]["complete"], out))
    if not meta["completeness"]["complete"]:
        pcli.say("NOT complete: %s. Finish the fetch (status shows 0 missing) and export again."
                 % meta["completeness"], err=True)
        return 1
    return 0


TAU_COLUMNS = ["index", "model", "percentile", "tau", "tau_with_est", "facts", "counted", "zero",
               "unknown", "docs_above_0", "est_mentions", "year_conflict"]


def _num(v):
    return "" if v is None else "%g" % v


def merge_dir(in_dir, out_dir, rule, n_shards=None):
    """Merge every shard_<i>_of_<n> export under in_dir; write out_dir. Returns the exit code."""
    dirs = sorted(str(p) for p in Path(in_dir).glob("shard_*_of_*") if p.is_dir())
    res = bc.merge(dirs, rule, n_shards)
    if res["problems"]:
        pcli.say("cannot merge %d shard export(s) from %s:" % (len(dirs), in_dir), err=True)
        for p in res["problems"]:
            pcli.say("  - %s" % p, err=True)
        return 1
    out = Path(out_dir)
    pcli.write_rows(out / "counts.csv", [bc.ROLE_COLUMN] + pc.CSV_COLUMNS, res["rows"])
    pcli.write_rows(out / "tau.csv", TAU_COLUMNS,
                    [dict(t, tau=_num(t["tau"]), tau_with_est=_num(t["tau_with_est"])) for t in res["tau"]])
    meta = dict(res["meta"], merge_git_head=git_head())
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    pcli.say("merged %s, cell %s: n_c rows %d (%d names), n_h rows %d (%d names) -> %s"
             % (", ".join(meta["shards"]), "/".join(rule.values()), meta["rows"]["c"], meta["names"]["c"],
                meta["rows"]["h"], meta["names"]["h"], out))
    for t in res["tau"]:
        if t["percentile"] == bc.TAU_P:
            pcli.say("  tau %d%% %-10s %s (with sampled pools %s; %d facts above 0 in docs, %d sampled, "
                     "%d unknown)" % (bc.TAU_P, t["model"], _num(t["tau"]), _num(t["tau_with_est"]),
                                      t["docs_above_0"], t["est_mentions"], t["unknown"]))
    return 0


def cmd_merge(args):
    rule = {"matcher": args.matcher, "half_width": args.half_width, "strings": args.strings}
    return merge_dir(args.inp, args.out, rule, args.shards)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="count your shard's cache, write numbers only")
    ex.add_argument("--cache", required=True, type=Path, help="your shard's window cache")
    ex.add_argument("--claims", default=str(ROOT / "results" / "bio_year_claims.csv"))
    ex.add_argument("--out", default=str(ROOT / "results" / "bio_counts"))
    mg = sub.add_parser("merge", help="check the shard exports, keep the counting rule's cell, set the second tau")
    mg.add_argument("--in", dest="inp", default=str(ROOT / "results" / "bio_counts"))
    mg.add_argument("--out", default=str(ROOT / "results" / "bio_counts" / "merged"))
    mg.add_argument("--shards", type=int, default=None, help="expected shard count (default: from the exports)")
    mg.add_argument("--matcher", default=bc.RULE["matcher"])
    mg.add_argument("--half-width", default=bc.RULE["half_width"])
    mg.add_argument("--strings", default=bc.RULE["strings"])
    args = ap.parse_args(argv)
    try:
        return cmd_export(args) if args.command == "export" else cmd_merge(args)
    except (pcli.CliError, sqlite3.Error, OSError) as e:
        pcli.say("error: %s" % e, err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
