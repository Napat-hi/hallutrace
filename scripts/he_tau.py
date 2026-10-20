r"""historical_events tau with the counting rule. Numbers and names only.

    .\.venv\Scripts\python.exe scripts\he_tau.py --cache <path to infinigram.sqlite>

Counts every historical_events name once (`pilot_counts.count_grid`), keeps the counting
rule's cell (near_or_cue / full / no_stripped) and writes results/slot_types/: he_counts.csv
(the cell, with short_form and in_headline), he_tau.csv (tau and tau_with_est per variant, model and cutoff,
5% to 20%), he_tau_summary.md and meta.json. Offline, read-only on the cache, no corpus text
in any output. Exit 0 done; 1 refusal or error (message on stderr, no traceback); 2 bad
command line. Refuses when the out dir already has outputs (pass --force to overwrite) and
when the name sets are not the expected sizes. See hallutrace/he_tau.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace import bio_counts as bc  # noqa: E402
from hallutrace import he_tau as ct  # noqa: E402
from hallutrace import pilot_counts as pc  # noqa: E402
from scripts import pilot_counts as pcli  # noqa: E402
from scripts.bio_counts import git_head  # noqa: E402

DEFAULT_CACHE = pcli.DEFAULT_CACHE
DEFAULT_PROMPTS = pcli.DEFAULT_PROMPTS
DEFAULT_OUT = ROOT / "results" / "slot_types"
SCRIPT_VERSION = "1"
COUNTS_NAME = "he_counts.csv"
TAU_NAME = "he_tau.csv"
SUMMARY_NAME = "he_tau_summary.md"
META_NAME = "meta.json"
OUTPUTS = (COUNTS_NAME, TAU_NAME, SUMMARY_NAME, META_NAME)
HASH_CHUNK = 1 << 20  # 1 MiB
DECISIONS = ["tau: nearest-rank percentile over facts with n_c above 0",
             "facts: the correct facts (birth and death years)",
             "counting: near_or_cue matcher, full window, no stripped short forms",
             "headline without the 38 short-form names; the 141 ambiguous names kept",
             "an unknown fact (a sampled pool with no match) enters neither tau"]


def _num(v):
    return "" if v is None else "%g" % v


def sizes_arg(text):
    """argparse type: "ALL,SHORT,AMBIGUOUS" -> (int, int, int)."""
    try:
        a, s, m = (int(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected three integers ALL,SHORT,AMBIGUOUS, got %r" % text)
    return a, s, m


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_parser():
    exp = ct.EXPECTED_SIZES
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="the window cache (opened read-only)")
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs in --out")
    ap.add_argument("--expect-sizes", type=sizes_arg, default=(exp["all"], exp["short"], exp["ambiguous"]),
                    metavar="ALL,SHORT,AMBIGUOUS", help="expected name set sizes (overlap must be 0)")
    return ap


def run(args):
    """Returns the exit code; raises CliError, sqlite3.Error, OSError or ValueError."""
    t0 = time.time()
    out = Path(args.out)
    present = [n for n in OUTPUTS if (out / n).exists()]
    if present and not args.force:
        raise pcli.CliError("%s already has %s; pass --force to overwrite" % (out, ", ".join(present)))

    years = pcli.load_years(args.prompts)
    data = pcli.load_cache(args.cache, pcli.ALL_NAMES)

    a, s, m = args.expect_sizes
    expected = {"all": a, "short": s, "ambiguous": m, "overlap": 0}
    sets = ct.name_sets(data["entities"])
    problems = ct.check_sizes(sets, expected)
    if problems:
        pcli.say("name set sizes: %s" % ", ".join("%s %d" % kv for kv in ct.set_sizes(sets).items()), err=True)
        raise pcli.CliError("name sets are not as expected: %s" % "; ".join(problems))
    if sorted(data["names"]) != sets["all"]:
        raise pcli.CliError("the names counted are not the active prompt names of the cache")
    pcli.check_years(years, data["names"])

    rows = pc.count_grid(data["entities"], data["finds"], data["plan"], data["windows"],
                         years, data["names"], data["indexes"])
    cell = bc.select_cell(rows, bc.RULE)
    counts = ct.annotate(cell, sets)
    tau = ct.tau_table(cell, sets)

    st = os.stat(args.cache)
    written_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cache_path = str(Path(args.cache).resolve())
    head = git_head()
    cache_sha = sha256_file(args.cache)
    run_seconds = round(time.time() - t0, 1)
    provenance = {"cache_path": cache_path, "cache_size_bytes": st.st_size, "cache_sha256": cache_sha,
                  "git_head": head, "script_version": SCRIPT_VERSION, "written_at": written_at,
                  "run_seconds": run_seconds, "cell_rows": len(cell)}
    sizes = ct.set_sizes(sets)
    meta = {
        "cache_path": cache_path, "cache_size_bytes": st.st_size, "cache_sha256": cache_sha,
        "cache_mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "git_head": head, "script_version": SCRIPT_VERSION, "rule": bc.RULE,
        "percents": list(bc.PERCENTS), "tau_percentile": ct.TAU_P, "near": pc.NEAR,
        "cap_full": pc.CAP_FULL, "set_sizes": sizes, "short_names": sets["short"],
        "variants": list(ct.VARIANTS), "headline_variant": ct.HEADLINE,
        "headline_definition": "open: tau or tau_with_est, not chosen",
        "cell_rows": len(cell), "grid_rows": len(rows), "names": len(data["names"]),
        "indexes": data["indexes"], "diagnostics": pc.diagnostics(data["plan"], data["windows"]),
        "decisions": DECISIONS, "run_seconds": run_seconds, "written_at": written_at,
    }

    summary = ct.summary_markdown(tau, sets, provenance)  # built before any file is written
    out.mkdir(parents=True, exist_ok=True)
    pcli.write_rows(out / COUNTS_NAME, ct.COUNTS_COLUMNS, counts)
    pcli.write_rows(out / TAU_NAME, ct.TAU_COLUMNS,
                    [dict(t, tau=_num(t["tau"]), tau_with_est=_num(t["tau_with_est"])) for t in tau])
    pcli.write_text(out / SUMMARY_NAME, summary)
    with open(out / META_NAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    pcli.say("name sets: %s" % ", ".join("%s %d" % kv for kv in sizes.items()))
    pcli.say("cell %s: %d rows (grid %d) -> %s" % ("/".join(bc.RULE.values()), len(cell), len(rows), out))
    for t in tau:
        if t["variant"] == ct.HEADLINE and t["percentile"] == ct.TAU_P:
            pcli.say("  %s tau %d%% %-10s tau %s, with sampled pools %s; docs_above_0 %d, est_mentions %d, "
                     "unknown %d" % (ct.HEADLINE, ct.TAU_P, t["model"], _num(t["tau"]),
                                     _num(t["tau_with_est"]), t["docs_above_0"], t["est_mentions"],
                                     t["unknown"]))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (pcli.CliError, sqlite3.Error, OSError, ValueError) as e:
        pcli.say("error: %s" % e, err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
