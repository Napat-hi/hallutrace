"""Count correct birth/death years in the infini-gram window cache (pilot preview, offline).

    count   write results/c_pilot/pilot_counts.csv and pilot_summary.md
    packet  write the hand-check packet (outputs/c_pilot_packet.csv, has corpus text),
            its README and results/c_pilot/pilot_packet_strata.csv
    score   read the labelled packet, write results/c_pilot/pilot_precision.md

    .venv\\Scripts\\python.exe scripts\\pilot_counts.py count
    .venv\\Scripts\\python.exe scripts\\pilot_counts.py packet
    .venv\\Scripts\\python.exe scripts\\pilot_counts.py score

Flags: --cache, --prompts, --out, --packet, --names (a JSON list path, or `all`),
--seed, --force.

Makes NO network request and never imports hallutrace.infinigram. The cache may be
written by a running fetch (WAL mode), so it is opened read-only
(file:...?mode=ro) and everything is read inside one transaction. All counting
logic lives in hallutrace/pilot_counts.py; this file only reads SQLite and files
and writes the outputs.

The packet holds copyrighted corpus text: it and its README go next to --packet
(outputs/, gitignored), and the script refuses a --packet under --out or under the
repo's results/. results/ gets numbers, names, years and ids only. Nothing prints
corpus text.

Exit codes: 0 done; 1 refusal or error (message on stderr, no traceback); 2 bad
command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# fetch_names is stdlib only, so it does not pull in hallutrace.infinigram (the HTTP client).
from hallutrace import fetch_names  # noqa: E402
from hallutrace import pilot_counts as pc  # noqa: E402

DEFAULT_PROMPTS = ROOT / "data" / "prompts.csv"
DEFAULT_CACHE = ROOT / "data" / "corpus_cache" / "infinigram.sqlite"
DEFAULT_OUT = ROOT / "results" / "c_pilot"
DEFAULT_PACKET = ROOT / "outputs" / "c_pilot_packet.csv"
DEFAULT_SEED = 20260930
DEFAULT_TASK = "historical_events"

ALL_NAMES = "all"  # --names all: every active prompt name in the cache
LOCK_RETRIES = 5  # retries after "database is locked", LOCK_SLEEP_S apart
LOCK_SLEEP_S = 2
BUSY_TIMEOUT_S = 60
IN_CHUNK = 500  # query strings per IN (...) list

COUNTS_NAME = "pilot_counts.csv"
SUMMARY_NAME = "pilot_summary.md"
STRATA_NAME = "pilot_packet_strata.csv"
PRECISION_NAME = "pilot_precision.md"


class CliError(Exception):
    """An expected failure: printed on stderr, exit code 1, no traceback."""


def say(msg="", err=False):
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


# -- read-only cache access ----------------------------------------------------------------
def _dict_row(cursor, row):
    return {d[0]: v for d, v in zip(cursor.description, row)}


def open_readonly(path):
    """A read-only connection (file:...?mode=ro) that returns rows as dicts named like
    the cache columns. Writes raise sqlite3.OperationalError. Never immutable=1 (that
    ignores the WAL and can read torn data), never a PRAGMA that writes. isolation_level
    is None so the caller issues BEGIN and COMMIT itself."""
    posix = Path(path).resolve().as_posix()
    conn = sqlite3.connect("file:%s?mode=ro" % quote(posix, safe="/:"), uri=True,
                           timeout=BUSY_TIMEOUT_S, isolation_level=None)
    conn.row_factory = _dict_row
    return conn


def _is_locked(exc):
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def load_cache(path, names_arg):
    """Everything the run needs, from one read transaction, as a dict: entities (all
    rows), finds/plan/windows (only the query strings of the selected names), names,
    indexes. Retries a locked database LOCK_RETRIES times, LOCK_SLEEP_S apart."""
    last = None
    for attempt in range(LOCK_RETRIES + 1):
        if attempt:
            say("cache is locked (%s), retry %d of %d in %d s"
                % (last, attempt, LOCK_RETRIES, LOCK_SLEEP_S), err=True)
            time.sleep(LOCK_SLEEP_S)
        try:
            return _read_snapshot(path, names_arg)
        except sqlite3.OperationalError as e:
            if not _is_locked(e):
                raise CliError("cannot read the cache %s: %s" % (path, e))
            last = e
    raise CliError("the cache %s stayed locked after %d retries (%s); is a fetch holding it? "
                   "Try again in a minute." % (path, LOCK_RETRIES, last))


def _read_snapshot(path, names_arg):
    conn = open_readonly(path)
    try:
        conn.execute("BEGIN")
        try:
            data = _read_all(conn, names_arg)
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        conn.execute("COMMIT")
        return data
    finally:
        conn.close()


def _read_all(conn, names_arg):
    entities = conn.execute(
        "SELECT prompt_name, query_string, source, qid, ambiguous_name, ambiguous_stripped, "
        "active FROM entities ORDER BY prompt_name, query_string").fetchall()
    row = conn.execute("SELECT value FROM run_meta WHERE key = 'pilot_names'").fetchone()
    names = resolve_names(names_arg, entities, row["value"] if row else None)
    chosen = set(names)
    # Inactive strings of a selected name are read too: their plan rows keep the
    # windows diagnostics honest. The strings are always bound parameters.
    qstrings = sorted({e["query_string"] for e in entities if e["prompt_name"] in chosen})
    present = [r["index_name"] for r in conn.execute("SELECT DISTINCT index_name FROM finds")]
    indexes = [i for i in pc.INDEX_LABELS if i in present] + sorted(
        i for i in present if i not in pc.INDEX_LABELS)
    tables = {
        "finds": "index_name, query_string, form, cnt, segment_by_shard_json",
        "plan": "index_name, query_string, form, s, rank, selected",
        "windows": ("index_name, query_string, form, s, rank, doc_ix, doc_len, disp_len, "
                    "needle_offset, spans_json, metadata_json"),
    }
    out = {t: [] for t in tables}
    for table, cols in tables.items():
        for index in indexes:
            for i in range(0, len(qstrings), IN_CHUNK):
                chunk = qstrings[i:i + IN_CHUNK]
                marks = ",".join("?" * len(chunk))  # only placeholders are formatted in
                out[table] += conn.execute(
                    "SELECT %s FROM %s WHERE index_name = ? AND query_string IN (%s)"
                    % (cols, table, marks), [index, *chunk]).fetchall()
    out.update(entities=entities, names=names, indexes=indexes)
    return out


def read_names_arg(spec):
    """None (use the stored pilot_names), ALL_NAMES, or the list read from a JSON file."""
    if spec is None:
        return None
    if spec == ALL_NAMES:
        return ALL_NAMES
    try:
        with open(spec, encoding="utf-8") as f:
            names = json.load(f)
    except OSError as e:
        raise CliError("cannot read the --names file %s: %s" % (spec, e))
    except ValueError as e:
        raise CliError("--names file %s is not valid JSON: %s" % (spec, e))
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise CliError("--names file %s must hold a JSON list of prompt names" % spec)
    return list(dict.fromkeys(names))


def resolve_names(names_arg, entities, stored):
    active = sorted({e["prompt_name"] for e in entities if e["active"]})
    if names_arg == ALL_NAMES:
        names = active
    elif names_arg is None:
        if stored is None:
            raise CliError("the cache has no pilot_names in run_meta (the pilot list is not "
                           "set): pass --names FILE or --names all")
        try:
            names = json.loads(stored)
        except ValueError as e:
            raise CliError("run_meta pilot_names is not valid JSON: %s" % e)
        if not isinstance(names, list):
            raise CliError("run_meta pilot_names is not a JSON list")
        names = list(dict.fromkeys(names))
    else:
        names = names_arg
    if not names:
        raise CliError("no prompt names selected")
    known = set(active)
    missing = [n for n in names if n not in known]
    if missing:
        raise CliError("%d selected names have no active query strings in the cache: %s"
                       % (len(missing), _some(missing)))
    return names


def _some(items, n=10):
    shown = ", ".join(repr(i) for i in items[:n])
    return shown + (" ... (%d more)" % (len(items) - n) if len(items) > n else "")


# -- prompts.csv ---------------------------------------------------------------------------
def load_years(prompts_path, task=DEFAULT_TASK):
    """{name: {"birth": [int, ...], "death": [int, ...]}} from task_meta name1/birth1/
    death1 and name2/birth2/death2 of every `task` prompt; every distinct year of both
    slots, so a name given two years by different prompts shows both."""
    fetch_names._raise_csv_limit()
    seen: dict = {}
    try:
        f = open(prompts_path, encoding="utf-8", newline="")
    except OSError as e:
        raise CliError("cannot read the prompts file %s: %s" % (prompts_path, e))
    with f:
        for row in csv.DictReader(f):
            if row.get("task") != task:
                continue
            try:
                meta = json.loads(row["task_meta"])
            except (KeyError, ValueError):
                raise CliError("prompt %s has no readable task_meta" % row.get("prompt_id"))
            for slot in ("1", "2"):
                name = (meta.get("name" + slot) or "").strip()
                if not name:
                    continue
                entry = seen.setdefault(name, {"birth": set(), "death": set()})
                for fact in pc.fact_match.FACTS:
                    try:
                        entry[fact].add(int(meta.get(fact + slot)))
                    except (TypeError, ValueError):
                        pass  # a slot without that year adds nothing
    return {n: {f: sorted(v[f]) for f in v} for n, v in seen.items()}


def check_years(years, names):
    """Every selected name must have at least one year (an error, not a skip)."""
    missing = [n for n in names if not (years.get(n, {}).get("birth") or years.get(n, {}).get("death"))]
    if missing:
        raise CliError("%d selected names have no birth or death year in the prompts file: %s"
                       % (len(missing), _some(missing)))


# -- files ---------------------------------------------------------------------------------
def is_under(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def check_packet_path(packet, out):
    """The packet holds corpus text: never inside --out or the repo's results/."""
    for parent, what in ((out, "the --out directory"), (ROOT / "results", "the repo's results/")):
        if is_under(packet, parent):
            raise CliError("refusing --packet %s: it is inside %s, and the packet holds "
                           "copyrighted corpus text (use a path under outputs/)" % (packet, what))


def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_rows(path, columns, rows, encoding="utf-8", quoting=csv.QUOTE_MINIMAL):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding, newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", quoting=quoting)
        w.writeheader()
        w.writerows(rows)


def read_rows(path):
    """(header, rows) of a CSV; utf-8-sig, so a BOM from Excel or the packet is dropped."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            rows = list(r)
            return list(r.fieldnames or []), rows
    except OSError as e:
        raise CliError("cannot read %s: %s" % (path, e))
    except (UnicodeDecodeError, csv.Error) as e:
        raise CliError("%s is not a readable UTF-8 CSV: %s" % (path, e))


def packet_has_labels(path):
    """True if the packet file holds any non-empty label. A file that cannot be read
    counts as labelled: its labels cannot be ruled out."""
    try:
        header, rows = read_rows(path)
    except CliError:
        return True
    if "label" not in header:
        return True
    return any((r.get("label") or "").strip() for r in rows)


def backup_path(packet):
    packet = Path(packet)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = packet.with_name("%s.%s.bak.csv" % (packet.stem, stamp))
    n = 1
    while dest.exists():  # two runs inside one second
        dest = packet.with_name("%s.%s-%d.bak.csv" % (packet.stem, stamp, n))
        n += 1
    return dest


def readme_path(packet):
    packet = Path(packet)
    stem = "c_pilot_packet" if packet.name == "c_pilot_packet.csv" else packet.stem
    return packet.with_name(stem + "_README.md")


def packet_readme(packet_rows, strata_rows, messages):
    """The labelling guide. Numbers and instructions only, no corpus text."""
    pop: dict = {}
    for s in strata_rows:
        k = (str(s["band"]), int(s["matched_cue"]))
        pop[k] = pop.get(k, 0) + int(s["population"])
    got: dict = {}
    for r in packet_rows:
        k = (str(r["band"]), int(r["matched_cue"]))
        got[k] = got.get(k, 0) + 1
    out = [
        "`correct`: the snippet states THIS person's birth year (fact = birth) or death year "
        "(fact = death) as that fact;",
        "`other_fact`: this person, but the year is about something else (an event, a "
        "publication, the other life date);",
        "`other_person`: the year, or the whole passage, is about someone else;",
        "`unclear`: the snippet does not let you decide. Judge from the snippet only.",
        "",
        "# How to label the pilot packet",
        "",
        "Each row of the packet CSV is one snippet. `[[...]]` marks the name, `<<...>>` the "
        "year being checked; the `fact` and `year` columns say which fact that year is "
        "supposed to be. Fill the `label` column with exactly one of the four labels above, "
        "in lower case. Leave a row empty to skip it. At least %d rows must be labelled "
        "before `score` will run." % pc.MIN_LABELLED,
        "",
        "## Opening and saving in Excel",
        "",
        "- Open the CSV in Excel (double click, or Data > From Text/CSV with file origin "
        "65001: UTF-8). The file starts with a byte order mark, so accents display correctly.",
        "- Save back to the SAME file with File > Save As > format \"CSV UTF-8 "
        "(Comma delimited) (*.csv)\". A plain \"CSV (Comma delimited)\" save loses non-ASCII "
        "characters.",
        "- Only the `label` column may change. Do not edit, sort, add or delete other "
        "columns or rows; `packet_id` ties each label to its row.",
        "- The packet holds copyrighted corpus text: it stays in outputs/, never in results/ "
        "and never in a commit.",
        "",
        "## Rows per stratum",
        "",
        "%d rows in the packet. A stratum is a distance band (characters between the name "
        "and the year) and whether the relation cue matched (matched_cue)." % len(packet_rows),
        "",
        "| band | matched_cue | rows in packet | candidates in population |",
        "|---|---|---|---|",
    ]
    for band in pc.BAND_LABELS:
        for cue in (0, 1):
            out.append("| %s | %d | %d | %d |" % (band, cue, got.get((band, cue), 0),
                                                   pop.get((band, cue), 0)))
    out += ["", "## Quotas", ""]
    out += ["- " + m for m in messages] if messages else ["All quotas met."]
    out.append("")
    return "\n".join(out)


# -- commands ------------------------------------------------------------------------------
def cmd_count(args):
    out = Path(args.out)
    years = load_years(args.prompts)
    data = load_cache(args.cache, read_names_arg(args.names))
    names, indexes = data["names"], data["indexes"]
    check_years(years, names)
    rows = pc.count_grid(data["entities"], data["finds"], data["plan"], data["windows"],
                         years, names, indexes)
    diag = pc.diagnostics(data["plan"], data["windows"])
    summary = pc.summary_markdown(
        rows, pc.truncated_side_share(data["windows"], indexes),
        pc.year_conflict_list(years, names), diag)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / COUNTS_NAME, pc.CSV_COLUMNS, rows)
    write_text(out / SUMMARY_NAME, summary)
    status = {s: sum(1 for r in rows if r["status"] == s) for s in ("counted", "zero", "unknown")}
    say("count: %d rows (%d names, %d indexes) -> %s" % (len(rows), len(names), len(indexes),
                                                         out / COUNTS_NAME))
    say("summary -> %s" % (out / SUMMARY_NAME))
    say("status: counted %(counted)d, zero %(zero)d, unknown %(unknown)d" % status)
    say("coverage: %d plan rows, %d not fetched, %d windows outside the plan"
        % (diag["plan_rows"], diag["plan_unfetched"], diag["extra_windows"]))
    return 0


def cmd_packet(args):
    out, packet = Path(args.out), Path(args.packet)
    check_packet_path(packet, out)
    # Decide about an existing packet before reading anything, so a refusal writes nothing.
    exists = packet.exists()
    if exists and not args.force and packet_has_labels(packet):
        raise CliError("refusing to overwrite %s: it has labels that would be lost. Score it "
                       "with `score`, move it away, or pass --force (the old file is copied "
                       "to a .bak.csv first)." % packet)
    years = load_years(args.prompts)
    data = load_cache(args.cache, read_names_arg(args.names))
    names = data["names"]
    check_years(years, names)
    chosen = set(names)
    entities = [e for e in data["entities"] if e["prompt_name"] in chosen]
    cands = pc.packet_candidates(entities, data["plan"], data["windows"], years, names)
    if not cands:
        raise CliError("no packet candidates: no fetched window has a birth or death year "
                       "within its text (nothing written)")
    packet_rows, strata_rows, messages = pc.select_packet(cands, args.seed)
    backup = None
    if exists and args.force:
        backup = backup_path(packet)
        shutil.copy2(packet, backup)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(packet, pc.PACKET_COLUMNS, packet_rows, encoding="utf-8-sig",
               quoting=csv.QUOTE_ALL)
    write_rows(out / STRATA_NAME, pc.STRATA_COLUMNS, strata_rows)
    readme = readme_path(packet)
    write_text(readme, packet_readme(packet_rows, strata_rows, messages))
    say("packet: %d rows from %d candidates (seed %d) -> %s"
        % (len(packet_rows), len(cands), args.seed, packet))
    if backup:
        say("old packet copied to %s" % backup)
    say("readme -> %s" % readme)
    say("strata (numbers only) -> %s" % (out / STRATA_NAME))
    for m in messages:
        say(m)
    return 0


def cmd_score(args):
    out, packet = Path(args.out), Path(args.packet)
    check_packet_path(packet, out)
    header, rows = read_rows(packet)
    missing = [c for c in pc.PACKET_COLUMNS if c not in header]
    if missing:
        raise CliError("%s lacks columns %s: was the file saved from Excel as CSV UTF-8 with "
                       "its header intact?" % (packet, ", ".join(missing)))
    strata_header, strata = read_rows(out / STRATA_NAME)
    try:
        text = pc.score_labels(rows, strata)
    except (pc.LabelError, pc.TooFewLabels) as e:
        raise CliError(str(e))
    except (KeyError, ValueError, TypeError) as e:
        raise CliError("cannot score %s: %s" % (packet, e))
    write_text(out / PRECISION_NAME, text)
    say("score: %d packet rows -> %s" % (len(rows), out / PRECISION_NAME))
    return 0


COMMANDS = {"count": cmd_count, "packet": cmd_packet, "score": cmd_score}


# -- entry point ---------------------------------------------------------------------------
def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="SQLite window cache, opened read-only (default: %(default)s)")
    common.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS,
                        help="prompts.csv (default: %(default)s)")
    common.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="results directory, created if missing (default: %(default)s)")
    common.add_argument("--packet", type=Path, default=DEFAULT_PACKET,
                        help="hand-check packet, has corpus text; never under --out or results/ "
                             "(default: %(default)s)")
    common.add_argument("--names", default=None, metavar="FILE|all",
                        help="a JSON list of prompt names, or `all` for every active prompt name "
                             "(default: the pilot_names stored in the cache)")
    common.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="packet draw seed (default: %(default)s)")
    common.add_argument("--force", action="store_true",
                        help="packet: overwrite a labelled packet, after copying it to .bak.csv")
    ap = argparse.ArgumentParser(
        description="Pilot counts of correct birth/death years in the window cache. Offline.")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, text in (
        ("count", "write pilot_counts.csv and pilot_summary.md"),
        ("packet", "write the hand-check packet, its README and pilot_packet_strata.csv"),
        ("score", "read the labelled packet, write pilot_precision.md (no cache needed)"),
    ):
        sub.add_parser(name, parents=[common], help=text, description=text)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except CliError as e:
        say("error: %s" % e, err=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
