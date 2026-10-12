"""Create the two SQLite files from scratch: the main store and the API cache.

    db/hallutrace.sqlite   the nine tables, with foreign keys, CHECK constraints
                           and indexes. Regenerable.
    db/api_cache.sqlite    the two API cache tables (Together, Semantic Scholar).
                           Not regenerable.

Usage (paths default to the repo's db/ directory): python scripts/init_db.py
"""

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace import cache, store  # noqa: E402

EPILOG = """\
What this will NOT do:
  * It never drops or overwrites anything. Every statement is CREATE ... IF NOT
    EXISTS, plus ADD COLUMN for a column added after the store was built
    (store.ADDED_COLUMNS), so running it against a store that already holds data
    adds only what is missing and leaves every existing row alone. An added
    column is NULL on the rows already there. Running it twice is a no-op.
    A change that cannot be made this way stops with a message telling you to
    rebuild; it never leaves a half-migrated store.
  * It has no reset option. To rebuild the main store from scratch, delete
    db/hallutrace.sqlite yourself, run this, then reload prompts and
    generations. That is safe: everything in it can be regenerated.
  * Never delete db/api_cache.sqlite to "start clean". It holds paid API answers
    that cannot be bought back once an endpoint retires (D-013).
  * It loads no data. Loading is store.load_prompts / store.load_generations.
"""


def table_names(conn: sqlite3.Connection) -> set:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {r[0] for r in rows}


def expected_schema(init) -> dict:
    """{table: {columns}} as init() builds it from scratch, in memory.

    An existing store is checked against this reference, so a store init()
    could not bring up to date is caught here rather than later.
    """
    scratch = sqlite3.connect(":memory:")
    init(scratch)
    schema = {
        t: {r[1] for r in scratch.execute('PRAGMA table_info("%s")' % t)}
        for t in table_names(scratch)
    }
    scratch.close()
    return schema


def build(label: str, path: Path, connect, init, schema_version=None) -> None:
    existed = path.exists()
    conn = connect(str(path))
    try:
        # a column comparison cannot see a stricter CHECK, so an existing store's
        # version is checked before init() stamps the current one onto it
        if schema_version is not None and existed:
            current_version = conn.execute("PRAGMA user_version").fetchone()[0]
            if table_names(conn) and current_version < schema_version:
                raise SystemExit(
                    "%s: %s is schema version %d, behind the current version %d. "
                    "This store predates a constraint change that cannot be made in "
                    "place (SQLite cannot alter an existing CHECK); rebuild it (delete "
                    "%s, run this again, then reload prompts and generations)."
                    % (label, path, current_version, schema_version, path)
                )
        # store.init_db() returns the columns it added to an older store;
        # cache.init_cache_db() returns None.
        added = init(conn) or []
        have = table_names(conn)
        want = expected_schema(init)
        missing = set(want) - have
        if missing:
            raise SystemExit("%s: still missing tables after init: %s" % (label, sorted(missing)))
        missing_columns = sorted(
            "%s.%s" % (t, c)
            for t, columns in want.items()
            for c in columns - {r[1] for r in conn.execute('PRAGMA table_info("%s")' % t)}
        )
        if missing_columns:
            raise SystemExit(
                "%s: %s still lacks %s after init. This store predates a column that "
                "cannot be added in place; rebuild it (delete %s, run this again, then "
                "reload prompts and generations)."
                % (label, path, ", ".join(missing_columns), path)
            )
        rows = {t: conn.execute('SELECT COUNT(*) FROM "%s"' % t).fetchone()[0] for t in sorted(want)}
    finally:
        conn.close()

    state = "already existed, anything missing was added, nothing dropped" if existed else "created"
    print("%s  %s  (%s)" % (label, path, state))
    print("  %d tables: %s" % (len(want), ", ".join(sorted(want))))
    if added:
        print("  columns added to the existing store: %s" % ", ".join(added))
    filled = {t: n for t, n in rows.items() if n}
    print("  rows already present: %s" % (filled if filled else "none"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Create db/hallutrace.sqlite (nine tables, foreign keys, indexes) and "
            "db/api_cache.sqlite (two cache tables). Safe to run repeatedly: it only "
            "creates what is missing."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", type=Path, default=ROOT / "db" / "hallutrace.sqlite",
                    help="main store (default: %(default)s)")
    ap.add_argument("--cache", type=Path, default=ROOT / "db" / "api_cache.sqlite",
                    help="API cache (default: %(default)s)")
    args = ap.parse_args(argv)

    build("main store", args.db, store.connect, store.init_db,
          schema_version=store.SCHEMA_VERSION)
    build("api cache ", args.cache, cache.connect_cache, cache.init_cache_db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
