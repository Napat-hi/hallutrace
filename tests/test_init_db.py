"""Tests for scripts/init_db.py's PRAGMA user_version guard.

No key, no network, temp files only. Run from the repo root:

    python -m tests.test_init_db

A column comparison cannot see a CHECK that got stricter, so build() checks
PRAGMA user_version on an existing store before init() runs; afterwards,
init_db() would already have stamped the current version onto it.
"""

import sqlite3
import sys
import tempfile
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace import store  # noqa: E402

_spec = importlib.util.spec_from_file_location("init_db", ROOT / "scripts" / "init_db.py")
idb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(idb)

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def version_of(path) -> int:
    return sqlite3.connect(str(path)).execute("PRAGMA user_version").fetchone()[0]


print("1. a store behind the current schema version is refused, not stamped forward")
tmpdir = Path(tempfile.mkdtemp(prefix="hallutrace_init_db_"))
path = tmpdir / "store.sqlite"

idb.build("main store", path, store.connect, store.init_db, schema_version=store.SCHEMA_VERSION)
conn = sqlite3.connect(str(path))
conn.execute("PRAGMA user_version = 1")
conn.commit()
conn.close()
check(version_of(path) == 1, "the store's version was rolled back to 1 for this test")

raised = None
try:
    idb.build("main store", path, store.connect, store.init_db, schema_version=store.SCHEMA_VERSION)
except SystemExit as e:
    raised = e
check(raised is not None, "build() refuses a store behind the current schema version")
message = str(raised) if raised else ""
check("schema version" in message and "rebuild" in message.lower(),
      "the refusal names the schema version and says to rebuild")
check(version_of(path) == 1,
      "the refusal happens before init() runs: version is still 1, not stamped forward")

print()
print("2. a fresh build ends at the current schema version")
tmpdir2 = Path(tempfile.mkdtemp(prefix="hallutrace_init_db_fresh_"))
path2 = tmpdir2 / "store.sqlite"

idb.build("main store", path2, store.connect, store.init_db, schema_version=store.SCHEMA_VERSION)
check(version_of(path2) == store.SCHEMA_VERSION,
      "a brand new store is stamped at the current SCHEMA_VERSION")

idb.build("main store", path2, store.connect, store.init_db, schema_version=store.SCHEMA_VERSION)
check(version_of(path2) == store.SCHEMA_VERSION,
      "running it again on an up-to-date store changes nothing")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
