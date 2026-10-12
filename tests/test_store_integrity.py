"""Audit tests for hallutrace/store.py: NULL and type handling in the conflict
check, transaction boundaries, lock behaviour, idempotency, and the rule that
no insert path may use OR IGNORE. No key, no network.

Run from the repo root: python -m tests.test_store_integrity

Section 6 pins the contract that the loaders own the transaction (BEGIN
IMMEDIATE before the `with conn:` block), so a batch is atomic whatever the
connection's isolation_level and a caller's own work is never committed or
rolled back by a loader.
"""

import ast
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from hallutrace import store

ROOT = Path(__file__).resolve().parent.parent
BATCH = ROOT / "results" / "pilot" / "olmo2-7b" / "olmo2-7b-b00000.jsonl"   # 64 real records
STORE_PY = ROOT / "hallutrace" / "store.py"

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


def raises(exc, fn, label, substr=None):
    try:
        fn()
    except exc as e:
        ok = substr is None or substr in str(e)
        check(ok, label if ok else "%s  (wrong message: %s)" % (label, e))
        return
    except BaseException as e:  # noqa: BLE001
        check(False, "%s  (raised %s instead: %s)" % (label, type(e).__name__, e))
        return
    check(False, label + "  (did not raise)")


tmp = Path(tempfile.mkdtemp(prefix="hallutrace_integrity_"))
_REAL_INSERT = store._insert_idempotent


def new_db(name=None, with_prompts=True):
    conn = store.connect(":memory:" if name is None else str(tmp / name))
    store.init_db(conn)
    if with_prompts:
        store.load_prompts(conn)
    return conn


def n_gen(conn):
    return conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]


def records():
    return [json.loads(line) for line in open(BATCH, encoding="utf-8") if line.strip()]


def write_batch(name, recs):
    out = tmp / name
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return out


class DieAfter:
    """Stands in for _insert_idempotent and raises on the k-th call."""

    def __init__(self, k, exc):
        self.k, self.exc, self.n = k, exc, 0

    def __call__(self, *a, **kw):
        self.n += 1
        if self.n == self.k:
            raise self.exc
        return _REAL_INSERT(*a, **kw)


def generation_values(rec):
    return tuple(int(rec[c]) if c == "escalated" else rec[c] for c in store.GENERATION_COLUMNS)


try:
    print("1. the conflict check is Python-side and NULL-safe")
    src = STORE_PY.read_text(encoding="utf-8")
    body = src[src.index("def _insert_idempotent"):src.index("def load_prompts")]
    check("tuple(existing) != tuple(values)" in body,
          "the comparison is a Python tuple comparison, not SQL '=' (NULL = NULL is NULL)")
    check(not re.search(r"WHERE[^\"']*\b(?:AND|OR)\b[^\"']*=", body),
          "the only SQL comparison is the single-key lookup, on a non-NULL primary key")

    conn = new_db()
    rec = records()[0]
    rec["response"], rec["finish_reason"], rec["n_output_tokens"] = None, "prompt_too_long", 0
    vals = generation_values(rec)
    eq(store._insert_idempotent(conn, "generations", "generation_id",
                                store.GENERATION_COLUMNS, vals), True,
       "a generation whose response is NULL inserts")
    eq(store._insert_idempotent(conn, "generations", "generation_id",
                                store.GENERATION_COLUMNS, vals), False,
       "loading it again is a no-op: NULL against NULL is 'same', not a false conflict")

    changed = dict(rec, response="", finish_reason="stop")
    raises(store.LoadConflict,
           lambda: store._insert_idempotent(conn, "generations", "generation_id",
                                            store.GENERATION_COLUMNS, generation_values(changed)),
           "NULL stored, empty string incoming, is a conflict (not silently equal)", "response")
    changed = dict(rec, response="x", finish_reason="stop")
    raises(store.LoadConflict,
           lambda: store._insert_idempotent(conn, "generations", "generation_id",
                                            store.GENERATION_COLUMNS, generation_values(changed)),
           "NULL stored, a real value incoming, is a conflict (a genuine difference is not missed)",
           "response")

    scratch = new_db(with_prompts=False)
    scratch.execute("CREATE TABLE t (k TEXT PRIMARY KEY, s TEXT, r REAL, i INTEGER)")
    cols = ("k", "s", "r", "i")
    tricky = [
        ("empty string", "", 0.0, 0),
        ("embedded NUL", "a\x00b", 0.1, 1),
        ("CRLF and U+2028", "a\r\nb c", 0.9, -1),
        ("emoji and accents", "café \U0001f600", 1.0, 2**31 - 1),
        ("digit-looking text", "007", 5e-324, 2**63 - 1),
        ("JSON int into REAL", "x", 1, 3),
        ("a 1 MB response", "y" * 1_000_000, 1.7976931348623157e308, 4),
    ]
    bad = []
    for i, (label, s, r, n) in enumerate(tricky):
        row = ("k%d" % i, s, r, n)
        first = store._insert_idempotent(scratch, "t", "k", cols, row)
        second = store._insert_idempotent(scratch, "t", "k", cols, row)
        if not (first is True and second is False):
            bad.append(label)
    check(not bad, "awkward strings and numbers round-trip and re-insert as identical (%d cases)%s"
          % (len(tricky), "" if not bad else ": " + ", ".join(bad)))

    nan_rec = dict(records()[1], temperature=float("nan"))
    raises(store.LoadError,
           lambda: store.load_generations(new_db(), [write_batch("nan.jsonl",
                                                                 [nan_rec] + records()[2:])]),
           "a NaN temperature is refused by the plan check before it could false-conflict "
           "(SQLite stores NaN as NULL)", "temperature")

    print()
    print("2. transactions: one file is one transaction, however the load dies")
    for label, exc in (("RuntimeError", RuntimeError("boom")),
                       ("KeyboardInterrupt", KeyboardInterrupt())):
        conn = new_db(label + ".sqlite")
        store._insert_idempotent = DieAfter(31, exc)
        try:
            store.load_generations(conn, [BATCH])
        except BaseException:  # noqa: BLE001
            pass
        finally:
            store._insert_idempotent = _REAL_INSERT
        eq(n_gen(conn), 0, "%s at row 31 of 64 leaves 0 rows, not 30" % label)
        eq(conn.in_transaction, False, "and leaves no transaction open on the connection")
        eq(store.load_generations(conn, [BATCH]).inserted, 64,
           "a clean rerun then inserts the whole batch")
        conn.close()

    path = tmp / "killed.sqlite"
    conn = store.connect(str(path))
    store.init_db(conn)
    store.load_prompts(conn)
    conn.close()
    child = (
        "import os, sys\n"
        "sys.path.insert(0, %r)\n"
        "os.chdir(%r)\n"
        "from hallutrace import store\n"
        "real = store._insert_idempotent\n"
        "n = [0]\n"
        "def die(*a, **kw):\n"
        "    n[0] += 1\n"
        "    if n[0] == 31:\n"
        "        os._exit(3)\n"
        "    return real(*a, **kw)\n"
        "store._insert_idempotent = die\n"
        "store.load_generations(store.connect(%r), [%r])\n"
    ) % (str(ROOT), str(ROOT), str(path), str(BATCH))
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
    eq(proc.returncode, 3, "the child process died at row 31 with no cleanup (os._exit)")
    conn = store.connect(str(path))
    eq(n_gen(conn), 0, "reopened after the death, the store holds 0 rows of that batch")
    eq(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok", "and passes integrity_check")
    eq(store.load_generations(conn, [BATCH]).inserted, 64, "a rerun after the death inserts all 64")
    conn.close()

    print()
    print("3. locks: what connect() waits for, and what a busy store does to a load")
    conn = store.connect(str(tmp / "t.sqlite"))
    timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    eq(timeout_ms, 30000,
       "connect() waits 30 s for a lock, stated explicitly (Python's own default is 5 s and "
       "SQLite's is 0, so a timeout=0 would slip in silently)")
    eq(store.BUSY_TIMEOUT_MS, 30000, "and store.BUSY_TIMEOUT_MS is the value it uses")
    conn.close()

    path = str(tmp / "busy.sqlite")
    setup = store.connect(path)
    store.init_db(setup)
    store.load_prompts(setup)
    setup.close()
    reader = store.connect(path)
    cur = reader.execute("SELECT prompt_id FROM prompts")
    cur.fetchone()                                    # a read statement left open
    loader = store.connect(path)
    loader.execute("PRAGMA busy_timeout = 100")
    raises(sqlite3.OperationalError, lambda: store.load_generations(loader, [BATCH]),
           "a reader holding an open statement makes the loader fail loudly at commit",
           "database is locked")
    # Close the cursor first: reader.close() alone does not release the read lock
    # while a cursor that read from it is still alive (Python 3.12, SQLite 3.49).
    cur.close()
    reader.close()
    eq(n_gen(store.connect(path)), 0, "and the batch is rolled back, not half stored")
    eq(store.load_generations(loader, [BATCH]).inserted, 64,
       "once the reader lets go the same connection loads the batch")
    loader.close()

    print()
    print("4. idempotency and refusal, tested at batch level")
    conn = new_db()
    first = store.load_generations(conn, [BATCH])
    eq((first.inserted, first.already_present), (64, 0), "first load: 64 inserted, 0 already present")
    second = store.load_generations(conn, [BATCH])
    eq((second.inserted, second.already_present), (0, 64),
       "second load of the same batch: 0 inserted, 64 already present")
    eq(n_gen(conn), 64, "and the table still holds exactly 64 rows")
    third = store.load_generations(conn, [BATCH, BATCH])
    eq((third.inserted, third.already_present), (0, 128), "the same file twice in one call: 0 and 128")

    planned_fields = {
        "seed": lambda r: r["seed"] + 1,
        "batch_id": lambda r: "olmo2-7b-b09999",
        "gen_config_hash": lambda r: "000000000000",
        "temperature": lambda r: 0.5,
        "top_p": lambda r: 0.5,
        "max_new_tokens": lambda r: 64,
        "escalated": lambda r: True,
        "sample_index": lambda r: r["sample_index"] + 1,
        "prompt_id": lambda r: "biographies-99999",
    }
    for field, wrong in planned_fields.items():
        conn = new_db()
        recs = records()
        recs[7][field] = wrong(recs[7])
        raises(store.LoadError,
               lambda: store.load_generations(conn, [write_batch("plan_%s.jsonl" % field, recs)]),
               "one record differing from the plan in `%s` refuses the whole batch" % field)
        eq(n_gen(conn), 0, "  and stores none of its 64 rows")

    conn = new_db()
    recs = records()
    del recs[3]["model_revision"]
    raises(store.LoadError,
           lambda: store.load_generations(conn, [write_batch("norev.jsonl", recs)]),
           "layer 1, the loader: a record with no model_revision is refused",
           "missing field(s): model_revision")

    values = list(generation_values(records()[3]))
    values[store.GENERATION_COLUMNS.index("model_revision")] = None
    raises(sqlite3.IntegrityError,
           lambda: _REAL_INSERT(conn, "generations", "generation_id",
                                store.GENERATION_COLUMNS, tuple(values)),
           "layer 2, the schema: with the loader bypassed, NOT NULL still refuses it",
           "NOT NULL constraint failed: generations.model_revision")
    cur = conn.execute(
        "INSERT OR IGNORE INTO generations (%s) VALUES (%s)"
        % (", ".join(store.GENERATION_COLUMNS), ", ".join("?" for _ in store.GENERATION_COLUMNS)),
        tuple(values),
    )
    check(cur.rowcount == 0 and n_gen(conn) == 0,
          "contrast: INSERT OR IGNORE swallows that same row without a word, which is exactly "
          "why the loader must never use it")
    conn.rollback()

    print()
    print("5. no insert path uses OR IGNORE or OR REPLACE")
    tree = ast.parse(STORE_PY.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            d = ast.get_docstring(node, clean=False)
            if d is not None:
                docstrings.add(d)
    offenders, inserts = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if text in docstrings or text.strip() in {d.strip() for d in docstrings}:
                continue
            if re.search(r"\bOR\s+(IGNORE|REPLACE)\b", text, re.I):
                offenders.append(text[:60])
            if re.search(r"\bINSERT\b", text, re.I) and "CREATE" not in text.upper():
                inserts.append(text)
    check(not offenders, "no SQL string in store.py says OR IGNORE or OR REPLACE%s"
          % ("" if not offenders else ": %r" % offenders))
    check(all("ON CONFLICT" in s and "DO NOTHING" in s for s in inserts) and len(inserts) == 1,
          "the single INSERT in store.py is ON CONFLICT ... DO NOTHING (%d found)" % len(inserts))

    print()
    print("6. the loaders own the transaction: a contract, tested from both sides")
    # 6a. atomicity no longer depends on the connection's isolation_level
    conn = new_db("iso_none.sqlite")
    conn.isolation_level = None
    store._insert_idempotent = DieAfter(31, RuntimeError("boom"))
    try:
        store.load_generations(conn, [BATCH])
    except RuntimeError:
        pass
    finally:
        store._insert_idempotent = _REAL_INSERT
    eq(n_gen(conn), 0,
       "under isolation_level=None, a failure at row 31 of 64 leaves 0 rows (it used to leave 30)")
    eq(conn.in_transaction, False, "  and leaves no transaction open")
    eq(store.load_generations(conn, [BATCH]).inserted, 64, "  and a rerun inserts the whole batch")
    conn.close()

    # 6b. a caller's open transaction is refused, and left exactly as it was
    for loader_name in ("load_generations", "load_prompts"):
        conn = new_db("open_%s.sqlite" % loader_name, with_prompts=(loader_name == "load_generations"))
        conn.execute("CREATE TABLE callers_scratch (x)")
        conn.commit()
        conn.execute("INSERT INTO callers_scratch VALUES ('callers pending row')")
        eq(conn.in_transaction, True, "%s: the caller has uncommitted work open" % loader_name)
        call = ((lambda: store.load_generations(conn, [BATCH])) if loader_name == "load_generations"
                else (lambda: store.load_prompts(conn)))
        raises(store.LoadError, call,
               "%s refuses to run on a connection with a transaction open" % loader_name,
               "no transaction open")
        eq(conn.in_transaction, True, "  the refusal did not roll the caller's transaction back")
        eq(conn.execute("SELECT COUNT(*) FROM callers_scratch").fetchone()[0], 1,
           "  the caller's pending row is still there")
        eq(n_gen(conn) + conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0],
           3200 if loader_name == "load_generations" else 0,
           "  and the loader stored nothing")
        conn.commit()
        if loader_name == "load_generations":
            eq(store.load_generations(conn, [BATCH]).inserted, 64,
               "  once the caller commits, the same call succeeds")
        else:
            eq(store.load_prompts(conn).inserted, 3200, "  once the caller commits, the same call succeeds")
        eq(conn.execute("SELECT COUNT(*) FROM callers_scratch").fetchone()[0], 1,
           "  and the caller's row was committed by the caller, not swallowed")
        conn.close()

    # 6c. Python 3.12's autocommit modes (the parameter does not exist before 3.12)
    if sys.version_info >= (3, 12):
        for mode in (True, False):
            c = sqlite3.connect(str(tmp / ("ac_%s.sqlite" % mode)), autocommit=mode)
            c.execute("PRAGMA foreign_keys = ON")
            store.init_db(c)
            raises(store.LoadError, lambda: store.load_generations(c, [BATCH]),
                   "autocommit=%s is refused" % mode,
                   "autocommit=True" if mode else "no transaction open")
            if mode is True:
                eq(c.in_transaction, False,
                   "  and no transaction is left open (autocommit=True would otherwise leave the "
                   "batch uncommitted and lose it on close, with no error)")
            c.close()
    else:
        print("   --   autocommit modes skipped: sqlite3.connect(autocommit=) needs Python 3.12+")

    # 6d. the write lock is taken up front, so a blocked loader does no work
    path = str(tmp / "upfront.sqlite")
    setup = store.connect(path)
    store.init_db(setup)
    store.load_prompts(setup)
    setup.close()
    holder = store.connect(path)
    holder.execute("BEGIN IMMEDIATE")
    blocked = store.connect(path)
    blocked.execute("PRAGMA busy_timeout = 100")
    calls = [0]

    def counting(*a, **kw):
        calls[0] += 1
        return _REAL_INSERT(*a, **kw)

    store._insert_idempotent = counting
    try:
        raises(sqlite3.OperationalError, lambda: store.load_generations(blocked, [BATCH]),
               "a second loader blocked by a writer fails with 'database is locked'",
               "database is locked")
    finally:
        store._insert_idempotent = _REAL_INSERT
    eq(calls[0], 0,
       "  and did so before inserting a single row: the lock is taken at the start of the "
       "batch, not discovered at commit")
    holder.rollback()
    holder.close()
    eq(store.load_generations(blocked, [BATCH]).inserted, 64,
       "  once the writer lets go, the blocked connection loads the batch")
    blocked.close()

    # 6e. records SQLite cannot store become LoadErrors with a file and a line
    def load_bad(name, mutate, expect):
        conn = new_db()
        recs = records()
        mutate(recs[7])                                  # record 8 of 64, so line 8
        out = write_batch(name, recs)
        try:
            store.load_generations(conn, [out])
        except store.LoadError as e:
            msg = str(e)
            check(expect in msg and name in msg and "line 8" in msg,
                  "%s -> %s" % (expect, msg.split(name, 1)[-1].lstrip(" :")[:90]))
        except BaseException as e:  # noqa: BLE001
            check(False, "%s raised %s instead of LoadError: %s" % (expect, type(e).__name__, e))
        else:
            check(False, "%s was not refused" % expect)
        eq(n_gen(conn), 0, "  and none of the batch's 64 rows were stored")

    load_bad("surrogate.jsonl", lambda r: r.update(response="abc\ud800def"),
             "response cannot be stored: lone surrogate U+D800 at offset 3")
    load_bad("surrogate_other.jsonl", lambda r: r.update(engine="vllm\udfff"),
             "engine cannot be stored: lone surrogate U+DFFF")
    load_bad("big.jsonl", lambda r: r.update(n_output_tokens=2**63),
             "n_output_tokens cannot be stored: integer 9223372036854775808 is outside")
    load_bad("small.jsonl", lambda r: r.update(n_output_tokens=-(2**63) - 1),
             "n_output_tokens cannot be stored: integer -9223372036854775809 is outside")

    for label, value in (("2**63 - 1", 2**63 - 1), ("-2**63", -(2**63))):
        conn = new_db()
        recs = records()
        recs[7]["n_output_tokens"] = value
        store.load_generations(conn, [write_batch("edge.jsonl", recs)])
        got = conn.execute("SELECT n_output_tokens FROM generations WHERE generation_id = ?",
                           (recs[7]["generation_id"],)).fetchone()[0]
        eq(got, value, "the boundary value %s loads and reads back exactly" % label)

finally:
    store._insert_idempotent = _REAL_INSERT
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
