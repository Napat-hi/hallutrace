"""Tests for hallutrace/store.py. No key, no network, in-memory SQLite only.

1. Foreign keys are enforced.
2. The decomposer_empty_reason CHECK constraints reject bad rows in both
   directions, including a reason on a row with NULL n_units.
3. stripped_tail keeps three states apart: NULL not yet processed, '' processed
   with nothing to drop, non-empty the dropped text.
4. A store built before stripped_tail existed gains the column, with its CHECK.
5. generations.n_units_kept cannot exceed n_units, and units.dropped_reason
   accepts only NULL or one of posthoc.py's four reasons.
"""

import sqlite3
import sys

from hallutrace import store

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def expect_integrity_error(fn, label):
    try:
        fn()
    except sqlite3.IntegrityError:
        check(True, label)
        return
    except Exception as e:  # noqa: BLE001
        check(False, label + "  (raised %s instead: %s)" % (type(e).__name__, e))
        return
    check(False, label + "  (did not raise)")


def fresh_db():
    conn = store.connect(":memory:")
    store.init_db(conn)
    return conn


def insert_prompt(conn, prompt_id="biographies-00142"):
    conn.execute(
        "INSERT INTO prompts (prompt_id, task, source_file, source_row, prompt, task_meta) "
        "VALUES (?, 'biographies', 'prompts_biographies.csv', 142, 'Tell me a bio.', '{}')",
        (prompt_id,),
    )
    conn.commit()


_gen_seq = [0]


# stripped_tail defaults to _OMIT, "do not name the column", so the helper also
# works on the older schema in section 4; None writes NULL explicitly.
_OMIT = object()


def insert_generation(conn, prompt_id="biographies-00142", n_units=None,
                       decomposer_empty_reason=None, stripped_tail=_OMIT,
                       n_units_kept=_OMIT):
    _gen_seq[0] += 1
    generation_id = "olmo2-7b|%s|%d" % (prompt_id, _gen_seq[0])
    columns = [
        "generation_id", "prompt_id", "model", "sample_index", "temperature",
        "top_p", "seed", "escalated", "response", "abstained", "n_units",
        "decomposer_empty_reason", "generated_at", "gen_config_hash", "batch_id",
        "finish_reason", "n_output_tokens", "max_new_tokens", "n_prompt_tokens",
        "engine", "gpu", "model_revision",
    ]
    values = [
        generation_id, prompt_id, "olmo2-7b", 0, 0.0, 1.0, 1234, 0, "a response",
        0, n_units, decomposer_empty_reason, "2026-09-18T00:00:00Z",
        "abc123def456", "b00000", "stop", 12, 128, 20, "vllm-0.11.0",
        "NVIDIA H100 80GB HBM", "deadbeef",
    ]
    if stripped_tail is not _OMIT:
        columns.append("stripped_tail")
        values.append(stripped_tail)
    if n_units_kept is not _OMIT:
        columns.append("n_units_kept")
        values.append(n_units_kept)
    conn.execute(
        "INSERT INTO generations (%s) VALUES (%s)"
        % (", ".join(columns), ", ".join("?" for _ in columns)),
        values,
    )
    conn.commit()
    return generation_id


print("1. foreign keys are actually enforced")
conn = fresh_db()
check(store.foreign_keys_enabled(conn),
      "PRAGMA foreign_keys reports ON for a connection opened through connect()")

expect_integrity_error(
    lambda: insert_generation(conn, prompt_id="does-not-exist"),
    "a generation whose prompt_id has no matching prompts row is rejected",
)

insert_prompt(conn)
gen_id = insert_generation(conn)
check(
    conn.execute("SELECT 1 FROM generations WHERE generation_id = ?", (gen_id,)).fetchone()
    is not None,
    "a generation with a valid prompt_id is accepted",
)

expect_integrity_error(
    lambda: conn.execute(
        "INSERT INTO units (unit_id, generation_id, unit_index, unit_text) "
        "VALUES ('u1', 'no-such-generation', 0, 'text')"
    ),
    "a unit whose generation_id has no matching generations row is rejected",
)

print()
print("2. decomposer_empty_reason: the CHECK constraint refuses bad rows")
conn = fresh_db()
insert_prompt(conn)

expect_integrity_error(
    lambda: insert_generation(conn, n_units=0, decomposer_empty_reason=None),
    "n_units = 0 with no reason is rejected",
)
expect_integrity_error(
    lambda: insert_generation(conn, n_units=5, decomposer_empty_reason="no_units_found"),
    "n_units > 0 with a reason is rejected",
)
expect_integrity_error(
    lambda: insert_generation(conn, n_units=0, decomposer_empty_reason="ran_no_claims"),
    "the old spelling ran_no_claims is rejected; the reason is no_units_found",
)
expect_integrity_error(
    lambda: insert_generation(conn, n_units=None, decomposer_empty_reason="abstained"),
    "n_units null with a reason is rejected: the original three-branch "
    "CHECK alone would pass this (it evaluates to NULL), so a second CHECK "
    "catches it",
)

# and the reverse: the constraint lets correct rows through
insert_generation(conn, n_units=None, decomposer_empty_reason=None)
check(True, "not-yet-decomposed (both null) is accepted")
insert_generation(conn, n_units=0, decomposer_empty_reason="empty_response")
check(True, "n_units = 0 with a reason is accepted")
insert_generation(conn, n_units=12, decomposer_empty_reason=None)
check(True, "n_units > 0 with no reason is accepted")
for reason in ("empty_response", "abstained", "no_units_found", "failed_parse", "failed_api"):
    insert_generation(conn, n_units=0, decomposer_empty_reason=reason)
check(True, "all five reasons are accepted")

print()
print("3. stripped_tail: three states, and the CHECK that keeps them honest")
conn = fresh_db()
insert_prompt(conn)


def stored_tail(conn, generation_id):
    return conn.execute(
        "SELECT stripped_tail FROM generations WHERE generation_id = ?",
        (generation_id,),
    ).fetchone()[0]


def count_where(conn, where):
    return conn.execute("SELECT COUNT(*) FROM generations WHERE " + where).fetchone()[0]


# the three states, each accepted
not_processed = insert_generation(conn)
check(True, "not yet processed: the column left unset, so NULL, is accepted")
nothing_stripped = insert_generation(conn, n_units=12, stripped_tail="")
check(True, "processed, nothing stripped: '' with n_units > 0 is accepted")
insert_generation(conn, n_units=9, stripped_tail=" and in 1842 she")
check(True, "processed, something stripped: the dropped text is accepted")

# and the three really are three: '' has to survive as distinct from NULL
check(stored_tail(conn, not_processed) is None,
      "the not-yet-processed row reads back as NULL")
check(stored_tail(conn, nothing_stripped) == "",
      "the nothing-stripped row reads back as '', not NULL")
check(count_where(conn, "stripped_tail IS NULL") == 1,
      "IS NULL selects only the not-yet-processed row")
check(count_where(conn, "stripped_tail = ''") == 1,
      "= '' selects only the processed-but-nothing-stripped row")
check(count_where(conn, "stripped_tail IS NOT NULL") == 2,
      "IS NOT NULL means processed: the '' row and the stripped row, not the NULL one")

# the CHECK: no strip result of any kind before decomposition ran
expect_integrity_error(
    lambda: insert_generation(conn, n_units=None, stripped_tail="a dropped clause"),
    "stripped text with n_units still null is rejected",
)
expect_integrity_error(
    lambda: insert_generation(conn, n_units=None, stripped_tail=""),
    "even '' with n_units still null is rejected: '' is a result, not an absence",
)
expect_integrity_error(
    lambda: conn.execute(
        "UPDATE generations SET stripped_tail = 'late' WHERE generation_id = ?",
        (not_processed,),
    ),
    "an UPDATE that strips an undecomposed row is rejected too, not only an INSERT",
)

# the converse is deliberately not enforced: see the comment in store.py
insert_generation(conn, n_units=0, decomposer_empty_reason="empty_response")
check(True, "decomposed with no strip result (stripped_tail NULL) is still allowed")

# the rows the strip emptied, as a query finds them
emptied = insert_generation(
    conn, n_units=0, decomposer_empty_reason="no_units_found",
    stripped_tail="She was the daughter of Lord",
)
found = [
    r[0] for r in conn.execute(
        "SELECT generation_id FROM generations "
        "WHERE stripped_tail IS NOT NULL AND stripped_tail <> '' "
        "AND n_units = 0 AND decomposer_empty_reason = 'no_units_found'"
    )
]
check(found == [emptied],
      "the emptied-by-the-strip case is one query away, and matches only that row")

print()
print("4. an older store gains stripped_tail and n_units_kept in place, CHECK and all")
# The older schema is store.py's own with both ADDED_COLUMNS columns removed, so
# it cannot drift from what store.py declares. Both go together: columns were
# only ever added in that order.
_schema_lines = store.SCHEMA_SQL.splitlines()
_strip_start = next(
    i for i, line in enumerate(_schema_lines) if line.lstrip().startswith("stripped_tail")
)
_strip_end = next(
    i for i, line in enumerate(_schema_lines)
    if i > _strip_start and line.lstrip().startswith("-- the five permitted reasons")
)
old_schema = "\n".join(_schema_lines[:_strip_start] + _schema_lines[_strip_end:])
check(
    len(old_schema.splitlines()) == len(_schema_lines) - (_strip_end - _strip_start),
    "the older schema is the current one minus exactly the stripped_tail and "
    "n_units_kept lines",
)
conn = store.connect(":memory:")
conn.executescript(old_schema)
conn.commit()
check("stripped_tail" not in store.column_names(conn, "generations"),
      "the older store starts without the column")
check("n_units_kept" not in store.column_names(conn, "generations"),
      "nor the column added after it")
insert_prompt(conn)
legacy = insert_generation(conn, n_units=7)

added = store.init_db(conn)
check(added == ["generations.stripped_tail", "generations.n_units_kept"],
      "init_db reports the columns it added, in order")
check("stripped_tail" in store.column_names(conn, "generations"),
      "the older store now has stripped_tail")
check("n_units_kept" in store.column_names(conn, "generations"),
      "and n_units_kept")
check(stored_tail(conn, legacy) is None,
      "the row already there reads NULL: not processed, not 'nothing stripped'")
check(
    conn.execute("SELECT n_units FROM generations WHERE generation_id = ?",
                 (legacy,)).fetchone()[0] == 7,
    "and it kept its other values: the migration adds, it does not rewrite",
)
expect_integrity_error(
    lambda: insert_generation(conn, n_units=None, stripped_tail="x"),
    "the migrated table enforces the CHECK, same as a freshly created one",
)


def table_info(conn, table):
    return [tuple(r) for r in conn.execute('PRAGMA table_info("%s")' % table)]


check(
    table_info(conn, "generations") == table_info(fresh_db(), "generations"),
    "migrated and freshly created generations have the same columns, types and order",
)
insert_generation(conn, n_units=0, decomposer_empty_reason="no_units_found",
                  stripped_tail="and then")
check(True, "and it accepts the states it should, so the CHECK is not blanket-rejecting")
check(store.init_db(conn) == [], "running init_db again adds nothing")
check(store.apply_additive_migrations(store.connect(":memory:")) == [],
      "migrating a database with no tables yet is a no-op, not a crash")

print()
print("5. n_units_kept and units.dropped_reason: the columns added with the second CHECK")
conn = fresh_db()
insert_prompt(conn)

expect_integrity_error(
    lambda: insert_generation(conn, n_units=3, n_units_kept=4),
    "n_units_kept > n_units is rejected",
)
expect_integrity_error(
    lambda: insert_generation(conn, n_units=None, n_units_kept=0),
    "n_units_kept set while n_units is still null is rejected",
)
insert_generation(conn, n_units=5, n_units_kept=5)
check(True, "n_units_kept == n_units is accepted")
insert_generation(conn, n_units=5, n_units_kept=0)
check(True, "n_units_kept == 0 with n_units > 0 is accepted: everything was dropped")
insert_generation(conn, n_units=5)
check(True, "n_units_kept left unset (not yet filtered) is accepted")

gen_id = insert_generation(conn, n_units=2, n_units_kept=1)
conn.execute(
    "INSERT INTO units (unit_id, generation_id, unit_index, unit_text, dropped_reason) "
    "VALUES ('u-kept', ?, 0, 'text', NULL)",
    (gen_id,),
)
conn.commit()
check(True, "dropped_reason NULL (kept) is accepted")
for i, reason in enumerate(("not_in_response", "none_token", "framing", "blank")):
    conn.execute(
        "INSERT INTO units (unit_id, generation_id, unit_index, unit_text, dropped_reason) "
        "VALUES (?, ?, ?, 'text', ?)",
        ("u-%s" % reason, gen_id, i + 1, reason),
    )
    conn.commit()
check(True, "all four of posthoc.py's _keep drop reasons are accepted")
expect_integrity_error(
    lambda: conn.execute(
        "INSERT INTO units (unit_id, generation_id, unit_index, unit_text, dropped_reason) "
        "VALUES ('u-bad', ?, 5, 'text', 'invented_reason')",
        (gen_id,),
    ),
    "a dropped_reason outside posthoc.py's four spellings is rejected",
)

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
