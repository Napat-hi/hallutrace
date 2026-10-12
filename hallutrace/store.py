"""The main SQLite store (db/hallutrace.sqlite): nine tables.

Everything here can be regenerated: prompts from scripts/draw_sample.py,
generations by replaying the plan, verdicts by rerunning the verifiers. The API
cache, which cannot, is a separate file (hallutrace/cache.py).

SQLite enforces foreign keys only when a connection asks, so open the store
through `connect()`, never sqlite3.connect() directly.

`init_db()` is idempotent and never drops anything; it also adds any
ADDED_COLUMNS column an older store lacks.

### Loaders

`load_prompts` puts data/prompts.csv into table 1; `load_generations` puts the
per-batch generation JSONL into table 2. Both are idempotent, so a crashed load
resumes by running it again.

- Idempotency uses INSERT ... ON CONFLICT DO NOTHING, never INSERT OR IGNORE.
  OR IGNORE also skips rows that violate NOT NULL or CHECK, which would
  quietly swallow a record with no model_revision.
- A row that already exists is compared with the incoming one. Identical is
  counted as already present; different raises LoadConflict rather than
  keeping a stale row.
- Every generation is checked against its planned row in
  data/generation_plan/{model}.csv (seed, id, batch and the other planned
  fields). The plan is read, never recomputed. A mismatch means something is
  wrong upstream and is refused at load time.
- One JSONL file is one plan batch and one transaction, validated and inserted
  whole, so the store only holds complete batches. The loader owns the
  transaction (BEGIN IMMEDIATE, then commit or rollback), so it raises
  LoadError, before touching anything, on a connection with a transaction
  already open or in autocommit mode. Open the connection through connect().
- BEGIN IMMEDIATE takes the write lock at the start of each batch, so a load
  blocked by another writer waits up to BUSY_TIMEOUT_MS before doing any work.
- A record SQLite cannot store (a lone surrogate, an integer outside 64 bits)
  is refused as a LoadError with its file and line.
- generations.model_revision is NOT NULL, and the loader requires it on every
  record.
- abstained, n_units, decomposer_empty_reason and stripped_tail are filled by
  the verification step and left NULL here.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

DEFAULT_DB_PATH = "db/hallutrace.sqlite"
DEFAULT_PROMPTS_PATH = "data/prompts.csv"
DEFAULT_PLAN_DIR = "data/generation_plan"

# how long a statement waits for a lock before "database is locked"
BUSY_TIMEOUT_MS = 30_000

# PRAGMA user_version, bumped when SCHEMA_SQL changes in a way the column names
# do not show (a new or tightened CHECK, which SQLite cannot ALTER in place).
# scripts/init_db.py reads it to refuse a store built before the change.
#
# 1: the original nine-table schema.
# 2: decomposer_empty_reason requires a non-NULL n_units; added
#    generations.n_units_kept and units.dropped_reason.
SCHEMA_VERSION = 2

# Booleans are INTEGER (0/1); SQLite has no native BOOLEAN type.
# task_meta is a JSON string; SQLite has no native JSON column type either.
# "index" (tables 7-8, the infini-gram index name) is a keyword and is quoted
# everywhere it appears as a column name.
SCHEMA_SQL = """
PRAGMA user_version = %d;

CREATE TABLE IF NOT EXISTS prompts (
    prompt_id   TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_row  INTEGER NOT NULL,
    prompt      TEXT NOT NULL,
    task_meta   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prompts_task ON prompts(task);

CREATE TABLE IF NOT EXISTS claims (
    claim_id   TEXT PRIMARY KEY,
    claim_key  TEXT NOT NULL UNIQUE,
    s          TEXT,
    r          TEXT,
    o_h        TEXT,
    o_c        TEXT,
    claim_kind TEXT
);

CREATE TABLE IF NOT EXISTS generations (
    generation_id           TEXT PRIMARY KEY,
    prompt_id               TEXT NOT NULL REFERENCES prompts(prompt_id),
    model                   TEXT NOT NULL,
    sample_index            INTEGER NOT NULL,
    temperature             REAL NOT NULL,
    top_p                   REAL NOT NULL,
    seed                    INTEGER NOT NULL,
    escalated               INTEGER NOT NULL CHECK (escalated IN (0, 1)),
    response                TEXT,
    abstained               INTEGER CHECK (abstained IS NULL OR abstained IN (0, 1)),
    n_units                 INTEGER CHECK (n_units IS NULL OR n_units >= 0),
    decomposer_empty_reason TEXT,
    generated_at            TEXT NOT NULL,
    gen_config_hash         TEXT NOT NULL,
    -- carried because every generation record has them
    batch_id                TEXT NOT NULL,
    finish_reason           TEXT NOT NULL,
    n_output_tokens         INTEGER NOT NULL,
    max_new_tokens          INTEGER NOT NULL,
    n_prompt_tokens         INTEGER NOT NULL,
    engine                  TEXT NOT NULL,
    gpu                     TEXT NOT NULL,
    model_revision          TEXT NOT NULL,
    -- the unfinished tail dropped before decomposition, as text. NULL: not yet
    -- processed; '': processed, nothing dropped; otherwise the dropped text (ask
    -- IS NULL, not truthiness). A strip result requires a decomposition result;
    -- the converse does not hold (a null response or an abstention skips it).
    stripped_tail           TEXT CHECK (stripped_tail IS NULL OR n_units IS NOT NULL),
    -- units left after posthoc.py's filter; n_units stays the decomposer's raw
    -- count, and this is NULL until the filter has run
    n_units_kept            INTEGER CHECK (
        n_units_kept IS NULL OR (n_units IS NOT NULL AND n_units_kept <= n_units)
    ),
    -- the five permitted reasons
    CHECK (
        decomposer_empty_reason IS NULL
        OR decomposer_empty_reason IN (
            'empty_response', 'abstained', 'no_units_found',
            'failed_parse', 'failed_api'
        )
    ),
    -- null while undecomposed, required at n_units = 0, forbidden once the
    -- decomposer found something
    CHECK (
        (n_units IS NULL AND decomposer_empty_reason IS NULL)
        OR (n_units = 0 AND decomposer_empty_reason IS NOT NULL)
        OR (n_units > 0 AND decomposer_empty_reason IS NULL)
    ),
    -- the CHECK above is NULL, which SQLite treats as satisfied, when n_units
    -- is NULL; this one keeps an undecomposed row from carrying a reason
    CHECK (n_units IS NOT NULL OR decomposer_empty_reason IS NULL)
);
CREATE INDEX IF NOT EXISTS idx_generations_prompt_id ON generations(prompt_id);
CREATE INDEX IF NOT EXISTS idx_generations_model ON generations(model);
CREATE INDEX IF NOT EXISTS idx_generations_batch_id ON generations(batch_id);

CREATE TABLE IF NOT EXISTS units (
    unit_id          TEXT PRIMARY KEY,
    generation_id    TEXT NOT NULL REFERENCES generations(generation_id),
    unit_index       INTEGER NOT NULL,
    unit_text        TEXT NOT NULL,
    -- nullable: claim_id is assigned by the claim_key rule
    claim_id         TEXT REFERENCES claims(claim_id),
    decomposer_model TEXT,
    verifier         TEXT,
    verifier_raw     TEXT,
    verdict          TEXT CHECK (
        verdict IS NULL OR verdict IN ('hallucinated', 'supported', 'unverifiable')
    ),
    verified_at      TEXT,
    -- why posthoc.py's filter dropped this unit; NULL means kept
    dropped_reason   TEXT CHECK (
        dropped_reason IS NULL OR dropped_reason IN (
            'not_in_response', 'none_token', 'framing', 'blank'
        )
    )
);
CREATE INDEX IF NOT EXISTS idx_units_generation_id ON units(generation_id);
CREATE INDEX IF NOT EXISTS idx_units_claim_id ON units(claim_id);
-- Phase 2 starts by selecting the hallucinated rows out of ~1.2M; partial
-- index keeps that a lookup instead of a full scan.
CREATE INDEX IF NOT EXISTS idx_units_hallucinated ON units(verdict)
    WHERE verdict = 'hallucinated';

CREATE TABLE IF NOT EXISTS stability (
    prompt_id           TEXT NOT NULL REFERENCES prompts(prompt_id),
    model               TEXT NOT NULL,
    n_samples           INTEGER NOT NULL,
    n_hallucinated      INTEGER NOT NULL,
    h                   REAL NOT NULL,
    dominant_claim_id   TEXT REFERENCES claims(claim_id),
    c                   REAL,
    h_at5               REAL NOT NULL,
    c_at5               REAL,
    n_hallucinated_at5  INTEGER NOT NULL,
    regime              TEXT CHECK (regime IS NULL OR regime IN ('S1', 'S2', 'S3')),
    greedy_hallucinated INTEGER NOT NULL CHECK (greedy_hallucinated IN (0, 1)),
    escalated           INTEGER NOT NULL CHECK (escalated IN (0, 1)),
    cluster_method      TEXT,
    c_cut               REAL,
    PRIMARY KEY (prompt_id, model)
);
CREATE INDEX IF NOT EXISTS idx_stability_model ON stability(model);

CREATE TABLE IF NOT EXISTS claim_stability (
    prompt_id   TEXT NOT NULL,
    model       TEXT NOT NULL,
    claim_id    TEXT NOT NULL REFERENCES claims(claim_id),
    n_present   INTEGER NOT NULL,
    claim_rate  REAL NOT NULL,
    is_dominant INTEGER NOT NULL CHECK (is_dominant IN (0, 1)),
    PRIMARY KEY (prompt_id, model, claim_id),
    FOREIGN KEY (prompt_id, model) REFERENCES stability(prompt_id, model)
);
CREATE INDEX IF NOT EXISTS idx_claim_stability_claim_id ON claim_stability(claim_id);

CREATE TABLE IF NOT EXISTS counts (
    "index"           TEXT NOT NULL,
    query_string      TEXT NOT NULL,
    query_form        TEXT NOT NULL,
    raw_count         INTEGER NOT NULL,
    tokens_subtracted INTEGER NOT NULL,
    clean_count       INTEGER NOT NULL,
    n_docs            INTEGER NOT NULL,
    n_domains         INTEGER NOT NULL,
    approx            INTEGER NOT NULL CHECK (approx IN (0, 1)),
    queried_at        TEXT NOT NULL,
    api_version       TEXT NOT NULL,
    -- this key is what makes the table a cache
    PRIMARY KEY ("index", query_form, query_string)
);

CREATE TABLE IF NOT EXISTS claim_corpus (
    claim_id     TEXT NOT NULL REFERENCES claims(claim_id),
    "index"      TEXT NOT NULL,
    n_h          INTEGER NOT NULL,
    n_c          INTEGER NOT NULL,
    n_docs_h     INTEGER NOT NULL,
    n_docs_c     INTEGER NOT NULL,
    n_domains_c  INTEGER NOT NULL,
    rate_h       REAL NOT NULL,
    rate_c       REAL NOT NULL,
    rho          REAL NOT NULL,
    phase2_type  TEXT,
    tau_used     REAL,
    rho_cut      REAL,
    rule_version TEXT,
    PRIMARY KEY (claim_id, "index")
);

CREATE TABLE IF NOT EXISTS labels (
    prompt_id          TEXT NOT NULL,
    model              TEXT NOT NULL,
    claim_id           TEXT NOT NULL REFERENCES claims(claim_id),
    phase2_type        TEXT,
    -- copied from stability.regime; same constraint, same reason
    regime             TEXT CHECK (regime IS NULL OR regime IN ('S1', 'S2', 'S3')),
    claim_rate         REAL,
    intervention_point TEXT,
    confidence         TEXT,
    coverage_downgrade INTEGER CHECK (coverage_downgrade IS NULL OR coverage_downgrade IN (0, 1, 2)),
    rule_version       TEXT,
    PRIMARY KEY (prompt_id, model, claim_id),
    -- implies a valid stability(prompt_id, model) row too, transitively,
    -- since claim_stability itself required one and nothing here is ever
    -- deleted
    FOREIGN KEY (prompt_id, model, claim_id)
        REFERENCES claim_stability(prompt_id, model, claim_id)
);
CREATE INDEX IF NOT EXISTS idx_labels_model ON labels(model);
CREATE INDEX IF NOT EXISTS idx_labels_claim_id ON labels(claim_id);
""" % SCHEMA_VERSION


def connect(path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the main store with foreign keys enforced and a 30 s lock wait.

    Every caller goes through this, never sqlite3.connect() directly: the FK
    pragma is per-connection and SQLite leaves it off unless asked. The
    connection keeps Python's default transaction handling, which the loaders
    rely on being untouched (see "Loaders" in the module docstring).
    """
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


# Columns added after the first stores were built, as (table, column, ALTER
# statement); init_db() runs those whose column is missing. A CHECK declared on
# an added column is enforced as in a fresh table, so keep each definition
# identical to SCHEMA_SQL's (tests/test_store_schema.py compares the two).
# Additive only: anything else is a rebuild, and scripts/init_db.py says so.
ADDED_COLUMNS = (
    (
        "generations",
        "stripped_tail",
        "ALTER TABLE generations ADD COLUMN stripped_tail TEXT "
        "CHECK (stripped_tail IS NULL OR n_units IS NOT NULL)",
    ),
    (
        "generations",
        "n_units_kept",
        "ALTER TABLE generations ADD COLUMN n_units_kept INTEGER "
        "CHECK (n_units_kept IS NULL OR (n_units IS NOT NULL AND n_units_kept <= n_units))",
    ),
    (
        "units",
        "dropped_reason",
        "ALTER TABLE units ADD COLUMN dropped_reason TEXT "
        "CHECK (dropped_reason IS NULL OR dropped_reason IN "
        "('not_in_response', 'none_token', 'framing', 'blank'))",
    ),
)


def column_names(conn: sqlite3.Connection, table: str) -> set:
    """The columns a table has right now, empty if the table does not exist."""
    return {r[1] for r in conn.execute('PRAGMA table_info("%s")' % table)}


def apply_additive_migrations(conn: sqlite3.Connection) -> list:
    """Add any ADDED_COLUMNS column the store does not have yet.

    Returns the "table.column" names it added, empty on an up-to-date store.
    Existing rows get NULL, which is the not-yet-processed state for every
    column here by construction.
    """
    added = []
    for table, column, statement in ADDED_COLUMNS:
        have = column_names(conn, table)
        if not have or column in have:
            continue
        conn.execute(statement)
        added.append("%s.%s" % (table, column))
    conn.commit()
    return added


def init_db(conn: sqlite3.Connection) -> list:
    """Create all nine tables and their indexes. Safe to call repeatedly.

    Returns what apply_additive_migrations() added, so a caller can report a
    store that was migrated rather than created.
    """
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return apply_additive_migrations(conn)


def foreign_keys_enabled(conn: sqlite3.Connection) -> bool:
    return conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

TASKS = (
    "biographies",
    "false_presuppositions",
    "historical_events",
    "rationalization_binary",
    "rationalization_numerical",
    "scientific_attribution",
    "simplification",
    "summarization",
)

PROMPT_COLUMNS = ("prompt_id", "task", "source_file", "source_row", "prompt", "task_meta")

# the fields of a generation record, each a column of generations
GENERATION_COLUMNS = (
    "generation_id", "prompt_id", "model", "sample_index", "temperature", "top_p",
    "seed", "escalated", "response", "generated_at", "gen_config_hash", "batch_id",
    "max_new_tokens", "n_prompt_tokens", "n_output_tokens", "finish_reason",
    "engine", "gpu", "model_revision",
)

FINISH_REASONS = ("length", "stop", "prompt_too_long")


class LoadError(Exception):
    """Input was refused. Nothing from the offending file/batch was stored."""


class LoadConflict(LoadError):
    """A row with this key already exists and differs from the incoming one."""


@dataclass
class LoadResult:
    inserted: int = 0
    already_present: int = 0

    def __iadd__(self, other: "LoadResult") -> "LoadResult":
        self.inserted += other.inserted
        self.already_present += other.already_present
        return self


def _begin_owned_transaction(conn: sqlite3.Connection, loader: str) -> None:
    """Open the transaction a loader will own, or refuse without touching anything.

    Call it before the `with conn:` block: a BEGIN failing inside the block would
    roll back the caller's own work. It refuses an open transaction and
    autocommit mode, where the batch would be left uncommitted with no error.
    """
    if conn.in_transaction:
        raise LoadError(
            "%s needs a connection with no transaction open, because it owns the "
            "transaction for each batch and would commit or roll back yours with it. "
            "Commit or roll back your own work first. (A connection opened with "
            "autocommit=False always has one open.)" % loader
        )
    if getattr(conn, "autocommit", None) is True:
        raise LoadError(
            "%s cannot run on a connection with autocommit=True: commit() and "
            "rollback() do nothing there, so it could not own a transaction and a "
            "batch would be left uncommitted with no error. Open the connection "
            "through store.connect()." % loader
        )
    conn.execute("BEGIN IMMEDIATE")


def _unstorable(value) -> str | None:
    """Why SQLite cannot store this value, or None if it can."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and not (-(2**63) <= value <= 2**63 - 1):
        return "integer %d is outside SQLite's 64-bit range" % value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as e:
            return "lone surrogate U+%04X at offset %d, which cannot be encoded as UTF-8" % (
                ord(value[e.start]), e.start)
    return None


def _insert_idempotent(conn, table, key_column, columns, values) -> bool:
    """Insert one row; True if inserted, False if an identical row was there.

    ON CONFLICT on the key only: NOT NULL, CHECK and FOREIGN KEY violations
    still raise. An existing row that differs raises LoadConflict.
    """
    cols = ", ".join(columns)
    marks = ", ".join("?" for _ in columns)
    cur = conn.execute(
        "INSERT INTO %s (%s) VALUES (%s) ON CONFLICT(%s) DO NOTHING"
        % (table, cols, marks, key_column),
        values,
    )
    if cur.rowcount == 1:
        return True
    key = values[columns.index(key_column)]
    existing = conn.execute(
        "SELECT %s FROM %s WHERE %s = ?" % (cols, table, key_column), (key,)
    ).fetchone()
    if tuple(existing) != tuple(values):
        differing = [c for c, a, b in zip(columns, existing, values) if a != b]
        raise LoadConflict(
            "%s %r is already stored and differs in: %s" % (table, key, ", ".join(differing))
        )
    return False


def load_prompts(conn: sqlite3.Connection, path: str = DEFAULT_PROMPTS_PATH) -> LoadResult:
    """data/prompts.csv into table 1, as one transaction. Idempotent."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != PROMPT_COLUMNS:
            raise LoadError(
                "%s: columns are %s, expected %s" % (path, reader.fieldnames, list(PROMPT_COLUMNS))
            )
        rows = list(reader)

    seen = set()
    records = []
    for n, row in enumerate(rows, start=1):
        where = "%s row %d" % (path, n)
        if None in row or any(v is None for v in row.values()):
            raise LoadError("%s: wrong number of fields" % where)
        if row["task"] not in TASKS:
            raise LoadError("%s: unknown task %r" % (where, row["task"]))
        try:
            source_row = int(row["source_row"])
        except ValueError:
            raise LoadError("%s: source_row %r is not an integer" % (where, row["source_row"]))
        expected = "%s-%05d" % (row["task"], source_row)
        if row["prompt_id"] != expected:
            raise LoadError("%s: prompt_id %r should be %r" % (where, row["prompt_id"], expected))
        if row["prompt_id"] in seen:
            raise LoadError("%s: duplicate prompt_id %r" % (where, row["prompt_id"]))
        seen.add(row["prompt_id"])
        if not row["prompt"]:
            raise LoadError("%s: empty prompt" % where)
        try:
            json.loads(row["task_meta"])
        except ValueError:
            raise LoadError("%s: task_meta is not valid JSON" % where)
        records.append(
            (row["prompt_id"], row["task"], row["source_file"], source_row,
             row["prompt"], row["task_meta"])
        )

    result = LoadResult()
    _begin_owned_transaction(conn, "load_prompts")
    with conn:
        for values in records:
            if _insert_idempotent(conn, "prompts", "prompt_id", PROMPT_COLUMNS, values):
                result.inserted += 1
            else:
                result.already_present += 1
    return result


class _Plan:
    """One model's generation plan: what was supposed to run."""

    def __init__(self, model: str, plan_dir: str):
        path = Path(plan_dir) / (model + ".csv")
        if not path.is_file():
            raise LoadError("no generation plan for model %r (looked for %s)" % (model, path))
        self.rows = {}
        self.batches = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                self.rows[r["generation_id"]] = r
                self.batches.setdefault(r["batch_id"], set()).add(r["generation_id"])


# record field -> how to read the planned value out of the plan CSV's strings
_PLANNED = {
    "prompt_id": str,
    "model": str,
    "batch_id": str,
    "gen_config_hash": str,
    "sample_index": int,
    "seed": int,
    "max_new_tokens": int,
    "temperature": float,
    "top_p": float,
    "escalated": lambda s: {"True": True, "False": False}[s],
}


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _nonempty_str(v):
    return isinstance(v, str) and v != ""


_TYPES = {
    "generation_id": _nonempty_str, "prompt_id": _nonempty_str, "model": _nonempty_str,
    "sample_index": _is_int, "temperature": _is_num, "top_p": _is_num, "seed": _is_int,
    "escalated": lambda v: isinstance(v, bool),
    "response": lambda v: v is None or isinstance(v, str),
    "generated_at": _nonempty_str, "gen_config_hash": _nonempty_str,
    "batch_id": _nonempty_str, "max_new_tokens": _is_int, "n_prompt_tokens": _is_int,
    "n_output_tokens": _is_int, "finish_reason": lambda v: v in FINISH_REASONS,
    "engine": _nonempty_str, "gpu": _nonempty_str, "model_revision": _nonempty_str,
}


def _validate_batch_file(path: Path, plans: dict, plan_dir: str) -> list:
    """Read one JSONL file and check every record. Returns insert-ready tuples."""
    parsed = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                parsed.append((lineno, json.loads(line)))
            except ValueError:
                raise LoadError("%s line %d: not valid JSON" % (path, lineno))
    if not parsed:
        raise LoadError("%s: no records" % path)

    ids = []
    batch_ids = set()
    values_out = []
    for lineno, rec in parsed:
        where = "%s line %d" % (path, lineno)
        if not isinstance(rec, dict):
            raise LoadError("%s: record is not an object" % where)
        missing = sorted(set(GENERATION_COLUMNS) - set(rec))
        extra = sorted(set(rec) - set(GENERATION_COLUMNS))
        if missing:
            raise LoadError("%s: missing field(s): %s" % (where, ", ".join(missing)))
        if extra:
            raise LoadError(
                "%s: unexpected field(s): %s (a new column needs a schema decision)"
                % (where, ", ".join(extra))
            )
        for field, ok in _TYPES.items():
            if not ok(rec[field]):
                raise LoadError("%s: bad value for %s: %r" % (where, field, rec[field]))
        for field in GENERATION_COLUMNS:
            why = _unstorable(rec[field])
            if why:
                raise LoadError("%s: %s cannot be stored: %s" % (where, field, why))
        try:
            datetime.fromisoformat(rec["generated_at"])
        except ValueError:
            raise LoadError("%s: generated_at %r is not ISO-8601" % (where, rec["generated_at"]))
        if (rec["response"] is None) != (rec["finish_reason"] == "prompt_too_long"):
            raise LoadError(
                "%s: response is null only when finish_reason is prompt_too_long "
                "(got response %s, finish_reason %r)"
                % (where, "null" if rec["response"] is None else "present", rec["finish_reason"])
            )

        expected_id = "%s|%s|%d" % (rec["model"], rec["prompt_id"], rec["sample_index"])
        if rec["generation_id"] != expected_id:
            raise LoadError(
                "%s: generation_id %r should be %r" % (where, rec["generation_id"], expected_id)
            )
        model = rec["model"]
        if model not in plans:
            plans[model] = _Plan(model, plan_dir)
        planned = plans[model].rows.get(rec["generation_id"])
        if planned is None:
            raise LoadError("%s: %r is not in the plan for %s" % (where, rec["generation_id"], model))
        for field, read in _PLANNED.items():
            want = read(planned[field])
            if rec[field] != want:
                raise LoadError(
                    "%s: %s is %r but the plan says %r for %s"
                    % (where, field, rec[field], want, rec["generation_id"])
                )

        ids.append(rec["generation_id"])
        batch_ids.add(rec["batch_id"])
        values_out.append(tuple(
            int(rec[c]) if c == "escalated" else rec[c] for c in GENERATION_COLUMNS
        ))

    if len(batch_ids) != 1:
        raise LoadError("%s: records span several batches: %s" % (path, sorted(batch_ids)))
    batch_id = next(iter(batch_ids))
    if len(ids) != len(set(ids)):
        raise LoadError("%s: duplicate generation_id within the batch" % path)
    planned_ids = plans[parsed[0][1]["model"]].batches[batch_id]
    if set(ids) != planned_ids:
        raise LoadError(
            "%s: batch %s is incomplete or wrong: %d planned, %d present, %d missing"
            % (path, batch_id, len(planned_ids), len(ids), len(planned_ids - set(ids)))
        )
    return values_out


def find_batch_files(*roots: str) -> list:
    """Every *.jsonl under the given directories, in a stable order."""
    files = []
    for root in roots:
        files.extend(Path(root).rglob("*.jsonl"))
    return sorted(files)


def load_generations(
    conn: sqlite3.Connection,
    paths: Iterable,
    plan_dir: str = DEFAULT_PLAN_DIR,
) -> LoadResult:
    """The per-batch generation JSONL into table 2. Idempotent.

    Each file is validated whole against the plan, then inserted in its own
    transaction. The first bad file raises; files before it stay loaded, so
    fixing the problem and running again resumes where it stopped.
    """
    plans: dict = {}
    total = LoadResult()
    for path in paths:
        path = Path(path)
        rows = _validate_batch_file(path, plans, plan_dir)
        result = LoadResult()
        _begin_owned_transaction(conn, "load_generations")
        try:
            with conn:
                for values in rows:
                    if _insert_idempotent(
                        conn, "generations", "generation_id", GENERATION_COLUMNS, values
                    ):
                        result.inserted += 1
                    else:
                        result.already_present += 1
        except sqlite3.IntegrityError as e:
            raise LoadError("%s: %s" % (path, e)) from e
        except LoadConflict as e:
            raise LoadConflict("%s: %s" % (path, e)) from e
        total += result
    return total
