"""Tests for the loaders in hallutrace/store.py. No key, no network.

Runs on the real files: data/prompts.csv (3,200 prompts) and the 90 pilot and
pilot_1 and pilot_2 JSONL batches in results/ (5,760 generations), plus broken copies made in
a temp directory to show the loader refuses them.

Run from the repo root: python -m tests.test_store_loaders
"""

import csv
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from hallutrace import store

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


def expect_refused(exc, fn, label, substr):
    """The loader must raise `exc`, and the message must say why."""
    try:
        fn()
    except exc as e:
        ok = substr in str(e)
        short = str(e).split(": ", 1)[-1]
        check(ok, "%s -> %s" % (label, short if ok else "wrong message: %s" % e))
        return
    except Exception as e:  # noqa: BLE001
        check(False, "%s  (raised %s instead: %s)" % (label, type(e).__name__, e))
        return
    check(False, label + "  (did not raise)")


def fingerprint(conn, table, order):
    h = hashlib.sha256()
    for row in conn.execute("SELECT * FROM %s ORDER BY %s" % (table, order)):
        h.update(repr(tuple(row)).encode("utf-8"))
    return h.hexdigest()


def count(conn, table):
    return conn.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]


def fresh(with_prompts=True):
    conn = store.connect(":memory:")
    store.init_db(conn)
    if with_prompts:
        store.load_prompts(conn)
    return conn


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def frozen_inputs():
    paths = [Path("data/prompts.csv")] + sorted(Path("data/generation_plan").glob("*.csv"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


INPUTS_BEFORE = frozen_inputs()
FILES = store.find_batch_files("results/pilot_1", "results/pilot_2")
tmp = Path(tempfile.mkdtemp(prefix="hallutrace_loader_test_"))
SRC = FILES[0]


def broken_copy(name, mutate):
    """A copy of one real batch with `mutate(records)` applied."""
    recs = read_jsonl(SRC)
    mutate(recs)
    out = tmp / name
    write_jsonl(out, recs)
    return out


try:
    print("0. the inputs are what we think they are")
    eq(len(FILES), 90, "90 pilot_1 and pilot_2 batch files")

    print()
    print("1. prompts: table 1")
    conn = fresh(with_prompts=False)
    first = store.load_prompts(conn)
    eq(first.inserted, 3200, "first load inserts all 3,200 prompts")
    eq(count(conn, "prompts"), 3200, "and the table holds 3,200 rows")
    before = fingerprint(conn, "prompts", "prompt_id")
    second = store.load_prompts(conn)
    eq((second.inserted, second.already_present), (0, 3200),
       "second load inserts nothing and finds all 3,200 already present")
    eq(fingerprint(conn, "prompts", "prompt_id"), before,
       "and the table content is byte-for-byte unchanged")

    multi = conn.execute("SELECT COUNT(*) FROM prompts WHERE prompt LIKE '%' || char(10) || '%'").fetchone()[0]
    check(multi > 0, "prompts with embedded newlines survived the CSV read (%d of them)" % multi)

    with open("data/prompts.csv", newline="", encoding="utf-8") as fh:
        source = list(csv.DictReader(fh))
    row = conn.execute("SELECT * FROM prompts WHERE prompt_id = ?", (source[0]["prompt_id"],)).fetchone()
    eq(row["task_meta"], source[0]["task_meta"], "task_meta is stored exactly as given, not re-serialised")

    def prompts_csv(name, mutate, fieldnames=None):
        rows = [dict(r) for r in source]
        mutate(rows)
        out = tmp / name
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames or list(source[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return str(out)

    p = prompts_csv("p_changed.csv", lambda rows: rows[0].update(prompt="a different prompt"))
    expect_refused(store.LoadConflict, lambda: store.load_prompts(conn, p),
                   "a re-drawn prompt under an existing id is refused, not skipped",
                   "differs in: prompt")
    p = prompts_csv("p_badid.csv", lambda rows: rows[0].update(prompt_id="biographies:1"))
    expect_refused(store.LoadError, lambda: store.load_prompts(fresh(False), p),
                   "the spec's colon id format is refused", "should be")
    p = prompts_csv("p_task.csv", lambda rows: rows[0].update(task="poetry"))
    expect_refused(store.LoadError, lambda: store.load_prompts(fresh(False), p),
                   "an unknown task is refused", "unknown task")
    p = prompts_csv("p_dup.csv", lambda rows: rows.append(dict(rows[0])))
    expect_refused(store.LoadError, lambda: store.load_prompts(fresh(False), p),
                   "a duplicate prompt_id is refused", "duplicate prompt_id")
    p = prompts_csv("p_hdr.csv", lambda rows: None, fieldnames=list(source[0].keys()))
    text = Path(p).read_text(encoding="utf-8").replace("task_meta", "meta", 1)
    Path(p).write_text(text, encoding="utf-8", newline="")
    expect_refused(store.LoadError, lambda: store.load_prompts(fresh(False), p),
                   "a wrong header is refused", "columns are")

    print()
    print("2. generations: table 2, real pilot data")
    conn = fresh()
    first = store.load_generations(conn, FILES)
    eq(first.inserted, 5760, "first load inserts all 5,760 real generations")
    eq(count(conn, "generations"), 5760, "and the table holds 5,760 rows")
    before = fingerprint(conn, "generations", "generation_id")
    second = store.load_generations(conn, FILES)
    eq((second.inserted, second.already_present), (0, 5760),
       "second load inserts nothing and finds all 5,760 already present")
    eq(fingerprint(conn, "generations", "generation_id"), before,
       "and the table content is byte-for-byte unchanged")
    per_model = dict(conn.execute("SELECT model, COUNT(*) FROM generations GROUP BY model").fetchall())
    eq(sorted(per_model.values()), [1152] * 5, "1,152 rows for each of the five models")

    mismatch = 0
    for f in FILES:
        for rec in read_jsonl(f):
            got = conn.execute("SELECT * FROM generations WHERE generation_id = ?",
                               (rec["generation_id"],)).fetchone()
            for col in store.GENERATION_COLUMNS:
                want = int(rec[col]) if col == "escalated" else rec[col]
                if got[col] != want:
                    mismatch += 1
    eq(mismatch, 0, "every stored column equals its JSONL value, for all 5,760 records")

    nulls = conn.execute(
        "SELECT COUNT(abstained), COUNT(n_units), COUNT(decomposer_empty_reason) FROM generations"
    ).fetchone()
    eq(tuple(nulls), (0, 0, 0), "abstained, n_units and decomposer_empty_reason are left for the verification step")
    eq(conn.execute("SELECT COUNT(*) FROM generations WHERE model_revision = ''").fetchone()[0], 0,
       "every row carries a model_revision")

    print()
    print("3. resume: a crashed load finishes where it stopped")
    conn = fresh()
    part = store.load_generations(conn, FILES[:10])
    eq(part.inserted, 640, "a load that stops after 10 batches stored 640 rows")
    rest = store.load_generations(conn, FILES)
    eq((rest.inserted, rest.already_present), (5120, 640),
       "rerunning over everything inserts only the 5,120 missing and skips the 640")
    eq(count(conn, "generations"), 5760, "final count is 5,760, no duplicates")
    oneshot = fresh()
    store.load_generations(oneshot, FILES)
    eq(fingerprint(conn, "generations", "generation_id"),
       fingerprint(oneshot, "generations", "generation_id"),
       "the resumed store is identical to one loaded in a single pass")

    good2, good3 = FILES[1], FILES[2]
    bad = tmp / "bad_second.jsonl"
    recs = read_jsonl(good2)
    recs[5]["seed"] += 1
    write_jsonl(bad, recs)
    conn = fresh()
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [FILES[0], bad, good3]),
                   "a bad file in the middle stops the load", "seed")
    eq(count(conn, "generations"), 64, "the batch before it stays loaded, the bad one left nothing")
    resumed = store.load_generations(conn, [FILES[0], good2, good3])
    eq((resumed.inserted, resumed.already_present), (128, 64),
       "after fixing it, the rerun resumes: 128 new, 64 already there")

    print()
    print("4. the plan is the record of what was supposed to run")
    conn = fresh()

    def seed_off(recs):
        recs[3]["seed"] += 1
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("seed.jsonl", seed_off)]),
                   "a seed that disagrees with the plan is refused", "seed is")

    def id_off(recs):
        recs[3]["generation_id"] = recs[3]["generation_id"][:-1] + "9"
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("id.jsonl", id_off)]),
                   "a generation_id that disagrees with its own fields is refused", "should be")

    def unplanned(recs):
        r = recs[3]
        r["prompt_id"] = "biographies-99999"
        r["generation_id"] = "%s|biographies-99999|%d" % (r["model"], r["sample_index"])
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("unplanned.jsonl", unplanned)]),
                   "an id that is not in the plan at all is refused", "not in the plan")

    def other_batch(recs):
        recs[3]["batch_id"] = "olmo2-7b-b00099"
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("batch.jsonl", other_batch)]),
                   "a record filed under the wrong batch is refused", "batch_id is")

    def other_hash(recs):
        recs[3]["gen_config_hash"] = "000000000000"
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("hash.jsonl", other_hash)]),
                   "a different gen_config_hash is refused", "gen_config_hash is")

    def short(recs):
        del recs[10]
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("short.jsonl", short)]),
                   "a batch with a planned generation missing is refused", "incomplete")

    def unknown_model(recs):
        for r in recs:
            r["model"] = "gpt-x"
            r["generation_id"] = "gpt-x|%s|%d" % (r["prompt_id"], r["sample_index"])
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("model.jsonl", unknown_model)]),
                   "a model with no plan file is refused", "no generation plan")

    eq(count(conn, "generations"), 0, "none of those refusals stored a single row")

    print()
    print("5. the contract: model_revision cannot be skipped, and nothing unexpected gets in")

    def no_rev(recs):
        del recs[0]["model_revision"]
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("norev.jsonl", no_rev)]),
                   "a record with no model_revision is refused", "missing field(s): model_revision")

    def null_rev(recs):
        recs[0]["model_revision"] = None
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("nullrev.jsonl", null_rev)]),
                   "a null model_revision is refused", "bad value for model_revision")

    def extra(recs):
        recs[0]["tokens_dropped"] = 0
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("extra.jsonl", extra)]),
                   "an unexpected field is refused, not dropped", "unexpected field(s): tokens_dropped")

    def null_response(recs):
        recs[0]["response"] = None
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("nullresp.jsonl", null_response)]),
                   "a null response without prompt_too_long is refused", "prompt_too_long")

    def bad_finish(recs):
        recs[0]["finish_reason"] = "timeout"
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [broken_copy("finish.jsonl", bad_finish)]),
                   "an unknown finish_reason is refused", "bad value for finish_reason")

    print()
    print("6. an existing row that differs is a conflict, not a skip")
    conn = fresh()
    store.load_generations(conn, [SRC])

    def new_response(recs):
        recs[0]["response"] = "a regenerated, different response"
    expect_refused(store.LoadConflict, lambda: store.load_generations(conn, [broken_copy("regen.jsonl", new_response)]),
                   "a re-downloaded batch with a changed response is refused", "differs in: response")
    stored = conn.execute("SELECT response FROM generations WHERE generation_id = ?",
                          (read_jsonl(SRC)[0]["generation_id"],)).fetchone()[0]
    eq(stored, read_jsonl(SRC)[0]["response"], "and the stored response is the original")

    print()
    print("7. foreign keys, and a failure halfway through a batch")
    conn = fresh(with_prompts=False)
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [SRC]),
                   "generations loaded before their prompts are refused", "FOREIGN KEY")

    conn = fresh()
    victim = read_jsonl(SRC)[40]["prompt_id"]
    conn.execute("DELETE FROM prompts WHERE prompt_id = ?", (victim,))
    conn.commit()
    expect_refused(store.LoadError, lambda: store.load_generations(conn, [SRC]),
                   "a batch whose 41st row has no prompt fails partway through", "FOREIGN KEY")
    eq(count(conn, "generations"), 0,
       "and the 40 rows inserted before the failure were rolled back with it")

    print()
    print("8. the inputs were only read")
    check(frozen_inputs() == INPUTS_BEFORE,
          "data/prompts.csv and data/generation_plan/*.csv are byte-identical after every load above")

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
