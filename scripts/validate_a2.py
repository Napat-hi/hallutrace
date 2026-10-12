r"""Checks a complete generation run.

Download the run first (a local copy, the volume is not touched):

    modal volume get hallutrace-data generations a2_out/

then, from the repo root:

    python scripts/validate_a2.py a2_out/generations

What it checks, and exits 1 if any fails:
    files        one JSONL per plan batch (300 per model, 1,500 in all), no `.tmp` left
    records      64 per batch file, 96,000 in all
    loader       every file loads through `store.load_generations` into a fresh
                 throwaway store: plan agreement, the 19-field contract, complete batches
    revisions    every record's `model_revision` equals `modal_app.MODEL_REVISIONS`

It also prints the `finish_reason` shares per model (how often the 128-token limit was hit).
The store is a temp file and is deleted afterwards; db/hallutrace.sqlite is not touched.
"""
import collections
import glob
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from hallutrace import store  # noqa: E402
from modal_app import MODEL_REVISIONS, SUBJECT_MODELS  # noqa: E402

BATCHES_PER_MODEL = 300
RECORDS_PER_BATCH = 64


def main(out_dir: str) -> int:
    failures = []
    files = sorted(glob.glob(os.path.join(out_dir, "*", "*.jsonl")))
    leftovers = glob.glob(os.path.join(out_dir, "*", "*.tmp"))
    per_model_files = collections.Counter(os.path.basename(os.path.dirname(f)) for f in files)
    print("files", len(files), dict(per_model_files))
    for m in SUBJECT_MODELS:
        if per_model_files[m] != BATCHES_PER_MODEL:
            failures.append("%s: %d batch files, expected %d" % (m, per_model_files[m], BATCHES_PER_MODEL))
    if leftovers:
        failures.append("%d .tmp files left, e.g. %s" % (len(leftovers), leftovers[0]))

    per_model = collections.Counter()
    finish = collections.Counter()
    bad_rev = 0
    for f in files:
        n = 0
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            n += 1
            per_model[r["model"]] += 1
            finish[(r["model"], r["finish_reason"])] += 1
            bad_rev += r["model_revision"] != MODEL_REVISIONS[r["model"]]
        if n != RECORDS_PER_BATCH:
            failures.append("%s: %d records, expected %d" % (f, n, RECORDS_PER_BATCH))
    total = sum(per_model.values())
    print("records per model", dict(per_model), "total", total)
    print("model_revision mismatches", bad_rev)
    if bad_rev:
        failures.append("%d records with a model_revision other than MODEL_REVISIONS" % bad_rev)
    for m in sorted(per_model):
        n = per_model[m]
        print("  %s: " % m + ", ".join(
            "%s %.1f%%" % (fr, 100 * finish[(m, fr)] / n) for fr in ("length", "stop", "prompt_too_long")))

    with tempfile.TemporaryDirectory() as tmp:
        conn = store.connect(os.path.join(tmp, "a2_check.sqlite"))
        try:
            store.init_db(conn)
            print("prompts", store.load_prompts(conn))
            # The loader owns its transactions; hand it a connection with none open.
            assert not conn.in_transaction
            try:
                print("generations", store.load_generations(conn, files))
            except (store.LoadError, store.LoadConflict) as e:
                failures.append("loader refused: %s" % e)
            rows = conn.execute("select count(*) from generations").fetchone()[0]
            print("rows in generations table", rows)
            if rows != total:
                failures.append("%d rows in the store, %d records in the files" % (rows, total))
        finally:
            conn.close()

    if failures:
        print("\nFAILED")
        for line in failures:
            print("  " + line)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
