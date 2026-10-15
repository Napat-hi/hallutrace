"""Supplementary tables T1 and T2 (scripts/supp_tables.py), on synthetic stability summaries.

    .\\.venv\\Scripts\\python.exe -m tests.test_supp_tables
"""

import csv
import json
import os
import shutil
import tempfile
import unittest

from scripts import supp_tables as st


def row(task, model, usable, hallucinated, zero_unit=0, abstention=0, empty=0, emptied=0, missing=0):
    n = usable + zero_unit + abstention + empty + emptied + missing
    return {"task": task, "model": model, "items": 2, "n_samples": n, "n_usable": usable,
            "n_hallucinated": hallucinated, "n_zero_unit": zero_unit, "n_abstention": abstention,
            "n_empty": empty, "n_emptied": emptied, "n_missing": missing, "S1": 1, "always_wrong": 1,
            "always_wrong_low_n": 0, "single_mention": 0, "no_h": 0,
            "response_ratio": round(usable / n, 4), "h_pooled": round(hallucinated / usable, 4),
            "h_pooled_halogen_zero_units": round(hallucinated / (usable + zero_unit), 4)}


def summaries():
    d1 = {"items": {"biographies|m1": row("biographies", "m1", 8, 6, abstention=2),
                    "historical_events|m1": row("historical_events", "m1", 10, 2),
                    "summarization|m1": row("summarization", "m1", 9, 3, zero_unit=1, empty=1)},
          "items_raw": {"summarization|m1": row("summarization", "m1", 9, 6, zero_unit=1, empty=1)}}
    d1_outputs = {"items": {"biographies|m1": row("biographies", "m1", 8, 7, abstention=2),
                            "historical_events|m1": row("historical_events", "m1", 10, 2),
                            "summarization|m1": row("summarization", "m1", 9, 3, zero_unit=1, empty=1)}}
    return d1, d1_outputs


class SuppTables(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unfiltered_h_by_task(self):
        t1, _ = st.tables(*summaries())
        by = {r["task"]: r for r in t1}
        self.assertEqual(by["biographies"]["h_unfiltered"], 0.875)       # from the --outputs run
        self.assertEqual(by["summarization"]["h_unfiltered"], 0.6667)    # from items_raw
        self.assertEqual(by["historical_events"]["h_unfiltered"], "")    # no filter
        self.assertEqual(by["biographies"]["h"], 0.75)

    def test_outcomes_add_up(self):
        _, t2 = st.tables(*summaries())
        for r in t2:
            parts = sum(r[c] for c in ("usable", "zero_unit", "abstention", "empty", "emptied", "missing"))
            self.assertEqual(parts, r["n_samples"])

    def test_refuses_outcomes_that_do_not_add_up(self):
        d1, d1_outputs = summaries()
        d1["items"]["summarization|m1"]["n_samples"] += 1
        with self.assertRaises(ValueError):
            st.tables(d1, d1_outputs)

    def test_cli_writes_both_files(self):
        d1, d1_outputs = summaries()
        for name, data in (("d1", d1), ("d1_outputs", d1_outputs)):
            os.makedirs(os.path.join(self.tmp, name))
            with open(os.path.join(self.tmp, name, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(data, f)
        out = os.path.join(self.tmp, "out")
        code = st.main(["--d1", os.path.join(self.tmp, "d1"), "--d1-outputs", os.path.join(self.tmp, "d1_outputs"),
                        "--out", out])
        self.assertEqual(code, 0)
        with open(os.path.join(out, "t1_h_by_task_model.csv"), encoding="utf-8") as f:
            self.assertEqual(next(csv.reader(f)), st.T1_COLUMNS)
        with open(os.path.join(out, "t2_outcomes_by_task_model.csv"), encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)


if __name__ == "__main__":
    unittest.main()
