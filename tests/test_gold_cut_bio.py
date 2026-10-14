"""hallutrace.gold_cut_bio: Biographies units, the join back to generation_id, articles.

Synthetic outputs, inputs and a tiny enwiki-shaped SQLite file. No Modal.
Run: python -m tests.test_gold_cut_bio
"""
import csv
import os
import shutil
import sqlite3
import tempfile
import unittest

from hallutrace import gold_cut as gc
from hallutrace import gold_cut_bio as gb


def _write(path, rows, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


class GoldCutBioTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="gcbio-")
        out, inp = os.path.join(self.root, "outputs_filtered"), os.path.join(self.root, "inputs")
        atoms = "| Ada Lovelace was a mathematician.\tTrue |\n| Ada Lovelace was born in 1900.\tFalse |\n"
        _write(os.path.join(out, "m", "m-b00000.csv"),
               [{"name": "Ada Lovelace", "response": "x", "atoms": atoms,
                 "hallucinated atoms": "| Ada Lovelace was born in 1900.\tFalse |\n", "dropped atoms": "{}"},
                {"name": "Nobody Known", "response": "y", "atoms": "| Nobody Known is a poet.\tFalse |\n",
                 "hallucinated atoms": "", "dropped atoms": "{}"}],
               ["name", "response", "atoms", "hallucinated atoms", "dropped atoms"])
        _write(os.path.join(inp, "m", "m-b00000.csv"),
               [{"name": "Ada Lovelace", "response": "x", "model": "m", "sample_index": 0,
                 "generation_id": "m|biographies-00001|0"},
                {"name": "Nobody Known", "response": "y", "model": "m", "sample_index": 0,
                 "generation_id": "m|biographies-00002|0"}],
               ["name", "response", "model", "sample_index", "generation_id"])
        self.db = os.path.join(self.root, "enwiki.db")
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE documents (title PRIMARY KEY, text)")
        conn.execute("INSERT INTO documents VALUES (?, ?)",
                     ("Ada Lovelace", "<s>Ada Lovelace was an English mathematician.</s>"
                      + gb.SPECIAL_SEPARATOR + "<s>She was born in 1815.</s>"))
        conn.commit()
        conn.close()
        self.out, self.inp = out, inp

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_units_and_labels(self):
        units = gb.build_units(self.out, self.inp)
        self.assertEqual([(u.unit_id, u.verifier_label) for u in units],
                         [("m|biographies-00001|0:0", gc.SUPPORTED), ("m|biographies-00001|0:1", gc.HALLUCINATED),
                          ("m|biographies-00002|0:0", gc.HALLUCINATED)])
        self.assertEqual(units[1].unit_text, "Ada Lovelace was born in 1900.")
        self.assertEqual(units[0].model, "m")
        self.assertFalse(any(u.would_drop_23 for u in units))

    def test_articles_read_only_and_missing_noted(self):
        units = gb.build_units(self.out, self.inp)
        before = os.path.getmtime(self.db)
        found, missing = gb.attach_articles(units, self.db)
        self.assertEqual((found, missing), (1, ["Nobody Known"]))
        self.assertEqual(units[0].context, "Ada Lovelace\n\nAda Lovelace was an English mathematician."
                                           "\n\nShe was born in 1815.")
        self.assertIn("No Wikipedia article", units[2].context)
        self.assertEqual(os.path.getmtime(self.db), before)

    def test_packets_have_no_verifier_column(self):
        units = gb.build_units(self.out, self.inp)
        d = gc.draw(gb.TASK, units, "test", n=3)
        gb.attach_articles(d.units, self.db)
        paths = gc.write_packets(d, os.path.join(self.root, "packets"))
        with open(paths["annotator_A.csv"], encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertNotIn("verifier_label", header)
        with open(paths["annotator_A.csv"], encoding="utf-8") as a, open(paths["annotator_B.csv"], encoding="utf-8") as b:
            self.assertEqual(a.read(), b.read())
        with open(paths["key.csv"], encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)


if __name__ == "__main__":
    unittest.main()
