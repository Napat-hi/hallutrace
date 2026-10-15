"""The year-judge packet (scripts/year_judge_packet.py), on synthetic claims and inputs.

    .\\.venv\\Scripts\\python.exe -m tests.test_year_judge_packet
"""

import csv
import os
import shutil
import tempfile
import unittest

from hallutrace import agreement as ag
from scripts import year_judge_packet as yj

CLAIM_COLS = ["generation_id", "model", "sample_index", "unit_id", "name", "kind", "label",
              "atom", "stated_years", "own_qid", "wikidata_years", "wikidata_match",
              "namesake_qids", "namesake_year_qids", "conflation_candidate"]


def claim(i, right, sample_index="1", wikidata="1950", model="m1"):
    return {"generation_id": "%s|biographies-%05d|%s" % (model, i, sample_index), "model": model,
            "sample_index": sample_index, "unit_id": "%s-b%05d" % (model, i % 3), "name": "P%d" % i,
            "kind": "birth", "label": "supported" if right else "hallucinated",
            "atom": "P%d was born in %s." % (i, "1950" if right else "1960"),
            "stated_years": "1950" if right else "1960", "own_qid": "Q%d" % i,
            "wikidata_years": wikidata, "wikidata_match": str(right),
            "namesake_qids": "", "namesake_year_qids": "", "conflation_candidate": "False"}


def population():
    rows = [claim(i, i % 2 == 0) for i in range(120)]
    rows += [claim(1000 + i, True, sample_index="0") for i in range(30)]      # greedy: out
    rows += [claim(2000 + i, False, wikidata="") for i in range(30)]          # no Wikidata: out
    return rows


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def write_inputs(self, claims):
        by = {}
        for c in claims:
            by.setdefault((c["model"], c["unit_id"]), []).append(c)
        for (model, unit), cs in by.items():
            os.makedirs(os.path.join(self.tmp, "inputs", model), exist_ok=True)
            with open(os.path.join(self.tmp, "inputs", model, unit + ".csv"), "w",
                      encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["name", "response", "model", "sample_index",
                                                  "generation_id"])
                w.writeheader()
                for c in cs:
                    w.writerow({"name": c["name"], "response": "Response of " + c["generation_id"],
                                "model": model, "sample_index": c["sample_index"],
                                "generation_id": c["generation_id"]})


class TestDraw(Base):
    def test_balanced_and_filtered(self):
        picked, pop = yj.draw(population())
        self.assertEqual(len(picked), 2 * yj.N_EACH)
        self.assertEqual(sum(c["wikidata_match"] == "True" for c in picked), yj.N_EACH)
        self.assertTrue(all(c["sample_index"] != "0" and c["wikidata_years"] for c in picked))
        self.assertEqual(pop, {"right": 60, "wrong": 60})

    def test_deterministic_and_shuffled(self):
        a, _ = yj.draw(population())
        b, _ = yj.draw(population())
        self.assertEqual([c["generation_id"] for c in a], [c["generation_id"] for c in b])
        first = [c["wikidata_match"] for c in a[:yj.N_EACH]]
        self.assertNotEqual(set(first), {"True"})   # right and wrong are mixed

    def test_repeated_atoms_are_one_decision(self):
        rows = population()
        dup = dict(rows[0])
        rows = rows + [dup, dict(dup)]          # the same year atom twice more in one response
        pop = yj.population(rows)
        keys = [(c["generation_id"], c["kind"], c["stated_years"]) for c in pop]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(pop), 120)

    def test_too_small(self):
        with self.assertRaises(ValueError):
            yj.draw([claim(i, True) for i in range(10)])


class TestPacket(Base):
    def build(self):
        claims = population()
        self.write_inputs(claims)
        picked, pop = yj.draw(claims)
        sheet, key = yj.rows(picked, yj.responses(self.tmp, picked))
        out = os.path.join(self.tmp, "packet")
        yj.write(out, sheet, key, yj.manifest(picked, pop))
        return out, sheet, key

    def test_sheet_hides_the_verdict(self):
        _, sheet, key = self.build()
        self.assertEqual(list(sheet[0]), yj.SHEET_COLUMNS)
        for col in ("verifier_label", "wikidata_years", "wikidata_match", "stated_years"):
            self.assertNotIn(col, sheet[0])
        for s, k in zip(sheet, key):
            self.assertEqual(s["item_id"], k["item_id"])
            self.assertEqual(k["flagged"], "1" if k["verifier_label"] == "hallucinated" else "0")

    def test_refuses_overwrite(self):
        out, sheet, key = self.build()
        with self.assertRaises(FileExistsError):
            yj.write(out, sheet, key, {})

    def test_missing_input_file(self):
        picked, _ = yj.draw(population())
        with self.assertRaises(FileNotFoundError):
            yj.responses(self.tmp, picked)

    def test_agreement_scores_it_with_one_annotator(self):
        out, sheet, key = self.build()
        lab = os.path.join(out, "annotator_a.csv")
        with open(lab, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=yj.SHEET_COLUMNS)
            w.writeheader()
            for s, k in zip(sheet, key):
                w.writerow(dict(s, annotator_label=k["verifier_label"]))   # gold = the judge
        res = ag.compute(ag.load(out, lab, lab))
        v = res["all_rows"]["verifier"]
        self.assertEqual(v["n"], 2 * yj.N_EACH)
        self.assertEqual(v["kappa"], 1.0)
        self.assertEqual(v["gate"], "accept")


if __name__ == "__main__":
    unittest.main()
