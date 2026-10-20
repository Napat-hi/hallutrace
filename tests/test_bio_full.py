"""hallutrace.bio_full: grouping Biographies units and splitting a group's output back.

No Modal, no GPU: builds small generation batches from real Biographies prompt ids,
fakes what verify_biographies writes, and checks every unit ends with the file a one-unit
call would have given. Run: python -m tests.test_bio_full
"""
import csv
import json
import os
import shutil
import tempfile
import unittest

from hallutrace import verifier_inputs as bx
from hallutrace import bio_full as bf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = bx.load_prompts(os.path.join(REPO, "data", "prompts.csv"))
BIO_IDS = sorted(p for p in PROMPTS if PROMPTS[p].task == "biographies")


def _record(model, pid, si, response):
    return {"generation_id": "%s|%s|%d" % (model, pid, si), "prompt_id": pid, "model": model,
            "sample_index": si, "response": response}


class BioFullTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="bio_full_test-")
        self.gen = os.path.join(self.root, "generations")
        self.out = os.path.join(self.root, "verifier")
        d = os.path.join(self.gen, "m")
        os.makedirs(d)
        # b0 and b1: sample 0; b2: sample 1, reusing b0's people
        batches = {"m-b00000": (0, BIO_IDS[:3]), "m-b00001": (0, BIO_IDS[3:6]),
                   "m-b00002": (1, BIO_IDS[:3])}
        for bid, (si, ids) in batches.items():
            with open(os.path.join(d, bid + ".jsonl"), "w", encoding="utf-8") as f:
                for k, pid in enumerate(ids):
                    text = "" if (bid == "m-b00001" and k == 0) else "Born in 1900. Died in 1950."
                    f.write(json.dumps(_record("m", pid, si, text)) + "\n")

    def tearDown(self):
        shutil.rmtree(self.root)

    def _units(self):
        return [u for u in bx.pending(self.gen, self.out, "") if u.task == bx.BIOGRAPHIES]

    def test_groups_never_mix_sample_index(self):
        groups = bf.group_units(self._units(), per_call=4)
        self.assertEqual([[u.unit_id for u in g] for g in groups],
                         [["m-b00000", "m-b00001"], ["m-b00002"]])
        self.assertEqual(len(bf.group_units(self._units(), per_call=1)), 3)

    def test_split_matches_units_and_clears_pending(self):
        for g in bf.group_units(self._units(), per_call=4):
            preps = [bx.prepare(u, PROMPTS) for u in g]
            inp, out = bf.group_paths(self.out, g)
            n = bf.write_group_input(preps, inp)
            rows = list(csv.DictReader(open(inp, encoding="utf-8")))
            self.assertEqual(n, len(rows))
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name", "response", "atoms", "hallucinated atoms", "model", "sample_index"])
                for r in rows:
                    w.writerow([r["name"], r["response"], "x", "", r["model"], r["sample_index"]])
            written = bf.split_output(out, preps, self.out)
            for p in preps:
                self.assertEqual(written[p.unit.unit_id], len(p.rows))
                got = list(csv.DictReader(open(bx.output_path(self.out, p.unit), encoding="utf-8")))
                self.assertEqual([r["name"] for r in got], [r["name"] for r in p.rows])
        self.assertEqual(self._units(), [])

    def test_empty_response_is_not_sent(self):
        g = bf.group_units(self._units(), per_call=4)[0]
        preps = [bx.prepare(u, PROMPTS) for u in g]
        self.assertEqual(sum(len(p.rows) for p in preps), 5)  # 6 records, one empty

    def test_group_with_nothing_to_score(self):
        u = self._units()[0]
        prep = bx.prepare(u, PROMPTS)
        prep.rows = []
        self.assertEqual(bf.split_output(None, [prep], self.out), {u.unit_id: 0})
        self.assertEqual(os.path.getsize(bx.output_path(self.out, u)), 0)

    def test_repeated_name_refused(self):
        units = self._units()
        preps = [bx.prepare(u, PROMPTS) for u in (units[0], units[2])]  # same people, two samples
        with self.assertRaises(ValueError):
            bf.write_group_input(preps, os.path.join(self.root, "x.csv"))

    def test_unknown_output_name_refused(self):
        g = bf.group_units(self._units(), per_call=4)[1]
        preps = [bx.prepare(u, PROMPTS) for u in g]
        path = os.path.join(self.root, "bad.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows([["name", "response"], ["Nobody At All", "x"]])
        with self.assertRaises(ValueError):
            bf.split_output(path, preps, self.out)


if __name__ == "__main__":
    unittest.main()
