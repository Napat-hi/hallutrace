"""Subsample check: subsets use stability's rule, overturns, held-out sample and greedy
pass, exact-year baseline and its two counting units, the CLI. Synthetic claims, no network."""
import csv
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hallutrace import stability as st  # noqa: E402
from hallutrace import subsample as ss  # noqa: E402


def claim(model, pid, kind, idx, years, own="1900", n=0):
    return {"generation_id": "%s|%s|%d" % (model, pid, idx), "model": model, "sample_index": str(idx),
            "unit_id": "u", "name": pid, "kind": kind, "label": "", "atom": "a%d" % n,
            "stated_years": years, "own_qid": "Q1", "wikidata_years": own, "wikidata_match": "",
            "namesake_qids": "", "namesake_year_qids": "", "conflation_candidate": ""}


def fixture():
    """p1: S1 with one right sample (5) and wrong 1901 x3: S2 on any subset without sample 5.
    p2: S3 on five (1950, 1960, 1970, 1980), greedy right. p3: never wrong, right in 1 and 2,
    greedy right. p4: single mention (1899), with a second atom repeating it in sample 1.
    p5: no Wikidata year, left out everywhere."""
    c = []
    for i in (1, 2, 3):
        c.append(claim("m", "p1", "birth", i, "1901"))
    c.append(claim("m", "p1", "birth", 5, "1900"))
    c.append(claim("m", "p1", "birth", 0, "1901"))
    for i, y in zip((1, 2, 3, 4), ("1950", "1960", "1970", "1980")):
        c.append(claim("m", "p2", "birth", i, y))
    c.append(claim("m", "p2", "birth", 0, "1900"))
    c.append(claim("m", "p3", "birth", 1, "1900"))
    c.append(claim("m", "p3", "birth", 2, "1900"))
    c.append(claim("m", "p3", "birth", 0, "1900"))
    c.append(claim("m", "p4", "birth", 1, "1899", n=0))
    c.append(claim("m", "p4", "birth", 1, "1899", n=1))
    c.append(claim("m", "p5", "birth", 1, "1800", own=""))
    return c


class TestSubsample(unittest.TestCase):
    def setUp(self):
        self.c = fixture()

    def test_five_samples_equal_stability(self):
        mine = ss.regimes(self.c, ss.SAMPLED)
        theirs = {ss.slot_key(r): r["regime"] for r in st.slot_items(self.c) if r["regime"]}
        self.assertEqual({k: r["regime"] for k, r in mine.items()}, theirs)
        self.assertEqual(theirs, {("m", "p1", "birth"): "S1", ("m", "p2", "birth"): "S3",
                                  ("m", "p3", "birth"): "S1", ("m", "p4", "birth"): st.SINGLE_MENTION})

    def test_subset_turns_s1_into_s2_and_counts_overturn(self):
        rows = ss.regimes(self.c, (1, 2, 3, 4))
        self.assertEqual(rows[("m", "p1", "birth")]["regime"], "S2")
        four = [r for r in ss.subset_table(self.c) if r["indexes"] == "1 2 3 4"][0]
        self.assertEqual(four["S2_S3_overturned"], 1)   # p1 is S1 on five
        self.assertEqual(four["S2_S3"], 2)               # p1 S2, p2 S3

    def test_table_shape_and_greedy_row(self):
        t = ss.subset_table(self.c)
        self.assertEqual([r["samples"] for r in t].count(3), 10)
        self.assertEqual([r["samples"] for r in t].count(4), 5)
        six = [r for r in t if r["samples"] == 6][0]
        self.assertTrue(six["indexes"].endswith("greedy"))
        five = [r for r in t if r["samples"] == 5][0]
        self.assertIsNone(five["S2_S3_overturned"])     # 0 by construction, left blank
        self.assertIsNone(six["overturn_rate"])
        self.assertEqual(five["share_inference"], 0.5)   # p1 at inference, p2 S3

    def test_held_out_sample(self):
        h = ss.held_out_sample(self.c)
        # p1 is S2 only on 1-4, and the left-out 5 is right; p2 is S3 on all five subsets,
        # left-out 5 silent, left-out 1 to 4 wrong
        self.assertEqual((h["slot_subsets"], h["right"], h["wrong"], h["silent"]), (6, 1, 4, 1))

    def test_held_out_greedy_uses_greedy_state(self):
        g = {r["class"]: r for r in ss.held_out_greedy(self.c)}
        self.assertEqual(g["S1 never wrong"]["greedy_right"], 1)                     # p3
        self.assertEqual(g["S1 with a wrong sample (at inference)"]["greedy_wrong"], 1)  # p1
        self.assertEqual(g["S2 or S3"]["greedy_right"], 1)                             # p2
        self.assertEqual(g["single mention"]["greedy_silent"], 1)                      # p4

    def test_exact_year_units(self):
        rows = {(r["scope"], r["unit"]): r for r in ss.exact_year_baseline(self.c)}
        claim_all = rows[("all slots with a stated year", "claim")]
        sample_all = rows[("all slots with a stated year", "sample")]
        self.assertEqual(claim_all["exact"], 3)          # p1 s5, p3 s1, p3 s2
        self.assertEqual(claim_all["one_above"], 3)      # 1901 x3
        self.assertEqual(claim_all["one_below"], 2)      # 1899, two atoms
        self.assertEqual(sample_all["one_below"], 1)     # same generation counts once
        self.assertEqual(claim_all["slots"], 4)          # p5 has no Wikidata year

    def test_cli_writes_and_refuses_overwrite(self):
        from scripts import subsample_check as cli
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "claims.csv")
            with open(path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.c[0]))
                w.writeheader()
                w.writerows(self.c)
            out = os.path.join(d, "out")
            self.assertEqual(cli.main(["--claims", path, "--out", out]), 0)
            self.assertEqual(sorted(os.listdir(out)), sorted(cli.FILES))
            self.assertEqual(cli.main(["--claims", path, "--out", out]), 1)
            self.assertEqual(cli.main(["--claims", path, "--out", out, "--force"]), 0)


if __name__ == "__main__":
    unittest.main()
