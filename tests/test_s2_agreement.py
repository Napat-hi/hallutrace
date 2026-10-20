"""First-two-wrong-sample agreement and its chance baseline (scripts/s2_agreement.py), synthetic.

    .\\.venv\\Scripts\\python.exe -m tests.test_s2_agreement
"""

import unittest

from scripts import s2_agreement as sa


def yc(i, years, pid, kind="birth", own="1869", model="m"):
    return {"generation_id": "%s|%s|%d" % (model, pid, i), "model": model, "sample_index": str(i),
            "name": pid, "kind": kind, "stated_years": years, "wikidata_years": own}


CLAIMS = (
    # A: S2, first two wrong samples agree
    [yc(1, "1870", "p1"), yc(2, "1870", "p1"), yc(3, "1870", "p1")]
    # B: S3, first two differ
    + [yc(1, "1871", "p2"), yc(2, "1872", "p2")]
    # C: S1, left out
    + [yc(1, "1869", "p3"), yc(2, "1870", "p3")]
    # D: S3, first two differ
    + [yc(1, "1870", "p4"), yc(2, "1871", "p4")]
    # E: S2 on death, its own baseline group
    + [yc(1, "1901", "p5", kind="death", own="1900"), yc(2, "1901", "p5", kind="death", own="1900")]
    # F: single mention, left out
    + [yc(3, "1870", "p6")]
)


class TestAgreement(unittest.TestCase):
    def test_counts(self):
        r = sa.agreement(CLAIMS)
        self.assertEqual(r["s2_s3_slots"], 4)
        self.assertEqual(r["first_two_wrong_agree"], 2)
        self.assertEqual(r["first_two_wrong_agree_share"], 0.5)

    def test_baseline_pairs_within_model_and_kind(self):
        r = sa.agreement(CLAIMS)
        # birth: A, B, D give 6 ordered pairs; D's first with A's second and B's first with
        # D's second agree. death has one slot, so no pair.
        self.assertEqual(r["baseline_pairs"], 6)
        self.assertEqual(r["baseline_pairs_agree"], 2)

    def test_right_year_makes_a_sample_right(self):
        seqs = sa.wrong_sequences([yc(1, "1869 1870", "p1"), yc(2, "1870", "p1")])
        self.assertEqual(seqs[("m", "p1", "birth")], [(1870,)])


if __name__ == "__main__":
    unittest.main()
