"""Tests for hallutrace/fetch_sampling.py. Pure functions: no network, no files.

Plain unittest. Every random draw is seeded, so the statistical checks are
deterministic: they pass or fail the same way on every run.
"""

import time
import unittest

from hallutrace.fetch_sampling import (
    BUCKETS,
    CAP_FULL,
    SAMPLE_N,
    SEED,
    choose_pilot,
    derive_seed,
    select_positions,
)

# 11 uneven shards summing to 3,100,000: the scale of "Charles Darwin" on OLMo 2.
BIG_SIZES = [1, 250000, 0, 400000, 9999, 700000, 5000, 600000, 100000, 1000000, 35000]
BIG_SEGMENTS = []
_start = 0
for _n in BIG_SIZES:
    BIG_SEGMENTS.append([_start, _start + _n])
    _start += _n + 7  # ranks need not start at 0 after the first shard


def segs(*sizes, start=0):
    """[start, start+size) per shard."""
    out = []
    for n in sizes:
        out.append([start, start + n])
    return out


class DeriveSeedTests(unittest.TestCase):
    def test_stable_value(self):
        # Hardcoded: a change here means seeds (and so every stored sample)
        # differ between versions or processes.
        self.assertEqual(
            derive_seed(20260929, "v4_olmo-mix-1124_llama", "Oberto Pallavicino"),
            3111643985862812885,
        )

    def test_sensitive_to_each_part(self):
        base = derive_seed(1, "idx", "Name")
        self.assertNotEqual(base, derive_seed(2, "idx", "Name"))
        self.assertNotEqual(base, derive_seed(1, "idx2", "Name"))
        self.assertNotEqual(base, derive_seed(1, "idx", "Name2"))
        self.assertNotEqual(base, derive_seed(1, "idx"))

    def test_part_boundaries_do_not_collide(self):
        self.assertNotEqual(derive_seed(1, "ab", "c"), derive_seed(1, "a", "bc"))

    def test_non_ascii(self):
        self.assertEqual(derive_seed(1, "Cederschiöld"), derive_seed(1, "Cederschiöld"))
        self.assertNotEqual(derive_seed(1, "Cederschiöld"), derive_seed(1, "Cederschiold"))


class SelectPositionsTests(unittest.TestCase):
    def test_deterministic_under_seed(self):
        a = select_positions(BIG_SEGMENTS, [], CAP_FULL, SAMPLE_N, 123)
        b = select_positions(BIG_SEGMENTS, [], CAP_FULL, SAMPLE_N, 123)
        self.assertEqual(a, b)

    def test_different_across_seeds(self):
        a = select_positions(BIG_SEGMENTS, [], CAP_FULL, SAMPLE_N, 123)
        b = select_positions(BIG_SEGMENTS, [], CAP_FULL, SAMPLE_N, 124)
        self.assertNotEqual(a, b)

    def test_sample_of_derived_seed_is_reproducible(self):
        seed = derive_seed(SEED, "v4_olmo-mix-1124_llama", "Charles Darwin")
        self.assertEqual(
            select_positions(BIG_SEGMENTS, segs(500, 300), CAP_FULL, SAMPLE_N, seed),
            select_positions(BIG_SEGMENTS, segs(500, 300), CAP_FULL, SAMPLE_N, seed),
        )

    def test_uniform_across_uneven_shards(self):
        # Shards of 10, 1000 and 100000 occurrences: each shard's share of the
        # draws must follow its share of the pool, not be equal per shard.
        segments = segs(10, 1000, 100000)
        total = 101010
        draws = 3
        repeats = 20000
        counts = [0, 0, 0]
        rank_sum_big = 0
        n_big = 0
        for i in range(repeats):
            for form, shard, rank in select_positions(segments, [], 0, draws, i):
                counts[shard] += 1
                if shard == 2:
                    rank_sum_big += rank
                    n_big += 1
        n = repeats * draws
        expect = [n * s / total for s in (10, 1000, 100000)]
        # ~5.9, ~594, ~59400. Generous bands (several standard deviations).
        self.assertLessEqual(counts[0], 25)
        self.assertGreater(counts[1], expect[1] * 0.75)
        self.assertLess(counts[1], expect[1] * 1.25)
        self.assertGreater(counts[2], expect[2] * 0.98)
        self.assertLess(counts[2], expect[2] * 1.02)
        # Ranks inside the big shard are uniform too: mean near the midpoint.
        mean_rank = rank_sum_big / n_big
        self.assertGreater(mean_rank, 100000 / 2 * 0.98)
        self.assertLess(mean_rank, 100000 / 2 * 1.02)

    def test_every_position_is_reachable(self):
        # Kills an off-by-one in the draw range: with range(total - 1) the
        # last-ranked occurrence of the last shard could never be sampled.
        for plain, glued in (
            ([[0, 3], [10, 14]], [[5, 7]]),      # the last-ranked occurrence is a glued one
            ([[0, 3], [10, 14], [20, 22]], []),  # ... or a plain one
            ([[0, 0], [4, 9]], [[0, 0]]),        # empty shards at both ends of the pool
        ):
            everything = select_positions(plain, glued, 1000, SAMPLE_N, 0)
            total = len(everything)
            self.assertGreater(total, 4)
            reached = set()
            for seed in range(3000):
                reached.update(select_positions(plain, glued, 0, 3, seed))
            self.assertEqual(reached, set(everything), (plain, glued))
            last_form, last_shard, _ = everything[-1]
            self.assertIn(everything[-1], reached)
            self.assertIn(everything[0], reached)

    def test_combined_plain_and_glued_pool(self):
        plain = [[100, 400], [50, 350]]  # 600 plain
        glued = [[7, 207], [0, 0], [30, 330]]  # 500 glued, one empty shard
        for seed in range(5):
            pos = select_positions(plain, glued, CAP_FULL, SAMPLE_N, seed)
            self.assertEqual(len(pos), SAMPLE_N)
            forms = {f for f, _, _ in pos}
            self.assertEqual(forms, {"plain", "glued"})
            for form, shard, rank in pos:
                seg = (plain if form == "plain" else glued)[shard]
                self.assertTrue(seg[0] <= rank < seg[1], (form, shard, rank))
                self.assertNotEqual(seg[0], seg[1], "drew from an empty shard")

    def test_form_share_follows_pool_share(self):
        # 900 plain vs 100 glued: about 90% of a large number of draws are plain.
        plain = segs(900)
        glued = segs(100)
        n_plain = 0
        n = 0
        for seed in range(400):
            for form, _, _ in select_positions(plain, glued, 0, 10, seed):
                n += 1
                n_plain += form == "plain"
        self.assertGreater(n_plain / n, 0.87)
        self.assertLess(n_plain / n, 0.93)

    def test_shard_index_refers_to_own_forms_list(self):
        # Plain has 1 shard, glued has 3: glued shard indices go up to 2, not
        # offset by the plain list length.
        plain = [[0, 5]]
        glued = [[0, 0], [0, 0], [10, 15]]
        pos = select_positions(plain, glued, CAP_FULL, SAMPLE_N, 1)
        self.assertEqual(
            pos,
            [("plain", 0, r) for r in range(0, 5)] + [("glued", 2, r) for r in range(10, 15)],
        )

    def test_all_returned_when_at_or_below_cap(self):
        plain = [[3, 10], [0, 4]]
        glued = [[20, 22]]
        want = (
            [("plain", 0, r) for r in range(3, 10)]
            + [("plain", 1, r) for r in range(0, 4)]
            + [("glued", 0, r) for r in range(20, 22)]
        )
        self.assertEqual(select_positions(plain, glued, CAP_FULL, SAMPLE_N, 9), want)
        self.assertEqual(len(want), 13)

    def test_exactly_cap_returns_all(self):
        pos = select_positions(segs(120, 50), segs(30), CAP_FULL, SAMPLE_N, 4)
        self.assertEqual(len(pos), CAP_FULL)
        self.assertEqual(len(set(pos)), CAP_FULL)

    def test_one_over_cap_returns_sample(self):
        pos = select_positions(segs(120, 50), segs(31), CAP_FULL, SAMPLE_N, 4)
        self.assertEqual(len(pos), SAMPLE_N)
        self.assertEqual(len(set(pos)), SAMPLE_N)

    def test_sorted_plain_before_glued(self):
        pos = select_positions(segs(500, 500), segs(500, 500), CAP_FULL, SAMPLE_N, 3)
        keys = [({"plain": 0, "glued": 1}[f], s, r) for f, s, r in pos]
        self.assertEqual(keys, sorted(keys))
        # "glued" < "plain" alphabetically, so this fails if the sort is by string.
        first_glued = next(i for i, p in enumerate(pos) if p[0] == "glued")
        self.assertTrue(all(p[0] == "plain" for p in pos[:first_glued]))

    def test_no_duplicates_ever(self):
        # A pool barely above the cap makes collisions likely if sampling were
        # with replacement.
        for seed in range(200):
            pos = select_positions(segs(100, 101), [], CAP_FULL, SAMPLE_N, seed)
            self.assertEqual(len(set(pos)), len(pos))
            self.assertEqual(len(pos), SAMPLE_N)

    def test_huge_counts_are_fast(self):
        self.assertEqual(sum(BIG_SIZES), 3100000)
        t0 = time.perf_counter()
        pos = select_positions(BIG_SEGMENTS, segs(1000, 200000), CAP_FULL, SAMPLE_N, 77)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 1.0)
        self.assertEqual(len(pos), SAMPLE_N)
        self.assertEqual(len(set(pos)), SAMPLE_N)
        for form, shard, rank in pos:
            seg = (BIG_SEGMENTS if form == "plain" else segs(1000, 200000))[shard]
            self.assertTrue(seg[0] <= rank < seg[1])

    def test_empty_pool(self):
        self.assertEqual(select_positions([], [], CAP_FULL, SAMPLE_N, 1), [])
        self.assertEqual(select_positions([[0, 0], [5, 5]], [[3, 3]], CAP_FULL, SAMPLE_N, 1), [])

    def test_sample_n_above_total_returns_all(self):
        pos = select_positions(segs(30), [], 10, 100, 1)
        self.assertEqual(pos, [("plain", 0, r) for r in range(30)])

    def test_malformed_segment_raises(self):
        with self.assertRaises(ValueError):
            select_positions([[10, 5]], [], CAP_FULL, SAMPLE_N, 1)
        with self.assertRaises(ValueError):
            select_positions([[0, 5]], [[9, 2]], CAP_FULL, SAMPLE_N, 1)
        with self.assertRaises(ValueError):
            select_positions([[1, 2, 3]], [], CAP_FULL, SAMPLE_N, 1)


class ChoosePilotTests(unittest.TestCase):
    @staticmethod
    def counts():
        # 12 names in each of the five buckets, plus a bucket of only 2 names
        # is exercised separately.
        out = {}
        for b, (lo, hi) in enumerate(BUCKETS):
            for i in range(12):
                cnt = lo + i if hi is None else lo + (hi - lo) * i // 11
                out["b%d_name%02d" % (b, i)] = cnt
        return out

    @staticmethod
    def bucket(name):
        return int(name[1])

    def test_covers_all_five_buckets(self):
        pilot = choose_pilot(self.counts(), 5)
        self.assertEqual(len(pilot), 20)
        self.assertEqual({self.bucket(n) for n in pilot}, {0, 1, 2, 3, 4})

    def test_respects_per_bucket(self):
        pilot = choose_pilot(self.counts(), 5, per_bucket=3)
        self.assertEqual(len(pilot), 15)
        for b in range(5):
            self.assertEqual(sum(self.bucket(n) == b for n in pilot), 3)

    def test_bucket_boundaries_are_inclusive(self):
        c = {"zero": 0, "one": 1, "ten": 10, "eleven": 11, "hundred": 100,
             "h1": 101, "thousand": 1000, "t1": 1001, "big": 3100000}
        pilot = choose_pilot(c, 1, per_bucket=10)
        self.assertEqual(
            pilot,
            ["zero", "one", "ten", "eleven", "hundred", "h1", "thousand", "big", "t1"],
        )

    def test_takes_all_of_a_small_bucket(self):
        c = self.counts()
        for k in [k for k in c if k.startswith("b0_")][2:]:
            del c[k]  # bucket 0 keeps only two names
        pilot = choose_pilot(c, 5)
        self.assertEqual([n for n in pilot if self.bucket(n) == 0], ["b0_name00", "b0_name01"])
        self.assertEqual(len(pilot), 18)

    def test_empty_bucket_is_skipped(self):
        c = {k: v for k, v in self.counts().items() if not k.startswith("b2_")}
        pilot = choose_pilot(c, 5)
        self.assertEqual({self.bucket(n) for n in pilot}, {0, 1, 3, 4})
        self.assertEqual(len(pilot), 16)

    def test_deterministic_and_seed_sensitive(self):
        self.assertEqual(choose_pilot(self.counts(), 5), choose_pilot(self.counts(), 5))
        self.assertNotEqual(choose_pilot(self.counts(), 5), choose_pilot(self.counts(), 6))

    def test_independent_of_dict_order(self):
        c = self.counts()
        reordered = dict(reversed(list(c.items())))
        self.assertNotEqual(list(c), list(reordered))
        self.assertEqual(choose_pilot(c, 5), choose_pilot(reordered, 5))

    def test_sorted_by_bucket_then_name(self):
        pilot = choose_pilot(self.counts(), 5)
        self.assertEqual(pilot, sorted(pilot, key=lambda n: (self.bucket(n), n)))

    def test_negative_count_raises(self):
        with self.assertRaises(ValueError):
            choose_pilot({"x": -1}, 5)


if __name__ == "__main__":
    unittest.main()
