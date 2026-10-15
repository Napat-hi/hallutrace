"""Tests for hallutrace/pilot_counts.py. Pure: rows are plain dicts, no SQLite, no files.

Plain unittest. Fixtures use invented text and the real hallutrace.fact_match.
Run: python -m tests.test_pilot_counts   or   python -m unittest tests.test_pilot_counts
"""

import json
import unittest

from hallutrace import pilot_counts as pc

IDX = "v4_olmo-mix-1124_llama"
IDX2 = "v4_dclm-baseline_llama"
EM_DASH = chr(0x2014)
YEARS = {"Ada Zed": {"birth": [1850], "death": [1900]}}


class Fx:
    """Builds entities / finds / plan / windows rows for a few invented strings."""

    def __init__(self):
        self.entities, self.finds, self.plan, self.windows = [], [], [], []

    def entity(self, name, q, source="name", active=1, amb_name=0, amb_stripped=0):
        self.entities.append({
            "prompt_name": name, "query_string": q, "source": source, "qid": "Q1",
            "ambiguous_name": amb_name, "ambiguous_stripped": amb_stripped, "active": active})

    def string(self, q, index=IDX, pool=None, selected=None, wins=(), plan_extra=0, glued=True):
        """One (index, string): finds rows, plan rows (one per win plus `plan_extra`
        never fetched) and windows rows. Each win is a dict: before, needle, after,
        doc_ix, meta, form."""
        pool = len(wins) + plan_extra if pool is None else pool
        self.finds.append({"index_name": index, "query_string": q, "form": "plain",
                           "cnt": pool, "segment_by_shard_json": json.dumps([[0, pool]])})
        if glued:
            self.finds.append({"index_name": index, "query_string": q, "form": "glued",
                               "cnt": 0, "segment_by_shard_json": json.dumps([[0, 0]])})
        if pool == 0:
            return  # a true zero: no plan, no windows
        selected = selected or ("all" if pool <= pc.CAP_FULL else "sample")
        for rank in range(len(wins) + plan_extra):
            self.plan.append({"index_name": index, "query_string": q, "form": "plain",
                              "s": 0, "rank": rank, "selected": selected})
        for rank, w in enumerate(wins):
            self.windows.append(self.window(q, index, rank, **w))

    @staticmethod
    def window(q, index, rank, before="", needle=None, after="", doc_ix=None, meta=None,
               form="plain", s=0):
        return {
            "index_name": index, "query_string": q, "form": form, "s": s, "rank": rank,
            "doc_ix": rank if doc_ix is None else doc_ix, "doc_len": 1000, "disp_len": 100,
            "needle_offset": 10,
            "spans_json": json.dumps([[before, None], [needle or q, "0"], [after, None]]),
            "metadata_json": meta,
        }

    def grid(self, names, years=YEARS, indexes=(IDX,)):
        return pc.count_grid(self.entities, self.finds, self.plan, self.windows, years,
                             names, list(indexes))


def cell(rows, name, index=IDX, fact="birth", width="full", matcher="bare", strings="all",
         year=None):
    got = [r for r in rows if r["prompt_name"] == name and r["index"] == index
           and r["fact"] == fact and r["half_width"] == width and r["matcher"] == matcher
           and r["strings"] == strings and (year is None or r["year"] == year)]
    assert len(got) == 1, "expected one row, got %d" % len(got)
    return got[0]


W_CUE = {"after": " was born in 1850 in Ely."}
W_FAR = {"after": " " + "x" * 300 + " in 1850."}
W_NONE = {"after": " nothing here."}


class GridBasicsTests(unittest.TestCase):
    def setUp(self):
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(W_CUE, doc_ix=1), dict(W_FAR, doc_ix=1), dict(W_NONE, doc_ix=2)])
        self.rows = fx.grid(["Ada Zed"])

    def test_csv_columns_are_the_row_keys(self):
        self.assertEqual(list(self.rows[0]), pc.CSV_COLUMNS)

    def test_row_count_and_order(self):
        # 1 name x 1 index x 2 facts x 6 widths x 3 matchers x 2 string sets
        self.assertEqual(len(self.rows), 72)
        keys = [(r["fact"], r["strings"], r["matcher"], pc.WIDTH_LABELS.index(r["half_width"]))
                for r in self.rows]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(self.rows[5]["half_width"], "full")

    def test_bare_and_cue_counts_and_docs(self):
        r = cell(self.rows, "Ada Zed")
        self.assertEqual((r["mentions_matched"], r["n_docs"], r["n_c"], r["n_c_unit"]), (2, 1, 1, "docs"))
        self.assertEqual(r["status"], "counted")
        c = cell(self.rows, "Ada Zed", matcher="cue")
        self.assertEqual((c["mentions_matched"], c["n_docs"]), (1, 1))

    def test_width_moves_the_count(self):
        self.assertEqual(cell(self.rows, "Ada Zed", width="100")["mentions_matched"], 1)
        self.assertEqual(cell(self.rows, "Ada Zed", width="200")["mentions_matched"], 1)
        self.assertEqual(cell(self.rows, "Ada Zed", width="400")["mentions_matched"], 2)

    def test_true_zero(self):
        r = cell(self.rows, "Ada Zed", fact="death")
        self.assertEqual((r["status"], r["n_c"], r["n_c_unit"], r["incomplete"]), ("zero", 0, "docs", 0))

    def test_coverage_and_pool(self):
        r = cell(self.rows, "Ada Zed")
        self.assertEqual((r["pool"], r["windows_planned"], r["windows_fetched"], r["coverage"]), (3, 3, 3, 1.0))
        self.assertEqual((r["selected"], r["short_year"]), ("all", 0))


class EstimateAndStatusTests(unittest.TestCase):
    def test_sample_is_estimated_per_string(self):
        fx = Fx()
        fx.entity("Bo Lee", "Bo Lee (singer)", "name")
        fx.entity("Bo Lee", "Bo Lee", "stripped")
        fx.string("Bo Lee (singer)", wins=[dict(W_CUE, doc_ix=1), dict(W_NONE, doc_ix=2)])
        sampled = [dict(W_CUE, doc_ix=10), dict(W_CUE, doc_ix=11)] + [dict(W_NONE, doc_ix=20 + i) for i in range(8)]
        fx.string("Bo Lee", pool=1000, wins=sampled)
        years = {"Bo Lee": {"birth": [1850], "death": []}}
        rows = fx.grid(["Bo Lee"], years)
        r = cell(rows, "Bo Lee")
        # name string is all: 1 match counted as is; stripped is sample: 2 / 10 x 1000 = 200
        self.assertEqual(r["est_mentions"], 201.0)
        self.assertEqual((r["selected"], r["status"], r["n_c"], r["n_c_unit"]), ("sample", "counted", 201.0, "est_mentions"))
        self.assertEqual((r["mentions_matched"], r["n_docs"], r["matched_name"], r["matched_stripped"]), (3, 3, 1, 2))
        self.assertEqual(r["overlap"], 1)
        detail = json.loads(r["per_string_json"])
        self.assertEqual([d["matched"] for d in detail], [2, 1] if detail[0]["q"] == "Bo Lee" else [1, 2])
        # no_stripped keeps the name string only, which is all and exact
        n = cell(rows, "Bo Lee", strings="no_stripped")
        self.assertEqual((n["n_c"], n["n_c_unit"], n["selected"], n["overlap"], n["n_strings"]), (1, "docs", "all", 0, 1))

    def test_unknown_only_if_every_string_has_zero_matches(self):
        fx = Fx()
        fx.entity("Cy Dun", "Cy Dun (poet)")
        fx.entity("Cy Dun", "Cy Dun", "stripped")
        fx.string("Cy Dun (poet)", wins=[dict(W_NONE), dict(W_NONE)])
        fx.string("Cy Dun", pool=1000, wins=[dict(W_NONE, doc_ix=i) for i in range(10)])
        years = {"Cy Dun": {"birth": [1850], "death": []}}
        rows = fx.grid(["Cy Dun"], years)
        r = cell(rows, "Cy Dun")
        self.assertEqual((r["status"], r["n_c"], r["n_c_unit"]), ("unknown", 0.0, "est_mentions"))
        # without the sampled string the rest is fully fetched: a true zero
        self.assertEqual(cell(rows, "Cy Dun", strings="no_stripped")["status"], "zero")

    def test_unfetched_planned_window_is_incomplete_never_zero(self):
        fx = Fx()
        fx.entity("Di Fay", "Di Fay")
        fx.string("Di Fay", wins=[dict(W_NONE)] * 3, plan_extra=2)
        years = {"Di Fay": {"birth": [1850], "death": []}}
        r = cell(fx.grid(["Di Fay"], years), "Di Fay")
        self.assertEqual((r["status"], r["incomplete"], r["coverage"]), ("unknown", 1, 0.6))
        self.assertEqual((r["windows_planned"], r["windows_fetched"]), (5, 3))

    def test_incomplete_with_a_match_is_counted_in_est_mentions(self):
        fx = Fx()
        fx.entity("Di Fay", "Di Fay")
        fx.string("Di Fay", wins=[dict(W_CUE), dict(W_NONE)], plan_extra=2)
        years = {"Di Fay": {"birth": [1850], "death": []}}
        r = cell(fx.grid(["Di Fay"], years), "Di Fay")
        self.assertEqual((r["status"], r["n_c_unit"], r["n_c"], r["incomplete"]), ("counted", "est_mentions", 1.0, 1))

    def test_string_without_finds_is_not_fetched_not_zero(self):
        fx = Fx()
        fx.entity("Ed Ng", "Ed Ng")
        years = {"Ed Ng": {"birth": [1850], "death": []}}
        r = cell(fx.grid(["Ed Ng"], years), "Ed Ng")
        self.assertEqual((r["status"], r["incomplete"], r["not_fetched_strings"]), ("unknown", 1, 1))

    def test_no_strings_after_no_stripped_does_not_crash(self):
        fx = Fx()
        fx.entity("Fo Ox", "Fo Ox", "stripped")
        fx.string("Fo Ox", wins=[dict(W_CUE)])
        years = {"Fo Ox": {"birth": [1850], "death": []}}
        rows = fx.grid(["Fo Ox"], years)
        r = cell(rows, "Fo Ox", strings="no_stripped")
        self.assertEqual((r["status"], r["n_strings"], r["incomplete"], r["n_c"]), ("unknown", 0, 1, ""))
        self.assertEqual(cell(rows, "Fo Ox")["status"], "counted")

    def test_pool_zero_is_a_true_zero_but_inactive_strings_are_only_counted(self):
        fx = Fx()
        fx.entity("Gi Ho", "Gi Ho")
        fx.entity("Gi Ho", "Gi Ho Alias", "alias", active=0)
        fx.string("Gi Ho", pool=0)
        years = {"Gi Ho": {"birth": [1850], "death": []}}
        r = cell(fx.grid(["Gi Ho"], years), "Gi Ho")
        self.assertEqual((r["status"], r["incomplete"], r["pool"], r["n_strings"], r["n_inactive_strings"]),
                         ("zero", 0, 0, 1, 1))

    def test_pool_from_segments_mismatch_and_glued_missing(self):
        fx = Fx()
        fx.entity("Hu Ip", "Hu Ip")
        fx.finds.append({"index_name": IDX, "query_string": "Hu Ip", "form": "plain", "cnt": 3,
                         "segment_by_shard_json": json.dumps([[0, 3], [10, 14]])})
        fx.finds.append({"index_name": IDX, "query_string": "Hu Ip", "form": "glued", "cnt": 2,
                         "segment_by_shard_json": json.dumps([[5, 7]])})
        years = {"Hu Ip": {"birth": [1850], "death": []}}
        r = cell(fx.grid(["Hu Ip"], years), "Hu Ip")
        # 3 + 4 + 2 = 9 by segments, cnt sum 5: flagged; planned 0 with pool > 0: incomplete
        self.assertEqual((r["pool"], r["pool_mismatch"], r["glued_missing"], r["incomplete"]), (9, 1, 0, 1))
        fx2 = Fx()
        fx2.entity("Hu Ip", "Hu Ip")
        fx2.string("Hu Ip", wins=[dict(W_CUE)], glued=False)
        r2 = cell(fx2.grid(["Hu Ip"], years), "Hu Ip")
        self.assertEqual((r2["glued_missing"], r2["pool_mismatch"], r2["status"]), (1, 0, "counted"))

    def test_extra_windows_are_ignored_and_reported(self):
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(W_NONE)])
        fx.windows.append(fx.window("Ada Zed", IDX, 99, **W_CUE))  # not in plan
        r = cell(fx.grid(["Ada Zed"]), "Ada Zed")
        self.assertEqual((r["mentions_matched"], r["windows_fetched"]), (0, 1))
        d = pc.diagnostics(fx.plan, fx.windows)
        self.assertEqual((d["extra_windows"], d["plan_rows"], d["plan_unfetched"]), (1, 1, 0))

    def test_short_year_and_ambiguity_flags(self):
        fx = Fx()
        fx.entity("Ivo Jae", "Ivo Jae", amb_name=1)
        fx.string("Ivo Jae", wins=[dict(after=" born 950 here")])
        years = {"Ivo Jae": {"birth": [950], "death": []}}
        r = cell(fx.grid(["Ivo Jae"], years), "Ivo Jae")
        self.assertEqual((r["short_year"], r["ambiguous_name"], r["ambiguous_stripped"], r["mentions_matched"]), (1, 1, 0, 1))


class DocsDedupDomainTests(unittest.TestCase):
    def test_n_docs_counts_distinct_documents_across_strings(self):
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed (chef)")
        fx.entity("Ada Zed", "Ada Zed", "stripped")
        # same document 5 in both strings (and twice in one), document 6 only in one
        fx.string("Ada Zed (chef)", wins=[dict(W_CUE, doc_ix=5), dict(W_CUE, doc_ix=5)])
        fx.string("Ada Zed", wins=[dict(W_CUE, doc_ix=5), dict(W_CUE, doc_ix=6)])
        r = cell(fx.grid(["Ada Zed"]), "Ada Zed")
        self.assertEqual((r["mentions_matched"], r["n_docs"]), (4, 2))
        # the same doc_ix on another index is another document
        fx.string("Ada Zed (chef)", index=IDX2, wins=[dict(W_CUE, doc_ix=5)])
        fx.string("Ada Zed", index=IDX2, wins=[dict(W_CUE, doc_ix=5)])
        rows = fx.grid(["Ada Zed"], indexes=[IDX, IDX2])
        self.assertEqual(cell(rows, "Ada Zed", index=IDX2)["n_docs"], 1)

    def test_doc_key_with_shard(self):
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(W_CUE, doc_ix=5, s=0), dict(W_CUE, doc_ix=5, s=1)])
        fx.plan = [dict(p, s=w["s"]) for p, w in zip(fx.plan, fx.windows)]
        try:
            pc.DOC_KEY_WITH_SHARD = True
            self.assertEqual(cell(fx.grid(["Ada Zed"]), "Ada Zed")["n_docs"], 2)
            pc.DOC_KEY_WITH_SHARD = False
            self.assertEqual(cell(fx.grid(["Ada Zed"]), "Ada Zed")["n_docs"], 1)
        finally:
            pc.DOC_KEY_WITH_SHARD = False

    def test_dedup_merges_identical_text_but_not_different_years(self):
        same = " was born in 1850 in Ely and later moved, died 1900."
        other = " was born in 1850 in Ely and later moved, died 1901."
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(after=same, doc_ix=1), dict(after=same, doc_ix=2),
                                   dict(after=other, doc_ix=3)])
        r = cell(fx.grid(["Ada Zed"]), "Ada Zed")
        self.assertEqual((r["n_docs"], r["n_docs_dedup"]), (3, 2))

    def test_dedup_counts_documents_not_windows(self):
        # one document, two matched windows with different text around the year: 1 and 1
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(after=" was born in 1850 in Ely, a chef.", doc_ix=7),
                                   dict(after=" born 1850 and later a baker in Kent.", doc_ix=7)])
        r = cell(fx.grid(["Ada Zed"]), "Ada Zed")
        self.assertEqual((r["mentions_matched"], r["n_docs"], r["n_docs_dedup"]), (2, 1, 1))

    def test_dedup_merges_documents_with_identical_text(self):
        same = " was born in 1850 in Ely and later moved."
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(after=same, doc_ix=1), dict(after=same, doc_ix=2)])
        r = cell(fx.grid(["Ada Zed"]), "Ada Zed")
        self.assertEqual((r["n_docs"], r["n_docs_dedup"]), (2, 1))

    def test_dedup_never_merges_different_years(self):
        a = " was born in 1850 in Ely and later moved, died 1900."
        b = " was born in 1850 in Ely and later moved, died 1901."
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(after=a, doc_ix=1), dict(after=b, doc_ix=2)])
        r = cell(fx.grid(["Ada Zed"]), "Ada Zed")
        self.assertEqual((r["n_docs"], r["n_docs_dedup"]), (2, 2))

    def test_dedup_chains_documents_transitively(self):
        # doc 1 and doc 2 share the key of the first text, doc 2 and doc 3 share the key of the
        # second: two windows in doc 2 (one per text), so all three are one group
        t1 = " was born in 1850 in Ely and later moved to the coast."
        t2 = " was born in 1850 in Ely and later moved to the city."
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(after=t1, doc_ix=1), dict(after=t1, doc_ix=2),
                                   dict(after=t2, doc_ix=2), dict(after=t2, doc_ix=3)])
        r = cell(fx.grid(["Ada Zed"]), "Ada Zed")
        self.assertEqual((r["mentions_matched"], r["n_docs"], r["n_docs_dedup"]), (4, 3, 1))

    def test_dedup_merges_documents_across_strings_of_one_name(self):
        same = " was born in 1850 in Ely and later moved."
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed (chef)")
        fx.entity("Ada Zed", "Ada Zed", "stripped")
        # the needle text differs per string, so put the shared text in `before` of a window whose
        # needle is the same words: same slice, same key
        fx.string("Ada Zed (chef)", wins=[dict(needle="Ada Zed", after=same, doc_ix=1)])
        fx.string("Ada Zed", wins=[dict(after=same, doc_ix=2)])
        r = cell(fx.grid(["Ada Zed"]), "Ada Zed")
        self.assertEqual((r["n_docs"], r["n_docs_dedup"]), (2, 1))

    def test_dedup_never_exceeds_docs_on_a_larger_fixture(self):
        texts = [" was born in 18%02d and died 1900." % k for k in (50, 50, 51, 50, 52)]
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        wins = []
        for i in range(60):
            wins.append(dict(after=" was born in 1850 in Ely." + " x" * (i % 4), doc_ix=i % 17))
            wins.append(dict(before="Sir ", after=texts[i % 5], doc_ix=(i * 7) % 23))
        fx.string("Ada Zed", wins=wins)
        rows = fx.grid(["Ada Zed"])
        self.assertTrue(rows)
        for r in rows:
            self.assertLessEqual(r["n_docs_dedup"], r["n_docs"])
        self.assertGreater(cell(rows, "Ada Zed")["n_docs"], 1)

    def test_domains_only_from_urls(self):
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        metas = [
            json.dumps({"linenum": 1, "metadata": {"url": "https://www.Example.com/a"}, "path": "p"}),
            json.dumps({"metadata": {"url": "http://news.example.org/x?q=1"}}),
            json.dumps({"metadata": {"url": "https://example.com/other"}}),
            None,
            "not json{",
            json.dumps({"metadata": {"id": "abc", "note": "ftp://x.y"}}),
        ]
        fx.string("Ada Zed", wins=[dict(W_CUE, meta=m) for m in metas])
        r = cell(fx.grid(["Ada Zed"]), "Ada Zed")
        # example.com twice (www. stripped, lowercased) + news.example.org
        self.assertEqual((r["n_domains"], r["matched_with_url"], r["url_share"]), (2, 3, 0.5))
        fx2 = Fx()
        fx2.entity("Ada Zed", "Ada Zed")
        fx2.string("Ada Zed", wins=[dict(W_CUE, meta=None), dict(W_CUE, meta="{{")])
        r2 = cell(fx2.grid(["Ada Zed"]), "Ada Zed")
        self.assertEqual((r2["n_domains"], r2["url_share"]), ("", 0.0))


class SharedAndConflictTests(unittest.TestCase):
    def test_shared_string_counts_for_both_names(self):
        fx = Fx()
        fx.entity("Kay Lin (actor)", "Kay Lin", "stripped")
        fx.entity("Kay Lin (judge)", "Kay Lin", "stripped")
        fx.entity("Kay Lin (judge)", "Kay Lin (judge)")
        fx.string("Kay Lin", wins=[dict(W_CUE)])
        fx.string("Kay Lin (judge)", wins=[dict(W_NONE)])
        years = {"Kay Lin (actor)": {"birth": [1850], "death": []},
                 "Kay Lin (judge)": {"birth": [1850], "death": []}}
        rows = fx.grid(["Kay Lin (actor)", "Kay Lin (judge)"], years)
        a, j = cell(rows, "Kay Lin (actor)"), cell(rows, "Kay Lin (judge)")
        self.assertEqual((a["mentions_matched"], a["shared_string"]), (1, 1))
        self.assertEqual((j["mentions_matched"], j["shared_string"]), (1, 1))
        self.assertEqual(cell(rows, "Kay Lin (judge)", strings="no_stripped")["shared_string"], 0)
        self.assertEqual(cell(rows, "Kay Lin (judge)", strings="no_stripped")["mentions_matched"], 0)

    def test_conflicting_years_give_two_rows(self):
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(after=" born 1850 or born 1851 here")])
        years = {"Ada Zed": {"birth": [1851, 1850], "death": [1900]}}
        rows = fx.grid(["Ada Zed"], years)
        r50 = cell(rows, "Ada Zed", year=1850)
        r51 = cell(rows, "Ada Zed", year=1851)
        self.assertEqual((r50["year_conflict"], r51["year_conflict"]), (1, 1))
        self.assertEqual((r50["mentions_matched"], r51["mentions_matched"]), (1, 1))
        self.assertEqual(cell(rows, "Ada Zed", fact="death")["year_conflict"], 0)
        self.assertEqual(pc.year_conflict_list(years, ["Ada Zed"]),
                         [{"prompt_name": "Ada Zed", "fact": "birth", "years": [1850, 1851]}])

    def test_needle_year_is_never_matched(self):
        fx = Fx()
        fx.entity("Pat Roe (footballer, born 1875)", "Pat Roe (footballer, born 1875)")
        fx.string("Pat Roe (footballer, born 1875)", wins=[dict(W_NONE)])
        years = {"Pat Roe (footballer, born 1875)": {"birth": [1875], "death": []}}
        r = cell(fx.grid(list(years), years), "Pat Roe (footballer, born 1875)")
        self.assertEqual(r["status"], "zero")


class TruncatedShareAndSummaryTests(unittest.TestCase):
    def test_truncated_side_share(self):
        fx = Fx()
        w = [
            Fx.window("q", IDX, 0, before="a" * 10, needle="N", after="b" * 2000),  # 1 of 2 short
            Fx.window("q", IDX, 1, before="a" * 1700, needle="N", after="b" * 1600),  # 0 of 2 short
            Fx.window("q", IDX2, 0, before="a", needle="N", after="b"),  # both short
        ]
        share = pc.truncated_side_share(w, [IDX, IDX2, "v4_piletrain_llama"])
        self.assertEqual(share, {IDX: 0.25, IDX2: 1.0})

    def _rows(self):
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(W_CUE, doc_ix=1, meta=json.dumps({"u": "https://a.example/x"})),
                                   dict(W_NONE, doc_ix=2)])
        years = {"Ada Zed": {"birth": [1850], "death": []}}
        return fx.grid(["Ada Zed"], years)

    def test_summary_heading_no_em_dash_and_numbers_from_rows(self):
        rows = self._rows()
        text = pc.summary_markdown(rows, {IDX: 0.25}, [], {"extra_windows": 4})
        self.assertTrue(text.startswith("# Pilot preview over 20 names, NOT tau"))
        self.assertNotIn(EM_DASH, text)
        self.assertIn("nearest-rank", text)
        # counted 1, zero 0, unknown 0, one docs fact, median 1, p10 1, one fact with n_c > 0
        # (p10 1), no est_mentions facts (median n/a)
        self.assertIn("| OLMo 2 | all | bare | full | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | n/a |", text)
        self.assertIn("| OLMo 2 | 1 | 1 | 100.0% | 25.0% |", text)  # url_share, truncated share
        self.assertIn("extra_windows: 4", text)

    def test_summary_accepts_rows_read_back_from_csv(self):
        rows = [{k: str(v) for k, v in r.items()} for r in self._rows()]
        text = pc.summary_markdown(rows, {}, [{"prompt_name": "Ada Zed", "fact": "birth", "years": [1850, 1851]}])
        self.assertIn("| OLMo 2 | all | bare | full | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | n/a |", text)
        self.assertIn("Ada Zed, birth: 1850, 1851", text)
        self.assertNotIn(EM_DASH, text)

    def test_summary_percentiles_are_docs_only_nearest_rank(self):
        # hand-computed: one true zero, docs n_c 1..12, and two counted est_mentions facts
        spec = [(0, "docs", "zero")] + [(v, "docs", "counted") for v in range(1, 13)]             + [(900.0, "est_mentions", "counted"), (100.0, "est_mentions", "counted")]
        rows = []
        for i, (n, unit, status) in enumerate(spec):
            r = {c: "" for c in pc.CSV_COLUMNS}
            r.update(prompt_name="n%d" % i, index=IDX, fact="birth", year=1850, half_width="full",
                     matcher="bare", strings="all", status=status, n_c=n, n_c_unit=unit,
                     mentions_matched=0 if status == "zero" else 1, year_conflict=0)
            rows.append(r)
        text = pc.summary_markdown(rows, {}, [])
        # incl. 0: 13 docs values 0..12: median = rank 7 -> 6, p10 = rank 2 -> 1.
        # n_c > 0: 12 values 1..12, p10 = rank 2 -> 2. The est_mentions facts (median of 100
        # and 900 = lower middle 100) are counted apart and never enter the docs columns.
        self.assertIn("| OLMo 2 | all | bare | full | 14 | 1 | 0 | 13 | 6 | 1 | 12 | 2 | 2 | 100 |", text)
        self.assertIn("median n_c (incl. 0)", text)
        self.assertIn("p10 n_c>0", text)
        self.assertIn("NOT tau", text)
        # width x matcher p10 table and the all-vs-no_stripped table carry the new columns
        self.assertIn("| OLMo 2 | bare | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | n/a (n=0) | 2 (n=12) |", text)

    def test_summary_reports_rows_where_dedup_exceeds_docs(self):
        rows = self._rows()
        self.assertIn("n_docs_dedup > n_docs: 0", pc.summary_markdown(rows, {}, []))
        bad = [dict(r, n_docs_dedup=r["n_docs"] + 1) for r in rows]
        self.assertIn("n_docs_dedup > n_docs: %d" % len(rows), pc.summary_markdown(bad, {}, []))


def cand(i, band, cue, source="name", index=None, fact=None, name=None):
    return {"index": index or "ix%d" % (i % 3), "prompt_name": name or "n%d" % (i % 9),
            "query_string": "q%03d" % i, "query_source": source, "form": "plain", "s": 0,
            "rank": i, "fact": fact or ("birth", "death")[i % 2], "year": 1850,
            "distance_chars": 5, "band": band, "matched_bare": 1, "matched_cue": cue,
            "snippet": "s"}


def population(per_stratum=20, stripped_every=None):
    out, i = [], 0
    for b in pc.BAND_LABELS:
        for cue in (0, 1):
            for k in range(per_stratum):
                src = "stripped" if stripped_every and k % stripped_every == 0 else "name"
                out.append(cand(i, b, cue, src))
                i += 1
    return out


class PacketCandidateTests(unittest.TestCase):
    def test_candidates_use_nearest_occurrence_and_mark_snippets(self):
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        far_then_near = dict(before="in 1900 ", after=" (born\n1850) and " + "x" * 500 + " born 1850 again")
        far_only = dict(after="y" * 700 + " 1850 z")
        fx.string("Ada Zed", wins=[far_then_near, far_only, dict(W_NONE)])
        c = pc.packet_candidates(fx.entities, fx.plan, fx.windows, YEARS, ["Ada Zed"])
        self.assertEqual(len(c), 3)  # window 0 birth + death, window 1 birth
        by = {(x["rank"], x["fact"]): x for x in c}
        b0, d0, b1 = by[(0, "birth")], by[(0, "death")], by[(1, "birth")]
        self.assertEqual((b0["band"], b0["matched_cue"], b0["matched_bare"]), ("0-100", 1, 1))
        self.assertEqual((d0["distance_chars"], d0["band"]), (1, "0-100"))
        self.assertEqual((b1["band"], b1["matched_cue"]), ("401-800", 0))
        for x in c:
            self.assertIn("[[Ada Zed]]", x["snippet"])
            self.assertIn("<<%d>>" % x["year"], x["snippet"])
            self.assertNotIn("\n", x["snippet"])
        self.assertEqual({x["query_source"] for x in c}, {"name"})

    def test_at_most_one_row_per_window_per_fact(self):
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[dict(after=" 1850 1850 1850")])
        c = pc.packet_candidates(fx.entities, fx.plan, fx.windows, YEARS, ["Ada Zed"])
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0]["distance_chars"], 1)


class SelectPacketTests(unittest.TestCase):
    def test_deterministic_and_covers_every_stratum(self):
        cands = population()
        p1, s1, m1 = pc.select_packet(cands, 20260930)
        p2, s2, m2 = pc.select_packet(list(reversed(cands)), 20260930)
        self.assertEqual((p1, s1, m1), (p2, s2, m2))
        self.assertNotEqual([r["query_string"] for r in p1],
                            [r["query_string"] for r in pc.select_packet(cands, 7)[0]])
        self.assertEqual(len(p1), 60)
        self.assertEqual([r["packet_id"] for r in p1][:2] + [p1[-1]["packet_id"]], ["P001", "P002", "P060"])
        counts = {}
        for r in p1:
            counts[(r["band"], r["matched_cue"])] = counts.get((r["band"], r["matched_cue"]), 0) + 1
        self.assertEqual(len(counts), 12)
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual(m1, ["quota unmet: 0 rows from stripped strings (need at least 10)"])
        self.assertTrue(all(r["label"] == "" for r in p1))
        self.assertEqual(sum(s["drawn"] for s in s1), 60)
        self.assertEqual(sum(s["population"] for s in s1), 240)

    def test_short_strata_taken_whole_and_gap_filled_from_largest(self):
        cands = [c for c in population(per_stratum=20, stripped_every=4)
                 if not (c["band"] == ">1600" and c["matched_cue"] == 1)]
        cands += [cand(1000 + i, ">1600", 1) for i in range(2)]  # this stratum holds only 2
        cands += [cand(2000 + i, "0-100", 0) for i in range(30)]  # now the largest stratum, 50
        packet, strata, msgs = pc.select_packet(cands, 1)
        self.assertEqual(len(packet), 60)
        drawn = {(s["band"], s["matched_cue"]): 0 for s in strata}
        for s in strata:
            drawn[(s["band"], s["matched_cue"])] += s["drawn"]
        self.assertEqual(drawn[(">1600", 1)], 2)
        self.assertEqual(drawn[("0-100", 0)], 5 + 3)  # the gap of 3 goes to the largest stratum
        self.assertEqual(msgs, [])

    def test_fewer_candidates_than_target_are_all_taken(self):
        cands = population(per_stratum=2)
        packet, _, msgs = pc.select_packet(cands, 3)
        self.assertEqual(len(packet), 24)
        self.assertTrue(any("candidates ran out" in m for m in msgs))

    def test_quota_met_by_swapping_and_unmet_quotas_reported(self):
        # 2 of every 20 candidates in a stratum are stripped: a plain draw of 5 gets ~0.5 of them
        cands = population(stripped_every=10)
        packet, strata, msgs = pc.select_packet(cands, 20260930)
        self.assertGreaterEqual(sum(r["query_source"] == "stripped" for r in packet), 10)
        self.assertEqual(len(packet), 60)
        self.assertEqual(len({r["packet_id"] for r in packet}), 60)
        self.assertEqual(sum(s["drawn"] for s in strata), 60)
        self.assertEqual(msgs, [])
        # no stripped candidates at all, one index, one fact, one name: everything unmet
        flat = [cand(i, b, cue, "name", "only", "birth", "solo")
                for i, (b, cue) in enumerate((b, c) for b in pc.BAND_LABELS for c in (0, 1) for _ in range(6))]
        _, _, msgs = pc.select_packet(flat, 5)
        self.assertEqual(len(msgs), 4)
        self.assertTrue(any("stripped" in m for m in msgs))
        self.assertTrue(any("indexes" in m for m in msgs))
        self.assertTrue(any("facts" in m for m in msgs))
        self.assertTrue(any("names" in m for m in msgs))

    def test_strata_rows_split_by_query_source(self):
        _, strata, _ = pc.select_packet(population(stripped_every=10), 2)
        first = [s for s in strata if s["band"] == "0-100" and s["matched_cue"] == 0]
        self.assertEqual({s["query_source"] for s in first}, {"name", "stripped"})
        self.assertEqual(sum(s["population"] for s in first), 20)
        self.assertEqual(list(strata[0]), pc.STRATA_COLUMNS)


def lab_rows(spec):
    """spec: [(band, cue, correct, other_person, other_fact, unclear)] -> packet rows."""
    rows, n = [], 0
    for band, cue, c, op, of, u in spec:
        for label, k in (("correct", c), ("other_person", op), ("other_fact", of), ("unclear", u)):
            for _ in range(k):
                n += 1
                rows.append({"packet_id": "P%03d" % n, "band": band, "matched_cue": cue,
                             "fact": ("birth", "death")[n % 2], "query_source": ("name", "stripped")[n % 2],
                             "label": label})
    return rows


STRATA = [
    {"band": "0-100", "matched_cue": 0, "query_source": "name", "population": 100, "drawn": 10},
    {"band": "0-100", "matched_cue": 1, "query_source": "name", "population": 100, "drawn": 10},
    {"band": "101-200", "matched_cue": 1, "query_source": "name", "population": 200, "drawn": 10},
    {"band": "401-800", "matched_cue": 0, "query_source": "name", "population": 300, "drawn": 10},
    {"band": "401-800", "matched_cue": 1, "query_source": "name", "population": 100, "drawn": 5},
]
# 0-100 cue1: 8 correct, 2 other_person, 1 unclear; 0-100 cue0: 5 correct, 5 other_fact;
# 101-200 cue1: 5 correct, 5 other_person; 401-800 cue1: 1 of 5; 401-800 cue0: 2 of 10.
SPEC = [("0-100", 1, 8, 2, 0, 1), ("0-100", 0, 5, 0, 5, 0), ("101-200", 1, 5, 5, 0, 0),
        ("401-800", 1, 1, 4, 0, 0), ("401-800", 0, 2, 8, 0, 0)]


class ScoreLabelsTests(unittest.TestCase):
    def test_refuses_under_40_labelled_rows(self):
        rows = lab_rows([("0-100", 1, 20, 19, 0, 0)])  # 39 labelled
        rows += [{"packet_id": "P900", "band": "0-100", "matched_cue": 1, "fact": "birth",
                  "query_source": "name", "label": "  "}]  # blank = not labelled
        with self.assertRaises(pc.TooFewLabels):
            pc.score_labels(rows, STRATA)
        rows += lab_rows([("0-100", 1, 1, 0, 0, 0)])
        pc.score_labels(rows, STRATA)  # 40 labelled: fine

    def test_unclear_counts_as_labelled(self):
        rows = lab_rows([("0-100", 1, 20, 10, 0, 10)])  # 40 labelled incl. 10 unclear
        text = pc.score_labels(rows, STRATA)
        self.assertIn("Labelled rows: 40", text)

    def test_rejects_unknown_label_with_its_packet_id(self):
        rows = lab_rows(SPEC)
        rows[3]["label"] = "maybe"
        rows[7]["label"] = "Correct!"
        with self.assertRaises(pc.LabelError) as cm:
            pc.score_labels(rows, STRATA)
        self.assertIn(rows[3]["packet_id"], str(cm.exception))
        self.assertIn(rows[7]["packet_id"], str(cm.exception))

    def test_trims_and_lowercases_labels(self):
        rows = lab_rows(SPEC)
        for r in rows:
            r["label"] = " " + r["label"].upper() + " "
        self.assertEqual(pc.score_labels(rows, STRATA), pc.score_labels(lab_rows(SPEC), STRATA))

    def test_hand_computed_precision(self):
        text = pc.score_labels(lab_rows(SPEC), STRATA)
        self.assertNotIn(EM_DASH, text)
        self.assertIn("Labelled rows: 46", text)
        line = {}
        for ln in text.splitlines():
            cells = [c.strip() for c in ln.strip("|").split("|")]
            line[tuple(cells[:2])] = cells
        # per band: 0-100 bare 13 of 20 scored = 0.650, cue 8 of 10 = 0.800 (unclear excluded)
        band = [ln for ln in text.splitlines() if ln.startswith("| 0-100 |")][0]
        self.assertIn("0.650 (correct 13 of 20 scored; labelled 21, unclear 1)", band)
        self.assertIn("0.800 (correct 8 of 10 scored; labelled 11, unclear 1)", band)
        self.assertIn("too few rows", [ln for ln in text.splitlines() if ln.startswith("| 201-400 |")][0])
        # rule width 100 bare: only band 0-100 inside
        r100 = line[("100", "bare")]
        self.assertIn("0.650", r100[4])
        self.assertEqual(r100[5], "0.650")
        # width 200 cue: weights pop 100 : 200, precisions 0.8 and 0.5 -> 0.6; unweighted 13 / 20
        r200 = line[("200", "cue")]
        self.assertTrue(r200[4].startswith("0.650"))
        self.assertEqual(r200[5], "0.600")
        # width 200 bare: pops 100+100 and 0+200 -> equal weights, 0.65 and 0.5 -> 0.575
        self.assertEqual(line[("200", "bare")][5], "0.575")
        # width 800 cue: bands 0-100, 101-200, 401-800 with pops 100, 200, 100 and 0.8, 0.5, 0.2 -> 0.5
        r800 = line[("800", "cue")]
        self.assertEqual(r800[5], "0.500")
        self.assertEqual(r800[6], "0-100, 101-200, 401-800")
        self.assertEqual(r800[7], "201-400")
        # full includes bands with no rows: they are excluded and listed
        self.assertEqual(line[("full", "cue")][7], "201-400, 801-1600, >1600")

    def test_per_fact_and_source_sections_and_counts(self):
        text = pc.score_labels(lab_rows(SPEC), STRATA)
        for head in ("## Per band and matcher", "## Per fact", "## Per query source",
                     "## Per candidate rule (width x matcher)"):
            self.assertIn(head, text)
        self.assertIn("| birth |", text)
        self.assertIn("| stripped |", text)
        self.assertIn("renormalised", text)

    def test_score_accepts_csv_strings(self):
        rows = [{k: str(v) for k, v in r.items()} for r in lab_rows(SPEC)]
        strata = [{k: str(v) for k, v in s.items()} for s in STRATA]
        self.assertEqual(pc.score_labels(rows, strata), pc.score_labels(lab_rows(SPEC), STRATA))


def _rule_check():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "scripts" / "pilot_rule_check.py"
    spec = importlib.util.spec_from_file_location("pilot_rule_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def occ(distance, length=4):
    return pc.fact_match.Occurrence("after", 0, length, distance)


class D024MatcherTests(unittest.TestCase):
    """`near_or_cue`, the counting rule: the year within NEAR characters of the
    name, or a cue anywhere in the window."""

    def test_parity_with_pilot_rule_check(self):
        rc = _rule_check()
        self.assertEqual(pc.NEAR, rc.NEAR)
        cases = {
            "near, no cue": [(occ(50), None)],
            "exactly 100": [(occ(100), None)],
            "101": [(occ(101), None)],
            "ends 101 to 104 out (in_width would drop it)": [(occ(98), None)],
            "far, no cue": [(occ(300), None)],
            "far, cue": [(occ(300), "born")],
            "several, none qualifies": [(occ(150), None), (occ(300), None)],
            "several, the far one has a cue": [(occ(150), None), (occ(900), "died")],
            "several, the near one qualifies": [(occ(60), None), (occ(400), None)],
            "no occurrence": [],
        }
        for label, entries in cases.items():
            with self.subTest(label):
                self.assertEqual(pc._nearest(entries, None, "near_or_cue") is not None,
                                 rc.window_matches(entries, "bare <= 100 OR cue (chosen)"))
        self.assertIsNotNone(pc._nearest(cases["exactly 100"], None, "near_or_cue"))
        self.assertIsNone(pc._nearest(cases["101"], None, "near_or_cue"))
        self.assertFalse(pc.fact_match.in_width(occ(98), pc.NEAR))  # why the rule uses distance
        self.assertIsNotNone(pc._nearest([(occ(98), None)], None, "near_or_cue"))

    def test_one_document_counts_once(self):
        fx = Fx()
        fx.entity("Ada Zed", "Ada Zed")
        fx.string("Ada Zed", wins=[{"after": " lived from 1850 on.", "doc_ix": 5},     # near, no cue
                                   {"after": " " + "x" * 300 + " was born in 1850.", "doc_ix": 5},  # far, cue
                                   dict(W_FAR, doc_ix=6)])                            # far, no cue
        r = cell(fx.grid(["Ada Zed"]), "Ada Zed", matcher="near_or_cue", strings="no_stripped")
        self.assertEqual((r["mentions_matched"], r["n_docs"], r["n_c"], r["status"]), (2, 1, 1, "counted"))

    def test_kept_out_of_the_precision_table(self):
        with self.assertRaises(ValueError):
            pc._matcher_rows([], "near_or_cue")
        self.assertNotIn("near_or_cue", pc.score_labels(lab_rows(SPEC), STRATA))


class NoNetworkTests(unittest.TestCase):
    def test_module_does_not_import_infinigram(self):
        import inspect
        src = inspect.getsource(pc)
        self.assertNotIn("infinigram", src)
        self.assertNotIn("sqlite3", src)
        self.assertNotIn(EM_DASH, src)


if __name__ == "__main__":
    unittest.main()
