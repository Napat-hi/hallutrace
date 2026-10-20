"""tests/test_fact_match.py  |  unit tests for hallutrace.fact_match (pure, no I/O)

Run: python -m tests.test_fact_match   or   python -m unittest tests.test_fact_match

En and em dashes are written as \\u2013 / \\u2014 escapes, never as literal characters.
"""

from __future__ import annotations

import unittest

from hallutrace import fact_match as fm
from hallutrace.fact_match import (
    CUE_GAP,
    Occurrence,
    cue_match,
    dedup_key,
    find_year,
    in_width,
    snippet,
    window_text,
)

EN = "\u2013"
EM = "\u2014"
PY = {"birth": 1931, "death": 2019}


def cue(before, needle, after, year, fact, person_years=PY, nth=0):
    """cue_match on the nth occurrence of `year` in before/after."""
    occs = find_year(before, needle, after, year)
    return cue_match(before + needle + after, occs[nth], year, fact, person_years)


class WindowTextTest(unittest.TestCase):
    def test_single_needle(self):
        spans = [["aa ", None], ["Bob", "0"], [" bb", None], [" cc", None]]
        self.assertEqual(window_text(spans), ("aa ", "Bob", " bb cc"))

    def test_needle_at_edges(self):
        self.assertEqual(window_text([["Bob", "0"], [" x", None]]), ("", "Bob", " x"))
        self.assertEqual(window_text([["x ", None], ["Bob", "0"]]), ("x ", "Bob", ""))

    def test_int_zero_label_accepted_but_not_none_or_false(self):
        self.assertEqual(window_text([["a", None], ["N", 0]]), ("a", "N", ""))
        with self.assertRaises(ValueError):
            window_text([["a", None], ["N", False]])

    def test_no_needle_raises(self):
        with self.assertRaises(ValueError):
            window_text([["a", None], ["b", None]])
        with self.assertRaises(ValueError):
            window_text([])

    def test_multiple_needles_first_when_offset_missing(self):
        spans = [["N1", "0"], [" mid ", None], ["N2", "0"]]
        self.assertEqual(window_text(spans)[1], "N1")
        self.assertEqual(window_text(spans, needle_offset=5)[1], "N1")
        self.assertEqual(window_text(spans, disp_len=10)[1], "N1")

    def test_multiple_needles_chosen_by_offset_fraction(self):
        # starts as fractions of the 20-character text: 0.0, 0.5, 0.9
        spans = [
            ["N1234", "0"], ["x" * 5, None], ["N5678", "0"], ["y" * 3, None], ["N9", "0"],
        ]
        self.assertEqual(sum(len(t) for t, _ in spans), 20)
        # starts 0, 10, 18 of 20 -> 0.0, 0.5, 0.9
        self.assertEqual(window_text(spans, 0, 100)[1], "N1234")
        self.assertEqual(window_text(spans, 50, 100)[1], "N5678")
        self.assertEqual(window_text(spans, 90, 100)[1], "N9")
        # offset is in tokens over disp_len tokens, not characters
        self.assertEqual(window_text(spans, 5, 10)[1], "N5678")

    def test_multiple_needles_tie_takes_first(self):
        spans = [["N1", "0"], ["ab", None], ["N2", "0"], ["cd", None]]
        # starts 0 and 4 of 8: fractions 0.0 and 0.5, target 0.25 is a tie
        self.assertEqual(window_text(spans, 1, 4)[1], "N1")

    def test_other_needles_digits_masked_same_length(self):
        spans = [
            ["Al (born 1875)", "0"], [" was in 1931 ", None], ["Al (born 1875)", "0"],
            [" then 1875.", None],
        ]
        before, needle, after = window_text(spans, 0, 1)
        self.assertEqual(needle, "Al (born 1875)")
        self.assertEqual(before, "")
        self.assertIn("Al (born ####)", after)
        self.assertIn("1931", after)
        self.assertEqual(len(before + needle + after), sum(len(t) for t, _ in spans))
        # the year inside the repeated needle is never found; the real one is
        occs = find_year(before, needle, after, 1875)
        self.assertEqual(len(occs), 1)
        self.assertEqual((before + needle + after)[occs[0].start:occs[0].end], "1875")
        self.assertEqual(find_year(before, needle, after, 1931)[0].side, "after")

    def test_masking_applies_before_chosen_needle_too(self):
        spans = [["Al 1875", "0"], [" mid ", None], ["Al 1875", "0"]]
        before, needle, after = window_text(spans, 9, 10)
        self.assertEqual((before, needle, after), ("Al ####" + " mid ", "Al 1875", ""))


class FindYearTest(unittest.TestCase):
    def test_whole_number_only(self):
        for text in ("x 19310 y", "x 11931 y", "x 19311931 y", "x 119310"):
            self.assertEqual(find_year("", "N", text, 1931), [], text)
        self.assertEqual(len(find_year("", "N", " in 1931.", 1931)), 1)
        self.assertEqual(len(find_year("", "N", "(1931)", 1931)), 1)
        self.assertEqual(len(find_year("", "N", "1931a", 1931)), 1)

    def test_year_inside_needle_never_found(self):
        needle = "Patrick O'Brien (footballer, born 1875)"
        self.assertEqual(find_year("", needle, "", 1875), [])

    def test_digit_touching_needle_blocks(self):
        self.assertEqual(find_year("1931", "5Bob", "", 1931), [])

    def test_side_offsets_and_distance(self):
        before, needle, after = "in 1931 by ", "Bob", " and 1931"
        occs = find_year(before, needle, after, 1931)
        self.assertEqual([o.side for o in occs], ["before", "after"])
        text = before + needle + after
        for o in occs:
            self.assertEqual(text[o.start:o.end], "1931")
        b, a = occs
        self.assertEqual(b.distance, len(" by "))
        self.assertEqual(a.distance, len(" and "))
        self.assertEqual(b.start, 3)
        self.assertEqual(a.start, len(before) + len(needle) + 5)

    def test_adjacent_to_needle_distance_zero(self):
        occs = find_year("1931 ", "Bob", " 1931", 1931)
        self.assertEqual([o.distance for o in occs], [1, 1])


class InWidthTest(unittest.TestCase):
    def occ(self, pad, side="after"):
        """Occurrence of 1931 with `pad` characters between it and the needle."""
        needle = "Bob"
        if side == "after":
            before, after = "", " " * pad + "1931"
        else:
            before, after = "1931" + " " * pad, ""
        (o,) = find_year(before, needle, after, 1931)
        self.assertEqual(o.distance, pad)
        return o

    def test_cut_by_characters_either_side(self):
        for side in ("after", "before"):
            self.assertTrue(in_width(self.occ(96, side), 100), side)  # far edge at 100
            self.assertFalse(in_width(self.occ(97, side), 100), side)  # one beyond
            self.assertTrue(in_width(self.occ(0, side), 4), side)
            self.assertFalse(in_width(self.occ(0, side), 3), side)

    def test_none_means_full(self):
        self.assertTrue(in_width(self.occ(50000), None))
        self.assertFalse(in_width(self.occ(50000), 1600))

    def test_standard_widths(self):
        o = self.occ(300)
        self.assertEqual([in_width(o, w) for w in (100, 200, 400, 800, 1600)],
                         [False, False, True, True, True])

    def test_direct_occurrence(self):
        o = Occurrence("after", 10, 14, 5)
        self.assertTrue(in_width(o, 9))
        self.assertFalse(in_width(o, 8))


class LifespanCueTest(unittest.TestCase):
    def both(self, text, py=PY):
        """(birth, death) for the first 1931 / 2019 occurrences in `text`."""
        b = cue("", "N", text, 1931, "birth", py) if find_year("", "N", text, 1931) else None
        d = cue("", "N", text, 2019, "death", py) if find_year("", "N", text, 2019) else None
        return b, d

    def test_full_lifespans_count_for_both(self):
        for form in (
            " (1931%s2019)" % EN,
            " (1931-2019)",
            " 1931%s2019" % EN,
            " (c. 1931 %s 2019)" % EN,
            " (1931 %s 2019)" % EM,
            " (ca. 1931-2019)",
            " (circa 1931 - c. 2019)",
        ):
            self.assertEqual(self.both(form), (True, True), form)

    def test_full_lifespan_also_gives_other_fact_cross_check(self):
        # birth year in the death slot is not a death cue
        self.assertFalse(cue("", "N", " (1931-2019)", 1931, "death"))
        self.assertFalse(cue("", "N", " (1931-2019)", 2019, "birth"))

    def test_range_with_one_persons_year_is_no_cue(self):
        text = " served 1931-1945"
        self.assertFalse(cue("", "N", text, 1931, "birth"))
        self.assertFalse(cue("", "N", text, 1931, "death"))
        text = " served 1945-2019"
        self.assertFalse(cue("", "N", text, 2019, "death"))
        self.assertFalse(cue("", "N", text, 2019, "birth"))

    def test_open_lifespan_is_birth_only(self):
        for text in (" (1931%s )" % EN, " (1931- )", " (1931-)", " (c. 1931-) is alive"):
            self.assertTrue(cue("", "N", text, 1931, "birth"), text)
            self.assertFalse(cue("", "N", text, 1931, "death"), text)

    def test_open_lifespan_needs_no_digit_after_dash(self):
        self.assertFalse(cue("", "N", " (1931-1945)", 1931, "birth"))
        self.assertFalse(cue("", "N", " (1931 - 1945)", 1931, "birth"))

    def test_lifespan_needs_person_years(self):
        text = " (1931-2019)"
        self.assertFalse(cue("", "N", text, 1931, "birth", {"birth": None, "death": 2019}))
        # birth known, death unknown: the full lifespan cannot be checked
        self.assertFalse(cue("", "N", text, 2019, "death", {"birth": 1931, "death": None}))
        # open lifespan still needs the birth year
        self.assertFalse(cue("", "N", " (1931- )", 1931, "birth", {"birth": None, "death": None}))

    def test_wrong_person_years_no_cue(self):
        self.assertFalse(cue("", "N", " (1931-2019)", 1931, "birth", {"birth": 1931, "death": 2020}))

    def test_lifespan_in_before_text(self):
        self.assertTrue(cue("(1931-2019) ", "Bob", "", 2019, "death"))
        self.assertTrue(cue("(1931-2019) ", "Bob", "", 1931, "birth"))


class WordCueTest(unittest.TestCase):
    def test_birth_cues(self):
        for text in (
            " b. 1931",
            " (b.1931)",
            " B. 1931",
            " born in 1931",
            " Born: 12 March 1931",
            " BORN 1931",
            " birth 1931",
            " Year of birth: 1931",
            " was born on 3 May, 1931",
        ):
            self.assertTrue(cue("", "N", text, 1931, "birth"), text)
            self.assertFalse(cue("", "N", text, 1931, "death"), text)

    def test_death_cues(self):
        for text in (
            " died 1931", " (d. 1931)", " d.1931", " Died in 1931", " death 1931",
            " passed away in 1931", " passed  away 1931", " DIED: 3 May 1931",
        ):
            self.assertTrue(cue("", "N", text, 1931, "death"), text)
            self.assertFalse(cue("", "N", text, 1931, "birth"), text)

    def test_born_then_died_year_is_death_not_birth(self):
        text = " born 1850, died 1931"
        self.assertTrue(cue("", "N", text, 1931, "death"))
        self.assertFalse(cue("", "N", text, 1931, "birth"))
        # and the first year is the birth
        self.assertTrue(cue("", "N", text, 1850, "birth", {"birth": 1850, "death": 1931}))

    def test_initial_b_is_not_a_cue(self):
        self.assertFalse(cue("", "N", "John B. Smith visited in 1931", 1931, "birth"))
        self.assertFalse(cue("", "N", " B. Smith ... 1931", 1931, "birth"))
        self.assertFalse(cue("", "N", " D. Smith 1931", 1931, "death"))

    def test_dot_cue_needs_word_boundary(self):
        self.assertFalse(cue("", "N", " Bob. 1931", 1931, "birth"))
        self.assertFalse(cue("", "N", " Todd. 1931", 1931, "death"))

    def test_whole_word_only(self):
        self.assertFalse(cue("", "N", " unborn 1931", 1931, "birth"))
        self.assertFalse(cue("", "N", " reborn 1931", 1931, "birth"))
        self.assertFalse(cue("", "N", " birthday 1931", 1931, "birth"))
        self.assertFalse(cue("", "N", " diedown 1931", 1931, "death"))

    def test_gap_limit(self):
        # the cue ends exactly CUE_GAP characters before the year: counts
        self.assertTrue(cue("", "N", " born" + " " * CUE_GAP + "1931", 1931, "birth"))
        self.assertFalse(cue("", "N", " born" + " " * (CUE_GAP + 1) + "1931", 1931, "birth"))

    def test_gap_limit_with_words(self):
        near = " born " + "a" * (CUE_GAP - 2) + " 1931"  # gap is exactly CUE_GAP
        far = " born " + "a" * (CUE_GAP - 1) + " 1931"  # one character too far
        self.assertTrue(cue("", "N", near, 1931, "birth"))
        self.assertFalse(cue("", "N", far, 1931, "birth"))
        text = " born in a very small town of the county of Springfield in 1931"
        self.assertFalse(cue("", "N", text, 1931, "birth"))

    def test_only_three_or_four_digit_numbers_block(self):
        self.assertTrue(cue("", "N", " born 12 March 1931", 1931, "birth"))
        self.assertFalse(cue("", "N", " born 1850 or 1931", 1931, "birth"))
        self.assertFalse(cue("", "N", " born 850 or 1931", 1931, "birth"))
        self.assertTrue(cue("", "N", " born 85 or 1931", 1931, "birth"))

    def test_case_insensitive(self):
        for w in ("born", "Born", "BORN", "bOrN"):
            self.assertTrue(cue("", "N", " %s 1931" % w, 1931, "birth"), w)

    def test_cue_before_needle_in_before_text(self):
        # the cue may sit in the needle text itself (needle ends with "born")
        before, needle, after = "", "Al, born", " 1931."
        self.assertTrue(cue(before, needle, after, 1931, "birth"))
        # and for an occurrence in before, the cue is before it
        self.assertTrue(cue("He was born 1931 in Rome. ", "Al", "", 1931, "birth"))

    def test_nee_is_not_a_cue(self):
        self.assertFalse(cue("", "N", " née Smith 1931", 1931, "birth"))

    def test_cue_that_ends_after_year_does_not_count(self):
        self.assertFalse(cue("", "N", " 1931 born", 1931, "birth"))

    def test_second_occurrence_uses_own_context(self):
        text = " in 1931 he left; born 1931"
        self.assertFalse(cue("", "N", text, 1931, "birth", nth=0))
        self.assertTrue(cue("", "N", text, 1931, "birth", nth=1))

    def test_cues_table_is_module_level(self):
        self.assertIn("born", fm.CUES["birth"])
        self.assertIn("passed away", fm.CUES["death"])
        self.assertEqual(fm.CUE_GAP, 40)
        with self.assertRaises(ValueError):
            cue_match("x 1931", Occurrence("after", 2, 6, 0), 1931, "marriage", PY)


class SnippetTest(unittest.TestCase):
    def test_marks_needle_and_year(self):
        before, needle, after = "Then ", "Bob", " was born in 1931 in Rome."
        (o,) = find_year(before, needle, after, 1931)
        self.assertEqual(
            snippet(before, needle, after, o),
            "Then [[Bob]] was born in <<1931>> in Rome.",
        )

    def test_year_before_needle(self):
        before, needle, after = "In 1931 came ", "Bob", " home."
        (o,) = find_year(before, needle, after, 1931)
        self.assertEqual(snippet(before, needle, after, o), "In <<1931>> came [[Bob]] home.")

    def test_whitespace_collapsed(self):
        before, needle, after = "a\n\nb\t", "Bob", "  c\r\n born\t\t 1931 \n"
        (o,) = find_year(before, needle, after, 1931)
        s = snippet(before, needle, after, o)
        self.assertEqual(s, "a b [[Bob]] c born <<1931>>")
        for ch in "\n\t\r":
            self.assertNotIn(ch, s)
        self.assertNotIn("  ", s)

    def test_near_form_pads_150_outside_only(self):
        before = "x" * 500
        after = " in 1931" + "y" * 500
        (o,) = find_year(before, "Bob", after, 1931)
        s = snippet(before, "Bob", after, o)
        self.assertTrue(s.startswith("x" * 150 + "[[Bob]] in <<1931>>"))
        self.assertTrue(s.endswith("<<1931>>" + "y" * 150))
        self.assertNotIn("[...]", s)

    def test_near_form_at_600_gap(self):
        after = " " + "w" * 599 + "1931" + "z" * 400
        # gap = 600 exactly
        (o,) = find_year("", "Bob", after, 1931)
        self.assertEqual(o.distance, 600)
        s = snippet("", "Bob", after, o)
        self.assertNotIn("[...]", s)
        self.assertTrue(s.endswith("<<1931>>" + "z" * 150))
        self.assertIn("w" * 599, s)

    def test_far_form_year_after(self):
        before = "b" * 300
        after = " " + "m" * 700 + " born 1931 " + "e" * 300
        (o,) = find_year(before, "Bob", after, 1931)
        self.assertGreater(o.distance, 600)
        s = snippet(before, "Bob", after, o)
        self.assertIn(" [...] ", s)
        left, right = s.split(" [...] ")
        self.assertEqual(left, "b" * 150 + "[[Bob]] " + "m" * 149)
        self.assertEqual(right, "m" * 144 + " born <<1931>> " + "e" * 149)

    def test_far_form_year_before_keeps_text_order(self):
        before = "s" * 200 + " 1931 " + "m" * 700
        after = " tail " + "t" * 300
        (o,) = find_year(before, "Bob", after, 1931)
        self.assertEqual(o.side, "before")
        self.assertGreater(o.distance, 600)
        s = snippet(before, "Bob", after, o)
        left, right = s.split(" [...] ")
        self.assertIn("<<1931>>", left)
        self.assertIn("[[Bob]]", right)
        self.assertTrue(left.startswith("s" * 149 + " <<1931>>"))
        self.assertTrue(right.endswith("[[Bob]] tail " + "t" * 144))

    def test_clipped_at_text_edges(self):
        (o,) = find_year("", "Bob", " 1931", 1931)
        self.assertEqual(snippet("", "Bob", " 1931", o), "[[Bob]] <<1931>>")


class DedupKeyTest(unittest.TestCase):
    def key(self, before, needle, after, year, nth=0):
        occs = find_year(before, needle, after, year)
        return dedup_key(before + needle + after, occs[nth])

    def test_same_text_from_different_windows_same_key(self):
        doc = "The life of Bob Smith. He was born in 1931 in Rome and later moved north."
        i = doc.index("Bob Smith")
        j = doc.index("Rome")
        # window A: needle "Bob Smith"; window B: needle "Rome"; same underlying text,
        # so the 200-character slice around the year is identical
        a = self.key(doc[:i], "Bob Smith", doc[i + 9:], 1931)
        b = self.key(doc[:j], "Rome", doc[j + 4:], 1931)
        self.assertEqual(a, b)

    def test_key_is_sha1_hex(self):
        k = dedup_key("in 1931 x", Occurrence("after", 3, 7, 0))
        self.assertEqual(len(k), 40)
        int(k, 16)

    def test_whitespace_and_case_insensitive(self):
        a = self.key("Born  In\n", "Bob", " 1931 X", 1931)
        b = self.key("born in ", "Bob", " 1931 x", 1931)
        self.assertEqual(a, b)

    def test_different_years_different_keys(self):
        a = self.key("", "Bob", " was born in 1931 in Rome.", 1931)
        b = self.key("", "Bob", " was born in 1932 in Rome.", 1932)
        self.assertNotEqual(a, b)

    def test_different_context_different_keys(self):
        a = self.key("", "Bob", " was born in 1931 in Rome.", 1931)
        b = self.key("", "Bob", " was born in 1931 in Paris.", 1931)
        self.assertNotEqual(a, b)

    def test_far_context_ignored(self):
        # the text 300 characters before the year is outside the 200-character slice
        a = self.key("a" * 50 + "c" * 300, "Bob", " born 1931 ok", 1931)
        b = self.key("b" * 50 + "c" * 300, "Bob", " born 1931 ok", 1931)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
