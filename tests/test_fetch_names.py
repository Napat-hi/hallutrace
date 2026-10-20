"""Tests for hallutrace.fetch_names: query strings for the infini-gram fetcher.

Small in-memory fixtures only, no data files and no network. Run from the repo root:

    python -m unittest tests.test_fetch_names -v
"""

import csv
import json
import os
import tempfile
import unittest

from hallutrace.claim_key import AliasTable
from hallutrace.fetch_names import (
    SOURCES,
    QueryString,
    fold_accents,
    glued_ids,
    load_names,
    query_strings,
    strip_disambiguator,
)

# Table keys are deliberately lowercased and punctuation-stripped, the way the
# real data/alias_table.json is. Nothing here may ever reach a query string.
ALIAS_TABLE = AliasTable(
    {
        "mahatma gandhi": ["Q1"],
        "m k gandhi": ["Q1"],
        "bapu": ["Q1"],
        "gandhi": ["Q1", "Q2"],
        "mohandas gandhi": ["Q1", "Q2"],
        "margareta cederschiold": ["Q3"],
        "bill hutchinson baseball": ["Q4"],
        "bill hutchinson": ["Q4", "Q5"],
        "william hutchinson": ["Q4"],
        "john smith": ["Q6", "Q7"],
        "jose martin footballer": ["Q8"],
        "jose martin": ["Q8", "Q9"],
    }
)

SNAPSHOT = {
    "entities": {
        "Q1": {
            "label": "Mahatma Gandhi",
            "aliases": ["Bapu", "M. K. Gandhi", "Gandhi", "Mohandas Gandhi", "Mahatma Gandhi"],
        },
        "Q3": {
            "label": "Margareta Cederschiöld",
            "aliases": ["Margareta Cederschiold"],
        },
        "Q4": {
            "label": "Bill Hutchinson",
            "aliases": ["William Hutchinson"],
        },
        "Q6": {"label": "John Smith", "aliases": ["Captain John Smith"]},
        "Q8": {"label": "José Martín", "aliases": []},
    }
}


def by_source(qs, source):
    return [q.string for q in qs if q.source == source]


class StripDisambiguatorTest(unittest.TestCase):
    def test_parenthetical_word(self):
        self.assertEqual(strip_disambiguator("Bill Hutchinson (baseball)"), "Bill Hutchinson")

    def test_died_year(self):
        self.assertEqual(strip_disambiguator("Brian Palmes (died 1519)"), "Brian Palmes")

    def test_no_parenthetical(self):
        self.assertIsNone(strip_disambiguator("Charles Darwin"))

    def test_parenthetical_not_at_end_is_left_alone(self):
        self.assertIsNone(strip_disambiguator("Anna (Ann) Smith"))

    def test_only_one_parenthetical_stripped(self):
        self.assertEqual(strip_disambiguator("Jo Lee (a) (b)"), "Jo Lee (a)")

    def test_nothing_before_the_parenthetical(self):
        self.assertIsNone(strip_disambiguator("(baseball)"))
        self.assertIsNone(strip_disambiguator(" (baseball)"))

    def test_needs_a_space_before_the_parenthetical(self):
        self.assertIsNone(strip_disambiguator("Lee(baseball)"))

    def test_surrounding_whitespace_ignored(self):
        self.assertEqual(strip_disambiguator("  Ann Lee (actress)  "), "Ann Lee")


class FoldAccentsTest(unittest.TestCase):
    def test_folds_and_keeps_case(self):
        self.assertEqual(fold_accents("Margareta Cederschiöld"), "Margareta Cederschiold")
        self.assertEqual(fold_accents("ÉMILE ZOLA"), "EMILE ZOLA")

    def test_keeps_punctuation(self):
        self.assertEqual(fold_accents("Jean-Luc Garçon (père)"), "Jean-Luc Garcon (pere)")

    def test_unchanged_is_none(self):
        self.assertIsNone(fold_accents("Charles Darwin"))

    def test_undecomposable_letters_are_not_invented_away(self):
        # NFKD leaves o-slash alone; folding must not transliterate it by guess.
        self.assertIsNone(fold_accents("Søren Kierkegaard"))
        self.assertEqual(fold_accents("Søren Kierkegård"), "Søren Kierkegard")

    def test_result_is_nfc(self):
        import unicodedata

        out = fold_accents("Ángel Ñandú")
        self.assertEqual(out, unicodedata.normalize("NFC", out))
        self.assertEqual(out, "Angel Nandu")


class QueryStringsTest(unittest.TestCase):
    def qs(self, name):
        return query_strings(name, ALIAS_TABLE, SNAPSHOT)

    def test_source_labels_and_order(self):
        qs = self.qs("Margareta Cederschiöld")
        self.assertEqual([q.source for q in qs], ["name", "folded"])
        self.assertTrue(all(q.source in SOURCES for q in qs))
        self.assertEqual(
            [q.string for q in qs], ["Margareta Cederschiöld", "Margareta Cederschiold"]
        )

    def test_name_is_first_and_as_written(self):
        qs = self.qs("Mahatma Gandhi")
        self.assertEqual(qs[0], QueryString("Mahatma Gandhi", "Mahatma Gandhi", "name", "Q1", False, False))

    def test_aliases_use_snapshot_casing_not_table_keys(self):
        strings = [q.string for q in self.qs("Mahatma Gandhi")]
        self.assertIn("M. K. Gandhi", strings)
        # The table key for that alias is "m k gandhi": it must never appear.
        for key in ("m k gandhi", "mahatma gandhi", "bapu"):
            self.assertNotIn(key, strings)
        for s in strings:
            self.assertEqual(s, s.strip())

    def test_aliases_skip_single_word_and_multi_qid_surfaces(self):
        qs = self.qs("Mahatma Gandhi")
        aliases = by_source(qs, "alias")
        self.assertEqual(aliases, ["M. K. Gandhi"])
        self.assertNotIn("Bapu", aliases)  # one word
        self.assertNotIn("Gandhi", aliases)  # one word and 2 QIDs
        self.assertNotIn("Mohandas Gandhi", aliases)  # two words, but 2 QIDs

    def test_alias_carries_the_entity_qid(self):
        (alias,) = [q for q in self.qs("Mahatma Gandhi") if q.source == "alias"]
        self.assertEqual(alias.qid, "Q1")
        self.assertFalse(alias.ambiguous_name)
        self.assertFalse(alias.ambiguous_stripped)

    def test_multi_qid_name_gets_no_aliases_and_is_flagged(self):
        qs = self.qs("John Smith")
        self.assertEqual([q.source for q in qs], ["name"])
        self.assertTrue(qs[0].ambiguous_name)
        self.assertIsNone(qs[0].qid)
        # The snapshot has aliases for Q6, but the name could be Q7 as well.
        self.assertNotIn("Captain John Smith", [q.string for q in qs])

    def test_unknown_name_gets_no_aliases(self):
        qs = self.qs("Nobody Atall")
        self.assertEqual(len(qs), 1)
        self.assertEqual(qs[0].source, "name")
        self.assertIsNone(qs[0].qid)
        self.assertFalse(qs[0].ambiguous_name)

    def test_single_qid_missing_from_snapshot_is_skipped_silently(self):
        table = AliasTable({"ada lovelace": ["Q99"]})
        qs = query_strings("Ada Lovelace", table, SNAPSHOT)
        self.assertEqual([q.source for q in qs], ["name"])
        self.assertEqual(qs[0].qid, "Q99")

    def test_stripped_form_is_ambiguous_and_has_no_qid(self):
        qs = self.qs("Bill Hutchinson (baseball)")
        self.assertEqual(qs[0].string, "Bill Hutchinson (baseball)")
        self.assertEqual(qs[0].qid, "Q4")
        self.assertFalse(qs[0].ambiguous_stripped)
        stripped = [q for q in qs if q.source == "stripped"]
        self.assertEqual([q.string for q in stripped], ["Bill Hutchinson"])
        self.assertIsNone(stripped[0].qid)
        self.assertTrue(stripped[0].ambiguous_stripped)

    def test_dedupe_first_source_wins(self):
        # The label "Bill Hutchinson" equals the stripped form and is also a
        # 2-QID surface; "William Hutchinson" is a fresh alias.
        qs = self.qs("Bill Hutchinson (baseball)")
        strings = [q.string for q in qs]
        self.assertEqual(len(strings), len(set(strings)))
        self.assertEqual(by_source(qs, "alias"), ["William Hutchinson"])
        # Mahatma Gandhi's label repeats in its own alias list.
        strings = [q.string for q in self.qs("Mahatma Gandhi")]
        self.assertEqual(strings.count("Mahatma Gandhi"), 1)
        self.assertEqual(
            [q.source for q in self.qs("Mahatma Gandhi") if q.string == "Mahatma Gandhi"], ["name"]
        )

    def test_alias_equal_to_folded_form_is_deduped(self):
        # The snapshot lists the folded spelling as an alias; the folded source
        # comes first, so it is not repeated.
        qs = self.qs("Margareta Cederschiöld")
        self.assertEqual(sum(q.string == "Margareta Cederschiold" for q in qs), 1)
        self.assertEqual(by_source(qs, "folded"), ["Margareta Cederschiold"])
        self.assertEqual(by_source(qs, "alias"), [])

    def test_folded_qid_comes_from_the_table(self):
        (folded,) = [q for q in self.qs("Margareta Cederschiöld") if q.source == "folded"]
        self.assertEqual(folded.qid, "Q3")

    def test_disambiguator_and_accents_fold_both_forms(self):
        qs = self.qs("José Martín (footballer)")
        self.assertEqual(
            [(q.string, q.source) for q in qs],
            [
                ("José Martín (footballer)", "name"),
                ("José Martín", "stripped"),
                ("Jose Martin (footballer)", "folded"),
                ("Jose Martin", "folded"),
            ],
        )
        # The folded stripped form mixes namesakes just like the stripped form.
        self.assertIsNone(qs[3].qid)
        self.assertTrue(qs[3].ambiguous_stripped)
        self.assertEqual(qs[2].qid, "Q8")
        # The label "José Martín" duplicates the stripped form and is dropped.
        self.assertEqual(by_source(qs, "alias"), [])

    def test_prompt_name_and_flags_on_every_string(self):
        for q in self.qs("Bill Hutchinson (baseball)"):
            self.assertEqual(q.prompt_name, "Bill Hutchinson (baseball)")
            self.assertFalse(q.ambiguous_name)

    def test_blank_and_padded_surfaces(self):
        snap = {"entities": {"Q1": {"label": "  Mahatma Gandhi ", "aliases": ["", "  ", " M. K. Gandhi "]}}}
        qs = query_strings("Mahatma Gandhi", ALIAS_TABLE, snap)
        self.assertEqual([q.string for q in qs], ["Mahatma Gandhi", "M. K. Gandhi"])


class GluedIdsTest(unittest.TestCase):
    def test_drops_leading_paren_token(self):
        tokens = ["▁(", "O", "berto", "▁Pal", "lav", "ic", "ino"]
        ids = [29898, 29949, 2496, 3793, 22288, 293, 1789]
        self.assertEqual(glued_ids(ids, tokens), ids[1:])

    def test_two_token_variant(self):
        tokens = ["▁", "(", "O", "berto"]
        self.assertEqual(glued_ids([1, 2, 3, 4], tokens), [3, 4])

    def test_rejects_plain_form(self):
        tokens = ["▁Ober", "to", "▁Pal"]
        with self.assertRaises(ValueError):
            glued_ids([1, 2, 3], tokens)

    def test_rejects_paren_without_space_marker(self):
        with self.assertRaises(ValueError):
            glued_ids([1, 2], ["(", "O"])

    def test_rejects_space_marker_then_other_token(self):
        with self.assertRaises(ValueError):
            glued_ids([1, 2], ["▁", "O"])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            glued_ids([], [])

    def test_rejects_nothing_left(self):
        with self.assertRaises(ValueError):
            glued_ids([1], ["▁("])

    def test_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            glued_ids([1, 2, 3], ["▁(", "O"])

    def test_does_not_mutate_inputs(self):
        ids, tokens = [1, 2, 3], ["▁(", "a", "b"]
        glued_ids(ids, tokens)
        self.assertEqual(ids, [1, 2, 3])
        self.assertEqual(tokens, ["▁(", "a", "b"])


class LoadNamesTest(unittest.TestCase):
    def write_csv(self, rows):
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        self.addCleanup(os.remove, path)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt_id", "task", "source_file", "source_row", "prompt", "task_meta"])
            for i, (task, meta, prompt) in enumerate(rows):
                w.writerow([f"p{i}", task, "x.csv", i, prompt, json.dumps(meta, ensure_ascii=False)])
        return path

    def test_unique_sorted_and_task_filtered(self):
        path = self.write_csv(
            [
                ("historical_events", {"name1": "Zed Zorn", "name2": "Ann Lee"}, "p"),
                ("historical_events", {"name1": "Ann Lee", "name2": "Bob Ray (actor)"}, "p"),
                ("biographies", {"name": "Ada Bio"}, "p"),
                ("biographies", {"name": "Ada Bio"}, "p"),
                ("biographies", {"name1": "Not A Biographies Field"}, "p"),
            ]
        )
        self.assertEqual(load_names(path), ["Ann Lee", "Bob Ray (actor)", "Zed Zorn"])
        # Biographies names its one subject `name`; a name1 there is not read.
        self.assertEqual(load_names(path, task="biographies"), ["Ada Bio"])
        self.assertEqual(load_names(path, task="nothing"), [])

    def test_non_ascii_round_trip(self):
        path = self.write_csv(
            [("historical_events", {"name1": "Margareta Cederschiöld", "name2": "Søren Å"}, "p")]
        )
        self.assertEqual(load_names(path), sorted(["Margareta Cederschiöld", "Søren Å"]))

    def test_long_prompt_text_and_missing_names(self):
        path = self.write_csv(
            [
                ("historical_events", {"name1": "Ann Lee", "name2": None}, "x" * 300_000),
                ("historical_events", {"name1": "", "birth1": 1}, "p"),
            ]
        )
        self.assertEqual(load_names(path), ["Ann Lee"])

    def test_multiline_prompt_and_extra_meta_keys(self):
        path = self.write_csv(
            [
                (
                    "historical_events",
                    {"name1": "Ann Lee", "name2": "Bo Ray", "birth1": 1, "similar_names": ["Q"]},
                    "line one\nline two, with \"quotes\"",
                )
            ]
        )
        self.assertEqual(load_names(path), ["Ann Lee", "Bo Ray"])


if __name__ == "__main__":
    unittest.main()
