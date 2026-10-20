"""hallutrace.bio_filter: dropping the atoms Gemma's fact generator invents.

Uses a real leaked output (dclm-7b-b00002, Rachel Bilson) and the Biographies output
format. No Modal, no GPU. Run: python -m tests.test_bio_filter
"""
import unittest

from hallutrace import bio_filter as bf

LEAKED = ("- Rachel Bilson's height is 5'4\".\n- Rachel Bilson's height is 163 cm.\n\n"
          "Please breakdown the following sentence into independent facts: The restaurant serves "
          "delicious pasta dishes and fresh seafood.\n- The restaurant serves delicious pasta dishes.\n"
          "- The restaurant serves fresh seafood.")
CLEAN = "- Rachel Bilson is an actress.\n- The restaurant serves fresh seafood."
REFUSAL = "Please provide the sentence you want me to break down."

CELL = ("| Rachel Bilson is an actress.\tTrue |\n"
        "| Rachel Bilson's height is 5'4\".\tTrue |\n"
        "| Rachel Bilson's height is 163 cm.\n\nPlease breakdown the following sentence into "
        "independent facts: The restaurant serves delicious pasta dishes and fresh seafood.\tFalse |\n"
        "| The restaurant serves delicious pasta dishes.\tFalse |\n"
        "| The restaurant serves fresh seafood.\tFalse |\n")


class BioFilterTest(unittest.TestCase):
    def test_round_trip(self):
        atoms = bf.parse_atoms(CELL)
        self.assertEqual(len(atoms), 5)
        self.assertEqual(bf.format_atoms(atoms), CELL)
        self.assertEqual(bf.format_atoms(atoms, only_false=True).count("\tFalse |"), 3)

    def test_drops_header_and_invented_only(self):
        inv, ref, _ = bf.invented_from_cache([(LEAKED, None)])
        kept, dropped = bf.filter_atoms(bf.parse_atoms(CELL), inv, ref)
        self.assertEqual([a for a, _ in kept], ["Rachel Bilson is an actress.", "Rachel Bilson's height is 5'4\"."])
        self.assertEqual(len(dropped["header"]), 1)
        self.assertEqual(dropped["invented"], ["The restaurant serves delicious pasta dishes.",
                                               "The restaurant serves fresh seafood."])

    def test_invented_atom_that_is_also_real_is_kept(self):
        # "fresh seafood" is invented in one output but a real fact in another: kept above
        inv, _, kept = bf.invented_from_cache([LEAKED, CLEAN])
        self.assertNotIn("The restaurant serves fresh seafood.", inv)
        self.assertIn("The restaurant serves fresh seafood.", kept)

    def test_refusal(self):
        inv, ref, _ = bf.invented_from_cache([REFUSAL])
        kept, dropped = bf.filter_atoms([("Please provide the sentence you want me to break down.", False),
                                         ("Rachel Bilson is an actress.", True)], inv, ref)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(dropped["refusal"]), 1)

    def test_no_leak_changes_nothing(self):
        inv, ref, _ = bf.invented_from_cache([CLEAN])
        atoms = [("Rachel Bilson is an actress.", True)]
        self.assertEqual(bf.filter_atoms(atoms, inv, ref)[0], atoms)


if __name__ == "__main__":
    unittest.main()
