"""README names every fallback flag make_game.py forwards, and says how many there are.

    python -I tests/test_docs_fallback_flags.py

`make_game.FALLBACK_FLAGS` is the set of translator fallbacks the build forwards and saves next to the translated
shaders. A flag added there but not to the README leaves the users of the device it exists for without a way to know
it exists: `--f16-interp-fallback` landed in #1407 for GPUs that cannot use per-vertex inputs (MoltenVK), and the
README kept listing four flags after the build forwarded five. Reads README.md and imports make_game only; no game
files, no GPU.
"""
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game

COUNT_WORDS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
               6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}
COUNT = re.compile(r"\b(" + "|".join(COUNT_WORDS.values()) + r") fallback flags are forwarded")


def missing_flags(readme, flags):
    """The flags `readme` never names in backticks."""
    return [flag for flag in flags if f"`{flag}`" not in readme]


def stated_count(readme):
    """The number word the README's "N fallback flags are forwarded" sentence starts with, or None."""
    m = COUNT.search(readme)
    return m.group(1) if m else None


class CheckerTest(unittest.TestCase):
    def test_reports_only_the_flag_the_text_does_not_name(self):
        self.assertEqual(missing_flags("`--a` and --b", ["--a", "--b"]), ["--b"])

    def test_reads_the_number_word(self):
        self.assertEqual(stated_count("Four fallback flags are forwarded to the translator"), "Four")
        self.assertIsNone(stated_count("The fallback flags are forwarded"))


class ReadmeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
            cls.readme = f.read()
        cls.flags = [flag for flag, _ in make_game.FALLBACK_FLAGS]

    def test_readme_names_every_forwarded_fallback(self):
        self.assertEqual(missing_flags(self.readme, self.flags), [],
                         "fallback flags make_game.py forwards that README.md never names")

    def test_readme_states_how_many_there_are(self):
        self.assertIn(len(self.flags), COUNT_WORDS, "add the new number word to COUNT_WORDS")
        self.assertEqual(stated_count(self.readme), COUNT_WORDS[len(self.flags)],
                         "README.md's fallback-flag count does not match make_game.FALLBACK_FLAGS")


if __name__ == "__main__":
    unittest.main()
