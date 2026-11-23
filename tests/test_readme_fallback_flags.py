"""README.md names every fallback flag `make_game.py` forwards to the translator.

`make_game.py`'s `FALLBACK_FLAGS` is what decides whether a work folder's translated shaders are reused: a flag
missing from it would not be recorded in `translator_flags.txt`, and a flag missing from README is one nobody knows
to pass. The README paragraph said "Four fallback flags" and listed four after `--f16-interp-fallback` and
`--pc-tail-ubo` were added (EGEN-README-FALLBACK-FLAGS), so this reads both files with `ast` and plain text
(standard library, no game data, no GPU).
"""
import ast
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
READ_ME = os.path.join(ROOT, "README.md")
MAKE_GAME = os.path.join(ROOT, "make_game.py")

NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
                9: "Nine", 10: "Ten"}
COUNTED = re.compile(r"^(\w+) fallback flags are forwarded to the translator", re.MULTILINE)
FLAG = re.compile(r"`(--[a-z0-9-]+)`")


def fallback_flags():
    """The flag names in make_game.py's FALLBACK_FLAGS tuple, read without importing the script."""
    with open(MAKE_GAME, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), MAKE_GAME)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "FALLBACK_FLAGS" for t in node.targets):
            return [entry.elts[0].value for entry in node.value.elts]
    raise AssertionError("make_game.py has no FALLBACK_FLAGS")


def readme_paragraph():
    """The README paragraph that lists the fallback flags, from its first line to the next blank line."""
    with open(READ_ME, encoding="utf-8") as handle:
        text = handle.read().replace("\r\n", "\n")
    match = COUNTED.search(text)
    assert match, "README.md has no paragraph starting '<n> fallback flags are forwarded to the translator'"
    start = match.start()
    end = text.find("\n\n", start)
    return match.group(1), text[start:end if end != -1 else len(text)]


class ReadmeFallbackFlagsTest(unittest.TestCase):
    def test_readme_names_every_fallback_flag(self):
        flags = fallback_flags()
        _, paragraph = readme_paragraph()
        listed = FLAG.findall(paragraph)
        self.assertEqual(sorted(listed), sorted(flags),
                         f"README's fallback paragraph lists {listed}; make_game.py FALLBACK_FLAGS is {flags}")

    def test_readme_counts_the_flags_it_lists(self):
        count, paragraph = readme_paragraph()
        listed = FLAG.findall(paragraph)
        self.assertEqual(count, NUMBER_WORDS.get(len(listed), count),
                         f"README says '{count} fallback flags' and lists {len(listed)}")


if __name__ == "__main__":
    unittest.main()
