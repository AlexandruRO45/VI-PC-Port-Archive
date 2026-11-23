"""Every page under docs/ has a row in the docs/README.md index.

    python -I tests/test_docs_index_covers_pages.py

A page nothing links to is invisible to a reader who starts at the index. `docs/dev-tools.md` and
`docs/gpu_test_results.md` reached `main` that way (#6590, #6720): the pull request added the page and no row, so the
index skipped it. The index is a table of relative links, so a top-level page is indexed when `docs/README.md` links
`<name>.md`; the index itself is the one page that needs no row.

Reads only the repository's Markdown; no game files, no build tools.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
INDEX = "README.md"
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def linked_targets(text):
    """The normalised paths the page links, dropping any #anchor or ?query."""
    return {os.path.normpath(target.split("#", 1)[0].split("?", 1)[0])
            for target in LINK.findall(text) if not target.startswith(("http://", "https://", "mailto:"))}


class EveryPageIsIndexed(unittest.TestCase):
    def test_a_page_with_no_row_in_the_index_is_named(self):
        with open(os.path.join(DOCS, INDEX), encoding="utf-8") as f:
            targets = linked_targets(f.read())
        pages = sorted(name for name in os.listdir(DOCS) if name.endswith(".md") and name != INDEX)
        missing = [name for name in pages if os.path.normpath(name) not in targets]
        self.assertEqual(missing, [], "docs/*.md pages with no row in docs/README.md")

    def test_the_index_links_pages_it_names(self):
        # A row that reads like a page but is not a link (a bare `<name>.md`) leaves the page findable by search only.
        with open(os.path.join(DOCS, INDEX), encoding="utf-8") as f:
            text = f.read()
        bare = re.findall(r"^\| (?!\[)[^|]*?(\w[\w.-]*\.md)\b", text, re.M)
        self.assertEqual(bare, [], "index rows that name a page without linking it")


if __name__ == "__main__":
    unittest.main()
