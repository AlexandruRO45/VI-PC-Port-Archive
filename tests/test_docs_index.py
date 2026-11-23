"""Every page under `docs/` is linked from `docs/README.md`.

    python -I tests/test_docs_index.py

A page nothing links is invisible to a reader who starts at the index: `docs/modding.md` was on `main` without a
row. The index also points into `docs/i18n/`, so only the link's file name is compared. Reads tracked Markdown
only; no game files.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
LINK = re.compile(r"\]\(([^)\s]+)\)")


def unlisted(index_text, pages):
    """Pages (names under `docs/`) that `index_text` never links to."""
    linked = {os.path.basename(target.split("#")[0]) for target in LINK.findall(index_text)}
    return sorted(page for page in pages if page not in linked)


class DocsIndexTest(unittest.TestCase):
    def test_every_page_is_linked(self):
        with open(os.path.join(DOCS, "README.md"), encoding="utf-8") as f:
            index = f.read()
        pages = sorted(name for name in os.listdir(DOCS) if name.endswith(".md") and name != "README.md")
        self.assertGreaterEqual(len(pages), 50)
        self.assertEqual(unlisted(index, pages), [],
                         "pages under docs/ that docs/README.md does not link: " +
                         ", ".join(unlisted(index, pages)))

    def test_an_unlinked_page_is_reported(self):
        index = "| [design.md](design.md) | x |\n| [i18n/README.md](i18n/README.md) | y |\n"
        self.assertEqual(unlisted(index, ["design.md", "modding.md"]), ["modding.md"])
        self.assertEqual(unlisted(index + "| [modding.md](modding.md#mods) | z |\n",
                                  ["design.md", "modding.md"]), [])


if __name__ == "__main__":
    unittest.main()
