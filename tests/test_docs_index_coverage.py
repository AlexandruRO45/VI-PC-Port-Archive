"""Every note under docs/ is named in the documentation index (docs/README.md).

AGENTS.md sends readers to docs/README.md for the notes; a page the index does not link is one nobody finds there
(dev-tools.md was the one page missing when this test was written). Reads the repository's own text only, no game data.
"""
import os
import tempfile
import unittest

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")


def missing_pages(docs_dir):
    """The note names in `docs_dir` that the index there (README.md) does not link."""
    with open(os.path.join(docs_dir, "README.md"), encoding="utf-8") as f:
        index = f.read()
    pages = sorted(name for name in os.listdir(docs_dir)
                   if name.endswith(".md") and name != "README.md" and os.path.isfile(os.path.join(docs_dir, name)))
    return [name for name in pages if f"]({name})" not in index]


class DocsIndexCoverageTest(unittest.TestCase):
    def test_every_note_is_linked_from_the_index(self):
        missing = missing_pages(DOCS)
        self.assertEqual([], missing, "docs/README.md does not link " + ", ".join(missing))

    def test_an_unlinked_note_is_reported(self):
        # The check above must fail on a page nobody indexed, not pass because it read nothing.
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8", newline="\n") as f:
                f.write("# Documentation\n\n| Document | Covers |\n|---|---|\n| [a.md](a.md) | linked |\n")
            for name in ("a.md", "b.md"):
                with open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n") as f:
                    f.write("# " + name + "\n")
            self.assertEqual(["b.md"], missing_pages(tmp))


if __name__ == "__main__":
    unittest.main()
