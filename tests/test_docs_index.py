"""docs/README.md links every page in docs/, and every row it lists exists.

    python -I tests/test_docs_index.py

`docs/README.md` is the only route into the notes, so a page merged without a row is unreachable from
the index and easy to miss when it later goes stale. `docs/modding.md` was that page: merged on its
own and never added, while the index has been edited since. Reads only `docs/*.md`; no game data.

The rules are exercised on a synthetic tree where one page has no row, one row has no page and one
page has two rows, so a checker that accepts everything, or parses nothing, cannot pass.
"""
import os
import re
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
INDEX = os.path.join(DOCS, "README.md")

ROW = re.compile(r"^\|\s*\[[^\]]*\]\((?:<([^>]+)>|([^)\s]+))")


def row_targets(text):
    """The pages the index's table rows link, as paths relative to docs/ (fragments dropped)."""
    return set(rows(text))


def rows(text):
    """One entry per table row, in order: a page listed twice appears twice. Fenced examples do not count."""
    found = []
    fence = None
    for line in text.splitlines():
        stripped = line.strip()
        if fence is None:
            opener = re.match(r"(`{3,}|~{3,})", stripped)
            if opener:
                fence = opener.group(1)
                continue
        elif stripped.startswith(fence[0] * len(fence)):
            fence = None
            continue
        else:
            continue
        match = ROW.match(stripped)
        if match:
            found.append(normalise((match.group(1) or match.group(2)).split("#", 1)[0]))
    return found


def normalise(target):
    return os.path.normpath(target).replace(os.sep, "/")


def pages(docs_dir):
    """The pages the index has to cover: every `.md` beside it, except the index itself."""
    return {name for name in os.listdir(docs_dir)
            if name.endswith(".md") and name != os.path.basename(INDEX)
            and os.path.isfile(os.path.join(docs_dir, name))}


def missing_rows(docs_dir, text):
    """Pages present in docs/ that no row links."""
    return sorted(pages(docs_dir) - row_targets(text))


def dangling_rows(docs_dir, text):
    """Rows whose target is neither a page in docs/ nor another existing file beside it."""
    return sorted(target for target in row_targets(text)
                  if target not in pages(docs_dir) and target != os.path.basename(INDEX)
                  and not os.path.exists(os.path.join(docs_dir, target)))


def duplicate_rows(text):
    """Pages linked from more than one row: the index would list the same page twice."""
    seen, twice = set(), []
    for target in rows(text):
        if target in seen and target.endswith(".md"):
            twice.append(target)
        seen.add(target)
    return sorted(twice)


def write_tree(root, names):
    for name in names:
        with open(os.path.join(root, name), "w", encoding="utf-8") as f:
            f.write("# page\n")


class IndexTest(unittest.TestCase):
    def setUp(self):
        with open(INDEX, encoding="utf-8") as f:
            self.text = f.read()

    def test_every_page_has_a_row(self):
        self.assertEqual(missing_rows(DOCS, self.text), [])

    def test_every_row_points_at_a_page(self):
        self.assertEqual(dangling_rows(DOCS, self.text), [])

    def test_no_page_is_listed_twice(self):
        self.assertEqual(duplicate_rows(self.text), [])

    def test_the_index_covers_the_pages(self):
        self.assertGreaterEqual(len(row_targets(self.text)), 40)

    def test_a_page_without_a_row_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tree(tmp, ["one.md", "two.md"])
            text = "| Document | Content |\n|---|---|\n| [one.md](one.md) | listed |\n"
            self.assertEqual(missing_rows(tmp, text), ["two.md"])
            self.assertEqual(missing_rows(tmp, text + "| [two.md](two.md) | listed |\n"), [])

    def test_a_row_inside_a_fenced_example_does_not_cover_a_page(self):
        for mark in ("```", "~~~"):
            with self.subTest(fence=mark), tempfile.TemporaryDirectory() as tmp:
                write_tree(tmp, ["one.md", "two.md"])
                text = ("| Document | Content |\n|---|---|\n| [one.md](one.md) | listed |\n"
                        f"{mark}markdown\n| [two.md](two.md) | example |\n{mark}\n")
                self.assertEqual(missing_rows(tmp, text), ["two.md"])
                self.assertEqual(missing_rows(tmp, text + "| [two.md](two.md) | listed |\n"), [])

    def test_a_row_without_a_page_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tree(tmp, ["one.md"])
            text = "| Document | Content |\n|---|---|\n| [one.md](one.md) | listed |\n"
            self.assertEqual(dangling_rows(tmp, text), [])
            self.assertEqual(dangling_rows(tmp, text + "| [gone.md](gone.md) | listed |\n"),
                             ["gone.md"])

    def test_a_page_with_two_rows_is_reported(self):
        text = ("| Document | Content |\n|---|---|\n"
                "| [one.md](one.md) | first |\n"
                "| [one.md](one.md) | again |\n")
        self.assertEqual(duplicate_rows(text), ["one.md"])


if __name__ == "__main__":
    unittest.main()
