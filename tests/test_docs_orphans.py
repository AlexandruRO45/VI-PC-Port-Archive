"""Every page under docs/ is linked from another Markdown page.

    python -I tests/test_docs_orphans.py

`docs/dev-tools.md` shipped with no inbound link: `docs/README.md` had no row for it and no other page
named it, so the only way to reach it was to guess the path. This reads the tracked Markdown files
(`git ls-files '*.md'`, with a directory walk when git is not available), resolves each relative link
against the page that holds it and reports a `docs/*.md` page that no other page reaches.

The checks are exercised on a synthetic tree and a hand-built link table: an orphan is reported, a page
with an inbound link is not, and a link that only appears inside a fenced sample does not count. No game
data is read.
"""
import os
import re
import subprocess
import tempfile
import unittest
from urllib.parse import unquote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FENCE = re.compile(r"```.*?```", re.S)
LINK = re.compile(r"!?\[[^\]]*\]\((?:<([^>]+)>|([^)\s]+))")
EXTERNAL = ("http://", "https://", "mailto:", "tel:", "data:")


def strip_fences(text):
    """Drop fenced code blocks: a path quoted in a sample is not a link."""
    return FENCE.sub("", text)


def targets(text):
    """Relative link and image targets on a page, fenced code excluded."""
    return [match.group(1) or match.group(2) for match in LINK.finditer(strip_fences(text))]


def markdown_files(root):
    """Tracked Markdown files as absolute paths; a directory walk without git."""
    try:
        result = subprocess.run(["git", "ls-files", "-z", "--", "*.md"], cwd=root, capture_output=True)
    except FileNotFoundError:
        result = None
    if result is not None and result.returncode == 0:
        return [os.path.join(root, os.fsdecode(name)) for name in result.stdout.split(b"\0") if name]
    found = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git", "third_party", "node_modules", "__pycache__")]
        found.extend(os.path.join(base, name) for name in names if name.endswith(".md"))
    return found


def read_page(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def reached_pages(files, reads=None):
    """Case-normalised real paths of every file a relative link on one of `files` reaches."""
    reads = reads or read_page
    reached = set()
    for source in files:
        try:
            text = reads(source)
        except OSError:
            continue
        for target in targets(text):
            if target.startswith(EXTERNAL) or target.startswith("#"):
                continue
            path = os.path.normpath(os.path.join(os.path.dirname(source),
                                                 unquote(target.split("#", 1)[0].split("?", 1)[0])))
            if os.path.normcase(path) != os.path.normcase(source):
                reached.add(os.path.normcase(path))
    return reached


def orphan_pages(docs_dir, files, reads=None):
    """Pages beside the index that no other page links, in name order."""
    reached = reached_pages(files, reads)
    return sorted(name for name in os.listdir(docs_dir)
                  if name.endswith(".md") and os.path.isfile(os.path.join(docs_dir, name))
                  and os.path.normcase(os.path.join(docs_dir, name)) not in reached)


class OrphanPages(unittest.TestCase):
    def test_every_docs_page_has_an_inbound_link(self):
        self.assertEqual(orphan_pages(os.path.join(ROOT, "docs"), markdown_files(ROOT)), [])

    def test_an_orphan_is_reported_and_a_linked_page_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = os.path.join(tmp, "docs")
            os.makedirs(docs)
            for name in ("README.md", "linked.md", "lost.md"):
                with open(os.path.join(docs, name), "w", encoding="utf-8") as f:
                    f.write("# page\n")
            index = os.path.join(docs, "README.md")
            pages = [index, os.path.join(docs, "linked.md"), os.path.join(docs, "lost.md")]
            reads = {index: "| [linked.md](linked.md) | a row |\n",
                     pages[1]: "# linked\n", pages[2]: "# lost\n",
                     os.path.join(tmp, "README.md"): "[docs](docs/README.md)\n"}
            self.assertEqual(orphan_pages(docs, pages + [os.path.join(tmp, "README.md")],
                                          reads.__getitem__), ["lost.md"])

    def test_a_link_inside_a_fenced_sample_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            docs = os.path.join(tmp, "docs")
            os.makedirs(docs)
            for name in ("README.md", "quoted.md"):
                with open(os.path.join(docs, name), "w", encoding="utf-8") as f:
                    f.write("# page\n")
            index = os.path.join(docs, "README.md")
            page = os.path.join(docs, "quoted.md")
            root_page = os.path.join(tmp, "README.md")
            reads = {index: "```\n[quoted.md](quoted.md)\n```\n", page: "# quoted\n",
                     root_page: "[docs](docs/README.md)\n"}
            self.assertEqual(orphan_pages(docs, [index, page, root_page], reads.__getitem__), ["quoted.md"])


if __name__ == "__main__":
    unittest.main()
