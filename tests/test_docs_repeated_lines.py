"""No two neighbouring prose lines in the Markdown docs start the same way.

    python -I tests/test_docs_repeated_lines.py

When two branches rewrite the same sentence and a merge keeps both sides, the page shows the sentence twice on
consecutive lines: the current text and the one it replaced, usually with a different number in it (docs/d3d12_vulkan.md
had "the one left needs `buffer_store_short`" twice, docs/animation.md gave both 350,251 and 382,781 for the same
export, docs/models.md said its test file runs 10 and 15 tests). A reader cannot tell which line is current.
`tests/test_docs_list_items.py` does this for list items and `tests/test_docs_table_rows.py` for table rows; this
flags two prose lines (neither a list item, table row, heading or code) within WINDOW lines of each other whose first
OPENING characters are equal. Reads tracked Markdown only; no game files.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = 3     # lines between the two (1 = neighbours)
OPENING = 40   # characters, after the indentation, that must match
LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s")


def repeated_lines(text):
    """[(line a, line b)] for prose lines within WINDOW lines whose first OPENING characters are equal."""
    found, in_fence, seen = [], False, {}
    for n, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        s = line.strip()
        if in_fence or len(s) < OPENING or s[0] in "|#>" or LIST_ITEM.match(s):
            continue
        key = s[:OPENING]
        if key in seen and n - seen[key] <= WINDOW:
            found.append((seen[key], n))
        seen[key] = n
    return found


def markdown_files():
    try:
        out = subprocess.run(["git", "ls-files", "-z", "*.md"], cwd=ROOT, capture_output=True, check=True).stdout
        files = [os.fsdecode(path) for path in out.split(b"\0") if path]
    except (OSError, subprocess.CalledProcessError):
        files = [os.path.relpath(os.path.join(r, f), ROOT).replace(os.sep, "/")
                 for r, d, fs in os.walk(ROOT) if ".git" not in r.split(os.sep) for f in fs if f.endswith(".md")]
    return [f for f in files if not f.startswith("third_party/")]


class RepeatedLines(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "requires git")
    def test_reads_tracked_markdown_with_spaces_and_unicode(self):
        names = ["space name.md", "café.md"]
        line = "A synthetic sentence long enough to trigger the repeated prose check."
        with tempfile.TemporaryDirectory() as root, patch.dict(globals(), ROOT=root):
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "core.quotePath", "true"], cwd=root, check=True)
            for name in names:
                (Path(root) / name).write_bytes((line + "\n" + line).encode("utf-8"))
            (Path(root) / "untracked.md").touch()
            subprocess.run(["git", "-c", "core.autocrlf=false", "add", "--", *names], cwd=root, check=True)

            paths = markdown_files()
            self.assertCountEqual(paths, names)
            for path in paths:
                with self.subTest(path=path):
                    self.assertEqual(repeated_lines((Path(root) / path).read_text(encoding="utf-8")), [(1, 2)])

    def test_checker_flags_neighbouring_repeats_only(self):
        a = "The disassembly count is unchanged: the same 42 programs have no invalid word, and"
        text = "\n".join([
            a + " 10 do.", a + " ten do.",                         # 1-2: repeated neighbours
            "x", "y", "z", "w",
            a + " later.",                                           # 7: four lines on, not flagged
            "- " + a, "- " + a,                                      # list items are tests/test_docs_list_items.py's
            "| " + a, "| " + a,                                      # table rows are tests/test_docs_table_rows.py's
            "```", a, a, "```",                                      # inside a code fence
            "short", "short",                                        # shorter than OPENING
        ])
        self.assertEqual(repeated_lines(text), [(1, 2)])

    def test_docs_have_no_repeated_neighbouring_lines(self):
        found = []
        for path in markdown_files():
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                found += [f"{path}:{a} and :{b}" for a, b in repeated_lines(f.read())]
        self.assertEqual(found, [], "a line repeated just below itself (keep the current one): " + "; ".join(found))


if __name__ == "__main__":
    unittest.main()
