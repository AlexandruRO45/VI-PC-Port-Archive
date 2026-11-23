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
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = 3     # lines between the two (1 = neighbours)
OPENING = 40   # characters, after the indentation, that must match
LIST_ITEM = re.compile(r"^(?:[-*+]|\d+[.)])\s")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def repeated_lines(text):
    """[(line a, line b)] for prose lines within WINDOW lines whose first OPENING characters are equal."""
    found, fence, seen = [], None, {}
    for n, line in enumerate(text.split("\n"), 1):
        marker = FENCE.match(line)
        if fence is not None:
            if (marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence)
                    and not marker[2].strip()):
                fence = None
            continue
        if marker and not (marker[1][0] == "`" and "`" in marker[2]):
            fence = marker[1]
            continue
        s = line.strip()
        if len(s) < OPENING or s[0] in "|#>" or LIST_ITEM.match(s):
            continue
        key = s[:OPENING]
        if key in seen and n - seen[key] <= WINDOW:
            found.append((seen[key], n))
        seen[key] = n
    return found


def markdown_files():
    try:
        out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = out.split()
    except (OSError, subprocess.CalledProcessError):
        files = [os.path.relpath(os.path.join(r, f), ROOT).replace(os.sep, "/")
                 for r, d, fs in os.walk(ROOT) if ".git" not in r.split(os.sep) for f in fs if f.endswith(".md")]
    return [f for f in files if not f.startswith("third_party/")]


class RepeatedLines(unittest.TestCase):
    def test_fenced_code_is_not_prose(self):
        a = "This repeated example line is long enough to trigger the prose checker."
        for opening, inner, closing in [
            ("~~~text", "```", "~~~"),
            ("````markdown", "```", "````"),
            ("~~~text", "~~", "~~~~"),
            ("```text", "```not a closing fence", "````"),
            ("   ~~~text", "~~~not a closing fence", "   ~~~"),
        ]:
            with self.subTest(opening=opening, inner=inner):
                text = "\n".join([opening, inner, a, a, closing, a, a])
                self.assertEqual(repeated_lines(text), [(6, 7)])

    def test_backtick_in_info_string_does_not_open_a_fence(self):
        a = "This repeated prose line is long enough to trigger the prose checker."
        self.assertEqual(repeated_lines("\n".join(["```invalid`info", a, a])), [(2, 3)])

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
