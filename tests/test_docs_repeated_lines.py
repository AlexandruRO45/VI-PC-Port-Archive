"""No two neighbouring prose lines, and no neighbouring block of lines, repeat in the Markdown docs.

    python -I tests/test_docs_repeated_lines.py

When two branches rewrite the same sentence and a merge keeps both sides, the page shows the sentence twice on
consecutive lines: the current text and the one it replaced, usually with a different number in it (docs/d3d12_vulkan.md
had "the one left needs `buffer_store_short`" twice, docs/animation.md gave both 350,251 and 382,781 for the same
export, docs/models.md said its test file runs 10 and 15 tests). A reader cannot tell which line is current.
`tests/test_docs_list_items.py` does this for list items and `tests/test_docs_table_rows.py` for table rows; this
flags two prose lines (neither a list item, table row, heading or code) within WINDOW lines of each other whose first
OPENING characters are equal. The same merge habit also pastes a whole section twice, heading, table and code
included, which the line check cannot see; `repeated_blocks` catches that one: a block of BLOCK lines or more that is
immediately followed by a copy of itself. Reads tracked Markdown only; no game files.
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW = 3     # lines between the two (1 = neighbours)
OPENING = 40   # characters, after the indentation, that must match
BLOCK = 6      # lines, for a block pasted twice in a row
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


def repeated_blocks(text, min_lines=BLOCK):
    """[(first line, second line, length)] for a block of lines immediately followed by a copy of itself."""
    lines = text.split("\n")
    where = {}
    for index, line in enumerate(lines):
        where.setdefault(line, []).append(index)
    found = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        for j in where[line]:
            if j - i < min_lines:
                continue
            length = 0
            while j + length < len(lines) and lines[i + length] == lines[j + length]:
                length += 1
            if length < j - i:      # the block is not followed by a full copy of itself
                continue
            length = j - i
            while length and not lines[i + length - 1].strip():   # a copied section ends with its blank line
                length -= 1
            found.append((i + 1, j + 1, length))
            break
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

    def test_block_checker_flags_a_repeated_block_only(self):
        block = ["First line of the block.", "", "Third line.", "Fourth line.", "Fifth line.", "Sixth line."]
        self.assertEqual(repeated_blocks("\n".join(["intro", ""] + block + block + ["", "outro"])), [(3, 9, 6)])
        short = ["one.", "two.", "three.", "four.", "five."]
        self.assertEqual(repeated_blocks("\n".join(short + short)), [], "under BLOCK lines")
        self.assertEqual(repeated_blocks("\n".join(block + ["between"] + block)), [], "not pasted next to itself")

    def test_docs_have_no_repeated_neighbouring_blocks(self):
        found = []
        for path in markdown_files():
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                found += [f"{path}:{a}-{a + n - 1} repeated at :{b}-{b + n - 1} ({n} lines)"
                          for a, b, n in repeated_blocks(f.read())]
        self.assertEqual(found, [], "a block pasted just below itself (keep the current one): " + "; ".join(found))


if __name__ == "__main__":
    unittest.main()
