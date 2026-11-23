"""No Markdown table names the same bold ID in two of its rows.

    python -I tests/test_docs_table_ids.py

Tables whose rows are keyed by a bold ID in the first cell (`| **M2** | ...`, the milestones in
docs/d3d12_vulkan.md section 6) are edited one row at a time. When two branches rewrite the same row and a merge
keeps both sides, the table has the row twice, the older text under or near the newer one: section 6 had M2, M3, M6
and M7 twice each, and its two M6 rows disagreed on how many FSSM programs pass. A reader cannot tell which copy is
current. Tables keyed by plain values (sizes, counts, names) repeat them on purpose and are not checked. Reads
tracked Markdown only; no game files.
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOLD_ID = re.compile(r"^\*\*[^*]+\*\*$")  # the whole first cell is bold
SEPARATOR = re.compile(r"^\|\s*:?-{3}")
FENCE_START = re.compile(r"^(`{3,}|~{3,})")


def repeated_ids(text):
    """[(line a, line b, id)] for two rows of one table whose first cells are the same bold ID."""
    found, seen, in_table = [], {}, False
    fence_char, fence_len = None, 0
    for n, line in enumerate(text.split("\n"), 1):
        stripped = line.lstrip()
        if fence_char is None:
            m = FENCE_START.match(stripped)
            if m:
                fence_char, fence_len = m.group(1)[0], len(m.group(1))
                in_table = False
                continue
        else:
            close_pat = r"^" + re.escape(fence_char) + r"{" + str(fence_len) + r",}\s*$"
            if re.match(close_pat, stripped):
                fence_char, fence_len = None, 0
                in_table = False
                continue
        in_fence = fence_char is not None
        row = not in_fence and line.startswith("|") and not SEPARATOR.match(line)
        if row and not in_table:
            seen = {}
        in_table = row or (not in_fence and SEPARATOR.match(line) is not None)
        if not row:
            continue
        cells = line.split("|")
        first = cells[1].strip() if len(cells) > 2 else ""
        if not BOLD_ID.match(first):
            continue
        if first in seen:
            found.append((seen[first], n, first))
        else:
            seen[first] = n
    return found


def markdown_files():
    try:
        out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = out.split()
    except (OSError, subprocess.CalledProcessError):
        files = [os.path.relpath(os.path.join(r, f), ROOT).replace(os.sep, "/")
                 for r, d, fs in os.walk(ROOT) if ".git" not in r for f in fs if f.endswith(".md")]
    return [f for f in files if not f.startswith("third_party/")]


class RepeatedTableIds(unittest.TestCase):
    def test_checker_flags_a_repeated_bold_id_only(self):
        text = "\n".join([
            "| # | Scope |", "|---|---|",
            "| **M2** | new text |", "| **M3** | x |", "| **M2** | old text |",  # repeated: flagged
            "| 1k | a |", "| 1k | b |",                                       # plain values: not checked
            "",
            "| # | Scope |", "|---|---|", "| **M2** | another table |",       # a new table starts over
            "```", "| **M3** | in a fence |", "| **M3** | in a fence |", "```",
            "````markdown", "```", "| **M4** | nested |", "| **M4** | nested |", "```", "````",
            "~~~", "| **M5** | tilde |", "| **M5** | tilde |", "~~~",
        ])
        self.assertEqual(repeated_ids(text), [(3, 5, "**M2**")])

    def test_docs_have_no_repeated_table_ids(self):
        found = []
        for path in markdown_files():
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                found += [f"{path}:{a} and :{b} ({name})" for a, b, name in repeated_ids(f.read())]
        self.assertEqual(found, [], "a table row repeated under the same bold ID (keep the current one): " +
                         "; ".join(found))


if __name__ == "__main__":
    unittest.main()
