"""Docs tables must not carry rows detached from a header (append-only drift)."""

import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


FENCE_LINE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def table_lines(text):
    """Blank fenced code without moving line numbers or joining adjacent tables."""
    lines = text.splitlines()
    fence = None
    for i, line in enumerate(lines):
        match = FENCE_LINE.match(line)
        if fence is None:
            if match and (match[1][0] == "~" or "`" not in match[2]):
                fence = match[1]
        elif (match and match[1][0] == fence[0] and len(match[1]) >= len(fence)
              and not match[2].strip(" \t")):
            fence = None
        else:
            lines[i] = ""
    return lines


class DocsTableRowsTest(unittest.TestCase):
    def test_pipe_rows_inside_code_fences_are_not_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page = root / "example.md"
            with mock.patch.dict(globals(), ROOT=root, DOCS=[page]):
                for opening, short, other in (("````", "```", "~~~"), ("~~~~", "~~~", "```")):
                    page.write_text(
                        f"   {opening}markdown\n{short}\n{other}\n| example |\n| example |\n"
                        f"{opening + opening[0]} \t\n\n| Header |\n| --- |\n| Actual row |\n",
                        encoding="utf-8", newline="\n",
                    )
                    for check in (self.test_no_orphan_table_rows, self.test_no_duplicate_rows_in_one_table):
                        with self.subTest(fence=opening, check=check.__name__):
                            check()

    def test_real_table_errors_after_a_fence_keep_original_line_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page = root / "example.md"
            prefix = "~~~text\nordinary code\n~~~\n\n"
            with mock.patch.dict(globals(), ROOT=root, DOCS=[page]):
                page.write_text(prefix + "| Orphan |\nProse\n", encoding="utf-8", newline="\n")
                with self.assertRaisesRegex(AssertionError, r"example\.md:5: table row without a table header"):
                    self.test_no_orphan_table_rows()
                page.write_text(prefix + "| Header |\n| --- |\n| Value |\n| Value |\n",
                                encoding="utf-8", newline="\n")
                with self.assertRaisesRegex(AssertionError, r"example\.md:7 and 8: duplicate row"):
                    self.test_no_duplicate_rows_in_one_table()

    def test_no_orphan_table_rows(self):
        for path in DOCS:
            lines = table_lines(path.read_text(encoding="utf-8"))
            for i, line in enumerate(lines):
                if not line.startswith("|"):
                    continue
                prev = next((x for x in reversed(lines[:i]) if x.strip()), "")
                if prev.startswith("|"):
                    continue
                nxt = next((x for x in lines[i + 1:i + 3] if x.strip()), "")
                if set(nxt.replace("|", "").replace(" ", "")) <= set("-:"):
                    continue  # this is a table header followed by its separator
                self.fail(f"{path.relative_to(ROOT)}:{i + 1}: table row without a table header")

    def test_no_duplicate_rows_in_one_table(self):
        for path in DOCS:
            lines = table_lines(path.read_text(encoding="utf-8"))
            block = []

            def check(rows):
                seen = {}
                for lineno, row in rows:
                    if row in seen:
                        self.fail(
                            f"{path.relative_to(ROOT)}:{seen[row]} and {lineno}: "
                            "duplicate row in the same table"
                        )
                    seen[row] = lineno

            for i, line in enumerate(lines, 1):
                if line.startswith("|"):
                    block.append((i, line))
                elif block:
                    check(block)
                    block = []
            check(block)


if __name__ == "__main__":
    unittest.main()
