"""Docs tables must not carry rows detached from a header (append-only drift)."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


class DocsTableRowsTest(unittest.TestCase):
    def test_no_orphan_table_rows(self):
        for path in DOCS:
            lines = path.read_text(encoding="utf-8").splitlines()
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
            lines = path.read_text(encoding="utf-8").splitlines()
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

    def test_row_has_as_many_cells_as_its_header(self):
        """GFM splits cells on every pipe that is not escaped, even inside a code span, so an unescaped `a | b`
        in a cell renders two cells and shifts the row's columns."""
        def cells_in(line):
            count, backslashes = 0, 0
            for ch in line:
                if ch == "\\":
                    backslashes += 1
                    continue
                if ch == "|" and not backslashes % 2:
                    count += 1
                backslashes = 0
            return count

        for path in DOCS:
            lines = path.read_text(encoding="utf-8").splitlines()
            header = None
            for i, line in enumerate(lines, 1):
                if not line.startswith("|"):
                    header = None
                    continue
                cells = cells_in(line)
                if header is None:
                    header = (cells, i)
                elif cells != header[0]:
                    self.fail(
                        f"{path.relative_to(ROOT)}:{i}: {cells - 1} cells in a table whose header at "
                        f"{header[1]} has {header[0] - 1}; escape a literal pipe as \\|"
                    )


if __name__ == "__main__":
    unittest.main()
