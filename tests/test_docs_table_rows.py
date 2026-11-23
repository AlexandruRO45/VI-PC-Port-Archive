"""Docs tables must not carry rows detached from a header (append-only drift)."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def is_table_separator(line: str) -> bool:
    """True when `line` is a Markdown table separator row (e.g. `|---|---|` or `| :---: | ---: |`)."""
    stripped = line.strip()
    if "|" not in stripped:
        return False
    # Strip at most one outer pipe per side so repeated outer pipes like `||---|`, `|---||`, or `||---||`
    # leave an empty outer cell and are rejected.
    inner = stripped.removeprefix("|").removesuffix("|")
    cells = inner.split("|")
    return bool(cells) and all(
        (c := cell.strip()) and set(c) <= set("-:") and "-" in c
        for cell in cells
    )


def find_orphan_table_rows(lines: list[str]) -> list[int]:
    """Return 1-based line numbers of `|`-prefixed rows that have neither a preceding `|` row nor a separator."""
    orphans = []
    prev_nonblank = ""
    for i, line in enumerate(lines):
        if line.startswith("|"):
            if not prev_nonblank.startswith("|"):
                nxt = next((x for x in lines[i + 1:i + 3] if x.strip()), "")
                if not is_table_separator(nxt):
                    orphans.append(i + 1)
        if line.strip():
            prev_nonblank = line
    return orphans


class DocsTableRowsTest(unittest.TestCase):
    def test_no_orphan_table_rows(self):
        for path in DOCS:
            lines = path.read_text(encoding="utf-8").splitlines()
            orphans = find_orphan_table_rows(lines)
            if orphans:
                self.fail(f"{path.relative_to(ROOT)}:{orphans[0]}: table row without a table header")

    def test_orphan_row_detector_catches_eof_blank_lines_and_non_separators(self):
        self.assertEqual(find_orphan_table_rows(["| A | B |", "|---|---|", "| 1 | 2 |"]), [])
        self.assertEqual(find_orphan_table_rows(["| A | B |", "| :---: | ---: |", "| 1 | 2 |"]), [])
        # Previously `set("") <= set("-:")` was True, missing orphan rows at EOF or before >=2 blank lines.
        self.assertEqual(find_orphan_table_rows(["# Title", "", "| orphan | row |"]), [3])
        self.assertEqual(find_orphan_table_rows(["# Title", "", "| orphan | row |", "", "", "Next"]), [3])
        # A thematic break without pipes, an empty-cell row, or repeated outer pipes is not a table separator.
        self.assertEqual(find_orphan_table_rows(["| orphan | row |", "---"]), [1])
        self.assertEqual(find_orphan_table_rows(["| orphan | row |", "|  |  |"]), [1])
        self.assertEqual(find_orphan_table_rows(["| orphan |", "||---|"]), [1])
        self.assertEqual(find_orphan_table_rows(["| orphan |", "|---||"]), [1])
        self.assertEqual(find_orphan_table_rows(["intro", "|| orphan ||", "||---||"]), [2])

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


if __name__ == "__main__":
    unittest.main()
