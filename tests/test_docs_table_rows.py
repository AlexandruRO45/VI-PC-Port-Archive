"""Docs tables must not carry rows detached from a header or a second row for one subject (append-only drift)."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]

# A column after the first whose header says what state each row is in ("Status", "implemented").
STATUS_COLUMN = re.compile(r"(?i)\bstatus\b|^(not )?implemented$")


def cells(line):
    """The cells of a table row, split on unescaped pipes."""
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def tables(lines):
    """Each run of table lines as (header, [(line number, row), ...]); header is None for a run without a separator."""
    block = []
    for i, line in enumerate([*lines, ""], 1):
        if line.startswith("|"):
            block.append((i, line))
            continue
        if block:
            sep = len(block) > 1 and set(block[1][1].replace("|", "").replace(" ", "")) <= set("-:")
            yield (block[0][1], block[2:]) if sep else (None, block)
            block = []


def keyed(path, header):
    """Tables whose first cell names the row's subject, so one subject has one row.

    Every table in a README (the status, platform and index tables), every two-column table (a name and its value)
    and every status-style table (a later column says Status or (not) implemented). Other tables may group
    consecutive rows under one first cell (offsets, revisions, GPUs with several sizes) and are not checked.
    """
    if path.name == "README.md":
        return True
    columns = cells(header)
    return len(columns) == 2 or any(STATUS_COLUMN.search(c) for c in columns[1:])


def repeated_first_cells(path, text):
    """A message for each row of a keyed table whose first cell an earlier row of the same table already has."""
    problems = []
    for header, rows in tables(text.splitlines()):
        if header is None or not keyed(path, header):
            continue
        seen = {}
        for lineno, row in rows:
            first = cells(row)[0]
            if first in seen:
                problems.append(
                    f"{path.as_posix()}:{seen[first]} and {lineno}: two rows for {first!r} in one table; keep the "
                    "one that matches main and fold any still-true fact only the other has into it"
                )
            else:
                seen[first] = lineno
    return problems


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

    def test_no_repeated_first_cell_in_keyed_tables(self):
        # Two merges that each rewrote one status row can leave both versions behind (PROC-313): a reader then
        # gets two answers for one component.
        problems = []
        for path in DOCS:
            problems += repeated_first_cells(path.relative_to(ROOT), path.read_text(encoding="utf-8"))
        if problems:
            self.fail("\n" + "\n".join(problems))

    def test_repeated_first_cell_check_scope(self):
        status = "| What | Status |\n|---|---|\n| Audio | old |\n| Models | ok |\n| Audio | new |\n"
        self.assertEqual(len(repeated_first_cells(Path("README.md"), status)), 1)
        self.assertIn("README.md:3 and 5: two rows for 'Audio'", repeated_first_cells(Path("README.md"), status)[0])
        grouped = "| Offset | Size | Field |\n|---|---|---|\n| `0x08` | 24 | a |\n| `0x08` | 4 | b |\n"
        self.assertEqual(repeated_first_cells(Path("docs/x.md"), grouped), [])
        self.assertEqual(len(repeated_first_cells(Path("docs/x.md"), grouped.replace("Field", "Status"))), 1)
        self.assertEqual(len(repeated_first_cells(Path("docs/x.md"), "| Check | Count |\n|---|---|\n| a | 1 |\n| a | 2 |\n")), 1)

if __name__ == "__main__":
    unittest.main()
