"""Docs tables must not carry rows detached from a header (append-only drift)."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = [README, *sorted((ROOT / "docs").rglob("*.md"))]


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

    def test_no_repeated_first_cell_in_readme_tables(self):
        # A repeated first cell in the README status table means a merge kept
        # both sides of the same component row (PROC-313: four "Xbox D3D12 →
        # Vulkan" rows with contradicting statuses). Only the README is checked:
        # docs tables group rows under a repeated first cell on purpose.
        lines = README.read_text(encoding="utf-8").splitlines()

        def check(rows):
            seen = {}
            for lineno, row in rows:
                first = [c.strip() for c in row.strip().strip("|").split("|")][0]
                if not first or set(first) <= set("-:"):
                    continue  # separator row; an empty first cell is a continuation
                if first in seen:
                    self.fail(
                        f"README.md:{seen[first]} and {lineno}: repeated first cell "
                        f"{first!r} in the same table (keep only the newest variant)"
                    )
                seen[first] = lineno

        block = []
        for i, line in enumerate(lines, 1):
            if line.startswith("|"):
                block.append((i, line))
            elif block:
                check(block)
                block = []
        check(block)


if __name__ == "__main__":
    unittest.main()
