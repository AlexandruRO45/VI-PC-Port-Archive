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

    def test_a_long_table_cell_is_unique(self):
        # A row whose first cell is a sentence has no key to tell it apart; two of them in one table is a merge
        # leftover (docs/d3d12_vulkan.md carried three copies of the BVH-builder row, with 42 and 51 passes).
        examined = 0
        for path in DOCS:
            lines = path.read_text(encoding="utf-8").splitlines()
            block = []
            for i, line in enumerate(lines + [""], 1):
                if line.startswith("|"):
                    cell = line.strip().strip("|").split("|")[0].strip()
                    if set(cell.replace(" ", "")) <= set("-:"):
                        continue
                    block.append((i, cell))
                    examined += len(cell) > 60
                    continue
                if len(block) > 1:
                    seen = {}
                    for lineno, cell in block[1:]:
                        if len(cell) > 60 and cell in seen:
                            self.fail(f"{path.relative_to(ROOT)}:{seen[cell]} and {lineno}: "
                                      "two rows with the same long first cell")
                        seen[cell] = lineno
                block = []
        self.assertGreater(examined, 50, "the check read almost no long table cells")


if __name__ == "__main__":
    unittest.main()
