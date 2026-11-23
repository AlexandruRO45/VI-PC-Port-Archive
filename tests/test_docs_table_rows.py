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

    def test_readme_status_table_has_one_row_per_topic(self):
        # Merges that keep both sides of the status table leave two rows for one topic with different text, which
        # the identical-row check above can't see (PROC-234: 16 rows for 6 topics on 561b65a8).
        lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
        tables, block = [], []
        for i, line in enumerate(lines, 1):
            if line.startswith("|"):
                block.append((i, line))
            elif block:
                tables.append(block)
                block = []
        if block:
            tables.append(block)
        status = [t for t in tables if any("Shaders: RDNA2" in row for _, row in t)]
        self.assertEqual(len(status), 1, "README.md: the status table (with the shader translation row) not found")
        seen = {}
        for lineno, row in status[0][2:]:   # skip the header and its separator
            topic = row.split("|")[1].strip()
            if topic in seen:
                self.fail(f"README.md:{seen[topic]} and {lineno}: two status rows for {topic!r}")
            seen[topic] = lineno


if __name__ == "__main__":
    unittest.main()
