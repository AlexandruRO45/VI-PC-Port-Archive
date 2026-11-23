"""Docs tables must not carry rows detached from a header (append-only drift)."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
STATUS_HEADER = re.compile(r"^\|\s*What\s*\|\s*Status\s*\|$")
FIRST_CELL = re.compile(r"^\|([^|]+)\|")


def status_subjects(lines):
    """First cells of the README "What | Status" table, without its header and separator."""
    subjects, inside = [], False
    for line in lines:
        if STATUS_HEADER.match(line):
            inside = True
        elif not inside:
            continue
        elif not line.startswith("|"):
            break
        elif set(line.replace("|", "").replace(" ", "")) <= set("-:"):
            continue
        else:
            match = FIRST_CELL.match(line)
            if match:
                subjects.append(match.group(1).strip())
    return subjects


def repeated(items):
    """The items that occur more than once, sorted."""
    return sorted({item for item in items if items.count(item) > 1})


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

    def test_status_table_names_each_subject_once(self):
        # Merge leftovers can leave two rows for one subject with different figures (issue #4066).
        lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
        subjects = status_subjects(lines)
        self.assertGreaterEqual(len(subjects), 10)
        self.assertEqual(repeated(subjects), [],
                         "the README project status table names a subject twice: " + ", ".join(repeated(subjects)))

    def test_a_repeated_subject_is_reported(self):
        table = ["| What | Status |", "|---|---|", "| A | yes |", "| B | no |", "| A | old |"]
        self.assertEqual(status_subjects(table), ["A", "B", "A"])
        self.assertEqual(repeated(status_subjects(table)), ["A"])


if __name__ == "__main__":
    unittest.main()
