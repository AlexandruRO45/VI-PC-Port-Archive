"""A form with a contract row of its own in docs/design.md must not be called pending elsewhere.

Usage:
    python -I tests/test_docs_design_pending.py

A row whose first cell is one backticked form (``| `image_sample_c_l` | ...``) owns that form's rule. Another row that
still says "`image_sample_c_l` is #2685" describes the same form as unstarted work, so a reader gets the merged rule and
the pending note from one table (the row 291/289 pair this test was written for).
"""

import re
import unittest
from pathlib import Path

DESIGN = Path(__file__).resolve().parents[1] / "docs" / "design.md"

OWN_ROW = re.compile(r"`([^`]+)`")
PENDING = re.compile(r"`([A-Za-z_][A-Za-z0-9_.]*)` is #(\d+)")


def rows_with_own_form(lines):
    """{form: line number} for every table row whose first cell is exactly one backticked form."""
    owned = {}
    for number, line in enumerate(lines, 1):
        if not line.startswith("|"):
            continue
        match = OWN_ROW.fullmatch(line.split("|")[1].strip())
        if match:
            owned.setdefault(match.group(1), number)
    return owned


class DesignPendingRowTest(unittest.TestCase):
    def test_no_pending_note_for_a_form_the_table_documents(self):
        lines = DESIGN.read_text(encoding="utf-8").splitlines()
        owned = rows_with_own_form(lines)
        self.assertGreater(len(owned), 5, "no single-form contract rows parsed: the row format changed")
        for number, line in enumerate(lines, 1):
            for match in PENDING.finditer(line):
                form, pr = match.group(1), match.group(2)
                self.assertNotIn(
                    form, owned,
                    f"docs/design.md:{number} calls `{form}` pending as #{pr}, "
                    f"but line {owned[form]} is its contract row",
                )


if __name__ == "__main__":
    unittest.main()
