"""The v_interp operand-guards contract row must agree with the p10/p20 vertex-delta row.

The SHADER-177 row used to carry two contradictory statements: p10/p20 "was Unsupported until the vertex-delta
row above" (which now implements them) and, one sentence later, "is Unsupported". The translator accepts p10/p20
with per-vertex inputs (`tools/rdna2spirv.py`), so only the first statement may remain.

    python -I tests/test_docs_interp_guard_row.py

Standard library only; no game data.
"""
import os
import unittest

DESIGN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "design.md")


class InterpGuardRow(unittest.TestCase):
    def test_p10_p20_are_not_described_as_currently_unsupported(self):
        with open(DESIGN, encoding="utf-8") as handle:
            row = next(line for line in handle.read().splitlines()
                       if line.startswith("| v_interp operand guards"))
        self.assertIn("was **Unsupported** until the vertex-delta row above", row)
        self.assertNotIn("with p10/p20 is **Unsupported**, naming the select", row)


if __name__ == "__main__":
    unittest.main()
