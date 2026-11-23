"""tools/vcmp16_opsel_probe.py measures every 16-bit compare it names, with a row that tells op_sel from the low halves.

    python -I tests/test_vcmp16_opsel_probe_rows.py

The probe's docstring says it measures `v_cmp_{lt,eq,gt}_{f16,u16,i16}` and that the rows' values make the two
hypotheses differ "on most of them". `i16 eq` had no opcode and no row, and the four `u16 eq` rows compared halves
that all differ (5 against 7), so every one of their encodings answered 0 with op_sel honoured and ignored alike: the
probe printed "(does not tell them apart)" for all of them, and a hardware run of those rows says nothing about the
rule. This uses the tool's own `answer()` and needs no GPU, spirv-as or game files.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import vcmp16_opsel_probe as probe  # noqa: E402

NAMED = [(op, kind) for kind in ("f16", "u16", "i16") for op in ("lt", "eq", "gt")]


class ProbeRowsTest(unittest.TestCase):
    def test_every_named_compare_has_an_opcode(self):
        for op, kind in NAMED:
            with self.subTest(compare=f"v_cmp_{op}_{kind}"):
                w0, w1 = probe.words(op, kind, (0, 0))
                self.assertEqual((w0 >> 16) & 0xFF, probe.OPCODES[(op, kind)])
                self.assertEqual((w1 >> 29) & 0x7, 0)   # no neg/abs unless the row asks for it

    def test_every_named_compare_has_a_row(self):
        have = {(row[1], row[2]) for row in probe.rows()}
        for op, kind in NAMED:
            with self.subTest(compare=f"v_cmp_{op}_{kind}"):
                self.assertIn((op, kind), have)

    def test_every_named_compare_has_a_row_that_tells_the_hypotheses_apart(self):
        telling = {(row[1], row[2]) for row in probe.rows()
                   if probe.answer(row, True) != probe.answer(row, False)}
        for op, kind in NAMED:
            with self.subTest(compare=f"v_cmp_{op}_{kind}"):
                self.assertIn((op, kind), telling)

    def test_row_names_are_unique(self):
        names = [row[0] for row in probe.rows()]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
