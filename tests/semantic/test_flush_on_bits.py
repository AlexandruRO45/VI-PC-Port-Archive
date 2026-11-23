"""Gen.flush_f32 tests the exponent field of the bits, not |x| < FLT_MIN on the float (Gen.flush_u).

v_mad_f32 flushes its operands, the rounded product and the sum (SHADER-118). x * 1.0 + -0.0 is x for a normal x and a
zero of x's sign for a subnormal one, so the edges of the flush show in the result: FLT_MIN and -FLT_MIN stay, the
largest subnormals of each sign become +0 and -0, and +-0 and +-inf pass. The translated module flushes with integer ops
only: no FAbs and no float compare. Hand-written programs, standard library only, no game files.

    python -I tests/semantic/test_flush_on_bits.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

ONE, NEG_ZERO = 0x3F800000, 0x80000000
CASES = [  # (x, x * 1.0 + -0.0 on RDNA2 with f32 denormals flushed)
    (0x00800000, 0x00800000),   # FLT_MIN: normal, kept
    (0x80800000, 0x80800000),   # -FLT_MIN
    (0x007FFFFF, 0x00000000),   # the largest positive subnormal: +0 (+0 + -0 = +0)
    (0x807FFFFF, 0x80000000),   # the largest negative subnormal: -0
    (0x00000001, 0x00000000),
    (0x00000000, 0x00000000),
    (0x80000000, 0x80000000),
    (0x7F800000, 0x7F800000),   # +inf
    (0xFF800000, 0xFF800000),   # -inf
    (0x3F800000, 0x3F800000),
]


def program(x):
    return [f"v_mov_b32 v1, {x:#x}", f"v_mov_b32 v2, {ONE:#x}", f"v_mov_b32 v3, {NEG_ZERO:#x}",
            "v_mad_f32 v0, v1, v2, v3", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]


class FlushOnBits(unittest.TestCase):
    def test_edges(self):
        for x, want in CASES:
            with self.subTest(x=hex(x)):
                got = support.mrt0_words(support.translate_lines("ps_flush_bits", program(x)))[0]
                self.assertEqual(got, want)

    def test_nan_stays_a_nan(self):
        got = support.mrt0_words(support.translate_lines("ps_flush_bits", program(0x7FC00000)))[0]
        self.assertEqual(got & 0x7F800000, 0x7F800000)
        self.assertTrue(got & 0x007FFFFF)

    def test_integer_ops_only(self):
        spv = support.translate_lines("ps_flush_bits", program(0x007FFFFF))
        self.assertNotIn(" FAbs ", spv)
        self.assertNotIn("OpFOrdLessThan", spv)
        # Three operand flushes, the product's and the sum's: one exponent mask each.
        self.assertEqual(sum(ln.endswith(" %c7f800000") and "OpBitwiseAnd %uint" in ln for ln in spv.splitlines()), 5)


if __name__ == "__main__":
    unittest.main()
