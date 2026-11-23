"""Value and hardware regression tests for v_min_f64 (IEEE-754 binary64 minimum, VOP3 modifiers):

  v_min_f64   D = min(S0, S1)   (53-bit significand precision, signed zeros, NaN suppression,
                                 neg/abs/clamp/mul:2/mul:4/div:2, infinities, and subnormals)

RDNA2 Architectural Semantics:
  1. Standard minimum: returns the smaller of S0 and S1 according to IEEE-754 binary64.
  2. Signed Zero Ordering: -0.0 < +0.0
       min(+0.0, -0.0) == -0.0, min(-0.0, +0.0) == -0.0
  3. NaN Suppression (IEEE 754-2008 minNum / GLSL.std.450 NMin):
       If one operand is NaN and the other is a valid number, the non-NaN number is returned.
       If both operands are NaN, a quiet NaN is returned.
  4. Infinities & Subnormals:
       -inf < finite numbers < +inf.
       Subnormal doubles are compared without flushing to zero.
  5. VOP3 Modifiers:
       Input modifiers: abs (|x|) clears sign bit 63; neg (-x) inverts sign bit 63.
       Output modifier: clamp clamps final result to [0.0, 1.0].
       Output scale: mul:2, mul:4, div:2 scale before clamp.

Standard library Python only; zero third-party dependencies; zero game data.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import spirv_interp  # noqa: E402
from support import mrt0_words, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
POS_ZERO = 0x0000000000000000
NEG_ZERO = 0x8000000000000000
POS_INF = 0x7FF0000000000000
NEG_INF = 0xFFF0000000000000
QNAN = 0x7FF8000000000000
SNAN = 0x7FF0000000000001
DBL_MAX = 0x7FEFFFFFFFFFFFFF
NEG_DBL_MAX = 0xFFEFFFFFFFFFFFFF
DBL_MIN = 0x0010000000000000
SUBNORMAL_MIN = 0x0000000000000001
SUBNORMAL_MAX = 0x000FFFFFFFFFFFFF


def f64_to_bits(x: float) -> int:
    """Pack a Python float into its IEEE-754 binary64 64-bit unsigned word."""
    try:
        return struct.unpack("<Q", struct.pack("<d", x))[0]
    except OverflowError:
        return POS_INF if x > 0 else NEG_INF


def bits_to_f64(bits: int) -> float:
    """Unpack a 64-bit unsigned word as an IEEE-754 binary64 Python float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def run_min_f64(
    a_bits: int,
    b_bits: int,
    *,
    setup: list[str] | None = None,
    operands: str = "v[0:1], v[2:3]",
) -> tuple[int, str]:
    """Translate and run `v_min_f64 v[4:5], <operands>` via translator; return (u64_bits, spv)."""
    if setup is None:
        setup = [
            f"v_mov_b32 v0, {a_bits & MASK32:#010x}",
            f"v_mov_b32 v1, {(a_bits >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {b_bits & MASK32:#010x}",
            f"v_mov_b32 v3, {(b_bits >> 32) & MASK32:#010x}",
        ]
    lines = list(setup) + [
        f"v_min_f64 v[4:5], {operands}",
        "exp mrt0 v4, v5, v4, v5 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_min_f64", lines)
    words = mrt0_words(spv)
    got = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
    return got, spv


class MinF64BasicAndPrecisionTests(unittest.TestCase):
    def test_basic_min(self):
        cases = [
            (2.0, 1.0, 1.0),
            (1.0, 2.0, 1.0),
            (5.5, 3.25, 3.25),
            (-10.0, 1.0, -10.0),
            (1.0, -10.0, -10.0),
            (-5.0, -2.0, -5.0),
            (-2.0, -5.0, -5.0),
            (100.0, 100.0, 100.0),
            (-42.0, -42.0, -42.0),
        ]
        for a, b, exp in cases:
            with self.subTest(a=a, b=b):
                got, _ = run_min_f64(f64_to_bits(a), f64_to_bits(b))
                self.assertEqual(got, f64_to_bits(exp))

    def test_precision_beyond_f32(self):
        cases = [
            (float(2**40 + 1), float(2**40 + 2), float(2**40 + 1)),
            (float(2**52 + 1), float(2**52 + 3), float(2**52 + 1)),
            (1.0 + 2.0**-52, 1.0, 1.0),
            (1.0 + 2.0**-52, 1.0 + 2.0**-51, 1.0 + 2.0**-52),
            (float(2**53), float(2**53 + 2), float(2**53)),
        ]
        for a, b, exp in cases:
            with self.subTest(a=a, b=b):
                got, _ = run_min_f64(f64_to_bits(a), f64_to_bits(b))
                self.assertEqual(got, f64_to_bits(exp))


class MinF64SignedZeroTests(unittest.TestCase):
    def test_signed_zeros(self):
        # RDNA2 strict signed zero ordering: -0.0 < +0.0
        self.assertEqual(run_min_f64(POS_ZERO, NEG_ZERO)[0], NEG_ZERO)
        self.assertEqual(run_min_f64(NEG_ZERO, POS_ZERO)[0], NEG_ZERO)
        self.assertEqual(run_min_f64(POS_ZERO, POS_ZERO)[0], POS_ZERO)
        self.assertEqual(run_min_f64(NEG_ZERO, NEG_ZERO)[0], NEG_ZERO)


class MinF64NanSuppressionTests(unittest.TestCase):
    def test_nan_suppression(self):
        # NaN operand returns the other non-NaN operand (IEEE 754-2008 / GLSL.std.450 NMin)
        val = f64_to_bits(42.5)
        self.assertEqual(run_min_f64(QNAN, val)[0], val)
        self.assertEqual(run_min_f64(val, QNAN)[0], val)
        self.assertEqual(run_min_f64(SNAN, val)[0], val)
        self.assertEqual(run_min_f64(val, SNAN)[0], val)

        # NaN with infinities
        self.assertEqual(run_min_f64(QNAN, NEG_INF)[0], NEG_INF)
        self.assertEqual(run_min_f64(POS_INF, QNAN)[0], POS_INF)
        self.assertEqual(run_min_f64(NEG_INF, SNAN)[0], NEG_INF)

        # Both NaN returns a NaN
        got, _ = run_min_f64(QNAN, SNAN)
        self.assertTrue(math.isnan(bits_to_f64(got)))


class MinF64InfinitiesAndSubnormalsTests(unittest.TestCase):
    def test_infinities(self):
        val = f64_to_bits(1.0e300)
        neg_val = f64_to_bits(-1.0e300)
        self.assertEqual(run_min_f64(POS_INF, val)[0], val)
        self.assertEqual(run_min_f64(val, POS_INF)[0], val)
        self.assertEqual(run_min_f64(NEG_INF, neg_val)[0], NEG_INF)
        self.assertEqual(run_min_f64(neg_val, NEG_INF)[0], NEG_INF)
        self.assertEqual(run_min_f64(NEG_INF, POS_INF)[0], NEG_INF)
        self.assertEqual(run_min_f64(POS_INF, NEG_INF)[0], NEG_INF)
        self.assertEqual(run_min_f64(DBL_MAX, POS_INF)[0], DBL_MAX)
        self.assertEqual(run_min_f64(NEG_DBL_MAX, NEG_INF)[0], NEG_INF)

    def test_subnormals(self):
        # IEEE mode preserves subnormal double precision
        sub1 = SUBNORMAL_MIN
        sub2 = SUBNORMAL_MIN + 1
        self.assertEqual(run_min_f64(sub1, sub2)[0], sub1)
        self.assertEqual(run_min_f64(sub2, sub1)[0], sub1)
        self.assertEqual(run_min_f64(SUBNORMAL_MAX, DBL_MIN)[0], SUBNORMAL_MAX)
        self.assertEqual(run_min_f64(sub1, POS_ZERO)[0], POS_ZERO)
        self.assertEqual(run_min_f64(NEG_ZERO, sub1)[0], NEG_ZERO)


class MinF64VOP3ModifiersTests(unittest.TestCase):
    def test_abs_and_neg_modifiers(self):
        a = f64_to_bits(-10.0)
        b = f64_to_bits(5.0)
        # min(|-10.0|, 5.0) = min(10.0, 5.0) = 5.0
        got, _ = run_min_f64(a, b, operands="|v[0:1]|, v[2:3]")
        self.assertEqual(got, f64_to_bits(5.0))

        # min(-10.0, -|5.0|) = min(-10.0, -5.0) = -10.0
        got, _ = run_min_f64(a, b, operands="v[0:1], -|v[2:3]|")
        self.assertEqual(got, f64_to_bits(-10.0))

        # min(-(-10.0), 5.0) = min(10.0, 5.0) = 5.0
        got, _ = run_min_f64(a, b, operands="-v[0:1], v[2:3]")
        self.assertEqual(got, f64_to_bits(5.0))

    def test_clamp_and_omod_modifiers(self):
        cases = [
            (0.8, 0.4, "v[0:1], v[2:3] mul:2", 0.8),
            (0.8, 0.4, "v[0:1], v[2:3] mul:4", 1.6),
            (0.8, 0.4, "v[0:1], v[2:3] div:2", 0.2),
            (2.5, 3.0, "v[0:1], v[2:3] clamp", 1.0),
            (-2.5, 0.5, "v[0:1], v[2:3] clamp", 0.0),
            (0.8, 0.4, "v[0:1], v[2:3] mul:2 clamp", 0.8),
            (0.8, 0.4, "v[0:1], v[2:3] mul:4 clamp", 1.0),
        ]
        for a_val, b_val, ops, exp in cases:
            with self.subTest(ops=ops):
                got, _ = run_min_f64(f64_to_bits(a_val), f64_to_bits(b_val), operands=ops)
                self.assertEqual(got, f64_to_bits(exp))


class MinF64OperandEncodingTests(unittest.TestCase):
    def test_sgpr_pair_and_inline_constants(self):
        a = f64_to_bits(10.0)
        b = f64_to_bits(25.0)

        # SGPR-pair src0 + VGPR-pair src1
        setup_sgpr = [
            f"s_mov_b32 s0, {a & MASK32:#010x}",
            f"s_mov_b32 s1, {(a >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {b & MASK32:#010x}",
            f"v_mov_b32 v3, {(b >> 32) & MASK32:#010x}",
        ]
        got, _ = run_min_f64(0, 0, setup=setup_sgpr, operands="s[0:1], v[2:3]")
        self.assertEqual(got, f64_to_bits(10.0))

        # Inline float constants: 0.5
        got, _ = run_min_f64(a, 0, operands="v[0:1], 0.5")
        self.assertEqual(got, f64_to_bits(0.5))

        # Inline float constant: -1.0
        got, _ = run_min_f64(0, b, operands="-1.0, v[2:3]")
        self.assertEqual(got, f64_to_bits(-1.0))


class MinF64SpirvStructureTests(unittest.TestCase):
    def test_spirv_structure(self):
        _, spv = run_min_f64(f64_to_bits(1.0), f64_to_bits(2.0))
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpExtInst %double %glsl NMin", spv)
        self.assertIn("%out_mrt0 = OpVariable", spv)


if __name__ == "__main__":
    unittest.main()
