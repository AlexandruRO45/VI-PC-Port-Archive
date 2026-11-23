"""Value tests for v_mul_f64 (IEEE-754 binary64 multiplication with VOP3 modifiers):

  v_mul_f64   D = S0 * S1   (53-bit significand precision, signed zeros, inf/NaN,
                             neg/abs/clamp/mul:2/mul:4/div:2)

Each case translates a hand-written pixel-shader listing with tools/rdna2spirv.py
and executes the resulting SPIR-V via tests/spirv_interp.py (which supports OpTypeFloat 64,
whereas spirv_mini_interp.py is 32-bit float only). Standard library only.
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
from support import translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
POS_ZERO = 0x0000000000000000
NEG_ZERO = 0x8000000000000000
POS_INF = 0x7FF0000000000000
NEG_INF = 0xFFF0000000000000
QNAN = 0x7FF8000000000000
DBL_MAX = 0x7FEFFFFFFFFFFFFF
NEG_DBL_MAX = 0xFFEFFFFFFFFFFFFF
DBL_MIN = 0x0010000000000000


def f64_to_bits(x: float) -> int:
    """Pack a Python float into its IEEE-754 binary64 64-bit unsigned word."""
    try:
        return struct.unpack("<Q", struct.pack("<d", x))[0]
    except OverflowError:
        return POS_INF if x > 0 else NEG_INF


def bits_to_f64(bits: int) -> float:
    """Unpack a 64-bit unsigned word as an IEEE-754 binary64 Python float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def run_mul_f64(
    a_bits: int,
    b_bits: int,
    *,
    setup: list[str] | None = None,
    operands: str = "v[0:1], v[2:3]",
    dst: str = "v[4:5]",
):
    """Translate and run `v_mul_f64 <dst>, <operands>` via spirv_interp; return (u64_bits, spv)."""
    if setup is None:
        setup = [
            f"v_mov_b32 v0, {a_bits & MASK32:#010x}",
            f"v_mov_b32 v1, {(a_bits >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {b_bits & MASK32:#010x}",
            f"v_mov_b32 v3, {(b_bits >> 32) & MASK32:#010x}",
        ]
    lines = list(setup)
    d_lo = int(dst.replace("v[", "").replace("]", "").split(":")[0])
    lines += [
        f"v_mul_f64 {dst}, {operands}",
        f"exp mrt0 v{d_lo}, v{d_lo+1}, v{d_lo}, v{d_lo+1} done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_mul_f64", lines)
    res = spirv_interp.run(spv)
    got = (res.vgpr[d_lo] & MASK32) | ((res.vgpr[d_lo + 1] & MASK32) << 32)
    return got, spv


class MulF64PrecisionTests(unittest.TestCase):
    def test_basic_products(self):
        cases = [
            (2.0, 3.0, 6.0),
            (1.5, 2.5, 3.75),
            (0.125, 0.25, 0.03125),
            (-4.0, 0.5, -2.0),
            (-3.5, -2.0, 7.0),
        ]
        for a, b, exp in cases:
            with self.subTest(a=a, b=b):
                got, _ = run_mul_f64(f64_to_bits(a), f64_to_bits(b))
                self.assertEqual(got, f64_to_bits(exp))

    def test_full_53bit_precision(self):
        cases = [
            (float(2**52 + 1), 1.0, float(2**52 + 1)),
            (float(2**26 + 1), float(2**26 + 1), float((2**26 + 1) * (2**26 + 1))),
            (1.0 + 2.0**-52, 1.0, 1.0 + 2.0**-52),
            (1.0 + 2.0**-51, 0.5, 0.5 + 2.0**-52),
        ]
        for a, b, exp in cases:
            with self.subTest(a=a, b=b):
                got, _ = run_mul_f64(f64_to_bits(a), f64_to_bits(b))
                self.assertEqual(got, f64_to_bits(exp))

    def test_power_of_two_scaling(self):
        cases = [
            (float(2**20), float(2**30), float(2**50)),
            (2.0**-20, 2.0**-30, 2.0**-50),
            (float(2**60), 2.0**-60, 1.0),
        ]
        for a, b, exp in cases:
            with self.subTest(a=a, b=b):
                got, _ = run_mul_f64(f64_to_bits(a), f64_to_bits(b))
                self.assertEqual(got, f64_to_bits(exp))


class MulF64SpecialValuesTests(unittest.TestCase):
    def test_signed_zeros(self):
        cases = [
            (POS_ZERO, POS_ZERO, POS_ZERO),
            (POS_ZERO, NEG_ZERO, NEG_ZERO),
            (NEG_ZERO, POS_ZERO, NEG_ZERO),
            (NEG_ZERO, NEG_ZERO, POS_ZERO),
            (POS_ZERO, f64_to_bits(5.0), POS_ZERO),
            (NEG_ZERO, f64_to_bits(5.0), NEG_ZERO),
            (POS_ZERO, f64_to_bits(-5.0), NEG_ZERO),
            (NEG_ZERO, f64_to_bits(-5.0), POS_ZERO),
        ]
        for a, b, exp in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                got, _ = run_mul_f64(a, b)
                self.assertEqual(got, exp)

    def test_infinities(self):
        two = f64_to_bits(2.0)
        neg_two = f64_to_bits(-2.0)
        cases = [
            (POS_INF, two, POS_INF),
            (POS_INF, neg_two, NEG_INF),
            (NEG_INF, two, NEG_INF),
            (NEG_INF, neg_two, POS_INF),
            (POS_INF, POS_INF, POS_INF),
            (POS_INF, NEG_INF, NEG_INF),
            (NEG_INF, NEG_INF, POS_INF),
        ]
        for a, b, exp in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                got, _ = run_mul_f64(a, b)
                self.assertEqual(got, exp)

    def test_zero_times_infinity_produces_nan(self):
        for z in (POS_ZERO, NEG_ZERO):
            for inf in (POS_INF, NEG_INF):
                with self.subTest(z=hex(z), inf=hex(inf)):
                    got, _ = run_mul_f64(z, inf)
                    self.assertTrue(math.isnan(bits_to_f64(got)), f"expected NaN, got {got:#018x}")

    def test_nan_propagation(self):
        one = f64_to_bits(1.0)
        zero = POS_ZERO
        inf = POS_INF
        for operand in (one, zero, inf, QNAN):
            with self.subTest(operand=hex(operand)):
                got, _ = run_mul_f64(QNAN, operand)
                self.assertTrue(math.isnan(bits_to_f64(got)), f"expected NaN, got {got:#018x}")


class MulF64ModifiersTests(unittest.TestCase):
    def test_source_modifiers(self):
        a = f64_to_bits(2.0)
        b = f64_to_bits(3.0)
        cases = [
            ("-v[0:1], v[2:3]", f64_to_bits(-6.0)),
            ("v[0:1], -v[2:3]", f64_to_bits(-6.0)),
            ("-v[0:1], -v[2:3]", f64_to_bits(6.0)),
            ("|v[0:1]|, |v[2:3]|", f64_to_bits(6.0)),
            ("-|v[0:1]|, v[2:3]", f64_to_bits(-6.0)),
        ]
        for ops, exp in cases:
            with self.subTest(ops=ops):
                got, _ = run_mul_f64(a, b, operands=ops)
                self.assertEqual(got, exp)

    def test_output_modifiers(self):
        cases = [
            (0.75, 0.5, "v[0:1], v[2:3] mul:2", 0.75),
            (0.75, 0.5, "v[0:1], v[2:3] mul:4", 1.5),
            (0.75, 0.5, "v[0:1], v[2:3] div:2", 0.1875),
            (0.75, 0.5, "v[0:1], v[2:3] clamp", 0.375),
            (1.5, 2.0, "v[0:1], v[2:3] clamp", 1.0),
            (-1.5, 2.0, "v[0:1], v[2:3] clamp", 0.0),
            (0.75, 1.0, "v[0:1], v[2:3] mul:2 clamp", 1.0),
            (1.5, 1.0, "v[0:1], v[2:3] div:2 clamp", 0.75),
        ]
        for a_val, b_val, ops, exp in cases:
            with self.subTest(ops=ops):
                got, _ = run_mul_f64(f64_to_bits(a_val), f64_to_bits(b_val), operands=ops)
                self.assertEqual(got, f64_to_bits(exp))


class MulF64OperandFormsTests(unittest.TestCase):
    def test_sgpr_pair_and_inline_constants(self):
        a = f64_to_bits(3.0)
        setup_s0 = [
            f"s_mov_b32 s0, {a & MASK32:#010x}",
            f"s_mov_b32 s1, {(a >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {f64_to_bits(4.0) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_bits(4.0) >> 32) & MASK32:#010x}",
        ]
        got, _ = run_mul_f64(0, 0, setup=setup_s0, operands="s[0:1], v[2:3]")
        self.assertEqual(got, f64_to_bits(12.0))

        got, _ = run_mul_f64(a, 0, operands="v[0:1], 0.5")
        self.assertEqual(got, f64_to_bits(1.5))

        got, _ = run_mul_f64(0, a, operands="-1.0, v[2:3]")
        self.assertEqual(got, f64_to_bits(-3.0))

    def test_register_aliasing_squaring(self):
        a = f64_to_bits(4.0)
        setup = [
            f"v_mov_b32 v0, {a & MASK32:#010x}",
            f"v_mov_b32 v1, {(a >> 32) & MASK32:#010x}",
        ]
        got, _ = run_mul_f64(0, 0, setup=setup, operands="v[0:1], v[0:1]", dst="v[0:1]")
        self.assertEqual(got, f64_to_bits(16.0))


class MulF64BoundariesTests(unittest.TestCase):
    def test_overflow_to_infinity(self):
        got, _ = run_mul_f64(DBL_MAX, f64_to_bits(2.0))
        self.assertEqual(got, POS_INF)

        got, _ = run_mul_f64(NEG_DBL_MAX, f64_to_bits(2.0))
        self.assertEqual(got, NEG_INF)

    def test_underflow_behavior(self):
        got, _ = run_mul_f64(f64_to_bits(2.0**-1074), f64_to_bits(0.5))
        self.assertEqual(got, POS_ZERO)


if __name__ == "__main__":
    unittest.main()
