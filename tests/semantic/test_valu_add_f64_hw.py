from __future__ import annotations

"""Semantic translation tests for RDNA2 double-precision floating-point addition (v_add_f64).

Translates synthetic pixel-shader listings using tools/rdna2spirv.py via tests/semantic/support.py,
executing the resulting SPIR-V through tests/spirv_interp.py (which supports OpTypeFloat 64) and
comparing against a bit-exact IEEE-754 binary64 reference model.
"""

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (HERE, os.path.join(ROOT, "tools"), os.path.join(ROOT, "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

import spirv_interp
from support import translate_lines

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
POS_ZERO = 0x0000000000000000
NEG_ZERO = 0x8000000000000000
POS_INF = 0x7FF0000000000000
NEG_INF = 0xFFF0000000000000
DBL_MAX = 0x7FEFFFFFFFFFFFFF
NEG_DBL_MAX = 0xFFEFFFFFFFFFFFFF
DBL_MIN_NORMAL = 0x0010000000000000
DBL_MIN_SUBNORMAL = 0x0000000000000001
QNAN = 0x7FF8000000000000


def f64_to_u64(val: float) -> int:
    """Pack a Python float into an unsigned 64-bit IEEE-754 representation."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF if val > 0.0 else NEG_INF


def u64_to_f64(bits: int) -> float:
    """Unpack an unsigned 64-bit word into a Python IEEE-754 binary64 float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def ref_add_f64(a_bits: int, b_bits: int) -> int:
    """Bit-exact reference mathematical oracle for IEEE-754 binary64 addition."""
    a = u64_to_f64(a_bits)
    b = u64_to_f64(b_bits)
    res = a + b
    return f64_to_u64(res)


def run_translation_add_f64(
    a_bits: int,
    b_bits: int,
    *,
    setup: list[str] | None = None,
    operands: str = "v[0:1], v[2:3]",
    dst_vgpr: int = 4,
    mode: int | None = None,
) -> tuple[int, str]:
    """Translate and execute a synthetic shader with v_add_f64, returning (u64_result, spv_text)."""
    if setup is None:
        setup = [
            f"v_mov_b32 v0, {a_bits & MASK32:#010x}",
            f"v_mov_b32 v1, {(a_bits >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {b_bits & MASK32:#010x}",
            f"v_mov_b32 v3, {(b_bits >> 32) & MASK32:#010x}",
        ]
    lines = list(setup)
    if mode is not None:
        lines.append(f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), {mode}")
    d_lo, d_hi = dst_vgpr, dst_vgpr + 1
    lines += [
        f"v_add_f64 v[{d_lo}:{d_hi}], {operands}",
        f"exp mrt0 v{d_lo}, v{d_hi}, v{d_lo}, v{d_hi} done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_add_f64_hw", lines)
    res = spirv_interp.run(spv)
    got = (res.vgpr[d_lo] & MASK32) | ((res.vgpr[d_hi] & MASK32) << 32)
    return got, spv


class TestVAddF64ReferenceOracleAndBasics(unittest.TestCase):
    """Verifies that translation of v_add_f64 matches reference oracle and emits valid SPIR-V."""

    def test_spirv_capabilities_and_opcode(self):
        got, spv = run_translation_add_f64(f64_to_u64(1.0), f64_to_u64(2.0))
        self.assertEqual(got, f64_to_u64(3.0))
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpCapability SignedZeroInfNanPreserve", spv)
        self.assertIn("OpCapability DenormPreserve", spv)
        self.assertIn("OpFAdd %double", spv)

    def test_basic_arithmetic_cases(self):
        test_pairs = [
            (0.0, 0.0),
            (1.0, 2.0),
            (10.5, -4.25),
            (1234567.875, 9876543.125),
            (-500.0, -250.0),
            (0.125, 0.0625),
            (1.0e15, 2.0e15),
            (1.0e-15, 2.0e-15),
        ]
        for a, b in test_pairs:
            with self.subTest(a=a, b=b):
                a_bits = f64_to_u64(a)
                b_bits = f64_to_u64(b)
                expected = ref_add_f64(a_bits, b_bits)
                got, _ = run_translation_add_f64(a_bits, b_bits)
                self.assertEqual(got, expected)


class TestVAddF64PrecisionBeyondF32(unittest.TestCase):
    """Verifies full 53-bit significand precision that would truncate in 32-bit float."""

    def test_53bit_mantissa_precision(self):
        cases = [
            (float(2**24), 1.0, float(2**24 + 1)),
            (float(2**40), 1.0, float(2**40 + 1)),
            (float(2**52), 1.0, float(2**52 + 1)),
            (1.0, 2.0**-52, 1.0 + 2.0**-52),
            # 2^53 + 1.0 rounds to 2^53 in ties-to-even
            (float(2**53), 1.0, float(2**53)),
        ]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                got, _ = run_translation_add_f64(f64_to_u64(a), f64_to_u64(b))
                self.assertEqual(got, f64_to_u64(expected))

    def test_catastrophic_cancellation(self):
        cases = [
            (float(2**50) + 1.0, -float(2**50), 1.0),
            (1.0 + 2.0**-52, -1.0, 2.0**-52),
            (987654.3210123456, -987654.3210123456, 0.0),
        ]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                got, _ = run_translation_add_f64(f64_to_u64(a), f64_to_u64(b))
                self.assertEqual(got, f64_to_u64(expected))


class TestVAddF64SpecialValuesAndIEEE754(unittest.TestCase):
    """Verifies signed zeros, infinities, subnormals, and NaN handling."""

    def test_signed_zeros(self):
        cases = [
            (POS_ZERO, POS_ZERO, POS_ZERO),
            (POS_ZERO, NEG_ZERO, POS_ZERO),
            (NEG_ZERO, POS_ZERO, POS_ZERO),
            (NEG_ZERO, NEG_ZERO, NEG_ZERO),
        ]
        for a_bits, b_bits, expected in cases:
            with self.subTest(a=hex(a_bits), b=hex(b_bits)):
                got, _ = run_translation_add_f64(a_bits, b_bits)
                self.assertEqual(got, expected)

    def test_infinities(self):
        one = f64_to_u64(1.0)
        cases = [
            (POS_INF, one, POS_INF),
            (NEG_INF, one, NEG_INF),
            (POS_INF, POS_INF, POS_INF),
            (NEG_INF, NEG_INF, NEG_INF),
        ]
        for a_bits, b_bits, expected in cases:
            with self.subTest(a=hex(a_bits), b=hex(b_bits)):
                got, _ = run_translation_add_f64(a_bits, b_bits)
                self.assertEqual(got, expected)

    def test_nan_generation_and_propagation(self):
        # +inf + -inf produces NaN
        got, _ = run_translation_add_f64(POS_INF, NEG_INF)
        self.assertTrue(math.isnan(u64_to_f64(got)), f"expected NaN, got {got:#018x}")

        # NaN + x = NaN
        one = f64_to_u64(1.0)
        got, _ = run_translation_add_f64(QNAN, one)
        self.assertTrue(math.isnan(u64_to_f64(got)), f"expected NaN, got {got:#018x}")

    def test_subnormal_numbers(self):
        # Adding smallest subnormals
        got, _ = run_translation_add_f64(DBL_MIN_SUBNORMAL, DBL_MIN_SUBNORMAL)
        self.assertEqual(got, 0x0000000000000002)

        # Subnormal adding up to normal boundary
        subnormal_max = 0x000FFFFFFFFFFFFF
        got, _ = run_translation_add_f64(subnormal_max, DBL_MIN_SUBNORMAL)
        self.assertEqual(got, DBL_MIN_NORMAL)


class TestVAddF64Modifiers(unittest.TestCase):
    """Verifies VOP3 modifiers: source negation, absolute value, output clamp, and multipliers."""

    def test_negation_and_absolute_value(self):
        a = f64_to_u64(2.5)
        b = f64_to_u64(1.25)

        # Source negation on src0: -2.5 + 1.25 = -1.25
        got, spv = run_translation_add_f64(a, b, operands="-v[0:1], v[2:3]")
        self.assertEqual(got, f64_to_u64(-1.25))
        self.assertIn("OpFNegate %double", spv)

        # Source negation on src1: 2.5 + (-1.25) = 1.25
        got, _ = run_translation_add_f64(a, b, operands="v[0:1], -v[2:3]")
        self.assertEqual(got, f64_to_u64(1.25))

        # Absolute value: |-2.5| + 1.25 = 3.75
        neg_a = f64_to_u64(-2.5)
        got, spv = run_translation_add_f64(neg_a, b, operands="|v[0:1]|, v[2:3]")
        self.assertEqual(got, f64_to_u64(3.75))
        self.assertIn("FAbs", spv)

        # Negate absolute value: -|-2.5| + 1.25 = -1.25
        got, _ = run_translation_add_f64(neg_a, b, operands="-|v[0:1]|, v[2:3]")
        self.assertEqual(got, f64_to_u64(-1.25))

    def test_output_modifiers(self):
        # mul:2, mul:4, div:2
        a = f64_to_u64(1.5)
        b = f64_u64 = f64_to_u64(0.5)

        # (1.5 + 0.5) * 2 = 4.0
        got, _ = run_translation_add_f64(a, b, operands="v[0:1], v[2:3] mul:2")
        self.assertEqual(got, f64_to_u64(4.0))

        # (1.5 + 0.5) * 4 = 8.0
        got, _ = run_translation_add_f64(a, b, operands="v[0:1], v[2:3] mul:4")
        self.assertEqual(got, f64_to_u64(8.0))

        # (1.5 + 0.5) / 2 = 1.0
        got, _ = run_translation_add_f64(a, b, operands="v[0:1], v[2:3] div:2")
        self.assertEqual(got, f64_to_u64(1.0))

        # clamp [0.0, 1.0]
        # (1.5 + 0.5) clamp = 1.0
        got, _ = run_translation_add_f64(a, b, operands="v[0:1], v[2:3] clamp")
        self.assertEqual(got, f64_to_u64(1.0))

        # (-1.5 + 0.5) clamp = 0.0
        neg_a = f64_to_u64(-1.5)
        got, _ = run_translation_add_f64(neg_a, b, operands="v[0:1], v[2:3] clamp")
        self.assertEqual(got, f64_to_u64(0.0))


class TestVAddF64RegisterAliasingAndOperands(unittest.TestCase):
    """Verifies in-place register updates, aliasing, and alternate operand formats."""

    def test_inplace_accumulation(self):
        # v[0:1] = v[0:1] + v[2:3]
        a = f64_to_u64(10.0)
        b = f64_to_u64(5.0)
        setup = [
            f"v_mov_b32 v0, {a & MASK32:#010x}",
            f"v_mov_b32 v1, {(a >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {b & MASK32:#010x}",
            f"v_mov_b32 v3, {(b >> 32) & MASK32:#010x}",
        ]
        got, _ = run_translation_add_f64(
            a, b, setup=setup, operands="v[0:1], v[2:3]", dst_vgpr=0
        )
        self.assertEqual(got, f64_to_u64(15.0))

    def test_self_addition(self):
        # v[0:1] = v[0:1] + v[0:1] (doubling in place)
        a = f64_to_u64(7.5)
        setup = [
            f"v_mov_b32 v0, {a & MASK32:#010x}",
            f"v_mov_b32 v1, {(a >> 32) & MASK32:#010x}",
        ]
        got, _ = run_translation_add_f64(
            a, a, setup=setup, operands="v[0:1], v[0:1]", dst_vgpr=0
        )
        self.assertEqual(got, f64_to_u64(15.0))

    def test_sgpr_pair_and_inline_constants(self):
        a = f64_to_u64(100.0)
        b = f64_to_u64(25.0)

        # SGPR src0 + VGPR src1
        setup = [
            f"s_mov_b32 s0, {a & MASK32:#010x}",
            f"s_mov_b32 s1, {(a >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {b & MASK32:#010x}",
            f"v_mov_b32 v3, {(b >> 32) & MASK32:#010x}",
        ]
        got, _ = run_translation_add_f64(0, 0, setup=setup, operands="s[0:1], v[2:3]")
        self.assertEqual(got, f64_to_u64(125.0))

        # Inline float constant 0.5
        got, _ = run_translation_add_f64(a, 0, operands="v[0:1], 0.5")
        self.assertEqual(got, f64_to_u64(100.5))

        # Inline float constant -1.0
        got, _ = run_translation_add_f64(0, b, operands="-1.0, v[2:3]")
        self.assertEqual(got, f64_to_u64(24.0))


class TestVAddF64ChainedCalculations(unittest.TestCase):
    """Verifies sequential chained operations across multiple instructions."""

    def test_chained_three_way_accumulation(self):
        # Sum four doubles: ((1.0 + 2.0) + 3.0) + 4.0 = 10.0
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x3ff00000",  # 1.0
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40000000",  # 2.0
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0x40080000",  # 3.0
            "v_mov_b32 v6, 0",
            "v_mov_b32 v7, 0x40100000",  # 4.0
            "v_add_f64 v[8:9], v[0:1], v[2:3]",     # 1.0 + 2.0 = 3.0
            "v_add_f64 v[10:11], v[8:9], v[4:5]",   # 3.0 + 3.0 = 6.0
            "v_add_f64 v[12:13], v[10:11], v[6:7]", # 6.0 + 4.0 = 10.0
            "exp mrt0 v12, v13, v12, v13 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_chained_f64", lines)
        res = spirv_interp.run(spv)
        got = (res.vgpr[12] & MASK32) | ((res.vgpr[13] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(10.0))


class TestVAddF64RoundingModes(unittest.TestCase):
    """Verifies MODE.FP_ROUND behavior: nearest-even (0) vs toward-zero (3)."""

    def test_mode_rounding_differences(self):
        one = 0x3FF0000000000000
        three_half_ulp = 0x3CB8000000000000  # 1.5 * 2^-52 = 3 * 2^-53
        got_even, _ = run_translation_add_f64(one, three_half_ulp, mode=0)
        got_zero, _ = run_translation_add_f64(one, three_half_ulp, mode=3)
        self.assertEqual(got_even, 0x3FF0000000000002)
        self.assertEqual(got_zero, 0x3FF0000000000001)

    def test_mode_overflow_differences(self):
        got_even, _ = run_translation_add_f64(DBL_MAX, DBL_MAX, mode=0)
        got_zero, _ = run_translation_add_f64(DBL_MAX, DBL_MAX, mode=3)
        self.assertEqual(got_even, POS_INF)
        self.assertEqual(got_zero, DBL_MAX)


if __name__ == "__main__":
    unittest.main()
