from __future__ import annotations

"""Translator-backed semantic tests for AMD RDNA2 v_ldexp_f64 (double-precision load exponent).

Instruction:
  v_ldexp_f64 dst, src0, src1
  D.f64 = S0.f64 * (2 ^ S1.i32)

Architecture:
  - RDNA2 Vector ALU (GFX10) binary64 scaling by signed 32-bit integer.
  - S0: 64-bit IEEE 754 binary64 floating-point operand (VGPR pair, SGPR pair, or literal).
  - S1: 32-bit signed two's complement integer exponent (VGPR, SGPR, or inline constant).
  - D:  64-bit IEEE 754 binary64 destination (VGPR pair).
  - Modifiers:
      * Input modifiers on S0: abs (|S0|), neg (-S0).
      * Output modifiers: clamp (clamps result to [0.0, 1.0]).
"""

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

SIGN_MASK_F64 = 0x8000000000000000
EXP_MASK_F64 = 0x7FF0000000000000
MANT_MASK_F64 = 0x000FFFFFFFFFFFFF
BIAS_F64 = 1023

POS_ZERO_F64 = 0x0000000000000000
NEG_ZERO_F64 = 0x8000000000000000
POS_INF_F64 = 0x7FF0000000000000
NEG_INF_F64 = 0xFFF0000000000000
QNAN_F64 = 0x7FF8000000000000
SNAN_F64 = 0x7FF0000000000001

ONE_F64 = 0x3FF0000000000000          # 1.0
TWO_F64 = 0x4000000000000000          # 2.0
HALF_F64 = 0x3FE0000000000000         # 0.5
MAX_FINITE_F64 = 0x7FEFFFFFFFFFFFFF   # ~1.7976931348623157e+308
MIN_NORMAL_F64 = 0x0010000000000000   # 2^-1022 (~2.2250738585072014e-308)
MAX_SUBNORMAL_F64 = 0x000FFFFFFFFFFFFF
MIN_SUBNORMAL_F64 = 0x0000000000000001 # 2^-1074 (~4.9406564584124654e-324)


def s32(val: int) -> int:
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def bits_to_f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    if math.isnan(val):
        return QNAN_F64
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return NEG_INF_F64 if val < 0.0 else POS_INF_F64


def run_v_ldexp_f64(
    src0_bits: int,
    src1_i32: int,
    *,
    op_line: str = "v_ldexp_f64 v[0:1], v[2:3], v4",
    setup_lines: list[str] | None = None,
) -> int:
    """Translate and run v_ldexp_f64 via support.translate_lines and mrt0_words."""
    lo = src0_bits & MASK32
    hi = (src0_bits >> 32) & MASK32
    n = src1_i32 & MASK32

    if setup_lines is None:
        lines = [
            f"v_mov_b32 v2, {lo:#x}",
            f"v_mov_b32 v3, {hi:#x}",
            f"v_mov_b32 v4, {n:#x}",
            op_line,
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
    else:
        lines = list(setup_lines) + [
            op_line,
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]

    spv = translate_lines("ps_ldexp_f64_hw", lines)
    words = mrt0_words(spv)
    return (words[1] << 32) | words[0]


class ValuLdexpF64HwTests(unittest.TestCase):
    def test_basic_powers_of_two(self):
        cases = [
            (1.0, 0, 1.0),
            (1.0, 1, 2.0),
            (1.0, 2, 4.0),
            (1.0, 3, 8.0),
            (1.0, -1, 0.5),
            (1.0, -2, 0.25),
            (1.0, -3, 0.125),
            (1.5, 2, 6.0),
            (-1.5, 2, -6.0),
            (3.0, -1, 1.5),
            (-3.0, -1, -1.5),
        ]
        for f_in, exp_shift, f_expected in cases:
            with self.subTest(f_in=f_in, exp_shift=exp_shift):
                got = run_v_ldexp_f64(f64_to_bits(f_in), exp_shift)
                expected = f64_to_bits(f_expected)
                self.assertEqual(got, expected)

    def test_zero_and_special_values(self):
        # Signed zeros pass through with sign preserved
        self.assertEqual(run_v_ldexp_f64(POS_ZERO_F64, 5), POS_ZERO_F64)
        self.assertEqual(run_v_ldexp_f64(POS_ZERO_F64, -5), POS_ZERO_F64)
        self.assertEqual(run_v_ldexp_f64(NEG_ZERO_F64, 5), NEG_ZERO_F64)
        self.assertEqual(run_v_ldexp_f64(NEG_ZERO_F64, -5), NEG_ZERO_F64)

        # Infinities pass through unchanged
        self.assertEqual(run_v_ldexp_f64(POS_INF_F64, 10), POS_INF_F64)
        self.assertEqual(run_v_ldexp_f64(POS_INF_F64, -10), POS_INF_F64)
        self.assertEqual(run_v_ldexp_f64(NEG_INF_F64, 10), NEG_INF_F64)
        self.assertEqual(run_v_ldexp_f64(NEG_INF_F64, -10), NEG_INF_F64)

        # NaNs pass through unchanged
        got_qnan = run_v_ldexp_f64(QNAN_F64, 1)
        self.assertEqual(got_qnan, QNAN_F64)
        got_snan = run_v_ldexp_f64(SNAN_F64, 1)
        self.assertEqual(got_snan, SNAN_F64)

    def test_overflow_and_underflow_boundaries(self):
        # Exponent scaling causing overflow to infinity
        self.assertEqual(run_v_ldexp_f64(ONE_F64, 1024), POS_INF_F64)
        self.assertEqual(run_v_ldexp_f64(ONE_F64 | SIGN_MASK_F64, 1024), NEG_INF_F64)
        self.assertEqual(run_v_ldexp_f64(MAX_FINITE_F64, 1), POS_INF_F64)
        self.assertEqual(run_v_ldexp_f64(MAX_FINITE_F64, 2000), POS_INF_F64)

        # Extreme large positive exponent clamped
        self.assertEqual(run_v_ldexp_f64(ONE_F64, 0x7FFFFFFF), POS_INF_F64)

        # Extreme underflow to signed zero
        self.assertEqual(run_v_ldexp_f64(ONE_F64, -2000), POS_ZERO_F64)
        self.assertEqual(run_v_ldexp_f64(ONE_F64 | SIGN_MASK_F64, -2000), NEG_ZERO_F64)
        self.assertEqual(run_v_ldexp_f64(ONE_F64, s32(0x80000000)), POS_ZERO_F64)

    def test_subnormal_scaling(self):
        # Subnormal input normalized and scaled up to normal range
        got_norm = run_v_ldexp_f64(MIN_SUBNORMAL_F64, 52)
        self.assertEqual(got_norm, MIN_NORMAL_F64)

        # Normal scaled down into subnormal range
        got_sub = run_v_ldexp_f64(MIN_NORMAL_F64, -52)
        self.assertEqual(got_sub, MIN_SUBNORMAL_F64)

        # Normal scaled down by 1 power of 2 into subnormal range (2^-1023)
        got_sub1 = run_v_ldexp_f64(MIN_NORMAL_F64, -1)
        self.assertEqual(got_sub1, 0x0008000000000000)

        # Max subnormal scaled up by 1 power of 2 into normal range
        got_sub_max = run_v_ldexp_f64(MAX_SUBNORMAL_F64, 1)
        self.assertEqual(got_sub_max, 0x001FFFFFFFFFFFFE)

    def test_input_modifiers(self):
        # abs modifier (|src0|)
        got_abs_pos = run_v_ldexp_f64(
            f64_to_bits(2.0), 1, op_line="v_ldexp_f64 v[0:1], |v[2:3]|, v4"
        )
        self.assertEqual(got_abs_pos, f64_to_bits(4.0))

        got_abs_neg = run_v_ldexp_f64(
            f64_to_bits(-2.0), 1, op_line="v_ldexp_f64 v[0:1], |v[2:3]|, v4"
        )
        self.assertEqual(got_abs_neg, f64_to_bits(4.0))

        # neg modifier (-src0)
        got_neg_pos = run_v_ldexp_f64(
            f64_to_bits(2.0), 1, op_line="v_ldexp_f64 v[0:1], -v[2:3], v4"
        )
        self.assertEqual(got_neg_pos, f64_to_bits(-4.0))

        got_neg_neg = run_v_ldexp_f64(
            f64_to_bits(-2.0), 1, op_line="v_ldexp_f64 v[0:1], -v[2:3], v4"
        )
        self.assertEqual(got_neg_neg, f64_to_bits(4.0))

        # neg abs modifier (-|src0|)
        got_negabs = run_v_ldexp_f64(
            f64_to_bits(-2.0), 1, op_line="v_ldexp_f64 v[0:1], -|v[2:3]|, v4"
        )
        self.assertEqual(got_negabs, f64_to_bits(-4.0))

    def test_clamp_output_modifier(self):
        # Result > 1.0 clamps to 1.0
        got_clamped_high = run_v_ldexp_f64(
            f64_to_bits(2.0), 1, op_line="v_ldexp_f64 v[0:1], v[2:3], v4 clamp"
        )
        self.assertEqual(got_clamped_high, ONE_F64)

        # Result < 0.0 clamps to +0.0
        got_clamped_low = run_v_ldexp_f64(
            f64_to_bits(-2.0), 1, op_line="v_ldexp_f64 v[0:1], v[2:3], v4 clamp"
        )
        self.assertEqual(got_clamped_low, POS_ZERO_F64)

        # Result in [0.0, 1.0] preserved
        got_clamped_mid = run_v_ldexp_f64(
            f64_to_bits(0.25), 1, op_line="v_ldexp_f64 v[0:1], v[2:3], v4 clamp"
        )
        self.assertEqual(got_clamped_mid, f64_to_bits(0.5))

    def test_sgpr_and_inline_constant_operands(self):
        # SGPR pair for S0 and inline constant for S1
        setup = [
            f"s_mov_b32 s0, {ONE_F64 & MASK32:#x}",
            f"s_mov_b32 s1, {(ONE_F64 >> 32) & MASK32:#x}",
        ]
        got_sgpr = run_v_ldexp_f64(
            ONE_F64, 3, op_line="v_ldexp_f64 v[0:1], s[0:1], 3", setup_lines=setup
        )
        self.assertEqual(got_sgpr, f64_to_bits(8.0))

    def test_cross_check_python_ldexp_sweep(self):
        # Sweep normal floats against Python math.ldexp
        sweep_values = [
            0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.141592653589793, 10.0, 100.0, 1e-10, 1e10,
        ]
        for v in sweep_values:
            for shift in (-5, -2, -1, 0, 1, 2, 5):
                with self.subTest(v=v, shift=shift):
                    got = run_v_ldexp_f64(f64_to_bits(v), shift)
                    expected = f64_to_bits(math.ldexp(v, shift))
                    self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
