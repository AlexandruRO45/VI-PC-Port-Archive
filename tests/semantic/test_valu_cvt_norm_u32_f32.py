"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_u32_f32.

Models IEEE 754-2008 / IEEE 754-2019 binary32 (single-precision float) to 32-bit unsigned normalized integer
(UNORM32) conversions and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  - v_cvt_norm_u32_f32: D0.u32 = f32_to_unorm(S0.f32)

RDNA2 Floating-Point Conversion ISA Rules:
  1. Register Widths and Destination Semantics:
     - v_cvt_norm_u32_f32 operates on a 32-bit single-precision floating-point source in S0[31:0].
     - Writes a 32-bit unsigned normalized integer (UNORM32) into D0[31:0].
     - Fully writes the 32-bit destination VGPR.
  2. Normalized Integer Conversion & Rounding:
     - Scales normalized range [0.0, 1.0] across full 32-bit integer range [0, 4294967295] (0xFFFFFFFF).
     - Rounding mode: IEEE 754 round-to-nearest-even (RNE) is applied to the scaled value (val * 4294967295.0).
     - Midpoint ties (e.g. 0.5 * 4294967295.0 = 2147483647.5) round to the nearest even integer (2147483648 = 0x80000000).
  3. Boundary Extremes & Special Values:
     - Values <= 0.0 (including +0.0, -0.0, negative finite values, and -infinity) saturate/clamp to 0.
     - Values >= 1.0 (including positive numbers > 1.0 and +infinity) saturate/clamp to 4294967295 (0xFFFFFFFF).
     - Subnormal single-precision inputs round to 0 without faulting.
  4. NaN Handling:
     - Any NaN input (quiet NaN or signaling NaN, positive or negative) converts to 0.
  5. VOP1 / VOP3 Encoding and Modifiers:
     - VOP1 (e32): Compact encoding with single source VGPR or SGPR or inline constant.
     - VOP3 (e64): Extended encoding supporting input modifiers:
       * abs (|x|): Clears sign bit before conversion.
       * neg (-x): Inverts sign bit before conversion.
       * neg(abs(x)): Inverts sign after abs (forcing negative, which subsequently yields 0).
       * clamp: Clamps converted result. For normalized conversion, output is already within [0, 0xFFFFFFFF].

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import math
import struct
import unittest
from fractions import Fraction
from typing import Any, List, Optional, Tuple, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary32 Bit Patterns
SIGN_MASK_F32: int = 0x80000000
EXP_MASK_F32: int = 0x7F800000
MANT_MASK_F32: int = 0x007FFFFF
QNAN_BIT_F32: int = 0x00400000

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
NEG_QNAN_F32: int = 0xFFC00000
SNAN_F32: int = 0x7F800001
NEG_SNAN_F32: int = 0xFF800001

ONE_F32: int = 0x3F800000          # 1.0f
NEG_ONE_F32: int = 0xBF800000      # -1.0f
HALF_F32: int = 0x3F000000         # 0.5f
NEG_HALF_F32: int = 0xBF000000     # -0.5f
QUARTER_F32: int = 0x3E800000      # 0.25f
THREE_QUARTERS_F32: int = 0x3F400000  # 0.75f
TWO_F32: int = 0x40000000          # 2.0f
NEG_TWO_F32: int = 0xC0000000      # -2.0f
TEN_F32: int = 0x41200000          # 10.0f
NEG_TEN_F32: int = 0xC1200000      # -10.0f

MAX_NORMAL_F32: int = 0x7F7FFFFF   # (2 - 2^-23) * 2^127 ~= 3.4028e38
MIN_NORMAL_F32: int = 0x00800000   # 2^-126 ~= 1.1755e-38
MAX_SUBNORMAL_F32: int = 0x007FFFFF # (1 - 2^-23) * 2^-126
MIN_SUBNORMAL_F32: int = 0x00000001 # 2^-149 ~= 1.4013e-45

# UNORM32 Constants
UNORM32_MIN: int = 0x00000000
UNORM32_MAX: int = 0xFFFFFFFF
UNORM32_HALF: int = 0x80000000
UNORM32_QUARTER: int = 0x40000000
UNORM32_THREE_QUARTERS: int = 0xC0000000


def f32_to_bits(val: float) -> int:
    """Pack Python float into 32-bit binary32 integer bit pattern."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0 else NEG_INF_F32


def bits_to_f32(bits: int) -> float:
    """Unpack 32-bit integer bit pattern into Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary32 NaN."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) != 0)


def is_qnan_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary32 Quiet NaN."""
    return is_nan_f32(bits) and bool(bits & QNAN_BIT_F32)


def is_snan_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary32 Signaling NaN."""
    return is_nan_f32(bits) and not bool(bits & QNAN_BIT_F32)


def is_inf_f32(bits: int) -> bool:
    """Return True if bits represent IEEE 754 binary32 infinity."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) == 0)


def is_subnormal_f32(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary32 subnormal value."""
    return ((bits & EXP_MASK_F32) == 0) and ((bits & MANT_MASK_F32) != 0)


def is_zero_f32(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F32) == 0


def apply_f32_modifiers(bits: int, abs_val: bool = False, neg: bool = False) -> int:
    """Applies VOP3 input modifiers: abs clears sign, neg inverts sign."""
    res = bits & MASK32
    if abs_val:
        res &= ~SIGN_MASK_F32
    if neg:
        res ^= SIGN_MASK_F32
    return res


def round_unorm32_ties_to_even(frac_val: Fraction) -> int:
    """Rounds an exact non-negative Fraction to nearest integer with ties to even."""
    n = frac_val.numerator
    d = frac_val.denominator
    q, r = divmod(n, d)
    rem_doubled = 2 * r
    if rem_doubled < d:
        return q
    elif rem_doubled > d:
        return q + 1
    else:
        # Exact tie: round to even integer
        return q if (q % 2 == 0) else q + 1


def ref_cvt_norm_u32_f32(
    src32: int,
    clamp: bool = False,
    neg: bool = False,
    abs_val: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_norm_u32_f32.

    Converts 32-bit single-precision floating-point value to 32-bit unsigned normalized integer.
    """
    s_mod = apply_f32_modifiers(src32, abs_val=abs_val, neg=neg)

    # 1. NaN inputs convert to 0
    if is_nan_f32(s_mod):
        return UNORM32_MIN

    # 2. Signed zeros convert to 0
    if is_zero_f32(s_mod):
        return UNORM32_MIN

    # 3. Negative values (including -inf) convert to 0
    if bool(s_mod & SIGN_MASK_F32):
        return UNORM32_MIN

    # 4. Positive infinity converts to 4294967295
    if is_inf_f32(s_mod):
        return UNORM32_MAX

    # 5. Finite non-negative floating-point values
    f_val = bits_to_f32(s_mod)
    if f_val <= 0.0:
        return UNORM32_MIN
    if f_val >= 1.0:
        return UNORM32_MAX

    # Exact scaling: f_val * 4294967295 with roundTiesToEven
    scaled_frac = Fraction(f_val) * 4294967295
    unorm_int = round_unorm32_ties_to_even(scaled_frac)

    # Clamp to [0, 4294967295]
    if unorm_int < UNORM32_MIN:
        unorm_int = UNORM32_MIN
    elif unorm_int > UNORM32_MAX:
        unorm_int = UNORM32_MAX

    return unorm_int


class TestValuCvtNormU32F32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_zero_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u32_f32(POS_ZERO_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_ZERO_F32), UNORM32_MIN)

    def test_unit_and_boundary_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u32_f32(ONE_F32), UNORM32_MAX)
        self.assertEqual(ref_cvt_norm_u32_f32(HALF_F32), UNORM32_HALF)
        self.assertEqual(ref_cvt_norm_u32_f32(TWO_F32), UNORM32_MAX)
        self.assertEqual(ref_cvt_norm_u32_f32(TEN_F32), UNORM32_MAX)
        self.assertEqual(ref_cvt_norm_u32_f32(MAX_NORMAL_F32), UNORM32_MAX)

    def test_negative_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_ONE_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_HALF_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_TWO_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_TEN_F32), UNORM32_MIN)

    def test_infinity_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u32_f32(POS_INF_F32), UNORM32_MAX)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_INF_F32), UNORM32_MIN)

    def test_nan_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u32_f32(QNAN_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_QNAN_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(SNAN_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_SNAN_F32), UNORM32_MIN)
        # NaN with custom payload
        custom_nan = 0x7FC00123
        self.assertEqual(ref_cvt_norm_u32_f32(custom_nan), UNORM32_MIN)

    def test_tie_breaking_nearest_even(self) -> None:
        # 0.5 * 4294967295 = 2147483647.5 -> rounds to 2147483648 (even)
        self.assertEqual(ref_cvt_norm_u32_f32(HALF_F32), UNORM32_HALF)
        self.assertEqual(ref_cvt_norm_u32_f32(HALF_F32) % 2, 0)

        # 0.25 * 4294967295 = 1073741823.75 -> rounds to 1073741824
        self.assertEqual(ref_cvt_norm_u32_f32(QUARTER_F32), UNORM32_QUARTER)

        # 0.75 * 4294967295 = 3221225471.25 -> rounds to 3221225471
        self.assertEqual(ref_cvt_norm_u32_f32(THREE_QUARTERS_F32), 3221225471)

    def test_input_modifiers(self) -> None:
        # abs clears negative sign
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_ONE_F32, abs_val=True), UNORM32_MAX)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_HALF_F32, abs_val=True), UNORM32_HALF)

        # neg inverts positive sign to negative -> 0
        self.assertEqual(ref_cvt_norm_u32_f32(ONE_F32, neg=True), UNORM32_MIN)

        # neg inverts negative sign to positive -> valid unorm
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_ONE_F32, neg=True), UNORM32_MAX)

        # neg(abs(x)) forces negative -> 0
        self.assertEqual(ref_cvt_norm_u32_f32(ONE_F32, abs_val=True, neg=True), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(NEG_ONE_F32, abs_val=True, neg=True), UNORM32_MIN)

    def test_subnormals(self) -> None:
        # Subnormals in f32 are all < 2^-126, which scale to < 2^-94, rounding to 0
        self.assertEqual(ref_cvt_norm_u32_f32(MIN_SUBNORMAL_F32), UNORM32_MIN)
        self.assertEqual(ref_cvt_norm_u32_f32(MAX_SUBNORMAL_F32), UNORM32_MIN)


if __name__ == "__main__":
    unittest.main()
