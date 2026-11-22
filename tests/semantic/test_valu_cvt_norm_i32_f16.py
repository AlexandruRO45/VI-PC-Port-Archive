"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_i32_f16.

Models IEEE 754-2008 / IEEE 754-2019 binary16 (half-precision float) to 32-bit signed normalized integer
(SNORM32) conversions and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  - v_cvt_norm_i32_f16: D0.i32 = f16_to_snorm(S0.f16)

RDNA2 Floating-Point Conversion ISA Rules:
  1. Register Widths and Destination Semantics:
     - v_cvt_norm_i32_f16 operates on 16-bit half-precision floating-point source in S0[15:0].
     - Writes a signed normalized 32-bit integer (SNORM32) into D0[31:0].
     - Fully writes the 32-bit destination VGPR.
  2. Normalized Signed Integer Conversion & Rounding:
     - Scales normalized range [-1.0, 1.0] across 32-bit signed integer range [-2147483647, 2147483647]
       using scale factor 2147483647.0 (2^31 - 1).
     - Rounding mode: IEEE 754 round-to-nearest-even (RNE) is applied to the scaled value (val * 2147483647.0).
     - Midpoint ties (e.g. 0.5 * 2147483647.0 = 1073741823.5) round to the nearest even integer (1073741824 = 0x40000000).
  3. Boundary Extremes & Special Values:
     - Values <= -1.0 (including negative values <= -1.0 and -infinity) saturate to -2147483647.
     - Values >= 1.0 (including positive numbers >= 1.0 and +infinity) saturate to 2147483647 (0x7FFFFFFF).
     - Subnormal half-precision inputs round to 0 without faulting.
     - Signed zeros (+0.0 and -0.0) convert to 0.
  4. NaN Handling:
     - Any NaN input (quiet NaN or signaling NaN, positive or negative) converts to 0.
  5. VOP1 / VOP3 Encoding and Modifiers:
     - VOP1 (e32): Compact encoding with single source VGPR or SGPR or inline constant.
     - VOP3 (e64): Extended encoding supporting input modifiers:
       * abs (|x|): Clears sign bit before conversion.
       * neg (-x): Inverts sign bit before conversion.
       * clamp: Clamps converted result to [0, 2147483647].

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import math
import struct
import unittest
from enum import Enum, auto
from fractions import Fraction
from typing import Any, List, Optional, Tuple, Union

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary16 Bit Patterns
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
FRAC_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200

POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
QNAN_F16: int = 0x7E00
NEG_QNAN_F16: int = 0xFE00
SNAN_F16: int = 0x7C01
NEG_SNAN_F16: int = 0xFC01

ONE_F16: int = 0x3C00          # 1.0
NEG_ONE_F16: int = 0xBC00      # -1.0
HALF_F16: int = 0x3800         # 0.5
NEG_HALF_F16: int = 0xB800     # -0.5
QUARTER_F16: int = 0x3400      # 0.25
NEG_QUARTER_F16: int = 0xB400  # -0.25
THREE_QUARTERS_F16: int = 0x3A00  # 0.75
NEG_THREE_QUARTERS_F16: int = 0xBA00  # -0.75
TWO_F16: int = 0x4000          # 2.0
NEG_TWO_F16: int = 0xC000      # -2.0
TEN_F16: int = 0x4900          # 10.0
NEG_TEN_F16: int = 0xC900      # -10.0

MAX_NORMAL_F16: int = 0x7BFF   # 65504.0
MIN_NORMAL_F16: int = 0x0400   # 2^-14 ~= 6.1035e-5
MAX_SUBNORMAL_F16: int = 0x03FF # (1023/1024) * 2^-14
MIN_SUBNORMAL_F16: int = 0x0001 # 2^-24 ~= 5.9605e-8

# SNORM32 Constants
SNORM32_MAX: int = 0x7FFFFFFF          # 2147483647
SNORM32_MIN: int = -2147483647         # -2147483647 (two's complement 0x80000001)
SNORM32_MIN_RAW: int = 0x80000001
SNORM32_ZERO: int = 0x00000000
SNORM32_HALF: int = 0x40000000         # 1073741824
SNORM32_NEG_HALF_RAW: int = 0xC0000000 # -1073741824 in 32-bit two's complement


def f16_to_bits(val: float) -> int:
    """Pack Python float into 16-bit binary16 integer bit pattern."""
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return POS_INF_F16 if val > 0 else NEG_INF_F16


def bits_to_f16(bits: int) -> float:
    """Unpack 16-bit integer bit pattern into Python float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def is_nan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Return True if bits represent IEEE 754 binary16 infinity."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & FRAC_MASK_F16) == 0)


def is_subnormal_f16(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 binary16 subnormal value."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & FRAC_MASK_F16) != 0)


def is_zero_f16(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK_F16) == 0


def apply_f16_modifiers(bits: int, abs_val: bool = False, neg: bool = False) -> int:
    """Applies VOP3 input modifiers: abs clears sign, neg inverts sign."""
    res = bits & MASK16
    if abs_val:
        res &= ~SIGN_MASK_F16
    if neg:
        res ^= SIGN_MASK_F16
    return res


def round_snorm32_ties_to_even(frac_val: Fraction) -> int:
    """Rounds an exact Fraction to nearest integer with ties to even."""
    sign = -1 if frac_val < 0 else 1
    pos_frac = abs(frac_val)
    n = pos_frac.numerator
    d = pos_frac.denominator
    q, r = divmod(n, d)
    rem_doubled = 2 * r
    if rem_doubled < d:
        mag = q
    elif rem_doubled > d:
        mag = q + 1
    else:
        # Exact tie: round to even integer
        mag = q if (q % 2 == 0) else q + 1
    return sign * mag


def ref_cvt_norm_i32_f16(
    src16: int,
    clamp: bool = False,
    neg: bool = False,
    abs_val: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_norm_i32_f16.

    Converts 16-bit half-precision floating-point value to 32-bit signed normalized integer.
    Returns 32-bit unsigned bit pattern (two's complement representation).
    """
    s_mod = apply_f16_modifiers(src16, abs_val=abs_val, neg=neg)

    # 1. NaN inputs convert to 0
    if is_nan_f16(s_mod):
        return SNORM32_ZERO

    # 2. Signed zeros convert to 0
    if is_zero_f16(s_mod):
        return SNORM32_ZERO

    # 3. Infinities
    if is_inf_f16(s_mod):
        is_neg = bool(s_mod & SIGN_MASK_F16)
        res_int = SNORM32_MIN if is_neg else SNORM32_MAX
        if clamp and res_int < 0:
            res_int = 0
        return res_int & MASK32

    # 4. Finite values
    f_val = bits_to_f16(s_mod)
    if f_val >= 1.0:
        res_int = SNORM32_MAX
    elif f_val <= -1.0:
        res_int = SNORM32_MIN
    else:
        scaled_frac = Fraction(f_val) * 2147483647
        res_int = round_snorm32_ties_to_even(scaled_frac)
        if res_int < SNORM32_MIN:
            res_int = SNORM32_MIN
        elif res_int > SNORM32_MAX:
            res_int = SNORM32_MAX

    if clamp and res_int < 0:
        res_int = 0

    return res_int & MASK32


class TestValuCvtNormI32F16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_zero_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_i32_f16(POS_ZERO_F16), SNORM32_ZERO)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_ZERO_F16), SNORM32_ZERO)

    def test_unit_and_boundary_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_i32_f16(ONE_F16), SNORM32_MAX)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_ONE_F16), SNORM32_MIN_RAW)
        self.assertEqual(ref_cvt_norm_i32_f16(HALF_F16), SNORM32_HALF)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_HALF_F16), SNORM32_NEG_HALF_RAW)
        self.assertEqual(ref_cvt_norm_i32_f16(TWO_F16), SNORM32_MAX)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_TWO_F16), SNORM32_MIN_RAW)

    def test_infinity_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_i32_f16(POS_INF_F16), SNORM32_MAX)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_INF_F16), SNORM32_MIN_RAW)

    def test_nan_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_i32_f16(QNAN_F16), SNORM32_ZERO)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_QNAN_F16), SNORM32_ZERO)
        self.assertEqual(ref_cvt_norm_i32_f16(SNAN_F16), SNORM32_ZERO)
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_SNAN_F16), SNORM32_ZERO)

    def test_tie_breaking_nearest_even(self) -> None:
        # 0.5 * 2147483647 = 1073741823.5 -> 1073741824 (even)
        self.assertEqual(ref_cvt_norm_i32_f16(HALF_F16), 1073741824)
        self.assertEqual(ref_cvt_norm_i32_f16(HALF_F16) % 2, 0)

        # -0.5 * 2147483647 = -1073741823.5 -> -1073741824 (even)
        raw_neg = ref_cvt_norm_i32_f16(NEG_HALF_F16)
        self.assertEqual(raw_neg, (-1073741824) & MASK32)

    def test_input_modifiers(self) -> None:
        # abs clears negative sign
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_ONE_F16, abs_val=True), SNORM32_MAX)
        # neg inverts sign
        self.assertEqual(ref_cvt_norm_i32_f16(ONE_F16, neg=True), SNORM32_MIN_RAW)
        # clamp modifier clips negatives to 0
        self.assertEqual(ref_cvt_norm_i32_f16(NEG_HALF_F16, clamp=True), SNORM32_ZERO)

if __name__ == "__main__":
    unittest.main()
