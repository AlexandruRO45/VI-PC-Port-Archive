"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_u16_f16.

Models IEEE 754-2008 / IEEE 754-2019 binary16 (half-precision float) to 16-bit unsigned normalized integer
(UNORM16) conversions and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  - v_cvt_norm_u16_f16: D0.u16 = f16_to_unorm(S0.f16)

RDNA2 Floating-Point Conversion ISA Rules:
  1. Register Widths and Destination Semantics:
     - v_cvt_norm_u16_f16 operates on 16-bit half-precision floating-point source in S0[15:0].
     - Writes an unsigned normalized 16-bit integer (UNORM16) into D0[15:0].
     - On AMD RDNA2 / GFX10, 16-bit destination VGPR operations preserve the high 16 bits D0[31:16].
  2. Normalized Integer Conversion & Rounding:
     - Scales normalized range [0.0, 1.0] across full 16-bit integer range [0, 65535].
     - Rounding mode: IEEE 754 round-to-nearest-even (RNE) is applied to the scaled value (val * 65535.0).
     - Midpoint ties (e.g. 0.5 * 65535.0 = 32767.5) round to the nearest even integer (32768 = 0x8000).
  3. Boundary Extremes & Special Values:
     - Values <= 0.0 (including +0.0, -0.0, negative finite values, and -infinity) saturate/clamp to 0.
     - Values >= 1.0 (including positive numbers > 1.0 and +infinity) saturate/clamp to 65535 (0xFFFF).
     - Subnormal half-precision inputs round to 0 or appropriate integer without faulting.
  4. NaN Handling:
     - Any NaN input (quiet NaN or signaling NaN, positive or negative) converts to 0.
  5. VOP1 / VOP3 Encoding and Modifiers:
     - VOP1 (e32): Compact encoding with single source VGPR or SGPR or inline constant.
     - VOP3 (e64): Extended encoding supporting input modifiers:
       * abs (|x|): Clears sign bit before conversion.
       * neg (-x): Inverts sign bit before conversion.
       * neg(abs(x)): Inverts sign after abs (forcing negative, which subsequently yields 0).
       * clamp: Clamps converted result. For normalized conversion, output is already within [0, 65535].

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import math
import struct
import unittest
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
THREE_QUARTERS_F16: int = 0x3A00  # 0.75
TWO_F16: int = 0x4000          # 2.0
NEG_TWO_F16: int = 0xC000      # -2.0
TEN_F16: int = 0x4900          # 10.0
NEG_TEN_F16: int = 0xC900      # -10.0

MAX_NORMAL_F16: int = 0x7BFF   # 65504.0
MIN_NORMAL_F16: int = 0x0400   # 2^-14 ~= 6.1035e-5
MAX_SUBNORMAL_F16: int = 0x03FF # (1023/1024) * 2^-14
MIN_SUBNORMAL_F16: int = 0x0001 # 2^-24 ~= 5.9605e-8

# UNORM16 Constants
UNORM16_MIN: int = 0x0000
UNORM16_MAX: int = 0xFFFF
UNORM16_HALF: int = 0x8000


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


def round_unorm16_ties_to_even(frac_val: Fraction) -> int:
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


def ref_cvt_norm_u16_f16(
    src16: int,
    clamp: bool = False,
    neg: bool = False,
    abs_val: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_norm_u16_f16.

    Converts 16-bit half-precision floating-point value to 16-bit unsigned normalized integer.
    """
    s_mod = apply_f16_modifiers(src16, abs_val=abs_val, neg=neg)

    # 1. NaN inputs convert to 0
    if is_nan_f16(s_mod):
        return UNORM16_MIN

    # 2. Signed zeros convert to 0
    if is_zero_f16(s_mod):
        return UNORM16_MIN

    # 3. Negative values (including -inf) convert to 0
    if bool(s_mod & SIGN_MASK_F16):
        return UNORM16_MIN

    # 4. Positive infinity converts to 65535
    if is_inf_f16(s_mod):
        return UNORM16_MAX

    # 5. Finite non-negative floating-point values
    f_val = bits_to_f16(s_mod)
    if f_val <= 0.0:
        return UNORM16_MIN
    if f_val >= 1.0:
        return UNORM16_MAX

    # Exact scaling: f_val * 65535 with roundTiesToEven
    scaled_frac = Fraction(f_val) * 65535
    unorm_int = round_unorm16_ties_to_even(scaled_frac)

    # Clamp to [0, 65535]
    if unorm_int < UNORM16_MIN:
        unorm_int = UNORM16_MIN
    elif unorm_int > UNORM16_MAX:
        unorm_int = UNORM16_MAX

    return unorm_int


class TestValuCvtNormU16F16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_zero_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u16_f16(POS_ZERO_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ZERO_F16), UNORM16_MIN)

    def test_unit_and_boundary_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u16_f16(ONE_F16), UNORM16_MAX)
        self.assertEqual(ref_cvt_norm_u16_f16(HALF_F16), UNORM16_HALF)
        self.assertEqual(ref_cvt_norm_u16_f16(TWO_F16), UNORM16_MAX)
        self.assertEqual(ref_cvt_norm_u16_f16(TEN_F16), UNORM16_MAX)
        self.assertEqual(ref_cvt_norm_u16_f16(MAX_NORMAL_F16), UNORM16_MAX)

    def test_negative_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ONE_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_HALF_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_TWO_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_TEN_F16), UNORM16_MIN)

    def test_infinity_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u16_f16(POS_INF_F16), UNORM16_MAX)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_INF_F16), UNORM16_MIN)

    def test_nan_conversions(self) -> None:
        self.assertEqual(ref_cvt_norm_u16_f16(QNAN_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_QNAN_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(SNAN_F16), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_SNAN_F16), UNORM16_MIN)
        # NaN with custom payload
        custom_nan = 0x7E42
        self.assertEqual(ref_cvt_norm_u16_f16(custom_nan), UNORM16_MIN)

    def test_tie_breaking_nearest_even(self) -> None:
        # 0.5 * 65535 = 32767.5 -> rounds to 32768 (even)
        self.assertEqual(ref_cvt_norm_u16_f16(HALF_F16), 32768)
        self.assertEqual(ref_cvt_norm_u16_f16(HALF_F16) % 2, 0)

        # 0.25 * 65535 = 16383.75 -> rounds to 16384
        self.assertEqual(ref_cvt_norm_u16_f16(QUARTER_F16), 16384)

        # 0.75 * 65535 = 49151.25 -> rounds to 49151
        self.assertEqual(ref_cvt_norm_u16_f16(THREE_QUARTERS_F16), 49151)

    def test_input_modifiers(self) -> None:
        # abs clears negative sign
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ONE_F16, abs_val=True), UNORM16_MAX)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_HALF_F16, abs_val=True), UNORM16_HALF)

        # neg inverts positive sign to negative -> 0
        self.assertEqual(ref_cvt_norm_u16_f16(ONE_F16, neg=True), UNORM16_MIN)

        # neg inverts negative sign to positive -> valid unorm
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ONE_F16, neg=True), UNORM16_MAX)

        # neg(abs(x)) forces negative -> 0
        self.assertEqual(ref_cvt_norm_u16_f16(ONE_F16, abs_val=True, neg=True), UNORM16_MIN)
        self.assertEqual(ref_cvt_norm_u16_f16(NEG_ONE_F16, abs_val=True, neg=True), UNORM16_MIN)

    def test_subnormals(self) -> None:
        # Smallest subnormal 2^-24 * 65535 ~= 0.0039 rounds to 0
        self.assertEqual(ref_cvt_norm_u16_f16(MIN_SUBNORMAL_F16), UNORM16_MIN)
        # Largest subnormal ~6.0975e-5 * 65535 ~= 3.996 rounds to 4
        self.assertEqual(ref_cvt_norm_u16_f16(MAX_SUBNORMAL_F16), 4)


if __name__ == "__main__":
    unittest.main()
