from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pknorm_u16_f32.

Models the AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) instruction:
    v_cvt_pknorm_u16_f32 D, S0, S1

Lowered Architectural Semantics:
    D[15:0]  = convert_f32_to_unorm16(S0)
    D[31:16] = convert_f32_to_unorm16(S1)

RDNA2 VALU Specification Rules:
  1. Packing & Destination:
     - Destination is a single 32-bit VGPR (v[d]).
     - Overwrites all 32 bits of the destination register atomically.
     - D[15:0] is derived exclusively from S0; D[31:16] is derived exclusively from S1.
  2. Conversion Mathematics:
     - Input S0 and S1 are IEEE 754-2008 single-precision floating-point numbers (binary32).
     - Each channel is clamped to [0.0, 1.0]:
         * Values <= 0.0 (including -0.0) clamp to 0.0 -> 0x0000.
         * Values >= 1.0 clamp to 1.0 -> 0xFFFF (65535).
         * +Infinity clamps to 1.0 -> 0xFFFF.
         * -Infinity clamps to 0.0 -> 0x0000.
     - Normalized scaling:
         * Product = x * 65535.0
         * Rounded to nearest integer with tie-breaking to even (IEEE 754 roundTiesToEven).
  3. NaN Semantics:
     - Any NaN input (Quiet NaN or Signaling NaN, positive or negative sign, any payload)
       evaluates to 0x0000 (as confirmed on RDNA2 hardware, SHADER-137 / SHADER-339).
  4. VOP3 Source Modifiers:
     - abs (|x|): Clears sign bit (bit 31) of f32 input before conversion.
     - neg (-x): Inverts sign bit (bit 31) of f32 input before conversion.
     - neg(abs(x)): Inverts sign bit after abs, forcing negative sign -> clamps to 0x0000.
     - clamp on instruction destination is rejected / unsupported.

Standard library Python only (python -I, struct, math, unittest). Zero game data.
"""

import math
import struct
import unittest

# Bitwise Masks & Architectural Constants
MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF
SIGN32: int = 0x8000_0000
ABS32: int = 0x7FFF_FFFF
EXP_MASK32: int = 0x7F80_0000
MANT_MASK32: int = 0x007F_FFFF
QNAN_BIT32: int = 0x0040_0000
CANONICAL_QNAN32: int = 0x7FC0_0000

UNORM16_MIN: int = 0x0000
UNORM16_MAX: int = 0xFFFF
SCALE_U16: float = 65535.0

# IEEE 754 binary32 Bit Patterns
POS_ZERO_F32: int = 0x0000_0000
NEG_ZERO_F32: int = 0x8000_0000
POS_ONE_F32: int = 0x3F80_0000          # 1.0f
NEG_ONE_F32: int = 0xBF80_0000          # -1.0f
POS_TWO_F32: int = 0x4000_0000          # 2.0f
NEG_TWO_F32: int = 0xC000_0000          # -2.0f
HALF_F32: int = 0x3F00_0000             # 0.5f
NEG_HALF_F32: int = 0xBF00_0000         # -0.5f
QUARTER_F32: int = 0x3E80_0000          # 0.25f
NEG_QUARTER_F32: int = 0xBE80_0000      # -0.25f
THREE_QUARTERS_F32: int = 0x3F40_0000   # 0.75f

POS_INF_F32: int = 0x7F80_0000          # +Infinity
NEG_INF_F32: int = 0xFF80_0000          # -Infinity
POS_QNAN_F32: int = 0x7FC0_0000         # Canonical Quiet NaN
NEG_QNAN_F32: int = 0xFFC0_0000         # Negative Quiet NaN
POS_SNAN_F32: int = 0x7F80_0001         # Signaling NaN
NEG_SNAN_F32: int = 0xFF80_0001         # Negative Signaling NaN
SUBNORM_MIN_F32: int = 0x0000_0001      # Smallest positive subnormal (2^-149)
SUBNORM_MAX_F32: int = 0x007F_FFFF      # Largest positive subnormal
NEG_SUBNORM_MIN_F32: int = 0x8000_0001
NEG_SUBNORM_MAX_F32: int = 0x807F_FFFF


def f32_to_bits(val: float) -> int:
    """Pack Python float into IEEE 754 binary32 unsigned 32-bit integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0.0 else NEG_INF_F32


def bits_to_f32(bits: int) -> float:
    """Unpack unsigned 32-bit integer into Python float (IEEE 754 binary32)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def split_f32_bits(bits: int) -> tuple[int, int, int]:
    """Unpack 32-bit word into (sign, exponent, fraction) components."""
    bits &= MASK32
    sign = (bits & SIGN32) >> 31
    exp = (bits & EXP_MASK32) >> 23
    mant = bits & MANT_MASK32
    return sign, exp, mant


def is_nan32(bits: int) -> bool:
    """Return True if 32-bit word represents an IEEE 754 NaN."""
    bits &= MASK32
    return ((bits & EXP_MASK32) == EXP_MASK32) and ((bits & MANT_MASK32) != 0)


def is_qnan32(bits: int) -> bool:
    """Return True if 32-bit word represents an IEEE 754 Quiet NaN."""
    bits &= MASK32
    return is_nan32(bits) and bool(bits & QNAN_BIT32)


def is_snan32(bits: int) -> bool:
    """Return True if 32-bit word represents an IEEE 754 Signaling NaN."""
    bits &= MASK32
    return is_nan32(bits) and not bool(bits & QNAN_BIT32)


def is_inf32(bits: int) -> bool:
    """Return True if 32-bit word represents positive or negative infinity."""
    bits &= MASK32
    return ((bits & EXP_MASK32) == EXP_MASK32) and ((bits & MANT_MASK32) == 0)


def is_subnormal32(bits: int) -> bool:
    """Return True if 32-bit word represents an IEEE 754 subnormal float."""
    bits &= MASK32
    return ((bits & EXP_MASK32) == 0) and ((bits & MANT_MASK32) != 0)


def is_zero32(bits: int) -> bool:
    """Return True if 32-bit word represents +0.0 or -0.0."""
    return (bits & ABS32) == 0


def apply_f32_modifiers(bits32: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply RDNA2 source modifiers (|src|, -src) to a 32-bit float word.

    In RDNA2 ISA, abs clears sign bit (bit 31) first, then neg inverts sign bit (bit 31).
    """
    res = bits32 & MASK32
    if abs_mod:
        res &= ABS32
    if neg_mod:
        res ^= SIGN32
    return res


def pack_u16_to_u32(lo_half: int, hi_half: int) -> int:
    """Pack two 16-bit integers into an unsigned 32-bit dword."""
    return ((hi_half & MASK16) << 16) | (lo_half & MASK16)


def extract_u16(val32: int, high_half: bool) -> int:
    """Extract low 16 bits (high_half=False) or high 16 bits (high_half=True)."""
    return ((val32 >> 16) & MASK16) if high_half else (val32 & MASK16)


def oracle_cvt_unorm16_f32(f32_bits: int) -> int:
    """Mathematical reference oracle converting a single f32 to unorm16.

    Semantics:
        - If input is NaN: returns 0x0000.
        - If input <= 0.0 (including -0.0, -inf): returns 0x0000.
        - If input >= 1.0 (including +inf): returns 0xFFFF.
        - Otherwise: product (x * 65535.0) evaluated in f32 (matching OpFMul)
                     is rounded to nearest even integer (roundTiesToEven) & 0xFFFF.
    """
    if is_nan32(f32_bits):
        return UNORM16_MIN
    val = bits_to_f32(f32_bits)
    if val <= 0.0:
        return UNORM16_MIN
    if val >= 1.0 or math.isinf(val):
        return UNORM16_MAX
    # Product is rounded to single-precision float first (OpFMul) as in shader ALU
    prod_bits = f32_to_bits(val * SCALE_U16)
    prod_f32 = bits_to_f32(prod_bits)
    return round(prod_f32) & MASK16


def oracle_v_cvt_pknorm_u16_f32(
    src0_bits: int,
    src1_bits: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Architectural reference oracle for RDNA2 v_cvt_pknorm_u16_f32.

    Packs cvt_unorm16(src0) into dst[15:0] and
          cvt_unorm16(src1) into dst[31:16],
    applying respective source modifiers.
    """
    s0_mod = apply_f32_modifiers(src0_bits, abs_mod=abs0, neg_mod=neg0)
    s1_mod = apply_f32_modifiers(src1_bits, abs_mod=abs1, neg_mod=neg1)

    lo = oracle_cvt_unorm16_f32(s0_mod)
    hi = oracle_cvt_unorm16_f32(s1_mod)

    return pack_u16_to_u32(lo, hi)


# ==============================================================================
# Unit Tests: Architectural Reference Model & Mathematical Oracle
# ==============================================================================

class TestValuCvtPknormU16F32Oracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_cvt_pknorm_u16_f32."""

    def test_basic_normal_conversions(self) -> None:
        # 0.0 -> 0x0000, 1.0 -> 0xFFFF
        self.assertEqual(oracle_cvt_unorm16_f32(POS_ZERO_F32), 0)
        self.assertEqual(oracle_cvt_unorm16_f32(POS_ONE_F32), 0xFFFF)

        # 0.5 -> 32768 (32767.5 rounds to even 32768)
        self.assertEqual(oracle_cvt_unorm16_f32(HALF_F32), 32768)

        # 0.25 -> 16384 (16383.75 rounds to 16384)
        self.assertEqual(oracle_cvt_unorm16_f32(QUARTER_F32), 16384)

        # 0.75 -> 49151 (49151.25 rounds to 49151)
        self.assertEqual(oracle_cvt_unorm16_f32(THREE_QUARTERS_F32), 49151)

    def test_channel_packing_and_ordering(self) -> None:
        # Low channel (S0) gets 1.0 -> 0xFFFF
        # High channel (S1) gets 0.25 -> 0x4000 (16384)
        res = oracle_v_cvt_pknorm_u16_f32(POS_ONE_F32, QUARTER_F32)
        expected = (16384 << 16) | 0xFFFF
        self.assertEqual(res, expected)
        self.assertEqual(extract_u16(res, high_half=False), 0xFFFF)
        self.assertEqual(extract_u16(res, high_half=True), 16384)

    def test_clamping_out_of_range(self) -> None:
        # Negative values clamp to 0x0000
        self.assertEqual(oracle_cvt_unorm16_f32(NEG_ONE_F32), 0)
        self.assertEqual(oracle_cvt_unorm16_f32(NEG_TWO_F32), 0)
        self.assertEqual(oracle_cvt_unorm16_f32(f32_to_bits(-100.0)), 0)

        # Values > 1.0 clamp to 0xFFFF
        self.assertEqual(oracle_cvt_unorm16_f32(POS_TWO_F32), 0xFFFF)
        self.assertEqual(oracle_cvt_unorm16_f32(f32_to_bits(1.5)), 0xFFFF)
        self.assertEqual(oracle_cvt_unorm16_f32(f32_to_bits(1000.0)), 0xFFFF)

    def test_infinities(self) -> None:
        self.assertEqual(oracle_cvt_unorm16_f32(POS_INF_F32), 0xFFFF)
        self.assertEqual(oracle_cvt_unorm16_f32(NEG_INF_F32), 0)
        res = oracle_v_cvt_pknorm_u16_f32(POS_INF_F32, NEG_INF_F32)
        self.assertEqual(res, 0x0000FFFF)

    def test_nan_evaluates_to_zero(self) -> None:
        # Quiet and signaling NaNs evaluate to 0
        self.assertEqual(oracle_cvt_unorm16_f32(POS_QNAN_F32), 0)
        self.assertEqual(oracle_cvt_unorm16_f32(NEG_QNAN_F32), 0)
        self.assertEqual(oracle_cvt_unorm16_f32(POS_SNAN_F32), 0)
        self.assertEqual(oracle_cvt_unorm16_f32(NEG_SNAN_F32), 0)
        # NaN with custom payload
        nan_payload = EXP_MASK32 | 0x001234
        self.assertEqual(oracle_cvt_unorm16_f32(nan_payload), 0)

    def test_signed_zeros(self) -> None:
        self.assertEqual(oracle_cvt_unorm16_f32(POS_ZERO_F32), 0)
        self.assertEqual(oracle_cvt_unorm16_f32(NEG_ZERO_F32), 0)
        self.assertEqual(oracle_v_cvt_pknorm_u16_f32(POS_ZERO_F32, NEG_ZERO_F32), 0)

    def test_round_ties_to_even_steps(self) -> None:
        # Step size is 1 / 65535.0
        # 0.5 / 65535.0 gives product 0.5 -> rounds to 0 (even)
        self.assertEqual(oracle_cvt_unorm16_f32(f32_to_bits(0.5 / 65535.0)), 0)
        # 1.5 / 65535.0 gives product 1.5 -> rounds to 2 (even)
        self.assertEqual(oracle_cvt_unorm16_f32(f32_to_bits(1.5 / 65535.0)), 2)
        # 2.5 / 65535.0 gives product 2.5 -> rounds to 2 (even)
        self.assertEqual(oracle_cvt_unorm16_f32(f32_to_bits(2.5 / 65535.0)), 2)
        # 3.5 / 65535.0 gives product 3.5 -> rounds to 4 (even)
        self.assertEqual(oracle_cvt_unorm16_f32(f32_to_bits(3.5 / 65535.0)), 4)
        # 4.5 / 65535.0 gives product 4.5 -> rounds to 4 (even)
        self.assertEqual(oracle_cvt_unorm16_f32(f32_to_bits(4.5 / 65535.0)), 4)

    def test_source_modifiers(self) -> None:
        # abs on negative input: |-0.25| = 0.25 -> 16384
        res_abs = oracle_v_cvt_pknorm_u16_f32(NEG_QUARTER_F32, POS_ZERO_F32, abs0=True)
        self.assertEqual(extract_u16(res_abs, high_half=False), 16384)

        # neg on positive input: -0.25 -> clamps to 0
        res_neg = oracle_v_cvt_pknorm_u16_f32(QUARTER_F32, POS_ZERO_F32, neg0=True)
        self.assertEqual(extract_u16(res_neg, high_half=False), 0)

        # abs then neg (-|x|): forces negative -> clamps to 0
        res_both = oracle_v_cvt_pknorm_u16_f32(QUARTER_F32, POS_ZERO_F32, abs0=True, neg0=True)
        self.assertEqual(extract_u16(res_both, high_half=False), 0)

        # Independent modifiers on S0 and S1
        # S0: |-0.5| = 0.5 -> 32768
        # S1: -(-1.0) = 1.0 -> 65535
        res_ind = oracle_v_cvt_pknorm_u16_f32(NEG_HALF_F32, NEG_ONE_F32, abs0=True, neg1=True)
        self.assertEqual(res_ind, (65535 << 16) | 32768)


if __name__ == "__main__":
    unittest.main()
