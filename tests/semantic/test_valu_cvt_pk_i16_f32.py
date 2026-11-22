from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pk_i16_f32.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the packed float-to-integer conversion instruction:
  v_cvt_pk_i16_f32 dst, src0, src1

RDNA2 VALU Instruction Specification:
  v_cvt_pk_i16_f32 D.u32, S0.f32, S1.f32
  Operands:
    - D:  32-bit VGPR destination (holding two packed signed 16-bit integers)
    - S0: 32-bit float source (VGPR/SGPR/literal/inline constant), providing low 16 bits
    - S1: 32-bit float source (VGPR/SGPR/literal/inline constant), providing high 16 bits

  Mathematical Operation:
    D[15:0]  = f32_to_i16(S0)
    D[31:16] = f32_to_i16(S1)

    where f32_to_i16(f) truncates single-precision IEEE 754 float toward zero (RTZ)
    and saturates to the signed 16-bit integer range [-32768, 32767]:
      - If f is NaN: returns 0 (0x0000) per DirectX / AMD RDNA2 specification.
      - If f >= 32767.0f (or +infinity): saturates to 32767 (0x7FFF).
      - If f <= -32768.0f (or -infinity): saturates to -32768 (0x8000).
      - Otherwise: returns int(f) & 0xFFFF (truncation toward zero).

    Packed Destination Dword:
      D = (f32_to_i16(S0) & 0xFFFF) | ((f32_to_i16(S1) & 0xFFFF) << 16)

  Key Architectural Properties:
    1. Independent saturation: S0 and S1 are converted and saturated independently.
    2. Truncation toward zero (RTZ): Fractional values round toward 0 (+1.9 -> 1, -1.9 -> -1).
    3. Low/High Packing: S0 occupies the lower 16-bit halfword D[15:0], while S1 occupies
       the upper 16-bit halfword D[31:16].
    4. Destination is always a VGPR: The full 32-bit destination register is overwritten.
    5. Special values: NaN produces 0; ±0.0 produces 0; subnormals truncate to 0.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import math
import struct
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT32: int = 0x80000000
SIGN_BIT16: int = 0x8000

INT16_MIN: int = -32768
INT16_MAX: int = 32767

INT16_MIN_U16: int = 0x8000
INT16_MAX_U16: int = 0x7FFF

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_ONE_F32: int = 0x3F800000
NEG_ONE_F32: int = 0xBF800000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned halfword as a signed two's-complement integer."""
    val &= MASK16
    return val - (1 << 16) if (val & SIGN_BIT16) else val


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 single-precision 32-bit unsigned integer."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0 else NEG_INF_F32


def is_nan_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision NaN."""
    bits &= MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_inf_f32(bits: int) -> bool:
    """True if 32-bit integer represents an IEEE 754 single-precision infinity."""
    bits &= MASK32
    return (bits & 0x7FFFFFFF) == 0x7F800000


def f32_to_i16(bits: int, *, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Convert float32 bit pattern to saturated signed 16-bit integer (unsigned 16-bit word)."""
    b = bits & MASK32
    if abs_mod:
        b &= ~SIGN_BIT32
    if neg_mod:
        b ^= SIGN_BIT32

    # NaN returns 0 per AMD RDNA2 / DirectX specification
    if is_nan_f32(b):
        return 0

    val = bits_to_f32(b)

    # Positive saturation / +infinity
    if math.isinf(val) and val > 0:
        return INT16_MAX_U16
    if math.isinf(val) and val < 0:
        return INT16_MIN_U16

    if val >= 32767.0:
        return INT16_MAX_U16
    if val <= -32768.0:
        return INT16_MIN_U16

    # Truncation toward zero
    truncated = int(val)
    return truncated & MASK16


def pack_i16_to_u32(lo_i16: int, hi_i16: int) -> int:
    """Pack two 16-bit integers into a 32-bit unsigned dword (lo in [15:0], hi in [31:16])."""
    return ((hi_i16 & MASK16) << 16) | (lo_i16 & MASK16)


def unpack_i16_from_u32(val32: int) -> tuple[int, int]:
    """Unpack a 32-bit unsigned dword into low and high 16-bit words (lo, hi)."""
    val = val32 & MASK32
    return val & MASK16, (val >> 16) & MASK16


def ref_cvt_pk_i16_f32(
    src0: int,
    src1: int,
    *,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pk_i16_f32.

    Converts two 32-bit single-precision floats into saturated 16-bit signed integers and
    packs them into a single 32-bit dword:
      D[15:0]  = f32_to_i16(S0)
      D[31:16] = f32_to_i16(S1)
    """
    lo = f32_to_i16(src0, abs_mod=abs0, neg_mod=neg0)
    hi = f32_to_i16(src1, abs_mod=abs1, neg_mod=neg1)
    return pack_i16_to_u32(lo, hi)


class TestValuCvtPkI16F32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_in_range_values_truncation(self) -> None:
        """Verify integer conversions and truncation toward zero."""
        test_cases = [
            (f32_to_bits(0.0), f32_to_bits(0.0), 0x00000000),
            (f32_to_bits(1.0), f32_to_bits(0.0), 0x00000001),
            (f32_to_bits(0.0), f32_to_bits(1.0), 0x00010000),
            (f32_to_bits(10.0), f32_to_bits(-10.0), 0xFFF6000A),
            (f32_to_bits(1.9), f32_to_bits(-1.9), 0xFFFF0001),   # RTZ: 1.9 -> 1, -1.9 -> -1
            (f32_to_bits(0.999), f32_to_bits(-0.999), 0x00000000), # RTZ to 0
            (f32_to_bits(32766.0), f32_to_bits(-32767.0), 0x80017FFE),
            (f32_to_bits(32767.0), f32_to_bits(-32768.0), 0x80007FFF),
        ]
        for s0, s1, expected in test_cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                res = ref_cvt_pk_i16_f32(s0, s1)
                self.assertEqual(res, expected)

    def test_positive_overflow_saturation(self) -> None:
        """Values >= 32767.0f and +infinity saturate to 32767 (0x7FFF)."""
        overflow_values = [
            f32_to_bits(32767.0),
            f32_to_bits(32768.0),
            f32_to_bits(65536.0),
            f32_to_bits(1e6),
            POS_INF_F32,
        ]
        for val in overflow_values:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_cvt_pk_i16_f32(val, 0), 0x00007FFF)
                self.assertEqual(ref_cvt_pk_i16_f32(0, val), 0x7FFF0000)
                self.assertEqual(ref_cvt_pk_i16_f32(val, val), 0x7FFF7FFF)

    def test_negative_underflow_saturation(self) -> None:
        """Values <= -32768.0f and -infinity saturate to -32768 (0x8000)."""
        underflow_values = [
            f32_to_bits(-32768.0),
            f32_to_bits(-32769.0),
            f32_to_bits(-65536.0),
            f32_to_bits(-1e6),
            NEG_INF_F32,
        ]
        for val in underflow_values:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_cvt_pk_i16_f32(val, 0), 0x00008000)
                self.assertEqual(ref_cvt_pk_i16_f32(0, val), 0x80000000)
                self.assertEqual(ref_cvt_pk_i16_f32(val, val), 0x80008000)

    def test_nan_returns_zero(self) -> None:
        """NaN values produce 0 (0x0000)."""
        for nan_val in [QNAN_F32, SNAN_F32, 0x7FC01234, 0xFFC00000, 0xFF800001]:
            with self.subTest(nan=hex(nan_val)):
                self.assertEqual(f32_to_i16(nan_val), 0)
                self.assertEqual(ref_cvt_pk_i16_f32(nan_val, nan_val), 0x00000000)

    def test_input_modifiers(self) -> None:
        """abs and neg modifiers apply prior to conversion."""
        # abs(-10.0f) -> 10 -> 0x000A
        neg_ten = f32_to_bits(-10.0)
        self.assertEqual(ref_cvt_pk_i16_f32(neg_ten, 0, abs0=True), 0x0000000A)

        # neg(10.0f) -> -10 -> 0xFFF6
        pos_ten = f32_to_bits(10.0)
        self.assertEqual(ref_cvt_pk_i16_f32(0, pos_ten, neg1=True), 0xFFF60000)


if __name__ == "__main__":
    unittest.main()
