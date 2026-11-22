from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pk_i16_i32.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the packed integer format conversion instruction:
  v_cvt_pk_i16_i32 dst, src0, src1

RDNA2 VALU Instruction Specification:
  v_cvt_pk_i16_i32 D.u32, S0.i32, S1.i32
  Operands:
    - D:  32-bit VGPR destination (holding two packed signed 16-bit integers)
    - S0: 32-bit source (VGPR/SGPR/literal/inline constant), providing low 16 bits
    - S1: 32-bit source (VGPR/SGPR/literal/inline constant), providing high 16 bits

  Mathematical Operation:
    D[15:0]  = sat16(S0)
    D[31:16] = sat16(S1)

    where sat16(x) clamps a signed 32-bit integer x into the signed 16-bit range:
      [-32768, 32767] ([-0x8000, 0x7FFF]):
        if x > 32767:
            sat16(x) = 32767 (0x7FFF)
        elif x < -32768:
            sat16(x) = -32768 (0x8000)
        else:
            sat16(x) = x & 0xFFFF

    Packed Destination Dword:
      D = (sat16(S0) & 0xFFFF) | ((sat16(S1) & 0xFFFF) << 16)

  Key Architectural Properties:
    1. Independent saturation: S0 and S1 are converted and saturated independently.
    2. Signed saturation: Saturation treats inputs as signed 32-bit two's-complement
       values. Positive overflow saturates to 0x7FFF (+32767), while negative underflow
       saturates to 0x8000 (-32768).
    3. Low/High Packing: S0 occupies the lower 16-bit halfword D[15:0], while S1 occupies
       the upper 16-bit halfword D[31:16].
    4. Destination is always a VGPR: The full 32-bit destination register is overwritten.
    5. Modifiers: Integer clamp is not supported / rejected as per RDNA2 ISA contract.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT32: int = 0x80000000
SIGN_BIT16: int = 0x8000

INT32_MIN: int = -0x80000000
INT32_MAX: int = 0x7FFFFFFF
INT16_MIN: int = -32768
INT16_MAX: int = 32767

INT16_MIN_U16: int = 0x8000
INT16_MAX_U16: int = 0x7FFF


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


def sat_i16(val: int) -> int:
    """Saturate a 32-bit integer to signed 16-bit range [-32768, 32767], returning 16-bit unsigned bits."""
    s = s32(val)
    if s > INT16_MAX:
        return INT16_MAX_U16
    if s < INT16_MIN:
        return INT16_MIN_U16
    return s & MASK16


def pack_i16_to_u32(lo_i16: int, hi_i16: int) -> int:
    """Pack two 16-bit integers into a 32-bit unsigned dword (lo in [15:0], hi in [31:16])."""
    return ((hi_i16 & MASK16) << 16) | (lo_i16 & MASK16)


def unpack_i16_from_u32(val32: int) -> tuple[int, int]:
    """Unpack a 32-bit unsigned dword into low and high 16-bit words (lo, hi)."""
    val = val32 & MASK32
    return val & MASK16, (val >> 16) & MASK16


def ref_cvt_pk_i16_i32(src0: int, src1: int) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pk_i16_i32.

    Converts two 32-bit signed integers into saturated 16-bit signed integers and
    packs them into a single 32-bit dword:
      D[15:0]  = sat16(S0)
      D[31:16] = sat16(S1)
    """
    lo = sat_i16(src0)
    hi = sat_i16(src1)
    return pack_i16_to_u32(lo, hi)


def ref_cvt_pk_i16_i32_explicit(src0: int, src1: int) -> int:
    """Alternative architectural verification oracle with explicit boundary branching."""
    s0 = s32(src0)
    s1 = s32(src1)

    if s0 > 32767:
        lo = 32767
    elif s0 < -32768:
        lo = -32768
    else:
        lo = s0

    if s1 > 32767:
        hi = 32767
    elif s1 < -32768:
        hi = -32768
    else:
        hi = s1

    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


class TestValuCvtPkI16I32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_in_range_values_pass_through(self) -> None:
        """Values strictly within [-32768, 32767] pass through without clamping."""
        test_cases = [
            (0, 0, 0x00000000),
            (1, 0, 0x00000001),
            (0, 1, 0x00010000),
            (42, 100, (100 << 16) | 42),
            (0x7FFF, 0x7FFF, 0x7FFF7FFF),
            (-1, -1, 0xFFFFFFFF),
            (-32768, -32768, 0x80008000),
            (-32768, 32767, 0x7FFF8000),
            (32767, -32768, 0x80007FFF),
            (-1, 0, 0x0000FFFF),
            (0, -1, 0xFFFF0000),
            (-2, -3, 0xFFFDFFFE),
        ]
        for s0, s1, expected in test_cases:
            with self.subTest(s0=s0, s1=s1):
                res = ref_cvt_pk_i16_i32(s0, s1)
                self.assertEqual(res, expected)
                self.assertEqual(ref_cvt_pk_i16_i32_explicit(s0, s1), expected)

    def test_upper_saturation_boundary(self) -> None:
        """Values exceeding 32767 saturate strictly to 32767 (0x7FFF)."""
        overflow_values = [
            32768,
            32769,
            40000,
            65535,
            65536,
            0x00010000,
            0x12345678,
            0x7FFFFFFF,
            INT32_MAX,
        ]
        for val in overflow_values:
            with self.subTest(val=val):
                # S0 overflow, S1 zero
                self.assertEqual(ref_cvt_pk_i16_i32(val, 0), 0x00007FFF)
                # S1 overflow, S0 zero
                self.assertEqual(ref_cvt_pk_i16_i32(0, val), 0x7FFF0000)
                # Both overflow
                self.assertEqual(ref_cvt_pk_i16_i32(val, val), 0x7FFF7FFF)
                self.assertEqual(ref_cvt_pk_i16_i32_explicit(val, val), 0x7FFF7FFF)

    def test_lower_saturation_boundary(self) -> None:
        """Values less than -32768 saturate strictly to -32768 (0x8000)."""
        underflow_values = [
            -32769,
            -32770,
            -40000,
            -65536,
            -100000,
            -2147483648,
            INT32_MIN,
            0x80000000,
        ]
        for val in underflow_values:
            with self.subTest(val=val):
                # S0 underflow, S1 zero
                self.assertEqual(ref_cvt_pk_i16_i32(val, 0), 0x00008000)
                # S1 underflow, S0 zero
                self.assertEqual(ref_cvt_pk_i16_i32(0, val), 0x80000000)
                # Both underflow
                self.assertEqual(ref_cvt_pk_i16_i32(val, val), 0x80008000)
                self.assertEqual(ref_cvt_pk_i16_i32_explicit(val, val), 0x80008000)

    def test_asymmetric_mixed_boundaries(self) -> None:
        """Verify cross-combination of upper saturation, lower saturation, and in-range."""
        # S0 upper saturated, S1 lower saturated
        self.assertEqual(ref_cvt_pk_i16_i32(0x7FFFFFFF, 0x80000000), 0x80007FFF)
        # S0 lower saturated, S1 upper saturated
        self.assertEqual(ref_cvt_pk_i16_i32(0x80000000, 0x7FFFFFFF), 0x7FFF8000)
        # S0 in-range positive, S1 upper saturated
        self.assertEqual(ref_cvt_pk_i16_i32(0x1234, 0x7FFFFFFF), 0x7FFF1234)
        # S0 in-range negative, S1 lower saturated
        self.assertEqual(ref_cvt_pk_i16_i32(-5, -50000), 0x8000FFFB)

    def test_channel_independence(self) -> None:
        """Mutations to S0 never affect high half D[31:16], and mutations to S1 never affect low half D[15:0]."""
        fixed_hi = 0x1234
        for s0_cand in [-100000, -32768, -1, 0, 1, 32767, 100000]:
            packed = ref_cvt_pk_i16_i32(s0_cand, fixed_hi)
            lo, hi = unpack_i16_from_u32(packed)
            self.assertEqual(hi, fixed_hi)
            self.assertEqual(lo, sat_i16(s0_cand))

        fixed_lo = 0x5678
        for s1_cand in [-100000, -32768, -1, 0, 1, 32767, 100000]:
            packed = ref_cvt_pk_i16_i32(fixed_lo, s1_cand)
            lo, hi = unpack_i16_from_u32(packed)
            self.assertEqual(lo, fixed_lo)
            self.assertEqual(hi, sat_i16(s1_cand))

    def test_oracle_equivalence_across_sweeps(self) -> None:
        """Exhaustively verify equivalence between ref_cvt_pk_i16_i32 and explicit branch model."""
        test_points = [
            -0x80000000, -0x7FFFFFFF, -65537, -65536, -32769, -32768, -32767,
            -1, 0, 1, 32766, 32767, 32768, 65535, 65536, 0x7FFFFFFE, 0x7FFFFFFF
        ]
        for a in test_points:
            for b in test_points:
                self.assertEqual(ref_cvt_pk_i16_i32(a, b), ref_cvt_pk_i16_i32_explicit(a, b))


if __name__ == "__main__":
    unittest.main()
