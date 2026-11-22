from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pk_u16_u32.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the vector integer conversion and packing instruction:
    v_cvt_pk_u16_u32 dst, src0, src1

RDNA2 VALU Instruction Specification:
    Instruction: v_cvt_pk_u16_u32 D.u, S0.u, S1.u
    Operands:
        - D:   32-bit VGPR destination
        - S0:  32-bit unsigned integer source operand (VGPR, SGPR, or literal/inline constant)
        - S1:  32-bit unsigned integer source operand (VGPR in VOP2; VGPR, SGPR, or constant in VOP3)

Mathematical Operation:
    Converts two 32-bit unsigned integers (S0 and S1) into two 16-bit unsigned integers
    with unsigned saturation (clamping to [0, 65535]), then packs them into a single 32-bit VGPR:
        D[15:0]  = min(S0.u, 0xFFFF)
        D[31:16] = min(S1.u, 0xFFFF)

    Packed result:
        D = (min(S1.u, 0xFFFF) << 16) | min(S0.u, 0xFFFF)

Key Architectural Properties:
    1. Unsigned Saturation:
       Any 32-bit integer value strictly greater than 65535 (0xFFFF) saturates to 0xFFFF.
       Values within the representable unsigned 16-bit range [0, 65535] are preserved exactly.
    2. Zero Extension / Truncation Immunity:
       Unlike raw bitfield masking or truncation (which wraps modulo 65536), v_cvt_pk_u16_u32
       strictly clamps out-of-range values to UINT16_MAX.
    3. Half-word Packing:
       S0 maps strictly to the lower 16 bits (bits 15:0) of the destination.
       S1 maps strictly to the upper 16 bits (bits 31:16) of the destination.
    4. Full 32-bit Register Overwrite:
       Unlike sub-dword / D16 instructions that preserve existing halves of a VGPR,
       v_cvt_pk_u16_u32 unconditionally overwrites all 32 bits of the destination VGPR.
    5. No Status Flag / VCC Side Effects:
       Unsigned saturation does not alter condition codes or raise arithmetic exceptions.
       VCC and EXEC masks are preserved.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import math
import struct
import unittest
from typing import Tuple

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000

U16_MIN: int = 0x0000
U16_MAX: int = 0xFFFF
U32_MIN: int = 0x0000_0000
U32_MAX: int = 0xFFFF_FFFF


# ==============================================================================
# Bit Manipulation and Reference Mathematical Oracles
# ==============================================================================

def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u16_sat(val: int) -> int:
    """Unsigned 32-bit to unsigned 16-bit saturating conversion.
    
    Values in [0, 65535] are returned unchanged.
    Values greater than 65535 saturate to 65535 (0xFFFF).
    """
    v = val & MASK32
    return U16_MAX if v > U16_MAX else v


def pack_u16(lo: int, hi: int) -> int:
    """Pack two 16-bit unsigned integers into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack_u16(dword: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit unsigned integers (lo, hi)."""
    dw = dword & MASK32
    return dw & MASK16, (dw >> 16) & MASK16


def ref_cvt_pk_u16_u32(src0: int, src1: int) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pk_u16_u32.
    
    Converts src0 and src1 from unsigned 32-bit to saturated unsigned 16-bit,
    packing src0 into bits [15:0] and src1 into bits [31:16].
    """
    lo_u16 = u16_sat(src0)
    hi_u16 = u16_sat(src1)
    return pack_u16(lo_u16, hi_u16)


def ref_cvt_pk_u16_u32_decomposed(src0: int, src1: int) -> Tuple[int, int]:
    """Decomposed architectural reference returning (lo_u16, hi_u16)."""
    return u16_sat(src0), u16_sat(src1)


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestValuCvtPkU16U32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_both_zero(self) -> None:
        self.assertEqual(ref_cvt_pk_u16_u32(0, 0), 0)
        self.assertEqual(ref_cvt_pk_u16_u32_decomposed(0, 0), (0, 0))

    def test_in_range_values_exact_preservation(self) -> None:
        s0 = 0x1234
        s1 = 0x5678
        expected = 0x56781234
        self.assertEqual(ref_cvt_pk_u16_u32(s0, s1), expected)
        self.assertEqual(ref_cvt_pk_u16_u32_decomposed(s0, s1), (0x1234, 0x5678))

    def test_boundary_max_u16(self) -> None:
        s0 = 0xFFFF
        s1 = 0xFFFF
        expected = 0xFFFF_FFFF
        self.assertEqual(ref_cvt_pk_u16_u32(s0, s1), expected)
        self.assertEqual(ref_cvt_pk_u16_u32_decomposed(s0, s1), (0xFFFF, 0xFFFF))

    def test_saturation_at_u16_max_plus_one(self) -> None:
        # 65536 (0x10000) must saturate to 0xFFFF, not truncate to 0x0000
        s0 = 0x10000
        s1 = 0x10001
        expected = 0xFFFF_FFFF
        self.assertEqual(ref_cvt_pk_u16_u32(s0, s1), expected)
        self.assertEqual(ref_cvt_pk_u16_u32_decomposed(s0, s1), (0xFFFF, 0xFFFF))

    def test_extreme_u32_saturation(self) -> None:
        extremes = [
            0x0001_0000,
            0x0002_0000,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_0000,
            0xFFFF_FFFF,
        ]
        for val in extremes:
            with self.subTest(val=hex(val)):
                self.assertEqual(u16_sat(val), 0xFFFF)
                self.assertEqual(ref_cvt_pk_u16_u32(val, 0), 0x0000_FFFF)
                self.assertEqual(ref_cvt_pk_u16_u32(0, val), 0xFFFF_0000)
                self.assertEqual(ref_cvt_pk_u16_u32(val, val), 0xFFFF_FFFF)

    def test_asymmetric_operands(self) -> None:
        # src0 in-range, src1 saturated
        self.assertEqual(ref_cvt_pk_u16_u32(42, 100_000), 0xFFFF_002A)
        # src0 saturated, src1 in-range
        self.assertEqual(ref_cvt_pk_u16_u32(100_000, 42), 0x002A_FFFF)

    def test_roundtrip_unpack_integrity(self) -> None:
        test_pairs = [
            (0, 0),
            (1, 2),
            (65535, 65535),
            (65536, 10),
            (10, 65536),
            (0xDEADBEEF, 0xCAFEBABE),
            (0xFFFFFFFF, 0xFFFFFFFF),
        ]
        for s0, s1 in test_pairs:
            packed = ref_cvt_pk_u16_u32(s0, s1)
            lo, hi = unpack_u16(packed)
            expected_lo, expected_hi = ref_cvt_pk_u16_u32_decomposed(s0, s1)
            self.assertEqual(lo, expected_lo)
            self.assertEqual(hi, expected_hi)

    def test_powers_of_two_saturation_boundary(self) -> None:
        for bit in range(32):
            val = 1 << bit
            sat = u16_sat(val)
            if bit < 16:
                self.assertEqual(sat, val)
            else:
                self.assertEqual(sat, 0xFFFF)


if __name__ == "__main__":
    unittest.main()
