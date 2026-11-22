from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pk_u32_u16.

Models AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics
for the vector integer unpacking and format conversion instruction:
    v_cvt_pk_u32_u16 dst, src

RDNA2 VALU Instruction Specification:
    Instruction: v_cvt_pk_u32_u16 D.u64, S0.u32 (or D.2u32, S0.u32, S1.u32)
    Operands:
        - D:   64-bit VGPR pair destination v[d:d+1] (two consecutive 32-bit VGPRs)
        - S0:  32-bit source (VGPR, SGPR, or literal/inline constant) containing packed 16-bit integers
        - S1:  Optional second 32-bit source for dual-source VOP2/VOP3 encodings

Mathematical Operation:
    Unpacks two 16-bit unsigned integers into two 32-bit unsigned integers with zero-extension:
        low_u16  = S0[15:0]
        high_u16 = S0[31:16] (or S1[15:0] in dual-source format)

        D0 (bits 31:0,  v[d])   = uint32(low_u16)  = low_u16
        D1 (bits 63:32, v[d+1]) = uint32(high_u16) = high_u16

    Combined 64-bit Destination:
        D = (uint32(high_u16) << 32) | uint32(low_u16)

Key Architectural Properties:
    1. Zero-Extension:
       Each 16-bit field is zero-extended to 32 bits. The upper 16 bits of each
       destination dword (D0[31:16] and D1[31:16]) are strictly zeroed.
    2. Multi-Register Write:
       Overwrites two consecutive 32-bit vector registers v[d] and v[d+1].
    3. Inverse / Complementary to v_cvt_pk_u16_u32:
       v_cvt_pk_u16_u32 packs two 32-bit integers into one 32-bit register (with saturation).
       v_cvt_pk_u32_u16 unpacks two 16-bit integers into two 32-bit registers (zero-extended).
    4. Exact Roundtrip Property:
       For any (u16_0, u16_1), v_cvt_pk_u32_u16(v_cvt_pk_u16_u32(u16_0, u16_1)) preserves
       both integers exactly.
    5. No Status Flag / VCC Side Effects:
       Unpacking does not alter condition codes, carry/borrow flags, or VCC.

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
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF

U16_MIN: int = 0x0000
U16_MAX: int = 0xFFFF
U32_MIN: int = 0x0000_0000
U32_MAX: int = 0xFFFF_FFFF


# ==============================================================================
# Reference Mathematical Oracles and Bit Operations
# ==============================================================================

def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range [0, 65535]."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def unpack_u16(dword: int) -> Tuple[int, int]:
    """Unpack a 32-bit dword into two 16-bit unsigned integers (lo, hi)."""
    dw = dword & MASK32
    return dw & MASK16, (dw >> 16) & MASK16


def ref_cvt_pk_u32_u16(src: int) -> Tuple[int, int]:
    """Reference mathematical oracle for RDNA2 v_cvt_pk_u32_u16.
    
    Unpacks low and high 16-bit halves of src into two zero-extended 32-bit unsigned integers.
    Returns (d0, d1) where d0 is the low dword and d1 is the high dword.
    """
    lo_u16, hi_u16 = unpack_u16(src)
    d0 = lo_u16  # zero-extended to 32 bits
    d1 = hi_u16  # zero-extended to 32 bits
    return d0, d1


def ref_cvt_pk_u32_u16_64(src: int) -> int:
    """Reference oracle returning the combined 64-bit integer (d1 << 32) | d0."""
    d0, d1 = ref_cvt_pk_u32_u16(src)
    return ((d1 & MASK32) << 32) | (d0 & MASK32)


def ref_cvt_pk_u32_u16_dual(
    src0: int,
    src1: int,
    op_sel: Tuple[int, int] = (0, 0),
) -> Tuple[int, int]:
    """Dual-source reference oracle with halfword selection.
    
    op_sel[0] selects low (0) or high (1) 16 bits of src0 for d0.
    op_sel[1] selects low (0) or high (1) 16 bits of src1 for d1.
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    d0 = (s0 >> 16) & MASK16 if op_sel[0] else s0 & MASK16
    d1 = (s1 >> 16) & MASK16 if op_sel[1] else s1 & MASK16
    return d0, d1


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestValuCvtPkU32U16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_both_zero(self) -> None:
        self.assertEqual(ref_cvt_pk_u32_u16(0), (0, 0))
        self.assertEqual(ref_cvt_pk_u32_u16_64(0), 0)

    def test_basic_values_unpacking(self) -> None:
        src = 0x56781234
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d0, 0x00001234)
        self.assertEqual(d1, 0x00005678)
        self.assertEqual(ref_cvt_pk_u32_u16_64(src), 0x00005678_00001234)

    def test_max_u16_boundary(self) -> None:
        src = 0xFFFF_FFFF
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d0, 0x0000_FFFF)
        self.assertEqual(d1, 0x0000_FFFF)
        self.assertEqual(ref_cvt_pk_u32_u16_64(src), 0x0000_FFFF_0000_FFFF)

    def test_isolated_low_half(self) -> None:
        src = 0x0000_ABCD
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d0, 0x0000_ABCD)
        self.assertEqual(d1, 0x0000_0000)

    def test_isolated_high_half(self) -> None:
        src = 0xABCD_0000
        d0, d1 = ref_cvt_pk_u32_u16(src)
        self.assertEqual(d0, 0x0000_0000)
        self.assertEqual(d1, 0x0000_ABCD)

    def test_upper_16_bits_strictly_zeroed(self) -> None:
        # Check that bits 31:16 of both d0 and d1 are always 0
        test_values = [
            0x0000_0000,
            0x0000_FFFF,
            0xFFFF_0000,
            0xFFFF_FFFF,
            0x1234_5678,
            0x8000_8000,
            0x7FFF_7FFF,
        ]
        for val in test_values:
            with self.subTest(val=hex(val)):
                d0, d1 = ref_cvt_pk_u32_u16(val)
                self.assertEqual(d0 & ~MASK16, 0)
                self.assertEqual(d1 & ~MASK16, 0)

    def test_dual_source_oracle(self) -> None:
        s0 = 0xAAAA_1111
        s1 = 0xBBBB_2222
        # op_sel (0, 0): selects low halves
        self.assertEqual(ref_cvt_pk_u32_u16_dual(s0, s1, (0, 0)), (0x1111, 0x2222))
        # op_sel (1, 0): selects high half of s0, low half of s1
        self.assertEqual(ref_cvt_pk_u32_u16_dual(s0, s1, (1, 0)), (0xAAAA, 0x2222))
        # op_sel (0, 1): selects low half of s0, high half of s1
        self.assertEqual(ref_cvt_pk_u32_u16_dual(s0, s1, (0, 1)), (0x1111, 0xBBBB))
        # op_sel (1, 1): selects high halves
        self.assertEqual(ref_cvt_pk_u32_u16_dual(s0, s1, (1, 1)), (0xAAAA, 0xBBBB))

    def test_powers_of_two_bit_positions(self) -> None:
        for bit in range(32):
            val = 1 << bit
            d0, d1 = ref_cvt_pk_u32_u16(val)
            if bit < 16:
                self.assertEqual(d0, 1 << bit)
                self.assertEqual(d1, 0)
            else:
                self.assertEqual(d0, 0)
                self.assertEqual(d1, 1 << (bit - 16))


if __name__ == "__main__":
    unittest.main()
