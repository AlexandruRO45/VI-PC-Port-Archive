from __future__ import annotations

import enum
import math
import struct
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

"""
RDNA2 VALU instruction semantic tests: v_cvt_pk_i32_i16.

Converts packed two 16-bit signed integers into two 32-bit signed integers.
Single-source format:
    v[dst:dst+1] = v_cvt_pk_i32_i16 src0
    dst[0] = sign_extend(src0[15:0])
    dst[1] = sign_extend(src0[31:16])

Dual-source format (VOP3 with op_sel):
    v[dst:dst+1] = v_cvt_pk_i32_i16 src0, src1, op_sel=[sel0, sel1]
    dst[0] = sign_extend(sel0 ? src0[31:16] : src0[15:0])
    dst[1] = sign_extend(sel1 ? src1[31:16] : src1[15:0])

Destination is always a pair of consecutive 32-bit VGPRs: v[d:d+1].
No condition codes or carry/borrow flags are set.
"""

MASK16 = 0xFFFF
MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF
SIGN16 = 0x8000
SIGN32 = 0x8000_0000


def sign_extend_16_to_32(val: int) -> int:
    """Sign-extend a 16-bit signed integer into a 32-bit unsigned representation."""
    val = val & MASK16
    if val & SIGN16:
        return (val | 0xFFFF_0000) & MASK32
    return val


def to_signed_32(val: int) -> int:
    """Convert unsigned 32-bit representation to Python signed int."""
    val = val & MASK32
    if val & SIGN32:
        return val - 0x1_0000_0000
    return val


def to_signed_16(val: int) -> int:
    """Convert unsigned 16-bit representation to Python signed int."""
    val = val & MASK16
    if val & SIGN16:
        return val - 0x1_0000
    return val


def ref_cvt_pk_i32_i16(src: int) -> Tuple[int, int]:
    """
    Reference mathematical oracle for single-source v_cvt_pk_i32_i16.
    Extracts two 16-bit signed integers from a 32-bit register and sign-extends them to 32-bit words.
    Returns (d0, d1) as unsigned 32-bit integers.
    """
    src32 = src & MASK32
    lo16 = src32 & MASK16
    hi16 = (src32 >> 16) & MASK16
    d0 = sign_extend_16_to_32(lo16)
    d1 = sign_extend_16_to_32(hi16)
    return d0, d1


def ref_cvt_pk_i32_i16_64(src: int) -> int:
    """Return combined 64-bit pair (d1 << 32) | d0 for single-source v_cvt_pk_i32_i16."""
    d0, d1 = ref_cvt_pk_i32_i16(src)
    return (d1 << 32) | d0


def ref_cvt_pk_i32_i16_dual(src0: int, src1: int, op_sel: Tuple[int, int] = (0, 0)) -> Tuple[int, int]:
    """
    Reference mathematical oracle for dual-source v_cvt_pk_i32_i16.
    op_sel[0]: select high half of src0 if 1, else low half
    op_sel[1]: select high half of src1 if 1, else low half
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32

    half0 = ((s0 >> 16) & MASK16) if (op_sel[0] & 1) else (s0 & MASK16)
    half1 = ((s1 >> 16) & MASK16) if (op_sel[1] & 1) else (s1 & MASK16)

    d0 = sign_extend_16_to_32(half0)
    d1 = sign_extend_16_to_32(half1)
    return d0, d1


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle and Architectural Model
# ==============================================================================

class TestValuCvtPkI32I16Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_both_zero(self) -> None:
        self.assertEqual(ref_cvt_pk_i32_i16(0), (0, 0))
        self.assertEqual(ref_cvt_pk_i32_i16_64(0), 0)

    def test_positive_signed_unpacking(self) -> None:
        src = 0x0042_0010
        d0, d1 = ref_cvt_pk_i32_i16(src)
        self.assertEqual(d0, 0x0000_0010)
        self.assertEqual(d1, 0x0000_0042)
        self.assertEqual(to_signed_32(d0), 16)
        self.assertEqual(to_signed_32(d1), 66)
        self.assertEqual(ref_cvt_pk_i32_i16_64(src), 0x0000_0042_0000_0010)

    def test_negative_signed_unpacking(self) -> None:
        src = 0xFFFF_FFFF  # Both halves are -1 (0xFFFF)
        d0, d1 = ref_cvt_pk_i32_i16(src)
        self.assertEqual(d0, 0xFFFF_FFFF)
        self.assertEqual(d1, 0xFFFF_FFFF)
        self.assertEqual(to_signed_32(d0), -1)
        self.assertEqual(to_signed_32(d1), -1)
        self.assertEqual(ref_cvt_pk_i32_i16_64(src), 0xFFFF_FFFF_FFFF_FFFF)

    def test_mixed_signs(self) -> None:
        src = 0xFFFF_0001  # lo is +1, hi is -1
        d0, d1 = ref_cvt_pk_i32_i16(src)
        self.assertEqual(d0, 0x0000_0001)
        self.assertEqual(d1, 0xFFFF_FFFF)
        self.assertEqual(to_signed_32(d0), 1)
        self.assertEqual(to_signed_32(d1), -1)

    def test_int16_min_max_extremes(self) -> None:
        # lo is -32768 (0x8000), hi is +32767 (0x7FFF)
        src = 0x7FFF_8000
        d0, d1 = ref_cvt_pk_i32_i16(src)
        self.assertEqual(d0, 0xFFFF_8000)
        self.assertEqual(d1, 0x0000_7FFF)
        self.assertEqual(to_signed_32(d0), -32768)
        self.assertEqual(to_signed_32(d1), 32767)

    def test_dual_source_oracle(self) -> None:
        s0 = 0x8001_0002  # hi is -32767, lo is +2
        s1 = 0x0004_FFFE  # hi is +4, lo is -2
        self.assertEqual(ref_cvt_pk_i32_i16_dual(s0, s1, (0, 0)), (2, 0xFFFF_FFFE))
        self.assertEqual(ref_cvt_pk_i32_i16_dual(s0, s1, (1, 0)), (0xFFFF_8001, 0xFFFF_FFFE))
        self.assertEqual(ref_cvt_pk_i32_i16_dual(s0, s1, (0, 1)), (2, 4))
        self.assertEqual(ref_cvt_pk_i32_i16_dual(s0, s1, (1, 1)), (0xFFFF_8001, 4))

    def test_powers_of_two_and_sign_propagation(self) -> None:
        for bit in range(16):
            val_lo = 1 << bit
            d0, d1 = ref_cvt_pk_i32_i16(val_lo)
            self.assertEqual(d1, 0)
            if bit == 15:
                self.assertEqual(d0, 0xFFFF_8000)
            else:
                self.assertEqual(d0, 1 << bit)

if __name__ == "__main__":
    unittest.main()
