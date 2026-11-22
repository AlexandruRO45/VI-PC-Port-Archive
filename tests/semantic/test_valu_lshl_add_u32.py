from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_lshl_add_u32.

AMD RDNA VALU Instruction Set Architecture (ISA):
    v_lshl_add_u32 D, S0, S1, S2
Operation:
    Performs a 32-bit logical shift left of S0 by the shift count specified in S1,
    then adds S2 to the result modulo 2^32:
        dst = ((S0 << (S1 & 0x1F)) + S2) & 0xFFFFFFFF

Modifiers and Semantics:
    - S0: 32-bit unsigned integer base operand to shift.
    - S1: 32-bit unsigned integer shift count; only the low 5 bits [4:0] are used (modulo 32).
    - S2: 32-bit unsigned integer addend added to the shifted result.
    - Full 32-bit modulo wrapping arithmetic on addition without saturation.
    - Bits shifted left past bit 31 are discarded (zero-fill on the right).

Architectural model:
    32-bit vector registers (VGPRs) and scalar registers (SGPRs).
    The destination is always a 32-bit vector register (VGPR).
    Unlike carry-generating instructions, v_lshl_add_u32 does not write to VCC.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF
SHIFT_MASK: int = 0x1F


def oracle_v_lshl_add_u32(src0_bits: int, src1_bits: int, src2_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 v_lshl_add_u32."""
    s0 = src0_bits & MASK32
    shift = src1_bits & SHIFT_MASK
    s2 = src2_bits & MASK32

    shifted = (s0 << shift) & MASK32
    return (shifted + s2) & MASK32


class TestValuLshlAddU32ModelAndOracle(unittest.TestCase):
    """Architectural model and reference mathematical oracle tests for v_lshl_add_u32."""

    def test_zero_shift_identity(self) -> None:
        # (s0 << 0) + s2 == s0 + s2
        res = oracle_v_lshl_add_u32(100, 0, 50)
        self.assertEqual(res, 150)

    def test_zero_add_identity(self) -> None:
        # (s0 << s1) + 0 == s0 << s1
        res = oracle_v_lshl_add_u32(0x00000001, 16, 0)
        self.assertEqual(res, 0x00010000)

    def test_shift_count_modulo_32(self) -> None:
        # S1 values 32, 64, 96 mask to shift count 0
        self.assertEqual(oracle_v_lshl_add_u32(0x1234, 32, 0), 0x1234)
        self.assertEqual(oracle_v_lshl_add_u32(0x1234, 64, 0), 0x1234)
        # S1 value 33 masks to shift count 1
        self.assertEqual(oracle_v_lshl_add_u32(0x0001, 33, 0), 0x0002)
        # S1 value 0xFFFFFFFF masks to 31
        self.assertEqual(oracle_v_lshl_add_u32(0x0001, 0xFFFFFFFF, 0), 0x80000000)

    def test_modulo_32_addition_overflow(self) -> None:
        # Addition wraps modulo 2^32
        res = oracle_v_lshl_add_u32(0x80000000, 0, 0x80000000)
        self.assertEqual(res, 0)

        # Shifted value + addend overflows 32 bits
        res2 = oracle_v_lshl_add_u32(0x40000000, 1, 0x00000001)
        self.assertEqual(res2, 0x80000001)

    def test_high_bits_discarded_on_shift(self) -> None:
        # Shifting 0x80000000 left by 1 results in 0 (overflow discarded)
        self.assertEqual(oracle_v_lshl_add_u32(0x80000000, 1, 0x00000005), 0x00000005)

    def test_combined_shift_and_add(self) -> None:
        # (0x00000003 << 4) + 0x0000000F = 0x00000030 + 0x0000000F = 0x0000003F
        res = oracle_v_lshl_add_u32(3, 4, 15)
        self.assertEqual(res, 0x3F)


if __name__ == "__main__":
    unittest.main()
