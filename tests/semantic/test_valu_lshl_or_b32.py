from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_lshl_or_b32.

AMD RDNA VALU Instruction Set Architecture (ISA):
    v_lshl_or_b32 D, S0, S1, S2
Operation:
    Performs a 32-bit logical shift left of S0 by the shift count specified in S1,
    then bitwise ORs S2 with the shifted result:
        dst = ((S0 << (S1 & 0x1F)) | S2) & 0xFFFFFFFF

Modifiers and Semantics:
    - S0: 32-bit unsigned integer base value to shift.
    - S1: 32-bit unsigned integer shift count; only low 5 bits [4:0] are used (modulo 32).
    - S2: 32-bit unsigned integer bitmask to OR with the shifted base.
    - Zero-fills vacated bits on the right.
    - Bits shifted left past bit 31 are discarded.

Architectural model:
    32-bit vector registers (VGPRs) and scalar registers (SGPRs).
    The destination is always a 32-bit vector register (VGPR).
    VALU instructions do not modify VCC unless explicitly defined as carry/comparison ops.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF
SHIFT_MASK: int = 0x1F


def oracle_v_lshl_or_b32(src0_bits: int, src1_bits: int, src2_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 v_lshl_or_b32."""
    s0 = src0_bits & MASK32
    shift = src1_bits & SHIFT_MASK
    s2 = src2_bits & MASK32

    shifted = (s0 << shift) & MASK32
    return shifted | s2


class TestValuLshlOrB32ModelAndOracle(unittest.TestCase):
    """Architectural model and reference mathematical oracle tests for v_lshl_or_b32."""

    def test_zero_shift_identity(self) -> None:
        # (s0 << 0) | s2 == s0 | s2
        res = oracle_v_lshl_or_b32(0x12340000, 0, 0x00005678)
        self.assertEqual(res, 0x12345678)

    def test_zero_or_identity(self) -> None:
        # (s0 << s1) | 0 == s0 << s1
        res = oracle_v_lshl_or_b32(0x00000001, 16, 0)
        self.assertEqual(res, 0x00010000)

    def test_shift_count_modulo_32(self) -> None:
        # S1 values 32, 64, 96 mask to shift count 0
        self.assertEqual(oracle_v_lshl_or_b32(0x1234, 32, 0), 0x1234)
        self.assertEqual(oracle_v_lshl_or_b32(0x1234, 64, 0), 0x1234)
        # S1 value 33 masks to shift count 1
        self.assertEqual(oracle_v_lshl_or_b32(0x0001, 33, 0), 0x0002)
        # S1 value 0xFFFFFFFF masks to 31
        self.assertEqual(oracle_v_lshl_or_b32(0x0001, 0xFFFFFFFF, 0), 0x80000000)

    def test_high_bits_discarded_on_shift(self) -> None:
        # (0x80000000 << 1) | 0x00000005 => 0x00000005
        self.assertEqual(oracle_v_lshl_or_b32(0x80000000, 1, 0x00000005), 0x00000005)

    def test_overlapping_bits_bitwise_or(self) -> None:
        # (0x0F << 4) | 0xFF = 0x0F0 | 0x0FF = 0x0FF
        res = oracle_v_lshl_or_b32(0x0F, 4, 0xFF)
        self.assertEqual(res, 0xFF)

    def test_pack_halves_via_shift_and_or(self) -> None:
        # (0x1234 << 16) | 0x5678 = 0x12345678
        res = oracle_v_lshl_or_b32(0x1234, 16, 0x5678)
        self.assertEqual(res, 0x12345678)


if __name__ == "__main__":
    unittest.main()
