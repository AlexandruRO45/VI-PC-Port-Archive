from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_add_lshl_u32.

Tests the vector fused add and logical shift left instruction:
    v_add_lshl_u32 D, S0, S1, S2
Lowered semantics:
    sum32 = (S0 + S1) & 0xFFFFFFFF
    shift = S2 & 0x1F
    dst   = (sum32 << shift) & 0xFFFFFFFF
Mathematical and architectural properties:
    - Addition: 32-bit modular addition with wraparound mod 2^32.
    - Shift: Logical shift left by (S2 & 31); high bits shifted past bit 31 are discarded,
      and vacant low bits are zero-filled.
    - Commutativity: S0 and S1 are interchangeable: v_add_lshl_u32(a, b, c) == v_add_lshl_u32(b, a, c).
    - Shift operand masking: Bits [31:5] of S2 are ignored by hardware and translator.
Architectural model:
    32-bit vector registers (VGPRs) and scalar registers (SGPRs). Overwrites all 32 bits of destination VGPR.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Sequence, Tuple, Union

# Architectural bitwise masks and constants
MASK32: int = 0xFFFF_FFFF
SHIFT_MASK: int = 0x1F
SIGN32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret unsigned 32-bit dword as signed two's complement integer."""
    val &= MASK32
    return val - 0x1_0000_0000 if (val & SIGN32) else val


def ref_add_lshl_u32(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical computation for v_add_lshl_u32: ((s0 + s1) mod 2^32) << (s2 & 31)."""
    sum32 = (s0 + s1) & MASK32
    shift = s2 & SHIFT_MASK
    return (sum32 << shift) & MASK32


def oracle_v_add_lshl_u32(src0: int, src1: int, src2: int) -> int:
    """Reference architectural oracle for RDNA2 v_add_lshl_u32 instruction."""
    return ref_add_lshl_u32(src0, src1, src2)


class TestAddLshlU32ModelAndOracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_add_lshl_u32."""

    def test_basic_add_lshl(self) -> None:
        # (10 + 20) << 2 = 30 << 2 = 120
        res = oracle_v_add_lshl_u32(10, 20, 2)
        self.assertEqual(res, 120)

    def test_shift_zero_identity(self) -> None:
        # Shift count 0 produces plain modular addition
        self.assertEqual(oracle_v_add_lshl_u32(100, 250, 0), 350)
        self.assertEqual(oracle_v_add_lshl_u32(0xFFFFFFFF, 1, 0), 0)

    def test_shift_masking_upper_bits(self) -> None:
        # Shift amount is strictly masked by 0x1F (only lower 5 bits count)
        base = oracle_v_add_lshl_u32(5, 5, 3)
        self.assertEqual(base, 10 << 3)
        # S2 with upper bits set: 32 + 3 = 35 -> shift 3
        self.assertEqual(oracle_v_add_lshl_u32(5, 5, 35), base)
        # S2 = 0x100 -> 256 & 31 = 0 -> shift 0
        self.assertEqual(oracle_v_add_lshl_u32(5, 5, 256), 10)
        # S2 = 0xFFFFFFFF -> 0xFFFFFFFF & 31 = 31 -> shift 31
        self.assertEqual(oracle_v_add_lshl_u32(1, 0, 0xFFFFFFFF), (1 << 31) & MASK32)

    def test_addition_modulo_overflow(self) -> None:
        # Addition wraps mod 2^32 before the shift
        s0 = 0x8000_0000
        s1 = 0x8000_0000
        # (0x80000000 + 0x80000000) mod 2^32 = 0
        self.assertEqual(oracle_v_add_lshl_u32(s0, s1, 5), 0)

        # 0xFFFFFFFF + 2 = 1 (mod 2^32) -> 1 << 4 = 16
        self.assertEqual(oracle_v_add_lshl_u32(0xFFFF_FFFF, 2, 4), 16)

    def test_shift_truncation_overflow(self) -> None:
        # Bits shifted past 31 are discarded
        # 0x00010000 << 16 = 0 (1 << 32 is truncated)
        self.assertEqual(oracle_v_add_lshl_u32(0x0001_0000, 0, 16), 0)
        # 0x40000000 << 1 = 0x80000000, << 2 = 0
        self.assertEqual(oracle_v_add_lshl_u32(0x4000_0000, 0, 1), 0x8000_0000)
        self.assertEqual(oracle_v_add_lshl_u32(0x4000_0000, 0, 2), 0)

    def test_commutativity_src0_src1(self) -> None:
        # S0 and S1 addition is strictly commutative
        pairs = [
            (1234, 5678, 4),
            (0xFFFF_0000, 0x0000_FFFF, 7),
            (0xDEAD_BEEF, 0xCAFE_BABE, 12),
        ]
        for a, b, c in pairs:
            self.assertEqual(oracle_v_add_lshl_u32(a, b, c), oracle_v_add_lshl_u32(b, a, c))

    def test_identity_zero_operands(self) -> None:
        # s0 = 0, s1 = 0 -> 0
        self.assertEqual(oracle_v_add_lshl_u32(0, 0, 15), 0)
        # s1 = 0, s2 = 0 -> identity pass-through of s0
        val = 0x1234_5678
        self.assertEqual(oracle_v_add_lshl_u32(val, 0, 0), val)
        self.assertEqual(oracle_v_add_lshl_u32(0, val, 0), val)


if __name__ == "__main__":
    unittest.main()
