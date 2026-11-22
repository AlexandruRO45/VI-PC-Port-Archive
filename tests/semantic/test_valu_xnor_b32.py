from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_xnor_b32.

Tests the vector bitwise XNOR (equivalence) instruction:
    v_xnor_b32 D, S0, S1
Lowered semantics:
    dst = ~(S0 ^ S1) & 0xFFFFFFFF
Mathematical and architectural properties:
    - Bitwise equivalence: Evaluates bitwise equality between S0 and S1.
      Bit k of destination is 1 if S0[k] == S1[k], and 0 if S0[k] != S1[k].
    - Commutativity: S0 and S1 are interchangeable: v_xnor_b32(a, b) == v_xnor_b32(b, a).
    - Algebraic identities:
        - Self-equivalence: v_xnor_b32(a, a) == 0xFFFFFFFF (all bits match).
        - Complement-equivalence: v_xnor_b32(a, ~a) == 0x00000000 (no bits match).
        - Identity with 0xFFFFFFFF: v_xnor_b32(a, 0xFFFFFFFF) == a (pass-through).
        - Inversion with 0x00000000: v_xnor_b32(a, 0) == (~a) & 0xFFFFFFFF (bitwise NOT).
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
SIGN32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret unsigned 32-bit dword as signed two's complement integer."""
    val &= MASK32
    return val - 0x1_0000_0000 if (val & SIGN32) else val


def ref_xnor_b32(s0: int, s1: int) -> int:
    """Reference mathematical computation for v_xnor_b32: ~(s0 ^ s1) & 0xFFFFFFFF."""
    u0 = s0 & MASK32
    u1 = s1 & MASK32
    return (~(u0 ^ u1)) & MASK32


def oracle_v_xnor_b32(src0: int, src1: int) -> int:
    """Reference architectural oracle for RDNA2 v_xnor_b32 instruction."""
    return ref_xnor_b32(src0, src1)


class TestXnorB32ModelAndOracle(unittest.TestCase):
    """Architectural model and mathematical reference oracle tests for v_xnor_b32."""

    def test_basic_xnor(self) -> None:
        # Alternating bits
        self.assertEqual(oracle_v_xnor_b32(0xAAAA_AAAA, 0xAAAA_AAAA), 0xFFFF_FFFF)
        self.assertEqual(oracle_v_xnor_b32(0xAAAA_AAAA, 0x5555_5555), 0x0000_0000)

    def test_identity_with_all_ones(self) -> None:
        # ~(a ^ 0xFFFFFFFF) == ~(~a) == a
        values = [0x0000_0000, 0x1234_5678, 0x8000_0000, 0xFFFF_FFFF]
        for v in values:
            self.assertEqual(oracle_v_xnor_b32(v, 0xFFFF_FFFF), v)
            self.assertEqual(oracle_v_xnor_b32(0xFFFF_FFFF, v), v)

    def test_inversion_with_zero(self) -> None:
        # ~(a ^ 0) == ~a
        self.assertEqual(oracle_v_xnor_b32(0x0000_0000, 0), 0xFFFF_FFFF)
        self.assertEqual(oracle_v_xnor_b32(0xFFFF_FFFF, 0), 0x0000_0000)
        self.assertEqual(oracle_v_xnor_b32(0x1234_5678, 0), (~0x1234_5678) & MASK32)

    def test_self_equivalence(self) -> None:
        patterns = [0x0000_0000, 0xCAFE_BABE, 0xDEAD_BEEF, 0xFFFF_FFFF]
        for p in patterns:
            self.assertEqual(oracle_v_xnor_b32(p, p), 0xFFFF_FFFF)

    def test_complement_equivalence(self) -> None:
        patterns = [0x0000_0000, 0x1234_5678, 0xAAAA_5555, 0xFFFF_FFFF]
        for p in patterns:
            comp = (~p) & MASK32
            self.assertEqual(oracle_v_xnor_b32(p, comp), 0x0000_0000)

    def test_bit_level_truth_table(self) -> None:
        # 0 XNOR 0 = 1
        # 0 XNOR 1 = 0
        # 1 XNOR 0 = 0
        # 1 XNOR 1 = 1
        self.assertEqual(oracle_v_xnor_b32(0, 0) & 1, 1)
        self.assertEqual(oracle_v_xnor_b32(0, 1) & 1, 0)
        self.assertEqual(oracle_v_xnor_b32(1, 0) & 1, 0)
        self.assertEqual(oracle_v_xnor_b32(1, 1) & 1, 1)

    def test_commutativity(self) -> None:
        pairs = [
            (0x1122_3344, 0x5566_7788),
            (0xDEAD_BEEF, 0x0123_4567),
            (0x8000_0000, 0x7FFF_FFFF),
        ]
        for a, b in pairs:
            self.assertEqual(oracle_v_xnor_b32(a, b), oracle_v_xnor_b32(b, a))


if __name__ == "__main__":
    unittest.main()
