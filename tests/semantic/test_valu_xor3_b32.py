from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_xor3_b32.

AMD RDNA VALU Instruction Set Architecture (ISA):
    v_xor3_b32 D, S0, S1, S2
Operation:
    Performs a 3-way bitwise exclusive OR across three 32-bit operands:
        dst = (S0 ^ S1 ^ S2) & 0xFFFFFFFF

Modifiers and Semantics:
    - S0, S1, S2: 32-bit unsigned integer values.
    - Full bitwise XOR associativity and commutativity.
    - Zero identity: S0 ^ 0 ^ 0 == S0.
    - Self-inversion / cancellation: S0 ^ S0 ^ S2 == S2.
    - Inversion via all-ones: S0 ^ 0xFFFFFFFF ^ 0 == ~S0 & 0xFFFFFFFF.
    - Bit parity calculation: computes odd bit parity across three inputs per bit position.

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


def oracle_v_xor3_b32(src0_bits: int, src1_bits: int, src2_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 v_xor3_b32."""
    s0 = src0_bits & MASK32
    s1 = src1_bits & MASK32
    s2 = src2_bits & MASK32
    return (s0 ^ s1 ^ s2) & MASK32


class TestValuXor3B32ModelAndOracle(unittest.TestCase):
    """Architectural model and reference mathematical oracle tests for v_xor3_b32."""

    def test_zero_identity(self) -> None:
        # S0 ^ 0 ^ 0 == S0
        res = oracle_v_xor3_b32(0x12345678, 0, 0)
        self.assertEqual(res, 0x12345678)

    def test_self_cancellation(self) -> None:
        # S ^ S ^ Other == Other
        val = 0xA5A5A5A5
        other = 0x5A5A5A5A
        self.assertEqual(oracle_v_xor3_b32(val, val, other), other)

    def test_triple_identity(self) -> None:
        # S ^ S ^ S == S
        val = 0x12345678
        self.assertEqual(oracle_v_xor3_b32(val, val, val), val)

    def test_all_ones_inversion(self) -> None:
        # S ^ 0xFFFFFFFF ^ 0 == ~S
        val = 0x00FF00FF
        self.assertEqual(oracle_v_xor3_b32(val, 0xFFFFFFFF, 0), 0xFF00FF00)

    def test_commutativity_and_associativity(self) -> None:
        a, b, c = 0x12345678, 0x9ABCDEF0, 0x5555AAAA
        res1 = oracle_v_xor3_b32(a, b, c)
        res2 = oracle_v_xor3_b32(c, a, b)
        res3 = oracle_v_xor3_b32(b, c, a)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)


if __name__ == "__main__":
    unittest.main()
