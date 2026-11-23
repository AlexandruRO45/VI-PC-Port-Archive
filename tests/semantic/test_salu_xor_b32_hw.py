"""Semantic reference tests and translator-backed validation for AMD RDNA2 s_xor_b32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, s_xor_b32 performs a 32-bit bitwise logical XOR:
      D.u32 = S0.u32 ^ S1.u32
      SCC = (D.u32 != 0)

  - Format: SOP2 (encoding opcode 0x12).
  - Operates on 32-bit scalar registers s[d] from s[s0] and s[s1], or 32-bit inline/literal
    constants, or special registers (e.g. exec_lo, vcc_lo).
  - Bitwise XOR yields 1 where source bits differ, and 0 where source bits match.
  - Scalar Condition Code (SCC) is set to 1 if the 32-bit result is non-zero, and 0 if all 32 bits are zero.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF


def oracle_xor_b32(s0: int, s1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_xor_b32: returns (result_u32, scc)."""
    s0_32 = s0 & MASK32
    s1_32 = s1 & MASK32
    res = (s0_32 ^ s1_32) & MASK32
    scc = 1 if res != 0 else 0
    return res, scc

class SXorB32ReferenceOracleTests(unittest.TestCase):
    def test_oracle_identity(self):
        # x ^ x == 0, SCC == 0
        res, scc = oracle_xor_b32(0x12345678, 0x12345678)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_oracle_zero_identity(self):
        # x ^ 0 == x
        res, scc = oracle_xor_b32(0x87654321, 0)
        self.assertEqual(res, 0x87654321)
        self.assertEqual(scc, 1)

    def test_oracle_inversion(self):
        # x ^ ~x == 0xFFFFFFFF, SCC == 1
        x = 0x55AA55AA
        not_x = (~x) & MASK32
        res, scc = oracle_xor_b32(x, not_x)
        self.assertEqual(res, MASK32)
        self.assertEqual(scc, 1)

    def test_oracle_commutativity(self):
        a, b = 0xF0F00F0F, 0x0F0FAFAF
        self.assertEqual(oracle_xor_b32(a, b), oracle_xor_b32(b, a))

    def test_oracle_zeros(self):
        self.assertEqual(oracle_xor_b32(0, 0), (0, 0))
        self.assertEqual(oracle_xor_b32(MASK32, MASK32), (0, 0))

if __name__ == "__main__":
    unittest.main()
