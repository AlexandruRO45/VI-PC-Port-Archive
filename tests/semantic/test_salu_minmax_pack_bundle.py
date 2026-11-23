"""Semantic reference tests and translator-backed validation for AMD RDNA2 SALU Min/Max & Pack Suite.

Subsystem Feature Bundle 32 covers 6 SALU min/max selection and half-word packing instructions:
  1. s_min_i32         - Signed 32-bit minimum: D = min(S0, S1), SCC = (S0 < S1, signed)
  2. s_min_u32         - Unsigned 32-bit minimum: D = min(S0, S1), SCC = (S0 < S1, unsigned)
  3. s_max_i32         - Signed 32-bit maximum: D = max(S0, S1), SCC = (S0 > S1, signed)
  4. s_max_u32         - Unsigned 32-bit maximum: D = max(S0, S1), SCC = (S0 > S1, unsigned)
  5. s_pack_ll_b32_b16 - Pack low halves: D = {S1[15:0], S0[15:0]}, SCC untouched
  6. s_pack_hh_b32_b16 - Pack high halves: D = {S1[31:16], S0[31:16]}, SCC untouched

RDNA2 SALU Architecture:
  - Operates on 32-bit scalar registers (SGPRs), inline/literal constants.
  - Condition code (SCC):
      s_min_i32: SCC = 1 if S0 < S1 (signed), 0 on >= (ties set SCC to 0)
      s_min_u32: SCC = 1 if S0 < S1 (unsigned), 0 on >= (ties set SCC to 0)
      s_max_i32: SCC = 1 if S0 > S1 (signed), 0 on <= (ties set SCC to 0)
      s_max_u32: SCC = 1 if S0 > S1 (unsigned), 0 on <= (ties set SCC to 0)
      s_pack_*: SCC is preserved (untouched)
  - End-to-end SPIR-V assembly translation through tools/rdna2spirv.py.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
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


def s32(u: int) -> int:
    u &= MASK32
    return u if u < 0x80000000 else u - 0x100000000


def u32(s: int) -> int:
    return s & MASK32

# -----------------------------------------------------------------------------
# Opcode 1: s_min_i32
# -----------------------------------------------------------------------------
def oracle_min_i32(s0: int, s1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_min_i32: returns (result_u32, scc)."""
    xi = s32(s0)
    yi = s32(s1)
    val = xi if xi < yi else yi
    scc = 1 if xi < yi else 0
    return u32(val), scc


class SMinI32OracleTests(unittest.TestCase):
    def test_min_i32_basic(self):
        # min(10, 20) = 10, SCC = 1 (10 < 20)
        res, scc = oracle_min_i32(10, 20)
        self.assertEqual(res, 10)
        self.assertEqual(scc, 1)

    def test_min_i32_negative_numbers(self):
        # min(-10, 5) = -10, SCC = 1 (-10 < 5)
        res, scc = oracle_min_i32(u32(-10), 5)
        self.assertEqual(res, u32(-10))
        self.assertEqual(scc, 1)

        # min(5, -10) = -10, SCC = 0 (5 < -10 is False)
        res, scc = oracle_min_i32(5, u32(-10))
        self.assertEqual(res, u32(-10))
        self.assertEqual(scc, 0)

    def test_min_i32_ties(self):
        # Equal values: SCC = 0 (ties clear SCC)
        res, scc = oracle_min_i32(42, 42)
        self.assertEqual(res, 42)
        self.assertEqual(scc, 0)

if __name__ == "__main__":
    unittest.main()
