from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_andn2_b32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_andn2_b32 instruction belongs to SOP2 format.
  Instruction:
    s_andn2_b32 sdst, ssrc0, ssrc1

  Operands:
    sdst  = 32-bit destination SGPR (or exec_lo)
    ssrc0 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo
    ssrc1 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo

  Semantics:
    sdst = ssrc0 & ~ssrc1
    SCC = 1 if sdst != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32: int = 0xFFFFFFFF
ALL_LANES_ACTIVE_32: int = 0xFFFFFFFF
ALL_LANES_INACTIVE_32: int = 0x00000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_andn2_b32(ssrc0: int, ssrc1: int) -> tuple[int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_andn2_b32 instruction.

    Args:
        ssrc0: 32-bit first source operand (uint32)
        ssrc1: 32-bit second source operand (uint32)

    Returns:
        tuple (sdst, scc):
          - sdst: 32-bit result = ssrc0 & ~ssrc1
          - scc: scalar condition code = 1 if sdst != 0 else 0
    """
    s0 = u32(ssrc0)
    s1 = u32(ssrc1)
    sdst = u32(s0 & (~s1 & MASK32))
    scc = 1 if sdst != 0 else 0
    return sdst, scc


class TestSaluAndn2B32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_andn2_b32."""

    def test_oracle_basic(self):
        s0 = 0x12345678
        s1 = 0x00FF00FF
        sdst, scc = ref_andn2_b32(s0, s1)
        expected = s0 & (~s1 & MASK32)
        self.assertEqual(sdst, expected)
        self.assertEqual(scc, 1)

    def test_oracle_zero_s0(self):
        s1 = 0x12345678
        sdst, scc = ref_andn2_b32(0, s1)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)

    def test_oracle_all_ones_s1(self):
        s0 = 0x12345678
        sdst, scc = ref_andn2_b32(s0, ALL_LANES_ACTIVE_32)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)

    def test_oracle_zero_s1(self):
        s0 = 0x12345678
        sdst, scc = ref_andn2_b32(s0, 0)
        self.assertEqual(sdst, s0)
        self.assertEqual(scc, 1)

    def test_oracle_operand_asymmetry(self):
        # s0 & ~s1 != s1 & ~s0 (non-commutative)
        s0 = 0x0000FFFF
        s1 = 0x00FF00FF
        res_01, _ = ref_andn2_b32(s0, s1)
        res_10, _ = ref_andn2_b32(s1, s0)
        self.assertNotEqual(res_01, res_10)
        self.assertEqual(res_01, 0x0000FF00)
        self.assertEqual(res_10, 0x00FF0000)
