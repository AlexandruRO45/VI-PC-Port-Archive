from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 v_lshlrev_b16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_lshlrev_b16 performs a 16-bit logical left shift with reversed operands.
  Instruction:
    v_lshlrev_b16 vd, src0, src1

  Operands:
    vd   = Destination VGPR (low 16 bits written, high 16 bits zeroed in VOP2 / e32)
    src0 = Shift amount (low 4 bits used, masked by & 0xF)
    src1 = 16-bit value to be shifted (low 16 bits used)

  Semantics:
    shift = src0 & 0xF
    val   = src1 & 0xFFFF
    vd    = (val << shift) & 0xFFFF

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

MASK16: int = 0xFFFF
SHIFT_MASK: int = 0xF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit integer range [0, 2^16 - 1]."""
    return val & MASK16


def ref_lshlrev_b16(src0: int, src1: int) -> int:
    """Exact architectural reference oracle for AMD RDNA2 v_lshlrev_b16.

    Args:
        src0: Shift amount (lower 4 bits used)
        src1: 16-bit value to shift

    Returns:
        16-bit shifted result = (src1 << (src0 & 0xF)) & 0xFFFF
    """
    shift = src0 & SHIFT_MASK
    val = u16(src1)
    return (val << shift) & MASK16


class TestValuLshlrevB16Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of v_lshlrev_b16."""

    def test_oracle_basic_shift(self):
        # 0x12 << 3 = 0x90
        res = ref_lshlrev_b16(3, 0x12)
        self.assertEqual(res, 0x90)

    def test_oracle_zero_shift(self):
        # 0xABCD << 0 = 0xABCD
        res = ref_lshlrev_b16(0, 0xABCD)
        self.assertEqual(res, 0xABCD)

    def test_oracle_max_shift_15(self):
        # 1 << 15 = 0x8000
        res = ref_lshlrev_b16(15, 1)
        self.assertEqual(res, 0x8000)

    def test_oracle_shift_count_masking(self):
        # Shift counts are masked by low 4 bits (& 0xF): 19 & 0xF = 3
        res = ref_lshlrev_b16(19, 0x12)
        self.assertEqual(res, 0x90)

    def test_oracle_overflow_truncation(self):
        # Shifting past bit 15 discards upper bits: 0x8000 << 1 = 0
        res = ref_lshlrev_b16(1, 0x8000)
        self.assertEqual(res, 0)

    def test_oracle_operand_asymmetry(self):
        # Reversed operands: src0 is shift count, src1 is value
        # 3 << 1 != 1 << 3
        res0 = ref_lshlrev_b16(1, 3)   # 3 << 1 = 6
        res1 = ref_lshlrev_b16(3, 1)   # 1 << 3 = 8
        self.assertNotEqual(res0, res1)
        self.assertEqual(res0, 6)
        self.assertEqual(res1, 8)
