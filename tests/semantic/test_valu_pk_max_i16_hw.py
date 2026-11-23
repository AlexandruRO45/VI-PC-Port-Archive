from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 v_pk_max_i16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_pk_max_i16 performs packed 16-bit signed integer maximum.
  Instruction:
    v_pk_max_i16 vd, src0, src1  (VOP3P format)

  Operands:
    vd   = Destination 32-bit VGPR containing two 16-bit signed integers [lo, hi]
    src0 = Source 32-bit operand containing two 16-bit signed integers [lo, hi]
    src1 = Source 32-bit operand containing two 16-bit signed integers [lo, hi]

  Semantics:
    lo0 = s16(src0[15:0]), hi0 = s16(src0[31:16])
    lo1 = s16(src1[15:0]), hi1 = s16(src1[31:16])
    res_lo = max(lo0, lo1) & 0xFFFF
    res_hi = max(hi0, hi1) & 0xFFFF
    vd = (res_hi << 16) | res_lo

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
MASK32: int = 0xFFFFFFFF
SIGN_BIT16: int = 0x8000


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned integer as signed two's complement."""
    val &= MASK16
    return val - 0x10000 if (val & SIGN_BIT16) else val


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit integer range [0, 2^16 - 1]."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_pk_max_i16(src0: int, src1: int) -> int:
    """Exact architectural reference oracle for AMD RDNA2 v_pk_max_i16.

    Args:
        src0: 32-bit packed pair of two signed 16-bit integers
        src1: 32-bit packed pair of two signed 16-bit integers

    Returns:
        32-bit packed pair with element-wise signed maximum
    """
    lo0 = s16(src0 & MASK16)
    hi0 = s16((src0 >> 16) & MASK16)
    lo1 = s16(src1 & MASK16)
    hi1 = s16((src1 >> 16) & MASK16)

    res_lo = max(lo0, lo1) & MASK16
    res_hi = max(hi0, hi1) & MASK16
    return (res_hi << 16) | res_lo


class TestValuPkMaxI16Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of v_pk_max_i16."""

    def test_oracle_mixed_signs(self):
        # src0: lo = -5 (0xFFFB), hi = +5 (0x0005) -> 0x0005FFFB
        # src1: lo = +5 (0x0005), hi = -5 (0xFFFB) -> 0xFFFB0005
        # Expected: lo = max(-5, +5) = +5 (0x0005), hi = max(+5, -5) = +5 (0x0005) -> 0x00050005
        res = ref_pk_max_i16(0x0005FFFB, 0xFFFB0005)
        self.assertEqual(res, 0x00050005)

    def test_oracle_both_negative(self):
        # lo: max(-10, -20) = -10 (0xFFF6)
        # hi: max(-30, -5) = -5 (0xFFFB)
        s0 = ((-30 & MASK16) << 16) | (-10 & MASK16)
        s1 = ((-5 & MASK16) << 16) | (-20 & MASK16)
        res = ref_pk_max_i16(s0, s1)
        expected = ((-5 & MASK16) << 16) | (-10 & MASK16)
        self.assertEqual(res, expected)

    def test_oracle_both_positive(self):
        s0 = (100 << 16) | 300
        s1 = (200 << 16) | 150
        res = ref_pk_max_i16(s0, s1)
        expected = (200 << 16) | 300
        self.assertEqual(res, expected)

    def test_oracle_signed_extremes(self):
        # -32768 (0x8000) vs 32767 (0x7FFF)
        s0 = (0x8000 << 16) | 0x7FFF
        s1 = (0x7FFF << 16) | 0x8000
        res = ref_pk_max_i16(s0, s1)
        expected = (0x7FFF << 16) | 0x7FFF
        self.assertEqual(res, expected)

    def test_oracle_commutativity(self):
        a = 0x12345678
        b = 0x87654321
        self.assertEqual(ref_pk_max_i16(a, b), ref_pk_max_i16(b, a))
