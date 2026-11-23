"""Semantic interpreter tests and translator verification for 16-bit arithmetic right shift with reversed operands (v_ashrrev_i16).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    py -I tests/semantic/test_valu_ashrrev_i16_hw.py
    py -m unittest discover -s tests/semantic -p test_valu_ashrrev_i16_hw.py
"""
from __future__ import annotations

import os
import re
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import support
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFFFFFF
MASK16: int = 0xFFFF
SIGN_BIT16: int = 0x8000
OLD_VDST: int = 0x5A5A1234


def s16(word: int, hi: int = 0) -> int:
    """Extract 16-bit signed integer in [-32768, 32767]."""
    h = (word >> 16 if hi else word) & MASK16
    return h - 0x10000 if (h & SIGN_BIT16) else h


def u16(word: int, hi: int = 0) -> int:
    """Extract 16-bit unsigned integer in [0, 65535]."""
    return (word >> 16 if hi else word) & MASK16


def ref_v_ashrrev_i16(
    shift_src0: int,
    val_src1: int,
    *,
    old_dst: int = OLD_VDST,
    e64: bool = False,
    op_sel: tuple[int, int, int] = (0, 0, 0),
) -> int:
    """Reference mathematical oracle for RDNA2 v_ashrrev_i16.

    Operand order is reversed:
      shift amount is src0 (low 4 bits, [3:0], range 0..15).
      value to be shifted is src1 (interpreted as signed 16-bit integer).
    VOP2 (_e32) writes to low 16 bits and zeroes high 16 bits.
    VOP3 (_e64) preserves untouched half according to op_sel[2].
    """
    sh = u16(shift_src0, op_sel[0]) & 0xF
    val = s16(val_src1, op_sel[1])
    res = (val >> sh) & MASK16

    if e64:
        if op_sel[2]:
            return (old_dst & 0xFFFF) | (res << 16)
        else:
            return (old_dst & 0xFFFF0000) | res
    return res

class TestVAshrrevI16OracleAndArchitecturalModel(unittest.TestCase):
    """Part 1: Reference mathematical oracle and architectural model tests."""

    def test_signed_16bit_extraction(self) -> None:
        self.assertEqual(s16(0x0000), 0)
        self.assertEqual(s16(0x0001), 1)
        self.assertEqual(s16(0x7FFF), 32767)
        self.assertEqual(s16(0x8000), -32768)
        self.assertEqual(s16(0xFFFF), -1)
        self.assertEqual(s16(0xFFF0), -16)

        # High half extraction
        self.assertEqual(s16(0xFFFF0000, hi=1), -1)
        self.assertEqual(s16(0x00200000, hi=1), 32)

    def test_reference_oracle_positive_values(self) -> None:
        # Shift 16 by 2 -> 4
        self.assertEqual(ref_v_ashrrev_i16(2, 16), 4)
        # Shift 1000 by 3 -> 125
        self.assertEqual(ref_v_ashrrev_i16(3, 1000), 125)
        # Shift by 0 -> unchanged
        self.assertEqual(ref_v_ashrrev_i16(0, 0x1234), 0x1234)

    def test_reference_oracle_negative_sign_preservation(self) -> None:
        # -16 (0xFFF0) >> 2 -> -4 (0xFFFC)
        self.assertEqual(ref_v_ashrrev_i16(2, 0xFFF0), 0xFFFC)
        # -1 (0xFFFF) >> any shift (0..15) -> -1 (0xFFFF)
        for sh in range(16):
            self.assertEqual(ref_v_ashrrev_i16(sh, 0xFFFF), 0xFFFF)
        # -32768 (0x8000) >> 1 -> -16384 (0xC000)
        self.assertEqual(ref_v_ashrrev_i16(1, 0x8000), 0xC000)
        # -32768 (0x8000) >> 15 -> -1 (0xFFFF)
        self.assertEqual(ref_v_ashrrev_i16(15, 0x8000), 0xFFFF)

    def test_reference_oracle_shift_masking_to_4_bits(self) -> None:
        # Shift amount is masked with 15 (& 0xF): 18 & 0xF = 2, so shifts by 2
        self.assertEqual(ref_v_ashrrev_i16(18, 16), 4)
        self.assertEqual(ref_v_ashrrev_i16(0xFFF2, 16), 4)

    def test_reference_oracle_vop3_op_sel_and_preservation(self) -> None:
        old = 0x12345678
        # Write to low half (op_sel[2]=0): preserves high half (0x1234)
        res_lo = ref_v_ashrrev_i16(1, 0x0020, old_dst=old, e64=True, op_sel=(0, 0, 0))
        self.assertEqual(res_lo, 0x12340010)

        # Write to high half (op_sel[2]=1): preserves low half (0x5678)
        res_hi = ref_v_ashrrev_i16(1, 0x0020, old_dst=old, e64=True, op_sel=(0, 0, 1))
        self.assertEqual(res_hi, 0x00105678)


if __name__ == '__main__':
    unittest.main()
