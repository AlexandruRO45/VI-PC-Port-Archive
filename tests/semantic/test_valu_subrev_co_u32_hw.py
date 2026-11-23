"""Semantic tests for RDNA2 v_subrev_co_u32 (vector reverse subtract with carry out).

RDNA2 VALU architecture contract:
- Opcode: v_subrev_co_u32 vdst, sdst, src0, src1 [clamp]
- Operation: D = src1 - src0 (unsigned 32-bit integer subtraction).
- Reverse operand order: computes src1 minus src0, contrasting with v_sub_co_u32 (src0 minus src1).
- Borrow out: borrow occurs if src1 < src0 (unsigned), writing 1 to sdst mask for active lanes.
- VOP3 clamp modifier: hardware saturates D to 0 upon borrow (underflow) while still reporting
  the real borrow bit in sdst. If no borrow occurs, D receives (src1 - src0) & 0xFFFFFFFF.
- EXEC masking: inactive lanes preserve their destination VGPR and contribute 0 to the sdst mask.
- Condition codes: VALU operations do not modify SCC.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------
M32 = 0xFFFFFFFF


def ref_subrev_co_u32(src0: int, src1: int, clamp: bool = False) -> tuple[int, int]:
    """Pure mathematical oracle matching RDNA2 hardware semantics for v_subrev_co_u32.

    Returns (dst, borrow) where:
    - dst is (src1 - src0) & 0xFFFFFFFF (or 0 if clamp and borrow occurred)
    - borrow is 1 if (src1 & 0xFFFFFFFF) < (src0 & 0xFFFFFFFF) else 0
    """
    s0 = src0 & M32
    s1 = src1 & M32
    diff = (s1 - s0) & M32
    borrow = 1 if s1 < s0 else 0
    dst = 0 if (clamp and borrow) else diff
    return dst, borrow


class TestSubrevCoU32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        dst, borrow = ref_subrev_co_u32(10, 25)
        self.assertEqual(dst, 15)
        self.assertEqual(borrow, 0)

    def test_borrow_underflow(self) -> None:
        dst, borrow = ref_subrev_co_u32(25, 10)
        self.assertEqual(dst, 0xFFFFFFF1)
        self.assertEqual(borrow, 1)

    def test_zero_identity(self) -> None:
        dst, borrow = ref_subrev_co_u32(0, 0)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 0)

        dst, borrow = ref_subrev_co_u32(12345, 12345)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 0)

    def test_boundary_opposites(self) -> None:
        # src1 = 0, src0 = 0xFFFFFFFF -> 0 - 0xFFFFFFFF = 1 with borrow
        dst, borrow = ref_subrev_co_u32(0xFFFFFFFF, 0)
        self.assertEqual(dst, 1)
        self.assertEqual(borrow, 1)

        # src1 = 0xFFFFFFFF, src0 = 0 -> 0xFFFFFFFF - 0 = 0xFFFFFFFF without borrow
        dst, borrow = ref_subrev_co_u32(0, 0xFFFFFFFF)
        self.assertEqual(dst, 0xFFFFFFFF)
        self.assertEqual(borrow, 0)

    def test_clamp_saturation(self) -> None:
        # Underflow with clamp saturates dst to 0, borrow remains 1
        dst, borrow = ref_subrev_co_u32(50, 20, clamp=True)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 1)

        # Normal subtraction with clamp remains diff, borrow is 0
        dst, borrow = ref_subrev_co_u32(20, 50, clamp=True)
        self.assertEqual(dst, 30)
        self.assertEqual(borrow, 0)

    def test_subtraction_reversibility_asymmetry(self) -> None:
        # v_subrev_co_u32 computes src1 - src0, which is anti-symmetric to src0 - src1
        d1, b1 = ref_subrev_co_u32(100, 40)
        self.assertEqual(d1, (40 - 100) & M32)
        self.assertEqual(b1, 1)
