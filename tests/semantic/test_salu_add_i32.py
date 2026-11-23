from __future__ import annotations

"""Semantic reference oracle, AST definitions, interpreter models, and translator tests for s_add_i32.

AMD RDNA2 / GFX10 SOP2 (Scalar ALU) 32-bit Signed Integer Addition:
    s_add_i32 sdst, ssrc0, ssrc1

Hardware Semantics:
    sdst = (ssrc0 + ssrc1) & 0xFFFFFFFF
    SCC  = 1 if signed 32-bit integer overflow occurs, else 0

RDNA2 Condition Code (SCC) Behavior for s_add_i32:
    - ssrc0 and ssrc1 are interpreted as signed two's-complement 32-bit integers in [-2^31, 2^31 - 1].
    - Signed overflow occurs when:
        * both operands are positive (bit 31 == 0) and the sum is negative (bit 31 == 1)
        * both operands are negative (bit 31 == 1) and the sum is positive (bit 31 == 0)
    - Bitwise formulation: ((ssrc0 ^ sum) & (ssrc1 ^ sum)) >> 31 == 1
    - Note that unsigned wrap (carry-out) without signed overflow (such as (-1) + 1 = 0)
      clears SCC to 0 on s_add_i32, in direct contrast to s_add_u32 which sets SCC to 1.

Standard library Python only; zero third-party dependencies; zero game data.
"""

import dataclasses
import enum
import math
import os
import random
import struct
import sys
import unittest
from typing import List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MASK32: int = 0xFFFFFFFF
INT32_MIN: int = -0x80000000       # -2147483648
INT32_MAX: int = 0x7FFFFFFF        # 2147483647
UINT32_INT_MIN: int = 0x80000000   # 2147483648 as unsigned

def to_u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit word [0, 0xFFFFFFFF]."""
    return val & MASK32


def to_s32(val: int) -> int:
    """Interpret 32-bit unsigned word as signed two's-complement integer in [-2^31, 2^31 - 1]."""
    u = val & MASK32
    return u - (1 << 32) if (u >> 31) else u


def ref_s_add_i32(s0: int, s1: int) -> tuple[int, int]:
    """Pure mathematical reference oracle for RDNA2 s_add_i32.

    Returns (dst_u32, scc_overflow).
    """
    v0 = to_s32(s0)
    v1 = to_s32(s1)
    tot = v0 + v1
    dst = tot & MASK32
    scc = 1 if (tot < INT32_MIN or tot > INT32_MAX) else 0
    return dst, scc


def ref_s_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for RDNA2 s_add_u32 for direct contrast (carry out of bit 31)."""
    u0 = to_u32(s0)
    u1 = to_u32(s1)
    tot = u0 + u1
    dst = tot & MASK32
    scc = 1 if tot > MASK32 else 0
    return dst, scc


class TestSAddI32ReferenceOracle(unittest.TestCase):
    """Mathematical reference oracle tests for RDNA2 s_add_i32."""

    def test_identity_element(self) -> None:
        for x in (0, 1, -1, INT32_MAX, INT32_MIN, 0x12345678, 0xEDCBA987):
            with self.subTest(x=x):
                dst, scc = ref_s_add_i32(x, 0)
                self.assertEqual(dst, to_u32(x))
                self.assertEqual(scc, 0)
                dst_rev, scc_rev = ref_s_add_i32(0, x)
                self.assertEqual(dst_rev, to_u32(x))
                self.assertEqual(scc_rev, 0)

    def test_commutativity(self) -> None:
        pairs = [
            (0, 0),
            (100, 200),
            (INT32_MAX, 1),
            (INT32_MIN, -1),
            (0xFFFFFFFF, 1),
            (0x7FFFFFFF, 0x7FFFFFFF),
            (0x80000000, 0x80000000),
            (0x12345678, 0x87654321),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(to_u32(a)), b=hex(to_u32(b))):
                self.assertEqual(ref_s_add_i32(a, b), ref_s_add_i32(b, a))

    def test_positive_overflow(self) -> None:
        # INT32_MAX + 1 -> 0x80000000 (signed negative), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MAX, 1)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 1)

        # INT32_MAX + INT32_MAX -> 0xFFFFFFFE (-2), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MAX, INT32_MAX)
        self.assertEqual(dst, 0xFFFFFFFE)
        self.assertEqual(scc, 1)

        # 2^30 + 2^30 -> 2^31 (0x80000000), overflow = 1
        dst, scc = ref_s_add_i32(0x40000000, 0x40000000)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 1)

    def test_negative_overflow(self) -> None:
        # INT32_MIN + (-1) -> 0x7FFFFFFF (positive), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MIN, -1)
        self.assertEqual(dst, INT32_MAX)
        self.assertEqual(scc, 1)

        # INT32_MIN + INT32_MIN -> 0x00000000 (0), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MIN, INT32_MIN)
        self.assertEqual(dst, 0)
        self.assertEqual(scc, 1)

        # -2^30 + (-2^30 - 1) -> 0x7FFFFFFF, overflow = 1
        dst, scc = ref_s_add_i32(-0x40000000, -0x40000001)
        self.assertEqual(dst, INT32_MAX)
        self.assertEqual(scc, 1)

    def test_unsigned_wrap_without_signed_overflow(self) -> None:
        # (-1) + 1 = 0: unsigned carry-out occurs, but signed -1 + 1 = 0 (no signed overflow)
        dst_i, scc_i = ref_s_add_i32(0xFFFFFFFF, 1)
        self.assertEqual(dst_i, 0)
        self.assertEqual(scc_i, 0)

        # Contrast with s_add_u32 which sets SCC on unsigned carry:
        dst_u, scc_u = ref_s_add_u32(0xFFFFFFFF, 1)
        self.assertEqual(dst_u, 0)
        self.assertEqual(scc_u, 1)

        # (-1) + (-1) = -2: unsigned wrap occurs, signed -1 + -1 = -2 (no signed overflow)
        dst_i2, scc_i2 = ref_s_add_i32(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(dst_i2, 0xFFFFFFFE)
        self.assertEqual(scc_i2, 0)

        dst_u2, scc_u2 = ref_s_add_u32(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(dst_u2, 0xFFFFFFFE)
        self.assertEqual(scc_u2, 1)

    def test_opposite_signs_no_overflow(self) -> None:
        # Opposite signs can never overflow 32-bit signed range
        dst, scc = ref_s_add_i32(INT32_MIN, INT32_MAX)
        self.assertEqual(dst, 0xFFFFFFFF)
        self.assertEqual(scc, 0)

        dst, scc = ref_s_add_i32(0x12345678, to_u32(-0x12345678))
        self.assertEqual(dst, 0)
        self.assertEqual(scc, 0)

if __name__ == "__main__":
    unittest.main()
