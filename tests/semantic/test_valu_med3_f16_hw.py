"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_med3_f16.

AMD RDNA2 / GFX10 Vector ALU (VALU) Architecture:
  - 16-bit Floating-Point 3-Operand Median:
    v_med3_f16 vdst, src0, src1, src2 [op_sel:[s0,s1,s2,d]] [clamp]
  - Operands:
    src0, src1, src2: 16-bit floating-point values selected from low (0) or high (1)
    16-bit half of the operand registers.
  - Mathematical Semantics:
    Without NaN:
      Computes median of 3 IEEE 754 half-precision values:
      NMin(NMax(z, NMin(x, y)), NMax(x, y))
    With any NaN operand:
      The hardware returns min3(x, y, z) with NaN-ignoring min (GLSL NMin),
      returning the minimum among all non-NaN operands, or NaN if all are NaN.
    With clamp modifier:
      Clamps the result to [0.0, 1.0] (0x0000 to 0x3C00).
    Destination:
      VOP3 opsel preserves the other 16-bit half of destination VGPR.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF

F16_POS_ZERO: int = 0x0000
F16_NEG_ZERO: int = 0x8000
F16_ONE: int = 0x3C00
F16_POS_INF: int = 0x7C00
F16_NEG_INF: int = 0xFC00
F16_QNAN: int = 0x7E00


def is_nan_f16(u16: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 half-precision NaN."""
    return (u16 & 0x7C00) == 0x7C00 and (u16 & 0x03FF) != 0


def f16_sort_key(u16: int) -> int:
    """Order f16 values with -inf < ... < -0.0 < +0.0 < ... < +inf."""
    sign = (u16 >> 15) & 1
    if sign:
        return -(u16 & 0x7FFF) - 1
    return u16 & 0x7FFF


def nmin_f16(a: int, b: int) -> int:
    """GLSL/SPIR-V NMin semantics for 16-bit float: ignores NaN unless both are NaN."""
    a_nan = is_nan_f16(a)
    b_nan = is_nan_f16(b)
    if a_nan and b_nan:
        return F16_QNAN
    if a_nan:
        return b
    if b_nan:
        return a
    return a if f16_sort_key(a) <= f16_sort_key(b) else b


def nmax_f16(a: int, b: int) -> int:
    """GLSL/SPIR-V NMax semantics for 16-bit float: ignores NaN unless both are NaN."""
    a_nan = is_nan_f16(a)
    b_nan = is_nan_f16(b)
    if a_nan and b_nan:
        return F16_QNAN
    if a_nan:
        return b
    if b_nan:
        return a
    return a if f16_sort_key(a) >= f16_sort_key(b) else b


def v_med3_f16_oracle(x: int, y: int, z: int, clamp: bool = False) -> int:
    """Mathematical reference oracle for RDNA2 v_med3_f16."""
    x &= MASK16
    y &= MASK16
    z &= MASK16
    if is_nan_f16(x) or is_nan_f16(y) or is_nan_f16(z):
        res = nmin_f16(nmin_f16(x, y), z)
    else:
        lo = nmin_f16(x, y)
        hi = nmax_f16(x, y)
        m1 = nmax_f16(z, lo)
        res = nmin_f16(m1, hi)
    if clamp:
        if not is_nan_f16(res):
            if f16_sort_key(res) < f16_sort_key(F16_POS_ZERO):
                res = F16_POS_ZERO
            elif f16_sort_key(res) > f16_sort_key(F16_ONE):
                res = F16_ONE
    return res & MASK16


class TestVMed3F16Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_standard_median(self) -> None:
        c1 = 0x3C00  # 1.0
        c2 = 0x4000  # 2.0
        c3 = 0x4200  # 3.0
        self.assertEqual(v_med3_f16_oracle(c1, c2, c3), c2)

    def test_oracle_permutations(self) -> None:
        c1 = 0x3C00  # 1.0
        c2 = 0x4000  # 2.0
        c3 = 0x4200  # 3.0
        for p in [
            (c1, c2, c3), (c1, c3, c2),
            (c2, c1, c3), (c2, c3, c1),
            (c3, c1, c2), (c3, c2, c1),
        ]:
            self.assertEqual(v_med3_f16_oracle(p[0], p[1], p[2]), c2)

    def test_oracle_signed_zeros(self) -> None:
        pos_zero = F16_POS_ZERO
        neg_zero = F16_NEG_ZERO
        one = F16_ONE
        # -0.0 < +0.0 < 1.0, median is +0.0
        self.assertEqual(v_med3_f16_oracle(pos_zero, neg_zero, one), pos_zero)

    def test_oracle_nan_operand_min3_fallback(self) -> None:
        nan = F16_QNAN
        c2 = 0x4000  # 2.0
        c3 = 0x4200  # 3.0
        self.assertEqual(v_med3_f16_oracle(nan, c2, c3), c2)
        self.assertEqual(v_med3_f16_oracle(nan, nan, c3), c3)
        self.assertTrue(is_nan_f16(v_med3_f16_oracle(nan, nan, nan)))

    def test_oracle_clamp_modifier(self) -> None:
        # 2.0, 3.0, 5.0 -> median 3.0, clamped to 1.0 (0x3C00)
        self.assertEqual(v_med3_f16_oracle(0x4000, 0x4200, 0x4500, clamp=True), F16_ONE)
        # -3.0, -2.0, -1.0 -> median -2.0, clamped to 0.0 (0x0000)
        self.assertEqual(v_med3_f16_oracle(0xC200, 0xC000, 0xBC00, clamp=True), F16_POS_ZERO)


if __name__ == "__main__":
    unittest.main()
