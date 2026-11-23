"""Semantic reference oracle and regression tests for RDNA2 v_dot4_i32_i8.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_dot4_i32_i8 is a VOP3P instruction computing
  a four-component dot product of 8-bit signed integers with a 32-bit integer addend:
      D.i32 = (S0.b0 * S1.b0) + (S0.b1 * S1.b1) + (S0.b2 * S1.b2) + (S0.b3 * S1.b3) + S2.i32   (mod 2^32)

  - S0 and S1 each hold four packed 8-bit signed integer components (range -128..127).
  - Each byte is sign-extended to 32 bits before multiplication.
  - S2 is a 32-bit signed integer addend/accumulator.
  - The sum of products lies in [-65024, 65536]. Added to S2, it wraps modulo 2^32.
  - With optional clamp modifier, the result saturates to signed 32-bit range [-2^31, 2^31-1].
  - Condition codes: Neither SCC nor VCC is modified.
  - Floating-point representations (NaN, Inf, denormals) are treated strictly as raw bit patterns;
    no flushing or quietening occurs.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import support  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

MASK32: int = 0xFFFFFFFF
INT32_MIN: int = -0x80000000
INT32_MAX: int = 0x7FFFFFFF


def u32(val: int) -> int:
    return val & MASK32


def s32(val: int) -> int:
    v = val & MASK32
    return v - 0x100000000 if v & 0x80000000 else v


def s8(val: int) -> int:
    v = val & 0xFF
    return v - 256 if v & 0x80 else v


def pack_i8(b0: int, b1: int, b2: int, b3: int) -> int:
    return ((b0 & 0xFF) |
            ((b1 & 0xFF) << 8) |
            ((b2 & 0xFF) << 16) |
            ((b3 & 0xFF) << 24)) & MASK32


def ref_dot4_i32_i8(s0: int, s1: int, s2: int, clamp: bool = False) -> int:
    """RDNA2 v_dot4_i32_i8 reference mathematical oracle."""
    b0_0, b0_1, b0_2, b0_3 = (s8(s0 >> (8 * i)) for i in range(4))
    b1_0, b1_1, b1_2, b1_3 = (s8(s1 >> (8 * i)) for i in range(4))
    dot = (b0_0 * b1_0) + (b0_1 * b1_1) + (b0_2 * b1_2) + (b0_3 * b1_3)
    acc = s32(s2)
    total = acc + dot
    if clamp:
        total = max(INT32_MIN, min(INT32_MAX, total))
    return total & MASK32

class TestValuDot4I32I8Oracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle unit tests."""

    def test_oracle_basic_products(self) -> None:
        # 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        s0 = pack_i8(1, 2, 3, 4)
        s1 = pack_i8(5, 6, 7, 8)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 70)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 1000), 1070)

    def test_oracle_commutativity_of_multiplicands(self) -> None:
        s0 = pack_i8(10, -20, 30, -40)
        s1 = pack_i8(-2, 3, -4, 5)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 123), ref_dot4_i32_i8(s1, s0, 123))

    def test_oracle_signed_byte_sign_extension(self) -> None:
        # 0xFF is -1, 0x80 is -128, 0x7F is 127
        s0 = pack_i8(0xFF, 0x80, 0x7F, 0x01)  # -1, -128, 127, 1
        s1 = pack_i8(0x01, 0x01, 0x01, 0x01)  # 1, 1, 1, 1
        # sum = (-1)*1 + (-128)*1 + 127*1 + 1*1 = -1 - 128 + 127 + 1 = -1
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 0xFFFFFFFF)

    def test_oracle_extreme_positive_dot(self) -> None:
        # Max positive product per byte: -128 * -128 = 16384. Four bytes = 65536.
        s0 = pack_i8(0x80, 0x80, 0x80, 0x80)
        s1 = pack_i8(0x80, 0x80, 0x80, 0x80)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 65536)

    def test_oracle_extreme_negative_dot(self) -> None:
        # Max negative product per byte: -128 * 127 = -16256. Four bytes = -65024.
        s0 = pack_i8(0x80, 0x80, 0x80, 0x80)
        s1 = pack_i8(0x7F, 0x7F, 0x7F, 0x7F)
        # -65024 in two's complement 32-bit: 0xFFFF0200
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 0xFFFF0200)

    def test_oracle_overflow_wrapping_without_clamp(self) -> None:
        # INT32_MAX + 65536 wraps around into negative
        s0 = pack_i8(0x80, 0x80, 0x80, 0x80)
        s1 = pack_i8(0x80, 0x80, 0x80, 0x80)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, INT32_MAX, clamp=False), 0x8000FFFF)

    def test_oracle_clamp_saturation(self) -> None:
        s0_pos = pack_i8(0x80, 0x80, 0x80, 0x80)
        s1_pos = pack_i8(0x80, 0x80, 0x80, 0x80)
        # Overflow saturates to INT32_MAX
        self.assertEqual(ref_dot4_i32_i8(s0_pos, s1_pos, INT32_MAX, clamp=True), INT32_MAX)

        s0_neg = pack_i8(0x80, 0x80, 0x80, 0x80)
        s1_neg = pack_i8(0x7F, 0x7F, 0x7F, 0x7F)
        # Underflow saturates to INT32_MIN
        self.assertEqual(ref_dot4_i32_i8(s0_neg, s1_neg, 0x80000000, clamp=True), 0x80000000)

    def test_oracle_zero_multiplicands(self) -> None:
        self.assertEqual(ref_dot4_i32_i8(0, 0x12345678, 0xDEADBEEF), 0xDEADBEEF)
        self.assertEqual(ref_dot4_i32_i8(0x12345678, 0, 0xCAFEBABE), 0xCAFEBABE)

if __name__ == "__main__":
    unittest.main()
