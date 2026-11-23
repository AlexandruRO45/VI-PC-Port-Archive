from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_frexp_exp_i32_f32.

Instruction definition:
    v_frexp_exp_i32_f32 vdst, vsrc0

Semantics:
    Extracts the binary exponent of a single-precision floating-point value S0,
    such that S0 = mantissa * 2^exponent with mantissa in [0.5, 1.0).
    The exponent is returned as a 32-bit signed two's complement integer in destination VGPR.
    For normal binary32 numbers (biased exponent e in 1..254):
        exp = e - 126
    For zero (0.0, -0.0), denormals, infinities (+inf, -inf), and NaNs:
        exp = 0

Zero game data, standard library Python only (python -I, struct, math, unittest).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000
EXP_MASK32: int = 0x00FF
EXP_SHIFT32: int = 23
MANT_MASK32: int = 0x007F_FFFF
EXP_BIAS_FREXP: int = 126

POS_ZERO: int = 0x0000_0000
NEG_ZERO: int = 0x8000_0000
POS_INF: int = 0x7F80_0000
NEG_INF: int = 0xFF80_0000
QNAN: int = 0x7FC0_0000
SNAN: int = 0x7F80_0001
MIN_NORMAL: int = 0x0080_0000       # 2^-126
MAX_NORMAL: int = 0x7F7F_FFFF       # approx 3.4028235e+38


# ==============================================================================
# Reference Mathematical Oracle
# ==============================================================================

def to_i32(val: int) -> int:
    """Interpret 32-bit integer as two's complement signed integer."""
    val32 = val & MASK32
    if val32 & SIGN_BIT32:
        return val32 - 0x1_0000_0000
    return val32


def f32_to_bits(f: float) -> int:
    """Pack Python float to 32-bit IEEE 754 bit pattern."""
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(u: int) -> float:
    """Unpack 32-bit integer bit pattern to Python float."""
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def is_normal_f32(u: int) -> bool:
    """Check if bit pattern represents a normal IEEE 754 single-precision float."""
    e = (u >> EXP_SHIFT32) & EXP_MASK32
    return 1 <= e <= 254


def ref_frexp_exp_i32_f32(src: int) -> int:
    """Architectural reference oracle for v_frexp_exp_i32_f32.
    Computes 32-bit signed exponent in [0.5, 1.0) representation."""
    u = src & MASK32
    e = (u >> EXP_SHIFT32) & EXP_MASK32
    if 1 <= e <= 254:
        exp = e - EXP_BIAS_FREXP
        return exp & MASK32
    return 0


@dataclass
class VGPRWord:
    raw: int = 0

    @property
    def u32(self) -> int:
        return self.raw & MASK32

    @property
    def i32(self) -> int:
        return to_i32(self.raw)

    @property
    def f32(self) -> float:
        return bits_to_f32(self.raw)


class TestFrexpExpI32F32Oracle(unittest.TestCase):
    def test_powers_of_two(self):
        # 1.0f = 0.5 * 2^1 -> exp = 1
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(1.0)), 1)
        # 0.5f = 0.5 * 2^0 -> exp = 0
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(0.5)), 0)
        # 2.0f = 0.5 * 2^2 -> exp = 2
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(2.0)), 2)
        # 4.0f = 0.5 * 2^3 -> exp = 3
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(4.0)), 3)
        # 0.25f = 0.5 * 2^-1 -> exp = -1 (0xFFFFFFFF)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(0.25)), 0xFFFF_FFFF)
        # 0.125f = 0.5 * 2^-2 -> exp = -2 (0xFFFFFFFE)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(0.125)), 0xFFFF_FFFE)

    def test_sign_invariance(self):
        # Negative floats have identical exponent to positive counterparts
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(-1.0)), 1)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(-0.5)), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(-2.0)), 2)
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(-0.25)), 0xFFFF_FFFF)

    def test_arbitrary_normals(self):
        # 1.5 = 0.75 * 2^1 -> exp = 1
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(1.5)), 1)
        # 3.0 = 0.75 * 2^2 -> exp = 2
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(3.0)), 2)
        # 100.0 = 0.78125 * 2^7 -> exp = 7
        self.assertEqual(ref_frexp_exp_i32_f32(f32_to_bits(100.0)), 7)

    def test_boundary_normals(self):
        # Minimum positive normal: e = 1 -> exp = 1 - 126 = -125 (0xFFFFFF83)
        self.assertEqual(ref_frexp_exp_i32_f32(MIN_NORMAL), 0xFFFF_FF83)
        self.assertEqual(to_i32(ref_frexp_exp_i32_f32(MIN_NORMAL)), -125)
        # Maximum positive normal: e = 254 -> exp = 254 - 126 = 128 (0x00000080)
        self.assertEqual(ref_frexp_exp_i32_f32(MAX_NORMAL), 128)
        # Negative boundary normals
        self.assertEqual(ref_frexp_exp_i32_f32(MIN_NORMAL | SIGN_BIT32), 0xFFFF_FF83)
        self.assertEqual(ref_frexp_exp_i32_f32(MAX_NORMAL | SIGN_BIT32), 128)

    def test_special_values_return_zero(self):
        # Zeros
        self.assertEqual(ref_frexp_exp_i32_f32(POS_ZERO), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(NEG_ZERO), 0)
        # Infinities
        self.assertEqual(ref_frexp_exp_i32_f32(POS_INF), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(NEG_INF), 0)
        # NaNs
        self.assertEqual(ref_frexp_exp_i32_f32(QNAN), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(SNAN), 0)
        # Subnormals (denormals)
        self.assertEqual(ref_frexp_exp_i32_f32(0x0000_0001), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(0x007F_FFFF), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(0x8000_0001), 0)
        self.assertEqual(ref_frexp_exp_i32_f32(0x807F_FFFF), 0)


if __name__ == '__main__':
    unittest.main()
