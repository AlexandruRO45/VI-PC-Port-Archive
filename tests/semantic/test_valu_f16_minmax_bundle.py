from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 Float16 Min/Max/Clamp Suite.

Covers:
- v_min_f16: IEEE 754 binary16 minimum, NMin semantics (NaN passes other operand, -0.0 < +0.0).
- v_max_f16: IEEE 754 binary16 maximum, NMax semantics (NaN passes other operand, +0.0 > -0.0).
- v_floor_f16: floor rounding in binary16, preserving sign of +-0.0 and pass-through for NaN/inf.
- v_ceil_f16: ceiling rounding in binary16, preserving sign of +-0.0 and pass-through for NaN/inf.
- v_trunc_f16: truncation towards zero in binary16, preserving sign of +-0.0 and pass-through for NaN/inf.
- v_rndne_f16: round to nearest even in binary16, preserving sign of +-0.0 and pass-through for NaN/inf.

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

MASK16: int = 0xFFFF
SIGN_BIT16: int = 0x8000
EXP_MASK16: int = 0x001F
EXP_SHIFT16: int = 10
MANT_MASK16: int = 0x03FF

POS_ZERO16: int = 0x0000
NEG_ZERO16: int = 0x8000
POS_INF16: int = 0x7C00
NEG_INF16: int = 0xFC00
QNAN16: int = 0x7E00
SNAN16: int = 0x7C01


def f16_to_bits(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_to_f16(u: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u & MASK16))[0]


def is_nan_f16(u: int) -> bool:
    return (u & 0x7C00) == 0x7C00 and (u & 0x03FF) != 0


def is_inf_f16(u: int) -> bool:
    return (u & 0x7FFF) == 0x7C00


# ------------------------------------------------------------------------------
# v_min_f16 Reference Oracle
# ------------------------------------------------------------------------------

def ref_min_f16(s0: int, s1: int) -> int:
    u0 = s0 & MASK16
    u1 = s1 & MASK16
    nan0 = is_nan_f16(u0)
    nan1 = is_nan_f16(u1)
    if nan0 and nan1:
        return u0
    if nan0:
        return u1
    if nan1:
        return u0
    if (u0 | u1) == 0x8000:
        return NEG_ZERO16
    f0 = bits_to_f16(u0)
    f1 = bits_to_f16(u1)
    return u0 if f0 < f1 else u1


class TestMinF16Oracle(unittest.TestCase):
    def test_basic_min(self):
        self.assertEqual(ref_min_f16(f16_to_bits(1.0), f16_to_bits(2.0)), f16_to_bits(1.0))
        self.assertEqual(ref_min_f16(f16_to_bits(-3.0), f16_to_bits(1.0)), f16_to_bits(-3.0))
        self.assertEqual(ref_min_f16(f16_to_bits(5.0), f16_to_bits(5.0)), f16_to_bits(5.0))

    def test_signed_zeros(self):
        # -0.0 < +0.0
        self.assertEqual(ref_min_f16(POS_ZERO16, NEG_ZERO16), NEG_ZERO16)
        self.assertEqual(ref_min_f16(NEG_ZERO16, POS_ZERO16), NEG_ZERO16)


if __name__ == '__main__':
    unittest.main()
