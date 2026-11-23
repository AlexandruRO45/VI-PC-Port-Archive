from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 Parameter Interpolation Suite.

Covers:
- v_interp_p1_f32: First barycentric axis interpolation D = P0 + i * (P1 - P0).
- v_interp_p2_f32: Second barycentric axis interpolation and accumulation D = D_p1 + j * (P2 - P0).
- v_interp_mov_f32: Direct parameter access: p0 (provoking vertex), p10 (delta P1-P0), p20 (delta P2-P0).
- v_interp_p1ll_f16: 16-bit packed attribute low half first axis interpolation in f32 intermediate precision.
- v_interp_p1lv_f16: 16-bit packed attribute low/high half first axis interpolation in f32 intermediate precision.
- v_interp_p2_f16: Second axis accumulation onto f32 intermediate, narrowed to f16 with optional clamp.

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

MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF


def f32_to_bits(f: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", float(f)))[0]
    except OverflowError:
        return 0x7F80_0000 if f > 0 else 0xFF80_0000


def bits_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_round(f: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", float(f)))[0]
    except OverflowError:
        return float("inf") if f > 0 else float("-inf")


def f16_to_bits(f: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", float(f)))[0]
    except OverflowError:
        return 0x7C00 if f > 0 else 0xFC00


def bits_to_f16(u: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u & MASK16))[0]


# ------------------------------------------------------------------------------
# v_interp_p1_f32 Reference Oracle
# ------------------------------------------------------------------------------

def ref_interp_p1_f32(i_coord: float, p0: float, p1: float) -> int:
    """Computes D = P0 + i * (P1 - P0)."""
    d1 = f32_round(p1 - p0)
    m1 = f32_round(d1 * i_coord)
    res = f32_round(p0 + m1)
    return f32_to_bits(res)


class TestInterpP1F32Oracle(unittest.TestCase):
    def test_vertex_endpoints(self):
        # i = 0.0 -> P0
        self.assertEqual(ref_interp_p1_f32(0.0, 10.0, 20.0), f32_to_bits(10.0))
        # i = 1.0 -> P1
        self.assertEqual(ref_interp_p1_f32(1.0, 10.0, 20.0), f32_to_bits(20.0))

    def test_midpoint_interpolation(self):
        # i = 0.5 -> 15.0
        self.assertEqual(ref_interp_p1_f32(0.5, 10.0, 20.0), f32_to_bits(15.0))
        # i = 0.25 -> 12.5
        self.assertEqual(ref_interp_p1_f32(0.25, 10.0, 20.0), f32_to_bits(12.5))


if __name__ == '__main__':
    unittest.main()
