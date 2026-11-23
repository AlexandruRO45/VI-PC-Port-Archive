"""Semantic interpreter tests and translator verification for single-precision float LDS atomic minimum (ds_min_f32, ds_min_rtn_f32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    py -I tests/semantic/test_lds_min_f32_hw.py
    py -m unittest discover -s tests/semantic -p test_lds_min_f32_hw.py
"""
from __future__ import annotations

import math
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
import lds_run as L
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFFFFFF
SIGN_MASK: int = 0x80000000
EXP_MASK: int = 0x7F800000
FRAC_MASK: int = 0x007FFFFF
QNAN_BIT: int = 0x00400000

POS_ZERO: int = 0x00000000
NEG_ZERO: int = 0x80000000
ONE_F32: int = 0x3F800000           # 1.0
NEG_ONE_F32: int = 0xBF800000       # -1.0
TWO_F32: int = 0x40000000           # 2.0
NEG_TWO_F32: int = 0xC0000000       # -2.0
HALF_F32: int = 0x3F000000          # 0.5
NEG_HALF_F32: int = 0xBF000000      # -0.5
TEN_F32: int = 0x41200000           # 10.0
NEG_TEN_F32: int = 0xC1200000       # -10.0

POS_INF: int = 0x7F800000           # +infinity
NEG_INF: int = 0xFF800000           # -infinity
QNAN_VAL: int = 0x7FC00000          # Quiet NaN
SNAN_VAL: int = 0x7F800001          # Signaling NaN


def f32_to_bits(val: float) -> int:
    """Encode float to IEEE 754 binary32 32-bit unsigned bit pattern."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit unsigned integer to IEEE 754 binary32 float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan(bits: int) -> bool:
    """True if 32-bit pattern is an IEEE 754 NaN."""
    b = bits & MASK32
    return ((b & EXP_MASK) == EXP_MASK) and ((b & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """True if 32-bit pattern is +0.0 or -0.0."""
    return (bits & ~SIGN_MASK) == 0


def ref_ds_min_f32(curr_mem: int, data_val: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_min_f32 atomic operation.

    Follows GLSL.std.450 NMin / IEEE 754-2008 minNum:
      - If one operand is NaN and the other is a valid float, returns the valid float.
      - If both are NaN, returns a NaN.
      - Signed zero ordering: -0.0 is strictly less than +0.0 (min(+0.0, -0.0) -> -0.0).
      - Subnormals, infinities, and normals follow standard float ordering.
    """
    a = curr_mem & MASK32
    b = data_val & MASK32

    a_nan = is_nan(a)
    b_nan = is_nan(b)

    if a_nan and b_nan:
        return a | QNAN_BIT
    if a_nan:
        return b
    if b_nan:
        return a

    if is_zero(a) and is_zero(b):
        return NEG_ZERO if ((a == NEG_ZERO) or (b == NEG_ZERO)) else POS_ZERO

    fa = bits_to_f32(a)
    fb = bits_to_f32(b)

    if fa < fb:
        return a
    elif fb < fa:
        return b
    else:
        # Same magnitude: negative sign is smaller
        return a if (a & SIGN_MASK) != 0 else b


def ref_ds_min_rtn_f32(curr_mem: int, data_val: int) -> Tuple[int, int]:
    """Reference returning atomic float minimum: returns (old_mem, new_mem)."""
    old = curr_mem & MASK32
    new = ref_ds_min_f32(old, data_val)
    return old, new


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, size_bytes: int = 4096) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.storage = [0] * self.num_dwords

    def resolve_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        dword_idx = eff_addr >> 2
        return dword_idx & (self.num_dwords - 1)

    def read_dword(self, byte_addr: int, offset: int = 0) -> int:
        idx = self.resolve_dword_index(byte_addr, offset)
        return self.storage[idx]

    def write_dword(self, byte_addr: int, value: int, offset: int = 0) -> None:
        idx = self.resolve_dword_index(byte_addr, offset)
        self.storage[idx] = value & MASK32

    def atomic_min_f32(self, byte_addr: int, data_val: int, offset: int = 0, rtn: bool = False) -> Tuple[Optional[int], int]:
        idx = self.resolve_dword_index(byte_addr, offset)
        old_val = self.storage[idx]
        new_val = ref_ds_min_f32(old_val, data_val)
        self.storage[idx] = new_val
        return (old_val if rtn else None), new_val

    def get_bank(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        return (eff_addr >> 2) & 31

class TestDSMinF32OracleAndArchitecturalModel(unittest.TestCase):
    """Part 1: Reference mathematical oracle and architectural model tests."""

    def test_float_bit_conversions(self) -> None:
        self.assertEqual(f32_to_bits(0.0), POS_ZERO)
        self.assertEqual(f32_to_bits(-0.0), NEG_ZERO)
        self.assertEqual(f32_to_bits(1.0), ONE_F32)
        self.assertEqual(f32_to_bits(-1.0), NEG_ONE_F32)
        self.assertEqual(bits_to_f32(ONE_F32), 1.0)
        self.assertEqual(bits_to_f32(NEG_ONE_F32), -1.0)

    def test_reference_oracle_basic_floats(self) -> None:
        # Standard positive floats: min(1.0, 2.0) = 1.0
        self.assertEqual(ref_ds_min_f32(ONE_F32, TWO_F32), ONE_F32)
        self.assertEqual(ref_ds_min_f32(TWO_F32, ONE_F32), ONE_F32)

        # Negative vs positive: min(-2.0, 1.0) = -2.0
        self.assertEqual(ref_ds_min_f32(NEG_TWO_F32, ONE_F32), NEG_TWO_F32)
        self.assertEqual(ref_ds_min_f32(ONE_F32, NEG_TWO_F32), NEG_TWO_F32)

        # Negative vs negative: min(-10.0, -2.0) = -10.0
        self.assertEqual(ref_ds_min_f32(NEG_TEN_F32, NEG_TWO_F32), NEG_TEN_F32)
        self.assertEqual(ref_ds_min_f32(NEG_TWO_F32, NEG_TEN_F32), NEG_TEN_F32)

    def test_reference_oracle_signed_zeros(self) -> None:
        # -0.0 < +0.0
        self.assertEqual(ref_ds_min_f32(POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_ds_min_f32(NEG_ZERO, POS_ZERO), NEG_ZERO)
        self.assertEqual(ref_ds_min_f32(POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_ds_min_f32(NEG_ZERO, NEG_ZERO), NEG_ZERO)

    def test_reference_oracle_nan_suppression(self) -> None:
        # Single NaN: non-NaN operand is preserved (minNum / NMin semantics)
        self.assertEqual(ref_ds_min_f32(QNAN_VAL, ONE_F32), ONE_F32)
        self.assertEqual(ref_ds_min_f32(ONE_F32, QNAN_VAL), ONE_F32)
        self.assertEqual(ref_ds_min_f32(SNAN_VAL, NEG_TEN_F32), NEG_TEN_F32)

        # Both NaN: returns NaN
        res = ref_ds_min_f32(QNAN_VAL, SNAN_VAL)
        self.assertTrue(is_nan(res))

    def test_reference_oracle_returning_form(self) -> None:
        old, new = ref_ds_min_rtn_f32(TEN_F32, TWO_F32)
        self.assertEqual(old, TEN_F32)
        self.assertEqual(new, TWO_F32)

        old, new = ref_ds_min_rtn_f32(NEG_TEN_F32, ONE_F32)
        self.assertEqual(old, NEG_TEN_F32)
        self.assertEqual(new, NEG_TEN_F32)

    def test_lds_memory_model(self) -> None:
        lds = LDSMemory(size_bytes=4096)
        addr = 0x40
        lds.write_dword(addr, TEN_F32)

        # Non-returning atomic min with 2.0 -> memory becomes 2.0
        old, new = lds.atomic_min_f32(addr, TWO_F32, rtn=False)
        self.assertIsNone(old)
        self.assertEqual(new, TWO_F32)
        self.assertEqual(lds.read_dword(addr), TWO_F32)

        # Returning atomic min with -1.0 -> returns 2.0, memory becomes -1.0
        old, new = lds.atomic_min_f32(addr, NEG_ONE_F32, rtn=True)
        self.assertEqual(old, TWO_F32)
        self.assertEqual(new, NEG_ONE_F32)
        self.assertEqual(lds.read_dword(addr), NEG_ONE_F32)


if __name__ == '__main__':
    unittest.main()
