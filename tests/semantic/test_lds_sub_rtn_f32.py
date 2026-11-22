from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_sub_rtn_f32.

Opcode:
  - ds_sub_rtn_f32 vdst, vaddr, vdata [offset:N]
      Atomic 32-bit single-precision floating-point subtraction on LDS memory with
      destination register return:
        effective_byte = (vaddr + offset) & 0xFFFFFFFF
        idx = (effective_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[idx]
        LDS[idx] = IEEE754_sub(old_val, vdata)
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in VGPR vaddr + optional unsigned 16-bit immediate byte offset.
  - Floating-point arithmetic: IEEE 754-2008 / 2019 binary32 single-precision rules.
  - Return value: Prior 32-bit value stored in memory at the targeted location is atomically
    returned and written to destination VGPR vdst.
"""

from dataclasses import dataclass
from enum import Enum
import math
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET16_MASK: int = 0xFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB LDS capacity in dwords
DEFAULT_NUM_LANES: int = 64
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
NUM_LDS_BANKS: int = 32

# IEEE 754 binary32 Architectural Constants and Bit Patterns
SIGN_MASK: int = 0x80000000
EXP_MASK: int = 0x7F800000
FRAC_MASK: int = 0x007FFFFF
QNAN_BIT: int = 0x00400000
CANONICAL_QNAN: int = 0x7FC00000

POS_ZERO_F32: int = 0x00000000       # +0.0
NEG_ZERO_F32: int = 0x80000000       # -0.0
ONE_F32: int = 0x3F800000            # 1.0
NEG_ONE_F32: int = 0xBF800000        # -1.0
TWO_F32: int = 0x40000000            # 2.0
NEG_TWO_F32: int = 0xC0000000        # -2.0
THREE_F32: int = 0x40400000          # 3.0
FIVE_F32: int = 0x40A00000           # 5.0
TEN_F32: int = 0x41200000            # 10.0
POS_INF_F32: int = 0x7F800000        # +inf
NEG_INF_F32: int = 0xFF800000        # -inf
QNAN_F32: int = 0x7FC00000           # Quiet NaN
SNAN_F32: int = 0x7F800001           # Signaling NaN
FLT_MIN: int = 0x00800000            # 2^-126
FLT_MAX: int = 0x7F7FFFFF            # (2 - 2^-23) * 2^127
SUBNORM_MIN: int = 0x00000001        # Smallest positive subnormal: 2^-149
SUBNORM_MAX: int = 0x007FFFFF        # Largest positive subnormal: (1 - 2^-23) * 2^-126


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def f32_to_bits(val: float) -> int:
    """Encode Python float to IEEE 754 single-precision 32-bit bit pattern."""
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF_F32 if val > 0 else NEG_INF_F32


def bits_to_f32(bits: int) -> float:
    """Decode 32-bit bit pattern to IEEE 754 single-precision Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 single-precision NaN."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_inf(bits: int) -> bool:
    """Return True if bits represent IEEE 754 single-precision infinity."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_zero(bits: int) -> bool:
    """Return True if bits represent IEEE 754 single-precision +0.0 or -0.0."""
    return (bits & ~SIGN_MASK) == 0


def is_subnormal(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormalized) float."""
    bits &= MASK32
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def oracle_sub_f32(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for IEEE 754 single-precision subtraction (a - b).

    Returns 32-bit unsigned integer bit pattern of the result.
    """
    a_bits &= MASK32
    b_bits &= MASK32

    # NaN propagation
    if is_nan(a_bits) or is_nan(b_bits):
        return CANONICAL_QNAN

    # Infinity arithmetic
    if is_inf(a_bits) and is_inf(b_bits):
        # inf - inf of same sign is NaN (+inf - +inf = NaN, -inf - -inf = NaN)
        if (a_bits & SIGN_MASK) == (b_bits & SIGN_MASK):
            return CANONICAL_QNAN
        return a_bits
    if is_inf(a_bits):
        return a_bits
    if is_inf(b_bits):
        # a - (+inf) = -inf, a - (-inf) = +inf
        return NEG_INF_F32 if (b_bits == POS_INF_F32) else POS_INF_F32

    # Zeros and equal operands
    if is_zero(a_bits) and is_zero(b_bits):
        # (+0.0) - (+0.0) = +0.0
        # (-0.0) - (-0.0) = +0.0
        # (+0.0) - (-0.0) = +0.0
        # (-0.0) - (+0.0) = -0.0
        if a_bits == NEG_ZERO_F32 and b_bits == POS_ZERO_F32:
            return NEG_ZERO_F32
        return POS_ZERO_F32

    fa = bits_to_f32(a_bits)
    fb = bits_to_f32(b_bits)
    res_f = fa - fb
    res_bits = f32_to_bits(res_f)

    # When x - x yields zero in round-to-nearest-even, the sign is positive
    if is_zero(res_bits) and fa == fb:
        return POS_ZERO_F32

    return res_bits


class LDSMemory:
    """Model of Local Data Share memory storage."""

    def __init__(self, size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_dwords}")
        self.size_dwords = size_dwords
        self.data: list[int] = [0] * size_dwords

    def read_dword(self, index: int) -> int:
        return self.data[index & (self.size_dwords - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.data[index & (self.size_dwords - 1)] = u32(val)


def ref_ds_sub_rtn_f32(
    mem: LDSMemory,
    vaddr: int,
    vdata_bits: int,
    offset: int = 0,
) -> int:
    """Reference mathematical oracle for ds_sub_rtn_f32.

    Atomically computes:
      old_val = LDS[idx]
      LDS[idx] = old_val - vdata (IEEE 754 single precision)
      return old_val
    """
    effective_byte = u32(vaddr + offset)
    idx = (effective_byte >> 2) & (mem.size_dwords - 1)
    old_val = mem.read_dword(idx)
    new_val = oracle_sub_f32(old_val, vdata_bits)
    mem.write_dword(idx, new_val)
    return old_val


class TestLdsSubRtnF32Oracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_dwords=1024)

    def test_basic_sub_and_return(self) -> None:
        # Pre-seed memory at dword 10 with 5.0
        self.mem.write_dword(10, FIVE_F32)
        # vaddr = 40 (byte 40 -> dword 10), subtract 2.0
        ret = ref_ds_sub_rtn_f32(self.mem, vaddr=40, vdata_bits=TWO_F32, offset=0)
        # Old value (5.0) returned
        self.assertEqual(ret, FIVE_F32)
        # Memory updated to 5.0 - 2.0 = 3.0
        self.assertEqual(self.mem.read_dword(10), THREE_F32)

    def test_sub_negative_result(self) -> None:
        self.mem.write_dword(5, ONE_F32)
        ret = ref_ds_sub_rtn_f32(self.mem, vaddr=20, vdata_bits=THREE_F32, offset=0)
        self.assertEqual(ret, ONE_F32)
        # 1.0 - 3.0 = -2.0
        self.assertEqual(self.mem.read_dword(5), NEG_TWO_F32)

    def test_sub_zero_preserves_val_and_returns(self) -> None:
        self.mem.write_dword(2, TEN_F32)
        ret = ref_ds_sub_rtn_f32(self.mem, vaddr=8, vdata_bits=POS_ZERO_F32, offset=0)
        self.assertEqual(ret, TEN_F32)
        self.assertEqual(self.mem.read_dword(2), TEN_F32)

    def test_address_offset_and_wrapping(self) -> None:
        # vaddr = 16, offset = 8 -> effective byte = 24 -> dword 6
        self.mem.write_dword(6, THREE_F32)
        ret = ref_ds_sub_rtn_f32(self.mem, vaddr=16, vdata_bits=ONE_F32, offset=8)
        self.assertEqual(ret, THREE_F32)
        self.assertEqual(self.mem.read_dword(6), TWO_F32)


if __name__ == "__main__":
    unittest.main()
