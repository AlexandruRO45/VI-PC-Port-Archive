from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_rsub_rtn_f32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Floating-Point Reverse Subtraction with Return:
Performs an atomic single-precision IEEE 754 floating-point reverse subtraction on 32-bit LDS memory
and returns the original (pre-operation) value:
    new_val = vdata0 - old_val
    vdst = old_val

Syntax:
    ds_rsub_rtn_f32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val_u32 = LDS[dword_index]
    new_val_u32 = ieee754_sub_f32(vdata0, old_val_u32)  # vdata0 is minuend, old_val is subtrahend
    LDS[dword_index] = new_val_u32
    vdst = old_val_u32
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

# IEEE 754 Single-Precision Constants (raw uint32 bit patterns)
POS_ZERO_F32: int = 0x0000_0000
NEG_ZERO_F32: int = 0x8000_0000
POS_INF_F32: int = 0x7F80_0000
NEG_INF_F32: int = 0xFF80_0000
QNAN_DEFAULT: int = 0x7FC0_0000
SNAN_DEFAULT: int = 0x7F80_0001
ONE_F32: int = 0x3F80_0000          # 1.0f
TWO_F32: int = 0x4000_0000          # 2.0f
THREE_F32: int = 0x4040_0000        # 3.0f
FIVE_F32: int = 0x40A0_0000         # 5.0f
TEN_F32: int = 0x4120_0000          # 10.0f


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK_32
    return val - (1 << 32) if (val & 0x8000_0000) else val


def u32_to_f32(val: int) -> float:
    """Unpack 32-bit unsigned dword to Python float."""
    return struct.unpack("<f", struct.pack("<I", u32(val)))[0]


def f32_to_u32(val: float) -> int:
    """Pack Python float to IEEE 754 single-precision 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def ieee754_rsub_f32(vdata0_u32: int, old_val_u32: int) -> int:
    """Bit-accurate IEEE 754 single-precision reverse subtraction: vdata0 - old_val.

    Matches RDNA2 hardware semantics where vdata0 is minuend and old_val is subtrahend.
    """
    a_u32 = u32(vdata0_u32)
    b_u32 = u32(old_val_u32)

    a_exp = (a_u32 >> 23) & 0xFF
    a_mant = a_u32 & 0x7F_FFFF
    b_exp = (b_u32 >> 23) & 0xFF
    b_mant = b_u32 & 0x7F_FFFF

    a_is_nan = (a_exp == 0xFF and a_mant != 0)
    b_is_nan = (b_exp == 0xFF and b_mant != 0)

    # Any NaN operand produces quiet NaN
    if a_is_nan or b_is_nan:
        return QNAN_DEFAULT

    a_is_inf = (a_exp == 0xFF and a_mant == 0)
    b_is_inf = (b_exp == 0xFF and b_mant == 0)
    a_sign = (a_u32 >> 31) & 1
    b_sign = (b_u32 >> 31) & 1

    # (+Inf) - (+Inf) or (-Inf) - (-Inf) is invalid -> NaN
    if a_is_inf and b_is_inf and (a_sign == b_sign):
        return QNAN_DEFAULT

    # If minuend is Inf, result is minuend
    if a_is_inf:
        return a_u32

    # If subtrahend is Inf, result is -subtrahend
    if b_is_inf:
        return b_u32 ^ 0x8000_0000

    # Float subtraction
    a_f = u32_to_f32(a_u32)
    b_f = u32_to_f32(b_u32)

    # Special handling for signed zeros:
    # (+0.0) - (+0.0) = +0.0
    # (-0.0) - (-0.0) = +0.0
    # (+0.0) - (-0.0) = +0.0
    # (-0.0) - (+0.0) = -0.0
    if a_f == 0.0 and b_f == 0.0:
        if a_sign == 1 and b_sign == 0:
            return NEG_ZERO_F32
        return POS_ZERO_F32

    res_f = a_f - b_f
    res_u32 = f32_to_u32(res_f)

    # In round-to-nearest-even, if difference of equal non-zero finite numbers is zero, sign is +0.0
    if a_f == b_f and res_f == 0.0:
        return POS_ZERO_F32

    return res_u32


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert effective byte address to aligned LDS dword index."""
    return (eff_addr >> 2) & (capacity_dwords - 1)


class LDSMemory:
    """Architectural model of Local Data Share (LDS) memory."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_dwords}")
        self.size_dwords = size_dwords
        self.data: List[int] = [0] * size_dwords

    def read_dword(self, index: int) -> int:
        return self.data[index & (self.size_dwords - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.data[index & (self.size_dwords - 1)] = u32(val)

    def atomic_rsub_rtn_f32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 32-bit float reverse subtraction on LDS with return.

        Returns:
            Tuple[int, int]: (old_val, new_val)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        old_val = self.read_dword(idx)
        new_val = ieee754_rsub_f32(vdata0, old_val)
        self.write_dword(idx, new_val)
        return old_val, new_val


def oracle_ds_rsub_rtn_f32(
    lds_bytes: bytearray,
    vaddr: int,
    vdata0: int,
    offset: int = 0,
    lds_size_bytes: int = LDS_SIZE_BYTES,
) -> Tuple[int, int]:
    """Pure bytearray reference mathematical oracle for RDNA2 ds_rsub_rtn_f32."""
    eff_addr = calc_lds_effective_address(vaddr, offset) & (lds_size_bytes - 1)
    aligned_addr = eff_addr & ~3  # drop low 2 bits
    old_val = struct.unpack_from("<I", lds_bytes, aligned_addr)[0]
    new_val = ieee754_rsub_f32(vdata0, old_val)
    struct.pack_into("<I", lds_bytes, aligned_addr, new_val)
    return old_val, new_val


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_dwords=1024)
        self.lds_bytes = bytearray(LDS_SIZE_BYTES)

    def test_basic_reverse_subtraction(self) -> None:
        # Initial memory has 2.0f, vdata0 is 5.0f
        # Reverse subtraction: new_val = vdata0 - old_val = 5.0f - 2.0f = 3.0f
        # Returns old_val = 2.0f
        self.mem.write_dword(10, TWO_F32)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(vaddr=40, vdata0=FIVE_F32, offset=0)
        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(new_val, THREE_F32)
        self.assertAlmostEqual(u32_to_f32(self.mem.read_dword(10)), 3.0, places=5)

    def test_reverse_subtraction_negative_result(self) -> None:
        # Initial memory has 10.0f, vdata0 is 3.0f
        # Reverse subtraction: 3.0f - 10.0f = -7.0f
        self.mem.write_dword(15, TEN_F32)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(vaddr=60, vdata0=THREE_F32, offset=0)
        self.assertEqual(old_val, TEN_F32)
        self.assertAlmostEqual(u32_to_f32(new_val), -7.0, places=5)
        self.assertAlmostEqual(u32_to_f32(self.mem.read_dword(15)), -7.0, places=5)

    def test_non_commutativity_difference_from_sub(self) -> None:
        # Crucial test distinguishing rsub from sub:
        # vdata0 = 2.0f, old_val = 5.0f
        # rsub: 2.0f - 5.0f = -3.0f
        # sub:  5.0f - 2.0f = +3.0f
        self.mem.write_dword(20, FIVE_F32)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(vaddr=80, vdata0=TWO_F32, offset=0)
        self.assertEqual(old_val, FIVE_F32)
        self.assertAlmostEqual(u32_to_f32(new_val), -3.0, places=5)

    def test_address_offset_and_alignment(self) -> None:
        # vaddr = 0x200, offset = 16 -> effective = 0x210 -> dword index = 0x84
        idx = 0x84
        self.mem.write_dword(idx, ONE_F32)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(vaddr=0x200, vdata0=FIVE_F32, offset=16)
        self.assertEqual(old_val, ONE_F32)
        self.assertAlmostEqual(u32_to_f32(new_val), 4.0, places=5)

        # Unaligned byte address 0x203 drops low 2 bits -> 0x200 (index 0x80)
        self.mem.write_dword(0x80, TWO_F32)
        old_val, new_val = self.mem.atomic_rsub_rtn_f32(vaddr=0x200, vdata0=TEN_F32, offset=3)
        self.assertEqual(old_val, TWO_F32)
        self.assertAlmostEqual(u32_to_f32(new_val), 8.0, places=5)

    def test_bytearray_oracle_equivalence(self) -> None:
        addr = 0x400
        struct.pack_into("<I", self.lds_bytes, addr, TWO_F32)
        old_val, new_val = oracle_ds_rsub_rtn_f32(self.lds_bytes, vaddr=addr, vdata0=FIVE_F32, offset=0)
        self.assertEqual(old_val, TWO_F32)
        self.assertEqual(new_val, THREE_F32)
        self.assertEqual(struct.unpack_from("<I", self.lds_bytes, addr)[0], THREE_F32)


if __name__ == "__main__":
    unittest.main()
