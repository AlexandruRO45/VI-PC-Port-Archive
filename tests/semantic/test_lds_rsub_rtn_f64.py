from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_rsub_rtn_f64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Floating-Point Reverse Subtraction with Return:
Performs an atomic double-precision IEEE 754-2008/2019 floating-point reverse subtraction on 64-bit LDS memory
and returns the original (pre-operation) 64-bit value:
    new_val = vdata0 - old_val
    vdst = old_val

Syntax:
    ds_rsub_rtn_f64 vdst, vaddr, vdata0 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val_u64 = LDS64[qword_index]
    new_val_u64 = ieee754_rsub_f64(vdata0, old_val_u64)  # vdata0 is minuend, old_val is subtrahend
    LDS64[qword_index] = new_val_u64
    vdst = old_val_u64
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_QWORDS: int = 8192         # 8,192 64-bit qwords
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

# IEEE 754 Binary64 Architectural Bitmasks & Constants
SIGN_MASK64: int = 0x8000_0000_0000_0000
EXP_MASK64: int = 0x7FF0_0000_0000_0000
FRAC_MASK64: int = 0x000F_FFFF_FFFF_FFFF
CANONICAL_QNAN64: int = 0x7FF8_0000_0000_0000
CANONICAL_SNAN64: int = 0x7FF0_0000_0000_0001
POS_ZERO_F64: int = 0x0000_0000_0000_0000
NEG_ZERO_F64: int = 0x8000_0000_0000_0000
POS_INF_F64: int = 0x7FF0_0000_0000_0000
NEG_INF_F64: int = 0xFFF0_0000_0000_0000

# Convenient float64 constants (raw uint64 bit patterns)
ONE_F64: int = 0x3FF0_0000_0000_0000        # 1.0
TWO_F64: int = 0x4000_0000_0000_0000        # 2.0
THREE_F64: int = 0x4008_0000_0000_0000      # 3.0
FIVE_F64: int = 0x4014_0000_0000_0000       # 5.0
TEN_F64: int = 0x4024_0000_0000_0000        # 10.0


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def u64_to_f64(val: int) -> float:
    """Unpack 64-bit unsigned qword to Python float."""
    return struct.unpack("<d", struct.pack("<Q", u64(val)))[0]


def f64_to_u64(val: float) -> int:
    """Pack Python float to IEEE 754 binary64 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return NEG_INF_F64 if val < 0 else POS_INF_F64


def ieee754_rsub_f64(vdata0_u64: int, old_val_u64: int) -> int:
    """Bit-accurate IEEE 754 double-precision reverse subtraction: vdata0 - old_val.

    Matches RDNA2 hardware semantics where vdata0 is minuend and old_val is subtrahend.
    """
    a_u64 = u64(vdata0_u64)
    b_u64 = u64(old_val_u64)

    a_exp = (a_u64 & EXP_MASK64) >> 52
    a_frac = a_u64 & FRAC_MASK64
    b_exp = (b_u64 & EXP_MASK64) >> 52
    b_frac = b_u64 & FRAC_MASK64

    a_is_nan = (a_exp == 0x7FF and a_frac != 0)
    b_is_nan = (b_exp == 0x7FF and b_frac != 0)

    # Any NaN operand produces canonical quiet NaN
    if a_is_nan or b_is_nan:
        return CANONICAL_QNAN64

    a_is_inf = (a_exp == 0x7FF and a_frac == 0)
    b_is_inf = (b_exp == 0x7FF and b_frac == 0)
    a_sign = (a_u64 >> 63) & 1
    b_sign = (b_u64 >> 63) & 1

    # (+Inf) - (+Inf) or (-Inf) - (-Inf) is invalid -> NaN
    if a_is_inf and b_is_inf and (a_sign == b_sign):
        return CANONICAL_QNAN64

    # If minuend is Inf, result is minuend
    if a_is_inf:
        return a_u64

    # If subtrahend is Inf, result is -subtrahend
    if b_is_inf:
        return b_u64 ^ SIGN_MASK64

    a_f = u64_to_f64(a_u64)
    b_f = u64_to_f64(b_u64)

    # Signed zeros:
    # (+0.0) - (+0.0) = +0.0
    # (-0.0) - (-0.0) = +0.0
    # (+0.0) - (-0.0) = +0.0
    # (-0.0) - (+0.0) = -0.0
    if a_f == 0.0 and b_f == 0.0:
        if a_sign == 1 and b_sign == 0:
            return NEG_ZERO_F64
        return POS_ZERO_F64

    res_f = a_f - b_f
    res_u64 = f64_to_u64(res_f)

    # In round-to-nearest-even, if difference of equal non-zero numbers is zero, sign is +0.0
    if a_f == b_f and res_f == 0.0:
        return POS_ZERO_F64

    return res_u64


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert effective byte address to aligned LDS 64-bit qword index."""
    return (eff_addr >> 3) & (capacity_qwords - 1)


class LDSMemory64:
    """Architectural model of Local Data Share (LDS) 64-bit memory."""

    def __init__(self, size_qwords: int = LDS_SIZE_QWORDS) -> None:
        if size_qwords <= 0 or (size_qwords & (size_qwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_qwords}")
        self.size_qwords = size_qwords
        self.data: List[int] = [0] * size_qwords

    def read_qword(self, index: int) -> int:
        return self.data[index & (self.size_qwords - 1)]

    def write_qword(self, index: int, val: int) -> None:
        self.data[index & (self.size_qwords - 1)] = u64(val)

    def atomic_rsub_rtn_f64(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 64-bit float reverse subtraction on LDS with return.

        Returns:
            Tuple[int, int]: (old_val, new_val)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_qword_index(eff_addr, self.size_qwords)
        old_val = self.read_qword(idx)
        new_val = ieee754_rsub_f64(vdata0, old_val)
        self.write_qword(idx, new_val)
        return old_val, new_val


def oracle_ds_rsub_rtn_f64(
    lds_bytes: bytearray,
    vaddr: int,
    vdata0: int,
    offset: int = 0,
    lds_size_bytes: int = LDS_SIZE_BYTES,
) -> Tuple[int, int]:
    """Pure bytearray reference mathematical oracle for RDNA2 ds_rsub_rtn_f64."""
    eff_addr = calc_lds_effective_address(vaddr, offset) & (lds_size_bytes - 1)
    aligned_addr = eff_addr & ~7  # drop low 3 bits (8-byte alignment)
    old_val = struct.unpack_from("<Q", lds_bytes, aligned_addr)[0]
    new_val = ieee754_rsub_f64(vdata0, old_val)
    struct.pack_into("<Q", lds_bytes, aligned_addr, new_val)
    return old_val, new_val


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory64(size_qwords=1024)
        self.lds_bytes = bytearray(LDS_SIZE_BYTES)

    def test_basic_reverse_subtraction(self) -> None:
        # Initial memory has 2.0, vdata0 is 5.0
        # Reverse subtraction: new_val = vdata0 - old_val = 5.0 - 2.0 = 3.0
        # Returns old_val = 2.0
        self.mem.write_qword(5, TWO_F64)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(vaddr=40, vdata0=FIVE_F64, offset=0)
        self.assertEqual(old_val, TWO_F64)
        self.assertEqual(new_val, THREE_F64)
        self.assertAlmostEqual(u64_to_f64(self.mem.read_qword(5)), 3.0, places=9)

    def test_reverse_subtraction_negative_result(self) -> None:
        # Initial memory has 10.0, vdata0 is 3.0
        # Reverse subtraction: 3.0 - 10.0 = -7.0
        self.mem.write_qword(12, TEN_F64)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(vaddr=96, vdata0=THREE_F64, offset=0)
        self.assertEqual(old_val, TEN_F64)
        self.assertAlmostEqual(u64_to_f64(new_val), -7.0, places=9)
        self.assertAlmostEqual(u64_to_f64(self.mem.read_qword(12)), -7.0, places=9)

    def test_non_commutativity_difference_from_sub(self) -> None:
        # vdata0 = 2.0, old_val = 5.0
        # rsub: 2.0 - 5.0 = -3.0
        # sub:  5.0 - 2.0 = +3.0
        self.mem.write_qword(15, FIVE_F64)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(vaddr=120, vdata0=TWO_F64, offset=0)
        self.assertEqual(old_val, FIVE_F64)
        self.assertAlmostEqual(u64_to_f64(new_val), -3.0, places=9)

    def test_address_offset_and_qword_alignment(self) -> None:
        # vaddr = 0x200, offset = 16 -> effective = 0x210 -> qword index = 0x210 >> 3 = 0x42
        idx = 0x42
        self.mem.write_qword(idx, ONE_F64)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(vaddr=0x200, vdata0=FIVE_F64, offset=16)
        self.assertEqual(old_val, ONE_F64)
        self.assertAlmostEqual(u64_to_f64(new_val), 4.0, places=9)

        # Unaligned byte address: 0x207 drops low 3 bits -> 0x200 (index 0x40)
        self.mem.write_qword(0x40, TWO_F64)
        old_val, new_val = self.mem.atomic_rsub_rtn_f64(vaddr=0x200, vdata0=TEN_F64, offset=7)
        self.assertEqual(old_val, TWO_F64)
        self.assertAlmostEqual(u64_to_f64(new_val), 8.0, places=9)

    def test_bytearray_oracle_equivalence(self) -> None:
        addr = 0x800
        struct.pack_into("<Q", self.lds_bytes, addr, TWO_F64)
        old_val, new_val = oracle_ds_rsub_rtn_f64(self.lds_bytes, vaddr=addr, vdata0=FIVE_F64, offset=0)
        self.assertEqual(old_val, TWO_F64)
        self.assertEqual(new_val, THREE_F64)
        self.assertEqual(struct.unpack_from("<Q", self.lds_bytes, addr)[0], THREE_F64)


if __name__ == "__main__":
    unittest.main()
