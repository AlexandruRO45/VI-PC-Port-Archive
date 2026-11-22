from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_cmpst_f64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Floating-Point Compare-and-Swap:
Performs an atomic IEEE 754 double-precision float comparison on LDS memory and swaps if equal.

Syntax:
    ds_cmpst_f64 vaddr, vdata0, vdata1 [offset:offset16] [gds]
    ds_cmpst_rtn_f64 vdst, vaddr, vdata0, vdata1 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    addr_lo = effective_byte_address & ~3
    old_raw64 = (LDS[addr_lo + 4] << 32) | LDS[addr_lo]
    if float_equal_f64(old_raw64, vdata0):
        LDS[addr_lo] = vdata1 & 0xFFFFFFFF
        LDS[addr_lo + 4] = (vdata1 >> 32) & 0xFFFFFFFF
    if returning:
        vdst = old_raw64
"""

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

# IEEE 754 Double-Precision Boundary Constants
POS_ZERO_F64: int = 0x0000_0000_0000_0000
NEG_ZERO_F64: int = 0x8000_0000_0000_0000
POS_INF_F64: int = 0x7FF0_0000_0000_0000
NEG_INF_F64: int = 0xFFF0_0000_0000_0000
QNAN_F64: int = 0x7FF8_0000_0000_0000
SNAN_F64: int = 0x7FF0_0000_0000_0001
MIN_NORMAL_F64: int = 0x0010_0000_0000_0000
MIN_SUBNORMAL_F64: int = 0x0000_0000_0000_0001
MAX_SUBNORMAL_F64: int = 0x000F_FFFF_FFFF_FFFF
MAX_NORMAL_F64: int = 0x7FEF_FFFF_FFFF_FFFF
ONE_F64: int = 0x3FF0_0000_0000_0000


def u32(val: int) -> int:
    return val & MASK32


def u64(val: int) -> int:
    return val & MASK64


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    val &= MASK64
    return val & MASK32, (val >> 32) & MASK32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    return ((hi & MASK32) << 32) | (lo & MASK32)


def bits_to_f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def is_f64_nan(bits: int) -> bool:
    return (bits & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000 and (bits & 0x000F_FFFF_FFFF_FFFF) != 0


def is_f64_subnormal(bits: int) -> bool:
    return (bits & 0x7FF0_0000_0000_0000) == 0 and (bits & 0x000F_FFFF_FFFF_FFFF) != 0


def float_equal_f64(a_bits: int, b_bits: int) -> bool:
    if is_f64_nan(a_bits) or is_f64_nan(b_bits):
        return False
    return bits_to_f64(a_bits) == bits_to_f64(b_bits)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    return (eff_addr >> 2) & (capacity_dwords - 1)


def ref_ds_cmpst_f64(
    current_mem: int,
    cmp_val: int,
    new_val: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_cmpst_f64.

    Compares current_mem with cmp_val using IEEE 754 double-precision float equality;
    if equal, replaces memory with new_val.
    Always returns (old_mem64, new_mem64, swapped).
    """
    old_raw = u64(current_mem)
    cmp_raw = u64(cmp_val)
    new_raw = u64(new_val)

    if float_equal_f64(old_raw, cmp_raw):
        return old_raw, new_raw, True
    return old_raw, old_raw, False


class LDSMemory:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_qword(self, eff_addr: int) -> int:
        idx_lo = calc_lds_dword_index(eff_addr, self.size_dwords)
        idx_hi = (idx_lo + 1) & (self.size_dwords - 1)
        return combine_u32s_to_u64(self.memory[idx_lo], self.memory[idx_hi])

    def write_qword(self, eff_addr: int, val64: int) -> None:
        idx_lo = calc_lds_dword_index(eff_addr, self.size_dwords)
        idx_hi = (idx_lo + 1) & (self.size_dwords - 1)
        lo, hi = split_u64_to_u32s(val64)
        self.memory[idx_lo] = lo
        self.memory[idx_hi] = hi

    def atomic_cmpst_f64(
        self,
        vaddr: int,
        cmp_val64: int,
        new_val64: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_mem, swapped = ref_ds_cmpst_f64(curr, cmp_val64, new_val64)
        if swapped:
            self.write_qword(eff_addr, new_mem)
        return old_val, new_mem, swapped



class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_finite_double_match(self) -> None:
        val = f64_to_bits(3.141592653589793)
        new_val = f64_to_bits(2.718281828459045)
        old, mem, swapped = ref_ds_cmpst_f64(val, val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, val)
        self.assertEqual(mem, new_val)

    def test_oracle_signed_zero_equality(self) -> None:
        # In IEEE 754, +0.0 == -0.0 is True!
        old, mem, swapped = ref_ds_cmpst_f64(NEG_ZERO_F64, POS_ZERO_F64, f64_to_bits(100.0))
        self.assertTrue(swapped)
        self.assertEqual(old, NEG_ZERO_F64)
        self.assertEqual(mem, f64_to_bits(100.0))

    def test_oracle_nan_unordered(self) -> None:
        old, mem, swapped = ref_ds_cmpst_f64(QNAN_F64, QNAN_F64, f64_to_bits(1.0))
        self.assertFalse(swapped)
        self.assertEqual(mem, QNAN_F64)

    def test_oracle_infinities(self) -> None:
        old, mem, swapped = ref_ds_cmpst_f64(POS_INF_F64, POS_INF_F64, 0)
        self.assertTrue(swapped)
        self.assertEqual(mem, 0)

        old2, mem2, swapped2 = ref_ds_cmpst_f64(POS_INF_F64, NEG_INF_F64, 0)
        self.assertFalse(swapped2)
        self.assertEqual(mem2, POS_INF_F64)

    def test_architectural_memory_atomic_cmpst_f64(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        addr = 0x80
        val = f64_to_bits(42.0)
        new_val = f64_to_bits(84.0)

        lds.write_qword(addr, val)
        old, mem, swapped = lds.atomic_cmpst_f64(addr, val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, val)
        self.assertEqual(mem, new_val)
        self.assertEqual(lds.read_qword(addr), new_val)



if __name__ == "__main__":
    unittest.main()
