from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_cmpst_f32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 32-bit Floating-Point Compare-and-Swap:
Performs an atomic IEEE 754 single-precision float comparison on LDS memory and swaps if equal.

Syntax:
    ds_cmpst_f32 vaddr, vdata0, vdata1 [offset:offset16] [gds]
    ds_cmpst_rtn_f32 vdst, vaddr, vdata0, vdata1 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_raw = LDS[dword_index]
    if float_equal_f32(old_raw, vdata0):
        LDS[dword_index] = vdata1
    if returning:
        vdst = old_raw
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
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask

# IEEE 754 Single-Precision Boundary Constants
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001
MIN_NORMAL_F32: int = 0x00800000
MIN_SUBNORMAL_F32: int = 0x00000001
MAX_SUBNORMAL_F32: int = 0x007FFFFF
MAX_NORMAL_F32: int = 0x7F7FFFFF
ONE_F32: int = 0x3F800000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def bits_to_f32(bits: int) -> float:
    """Interpret 32-bit unsigned integer as IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Pack IEEE 754 single-precision float into 32-bit unsigned integer."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_f32_nan(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 NaN."""
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_f32_subnormal(bits: int) -> bool:
    """Test if 32-bit bit pattern is an IEEE 754 subnormal float."""
    return (bits & 0x7F800000) == 0 and (bits & 0x007FFFFF) != 0


def float_equal_f32(a_bits: int, b_bits: int) -> bool:
    """IEEE 754 single-precision equality comparison.

    NaNs compare unordered (always False).
    Signed zeros (+0.0 and -0.0) compare equal.
    """
    if is_f32_nan(a_bits) or is_f32_nan(b_bits):
        return False
    return bits_to_f32(a_bits) == bits_to_f32(b_bits)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def ref_ds_cmpst_f32(
    current_mem: int,
    cmp_val: int,
    new_val: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_cmpst_f32.

    Compares current_mem with cmp_val using IEEE 754 float equality;
    if equal, replaces memory with new_val.
    Always returns (old_mem, new_mem, swapped).
    """
    old_raw = u32(current_mem)
    cmp_raw = u32(cmp_val)
    new_raw = u32(new_val)

    if float_equal_f32(old_raw, cmp_raw):
        return old_raw, new_raw, True
    return old_raw, old_raw, False


class LDSMemory:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        self.memory[idx] = u32(val)

    def atomic_cmpst_f32(
        self,
        vaddr: int,
        cmp_val: int,
        new_val: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        curr = self.memory[idx]
        old_val, new_mem, swapped = ref_ds_cmpst_f32(curr, cmp_val, new_val)
        self.memory[idx] = new_mem
        return old_val, new_mem, swapped



class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_finite_float_match(self) -> None:
        val = f32_to_bits(3.14159)
        new_val = f32_to_bits(2.71828)
        old, mem, swapped = ref_ds_cmpst_f32(val, val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, val)
        self.assertEqual(mem, new_val)

    def test_oracle_finite_float_mismatch(self) -> None:
        val = f32_to_bits(10.0)
        cmp_val = f32_to_bits(20.0)
        old, mem, swapped = ref_ds_cmpst_f32(val, cmp_val, f32_to_bits(50.0))
        self.assertFalse(swapped)
        self.assertEqual(old, val)
        self.assertEqual(mem, val)

    def test_oracle_signed_zero_equality(self) -> None:
        # In IEEE 754, +0.0 == -0.0 is True!
        old, mem, swapped = ref_ds_cmpst_f32(NEG_ZERO_F32, POS_ZERO_F32, f32_to_bits(1.0))
        self.assertTrue(swapped)
        self.assertEqual(old, NEG_ZERO_F32)
        self.assertEqual(mem, f32_to_bits(1.0))

        old2, mem2, swapped2 = ref_ds_cmpst_f32(POS_ZERO_F32, NEG_ZERO_F32, f32_to_bits(2.0))
        self.assertTrue(swapped2)
        self.assertEqual(old2, POS_ZERO_F32)
        self.assertEqual(mem2, f32_to_bits(2.0))

    def test_oracle_nan_unordered_rejection(self) -> None:
        # NaN == NaN evaluates to False
        old, mem, swapped = ref_ds_cmpst_f32(QNAN_F32, QNAN_F32, f32_to_bits(1.0))
        self.assertFalse(swapped)
        self.assertEqual(mem, QNAN_F32)

    def test_oracle_infinities(self) -> None:
        # +inf == +inf is True, -inf == -inf is True, +inf == -inf is False
        old, mem, swapped = ref_ds_cmpst_f32(POS_INF_F32, POS_INF_F32, 0)
        self.assertTrue(swapped)
        self.assertEqual(mem, 0)

        old2, mem2, swapped2 = ref_ds_cmpst_f32(POS_INF_F32, NEG_INF_F32, 0)
        self.assertFalse(swapped2)
        self.assertEqual(mem2, POS_INF_F32)

    def test_effective_address_and_alignment(self) -> None:
        eff = calc_lds_effective_address(0x1000, 0x14)
        self.assertEqual(eff, 0x1014)
        self.assertEqual(calc_lds_dword_index(eff), 0x1014 >> 2)

    def test_architectural_memory_atomic_cmpst_f32(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        addr = 0x80
        val = f32_to_bits(42.0)
        new_val = f32_to_bits(84.0)

        lds.write_dword(addr, val)
        old, mem, swapped = lds.atomic_cmpst_f32(addr, val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, val)
        self.assertEqual(mem, new_val)
        self.assertEqual(lds.read_dword(addr), new_val)



if __name__ == "__main__":
    unittest.main()
