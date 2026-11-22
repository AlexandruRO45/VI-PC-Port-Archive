from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_cmpst_rtn_b64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Compare-and-Swap with Return:
Performs a 64-bit atomic compare-and-swap on LDS memory and returns the original 64-bit value.

Syntax:
    ds_cmpst_rtn_b64 vdst, vaddr, vdata0, vdata1 [offset:offset16] [gds]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    addr_lo = effective_byte_address & ~3
    old_lo = LDS[addr_lo]
    old_hi = LDS[addr_lo + 4]
    old_val64 = (old_hi << 32) | old_lo
    if old_val64 == vdata0:
        LDS[addr_lo] = vdata1 & 0xFFFFFFFF
        LDS[addr_lo + 4] = (vdata1 >> 32) & 0xFFFFFFFF
    vdst = old_val64
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


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned integer as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & 0x8000_0000_0000_0000) else val


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit unsigned integer into (lo, hi) 32-bit words."""
    val &= MASK64
    return val & MASK32, (val >> 32) & MASK32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (lo, hi) 32-bit words into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def ref_ds_cmpst_rtn_b64(
    current_mem64: int,
    cmp_val64: int,
    new_val64: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_cmpst_rtn_b64.

    Compares current_mem64 with cmp_val64; if equal, replaces memory with new_val64.
    Always returns the pre-operation original memory value.

    Returns:
        Tuple[int, int, bool]: (returned_old_val, new_mem_val, swapped)
    """
    old_val = u64(current_mem64)
    cmp_masked = u64(cmp_val64)
    new_masked = u64(new_val64)

    if old_val == cmp_masked:
        return old_val, new_masked, True
    return old_val, old_val, False


class LDSMemory:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        self.memory[idx] = u32(val)

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword (two consecutive dwords) from LDS."""
        idx_lo = calc_lds_dword_index(eff_addr, self.size_dwords)
        idx_hi = (idx_lo + 1) & (self.size_dwords - 1)
        return combine_u32s_to_u64(self.memory[idx_lo], self.memory[idx_hi])

    def write_qword(self, eff_addr: int, val64: int) -> None:
        """Write a 64-bit qword (two consecutive dwords) to LDS."""
        idx_lo = calc_lds_dword_index(eff_addr, self.size_dwords)
        idx_hi = (idx_lo + 1) & (self.size_dwords - 1)
        lo, hi = split_u64_to_u32s(val64)
        self.memory[idx_lo] = lo
        self.memory[idx_hi] = hi

    def atomic_cmpst_rtn_b64(
        self,
        vaddr: int,
        cmp_val64: int,
        new_val64: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic 64-bit compare-and-swap on LDS.

        Returns:
            Tuple[int, int, bool]: (old_val, new_mem, swapped)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b64(curr, cmp_val64, new_val64)
        if swapped:
            self.write_qword(eff_addr, new_mem)
        return old_val, new_mem, swapped


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_64bit_match_and_swap(self) -> None:
        val = 0x1122_3344_5566_7788
        new_val = 0x99AA_BBCC_DDEE_FF00
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b64(val, val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old_val, val)
        self.assertEqual(new_mem, new_val)

    def test_oracle_64bit_low_dword_mismatch(self) -> None:
        val = 0x1122_3344_5566_7788
        cmp_val = 0x1122_3344_5566_0000 # Low dword differs
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b64(val, cmp_val, 0x9999)
        self.assertFalse(swapped)
        self.assertEqual(old_val, val)
        self.assertEqual(new_mem, val)

    def test_oracle_64bit_high_dword_mismatch(self) -> None:
        val = 0x1122_3344_5566_7788
        cmp_val = 0x0000_0000_5566_7788 # High dword differs
        old_val, new_mem, swapped = ref_ds_cmpst_rtn_b64(val, cmp_val, 0x9999)
        self.assertFalse(swapped)
        self.assertEqual(old_val, val)
        self.assertEqual(new_mem, val)

    def test_dword_splitting_and_combining(self) -> None:
        val = 0xDEAD_BEEF_CAFE_BABE
        lo, hi = split_u64_to_u32s(val)
        self.assertEqual(lo, 0xCAFE_BABE)
        self.assertEqual(hi, 0xDEAD_BEEF)
        self.assertEqual(combine_u32s_to_u64(lo, hi), val)

    def test_effective_address_calculation(self) -> None:
        eff = calc_lds_effective_address(0x4000, 0x20)
        self.assertEqual(eff, 0x4020)

        eff_wrap = calc_lds_effective_address(0xFFFFFFFF, 1)
        self.assertEqual(eff_wrap, 0)

    def test_architectural_memory_qword_read_write(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        addr = 0x80
        val = 0x0123_4567_89AB_CDEF
        lds.write_qword(addr, val)

        self.assertEqual(lds.read_qword(addr), val)
        # Check individual dwords: lo at idx, hi at idx+1
        idx = calc_lds_dword_index(addr, 1024)
        self.assertEqual(lds.memory[idx], 0x89AB_CDEF)
        self.assertEqual(lds.memory[idx + 1], 0x0123_4567)

    def test_architectural_memory_64bit_boundary_wrap(self) -> None:
        # Wrap across LDS capacity: last dword at size - 1, second dword wraps to 0
        lds = LDSMemory(size_dwords=1024)
        last_byte_addr = (1024 - 1) * 4 # 0xFFC
        val = 0xAAAA_BBBB_CCCC_DDDD
        lds.write_qword(last_byte_addr, val)

        self.assertEqual(lds.read_qword(last_byte_addr), val)
        self.assertEqual(lds.memory[1023], 0xCCCC_DDDD)
        self.assertEqual(lds.memory[0], 0xAAAA_BBBB)

    def test_architectural_atomic_cmpst_b64(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        addr = 0x100
        initial_val = 0x5555_6666_7777_8888
        new_val = 0x1111_2222_3333_4444

        lds.write_qword(addr, initial_val)

        # Successful 64-bit CAS
        old, mem, swapped = lds.atomic_cmpst_rtn_b64(vaddr=addr, cmp_val64=initial_val, new_val64=new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, initial_val)
        self.assertEqual(mem, new_val)
        self.assertEqual(lds.read_qword(addr), new_val)

        # Failed 64-bit CAS
        old2, mem2, swapped2 = lds.atomic_cmpst_rtn_b64(vaddr=addr, cmp_val64=0x9999, new_val64=0xAAAA)
        self.assertFalse(swapped2)
        self.assertEqual(old2, new_val)
        self.assertEqual(mem2, new_val)
        self.assertEqual(lds.read_qword(addr), new_val)


if __name__ == "__main__":
    unittest.main()
