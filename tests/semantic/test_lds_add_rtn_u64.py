from __future__ import annotations

"""Reference oracle and architectural model for ds_add_rtn_u64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Unsigned Addition with Return:
Performs a 64-bit atomic addition on LDS memory and returns the original (pre-operation) value.

Syntax:
    ds_add_rtn_u64 vdst, vaddr, vdata0 [offset:offset16] [gds]

Operands:
    vdst: VGPR pair (v[d:d+1]) receiving the pre-operation 64-bit qword.
    vaddr: VGPR (v[a]) containing the byte address into LDS.
    vdata0: VGPR pair (v[s:s+1]) or 64-bit immediate to add.
    offset: 16-bit unsigned immediate byte offset.

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    dword_index_hi = (dword_index + 1) & (lds_dwords - 1)
    old_val_64 = (LDS[dword_index_hi] << 32) | LDS[dword_index]
    new_val_64 = (old_val_64 + vdata0) & 0xFFFFFFFFFFFFFFFF
    LDS[dword_index] = new_val_64 & 0xFFFFFFFF
    LDS[dword_index_hi] = (new_val_64 >> 32) & 0xFFFFFFFF
    vdst = old_val_64
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
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF  # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK_64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK_64
    return val - (1 << 64) if (val & 0x8000_0000_0000_0000) else val


def split_u64_to_u32s(val: int) -> Tuple[int, int]:
    """Split 64-bit qword into (low_u32, high_u32)."""
    val = u64(val)
    return val & MASK_32, (val >> 32) & MASK_32


def combine_u32s_to_u64(lo: int, hi: int) -> int:
    """Combine (low_u32, high_u32) into a 64-bit qword."""
    return ((hi & MASK_32) << 32) | (lo & MASK_32)


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index."""
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0..31) for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_add_rtn_u64(current_mem: int, data: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_add_rtn_u64.

    Returns:
        (old_val, new_val)
    """
    old_val = u64(current_mem)
    add_val = u64(data)
    new_val = u64(old_val + add_val)
    return old_val, new_val


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords = capacity_dwords
        self.memory = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        self.memory[idx] = u32(val)

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword (2 consecutive dwords) from LDS."""
        idx_lo = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        idx_hi = (idx_lo + 1) & (self.capacity_dwords - 1)
        return combine_u32s_to_u64(self.memory[idx_lo], self.memory[idx_hi])

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword (2 consecutive dwords) to LDS."""
        idx_lo = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        idx_hi = (idx_lo + 1) & (self.capacity_dwords - 1)
        lo, hi = split_u64_to_u32s(val)
        self.memory[idx_lo] = lo
        self.memory[idx_hi] = hi

    def atomic_add_u64(self, vaddr: int, data: int, offset: int = 0) -> Tuple[int, int]:
        """Execute atomic 64-bit unsigned addition with return on LDS.

        Returns (old_val, new_val).
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_val = oracle_ds_add_rtn_u64(curr, data)
        self.write_qword(eff_addr, new_val)
        return old_val, new_val


class TestLdsAddRtnU64ReferenceOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSModel()

    def test_add_identity_zero(self) -> None:
        """Adding 0 to 64-bit integer leaves value unchanged and returns original."""
        addr = 0x40
        init_val = 0x1234_5678_9ABC_DEF0
        self.lds.write_qword(addr, init_val)
        old_val, new_val = self.lds.atomic_add_u64(addr, 0)
        self.assertEqual(old_val, init_val)
        self.assertEqual(new_val, init_val)
        self.assertEqual(self.lds.read_qword(addr), init_val)

    def test_add_no_carry(self) -> None:
        """Direct 64-bit addition without carry across dword boundary."""
        addr = 0x80
        val1 = 0x1111_2222_1111_2222
        val2 = 0x2222_1111_2222_1111
        expected = val1 + val2
        self.lds.write_qword(addr, val1)
        old_val, new_val = self.lds.atomic_add_u64(addr, val2)
        self.assertEqual(old_val, val1)
        self.assertEqual(new_val, expected)
        self.assertEqual(self.lds.read_qword(addr), expected)

    def test_add_carry_across_dword_boundary(self) -> None:
        """Carry propagates from low 32 bits into high 32 bits: 0x0_FFFF_FFFF + 1."""
        addr = 0x100
        val1 = 0x0000_0000_FFFF_FFFF
        add_val = 0x0000_0000_0000_0001
        expected = 0x0000_0001_0000_0000
        self.lds.write_qword(addr, val1)
        old_val, new_val = self.lds.atomic_add_u64(addr, add_val)
        self.assertEqual(old_val, val1)
        self.assertEqual(new_val, expected)
        self.assertEqual(self.lds.read_qword(addr), expected)

    def test_add_overflow_modulo_2_pow_64(self) -> None:
        """Unsigned 64-bit addition overflow wraps strictly modulo 2^64: MASK_64 + 1 == 0."""
        addr = 0x120
        self.lds.write_qword(addr, MASK_64)
        old_val, new_val = self.lds.atomic_add_u64(addr, 1)
        self.assertEqual(old_val, MASK_64)
        self.assertEqual(new_val, 0)
        self.assertEqual(self.lds.read_qword(addr), 0)

    def test_offset_calculation_and_wrap(self) -> None:
        """Offset is 16-bit unsigned and added to vaddr modulo 2^32."""
        base = 0x1000
        offset = 0x28
        eff = calc_lds_effective_address(base, offset)
        self.assertEqual(eff, 0x1028)

        eff_masked = calc_lds_effective_address(base, 0x1_0008)
        self.assertEqual(eff_masked, 0x1008)

        eff_wrap = calc_lds_effective_address(0xFFFF_FFF0, 0x0020)
        self.assertEqual(eff_wrap, 0x0000_0010)

    def test_dword_alignment_truncation(self) -> None:
        """Low 2 bits of effective address are ignored for dword indexing."""
        idx0 = calc_lds_dword_index(0x400)
        idx1 = calc_lds_dword_index(0x401)
        idx2 = calc_lds_dword_index(0x402)
        idx3 = calc_lds_dword_index(0x403)
        self.assertEqual(idx0, 0x100)
        self.assertEqual(idx1, 0x100)
        self.assertEqual(idx2, 0x100)
        self.assertEqual(idx3, 0x100)

    def test_bank_distribution(self) -> None:
        """32 consecutive dwords map to 32 distinct LDS banks [0..31]."""
        banks = [calc_lds_bank(i * 4) for i in range(32)]
        self.assertEqual(banks, list(range(32)))
        self.assertEqual(calc_lds_bank(32 * 4), 0)

    def test_lds_capacity_wrap(self) -> None:
        """LDS access wraps modulo LDS capacity (default 64 KiB = 16384 dwords)."""
        addr_base = 0x10
        addr_wrapped = addr_base + LDS_SIZE_BYTES
        idx_base = calc_lds_dword_index(addr_base)
        idx_wrapped = calc_lds_dword_index(addr_wrapped)
        self.assertEqual(idx_base, idx_wrapped)


if __name__ == "__main__":
    unittest.main()
