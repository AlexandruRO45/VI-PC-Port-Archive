"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_max_rtn_u32.

Models AMD RDNA2 / GFX10 Local Data Share (LDS) atomic unsigned 32-bit maximum with return:
    ds_max_rtn_u32 vdst, vaddr, vdata [offset:offset16]

RDNA2 LDS / Unsigned 32-bit Integer ISA Semantics:
  1. Local Data Share Architecture:
     LDS is an on-chip, low-latency scratchpad memory shared across all threads in a workgroup.
     Addressing is byte-addressed, but memory is organized into 32-bit dwords across 32 banks.
     LDS accesses are dword-aligned in hardware: bits [1:0] of the effective byte address are dropped.
     Addresses wrap modulo the allocated power-of-two LDS capacity.
  2. Addressing and Effective Offset:
     effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
     dword_index = (effective_byte_address >> 2) & (capacity_dwords - 1)
  3. Atomic Unsigned Maximum & Return Value:
     old_val = LDS[dword_index]
     new_val = max(old_val, vdata)  # unsigned 32-bit integer comparison
     LDS[dword_index] = new_val
     vdst = old_val  # previous memory value returned to destination VGPR
  4. Unsigned Integer Ordering:
     Operands are treated as unsigned 32-bit integers in [0, 2^32 - 1].
     Strictly: 0 < 1 < 0x7FFFFFFF < 0x80000000 < 0xFFFFFFFF.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 LDS Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent banks
MASK32: int = 0xFFFFFFFF            # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask
SIGN_BIT32: int = 0x80000000        # 32-bit sign bit / MSB


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS with 16-bit offset."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are dropped by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0-31) for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_max_u32(a: int, b: int) -> int:
    """Reference pure unsigned 32-bit maximum."""
    ua = a & MASK32
    ub = b & MASK32
    return ua if ua >= ub else ub


def oracle_ds_max_rtn_u32(
    current_mem: int,
    vdata: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_max_rtn_u32.

    Returns:
        (returned_old_val, new_mem_val, modified)
    """
    old_val = current_mem & MASK32
    val = vdata & MASK32
    new_val = oracle_max_u32(old_val, val)
    return old_val, new_val, (new_val != old_val)


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_dwords: int = LDS_SIZE_DWORDS) -> None:
        if capacity_dwords <= 0 or (capacity_dwords & (capacity_dwords - 1)) != 0:
            raise ValueError(f"capacity_dwords must be a power of two, got {capacity_dwords}")
        self.capacity_dwords = capacity_dwords
        self.memory: List[int] = [0] * capacity_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        self.memory[idx] = val & MASK32

    def atomic_max_rtn_u32(
        self,
        vaddr: int,
        vdata: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic unsigned maximum on LDS dword returning previous value.

        Returns:
            (old_val, new_val, modified)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_val, modified = oracle_ds_max_rtn_u32(curr, vdata)
        self.memory[idx] = new_val
        return old_val, new_val, modified


class TestLdsMaxRtnU32ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_max_u32_standard(self) -> None:
        self.assertEqual(oracle_max_u32(10, 20), 20)
        self.assertEqual(oracle_max_u32(50, 30), 50)
        self.assertEqual(oracle_max_u32(100, 100), 100)

    def test_oracle_max_u32_unsigned_boundaries(self) -> None:
        # Unsigned 0 is less than 0xFFFFFFFF
        self.assertEqual(oracle_max_u32(0, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(oracle_max_u32(0xFFFFFFFF, 0), 0xFFFFFFFF)
        # 0x80000000 (signed -2147483648) > 0x7FFFFFFF (signed +2147483647) in unsigned comparison
        self.assertEqual(oracle_max_u32(0x80000000, 0x7FFFFFFF), 0x80000000)
        self.assertEqual(oracle_max_u32(0x7FFFFFFF, 0x80000000), 0x80000000)
        # Extremes
        self.assertEqual(oracle_max_u32(0, 0), 0)
        self.assertEqual(oracle_max_u32(0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_oracle_ds_max_rtn_u32_return_and_modify(self) -> None:
        old, new, mod = oracle_ds_max_rtn_u32(100, 200)
        self.assertEqual(old, 100)
        self.assertEqual(new, 200)
        self.assertTrue(mod)

        old, new, mod = oracle_ds_max_rtn_u32(500, 300)
        self.assertEqual(old, 500)
        self.assertEqual(new, 500)
        self.assertFalse(mod)

        old, new, mod = oracle_ds_max_rtn_u32(500, 500)
        self.assertEqual(old, 500)
        self.assertEqual(new, 500)
        self.assertFalse(mod)

    def test_effective_address_calculation(self) -> None:
        # vaddr + offset
        eff = calc_lds_effective_address(0x1000, 0x0040)
        self.assertEqual(eff, 0x1040)
        # 16-bit offset masking
        eff = calc_lds_effective_address(0x1000, 0x10040)
        self.assertEqual(eff, 0x1040)
        # 32-bit modulo wrapping
        eff = calc_lds_effective_address(0xFFFFFFFF, 0x0001)
        self.assertEqual(eff, 0x0000)

    def test_dword_alignment_and_bit_truncation(self) -> None:
        # LDS drops low 2 bits
        idx0 = calc_lds_dword_index(0x00)
        idx1 = calc_lds_dword_index(0x01)
        idx2 = calc_lds_dword_index(0x02)
        idx3 = calc_lds_dword_index(0x03)
        self.assertEqual(idx0, 0)
        self.assertEqual(idx1, 0)
        self.assertEqual(idx2, 0)
        self.assertEqual(idx3, 0)

        idx4 = calc_lds_dword_index(0x04)
        self.assertEqual(idx4, 1)

    def test_lds_capacity_wrapping(self) -> None:
        capacity = 1024  # 1024 dwords = 4096 bytes
        eff_addr = 4096
        self.assertEqual(calc_lds_dword_index(eff_addr, capacity), 0)
        eff_addr2 = 4096 + 12
        self.assertEqual(calc_lds_dword_index(eff_addr2, capacity), 3)

    def test_bank_indexing(self) -> None:
        self.assertEqual(calc_lds_bank(0x00), 0)
        self.assertEqual(calc_lds_bank(0x04), 1)
        self.assertEqual(calc_lds_bank(0x7C), 31)
        self.assertEqual(calc_lds_bank(0x80), 0)

    def test_lds_model_atomic_max_rtn_u32(self) -> None:
        lds = LDSModel(capacity_dwords=16384)
        addr = 0x200
        lds.write_dword(addr, 1000)
        self.assertEqual(lds.read_dword(addr), 1000)

        # Atomic max with smaller value -> old returned, memory unchanged
        old_val, new_val, mod = lds.atomic_max_rtn_u32(addr, 500)
        self.assertEqual(old_val, 1000)
        self.assertEqual(new_val, 1000)
        self.assertFalse(mod)
        self.assertEqual(lds.read_dword(addr), 1000)

        # Atomic max with larger value -> old returned, memory updated
        old_val2, new_val2, mod2 = lds.atomic_max_rtn_u32(addr, 2500)
        self.assertEqual(old_val2, 1000)
        self.assertEqual(new_val2, 2500)
        self.assertTrue(mod2)
        self.assertEqual(lds.read_dword(addr), 2500)

        # Atomic max with immediate offset
        old_val3, new_val3, mod3 = lds.atomic_max_rtn_u32(addr, 5000, offset=4)
        # offset=4 means addr 0x204 (dword index 0x81) which was initially 0
        self.assertEqual(old_val3, 0)
        self.assertEqual(new_val3, 5000)
        self.assertTrue(mod3)
        self.assertEqual(lds.read_dword(0x204), 5000)
        # original dword 0x200 remains 2500
        self.assertEqual(lds.read_dword(addr), 2500)


if __name__ == "__main__":
    unittest.main()
