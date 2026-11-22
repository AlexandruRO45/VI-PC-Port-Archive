"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_min_rtn_u64.

Models AMD RDNA2 / GFX10 Local Data Share (LDS) atomic unsigned 64-bit minimum with return:
    ds_min_rtn_u64 v[dst_lo:dst_hi], vaddr, v[data_lo:data_hi] [offset:offset16]

RDNA2 LDS / Unsigned 64-bit Integer ISA Semantics:
  1. Local Data Share Architecture:
     LDS is an on-chip, low-latency scratchpad memory shared across all threads in a workgroup.
     Addressing is byte-addressed, but organized in 32-bit dwords across 32 banks.
     64-bit operations access two consecutive 32-bit dwords (8 bytes = 1 qword).
     Accesses are aligned in hardware: bits [2:0] (qword) or [1:0] (dword) of the effective byte
     address determine storage locations.
     Addresses wrap modulo the allocated power-of-two LDS capacity.
  2. Addressing and Effective Offset:
     effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
     qword_index = (effective_byte_address >> 3) & (capacity_qwords - 1)
  3. Atomic Unsigned 64-bit Minimum & Return Value:
     old_val = LDS[qword_index]
     new_val = min(old_val, vdata)  # unsigned 64-bit integer comparison
     LDS[qword_index] = new_val
     vdst = old_val  # previous 64-bit memory value returned to destination VGPR pair
  4. Unsigned Integer Ordering:
     Operands are treated as unsigned 64-bit integers in [0, 2^64 - 1].
     Strictly: 0 < 1 < 0x7FFFFFFFFFFFFFFF < 0x8000000000000000 < 0xFFFFFFFFFFFFFFFF.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Set, Tuple, Union

# RDNA2 LDS Architectural Constants
LDS_SIZE_BYTES: int = 65536             # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384            # 16,384 32-bit dwords
LDS_SIZE_QWORDS: int = 8192             # 8,192 64-bit qwords
LDS_BANK_COUNT: int = 32                # 32 independent memory banks
MASK32: int = 0xFFFFFFFF                # 32-bit unsigned mask
MASK64: int = 0xFFFFFFFFFFFFFFFF        # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF             # 16-bit immediate byte offset mask
SIGN_BIT64: int = 0x8000000000000000    # 64-bit sign bit / MSB


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS with 16-bit offset."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index.

    64-bit LDS accesses drop the low 3 bits for qword alignment.
    The index wraps modulo the power-of-two qword capacity.
    """
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0-31) for the low dword of an address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_min_u64(a: int, b: int) -> int:
    """Reference pure unsigned 64-bit minimum."""
    ua = a & MASK64
    ub = b & MASK64
    return ua if ua <= ub else ub


def oracle_ds_min_rtn_u64(
    current_mem: int,
    vdata: int,
) -> Tuple[int, int, bool]:
    """Pure mathematical reference oracle for ds_min_rtn_u64.

    Returns:
        (returned_old_val, new_mem_val, modified)
    """
    old_val = current_mem & MASK64
    val = vdata & MASK64
    new_val = oracle_min_u64(old_val, val)
    return old_val, new_val, (new_val != old_val)


class LDSModel64:
    """Architectural memory model representing the Local Data Share scratchpad for 64-bit access."""

    def __init__(self, capacity_bytes: int = LDS_SIZE_BYTES) -> None:
        if capacity_bytes <= 0 or (capacity_bytes & (capacity_bytes - 1)) != 0:
            raise ValueError(f"capacity_bytes must be a power of two, got {capacity_bytes}")
        if capacity_bytes < 8:
            raise ValueError(f"capacity_bytes must be at least 8 bytes, got {capacity_bytes}")
        self.capacity_bytes = capacity_bytes
        self.capacity_qwords = capacity_bytes // 8
        self.memory: List[int] = [0] * self.capacity_qwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        return self.memory[idx]

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        self.memory[idx] = val & MASK64

    def atomic_min_rtn_u64(
        self,
        vaddr: int,
        vdata: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic unsigned 64-bit minimum on LDS returning previous value.

        Returns:
            (old_val, new_val, modified)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        curr = self.memory[idx]
        old_val, new_val, modified = oracle_ds_min_rtn_u64(curr, vdata)
        self.memory[idx] = new_val
        return old_val, new_val, modified


class TestLdsMinRtnU64ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_min_u64_standard(self) -> None:
        self.assertEqual(oracle_min_u64(10, 20), 10)
        self.assertEqual(oracle_min_u64(50, 30), 30)
        self.assertEqual(oracle_min_u64(100, 100), 100)

    def test_oracle_min_u64_unsigned_boundaries(self) -> None:
        # Unsigned 0 is less than 0xFFFFFFFFFFFFFFFF
        self.assertEqual(oracle_min_u64(0, 0xFFFFFFFFFFFFFFFF), 0)
        self.assertEqual(oracle_min_u64(0xFFFFFFFFFFFFFFFF, 0), 0)
        # 0x7FFFFFFFFFFFFFFF < 0x8000000000000000 in unsigned comparison
        self.assertEqual(oracle_min_u64(0x7FFFFFFFFFFFFFFF, 0x8000000000000000), 0x7FFFFFFFFFFFFFFF)
        self.assertEqual(oracle_min_u64(0x8000000000000000, 0x7FFFFFFFFFFFFFFF), 0x7FFFFFFFFFFFFFFF)
        # Extremes
        self.assertEqual(oracle_min_u64(0, 0), 0)
        self.assertEqual(oracle_min_u64(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF), 0xFFFFFFFFFFFFFFFF)

    def test_oracle_ds_min_rtn_u64_return_and_modify(self) -> None:
        old, new, mod = oracle_ds_min_rtn_u64(200, 100)
        self.assertEqual(old, 200)
        self.assertEqual(new, 100)
        self.assertTrue(mod)

        old, new, mod = oracle_ds_min_rtn_u64(300, 500)
        self.assertEqual(old, 300)
        self.assertEqual(new, 300)
        self.assertFalse(mod)

        old, new, mod = oracle_ds_min_rtn_u64(500, 500)
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

    def test_qword_alignment_and_bit_truncation(self) -> None:
        # LDS 64-bit drops low 3 bits
        for b in range(8):
            self.assertEqual(calc_lds_qword_index(b), 0)
        self.assertEqual(calc_lds_qword_index(8), 1)
        self.assertEqual(calc_lds_qword_index(16), 2)

    def test_lds_capacity_wrapping(self) -> None:
        capacity = 1024  # 1024 bytes = 128 qwords
        eff_addr = 1024
        self.assertEqual(calc_lds_qword_index(eff_addr, capacity_qwords=128), 0)
        eff_addr2 = 1024 + 16
        self.assertEqual(calc_lds_qword_index(eff_addr2, capacity_qwords=128), 2)

    def test_bank_indexing(self) -> None:
        self.assertEqual(calc_lds_bank(0x00), 0)
        self.assertEqual(calc_lds_bank(0x04), 1)
        self.assertEqual(calc_lds_bank(0x7C), 31)
        self.assertEqual(calc_lds_bank(0x80), 0)

    def test_lds_model_atomic_min_rtn_u64(self) -> None:
        lds = LDSModel64(capacity_bytes=65536)
        addr = 0x200
        initial_val = 0x0000000500000000
        lds.write_qword(addr, initial_val)
        self.assertEqual(lds.read_qword(addr), initial_val)

        # Atomic min with larger value -> old returned, memory unchanged
        larger_val = 0x0000000800000000
        old_val, new_val, mod = lds.atomic_min_rtn_u64(addr, larger_val)
        self.assertEqual(old_val, initial_val)
        self.assertEqual(new_val, initial_val)
        self.assertFalse(mod)
        self.assertEqual(lds.read_qword(addr), initial_val)

        # Atomic min with smaller value -> old returned, memory updated
        smaller_val = 0x0000000200000000
        old_val2, new_val2, mod2 = lds.atomic_min_rtn_u64(addr, smaller_val)
        self.assertEqual(old_val2, initial_val)
        self.assertEqual(new_val2, smaller_val)
        self.assertTrue(mod2)
        self.assertEqual(lds.read_qword(addr), smaller_val)

        # Atomic min with immediate offset
        offset = 8  # next qword at 0x208
        old_val3, new_val3, mod3 = lds.atomic_min_rtn_u64(addr, 0x1234, offset=offset)
        # Initially qword at 0x208 is 0; 0 is smaller than 0x1234, so result is 0
        self.assertEqual(old_val3, 0)
        self.assertEqual(new_val3, 0)
        self.assertFalse(mod3)
        self.assertEqual(lds.read_qword(0x208), 0)


if __name__ == "__main__":
    unittest.main()
