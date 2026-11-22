"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_nor_rtn_b32.

Models 32-bit bitwise NOR and AMD RDNA2 / GFX10 Local Data Share (LDS)
architectural semantics:
  - ds_nor_rtn_b32: Atomic 32-bit bitwise NOR returning previous value to destination VGPR.
  - ds_nor_b32: Atomic 32-bit bitwise NOR discarding previous value.

RDNA2 LDS / Bitwise NOR ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip scratchpad memory shared across all threads in a workgroup.
     Addressing is byte-addressed but dword-aligned in hardware (low 2 bits dropped).
     Out-of-bounds addresses wrap within the allocated power-of-two LDS slice.
     LDS is organized into 32 memory banks (bank = dword_index % 32).
  2. Bitwise NOR Logic:
     Operands are treated as 32-bit bitmasks.
     nor(A, B) = ~(A | B) & 0xFFFFFFFF.
     Bitwise truth table:
       0 NOR 0 = 1
       0 NOR 1 = 0
       1 NOR 0 = 0
       1 NOR 1 = 0
     0 NOR 0 = 0xFFFFFFFF; 0xFFFFFFFF NOR any = 0; 0 NOR x = ~x.
  3. Atomic Operations & Return Semantics:
     Returning variants (ds_nor_rtn_b32) atomically fetch the previous 32-bit value,
     write the bitwise NOR into LDS, and return the previous value to the destination VGPR.
     Non-returning variants (ds_nor_b32) perform the atomic NOR and discard the old value.

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

MASK32: int = 0xFFFFFFFF
UINT32_MIN: int = 0x00000000
UINT32_MAX: int = 0xFFFFFFFF
OFFSET16_MASK: int = 0xFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384  # 16,384 32-bit dwords
NUM_LDS_BANKS: int = 32


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def oracle_nor_b32(a: int, b: int) -> int:
    """Reference 32-bit bitwise NOR returning 32-bit unsigned bit pattern."""
    ua = a & MASK32
    ub = b & MASK32
    return (~(ua | ub)) & MASK32


def ref_ds_nor_rtn_b32(current_mem: int, operand: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_nor_rtn_b32.

    Returns:
        (old_val, new_mem) as unsigned 32-bit integers.
    """
    old_val = u32(current_mem)
    new_mem = oracle_nor_b32(old_val, operand)
    return old_val, new_mem


class LDSMemory:
    """Architectural simulation model of AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of 2, got {size_bytes}")
        self.size_bytes: int = size_bytes
        self.num_dwords: int = size_bytes // 4
        self.dwords: List[int] = [0] * self.num_dwords

    def get_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        """Compute effective wrapped dword index, dropping bottom 2 unaligned bits."""
        effective_byte = (byte_addr + (offset & OFFSET16_MASK)) & MASK32
        return (effective_byte >> 2) & (self.num_dwords - 1)

    def get_bank(self, dword_idx: int) -> int:
        """Compute memory bank index for the given dword (0 to 31)."""
        return dword_idx % NUM_LDS_BANKS

    def read_dword(self, dword_idx: int) -> int:
        """Read a 32-bit dword from LDS."""
        return self.dwords[dword_idx & (self.num_dwords - 1)]

    def write_dword(self, dword_idx: int, val: int) -> None:
        """Write a 32-bit dword into LDS."""
        self.dwords[dword_idx & (self.num_dwords - 1)] = u32(val)

    def atomic_nor_b32(self, byte_addr: int, val: int, offset: int = 0) -> int:
        """Execute atomic bitwise NOR at (byte_addr + offset), returning new value."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_nor_b32(old_val, val)
        self.dwords[idx] = new_val
        return new_val

    def atomic_nor_rtn_b32(self, byte_addr: int, val: int, offset: int = 0) -> Tuple[int, int]:
        """Execute atomic bitwise NOR at (byte_addr + offset), returning (old_val, new_val)."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        new_val = oracle_nor_b32(old_val, val)
        self.dwords[idx] = new_val
        return old_val, new_val


class TestLDSNorRtnB32ModelAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)

    def test_oracle_nor_b32_truth_table(self) -> None:
        # 0 NOR 0 = 0xFFFFFFFF
        self.assertEqual(oracle_nor_b32(0, 0), 0xFFFFFFFF)
        # 0xFFFFFFFF NOR 0 = 0
        self.assertEqual(oracle_nor_b32(0xFFFFFFFF, 0), 0)
        # 0 NOR 0xFFFFFFFF = 0
        self.assertEqual(oracle_nor_b32(0, 0xFFFFFFFF), 0)
        # 0xFFFFFFFF NOR 0xFFFFFFFF = 0
        self.assertEqual(oracle_nor_b32(0xFFFFFFFF, 0xFFFFFFFF), 0)

    def test_oracle_nor_b32_bit_patterns(self) -> None:
        # Alternating patterns: 0x55555555 | 0xAAAAAAAA = 0xFFFFFFFF -> ~(0xFFFFFFFF) = 0
        self.assertEqual(oracle_nor_b32(0x55555555, 0xAAAAAAAA), 0)
        # 0x55555555 NOR 0 = ~0x55555555 = 0xAAAAAAAA
        self.assertEqual(oracle_nor_b32(0x55555555, 0), 0xAAAAAAAA)
        # 0x0F0F0F0F NOR 0x00FF00FF = ~(0x0FFF0FFF) = 0xF000F000
        self.assertEqual(oracle_nor_b32(0x0F0F0F0F, 0x00FF00FF), 0xF000F000)

    def test_ref_ds_nor_rtn_b32(self) -> None:
        old_val, new_val = ref_ds_nor_rtn_b32(0x00000000, 0x00000000)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 0xFFFFFFFF)

        old_val2, new_val2 = ref_ds_nor_rtn_b32(0x12345678, 0x87654321)
        expected = (~(0x12345678 | 0x87654321)) & MASK32
        self.assertEqual(old_val2, 0x12345678)
        self.assertEqual(new_val2, expected)

    def test_lds_memory_addressing_and_wrapping(self) -> None:
        # Byte address 0x10 -> dword index 4
        self.assertEqual(self.lds.get_dword_index(0x10), 4)
        # Unaligned low 2 bits are dropped (0x13 -> dword index 4)
        self.assertEqual(self.lds.get_dword_index(0x13), 4)
        # Offset addition
        self.assertEqual(self.lds.get_dword_index(0x10, offset=4), 5)
        # Wrap around 64 KiB (65536 bytes = 16384 dwords)
        self.assertEqual(self.lds.get_dword_index(65536), 0)
        self.assertEqual(self.lds.get_dword_index(65540), 1)

    def test_lds_memory_banks(self) -> None:
        for i in range(64):
            self.assertEqual(self.lds.get_bank(i), i % NUM_LDS_BANKS)

    def test_lds_read_write(self) -> None:
        self.lds.write_dword(10, 0xAABBCCDD)
        self.assertEqual(self.lds.read_dword(10), 0xAABBCCDD)

    def test_lds_atomic_nor_operations(self) -> None:
        # Set initial value to 0x55555555 at dword 8 (byte addr 32)
        self.lds.write_dword(8, 0x55555555)
        old_val, new_val = self.lds.atomic_nor_rtn_b32(32, 0)
        self.assertEqual(old_val, 0x55555555)
        self.assertEqual(new_val, 0xAAAAAAAA)
        self.assertEqual(self.lds.read_dword(8), 0xAAAAAAAA)

        # Atomic nor with 0xAAAAAAAA: ~(0xAAAAAAAA | 0xAAAAAAAA) = ~0xAAAAAAAA = 0x55555555
        old_val2, new_val2 = self.lds.atomic_nor_rtn_b32(32, 0xAAAAAAAA)
        self.assertEqual(old_val2, 0xAAAAAAAA)
        self.assertEqual(new_val2, 0x55555555)
        self.assertEqual(self.lds.read_dword(8), 0x55555555)


if __name__ == "__main__":
    unittest.main()
