from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_max_src2_u32.

RDNA2 / GFX10 Local Data Share (LDS) Atomic Dual-Source Unsigned 32-bit Maximum:
Atomically reads two 32-bit operands from LDS memory locations A and B (computed
via base address VGPR and two 8-bit dword immediate offsets offset0 and offset1),
computes their unsigned 32-bit maximum, and writes the maximum back into LDS location A.
Location B remains unmodified unless Location A and Location B are identical.
Non-returning: VGPRs are not modified.

Syntax:
    ds_max_src2_u32 vaddr [offset0:offset8] [offset1:offset8]

Architectural Semantics:
    addr_a = (vaddr + offset0 * 4) & 0xFFFFFFFF
    addr_b = (vaddr + offset1 * 4) & 0xFFFFFFFF
    idx_a = (addr_a >> 2) & (lds_dwords - 1)
    idx_b = (addr_b >> 2) & (lds_dwords - 1)
    val_a = LDS[idx_a]
    val_b = LDS[idx_b]
    new_val_a = max(val_a, val_b)
    LDS[idx_a] = new_val_a

Unsigned 32-bit Integer Properties:
    - Range: [0, 2^32 - 1] ([0, 4294967295])
    - UINT32_MIN = 0
    - UINT32_MAX = 0xFFFFFFFF
    - Unsigned comparison: 0xFFFFFFFF > 0, 0x80000000 > 0x7FFFFFFF
"""

import math
import re
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
MASK32: int = 0xFFFF_FFFF           # 32-bit unsigned mask

UINT32_MIN: int = 0
UINT32_MAX: int = 0xFFFF_FFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def ref_ds_max_src2_u32(val_a: int, val_b: int) -> int:
    """Pure mathematical reference oracle for ds_max_src2_u32.

    Computes:
        max(val_a, val_b) in unsigned 32-bit integer arithmetic.
    """
    ua = u32(val_a)
    ub = u32(val_b)
    return max(ua, ub)


def calc_lds_src2_addresses(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_dwords: int = LDS_SIZE_DWORDS,
) -> Tuple[int, int, int, int]:
    """Calculate byte addresses and dword indices for address A and address B.

    offset0 and offset1 are 8-bit unsigned integers representing dword counts (scaled by 4).
    """
    byte_addr_a = (vaddr + ((offset0 & 0xFF) * 4)) & MASK32
    byte_addr_b = (vaddr + ((offset1 & 0xFF) * 4)) & MASK32
    dword_idx_a = (byte_addr_a >> 2) & (lds_dwords - 1)
    dword_idx_b = (byte_addr_b >> 2) & (lds_dwords - 1)
    return byte_addr_a, byte_addr_b, dword_idx_a, dword_idx_b


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0..31) for an effective byte address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


class LDSMemory:
    """Architectural memory model representing LDS storage with dual-source atomic operations."""

    def __init__(self, size_dwords: int = LDS_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a positive power of two, got {size_dwords}")
        self.size_dwords = size_dwords
        self.memory = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        idx = (eff_addr >> 2) & (self.size_dwords - 1)
        return self.memory[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        idx = (eff_addr >> 2) & (self.size_dwords - 1)
        self.memory[idx] = u32(val)

    def atomic_max_src2_u32(
        self,
        vaddr: int,
        offset0: int = 0,
        offset1: int = 0,
    ) -> int:
        """Execute atomic dual-source unsigned maximum on LDS: LDS[A] = max(LDS[A], LDS[B])."""
        _, _, idx_a, idx_b = calc_lds_src2_addresses(vaddr, offset0, offset1, self.size_dwords)
        val_a = self.memory[idx_a]
        val_b = self.memory[idx_b]
        new_val_a = ref_ds_max_src2_u32(val_a, val_b)
        self.memory[idx_a] = new_val_a
        return new_val_a


class TestLDSMaxSrc2U32OracleAndModel(unittest.TestCase):
    """Phase 1: Unit tests for unsigned 32-bit dual-source oracle and LDS model."""

    def test_oracle_unsigned_basic_ordering(self) -> None:
        self.assertEqual(ref_ds_max_src2_u32(10, 20), 20)
        self.assertEqual(ref_ds_max_src2_u32(20, 10), 20)
        self.assertEqual(ref_ds_max_src2_u32(500, 500), 500)

    def test_oracle_high_bit_unsigned_semantics(self) -> None:
        # In unsigned: 0x80000000 > 0x7FFFFFFF (unlike signed where 0x80000000 is negative)
        self.assertEqual(ref_ds_max_src2_u32(0x7FFFFFFF, 0x80000000), 0x80000000)
        self.assertEqual(ref_ds_max_src2_u32(0x80000000, 0x7FFFFFFF), 0x80000000)

    def test_oracle_uint32_extremes(self) -> None:
        self.assertEqual(ref_ds_max_src2_u32(0, UINT32_MAX), UINT32_MAX)
        self.assertEqual(ref_ds_max_src2_u32(UINT32_MAX, 0), UINT32_MAX)
        self.assertEqual(ref_ds_max_src2_u32(UINT32_MAX - 1, UINT32_MAX), UINT32_MAX)

    def test_calc_lds_src2_addresses_offsets(self) -> None:
        base = 0x100
        byte_a, byte_b, idx_a, idx_b = calc_lds_src2_addresses(base, offset0=2, offset1=4)
        self.assertEqual(byte_a, 0x100 + 2 * 4)  # 0x108
        self.assertEqual(byte_b, 0x100 + 4 * 4)  # 0x110
        self.assertEqual(idx_a, 0x108 >> 2)
        self.assertEqual(idx_b, 0x110 >> 2)

    def test_lds_memory_atomic_max_src2_u32(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        base = 0x40
        # Initialize location A (base + 0*4) with 100, location B (base + 2*4) with 250
        lds.write_dword(base, 100)
        lds.write_dword(base + 8, 250)

        res = lds.atomic_max_src2_u32(base, offset0=0, offset1=2)
        self.assertEqual(res, 250)
        # Location A updated with max(100, 250) = 250
        self.assertEqual(lds.read_dword(base), 250)
        # Location B remains 250
        self.assertEqual(lds.read_dword(base + 8), 250)

    def test_lds_memory_atomic_max_src2_u32_identical_addresses(self) -> None:
        lds = LDSMemory(size_dwords=1024)
        base = 0x60
        lds.write_dword(base, 77)

        res = lds.atomic_max_src2_u32(base, offset0=1, offset1=1)
        self.assertEqual(res, 0)  # Memory at base + 4 was 0, max(0, 0) = 0


if __name__ == "__main__":
    unittest.main()
