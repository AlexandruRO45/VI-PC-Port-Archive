"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_condxchg32_rtn_b64.

Models 64-bit Local Data Share (LDS) conditional exchange (compare-and-swap)
and AMD RDNA2 / GFX10 Local Data Share (LDS) architectural semantics:
  - ds_condxchg32_rtn_b64 (also known as ds_cmpst_rtn_b64):
    Atomic 64-bit conditional exchange on LDS storage returning previous 64-bit value.

RDNA2 LDS / 64-Bit Conditional Exchange ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip scratchpad memory shared across all threads in a workgroup.
     A 64-bit element occupies two consecutive 32-bit dwords: low dword at dword_idx,
     high dword at (dword_idx + 1).
     Addressing is byte-addressed but dword-aligned in hardware (low 2 bits dropped).
     Out-of-bounds addresses wrap within the allocated power-of-two LDS slice.
     LDS is organized into 32 memory banks (bank = dword_index % 32).
  2. 64-Bit Conditional Exchange Semantics:
     Memory value M[addr] is compared with 64-bit cmp_val.
     If M[addr] == cmp_val:
         M[addr] = new_val
         exchanged = True
     Else:
         exchanged = False
  3. Return Semantics:
     The instruction atomically reads the existing 64-bit qword from LDS and writes
     it into destination VGPR pair vdst[0:1] regardless of whether the comparison succeeded.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Tuple, Set

MASK64: int = 0xFFFFFFFFFFFFFFFF
MASK32: int = 0xFFFFFFFF
ALL_ONES_B64: int = 0xFFFFFFFFFFFFFFFF
ALL_ZEROS_B64: int = 0x0000000000000000

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384
NUM_LDS_BANKS: int = 32


def split64(val: int) -> Tuple[int, int]:
    """Split 64-bit integer into (lo32, hi32)."""
    return val & MASK32, (val >> 32) & MASK32


def join64(lo: int, hi: int) -> int:
    """Combine two 32-bit integers into unsigned 64-bit integer."""
    return ((lo & MASK32) | ((hi & MASK32) << 32)) & MASK64


def oracle_condxchg64(
    mem_val: int,
    cmp_val: int,
    new_val: int,
) -> Tuple[int, int, bool]:
    """Reference mathematical oracle for 64-bit conditional exchange.
    
    Returns:
        (old_val, new_mem_val, exchanged)
    """
    m = mem_val & MASK64
    c = cmp_val & MASK64
    n = new_val & MASK64

    if m == c:
        return m, n, True
    return m, m, False


class LDSMemory:
    """Architectural simulation model of AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of 2, got {size_bytes}")
        self.size_bytes: int = size_bytes
        self.num_dwords: int = size_bytes // 4
        self.dwords: List[int] = [0] * self.num_dwords

    def get_dword_indices(self, byte_addr: int, offset: int = 0) -> Tuple[int, int]:
        """Compute effective wrapped dword indices for a 64-bit (2 dword) access."""
        effective_byte = (byte_addr + offset) & MASK32
        base_idx = (effective_byte >> 2) & (self.num_dwords - 1)
        hi_idx = (base_idx + 1) & (self.num_dwords - 1)
        return base_idx, hi_idx

    def get_banks(self, idx0: int, idx1: int) -> Tuple[int, int]:
        """Compute memory bank indices for the given dword pair."""
        return idx0 % NUM_LDS_BANKS, idx1 % NUM_LDS_BANKS

    def read_qword(self, byte_addr: int, offset: int = 0) -> int:
        """Read a 64-bit qword from LDS."""
        idx0, idx1 = self.get_dword_indices(byte_addr, offset)
        lo = self.dwords[idx0]
        hi = self.dwords[idx1]
        return join64(lo, hi)

    def write_qword(self, byte_addr: int, val: int, offset: int = 0) -> None:
        """Write a 64-bit qword into LDS."""
        idx0, idx1 = self.get_dword_indices(byte_addr, offset)
        lo, hi = split64(val)
        self.dwords[idx0] = lo
        self.dwords[idx1] = hi

    def atomic_condxchg64(
        self,
        byte_addr: int,
        cmp_val: int,
        new_val: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic 64-bit conditional exchange at (byte_addr + offset)."""
        idx0, idx1 = self.get_dword_indices(byte_addr, offset)
        lo = self.dwords[idx0]
        hi = self.dwords[idx1]
        old_val = join64(lo, hi)

        old, new_mem, exchanged = oracle_condxchg64(old_val, cmp_val, new_val)
        if exchanged:
            new_lo, new_hi = split64(new_mem)
            self.dwords[idx0] = new_lo
            self.dwords[idx1] = new_hi

        return old, new_mem, exchanged


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for reference 64-bit conditional exchange oracle and LDS model."""

    def test_split_and_join_roundtrip(self) -> None:
        values = [0x0, 0x1, 0xFFFFFFFF, 0x123456789ABCDEF0, 0xFFFFFFFFFFFFFFFF]
        for v in values:
            with self.subTest(v=hex(v)):
                lo, hi = split64(v)
                self.assertEqual(join64(lo, hi), v)

    def test_oracle_match_and_mismatch(self) -> None:
        val_a = 0x1111222233334444
        val_b = 0x5555666677778888
        val_c = 0x9999AAAABBBBCCCC

        # Match -> exchanges to val_c
        old, mem, swapped = oracle_condxchg64(val_a, val_a, val_c)
        self.assertTrue(swapped)
        self.assertEqual(old, val_a)
        self.assertEqual(mem, val_c)

        # Mismatch -> preserves val_a
        old2, mem2, swapped2 = oracle_condxchg64(val_a, val_b, val_c)
        self.assertFalse(swapped2)
        self.assertEqual(old2, val_a)
        self.assertEqual(mem2, val_a)

    def test_oracle_partial_mismatch_low_vs_high(self) -> None:
        # Match only in low 32 bits
        val1 = 0xAAAAAAAAAAAAAAAA_12345678
        val2 = 0xBBBBBBBBBBBBBBBB_12345678
        old, mem, swapped = oracle_condxchg64(val1, val2, 0)
        self.assertFalse(swapped)

        # Match only in high 32 bits
        val3 = 0x12345678_AAAAAAAAAAAAAAAA
        val4 = 0x12345678_BBBBBBBBBBBBBBBB
        old2, mem2, swapped2 = oracle_condxchg64(val3, val4, 0)
        self.assertFalse(swapped2)

    def test_lds_dword_indices_and_alignment(self) -> None:
        lds = LDSMemory(65536)
        # Byte addresses 0..3 map to dwords 0 and 1
        for b in range(4):
            idx0, idx1 = lds.get_dword_indices(b)
            self.assertEqual(idx0, 0)
            self.assertEqual(idx1, 1)

        # Byte address 4 maps to dwords 1 and 2
        idx0, idx1 = lds.get_dword_indices(4)
        self.assertEqual(idx0, 1)
        self.assertEqual(idx1, 2)

    def test_lds_offset_addition(self) -> None:
        lds = LDSMemory(65536)
        idx0, idx1 = lds.get_dword_indices(0, offset=8)
        self.assertEqual(idx0, 2)
        self.assertEqual(idx1, 3)

    def test_lds_out_of_bounds_wrapping(self) -> None:
        lds = LDSMemory(65536)
        idx0, idx1 = lds.get_dword_indices(65536)
        self.assertEqual(idx0, 0)
        self.assertEqual(idx1, 1)

        # Wrapping at highest dword boundary
        idx0_end, idx1_end = lds.get_dword_indices(65532)
        self.assertEqual(idx0_end, 16383)
        self.assertEqual(idx1_end, 0)  # Wraps around to 0

    def test_lds_bank_calculation(self) -> None:
        lds = LDSMemory(65536)
        idx0, idx1 = lds.get_dword_indices(0)
        bank0, bank1 = lds.get_banks(idx0, idx1)
        self.assertEqual(bank0, 0)
        self.assertEqual(bank1, 1)

    def test_lds_read_write_qword(self) -> None:
        lds = LDSMemory(65536)
        test_val = 0xCAFEBABE_DEADBEEF
        lds.write_qword(0x20, test_val)
        self.assertEqual(lds.read_qword(0x20), test_val)
        # Check raw dwords
        self.assertEqual(lds.dwords[8], 0xDEADBEEF)
        self.assertEqual(lds.dwords[9], 0xCAFEBABE)

    def test_lds_atomic_condxchg64_execution(self) -> None:
        lds = LDSMemory(65536)
        initial_val = 0x12345678_9ABCDEF0
        new_val = 0xFEDCBA98_76543210
        lds.write_qword(40, initial_val)

        # Successful CAS
        old, mem, swapped = lds.atomic_condxchg64(40, initial_val, new_val)
        self.assertTrue(swapped)
        self.assertEqual(old, initial_val)
        self.assertEqual(mem, new_val)
        self.assertEqual(lds.read_qword(40), new_val)

        # Failed CAS
        old2, mem2, swapped2 = lds.atomic_condxchg64(40, initial_val, 0)
        self.assertFalse(swapped2)
        self.assertEqual(old2, new_val)
        self.assertEqual(mem2, new_val)
        self.assertEqual(lds.read_qword(40), new_val)


if __name__ == "__main__":
    unittest.main()
