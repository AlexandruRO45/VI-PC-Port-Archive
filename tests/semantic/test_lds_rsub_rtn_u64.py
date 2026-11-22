"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_rsub_rtn_u64.

Models unsigned 64-bit integer reverse-subtraction and AMD RDNA2 / GFX10 Local Data Share (LDS)
architectural semantics:
  - ds_rsub_rtn_u64: Atomic reverse subtraction of unsigned 64-bit integer in LDS memory,
    returning the previous memory content into destination VGPR pair v[d:d+1].

Syntax:
    ds_rsub_rtn_u64 vdst, vaddr, vdata0 [offset:offset16]

Mathematical Specification:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val = LDS_qword[qword_index]
    new_mem = (vdata0 - old_val) & 0xFFFFFFFFFFFFFFFF
    LDS_qword[qword_index] = new_mem
    vdst = old_val

Key Architectural Characteristics:
  1. Reverse Subtraction (RSUB):
     Computes (vdata0 - old_val) rather than (old_val - vdata0).
     The incoming 64-bit operand vdata0 serves as the minuend, and the current 64-bit memory
     content serves as the subtrahend.
  2. Return Value Semantics (RTN):
     Atomically fetches the pre-arithmetic 64-bit value of LDS[qword_index] and returns it
     into the destination 64-bit VGPR pair (vdst).
  3. Addressing and Alignment:
     Local Data Share is byte-addressed but qword-aligned (8-byte aligned) for 64-bit operations.
     The bottom 3 bits of the effective address are dropped (eff_addr >> 3).
     Immediate offset is an unsigned 16-bit integer (0 <= offset <= 65535).
  4. LDS Capacity:
     Standard allocation is 64 KiB (8,192 64-bit qwords).

Standard library Python only; zero external dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import struct
import sys
from typing import Any, Tuple
import unittest

# Architectural Constants for RDNA2 Local Data Share (LDS) 64-bit
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64: int = 0x8000000000000000
OFFSET16_MASK: int = 0xFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536      # 64 KiB default capacity
DEFAULT_LDS_NUM_QWORDS: int = 8192       # 8,192 64-bit qwords
DEFAULT_NUM_VGPRS: int = 256             # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106             # Standard GFX10 scalar register file size


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword [0, 2^64 - 1]."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_BIT64) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS with 16-bit offset."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = DEFAULT_LDS_NUM_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS qword index.

    LDS 64-bit operations drop the lowest 3 bits and wrap modulo the power-of-two qword capacity.
    """
    return (eff_addr >> 3) & (capacity_qwords - 1)


def ref_ds_rsub_rtn_u64(current_mem: int, vdata0: int) -> tuple[int, int]:
    """Pure mathematical reference oracle for ds_rsub_rtn_u64.

    Computes:
        old_val = current_mem & 0xFFFFFFFFFFFFFFFF
        new_mem = (vdata0 - old_val) & 0xFFFFFFFFFFFFFFFF

    Returns:
        (old_val, new_mem)
    """
    old_val = u64(current_mem)
    new_mem = u64(u64(vdata0) - old_val)
    return old_val, new_mem


class LDSMemory64:
    """Architectural memory model simulating AMD RDNA2 Local Data Share (LDS) for 64-bit operations."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"size_bytes must be a positive power of two, got {size_bytes}")
        if size_bytes < 8:
            raise ValueError(f"size_bytes must be at least 8 bytes for 64-bit LDS, got {size_bytes}")
        self.size_bytes: int = size_bytes
        self.num_qwords: int = size_bytes // 8
        self.qwords: list[int] = [0] * self.num_qwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.num_qwords)
        return self.qwords[idx]

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.num_qwords)
        self.qwords[idx] = u64(val)

    def atomic_rsub_rtn_u64(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> tuple[int, int]:
        """Execute atomic 64-bit integer reverse subtraction with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_qword_index(eff_addr, self.num_qwords)
        curr = self.qwords[idx]
        old_val, new_mem = ref_ds_rsub_rtn_u64(curr, vdata0)
        self.qwords[idx] = new_mem
        return old_val, new_mem


class TestLDSMemoryAndOracle64(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS 64-bit memory model."""

    def test_oracle_basic_rsub_u64(self) -> None:
        # 500 - 100 = 400
        old_val, new_mem = ref_ds_rsub_rtn_u64(current_mem=100, vdata0=500)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, 400)

        # Reverse order: 100 - 500 = -400 mod 2^64
        old_val, new_mem = ref_ds_rsub_rtn_u64(current_mem=500, vdata0=100)
        self.assertEqual(old_val, 500)
        self.assertEqual(new_mem, u64(-400))
        self.assertEqual(new_mem, (100 - 500) & MASK64)

    def test_oracle_identity(self) -> None:
        for x in (0, 1, 42, 0x123456789ABCDEF0, 0xFFFFFFFFFFFFFFFF):
            old_val, new_mem = ref_ds_rsub_rtn_u64(current_mem=x, vdata0=x)
            self.assertEqual(old_val, x & MASK64)
            self.assertEqual(new_mem, 0)

    def test_oracle_zero_cases(self) -> None:
        # 0 - 0 = 0
        self.assertEqual(ref_ds_rsub_rtn_u64(0, 0), (0, 0))
        # vdata0 - 0 = vdata0
        self.assertEqual(ref_ds_rsub_rtn_u64(0, 0xABCDEF0123456789), (0, 0xABCDEF0123456789))
        # 0 - current_mem = -current_mem mod 2^64
        self.assertEqual(ref_ds_rsub_rtn_u64(0x1000, 0), (0x1000, u64(-0x1000)))

    def test_oracle_wraparound(self) -> None:
        # 0 - 0xFFFFFFFFFFFFFFFF = 1
        self.assertEqual(ref_ds_rsub_rtn_u64(0xFFFFFFFFFFFFFFFF, 0), (0xFFFFFFFFFFFFFFFF, 1))
        # 0xFFFFFFFFFFFFFFFF - 0 = 0xFFFFFFFFFFFFFFFF
        self.assertEqual(ref_ds_rsub_rtn_u64(0, 0xFFFFFFFFFFFFFFFF), (0, 0xFFFFFFFFFFFFFFFF))
        # 1 - 2 = 0xFFFFFFFFFFFFFFFF
        self.assertEqual(ref_ds_rsub_rtn_u64(2, 1), (2, 0xFFFFFFFFFFFFFFFF))

    def test_lds_memory64_initialization(self) -> None:
        mem = LDSMemory64(65536)
        self.assertEqual(mem.size_bytes, 65536)
        self.assertEqual(mem.num_qwords, 8192)
        self.assertEqual(len(mem.qwords), 8192)

        for invalid_size in (0, -64, 4, 1000, 30000, 65537):
            with self.assertRaises(ValueError):
                LDSMemory64(invalid_size)

    def test_lds_memory64_read_write(self) -> None:
        mem = LDSMemory64(1024)  # 128 qwords
        mem.write_qword(0x00, 0x1122334455667788)
        self.assertEqual(mem.read_qword(0x00), 0x1122334455667788)

        mem.write_qword(0x40, 0xAABBCCDDEEFF0011)
        self.assertEqual(mem.read_qword(0x40), 0xAABBCCDDEEFF0011)

    def test_lds_memory64_unaligned_byte_address(self) -> None:
        mem = LDSMemory64(1024)
        # Low 3 bits dropped: 0x100 through 0x107 all map to qword index 0x100 >> 3
        mem.write_qword(0x100, 0xDEADBEEFFEEDCAFE)
        for b in range(8):
            self.assertEqual(mem.read_qword(0x100 + b), 0xDEADBEEFFEEDCAFE)

    def test_lds_memory64_address_wrapping(self) -> None:
        mem = LDSMemory64(1024)  # 128 qwords (1024 bytes)
        mem.write_qword(0, 0xCAFEBABEDEADBEEF)
        self.assertEqual(mem.read_qword(1024), 0xCAFEBABEDEADBEEF)
        self.assertEqual(mem.read_qword(2048), 0xCAFEBABEDEADBEEF)

    def test_lds_atomic_rsub_rtn_u64(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x20, 100)

        # 250 - 100 = 150
        old_val, new_mem = mem.atomic_rsub_rtn_u64(vaddr=0x20, vdata0=250)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, 150)
        self.assertEqual(mem.read_qword(0x20), 150)

        # 50 - 150 = -100 mod 2^64
        old_val2, new_mem2 = mem.atomic_rsub_rtn_u64(vaddr=0x20, vdata0=50)
        self.assertEqual(old_val2, 150)
        self.assertEqual(new_mem2, u64(-100))
        self.assertEqual(mem.read_qword(0x20), u64(-100))

    def test_lds_atomic_rsub_rtn_u64_with_offset(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x50, 40)

        # vaddr = 0x48, offset = 0x08 -> eff_addr = 0x50
        old_val, new_mem = mem.atomic_rsub_rtn_u64(vaddr=0x48, vdata0=100, offset=0x08)
        self.assertEqual(old_val, 40)
        self.assertEqual(new_mem, 60)
        self.assertEqual(mem.read_qword(0x50), 60)


if __name__ == "__main__":
    unittest.main()
