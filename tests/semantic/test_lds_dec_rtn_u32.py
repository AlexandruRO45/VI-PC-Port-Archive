from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_dec_rtn_u32.

Opcode:
  - ds_dec_rtn_u32 vdst, vaddr, vdata0 [offset:offset16] [gds]
      Atomic 32-bit decrement on LDS memory with destination register return:
        effective_byte = (vaddr + offset16) & 0xFFFFFFFF
        idx = (effective_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[idx]
        if old_val == 0 or old_val > vdata0:
            new_val = vdata0
        else:
            new_val = (old_val - 1) & 0xFFFFFFFF
        LDS[idx] = new_val
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in VGPR vaddr + unsigned 16-bit immediate byte offset (offset:offset16).
    Dword-aligned: hardware drops low 2 bits when indexing 32-bit dwords in LDS.
  - Modulo/wrapping arithmetic:
      If old_val == 0 or old_val > vdata0: wraps to vdata0.
      Else: decrements by 1 (old_val - 1).
  - Return value: Previous value stored in memory at the targeted location is atomically
    returned and written to destination VGPR vdst.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import struct
from typing import Dict, List, Optional, Tuple, Union
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
LDS_DEFAULT_SIZE_BYTES: int = 65536   # 64 KiB default Workgroup LDS allocation
LDS_DEFAULT_SIZE_DWORDS: int = 16384  # 16,384 32-bit dwords
OFFSET16_MASK: int = 0xFFFF
DEFAULT_NUM_LANES: int = 64           # Wave64 default width
DEFAULT_NUM_VGPRS: int = 256          # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106          # Standard GFX10 scalar register file size

# IEEE 754 Boundary Constants
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


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate byte offset.
    Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_DEFAULT_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def ref_ds_dec_u32(old_val: int, vdata0: int) -> int:
    """Pure reference mathematical formula for ds_dec_u32.

    If old_val == 0 or old_val > vdata0:
        new_val = vdata0
    else:
        new_val = (old_val - 1) & 0xFFFFFFFF
    """
    a = u32(old_val)
    b = u32(vdata0)
    if a == 0 or a > b:
        return b
    return u32(a - 1)


class LDSMemory:
    """Architectural model of Local Data Share memory storage."""

    def __init__(self, size_dwords: int = LDS_DEFAULT_SIZE_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_dwords}")
        self.size_dwords = size_dwords
        self.data: List[int] = [0] * size_dwords

    def read_dword(self, index: int) -> int:
        return self.data[index & (self.size_dwords - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.data[index & (self.size_dwords - 1)] = u32(val)

    def atomic_dec_rtn_u32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 32-bit decrement with return on LDS memory.

        Returns:
            Tuple[int, int]: (old_val, new_val)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        old_val = self.read_dword(idx)
        new_val = ref_ds_dec_u32(old_val, vdata0)
        self.write_dword(idx, new_val)
        return old_val, new_val


def oracle_ds_dec_rtn_u32(
    lds_mem: bytearray,
    vaddr: int,
    vdata0: int,
    offset: int = 0,
    lds_size_bytes: int = LDS_DEFAULT_SIZE_BYTES,
) -> Tuple[int, int]:
    """Reference bytearray mathematical oracle for RDNA2 ds_dec_rtn_u32.

    Computes effective address, reads previous 32-bit dword, applies wrapping
    decrement logic, writes updated 32-bit dword back, and returns (old_val, new_val).
    """
    eff_addr = calc_lds_effective_address(vaddr, offset) & (lds_size_bytes - 1)
    aligned_addr = eff_addr & ~3  # dword aligned (drop low 2 bits)
    old_val = struct.unpack_from("<I", lds_mem, aligned_addr)[0]
    new_val = ref_ds_dec_u32(old_val, vdata0)
    struct.pack_into("<I", lds_mem, aligned_addr, new_val)
    return old_val, new_val


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_dwords=1024)
        self.lds_bytes = bytearray(LDS_DEFAULT_SIZE_BYTES)

    def test_normal_decrement_within_bounds(self) -> None:
        # Initial 5, bound 10 -> decrements to 4, returns old 5
        self.mem.write_dword(10, 5)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=40, vdata0=10, offset=0)
        self.assertEqual(old_val, 5)
        self.assertEqual(new_val, 4)
        self.assertEqual(self.mem.read_dword(10), 4)

        # Initial 1, bound 10 -> decrements to 0, returns old 1
        self.mem.write_dword(10, 1)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=40, vdata0=10, offset=0)
        self.assertEqual(old_val, 1)
        self.assertEqual(new_val, 0)
        self.assertEqual(self.mem.read_dword(10), 0)

    def test_wrapping_at_zero(self) -> None:
        # Initial 0, bound 10 -> wraps to 10, returns old 0
        self.mem.write_dword(20, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=80, vdata0=10, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 10)
        self.assertEqual(self.mem.read_dword(20), 10)

    def test_wrapping_when_strictly_greater_than_bound(self) -> None:
        # Initial 15, bound 10 -> strictly greater, wraps to 10, returns old 15
        self.mem.write_dword(30, 15)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=120, vdata0=10, offset=0)
        self.assertEqual(old_val, 15)
        self.assertEqual(new_val, 10)
        self.assertEqual(self.mem.read_dword(30), 10)

        # Initial MASK32, bound 10 -> wraps to 10
        self.mem.write_dword(30, MASK32)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=120, vdata0=10, offset=0)
        self.assertEqual(old_val, MASK32)
        self.assertEqual(new_val, 10)
        self.assertEqual(self.mem.read_dword(30), 10)

    def test_decrement_when_equal_to_bound(self) -> None:
        # Initial 10, bound 10 -> not 0 and not > 10, so normal decrement to 9
        self.mem.write_dword(40, 10)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=160, vdata0=10, offset=0)
        self.assertEqual(old_val, 10)
        self.assertEqual(new_val, 9)
        self.assertEqual(self.mem.read_dword(40), 9)

    def test_boundary_bound_zero(self) -> None:
        # bound = 0:
        # if old == 0: wraps to 0
        self.mem.write_dword(5, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=20, vdata0=0, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, 0)

        # if old == 1: 1 > 0, wraps to 0
        self.mem.write_dword(5, 1)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=20, vdata0=0, offset=0)
        self.assertEqual(old_val, 1)
        self.assertEqual(new_val, 0)

        # if old == MASK32: MASK32 > 0, wraps to 0
        self.mem.write_dword(5, MASK32)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=20, vdata0=0, offset=0)
        self.assertEqual(old_val, MASK32)
        self.assertEqual(new_val, 0)

    def test_boundary_bound_max_u32(self) -> None:
        # bound = MASK32:
        # if old == 0: wraps to MASK32
        self.mem.write_dword(6, 0)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=24, vdata0=MASK32, offset=0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, MASK32)

        # if old == MASK32: not > MASK32 and not 0, decrements to MASK32 - 1
        self.mem.write_dword(6, MASK32)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=24, vdata0=MASK32, offset=0)
        self.assertEqual(old_val, MASK32)
        self.assertEqual(new_val, MASK32 - 1)

    def test_countdown_cycle_simulation(self) -> None:
        # Countdown loop from 3 down to 0 and wrapping back to 3 with vdata0 = 3
        bound = 3
        idx = 15
        self.mem.write_dword(idx, bound)

        expected_sequence = [
            (3, 2),  # step 1
            (2, 1),  # step 2
            (1, 0),  # step 3
            (0, 3),  # step 4: wraps from 0 back to bound
            (3, 2),  # step 5: cycles again
        ]
        for expected_old, expected_new in expected_sequence:
            old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=idx * 4, vdata0=bound)
            self.assertEqual(old_val, expected_old)
            self.assertEqual(new_val, expected_new)
            self.assertEqual(self.mem.read_dword(idx), expected_new)

    def test_address_offset_and_dword_alignment(self) -> None:
        # vaddr = 0x1000, offset = 16 -> eff = 0x1010 -> dword index = 0x404 & 1023 = 4
        self.mem.write_dword(4, 50)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=0x1000, vdata0=100, offset=16)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_val, 49)
        self.assertEqual(self.mem.read_dword(4), 49)

        # Low 2 bits of address ignored (unaligned byte offset 7 -> 0x1007 -> aligned to 0x1004)
        self.mem.write_dword(1, 25)
        old_val, new_val = self.mem.atomic_dec_rtn_u32(vaddr=0x1000, vdata0=100, offset=7)
        # 0x1007 >> 2 = 0x401 -> dword index 1
        self.assertEqual(old_val, 25)
        self.assertEqual(new_val, 24)
        self.assertEqual(self.mem.read_dword(1), 24)

    def test_bytearray_oracle_equivalence(self) -> None:
        addr = 0x200
        struct.pack_into("<I", self.lds_bytes, addr, 77)
        old_val, new_val = oracle_ds_dec_rtn_u32(self.lds_bytes, vaddr=addr, vdata0=50)
        # 77 > 50 -> wraps to 50
        self.assertEqual(old_val, 77)
        self.assertEqual(new_val, 50)
        self.assertEqual(struct.unpack_from("<I", self.lds_bytes, addr)[0], 50)


if __name__ == "__main__":
    unittest.main()
