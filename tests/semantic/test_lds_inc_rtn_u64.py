from __future__ import annotations

"""Semantic reference oracle and architectural model for ds_inc_rtn_u64.

RDNA2 / GFX10 Local Data Share (LDS) Atomic 64-bit Increment with Return:
Performs a 64-bit atomic wrapping increment on LDS memory and returns the original 64-bit value.

Syntax:
    ds_inc_rtn_u64 vdst, vaddr, vdata0 [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + (offset16 & 0xFFFF)) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val = LDS[qword_index]
    new_mem = 0 if old_val >= (vdata0 & 0xFFFFFFFFFFFFFFFF) else (old_val + 1) & 0xFFFFFFFFFFFFFFFF
    LDS[qword_index] = new_mem
    vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Capacity: 64 KiB default capacity (8,192 64-bit qwords / 16,384 32-bit dwords).
  - Addressing: Base byte address in VGPR vaddr + 16-bit unsigned immediate offset.
  - Alignment: 64-bit qword aligned; low 3 bits of byte address are truncated (eff_addr >> 3).
  - Wrap Condition: Unsigned 64-bit comparison: if old_val >= wrap_limit then new_mem = 0,
    else new_mem = (old_val + 1) mod 2^64.
  - Return: The pre-operation 64-bit value is returned to VGPR pair vdst.
"""

import struct
import unittest
from typing import Tuple

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536          # 64 KiB capacity
LDS_SIZE_DWORDS: int = 16384         # 16,384 32-bit dwords
LDS_SIZE_QWORDS: int = 8192          # 8,192 64-bit qwords
LDS_BANK_COUNT: int = 32             # 32 memory banks
MASK_32: int = 0xFFFF_FFFF           # 32-bit unsigned mask
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF          # 16-bit offset mask
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106
DEFAULT_NUM_LANES: int = 64


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK_64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as signed two's-complement integer."""
    val &= MASK_64
    return val - (1 << 64) if (val & (1 << 63)) else val


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index."""
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the starting LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_inc_rtn_u64(
    current_mem: int,
    wrap_limit: int,
) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_inc_rtn_u64.

    Computes:
        old_val = current_mem & 0xFFFFFFFFFFFFFFFF
        limit = wrap_limit & 0xFFFFFFFFFFFFFFFF
        new_mem = 0 if old_val >= limit else (old_val + 1) & 0xFFFFFFFFFFFFFFFF

    Returns:
        (old_val, new_mem)
    """
    old_val = u64(current_mem)
    limit = u64(wrap_limit)
    new_mem = 0 if old_val >= limit else u64(old_val + 1)
    return old_val, new_mem


class LDSModel64:
    """Architectural memory model representing the Local Data Share scratchpad with 64-bit access."""

    def __init__(self, capacity_qwords: int = LDS_SIZE_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a positive power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.memory = [0] * capacity_qwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        return self.memory[idx]

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        self.memory[idx] = u64(val)

    def atomic_inc_rtn_u64(
        self,
        vaddr: int,
        wrap_limit: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 64-bit wrapping increment with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_qword_index(eff_addr, self.capacity_qwords)
        curr = self.memory[idx]
        old_val, new_mem = oracle_ds_inc_rtn_u64(curr, wrap_limit)
        self.memory[idx] = new_mem
        return old_val, new_mem


class TestReferenceOracleAndLDSModel64(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model for ds_inc_rtn_u64."""

    def test_oracle_normal_increment_u64(self) -> None:
        old, new_mem = oracle_ds_inc_rtn_u64(0, 5)
        self.assertEqual(old, 0)
        self.assertEqual(new_mem, 1)

        old, new_mem = oracle_ds_inc_rtn_u64(4, 5)
        self.assertEqual(old, 4)
        self.assertEqual(new_mem, 5)

        old, new_mem = oracle_ds_inc_rtn_u64(0x1_0000_0000, 0x2_0000_0000)
        self.assertEqual(old, 0x1_0000_0000)
        self.assertEqual(new_mem, 0x1_0000_0001)

    def test_oracle_wrap_on_equal_threshold_u64(self) -> None:
        old, new_mem = oracle_ds_inc_rtn_u64(5, 5)
        self.assertEqual(old, 5)
        self.assertEqual(new_mem, 0)

        limit_64 = 0x1234_5678_9ABC_DEF0
        old, new_mem = oracle_ds_inc_rtn_u64(limit_64, limit_64)
        self.assertEqual(old, limit_64)
        self.assertEqual(new_mem, 0)

    def test_oracle_wrap_on_greater_threshold_u64(self) -> None:
        old, new_mem = oracle_ds_inc_rtn_u64(6, 5)
        self.assertEqual(old, 6)
        self.assertEqual(new_mem, 0)

        old, new_mem = oracle_ds_inc_rtn_u64(0xFFFF_FFFF_FFFF_FFFF, 100)
        self.assertEqual(old, 0xFFFF_FFFF_FFFF_FFFF)
        self.assertEqual(new_mem, 0)

    def test_oracle_zero_wrap_limit_u64(self) -> None:
        old, new_mem = oracle_ds_inc_rtn_u64(0, 0)
        self.assertEqual(old, 0)
        self.assertEqual(new_mem, 0)

        old, new_mem = oracle_ds_inc_rtn_u64(42, 0)
        self.assertEqual(old, 42)
        self.assertEqual(new_mem, 0)

    def test_oracle_max_u64_limit(self) -> None:
        old, new_mem = oracle_ds_inc_rtn_u64(0, MASK_64)
        self.assertEqual(old, 0)
        self.assertEqual(new_mem, 1)

        old, new_mem = oracle_ds_inc_rtn_u64(MASK_64 - 1, MASK_64)
        self.assertEqual(old, MASK_64 - 1)
        self.assertEqual(new_mem, MASK_64)

        old, new_mem = oracle_ds_inc_rtn_u64(MASK_64, MASK_64)
        self.assertEqual(old, MASK_64)
        self.assertEqual(new_mem, 0)

    def test_effective_address_and_offset_u64(self) -> None:
        eff = calc_lds_effective_address(0x1000, 0x0040)
        self.assertEqual(eff, 0x1040)

        # 32-bit modulo wrapping
        eff_wrap = calc_lds_effective_address(0xFFFF_FFF8, 0x0010)
        self.assertEqual(eff_wrap, 0x0008)

        # 16-bit offset mask
        eff_mask = calc_lds_effective_address(0x2000, 0x1_0020)
        self.assertEqual(eff_mask, 0x2020)

    def test_qword_alignment_truncation(self) -> None:
        # Low 3 bits truncated (64-bit qword alignment)
        for offset in range(8):
            self.assertEqual(calc_lds_qword_index(0x1000 + offset), 0x200)
        self.assertEqual(calc_lds_qword_index(0x1008), 0x201)

    def test_qword_capacity_wrapping(self) -> None:
        capacity_bytes = LDS_SIZE_QWORDS * 8
        eff = calc_lds_effective_address(capacity_bytes + 0x80)
        idx = calc_lds_qword_index(eff, LDS_SIZE_QWORDS)
        self.assertEqual(idx, 0x10)

    def test_lds_model_atomic_inc_rtn_u64(self) -> None:
        lds = LDSModel64(capacity_qwords=512)
        addr = 0x80

        lds.write_qword(addr, 0x1_0000_0000)
        old, new_mem = lds.atomic_inc_rtn_u64(vaddr=addr, wrap_limit=0x2_0000_0000, offset=0)
        self.assertEqual(old, 0x1_0000_0000)
        self.assertEqual(new_mem, 0x1_0000_0001)
        self.assertEqual(lds.read_qword(addr), 0x1_0000_0001)

        # Increment with offset
        old, new_mem = lds.atomic_inc_rtn_u64(vaddr=0x78, wrap_limit=0x2_0000_0000, offset=8)
        self.assertEqual(old, 0x1_0000_0001)
        self.assertEqual(new_mem, 0x1_0000_0002)
        self.assertEqual(lds.read_qword(addr), 0x1_0000_0002)

        # Hit wrap condition
        old, new_mem = lds.atomic_inc_rtn_u64(vaddr=addr, wrap_limit=0x1_0000_0002, offset=0)
        self.assertEqual(old, 0x1_0000_0002)
        self.assertEqual(new_mem, 0)
        self.assertEqual(lds.read_qword(addr), 0)


if __name__ == "__main__":
    unittest.main()
