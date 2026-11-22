"""Semantic reference oracle and architectural model for ds_nand_rtn_b64.

RDNA2 / GFX10 Local Data Share (LDS) 64-Bit Bitwise Atomic NAND with Return:
Performs a 64-bit atomic bitwise NAND operation on LDS memory and returns
the previous memory value to the destination VGPR pair v[dst:dst+1].

Syntax:
    ds_nand_rtn_b64 v[dst:dst+1], vaddr, v[data:data+1] [offset:offset16]
    ds_nand_b64 vaddr, v[data:data+1] [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_capacity_qwords - 1)
    old_val = LDS[qword_index]
    new_val = ~(old_val & vdata) & 0xFFFFFFFFFFFFFFFF
    LDS[qword_index] = new_val
    v[dst:dst+1] = old_val
"""
from __future__ import annotations

import unittest
from typing import Tuple

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_SIZE_QWORDS: int = 8192         # 8,192 64-bit qwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.
    
    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index.
    
    64-bit LDS accesses are 8-byte aligned; the low 3 bits are ignored by hardware.
    The index wraps modulo the power-of-two qword capacity.
    """
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_nand_b64(current_mem: int, nand_val: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_nand_rtn_b64.
    
    Returns:
        (returned_old_val, new_mem_val)
    """
    old_val = current_mem & MASK_64
    vdata = nand_val & MASK_64
    new_mem = (~(old_val & vdata)) & MASK_64
    return old_val, new_mem


class LDSModel:
    """Architectural memory model representing the Local Data Share scratchpad."""

    def __init__(self, capacity_qwords: int = LDS_SIZE_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.capacity_dwords = capacity_qwords * 2
        self.memory = [0] * self.capacity_dwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        dword_idx = ((eff_addr >> 3) << 1) & (self.capacity_dwords - 1)
        low = self.memory[dword_idx]
        high = self.memory[dword_idx + 1]
        return ((high << 32) | low) & MASK_64

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        dword_idx = ((eff_addr >> 3) << 1) & (self.capacity_dwords - 1)
        self.memory[dword_idx] = val & MASK_32
        self.memory[dword_idx + 1] = (val >> 32) & MASK_32

    def atomic_nand_b64(
        self,
        vaddr: int,
        nand_val: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic bitwise NAND on LDS 64-bit qword, returning (old_val, new_val)."""
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_mem = oracle_ds_nand_b64(curr, nand_val)
        self.write_qword(eff_addr, new_mem)
        return old_val, new_mem


class TestLdsNandRtnB64ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_nand_with_zero(self) -> None:
        """NAND with 0 always sets all 64 bits (~0 == 0xFFFFFFFFFFFFFFFF)."""
        old, mem = oracle_ds_nand_b64(0x12345678_9ABCDEF0, 0x00000000_00000000)
        self.assertEqual(old, 0x12345678_9ABCDEF0)
        self.assertEqual(mem, MASK_64)

    def test_oracle_nand_with_all_ones_inversion(self) -> None:
        """NAND with MASK_64 computes 64-bit bitwise NOT (~x)."""
        val = 0xAA55AA55_55AAAA55
        old, mem = oracle_ds_nand_b64(val, MASK_64)
        self.assertEqual(old, val)
        self.assertEqual(mem, 0x55AA55AA_AA5555AA)

    def test_oracle_self_nand_inversion(self) -> None:
        """x NAND x computes ~x across 64 bits."""
        val = 0x12345678_9ABCDEF0
        old, mem = oracle_ds_nand_b64(val, val)
        self.assertEqual(old, val)
        self.assertEqual(mem, (~val) & MASK_64)

    def test_oracle_disjoint_bitmasks(self) -> None:
        """NAND of disjoint 64-bit masks produces MASK_64."""
        mask_a = 0xFFFF_0000_0000_0000
        mask_b = 0x0000_0000_FFFF_0000
        old, mem = oracle_ds_nand_b64(mask_a, mask_b)
        self.assertEqual(old, mask_a)
        self.assertEqual(mem, MASK_64)

    def test_oracle_overlapping_bitmasks(self) -> None:
        """NAND of overlapping 64-bit masks inverts only common bits."""
        val_a = 0x0000_0000_FFFF_FFFF
        val_b = 0x0000_FFFF_FFFF_0000
        # val_a & val_b = 0x0000_0000_FFFF_0000
        old, mem = oracle_ds_nand_b64(val_a, val_b)
        self.assertEqual(old, val_a)
        expected = (~0x0000_0000_FFFF_0000) & MASK_64
        self.assertEqual(mem, expected)

    def test_effective_address_calculation_with_offset(self) -> None:
        """Effective address correctly adds base and 16-bit offset."""
        eff = calc_lds_effective_address(0x1000, 0x0040)
        self.assertEqual(eff, 0x1040)
        self.assertEqual(calc_lds_qword_index(eff), 0x1040 >> 3)

    def test_effective_address_wrapping_modulo_2_32(self) -> None:
        """Effective address wraps cleanly at 2^32 boundary."""
        eff = calc_lds_effective_address(0xFFFF_FFF8, 0x0010)
        self.assertEqual(eff, 0x0000_0008)

    def test_qword_alignment_drops_low_3_bits(self) -> None:
        """Hardware drops low 3 bits; byte offsets 0..7 all map to same qword."""
        idx_base = calc_lds_qword_index(0x400)
        for byte_off in range(8):
            self.assertEqual(calc_lds_qword_index(0x400 + byte_off), idx_base)
        self.assertEqual(idx_base, 0x400 >> 3)

    def test_lds_capacity_wrapping(self) -> None:
        """Addresses past capacity wrap modulo capacity_qwords."""
        cap_bytes = 65536
        idx_low = calc_lds_qword_index(0x100)
        idx_wrapped = calc_lds_qword_index(0x100 + cap_bytes)
        self.assertEqual(idx_low, idx_wrapped)

    def test_model_atomic_nand_returns_pre_op_value(self) -> None:
        """LDSModel atomic_nand_b64 updates memory and returns the pre-operation value."""
        lds = LDSModel()
        addr = 0x200
        lds.write_qword(addr, 0x00FF00FF_00FF00FF)

        old_val, new_val = lds.atomic_nand_b64(addr, 0x0000FFFF_0000FFFF)
        # old & data = 0x000000FF_000000FF
        # new = ~0x000000FF_000000FF = 0xFFFFFF00_FFFFFF00
        self.assertEqual(old_val, 0x00FF00FF_00FF00FF)
        self.assertEqual(new_val, 0xFFFFFF00_FFFFFF00)
        self.assertEqual(lds.read_qword(addr), 0xFFFFFF00_FFFFFF00)


if __name__ == "__main__":
    unittest.main()
