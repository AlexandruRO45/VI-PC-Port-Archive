"""Semantic reference oracle and architectural model for ds_nand_rtn_b32.

RDNA2 / GFX10 Local Data Share (LDS) Bitwise Atomic NAND with Return:
Performs a 32-bit atomic bitwise NAND operation on LDS memory and returns
the previous memory value to the destination VGPR.

Syntax:
    ds_nand_rtn_b32 vdst, vaddr, vdata [offset:offset16]
    ds_nand_b32 vaddr, vdata [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_capacity_dwords - 1)
    old_val = LDS[dword_index]
    new_val = ~(old_val & vdata) & 0xFFFFFFFF
    LDS[dword_index] = new_val
    vdst = old_val
"""
from __future__ import annotations

import unittest
from typing import Tuple

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.
    
    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = LDS_SIZE_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.
    
    LDS addresses are dword-aligned; the low 2 bits are ignored by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index for an effective address."""
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_nand_b32(current_mem: int, nand_val: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_nand_rtn_b32.
    
    Returns:
        (returned_old_val, new_mem_val)
    """
    old_val = current_mem & MASK_32
    vdata = nand_val & MASK_32
    new_mem = (~(old_val & vdata)) & MASK_32
    return old_val, new_mem


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
        self.memory[idx] = val & MASK_32

    def atomic_nand_b32(
        self,
        vaddr: int,
        nand_val: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic bitwise NAND on LDS dword, returning (old_val, new_val)."""
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.capacity_dwords)
        curr = self.memory[idx]
        old_val, new_mem = oracle_ds_nand_b32(curr, nand_val)
        self.memory[idx] = new_mem
        return old_val, new_mem


class TestLdsNandRtnB32ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and architectural model."""

    def test_oracle_nand_with_zero(self) -> None:
        """NAND with 0 always sets all 32 bits (~0 == 0xFFFFFFFF)."""
        old, mem = oracle_ds_nand_b32(0x12345678, 0x00000000)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(mem, 0xFFFFFFFF)

        old2, mem2 = oracle_ds_nand_b32(0x00000000, 0x00000000)
        self.assertEqual(old2, 0x00000000)
        self.assertEqual(mem2, 0xFFFFFFFF)

    def test_oracle_nand_with_all_ones_inversion(self) -> None:
        """NAND with 0xFFFFFFFF computes bitwise NOT (~x)."""
        val = 0xAA55AA55
        old, mem = oracle_ds_nand_b32(val, 0xFFFFFFFF)
        self.assertEqual(old, val)
        self.assertEqual(mem, 0x55AA55AA)

    def test_oracle_self_nand_inversion(self) -> None:
        """x NAND x computes ~x."""
        val = 0x12345678
        old, mem = oracle_ds_nand_b32(val, val)
        self.assertEqual(old, val)
        self.assertEqual(mem, (~val) & MASK_32)

    def test_oracle_disjoint_bitmasks(self) -> None:
        """NAND of disjoint bitmasks produces 0xFFFFFFFF because AND is 0."""
        mask_a = 0xAAAA0000
        mask_b = 0x00005555
        old, mem = oracle_ds_nand_b32(mask_a, mask_b)
        self.assertEqual(old, mask_a)
        self.assertEqual(mem, 0xFFFFFFFF)

    def test_oracle_overlapping_bitmasks(self) -> None:
        """NAND of overlapping bitmasks correctly inverts common set bits."""
        val_a = 0b1100_1010
        val_b = 0b0110_0111
        # val_a & val_b = 0b0100_0010
        # ~(val_a & val_b) has 0 at bits 1 and 6, 1 elsewhere
        old, mem = oracle_ds_nand_b32(val_a, val_b)
        self.assertEqual(old, val_a)
        expected = (~0b0100_0010) & MASK_32
        self.assertEqual(mem, expected)

    def test_effective_address_calculation_with_offset(self) -> None:
        """Effective address correctly adds base and 16-bit offset."""
        eff = calc_lds_effective_address(0x1000, 0x0024)
        self.assertEqual(eff, 0x1024)
        self.assertEqual(calc_lds_dword_index(eff), 0x1024 >> 2)

    def test_effective_address_wrapping_modulo_2_32(self) -> None:
        """Effective address wraps cleanly at 2^32 boundary."""
        eff = calc_lds_effective_address(0xFFFF_FFFE, 0x0004)
        self.assertEqual(eff, 0x0000_0002)

    def test_dword_alignment_drops_low_bits(self) -> None:
        """Hardware drops low 2 bits; byte offsets 0, 1, 2, 3 map to same dword."""
        idx0 = calc_lds_dword_index(0x400)
        idx1 = calc_lds_dword_index(0x401)
        idx2 = calc_lds_dword_index(0x402)
        idx3 = calc_lds_dword_index(0x403)
        self.assertEqual(idx0, idx1)
        self.assertEqual(idx1, idx2)
        self.assertEqual(idx2, idx3)
        self.assertEqual(idx0, 0x100)

    def test_bank_distribution_across_32_banks(self) -> None:
        """Sequential dwords map cleanly across all 32 LDS banks."""
        banks = [calc_lds_bank(i * 4) for i in range(32)]
        self.assertEqual(banks, list(range(32)))

    def test_lds_capacity_wrapping(self) -> None:
        """Addresses past capacity wrap modulo capacity_dwords."""
        cap_bytes = 65536
        idx_low = calc_lds_dword_index(0x100)
        idx_wrapped = calc_lds_dword_index(0x100 + cap_bytes)
        self.assertEqual(idx_low, idx_wrapped)

    def test_model_atomic_nand_returns_pre_op_value(self) -> None:
        """LDSModel atomic_nand_b32 updates memory and returns the pre-operation value."""
        lds = LDSModel()
        addr = 0x200
        lds.write_dword(addr, 0x00FF00FF)

        old_val, new_val = lds.atomic_nand_b32(addr, 0x0000FFFF)
        # old & data = 0x000000FF
        # new = ~0x000000FF = 0xFFFFFF00
        self.assertEqual(old_val, 0x00FF00FF)
        self.assertEqual(new_val, 0xFFFFFF00)
        self.assertEqual(lds.read_dword(addr), 0xFFFFFF00)


if __name__ == "__main__":
    unittest.main()
