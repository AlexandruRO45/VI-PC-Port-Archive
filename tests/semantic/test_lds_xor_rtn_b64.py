"""Semantic reference oracle and architectural model for ds_xor_rtn_b64.

RDNA2 / GFX10 Local Data Share (LDS) 64-Bit Bitwise Atomic XOR with Return:
Performs a 64-bit atomic bitwise XOR operation on LDS memory and returns
the previous memory value to the destination VGPR pair v[dst:dst+1].

Syntax:
    ds_xor_rtn_b64 v[dst:dst+1], vaddr, v[data:data+1] [offset:offset16]
    ds_xor_b64 vaddr, v[data:data+1] [offset:offset16]

Semantics:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_capacity_qwords - 1)
    old_val = LDS[qword_index]
    new_mem = (old_val ^ vdata) & 0xFFFFFFFFFFFFFFFF
    LDS[qword_index] = new_mem
    v[dst:dst+1] = old_val
"""
from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Union

# RDNA2 Local Data Share (LDS) Architectural Constants
LDS_SIZE_BYTES: int = 65536         # 64 KiB default capacity
LDS_SIZE_DWORDS: int = 16384        # 16,384 32-bit dwords
LDS_SIZE_QWORDS: int = 8192         # 8,192 64-bit qwords
LDS_BANK_COUNT: int = 32            # 32 independent memory banks
MASK_32: int = 0xFFFF_FFFF          # 32-bit unsigned mask
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF # 64-bit unsigned mask
OFFSET16_MASK: int = 0xFFFF         # 16-bit immediate byte offset mask


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    return (vaddr + (offset & OFFSET16_MASK)) & MASK_32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = LDS_SIZE_QWORDS) -> int:
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    return (eff_addr >> 2) & (LDS_BANK_COUNT - 1)


def oracle_ds_xor_b64(current_mem: int, xor_val: int) -> Tuple[int, int]:
    old_val = current_mem & MASK_64
    vdata = xor_val & MASK_64
    new_mem = (old_val ^ vdata) & MASK_64
    return old_val, new_mem


class LDSModel:
    def __init__(self, capacity_qwords: int = LDS_SIZE_QWORDS) -> None:
        if capacity_qwords <= 0 or (capacity_qwords & (capacity_qwords - 1)) != 0:
            raise ValueError(f"capacity_qwords must be a power of two, got {capacity_qwords}")
        self.capacity_qwords = capacity_qwords
        self.capacity_dwords = capacity_qwords * 2
        self.memory = [0] * self.capacity_dwords

    def read_qword(self, eff_addr: int) -> int:
        dword_idx = ((eff_addr >> 3) << 1) & (self.capacity_dwords - 1)
        low = self.memory[dword_idx]
        high = self.memory[dword_idx + 1]
        return ((high << 32) | low) & MASK_64

    def write_qword(self, eff_addr: int, val: int) -> None:
        dword_idx = ((eff_addr >> 3) << 1) & (self.capacity_dwords - 1)
        self.memory[dword_idx] = val & MASK_32
        self.memory[dword_idx + 1] = (val >> 32) & MASK_32

    def atomic_xor_b64(
        self,
        vaddr: int,
        xor_val: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        eff_addr = calc_lds_effective_address(vaddr, offset)
        curr = self.read_qword(eff_addr)
        old_val, new_mem = oracle_ds_xor_b64(curr, xor_val)
        self.write_qword(eff_addr, new_mem)
        return old_val, new_mem


class TestLdsXorRtnB64ReferenceOracle(unittest.TestCase):
    def test_oracle_xor_with_zero(self) -> None:
        old, mem = oracle_ds_xor_b64(0x12345678_9ABCDEF0, 0)
        self.assertEqual(old, 0x12345678_9ABCDEF0)
        self.assertEqual(mem, 0x12345678_9ABCDEF0)

    def test_oracle_xor_self_cancels_to_zero(self) -> None:
        val = 0xAA55AA55_55AAAA55
        old, mem = oracle_ds_xor_b64(val, val)
        self.assertEqual(old, val)
        self.assertEqual(mem, 0)

    def test_oracle_xor_with_all_ones_inversion(self) -> None:
        val = 0x12345678_9ABCDEF0
        old, mem = oracle_ds_xor_b64(val, MASK_64)
        self.assertEqual(old, val)
        self.assertEqual(mem, (~val) & MASK_64)

    def test_oracle_double_xor_restores_original(self) -> None:
        val = 0xCAFEBABE_DEADBEEF
        key = 0x11223344_55667788
        old1, enc = oracle_ds_xor_b64(val, key)
        old2, dec = oracle_ds_xor_b64(enc, key)
        self.assertEqual(dec, val)

    def test_effective_address_calculation_with_offset(self) -> None:
        eff = calc_lds_effective_address(0x1000, 0x0040)
        self.assertEqual(eff, 0x1040)
        self.assertEqual(calc_lds_qword_index(eff), 0x1040 >> 3)

    def test_effective_address_wrapping_modulo_2_32(self) -> None:
        eff = calc_lds_effective_address(0xFFFF_FFF8, 0x0010)
        self.assertEqual(eff, 0x0000_0008)

    def test_qword_alignment_drops_low_3_bits(self) -> None:
        idx_base = calc_lds_qword_index(0x400)
        for byte_off in range(8):
            self.assertEqual(calc_lds_qword_index(0x400 + byte_off), idx_base)
        self.assertEqual(idx_base, 0x400 >> 3)

    def test_lds_capacity_wrapping(self) -> None:
        cap_bytes = 65536
        idx_low = calc_lds_qword_index(0x100)
        idx_wrapped = calc_lds_qword_index(0x100 + cap_bytes)
        self.assertEqual(idx_low, idx_wrapped)

    def test_model_atomic_xor_returns_pre_op_value(self) -> None:
        lds = LDSModel()
        addr = 0x200
        lds.write_qword(addr, 0x00FF00FF_00FF00FF)

        old_val, new_val = lds.atomic_xor_b64(addr, 0x0000FFFF_0000FFFF)
        self.assertEqual(old_val, 0x00FF00FF_00FF00FF)
        self.assertEqual(new_val, 0x00FFFF00_00FFFF00)
        self.assertEqual(lds.read_qword(addr), 0x00FFFF00_00FFFF00)

if __name__ == "__main__":
    unittest.main()
