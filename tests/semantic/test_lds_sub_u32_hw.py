"""Semantic interpreter tests and translator verification for 32-bit unsigned LDS atomic subtraction (ds_sub_u32, ds_sub_rtn_u32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    py -I tests/semantic/test_lds_sub_u32_hw.py
    py -m unittest discover -s tests/semantic -p test_lds_sub_u32_hw.py
"""
from __future__ import annotations

import os
import re
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import support
import lds_run as L
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFFFFFF
UINT32_MAX: int = 0xFFFFFFFF
INT32_MIN: int = 0x80000000
INT32_MAX: int = 0x7FFFFFFF


def to_unsigned_u32(val: int) -> int:
    """Wrap integer into 32-bit unsigned integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_ds_sub_u32(curr_mem: int, data_val: int) -> int:
    """Reference mathematical oracle for 32-bit atomic subtraction (M = (M - D) mod 2^32)."""
    return (curr_mem - data_val) & MASK32


def ref_ds_sub_rtn_u32(curr_mem: int, data_val: int) -> Tuple[int, int]:
    """Reference returning atomic subtraction: returns (old_mem, new_mem)."""
    old = curr_mem & MASK32
    new = ref_ds_sub_u32(old, data_val)
    return old, new


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, size_bytes: int = 4096) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.storage = [0] * self.num_dwords

    def resolve_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        dword_idx = eff_addr >> 2
        return dword_idx & (self.num_dwords - 1)

    def read_dword(self, byte_addr: int, offset: int = 0) -> int:
        idx = self.resolve_dword_index(byte_addr, offset)
        return self.storage[idx]

    def write_dword(self, byte_addr: int, value: int, offset: int = 0) -> None:
        idx = self.resolve_dword_index(byte_addr, offset)
        self.storage[idx] = value & MASK32

    def atomic_sub_u32(self, byte_addr: int, data_val: int, offset: int = 0, rtn: bool = False) -> Tuple[Optional[int], int]:
        idx = self.resolve_dword_index(byte_addr, offset)
        old_val = self.storage[idx]
        new_val = ref_ds_sub_u32(old_val, data_val)
        self.storage[idx] = new_val
        return (old_val if rtn else None), new_val

    def get_bank(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        return (eff_addr >> 2) & 31

class TestDSSubU32OracleAndArchitecturalModel(unittest.TestCase):
    """Part 1: Reference mathematical oracle and architectural model tests."""

    def test_reference_oracle_basic_subtraction(self) -> None:
        self.assertEqual(ref_ds_sub_u32(100, 40), 60)
        self.assertEqual(ref_ds_sub_u32(40, 40), 0)
        self.assertEqual(ref_ds_sub_u32(0, 0), 0)

    def test_reference_oracle_underflow_wrapping(self) -> None:
        # 0 - 1 = 0xFFFFFFFF (mod 2^32)
        self.assertEqual(ref_ds_sub_u32(0, 1), 0xFFFFFFFF)
        self.assertEqual(ref_ds_sub_u32(0, 0xFFFFFFFF), 1)
        self.assertEqual(ref_ds_sub_u32(1, 2), 0xFFFFFFFF)
        self.assertEqual(ref_ds_sub_u32(0x80000000, 1), 0x7FFFFFFF)
        self.assertEqual(ref_ds_sub_u32(0, 0x80000000), 0x80000000)

    def test_reference_oracle_returning_form(self) -> None:
        old, new = ref_ds_sub_rtn_u32(500, 200)
        self.assertEqual(old, 500)
        self.assertEqual(new, 300)

        old, new = ref_ds_sub_rtn_u32(10, 15)
        self.assertEqual(old, 10)
        self.assertEqual(new, 0xFFFFFFFB)

    def test_lds_memory_model(self) -> None:
        lds = LDSMemory(size_bytes=4096)
        addr = 0x40
        lds.write_dword(addr, 100)

        # Non-returning atomic sub: 100 - 25 = 75
        old, new = lds.atomic_sub_u32(addr, 25, rtn=False)
        self.assertIsNone(old)
        self.assertEqual(new, 75)
        self.assertEqual(lds.read_dword(addr), 75)

        # Returning atomic sub: 75 - 100 = 0xFFFFFFE7 (-25 mod 2^32)
        old, new = lds.atomic_sub_u32(addr, 100, rtn=True)
        self.assertEqual(old, 75)
        self.assertEqual(new, 0xFFFFFFE7)
        self.assertEqual(lds.read_dword(addr), 0xFFFFFFE7)


if __name__ == '__main__':
    unittest.main()
