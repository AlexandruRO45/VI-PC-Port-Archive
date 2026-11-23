"""Semantic interpreter tests and translator verification for 32-bit unsigned LDS atomic increment with wrap (ds_inc_u32, ds_inc_rtn_u32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    py -I tests/semantic/test_lds_inc_u32_hw.py
    py -m unittest discover -s tests/semantic -p test_lds_inc_u32_hw.py
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


def ref_ds_inc_u32(curr_mem: int, data_val: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_inc_u32: M = (M >= DATA) ? 0 : M + 1 (unsigned)."""
    cur = curr_mem & MASK32
    data = data_val & MASK32
    return 0 if cur >= data else (cur + 1) & MASK32


def ref_ds_inc_rtn_u32(curr_mem: int, data_val: int) -> Tuple[int, int]:
    """Reference returning atomic increment with wrap: returns (old_mem, new_mem)."""
    old = curr_mem & MASK32
    new = ref_ds_inc_u32(old, data_val)
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

    def atomic_inc_u32(self, byte_addr: int, data_val: int, offset: int = 0, rtn: bool = False) -> Tuple[Optional[int], int]:
        idx = self.resolve_dword_index(byte_addr, offset)
        old_val = self.storage[idx]
        new_val = ref_ds_inc_u32(old_val, data_val)
        self.storage[idx] = new_val
        return (old_val if rtn else None), new_val

    def get_bank(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        return (eff_addr >> 2) & 31

class TestDSIncU32OracleAndArchitecturalModel(unittest.TestCase):
    """Part 1: Reference mathematical oracle and architectural model tests."""

    def test_reference_oracle_basic_increments(self) -> None:
        # Standard increment: cur < data
        self.assertEqual(ref_ds_inc_u32(0, 10), 1)
        self.assertEqual(ref_ds_inc_u32(1, 10), 2)
        self.assertEqual(ref_ds_inc_u32(8, 10), 9)

    def test_reference_oracle_wrap_resets(self) -> None:
        # Wrap when cur == data
        self.assertEqual(ref_ds_inc_u32(10, 10), 0)

        # Wrap when cur > data
        self.assertEqual(ref_ds_inc_u32(15, 10), 0)

        # When data == 0: cur >= 0 is always True (unsigned), so always resets to 0
        self.assertEqual(ref_ds_inc_u32(0, 0), 0)
        self.assertEqual(ref_ds_inc_u32(5, 0), 0)

    def test_reference_oracle_returning_form(self) -> None:
        old, new = ref_ds_inc_rtn_u32(3, 5)
        self.assertEqual(old, 3)
        self.assertEqual(new, 4)

        old, new = ref_ds_inc_rtn_u32(5, 5)
        self.assertEqual(old, 5)
        self.assertEqual(new, 0)

    def test_lds_memory_model(self) -> None:
        lds = LDSMemory(size_bytes=4096)
        addr = 0x40
        lds.write_dword(addr, 4)

        # Non-returning atomic inc with data=10: 4 < 10 -> 5
        old, new = lds.atomic_inc_u32(addr, 10, rtn=False)
        self.assertIsNone(old)
        self.assertEqual(new, 5)
        self.assertEqual(lds.read_dword(addr), 5)

        # Returning atomic inc with data=5: 5 >= 5 -> resets to 0
        old, new = lds.atomic_inc_u32(addr, 5, rtn=True)
        self.assertEqual(old, 5)
        self.assertEqual(new, 0)
        self.assertEqual(lds.read_dword(addr), 0)


if __name__ == '__main__':
    unittest.main()
