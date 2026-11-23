"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_wrxchg_rtn_b32.

AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
  - LDS is an on-chip, software-managed, high-throughput scratchpad memory shared across
    all work-items (lanes/threads) within a workgroup.
  - Single 32-bit Atomic Exchange / Swap:
    - ds_wrxchg_rtn_b32 vdst, vaddr, vdata [offset:N]:
      Returning form (atomic swap):
        eff_byte = (vaddr + offset) & 0xFFFFFFFF
        dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[dword_idx]
        LDS[dword_idx] = vdata & 0xFFFFFFFF
        vdst = old_val
      Atomically swaps the memory contents with vdata and stores pre-atomic value in vdst.
  - Active lane masking:
    - Vector operations governed by EXEC mask (32-bit for Wave32, 64-bit for Wave64).
    - Inactive lanes do not mutate LDS storage and leave destination VGPRs unmodified.
  - Hardware wrapping & alignment:
    - Byte addresses are shifted right by 2, dropping unaligned low bits [1:0].
    - Resulting dword index is masked with (lds_dwords - 1).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import random
import struct
import sys
import unittest
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lds_run as L  # noqa: E402
import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB default LDS allocation in dwords


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, num_dwords: int = DEFAULT_LDS_DWORDS, bounds_check: bool = False) -> None:
        if num_dwords & (num_dwords - 1) != 0:
            raise ValueError("num_dwords must be a power of two")
        self.num_dwords: int = num_dwords
        self.size_bytes: int = num_dwords * 4
        self.dwords: List[int] = [0] * num_dwords
        self.bounds_check: bool = bounds_check

    def read_dword(self, byte_addr: int) -> int:
        eff = byte_addr & MASK32
        idx = eff >> 2
        if self.bounds_check and idx >= self.num_dwords:
            return 0
        idx &= self.num_dwords - 1
        return self.dwords[idx]

    def write_dword(self, byte_addr: int, val: int) -> None:
        eff = byte_addr & MASK32
        idx = eff >> 2
        if self.bounds_check and idx >= self.num_dwords:
            return
        idx &= self.num_dwords - 1
        self.dwords[idx] = val & MASK32

    def atomic_xchg(self, byte_addr: int, data: int) -> int:
        """Atomic exchange (swap). Returns pre-atomic dword value."""
        eff = byte_addr & MASK32
        idx = eff >> 2
        if self.bounds_check and idx >= self.num_dwords:
            return 0
        idx &= self.num_dwords - 1
        old = self.dwords[idx]
        self.dwords[idx] = data & MASK32
        return old


def ds_wrxchg_rtn_b32_oracle(old_val: int, data: int) -> Tuple[int, int]:
    """Pure reference oracle for 32-bit atomic swap. Returns (old_val, new_val)."""
    return (old_val & MASK32), (data & MASK32)


def ds_wrxchg_rtn_b32_execute_model(
    lds: LDSMemory, vaddr: int, vdata: int, offset: int = 0
) -> Tuple[int, int]:
    """Executes returning ds_wrxchg_rtn_b32. Returns (old_val, new_val)."""
    eff_byte = (vaddr + offset) & MASK32
    old_val = lds.atomic_xchg(eff_byte, vdata)
    new_val = lds.read_dword(eff_byte)
    return old_val, new_val


class TestDSWrxchgRtnB32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_swap_values(self) -> None:
        test_pairs = [
            (0x12345678, 0x9ABCDEF0),
            (0x00000000, 0xFFFFFFFF),
            (0xAAAAAAAA, 0x55555555),
            (0xCAFEBABE, 0xDEADBEEF),
        ]
        for old, new in test_pairs:
            with self.subTest(old=hex(old), new=hex(new)):
                returned_old, left_new = ds_wrxchg_rtn_b32_oracle(old, new)
                self.assertEqual(returned_old, old)
                self.assertEqual(left_new, new)

    def test_lds_memory_address_wrapping_and_alignment(self) -> None:
        lds = LDSMemory(num_dwords=1024)  # 4 KiB
        lds.write_dword(0x40, 0xCAFEBABE)
        for off in (0, 1, 2, 3):
            self.assertEqual(lds.read_dword(0x40 + off), 0xCAFEBABE)
        self.assertEqual(lds.read_dword(0x40 + 4096), 0xCAFEBABE)

    def test_model_execution_swap(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x100, 0x11112222)
        old_val, new_val = ds_wrxchg_rtn_b32_execute_model(lds, 0x100, 0x33334444)
        self.assertEqual(old_val, 0x11112222)
        self.assertEqual(new_val, 0x33334444)
        self.assertEqual(lds.read_dword(0x100), 0x33334444)


if __name__ == "__main__":
    unittest.main()
