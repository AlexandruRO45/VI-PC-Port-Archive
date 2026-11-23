"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_mskor_b32 and ds_mskor_rtn_b32.

AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
  - LDS is an on-chip, software-managed, high-throughput scratchpad memory shared across
    all work-items (lanes/threads) within a workgroup.
  - 32-bit Atomic Masked OR (Bitfield Insert / Mask-and-OR):
    - RDNA2 ISA: tmp = M[A]; M[A] = (tmp & ~MASK) | DATA
    - ds_mskor_b32 vaddr, vmask, vdata [offset:N]:
      Non-returning form:
        eff_byte = (vaddr + offset) & 0xFFFFFFFF
        dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[dword_idx]
        LDS[dword_idx] = ((old_val & ~vmask) | vdata) & 0xFFFFFFFF
      The prior value at LDS[dword_idx] is discarded.
    - ds_mskor_rtn_b32 vdst, vaddr, vmask, vdata [offset:N]:
      Returning form:
        Atomically updates memory according to the same formula and returns pre-atomic
        value old_val in destination VGPR vdst.
  - Active lane masking:
    - Governed by EXEC mask (Wave32 / Wave64). Inactive lanes do not mutate LDS storage
      and leave destination VGPRs unmodified.
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
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB default LDS allocation in dwords


def ds_mskor_b32_oracle(old_val: int, mask: int, data: int) -> int:
    """Pure reference oracle for RDNA2 ds_mskor_b32: (old & ~mask) | data."""
    old = old_val & MASK32
    m = mask & MASK32
    d = data & MASK32
    return ((old & (~m & MASK32)) | d) & MASK32


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, num_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if num_dwords & (num_dwords - 1) != 0:
            raise ValueError("num_dwords must be a power of two")
        self.num_dwords: int = num_dwords
        self.size_bytes: int = num_dwords * 4
        self.dwords: List[int] = [0] * num_dwords

    def read_dword(self, byte_addr: int) -> int:
        idx = (byte_addr & MASK32) >> 2
        idx &= self.num_dwords - 1
        return self.dwords[idx]

    def write_dword(self, byte_addr: int, val: int) -> None:
        idx = (byte_addr & MASK32) >> 2
        idx &= self.num_dwords - 1
        self.dwords[idx] = val & MASK32

    def atomic_mskor(self, byte_addr: int, mask: int, data: int) -> int:
        """Atomic masked OR. Returns pre-atomic dword value."""
        idx = (byte_addr & MASK32) >> 2
        idx &= self.num_dwords - 1
        old = self.dwords[idx]
        self.dwords[idx] = ds_mskor_b32_oracle(old, mask, data)
        return old


def ds_mskor_b32_execute_model(
    lds: LDSMemory, vaddr: int, vmask: int, vdata: int, offset: int = 0
) -> Tuple[Optional[int], int]:
    """Executes non-returning ds_mskor_b32. Returns (None, new_val)."""
    eff_byte = (vaddr + offset) & MASK32
    old_val = lds.atomic_mskor(eff_byte, vmask, vdata)
    new_val = lds.read_dword(eff_byte)
    return None, new_val


def ds_mskor_rtn_b32_execute_model(
    lds: LDSMemory, vaddr: int, vmask: int, vdata: int, offset: int = 0
) -> Tuple[int, int]:
    """Executes returning ds_mskor_rtn_b32. Returns (old_val, new_val)."""
    eff_byte = (vaddr + offset) & MASK32
    old_val = lds.atomic_mskor(eff_byte, vmask, vdata)
    new_val = lds.read_dword(eff_byte)
    return old_val, new_val


class TestDSMskorB32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_oracle_byte_replacement(self) -> None:
        # Replace second byte (bits [23:16]) of 0x12345678 with 0xAA -> 0x12AA5678
        res = ds_mskor_b32_oracle(0x12345678, 0x00FF0000, 0x00AA0000)
        self.assertEqual(res, 0x12AA5678)

    def test_oracle_clear_bits_when_data_is_zero(self) -> None:
        # mask = 0x0F0F0F0F, data = 0 -> clears masked nibbles
        res = ds_mskor_b32_oracle(0xFFFFFFFF, 0x0F0F0F0F, 0x00000000)
        self.assertEqual(res, 0xF0F0F0F0)

    def test_oracle_set_bits_when_mask_matches_data(self) -> None:
        # mask = 0x0F0F0F0F, data = 0x0F0F0F0F -> sets masked nibbles
        res = ds_mskor_b32_oracle(0x00000000, 0x0F0F0F0F, 0x0F0F0F0F)
        self.assertEqual(res, 0x0F0F0F0F)

    def test_lds_memory_address_wrapping_and_alignment(self) -> None:
        lds = LDSMemory(num_dwords=1024)  # 4 KiB
        lds.write_dword(0x40, 0x12345678)
        for off in (0, 1, 2, 3):
            self.assertEqual(lds.read_dword(0x40 + off), 0x12345678)
        self.assertEqual(lds.read_dword(0x40 + 4096), 0x12345678)

    def test_model_non_returning_and_returning(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        lds.write_dword(0x100, 0x12345678)

        old, new = ds_mskor_rtn_b32_execute_model(lds, 0x100, 0x00FF0000, 0x00AA0000)
        self.assertEqual(old, 0x12345678)
        self.assertEqual(new, 0x12AA5678)

        rtn, new2 = ds_mskor_b32_execute_model(lds, 0x100, 0x000000FF, 0x00000099)
        self.assertIsNone(rtn)
        self.assertEqual(new2, 0x12AA5699)


if __name__ == "__main__":
    unittest.main()
