from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_sub_rtn_u32.

Opcode:
  - ds_sub_rtn_u32 vdst, vaddr, vdata [offset:N]
      Atomic 32-bit subtraction on LDS memory with destination register return:
        effective_byte = (vaddr + offset) & 0xFFFFFFFF
        idx = (effective_byte >> 2) & (lds_dwords - 1)
        old_val = LDS[idx]
        LDS[idx] = (old_val - vdata) & 0xFFFFFFFF
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in VGPR vaddr + optional unsigned immediate byte offset.
  - Modulo arithmetic: Subtraction wraps strictly modulo 2^32 (e.g. 0 - 1 = 0xFFFFFFFF).
  - Return value: Prior value stored in memory at the targeted location is atomically
    returned and written to destination VGPR vdst.
"""

from dataclasses import dataclass
from enum import Enum
import math
import struct
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB LDS capacity in dwords
DEFAULT_NUM_LANES: int = 64
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106

# IEEE 754 Boundary Constants
POS_ZERO_F32 = 0x00000000
NEG_ZERO_F32 = 0x80000000
POS_INF_F32 = 0x7F800000
NEG_INF_F32 = 0xFF800000
QNAN_F32 = 0x7FC00000
SNAN_F32 = 0x7F800001
MIN_NORMAL_F32 = 0x00800000
MIN_SUBNORMAL_F32 = 0x00000001
MAX_SUBNORMAL_F32 = 0x007FFFFF
MAX_NORMAL_F32 = 0x7F7FFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


class LDSMemory:
    """Model of Local Data Share memory storage."""

    def __init__(self, size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_dwords}")
        self.size_dwords = size_dwords
        self.data: list[int] = [0] * size_dwords

    def read_dword(self, index: int) -> int:
        return self.data[index & (self.size_dwords - 1)]

    def write_dword(self, index: int, val: int) -> None:
        self.data[index & (self.size_dwords - 1)] = u32(val)


def ref_ds_sub_rtn_u32(
    mem: LDSMemory,
    vaddr: int,
    vdata: int,
    offset: int = 0,
) -> int:
    """Reference mathematical oracle for ds_sub_rtn_u32.

    Atomically computes:
      old_val = LDS[idx]
      LDS[idx] = (old_val - vdata) mod 2^32
      return old_val
    """
    effective_byte = u32(vaddr + offset)
    idx = (effective_byte >> 2) & (mem.size_dwords - 1)
    old_val = mem.read_dword(idx)
    new_val = u32(old_val - u32(vdata))
    mem.write_dword(idx, new_val)
    return old_val


class TestLdsSubRtnU32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_dwords=1024)

    def test_basic_sub_and_return(self) -> None:
        self.mem.write_dword(10, 100)
        ret = ref_ds_sub_rtn_u32(self.mem, vaddr=40, vdata=30, offset=0)
        self.assertEqual(ret, 100)
        self.assertEqual(self.mem.read_dword(10), 70)

    def test_sub_underflow_wrap(self) -> None:
        self.mem.write_dword(5, 5)
        ret = ref_ds_sub_rtn_u32(self.mem, vaddr=20, vdata=10, offset=0)
        self.assertEqual(ret, 5)
        self.assertEqual(self.mem.read_dword(5), MASK32 - 4)

    def test_sub_zero_preserves_val_and_returns(self) -> None:
        self.mem.write_dword(2, 42)
        ret = ref_ds_sub_rtn_u32(self.mem, vaddr=8, vdata=0, offset=0)
        self.assertEqual(ret, 42)
        self.assertEqual(self.mem.read_dword(2), 42)

    def test_address_offset_and_wrapping(self) -> None:
        self.mem.write_dword(6, 0x12345678)
        ret = ref_ds_sub_rtn_u32(self.mem, vaddr=16, vdata=0x11111111, offset=8)
        self.assertEqual(ret, 0x12345678)
        self.assertEqual(self.mem.read_dword(6), 0x01234567)


if __name__ == "__main__":
    unittest.main()
