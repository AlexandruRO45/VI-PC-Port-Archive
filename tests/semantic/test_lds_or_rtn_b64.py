from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_or_rtn_b64.

Opcode:
  - ds_or_rtn_b64 vdst, vaddr, vdata [offset:N]
      Atomic 64-bit bitwise OR on LDS memory with 64-bit destination register return:
        effective_byte = (vaddr + offset) & 0xFFFFFFFF
        idx = (effective_byte & (lds_bytes - 1)) & ~7
        old_val = LDS_qword[idx]
        LDS_qword[idx] = old_val | vdata
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in 32-bit VGPR vaddr + optional unsigned immediate byte offset.
  - Alignment: 64-bit access is 8-byte aligned (low 3 bits of byte address masked out).
  - Return value: Prior 64-bit value stored in memory at the targeted location is atomically
    returned and written to destination VGPR pair v[dst:dst+1].
  - Flags: LDS operations do not alter VCC or SCC condition flags.
  - Untyped bitvector: Operates on raw 64-bit qwords without IEEE 754 interpretation or NaN quietening.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB LDS capacity in bytes
DEFAULT_NUM_LANES: int = 64
DEFAULT_NUM_VGPRS: int = 256
DEFAULT_NUM_SGPRS: int = 106

# IEEE 754 Double-Precision Boundary Constants
POS_ZERO_F64 = 0x0000000000000000
NEG_ZERO_F64 = 0x8000000000000000
POS_INF_F64 = 0x7FF0000000000000
NEG_INF_F64 = 0xFFF0000000000000
QNAN_F64 = 0x7FF8000000000000
SNAN_F64 = 0x7FF0000000000001
MIN_NORMAL_F64 = 0x0010000000000000
MIN_SUBNORMAL_F64 = 0x0000000000000001
MAX_SUBNORMAL_F64 = 0x000FFFFFFFFFFFFF
MAX_NORMAL_F64 = 0x7FEFFFFFFFFFFFFF
ALL_ONES_F64 = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & 0x8000000000000000) else val


class LDSMemory:
    """Model of Local Data Share memory storage with 64-bit qword access."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_bytes}")
        self.size_bytes = size_bytes
        self.mem: bytearray = bytearray(size_bytes)

    def read_qword(self, byte_addr: int) -> int:
        idx = (byte_addr & (self.size_bytes - 1)) & ~7
        return struct.unpack_from("<Q", self.mem, idx)[0]

    def write_qword(self, byte_addr: int, val: int) -> None:
        idx = (byte_addr & (self.size_bytes - 1)) & ~7
        struct.pack_into("<Q", self.mem, idx, u64(val))


def ref_ds_or_rtn_b64(
    mem: LDSMemory,
    vaddr: int,
    vdata: int,
    offset: int = 0,
) -> int:
    """Reference mathematical oracle for ds_or_rtn_b64.

    Atomically computes:
      effective_byte = (vaddr + offset) mod 2^32
      old_val = LDS[effective_byte]
      LDS[effective_byte] = old_val | vdata
      return old_val
    """
    effective_byte = u32(vaddr + offset)
    old_val = mem.read_qword(effective_byte)
    new_val = u64(old_val | u64(vdata))
    mem.write_qword(effective_byte, new_val)
    return old_val


class TestLdsOrRtnB64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = LDSMemory(size_bytes=4096)

    def test_basic_or_and_return(self) -> None:
        # Pre-seed memory at byte 16 with 0x00FF00FF00FF00FF
        self.mem.write_qword(16, 0x00FF00FF00FF00FF)
        # vaddr = 16, OR with 0xFF00FF00FF00FF00
        ret = ref_ds_or_rtn_b64(self.mem, vaddr=16, vdata=0xFF00FF00FF00FF00, offset=0)
        # Old value returned
        self.assertEqual(ret, 0x00FF00FF00FF00FF)
        # Memory updated to 0xFFFFFFFFFFFFFFFF
        self.assertEqual(self.mem.read_qword(16), 0xFFFFFFFFFFFFFFFF)

    def test_or_with_zero_preserves_memory_and_returns_old(self) -> None:
        self.mem.write_qword(32, 0x123456789ABCDEF0)
        ret = ref_ds_or_rtn_b64(self.mem, vaddr=32, vdata=0, offset=0)
        self.assertEqual(ret, 0x123456789ABCDEF0)
        self.assertEqual(self.mem.read_qword(32), 0x123456789ABCDEF0)

    def test_or_with_all_ones_sets_all_bits(self) -> None:
        self.mem.write_qword(40, 0)
        ret = ref_ds_or_rtn_b64(self.mem, vaddr=40, vdata=ALL_ONES_F64, offset=0)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_qword(40), ALL_ONES_F64)

    def test_address_offset_and_8byte_alignment(self) -> None:
        # vaddr = 24, offset = 16 -> effective = 40 (aligned)
        self.mem.write_qword(40, 0x000000000000FFFF)
        # unaligned byte 43 with ~7 masking resolves to 40
        ret = ref_ds_or_rtn_b64(self.mem, vaddr=27, vdata=0xFFFF000000000000, offset=16)
        self.assertEqual(ret, 0x000000000000FFFF)
        self.assertEqual(self.mem.read_qword(40), 0xFFFF00000000FFFF)


if __name__ == "__main__":
    unittest.main()
