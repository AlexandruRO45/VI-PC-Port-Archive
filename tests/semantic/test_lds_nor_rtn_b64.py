"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_nor_rtn_b64.

Opcode:
  - ds_nor_rtn_b64 vdst, vaddr, vdata [offset:N]
      Atomic 64-bit bitwise NOR on LDS memory with 64-bit destination register return:
        effective_byte = (vaddr + offset) & 0xFFFFFFFF
        idx = (effective_byte & (lds_bytes - 1)) & ~7
        old_val = LDS_qword[idx]
        LDS_qword[idx] = ~(old_val | vdata) & 0xFFFFFFFFFFFFFFFF
        vdst = old_val

Architecture & Specification:
  - Target: AMD RDNA2 / GFX10 Local Data Share (LDS).
  - Memory: On-chip software-managed scratchpad shared among threads of a workgroup.
  - Addressing: Base byte address in 32-bit VGPR vaddr + optional unsigned immediate byte offset.
  - Alignment: 64-bit access is 8-byte aligned (low 3 bits of byte address masked out).
  - Return value: Prior 64-bit value stored in memory at the targeted location is atomically
    returned and written to destination VGPR pair v[dst:dst+1].
  - Flags: LDS operations do not alter VCC or SCC condition flags.
  - Bitwise NOR: Operates on raw 64-bit qwords without IEEE 754 interpretation or NaN quietening.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Set, Tuple, Union

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET16_MASK: int = 0xFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB LDS capacity in bytes
DEFAULT_NUM_VGPRS: int = 256
NUM_LDS_BANKS: int = 32

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


def oracle_nor_b64(a: int, b: int) -> int:
    """Reference 64-bit bitwise NOR returning 64-bit unsigned bit pattern."""
    ua = a & MASK64
    ub = b & MASK64
    return (~(ua | ub)) & MASK64


def ref_ds_nor_rtn_b64(current_mem: int, operand: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_nor_rtn_b64.

    Returns:
        (old_val, new_mem) as unsigned 64-bit integers.
    """
    old_val = u64(current_mem)
    new_mem = oracle_nor_b64(old_val, operand)
    return old_val, new_mem


class LDSMemory:
    """Model of Local Data Share memory storage with 64-bit qword access."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two: {size_bytes}")
        self.size_bytes = size_bytes
        self.mem: bytearray = bytearray(size_bytes)

    def get_qword_offset(self, byte_addr: int, offset: int = 0) -> int:
        """Calculate effective wrapped byte address, aligned to 8-byte qword boundary."""
        eff_addr = (byte_addr + (offset & OFFSET16_MASK)) & MASK32
        return (eff_addr & (self.size_bytes - 1)) & ~7

    def read_qword(self, byte_addr: int, offset: int = 0) -> int:
        idx = self.get_qword_offset(byte_addr, offset)
        return struct.unpack_from("<Q", self.mem, idx)[0]

    def write_qword(self, byte_addr: int, val: int, offset: int = 0) -> None:
        idx = self.get_qword_offset(byte_addr, offset)
        struct.pack_into("<Q", self.mem, idx, u64(val))

    def atomic_nor_b64(self, byte_addr: int, val: int, offset: int = 0) -> int:
        idx = self.get_qword_offset(byte_addr, offset)
        old_val = struct.unpack_from("<Q", self.mem, idx)[0]
        new_val = oracle_nor_b64(old_val, val)
        struct.pack_into("<Q", self.mem, idx, new_val)
        return new_val

    def atomic_nor_rtn_b64(self, byte_addr: int, val: int, offset: int = 0) -> Tuple[int, int]:
        idx = self.get_qword_offset(byte_addr, offset)
        old_val = struct.unpack_from("<Q", self.mem, idx)[0]
        new_val = oracle_nor_b64(old_val, val)
        struct.pack_into("<Q", self.mem, idx, new_val)
        return old_val, new_val


class TestLdsNorRtnB64Oracle(unittest.TestCase):
    """Unit tests for 64-bit mathematical NOR oracle and LDS architectural model."""

    def setUp(self) -> None:
        self.lds = LDSMemory(DEFAULT_LDS_SIZE_BYTES)

    def test_oracle_nor_b64_truth_table(self) -> None:
        # 0 NOR 0 = 0xFFFFFFFFFFFFFFFF
        self.assertEqual(oracle_nor_b64(0, 0), MASK64)
        # MASK64 NOR 0 = 0
        self.assertEqual(oracle_nor_b64(MASK64, 0), 0)
        # 0 NOR MASK64 = 0
        self.assertEqual(oracle_nor_b64(0, MASK64), 0)
        # MASK64 NOR MASK64 = 0
        self.assertEqual(oracle_nor_b64(MASK64, MASK64), 0)

    def test_oracle_nor_b64_bit_patterns(self) -> None:
        # Alternating patterns
        self.assertEqual(oracle_nor_b64(0x5555555555555555, 0xAAAAAAAAAAAAAAAA), 0)
        self.assertEqual(oracle_nor_b64(0x5555555555555555, 0), 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(oracle_nor_b64(0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF), 0xF000F000F000F000)

    def test_ref_ds_nor_rtn_b64(self) -> None:
        old_val, new_val = ref_ds_nor_rtn_b64(0, 0)
        self.assertEqual(old_val, 0)
        self.assertEqual(new_val, MASK64)

        old_val2, new_val2 = ref_ds_nor_rtn_b64(0x0123456789ABCDEF, 0xFEDCBA9876543210)
        expected = (~(0x0123456789ABCDEF | 0xFEDCBA9876543210)) & MASK64
        self.assertEqual(old_val2, 0x0123456789ABCDEF)
        self.assertEqual(new_val2, expected)

    def test_lds_qword_alignment_and_wrapping(self) -> None:
        # Byte addresses 0 to 7 all map to qword index 0
        for b in range(8):
            self.assertEqual(self.lds.get_qword_offset(b), 0)
        # Byte addresses 8 to 15 map to offset 8
        for b in range(8, 16):
            self.assertEqual(self.lds.get_qword_offset(b), 8)
        # Immediate offset addition
        self.assertEqual(self.lds.get_qword_offset(8, offset=8), 16)
        # Wrap around 64 KiB (65536)
        self.assertEqual(self.lds.get_qword_offset(65536), 0)
        self.assertEqual(self.lds.get_qword_offset(65544), 8)

    def test_lds_read_write_qword(self) -> None:
        self.lds.write_qword(16, 0xDEADBEEFCAFEBABE)
        self.assertEqual(self.lds.read_qword(16), 0xDEADBEEFCAFEBABE)

    def test_lds_atomic_nor_b64(self) -> None:
        self.lds.write_qword(32, 0x5555555555555555)
        old_val, new_val = self.lds.atomic_nor_rtn_b64(32, 0)
        self.assertEqual(old_val, 0x5555555555555555)
        self.assertEqual(new_val, 0xAAAAAAAAAAAAAAAA)
        self.assertEqual(self.lds.read_qword(32), 0xAAAAAAAAAAAAAAAA)


if __name__ == "__main__":
    unittest.main()
