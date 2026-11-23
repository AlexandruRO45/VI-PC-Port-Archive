"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_read_i16.

AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
  - LDS is an on-chip, software-managed, high-throughput scratchpad memory shared across
    all work-items (lanes/threads) within a workgroup.
  - Sub-dword 16-bit Signed Read:
    - ds_read_i16 vdst, vaddr [offset:N]:
      eff_byte = (vaddr + offset) & 0xFFFFFFFF
      Loads 16-bit halfword from eff_byte in LDS memory and sign-extends bit 15 into bits [31:16]
      of the destination VGPR vdst.
      Hardware supports unaligned access: reading at byte offset 3 of a dword crosses the dword
      boundary and loads byte 0 of the adjacent dword as the upper 8 bits.
  - Active lane masking:
    - Governed by EXEC mask (Wave32 / Wave64). Inactive lanes do not mutate destination VGPRs.
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
MASK16: int = 0xFFFF
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB default LDS allocation in dwords


def sign_extend_i16(val16: int) -> int:
    """Sign-extends a 16-bit integer to a 32-bit two's-complement integer."""
    v = val16 & MASK16
    if v & 0x8000:
        return (v | 0xFFFF0000) & MASK32
    return v


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, num_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if num_dwords & (num_dwords - 1) != 0:
            raise ValueError("num_dwords must be a power of two")
        self.num_dwords: int = num_dwords
        self.size_bytes: int = num_dwords * 4
        self.bytes: bytearray = bytearray(self.size_bytes)

    def write_dword(self, byte_addr: int, val: int) -> None:
        idx = (byte_addr & (self.size_bytes - 1)) & ~3
        struct.pack_into("<I", self.bytes, idx, val & MASK32)

    def write_u16(self, byte_addr: int, val: int) -> None:
        b0 = byte_addr & (self.size_bytes - 1)
        b1 = (byte_addr + 1) & (self.size_bytes - 1)
        self.bytes[b0] = val & 0xFF
        self.bytes[b1] = (val >> 8) & 0xFF

    def read_i16(self, byte_addr: int) -> int:
        """Reads 16-bit halfword with unaligned boundary support and sign extension."""
        b0 = byte_addr & (self.size_bytes - 1)
        b1 = (byte_addr + 1) & (self.size_bytes - 1)
        raw16 = self.bytes[b0] | (self.bytes[b1] << 8)
        return sign_extend_i16(raw16)

    def read_dword(self, byte_addr: int) -> int:
        idx = (byte_addr & (self.size_bytes - 1)) & ~3
        return struct.unpack_from("<I", self.bytes, idx)[0]


class TestDSReadI16Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model tests."""

    def test_sign_extend_positive_values(self) -> None:
        test_values = [0, 1, 42, 0x00FF, 0x7FFE, 0x7FFF]
        for v in test_values:
            with self.subTest(val=hex(v)):
                self.assertEqual(sign_extend_i16(v), v)

    def test_sign_extend_negative_values(self) -> None:
        test_values = [
            (0xFFFF, 0xFFFFFFFF),       # -1
            (0xFFFE, 0xFFFFFFFE),       # -2
            (0x8000, 0xFFFF8000),       # -32768
            (0x8001, 0xFFFF8001),       # -32767
            (0xC000, 0xFFFFC000),       # -16384
        ]
        for raw, want in test_values:
            with self.subTest(raw=hex(raw)):
                self.assertEqual(sign_extend_i16(raw), want)

    def test_lds_memory_aligned_and_unaligned_read(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        # dword at 0x40 = 0x80007FFF (low 16: +32767, high 16: -32768)
        lds.write_dword(0x40, 0x80007FFF)

        # Offset 0: loads low halfword 0x7FFF -> +32767
        self.assertEqual(lds.read_i16(0x40), 0x00007FFF)
        # Offset 2: loads high halfword 0x8000 -> -32768 (0xFFFF8000)
        self.assertEqual(lds.read_i16(0x42), 0xFFFF8000)

    def test_lds_memory_cross_dword_read(self) -> None:
        lds = LDSMemory(num_dwords=1024)
        # dword at 0x40 ends with byte 0xAA (at 0x43)
        # dword at 0x44 starts with byte 0xBB (at 0x44)
        # raw 16-bit value = 0xBBAA -> sign bit is 1 (0xB >= 8) -> sign extended
        lds.write_dword(0x40, 0xAA000000)
        lds.write_dword(0x44, 0x000000BB)
        val = lds.read_i16(0x43)
        self.assertEqual(val, 0xFFFFBBAA)


if __name__ == "__main__":
    unittest.main()
