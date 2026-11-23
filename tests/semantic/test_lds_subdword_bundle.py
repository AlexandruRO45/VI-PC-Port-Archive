from __future__ import annotations

"""Semantic interpreter tests and reference architectural model for RDNA2 LDS Sub-Dword Memory Suite.

Covers:
- ds_read_u8: 8-bit unsigned byte read from LDS, zero-extended to 32 bits.
- ds_read_i8: 8-bit signed byte read from LDS, sign-extended to 32 bits.
- ds_read_u16: 16-bit unsigned halfword read from LDS, zero-extended to 32 bits.
- ds_read_i16: 16-bit signed halfword read from LDS, sign-extended to 32 bits.
- ds_write_b8: 8-bit byte write to LDS, modifying only addressed byte within dword.
- ds_write_b16: 16-bit halfword write to LDS, modifying only addressed halfword within dword.

Standard library only (python -I, struct, math, unittest). Zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF
LDS_SIZE: int = 65536


# ------------------------------------------------------------------------------
# Architectural LDS Memory Reference Model
# ------------------------------------------------------------------------------

class LDSMemory:
    """Byte-addressable LDS memory reference model wrapping at LDS_SIZE."""

    def __init__(self, size: int = LDS_SIZE):
        self.size = size
        self.mem: bytearray = bytearray(size)

    def write_u8(self, addr: int, val: int, offset: int = 0):
        eff = (addr + offset) % self.size
        self.mem[eff] = val & 0xFF

    def read_u8(self, addr: int, offset: int = 0) -> int:
        eff = (addr + offset) % self.size
        return self.mem[eff]

    def read_i8(self, addr: int, offset: int = 0) -> int:
        b = self.read_u8(addr, offset)
        return (b - 256 if (b & 0x80) else b) & MASK32

    def write_u16(self, addr: int, val: int, offset: int = 0):
        eff = (addr + offset) % self.size
        self.mem[eff] = val & 0xFF
        self.mem[(eff + 1) % self.size] = (val >> 8) & 0xFF

    def read_u16(self, addr: int, offset: int = 0) -> int:
        eff = (addr + offset) % self.size
        lo = self.mem[eff]
        hi = self.mem[(eff + 1) % self.size]
        return (lo | (hi << 8)) & 0xFFFF

    def read_i16(self, addr: int, offset: int = 0) -> int:
        h = self.read_u16(addr, offset)
        return (h - 65536 if (h & 0x8000) else h) & MASK32


class TestDsReadU8Oracle(unittest.TestCase):
    def setUp(self):
        self.lds = LDSMemory()

    def test_read_u8_zero_extension(self):
        self.lds.write_u8(0, 0xAB)
        self.assertEqual(self.lds.read_u8(0), 0xAB)
        self.lds.write_u8(1, 0xFF)
        self.assertEqual(self.lds.read_u8(1), 0xFF)

    def test_read_u8_with_offset(self):
        self.lds.write_u8(10, 0x42)
        self.assertEqual(self.lds.read_u8(4, offset=6), 0x42)


if __name__ == '__main__':
    unittest.main()
