"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_smax_x4.

Models RDNA2 / GFX10 128-bit (4-DWORD) untyped buffer memory atomic signed maximum:
  buffer_atomic_smax_x4 vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR quad v[d:d+3] containing the 128-bit signed integer operand
       (packed in little-endian order: v[d]=b31..0, v[d+1]=b63..32, v[d+2]=b95..64, v[d+3]=b127..96).
       If 'glc' is enabled, the 128-bit pre-operation value at memory is returned to v[d:d+3].
       If 'glc' is omitted, vdata registers retain their input values.
     - vaddr: VGPR containing byte or index offset (when offen or idxen is enabled).
     - srsrc: 128-bit Buffer Resource Descriptor (4 SGPRs s[r:r+3]):
         Dword 0: Base address low [31:0]
         Dword 1: Base address high [15:0], STRIDE [31:16]
         Dword 2: NUM_RECORDS (buffer limit / capacity in bytes)
         Dword 3: Format, element size, resource type
     - soffset: Scalar SGPR or immediate byte offset.
     - inst_offset: 12-bit unsigned immediate byte offset [0, 4095].
  2. Addressing and Bounds Checking:
     effective_offset = (soffset & 0xFFFFFFFF) + (inst_offset & 0xFFF) + (vaddr & 0xFFFFFFFF if offen else 0)
     If effective_offset + 16 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic smax is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to v[d:d+3].
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory_128[effective_address]
       new_val = max(s128(old_val), s128(vdata))
       memory_128[effective_address] = u128(new_val)
       if glc: vdata = old_val
  4. Inactive Lanes (EXEC mask):
     Lanes disabled by EXEC mask never touch memory and do not modify their VGPRs.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
MASK128: int = (1 << 128) - 1
SIGN_BIT128: int = 1 << 127


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def u128(val: int) -> int:
    """Normalize integer to unsigned 128-bit range [0, 2^128 - 1]."""
    return val & MASK128


def s128(val: int) -> int:
    """Interpret a 128-bit unsigned integer as signed two's-complement."""
    val &= MASK128
    return val - (1 << 128) if (val & SIGN_BIT128) else val


def pack_u128(d0: int, d1: int, d2: int, d3: int) -> int:
    """Pack four 32-bit dwords into a 128-bit integer (little-endian)."""
    return ((d3 & MASK32) << 96) | ((d2 & MASK32) << 64) | ((d1 & MASK32) << 32) | (d0 & MASK32)


def unpack_u128(val: int) -> tuple[int, int, int, int]:
    """Unpack a 128-bit integer into four 32-bit dwords (little-endian)."""
    val &= MASK128
    return (
        val & MASK32,
        (val >> 32) & MASK32,
        (val >> 64) & MASK32,
        (val >> 96) & MASK32,
    )


@dataclasses.dataclass(frozen=True)
class BufferResourceDescriptor:
    """AMD GFX10 128-bit Buffer Resource Descriptor (V#)."""
    base_address: int
    stride: int
    num_records: int
    format_flags: int = 0

    @classmethod
    def from_dwords(cls, d0: int, d1: int, d2: int, d3: int) -> BufferResourceDescriptor:
        base_lo = d0 & MASK32
        base_hi = d1 & 0xFFFF
        base = (base_hi << 32) | base_lo
        stride = (d1 >> 16) & 0xFFFF
        num_records = d2 & MASK32
        return cls(base_address=base, stride=stride, num_records=num_records, format_flags=d3)

    def to_dwords(self) -> tuple[int, int, int, int]:
        d0 = self.base_address & MASK32
        d1 = ((self.base_address >> 32) & 0xFFFF) | ((self.stride & 0xFFFF) << 16)
        d2 = self.num_records & MASK32
        d3 = self.format_flags & MASK32
        return d0, d1, d2, d3


class BufferMemory:
    """Byte-addressable memory buffer simulating GPU global / buffer device memory."""

    def __init__(self, size: int = 65536) -> None:
        self.data: bytearray = bytearray(size)

    def read_u32(self, addr: int) -> int:
        if addr < 0 or addr + 4 > len(self.data):
            return 0
        return struct.unpack_from("<I", self.data, addr)[0]

    def write_u32(self, addr: int, val: int) -> None:
        if addr < 0 or addr + 4 > len(self.data):
            return
        struct.pack_into("<I", self.data, addr, val & MASK32)

    def read_u64(self, addr: int) -> int:
        if addr < 0 or addr + 8 > len(self.data):
            return 0
        return struct.unpack_from("<Q", self.data, addr)[0]

    def write_u64(self, addr: int, val: int) -> None:
        if addr < 0 or addr + 8 > len(self.data):
            return
        struct.pack_into("<Q", self.data, addr, val & MASK64)

    def read_u128(self, addr: int) -> int:
        if addr < 0 or addr + 16 > len(self.data):
            return 0
        lo, hi = struct.unpack_from("<QQ", self.data, addr)
        return (hi << 64) | lo

    def write_u128(self, addr: int, val: int) -> None:
        if addr < 0 or addr + 16 > len(self.data):
            return
        lo = val & MASK64
        hi = (val >> 64) & MASK64
        struct.pack_into("<QQ", self.data, addr, lo, hi)


def ref_buffer_atomic_smax_x4(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    smax_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_smax_x4."""
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check: 128-bit access requires 16 contiguous bytes
    if effective_offset + 16 > desc.num_records:
        return (0 if glc else (smax_val & MASK128)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u128(phys_addr)
    new_val = max(s128(old_val), s128(smax_val))
    mem.write_u128(phys_addr, u128(new_val))

    ret_val = old_val if glc else (smax_val & MASK128)
    return ret_val, True

class TestBufferAtomicSmaxX4Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_smax_x4_positive_values(self) -> None:
        val_100 = 100
        val_250 = 250
        self.mem.write_u128(64, val_100)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            smax_val=val_250,
            glc=False,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_250)
        self.assertEqual(self.mem.read_u128(64), val_250)

    def test_basic_smax_x4_negative_vs_positive(self) -> None:
        val_neg50 = (-50) & MASK128
        val_pos20 = 20
        self.mem.write_u128(64, val_neg50)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            smax_val=val_pos20,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_neg50)
        self.assertEqual(self.mem.read_u128(64), val_pos20)

    def test_basic_smax_x4_two_negatives(self) -> None:
        val_neg100 = (-100) & MASK128
        val_neg20 = (-20) & MASK128
        self.mem.write_u128(128, val_neg100)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=128,
            smax_val=val_neg20,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_neg100)
        self.assertEqual(self.mem.read_u128(128), val_neg20)

    def test_smax_x4_no_update_when_smaller(self) -> None:
        val_500 = 500
        val_200 = 200
        self.mem.write_u128(32, val_500)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=32,
            smax_val=val_200,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, val_500)
        self.assertEqual(self.mem.read_u128(32), val_500)

    def test_smax_x4_oob_rejection(self) -> None:
        canary = pack_u128(0x11111111, 0x22222222, 0x33333333, 0x44444444)
        self.mem.write_u128(1010, canary)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1010,
            smax_val=pack_u128(0x55555555, 0x66666666, 0x77777777, 0x88888888),
            glc=True,
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u128(1010), canary)

    def test_smax_x4_boundary_exact_fit(self) -> None:
        self.mem.write_u128(1008, 1000)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1008,
            smax_val=5000,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 1000)
        self.assertEqual(self.mem.read_u128(1008), 5000)

    def test_smax_x4_offset_combination(self) -> None:
        self.mem.write_u128(208, 1234)
        ret, in_bounds = ref_buffer_atomic_smax_x4(
            self.mem,
            self.desc,
            soffset=128,
            vaddr=16,
            smax_val=5678,
            inst_offset=64,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 1234)
        self.assertEqual(self.mem.read_u128(208), 5678)

    def test_pack_unpack_u128_roundtrip(self) -> None:
        d0, d1, d2, d3 = 0xDEADBEEF, 0xCAFEBABE, 0x01234567, 0x89ABCDEF
        packed = pack_u128(d0, d1, d2, d3)
        unpacked = unpack_u128(packed)
        self.assertEqual(unpacked, (d0, d1, d2, d3))

if __name__ == "__main__":
    unittest.main()
