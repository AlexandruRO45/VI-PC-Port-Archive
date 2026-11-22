"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_smax_x2.

Models AMD RDNA2 / GFX10 MUBUF 64-bit untyped buffer memory atomic signed maximum:
  buffer_atomic_smax_x2 vdata, vaddr, srsrc, soffset [offset:N] [offen] [idxen] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR pair v[d:d+1] containing the 64-bit signed integer operand.
       If 'glc' is enabled, the 64-bit pre-operation value at memory is returned to v[d:d+1].
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
     If effective_offset + 8 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic smax is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to v[d:d+1].
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory_64[effective_address]
       new_val = max(s64(old_val), s64(vdata))
       memory_64[effective_address] = u64(new_val)
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
SIGN_BIT64: int = 0x8000000000000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned integer as signed two's-complement."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_BIT64) else val


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


def ref_buffer_atomic_smax_x2(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    smax_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_smax_x2.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS (64-bit access requires 8 bytes)
    if effective_offset + 8 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc else smax_val
        return (0 if glc else (smax_val & MASK64)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u64(phys_addr)
    new_val = max(s64(old_val), s64(smax_val))
    mem.write_u64(phys_addr, u64(new_val))

    ret_val = old_val if glc else (smax_val & MASK64)
    return ret_val, True


class TestBufferAtomicSmaxX2Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_smax_x2_positive_values(self) -> None:
        self.mem.write_u64(64, 100)
        ret, in_bounds = ref_buffer_atomic_smax_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            smax_val=250,
            glc=False,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(self.mem.read_u64(64), 250)
        self.assertEqual(ret, 250)

    def test_basic_smax_x2_negative_vs_positive(self) -> None:
        # Memory holds -50 (signed), smax_val is 20 -> max(-50, 20) = 20
        self.mem.write_u64(64, u64(-50))
        ret, in_bounds = ref_buffer_atomic_smax_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            smax_val=20,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, u64(-50))
        self.assertEqual(self.mem.read_u64(64), 20)

    def test_basic_smax_x2_two_negatives(self) -> None:
        # Memory holds -100, smax_val is -20 -> max(-100, -20) = -20
        self.mem.write_u64(128, u64(-100))
        ret, in_bounds = ref_buffer_atomic_smax_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=128,
            smax_val=u64(-20),
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, u64(-100))
        self.assertEqual(self.mem.read_u64(128), u64(-20))

    def test_smax_x2_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u64(200, 500)
        ret, in_bounds = ref_buffer_atomic_smax_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=200,
            smax_val=100,
            glc=True,
        )
        self.assertTrue(in_bounds)
        # glc returns 500
        self.assertEqual(ret, 500)
        # memory retains 500 since max(500, 100) = 500
        self.assertEqual(self.mem.read_u64(200), 500)

    def test_smax_x2_oob_rejection(self) -> None:
        # num_records is 1024. Offset 1020 has 1020 + 8 = 1028 > 1024 (OOB)
        self.mem.write_u64(1020, 777)
        ret, in_bounds = ref_buffer_atomic_smax_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1020,
            smax_val=999,
            glc=True,
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u64(1020), 777)

    def test_smax_x2_boundary_exact_fit(self) -> None:
        # Offset 1016 has 1016 + 8 == 1024 (exact fit, in bounds)
        self.mem.write_u64(1016, 50)
        ret, in_bounds = ref_buffer_atomic_smax_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1016,
            smax_val=80,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 50)
        self.assertEqual(self.mem.read_u64(1016), 80)

    def test_smax_x2_offset_combination(self) -> None:
        # Total offset = soffset(128) + inst_offset(64) + vaddr(8) = 200
        self.mem.write_u64(200, 10)
        ret, in_bounds = ref_buffer_atomic_smax_x2(
            self.mem,
            self.desc,
            soffset=128,
            vaddr=8,
            smax_val=40,
            inst_offset=64,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 10)
        self.assertEqual(self.mem.read_u64(200), 40)

    def test_smax_x2_signed_extremes(self) -> None:
        min_int64 = 0x8000000000000000  # -9223372036854775808
        max_int64 = 0x7FFFFFFFFFFFFFFF  # 9223372036854775807
        self.mem.write_u64(0, min_int64)
        ret, in_bounds = ref_buffer_atomic_smax_x2(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=0,
            smax_val=max_int64,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, min_int64)
        self.assertEqual(self.mem.read_u64(0), max_int64)

    def test_descriptor_roundtrip(self) -> None:
        desc = BufferResourceDescriptor(
            base_address=0x00007FFF12340000,
            stride=32,
            num_records=131072,
            format_flags=0x40,
        )
        d0, d1, d2, d3 = desc.to_dwords()
        restored = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)
        self.assertEqual(desc, restored)


if __name__ == "__main__":
    unittest.main()
