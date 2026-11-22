"""Semantic interpreter tests and reference mathematical oracle for RDNA2 buffer_atomic_swap.

Models AMD RDNA2 / GFX10 MUBUF untyped buffer memory atomic swap:
  buffer_atomic_swap vdata, vaddr, srsrc, soffset [offset:N] [glc] [slc]

RDNA2 Buffer Atomic Architecture & Semantics:
  1. Instruction Operands:
     - vdata: VGPR containing the 32-bit dword value to swap into memory.
       If 'glc' (Globally Coherent) is enabled, the pre-operation dword value
       at memory is returned to vdata.
     - vaddr: VGPR containing the byte or index offset (when offen or idxen is enabled).
     - srsrc: 128-bit Buffer Resource Descriptor (4 SGPRs s[r:r+3]):
         Dword 0: Base address low [31:0]
         Dword 1: Base address high [15:0], STRIDE [31:16]
         Dword 2: NUM_RECORDS (buffer limit / capacity in bytes)
         Dword 3: Format, element size, resource type
     - soffset: Scalar SGPR or immediate byte offset.
     - inst_offset: 12-bit unsigned immediate byte offset [0, 4095].
  2. Addressing and Bounds Checking:
     effective_offset = soffset + inst_offset + (vaddr if offen else 0)
     If effective_offset + 4 > NUM_RECORDS:
       Access is Out-Of-Bounds (OOB).
       The atomic swap is suppressed (memory is NOT modified).
       If 'glc' is set, 0 is returned to vdata.
  3. Atomic Operation:
     For in-bounds accesses:
       old_val = memory[effective_address]
       new_val = vdata
       memory[effective_address] = new_val
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
SIGN_BIT32: int = 0x80000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


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


def ref_buffer_atomic_swap(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    swap_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_swap.

    Returns:
      (return_val, was_in_bounds)
    """
    effective_offset = (soffset & MASK32) + (inst_offset & 0xFFF)
    if offen:
        effective_offset += (vaddr & MASK32)

    # Hardware bounds check against NUM_RECORDS
    if effective_offset + 4 > desc.num_records:
        # Out-of-bounds: memory untouched, returns 0 if glc
        return (0 if glc else (swap_val & MASK32)), False

    phys_addr = (desc.base_address + effective_offset) & MASK64
    old_val = mem.read_u32(phys_addr)
    new_val = swap_val & MASK32
    mem.write_u32(phys_addr, new_val)

    ret_val = old_val if glc else (swap_val & MASK32)
    return ret_val, True


class TestBufferAtomicSwapOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def setUp(self) -> None:
        self.mem = BufferMemory(size=4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=1024,
        )

    def test_basic_atomic_swap_in_bounds(self) -> None:
        self.mem.write_u32(64, 0x12345678)
        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=64,
            swap_val=0x9ABCDEF0,
            inst_offset=0,
            glc=False,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret_val, 0x9ABCDEF0)
        self.assertEqual(self.mem.read_u32(64), 0x9ABCDEF0)

    def test_atomic_swap_glc_returns_pre_op_value(self) -> None:
        self.mem.write_u32(128, 0xCAFEBABE)
        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=128,
            swap_val=0xDEADBEEF,
            inst_offset=0,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret_val, 0xCAFEBABE)
        self.assertEqual(self.mem.read_u32(128), 0xDEADBEEF)

    def test_out_of_bounds_rejection(self) -> None:
        self.mem.write_u32(1020, 0x11111111)
        # 1024 + 4 > 1024 (NUM_RECORDS) -> OOB
        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=1024,
            swap_val=0x99999999,
            glc=True,
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret_val, 0)
        # Memory must remain untouched
        self.assertEqual(self.mem.read_u32(1020), 0x11111111)

    def test_swap_identical_value(self) -> None:
        self.mem.write_u32(32, 0x55555555)
        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            self.desc,
            soffset=0,
            vaddr=32,
            swap_val=0x55555555,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret_val, 0x55555555)
        self.assertEqual(self.mem.read_u32(32), 0x55555555)

    def test_inst_offset_and_soffset_combination(self) -> None:
        self.mem.write_u32(256 + 16 + 8, 0x42)
        ret_val, in_bounds = ref_buffer_atomic_swap(
            self.mem,
            self.desc,
            soffset=256,
            vaddr=16,
            swap_val=0x1337,
            inst_offset=8,
            glc=True,
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret_val, 0x42)
        self.assertEqual(self.mem.read_u32(256 + 16 + 8), 0x1337)


if __name__ == "__main__":
    unittest.main()
