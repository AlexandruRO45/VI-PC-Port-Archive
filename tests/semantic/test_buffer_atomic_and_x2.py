from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64 = 1 << 63


# ---------------------------------------------------------------------------
# Reference Mathematical Oracle and Architectural Model
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BufferResourceDescriptor:
    """RDNA2 128-bit Buffer Resource Descriptor (V# / SQ_BUF_RSRC_WORD0..3)."""
    base_address: int
    stride: int
    num_records: int
    format_flags: int = 0

    @classmethod
    def from_dwords(cls, d0: int, d1: int, d2: int, d3: int) -> BufferResourceDescriptor:
        base = (d0 & MASK32) | ((d1 & 0xFFFF) << 32)
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


def ref_buffer_atomic_and_x2(
    mem: BufferMemory,
    desc: BufferResourceDescriptor,
    soffset: int,
    vaddr: int,
    and_val: int,
    inst_offset: int = 0,
    glc: bool = False,
    offen: bool = True,
) -> tuple[int, bool]:
    """Pure reference oracle for RDNA2 buffer_atomic_and_x2.

    Computes 64-bit untyped buffer atomic bitwise AND:
      mem[addr] = mem[addr] & and_val

    Returns:
      (return_val, was_in_bounds)
      where return_val is the pre-operation memory value if glc=True, else 0.
    """
    effective_offset = (vaddr if offen else 0) + soffset + inst_offset
    if effective_offset < 0 or effective_offset + 8 > desc.num_records:
        return 0, False

    phys_addr = desc.base_address + effective_offset
    prev_val = mem.read_u64(phys_addr)
    new_val = (prev_val & and_val) & MASK64
    mem.write_u64(phys_addr, new_val)

    ret_val = prev_val if glc else 0
    return ret_val, True


class TestBufferAtomicAndX2Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and memory models."""

    def setUp(self) -> None:
        self.mem = BufferMemory(4096)
        self.desc = BufferResourceDescriptor(
            base_address=0,
            stride=0,
            num_records=4096,
        )

    def test_oracle_basic_and_with_glc(self) -> None:
        self.mem.write_u64(64, 0xFF00FF00FF00FF00)
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, self.desc, soffset=0, vaddr=64, and_val=0x0F0F0F0F0F0F0F0F, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xFF00FF00FF00FF00)
        self.assertEqual(self.mem.read_u64(64), 0x0F000F000F000F00)

    def test_oracle_glc_false_returns_zero(self) -> None:
        self.mem.write_u64(64, 0x1234567890ABCDEF)
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, self.desc, soffset=0, vaddr=64, and_val=0x00000000FFFFFFFF, glc=False
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u64(64), 0x0000000090ABCDEF)

    def test_oracle_out_of_bounds_high(self) -> None:
        desc = BufferResourceDescriptor(base_address=0, stride=0, num_records=100)
        self.mem.write_u64(96, 0xAAAAAAAAAAAAAAAA)
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, desc, soffset=0, vaddr=96, and_val=0x5555555555555555, glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)
        self.assertEqual(self.mem.read_u64(96), 0xAAAAAAAAAAAAAAAA)

    def test_oracle_out_of_bounds_negative(self) -> None:
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, self.desc, soffset=-8, vaddr=0, and_val=0xFF, glc=True
        )
        self.assertFalse(in_bounds)
        self.assertEqual(ret, 0)

    def test_oracle_descriptor_dwords_roundtrip(self) -> None:
        desc = BufferResourceDescriptor(
            base_address=0x123456789000,
            stride=64,
            num_records=16384,
            format_flags=0x80000000,
        )
        d0, d1, d2, d3 = desc.to_dwords()
        restored = BufferResourceDescriptor.from_dwords(d0, d1, d2, d3)
        self.assertEqual(desc.base_address, restored.base_address)
        self.assertEqual(desc.stride, restored.stride)
        self.assertEqual(desc.num_records, restored.num_records)
        self.assertEqual(desc.format_flags, restored.format_flags)

    def test_oracle_all_ones_identity_and_zero_annihilation(self) -> None:
        self.mem.write_u64(128, 0xDEADBEEFCAFEBABE)
        # Bitwise AND with all 1s preserves value
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, self.desc, soffset=0, vaddr=128, and_val=MASK64, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xDEADBEEFCAFEBABE)
        self.assertEqual(self.mem.read_u64(128), 0xDEADBEEFCAFEBABE)

        # Bitwise AND with 0 clears to 0
        ret, in_bounds = ref_buffer_atomic_and_x2(
            self.mem, self.desc, soffset=0, vaddr=128, and_val=0, glc=True
        )
        self.assertTrue(in_bounds)
        self.assertEqual(ret, 0xDEADBEEFCAFEBABE)
        self.assertEqual(self.mem.read_u64(128), 0)


if __name__ == "__main__":
    unittest.main()
