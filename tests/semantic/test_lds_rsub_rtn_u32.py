from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_rsub_rtn_u32.

Models unsigned 32-bit integer reverse-subtraction and AMD RDNA2 / GFX10 Local Data Share (LDS)
architectural semantics:
  - ds_rsub_rtn_u32: Atomic reverse subtraction of unsigned 32-bit integer in LDS memory,
    returning the previous memory content into destination VGPR.

Syntax:
    ds_rsub_rtn_u32 vdst, vaddr, vdata0 [offset:offset16] [gds]

Mathematical Specification:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
    old_val = LDS[dword_index]
    new_mem = (vdata0 - old_val) & 0xFFFFFFFF
    LDS[dword_index] = new_mem
    vdst = old_val

Key Architectural Characteristics:
  1. Reverse Subtraction (RSUB):
     Computes (vdata0 - old_val) rather than (old_val - vdata0).
     The incoming operand vdata0 serves as the minuend, and the current memory content
     serves as the subtrahend.
  2. Return Value Semantics (RTN):
     Atomically fetches the pre-arithmetic value of LDS[dword_index] and returns it
     into the destination VGPR (vdst).
  3. Addressing and Alignment:
     Local Data Share is byte-addressed but dword-aligned for 32-bit operations.
     The bottom 2 bits of the effective address are dropped (eff_addr >> 2).
     Immediate offset is an unsigned 16-bit integer (0 <= offset <= 65535).
  4. LDS Capacity and Banking:
     Standard allocation is 64 KiB (16,384 dwords) organized into 32 memory banks.
     Bank index = dword_index % 32.

Standard library Python only; zero external dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import unittest
from typing import Tuple

# RDNA2 Local Data Share (LDS) Architectural Constants
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET16_MASK: int = 0xFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536      # 64 KiB default capacity
DEFAULT_LDS_NUM_DWORDS: int = 16384      # 16,384 32-bit dwords
NUM_LDS_BANKS: int = 32                  # 32 memory banks
DEFAULT_NUM_VGPRS: int = 256             # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106             # Standard GFX10 scalar register file size


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS.

    The offset is an unsigned 16-bit immediate. Address wrapping is strictly modulo 2^32.
    """
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = DEFAULT_LDS_NUM_DWORDS) -> int:
    """Convert an effective byte address to an aligned LDS dword index.

    LDS addresses are dword-aligned; the lowest 2 bits are dropped by hardware.
    The index wraps modulo the power-of-two dword capacity.
    """
    return (eff_addr >> 2) & (capacity_dwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for an effective address."""
    return (eff_addr >> 2) % NUM_LDS_BANKS


def ref_ds_rsub_rtn_u32(current_mem: int, vdata0: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_rsub_rtn_u32.

    Computes:
        old_val = current_mem & 0xFFFFFFFF
        new_mem = (vdata0 - old_val) & 0xFFFFFFFF

    Returns:
        (old_val, new_mem)
    """
    old_val = u32(current_mem)
    new_mem = u32(u32(vdata0) - old_val)
    return old_val, new_mem


class LDSMemory:
    """Architectural memory model simulating AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"size_bytes must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.dwords = [0] * self.num_dwords

    def read_dword(self, eff_addr: int) -> int:
        """Read a 32-bit dword from LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        return self.dwords[idx]

    def write_dword(self, eff_addr: int, val: int) -> None:
        """Write a 32-bit dword to LDS given an effective byte address."""
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        self.dwords[idx] = u32(val)

    def atomic_rsub_rtn_u32(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 32-bit integer reverse subtraction with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_dword_index(eff_addr, self.num_dwords)
        curr = self.dwords[idx]
        old_val, new_mem = ref_ds_rsub_rtn_u32(curr, vdata0)
        self.dwords[idx] = new_mem
        return old_val, new_mem


class TestLDSMemoryAndOracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS memory model."""

    def test_oracle_basic_rsub(self) -> None:
        # 50 - 10 = 40
        old_val, new_mem = ref_ds_rsub_rtn_u32(current_mem=10, vdata0=50)
        self.assertEqual(old_val, 10)
        self.assertEqual(new_mem, 40)

        # Reverse order: 10 - 50 = -40 = 0xFFFFFFD8
        old_val, new_mem = ref_ds_rsub_rtn_u32(current_mem=50, vdata0=10)
        self.assertEqual(old_val, 50)
        self.assertEqual(new_mem, u32(-40))
        self.assertEqual(new_mem, 0xFFFFFFD8)

    def test_oracle_identity(self) -> None:
        # X - X = 0
        for x in (0, 1, 42, 0x12345678, 0xFFFFFFFF):
            old_val, new_mem = ref_ds_rsub_rtn_u32(current_mem=x, vdata0=x)
            self.assertEqual(old_val, x & MASK32)
            self.assertEqual(new_mem, 0)

    def test_oracle_zero_cases(self) -> None:
        # 0 - 0 = 0
        self.assertEqual(ref_ds_rsub_rtn_u32(0, 0), (0, 0))
        # vdata0 - 0 = vdata0
        self.assertEqual(ref_ds_rsub_rtn_u32(0, 500), (0, 500))
        # 0 - current_mem = -current_mem
        self.assertEqual(ref_ds_rsub_rtn_u32(500, 0), (500, u32(-500)))

    def test_oracle_wraparound(self) -> None:
        # 0 - 0xFFFFFFFF = 1
        self.assertEqual(ref_ds_rsub_rtn_u32(0xFFFFFFFF, 0), (0xFFFFFFFF, 1))
        # 0xFFFFFFFF - 0 = 0xFFFFFFFF
        self.assertEqual(ref_ds_rsub_rtn_u32(0, 0xFFFFFFFF), (0, 0xFFFFFFFF))
        # 1 - 2 = 0xFFFFFFFF
        self.assertEqual(ref_ds_rsub_rtn_u32(2, 1), (2, 0xFFFFFFFF))

    def test_lds_memory_initialization(self) -> None:
        mem = LDSMemory(65536)
        self.assertEqual(mem.size_bytes, 65536)
        self.assertEqual(mem.num_dwords, 16384)
        self.assertEqual(len(mem.dwords), 16384)

        # Invalid non-power-of-two or non-positive sizes
        for invalid_size in (0, -64, 1000, 30000, 65537):
            with self.assertRaises(ValueError):
                LDSMemory(invalid_size)

    def test_lds_memory_read_write(self) -> None:
        mem = LDSMemory(1024)  # 256 dwords
        mem.write_dword(0x00, 0x12345678)
        self.assertEqual(mem.read_dword(0x00), 0x12345678)

        mem.write_dword(0x40, 0xABCDEF01)
        self.assertEqual(mem.read_dword(0x40), 0xABCDEF01)

    def test_lds_memory_unaligned_byte_address(self) -> None:
        mem = LDSMemory(1024)
        # Low 2 bits dropped: 0x100, 0x101, 0x102, 0x103 all map to dword index 0x100 >> 2
        mem.write_dword(0x100, 0xDEADBEEF)
        self.assertEqual(mem.read_dword(0x101), 0xDEADBEEF)
        self.assertEqual(mem.read_dword(0x102), 0xDEADBEEF)
        self.assertEqual(mem.read_dword(0x103), 0xDEADBEEF)

    def test_lds_memory_address_wrapping(self) -> None:
        mem = LDSMemory(1024)  # 256 dwords (1024 bytes)
        # Byte address 1024 wraps to 0
        mem.write_dword(0, 0xCAFE)
        self.assertEqual(mem.read_dword(1024), 0xCAFE)
        self.assertEqual(mem.read_dword(2048), 0xCAFE)

    def test_lds_memory_bank_calculation(self) -> None:
        # Bank is (eff_addr >> 2) % 32
        self.assertEqual(calc_lds_bank(0), 0)
        self.assertEqual(calc_lds_bank(4), 1)
        self.assertEqual(calc_lds_bank(124), 31)
        self.assertEqual(calc_lds_bank(128), 0)  # Next bank cycle

    def test_lds_atomic_rsub_rtn_u32(self) -> None:
        mem = LDSMemory(1024)
        mem.write_dword(0x20, 100)

        # Execute rsub with vdata0=250: old=100, new = 250 - 100 = 150
        old_val, new_mem = mem.atomic_rsub_rtn_u32(vaddr=0x20, vdata0=250)
        self.assertEqual(old_val, 100)
        self.assertEqual(new_mem, 150)
        self.assertEqual(mem.read_dword(0x20), 150)

        # Subsequent rsub with vdata0=50: old=150, new = 50 - 150 = -100 = 0xFFFFFF9C
        old_val2, new_mem2 = mem.atomic_rsub_rtn_u32(vaddr=0x20, vdata0=50)
        self.assertEqual(old_val2, 150)
        self.assertEqual(new_mem2, u32(-100))
        self.assertEqual(mem.read_dword(0x20), u32(-100))

    def test_lds_atomic_rsub_rtn_with_offset(self) -> None:
        mem = LDSMemory(1024)
        mem.write_dword(0x50, 40)

        # vaddr = 0x40, offset = 0x10 -> eff_addr = 0x50
        old_val, new_mem = mem.atomic_rsub_rtn_u32(vaddr=0x40, vdata0=100, offset=0x10)
        self.assertEqual(old_val, 40)
        self.assertEqual(new_mem, 60)
        self.assertEqual(mem.read_dword(0x50), 60)


if __name__ == "__main__":
    unittest.main()
