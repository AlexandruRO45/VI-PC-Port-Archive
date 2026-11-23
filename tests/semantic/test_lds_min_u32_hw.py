from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv  # noqa: E402
    import support  # noqa: E402
    import lds_run  # noqa: E402
except Exception:  # noqa: E722
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore
    lds_run = None  # type: ignore

MASK32: int = 0xFFFFFFFF
SIGN_BIT32: int = 0x80000000
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB capacity in 32-bit dwords


# ---------------------------------------------------------------------------
# Reference Mathematical Oracle and Architectural Model
# ---------------------------------------------------------------------------

def ref_ds_min_u32(curr_val: int, data_val: int) -> int:
    """Pure mathematical reference oracle for 32-bit unsigned atomic minimum:

    Computes:
      LDS[addr] = min(LDS[addr], data_val) (unsigned 32-bit comparison)

    Properties:
      - Identity: min(x, 0xFFFFFFFF) == x
      - Minimum zero: min(x, 0) == 0
      - Idempotence: min(x, x) == x
      - Commutativity: min(x, y) == min(y, x)
      - Monotonicity: min(x, y) <= x (memory contents never increase)
    """
    c = curr_val & MASK32
    d = data_val & MASK32
    return min(c, d)


def calc_lds_effective_address(vaddr: int, offset16: int = 0) -> int:
    """Calculates 32-bit effective byte address with 16-bit unsigned offset."""
    return (vaddr + (offset16 & 0xFFFF)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = DEFAULT_LDS_DWORDS) -> int:
    """Calculates aligned dword index, dropping unaligned low 2 bits and wrapping modulo capacity."""
    return (eff_addr >> 2) & (capacity_dwords - 1)


class LDSMemory:
    """Architectural model of Local Data Share (LDS) scratchpad memory."""

    def __init__(self, size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a power of two, got {size_dwords}")
        self.size_dwords: int = size_dwords
        self.data: list[int] = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        return self.data[idx] & MASK32

    def write_dword(self, eff_addr: int, val: int) -> None:
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        self.data[idx] = val & MASK32

    def atomic_min_u32(self, vaddr: int, data_val: int, offset16: int = 0) -> int:
        """Executes non-returning 32-bit atomic unsigned min; returns updated memory dword."""
        eff_addr = calc_lds_effective_address(vaddr, offset16)
        curr = self.read_dword(eff_addr)
        updated = ref_ds_min_u32(curr, data_val)
        self.write_dword(eff_addr, updated)
        return updated


class TestLdsMinU32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural LDS models."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=1024)

    def test_oracle_basic_min(self) -> None:
        self.lds.write_dword(0x40, 100)
        updated = self.lds.atomic_min_u32(0x40, 50)
        self.assertEqual(updated, 50)
        self.assertEqual(self.lds.read_dword(0x40), 50)

        # Larger operand leaves memory unchanged
        updated_larger = self.lds.atomic_min_u32(0x40, 75)
        self.assertEqual(updated_larger, 50)
        self.assertEqual(self.lds.read_dword(0x40), 50)

    def test_oracle_identity_all_ones(self) -> None:
        cases = [0x0, 0x1, 0x12345678, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_ds_min_u32(val, MASK32), val)

    def test_oracle_zero_minimum(self) -> None:
        cases = [0x0, 0x1, 0x55555555, 0xAAAAAAAA, 0x80000000, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_ds_min_u32(val, 0), 0)

    def test_oracle_idempotence(self) -> None:
        cases = [0x0, 0x10203040, 0x55AA55AA, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_ds_min_u32(val, val), val)

    def test_oracle_unsigned_ordering(self) -> None:
        # In unsigned 32-bit: 0x7FFFFFFF < 0x80000000
        self.assertEqual(ref_ds_min_u32(0x7FFFFFFF, 0x80000000), 0x7FFFFFFF)
        # 0 < 0xFFFFFFFF
        self.assertEqual(ref_ds_min_u32(0xFFFFFFFF, 0), 0)

    def test_oracle_address_calculation_with_offset(self) -> None:
        self.lds.write_dword(0x44, 80)
        # vaddr=0x40, offset=4 -> eff_addr=0x44
        updated = self.lds.atomic_min_u32(0x40, 20, offset16=4)
        self.assertEqual(updated, 20)
        self.assertEqual(self.lds.read_dword(0x44), 20)

    def test_oracle_address_dword_alignment(self) -> None:
        self.lds.write_dword(0x40, 100)
        self.lds.atomic_min_u32(0x41, 80)
        self.lds.atomic_min_u32(0x42, 60)
        self.lds.atomic_min_u32(0x43, 70)
        self.assertEqual(self.lds.read_dword(0x40), 60)

    def test_oracle_capacity_wrapping(self) -> None:
        self.lds.write_dword(0, 200)
        self.lds.atomic_min_u32(4096, 100)
        self.assertEqual(self.lds.read_dword(0), 100)


if __name__ == "__main__":
    unittest.main()
