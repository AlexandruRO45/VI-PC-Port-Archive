"""Semantic interpreter tests and reference mathematical oracle for ds_min_u32.

AMD RDNA2 / GFX10 LDS (Local Data Share) unsigned 32-bit atomic minimum.
Specification & Hardware Semantics:
  * Non-returning form: `ds_min_u32 vaddr, vdata0 [offset:offset16]`
      Computes effective byte address: eff_addr = (vaddr + offset16) & 0xFFFFFFFF
      Dword index: idx = (eff_addr >> 2) & (lds_dwords - 1)
      Atomic operation: LDS[idx] = min_u32(LDS[idx], vdata0)
      Unsigned 32-bit comparison: 0x00000000 < 0x80000000 < 0xFFFFFFFF
      Non-returning: prior value in LDS[idx] is discarded, destination register is untouched.
  * Returning form: `ds_min_rtn_u32 vdst, vaddr, vdata0 [offset:offset16]`
      Computes effective byte address and updates LDS[idx] identically.
      Returns prior 32-bit value into vdst.
  * In SPIR-V, translated via OpAtomicUMin on Workgroup memory array.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import random
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import support
    import lds_run as L
    TRANSLATOR_AVAILABLE = True
except (ImportError, TypeError):
    support = None
    L = None
    TRANSLATOR_AVAILABLE = False

MASK32 = 0xFFFFFFFF
LDS_SIZE_BYTES_DEFAULT = 4096
LDS_DWORDS_DEFAULT = 1024


def ref_min_u32(a: int, b: int) -> int:
    """Reference unsigned 32-bit minimum oracle."""
    ua = a & MASK32
    ub = b & MASK32
    return ua if ua <= ub else ub


class LDSModel32:
    """Architectural memory model of RDNA2 Local Data Share (32-bit dwords)."""

    def __init__(self, size_bytes: int = LDS_SIZE_BYTES_DEFAULT):
        assert size_bytes > 0 and (size_bytes & (size_bytes - 1)) == 0, "LDS size must be power of two"
        self.size_bytes = size_bytes
        self.lds_dwords = size_bytes >> 2
        self.memory: dict[int, int] = {}

    def read_dword(self, byte_addr: int) -> int:
        idx = ((byte_addr & MASK32) >> 2) & (self.lds_dwords - 1)
        return self.memory.get(idx, 0)

    def write_dword(self, byte_addr: int, val: int) -> None:
        idx = ((byte_addr & MASK32) >> 2) & (self.lds_dwords - 1)
        self.memory[idx] = val & MASK32

    def ds_min_u32(self, vaddr: int, vdata0: int, offset: int = 0) -> int:
        eff_addr = (vaddr + offset) & MASK32
        idx = (eff_addr >> 2) & (self.lds_dwords - 1)
        old_val = self.memory.get(idx, 0)
        self.memory[idx] = ref_min_u32(old_val, vdata0)
        return old_val


class TestDsMinU32ReferenceOracle(unittest.TestCase):
    """Test suite verifying the reference mathematical oracle and architectural LDS model."""

    def test_ref_min_u32_truth_table(self):
        cases = [
            (0, 0, 0),
            (0, 1, 0),
            (1, 0, 0),
            (5, 10, 5),
            (10, 5, 5),
            (0x7FFFFFFF, 0x80000000, 0x7FFFFFFF),  # Unsigned: 0x7FFFFFFF < 0x80000000
            (0x80000000, 0x7FFFFFFF, 0x7FFFFFFF),
            (0, 0xFFFFFFFF, 0),
            (0xFFFFFFFF, 0, 0),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x80000000, 0x80000001, 0x80000000),
            (0x80000001, 0x80000000, 0x80000000),
        ]
        for a, b, expected in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ref_min_u32(a, b), expected)

    def test_lds_model_basic_atomic_min(self):
        lds = LDSModel32(4096)
        lds.write_dword(0x100, 100)
        old = lds.ds_min_u32(0x100, 42)
        self.assertEqual(old, 100)
        self.assertEqual(lds.read_dword(0x100), 42)

    def test_lds_model_offset_addition_and_alignment(self):
        lds = LDSModel32(4096)
        lds.write_dword(0x44, 500)
        # vaddr 0x40 + offset 4 = 0x44
        old = lds.ds_min_u32(0x40, 250, offset=4)
        self.assertEqual(old, 500)
        self.assertEqual(lds.read_dword(0x44), 250)
        # Unaligned byte address: low 2 bits are dropped (dword aligned)
        lds.ds_min_u32(0x45, 100, offset=1)  # eff_addr = 0x46 -> dword idx 0x44 >> 2
        self.assertEqual(lds.read_dword(0x44), 100)

    def test_lds_model_address_wrapping(self):
        lds = LDSModel32(4096)  # 1024 dwords -> 0x1000 bytes
        lds.write_dword(0x0, 999)
        # 0x1000 wraps to 0x0
        lds.ds_min_u32(0x1000, 50)
        self.assertEqual(lds.read_dword(0x0), 50)
