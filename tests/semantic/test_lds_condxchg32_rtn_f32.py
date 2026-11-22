"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_condxchg32_rtn_f32.

Models single-precision floating-point (IEEE 754 binary32) conditional exchange
and AMD RDNA2 / GFX10 Local Data Share (LDS) architectural semantics:
  - ds_condxchg32_rtn_f32 (also known as ds_cmpst_rtn_f32):
    Atomic conditional exchange on 32-bit LDS storage using IEEE 754 single-precision
    floating-point comparison.

RDNA2 LDS / Floating-Point Conditional Exchange ISA Rules:
  1. Local Data Share Architecture:
     LDS is an on-chip scratchpad memory shared across all threads in a workgroup.
     Addressing is byte-addressed but dword-aligned in hardware (low 2 bits dropped).
     Out-of-bounds addresses wrap within the allocated power-of-two LDS slice.
     LDS is organized into 32 memory banks (bank = dword_index % 32).
  2. Floating-Point Comparison Semantics:
     The memory value and comparison value are interpreted as IEEE 754 binary32 floats.
     - Equality (+0.0 == -0.0): In IEEE 754, positive and negative zero compare equal.
       If memory contains -0.0 and cmp_val is +0.0, the exchange succeeds.
     - Not-a-Number (NaN != NaN): NaNs compare unordered; NaN == NaN evaluates to False.
       If cmp_val or memory is NaN, the exchange never occurs.
     - Infinities: +INF == +INF evaluates to True, -INF == -INF evaluates to True.
     - Subnormals: Compared according to IEEE 754 binary32 float values.
  3. Return Semantics:
     The instruction atomically reads the existing 32-bit dword from LDS and writes
     it into destination VGPR vdst regardless of whether the comparison succeeded.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import struct
import unittest
from typing import Dict, List, Optional, Tuple, Set

MASK32: int = 0xFFFFFFFF
SIGN_MASK: int = 0x80000000
EXP_MASK: int = 0x7F800000
FRAC_MASK: int = 0x007FFFFF
QNAN_BIT: int = 0x00400000

# Canonical IEEE 754 bit representations
POS_ZERO_RAW: int = 0x00000000
NEG_ZERO_RAW: int = 0x80000000
POS_INF_RAW: int = 0x7F800000
NEG_INF_RAW: int = 0xFF800000
QNAN_RAW: int = 0x7FC00000
SNAN_RAW: int = 0x7F800001
ONE_RAW: int = 0x3F800000      # 1.0f
TWO_RAW: int = 0x40000000      # 2.0f
THREE_RAW: int = 0x40400000    # 3.0f

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB
DEFAULT_LDS_NUM_DWORDS: int = 16384
NUM_LDS_BANKS: int = 32


def raw_u32_to_f32(raw: int) -> float:
    """Convert raw 32-bit unsigned integer to IEEE 754 single-precision float."""
    return struct.unpack("<f", struct.pack("<I", raw & MASK32))[0]


def f32_to_raw_u32(val: float) -> int:
    """Convert Python float to 32-bit IEEE 754 single-precision bit pattern."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def is_nan_raw(raw: int) -> bool:
    """Check if 32-bit bit pattern represents an IEEE 754 NaN."""
    return (raw & EXP_MASK) == EXP_MASK and (raw & FRAC_MASK) != 0


def float_equal_f32(a_raw: int, b_raw: int) -> bool:
    """Evaluate IEEE 754 binary32 equality between two raw bit patterns.
    
    Returns True if fa == fb according to IEEE 754 rules.
    Specifically:
      - Any NaN operand yields False (including NaN == NaN).
      - +0.0 == -0.0 yields True.
      - Exact value equality for finite numbers and infinities.
    """
    if is_nan_raw(a_raw) or is_nan_raw(b_raw):
        return False
    fa = raw_u32_to_f32(a_raw)
    fb = raw_u32_to_f32(b_raw)
    return fa == fb


def oracle_condxchg32_f32(
    mem_raw: int,
    cmp_raw: int,
    new_raw: int,
) -> Tuple[int, int, bool]:
    """Reference mathematical oracle for ds_condxchg32_rtn_f32.
    
    Returns:
        (old_val, new_mem_val, exchanged)
    """
    old_val = mem_raw & MASK32
    c_raw = cmp_raw & MASK32
    n_raw = new_raw & MASK32

    if float_equal_f32(old_val, c_raw):
        return old_val, n_raw, True
    return old_val, old_val, False


class LDSMemory:
    """Architectural simulation model of AMD RDNA2 Local Data Share (LDS)."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of 2, got {size_bytes}")
        self.size_bytes: int = size_bytes
        self.num_dwords: int = size_bytes // 4
        self.dwords: List[int] = [0] * self.num_dwords

    def get_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        """Compute effective wrapped dword index, dropping bottom 2 unaligned bits."""
        effective_byte = (byte_addr + offset) & MASK32
        return (effective_byte >> 2) & (self.num_dwords - 1)

    def get_bank(self, dword_idx: int) -> int:
        """Compute memory bank index for the given dword (0 to 31)."""
        return dword_idx % NUM_LDS_BANKS

    def read_dword(self, dword_idx: int) -> int:
        """Read a 32-bit dword from LDS."""
        return self.dwords[dword_idx & (self.num_dwords - 1)]

    def write_dword(self, dword_idx: int, val: int) -> None:
        """Write a 32-bit dword into LDS."""
        self.dwords[dword_idx & (self.num_dwords - 1)] = val & MASK32

    def atomic_condxchg32_f32(
        self,
        byte_addr: int,
        cmp_val: int,
        new_val: int,
        offset: int = 0,
    ) -> Tuple[int, int, bool]:
        """Execute atomic conditional exchange at (byte_addr + offset)."""
        idx = self.get_dword_index(byte_addr, offset)
        old_val = self.dwords[idx]
        old_val, new_mem, exchanged = oracle_condxchg32_f32(old_val, cmp_val, new_val)
        self.dwords[idx] = new_mem
        return old_val, new_mem, exchanged


class TestReferenceOracleAndArchitecturalModel(unittest.TestCase):
    """Unit tests for reference floating-point conditional exchange oracle and LDS model."""

    def test_oracle_finite_matches_and_mismatches(self) -> None:
        # Match: 1.0 == 1.0 -> exchange to 2.0
        old, mem, swapped = oracle_condxchg32_f32(ONE_RAW, ONE_RAW, TWO_RAW)
        self.assertTrue(swapped)
        self.assertEqual(old, ONE_RAW)
        self.assertEqual(mem, TWO_RAW)

        # Mismatch: 1.0 != 2.0 -> no exchange
        old, mem, swapped = oracle_condxchg32_f32(ONE_RAW, TWO_RAW, THREE_RAW)
        self.assertFalse(swapped)
        self.assertEqual(old, ONE_RAW)
        self.assertEqual(mem, ONE_RAW)

    def test_oracle_signed_zero_equality(self) -> None:
        # IEEE 754: +0.0 == -0.0 is True
        old, mem, swapped = oracle_condxchg32_f32(NEG_ZERO_RAW, POS_ZERO_RAW, ONE_RAW)
        self.assertTrue(swapped)
        self.assertEqual(old, NEG_ZERO_RAW)
        self.assertEqual(mem, ONE_RAW)

        old2, mem2, swapped2 = oracle_condxchg32_f32(POS_ZERO_RAW, NEG_ZERO_RAW, TWO_RAW)
        self.assertTrue(swapped2)
        self.assertEqual(old2, POS_ZERO_RAW)
        self.assertEqual(mem2, TWO_RAW)

    def test_oracle_nan_unordered_rejection(self) -> None:
        # NaN == NaN is always False in IEEE 754
        old, mem, swapped = oracle_condxchg32_f32(QNAN_RAW, QNAN_RAW, ONE_RAW)
        self.assertFalse(swapped)
        self.assertEqual(old, QNAN_RAW)
        self.assertEqual(mem, QNAN_RAW)

        # Signaling NaN rejection
        old_s, mem_s, swapped_s = oracle_condxchg32_f32(SNAN_RAW, SNAN_RAW, TWO_RAW)
        self.assertFalse(swapped_s)
        self.assertEqual(old_s, SNAN_RAW)
        self.assertEqual(mem_s, SNAN_RAW)

    def test_oracle_infinities(self) -> None:
        # +inf == +inf
        old, mem, swapped = oracle_condxchg32_f32(POS_INF_RAW, POS_INF_RAW, ONE_RAW)
        self.assertTrue(swapped)
        self.assertEqual(mem, ONE_RAW)

        # +inf != -inf
        old, mem, swapped = oracle_condxchg32_f32(POS_INF_RAW, NEG_INF_RAW, ONE_RAW)
        self.assertFalse(swapped)
        self.assertEqual(mem, POS_INF_RAW)

    def test_lds_dword_indexing_and_alignment(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.get_dword_index(0), 0)
        self.assertEqual(lds.get_dword_index(1), 0)
        self.assertEqual(lds.get_dword_index(2), 0)
        self.assertEqual(lds.get_dword_index(3), 0)
        self.assertEqual(lds.get_dword_index(4), 1)

    def test_lds_offset_and_wrapping(self) -> None:
        lds = LDSMemory(65536)
        self.assertEqual(lds.get_dword_index(0, offset=8), 2)
        self.assertEqual(lds.get_dword_index(65536), 0)

    def test_lds_bank_distribution(self) -> None:
        lds = LDSMemory(65536)
        for i in range(32):
            self.assertEqual(lds.get_bank(i), i)
        self.assertEqual(lds.get_bank(32), 0)

    def test_lds_atomic_condxchg32_f32_execution(self) -> None:
        lds = LDSMemory(65536)
        lds.write_dword(5, ONE_RAW)

        # Match at dword 5
        old, mem, swapped = lds.atomic_condxchg32_f32(byte_addr=20, cmp_val=ONE_RAW, new_val=TWO_RAW)
        self.assertTrue(swapped)
        self.assertEqual(old, ONE_RAW)
        self.assertEqual(lds.read_dword(5), TWO_RAW)

        # Mismatch at dword 5
        old2, mem2, swapped2 = lds.atomic_condxchg32_f32(byte_addr=20, cmp_val=THREE_RAW, new_val=ONE_RAW)
        self.assertFalse(swapped2)
        self.assertEqual(old2, TWO_RAW)
        self.assertEqual(lds.read_dword(5), TWO_RAW)


if __name__ == "__main__":
    unittest.main()
