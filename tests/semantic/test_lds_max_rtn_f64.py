from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 ds_max_rtn_f64.

Models IEEE 754-2008 / IEEE 754-2019 binary64 double-precision maxNum and AMD RDNA2 / GFX10
Local Data Share (LDS) architectural semantics:
  - ds_max_rtn_f64: Atomic floating-point 64-bit maximum on Local Data Share (LDS),
    returning the previous 64-bit memory content into destination VGPR pair.

Syntax:
    ds_max_rtn_f64 vdst, vaddr, vdata0 [offset:offset16] [gds]

Mathematical Specification:
    effective_byte_address = (vaddr + offset16) & 0xFFFFFFFF
    qword_index = (effective_byte_address >> 3) & (lds_qwords - 1)
    old_val = LDS64[qword_index]
    new_mem = oracle_max_f64(old_val, vdata0)
    LDS64[qword_index] = new_mem
    vdst = old_val

Key Architectural Characteristics:
  1. IEEE 754-2008 / IEEE 754-2019 maxNum Semantics (binary64):
     - NaN Suppression: If one operand is NaN and the other is a valid float, the valid float
       is returned. If both operands are NaN, a quiet NaN is returned.
     - Signed Zero Ordering: -0.0 is strictly less than +0.0:
       max(+0.0, -0.0) = +0.0, max(-0.0, +0.0) = +0.0.
     - Infinities & Subnormals: -INF < all finite numbers < +INF.
       Subnormal double-precision values are preserved without flushing.
  2. Return Value Semantics (RTN):
     Atomically fetches the pre-operation 64-bit value from LDS64[qword_index] and writes it
     into the destination VGPR pair (vdst:vdst+1).
  3. Addressing and Alignment:
     Local Data Share is byte-addressed but qword-aligned for 64-bit operations (eff_addr >> 3).
     The bottom 3 bits of the effective address are dropped by hardware.
     Immediate offset is an unsigned 16-bit integer (0 <= offset <= 65535).
  4. LDS Capacity and Banking:
     Standard allocation is 64 KiB (8,192 64-bit qwords) across 32 memory banks.
     Bank index = (effective_byte_address >> 2) % 32.

Standard library Python only; zero external dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import dataclasses
import enum
import math
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

# IEEE 754 binary64 architectural bitmasks
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
OFFSET16_MASK: int = 0xFFFF
SIGN_MASK64: int = 0x8000000000000000
EXP_MASK64: int = 0x7FF0000000000000
FRAC_MASK64: int = 0x000FFFFFFFFFFFFF
QNAN_BIT64: int = 0x0008000000000000
CANONICAL_QNAN64: int = 0x7FF8000000000000

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO64: int = 0x0000000000000000
NEG_ZERO64: int = 0x8000000000000000
ONE_F64: int = 0x3FF0000000000000           # 1.0
NEG_ONE_F64: int = 0xBFF0000000000000       # -1.0
HALF_F64: int = 0x3FE0000000000000          # 0.5
NEG_HALF_F64: int = 0xBFE0000000000000      # -0.5
TWO_F64: int = 0x4000000000000000           # 2.0
NEG_TWO_F64: int = 0xC000000000000000       # -2.0
THREE_F64: int = 0x4008000000000000         # 3.0
TEN_F64: int = 0x4024000000000000           # 10.0
NEG_TEN_F64: int = 0xC024000000000000       # -10.0

DBL_MIN: int = 0x0010000000000000           # Smallest positive normal: 2^-1022
NEG_DBL_MIN: int = 0x8010000000000000       # -2^-1022
DBL_MAX: int = 0x7FEFFFFFFFFFFFFF           # Largest finite normal double
NEG_DBL_MAX: int = 0xFFEFFFFFFFFFFFFF
POS_INF64: int = 0x7FF0000000000000         # +infinity
NEG_INF64: int = 0xFFF0000000000000         # -infinity
QNAN_PAYLOAD_A64: int = 0x7FF8000000000001  # Quiet NaN with payload A
QNAN_PAYLOAD_B64: int = 0xFFF8000000000002  # Quiet NaN with payload B
SNAN_PAYLOAD_A64: int = 0x7FF0000000000001  # Signaling NaN
SNAN_PAYLOAD_B64: int = 0xFFF0000000000002  # Signaling NaN
SUBNORM_MIN64: int = 0x0000000000000001     # Smallest positive subnormal: 2^-1074
SUBNORM_MAX64: int = 0x000FFFFFFFFFFFFF     # Largest positive subnormal
NEG_SUBNORM_MIN64: int = 0x8000000000000001
NEG_SUBNORM_MAX64: int = 0x800FFFFFFFFFFFFF

DEFAULT_LDS_SIZE_BYTES: int = 65536  # 64 KiB default capacity
DEFAULT_LDS_NUM_QWORDS: int = 8192   # 8,192 64-bit qwords
NUM_LDS_BANKS: int = 32              # 32 memory banks
DEFAULT_NUM_VGPRS: int = 256         # Standard GFX10 vector register file size
DEFAULT_NUM_SGPRS: int = 106         # Standard GFX10 scalar register file size


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword [0, 2^64 - 1]."""
    return val & MASK64


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_MASK64) else val


def f64_to_bits(val: float) -> int:
    """Encode float to IEEE 754 double-precision 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF64 if val > 0 else NEG_INF64


def bits_to_f64(bits: int) -> float:
    """Decode 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def split_f64_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack 64-bit integer into (sign, exponent, fraction) components."""
    bits &= MASK64
    sign = (bits & SIGN_MASK64) >> 63
    exponent = (bits & EXP_MASK64) >> 52
    fraction = bits & FRAC_MASK64
    return sign, exponent, fraction


def is_nan64(bits: int) -> bool:
    """Return True if 64-bit integer represents an IEEE 754 double NaN."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == EXP_MASK64) and ((bits & FRAC_MASK64) != 0)


def is_qnan64(bits: int) -> bool:
    """Return True if 64-bit integer represents an IEEE 754 double Quiet NaN."""
    return is_nan64(bits) and bool(bits & QNAN_BIT64)


def is_snan64(bits: int) -> bool:
    """Return True if 64-bit integer represents an IEEE 754 double Signaling NaN."""
    return is_nan64(bits) and not bool(bits & QNAN_BIT64)


def is_inf64(bits: int) -> bool:
    """Return True if 64-bit integer represents IEEE 754 double positive or negative infinity."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == EXP_MASK64) and ((bits & FRAC_MASK64) == 0)


def is_subnormal64(bits: int) -> bool:
    """Return True if 64-bit integer represents a subnormal double-precision float."""
    bits &= MASK64
    return ((bits & EXP_MASK64) == 0) and ((bits & FRAC_MASK64) != 0)


def is_zero64(bits: int) -> bool:
    """Return True if 64-bit integer represents positive or negative zero."""
    return (bits & ~SIGN_MASK64) == 0


def oracle_max_f64(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_max_f64 atomic operation.

    Follows IEEE 754-2008/2019 maxNum for binary64:
      - If one operand is NaN and the other is a valid float, returns the valid float.
      - If both are NaN, returns a quiet NaN (preserving payload or quietened).
      - Signed zero ordering: -0.0 < +0.0 (max(+0.0, -0.0) -> +0.0).
      - Subnormals and infinities preserved.
    """
    a = a_bits & MASK64
    b = b_bits & MASK64

    a_is_nan = is_nan64(a)
    b_is_nan = is_nan64(b)

    if a_is_nan and b_is_nan:
        return a | QNAN_BIT64
    if a_is_nan:
        return b
    if b_is_nan:
        return a

    # Signed zeros
    if is_zero64(a) and is_zero64(b):
        return POS_ZERO64 if ((a == POS_ZERO64) or (b == POS_ZERO64)) else NEG_ZERO64

    fa = bits_to_f64(a)
    fb = bits_to_f64(b)

    if fa > fb:
        return a
    elif fb > fa:
        return b
    else:
        # Numerically equal: positive sign bit is strictly larger (+0.0 > -0.0)
        if (a & SIGN_MASK64) == 0:
            return a
        if (b & SIGN_MASK64) == 0:
            return b
        return a


def ref_ds_max_rtn_f64(current_mem: int, vdata0: int) -> Tuple[int, int]:
    """Pure mathematical reference oracle for ds_max_rtn_f64.

    Computes:
        old_val = current_mem & 0xFFFFFFFFFFFFFFFF
        new_mem = oracle_max_f64(old_val, vdata0)

    Returns:
        (old_val, new_mem)
    """
    old_val = u64(current_mem)
    new_mem = oracle_max_f64(old_val, vdata0)
    return old_val, new_mem


def calc_lds_effective_address(vaddr: int, offset: int = 0) -> int:
    """Calculate the 32-bit effective byte address into LDS."""
    return (vaddr + (offset & OFFSET16_MASK)) & MASK32


def calc_lds_qword_index(eff_addr: int, capacity_qwords: int = DEFAULT_LDS_NUM_QWORDS) -> int:
    """Convert an effective byte address to an aligned LDS 64-bit qword index."""
    return (eff_addr >> 3) & (capacity_qwords - 1)


def calc_lds_bank(eff_addr: int) -> int:
    """Determine the LDS memory bank index (0 to 31) for an effective address."""
    return (eff_addr >> 2) % NUM_LDS_BANKS


class LDSMemory64:
    """Architectural memory model simulating AMD RDNA2 Local Data Share (LDS) for 64-bit operations."""

    def __init__(self, size_bytes: int = DEFAULT_LDS_SIZE_BYTES) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"size_bytes must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_qwords = size_bytes // 8
        self.qwords = [0] * self.num_qwords

    def read_qword(self, eff_addr: int) -> int:
        """Read a 64-bit qword from LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.num_qwords)
        return self.qwords[idx]

    def write_qword(self, eff_addr: int, val: int) -> None:
        """Write a 64-bit qword to LDS given an effective byte address."""
        idx = calc_lds_qword_index(eff_addr, self.num_qwords)
        self.qwords[idx] = u64(val)

    def atomic_max_rtn_f64(
        self,
        vaddr: int,
        vdata0: int,
        offset: int = 0,
    ) -> Tuple[int, int]:
        """Execute atomic 64-bit double maximum with return on LDS.

        Returns:
            (old_val, new_mem)
        """
        eff_addr = calc_lds_effective_address(vaddr, offset)
        idx = calc_lds_qword_index(eff_addr, self.num_qwords)
        curr = self.qwords[idx]
        old_val, new_mem = ref_ds_max_rtn_f64(curr, vdata0)
        self.qwords[idx] = new_mem
        return old_val, new_mem


class TestLDSMemoryAndOracle64(unittest.TestCase):
    """Unit tests for reference mathematical oracle and architectural LDS 64-bit memory model."""

    def test_oracle_standard_doubles(self) -> None:
        # max(1.0, 2.0) = 2.0
        self.assertEqual(oracle_max_f64(ONE_F64, TWO_F64), TWO_F64)
        self.assertEqual(oracle_max_f64(TWO_F64, ONE_F64), TWO_F64)

        # max(10.0, -2.0) = 10.0
        self.assertEqual(oracle_max_f64(TEN_F64, NEG_TWO_F64), TEN_F64)
        self.assertEqual(oracle_max_f64(NEG_TWO_F64, TEN_F64), TEN_F64)

        # max(-10.0, -2.0) = -2.0
        self.assertEqual(oracle_max_f64(NEG_TEN_F64, NEG_TWO_F64), NEG_TWO_F64)

    def test_oracle_signed_zeros(self) -> None:
        # max(+0.0, -0.0) = +0.0
        self.assertEqual(oracle_max_f64(POS_ZERO64, NEG_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_ZERO64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(POS_ZERO64, POS_ZERO64), POS_ZERO64)
        self.assertEqual(oracle_max_f64(NEG_ZERO64, NEG_ZERO64), NEG_ZERO64)

    def test_oracle_infinities(self) -> None:
        self.assertEqual(oracle_max_f64(POS_INF64, ONE_F64), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, ONE_F64), ONE_F64)
        self.assertEqual(oracle_max_f64(POS_INF64, NEG_INF64), POS_INF64)
        self.assertEqual(oracle_max_f64(NEG_INF64, NEG_INF64), NEG_INF64)
        self.assertEqual(oracle_max_f64(POS_INF64, POS_INF64), POS_INF64)

    def test_oracle_nan_suppression(self) -> None:
        self.assertEqual(oracle_max_f64(QNAN_PAYLOAD_A64, ONE_F64), ONE_F64)
        self.assertEqual(oracle_max_f64(ONE_F64, QNAN_PAYLOAD_A64), ONE_F64)
        self.assertEqual(oracle_max_f64(SNAN_PAYLOAD_A64, NEG_TEN_F64), NEG_TEN_F64)
        self.assertEqual(oracle_max_f64(NEG_TEN_F64, SNAN_PAYLOAD_A64), NEG_TEN_F64)

        res_nan = oracle_max_f64(QNAN_PAYLOAD_A64, QNAN_PAYLOAD_B64)
        self.assertTrue(is_qnan64(res_nan))

    def test_oracle_subnormals(self) -> None:
        self.assertEqual(oracle_max_f64(SUBNORM_MIN64, DBL_MIN), DBL_MIN)
        self.assertEqual(oracle_max_f64(SUBNORM_MAX64, SUBNORM_MIN64), SUBNORM_MAX64)
        self.assertEqual(oracle_max_f64(NEG_SUBNORM_MAX64, POS_ZERO64), POS_ZERO64)

    def test_ref_ds_max_rtn_f64(self) -> None:
        old_val, new_mem = ref_ds_max_rtn_f64(ONE_F64, TWO_F64)
        self.assertEqual(old_val, ONE_F64)
        self.assertEqual(new_mem, TWO_F64)

        old_val, new_mem = ref_ds_max_rtn_f64(TEN_F64, ONE_F64)
        self.assertEqual(old_val, TEN_F64)
        self.assertEqual(new_mem, TEN_F64)

    def test_lds_memory_initialization(self) -> None:
        mem = LDSMemory64(65536)
        self.assertEqual(mem.size_bytes, 65536)
        self.assertEqual(mem.num_qwords, 8192)
        self.assertEqual(len(mem.qwords), 8192)

        for invalid_size in (0, -64, 1000, 30000, 65537):
            with self.assertRaises(ValueError):
                LDSMemory64(invalid_size)

    def test_lds_memory_read_write(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x00, ONE_F64)
        self.assertEqual(mem.read_qword(0x00), ONE_F64)
        mem.write_qword(0x40, NEG_TEN_F64)
        self.assertEqual(mem.read_qword(0x40), NEG_TEN_F64)

    def test_lds_memory_unaligned_byte_address(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x100, TWO_F64)
        for off in range(8):
            self.assertEqual(mem.read_qword(0x100 + off), TWO_F64)

    def test_lds_atomic_max_rtn_f64(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x20, ONE_F64)

        # max(1.0, 10.0) = 10.0
        old_val, new_mem = mem.atomic_max_rtn_f64(vaddr=0x20, vdata0=TEN_F64)
        self.assertEqual(old_val, ONE_F64)
        self.assertEqual(new_mem, TEN_F64)
        self.assertEqual(mem.read_qword(0x20), TEN_F64)

        # max(10.0, 3.0) = 10.0
        old_val2, new_mem2 = mem.atomic_max_rtn_f64(vaddr=0x20, vdata0=THREE_F64)
        self.assertEqual(old_val2, TEN_F64)
        self.assertEqual(new_mem2, TEN_F64)
        self.assertEqual(mem.read_qword(0x20), TEN_F64)

    def test_lds_atomic_max_rtn_with_offset(self) -> None:
        mem = LDSMemory64(1024)
        mem.write_qword(0x50, NEG_TEN_F64)

        # vaddr = 0x40, offset = 0x10 -> eff_addr = 0x50
        old_val, new_mem = mem.atomic_max_rtn_f64(vaddr=0x40, vdata0=TWO_F64, offset=0x10)
        self.assertEqual(old_val, NEG_TEN_F64)
        self.assertEqual(new_mem, TWO_F64)
        self.assertEqual(mem.read_qword(0x50), TWO_F64)


if __name__ == "__main__":
    unittest.main()
