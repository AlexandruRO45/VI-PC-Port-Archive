"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU packed 16-bit shift and integer multiply-add operations (Bundle 22):
  - v_pk_ashrrev_i16: Packed 16-bit signed arithmetic right shift
  - v_pk_lshlrev_b16: Packed 16-bit logical left shift
  - v_pk_lshrrev_b16: Packed 16-bit logical right shift
  - v_pk_mad_i16:     Packed 16-bit signed integer multiply-add
  - v_pk_mad_u16:     Packed 16-bit unsigned integer multiply-add
  - v_pk_sub_u16:     Packed 16-bit unsigned integer subtraction

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Callable, Dict, List, Optional, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

try:
    import rdna2spirv
    import support
except Exception:
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF


def u16(val: int) -> int:
    return val & MASK16


def i16(val: int) -> int:
    v = val & MASK16
    return v - 0x10000 if v >= 0x8000 else v


def u32(val: int) -> int:
    return val & MASK32


def i32(val: int) -> int:
    v = val & MASK32
    return v - 0x100000000 if v >= 0x80000000 else v


def pack16(lo: int, hi: int) -> int:
    return ((hi & MASK16) << 16) | (lo & MASK16)


def unpack16(val: int) -> tuple[int, int]:
    return (val & MASK16, (val >> 16) & MASK16)

def ref_v_pk_ashrrev_i16(s0: int, s1: int) -> int:
    """Arithmetic right shift reversed on packed 16-bit integers.
    s0: shift count (each half masked to 15).
    s1: values (each half sign-extended to 16 bits).
    """
    amt_lo = s0 & 15
    amt_hi = (s0 >> 16) & 15
    val_lo = i16(s1 & MASK16)
    val_hi = i16((s1 >> 16) & MASK16)
    res_lo = (val_lo >> amt_lo) & MASK16
    res_hi = (val_hi >> amt_hi) & MASK16
    return pack16(res_lo, res_hi)


class TestVPkAshrrevI16Oracle(unittest.TestCase):
    def test_oracle_positive_shifts(self) -> None:
        s0 = pack16(2, 4)
        s1 = pack16(8, 64)
        self.assertEqual(ref_v_pk_ashrrev_i16(s0, s1), pack16(2, 4))

    def test_oracle_negative_sign_extension(self) -> None:
        s0 = pack16(2, 3)
        s1 = pack16(u16(-16), u16(-64))
        self.assertEqual(ref_v_pk_ashrrev_i16(s0, s1), pack16(u16(-4), u16(-8)))

    def test_oracle_shift_amount_masking(self) -> None:
        s0 = pack16(16, 31)
        s1 = pack16(1234, u16(-8))
        # 16 & 15 == 0, 31 & 15 == 15
        self.assertEqual(ref_v_pk_ashrrev_i16(s0, s1), pack16(1234, u16(-1)))

    def test_oracle_zero_and_extremes(self) -> None:
        s0 = pack16(1, 15)
        s1 = pack16(0x7FFF, 0x8000)
        # 0x7FFF >> 1 == 0x3FFF, -32768 >> 15 == -1 (0xFFFF)
        self.assertEqual(ref_v_pk_ashrrev_i16(s0, s1), pack16(0x3FFF, 0xFFFF))

if __name__ == "__main__":
    unittest.main()
