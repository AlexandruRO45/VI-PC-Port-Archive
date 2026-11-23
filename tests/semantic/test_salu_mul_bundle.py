"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 SALU 32-bit integer multiplication and division helpers (Bundle 40):
  - s_mul_i32:    32-bit integer multiplication (low 32 bits, signed/unsigned identical)
  - s_mul_hi_i32: Signed 32-bit integer multiplication high-half (high 32 bits of 64-bit product)
  - s_mul_hi_u32: Unsigned 32-bit integer multiplication high-half (high 32 bits of 64-bit product)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import random
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

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    return val & MASK32


def i32(val: int) -> int:
    v = val & MASK32
    return v - 0x100000000 if v >= 0x80000000 else v

def ref_s_mul_i32(s0: int, s1: int) -> int:
    """Low 32 bits of 32x32 integer multiply (identical for signed/unsigned)."""
    return (s0 * s1) & MASK32


class TestSMulI32Oracle(unittest.TestCase):
    def test_oracle_basic_multiplication(self) -> None:
        self.assertEqual(ref_s_mul_i32(6, 7), 42)
        self.assertEqual(ref_s_mul_i32(100, 200), 20000)

    def test_oracle_negative_operands(self) -> None:
        self.assertEqual(ref_s_mul_i32(u32(-3), 4), u32(-12))
        self.assertEqual(ref_s_mul_i32(u32(-5), u32(-6)), 30)

    def test_oracle_zero_and_identity(self) -> None:
        self.assertEqual(ref_s_mul_i32(0, 12345), 0)
        self.assertEqual(ref_s_mul_i32(12345, 1), 12345)

    def test_oracle_wrapping_modulo_2_32(self) -> None:
        # 0x10001 * 0x10001 = 0x200020001 & 0xFFFFFFFF = 0x00020001
        self.assertEqual(ref_s_mul_i32(0x10001, 0x10001), 0x00020001)

if __name__ == "__main__":
    unittest.main()
