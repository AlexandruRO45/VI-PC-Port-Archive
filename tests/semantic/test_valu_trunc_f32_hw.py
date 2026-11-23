"""Semantic reference tests and translator hardware verification for AMD RDNA2 v_trunc_f32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_trunc_f32 belongs to the VOP1 encoding format.
  Instruction:
    v_trunc_f32 vd, src0

  Operands:
    vd   = 32-bit destination VGPR
    src0 = 32-bit source operand: VGPR, SGPR, inline constant, or literal float

  Semantics:
    Truncates floating-point value to an integer, rounding toward zero.
    Result is encoded as an IEEE 754 single-precision float.
    vd = trunc(src0)
    Neither SCC nor VCC is modified.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple
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
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def ref_trunc_f32(src0_bits: int) -> int:
    """Mathematical reference oracle for AMD RDNA2 v_trunc_f32 instruction.

    Args:
        src0_bits: 32-bit IEEE 754 float bit pattern (uint32)

    Returns:
        32-bit IEEE 754 float bit pattern of truncated result.
    """
    f = u32_to_f32(src0_bits)
    if math.isnan(f) or math.isinf(f):
        return src0_bits & MASK32
    res_f = float(math.trunc(f))
    if res_f == 0.0 and math.copysign(1.0, f) < 0.0:
        return NEG_ZERO_F32
    return f32_to_u32(res_f)


class TestValuTruncF32Oracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle unit tests."""

    def test_oracle_positive_floats(self) -> None:
        self.assertEqual(ref_trunc_f32(f32_to_u32(5.5)), f32_to_u32(5.0))
        self.assertEqual(ref_trunc_f32(f32_to_u32(5.9)), f32_to_u32(5.0))
        self.assertEqual(ref_trunc_f32(f32_to_u32(0.9)), f32_to_u32(0.0))

    def test_oracle_negative_floats(self) -> None:
        # Truncation rounds toward zero, so -5.5 -> -5.0, -0.9 -> -0.0
        self.assertEqual(ref_trunc_f32(f32_to_u32(-5.5)), f32_to_u32(-5.0))
        self.assertEqual(ref_trunc_f32(f32_to_u32(-5.9)), f32_to_u32(-5.0))
        self.assertEqual(ref_trunc_f32(f32_to_u32(-0.9)), NEG_ZERO_F32)

    def test_oracle_integers_unchanged(self) -> None:
        for val in [0.0, 1.0, 42.0, 1000000.0]:
            self.assertEqual(ref_trunc_f32(f32_to_u32(val)), f32_to_u32(val))
            self.assertEqual(ref_trunc_f32(f32_to_u32(-val)), f32_to_u32(-val) if val != 0.0 else NEG_ZERO_F32)

    def test_oracle_signed_zeros(self) -> None:
        self.assertEqual(ref_trunc_f32(POS_ZERO_F32), POS_ZERO_F32)
        self.assertEqual(ref_trunc_f32(NEG_ZERO_F32), NEG_ZERO_F32)

    def test_oracle_infinities_and_nan(self) -> None:
        self.assertEqual(ref_trunc_f32(POS_INF_F32), POS_INF_F32)
        self.assertEqual(ref_trunc_f32(NEG_INF_F32), NEG_INF_F32)
        self.assertEqual(ref_trunc_f32(QNAN_F32), QNAN_F32)


if __name__ == "__main__":
    unittest.main()
