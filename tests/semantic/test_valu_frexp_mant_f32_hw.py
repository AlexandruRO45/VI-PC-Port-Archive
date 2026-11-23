"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_frexp_mant_f32.

Models AMD RDNA2 / GFX10 Vector ALU (VALU) v_frexp_mant_f32 normalized mantissa extraction semantics:
  - Extracts the normalized floating-point mantissa of a single-precision (binary32) float.
  - Normal input with biased exponent 1..254: mantissa placed in interval [0.5, 1.0)
    by replacing exponent field with 126 (0x3F000000), preserving sign and fractional bits.
  - Zero and subnormals/denormals are flushed to signed zero (+-0.0), preserving sign bit.
  - Infinities and NaNs (biased exponent 255) are passed through unchanged.
  - VOP1 encoding with single source and destination VGPR.
  - VALU operation: scalar condition code (SCC) and vector condition code (VCC) remain untouched.

Standard library Python only (unittest, struct, math, dataclasses, enum), strictly zero game data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Optional, Sequence, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def f32_to_u32(val: float) -> int:
    """Pack Python float as IEEE 754 binary32 uint32 bits."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def u32_to_f32(bits: int) -> float:
    """Unpack uint32 bits as Python float (IEEE 754 binary32)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def ref_frexp_mant_f32(x: int) -> int:
    """Architectural reference oracle for RDNA2 v_frexp_mant_f32.

    Returns the normalized mantissa in [0.5, 1.0) with biased exponent 126.
    Zero and denormals flush to +-0.0. Infinities and NaNs pass through unchanged.
    """
    x &= MASK32
    e = (x >> 23) & 0xFF
    sgn = x & 0x80000000
    if e == 0:
        return sgn
    elif e == 255:
        return x
    else:
        keep = x & 0x807FFFFF
        return (keep | 0x3F000000) & MASK32


def run_ps_frexp_mant_f32(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_mant_f32_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_powers_of_two(self):
        # 1.0 (0x3F800000) -> 0.5 (0x3F000000)
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(1.0)), f32_to_u32(0.5))
        # 2.0 (0x40000000) -> 0.5 (0x3F000000)
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(2.0)), f32_to_u32(0.5))
        # 4.0 (0x40800000) -> 0.5 (0x3F000000)
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(4.0)), f32_to_u32(0.5))
        # 0.5 (0x3F000000) -> 0.5 (0x3F000000)
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(0.5)), f32_to_u32(0.5))

    def test_fractional_numbers(self):
        # 3.0 (0x40400000) = 0.75 * 2^2 -> mantissa 0.75 (0x3F400000)
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(3.0)), f32_to_u32(0.75))
        # 1.5 (0x3FC00000) = 0.75 * 2^1 -> mantissa 0.75 (0x3F400000)
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(1.5)), f32_to_u32(0.75))

    def test_negative_numbers(self):
        # -1.0 (0xBF800000) -> -0.5 (0xBF000000)
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(-1.0)), f32_to_u32(-0.5))
        # -3.0 (0xC0400000) -> -0.75 (0xBF400000)
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(-3.0)), f32_to_u32(-0.75))

    def test_translator_basic_execution(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x40000000",  # 2.0
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words[0], f32_to_u32(0.5))

    def test_translator_fractional_execution(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x40400000",  # 3.0
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words[0], f32_to_u32(0.75))


if __name__ == "__main__":
    unittest.main()
