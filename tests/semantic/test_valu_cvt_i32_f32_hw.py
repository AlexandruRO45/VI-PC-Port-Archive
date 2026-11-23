from __future__ import annotations

"""Semantic reference oracle and architectural model for v_cvt_i32_f32.

RDNA2 / GFX10 VOP1 Floating-Point to Signed 32-bit Integer Conversion:
Converts an IEEE 754 single-precision float (32-bit) to a signed 32-bit integer with truncation
toward zero.

Saturation and Edge Case Behavior (measured on RDNA2 / gfx1036):
- NaN inputs (both quiet and signaling) produce 0.
- Values >= 2^31 (including +infinity) saturate to 0x7FFFFFFF (+2147483647).
- Values <= -2^31 (including -infinity) saturate to 0x80000000 (-2147483648).
- Representable values in (-2^31, 2^31) truncate toward zero (round-toward-zero).
- Floating-point +/-0.0 both produce 0.

Syntax:
    v_cvt_i32_f32 vdst, vsrc
    v_cvt_i32_f32_e32 vdst, vsrc
    v_cvt_i32_f32_e64 vdst, [mods] vsrc
"""

import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK_32: int = 0xFFFF_FFFF
INT32_MAX: int = 0x7FFF_FFFF
INT32_MIN_UNSIGNED: int = 0x8000_0000


def f32_to_bits(f: float) -> int:
    """Pack Python float to IEEE 754 uint32 bits."""
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return 0x7F80_0000 if f > 0 else 0xFF80_0000


def bits_to_f32(bits: int) -> float:
    """Unpack IEEE 754 uint32 bits to Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK_32))[0]


def oracle_cvt_i32_f32(bits: int) -> int:
    """Reference mathematical oracle for v_cvt_i32_f32."""
    f = bits_to_f32(bits)
    if math.isnan(f):
        return 0
    if f >= 2147483648.0:
        return INT32_MAX
    if f <= -2147483648.0:
        return INT32_MIN_UNSIGNED
    val = math.trunc(f)
    return val & MASK_32


def run_translator_cvt_i32_f32(
    src_bits: int,
    name: str = "ps_cvt_i32_f32",
    op: str = "v_cvt_i32_f32",
    src_expr: Optional[str] = None,
) -> int:
    """Translate and run a minimal shader through tools/rdna2spirv and return result word."""
    if src_expr is None:
        lines = [
            f"v_mov_b32 v1, {src_bits & MASK_32:#x}",
            f"{op} v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
    else:
        lines = [
            f"v_mov_b32 v1, {src_bits & MASK_32:#x}",
            f"{op} v0, {src_expr}",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestOracleAndHardwareModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_positive_integers_and_truncation(self) -> None:
        """Verify standard positive float conversion with truncation toward zero."""
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(1.0)), 1)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(1.9999)), 1)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(42.75)), 42)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(100.25)), 100)

    def test_negative_integers_and_truncation(self) -> None:
        """Verify negative float conversion with truncation toward zero."""
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(-1.0)), 0xFFFF_FFFF)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(-1.9999)), 0xFFFF_FFFF)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(-42.75)), (-42) & MASK_32)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(-100.25)), (-100) & MASK_32)

    def test_zeros_and_subnormals(self) -> None:
        """Verify +0.0, -0.0 and subnormals convert to zero."""
        self.assertEqual(oracle_cvt_i32_f32(0x0000_0000), 0)
        self.assertEqual(oracle_cvt_i32_f32(0x8000_0000), 0)
        self.assertEqual(oracle_cvt_i32_f32(0x0000_0001), 0)
        self.assertEqual(oracle_cvt_i32_f32(0x8000_0001), 0)

    def test_saturation_limits(self) -> None:
        """Verify positive and negative overflow saturation."""
        self.assertEqual(oracle_cvt_i32_f32(0x4F00_0000), INT32_MAX)           # 2^31
        self.assertEqual(oracle_cvt_i32_f32(0xCF00_0000), INT32_MIN_UNSIGNED)  # -2^31
        self.assertEqual(oracle_cvt_i32_f32(0x7F80_0000), INT32_MAX)           # +inf
        self.assertEqual(oracle_cvt_i32_f32(0xFF80_0000), INT32_MIN_UNSIGNED)  # -inf

    def test_nan_produces_zero(self) -> None:
        """Verify NaNs produce zero according to RDNA2 hardware semantics."""
        self.assertEqual(oracle_cvt_i32_f32(0x7FC0_0000), 0)  # +qNaN
        self.assertEqual(oracle_cvt_i32_f32(0xFFC0_0000), 0)  # -qNaN
        self.assertEqual(oracle_cvt_i32_f32(0x7F80_0001), 0)  # +sNaN
        self.assertEqual(oracle_cvt_i32_f32(0xFF80_0001), 0)  # -sNaN

    def test_oracle_matches_translator(self) -> None:
        """Verify oracle directly against translator execution."""
        cases = [
            0x0000_0000,
            0x8000_0000,
            f32_to_bits(1.0),
            f32_to_bits(-1.0),
            f32_to_bits(42.75),
            f32_to_bits(-42.75),
            0x4F00_0000,
            0xCF00_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
        ]
        for idx, bits in enumerate(cases):
            expected = oracle_cvt_i32_f32(bits)
            translated = run_translator_cvt_i32_f32(bits, name=f"ps_oracle_{idx}")
            self.assertEqual(translated, expected, f"Mismatch on bits {bits:#010x}")
