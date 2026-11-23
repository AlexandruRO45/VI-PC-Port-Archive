"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_bfe_i32.

Models AMD RDNA2 / GFX10 Vector ALU (VALU) v_bfe_i32 signed bitfield extract semantics:
  - Computes signed bitfield extract: D = (S0.i >> (S1.u & 31)) with bitfield length (S2.u & 31)
  - Extracted field is sign-extended to 32-bit two's-complement integer.
  - If width is 0, the result is 0.
  - VOP3 encoding with VGPR destination and VGPR/SGPR/literal sources.
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


def ref_bfe_i32(base: int, offset: int, width: int) -> int:
    """Architectural reference oracle for RDNA2 v_bfe_i32 signed bitfield extract."""
    base &= MASK32
    offset &= 31
    width &= 31
    if width == 0:
        return 0
    signed = base - (1 << 32) if (base >> 31) else base
    field = (signed >> offset) & ((1 << width) - 1)
    if field & (1 << (width - 1)):
        field -= 1 << width
    return field & MASK32


def run_ps_bfe_i32(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_bfe_i32_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_basic_signed_bitfield_extract(self):
        # Extract positive subfield: 0x12345678, offset=4, width=8 -> 0x67
        self.assertEqual(ref_bfe_i32(0x12345678, 4, 8), 0x67)
        # Extract negative subfield with sign extension: 0xFFFFFFF0, offset=2, width=4 -> 0xFFFFFFFC (-4)
        self.assertEqual(ref_bfe_i32(0xFFFFFFF0, 2, 4), 0xFFFFFFFC)
        # Extract single sign bit: 0x80000000, offset=31, width=1 -> 0xFFFFFFFF (-1)
        self.assertEqual(ref_bfe_i32(0x80000000, 31, 1), 0xFFFFFFFF)
        # Extract zero bit from MSB: 0x7FFFFFFF, offset=31, width=1 -> 0
        self.assertEqual(ref_bfe_i32(0x7FFFFFFF, 31, 1), 0)

    def test_width_zero_yields_zero(self):
        self.assertEqual(ref_bfe_i32(0xFFFFFFFF, 0, 0), 0)
        self.assertEqual(ref_bfe_i32(0x12345678, 12, 0), 0)
        self.assertEqual(ref_bfe_i32(0x80000000, 31, 0), 0)

    def test_translator_basic_execution(self):
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 4",
            "v_mov_b32 v2, 8",
            "v_bfe_i32 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], 0x67)

    def test_translator_sign_extended_negative(self):
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xFFFFFFF0",
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 4",
            "v_bfe_i32 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], 0xFFFFFFFC)


if __name__ == "__main__":
    unittest.main()
