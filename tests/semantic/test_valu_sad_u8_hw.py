"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_sad_u8.

Models AMD RDNA2 / GFX10 Vector ALU (VALU) v_sad_u8 sum of absolute differences semantics:
  - Computes unsigned byte-level sum of absolute differences plus accumulator:
      D = S2 + |S0.byte0 - S1.byte0| + |S0.byte1 - S1.byte1| + |S0.byte2 - S1.byte2| + |S0.byte3 - S1.byte3|
  - Each byte difference is unsigned (the absolute diff of two 8-bit unsigned values).
  - The accumulator S2 and the final result are 32-bit unsigned (wrapping).
  - VOP3 encoding with 3 input operands (VGPR, SGPR, inline constants, 32-bit literals).
  - VALU operation: does not modify SCC or VCC.

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
MASK8 = 0xFF


def ref_sad_u8(s0: int, s1: int, s2: int) -> int:
    """Architectural reference oracle for RDNA2 v_sad_u8.

    Returns s2 + sum(|s0.byte_i - s1.byte_i| for i in 0..3), 32-bit unsigned wrapping.
    """
    s0 &= MASK32
    s1 &= MASK32
    s2 &= MASK32
    total = s2
    for shift in (0, 8, 16, 24):
        b0 = (s0 >> shift) & MASK8
        b1 = (s1 >> shift) & MASK8
        total += abs(b0 - b1)
    return total & MASK32


def run_ps_sad_u8(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_sad_u8_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_basic_sad_all_zero_diff(self):
        # Identical bytes: all diffs are 0
        self.assertEqual(ref_sad_u8(0x01020304, 0x01020304, 0), 0)
        # With accumulator
        self.assertEqual(ref_sad_u8(0x01020304, 0x01020304, 100), 100)

    def test_basic_sad_known_values(self):
        # 0x04030201 vs 0x00000000: sum = 4+3+2+1 = 10
        self.assertEqual(ref_sad_u8(0x04030201, 0x00000000, 0), 10)
        # 0x00000000 vs 0xFFFFFFFF: each byte diff is 255, sum = 1020
        self.assertEqual(ref_sad_u8(0x00000000, 0xFFFFFFFF, 0), 1020)

    def test_accumulator_addition(self):
        # 0x04030201 vs 0: sum=10, acc=50 -> result=60
        self.assertEqual(ref_sad_u8(0x04030201, 0x00000000, 50), 60)

    def test_translator_basic_execution(self):
        words = run_ps_sad_u8([
            "v_mov_b32 v0, 0x04030201",
            "v_mov_b32 v1, 0x00000000",
            "v_mov_b32 v2, 0",
            "v_sad_u8 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], 10)

    def test_translator_with_accumulator(self):
        words = run_ps_sad_u8([
            "v_mov_b32 v0, 0x04030201",
            "v_mov_b32 v1, 0x00000000",
            "v_mov_b32 v2, 50",
            "v_sad_u8 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], 60)


if __name__ == "__main__":
    unittest.main()
