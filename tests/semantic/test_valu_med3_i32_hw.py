"""Hardware-accurate semantic tests and mathematical reference model for RDNA2 v_med3_i32.

RDNA2 / GFX10 Vector ALU Signed 32-bit Integer Three-Operand Median:
Performs signed integer comparison across three 32-bit operands and returns the median:
    vdst = median_i32(src0, src1, src2)

Syntax:
    v_med3_i32 vdst, src0, src1, src2

Semantics:
    Interprets src0, src1, src2 as 32-bit two's complement signed integers in [-2^31, 2^31 - 1].
    Sorts the three integers in non-decreasing order: s0 <= s1 <= s2.
    vdst = s1 (the median element), represented as a 32-bit unsigned dword (& 0xFFFFFFFF).
    Equivalent to clamp(src2, min(src0, src1), max(src0, src1)) in signed integer arithmetic.

Translator Integration:
    Exercises tools/rdna2spirv.py via support.translate_lines and executes the generated
    SPIR-V shaders via support.mrt0_words.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
from enum import Enum, auto
import itertools
import math
import os
import struct
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
import rdna2spirv  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402

MASK32: int = 0xFFFFFFFF
SIGN_BIT32: int = 0x80000000
INT32_MIN: int = -0x80000000
INT32_MAX: int = 0x7FFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def ref_med3_i32(s0: int, s1: int, s2: int) -> int:
    """Pure mathematical reference oracle for signed 32-bit median."""
    sorted_vals = sorted([s32(s0), s32(s1), s32(s2)])
    return u32(sorted_vals[1])


def hw_clamp_med3_i32(s0: int, s1: int, s2: int) -> int:
    """Hardware clamp-equivalent model matching GLSL SClamp(z, SMin(x, y), SMax(x, y))."""
    lo = min(s32(s0), s32(s1))
    hi = max(s32(s0), s32(s1))
    clamped = min(max(s32(s2), lo), hi)
    return u32(clamped)


def run_shader_med3_i32(
    src0_val: int,
    src1_val: int,
    src2_val: int,
    vdst_reg: int = 0,
    v0_reg: int = 1,
    v1_reg: int = 2,
    v2_reg: int = 3,
) -> int:
    """Translate and execute a synthetic pixel shader using v_med3_i32, returning mrt0.x."""
    lines = [
        f"v_mov_b32_e32 v{v0_reg}, {u32(src0_val):#x}",
        f"v_mov_b32_e32 v{v1_reg}, {u32(src1_val):#x}",
        f"v_mov_b32_e32 v{v2_reg}, {u32(src2_val):#x}",
        f"v_med3_i32 v{vdst_reg}, v{v0_reg}, v{v1_reg}, v{v2_reg}",
        f"exp mrt0, v{vdst_reg}, v{vdst_reg}, v{vdst_reg}, v{vdst_reg} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_med3_i32_test", lines)
    words = support.mrt0_words(spv)
    return words[0]


class TestMed3I32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_oracle_basic_integers(self):
        cases = [
            (10, 20, 30, 20),
            (30, 10, 20, 20),
            (-5, 0, 5, 0),
            (-100, -200, -50, -100),
            (7, 7, 7, 7),
            (12, 12, 5, 12),
            (5, 12, 12, 12),
        ]
        for s0, s1, s2, expected in cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                self.assertEqual(ref_med3_i32(s0, s1, s2), u32(expected))
                self.assertEqual(hw_clamp_med3_i32(s0, s1, s2), u32(expected))

    def test_oracle_permutation_symmetry(self):
        triples = [
            (5, -10, 25),
            (-50, 0, 100),
            (0x7FFFFFFF, -1, -0x80000000),
            (42, 42, -99),
        ]
        for trip in triples:
            expected = ref_med3_i32(*trip)
            for perm in itertools.permutations(trip):
                with self.subTest(perm=perm):
                    self.assertEqual(ref_med3_i32(*perm), expected)
                    self.assertEqual(hw_clamp_med3_i32(*perm), expected)

    def test_translator_single_invocation(self):
        result = run_shader_med3_i32(5, 0x80000000, 7)
        self.assertEqual(result, 5)
        self.assertEqual(result, ref_med3_i32(5, 0x80000000, 7))


if __name__ == "__main__":
    unittest.main()
