from __future__ import annotations

"""Semantic reference oracle and architectural model for v_min3_i32.

RDNA2 / GFX10 VOP3 3-Operand Signed 32-Bit Integer Minimum:
Computes the signed minimum of three 32-bit two's complement integer operands:
    D.i = min(S0.i, S1.i, S2.i)

Syntax:
    v_min3_i32 vdst, vsrc0, vsrc1, vsrc2

Semantics:
    Interprets S0, S1, S2 as 32-bit signed two's complement integers [-2147483648, 2147483647].
    Result is the minimum of all three inputs, stored as a 32-bit word into vdst.
    Does not modify EXEC, VCC, or SCC condition codes.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible.
"""

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000
INT32_MIN: int = -0x8000_0000      # -2147483648
INT32_MAX: int = 0x7FFF_FFFF       #  2147483647


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword [0, 0xFFFFFFFF]."""
    return val & MASK32


def to_signed_i32(val: int) -> int:
    """Interpret raw 32-bit unsigned dword as two's complement signed 32-bit integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def from_signed_i32(val: int) -> int:
    """Pack signed integer into unsigned 32-bit dword pattern."""
    return val & MASK32


def oracle_min3_i32(s0: int, s1: int, s2: int) -> int:
    """Bit-accurate RDNA2 signed 32-bit integer minimum matching hardware semantics."""
    v0 = to_signed_i32(s0)
    v1 = to_signed_i32(s1)
    v2 = to_signed_i32(s2)
    return min(v0, v1, v2) & MASK32


def run_translator_min3_i32(
    s0: int,
    s1: int,
    s2: int,
    name: str = "ps_min3_i32",
) -> int:
    """Translate and run a minimal shader through tools/rdna2spirv and return result word."""
    lines = [
        f"v_mov_b32 v1, {u32(s0):#x}",
        f"v_mov_b32 v2, {u32(s1):#x}",
        f"v_mov_b32 v3, {u32(s2):#x}",
        "v_min3_i32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestOracleAndHardwareModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_basic_positive_integers(self) -> None:
        """Verify min(10, 20, 30) = 10 across all operand permutations."""
        self.assertEqual(oracle_min3_i32(10, 20, 30), 10)
        self.assertEqual(oracle_min3_i32(30, 10, 20), 10)
        self.assertEqual(oracle_min3_i32(20, 30, 10), 10)

    def test_negative_two_complement(self) -> None:
        """Verify signed comparison: -10 < -5 < 5."""
        s_neg10 = from_signed_i32(-10)
        s_neg5 = from_signed_i32(-5)
        self.assertEqual(oracle_min3_i32(s_neg10, s_neg5, 5), s_neg10)
        self.assertEqual(oracle_min3_i32(5, s_neg5, s_neg10), s_neg10)

    def test_signed_vs_unsigned_divergence(self) -> None:
        """Verify that 0xFFFFFFFF is treated as -1 (less than 1), not 4294967295."""
        val = oracle_min3_i32(0xFFFF_FFFF, 1, 2)
        self.assertEqual(val, 0xFFFF_FFFF)

    def test_extrema_int32_min_max(self) -> None:
        """Verify INT32_MIN is strictly less than 0 and INT32_MAX."""
        u_min = from_signed_i32(INT32_MIN)
        u_max = from_signed_i32(INT32_MAX)
        self.assertEqual(oracle_min3_i32(u_min, 0, u_max), u_min)
        self.assertEqual(oracle_min3_i32(0, u_max, u_min), u_min)

    def test_oracle_matches_translator(self) -> None:
        """Verify reference oracle matches translator output directly."""
        s0, s1, s2 = 100, from_signed_i32(-250), 42
        expected = oracle_min3_i32(s0, s1, s2)
        translated = run_translator_min3_i32(s0, s1, s2, name="ps_min3_oracle_check")
        self.assertEqual(translated, expected)
