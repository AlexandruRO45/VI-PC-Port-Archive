from __future__ import annotations

"""Hardware-verified semantic reference and translator regression suite for RDNA2 v_add3_u32.

Tests v_add3_u32 (three-operand 32-bit unsigned addition) against mathematical reference semantics
and translated SPIR-V shader executions via rdna2spirv and spirv_mini_interp.
"""

import dataclasses
import enum
import os
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def oracle_add3_u32(a: int, b: int, c: int) -> int:
    """Reference mathematical oracle for RDNA2 v_add3_u32: (a + b + c) mod 2^32."""
    return (u32(a) + u32(b) + u32(c)) & MASK32


_TRANSLATE_ID: int = 0


def run_translated_add3_u32(src0: int, src1: int, src2: int) -> int:
    """Execute v_add3_u32 through rdna2spirv translation and spirv_mini_interp execution."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_add3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_add3_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


class TestValuAdd3U32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_add3_u32."""

    def test_oracle_basic_addition(self) -> None:
        self.assertEqual(oracle_add3_u32(10, 20, 30), 60)
        self.assertEqual(oracle_add3_u32(0, 0, 0), 0)
        self.assertEqual(oracle_add3_u32(1, 2, 3), 6)
        self.assertEqual(oracle_add3_u32(100, 200, 300), 600)

    def test_oracle_wrapping(self) -> None:
        self.assertEqual(oracle_add3_u32(MASK32, 1, 0), 0)
        self.assertEqual(oracle_add3_u32(MASK32, 1, 5), 5)
        self.assertEqual(oracle_add3_u32(MASK32, MASK32, MASK32), MASK32 - 2)

    def test_oracle_matches_translated_execution(self) -> None:
        triples = [
            (0, 0, 0),
            (10, 20, 30),
            (1000, 2000, 3000),
            (0x12345678, 0x11111111, 0x22222222),
            (MASK32, 1, 0),
            (MASK32, MASK32, 1),
            (0x7FFFFFFF, 1, 0x80000000),
            (0xAAAAAAAA, 0x55555555, 1),
        ]
        for a, b, c in triples:
            expected = oracle_add3_u32(a, b, c)
            actual = run_translated_add3_u32(a, b, c)
            self.assertEqual(actual, expected)

    def test_oracle_commutativity_and_associativity(self) -> None:
        a, b, c = 0x1234, 0x5678, 0x9ABC
        res1 = oracle_add3_u32(a, b, c)
        res2 = oracle_add3_u32(c, a, b)
        res3 = oracle_add3_u32(b, c, a)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)


if __name__ == "__main__":
    unittest.main()
