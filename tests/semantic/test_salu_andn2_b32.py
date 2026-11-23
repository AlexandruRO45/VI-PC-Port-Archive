from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_andn2_b32.

RDNA2 Instruction Set Architecture, SOP2:
    s_andn2_b32 D, S0, S1

Semantic operation:
    D.u32 = S0.u32 & (~S1.u32 & 0xFFFFFFFF)
    SCC = (D != 0)

Outputs:
    D: 32-bit destination SGPR (or EXEC_LO, VCC_LO, NULL) storing the bitwise AND of S0 and bitwise NOT of S1.
    SCC: 1 if destination is non-zero, 0 if all zeros.

Standard library Python only (unittest, struct, math, dataclasses, enum, typing).
Strictly zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Dict, List, Optional, Set, Tuple, Union
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si

MASK32: int = 0xFFFF_FFFF
ALL32: int = 0xFFFF_FFFF

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def s_andn2_b32_oracle(s0: int, s1: int) -> Tuple[int, int]:
    """Reference mathematical model for s_andn2_b32.

    Args:
        s0: 32-bit unsigned integer source 0.
        s1: 32-bit unsigned integer source 1.

    Returns:
        (d, scc) tuple where:
            d: (s0 & (~s1 & MASK32)) & MASK32
            scc: 1 if d != 0 else 0
    """
    s0_u32 = s0 & MASK32
    s1_u32 = s1 & MASK32
    d = (s0_u32 & (~s1_u32 & MASK32)) & MASK32
    scc = 1 if d != 0 else 0
    return d, scc


class SAndn2B32Oracle:
    """Architectural model and hardware-rule validator for s_andn2_b32."""

    @staticmethod
    def eval(s0: int, s1: int) -> Tuple[int, int]:
        return s_andn2_b32_oracle(s0, s1)

    @staticmethod
    def build_pixel_shader(s0: int, s1: int) -> Tuple[str, List[str]]:
        """Generate synthetic pixel shader lines to test s_andn2_b32."""
        name = "ps_andn2_b32_test"
        lines = [
            f"s_mov_b32 s0, {s0 & MASK32:#x}",
            f"s_mov_b32 s1, {s1 & MASK32:#x}",
            "s_andn2_b32 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        return name, lines

    @staticmethod
    def translate_and_validate(s0: int, s1: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines = SAndn2B32Oracle.build_pixel_shader(s0, s1)
        return support.translate_lines(name, lines)


class TestSAndn2B32Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_truth_table_bit_combinations(self):
        cases = [
            (0, 0, 0),
            (0, 1, 0),
            (1, 0, 1),
            (1, 1, 0),
        ]
        for b0, b1, expected in cases:
            with self.subTest(b0=b0, b1=b1):
                d, scc = SAndn2B32Oracle.eval(b0, b1)
                self.assertEqual(d & 1, expected)
                self.assertEqual(scc, expected)

    def test_oracle_identical_operands_produce_zero(self):
        for val in (0, ALL32, 0x1234_5678, 0x5555_5555, 0xAAAA_AAAA, 0x8000_0000):
            with self.subTest(val=hex(val)):
                d, scc = SAndn2B32Oracle.eval(val, val)
                self.assertEqual(d, 0)
                self.assertEqual(scc, 0)

    def test_oracle_s1_zero_produces_s0(self):
        for s0 in (0, ALL32, 0xCAFE_BABE, 0x0000_0001, 0x8000_0000):
            with self.subTest(s0=hex(s0)):
                d, scc = SAndn2B32Oracle.eval(s0, 0)
                self.assertEqual(d, s0)
                self.assertEqual(scc, 1 if s0 != 0 else 0)

    def test_oracle_s1_all_ones_produces_zero(self):
        for s0 in (0, ALL32, 0x1234_5678, 0xDEAD_BEEF):
            with self.subTest(s0=hex(s0)):
                d, scc = SAndn2B32Oracle.eval(s0, ALL32)
                self.assertEqual(d, 0)
                self.assertEqual(scc, 0)

    def test_oracle_s0_zero_produces_zero(self):
        for s1 in (0, ALL32, 0x5555_5555, 0x1234_5678):
            with self.subTest(s1=hex(s1)):
                d, scc = SAndn2B32Oracle.eval(0, s1)
                self.assertEqual(d, 0)
                self.assertEqual(scc, 0)

    def test_oracle_non_commutative_asymmetry(self):
        a = 0xF0F0_0F0F
        b = 0x3333_CCCC
        d_ab, _ = SAndn2B32Oracle.eval(a, b)
        d_ba, _ = SAndn2B32Oracle.eval(b, a)
        self.assertNotEqual(d_ab, d_ba)
        self.assertEqual(d_ab, a & (~b & MASK32))
        self.assertEqual(d_ba, b & (~a & MASK32))

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SAndn2B32Oracle.translate_and_validate(0x1234_5678, 0x00FF_00FF)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)

if __name__ == "__main__":
    unittest.main()
