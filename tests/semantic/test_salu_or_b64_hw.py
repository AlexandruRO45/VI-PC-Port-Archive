from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_or_b64.

RDNA2 Instruction Set Architecture, SOP2:
    s_or_b64 D, S0, S1

Semantic operation:
    D.u64 = S0.u64 | S1.u64
    SCC = (D != 0)

Outputs:
    D: 64-bit destination SGPR pair (or EXEC/VCC) storing the bitwise OR of S0 and S1.
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
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
ALL64: int = 0xFFFF_FFFF_FFFF_FFFF

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def s_or_b64_oracle(s0: int, s1: int) -> Tuple[int, int]:
    """Reference mathematical model for s_or_b64.

    Args:
        s0: 64-bit unsigned integer source 0.
        s1: 64-bit unsigned integer source 1.

    Returns:
        (d, scc) tuple where:
            d: (s0 | s1) & MASK64
            scc: 1 if d != 0 else 0
    """
    s0_u64 = s0 & MASK64
    s1_u64 = s1 & MASK64
    d = (s0_u64 | s1_u64) & MASK64
    scc = 1 if d != 0 else 0
    return d, scc


class SOrB64Oracle:
    """Architectural model and hardware-rule validator for s_or_b64."""

    @staticmethod
    def eval(s0: int, s1: int) -> Tuple[int, int]:
        return s_or_b64_oracle(s0, s1)

    @staticmethod
    def build_compute_shader(s0: int, s1: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test s_or_b64."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_or_b64_hw"
        lines = [
            "s_mov_b64 exec, -1",
            f"s_mov_b32 s2, {s0 & MASK32:#x}",
            f"s_mov_b32 s3, {(s0 >> 32) & MASK32:#x}",
            f"s_mov_b32 s4, {s1 & MASK32:#x}",
            f"s_mov_b32 s5, {(s1 >> 32) & MASK32:#x}",
            "s_or_b64 s[6:7], s[2:3], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, s1: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = SOrB64Oracle.build_compute_shader(s0, s1)
        return support.translate_lines(name, lines, fssm=fssm)


class TestSOrB64Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_truth_table_bit_combinations(self):
        # Truth table for bitwise OR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1
        cases = [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 1),
        ]
        for b0, b1, expected in cases:
            with self.subTest(b0=b0, b1=b1):
                d, scc = SOrB64Oracle.eval(b0, b1)
                self.assertEqual(d & 1, expected)
                self.assertEqual(scc, expected)

    def test_oracle_identity_zero(self):
        # x | 0 = x
        for val in (0, ALL64, 0x1234_5678_9ABC_DEF0, 0x5555_5555_5555_5555):
            with self.subTest(val=hex(val)):
                d, scc = SOrB64Oracle.eval(val, 0)
                self.assertEqual(d, val)
                self.assertEqual(scc, 1 if val != 0 else 0)

    def test_oracle_annihilator_all_ones(self):
        # x | ALL64 = ALL64
        for val in (0, ALL64, 0x1234_5678_9ABC_DEF0, 0xAAAA_AAAA_AAAA_AAAA):
            with self.subTest(val=hex(val)):
                d, scc = SOrB64Oracle.eval(val, ALL64)
                self.assertEqual(d, ALL64)
                self.assertEqual(scc, 1)

    def test_oracle_idempotence(self):
        # x | x = x
        for val in (0, ALL64, 0xCAFE_BABE_DEAD_BEEF):
            with self.subTest(val=hex(val)):
                d, scc = SOrB64Oracle.eval(val, val)
                self.assertEqual(d, val)
                self.assertEqual(scc, 1 if val != 0 else 0)

    def test_oracle_commutative(self):
        # s0 | s1 == s1 | s0
        s0, s1 = 0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321
        d1, scc1 = SOrB64Oracle.eval(s0, s1)
        d2, scc2 = SOrB64Oracle.eval(s1, s0)
        self.assertEqual(d1, d2)
        self.assertEqual(scc1, scc2)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SOrB64Oracle.translate_and_validate(0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpFunction", spv)
        self.assertIn("OpReturn", spv)

if __name__ == "__main__":
    unittest.main()
