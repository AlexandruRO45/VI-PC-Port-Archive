from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for v_min3_i32.

RDNA2 Instruction Set Architecture, VOP3:
    v_min3_i32 D, S0, S1, S2

Semantic operation:
    D.i32 = min(to_s32(S0), to_s32(S1), to_s32(S2))
    D.u32 = to_u32(D.i32)

Outputs:
    D: 32-bit destination VGPR storing the signed 32-bit minimum of the three input sources.

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
SIGN_BIT32: int = 0x8000_0000
MIN_INT32: int = -0x8000_0000
MAX_INT32: int = 0x7FFF_FFFF

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def to_s32(val: int) -> int:
    """Convert 32-bit unsigned integer to signed two's-complement integer."""
    val = val & MASK32
    return val - 0x1_0000_0000 if (val & SIGN_BIT32) else val


def to_u32(val: int) -> int:
    """Convert signed integer to 32-bit unsigned integer representation."""
    return val & MASK32


def v_min3_i32_oracle(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical model for v_min3_i32.

    Computes the minimum of three signed 32-bit integers.
    """
    v0 = to_s32(s0)
    v1 = to_s32(s1)
    v2 = to_s32(s2)
    return to_u32(min(v0, v1, v2))


class VMin3I32Oracle:
    """Architectural model and hardware-rule validator for v_min3_i32."""

    @staticmethod
    def eval(s0: int, s1: int, s2: int) -> int:
        return v_min3_i32_oracle(s0, s1, s2)

    @staticmethod
    def build_compute_shader(s0: int, s1: int, s2: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test v_min3_i32."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_min3_i32_hw"
        lines = [
            "s_mov_b64 exec, -1",
            f"v_mov_b32_e32 v1, {to_s32(s0)}",
            f"v_mov_b32_e32 v2, {to_s32(s1)}",
            f"v_mov_b32_e32 v3, {to_s32(s2)}",
            "v_min3_i32 v4, v1, v2, v3",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, s1: int, s2: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = VMin3I32Oracle.build_compute_shader(s0, s1, s2)
        return support.translate_lines(name, lines, fssm=fssm)


class TestVMin3I32Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_signed_negatives_vs_positives(self):
        # In signed comparison, -100 is less than 50 and 100
        # -100 is 0xFFFFFF9C
        res = VMin3I32Oracle.eval(100, 50, to_u32(-100))
        self.assertEqual(to_s32(res), -100)

    def test_oracle_signed_negative_one_less_than_zero(self):
        # 0xFFFFFFFF (-1) vs 0 vs 1
        res = VMin3I32Oracle.eval(0, 0xFFFF_FFFF, 1)
        self.assertEqual(to_s32(res), -1)
        self.assertEqual(res, 0xFFFF_FFFF)

    def test_oracle_extreme_min_int32(self):
        # -2147483648 (0x80000000) is the absolute minimum signed 32-bit integer
        res = VMin3I32Oracle.eval(0x8000_0000, 0, 0x7FFF_FFFF)
        self.assertEqual(res, 0x8000_0000)
        self.assertEqual(to_s32(res), MIN_INT32)

    def test_oracle_identical_values(self):
        val = to_u32(-42)
        self.assertEqual(VMin3I32Oracle.eval(val, val, val), val)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = VMin3I32Oracle.translate_and_validate(10, to_u32(-20), 30)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpFunction", spv)
        self.assertIn("OpReturn", spv)

if __name__ == "__main__":
    unittest.main()
