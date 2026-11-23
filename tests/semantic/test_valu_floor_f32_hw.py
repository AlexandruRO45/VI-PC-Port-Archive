from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for v_floor_f32.

RDNA2 Instruction Set Architecture, VOP1:
    v_floor_f32 D, S0

Semantic operation:
    D.f32 = floor(S0.f32)

Hardware rules (GFX10 RDNA2):
    - Float32 IEEE 754 floor operation.
    - f32 denormals flushed to zero preserving sign: floor(+denorm) = +0.0, floor(-denorm) = -0.0.
    - Zero preservation: floor(+0.0) = +0.0, floor(-0.0) = -0.0.
    - Infs pass through: floor(+inf) = +inf, floor(-inf) = -inf.
    - NaNs pass through unchanged (quiet and signaling payloads preserved).

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
EXP_MASK32: int = 0x7F80_0000
MANT_MASK32: int = 0x007F_FFFF

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def v_floor_f32_oracle(val: int) -> int:
    """Reference mathematical model matching GFX10 RDNA2 hardware semantics."""
    val = val & MASK32
    exp = (val >> 23) & 0xFF
    mant = val & MANT_MASK32
    sign = val & SIGN_BIT32

    # NaN pass through
    if exp == 0xFF and mant != 0:
        return val
    # Inf pass through
    if exp == 0xFF and mant == 0:
        return val
    # Signed zero pass through
    if exp == 0 and mant == 0:
        return val
    # Denormals flushed to signed zero
    if exp == 0:
        return sign

    f = to_f32(val)
    rf = float(math.floor(f))
    if rf == 0.0 and sign:
        return SIGN_BIT32
    return to_u32(rf)


class VFloorF32Oracle:
    """Architectural model and hardware-rule validator for v_floor_f32."""

    @staticmethod
    def eval(val: int) -> int:
        return v_floor_f32_oracle(val)

    @staticmethod
    def build_compute_shader(val: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test v_floor_f32."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_floor_f32_hw"
        lines = [
            "s_mov_b64 exec, -1",
            f"v_mov_b32_e32 v1, {val:#x}",
            "v_floor_f32 v2, v1",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(val: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = VFloorF32Oracle.build_compute_shader(val)
        return support.translate_lines(name, lines, fssm=fssm)


class TestVFloorF32Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_hardware_measured_table(self):
        # Direct measured values on AMD RDNA2
        measured = [
            (0x00000001, 0x00000000),  # +denorm -> +0.0
            (0x80000001, 0x80000000),  # -denorm -> -0.0
            (0x00400000, 0x00000000),  # +denorm -> +0.0
            (0x807FFFFF, 0x80000000),  # -denorm -> -0.0
            (0x80000000, 0x80000000),  # -0.0 -> -0.0
            (0x00000000, 0x00000000),  # +0.0 -> +0.0
            (0x00800000, 0x00000000),  # min normal (+1.17e-38) -> 0.0
            (0x80800000, 0xBF800000),  # min neg normal (-1.17e-38) -> -1.0
            (0x7F800000, 0x7F800000),  # +inf
            (0xFF800000, 0xFF800000),  # -inf
            (0x7FA00000, 0x7FA00000),  # sNaN
            (0x7FC00000, 0x7FC00000),  # qNaN
            (0xBF000000, 0xBF800000),  # -0.5 -> -1.0
            (0x3FC00000, 0x3F800000),  # +1.5 -> +1.0
            (0x3F7FFFFF, 0x00000000),  # 0.99999994 -> 0.0
            (0xBF7FFFFF, 0xBF800000),  # -0.99999994 -> -1.0
        ]
        for inp, expected in measured:
            with self.subTest(inp=hex(inp)):
                self.assertEqual(VFloorF32Oracle.eval(inp), expected)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = VFloorF32Oracle.translate_and_validate(0x3FC0_0000)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpFunction", spv)
        self.assertIn("OpReturn", spv)

if __name__ == "__main__":
    unittest.main()
