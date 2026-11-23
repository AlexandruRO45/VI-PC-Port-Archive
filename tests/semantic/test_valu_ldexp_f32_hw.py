from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for v_ldexp_f32.

RDNA2 Instruction Set Architecture, VOP2/VOP3:
    v_ldexp_f32 D, S0, S1

Semantic operation:
    D.f32 = S0.f32 * 2^(S1.i32)

Bit-exact binary32 semantics:
    - Normal inputs (biased exponent 1..254): exponent field is incremented by clamped S1 in [-512, 512].
    - Zero and denormals (biased exponent 0): flushed to signed zero (+0.0 / -0.0).
    - Infinities and NaNs (biased exponent 255): preserved bit-exact without modification.
    - Exponent overflow (> 254): saturates to +/-infinity with original sign.
    - Exponent underflow (<= 0): saturates to +/-zero with original sign.

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
SIGN_MASK: int = 0x8000_0000
EXP_MASK: int = 0x7F80_0000
MANT_MASK: int = 0x007F_FFFF
POS_INF: int = 0x7F80_0000
NEG_INF: int = 0xFF80_0000

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def ref_ldexp_f32(s0_u32: int, s1_i32: int) -> int:
    """Reference mathematical model for v_ldexp_f32 matching RDNA2 hardware semantics."""
    x = s0_u32 & MASK32
    sgn = x & SIGN_MASK
    e = (x >> 23) & 0xFF

    # Zero and denormals flush to signed zero (+-0.0)
    if e == 0:
        return sgn
    # Infinities and NaNs pass through unchanged
    if e == 255:
        return x

    # Normal input: clamp s1 to [-512, 512]
    if s1_i32 >= 0x8000_0000:
        s1_val = s1_i32 - 0x1_0000_0000
    else:
        s1_val = s1_i32

    clamped_n = max(-512, min(512, s1_val))
    new_e = e + clamped_n

    if 1 <= new_e <= 254:
        return sgn | (new_e << 23) | (x & MANT_MASK)
    elif new_e > 254:
        return sgn | POS_INF
    else:
        return sgn


class VLdexpF32Oracle:
    """Architectural model and hardware-rule validator for v_ldexp_f32."""

    @staticmethod
    def eval(s0_u32: int, s1_i32: int) -> int:
        return ref_ldexp_f32(s0_u32, s1_i32)

    @staticmethod
    def build_shader(s0_u32: int, s1_i32: int) -> Tuple[str, List[str]]:
        name = "ps_ldexp_f32_hw"
        lines = [
            f"v_mov_b32_e32 v0, {s0_u32:#x}",
            f"s_mov_b32 s0, {s1_i32:#x}",
            "v_ldexp_f32 v2, v0, s0",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        return name, lines

    @staticmethod
    def translate_and_validate(s0_u32: int, s1_i32: int) -> str:
        name, lines = VLdexpF32Oracle.build_shader(s0_u32, s1_i32)
        return support.translate_lines(name, lines)

    @staticmethod
    def execute_and_verify(s0_u32: int, s1_i32: int) -> int:
        spv = VLdexpF32Oracle.translate_and_validate(s0_u32, s1_i32)
        out = support.mrt0_words(spv)
        return out[0]


class TestVLdexpF32Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_identity_scale_zero(self):
        # 1.0f * 2^0 = 1.0f (0x3F80_0000)
        one = 0x3F80_0000
        res = VLdexpF32Oracle.eval(one, 0)
        self.assertEqual(res, one)

    def test_oracle_positive_scale(self):
        # 1.0f * 2^2 = 4.0f (0x4080_0000)
        one = 0x3F80_0000
        res = VLdexpF32Oracle.eval(one, 2)
        self.assertEqual(res, 0x4080_0000)

    def test_oracle_negative_scale(self):
        # 1.0f * 2^(-2) = 0.25f (0x3E80_0000)
        one = 0x3F80_0000
        res = VLdexpF32Oracle.eval(one, -2)
        self.assertEqual(res, 0x3E80_0000)

    def test_oracle_overflow_to_infinity(self):
        one = 0x3F80_0000
        res = VLdexpF32Oracle.eval(one, 150)
        self.assertEqual(res, POS_INF)

    def test_oracle_underflow_to_zero(self):
        one = 0x3F80_0000
        res = VLdexpF32Oracle.eval(one, -150)
        self.assertEqual(res, 0)

    def test_translator_and_execution_pipeline(self):
        """Verify translation and execution of 1.0f * 2^2 = 4.0f through rdna2spirv and spirv_mini_interp."""
        one = 0x3F80_0000
        got = VLdexpF32Oracle.execute_and_verify(one, 2)
        self.assertEqual(got, 0x4080_0000)

if __name__ == "__main__":
    unittest.main()
