"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_min3_f32.

Models IEEE 754-2008 / IEEE 754-2019 binary32 single-precision 3-way minimum and AMD RDNA2 / GFX10
Vector ALU (VALU) v_min3_f32 semantics:
  - Computes the minimum of three 32-bit IEEE 754 floating-point values: D = min(S0, S1, S2).
  - VOP3 / VOP3A encoding supports input modifiers (negation, absolute value, neg-abs) on each source.
  - Supports output clamp modifier (clamp to [0.0, 1.0]).
  - Evaluates with IEEE 754-2008 minNum / GLSL.std.450 NMin NaN skipping:
      * If one or two operands are NaN, the non-NaN operand is chosen.
      * If all three operands are NaN, NaN is returned.
  - Signed zero ordering: -0.0 is strictly less than +0.0 (0x80000000 < 0x00000000).

Synthesizes test listings, invokes tools/rdna2spirv.py via tests/semantic/support.py,
executes translated SPIR-V shaders in tests/semantic/spirv_mini_interp.py, and compares mrt0 outputs.

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
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
QNAN = 0x7FC00000
SNAN = 0x7F800001
NEG_QNAN = 0xFFC00000
SUBNORM_MIN = 0x00000001
SUBNORM_MAX = 0x007FFFFF
ONE_F32 = 0x3F800000
TWO_F32 = 0x40000000
THREE_F32 = 0x40400000
HALF_F32 = 0x3F000000


def u32_to_f32(val: int) -> float:
    return struct.unpack("<f", struct.pack("<I", val & MASK32))[0]


def f32_to_u32(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def is_nan_u32(val: int) -> bool:
    val &= MASK32
    return (val & 0x7F800000) == 0x7F800000 and (val & 0x007FFFFF) != 0


def is_quiet_nan_u32(val: int) -> bool:
    val &= MASK32
    return (val & 0x7FC00000) == 0x7FC00000


def apply_src_mods(val: int, neg: bool, abs_: bool) -> int:
    val &= MASK32
    if abs_:
        val &= 0x7FFFFFFF
    if neg:
        val ^= 0x80000000
    return val


def ref_min2_f32(u0: int, u1: int) -> int:
    """RDNA2 NMin binary32 minimum between two IEEE-754 bit representations."""
    nan0 = is_nan_u32(u0)
    nan1 = is_nan_u32(u1)
    if nan0 and nan1:
        return QNAN
    if nan0:
        return u1
    if nan1:
        return u0

    # Signed zero comparison: -0.0 is strictly less than +0.0
    if (u0 & 0x7FFFFFFF) == 0 and (u1 & 0x7FFFFFFF) == 0:
        if (u0 & 0x80000000) != 0 or (u1 & 0x80000000) != 0:
            return NEG_ZERO
        return POS_ZERO

    f0 = u32_to_f32(u0)
    f1 = u32_to_f32(u1)
    if f0 < f1:
        return u0
    elif f1 < f0:
        return u1
    else:
        # Equal non-zero floats: preserve u0
        return u0


def ref_min3_f32(
    s0: int,
    s1: int,
    s2: int,
    clamp: bool = False,
    neg: Tuple[bool, bool, bool] = (False, False, False),
    abs_: Tuple[bool, bool, bool] = (False, False, False),
) -> int:
    """Architectural reference oracle for RDNA2 v_min3_f32."""
    u0 = apply_src_mods(s0, neg[0], abs_[0])
    u1 = apply_src_mods(s1, neg[1], abs_[1])
    u2 = apply_src_mods(s2, neg[2], abs_[2])

    res = ref_min2_f32(ref_min2_f32(u0, u1), u2)

    if clamp:
        if is_nan_u32(res):
            return POS_ZERO
        f_res = u32_to_f32(res)
        if f_res < 0.0:
            return POS_ZERO
        elif f_res > 1.0:
            return ONE_F32
        else:
            return res
    return res


def run_ps_min3(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_min3_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_basic_ordering(self):
        # min(1.0, 2.0, 3.0) -> 1.0
        r = ref_min3_f32(ONE_F32, TWO_F32, THREE_F32)
        self.assertEqual(r, ONE_F32)

        # min(3.0, 1.0, 2.0) -> 1.0
        r2 = ref_min3_f32(THREE_F32, ONE_F32, TWO_F32)
        self.assertEqual(r2, ONE_F32)

        # min(3.0, 2.0, 1.0) -> 1.0
        r3 = ref_min3_f32(THREE_F32, TWO_F32, ONE_F32)
        self.assertEqual(r3, ONE_F32)

    def test_translator_basic_execution(self):
        words = run_ps_min3([
            f"v_mov_b32 v0, {ONE_F32:#x}",
            f"v_mov_b32 v1, {TWO_F32:#x}",
            f"v_mov_b32 v2, {THREE_F32:#x}",
            "v_min3_f32 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], ONE_F32)

    def test_clamp_output_modifier(self):
        # min(2.0, 3.0, 4.0) with clamp -> 1.0
        r = ref_min3_f32(TWO_F32, THREE_F32, f32_to_u32(4.0), clamp=True)
        self.assertEqual(r, ONE_F32)

        words = run_ps_min3([
            f"v_mov_b32 v0, {TWO_F32:#x}",
            f"v_mov_b32 v1, {THREE_F32:#x}",
            f"v_mov_b32 v2, {f32_to_u32(4.0):#x}",
            "v_min3_f32 v3, v0, v1, v2 clamp",
        ])
        self.assertEqual(words[0], ONE_F32)

    def test_input_modifiers(self):
        # min(-|2.0|, 3.0, 4.0) -> -2.0
        v0 = TWO_F32
        v1 = THREE_F32
        v2 = f32_to_u32(4.0)
        expected = f32_to_u32(-2.0)
        r = ref_min3_f32(v0, v1, v2, neg=(True, False, False), abs_=(True, False, False))
        self.assertEqual(r, expected)

        words = run_ps_min3([
            f"v_mov_b32 v0, {v0:#x}",
            f"v_mov_b32 v1, {v1:#x}",
            f"v_mov_b32 v2, {v2:#x}",
            "v_min3_f32 v3, -|v0|, v1, v2",
        ])
        self.assertEqual(words[0], expected)


if __name__ == "__main__":
    unittest.main()
