"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_med3_f32.

Models IEEE 754-2008 / IEEE 754-2019 binary32 single-precision 3-way median and AMD RDNA2 / GFX10
Vector ALU (VALU) v_med3_f32 semantics:
  - Computes the median of three 32-bit IEEE 754 floating-point values: D = med3(S0, S1, S2).
  - VOP3 / VOP3A encoding supports input modifiers (negation, absolute value, neg-abs) on each source.
  - Supports output clamp modifier (clamp to [0.0, 1.0]).
  - With any NaN operand: RDNA2 hardware returns min3(S0, S1, S2) with NaN-ignoring min semantics.
    If all three operands are NaN, quiet NaN is returned.
  - Without NaN: returns the median.
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


def apply_src_mods(val: int, neg: bool, abs_: bool) -> int:
    val &= MASK32
    if abs_:
        val &= 0x7FFFFFFF
    if neg:
        val ^= 0x80000000
    return val


def cmp_f32_key(u: int) -> Tuple[float, float]:
    f = u32_to_f32(u)
    sign = -1.0 if (u & 0x80000000) else 1.0
    return (f, sign)


def ref_min3_f32_val(u0: int, u1: int, u2: int) -> int:
    nans = [is_nan_u32(x) for x in (u0, u1, u2)]
    if all(nans):
        return QNAN
    valid = [x for x, is_n in zip((u0, u1, u2), nans) if not is_n]
    valid.sort(key=cmp_f32_key)
    return valid[0]


def ref_med3_f32(
    s0: int,
    s1: int,
    s2: int,
    clamp: bool = False,
    neg: Tuple[bool, bool, bool] = (False, False, False),
    abs_: Tuple[bool, bool, bool] = (False, False, False),
) -> int:
    """Architectural reference oracle for RDNA2 v_med3_f32."""
    u0 = apply_src_mods(s0, neg[0], abs_[0])
    u1 = apply_src_mods(s1, neg[1], abs_[1])
    u2 = apply_src_mods(s2, neg[2], abs_[2])

    nan0 = is_nan_u32(u0)
    nan1 = is_nan_u32(u1)
    nan2 = is_nan_u32(u2)

    # If any operand is NaN, hardware evaluates min3(u0, u1, u2) with NaN-skipping
    if nan0 or nan1 or nan2:
        res = ref_min3_f32_val(u0, u1, u2)
    else:
        triplet = [u0, u1, u2]
        triplet.sort(key=cmp_f32_key)
        res = triplet[1]

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


def run_ps_med3(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_med3_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_basic_ordering_and_permutations(self):
        # med3(1.0, 2.0, 3.0) -> 2.0 in all permutations
        p1 = ref_med3_f32(ONE_F32, TWO_F32, THREE_F32)
        p2 = ref_med3_f32(THREE_F32, ONE_F32, TWO_F32)
        p3 = ref_med3_f32(TWO_F32, THREE_F32, ONE_F32)
        self.assertEqual(p1, TWO_F32)
        self.assertEqual(p2, TWO_F32)
        self.assertEqual(p3, TWO_F32)

    def test_translator_basic_execution(self):
        words = run_ps_med3([
            f"v_mov_b32 v0, {ONE_F32:#x}",
            f"v_mov_b32 v1, {TWO_F32:#x}",
            f"v_mov_b32 v2, {THREE_F32:#x}",
            "v_med3_f32 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], TWO_F32)

    def test_clamp_output_modifier(self):
        # med3(0.5, 2.0, 3.0) -> 2.0, clamped -> 1.0
        r = ref_med3_f32(HALF_F32, TWO_F32, THREE_F32, clamp=True)
        self.assertEqual(r, ONE_F32)

        words = run_ps_med3([
            f"v_mov_b32 v0, {HALF_F32:#x}",
            f"v_mov_b32 v1, {TWO_F32:#x}",
            f"v_mov_b32 v2, {THREE_F32:#x}",
            "v_med3_f32 v3, v0, v1, v2 clamp",
        ])
        self.assertEqual(words[0], ONE_F32)

    def test_input_modifiers(self):
        # med3(-|2.0|, 1.0, 3.0) -> med3(-2.0, 1.0, 3.0) -> 1.0
        v0 = TWO_F32
        v1 = ONE_F32
        v2 = THREE_F32
        expected = ONE_F32
        r = ref_med3_f32(v0, v1, v2, neg=(True, False, False), abs_=(True, False, False))
        self.assertEqual(r, expected)

        words = run_ps_med3([
            f"v_mov_b32 v0, {v0:#x}",
            f"v_mov_b32 v1, {v1:#x}",
            f"v_mov_b32 v2, {v2:#x}",
            "v_med3_f32 v3, -|v0|, v1, v2",
        ])
        self.assertEqual(words[0], expected)


if __name__ == "__main__":
    unittest.main()
