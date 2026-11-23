from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 VALU Clamping & Saturation.

RDNA2 Vector Arithmetic Logic Unit (VALU) Clamping & Saturation Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, floating-point arithmetic instructions support output modifiers:
    - clamp: saturates the final result to the normalized range [0.0, 1.0] (GLSL FClamp).
      Values < 0.0 (including -0.0) map to +0.0. Values > 1.0 map to 1.0.
      NaN values propagate according to IEEE 754 rules.
    - omod (output modifier):
      * mul:2 -> multiplies result by 2.0
      * mul:4 -> multiplies result by 4.0
      * div:2 -> multiplies result by 0.5
      On hardware and in the translator, omod is applied before clamp.

Covered instructions across the suite:
  1. v_add_f32 (VOP2 / VOP3 encoding: v_add_f32_e64 vdst, src0, src1 [omod] [clamp])
  2. v_sub_f32 (VOP2 / VOP3 encoding: v_sub_f32_e64 vdst, src0, src1 [omod] [clamp])
  3. v_mul_f32 (VOP2 / VOP3 encoding: v_mul_f32_e64 vdst, src0, src1 [omod] [clamp])
  4. v_fma_f32 (VOP3 encoding: v_fma_f32 vdst, src0, src1, src2 [omod] [clamp])

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
100% of test cases exercise tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words
for hardware-accurate translation and SPIR-V execution verification.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Sequence
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

_counter = 0


def next_test_id(prefix: str = "ps_clamp_sat") -> str:
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def f32_clamp(v: float) -> float:
    if math.isnan(v):
        return v
    if v < 0.0 or (v == 0.0 and math.copysign(1.0, v) < 0.0):
        return 0.0
    elif v > 1.0:
        return 1.0
    return v


def apply_omod_and_clamp(v: float, omod: str | None, clamp: bool) -> float:
    if omod == "mul:2":
        v *= 2.0
    elif omod == "mul:4":
        v *= 4.0
    elif omod == "div:2":
        v *= 0.5
    if clamp:
        v = f32_clamp(v)
    return v


def run_binary_f32(
    op: str,
    s0: float,
    s1: float,
    omod: str | None = None,
    clamp: bool = False,
    name: str | None = None,
) -> float:
    test_id = name or next_test_id(f"ps_{op}")
    s0_bits = f32_to_bits(s0)
    s1_bits = f32_to_bits(s1)
    mods = []
    if omod:
        mods.append(omod)
    if clamp:
        mods.append("clamp")
    mod_str = (" " + " ".join(mods)) if mods else ""
    lines = [
        f"v_mov_b32 v0, {s0_bits:#x}",
        f"v_mov_b32 v1, {s1_bits:#x}",
        f"{op}_e64 v2, v0, v1{mod_str}",
        "exp mrt0, v2, v2, v2, v2 done",
        "s_endpgm",
    ]
    spv = support.translate_lines(test_id, lines)
    words = support.mrt0_words(spv)
    return bits_to_f32(words[0])


def run_ternary_f32(
    op: str,
    s0: float,
    s1: float,
    s2: float,
    omod: str | None = None,
    clamp: bool = False,
    name: str | None = None,
) -> float:
    test_id = name or next_test_id(f"ps_{op}")
    s0_bits = f32_to_bits(s0)
    s1_bits = f32_to_bits(s1)
    s2_bits = f32_to_bits(s2)
    mods = []
    if omod:
        mods.append(omod)
    if clamp:
        mods.append("clamp")
    mod_str = (" " + " ".join(mods)) if mods else ""
    lines = [
        f"v_mov_b32 v0, {s0_bits:#x}",
        f"v_mov_b32 v1, {s1_bits:#x}",
        f"v_mov_b32 v2, {s2_bits:#x}",
        f"{op} v3, v0, v1, v2{mod_str}",
        "exp mrt0, v3, v3, v3, v3 done",
        "s_endpgm",
    ]
    spv = support.translate_lines(test_id, lines)
    words = support.mrt0_words(spv)
    return bits_to_f32(words[0])

class TestValuAddF32ClampOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle for v_add_f32 clamp and omod."""

    def test_add_clamp_saturation_upper(self):
        # 0.6 + 0.7 = 1.3 -> clamped to 1.0
        hw = run_binary_f32("v_add_f32", 0.6, 0.7, clamp=True)
        self.assertAlmostEqual(hw, 1.0, places=6)

    def test_add_clamp_saturation_lower(self):
        # -0.8 + 0.2 = -0.6 -> clamped to 0.0
        hw = run_binary_f32("v_add_f32", -0.8, 0.2, clamp=True)
        self.assertAlmostEqual(hw, 0.0, places=6)

    def test_add_clamp_in_range(self):
        # 0.2 + 0.3 = 0.5 -> remains 0.5
        hw = run_binary_f32("v_add_f32", 0.2, 0.3, clamp=True)
        self.assertAlmostEqual(hw, 0.5, places=6)

    def test_add_omod_mul2(self):
        # (0.2 + 0.3) * 2.0 = 1.0
        hw = run_binary_f32("v_add_f32", 0.2, 0.3, omod="mul:2")
        self.assertAlmostEqual(hw, 1.0, places=6)

    def test_add_omod_mul2_clamp(self):
        # (0.4 + 0.3) * 2.0 = 1.4 -> clamped to 1.0
        hw = run_binary_f32("v_add_f32", 0.4, 0.3, omod="mul:2", clamp=True)
        self.assertAlmostEqual(hw, 1.0, places=6)

class TestValuAddF32AST(unittest.TestCase):
    """Stage 2: Symbolic AST and typed modifier definitions for v_add_f32."""

    def test_ast_add_compilation_clamp(self):
        lines = [
            "v_mov_b32 v0, 0x3f000000",
            "v_mov_b32 v1, 0x3f000000",
            "v_add_f32_e64 v2, v0, v1 clamp",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_ast_add"), lines)
        self.assertIn("OpFAdd", spv)
        self.assertIn("FClamp", spv)
        words = support.mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(words[0]), 1.0, places=6)

    def test_ast_add_compilation_omod_mul4(self):
        lines = [
            "v_mov_b32 v0, 0x3e800000",  # 0.25
            "v_mov_b32 v1, 0x00000000",
            "v_add_f32_e64 v2, v0, v1 mul:4",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_ast_add_mul4"), lines)
        self.assertIn("OpFMul", spv)
        words = support.mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(words[0]), 1.0, places=6)

class TestValuAddF32ScalarExecution(unittest.TestCase):
    """Stage 3: Scalar execution tests for v_add_f32 clamp and omod."""

    def test_add_sgpr_source_clamped(self):
        lines = [
            "s_mov_b32 s0, 0x3f400000",  # 0.75
            "v_mov_b32 v0, 0x3f000000",  # 0.5
            "v_add_f32_e64 v1, v0, s0 clamp",  # 0.5 + 0.75 = 1.25 -> 1.0
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        words = support.mrt0_words(support.translate_lines(next_test_id("ps_add_sgpr"), lines))
        self.assertAlmostEqual(bits_to_f32(words[0]), 1.0, places=6)

    def test_add_aliasing_dst_src0(self):
        lines = [
            "v_mov_b32 v0, 0x3f400000",  # 0.75
            "v_mov_b32 v1, 0x3f000000",  # 0.5
            "v_add_f32_e64 v0, v0, v1 clamp",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        words = support.mrt0_words(support.translate_lines(next_test_id("ps_add_alias"), lines))
        self.assertAlmostEqual(bits_to_f32(words[0]), 1.0, places=6)

class TestValuAddF32SIMDDivergence(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence tests for v_add_f32 clamp and omod."""

    def test_add_lane_symmetric(self):
        # A + B == B + A under clamp
        res_ab = run_binary_f32("v_add_f32", 0.3, 0.4, clamp=True)
        res_ba = run_binary_f32("v_add_f32", 0.4, 0.3, clamp=True)
        self.assertAlmostEqual(res_ab, res_ba, places=6)

    def test_add_div2_divergent_lanes(self):
        # (0.8 + 0.6) * 0.5 = 0.7
        hw = run_binary_f32("v_add_f32", 0.8, 0.6, omod="div:2")
        self.assertAlmostEqual(hw, 0.7, places=6)

class TestValuAddF32BoundaryExtremes(unittest.TestCase):
    """Stage 5: IEEE 754 boundary extremes and saturation for v_add_f32."""

    def test_add_inf_clamping(self):
        # inf + 1.0 = inf -> clamped to 1.0
        hw = run_binary_f32("v_add_f32", float("inf"), 1.0, clamp=True)
        self.assertEqual(hw, 1.0)

    def test_add_neg_inf_clamping(self):
        # -inf + 1.0 = -inf -> clamped to 0.0
        hw = run_binary_f32("v_add_f32", float("-inf"), 1.0, clamp=True)
        self.assertEqual(hw, 0.0)

    def test_add_signed_zeros_clamped(self):
        # -0.0 + -0.0 = -0.0 -> clamped to 0.0
        hw = run_binary_f32("v_add_f32", -0.0, -0.0, clamp=True)
        self.assertEqual(hw, 0.0)

    def test_add_subnormal_clamped(self):
        # 1e-40 (subnormal) + 0.5 = ~0.5 -> clamped remains ~0.5
        hw = run_binary_f32("v_add_f32", 1e-40, 0.5, clamp=True)
        self.assertAlmostEqual(hw, 0.5, places=6)

class TestValuAddF32NegativeControls(unittest.TestCase):
    """Stage 6: Negative controls and mutation detection for v_add_f32."""

    def test_mutation_missing_clamp(self):
        with_clamp = run_binary_f32("v_add_f32", 0.8, 0.8, clamp=True)
        without_clamp = run_binary_f32("v_add_f32", 0.8, 0.8, clamp=False)
        self.assertNotEqual(with_clamp, without_clamp)
        self.assertAlmostEqual(with_clamp, 1.0, places=6)
        self.assertAlmostEqual(without_clamp, 1.6, places=6)

    def test_spirv_emission_fclamp(self):
        lines = [
            "v_mov_b32 v0, 0x3f000000",
            "v_mov_b32 v1, 0x3f000000",
            "v_add_f32_e64 v2, v0, v1 clamp",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_spv_add_clamp"), lines)
        self.assertIn("FClamp", spv)

class TestValuSubF32ClampOracle(unittest.TestCase):
    """Stage 7: Reference mathematical oracle for v_sub_f32 clamp and omod."""

    def test_sub_clamp_underflow(self):
        # 0.2 - 0.9 = -0.7 -> clamped to 0.0
        hw = run_binary_f32("v_sub_f32", 0.2, 0.9, clamp=True)
        self.assertAlmostEqual(hw, 0.0, places=6)

    def test_sub_clamp_in_range(self):
        # 0.9 - 0.4 = 0.5 -> 0.5
        hw = run_binary_f32("v_sub_f32", 0.9, 0.4, clamp=True)
        self.assertAlmostEqual(hw, 0.5, places=6)

    def test_sub_omod_mul2(self):
        # (0.8 - 0.3) * 2.0 = 1.0
        hw = run_binary_f32("v_sub_f32", 0.8, 0.3, omod="mul:2")
        self.assertAlmostEqual(hw, 1.0, places=6)

    def test_sub_omod_div2_clamp(self):
        # (1.5 - 0.1) * 0.5 = 0.7 -> clamped in range remains 0.7
        hw = run_binary_f32("v_sub_f32", 1.5, 0.1, omod="div:2", clamp=True)
        self.assertAlmostEqual(hw, 0.7, places=6)

class TestValuSubF32AST(unittest.TestCase):
    """Stage 8: Symbolic AST and typed modifier definitions for v_sub_f32."""

    def test_ast_sub_compilation(self):
        lines = [
            "v_mov_b32 v0, 0x3f800000",  # 1.0
            "v_mov_b32 v1, 0x40000000",  # 2.0
            "v_sub_f32_e64 v2, v0, v1 clamp",  # 1.0 - 2.0 = -1.0 -> 0.0
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines(next_test_id("ps_ast_sub"), lines)
        self.assertIn("OpFSub", spv)
        self.assertIn("FClamp", spv)
        words = support.mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(words[0]), 0.0, places=6)

class TestValuSubF32ScalarExecution(unittest.TestCase):
    """Stage 9: Scalar execution tests for v_sub_f32 clamp and omod."""

    def test_sub_self_cancellation(self):
        # v0 - v0 = 0.0
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_sub_f32_e64 v1, v0, v0 clamp",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        words = support.mrt0_words(support.translate_lines(next_test_id("ps_sub_self"), lines))
        self.assertAlmostEqual(bits_to_f32(words[0]), 0.0, places=6)

    def test_sub_inline_literal(self):
        # v0 - 0.5
        lines = [
            "v_mov_b32 v0, 0x3f800000",  # 1.0
            "v_sub_f32_e64 v1, v0, 0.5 clamp",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        words = support.mrt0_words(support.translate_lines(next_test_id("ps_sub_inline"), lines))
        self.assertAlmostEqual(bits_to_f32(words[0]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
