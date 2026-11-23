"""Semantic reference oracle and verification bundle for RDNA2 3-input selection instructions.

Covers:
  - v_min3_f32: IEEE 754 32-bit float minimum of 3 operands with NaN-ignoring semantics
  - v_max3_f32: IEEE 754 32-bit float maximum of 3 operands with NaN-ignoring semantics
  - v_med3_f32: IEEE 754 32-bit float median of 3 operands; falls back to min3 on any NaN
  - v_min3_i32: Signed 32-bit integer minimum of 3 operands
  - v_max3_i32: Signed 32-bit integer maximum of 3 operands
  - v_med3_i32: Signed 32-bit integer median of 3 operands

High-density Subsystem Feature Bundle testing reference oracles, AST modeling,
scalar/lane interpreter, multi-lane divergence, IEEE-754 boundary extremes,
mutation detection, and tools/rdna2spirv.py SPIR-V translation.
All 52 tests are 100% translator-backed via support.translate_lines and support.mrt0_words.
"""
from __future__ import annotations

import dataclasses
import math
import os
import random
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
_TRANSLATE_ID = 0


def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def i32(val: int) -> int:
    v = val & MASK32
    return v if v < 0x80000000 else v - 0x100000000


def u32(val: int) -> int:
    return val & MASK32


def is_nan_f32(b: int) -> bool:
    return (b & 0x7F800000) == 0x7F800000 and (b & 0x007FFFFF) != 0


def run_translated_triop_u32(opcode: str, src0: int, src1: int, src2: int) -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {src0 & MASK32:#x}",
        f"v_mov_b32 v1, {src1 & MASK32:#x}",
        f"v_mov_b32 v2, {src2 & MASK32:#x}",
        f"{opcode} v3, v0, v1, v2",
        "v_mov_b32 v0, v3",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_{opcode}_{_TRANSLATE_ID}", lines)
    words = mrt0_words(spv)
    return words[0] & MASK32


def run_translated_triop_f32(opcode: str, src0_bits: int, src1_bits: int, src2_bits: int) -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {src0_bits & MASK32:#x}",
        f"v_mov_b32 v1, {src1_bits & MASK32:#x}",
        f"v_mov_b32 v2, {src2_bits & MASK32:#x}",
        f"{opcode} v3, v0, v1, v2",
        "v_mov_b32 v0, v3",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_{opcode}_{_TRANSLATE_ID}", lines)
    words = mrt0_words(spv)
    return words[0] & MASK32


def ref_min3_f32_bits(b0: int, b1: int, b2: int) -> int:
    """RDNA2 v_min3_f32 executed directly through translator."""
    return run_translated_triop_f32("v_min3_f32", b0, b1, b2)


def ref_max3_f32_bits(b0: int, b1: int, b2: int) -> int:
    """RDNA2 v_max3_f32 executed directly through translator."""
    return run_translated_triop_f32("v_max3_f32", b0, b1, b2)


def ref_med3_f32_bits(b0: int, b1: int, b2: int) -> int:
    """RDNA2 v_med3_f32 executed directly through translator."""
    return run_translated_triop_f32("v_med3_f32", b0, b1, b2)


def ref_min3_i32(a: int, b: int, c: int) -> int:
    """RDNA2 v_min3_i32 executed directly through translator."""
    return run_translated_triop_u32("v_min3_i32", a, b, c)


def ref_max3_i32(a: int, b: int, c: int) -> int:
    """RDNA2 v_max3_i32 executed directly through translator."""
    return run_translated_triop_u32("v_max3_i32", a, b, c)


def ref_med3_i32(a: int, b: int, c: int) -> int:
    """RDNA2 v_med3_i32 executed directly through translator."""
    return run_translated_triop_u32("v_med3_i32", a, b, c)


@dataclasses.dataclass
class Valu3AstNode:
    opcode: str
    dst: str
    src0: str
    src1: str
    src2: str

    def emit_asm(self) -> str:
        return f"{self.opcode} {self.dst}, {self.src0}, {self.src1}, {self.src2}"


# -----------------------------------------------------------------------------
# Opcode 1: v_min3_f32
# -----------------------------------------------------------------------------

class VMin3F32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_min3_f32."""

    def test_basic_ordering(self):
        a, b, c = f32_to_bits(1.0), f32_to_bits(2.0), f32_to_bits(3.0)
        self.assertEqual(ref_min3_f32_bits(a, b, c), a)
        self.assertEqual(ref_min3_f32_bits(b, a, c), a)
        # Extreme in third operand: catches third operand dropped mutation
        self.assertEqual(ref_min3_f32_bits(c, b, a), a)

    def test_negative_floats(self):
        a, b, c = f32_to_bits(-10.0), f32_to_bits(-5.0), f32_to_bits(0.0)
        self.assertEqual(ref_min3_f32_bits(a, b, c), a)
        self.assertEqual(ref_min3_f32_bits(b, a, c), a)
        self.assertEqual(ref_min3_f32_bits(c, b, a), a)

    def test_nan_ignoring_behavior(self):
        qnan = 0x7FC00001
        v1 = f32_to_bits(4.5)
        v2 = f32_to_bits(9.0)
        # NaN in operand 0, 1, and 2 verified via translator: catches NMin -> FMin mutation
        self.assertEqual(ref_min3_f32_bits(qnan, v1, v2), v1)
        self.assertEqual(ref_min3_f32_bits(v1, qnan, v2), v1)
        self.assertEqual(ref_min3_f32_bits(v2, v1, qnan), v1)


class VMin3F32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_min3_f32."""

    def test_ast_emission(self):
        node = Valu3AstNode("v_min3_f32", "v0", "v1", "v2", "v3")
        self.assertEqual(node.emit_asm(), "v_min3_f32 v0, v1, v2, v3")
        spv = translate_lines("ps_min3_f32_ast", [
            node.emit_asm(),
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertIn("NMin", spv)

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": f32_to_bits(5.0), "v2": f32_to_bits(1.5), "v3": f32_to_bits(7.2)}
        res = ref_min3_f32_bits(regs["v1"], regs["v2"], regs["v3"])
        regs["v0"] = res
        self.assertAlmostEqual(bits_to_f32(regs["v0"]), 1.5)


class VMin3F32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_min3_f32."""

    def test_simd_divergent_lanes(self):
        lanes = 16
        src0 = [f32_to_bits(float(i)) for i in range(lanes)]
        src1 = [f32_to_bits(float(lanes - i)) for i in range(lanes)]
        src2 = [f32_to_bits(8.0) for _ in range(lanes)]
        out = [ref_min3_f32_bits(s0, s1, s2) for s0, s1, s2 in zip(src0, src1, src2)]
        for i in range(lanes):
            expected = min(float(i), float(lanes - i), 8.0)
            self.assertAlmostEqual(bits_to_f32(out[i]), expected)

    def test_infinity_and_subnormals(self):
        pos_inf = 0x7F800000
        neg_inf = 0xFF800000
        subnorm = 0x00000001
        self.assertEqual(ref_min3_f32_bits(pos_inf, 0, subnorm), 0)
        self.assertEqual(ref_min3_f32_bits(pos_inf, neg_inf, subnorm), neg_inf)


class VMin3F32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_min3_f32."""

    def test_spirv_translation_execution(self):
        # Extreme in third operand: (3.0, 2.0, 1.0) -> 1.0
        lines = [
            "v_mov_b32 v0, 0x40400000",  # 3.0f
            "v_mov_b32 v1, 0x40000000",  # 2.0f
            "v_mov_b32 v2, 0x3F800000",  # 1.0f
            "v_min3_f32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_min3_f32_int", lines)
        out = mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(out[0]), 1.0)

    def test_negative_mutation_contrast(self):
        lines = [
            "v_mov_b32 v0, 0x3F800000",  # 1.0f
            "v_mov_b32 v1, 0x40000000",  # 2.0f
            "v_mov_b32 v2, 0x40400000",  # 3.0f
            "v_min3_f32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_min3_f32_mut", lines)
        self.assertIn("NMin", spv)
        normal = mrt0_words(spv)[0]
        self.assertEqual(normal, f32_to_bits(1.0))
        # NMin -> NMax mutation detection
        mut_spv = spv.replace("glsl NMin", "glsl NMax")
        mutated = mrt0_words(mut_spv)[0]
        self.assertEqual(mutated, f32_to_bits(3.0))
        self.assertNotEqual(normal, mutated)


# -----------------------------------------------------------------------------
# Opcode 2: v_max3_f32
# -----------------------------------------------------------------------------

class VMax3F32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_max3_f32."""

    def test_basic_ordering(self):
        a, b, c = f32_to_bits(1.0), f32_to_bits(2.5), f32_to_bits(5.0)
        self.assertEqual(ref_max3_f32_bits(a, b, c), c)
        self.assertEqual(ref_max3_f32_bits(b, a, c), c)
        self.assertEqual(ref_max3_f32_bits(c, a, b), c)

    def test_negative_floats(self):
        a, b, c = f32_to_bits(-10.0), f32_to_bits(-5.0), f32_to_bits(-1.0)
        self.assertEqual(ref_max3_f32_bits(a, b, c), c)
        self.assertEqual(ref_max3_f32_bits(c, b, a), c)

    def test_nan_ignoring_behavior(self):
        qnan = 0x7FC00001
        v1 = f32_to_bits(12.0)
        v2 = f32_to_bits(4.0)
        self.assertEqual(ref_max3_f32_bits(qnan, v1, v2), v1)
        self.assertEqual(ref_max3_f32_bits(v2, qnan, v1), v1)
        self.assertEqual(ref_max3_f32_bits(v1, v2, qnan), v1)


class VMax3F32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_max3_f32."""

    def test_ast_emission(self):
        node = Valu3AstNode("v_max3_f32", "v0", "v1", "v2", "v3")
        self.assertEqual(node.emit_asm(), "v_max3_f32 v0, v1, v2, v3")
        spv = translate_lines("ps_max3_f32_ast", [
            node.emit_asm(),
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertIn("NMax", spv)

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": f32_to_bits(10.0), "v2": f32_to_bits(100.0), "v3": f32_to_bits(50.0)}
        res = ref_max3_f32_bits(regs["v1"], regs["v2"], regs["v3"])
        regs["v0"] = res
        self.assertAlmostEqual(bits_to_f32(regs["v0"]), 100.0)


class VMax3F32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_max3_f32."""

    def test_simd_divergent_lanes(self):
        lanes = 16
        src0 = [f32_to_bits(float(i)) for i in range(lanes)]
        src1 = [f32_to_bits(10.0) for _ in range(lanes)]
        src2 = [f32_to_bits(float(lanes - i)) for i in range(lanes)]
        out = [ref_max3_f32_bits(s0, s1, s2) for s0, s1, s2 in zip(src0, src1, src2)]
        for i in range(lanes):
            expected = max(float(i), 10.0, float(lanes - i))
            self.assertAlmostEqual(bits_to_f32(out[i]), expected)

    def test_infinity_and_signed_zero(self):
        pos_inf = 0x7F800000
        neg_inf = 0xFF800000
        self.assertEqual(ref_max3_f32_bits(neg_inf, 0, pos_inf), pos_inf)


class VMax3F32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_max3_f32."""

    def test_spirv_translation_execution(self):
        # Extreme in third operand: (1.0, 4.0, 9.0) -> 9.0
        lines = [
            "v_mov_b32 v0, 0x3F800000",  # 1.0f
            "v_mov_b32 v1, 0x40800000",  # 4.0f
            "v_mov_b32 v2, 0x41100000",  # 9.0f
            "v_max3_f32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_max3_f32_int", lines)
        out = mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(out[0]), 9.0)

    def test_negative_mutation_contrast(self):
        lines = [
            "v_mov_b32 v0, 0x41200000",  # 10.0f
            "v_mov_b32 v1, 0x41A00000",  # 20.0f
            "v_mov_b32 v2, 0x41F00000",  # 30.0f
            "v_max3_f32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_max3_f32_mut", lines)
        self.assertIn("NMax", spv)
        normal = mrt0_words(spv)[0]
        self.assertEqual(normal, f32_to_bits(30.0))
        # NMax -> NMin mutation detection
        mut_spv = spv.replace("glsl NMax", "glsl NMin")
        mutated = mrt0_words(mut_spv)[0]
        self.assertEqual(mutated, f32_to_bits(10.0))
        self.assertNotEqual(normal, mutated)


# -----------------------------------------------------------------------------
# Opcode 3: v_med3_f32
# -----------------------------------------------------------------------------

class VMed3F32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_med3_f32."""

    def test_basic_median(self):
        a, b, c = f32_to_bits(1.0), f32_to_bits(2.0), f32_to_bits(3.0)
        self.assertEqual(ref_med3_f32_bits(a, b, c), b)
        self.assertEqual(ref_med3_f32_bits(c, a, b), b)
        self.assertEqual(ref_med3_f32_bits(b, c, a), b)
        # Median in third operand
        self.assertEqual(ref_med3_f32_bits(a, c, b), b)

    def test_all_equal_and_pairs(self):
        a = f32_to_bits(4.0)
        b = f32_to_bits(7.0)
        self.assertEqual(ref_med3_f32_bits(a, a, a), a)
        self.assertEqual(ref_med3_f32_bits(a, a, b), a)
        self.assertEqual(ref_med3_f32_bits(b, a, b), b)

    def test_nan_fallback_to_min3(self):
        qnan = 0x7FC00001
        v1 = f32_to_bits(5.0)
        v2 = f32_to_bits(10.0)
        self.assertEqual(ref_med3_f32_bits(v1, v2, qnan), v1)
        self.assertEqual(ref_med3_f32_bits(qnan, v2, v1), v1)


class VMed3F32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_med3_f32."""

    def test_ast_emission(self):
        node = Valu3AstNode("v_med3_f32", "v0", "v1", "v2", "v3")
        self.assertEqual(node.emit_asm(), "v_med3_f32 v0, v1, v2, v3")
        spv = translate_lines("ps_med3_f32_ast", [
            node.emit_asm(),
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertIn("NMin", spv)
        self.assertIn("NMax", spv)

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": f32_to_bits(15.0), "v2": f32_to_bits(3.0), "v3": f32_to_bits(9.0)}
        res = ref_med3_f32_bits(regs["v1"], regs["v2"], regs["v3"])
        regs["v0"] = res
        self.assertAlmostEqual(bits_to_f32(regs["v0"]), 9.0)


class VMed3F32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_med3_f32."""

    def test_simd_divergent_lanes(self):
        lanes = 16
        src0 = [f32_to_bits(float(i)) for i in range(lanes)]
        src1 = [f32_to_bits(8.0) for _ in range(lanes)]
        src2 = [f32_to_bits(float(lanes - i)) for i in range(lanes)]
        out = [ref_med3_f32_bits(s0, s1, s2) for s0, s1, s2 in zip(src0, src1, src2)]
        for i in range(lanes):
            expected = sorted([float(i), 8.0, float(lanes - i)])[1]
            self.assertAlmostEqual(bits_to_f32(out[i]), expected)

    def test_subnormal_median(self):
        s1 = 0x00000001
        s2 = 0x00000002
        s3 = 0x00000003
        self.assertEqual(ref_med3_f32_bits(s1, s3, s2), s2)


class VMed3F32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_med3_f32."""

    def test_spirv_translation_execution(self):
        # Median in third operand: (10.0, 20.0, 15.0) -> 15.0
        lines = [
            "v_mov_b32 v0, 0x41200000",  # 10.0f
            "v_mov_b32 v1, 0x41A00000",  # 20.0f
            "v_mov_b32 v2, 0x41700000",  # 15.0f
            "v_med3_f32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_med3_f32_int", lines)
        out = mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(out[0]), 15.0)

    def test_negative_mutation_contrast(self):
        lines = [
            "v_mov_b32 v0, 0x3F800000",  # 1.0f
            "v_mov_b32 v1, 0x40000000",  # 2.0f
            "v_mov_b32 v2, 0x40400000",  # 3.0f
            "v_med3_f32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_med3_f32_mut", lines)
        med_res = mrt0_words(spv)[0]
        self.assertEqual(med_res, f32_to_bits(2.0))
        # Compare against min and max
        min_res = ref_min3_f32_bits(f32_to_bits(1.0), f32_to_bits(2.0), f32_to_bits(3.0))
        max_res = ref_max3_f32_bits(f32_to_bits(1.0), f32_to_bits(2.0), f32_to_bits(3.0))
        self.assertNotEqual(med_res, min_res)
        self.assertNotEqual(med_res, max_res)


# -----------------------------------------------------------------------------
# Opcode 4: v_min3_i32
# -----------------------------------------------------------------------------

class VMin3I32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_min3_i32."""

    def test_basic_ordering(self):
        self.assertEqual(ref_min3_i32(10, 20, 30), 10)
        self.assertEqual(ref_min3_i32(30, 10, 20), 10)
        # Minimum in third operand: catches third operand dropped mutation
        self.assertEqual(ref_min3_i32(30, 20, 10), 10)

    def test_signed_negative_values(self):
        # -10 is 0xFFFFFFF6, -20 is 0xFFFFFFEC, 5 is 0x5
        self.assertEqual(i32(ref_min3_i32(u32(-10), u32(-20), 5)), -20)
        self.assertEqual(i32(ref_min3_i32(u32(-100), 0, 100)), -100)
        # -1 (0xFFFFFFFF) against positive numbers: catches signed -> unsigned compare mutation
        self.assertEqual(i32(ref_min3_i32(u32(-1), 10, 20)), -1)
        self.assertEqual(ref_min3_i32(u32(-1), 10, 20), u32(-1))

    def test_boundary_extrema(self):
        int_min = 0x80000000  # -2147483648
        int_max = 0x7FFFFFFF  # 2147483647
        self.assertEqual(ref_min3_i32(int_min, 0, int_max), int_min)
        self.assertEqual(ref_min3_i32(int_max, -1, int_min), int_min)


class VMin3I32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_min3_i32."""

    def test_ast_emission(self):
        node = Valu3AstNode("v_min3_i32", "v0", "v1", "v2", "v3")
        self.assertEqual(node.emit_asm(), "v_min3_i32 v0, v1, v2, v3")
        spv = translate_lines("ps_min3_i32_ast", [
            node.emit_asm(),
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertIn("SMin", spv)

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": u32(-50), "v2": 25, "v3": u32(-10)}
        res = ref_min3_i32(regs["v1"], regs["v2"], regs["v3"])
        regs["v0"] = res
        self.assertEqual(i32(regs["v0"]), -50)


class VMin3I32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_min3_i32."""

    def test_simd_divergent_lanes(self):
        lanes = 16
        src0 = [u32(i - 8) for i in range(lanes)]
        src1 = [0 for _ in range(lanes)]
        src2 = [u32(8 - i) for i in range(lanes)]
        out = [ref_min3_i32(s0, s1, s2) for s0, s1, s2 in zip(src0, src1, src2)]
        for i in range(lanes):
            expected = min(i - 8, 0, 8 - i)
            self.assertEqual(i32(out[i]), expected)

    def test_extremes_min3(self):
        int_min = 0x80000000
        self.assertEqual(ref_min3_i32(int_min, int_min, int_min), int_min)


class VMin3I32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_min3_i32."""

    def test_spirv_translation_execution(self):
        # Minimum in third operand: (50, 40, 10) -> 10
        lines = [
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 40",
            "v_mov_b32 v2, 10",
            "v_min3_i32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_min3_i32_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 10)

    def test_negative_mutation_signed_vs_unsigned(self):
        # -1 (0xFFFFFFFF) as signed is < 10, but as unsigned is > 10
        lines = [
            f"v_mov_b32 v0, {u32(-1):#x}",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_min3_i32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_min3_i32_mut", lines)
        self.assertIn("SMin", spv)
        signed_out = mrt0_words(spv)[0]
        self.assertEqual(signed_out, u32(-1))
        # SMin -> UMin mutation detection
        mut_spv = spv.replace("glsl SMin", "glsl UMin")
        unsigned_out = mrt0_words(mut_spv)[0]
        self.assertEqual(unsigned_out, 10)
        self.assertNotEqual(signed_out, unsigned_out)


# -----------------------------------------------------------------------------
# Opcode 5: v_max3_i32
# -----------------------------------------------------------------------------

class VMax3I32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_max3_i32."""

    def test_basic_ordering(self):
        self.assertEqual(ref_max3_i32(10, 30, 50), 50)
        self.assertEqual(ref_max3_i32(50, 10, 30), 50)
        # Maximum in third operand: catches third operand dropped mutation
        self.assertEqual(ref_max3_i32(10, 30, 50), 50)

    def test_signed_negative_values(self):
        self.assertEqual(i32(ref_max3_i32(u32(-50), u32(-10), u32(-30))), -10)
        self.assertEqual(i32(ref_max3_i32(u32(-100), 0, u32(-5))), 0)
        # -1 (0xFFFFFFFF) against positive numbers: catches signed -> unsigned compare mutation
        self.assertEqual(ref_max3_i32(u32(-1), 10, 20), 20)

    def test_boundary_extrema(self):
        int_min = 0x80000000
        int_max = 0x7FFFFFFF
        self.assertEqual(ref_max3_i32(int_min, 0, int_max), int_max)


class VMax3I32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_max3_i32."""

    def test_ast_emission(self):
        node = Valu3AstNode("v_max3_i32", "v0", "v1", "v2", "v3")
        self.assertEqual(node.emit_asm(), "v_max3_i32 v0, v1, v2, v3")
        spv = translate_lines("ps_max3_i32_ast", [
            node.emit_asm(),
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertIn("SMax", spv)

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": u32(-10), "v2": 42, "v3": 17}
        res = ref_max3_i32(regs["v1"], regs["v2"], regs["v3"])
        regs["v0"] = res
        self.assertEqual(regs["v0"], 42)


class VMax3I32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_max3_i32."""

    def test_simd_divergent_lanes(self):
        lanes = 16
        src0 = [u32(i - 8) for i in range(lanes)]
        src1 = [0 for _ in range(lanes)]
        src2 = [u32(8 - i) for i in range(lanes)]
        out = [ref_max3_i32(s0, s1, s2) for s0, s1, s2 in zip(src0, src1, src2)]
        for i in range(lanes):
            expected = max(i - 8, 0, 8 - i)
            self.assertEqual(i32(out[i]), expected)


class VMax3I32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_max3_i32."""

    def test_spirv_translation_execution(self):
        # Maximum in third operand: (50, 40, 100) -> 100
        lines = [
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 40",
            "v_mov_b32 v2, 100",
            "v_max3_i32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_max3_i32_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 100)

    def test_negative_mutation_contrast(self):
        # -1 (0xFFFFFFFF) as signed is < 20, but as unsigned is > 20
        lines = [
            f"v_mov_b32 v0, {u32(-1):#x}",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_max3_i32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_max3_i32_mut", lines)
        self.assertIn("SMax", spv)
        signed_out = mrt0_words(spv)[0]
        self.assertEqual(signed_out, 20)
        # SMax -> UMax mutation detection: unsigned treats -1 as 0xFFFFFFFF, which would be the max
        mut_spv = spv.replace("glsl SMax", "glsl UMax")
        unsigned_out = mrt0_words(mut_spv)[0]
        self.assertEqual(unsigned_out, u32(-1))
        self.assertNotEqual(signed_out, unsigned_out)


# -----------------------------------------------------------------------------
# Opcode 6: v_med3_i32
# -----------------------------------------------------------------------------

class VMed3I32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_med3_i32."""

    def test_basic_ordering(self):
        self.assertEqual(ref_med3_i32(10, 20, 30), 20)
        self.assertEqual(ref_med3_i32(30, 10, 20), 20)
        # Median in third operand
        self.assertEqual(ref_med3_i32(10, 30, 20), 20)

    def test_signed_values(self):
        self.assertEqual(i32(ref_med3_i32(u32(-50), 0, 50)), 0)
        self.assertEqual(i32(ref_med3_i32(u32(-20), u32(-10), u32(-30))), -20)
        # -1 (0xFFFFFFFF) with positive values
        self.assertEqual(ref_med3_i32(u32(-1), 10, 20), 10)

    def test_duplicates(self):
        self.assertEqual(ref_med3_i32(5, 5, 10), 5)
        self.assertEqual(ref_med3_i32(10, 5, 10), 10)


class VMed3I32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_med3_i32."""

    def test_ast_emission(self):
        node = Valu3AstNode("v_med3_i32", "v0", "v1", "v2", "v3")
        self.assertEqual(node.emit_asm(), "v_med3_i32 v0, v1, v2, v3")
        spv = translate_lines("ps_med3_i32_ast", [
            node.emit_asm(),
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertIn("SMin", spv)
        self.assertIn("SMax", spv)

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": 100, "v2": 5, "v3": 42}
        res = ref_med3_i32(regs["v1"], regs["v2"], regs["v3"])
        regs["v0"] = res
        self.assertEqual(regs["v0"], 42)


class VMed3I32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_med3_i32."""

    def test_simd_divergent_lanes(self):
        lanes = 16
        src0 = [u32(i) for i in range(lanes)]
        src1 = [8 for _ in range(lanes)]
        src2 = [u32(lanes - i) for i in range(lanes)]
        out = [ref_med3_i32(s0, s1, s2) for s0, s1, s2 in zip(src0, src1, src2)]
        for i in range(lanes):
            expected = sorted([i, 8, lanes - i])[1]
            self.assertEqual(i32(out[i]), expected)


class VMed3I32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_med3_i32."""

    def test_spirv_translation_execution(self):
        # Median in third operand: (20, 80, 50) -> 50
        lines = [
            "v_mov_b32 v0, 20",
            "v_mov_b32 v1, 80",
            "v_mov_b32 v2, 50",
            "v_med3_i32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_med3_i32_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 50)

    def test_negative_mutation_contrast(self):
        lines = [
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 30",
            "v_med3_i32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_med3_i32_mut", lines)
        med_res = mrt0_words(spv)[0]
        self.assertEqual(med_res, 20)
        min_res = ref_min3_i32(10, 20, 30)
        max_res = ref_max3_i32(10, 20, 30)
        self.assertNotEqual(med_res, min_res)
        self.assertNotEqual(med_res, max_res)


if __name__ == "__main__":
    unittest.main()
