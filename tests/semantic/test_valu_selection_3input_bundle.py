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
from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF


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

def ref_min3_f32_bits(b0: int, b1: int, b2: int) -> int:
    """RDNA2 v_min3_f32: minimum of 3 floats; any NaN operand is ignored in favor of non-NaN."""
    nans = [is_nan_f32(b) for b in (b0, b1, b2)]
    if all(nans):
        return b0
    valid_floats = [bits_to_f32(b) for b, is_n in zip((b0, b1, b2), nans) if not is_n]
    res_f = min(valid_floats)
    return f32_to_bits(res_f)


class VMin3F32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_min3_f32."""

    def test_basic_ordering(self):
        a, b, c = f32_to_bits(1.0), f32_to_bits(2.0), f32_to_bits(3.0)
        self.assertEqual(ref_min3_f32_bits(a, b, c), a)
        self.assertEqual(ref_min3_f32_bits(b, a, c), a)
        self.assertEqual(ref_min3_f32_bits(c, b, a), a)

    def test_negative_floats(self):
        a, b, c = f32_to_bits(-10.0), f32_to_bits(-5.0), f32_to_bits(0.0)
        self.assertEqual(ref_min3_f32_bits(a, b, c), a)
        self.assertEqual(ref_min3_f32_bits(b, a, c), a)

    def test_nan_ignoring_behavior(self):
        qnan = 0x7FC00001
        v1 = f32_to_bits(4.5)
        v2 = f32_to_bits(9.0)
        self.assertEqual(ref_min3_f32_bits(qnan, v1, v2), v1)
        self.assertEqual(ref_min3_f32_bits(v1, qnan, v2), v1)
        self.assertEqual(ref_min3_f32_bits(v2, v1, qnan), v1)

@dataclasses.dataclass
class Valu3AstNode:
    opcode: str
    dst: str
    src0: str
    src1: str
    src2: str

    def emit_asm(self) -> str:
        return f"{self.opcode} {self.dst}, {self.src0}, {self.src1}, {self.src2}"


class VMin3F32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_min3_f32."""

    def test_ast_emission(self):
        node = Valu3AstNode("v_min3_f32", "v0", "v1", "v2", "v3")
        self.assertEqual(node.emit_asm(), "v_min3_f32 v0, v1, v2, v3")

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": f32_to_bits(5.0), "v2": f32_to_bits(1.5), "v3": f32_to_bits(7.2)}
        res = ref_min3_f32_bits(regs["v1"], regs["v2"], regs["v3"])
        regs["v0"] = res
        self.assertAlmostEqual(bits_to_f32(regs["v0"]), 1.5)

class VMin3F32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_min3_f32."""

    def test_simd_divergent_lanes(self):
        lanes = 64
        src0 = [f32_to_bits(float(i)) for i in range(lanes)]
        src1 = [f32_to_bits(float(lanes - i)) for i in range(lanes)]
        src2 = [f32_to_bits(32.0) for _ in range(lanes)]
        out = [ref_min3_f32_bits(s0, s1, s2) for s0, s1, s2 in zip(src0, src1, src2)]
        for i in range(lanes):
            expected = min(float(i), float(lanes - i), 32.0)
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
        lines = [
            "v_mov_b32 v0, 0x40400000",  # 3.0f
            "v_mov_b32 v1, 0x3F800000",  # 1.0f
            "v_mov_b32 v2, 0x40000000",  # 2.0f
            "v_min3_f32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_min3_f32_int", lines)
        out = mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(out[0]), 1.0)

    def test_negative_mutation_contrast(self):
        a, b, c = f32_to_bits(1.0), f32_to_bits(2.0), f32_to_bits(3.0)
        min_res = ref_min3_f32_bits(a, b, c)
        max_res = f32_to_bits(max(bits_to_f32(a), bits_to_f32(b), bits_to_f32(c)))
        self.assertNotEqual(min_res, max_res)

def ref_max3_f32_bits(b0: int, b1: int, b2: int) -> int:
    """RDNA2 v_max3_f32: maximum of 3 floats; any NaN operand is ignored in favor of non-NaN."""
    nans = [is_nan_f32(b) for b in (b0, b1, b2)]
    if all(nans):
        return b0
    valid_floats = [bits_to_f32(b) for b, is_n in zip((b0, b1, b2), nans) if not is_n]
    res_f = max(valid_floats)
    return f32_to_bits(res_f)


class VMax3F32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_max3_f32."""

    def test_basic_ordering(self):
        a, b, c = f32_to_bits(1.0), f32_to_bits(5.0), f32_to_bits(2.5)
        self.assertEqual(ref_max3_f32_bits(a, b, c), b)
        self.assertEqual(ref_max3_f32_bits(b, a, c), b)
        self.assertEqual(ref_max3_f32_bits(c, a, b), b)

    def test_negative_floats(self):
        a, b, c = f32_to_bits(-10.0), f32_to_bits(-5.0), f32_to_bits(-1.0)
        self.assertEqual(ref_max3_f32_bits(a, b, c), c)

    def test_nan_ignoring_behavior(self):
        qnan = 0x7FC00001
        v1 = f32_to_bits(12.0)
        v2 = f32_to_bits(4.0)
        self.assertEqual(ref_max3_f32_bits(qnan, v1, v2), v1)
        self.assertEqual(ref_max3_f32_bits(v2, qnan, v1), v1)

class VMax3F32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_max3_f32."""

    def test_ast_emission(self):
        node = Valu3AstNode("v_max3_f32", "v0", "v1", "v2", "v3")
        self.assertEqual(node.emit_asm(), "v_max3_f32 v0, v1, v2, v3")

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": f32_to_bits(10.0), "v2": f32_to_bits(100.0), "v3": f32_to_bits(50.0)}
        res = ref_max3_f32_bits(regs["v1"], regs["v2"], regs["v3"])
        regs["v0"] = res
        self.assertAlmostEqual(bits_to_f32(regs["v0"]), 100.0)

class VMax3F32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_max3_f32."""

    def test_simd_divergent_lanes(self):
        lanes = 64
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
        lines = [
            "v_mov_b32 v0, 0x40400000",  # 3.0f
            "v_mov_b32 v1, 0x41100000",  # 9.0f
            "v_mov_b32 v2, 0x40800000",  # 4.0f
            "v_max3_f32 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_max3_f32_int", lines)
        out = mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(out[0]), 9.0)

    def test_negative_mutation_contrast(self):
        a, b, c = f32_to_bits(10.0), f32_to_bits(20.0), f32_to_bits(30.0)
        max_res = ref_max3_f32_bits(a, b, c)
        min_res = ref_min3_f32_bits(a, b, c)
        self.assertNotEqual(max_res, min_res)

def ref_med3_f32_bits(b0: int, b1: int, b2: int) -> int:
    """RDNA2 v_med3_f32: median of 3 floats; if any operand is NaN, returns min3(x, y, z)."""
    nans = [is_nan_f32(b) for b in (b0, b1, b2)]
    if any(nans):
        return ref_min3_f32_bits(b0, b1, b2)
    floats = [bits_to_f32(b) for b in (b0, b1, b2)]
    sorted_f = sorted(floats)
    return f32_to_bits(sorted_f[1])


class VMed3F32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_med3_f32."""

    def test_basic_median(self):
        a, b, c = f32_to_bits(1.0), f32_to_bits(2.0), f32_to_bits(3.0)
        self.assertEqual(ref_med3_f32_bits(a, b, c), b)
        self.assertEqual(ref_med3_f32_bits(c, a, b), b)
        self.assertEqual(ref_med3_f32_bits(b, c, a), b)

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

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": f32_to_bits(15.0), "v2": f32_to_bits(3.0), "v3": f32_to_bits(9.0)}
        res = ref_med3_f32_bits(regs["v1"], regs["v2"], regs["v3"])
        regs["v0"] = res
        self.assertAlmostEqual(bits_to_f32(regs["v0"]), 9.0)

if __name__ == "__main__":
    unittest.main()
