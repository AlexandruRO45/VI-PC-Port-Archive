"""Semantic reference oracle and verification bundle for RDNA2 16-bit integer multiplication and MAD.

Covers:
  - v_mul_lo_u16: 16-bit unsigned integer multiplication (low 16 bits)
  - v_mad_i16: 16-bit signed integer multiply-add: D = S0.i16 * S1.i16 + S2.i16
  - v_mad_u16: 16-bit unsigned integer multiply-add: D = S0.u16 * S1.u16 + S2.u16
  - v_mad_i32_i16: 16-bit signed multiplication with 32-bit signed accumulator
  - v_mad_u32_u16: 16-bit unsigned multiplication with 32-bit unsigned accumulator

High-density Subsystem Feature Bundle testing reference oracles, AST modeling,
scalar/lane interpreter, multi-lane divergence, 16/32-bit sign extension,
mutation detection, and tools/rdna2spirv.py SPIR-V translation.
"""
from __future__ import annotations

import dataclasses
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
MASK16 = 0xFFFF


def u16(val: int) -> int:
    return val & MASK16


def i16(val: int) -> int:
    v = val & MASK16
    return v if v < 0x8000 else v - 0x10000


def u32(val: int) -> int:
    return val & MASK32


def i32(val: int) -> int:
    v = val & MASK32
    return v if v < 0x80000000 else v - 0x100000000


def ref_mul_lo_u16(s0: int, s1: int) -> int:
    return u16(u16(s0) * u16(s1))


def ref_mad_i16(s0: int, s1: int, s2: int) -> int:
    return u16(i16(s0) * i16(s1) + i16(s2))


def ref_mad_u16(s0: int, s1: int, s2: int) -> int:
    return u16(u16(s0) * u16(s1) + u16(s2))


def ref_mad_i32_i16(s0: int, s1: int, s2: int) -> int:
    return u32(i16(s0) * i16(s1) + i32(s2))


def ref_mad_u32_u16(s0: int, s1: int, s2: int) -> int:
    return u32(u16(s0) * u16(s1) + u32(s2))

class VMulLoU16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_mul_lo_u16."""

    def test_basic_multiplication(self):
        self.assertEqual(ref_mul_lo_u16(10, 20), 200)
        self.assertEqual(ref_mul_lo_u16(100, 50), 5000)

    def test_multiplication_overflow_wrapping(self):
        # 1000 * 1000 = 1000000 = 0xF4240 -> low 16 bits: 0x4240 = 16960
        self.assertEqual(ref_mul_lo_u16(1000, 1000), 16960)

    def test_identity_and_zero(self):
        self.assertEqual(ref_mul_lo_u16(1234, 1), 1234)
        self.assertEqual(ref_mul_lo_u16(1234, 0), 0)

@dataclasses.dataclass
class Mul16AstNode:
    opcode: str
    dst: str
    src0: str
    src1: str
    src2: Optional[str] = None

    def emit_asm(self) -> str:
        if self.src2 is not None:
            return f"{self.opcode} {self.dst}, {self.src0}, {self.src1}, {self.src2}"
        return f"{self.opcode} {self.dst}, {self.src0}, {self.src1}"


class VMulLoU16AstTests(unittest.TestCase):
    """AST modeling for v_mul_lo_u16."""

    def test_ast_emission(self):
        node = Mul16AstNode("v_mul_lo_u16", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_mul_lo_u16 v0, v1, v2")

class VMulLoU16InterpreterTests(unittest.TestCase):
    """Interpreter simulation for v_mul_lo_u16."""

    def test_scalar_simulation(self):
        regs = {"v1": 25, "v2": 4}
        res = ref_mul_lo_u16(regs["v1"], regs["v2"])
        regs["v0"] = res
        self.assertEqual(regs["v0"], 100)

class VMulLoU16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_mul_lo_u16."""

    def test_simd_divergent_lanes(self):
        lanes = 64
        src0 = [i for i in range(lanes)]
        src1 = [10 for _ in range(lanes)]
        out = [ref_mul_lo_u16(s0, s1) for s0, s1 in zip(src0, src1)]
        for i in range(lanes):
            self.assertEqual(out[i], i * 10)

    def test_max_u16_boundary(self):
        self.assertEqual(ref_mul_lo_u16(0xFFFF, 0xFFFF), 1)

class VMulLoU16TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_mul_lo_u16."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 15",
            "v_mov_b32 v1, 20",
            "v_mul_lo_u16 v2, v0, v1",
            "v_mov_b32 v0, v2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mul_lo_u16_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0] & 0xFFFF, 300)

    def test_negative_mutation_contrast(self):
        a, b = 10, 20
        res_mul = ref_mul_lo_u16(a, b)
        res_add = u16(a + b)
        self.assertNotEqual(res_mul, res_add)

class VMadI16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_mad_i16."""

    def test_basic_signed_mad(self):
        # 10 * 20 + 5 = 205
        self.assertEqual(ref_mad_i16(10, 20, 5), 205)

    def test_negative_operands(self):
        # -2 * 10 + 5 = -15 = 0xFFF1
        self.assertEqual(ref_mad_i16(u16(-2), 10, 5), 0xFFF1)
        # -5 * -4 + (-10) = 20 - 10 = 10
        self.assertEqual(ref_mad_i16(u16(-5), u16(-4), u16(-10)), 10)

class VMadI16AstTests(unittest.TestCase):
    """AST modeling for v_mad_i16."""

    def test_ast_emission(self):
        node = Mul16AstNode("v_mad_i16", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_mad_i16 v3, v0, v1, v2")

class VMadI16InterpreterTests(unittest.TestCase):
    """Interpreter simulation for v_mad_i16."""

    def test_scalar_simulation(self):
        regs = {"v0": u16(-3), "v1": 10, "v2": 2}
        res = ref_mad_i16(regs["v0"], regs["v1"], regs["v2"])
        regs["v3"] = res
        # -3 * 10 + 2 = -28 = 0xFFE4
        self.assertEqual(regs["v3"], 0xFFE4)

class VMadI16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_mad_i16."""

    def test_simd_divergent_lanes(self):
        lanes = 64
        src0 = [u16(i - 32) for i in range(lanes)]
        src1 = [2 for _ in range(lanes)]
        src2 = [10 for _ in range(lanes)]
        out = [ref_mad_i16(s0, s1, s2) for s0, s1, s2 in zip(src0, src1, src2)]
        for i in range(lanes):
            expected = u16((i - 32) * 2 + 10)
            self.assertEqual(out[i], expected)

class VMadI16TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_mad_i16."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0xFFFE",  # -2 in i16
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 5",
            "v_mad_i16 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mad_i16_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0] & 0xFFFF, 0xFFF1)

    def test_negative_mutation_contrast(self):
        a, b, c = 10, 20, 5
        mad_res = ref_mad_i16(a, b, c)
        add_res = u16(a + b + c)
        self.assertNotEqual(mad_res, add_res)
        self.assertNotEqual(mad_res, ref_mul_lo_u16(a, b))

class VMadU16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_mad_u16."""

    def test_basic_unsigned_mad(self):
        self.assertEqual(ref_mad_u16(10, 20, 5), 205)

    def test_unsigned_wrapping(self):
        # 1000 * 100 + 50 = 100050 = 0x186D2 -> 0x86D2 = 34514
        self.assertEqual(ref_mad_u16(1000, 100, 50), 34514)

class VMadU16AstTests(unittest.TestCase):
    """AST modeling for v_mad_u16."""

    def test_ast_emission(self):
        node = Mul16AstNode("v_mad_u16", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_mad_u16 v3, v0, v1, v2")

class VMadU16InterpreterTests(unittest.TestCase):
    """Interpreter simulation for v_mad_u16."""

    def test_scalar_simulation(self):
        regs = {"v0": 100, "v1": 2, "v2": 33}
        res = ref_mad_u16(regs["v0"], regs["v1"], regs["v2"])
        regs["v3"] = res
        self.assertEqual(regs["v3"], 233)

class VMadU16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_mad_u16."""

    def test_simd_divergent_lanes(self):
        lanes = 64
        src0 = [i for i in range(lanes)]
        src1 = [5 for _ in range(lanes)]
        src2 = [100 for _ in range(lanes)]
        out = [ref_mad_u16(s0, s1, s2) for s0, s1, s2 in zip(src0, src1, src2)]
        for i in range(lanes):
            self.assertEqual(out[i], i * 5 + 100)

class VMadU16TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_mad_u16."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 20",
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 40",
            "v_mad_u16 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mad_u16_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0] & 0xFFFF, 640)

    def test_negative_mutation_contrast(self):
        a, b, c = 10, 10, 10
        mad_res = ref_mad_u16(a, b, c)
        add_res = u16(a + b + c)
        self.assertNotEqual(mad_res, add_res)
        self.assertNotEqual(mad_res, ref_mul_lo_u16(a, b))

class VMadI32I16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_mad_i32_i16."""

    def test_signed_multiplication_32bit_addend(self):
        # -1000 * 2000 + 50 = -1999950
        res = ref_mad_i32_i16(u16(-1000), 2000, 50)
        self.assertEqual(i32(res), -1999950)

    def test_positive_full_range(self):
        # 10000 * 10000 + 100 = 100000100
        res = ref_mad_i32_i16(10000, 10000, 100)
        self.assertEqual(res, 100000100)

class VMadI32I16AstTests(unittest.TestCase):
    """AST modeling for v_mad_i32_i16."""

    def test_ast_emission(self):
        node = Mul16AstNode("v_mad_i32_i16", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_mad_i32_i16 v3, v0, v1, v2")

if __name__ == "__main__":
    unittest.main()
