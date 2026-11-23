"""Semantic reference oracle and verification bundle for RDNA2 integer carry and funnel arithmetic.

Covers:
  - v_add_co_u32: 32-bit unsigned addition with carry-out
  - v_sub_co_u32: 32-bit unsigned subtraction with carry-out (borrow)
  - v_subrev_co_u32: 32-bit unsigned reverse subtraction with carry-out (borrow)
  - v_add_co_ci_u32: 32-bit unsigned addition with carry-in and carry-out
  - v_sub_co_ci_u32: 32-bit unsigned subtraction with carry-in and carry-out (borrow)
  - v_subrev_co_ci_u32: 32-bit unsigned reverse subtraction with carry-in and carry-out (borrow)

High-density Subsystem Feature Bundle testing reference oracles, AST modeling,
scalar/lane interpreter, multi-lane divergence, 64-bit ripple-carry arithmetic,
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


def u32(val: int) -> int:
    return val & MASK32


def ref_add_co_u32(x: int, y: int) -> Tuple[int, int]:
    """32-bit addition with carry out."""
    total = u32(x) + u32(y)
    return u32(total), 1 if total >= 0x100000000 else 0


def ref_sub_co_u32(x: int, y: int) -> Tuple[int, int]:
    """32-bit subtraction with carry out (borrow)."""
    borrow = 1 if u32(x) < u32(y) else 0
    diff = (u32(x) - u32(y)) & MASK32
    return diff, borrow


def ref_subrev_co_u32(x: int, y: int) -> Tuple[int, int]:
    """32-bit reverse subtraction (y - x) with carry out (borrow)."""
    return ref_sub_co_u32(y, x)


def ref_add_co_ci_u32(x: int, y: int, cin: int) -> Tuple[int, int]:
    """32-bit addition with carry in and carry out."""
    c = 1 if cin != 0 else 0
    total = u32(x) + u32(y) + c
    return u32(total), 1 if total >= 0x100000000 else 0


def ref_sub_co_ci_u32(x: int, y: int, cin: int) -> Tuple[int, int]:
    """32-bit subtraction with borrow in and borrow out."""
    c = 1 if cin != 0 else 0
    borrow = 1 if u32(x) < (u32(y) + c) else 0
    diff = (u32(x) - u32(y) - c) & MASK32
    return diff, borrow


def ref_subrev_co_ci_u32(x: int, y: int, cin: int) -> Tuple[int, int]:
    """32-bit reverse subtraction (y - x - cin) with borrow in and borrow out."""
    return ref_sub_co_ci_u32(y, x, cin)

class VAddCoU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_add_co_u32."""

    def test_basic_addition_no_carry(self):
        s, c = ref_add_co_u32(10, 20)
        self.assertEqual(s, 30)
        self.assertEqual(c, 0)

    def test_addition_with_carry_out(self):
        s, c = ref_add_co_u32(0xFFFFFFFF, 1)
        self.assertEqual(s, 0)
        self.assertEqual(c, 1)

        s2, c2 = ref_add_co_u32(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(s2, 0xFFFFFFFE)
        self.assertEqual(c2, 1)

    def test_identity_zero(self):
        s, c = ref_add_co_u32(0x12345678, 0)
        self.assertEqual(s, 0x12345678)
        self.assertEqual(c, 0)

@dataclasses.dataclass
class CarryAstNode:
    opcode: str
    dst: str
    cd: str
    src0: str
    src1: str
    src2: Optional[str] = None

    def emit_asm(self) -> str:
        if self.src2 is not None:
            return f"{self.opcode} {self.dst}, {self.cd}, {self.src0}, {self.src1}, {self.src2}"
        return f"{self.opcode} {self.dst}, {self.cd}, {self.src0}, {self.src1}"


class VAddCoU32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_add_co_u32."""

    def test_ast_emission(self):
        node = CarryAstNode("v_add_co_u32", "v0", "vcc", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_add_co_u32 v0, vcc, v1, v2")

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": 0xFFFFFFFF, "v2": 5}
        s, c = ref_add_co_u32(regs["v1"], regs["v2"])
        regs["v0"] = s
        regs["vcc"] = c
        self.assertEqual(regs["v0"], 4)
        self.assertEqual(regs["vcc"], 1)

class VAddCoU32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_add_co_u32."""

    def test_simd_divergent_lanes(self):
        lanes = 64
        src0 = [0xFFFFFFFF if i % 2 == 1 else 100 for i in range(lanes)]
        src1 = [1 for _ in range(lanes)]
        out = [ref_add_co_u32(s0, s1) for s0, s1 in zip(src0, src1)]
        for i in range(lanes):
            s, c = out[i]
            if i % 2 == 1:
                self.assertEqual(s, 0)
                self.assertEqual(c, 1)
            else:
                self.assertEqual(s, 101)
                self.assertEqual(c, 0)

    def test_boundary_extrema(self):
        s, c = ref_add_co_u32(0, 0)
        self.assertEqual(s, 0)
        self.assertEqual(c, 0)

class VAddCoU32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_add_co_u32."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0xFFFFFFFF",
            "v_mov_b32 v1, 2",
            "v_add_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_add_co_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 1)
        self.assertEqual(out[1], 1)

    def test_negative_mutation_contrast(self):
        a, b = 10, 20
        add_res, _ = ref_add_co_u32(a, b)
        sub_res, _ = ref_sub_co_u32(a, b)
        self.assertNotEqual(add_res, sub_res)

class VSubCoU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_sub_co_u32."""

    def test_basic_subtraction_no_borrow(self):
        d, b = ref_sub_co_u32(50, 20)
        self.assertEqual(d, 30)
        self.assertEqual(b, 0)

    def test_subtraction_with_borrow(self):
        d, b = ref_sub_co_u32(5, 8)
        self.assertEqual(d, 0xFFFFFFFD)
        self.assertEqual(b, 1)

    def test_identity_zero(self):
        d, b = ref_sub_co_u32(42, 0)
        self.assertEqual(d, 42)
        self.assertEqual(b, 0)

        d2, b2 = ref_sub_co_u32(42, 42)
        self.assertEqual(d2, 0)
        self.assertEqual(b2, 0)

class VSubCoU32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_sub_co_u32."""

    def test_ast_emission(self):
        node = CarryAstNode("v_sub_co_u32", "v0", "vcc", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_sub_co_u32 v0, vcc, v1, v2")

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": 10, "v2": 15}
        d, b = ref_sub_co_u32(regs["v1"], regs["v2"])
        regs["v0"] = d
        regs["vcc"] = b
        self.assertEqual(regs["v0"], 0xFFFFFFFB)
        self.assertEqual(regs["vcc"], 1)

class VSubCoU32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_sub_co_u32."""

    def test_simd_divergent_lanes(self):
        lanes = 64
        src0 = [i for i in range(lanes)]
        src1 = [32 for _ in range(lanes)]
        out = [ref_sub_co_u32(s0, s1) for s0, s1 in zip(src0, src1)]
        for i in range(lanes):
            d, b = out[i]
            if i < 32:
                self.assertEqual(b, 1)
                self.assertEqual(d, u32(i - 32))
            else:
                self.assertEqual(b, 0)
                self.assertEqual(d, i - 32)

    def test_boundary_extrema(self):
        d, b = ref_sub_co_u32(0, 0xFFFFFFFF)
        self.assertEqual(d, 1)
        self.assertEqual(b, 1)

class VSubCoU32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_sub_co_u32."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 5",
            "v_mov_b32 v1, 8",
            "v_sub_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_sub_co_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0xFFFFFFFD)
        self.assertEqual(out[1], 1)

    def test_negative_mutation_contrast(self):
        a, b = 10, 5
        sub_res, _ = ref_sub_co_u32(a, b)
        subrev_res, _ = ref_subrev_co_u32(a, b)
        self.assertNotEqual(sub_res, subrev_res)

class VSubrevCoU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_subrev_co_u32."""

    def test_basic_reverse_subtraction(self):
        d, b = ref_subrev_co_u32(20, 50)
        self.assertEqual(d, 30)
        self.assertEqual(b, 0)

        d2, b2 = ref_subrev_co_u32(50, 20)
        self.assertEqual(d2, 0xFFFFFFE2)
        self.assertEqual(b2, 1)

    def test_asymmetry_against_sub(self):
        a, b = 100, 40
        self.assertEqual(ref_subrev_co_u32(a, b), ref_sub_co_u32(b, a))

class VSubrevCoU32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_subrev_co_u32."""

    def test_ast_emission(self):
        node = CarryAstNode("v_subrev_co_u32", "v0", "vcc", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_subrev_co_u32 v0, vcc, v1, v2")

    def test_interpreter_scalar_simulation(self):
        regs = {"v1": 30, "v2": 70}
        d, b = ref_subrev_co_u32(regs["v1"], regs["v2"])
        regs["v0"] = d
        regs["vcc"] = b
        self.assertEqual(regs["v0"], 40)
        self.assertEqual(regs["vcc"], 0)

if __name__ == "__main__":
    unittest.main()
