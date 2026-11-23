"""Semantic reference oracle and verification bundle for RDNA2 integer carry arithmetic.

Covers:
  - v_add_co_u32: 32-bit unsigned addition with carry-out
  - v_sub_co_u32: 32-bit unsigned subtraction with carry-out (borrow)
  - v_subrev_co_u32: 32-bit unsigned reverse subtraction with carry-out (borrow)
  - v_add_co_ci_u32: 32-bit unsigned addition with carry-in and carry-out
  - v_sub_co_ci_u32: 32-bit unsigned subtraction with carry-in and carry-out (borrow)
  - v_subrev_co_ci_u32: 32-bit unsigned reverse subtraction with carry-in and carry-out (borrow)

All test methods compile assembly listings directly through tools/rdna2spirv.py
via support.translate_lines and execute in spirv_mini_interp via support.mrt0_words.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import unittest
from typing import List, Optional, Tuple

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


def execute_co_u32(op: str, x: int, y: int, name: str = "ps_co") -> Tuple[int, int]:
    """Compile and execute a 32-bit carry operation shader, returning (res, carry_out)."""
    lines = [
        f"v_mov_b32 v0, {x:#x}",
        f"v_mov_b32 v1, {y:#x}",
        f"{op} v2, vcc, v0, v1",
        "v_cndmask_b32 v3, 0, 1, vcc",
        "exp mrt0 v2, v3, v0, v1 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(name, lines)
    out = mrt0_words(spv)
    return out[0], out[1]


def execute_co_ci_u32(op: str, x: int, y: int, cin: int, name: str = "ps_co_ci") -> Tuple[int, int]:
    """Compile and execute a 32-bit carry-in/carry-out operation shader, returning (res, carry_out)."""
    cin_setup = "v_cmp_ne_u32 vcc, 1, 0" if cin else "v_cmp_ne_u32 vcc, 0, 0"
    lines = [
        f"v_mov_b32 v0, {x:#x}",
        f"v_mov_b32 v1, {y:#x}",
        cin_setup,
        f"{op} v2, vcc, v0, v1, vcc",
        "v_cndmask_b32 v3, 0, 1, vcc",
        "exp mrt0 v2, v3, v0, v1 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(name, lines)
    out = mrt0_words(spv)
    return out[0], out[1]


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


# =============================================================================
# 1. v_add_co_u32
# =============================================================================

class VAddCoU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_add_co_u32 backed by translator execution."""

    def test_basic_addition_no_carry(self):
        s, c = execute_co_u32("v_add_co_u32", 10, 20, "ps_add_co_basic")
        self.assertEqual((s, c), ref_add_co_u32(10, 20))
        self.assertEqual((s, c), (30, 0))

    def test_addition_with_carry_out(self):
        s, c = execute_co_u32("v_add_co_u32", 0xFFFFFFFF, 1, "ps_add_co_c1")
        self.assertEqual((s, c), (0, 1))

        s2, c2 = execute_co_u32("v_add_co_u32", 0xFFFFFFFF, 0xFFFFFFFF, "ps_add_co_c2")
        self.assertEqual((s2, c2), (0xFFFFFFFE, 1))

    def test_identity_zero(self):
        s, c = execute_co_u32("v_add_co_u32", 0x12345678, 0, "ps_add_co_id")
        self.assertEqual((s, c), (0x12345678, 0))


class VAddCoU32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_add_co_u32."""

    def test_ast_emission(self):
        node = CarryAstNode("v_add_co_u32", "v2", "vcc", "v0", "v1")
        lines = [
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 75",
            node.emit_asm(),
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v2, v3, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_add_co_ast", lines)
        self.assertIn("OpIAddCarry", spv)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (125, 0))

    def test_interpreter_scalar_simulation(self):
        lines = [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_mov_b32 v2, 5",
            "v_add_co_u32 v0, vcc, v1, v2",
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v0, v3, v1, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_add_co_sim", lines)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (4, 1))


class VAddCoU32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_add_co_u32."""

    def test_simd_divergent_lanes(self):
        cases = [(0xFFFFFFFF, 1), (100, 1), (0x7FFFFFFF, 1), (0x80000000, 0x80000000)]
        for idx, (x, y) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y)):
                s, c = execute_co_u32("v_add_co_u32", x, y, f"ps_add_co_div_{idx}")
                self.assertEqual((s, c), ref_add_co_u32(x, y))

    def test_boundary_extrema(self):
        cases = [(0, 0), (0, 0xFFFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF), (0x80000000, 0)]
        for idx, (x, y) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y)):
                s, c = execute_co_u32("v_add_co_u32", x, y, f"ps_add_co_ext_{idx}")
                self.assertEqual((s, c), ref_add_co_u32(x, y))


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
        s, c = execute_co_u32("v_add_co_u32", 10, 20, "ps_add_co_mut_add")
        d, b = execute_co_u32("v_sub_co_u32", 10, 20, "ps_add_co_mut_sub")
        self.assertNotEqual(s, d)
        self.assertNotEqual(c, b)
        spv = translate_lines("ps_add_co_opcode_chk", ["v_add_co_u32 v0, vcc, v1, v2", "s_endpgm"])
        self.assertIn("OpIAddCarry", spv)
        self.assertNotIn("OpISubBorrow", spv)


# =============================================================================
# 2. v_sub_co_u32
# =============================================================================

class VSubCoU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_sub_co_u32 backed by translator execution."""

    def test_basic_subtraction_no_borrow(self):
        d, b = execute_co_u32("v_sub_co_u32", 50, 20, "ps_sub_co_basic")
        self.assertEqual((d, b), (30, 0))
        self.assertEqual((d, b), ref_sub_co_u32(50, 20))

    def test_subtraction_with_borrow(self):
        d, b = execute_co_u32("v_sub_co_u32", 5, 8, "ps_sub_co_b1")
        self.assertEqual((d, b), (0xFFFFFFFD, 1))

    def test_identity_zero(self):
        d, b = execute_co_u32("v_sub_co_u32", 42, 0, "ps_sub_co_id1")
        self.assertEqual((d, b), (42, 0))

        d2, b2 = execute_co_u32("v_sub_co_u32", 42, 42, "ps_sub_co_id2")
        self.assertEqual((d2, b2), (0, 0))


class VSubCoU32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_sub_co_u32."""

    def test_ast_emission(self):
        node = CarryAstNode("v_sub_co_u32", "v2", "vcc", "v0", "v1")
        lines = [
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 40",
            node.emit_asm(),
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v2, v3, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_sub_co_ast", lines)
        self.assertIn("OpISubBorrow", spv)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (60, 0))

    def test_interpreter_scalar_simulation(self):
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 15",
            "v_sub_co_u32 v0, vcc, v1, v2",
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v0, v3, v1, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_sub_co_sim", lines)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (0xFFFFFFFB, 1))


class VSubCoU32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_sub_co_u32."""

    def test_simd_divergent_lanes(self):
        cases = [(10, 32), (32, 32), (50, 32), (0, 1)]
        for idx, (x, y) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y)):
                d, b = execute_co_u32("v_sub_co_u32", x, y, f"ps_sub_co_div_{idx}")
                self.assertEqual((d, b), ref_sub_co_u32(x, y))

    def test_boundary_extrema(self):
        cases = [(0, 0), (0, 0xFFFFFFFF), (0xFFFFFFFF, 0), (0x80000000, 0x7FFFFFFF)]
        for idx, (x, y) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y)):
                d, b = execute_co_u32("v_sub_co_u32", x, y, f"ps_sub_co_ext_{idx}")
                self.assertEqual((d, b), ref_sub_co_u32(x, y))


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
        d_sub, b_sub = execute_co_u32("v_sub_co_u32", 10, 5, "ps_sub_co_mut_sub")
        d_rev, b_rev = execute_co_u32("v_subrev_co_u32", 10, 5, "ps_sub_co_mut_rev")
        self.assertNotEqual(d_sub, d_rev)
        spv = translate_lines("ps_sub_co_opcode_chk", ["v_sub_co_u32 v0, vcc, v1, v2", "s_endpgm"])
        self.assertIn("OpISubBorrow", spv)
        self.assertNotIn("OpIAddCarry", spv)


# =============================================================================
# 3. v_subrev_co_u32
# =============================================================================

class VSubrevCoU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_subrev_co_u32 backed by translator execution."""

    def test_basic_reverse_subtraction(self):
        d, b = execute_co_u32("v_subrev_co_u32", 20, 50, "ps_subrev_co_b1")
        self.assertEqual((d, b), (30, 0))

        d2, b2 = execute_co_u32("v_subrev_co_u32", 50, 20, "ps_subrev_co_b2")
        self.assertEqual((d2, b2), (0xFFFFFFE2, 1))

    def test_asymmetry_against_sub(self):
        d_rev, b_rev = execute_co_u32("v_subrev_co_u32", 100, 40, "ps_subrev_asym1")
        d_sub, b_sub = execute_co_u32("v_sub_co_u32", 40, 100, "ps_subrev_asym2")
        self.assertEqual((d_rev, b_rev), (d_sub, b_sub))
        self.assertEqual((d_rev, b_rev), (0xFFFFFFC4, 1))


class VSubrevCoU32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_subrev_co_u32."""

    def test_ast_emission(self):
        node = CarryAstNode("v_subrev_co_u32", "v2", "vcc", "v0", "v1")
        lines = [
            "v_mov_b32 v0, 30",
            "v_mov_b32 v1, 70",
            node.emit_asm(),
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v2, v3, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_subrev_co_ast", lines)
        self.assertIn("OpISubBorrow", spv)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (40, 0))

    def test_interpreter_scalar_simulation(self):
        lines = [
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 70",
            "v_subrev_co_u32 v0, vcc, v1, v2",
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v0, v3, v1, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_subrev_co_sim", lines)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (40, 0))


class VSubrevCoU32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_subrev_co_u32."""

    def test_simd_divergent_lanes(self):
        cases = [(32, 10), (32, 32), (32, 50), (1, 0)]
        for idx, (x, y) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y)):
                d, b = execute_co_u32("v_subrev_co_u32", x, y, f"ps_subrev_div_{idx}")
                self.assertEqual((d, b), ref_subrev_co_u32(x, y))

    def test_boundary_extrema(self):
        cases = [(0, 0), (0xFFFFFFFF, 0), (0, 0xFFFFFFFF), (0x7FFFFFFF, 0x80000000)]
        for idx, (x, y) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y)):
                d, b = execute_co_u32("v_subrev_co_u32", x, y, f"ps_subrev_ext_{idx}")
                self.assertEqual((d, b), ref_subrev_co_u32(x, y))


class VSubrevCoU32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_subrev_co_u32."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 5",
            "v_mov_b32 v1, 8",
            "v_subrev_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_subrev_co_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 3)
        self.assertEqual(out[1], 0)

    def test_negative_mutation_contrast(self):
        d_sub, _ = execute_co_u32("v_sub_co_u32", 15, 25, "ps_subrev_mut_sub")
        d_rev, _ = execute_co_u32("v_subrev_co_u32", 15, 25, "ps_subrev_mut_rev")
        self.assertNotEqual(d_sub, d_rev)
        self.assertEqual(d_rev, 10)
        self.assertEqual(d_sub, 0xFFFFFFF6)


# =============================================================================
# 4. v_add_co_ci_u32
# =============================================================================

class VAddCoCiU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_add_co_ci_u32 backed by translator execution."""

    def test_addition_with_carry_in_zero(self):
        s, c = execute_co_ci_u32("v_add_co_ci_u32", 10, 20, 0, "ps_add_ci_0")
        self.assertEqual((s, c), (30, 0))
        self.assertEqual((s, c), ref_add_co_ci_u32(10, 20, 0))

    def test_addition_with_carry_in_one(self):
        s, c = execute_co_ci_u32("v_add_co_ci_u32", 10, 20, 1, "ps_add_ci_1")
        self.assertEqual((s, c), (31, 0))
        self.assertEqual((s, c), ref_add_co_ci_u32(10, 20, 1))

    def test_ripple_carry_overflow(self):
        s, c = execute_co_ci_u32("v_add_co_ci_u32", 0xFFFFFFFF, 0, 1, "ps_add_ci_ovf")
        self.assertEqual((s, c), (0, 1))


class VAddCoCiU32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_add_co_ci_u32."""

    def test_ast_emission(self):
        node = CarryAstNode("v_add_co_ci_u32", "v2", "vcc", "v0", "v1", "vcc")
        lines = [
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 200",
            "v_cmp_ne_u32 vcc, 1, 0",
            node.emit_asm(),
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v2, v3, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_add_ci_ast", lines)
        self.assertIn("OpIAddCarry", spv)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (301, 0))

    def test_interpreter_scalar_simulation(self):
        lines = [
            "v_mov_b32 v1, 0x7FFFFFFF",
            "v_mov_b32 v2, 0x7FFFFFFF",
            "v_cmp_ne_u32 vcc, 1, 0",
            "v_add_co_ci_u32 v0, vcc, v1, v2, vcc",
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v0, v3, v1, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_add_ci_sim", lines)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (0xFFFFFFFF, 0))


class VAddCoCiU32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_add_co_ci_u32."""

    def test_simd_divergent_lanes(self):
        cases = [(0xFFFFFFFF, 0, 1), (0xFFFFFFFF, 0, 0), (0xFFFFFFFF, 1, 1), (0x7FFFFFFF, 0x80000000, 1)]
        for idx, (x, y, ci) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y), ci=ci):
                s, c = execute_co_ci_u32("v_add_co_ci_u32", x, y, ci, f"ps_add_ci_div_{idx}")
                self.assertEqual((s, c), ref_add_co_ci_u32(x, y, ci))

    def test_boundary_extrema(self):
        cases = [(0, 0, 0), (0, 0, 1), (0xFFFFFFFF, 0xFFFFFFFF, 1), (0x80000000, 0x7FFFFFFF, 1)]
        for idx, (x, y, ci) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y), ci=ci):
                s, c = execute_co_ci_u32("v_add_co_ci_u32", x, y, ci, f"ps_add_ci_ext_{idx}")
                self.assertEqual((s, c), ref_add_co_ci_u32(x, y, ci))


class VAddCoCiU32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_add_co_ci_u32."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0xFFFFFFFF",
            "v_mov_b32 v1, 0",
            "v_add_co_u32 v2, vcc, v0, 1",
            "v_add_co_ci_u32 v3, vcc, v1, 0, vcc",
            "v_mov_b32 v0, v2",
            "v_mov_b32 v1, v3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_add_co_ci_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 1)

    def test_negative_mutation_contrast(self):
        s0, _ = execute_co_ci_u32("v_add_co_ci_u32", 10, 10, 0, "ps_add_ci_mut_0")
        s1, _ = execute_co_ci_u32("v_add_co_ci_u32", 10, 10, 1, "ps_add_ci_mut_1")
        self.assertNotEqual(s0, s1)
        self.assertEqual(s1 - s0, 1)


# =============================================================================
# 5. v_sub_co_ci_u32
# =============================================================================

class VSubCoCiU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_sub_co_ci_u32 backed by translator execution."""

    def test_subtraction_with_borrow_in_zero(self):
        d, b = execute_co_ci_u32("v_sub_co_ci_u32", 50, 20, 0, "ps_sub_ci_0")
        self.assertEqual((d, b), (30, 0))
        self.assertEqual((d, b), ref_sub_co_ci_u32(50, 20, 0))

    def test_subtraction_with_borrow_in_one(self):
        d, b = execute_co_ci_u32("v_sub_co_ci_u32", 50, 20, 1, "ps_sub_ci_1")
        self.assertEqual((d, b), (29, 0))
        self.assertEqual((d, b), ref_sub_co_ci_u32(50, 20, 1))

    def test_ripple_borrow_chain(self):
        d, b = execute_co_ci_u32("v_sub_co_ci_u32", 0, 0, 1, "ps_sub_ci_rip")
        self.assertEqual((d, b), (0xFFFFFFFF, 1))


class VSubCoCiU32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_sub_co_ci_u32."""

    def test_ast_emission(self):
        node = CarryAstNode("v_sub_co_ci_u32", "v2", "vcc", "v0", "v1", "vcc")
        lines = [
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 30",
            "v_cmp_ne_u32 vcc, 1, 0",
            node.emit_asm(),
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v2, v3, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_sub_ci_ast", lines)
        self.assertIn("OpISubBorrow", spv)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (69, 0))

    def test_interpreter_scalar_simulation(self):
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 5",
            "v_cmp_ne_u32 vcc, 1, 0",
            "v_sub_co_ci_u32 v0, vcc, v1, v2, vcc",
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v0, v3, v1, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_sub_ci_sim", lines)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (4, 0))


class VSubCoCiU32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_sub_co_ci_u32."""

    def test_simd_divergent_lanes(self):
        cases = [(10, 10, 1), (10, 10, 0), (0, 0, 1), (5, 10, 1)]
        for idx, (x, y, ci) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y), ci=ci):
                d, b = execute_co_ci_u32("v_sub_co_ci_u32", x, y, ci, f"ps_sub_ci_div_{idx}")
                self.assertEqual((d, b), ref_sub_co_ci_u32(x, y, ci))

    def test_boundary_extrema(self):
        cases = [(0, 0, 0), (0, 0xFFFFFFFF, 1), (0xFFFFFFFF, 0, 1), (0x80000000, 0x80000000, 1)]
        for idx, (x, y, ci) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y), ci=ci):
                d, b = execute_co_ci_u32("v_sub_co_ci_u32", x, y, ci, f"ps_sub_ci_ext_{idx}")
                self.assertEqual((d, b), ref_sub_co_ci_u32(x, y, ci))


class VSubCoCiU32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_sub_co_ci_u32."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_sub_co_u32 v2, vcc, v0, 1",
            "v_sub_co_ci_u32 v3, vcc, v1, 0, vcc",
            "v_mov_b32 v0, v2",
            "v_mov_b32 v1, v3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_sub_co_ci_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0xFFFFFFFF)
        self.assertEqual(out[1], 0xFFFFFFFF)

    def test_negative_mutation_contrast(self):
        d0, _ = execute_co_ci_u32("v_sub_co_ci_u32", 10, 5, 0, "ps_sub_ci_mut_0")
        d1, _ = execute_co_ci_u32("v_sub_co_ci_u32", 10, 5, 1, "ps_sub_ci_mut_1")
        self.assertNotEqual(d0, d1)
        self.assertEqual(d0 - d1, 1)


# =============================================================================
# 6. v_subrev_co_ci_u32
# =============================================================================

class VSubrevCoCiU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_subrev_co_ci_u32 backed by translator execution."""

    def test_basic_reverse_subtraction_with_borrow_in(self):
        d, b = execute_co_ci_u32("v_subrev_co_ci_u32", 20, 50, 0, "ps_subrev_ci_0")
        self.assertEqual((d, b), (30, 0))
        self.assertEqual((d, b), ref_subrev_co_ci_u32(20, 50, 0))

        d2, b2 = execute_co_ci_u32("v_subrev_co_ci_u32", 20, 50, 1, "ps_subrev_ci_1")
        self.assertEqual((d2, b2), (29, 0))
        self.assertEqual((d2, b2), ref_subrev_co_ci_u32(20, 50, 1))

    def test_borrow_in_overflow(self):
        d, b = execute_co_ci_u32("v_subrev_co_ci_u32", 1, 0, 1, "ps_subrev_ci_ovf")
        self.assertEqual((d, b), (0xFFFFFFFE, 1))


class VSubrevCoCiU32AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_subrev_co_ci_u32."""

    def test_ast_emission(self):
        node = CarryAstNode("v_subrev_co_ci_u32", "v2", "vcc", "v0", "v1", "vcc")
        lines = [
            "v_mov_b32 v0, 20",
            "v_mov_b32 v1, 50",
            "v_cmp_ne_u32 vcc, 1, 0",
            node.emit_asm(),
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v2, v3, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_subrev_ci_ast", lines)
        self.assertIn("OpISubBorrow", spv)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (29, 0))

    def test_interpreter_scalar_simulation(self):
        lines = [
            "v_mov_b32 v1, 15",
            "v_mov_b32 v2, 25",
            "v_cmp_ne_u32 vcc, 1, 0",
            "v_subrev_co_ci_u32 v0, vcc, v1, v2, vcc",
            "v_cndmask_b32 v3, 0, 1, vcc",
            "exp mrt0 v0, v3, v1, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_subrev_ci_sim", lines)
        out = mrt0_words(spv)
        self.assertEqual((out[0], out[1]), (9, 0))


class VSubrevCoCiU32SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_subrev_co_ci_u32."""

    def test_simd_divergent_lanes(self):
        cases = [(10, 20, 1), (10, 20, 0), (20, 10, 1), (0, 0, 1)]
        for idx, (x, y, ci) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y), ci=ci):
                d, b = execute_co_ci_u32("v_subrev_co_ci_u32", x, y, ci, f"ps_subrev_ci_div_{idx}")
                self.assertEqual((d, b), ref_subrev_co_ci_u32(x, y, ci))

    def test_boundary_extrema(self):
        cases = [(0, 0, 0), (0xFFFFFFFF, 0, 1), (0, 0xFFFFFFFF, 1), (0x80000000, 0x80000000, 1)]
        for idx, (x, y, ci) in enumerate(cases):
            with self.subTest(idx=idx, x=hex(x), y=hex(y), ci=ci):
                d, b = execute_co_ci_u32("v_subrev_co_ci_u32", x, y, ci, f"ps_subrev_ci_ext_{idx}")
                self.assertEqual((d, b), ref_subrev_co_ci_u32(x, y, ci))


class VSubrevCoCiU32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_subrev_co_ci_u32."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 1",
            "v_mov_b32 v1, 0",
            "v_sub_co_u32 v2, vcc, v1, 1",
            "v_subrev_co_ci_u32 v3, vcc, v0, 5, vcc",
            "v_mov_b32 v0, v2",
            "v_mov_b32 v1, v3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_subrev_co_ci_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0xFFFFFFFF)
        self.assertEqual(out[1], 3)

    def test_negative_mutation_contrast(self):
        d_sub, _ = execute_co_ci_u32("v_sub_co_ci_u32", 10, 20, 1, "ps_subrev_ci_mut_sub")
        d_subrev, _ = execute_co_ci_u32("v_subrev_co_ci_u32", 10, 20, 1, "ps_subrev_ci_mut_rev")
        self.assertNotEqual(d_sub, d_subrev)
        self.assertEqual(d_subrev, 9)
        self.assertEqual(d_sub, 0xFFFFFFF5)


if __name__ == "__main__":
    unittest.main()
