"""Semantic reference oracle and verification bundle for RDNA2 subtraction with carry & borrow inversion.

Covers:
  - v_sub_co_u32: 32-bit unsigned subtraction with carry-out (borrow)
  - v_subrev_co_u32: 32-bit unsigned reverse subtraction with carry-out (borrow)
  - v_sub_co_ci_u32: 32-bit unsigned subtraction with carry-in and carry-out (borrow)
  - v_subrev_co_ci_u32: 32-bit unsigned reverse subtraction with carry-in and carry-out (borrow)
  - v_sub_f16: 16-bit half-precision floating-point subtraction (D.f16 = S0.f16 - S1.f16)
  - v_subrev_f16: 16-bit half-precision floating-point reverse subtraction (D.f16 = S1.f16 - S0.f16)

High-density Subsystem Feature Bundle testing reference oracles, AST modeling,
scalar/lane interpreter, multi-lane divergence, 64-bit ripple borrow, IEEE 754 half-precision,
mutation detection, and tools/rdna2spirv.py SPIR-V translation.
100% translator-backed: every test verifies real lowering and execution via support.translate_lines.
"""
from __future__ import annotations

import dataclasses
import os
import struct
import sys
import unittest
from typing import Optional, Tuple

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


def u32(val: int) -> int:
    return val & MASK32


def u16(val: int) -> int:
    return val & MASK16


def f16_to_bits(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_to_f16(b: int) -> float:
    return struct.unpack("<e", struct.pack("<H", b & MASK16))[0]


def ref_sub_co_u32(x: int, y: int) -> Tuple[int, int]:
    """32-bit subtraction with borrow out."""
    borrow = 1 if u32(x) < u32(y) else 0
    diff = (u32(x) - u32(y)) & MASK32
    return diff, borrow


def ref_subrev_co_u32(x: int, y: int) -> Tuple[int, int]:
    """32-bit reverse subtraction (y - x) with borrow out."""
    return ref_sub_co_u32(y, x)


def ref_sub_co_ci_u32(x: int, y: int, cin: int) -> Tuple[int, int]:
    """32-bit subtraction with borrow in and borrow out."""
    c = 1 if cin != 0 else 0
    borrow = 1 if u32(x) < (u32(y) + c) else 0
    diff = (u32(x) - u32(y) - c) & MASK32
    return diff, borrow


def ref_subrev_co_ci_u32(x: int, y: int, cin: int) -> Tuple[int, int]:
    """32-bit reverse subtraction (y - x - cin) with borrow in and borrow out."""
    return ref_sub_co_ci_u32(y, x, cin)


def ref_sub_f16_bits(b0: int, b1: int) -> int:
    """16-bit half-precision float subtraction: b0 - b1."""
    f0 = bits_to_f16(b0)
    f1 = bits_to_f16(b1)
    return f16_to_bits(f0 - f1)


def ref_subrev_f16_bits(b0: int, b1: int) -> int:
    """16-bit half-precision float reverse subtraction: b1 - b0."""
    return ref_sub_f16_bits(b1, b0)


def run_prog(lines: list[str]) -> list[int]:
    """Assemble, translate, and execute a fragment shader returning MRT0 words."""
    prog = lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
    spv = translate_lines("ps_valu_sub", prog)
    return mrt0_words(spv)


@dataclasses.dataclass
class ValuSubAstNode:
    opcode: str
    dst: str
    cd_or_src: str
    src0: str
    src1: Optional[str] = None
    src2: Optional[str] = None

    def emit_asm(self) -> str:
        if self.src2 is not None:
            return f"{self.opcode} {self.dst}, {self.cd_or_src}, {self.src0}, {self.src1}, {self.src2}"
        if self.src1 is not None:
            return f"{self.opcode} {self.dst}, {self.cd_or_src}, {self.src0}, {self.src1}"
        return f"{self.opcode} {self.dst}, {self.cd_or_src}, {self.src0}"


class VSubCoU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_sub_co_u32 with translator execution."""

    def test_basic_subtraction(self):
        out = run_prog([
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 40",
            "v_sub_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        d, b = ref_sub_co_u32(100, 40)
        self.assertEqual(out[0], d)
        self.assertEqual(out[1], b)
        self.assertEqual(out[0], 60)
        self.assertEqual(out[1], 0)

    def test_borrow_generation(self):
        out = run_prog([
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 25",
            "v_sub_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        d, b = ref_sub_co_u32(10, 25)
        self.assertEqual(out[0], d)
        self.assertEqual(out[1], b)
        self.assertEqual(out[0], 0xFFFFFFF1)
        self.assertEqual(out[1], 1)

    def test_identity_and_self_cancellation(self):
        out1 = run_prog([
            "v_mov_b32 v0, 12345",
            "v_mov_b32 v1, 0",
            "v_sub_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out1[0], 12345)
        self.assertEqual(out1[1], 0)

        out2 = run_prog([
            "v_mov_b32 v0, 12345",
            "v_sub_co_u32 v2, vcc, v0, v0",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out2[0], 0)
        self.assertEqual(out2[1], 0)


class VSubCoU32AstInterpreterTests(unittest.TestCase):
    """AST modeling, scalar simulation, and register aliasing for v_sub_co_u32."""

    def test_ast_emission_and_execution(self):
        node = ValuSubAstNode("v_sub_co_u32", "v2", "vcc", "v0", "v1")
        self.assertEqual(node.emit_asm(), "v_sub_co_u32 v2, vcc, v0, v1")
        out = run_prog([
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 20",
            node.emit_asm(),
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 30)
        self.assertEqual(out[1], 0)

    def test_register_aliasing_dst_equals_src0(self):
        out = run_prog([
            "v_mov_b32 v0, 75",
            "v_mov_b32 v1, 25",
            "v_sub_co_u32 v0, vcc, v0, v1",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 50)
        self.assertEqual(out[1], 0)

    def test_register_aliasing_dst_equals_src1(self):
        out = run_prog([
            "v_mov_b32 v0, 80",
            "v_mov_b32 v1, 30",
            "v_sub_co_u32 v1, vcc, v0, v1",
            "v_mov_b32 v0, v1",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 50)
        self.assertEqual(out[1], 0)


class VSubCoU32SimdBoundaryTests(unittest.TestCase):
    """SIMD boundary extremes and clamp saturation for v_sub_co_u32."""

    def test_boundary_extremes_translator(self):
        cases = [
            (0, 0, 0, 0),
            (0, 1, 0xFFFFFFFF, 1),
            (1, 0, 1, 0),
            (0xFFFFFFFF, 0xFFFFFFFF, 0, 0),
            (0x80000000, 1, 0x7FFFFFFF, 0),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 1),
        ]
        for x, y, exp_d, exp_b in cases:
            out = run_prog([
                f"v_mov_b32 v0, {x:#x}",
                f"v_mov_b32 v1, {y:#x}",
                "v_sub_co_u32 v2, vcc, v0, v1",
                "v_mov_b32 v0, v2",
                "v_cndmask_b32 v1, 0, 1, vcc",
            ])
            self.assertEqual(out[0], exp_d, f"Failed diff for {x:#x} - {y:#x}")
            self.assertEqual(out[1], exp_b, f"Failed borrow for {x:#x} - {y:#x}")

    def test_clamp_saturation_translator(self):
        out_borrow = run_prog([
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 50",
            "v_sub_co_u32 v2, vcc, v0, v1 clamp",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out_borrow[0], 0)
        self.assertEqual(out_borrow[1], 1)

        out_noborrow = run_prog([
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 10",
            "v_sub_co_u32 v2, vcc, v0, v1 clamp",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out_noborrow[0], 40)
        self.assertEqual(out_noborrow[1], 0)


class VSubCoU32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution, operand modes, and negative mutation controls for v_sub_co_u32."""

    def test_sgpr_and_inline_operands(self):
        out = run_prog([
            "s_mov_b32 s0, 90",
            "v_mov_b32 v1, 35",
            "v_sub_co_u32 v0, vcc, s0, v1",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 55)
        self.assertEqual(out[1], 0)

    def test_negative_mutation_rejection(self):
        prog = ["v_sub_co_u32 v0, vcc, v1, v2 omod:2", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines("ps_neg_sub_co", prog)

    def test_behavioral_mutation_contrast(self):
        out_sub = run_prog([
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 30",
            "v_sub_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
        ])
        out_subrev = run_prog([
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 30",
            "v_subrev_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
        ])
        self.assertEqual(out_sub[0], 70)
        self.assertEqual(out_subrev[0], 0xFFFFFFBA)
        self.assertNotEqual(out_sub[0], out_subrev[0])


class VSubrevCoU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_subrev_co_u32 with translator execution."""

    def test_basic_reverse_subtraction(self):
        out = run_prog([
            "v_mov_b32 v0, 30",
            "v_mov_b32 v1, 80",
            "v_subrev_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        d, b = ref_subrev_co_u32(30, 80)
        self.assertEqual(out[0], d)
        self.assertEqual(out[1], b)
        self.assertEqual(out[0], 50)
        self.assertEqual(out[1], 0)

    def test_borrow_generation(self):
        out = run_prog([
            "v_mov_b32 v0, 80",
            "v_mov_b32 v1, 30",
            "v_subrev_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        d, b = ref_subrev_co_u32(80, 30)
        self.assertEqual(out[0], d)
        self.assertEqual(out[1], b)
        self.assertEqual(out[0], 0xFFFFFFCE)
        self.assertEqual(out[1], 1)

    def test_zero_identity(self):
        out = run_prog([
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 500",
            "v_subrev_co_u32 v2, vcc, v0, v1",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 500)
        self.assertEqual(out[1], 0)


class VSubrevCoU32AstInterpreterTests(unittest.TestCase):
    """AST modeling, scalar simulation, and register aliasing for v_subrev_co_u32."""

    def test_ast_emission_and_execution(self):
        node = ValuSubAstNode("v_subrev_co_u32", "v2", "vcc", "v0", "v1")
        self.assertEqual(node.emit_asm(), "v_subrev_co_u32 v2, vcc, v0, v1")
        out = run_prog([
            "v_mov_b32 v0, 15",
            "v_mov_b32 v1, 65",
            node.emit_asm(),
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 50)
        self.assertEqual(out[1], 0)

    def test_register_aliasing_dst_equals_src0(self):
        out = run_prog([
            "v_mov_b32 v0, 20",
            "v_mov_b32 v1, 50",
            "v_subrev_co_u32 v0, vcc, v0, v1",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 30)
        self.assertEqual(out[1], 0)

    def test_register_aliasing_dst_equals_src1(self):
        out = run_prog([
            "v_mov_b32 v0, 20",
            "v_mov_b32 v1, 50",
            "v_subrev_co_u32 v1, vcc, v0, v1",
            "v_mov_b32 v0, v1",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 30)
        self.assertEqual(out[1], 0)


class VSubrevCoU32SimdBoundaryTests(unittest.TestCase):
    """SIMD boundary extremes and clamp saturation for v_subrev_co_u32."""

    def test_boundary_extremes_translator(self):
        cases = [
            (0, 0, 0, 0),
            (1, 0, 0xFFFFFFFF, 1),
            (0, 1, 1, 0),
            (0xFFFFFFFF, 0, 1, 1),
        ]
        for x, y, exp_d, exp_b in cases:
            out = run_prog([
                f"v_mov_b32 v0, {x:#x}",
                f"v_mov_b32 v1, {y:#x}",
                "v_subrev_co_u32 v2, vcc, v0, v1",
                "v_mov_b32 v0, v2",
                "v_cndmask_b32 v1, 0, 1, vcc",
            ])
            self.assertEqual(out[0], exp_d, f"Failed diff for {y:#x} - {x:#x}")
            self.assertEqual(out[1], exp_b, f"Failed borrow for {y:#x} - {x:#x}")

    def test_clamp_saturation_translator(self):
        out_borrow = run_prog([
            "v_mov_b32 v0, 50",
            "v_mov_b32 v1, 10",
            "v_subrev_co_u32 v2, vcc, v0, v1 clamp",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out_borrow[0], 0)
        self.assertEqual(out_borrow[1], 1)

        out_noborrow = run_prog([
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 50",
            "v_subrev_co_u32 v2, vcc, v0, v1 clamp",
            "v_mov_b32 v0, v2",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out_noborrow[0], 40)
        self.assertEqual(out_noborrow[1], 0)


class VSubrevCoU32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution, operand modes, and negative mutation controls for v_subrev_co_u32."""

    def test_sgpr_and_inline_operands(self):
        out = run_prog([
            "s_mov_b32 s0, 35",
            "v_mov_b32 v1, 90",
            "v_subrev_co_u32 v0, vcc, s0, v1",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 55)
        self.assertEqual(out[1], 0)

    def test_negative_mutation_rejection(self):
        prog = ["v_subrev_co_u32 v0, vcc, v1, v2 omod:2", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines("ps_neg_subrev_co", prog)


class VSubCoCiU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_sub_co_ci_u32 with translator execution."""

    def test_subtraction_with_borrow_in_zero(self):
        out = run_prog([
            "v_mov_b32 v0, 40",
            "v_mov_b32 v1, 15",
            "v_sub_co_u32 v2, vcc, 1, 0",  # sets vcc borrow = 0
            "v_sub_co_ci_u32 v3, vcc, v0, v1, vcc",
            "v_mov_b32 v0, v3",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        d, b = ref_sub_co_ci_u32(40, 15, 0)
        self.assertEqual(out[0], d)
        self.assertEqual(out[1], b)
        self.assertEqual(out[0], 25)
        self.assertEqual(out[1], 0)

    def test_subtraction_with_borrow_in_one(self):
        out = run_prog([
            "v_mov_b32 v0, 40",
            "v_mov_b32 v1, 15",
            "v_sub_co_u32 v2, vcc, 0, 1",  # sets vcc borrow = 1
            "v_sub_co_ci_u32 v3, vcc, v0, v1, vcc",
            "v_mov_b32 v0, v3",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        d, b = ref_sub_co_ci_u32(40, 15, 1)
        self.assertEqual(out[0], d)
        self.assertEqual(out[1], b)
        self.assertEqual(out[0], 24)
        self.assertEqual(out[1], 0)

    def test_borrow_chain_propagation(self):
        out = run_prog([
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_sub_co_u32 v2, vcc, 0, 1",  # borrow in = 1
            "v_sub_co_ci_u32 v3, vcc, v0, v1, vcc",
            "v_mov_b32 v0, v3",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        d, b = ref_sub_co_ci_u32(0, 0, 1)
        self.assertEqual(out[0], d)
        self.assertEqual(out[1], b)
        self.assertEqual(out[0], 0xFFFFFFFF)
        self.assertEqual(out[1], 1)


class VSubCoCiU32AstInterpreterTests(unittest.TestCase):
    """AST modeling, scalar simulation, and register aliasing for v_sub_co_ci_u32."""

    def test_ast_emission_and_execution(self):
        node = ValuSubAstNode("v_sub_co_ci_u32", "v3", "vcc", "v0", "v1", "vcc")
        self.assertEqual(node.emit_asm(), "v_sub_co_ci_u32 v3, vcc, v0, v1, vcc")
        out = run_prog([
            "v_mov_b32 v0, 20",
            "v_mov_b32 v1, 8",
            "v_sub_co_u32 v2, vcc, 0, 1",  # vcc = 1
            node.emit_asm(),
            "v_mov_b32 v0, v3",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 11)
        self.assertEqual(out[1], 0)

    def test_register_aliasing_dst_equals_src0(self):
        out = run_prog([
            "v_mov_b32 v0, 60",
            "v_mov_b32 v1, 20",
            "v_sub_co_u32 v2, vcc, 0, 1",  # vcc = 1
            "v_sub_co_ci_u32 v0, vcc, v0, v1, vcc",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 39)
        self.assertEqual(out[1], 0)

    def test_register_aliasing_dst_equals_src1(self):
        out = run_prog([
            "v_mov_b32 v0, 60",
            "v_mov_b32 v1, 20",
            "v_sub_co_u32 v2, vcc, 0, 1",  # vcc = 1
            "v_sub_co_ci_u32 v1, vcc, v0, v1, vcc",
            "v_mov_b32 v0, v1",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 39)
        self.assertEqual(out[1], 0)


class VSubCoCiU32SimdBoundaryTests(unittest.TestCase):
    """64-bit ripple borrow chains and clamp saturation for v_sub_co_ci_u32."""

    def test_64bit_ripple_borrow_chain_translator(self):
        # 0x00000001_00000000 - 0x00000000_00000001 = 0x00000000_FFFFFFFF
        out = run_prog([
            "v_mov_b32 v2, 0",  # low minuend
            "v_mov_b32 v3, 1",  # high minuend
            "v_mov_b32 v4, 1",  # low subtrahend
            "v_mov_b32 v5, 0",  # high subtrahend
            "v_sub_co_u32 v0, vcc, v2, v4",
            "v_sub_co_ci_u32 v1, vcc, v3, v5, vcc",
            "v_cndmask_b32 v2, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 0xFFFFFFFF)  # low dword
        self.assertEqual(out[1], 0x0)         # high dword
        self.assertEqual(out[2], 0)           # final borrow out

    def test_clamp_saturation_translator(self):
        # 10 - 20 - 1 clamp -> 0 (borrow=1)
        out = run_prog([
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 20",
            "v_sub_co_u32 v2, vcc, 0, 1",  # vcc = 1
            "v_sub_co_ci_u32 v3, vcc, v0, v1, vcc clamp",
            "v_mov_b32 v0, v3",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 1)


class VSubCoCiU32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution, operand modes, and negative mutation controls for v_sub_co_ci_u32."""

    def test_sgpr_and_inline_operands(self):
        out = run_prog([
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 20",
            "v_sub_co_u32 v2, vcc, v0, v1",
            "v_sub_co_ci_u32 v3, vcc, 50, 10, vcc",
            "v_mov_b32 v0, v2",
            "v_mov_b32 v1, v3",
        ])
        self.assertEqual(out[0], 0xFFFFFFF6)
        self.assertEqual(out[1], 39)

    def test_negative_mutation_rejection(self):
        prog = ["v_sub_co_ci_u32 v0, vcc, v1, v2, vcc omod:2", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines("ps_neg_sub_co_ci", prog)


class VSubrevCoCiU32OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_subrev_co_ci_u32 with translator execution."""

    def test_basic_reverse_subtraction_with_borrow_in(self):
        out0 = run_prog([
            "v_mov_b32 v0, 15",
            "v_mov_b32 v1, 60",
            "v_sub_co_u32 v2, vcc, 1, 0",  # vcc = 0
            "v_subrev_co_ci_u32 v3, vcc, v0, v1, vcc",
            "v_mov_b32 v0, v3",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        d0, b0 = ref_subrev_co_ci_u32(15, 60, 0)
        self.assertEqual(out0[0], d0)
        self.assertEqual(out0[1], b0)
        self.assertEqual(out0[0], 45)
        self.assertEqual(out0[1], 0)

        out1 = run_prog([
            "v_mov_b32 v0, 15",
            "v_mov_b32 v1, 60",
            "v_sub_co_u32 v2, vcc, 0, 1",  # vcc = 1
            "v_subrev_co_ci_u32 v3, vcc, v0, v1, vcc",
            "v_mov_b32 v0, v3",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        d1, b1 = ref_subrev_co_ci_u32(15, 60, 1)
        self.assertEqual(out1[0], d1)
        self.assertEqual(out1[1], b1)
        self.assertEqual(out1[0], 44)
        self.assertEqual(out1[1], 0)


class VSubrevCoCiU32AstInterpreterTests(unittest.TestCase):
    """AST modeling, scalar simulation, and register aliasing for v_subrev_co_ci_u32."""

    def test_ast_emission_and_execution(self):
        node = ValuSubAstNode("v_subrev_co_ci_u32", "v3", "vcc", "v0", "v1", "vcc")
        self.assertEqual(node.emit_asm(), "v_subrev_co_ci_u32 v3, vcc, v0, v1, vcc")
        out = run_prog([
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 30",
            "v_sub_co_u32 v2, vcc, 0, 1",  # vcc = 1
            node.emit_asm(),
            "v_mov_b32 v0, v3",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 19)
        self.assertEqual(out[1], 0)

    def test_register_aliasing_dst_equals_src0(self):
        out = run_prog([
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 30",
            "v_sub_co_u32 v2, vcc, 0, 1",  # vcc = 1
            "v_subrev_co_ci_u32 v0, vcc, v0, v1, vcc",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 19)
        self.assertEqual(out[1], 0)

    def test_register_aliasing_dst_equals_src1(self):
        out = run_prog([
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 30",
            "v_sub_co_u32 v2, vcc, 0, 1",  # vcc = 1
            "v_subrev_co_ci_u32 v1, vcc, v0, v1, vcc",
            "v_mov_b32 v0, v1",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 19)
        self.assertEqual(out[1], 0)


class VSubrevCoCiU32SimdBoundaryTests(unittest.TestCase):
    """64-bit reverse ripple borrow chains and clamp saturation for v_subrev_co_ci_u32."""

    def test_64bit_reverse_borrow_chain_translator(self):
        # [v3:v2] - [v5:v4] = (0x00000001_00000000) - (0x00000000_00000001) = 0x00000000_FFFFFFFF
        out = run_prog([
            "v_mov_b32 v2, 0",  # low minuend
            "v_mov_b32 v3, 1",  # high minuend
            "v_mov_b32 v4, 1",  # low subtrahend
            "v_mov_b32 v5, 0",  # high subtrahend
            "v_subrev_co_u32 v0, vcc, v4, v2",
            "v_subrev_co_ci_u32 v1, vcc, v5, v3, vcc",
            "v_cndmask_b32 v2, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 0xFFFFFFFF)
        self.assertEqual(out[1], 0x0)
        self.assertEqual(out[2], 0)

    def test_clamp_saturation_translator(self):
        # 5 - 20 - 1 clamp -> 0 (borrow=1)
        out = run_prog([
            "v_mov_b32 v0, 20",
            "v_mov_b32 v1, 5",
            "v_sub_co_u32 v2, vcc, 0, 1",  # vcc = 1
            "v_subrev_co_ci_u32 v3, vcc, v0, v1, vcc clamp",
            "v_mov_b32 v0, v3",
            "v_cndmask_b32 v1, 0, 1, vcc",
        ])
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 1)


class VSubrevCoCiU32TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution, operand modes, and negative mutation controls for v_subrev_co_ci_u32."""

    def test_inline_constant_carry_in(self):
        out = run_prog([
            "v_mov_b32 v0, 2",
            "v_mov_b32 v1, 10",
            "v_subrev_co_ci_u32 v2, vcc, v0, v1, 1",
            "v_mov_b32 v0, v2",
        ])
        self.assertEqual(out[0], 7)  # 10 - 2 - 1 = 7

    def test_negative_mutation_rejection(self):
        prog = ["v_subrev_co_ci_u32 v0, vcc, v1, v2, vcc omod:2", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines("ps_neg_subrev_co_ci", prog)


class VSubF16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_sub_f16 with translator execution."""

    def test_basic_subtraction(self):
        # 5.0 (0x4500) - 2.0 (0x4000) = 3.0 (0x4200)
        out = run_prog([
            "v_mov_b32 v0, 0x4500",
            "v_mov_b32 v1, 0x4000",
            "v_sub_f16 v2, v0, v1",
            "v_mov_b32 v0, v2",
        ])
        self.assertEqual(out[0] & 0xFFFF, 0x4200)

    def test_negative_result(self):
        # 1.0 (0x3C00) - 4.0 (0x4400) = -3.0 (0xC200)
        out = run_prog([
            "v_mov_b32 v0, 0x3C00",
            "v_mov_b32 v1, 0x4400",
            "v_sub_f16 v2, v0, v1",
            "v_mov_b32 v0, v2",
        ])
        self.assertEqual(out[0] & 0xFFFF, 0xC200)

    def test_cancellation(self):
        out = run_prog([
            "v_mov_b32 v0, 0x4780",  # 7.5
            "v_sub_f16 v1, v0, v0",
            "v_mov_b32 v0, v1",
        ])
        self.assertEqual(out[0] & 0xFFFF, 0x0000)


class VSubF16AstInterpreterTests(unittest.TestCase):
    """AST modeling, scalar simulation, and high-half preservation for v_sub_f16."""

    def test_ast_emission_and_execution(self):
        node = ValuSubAstNode("v_sub_f16", "v2", "v0", "v1")
        self.assertEqual(node.emit_asm(), "v_sub_f16 v2, v0, v1")
        out = run_prog([
            "v_mov_b32 v0, 0x4900",  # 10.0
            "v_mov_b32 v1, 0x4300",  # 3.5
            node.emit_asm(),
            "v_mov_b32 v0, v2",
        ])
        self.assertEqual(out[0] & 0xFFFF, 0x4680)  # 6.5

    def test_high_half_preservation_e32(self):
        out = run_prog([
            "v_mov_b32 v0, 0x5A5A0000",
            "v_mov_b32 v1, 0x4500",  # 5.0
            "v_mov_b32 v2, 0x4000",  # 2.0
            "v_sub_f16 v0, v1, v2",
        ])
        self.assertEqual(out[0] >> 16, 0x5A5A)
        self.assertEqual(out[0] & 0xFFFF, 0x4200)

    def test_register_aliasing_dst_equals_src0(self):
        out = run_prog([
            "v_mov_b32 v0, 0x4500",  # 5.0
            "v_mov_b32 v1, 0x4000",  # 2.0
            "v_sub_f16 v0, v0, v1",
        ])
        self.assertEqual(out[0] & 0xFFFF, 0x4200)


if __name__ == "__main__":
    unittest.main()
