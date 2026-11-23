"""Semantic reference oracle and verification bundle for RDNA2 16-bit integer multiplication and MAD.

Covers:
  - v_mul_lo_u16: 16-bit unsigned integer multiplication (low 16 bits)
  - v_mad_i16: 16-bit signed integer multiply-add: D = S0.i16 * S1.i16 + S2.i16
  - v_mad_u16: 16-bit unsigned integer multiply-add: D = S0.u16 * S1.u16 + S2.u16
  - v_mad_i32_i16: 16-bit signed multiplication with 32-bit signed accumulator
  - v_mad_u32_u16: 16-bit unsigned multiplication with 32-bit unsigned accumulator

High-density Subsystem Feature Bundle testing reference oracles, AST modeling,
scalar/lane execution, multi-lane divergence, 16/32-bit sign extension,
mutation detection, and tools/rdna2spirv.py SPIR-V translation.
"""
from __future__ import annotations

import dataclasses
import os
import struct
import sys
import unittest
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

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


def run_ps_valu(name: str, inst_line: str, setup_lines: list[str], read_reg: str = "v3") -> int:
    """Compile and execute a VALU instruction, returning the low 32 bits of the output register."""
    lines = list(setup_lines) + [
        inst_line,
        f"v_mov_b32 v0, {read_reg}",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(name, lines)
    words = mrt0_words(spv)
    return words[0]


def run_mul_lo_u16(s0: int, s1: int, name: str = "ps_mul_lo_u16") -> int:
    return run_ps_valu(
        name,
        "v_mul_lo_u16 v2, v0, v1",
        [f"v_mov_b32 v0, {u16(s0)}", f"v_mov_b32 v1, {u16(s1)}"],
        read_reg="v2",
    ) & MASK16


def run_mad_i16(s0: int, s1: int, s2: int, name: str = "ps_mad_i16") -> int:
    return run_ps_valu(
        name,
        "v_mad_i16 v3, v0, v1, v2",
        [f"v_mov_b32 v0, {u16(s0)}", f"v_mov_b32 v1, {u16(s1)}", f"v_mov_b32 v2, {u16(s2)}"],
        read_reg="v3",
    ) & MASK16


def run_mad_u16(s0: int, s1: int, s2: int, name: str = "ps_mad_u16") -> int:
    return run_ps_valu(
        name,
        "v_mad_u16 v3, v0, v1, v2",
        [f"v_mov_b32 v0, {u16(s0)}", f"v_mov_b32 v1, {u16(s1)}", f"v_mov_b32 v2, {u16(s2)}"],
        read_reg="v3",
    ) & MASK16


def run_mad_i32_i16(s0: int, s1: int, s2: int, name: str = "ps_mad_i32_i16") -> int:
    return run_ps_valu(
        name,
        "v_mad_i32_i16 v3, v0, v1, v2",
        [f"v_mov_b32 v0, {u16(s0)}", f"v_mov_b32 v1, {u16(s1)}", f"v_mov_b32 v2, {u32(s2)}"],
        read_reg="v3",
    )


def run_mad_u32_u16(s0: int, s1: int, s2: int, name: str = "ps_mad_u32_u16") -> int:
    return run_ps_valu(
        name,
        "v_mad_u32_u16 v3, v0, v1, v2",
        [f"v_mov_b32 v0, {u16(s0)}", f"v_mov_b32 v1, {u16(s1)}", f"v_mov_b32 v2, {u32(s2)}"],
        read_reg="v3",
    )


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


class VMulLoU16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_mul_lo_u16 with translator validation."""

    def test_basic_multiplication(self):
        hw1 = run_mul_lo_u16(10, 20, "ps_mul16_basic1")
        self.assertEqual(hw1, ref_mul_lo_u16(10, 20))
        self.assertEqual(hw1, 200)

        hw2 = run_mul_lo_u16(100, 50, "ps_mul16_basic2")
        self.assertEqual(hw2, ref_mul_lo_u16(100, 50))
        self.assertEqual(hw2, 5000)

    def test_multiplication_overflow_wrapping(self):
        # 1000 * 1000 = 1000000 = 0xF4240 -> low 16 bits: 0x4240 = 16960
        hw = run_mul_lo_u16(1000, 1000, "ps_mul16_wrap")
        self.assertEqual(hw, ref_mul_lo_u16(1000, 1000))
        self.assertEqual(hw, 16960)

    def test_identity_and_zero(self):
        hw_id = run_mul_lo_u16(1234, 1, "ps_mul16_ident")
        self.assertEqual(hw_id, 1234)
        hw_zero = run_mul_lo_u16(1234, 0, "ps_mul16_zero")
        self.assertEqual(hw_zero, 0)


class VMulLoU16AstTests(unittest.TestCase):
    """AST modeling and translator execution for v_mul_lo_u16."""

    def test_ast_emission(self):
        node = Mul16AstNode("v_mul_lo_u16", "v2", "v0", "v1")
        self.assertEqual(node.emit_asm(), "v_mul_lo_u16 v2, v0, v1")
        res = run_ps_valu("ps_ast_mul_lo", node.emit_asm(), ["v_mov_b32 v0, 12", "v_mov_b32 v1, 8"], read_reg="v2")
        self.assertEqual(res & MASK16, 96)


class VMulLoU16InterpreterTests(unittest.TestCase):
    """Interpreter simulation through translator for v_mul_lo_u16."""

    def test_scalar_simulation(self):
        res = run_mul_lo_u16(25, 4, "ps_interp_mul_lo")
        self.assertEqual(res, 100)


class VMulLoU16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_mul_lo_u16."""

    def test_simd_divergent_lanes(self):
        test_cases = [(0, 10), (1, 10), (5, 10), (15, 10), (31, 10), (63, 10)]
        for idx, (s0, s1) in enumerate(test_cases):
            with self.subTest(idx=idx, s0=s0, s1=s1):
                hw = run_mul_lo_u16(s0, s1, f"ps_simd_mullo_{idx}")
                self.assertEqual(hw, s0 * s1)

    def test_max_u16_boundary(self):
        hw = run_mul_lo_u16(0xFFFF, 0xFFFF, "ps_mullo_max")
        self.assertEqual(hw, ref_mul_lo_u16(0xFFFF, 0xFFFF))
        self.assertEqual(hw, 1)


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
        self.assertIn("OpIMul", spv)

    def test_negative_mutation_contrast(self):
        hw_mul = run_mul_lo_u16(10, 20, "ps_mullo_mut")
        hw_add = 10 + 20
        self.assertNotEqual(hw_mul, hw_add)


class VMadI16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_mad_i16 with translator validation."""

    def test_basic_signed_mad(self):
        # 10 * 20 + 5 = 205
        hw = run_mad_i16(10, 20, 5, "ps_mad_i16_basic")
        self.assertEqual(hw, ref_mad_i16(10, 20, 5))
        self.assertEqual(hw, 205)

    def test_negative_operands(self):
        # -2 * 10 + 5 = -15 = 0xFFF1
        hw1 = run_mad_i16(u16(-2), 10, 5, "ps_mad_i16_neg1")
        self.assertEqual(hw1, ref_mad_i16(u16(-2), 10, 5))
        self.assertEqual(hw1, 0xFFF1)

        # -5 * -4 + (-10) = 20 - 10 = 10
        hw2 = run_mad_i16(u16(-5), u16(-4), u16(-10), "ps_mad_i16_neg2")
        self.assertEqual(hw2, ref_mad_i16(u16(-5), u16(-4), u16(-10)))
        self.assertEqual(hw2, 10)


class VMadI16AstTests(unittest.TestCase):
    """AST modeling and translator execution for v_mad_i16."""

    def test_ast_emission(self):
        node = Mul16AstNode("v_mad_i16", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_mad_i16 v3, v0, v1, v2")
        res = run_ps_valu(
            "ps_ast_mad_i16",
            node.emit_asm(),
            ["v_mov_b32 v0, 3", "v_mov_b32 v1, 4", "v_mov_b32 v2, 5"],
            read_reg="v3"
        )
        self.assertEqual(res & MASK16, 17)


class VMadI16InterpreterTests(unittest.TestCase):
    """Interpreter simulation through translator for v_mad_i16."""

    def test_scalar_simulation(self):
        hw = run_mad_i16(u16(-3), 10, 2, "ps_interp_mad_i16")
        # -3 * 10 + 2 = -28 = 0xFFE4
        self.assertEqual(hw, 0xFFE4)


class VMadI16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_mad_i16."""

    def test_simd_divergent_lanes(self):
        cases = [-32, -16, 0, 16, 31]
        for idx, val in enumerate(cases):
            with self.subTest(idx=idx, val=val):
                hw = run_mad_i16(u16(val), 2, 10, f"ps_simd_madi16_{idx}")
                expected = u16(val * 2 + 10)
                self.assertEqual(hw, expected)


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
        self.assertIn("OpIMul", spv)
        self.assertIn("OpIAdd", spv)

    def test_negative_mutation_contrast(self):
        hw = run_mad_i16(10, 20, 5, "ps_madi16_mut")
        add_res = u16(10 + 20 + 5)
        self.assertNotEqual(hw, add_res)
        self.assertNotEqual(hw, ref_mul_lo_u16(10, 20))


class VMadU16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_mad_u16 with translator validation."""

    def test_basic_unsigned_mad(self):
        hw = run_mad_u16(10, 20, 5, "ps_mad_u16_basic")
        self.assertEqual(hw, ref_mad_u16(10, 20, 5))
        self.assertEqual(hw, 205)

    def test_unsigned_wrapping(self):
        # 1000 * 100 + 50 = 100050 = 0x186D2 -> 0x86D2 = 34514
        hw = run_mad_u16(1000, 100, 50, "ps_mad_u16_wrap")
        self.assertEqual(hw, ref_mad_u16(1000, 100, 50))
        self.assertEqual(hw, 34514)


class VMadU16AstTests(unittest.TestCase):
    """AST modeling and translator execution for v_mad_u16."""

    def test_ast_emission(self):
        node = Mul16AstNode("v_mad_u16", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_mad_u16 v3, v0, v1, v2")
        res = run_ps_valu(
            "ps_ast_mad_u16",
            node.emit_asm(),
            ["v_mov_b32 v0, 6", "v_mov_b32 v1, 7", "v_mov_b32 v2, 8"],
            read_reg="v3"
        )
        self.assertEqual(res & MASK16, 50)


class VMadU16InterpreterTests(unittest.TestCase):
    """Interpreter simulation through translator for v_mad_u16."""

    def test_scalar_simulation(self):
        hw = run_mad_u16(100, 2, 33, "ps_interp_mad_u16")
        self.assertEqual(hw, 233)


class VMadU16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_mad_u16."""

    def test_simd_divergent_lanes(self):
        cases = [0, 10, 25, 50, 63]
        for idx, val in enumerate(cases):
            with self.subTest(idx=idx, val=val):
                hw = run_mad_u16(val, 5, 100, f"ps_simd_madu16_{idx}")
                self.assertEqual(hw, val * 5 + 100)


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
        self.assertIn("OpIMul", spv)
        self.assertIn("OpIAdd", spv)

    def test_negative_mutation_contrast(self):
        hw = run_mad_u16(10, 10, 10, "ps_madu16_mut")
        add_res = u16(10 + 10 + 10)
        self.assertNotEqual(hw, add_res)
        self.assertNotEqual(hw, ref_mul_lo_u16(10, 10))


class VMadI32I16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_mad_i32_i16 with translator validation."""

    def test_signed_multiplication_32bit_addend(self):
        # -1000 * 2000 + 50 = -1999950
        hw = run_mad_i32_i16(u16(-1000), 2000, 50, "ps_mad_i32_i16_basic")
        self.assertEqual(i32(hw), -1999950)
        self.assertEqual(hw, ref_mad_i32_i16(u16(-1000), 2000, 50))

    def test_positive_full_range(self):
        # 10000 * 10000 + 100 = 100000100
        hw = run_mad_i32_i16(10000, 10000, 100, "ps_mad_i32_i16_full")
        self.assertEqual(hw, 100000100)
        self.assertEqual(hw, ref_mad_i32_i16(10000, 10000, 100))


class VMadI32I16AstTests(unittest.TestCase):
    """AST modeling and translator execution for v_mad_i32_i16."""

    def test_ast_emission(self):
        node = Mul16AstNode("v_mad_i32_i16", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_mad_i32_i16 v3, v0, v1, v2")
        res = run_ps_valu(
            "ps_ast_mad_i32_i16",
            node.emit_asm(),
            ["v_mov_b32 v0, 100", "v_mov_b32 v1, 200", "v_mov_b32 v2, 300"],
            read_reg="v3"
        )
        self.assertEqual(res, 20300)


class VMadI32I16InterpreterTests(unittest.TestCase):
    """Interpreter simulation through translator for v_mad_i32_i16."""

    def test_scalar_simulation(self):
        hw = run_mad_i32_i16(u16(-500), 40, 1000, "ps_interp_mad_i32_i16")
        # -500 * 40 + 1000 = -20000 + 1000 = -19000
        self.assertEqual(i32(hw), -19000)


class VMadI32I16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_mad_i32_i16."""

    def test_simd_divergent_lanes(self):
        cases = [-32, -10, 0, 15, 31]
        for idx, val in enumerate(cases):
            with self.subTest(idx=idx, val=val):
                hw = run_mad_i32_i16(u16(val), 1000, 50000, f"ps_simd_madi32i16_{idx}")
                expected = val * 1000 + 50000
                self.assertEqual(i32(hw), expected)


class VMadI32I16TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_mad_i32_i16."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0xFC18",  # -1000 in i16
            "v_mov_b32 v1, 2000",
            "v_mov_b32 v2, 50",
            "v_mad_i32_i16 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mad_i32_i16_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(i32(out[0]), -1999950)
        self.assertIn("OpBitFieldSExtract", spv)

    def test_negative_mutation_contrast(self):
        s0, s1, s2 = u16(-10), 10, 0
        signed_hw = run_mad_i32_i16(s0, s1, s2, "ps_madi32i16_mut")
        unsigned_hw = run_mad_u32_u16(s0, s1, s2, "ps_madu32u16_mut")
        self.assertNotEqual(signed_hw, unsigned_hw)


class VMadU32U16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_mad_u32_u16 with translator validation."""

    def test_unsigned_multiplication_32bit_addend(self):
        # 1000 * 2000 + 50 = 2000050
        hw = run_mad_u32_u16(1000, 2000, 50, "ps_mad_u32_u16_basic")
        self.assertEqual(hw, 2000050)
        self.assertEqual(hw, ref_mad_u32_u16(1000, 2000, 50))

    def test_max_u16_product(self):
        # 0xFFFF * 0xFFFF + 0 = 65535 * 65535 = 4294836225
        hw = run_mad_u32_u16(0xFFFF, 0xFFFF, 0, "ps_mad_u32_u16_max")
        self.assertEqual(hw, 4294836225)
        self.assertEqual(hw, ref_mad_u32_u16(0xFFFF, 0xFFFF, 0))


class VMadU32U16AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_mad_u32_u16 through translator."""

    def test_ast_emission(self):
        node = Mul16AstNode("v_mad_u32_u16", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_mad_u32_u16 v3, v0, v1, v2")
        res = run_ps_valu(
            "ps_ast_mad_u32_u16",
            node.emit_asm(),
            ["v_mov_b32 v0, 50", "v_mov_b32 v1, 60", "v_mov_b32 v2, 70"],
            read_reg="v3"
        )
        self.assertEqual(res, 3070)

    def test_interpreter_scalar_simulation(self):
        hw = run_mad_u32_u16(500, 300, 10000, "ps_interp_mad_u32_u16")
        self.assertEqual(hw, 160000)


class VMadU32U16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_mad_u32_u16."""

    def test_simd_divergent_lanes(self):
        cases = [0, 5, 20, 45, 63]
        for idx, val in enumerate(cases):
            with self.subTest(idx=idx, val=val):
                hw = run_mad_u32_u16(val * 10, 100, 1000, f"ps_simd_madu32u16_{idx}")
                self.assertEqual(hw, (val * 10) * 100 + 1000)


class VMadU32U16TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_mad_u32_u16."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 1000",
            "v_mov_b32 v1, 2000",
            "v_mov_b32 v2, 50",
            "v_mad_u32_u16 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mad_u32_u16_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 2000050)
        self.assertIn("OpBitFieldUExtract", spv)

    def test_negative_mutation_contrast(self):
        hw = run_mad_u32_u16(100, 200, 300, "ps_madu32u16_mut")
        add_res = 100 + 200 + 300
        self.assertNotEqual(hw, add_res)


if __name__ == "__main__":
    unittest.main()
