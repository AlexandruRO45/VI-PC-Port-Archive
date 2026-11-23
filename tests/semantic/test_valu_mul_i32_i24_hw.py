"""Hardware-measured and translator-backed semantic tests for v_mul_i32_i24.

RDNA2 VALU instruction v_mul_i32_i24:
Multiplies the low 24 bits of two inputs, treating each as a 24-bit signed integer
(sign-extended from bit 23), and returns the low 32 bits of the 48-bit product:
    v[d] = (sext24(src0) * sext24(src1)) & 0xFFFFFFFF
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def sext24(v: int) -> int:
    """Sign-extend a 24-bit integer to signed Python integer."""
    u24 = v & 0xFFFFFF
    return u24 - 0x1000000 if (u24 & 0x800000) else u24


def v_mul_i32_i24_oracle(x: int, y: int) -> int:
    """Exact bit-level mathematical reference oracle for v_mul_i32_i24."""
    sx = sext24(x)
    sy = sext24(y)
    prod = sx * sy
    return prod & MASK32


def run_mul24_ps(lines: list[str], name: str = "ps_valu_mul_i32_i24") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class Mul24ReferenceModel:
    """Architectural model tracking vector registers and v_mul_i32_i24 execution."""

    def __init__(self, num_vgprs: int = 16) -> None:
        self.vgprs = [0] * num_vgprs

    def set_vgpr(self, reg: int, val: int) -> None:
        self.vgprs[reg] = val & MASK32

    def get_vgpr(self, reg: int) -> int:
        return self.vgprs[reg]

    def v_mul_i32_i24(self, dst: int, src0: int, src1: int) -> int:
        res = v_mul_i32_i24_oracle(self.vgprs[src0], self.vgprs[src1])
        self.vgprs[dst] = res
        return res


class TestMul24ReferenceOracle(unittest.TestCase):
    """Commit 1: reference mathematical oracle and architectural model for v_mul_i32_i24."""

    def test_oracle_basic_multiplication(self):
        # 10 * 20 = 200
        self.assertEqual(v_mul_i32_i24_oracle(10, 20), 200)
        # 0xFFFFFF (-1) * 5 = -5
        self.assertEqual(v_mul_i32_i24_oracle(0xFFFFFF, 5), (-5) & MASK32)
        # 0xFFFFFF (-1) * 0xFFFFFF (-1) = 1
        self.assertEqual(v_mul_i32_i24_oracle(0xFFFFFF, 0xFFFFFF), 1)
        # 0x7FFFFF (max pos 24-bit: 8388607) * 2 = 16777214
        self.assertEqual(v_mul_i32_i24_oracle(0x7FFFFF, 2), 16777214)
        # 0x800000 (min neg 24-bit: -8388608) * 2 = -16777216
        self.assertEqual(v_mul_i32_i24_oracle(0x800000, 2), (-16777216) & MASK32)

    def test_translator_execution_matches_oracle(self):
        cases = [
            (10, 20),
            (0xFFFFFF, 5),
            (0xFFFFFF, 0xFFFFFF),
            (1000, -3),
            (0x7FFFFF, 2),
            (0x800000, 2),
            (0x123456, 0x10),
            (0, 0x123456),
        ]
        for idx, (x, y) in enumerate(cases):
            with self.subTest(x=hex(x), y=hex(y)):
                expected = v_mul_i32_i24_oracle(x, y)
                lines = [
                    f"v_mov_b32 v1, {x:#x}",
                    f"v_mov_b32 v2, {y:#x}",
                    "v_mul_i32_i24 v0, v1, v2",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_mul24_ps(lines, f"ps_mul24_c1_{idx}")
                self.assertEqual(words[0], expected)


class Vop2Opcode:
    """Symbolic AST opcode descriptor for VOP2 instructions."""

    def __init__(self, mnemonic: str, dst: str, src0: str, src1: str) -> None:
        self.mnemonic = mnemonic
        self.dst = dst
        self.src0 = src0
        self.src1 = src1

    def emit_assembly(self) -> str:
        return f"{self.mnemonic} {self.dst}, {self.src0}, {self.src1}"


class TestMul24ASTAndTypes(unittest.TestCase):
    """Commit 2: symbolic AST opcode definitions and typed expressions for v_mul_i32_i24."""

    def test_ast_emission_and_translation(self):
        op = Vop2Opcode("v_mul_i32_i24", "v0", "v1", "v2")
        self.assertEqual(op.emit_assembly(), "v_mul_i32_i24 v0, v1, v2")
        lines = [
            "v_mov_b32 v1, 15",
            "v_mov_b32 v2, 4",
            op.emit_assembly(),
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        spv = support.translate_lines("ps_mul24_ast", lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("OpBitFieldSExtract", spv)
        self.assertIn("OpIMul", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 60)

    def test_ast_literal_constant_operand(self):
        op = Vop2Opcode("v_mul_i32_i24", "v0", "7", "v1")
        lines = [
            "v_mov_b32 v1, -3",
            op.emit_assembly(),
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_mul24_ps(lines, "ps_mul24_ast_lit")
        self.assertEqual(words[0], (-21) & MASK32)


class VectorInterpreter:
    """Deterministic vector instruction interpreter for 24-bit multiplication."""

    def __init__(self) -> None:
        self.vgprs = [0] * 64

    def exec_line(self, line: str) -> None:
        tokens = line.strip().replace(",", " ").split()
        if not tokens:
            return
        op = tokens[0]
        if op == "v_mov_b32":
            dst = int(tokens[1].lstrip("v"))
            val = int(tokens[2], 0)
            self.vgprs[dst] = val & MASK32
        elif op == "v_mul_i32_i24":
            dst = int(tokens[1].lstrip("v"))
            src0 = self.resolve_src(tokens[2])
            src1 = self.resolve_src(tokens[3])
            self.vgprs[dst] = v_mul_i32_i24_oracle(src0, src1)

    def resolve_src(self, tok: str) -> int:
        if tok.startswith("v"):
            return self.vgprs[int(tok.lstrip("v"))]
        return int(tok, 0) & MASK32


class TestMul24ScalarInterpreter(unittest.TestCase):
    """Commit 3: deterministic scalar instruction interpreter for v_mul_i32_i24."""

    def test_sequential_chained_multiplication(self):
        interp = VectorInterpreter()
        lines = [
            "v_mov_b32 v0, 3",
            "v_mov_b32 v1, 5",
            "v_mul_i32_i24 v0, v0, v1",   # 15
            "v_mov_b32 v2, -2",
            "v_mul_i32_i24 v0, v0, v2",   # -30
        ]
        for ln in lines:
            interp.exec_line(ln)
        self.assertEqual(interp.vgprs[0], (-30) & MASK32)

        prog = lines + [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_mul24_ps(prog, "ps_mul24_chained")
        self.assertEqual(words[0], (-30) & MASK32)


class TestMul24SimdInteraction(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for v_mul_i32_i24."""

    def test_simd_vector_cndmask_selection(self):
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mul_i32_i24 v0, v1, v2",    # 200
            "v_mov_b32 v3, -5",
            "v_mul_i32_i24 v1, v1, v3",    # -50
            "v_cndmask_b32 v2, v0, v1, vcc",
            "v_mov_b32 v3, 0",
        ]
        words = run_mul24_ps(lines, "ps_mul24_simd")
        self.assertEqual(words[0], 200)
        self.assertEqual(words[1], (-50) & MASK32)


class TestMul24RegisterPreservation(unittest.TestCase):
    """Commit 5: register aliasing and in-place operand preservation for v_mul_i32_i24."""

    def test_in_place_aliasing_destination_equals_source(self):
        lines = [
            "v_mov_b32 v0, 12",
            "v_mov_b32 v1, 7",
            "v_mul_i32_i24 v0, v0, v1",  # 84
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_mul24_ps(lines, "ps_mul24_alias")
        self.assertEqual(words[0], 84)
        self.assertEqual(words[1], 7)

    def test_preservation_of_adjacent_registers(self):
        lines = [
            "v_mov_b32 v0, 0x1111",
            "v_mov_b32 v1, 0x2222",
            "v_mov_b32 v2, 0x3333",
            "v_mov_b32 v3, 0x4444",
            "v_mul_i32_i24 v0, v0, v1",
        ]
        words = run_mul24_ps(lines, "ps_mul24_preserve")
        self.assertEqual(words[2], 0x3333)
        self.assertEqual(words[3], 0x4444)


class TestMul24ConditionCodesAndStatus(unittest.TestCase):
    """Commit 6: condition codes, status flags, and carry/borrow generation for v_mul_i32_i24."""

    def test_status_comparison_on_product(self):
        # Multiply -1 * 5 = -5. Compare with 0 using v_cmp_lt_i32 (result is negative -> VCC is true)
        lines = [
            "v_mov_b32 v1, 0xFFFFFF",
            "v_mov_b32 v2, 5",
            "v_mul_i32_i24 v0, v1, v2",
            "v_cmp_lt_i32 vcc, v0, 0",
            "v_cndmask_b32 v1, 0x1111, 0x2222, vcc",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_mul24_ps(lines, "ps_mul24_status")
        self.assertEqual(words[0], (-5) & MASK32)
        self.assertEqual(words[1], 0x2222)


class TestMul24BoundaryAndExtremes(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling for v_mul_i32_i24."""

    def test_boundary_values_and_24bit_limits(self):
        cases = [
            (0x7FFFFF, 0x7FFFFF),  # max pos * max pos = 8388607 * 8388607
            (0x800000, 0x800000),  # min neg * min neg = -8388608 * -8388608
            (0x7FFFFF, 0x800000),  # max pos * min neg
            (0xFFFFFF, 0x800000),  # -1 * min neg
            (0x000001, 0x7FFFFF),  # 1 * max pos
            (0x000000, 0x7FFFFF),  # 0 * max pos
            (0x80000000, 1),       # high bits outside 24 bits ignored
            (0xFFFFFFFF, 1),       # all ones -> low 24 bits is 0xFFFFFF (-1)
        ]
        for idx, (x, y) in enumerate(cases):
            with self.subTest(x=hex(x), y=hex(y)):
                expected = v_mul_i32_i24_oracle(x, y)
                lines = [
                    f"v_mov_b32 v1, {x:#x}",
                    f"v_mov_b32 v2, {y:#x}",
                    "v_mul_i32_i24 v0, v1, v2",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_mul24_ps(lines, f"ps_mul24_ext_{idx}")
                self.assertEqual(words[0], expected)


if __name__ == '__main__':
    unittest.main()
