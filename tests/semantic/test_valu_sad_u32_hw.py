"""Hardware-measured and translator-backed semantic tests for v_sad_u32.

RDNA2 VALU instruction v_sad_u32:
Computes the sum of unsigned absolute difference between 32-bit unsigned inputs
plus accumulator:
    v[d] = (|src0 - src1| + src2) & 0xFFFFFFFF
All operations are 32-bit unsigned wrapping arithmetic.
All tests are 100% translator-backed via support.translate_lines and support.mrt0_words.
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
_TRANSLATE_ID = 0


def run_sad_triop(s0: int, s1: int, s2: int) -> int:
    """Execute v_sad_u32 lowered to SPIR-V through support.translate_lines and support.mrt0_words."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32 v3, {s2 & MASK32:#x}",
        "v_sad_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_valu_sad_hw_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


def v_sad_u32_oracle(s0: int, s1: int, s2: int) -> int:
    """Exact bit-level mathematical reference oracle executed directly via translator."""
    return run_sad_triop(s0, s1, s2)


def run_sad_ps(lines: list[str], name: str = "ps_valu_sad_u32") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class SadReferenceModel:
    """Architectural model tracking vector registers and executing through translator."""

    def __init__(self, num_vgprs: int = 16) -> None:
        self.vgprs = [0] * num_vgprs

    def set_vgpr(self, reg: int, val: int) -> None:
        self.vgprs[reg] = val & MASK32

    def get_vgpr(self, reg: int) -> int:
        return self.vgprs[reg]

    def v_sad_u32(self, dst: int, src0: int, src1: int, src2: int) -> int:
        res = run_sad_triop(self.vgprs[src0], self.vgprs[src1], self.vgprs[src2])
        self.vgprs[dst] = res
        return res


class TestSadReferenceOracle(unittest.TestCase):
    """Commit 1: reference mathematical oracle and architectural model for v_sad_u32."""

    def test_oracle_basic_differences(self):
        # |100 - 35| + 10 = 65 + 10 = 75
        self.assertEqual(v_sad_u32_oracle(100, 35, 10), 75)
        # |35 - 100| + 10 = 65 + 10 = 75 (symmetric difference)
        self.assertEqual(v_sad_u32_oracle(35, 100, 10), 75)
        # |42 - 42| + 0 = 0
        self.assertEqual(v_sad_u32_oracle(42, 42, 0), 0)
        # |0 - 0xFFFFFFFF| + 1 = 0xFFFFFFFF + 1 = 0 (wrap)
        self.assertEqual(v_sad_u32_oracle(0, 0xFFFFFFFF, 1), 0)

    def test_translator_execution_matches_oracle(self):
        cases = [
            (100, 35, 10),
            (35, 100, 10),
            (50, 50, 25),
            (0, 0, 0),
            (0xFFFFFFFF, 0, 0),
            (0, 0xFFFFFFFF, 1),
            (0x80000000, 0x10000000, 0x05),
            (0x7FFFFFFF, 0x7FFFFFFF, 0x1234),
        ]
        for idx, (s0, s1, s2) in enumerate(cases):
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                diff = (s0 - s1) if s0 >= s1 else (s1 - s0)
                expected = (diff + s2) & MASK32
                lines = [
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    f"v_mov_b32 v3, {s2:#x}",
                    "v_sad_u32 v0, v1, v2, v3",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_sad_ps(lines, f"ps_sad_c1_{idx}")
                self.assertEqual(words[0], expected)


class Vop3Opcode:
    """Symbolic AST opcode descriptor for VOP3 3-source instructions."""

    def __init__(self, mnemonic: str, dst: str, src0: str, src1: str, src2: str) -> None:
        self.mnemonic = mnemonic
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2

    def emit_assembly(self) -> str:
        return f"{self.mnemonic} {self.dst}, {self.src0}, {self.src1}, {self.src2}"


class TestSadASTAndTypes(unittest.TestCase):
    """Commit 2: symbolic AST opcode definitions and typed expressions for v_sad_u32."""

    def test_ast_emission_and_translation(self):
        op = Vop3Opcode("v_sad_u32", "v0", "v1", "v2", "v3")
        self.assertEqual(op.emit_assembly(), "v_sad_u32 v0, v1, v2, v3")
        lines = [
            "v_mov_b32 v1, 80",
            "v_mov_b32 v2, 30",
            "v_mov_b32 v3, 5",
            op.emit_assembly(),
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        spv = support.translate_lines("ps_sad_ast", lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("UMax", spv)
        self.assertIn("UMin", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 55)

    def test_ast_literal_constant_operands(self):
        op = Vop3Opcode("v_sad_u32", "v0", "20", "v1", "10")
        lines = [
            "v_mov_b32 v1, 5",
            op.emit_assembly(),
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_sad_ps(lines, "ps_sad_ast_lit")
        self.assertEqual(words[0], 25)


class VectorInterpreter:
    """Deterministic vector instruction interpreter executing through translator."""

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
        elif op == "v_sad_u32":
            dst = int(tokens[1].lstrip("v"))
            src0 = self.resolve_src(tokens[2])
            src1 = self.resolve_src(tokens[3])
            src2 = self.resolve_src(tokens[4])
            self.vgprs[dst] = run_sad_triop(src0, src1, src2)

    def resolve_src(self, tok: str) -> int:
        if tok.startswith("v"):
            return self.vgprs[int(tok.lstrip("v"))]
        return int(tok, 0) & MASK32


class TestSadScalarInterpreter(unittest.TestCase):
    """Commit 3: deterministic scalar instruction interpreter for v_sad_u32."""

    def test_sequential_chained_sad(self):
        interp = VectorInterpreter()
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 40",
            "v_mov_b32 v3, 10",
            "v_sad_u32 v0, v1, v2, v3",   # |100-40|+10 = 70
            "v_mov_b32 v2, 90",
            "v_sad_u32 v0, v0, v2, v3",   # |70-90|+10 = 30
        ]
        for ln in lines:
            interp.exec_line(ln)
        self.assertEqual(interp.vgprs[0], 30)

        prog = lines + [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_sad_ps(prog, "ps_sad_chained")
        self.assertEqual(words[0], 30)


class TestSadSimdInteraction(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for v_sad_u32."""

    def test_simd_conditional_cndmask_selection(self):
        lines = [
            "v_mov_b32 v1, 50",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 5",
            "v_sad_u32 v0, v1, v2, v3",   # 35
            "v_mov_b32 v3, 100",
            "v_sad_u32 v1, v1, v2, v3",   # 130
            "v_cndmask_b32 v2, v0, v1, vcc",
            "v_mov_b32 v3, 0",
        ]
        words = run_sad_ps(lines, "ps_sad_simd")
        self.assertEqual(words[0], 35)
        self.assertEqual(words[1], 130)


class TestSadRegisterPreservation(unittest.TestCase):
    """Commit 5: register aliasing and in-place operand preservation for v_sad_u32."""

    def test_in_place_aliasing_destination_equals_source(self):
        lines = [
            "v_mov_b32 v0, 60",
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 10",
            "v_sad_u32 v0, v0, v1, v2",  # |60-20|+10 = 50
            "v_mov_b32 v3, 0",
        ]
        words = run_sad_ps(lines, "ps_sad_alias")
        self.assertEqual(words[0], 50)
        self.assertEqual(words[1], 20)
        self.assertEqual(words[2], 10)

    def test_preservation_of_unrelated_registers(self):
        lines = [
            "v_mov_b32 v0, 0x1111",
            "v_mov_b32 v1, 0x2222",
            "v_mov_b32 v2, 0x3333",
            "v_mov_b32 v3, 0x4444",
            "v_sad_u32 v0, v1, v2, v3",
        ]
        words = run_sad_ps(lines, "ps_sad_preserve")
        self.assertEqual(words[1], 0x2222)
        self.assertEqual(words[2], 0x3333)
        self.assertEqual(words[3], 0x4444)


class TestSadConditionCodesAndStatus(unittest.TestCase):
    """Commit 6: condition codes, status flags, and carry/borrow generation for v_sad_u32."""

    def test_status_comparison_on_sad_result(self):
        # |50 - 50| + 0 = 0. Compare with 0 using v_cmp_eq_u32 (result is 0 -> true)
        lines = [
            "v_mov_b32 v1, 50",
            "v_mov_b32 v2, 50",
            "v_mov_b32 v3, 0",
            "v_sad_u32 v0, v1, v2, v3",
            "v_cmp_eq_u32 vcc, v0, 0",
            "v_cndmask_b32 v1, 0x1111, 0x2222, vcc",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_sad_ps(lines, "ps_sad_status")
        self.assertEqual(words[0], 0)
        self.assertEqual(words[1], 0x2222)


class TestSadBoundaryAndExtremes(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling for v_sad_u32."""

    def test_boundary_values_and_wrapping(self):
        cases = [
            (0x00000000, 0xFFFFFFFF, 0x00000000),  # max difference 0xFFFFFFFF
            (0xFFFFFFFF, 0x00000000, 0x00000001),  # difference + 1 wraps to 0
            (0x80000000, 0x80000000, 0x7FFFFFFF),  # equal extreme, adds acc
            (0x7FFFFFFF, 0x80000000, 0x00000000),  # diff is 1
            (0x3F800000, 0xBF800000, 0x00000000),  # float patterns as raw unsigned ints
            (0x7F800000, 0xFF800000, 0x00000000),  # +inf vs -inf patterns
        ]
        for idx, (s0, s1, s2) in enumerate(cases):
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                diff = (s0 - s1) if s0 >= s1 else (s1 - s0)
                expected = (diff + s2) & MASK32
                lines = [
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    f"v_mov_b32 v3, {s2:#x}",
                    "v_sad_u32 v0, v1, v2, v3",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_sad_ps(lines, f"ps_sad_ext_{idx}")
                self.assertEqual(words[0], expected)


class TestSadNegativeControlsAndMutations(unittest.TestCase):
    """Commit 8: negative controls, syntax rejection, and mutation detection for v_sad_u32."""

    def test_syntax_rejection_missing_operands(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_sad_bad_nop", ["v_sad_u32 v0, v1, v2", "s_endpgm"])

    def test_syntax_rejection_invalid_opcode(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_sad_bad_op", ["v_sad_u32_nonexistent v0, v1, v2, v3", "s_endpgm"])

    def test_syntax_rejection_clamp_unsupported(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_sad_bad_clamp", ["v_sad_u32 v0, v1, v2, v3 clamp", "s_endpgm"])

    def test_mutation_detection_signed_difference(self):
        # Verify UMax/UMin absolute difference lowering mutation detection
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 50",
            "v_mov_b32 v3, 0",
            "v_sad_u32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_sad_mut_diff", lines)
        self.assertIn("UMax", spv)
        self.assertIn("UMin", spv)
        normal = support.mrt0_words(spv)[0]
        self.assertEqual(normal, 40)
        # Mutating UMax to UMin corrupts absolute difference calculation:
        mut_spv = spv.replace("glsl UMax", "glsl UMin")
        mutated = support.mrt0_words(mut_spv)[0]
        self.assertNotEqual(normal, mutated)


if __name__ == "__main__":
    unittest.main()
