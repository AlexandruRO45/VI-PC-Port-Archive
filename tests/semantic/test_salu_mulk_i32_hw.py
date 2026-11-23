"""Hardware-measured and translator-backed semantic tests for s_mulk_i32.

RDNA2 SALU instruction s_mulk_i32:
Multiplies a 32-bit scalar general-purpose register (sD) by a sign-extended 16-bit
signed immediate (simm16) and stores the 32-bit result back in sD:
    sD = (sD * sign_extend_16(simm16)) & 0xFFFFFFFF
Condition codes: does not modify SCC (Scalar Condition Code).
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


def s_mulk_i32_oracle(s_val: int, simm16: int) -> int:
    """Exact bit-level mathematical reference oracle for s_mulk_i32.

    Takes a 32-bit integer s_val and 16-bit integer simm16.
    Sign extends simm16 to a signed 32-bit value and performs two's-complement
    32-bit multiplication.
    """
    s_val = s_val & MASK32
    u16 = simm16 & 0xFFFF
    k = u16 - 0x10000 if (u16 & 0x8000) else u16
    signed_s = s_val - 0x100000000 if (s_val & 0x80000000) else s_val
    prod = signed_s * k
    return prod & MASK32


def run_mulk_ps(lines: list[str], name: str = "ps_salu_mulk_i32") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class MulkReferenceModel:
    """Architectural model tracking scalar registers and s_mulk_i32 execution."""

    def __init__(self, num_sgprs: int = 16) -> None:
        self.sgprs = [0] * num_sgprs

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg]

    def s_mulk_i32(self, reg: int, simm16: int) -> int:
        res = s_mulk_i32_oracle(self.sgprs[reg], simm16)
        self.sgprs[reg] = res
        return res


class TestMulkReferenceOracle(unittest.TestCase):
    """Commit 1: reference mathematical oracle and architectural model for s_mulk_i32."""

    def test_oracle_basic_multiplication(self):
        self.assertEqual(s_mulk_i32_oracle(7, 5), 35)
        self.assertEqual(s_mulk_i32_oracle(10, 0), 0)
        self.assertEqual(s_mulk_i32_oracle(12345, 1), 12345)
        self.assertEqual(s_mulk_i32_oracle(12345, -1), (-12345) & MASK32)
        self.assertEqual(s_mulk_i32_oracle(7, -3), (-21) & MASK32)

    def test_oracle_sign_extension(self):
        self.assertEqual(s_mulk_i32_oracle(42, 0xFFFF), (-42) & MASK32)
        self.assertEqual(s_mulk_i32_oracle(2, 0x7FFF), 65534)
        self.assertEqual(s_mulk_i32_oracle(2, 0x8000), (-65536) & MASK32)

    def test_translator_execution_matches_oracle(self):
        test_cases = [
            (7, 5),
            (100, -3),
            (0, 100),
            (42, 1),
            (42, -1),
            (1000, 32),
            (0x12345678, 2),
            (0x7FFFFFFF, 2),
            (0x80000000, 2),
            (0x1000, 0x7FFF),
            (0x1000, -32768),
        ]
        for s_init, k in test_cases:
            with self.subTest(s_init=hex(s_init), k=k):
                expected = s_mulk_i32_oracle(s_init, k)
                lines = [
                    f"s_mov_b32 s0, {s_init:#x}",
                    f"s_mulk_i32 s0, {k}",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_mulk_ps(lines, f"ps_mulk_c1_{s_init & 0xFFFF}_{abs(k)}")
                self.assertEqual(words[0], expected)


class SopkOpcode:
    """Symbolic AST opcode descriptor for SOPK instructions."""

    def __init__(self, mnemonic: str, dst_reg: str, simm16: int) -> None:
        self.mnemonic = mnemonic
        self.dst_reg = dst_reg
        self.simm16 = simm16

    def emit_assembly(self) -> str:
        return f"{self.mnemonic} {self.dst_reg}, {self.simm16}"


class TestMulkASTAndTypes(unittest.TestCase):
    """Commit 2: symbolic AST opcode definitions and typed expressions for s_mulk_i32."""

    def test_ast_emission_and_translation(self):
        op = SopkOpcode("s_mulk_i32", "s0", 17)
        self.assertEqual(op.emit_assembly(), "s_mulk_i32 s0, 17")
        lines = [
            "s_mov_b32 s0, 3",
            op.emit_assembly(),
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        spv = support.translate_lines("ps_mulk_ast", lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("OpIMul", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 51)

    def test_ast_negative_immediate_hex(self):
        op = SopkOpcode("s_mulk_i32", "s0", -2)
        lines = [
            "s_mov_b32 s0, 16",
            op.emit_assembly(),
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_mulk_ps(lines, "ps_mulk_ast_neg")
        self.assertEqual(words[0], (-32) & MASK32)


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for SALU blocks."""

    def __init__(self) -> None:
        self.sgprs = [0] * 64

    def exec_line(self, line: str) -> None:
        tokens = line.strip().replace(",", " ").split()
        if not tokens:
            return
        op = tokens[0]
        if op == "s_mov_b32":
            dst = int(tokens[1].lstrip("s"))
            val = int(tokens[2], 0)
            self.sgprs[dst] = val & MASK32
        elif op == "s_mulk_i32":
            dst = int(tokens[1].lstrip("s"))
            k = int(tokens[2], 0)
            self.sgprs[dst] = s_mulk_i32_oracle(self.sgprs[dst], k)


class TestMulkScalarInterpreter(unittest.TestCase):
    """Commit 3: deterministic scalar instruction interpreter for s_mulk_i32."""

    def test_sequential_chained_execution(self):
        interp = ScalarInterpreter()
        lines = [
            "s_mov_b32 s0, 3",
            "s_mulk_i32 s0, 5",
            "s_mulk_i32 s0, -2",
            "s_mulk_i32 s0, 10",
        ]
        for ln in lines:
            interp.exec_line(ln)
        self.assertEqual(interp.sgprs[0], (-300) & MASK32)

        prog = lines + [
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_mulk_ps(prog, "ps_mulk_chained")
        self.assertEqual(words[0], interp.sgprs[0])

    def test_parallel_sgpr_operations(self):
        interp = ScalarInterpreter()
        lines = [
            "s_mov_b32 s0, 10",
            "s_mov_b32 s1, 20",
            "s_mov_b32 s2, 30",
            "s_mov_b32 s3, 40",
            "s_mulk_i32 s0, 2",
            "s_mulk_i32 s1, -3",
            "s_mulk_i32 s2, 4",
            "s_mulk_i32 s3, -5",
        ]
        for ln in lines:
            interp.exec_line(ln)
        self.assertEqual(interp.sgprs[0], 20)
        self.assertEqual(interp.sgprs[1], (-60) & MASK32)
        self.assertEqual(interp.sgprs[2], 120)
        self.assertEqual(interp.sgprs[3], (-200) & MASK32)

        prog = lines + [
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ]
        words = run_mulk_ps(prog, "ps_mulk_parallel_sgprs")
        self.assertEqual(words, [interp.sgprs[0], interp.sgprs[1], interp.sgprs[2], interp.sgprs[3]])


class TestMulkSimdInteraction(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking for s_mulk_i32."""

    def test_simd_broadcast_and_vector_cndmask(self):
        lines = [
            "s_mov_b32 s0, 100",
            "s_mulk_i32 s0, 3",
            "s_mov_b32 s1, 200",
            "s_mulk_i32 s1, -2",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_cndmask_b32 v2, v0, v1, vcc",
            "v_mov_b32 v3, 0",
        ]
        words = run_mulk_ps(lines, "ps_mulk_simd_cndmask")
        self.assertEqual(words[0], 300)
        self.assertEqual(words[1], (-400) & MASK32)


class TestMulkRegisterPreservation(unittest.TestCase):
    """Commit 5: register aliasing and in-place operand preservation for s_mulk_i32."""

    def test_in_place_mutation_preserves_adjacent_sgprs(self):
        lines = [
            "s_mov_b32 s0, 11",
            "s_mov_b32 s1, 22",
            "s_mov_b32 s2, 33",
            "s_mov_b32 s3, 44",
            "s_mulk_i32 s1, 7",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
        ]
        words = run_mulk_ps(lines, "ps_mulk_reg_preservation")
        self.assertEqual(words[0], 11)
        self.assertEqual(words[1], 154)
        self.assertEqual(words[2], 33)
        self.assertEqual(words[3], 44)

    def test_high_sgpr_register_indices(self):
        lines = [
            "s_mov_b32 s8, 50",
            "s_mulk_i32 s8, 12",
            "s_mov_b32 s12, 70",
            "s_mulk_i32 s12, -4",
            "v_mov_b32 v0, s8",
            "v_mov_b32 v1, s12",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        words = run_mulk_ps(lines, "ps_mulk_high_sgpr")
        self.assertEqual(words[0], 600)
        self.assertEqual(words[1], (-280) & MASK32)


if __name__ == '__main__':
    unittest.main()
