from __future__ import annotations

"""Semantic reference and translator regression suite for RDNA2 v_min3_u32.

Tests v_min3_u32 (three-operand 32-bit unsigned integer minimum) against mathematical reference
semantics, SIMD divergence execution, and translated SPIR-V shaders via rdna2spirv and spirv_mini_interp.
100% translator-backed execution; zero mock AST or virtual CPU/SIMD models.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import itertools
import os
import struct
import sys
import unittest
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def oracle_min3_u32(a: int, b: int, c: int) -> int:
    """Reference mathematical oracle for RDNA2 v_min3_u32: min(unsigned a, unsigned b, unsigned c)."""
    return min(u32(a), u32(b), u32(c))


_TRANSLATE_ID: int = 0


def run_translated_min3_u32(src0: int, src1: int, src2: int, name: Optional[str] = None) -> int:
    """Execute v_min3_u32 through rdna2spirv translation and spirv_mini_interp execution."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    prog_name = name or f"ps_min3_trans_{_TRANSLATE_ID}"
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_min3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(prog_name, lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


class TestValuMin3U32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_min3_u32."""

    def test_oracle_basic_minimum(self) -> None:
        cases = [
            (10, 20, 30, 10),
            (30, 10, 20, 10),
            (20, 30, 10, 10),
            (0, 0, 0, 0),
            (1, 2, 3, 1),
            (100, 200, 300, 100),
        ]
        for a, b, c, expected in cases:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(oracle_min3_u32(a, b, c), expected)
                hw = run_translated_min3_u32(a, b, c)
                self.assertEqual(hw, expected)

    def test_oracle_unsigned_vs_signed_divergence(self) -> None:
        # In signed math, 0xFFFFFFFF (-1) < 10, but in unsigned math 0xFFFFFFFF is 4294967295 > 10.
        self.assertEqual(oracle_min3_u32(MASK32, 10, 20), 10)
        hw1 = run_translated_min3_u32(MASK32, 10, 20)
        self.assertEqual(hw1, 10)
        self.assertNotEqual(hw1, MASK32)  # SMin would return MASK32 (-1)

        self.assertEqual(oracle_min3_u32(0x8000_0000, 0x7FFF_FFFF, 0), 0)
        hw2 = run_translated_min3_u32(0x8000_0000, 0x7FFF_FFFF, 0)
        self.assertEqual(hw2, 0)

        self.assertEqual(oracle_min3_u32(0x8000_0000, 0x9000_0000, MASK32), 0x8000_0000)
        hw3 = run_translated_min3_u32(0x8000_0000, 0x9000_0000, MASK32)
        self.assertEqual(hw3, 0x8000_0000)
        self.assertNotEqual(hw3, MASK32)  # SMin would return MASK32 (-1)

    def test_oracle_matches_translated_execution(self) -> None:
        triples = [
            (0, 0, 0),
            (10, 20, 30),
            (1000, 2000, 3000),
            (0x12345678, 0x11111111, 0x22222222),
            (MASK32, 10, 20),
            (MASK32, MASK32, 1),
            (0x7FFFFFFF, 1, 0x80000000),
            (0xAAAAAAAA, 0x55555555, 1),
            (0, MASK32, 0x80000000),
            (MASK32, 0, MASK32),
            (MASK32, MASK32, MASK32),
            (1, 1, 1),
        ]
        for a, b, c in triples:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                expected = oracle_min3_u32(a, b, c)
                actual = run_translated_min3_u32(a, b, c)
                self.assertEqual(actual, expected)

    def test_oracle_permutations_and_idempotence(self) -> None:
        a, b, c = 0x1234, 0x5678, 0x9ABC
        res1 = run_translated_min3_u32(a, b, c)
        res2 = run_translated_min3_u32(c, a, b)
        res3 = run_translated_min3_u32(b, c, a)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)
        self.assertEqual(res1, a)

        hw_same = run_translated_min3_u32(a, a, a)
        self.assertEqual(hw_same, a)


class TestValuMin3U32InstructionForms(unittest.TestCase):
    """Stage 2: Instruction operand variations and translator emission for v_min3_u32."""

    def test_instruction_vgpr_operands(self) -> None:
        lines = [
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 10",
            "v_mov_b32 v3, 20",
            "v_min3_u32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inst_vgprs", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 10)

    def test_instruction_with_literals_and_sgprs(self) -> None:
        lines = [
            "s_mov_b32 s0, 200",
            "v_mov_b32 v5, 300",
            "v_min3_u32 v4, s0, 0x100, v5",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inst_sgpr_lit", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 200)

    def test_instruction_inline_constant(self) -> None:
        lines = [
            "v_mov_b32 v1, 50",
            "v_mov_b32 v2, 100",
            "v_min3_u32 v0, 10, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inst_inline", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 10)

    def test_instruction_emits_valid_spirv(self) -> None:
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 50",
            "v_mov_b32 v3, 200",
            "v_min3_u32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_min3", lines)
        self.assertIn("UMin", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)


class TestValuMin3U32SequentialExecution(unittest.TestCase):
    """Stage 3: Sequential and chained instruction shader execution for v_min3_u32."""

    def test_sequential_step_basic(self) -> None:
        hw = run_translated_min3_u32(50, 20, 80)
        self.assertEqual(hw, 20)
        self.assertEqual(hw, oracle_min3_u32(50, 20, 80))

    def test_sequential_step_sgpr_and_imm(self) -> None:
        lines = [
            "s_mov_b32 s4, 100",
            "v_mov_b32 v1, 50",
            "v_min3_u32 v0, s4, 25, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_step_sgpr_imm", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 25)

    def test_sequential_matches_translated_shader(self) -> None:
        test_cases = [
            (10, 20, 30),
            (100, 20, 300),
            (0xFFFF_0000, 0x0000_FFFF, 0x1234),
            (0x8000_0000, 0x4000_0000, 0x7000_0000),
            (0, 0, 0),
            (MASK32, MASK32, 0),
        ]
        for a, b, c in test_cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                trans_res = run_translated_min3_u32(a, b, c)
                self.assertEqual(trans_res, oracle_min3_u32(a, b, c))

    def test_chained_min3_u32_in_shader(self) -> None:
        lines = [
            "v_mov_b32 v1, 50",
            "v_mov_b32 v2, 80",
            "v_mov_b32 v3, 30",
            "v_min3_u32 v4, v1, v2, v3",
            "v_mov_b32 v5, 100",
            "v_min3_u32 v0, v4, v5, 25",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_chained", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 25)
        self.assertEqual(words[0], oracle_min3_u32(oracle_min3_u32(50, 80, 30), 100, 25))


class TestValuMin3U32MultiLaneExecution(unittest.TestCase):
    """Stage 4: Multi-lane divergence sweeps and coverage for v_min3_u32."""

    def test_simd32_lane_sweeps(self) -> None:
        for lane in range(32):
            s0 = 100 + lane
            s1 = 50 + lane * 2
            s2 = 200 - lane
            hw = run_translated_min3_u32(s0, s1, s2)
            expected = oracle_min3_u32(s0, s1, s2)
            self.assertEqual(hw, expected)

    def test_simd64_divergence_sweeps(self) -> None:
        for lane in range(64):
            s0 = lane * 17
            s1 = 500 - lane * 5
            s2 = 250
            hw = run_translated_min3_u32(s0, s1, s2)
            expected = oracle_min3_u32(s0, s1, s2)
            self.assertEqual(hw, expected)

    def test_simd_translator_active_mask_preservation(self) -> None:
        lines = [
            "v_mov_b32 v1, 15",
            "v_mov_b32 v2, 5",
            "v_mov_b32 v3, 25",
            "v_min3_u32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_simd_min3", lines)
        self.assertIn("OpExtInst", spv)
        self.assertIn("UMin", spv)
        self.assertEqual(support.mrt0_words(spv)[0], 5)


class TestValuMin3U32RegisterAliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation for v_min3_u32."""

    def test_in_place_dst_equals_src0(self) -> None:
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 300",
            "v_min3_u32 v1, v1, v2, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_src0", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 20)

    def test_in_place_dst_equals_src1(self) -> None:
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_mov_b32 v3, 30",
            "v_min3_u32 v2, v1, v2, v3",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_src1", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 30)

    def test_in_place_dst_equals_src2(self) -> None:
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 200",
            "v_mov_b32 v3, 300",
            "v_min3_u32 v3, v1, v2, v3",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_src2", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 10)

    def test_in_place_all_operands_aliased(self) -> None:
        lines = [
            "v_mov_b32 v1, 42",
            "v_min3_u32 v1, v1, v1, v1",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_all", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 42)

    def test_in_place_translated_shader(self) -> None:
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 50",
            "v_mov_b32 v3, 300",
            "v_min3_u32 v1, v1, v2, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_inplace_min3", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)

    def test_surrounding_registers_unmodified(self) -> None:
        lines = [
            "v_mov_b32 v0, 0xaaaaaaaa",
            "v_mov_b32 v4, 0xbbbbbbbb",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 30",
            "v_min3_u32 v2, v1, v2, v3",
            "exp mrt0 v0, v2, v4, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_surround_unmod", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xAAAAAAAA)
        self.assertEqual(words[1], 10)
        self.assertEqual(words[2], 0xBBBBBBBB)


class TestValuMin3U32ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and downstream branching for v_min3_u32."""

    def test_downstream_zero_flag_branching(self) -> None:
        lines = [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 10",
            "v_mov_b32 v3, 20",
            "v_min3_u32 v0, v1, v2, v3",
            "v_cmp_eq_u32 vcc, 0, v0",
            "v_cndmask_b32_e64 v4, 0x1111, 0x2222, vcc",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_zero_flag", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x2222)

    def test_downstream_msb_negative_flag_branching(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0x90000000",
            "v_mov_b32 v3, 0xa0000000",
            "v_min3_u32 v0, v1, v2, v3",
            "v_cmp_ge_u32 vcc, v0, 0x80000000",
            "v_cndmask_b32_e64 v4, 0x3333, 0x4444, vcc",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_msb_flag", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x4444)

    def test_downstream_equality_two_way_tie_branching(self) -> None:
        lines = [
            "v_mov_b32 v1, 15",
            "v_mov_b32 v2, 15",
            "v_mov_b32 v3, 30",
            "v_min3_u32 v0, v1, v2, v3",
            "v_cmp_eq_u32 vcc, v0, v1",
            "v_cndmask_b32_e64 v4, 0x5555, 0x6666, vcc",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_two_way_tie", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x6666)

    def test_downstream_three_way_tie_branching(self) -> None:
        lines = [
            "v_mov_b32 v1, 42",
            "v_mov_b32 v2, 42",
            "v_mov_b32 v3, 42",
            "v_min3_u32 v0, v1, v2, v3",
            "v_cmp_eq_u32 vcc, v0, 42",
            "v_cndmask_b32_e64 v4, 0x7777, 0x8888, vcc",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_three_way_tie", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x8888)

    def test_extrema_boundary_status(self) -> None:
        hw = run_translated_min3_u32(MASK32, 0, MASK32)
        self.assertEqual(hw, 0)
        self.assertEqual(hw, oracle_min3_u32(MASK32, 0, MASK32))


class TestValuMin3U32IEEE754EdgeCases(unittest.TestCase):
    """Stage 7: Boundary extremes, subnormals, and NaN handling for v_min3_u32."""

    def test_integer_boundary_extremes(self) -> None:
        extremes = [
            (0, 1, 2, 0),
            (MASK32, MASK32 - 1, MASK32, MASK32 - 1),
            (0x8000_0000, 0x7FFF_FFFF, 0x8000_0001, 0x7FFF_FFFF),
            (0x1000_0000, 0x2000_0000, 0x4000_0000, 0x1000_0000),
            (0, 0, MASK32, 0),
            (MASK32, MASK32, MASK32, MASK32),
            (0, MASK32, 1, 0),
        ]
        for a, b, c, expected in extremes:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(oracle_min3_u32(a, b, c), expected)
                hw = run_translated_min3_u32(a, b, c)
                self.assertEqual(hw, expected)

    def test_float_bitpatterns_treated_as_integers(self) -> None:
        pos_inf = struct.unpack(">I", struct.pack(">f", float("inf")))[0]   # 0x7F80_0000
        neg_inf = struct.unpack(">I", struct.pack(">f", float("-inf")))[0]  # 0xFF80_0000
        qnan = struct.unpack(">I", struct.pack(">f", float("nan")))[0]      # 0x7FC0_0000

        self.assertEqual(oracle_min3_u32(pos_inf, neg_inf, qnan), pos_inf)
        res_trans = run_translated_min3_u32(pos_inf, neg_inf, qnan)
        self.assertEqual(res_trans, pos_inf)

    def test_float_signed_zeros_treated_as_integers(self) -> None:
        pos_zero = 0x0000_0000  # +0.0
        neg_zero = 0x8000_0000  # -0.0
        self.assertEqual(oracle_min3_u32(pos_zero, neg_zero, 100), pos_zero)
        self.assertEqual(run_translated_min3_u32(pos_zero, neg_zero, 100), pos_zero)

    def test_subnormal_bitpatterns_treated_as_integers(self) -> None:
        subnormal_min = 1
        subnormal_max = 0x007F_FFFF
        normal_min = 0x0080_0000
        self.assertEqual(oracle_min3_u32(subnormal_min, subnormal_max, normal_min), subnormal_min)
        self.assertEqual(run_translated_min3_u32(subnormal_min, subnormal_max, normal_min), subnormal_min)


class TestValuMin3U32NegativeControls(unittest.TestCase):
    """Stage 8: Negative controls, syntax rejection, and mutation detection for v_min3_u32."""

    def test_translator_syntax_rejection_missing_op(self) -> None:
        prog = [
            "v_min3_u32 v0, v1, v2",
            "s_endpgm",
        ]
        with self.assertRaises((ValueError, rdna2spirv.Unsupported, IndexError)):
            support.translate_lines("ps_min3_missing_op", prog)

    def test_translator_syntax_rejection_unknown_operand(self) -> None:
        prog = [
            "v_min3_u32 v0, unknown_op, v2, v3",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_min3_unknown_op", prog)

    def test_unsupported_64bit_register_rejected(self) -> None:
        prog = [
            "v_min3_u32 v0, v[1:2], v3, v4",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_min3_64bit", prog)

    def test_translator_structural_umin_lowering(self) -> None:
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 30",
            "v_min3_u32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_min3_struct", lines)
        self.assertIn("UMin", spv)
        self.assertNotIn("SMin", spv)
        self.assertNotIn("UMax", spv)
        self.assertNotIn("SMax", spv)
        self.assertEqual(spv.count("UMin"), 2)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 10)

    def test_mutation_detection_and_oracle_rejection(self) -> None:
        a, b, c = MASK32, 10, 20
        hw = run_translated_min3_u32(a, b, c)
        self.assertEqual(hw, 10)
        self.assertNotEqual(hw, MASK32)
        self.assertNotEqual(hw, s32(a) & MASK32)

        hw2 = run_translated_min3_u32(0x8000_0000, 100, 200)
        self.assertEqual(hw2, 100)
        self.assertNotEqual(hw2, 0x8000_0000)


if __name__ == "__main__":
    unittest.main()
