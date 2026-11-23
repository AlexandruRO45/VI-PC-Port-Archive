from __future__ import annotations

"""Translator-backed semantic tests for RDNA2 v_max3_i32.

Instruction definition:
    v_max3_i32 vdst, vsrc0, vsrc1, vsrc2 [clamp]

Semantics:
    3-operand signed 32-bit integer maximum (RDNA2 VOP3).
    D.i32 = max(S0.i32, S1.i32, S2.i32)
    Operands are interpreted as two's complement signed 32-bit integers in [-2^31, 2^31 - 1].
    The largest algebraic value is written to the 32-bit destination VGPR.
    Optional clamp modifier clamps negative results to 0.

100% translator-backed execution exercising tools/rdna2spirv.py and support.mrt0_words.
Strictly zero game data.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32: int = 0xFFFF_FFFF
INT32_MIN: int = -2147483648     # -2^31 (0x80000000)
INT32_MAX: int = 2147483647      # 2^31 - 1 (0x7FFFFFFF)


def run_hw_max3_i32(s0: int, s1: int, s2: int, name: str = "ps_max3_i32") -> int:
    """Translates and executes v_max3_i32 via tools/rdna2spirv.py and returns mrt0 dword."""
    lines = [
        f"v_mov_b32_e32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32_e32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32_e32 v3, {s2 & MASK32:#x}",
        "v_max3_i32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestMax3I32BasicExecution(unittest.TestCase):
    def test_all_positive_values(self):
        self.assertEqual(run_hw_max3_i32(10, 20, 30, name="ps_or_pos1"), 30)
        self.assertEqual(run_hw_max3_i32(100, 50, 75, name="ps_or_pos2"), 100)
        self.assertEqual(run_hw_max3_i32(1, 2, 3, name="ps_or_pos3"), 3)

    def test_mixed_signs(self):
        # max(-5, 10, -2) = 10
        self.assertEqual(run_hw_max3_i32(-5, 10, -2, name="ps_or_mix1"), 10)
        # max(-100, -200, 0) = 0
        self.assertEqual(run_hw_max3_i32(-100, -200, 0, name="ps_or_mix2"), 0)
        # max(-1, 0, 1) = 1
        self.assertEqual(run_hw_max3_i32(-1, 0, 1, name="ps_or_mix3"), 1)

    def test_all_negative_values(self):
        # max(-10, -5, -20) = -5 (0xFFFFFFFB)
        self.assertEqual(run_hw_max3_i32(-10, -5, -20, name="ps_or_neg1"), 0xFFFF_FFFB)
        # max(-1, -2, -3) = -1 (0xFFFFFFFF)
        self.assertEqual(run_hw_max3_i32(-1, -2, -3, name="ps_or_neg2"), 0xFFFF_FFFF)
        # max(-50, -100, -75) = -50 (0xFFFFFFCE)
        self.assertEqual(run_hw_max3_i32(-50, -100, -75, name="ps_or_neg3"), 0xFFFF_FFCE)

    def test_identical_operands(self):
        self.assertEqual(run_hw_max3_i32(42, 42, 42, name="ps_or_id1"), 42)
        self.assertEqual(run_hw_max3_i32(-7, -7, -7, name="ps_or_id2"), 0xFFFF_FFF9)
        self.assertEqual(run_hw_max3_i32(0, 0, 0, name="ps_or_id3"), 0)


class TestMax3I32BoundaryExtremes(unittest.TestCase):
    def test_int32_min_max(self):
        # max(INT32_MIN, 0, INT32_MAX) = INT32_MAX
        self.assertEqual(run_hw_max3_i32(INT32_MIN, 0, INT32_MAX, name="ps_or_bnd1"), INT32_MAX & MASK32)
        # max(INT32_MIN, INT32_MIN + 1, -1) = -1
        self.assertEqual(run_hw_max3_i32(INT32_MIN, INT32_MIN + 1, -1, name="ps_or_bnd2"), 0xFFFF_FFFF)
        # max(INT32_MIN, INT32_MIN, INT32_MIN) = INT32_MIN
        self.assertEqual(run_hw_max3_i32(INT32_MIN, INT32_MIN, INT32_MIN, name="ps_or_bnd3"), 0x8000_0000)
        # max(INT32_MAX, INT32_MAX - 1, INT32_MAX - 2) = INT32_MAX
        self.assertEqual(run_hw_max3_i32(INT32_MAX, INT32_MAX - 1, INT32_MAX - 2, name="ps_or_bnd4"), INT32_MAX & MASK32)

    def test_ieee754_bit_patterns_as_signed_integers(self):
        # Raw IEEE 754 float bit patterns interpreted as signed two's complement integers:
        # Float -0.0f: 0x80000000 -> -2147483648 (INT32_MIN)
        # Float +0.0f: 0x00000000 -> 0
        # Float 1.0f:  0x3F800000 -> 1065353216
        # max(-0.0f, +0.0f, 1.0f) = 1.0f (0x3F800000)
        self.assertEqual(run_hw_max3_i32(0x8000_0000, 0x0000_0000, 0x3F80_0000, name="ps_bnd_flt1"), 0x3F80_0000)
        # max(-0.0f, -1, -100) = max(-2147483648, -1, -100) = -1 (0xFFFFFFFF)
        self.assertEqual(run_hw_max3_i32(0x8000_0000, -1, -100, name="ps_bnd_flt2"), 0xFFFF_FFFF)


class TestMax3I32OperandVariants(unittest.TestCase):
    def test_sgpr_and_literal_operands(self):
        lines = [
            "s_mov_b32 s0, 10",
            "v_mov_b32_e32 v1, 20",
            "v_max3_i32 v0, s0, v1, -30",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_op_sgpr_lit", lines)
        self.assertIn("OpExtInst %uint %glsl SMax", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)

    def test_inline_constants(self):
        lines = [
            "v_mov_b32_e32 v1, -10",
            "v_mov_b32_e32 v2, 5",
            "v_max3_i32 v0, 0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_op_inline_const", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 5)

    def test_vop3_three_operand_variations(self):
        # Varying combinations of inputs
        self.assertEqual(run_hw_max3_i32(-10, -20, -5, name="ps_var_neg"), 0xFFFF_FFFB)
        self.assertEqual(run_hw_max3_i32(10, 5, 2, name="ps_var_pos"), 10)


class TestMax3I32RegisterAliasing(unittest.TestCase):
    def test_dst_aliases_src0(self):
        lines = [
            "v_mov_b32_e32 v0, 10",
            "v_mov_b32_e32 v1, 50",
            "v_mov_b32_e32 v2, 20",
            "v_max3_i32 v0, v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_s0", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)

    def test_dst_aliases_src1(self):
        lines = [
            "v_mov_b32_e32 v1, 10",
            "v_mov_b32_e32 v0, 50",
            "v_mov_b32_e32 v2, 20",
            "v_max3_i32 v0, v1, v0, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_s1", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)

    def test_dst_aliases_src2(self):
        lines = [
            "v_mov_b32_e32 v0, 50",
            "v_mov_b32_e32 v1, 10",
            "v_mov_b32_e32 v2, 20",
            "v_max3_i32 v0, v1, v2, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_s2", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)

    def test_dst_aliases_all_sources(self):
        lines = [
            "v_mov_b32_e32 v0, -2",
            "v_max3_i32 v0, v0, v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_all", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFF_FFFE)


class TestMax3I32StatusFlags(unittest.TestCase):
    def test_scc_invariance(self):
        lines = [
            "s_mov_b32 s3, 1",
            "s_cmp_eq_u32 s3, 1",  # sets SCC to 1
            "v_mov_b32_e32 v1, -5",
            "v_mov_b32_e32 v2, 10",
            "v_mov_b32_e32 v3, 2",
            "v_max3_i32 v0, v1, v2, v3",
            "s_cselect_b32 s5, 0xCAFE, 0xBEEF",  # reads SCC: must select 0xCAFE
            "v_mov_b32_e32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_scc_invariance", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 10)
        self.assertEqual(words[1], 0xCAFE)

    def test_exec_mask_invariance(self):
        hw = run_hw_max3_i32(-5, 10, 2, name="ps_exec_invariance")
        self.assertEqual(hw, 10)


class TestMax3I32MutationDetection(unittest.TestCase):
    def test_mutation_detection_smax_to_smin(self):
        lines = [
            "v_mov_b32_e32 v1, -5",
            "v_mov_b32_e32 v2, 10",
            "v_mov_b32_e32 v3, -2",
            "v_max3_i32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_smax", lines)
        correct_word = support.mrt0_words(spv)[0]
        self.assertEqual(correct_word, 10)

        mutated_spv = spv.replace("glsl SMax", "glsl SMin")
        mutated_word = support.mrt0_words(mutated_spv)[0]
        self.assertEqual(mutated_word, 0xFFFF_FFFB)  # -5
        self.assertNotEqual(correct_word, mutated_word)

    def test_mutation_detection_smax_to_umax(self):
        lines = [
            "v_mov_b32_e32 v1, -5",
            "v_mov_b32_e32 v2, 10",
            "v_mov_b32_e32 v3, -2",
            "v_max3_i32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_umax", lines)
        correct_word = support.mrt0_words(spv)[0]
        self.assertEqual(correct_word, 10)

        mutated_spv = spv.replace("glsl SMax", "glsl UMax")
        mutated_word = support.mrt0_words(mutated_spv)[0]
        self.assertEqual(mutated_word, 0xFFFF_FFFE)  # 0xFFFFFFFE (-2) is largest unsigned
        self.assertNotEqual(correct_word, mutated_word)

    def test_translation_matrix(self):
        cases = [
            (-5, 10, -2, 10),
            (-10, -5, -20, 0xFFFF_FFFB),
            (0x8000_0000, 0, 0x7FFF_FFFF, 0x7FFF_FFFF),
            (-1, -1, -1, 0xFFFF_FFFF),
            (100, 200, 300, 300),
            (-50, -10, -20, 0xFFFF_FFF6),
            (0, -100, 50, 50),
        ]
        for s0, s1, s2, expected in cases:
            hw = run_hw_max3_i32(s0, s1, s2, name=f"ps_matrix_{s0}_{s1}_{s2}".replace("-", "n"))
            self.assertEqual(hw, expected)


if __name__ == '__main__':
    unittest.main()
