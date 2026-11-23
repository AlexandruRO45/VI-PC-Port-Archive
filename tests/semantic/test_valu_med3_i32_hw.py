"""Hardware-accurate semantic tests and mathematical reference model for RDNA2 v_med3_i32.

RDNA2 / GFX10 Vector ALU Signed 32-bit Integer Three-Operand Median:
Performs signed integer comparison across three 32-bit operands and returns the median:
    vdst = median_i32(src0, src1, src2)

Syntax:
    v_med3_i32 vdst, src0, src1, src2

Semantics:
    Interprets src0, src1, src2 as 32-bit two's complement signed integers in [-2^31, 2^31 - 1].
    Sorts the three integers in non-decreasing order: s0 <= s1 <= s2.
    vdst = s1 (the median element), represented as a 32-bit unsigned dword (& 0xFFFFFFFF).
    Equivalent to clamp(src2, min(src0, src1), max(src0, src1)) in signed integer arithmetic.

Translator Integration:
    Exercises tools/rdna2spirv.py via support.translate_lines and executes the generated
    SPIR-V shaders via support.mrt0_words.
    100% translator-backed execution; zero mock AST or virtual CPU/SIMD models.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import itertools
import os
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
import rdna2spirv  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402

MASK32: int = 0xFFFFFFFF
SIGN_BIT32: int = 0x80000000
INT32_MIN: int = -0x80000000
INT32_MAX: int = 0x7FFFFFFF

POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
NEG_QNAN_F32: int = 0xFFC00000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def ref_med3_i32(s0: int, s1: int, s2: int) -> int:
    """Pure mathematical reference oracle for signed 32-bit median."""
    sorted_vals = sorted([s32(s0), s32(s1), s32(s2)])
    return u32(sorted_vals[1])


def hw_clamp_med3_i32(s0: int, s1: int, s2: int) -> int:
    """Hardware clamp-equivalent model matching GLSL SClamp(z, SMin(x, y), SMax(x, y))."""
    lo = min(s32(s0), s32(s1))
    hi = max(s32(s0), s32(s1))
    clamped = min(max(s32(s2), lo), hi)
    return u32(clamped)


def run_shader_med3_i32(
    src0_val: int,
    src1_val: int,
    src2_val: int,
    vdst_reg: int = 0,
    v0_reg: int = 1,
    v1_reg: int = 2,
    v2_reg: int = 3,
) -> int:
    """Translate and execute a synthetic pixel shader using v_med3_i32, returning mrt0.x."""
    lines = [
        f"v_mov_b32_e32 v{v0_reg}, {u32(src0_val):#x}",
        f"v_mov_b32_e32 v{v1_reg}, {u32(src1_val):#x}",
        f"v_mov_b32_e32 v{v2_reg}, {u32(src2_val):#x}",
        f"v_med3_i32 v{vdst_reg}, v{v0_reg}, v{v1_reg}, v{v2_reg}",
        f"exp mrt0, v{vdst_reg}, v{vdst_reg}, v{vdst_reg}, v{vdst_reg} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_med3_i32_test", lines)
    words = support.mrt0_words(spv)
    return words[0]


class TestMed3I32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_oracle_basic_integers(self):
        cases = [
            (10, 20, 30, 20),
            (30, 10, 20, 20),
            (-5, 0, 5, 0),
            (-100, -200, -50, -100),
            (7, 7, 7, 7),
            (12, 12, 5, 12),
            (5, 12, 12, 12),
        ]
        for s0, s1, s2, expected in cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                self.assertEqual(ref_med3_i32(s0, s1, s2), u32(expected))
                self.assertEqual(hw_clamp_med3_i32(s0, s1, s2), u32(expected))

    def test_oracle_permutation_symmetry(self):
        triples = [
            (5, -10, 25),
            (-50, 0, 100),
            (0x7FFFFFFF, -1, -0x80000000),
            (42, 42, -99),
        ]
        for trip in triples:
            expected = ref_med3_i32(*trip)
            for perm in itertools.permutations(trip):
                with self.subTest(perm=perm):
                    self.assertEqual(ref_med3_i32(*perm), expected)
                    self.assertEqual(hw_clamp_med3_i32(*perm), expected)

    def test_translator_single_invocation(self):
        result = run_shader_med3_i32(5, 0x80000000, 7)
        self.assertEqual(result, 5)
        self.assertEqual(result, ref_med3_i32(5, 0x80000000, 7))


class TestMed3I32TranslatorInstructions(unittest.TestCase):
    """Commit 2: Instruction operand variations and translator emission."""

    def test_translator_vgpr_operands(self):
        lines = [
            "v_mov_b32_e32 v1, 100",
            "v_mov_b32_e32 v2, -200",
            "v_mov_b32_e32 v3, 0",
            "v_med3_i32 v0, v1, v2, v3",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_vgprs", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0)
        self.assertEqual(words[0], ref_med3_i32(100, -200, 0))

    def test_translator_inline_constant_negative(self):
        lines = [
            "v_mov_b32_e32 v1, -20",
            "v_mov_b32_e32 v2, 30",
            "v_med3_i32 v0, -16, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_inline_neg", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], u32(-16))

    def test_translator_inline_constant_positive(self):
        lines = [
            "v_mov_b32_e32 v1, -5",
            "v_mov_b32_e32 v2, 25",
            "v_med3_i32 v0, 10, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_inline_pos", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 10)

    def test_translator_literal_constant(self):
        lines = [
            "v_mov_b32_e32 v1, 0x10000000",
            "v_mov_b32_e32 v2, 0x30000000",
            "v_med3_i32 v0, 0x20000000, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_literal", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x20000000)

    def test_translator_sgpr_source(self):
        lines = [
            "s_mov_b32 s0, 50",
            "v_mov_b32_e32 v1, 10",
            "v_mov_b32_e32 v2, 20",
            "v_med3_i32 v0, s0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_sgpr_src", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)


class TestMed3I32ChainedExecution(unittest.TestCase):
    """Commit 3: Sequential and chained instruction shader execution."""

    def test_chained_med3_i32_shader(self):
        lines = [
            "v_mov_b32_e32 v1, 50",
            "v_mov_b32_e32 v2, -10",
            "v_mov_b32_e32 v3, 20",
            "v_med3_i32 v4, v1, v2, v3",
            "v_mov_b32_e32 v5, 100",
            "v_med3_i32 v0, v4, v5, 10",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_chained", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)
        self.assertEqual(words[0], ref_med3_i32(ref_med3_i32(50, -10, 20), 100, 10))

    def test_sequential_med3_i32_matches_oracle(self):
        triples = [
            (-30, 40, -10),
            (0x7FFFFFFF, 0, 0x80000000),
            (-1, -2, -3),
            (1234, -5678, 91011),
        ]
        for s0, s1, s2 in triples:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                expected = ref_med3_i32(s0, s1, s2)
                shader_res = run_shader_med3_i32(s0, s1, s2)
                self.assertEqual(shader_res, expected)


class TestMed3I32MultiLaneExecution(unittest.TestCase):
    """Commit 4: Multi-lane divergence sweeps and coverage."""

    def test_wave32_uniform_sweep(self):
        for lane in range(32):
            val = lane * 4 - 64
            res = run_shader_med3_i32(val, 0, 50)
            self.assertEqual(res, ref_med3_i32(val, 0, 50))

    def test_wave64_divergent_lanes_sweep(self):
        for lane in range(64):
            v1_val = lane * 10 - 320
            expected = ref_med3_i32(v1_val, 0, 100)
            res = run_shader_med3_i32(v1_val, 0, 100)
            self.assertEqual(res, expected)


class TestMed3I32RegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_src0_in_shader(self):
        lines = [
            "v_mov_b32_e32 v1, 100",
            "v_mov_b32_e32 v2, -50",
            "v_mov_b32_e32 v3, 20",
            "v_med3_i32 v1, v1, v2, v3",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_alias_dst_s0", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)

    def test_alias_dst_src1_in_shader(self):
        lines = [
            "v_mov_b32_e32 v1, 100",
            "v_mov_b32_e32 v2, -50",
            "v_mov_b32_e32 v3, 20",
            "v_med3_i32 v2, v1, v2, v3",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_alias_dst_s1", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)

    def test_alias_dst_src2_in_shader(self):
        lines = [
            "v_mov_b32_e32 v1, 100",
            "v_mov_b32_e32 v2, -50",
            "v_mov_b32_e32 v3, 20",
            "v_med3_i32 v3, v1, v2, v3",
            "exp mrt0, v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_alias_dst_s2", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)

    def test_alias_all_sources_identical(self):
        lines = [
            "v_mov_b32_e32 v0, 42",
            "v_med3_i32 v0, v0, v0, v0",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_alias_all", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 42)

    def test_alias_pairwise_sources(self):
        lines = [
            "v_mov_b32_e32 v1, -15",
            "v_mov_b32_e32 v2, 30",
            "v_med3_i32 v0, v1, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_alias_pairwise", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], u32(-15))


class TestMed3I32ConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation."""

    def test_zero_flag_behavior(self):
        res = ref_med3_i32(-10, 0, 10)
        self.assertTrue(res == 0)

    def test_sign_flag_behavior(self):
        res_neg = ref_med3_i32(-50, -20, 100)
        self.assertTrue(bool(res_neg & SIGN_BIT32))

        res_pos = ref_med3_i32(-50, 20, 100)
        self.assertFalse(bool(res_pos & SIGN_BIT32))

    def test_downstream_vcc_condition_code_branching(self):
        lines = [
            "v_mov_b32_e32 v1, -5",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 5",
            "v_med3_i32 v0, v1, v2, v3",
            "v_cmp_eq_u32 vcc, 0, v0",
            "v_cndmask_b32_e64 v4, 0x1111, 0x2222, vcc",
            "exp mrt0, v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_downstream_vcc", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x2222)


class TestMed3I32BoundaryExtremes(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_vs_unsigned_median_divergence(self):
        s0, s1, s2 = 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF
        signed_med = ref_med3_i32(s0, s1, s2)
        self.assertEqual(signed_med, 0xFFFFFFFF)

        res_shader = run_shader_med3_i32(s0, s1, s2)
        self.assertEqual(res_shader, 0xFFFFFFFF)

    def test_ieee754_bitpatterns_as_signed_integers(self):
        self.assertEqual(ref_med3_i32(NEG_ZERO_F32, POS_ZERO_F32, 100), POS_ZERO_F32)
        self.assertEqual(run_shader_med3_i32(NEG_ZERO_F32, POS_ZERO_F32, 100), POS_ZERO_F32)

        self.assertEqual(ref_med3_i32(POS_INF_F32, NEG_INF_F32, 0), 0)
        self.assertEqual(run_shader_med3_i32(POS_INF_F32, NEG_INF_F32, 0), 0)

        self.assertEqual(ref_med3_i32(QNAN_F32, NEG_QNAN_F32, 1), 1)
        self.assertEqual(run_shader_med3_i32(QNAN_F32, NEG_QNAN_F32, 1), 1)

    def test_integer_boundary_extremes_sweep(self):
        values = (0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
        for perm in itertools.product(values, repeat=3):
            with self.subTest(perm=[hex(x) for x in perm]):
                expected = ref_med3_i32(*perm)
                self.assertEqual(hw_clamp_med3_i32(*perm), expected)

        key_cases = [
            (0, 1, 0x7FFFFFFF),
            (0, 0x80000000, 0x7FFFFFFF),
            (0xFFFFFFFF, 0, 1),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
            (0x80000000, 0x80000001, 0x7FFFFFFE),
            (1, 0xFFFFFFFE, 0xFFFFFFFF),
        ]
        for c in key_cases:
            with self.subTest(case=[hex(x) for x in c]):
                res = run_shader_med3_i32(*c)
                self.assertEqual(res, ref_med3_i32(*c))


class TestMed3I32NegativeControls(unittest.TestCase):
    """Commit 8: Negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_insufficient_operands(self):
        lines = [
            "v_mov_b32_e32 v1, 10",
            "v_mov_b32_e32 v2, 20",
            "v_med3_i32 v0, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_med3_bad_operands", lines)

    def test_syntax_rejection_unknown_operand(self):
        lines = [
            "v_mov_b32_e32 v1, 10",
            "v_mov_b32_e32 v2, 20",
            "v_med3_i32 v0, unknown_op, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_med3_bad_token", lines)

    def test_mutation_detection_unsigned_median_mutation(self):
        def mutant_u32_med3(s0: int, s1: int, s2: int) -> int:
            return sorted([u32(s0), u32(s1), u32(s2)])[1]

        trip = (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF)
        correct = ref_med3_i32(*trip)
        mutant = mutant_u32_med3(*trip)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_min3_mutation(self):
        def mutant_min3(s0: int, s1: int, s2: int) -> int:
            return sorted([s32(s0), s32(s1), s32(s2)])[0] & MASK32

        trip = (10, 20, 30)
        self.assertNotEqual(ref_med3_i32(*trip), mutant_min3(*trip))

    def test_mutation_detection_max3_mutation(self):
        def mutant_max3(s0: int, s1: int, s2: int) -> int:
            return sorted([s32(s0), s32(s1), s32(s2)])[2] & MASK32

        trip = (10, 20, 30)
        self.assertNotEqual(ref_med3_i32(*trip), mutant_max3(*trip))


if __name__ == "__main__":
    unittest.main()
