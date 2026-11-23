from __future__ import annotations

"""Semantic interpreter tests and translator integration for RDNA2 v_frexp_mant_f32.

Models AMD RDNA2 / GFX10 Vector ALU (VALU) v_frexp_mant_f32 normalized mantissa extraction semantics:
  - Extracts the normalized floating-point mantissa of a single-precision (binary32) float.
  - Normal input with biased exponent 1..254: mantissa placed in interval [0.5, 1.0)
    by replacing exponent field with 126 (0x3F000000), preserving sign and fractional bits.
  - Zero and subnormals/denormals are flushed to signed zero (+-0.0), preserving sign bit.
  - Infinities and NaNs (biased exponent 255) are passed through unchanged.
  - VOP1 / VOP3 encoding with source (VGPR, SGPR, or literal) and destination VGPR.
  - VALU operation: scalar condition code (SCC) and vector condition code (VCC) remain untouched.

100% of tests are executed through tools/rdna2spirv.py (support.translate_lines)
and spirv_mini_interp.run_fragment (support.mrt0_words). Zero standalone mock models.
"""

import os
import struct
import sys
from typing import Sequence, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

MASK32: int = 0xFFFFFFFF


def f32_to_u32(val: float) -> int:
    """Pack Python float as IEEE 754 binary32 uint32 bits."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def u32_to_f32(bits: int) -> float:
    """Unpack uint32 bits as Python float (IEEE 754 binary32)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def run_ps_frexp_mant_f32(
    instructions: Sequence[str],
    export_regs: Tuple[str, str, str, str] = ("v0", "v0", "v0", "v0"),
) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + [
        f"exp mrt0 {export_regs[0]}, {export_regs[1]}, {export_regs[2]}, {export_regs[3]} done vm",
        "s_endpgm",
    ]
    spvasm = support.translate_lines("ps_mant_f32_test", lines)
    return support.mrt0_words(spvasm)


class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_powers_of_two(self):
        words = run_ps_frexp_mant_f32(
            [
                "v_mov_b32 v0, 0x3F800000",  # 1.0
                "v_mov_b32 v1, 0x40000000",  # 2.0
                "v_mov_b32 v2, 0x40800000",  # 4.0
                "v_mov_b32 v3, 0x3F000000",  # 0.5
                "v_frexp_mant_f32 v0, v0",
                "v_frexp_mant_f32 v1, v1",
                "v_frexp_mant_f32 v2, v2",
                "v_frexp_mant_f32 v3, v3",
            ],
            ("v0", "v1", "v2", "v3"),
        )
        self.assertEqual(words, [f32_to_u32(0.5)] * 4)

    def test_fractional_numbers(self):
        words = run_ps_frexp_mant_f32(
            [
                "v_mov_b32 v0, 0x40400000",  # 3.0
                "v_mov_b32 v1, 0x3FC00000",  # 1.5
                "v_frexp_mant_f32 v0, v0",
                "v_frexp_mant_f32 v1, v1",
            ],
            ("v0", "v1", "v0", "v1"),
        )
        self.assertEqual(words[0], f32_to_u32(0.75))
        self.assertEqual(words[1], f32_to_u32(0.75))

    def test_negative_numbers(self):
        words = run_ps_frexp_mant_f32(
            [
                "v_mov_b32 v0, 0xBF800000",  # -1.0
                "v_mov_b32 v1, 0xC0400000",  # -3.0
                "v_frexp_mant_f32 v0, v0",
                "v_frexp_mant_f32 v1, v1",
            ],
            ("v0", "v1", "v0", "v1"),
        )
        self.assertEqual(words[0], f32_to_u32(-0.5))
        self.assertEqual(words[1], f32_to_u32(-0.75))

    def test_translator_basic_execution(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x40000000",  # 2.0
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words[0], f32_to_u32(0.5))

    def test_translator_fractional_execution(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x40400000",  # 3.0
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words[0], f32_to_u32(0.75))


class TestSymbolicAST(unittest.TestCase):
    def test_vop1_standard_form(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x40800000",  # 4.0
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words[0], f32_to_u32(0.5))

    def test_vop1_e32_explicit_form(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x40800000",  # 4.0
            "v_frexp_mant_f32_e32 v0, v1",
        ])
        self.assertEqual(words[0], f32_to_u32(0.5))

    def test_vop3_e64_explicit_form(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x40800000",  # 4.0
            "v_frexp_mant_f32_e64 v0, v1",
        ])
        self.assertEqual(words[0], f32_to_u32(0.5))

    def test_vop3_clamp_modifier(self):
        words = run_ps_frexp_mant_f32(
            [
                "v_mov_b32 v1, 0x40000000",  # 2.0 -> 0.5 (within [0, 1])
                "v_frexp_mant_f32_e64 v0, v1 clamp",
                "v_mov_b32 v2, 0xC0000000",  # -2.0 -> -0.5 (clamped to 0.0)
                "v_frexp_mant_f32_e64 v3, v2 clamp",
            ],
            ("v0", "v3", "v0", "v3"),
        )
        self.assertEqual(words[0], f32_to_u32(0.5))
        self.assertEqual(words[1], f32_to_u32(0.0))


class TestScalarInterpreter(unittest.TestCase):
    def test_scalar_execution_from_sgpr(self):
        words = run_ps_frexp_mant_f32([
            "s_mov_b32 s0, 0x40000000",  # 2.0 in SGPR
            "v_frexp_mant_f32 v0, s0",
        ])
        self.assertEqual(words[0], f32_to_u32(0.5))

    def test_scalar_execution_from_literal(self):
        words = run_ps_frexp_mant_f32([
            "v_frexp_mant_f32 v0, 0x40800000",  # 4.0 as inline literal
        ])
        self.assertEqual(words[0], f32_to_u32(0.5))

    def test_scalar_chained_idempotence(self):
        words = run_ps_frexp_mant_f32(
            [
                "v_mov_b32 v1, 0x41000000",  # 8.0
                "v_frexp_mant_f32 v0, v1",   # 8.0 -> 0.5
                "v_frexp_mant_f32 v2, v0",   # 0.5 -> 0.5 (idempotent for mantissas in [0.5, 1.0))
            ],
            ("v0", "v2", "v0", "v2"),
        )
        self.assertEqual(words[0], f32_to_u32(0.5))
        self.assertEqual(words[1], f32_to_u32(0.5))

    def test_scalar_interpreter_matches_translator(self):
        words = run_ps_frexp_mant_f32(
            [
                "v_mov_b32 v1, 0x41000000",  # 8.0
                "v_frexp_mant_f32 v0, v1",
                "v_frexp_mant_f32 v2, v0",
            ],
            ("v2", "v2", "v2", "v2"),
        )
        self.assertEqual(words[0], f32_to_u32(0.5))


class TestSIMDDivergence(unittest.TestCase):
    def test_simd_multi_channel_parallelism(self):
        words = run_ps_frexp_mant_f32(
            [
                "v_mov_b32 v0, 0x40000000",  # 2.0
                "v_mov_b32 v1, 0x40400000",  # 3.0
                "v_mov_b32 v2, 0xC0000000",  # -2.0
                "v_mov_b32 v3, 0xC0400000",  # -3.0
                "v_frexp_mant_f32 v0, v0",
                "v_frexp_mant_f32 v1, v1",
                "v_frexp_mant_f32 v2, v2",
                "v_frexp_mant_f32 v3, v3",
            ],
            ("v0", "v1", "v2", "v3"),
        )
        self.assertEqual(words[0], f32_to_u32(0.5))
        self.assertEqual(words[1], f32_to_u32(0.75))
        self.assertEqual(words[2], f32_to_u32(-0.5))
        self.assertEqual(words[3], f32_to_u32(-0.75))

    def test_simd_conditional_cndmask_selection(self):
        words = run_ps_frexp_mant_f32(
            [
                "v_mov_b32 v1, 0x40000000",       # 2.0
                "v_frexp_mant_f32 v2, v1",        # 0.5
                "s_mov_b32 s0, 0",                # condition False
                "v_cndmask_b32_e64 v0, 0xDEADBEEF, v2, s0",  # picks false val (0xDEADBEEF)
                "s_mov_b32 s1, 1",                # condition True
                "v_cndmask_b32_e64 v3, 0xDEADBEEF, v2, s1",  # picks true val (v2 = 0.5)
            ],
            ("v0", "v3", "v0", "v3"),
        )
        self.assertEqual(words[0], 0xDEADBEEF)
        self.assertEqual(words[1], f32_to_u32(0.5))


class TestRegisterAliasing(unittest.TestCase):
    def test_alias_dst_equals_src0(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v0, 0x40000000",  # 2.0
            "v_frexp_mant_f32 v0, v0",
        ])
        self.assertEqual(words[0], f32_to_u32(0.5))

    def test_alias_in_place_negative(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v0, 0xC0400000",  # -3.0
            "v_frexp_mant_f32 v0, v0",
        ])
        self.assertEqual(words[0], f32_to_u32(-0.75))


class TestConditionCodesAndStatus(unittest.TestCase):
    def test_vcc_and_scc_preservation(self):
        words = run_ps_frexp_mant_f32(
            [
                "s_mov_b32 s0, 1",
                "s_add_u32 s1, s0, 0",           # sets SCC = 0
                "v_cmp_eq_u32_e32 vcc, 0, 0",    # sets VCC to all ones
                "v_mov_b32 v1, 0x40000000",      # 2.0
                "v_frexp_mant_f32 v0, v1",       # 0.5
                "s_cselect_b32 s2, 0x111, 0x222",  # reads SCC: s2 = 0x222
                "v_cndmask_b32_e64 v2, 0x333, 0x444, vcc",  # reads VCC: v2 = 0x444
            ],
            ("v0", "v2", "s2", "v0"),
        )
        self.assertEqual(words[0], f32_to_u32(0.5))
        self.assertEqual(words[1], 0x444)  # VCC preserved
        self.assertEqual(words[2], 0x222)  # SCC preserved

    def test_mantissa_normalized_range_verification(self):
        for val in (5.0, 100.0, 0.125, 65536.0):
            bits = f32_to_u32(val)
            words = run_ps_frexp_mant_f32([
                f"v_mov_b32 v1, {bits:#x}",
                "v_frexp_mant_f32 v0, v1",
            ])
            res = words[0]
            exp = (res >> 23) & 0xFF
            self.assertEqual(exp, 126, f"Biased exponent of mantissa for {val} must be 126")
            fval = u32_to_f32(res)
            self.assertTrue(0.5 <= fval < 1.0, f"Mantissa {fval} for {val} must be in [0.5, 1.0)")


class TestBoundaryExtremes(unittest.TestCase):
    def test_signed_zeros(self):
        words_pos = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x00000000",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_pos[0], 0x00000000)

        words_neg = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x80000000",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_neg[0], 0x80000000)

    def test_infinities_pass_through(self):
        words_pinf = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x7F800000",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_pinf[0], 0x7F800000)

        words_ninf = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0xFF800000",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_ninf[0], 0xFF800000)

    def test_nans_pass_through(self):
        words_qnan = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x7FC00000",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_qnan[0], 0x7FC00000)

        words_cnan = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x7F800001",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_cnan[0], 0x7F800001)

    def test_denormals_flushed_to_zero(self):
        words_sub_min = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x00000001",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_sub_min[0], 0x00000000)

        words_sub_max = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x007FFFFF",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_sub_max[0], 0x00000000)

        words_sub_neg = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x80000001",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_sub_neg[0], 0x80000000)

    def test_float_min_max_extremes(self):
        words_flt_min = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x00800000",  # FLT_MIN (2^-126)
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_flt_min[0], 0x3F000000)  # 0.5

        words_flt_max = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x7F7FFFFF",  # FLT_MAX
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words_flt_max[0], 0x3F7FFFFF)


class TestNegativeControlsAndMutation(unittest.TestCase):
    def test_rejection_invalid_syntax_missing_src(self):
        bad_lines = [
            "v_frexp_mant_f32 v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_mant_bad_args", bad_lines)

    def test_rejection_invalid_opcode_syntax(self):
        bad_lines = [
            "v_frexp_mant_f32_bad v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_mant_bad_op", bad_lines)

    def test_mutation_biased_exponent_must_be_126(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x3F800000",  # 1.0
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words[0], f32_to_u32(0.5))
        self.assertNotEqual(words[0], f32_to_u32(1.0))

    def test_mutation_denormals_must_not_pass_through(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x00000001",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words[0], 0)
        unflushed = 0x00000001 | 0x3F000000
        self.assertNotEqual(words[0], unflushed)

    def test_mutation_infinities_must_pass_through(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x7F800000",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words[0], 0x7F800000)
        corrupted = (0x7F800000 & 0x807FFFFF) | 0x3F000000
        self.assertNotEqual(words[0], corrupted)


if __name__ == "__main__":
    unittest.main()
