from __future__ import annotations

"""Translator-backed regression tests for RDNA2 Float32 Exponential/Mantissa/Scale Suite.

Covers:
- v_fract_f32: fractional part x - floor(x), denormals flush to zero, inf gives -qNaN, clamped below 1.0.
- v_frexp_exp_i32_f32: binary exponent e - 126 for normal float, 0 for zero/denorm/inf/nan.
- v_frexp_mant_f32: mantissa in [0.5, 1.0) with sign preserved, 0 for zero/denorm, pass-through for inf/nan.
- v_ldexp_f32: computes S0 * 2^S1 with exponent clamped to [-512, 512], overflow to inf, underflow to 0.

Every single test translates real RDNA2 assembly via support.translate_lines and verifies
execution via support.mrt0_words. Zero standalone mock models.
Standard library only (python -I, struct, math, unittest). Zero game data.
"""

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000
EXP_MASK32: int = 0x00FF
EXP_SHIFT32: int = 23
MANT_MASK32: int = 0x007F_FFFF

POS_ZERO: int = 0x0000_0000
NEG_ZERO: int = 0x8000_0000
POS_INF: int = 0x7F80_0000
NEG_INF: int = 0xFF80_0000
QNAN: int = 0x7FC0_0000
SNAN: int = 0x7F80_0001
MIN_NORMAL: int = 0x0080_0000
MAX_NORMAL: int = 0x7F7F_FFFF
LARGEST_BELOW_ONE: int = 0x3F7F_FFFF


def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def run_fract(val: int) -> int:
    prog = [
        f"v_mov_b32 v1, {val:#x}",
        "v_fract_f32 v0, v1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_fract", prog)
    return support.mrt0_words(spv)[0]


def run_frexp_exp(val: int) -> int:
    prog = [
        f"v_mov_b32 v1, {val:#x}",
        "v_frexp_exp_i32_f32 v0, v1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_frexp_exp", prog)
    return support.mrt0_words(spv)[0]


def run_frexp_mant(val: int) -> int:
    prog = [
        f"v_mov_b32 v1, {val:#x}",
        "v_frexp_mant_f32 v0, v1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_frexp_mant", prog)
    return support.mrt0_words(spv)[0]


def run_ldexp(s0: int, s1: int) -> int:
    prog = [
        f"v_mov_b32 v1, {s0:#x}",
        f"v_mov_b32 v2, {s1}",
        "v_ldexp_f32 v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_ldexp", prog)
    return support.mrt0_words(spv)[0]


# ------------------------------------------------------------------------------
# v_fract_f32 Tests
# ------------------------------------------------------------------------------

class TestFractF32Oracle(unittest.TestCase):
    def test_positive_fractions(self):
        self.assertEqual(run_fract(f32_to_bits(1.25)), f32_to_bits(0.25))
        self.assertEqual(run_fract(f32_to_bits(5.75)), f32_to_bits(0.75))
        self.assertEqual(run_fract(f32_to_bits(10.0)), 0)

    def test_negative_fractions(self):
        self.assertEqual(run_fract(f32_to_bits(-0.25)), f32_to_bits(0.75))
        self.assertEqual(run_fract(f32_to_bits(-1.5)), f32_to_bits(0.5))

    def test_special_cases(self):
        self.assertEqual(run_fract(POS_ZERO), 0)
        self.assertEqual(run_fract(NEG_ZERO), 0)
        self.assertEqual(run_fract(POS_INF), 0xFFC0_0000)
        self.assertEqual(run_fract(NEG_INF), 0xFFC0_0000)
        self.assertEqual(run_fract(QNAN), QNAN)
        self.assertEqual(run_fract(0x0000_0001), 0)


class TestFractF32AST(unittest.TestCase):
    def test_ast_construction(self):
        prog = [
            "v_mov_b32 v1, 0x3f800000",
            "v_fract_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_fract_ast", prog)
        self.assertIn("Fract", spv)

    def test_ast_invalid_dst(self):
        prog = [
            "v_fract_f32 v0, v[0:1]",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_fract_bad_op", prog)


class TestFractF32ScalarInterpreter(unittest.TestCase):
    def test_scalar_fract_execution(self):
        prog = [
            f"v_mov_b32 v2, {f32_to_bits(3.75):#x}",
            "v_fract_f32 v0, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_fract_scalar", prog)
        self.assertEqual(support.mrt0_words(spv)[0], f32_to_bits(0.75))


class TestFractF32SIMDDivergence(unittest.TestCase):
    def test_divergence_and_exec_masking(self):
        prog = [
            f"v_mov_b32 v1, {f32_to_bits(1.125):#x}",
            f"v_mov_b32 v2, {f32_to_bits(2.25):#x}",
            f"v_mov_b32 v3, {f32_to_bits(3.5):#x}",
            f"v_mov_b32 v4, {f32_to_bits(4.75):#x}",
            "v_fract_f32 v5, v1",
            "v_fract_f32 v6, v2",
            "v_fract_f32 v7, v3",
            "v_fract_f32 v8, v4",
            "exp mrt0 v5, v6, v7, v8 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_fract_simd", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words, [f32_to_bits(0.125), f32_to_bits(0.25), f32_to_bits(0.5), f32_to_bits(0.75)])


class TestFractF32IEEE754Handling(unittest.TestCase):
    def test_clamp_below_one_on_tiny_negative(self):
        tiny_neg = f32_to_bits(-1e-9)
        self.assertEqual(run_fract(tiny_neg), LARGEST_BELOW_ONE)

    def test_subnormals_flush_to_zero(self):
        for sub in (0x0000_0001, 0x0040_0000, 0x007F_FFFF, 0x8000_0001, 0x807F_FFFF):
            with self.subTest(sub=hex(sub)):
                self.assertEqual(run_fract(sub), 0)

    def test_nans_pass_through(self):
        for nan in (QNAN, SNAN, 0x7FC0_1234, 0xFFC0_ABCD):
            with self.subTest(nan=hex(nan)):
                self.assertEqual(run_fract(nan), nan)


class TestFractF32RejectionAndHardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        prog = [
            "v_fract_f32 v0, v1 bad_modifier",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_fract_syntax_rej", prog)

    def test_hardware_translator_execution(self):
        cases = [
            (f32_to_bits(2.25), f32_to_bits(0.25)),
            (f32_to_bits(-0.25), f32_to_bits(0.75)),
            (POS_ZERO, 0),
            (NEG_ZERO, 0),
            (POS_INF, 0xFFC0_0000),
            (NEG_INF, 0xFFC0_0000),
            (QNAN, QNAN),
            (0x0000_0001, 0),
        ]
        for src_val, expected in cases:
            with self.subTest(src=hex(src_val)):
                self.assertEqual(run_fract(src_val), expected)


# ------------------------------------------------------------------------------
# v_frexp_exp_i32_f32 Tests
# ------------------------------------------------------------------------------

class TestFrexpExpI32F32Oracle(unittest.TestCase):
    def test_powers_of_two(self):
        self.assertEqual(run_frexp_exp(f32_to_bits(1.0)), 1)
        self.assertEqual(run_frexp_exp(f32_to_bits(0.5)), 0)
        self.assertEqual(run_frexp_exp(f32_to_bits(2.0)), 2)
        self.assertEqual(run_frexp_exp(f32_to_bits(0.25)), 0xFFFF_FFFF)

    def test_sign_invariance(self):
        self.assertEqual(run_frexp_exp(f32_to_bits(-1.0)), 1)
        self.assertEqual(run_frexp_exp(f32_to_bits(-2.0)), 2)

    def test_specials_yield_zero(self):
        self.assertEqual(run_frexp_exp(POS_ZERO), 0)
        self.assertEqual(run_frexp_exp(NEG_ZERO), 0)
        self.assertEqual(run_frexp_exp(POS_INF), 0)
        self.assertEqual(run_frexp_exp(NEG_INF), 0)
        self.assertEqual(run_frexp_exp(QNAN), 0)
        self.assertEqual(run_frexp_exp(0x0000_0001), 0)


class TestFrexpExpI32F32AST(unittest.TestCase):
    def test_ast_construction(self):
        prog = [
            "v_mov_b32 v1, 0x3f800000",
            "v_frexp_exp_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_exp_ast", prog)
        self.assertIn("OpBitFieldUExtract", spv)


class TestFrexpExpI32F32ScalarInterpreter(unittest.TestCase):
    def test_scalar_execution(self):
        prog = [
            f"v_mov_b32 v1, {f32_to_bits(4.0):#x}",
            "v_frexp_exp_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_exp_scalar", prog)
        self.assertEqual(support.mrt0_words(spv)[0], 3)


class TestFrexpExpI32F32SIMDDivergence(unittest.TestCase):
    def test_simd_execution(self):
        prog = [
            f"v_mov_b32 v1, {f32_to_bits(1.0):#x}",
            f"v_mov_b32 v2, {f32_to_bits(2.0):#x}",
            f"v_mov_b32 v3, {f32_to_bits(4.0):#x}",
            f"v_mov_b32 v4, {f32_to_bits(0.5):#x}",
            "v_frexp_exp_i32_f32 v5, v1",
            "v_frexp_exp_i32_f32 v6, v2",
            "v_frexp_exp_i32_f32 v7, v3",
            "v_frexp_exp_i32_f32 v8, v4",
            "exp mrt0 v5, v6, v7, v8 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_exp_simd", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words, [1, 2, 3, 0])


class TestFrexpExpI32F32IEEE754Handling(unittest.TestCase):
    def test_boundary_normals(self):
        self.assertEqual(run_frexp_exp(MIN_NORMAL), 0xFFFF_FF83)
        self.assertEqual(run_frexp_exp(MAX_NORMAL), 128)

    def test_all_special_cases_zero(self):
        for u in (POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, QNAN, SNAN, 0x0000_0001, 0x007F_FFFF):
            with self.subTest(val=hex(u)):
                self.assertEqual(run_frexp_exp(u), 0)


class TestFrexpExpI32F32RejectionAndHardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        prog = [
            "v_frexp_exp_i32_f32 v0, v[0:1]",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_frexp_exp_rej", prog)

    def test_hardware_translator_execution(self):
        cases = [
            (f32_to_bits(1.0), 1),
            (f32_to_bits(0.5), 0),
            (f32_to_bits(2.0), 2),
            (f32_to_bits(0.25), 0xFFFF_FFFF),
            (POS_ZERO, 0),
            (POS_INF, 0),
            (QNAN, 0),
            (0x0000_0001, 0),
            (MIN_NORMAL, 0xFFFF_FF83),
            (MAX_NORMAL, 128),
        ]
        for src_val, expected in cases:
            with self.subTest(src=hex(src_val)):
                self.assertEqual(run_frexp_exp(src_val), expected)


# ------------------------------------------------------------------------------
# v_frexp_mant_f32 Tests
# ------------------------------------------------------------------------------

class TestFrexpMantF32Oracle(unittest.TestCase):
    def test_powers_of_two(self):
        self.assertEqual(run_frexp_mant(f32_to_bits(1.0)), f32_to_bits(0.5))
        self.assertEqual(run_frexp_mant(f32_to_bits(2.0)), f32_to_bits(0.5))
        self.assertEqual(run_frexp_mant(f32_to_bits(0.25)), f32_to_bits(0.5))

    def test_negative_normals_preserve_sign(self):
        self.assertEqual(run_frexp_mant(f32_to_bits(-1.0)), f32_to_bits(-0.5))
        self.assertEqual(run_frexp_mant(f32_to_bits(-4.0)), f32_to_bits(-0.5))

    def test_special_cases(self):
        self.assertEqual(run_frexp_mant(POS_ZERO), POS_ZERO)
        self.assertEqual(run_frexp_mant(NEG_ZERO), NEG_ZERO)
        self.assertEqual(run_frexp_mant(POS_INF), POS_INF)
        self.assertEqual(run_frexp_mant(NEG_INF), NEG_INF)
        self.assertEqual(run_frexp_mant(QNAN), QNAN)
        self.assertEqual(run_frexp_mant(0x0000_0001), POS_ZERO)
        self.assertEqual(run_frexp_mant(0x8000_0001), NEG_ZERO)


class TestFrexpMantF32AST(unittest.TestCase):
    def test_ast_construction(self):
        prog = [
            "v_mov_b32 v1, 0x3f800000",
            "v_frexp_mant_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_mant_ast", prog)
        self.assertIn("OpBitwiseOr", spv)


class TestFrexpMantF32ScalarInterpreter(unittest.TestCase):
    def test_scalar_execution(self):
        prog = [
            f"v_mov_b32 v1, {f32_to_bits(16.0):#x}",
            "v_frexp_mant_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_mant_scalar", prog)
        self.assertEqual(support.mrt0_words(spv)[0], f32_to_bits(0.5))


class TestFrexpMantF32SIMDDivergence(unittest.TestCase):
    def test_simd_execution(self):
        prog = [
            f"v_mov_b32 v1, {f32_to_bits(1.0):#x}",
            f"v_mov_b32 v2, {f32_to_bits(-2.0):#x}",
            f"v_mov_b32 v3, {f32_to_bits(3.0):#x}",
            f"v_mov_b32 v4, {f32_to_bits(-0.25):#x}",
            "v_frexp_mant_f32 v5, v1",
            "v_frexp_mant_f32 v6, v2",
            "v_frexp_mant_f32 v7, v3",
            "v_frexp_mant_f32 v8, v4",
            "exp mrt0 v5, v6, v7, v8 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_mant_simd", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words, [f32_to_bits(0.5), f32_to_bits(-0.5), f32_to_bits(0.75), f32_to_bits(-0.5)])


class TestFrexpMantF32IEEE754Handling(unittest.TestCase):
    def test_subnormals_flush_to_signed_zero(self):
        self.assertEqual(run_frexp_mant(0x0000_0001), POS_ZERO)
        self.assertEqual(run_frexp_mant(0x007F_FFFF), POS_ZERO)
        self.assertEqual(run_frexp_mant(0x8000_0001), NEG_ZERO)
        self.assertEqual(run_frexp_mant(0x807F_FFFF), NEG_ZERO)

    def test_inf_and_nan_pass_through(self):
        for val in (POS_INF, NEG_INF, QNAN, SNAN, 0x7FC0_FFFF):
            with self.subTest(val=hex(val)):
                self.assertEqual(run_frexp_mant(val), val)


class TestFrexpMantF32RejectionAndHardwareTranslator(unittest.TestCase):
    def test_hardware_translator_execution(self):
        cases = [
            (f32_to_bits(1.0), f32_to_bits(0.5)),
            (f32_to_bits(2.0), f32_to_bits(0.5)),
            (f32_to_bits(-1.0), f32_to_bits(-0.5)),
            (POS_ZERO, POS_ZERO),
            (NEG_ZERO, NEG_ZERO),
            (POS_INF, POS_INF),
            (NEG_INF, NEG_INF),
            (QNAN, QNAN),
            (0x0000_0001, POS_ZERO),
            (0x8000_0001, NEG_ZERO),
        ]
        for src_val, expected in cases:
            with self.subTest(src=hex(src_val)):
                self.assertEqual(run_frexp_mant(src_val), expected)


# ------------------------------------------------------------------------------
# v_ldexp_f32 Tests
# ------------------------------------------------------------------------------

class TestLdexpF32Oracle(unittest.TestCase):
    def test_basic_scaling(self):
        self.assertEqual(run_ldexp(f32_to_bits(0.5), 2), f32_to_bits(2.0))
        self.assertEqual(run_ldexp(f32_to_bits(1.0), -1), f32_to_bits(0.5))

    def test_overflow_and_underflow(self):
        self.assertEqual(run_ldexp(f32_to_bits(1.0), 200), POS_INF)
        self.assertEqual(run_ldexp(f32_to_bits(1.0), -200), POS_ZERO)
        self.assertEqual(run_ldexp(f32_to_bits(-1.0), -200), NEG_ZERO)

    def test_specials_invariance(self):
        self.assertEqual(run_ldexp(POS_ZERO, 10), POS_ZERO)
        self.assertEqual(run_ldexp(NEG_ZERO, 10), NEG_ZERO)
        self.assertEqual(run_ldexp(POS_INF, -5), POS_INF)
        self.assertEqual(run_ldexp(QNAN, 5), QNAN)


class TestLdexpF32AST(unittest.TestCase):
    def test_ast_construction(self):
        prog = [
            "v_mov_b32 v1, 0x3f800000",
            "v_mov_b32 v2, 1",
            "v_ldexp_f32 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_ast", prog)
        self.assertIn("OpExtInst", spv)


class TestLdexpF32ScalarInterpreter(unittest.TestCase):
    def test_scalar_execution(self):
        prog = [
            f"v_mov_b32 v1, {f32_to_bits(1.0):#x}",
            "v_mov_b32 v2, 3",
            "v_ldexp_f32 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_scalar", prog)
        self.assertEqual(support.mrt0_words(spv)[0], f32_to_bits(8.0))


class TestLdexpF32SIMDDivergence(unittest.TestCase):
    def test_simd_execution(self):
        prog = [
            f"v_mov_b32 v1, {f32_to_bits(1.0):#x}",
            "v_mov_b32 v2, 1",
            "v_mov_b32 v3, 2",
            "v_mov_b32 v4, -1",
            "v_mov_b32 v5, 0",
            "v_ldexp_f32 v6, v1, v2",
            "v_ldexp_f32 v7, v1, v3",
            "v_ldexp_f32 v8, v1, v4",
            "v_ldexp_f32 v9, v1, v5",
            "exp mrt0 v6, v7, v8, v9 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_simd", prog)
        words = support.mrt0_words(spv)
        self.assertEqual(words, [f32_to_bits(2.0), f32_to_bits(4.0), f32_to_bits(0.5), f32_to_bits(1.0)])


class TestLdexpF32IEEE754Handling(unittest.TestCase):
    def test_extreme_shift_clamping(self):
        self.assertEqual(run_ldexp(f32_to_bits(1.0), 1000), POS_INF)
        self.assertEqual(run_ldexp(f32_to_bits(1.0), -1000), POS_ZERO)
        self.assertEqual(run_ldexp(f32_to_bits(-1.0), 1000), NEG_INF)
        self.assertEqual(run_ldexp(f32_to_bits(-1.0), -1000), NEG_ZERO)

    def test_subnormal_inputs_flush_to_zero(self):
        self.assertEqual(run_ldexp(0x0000_0001, 10), POS_ZERO)
        self.assertEqual(run_ldexp(0x8000_0001, 10), NEG_ZERO)


class TestLdexpF32RejectionAndHardwareTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        prog = [
            "v_ldexp_f32 v0, v[0:1], v2",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_ldexp_rej", prog)

    def test_hardware_translator_execution(self):
        cases = [
            (f32_to_bits(0.5), 2, f32_to_bits(2.0)),
            (f32_to_bits(1.0), -1, f32_to_bits(0.5)),
            (f32_to_bits(1.0), 200, POS_INF),
            (f32_to_bits(1.0), -200, POS_ZERO),
            (f32_to_bits(-1.0), -200, NEG_ZERO),
            (POS_ZERO, 10, POS_ZERO),
            (NEG_ZERO, 10, NEG_ZERO),
            (POS_INF, -5, POS_INF),
            (QNAN, 5, QNAN),
            (0x0000_0001, 5, POS_ZERO),
        ]
        for s0, s1, expected in cases:
            with self.subTest(s0=hex(s0), s1=s1):
                self.assertEqual(run_ldexp(s0, s1), expected)


if __name__ == '__main__':
    unittest.main()
