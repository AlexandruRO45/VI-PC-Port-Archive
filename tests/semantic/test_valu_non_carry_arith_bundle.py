"""Bundle 44: VALU Non-Carry Integer Arithmetic Suite.

Tests for 5 VALU integer arithmetic instructions without carry:
- v_add_nc_u32: 32-bit unsigned addition without carry (wrapping or saturated clamp)
- v_add_nc_i32: 32-bit signed addition without carry (wrapping or signed saturated clamp)
- v_sub_nc_u32: 32-bit unsigned subtraction without borrow (wrapping or saturated clamp)
- v_sub_nc_i32: 32-bit signed subtraction without borrow (wrapping or signed saturated clamp)
- v_subrev_nc_u32: 32-bit unsigned reverse subtraction without borrow (src1 - src0)

Standard library only; zero game data.
"""
from __future__ import annotations

import os
import sys
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

MASK32 = 0xFFFFFFFF


def run_bundle_ps(lines: list[str], name: str = "ps_nc_arith") -> list[int]:
    """Helper to translate and run a pixel shader; returns MRT0 four 32-bit words."""
    all_lines = list(lines) + [
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


def oracle_add_nc_u32(s0: int, s1: int, clamp: bool = False) -> int:
    tot = (s0 & MASK32) + (s1 & MASK32)
    if clamp:
        return min(tot, MASK32)
    return tot & MASK32


def oracle_add_nc_i32(s0: int, s1: int, clamp: bool = False) -> int:
    s0, s1 = s0 & MASK32, s1 & MASK32
    i0 = s0 - 0x100000000 if s0 & 0x80000000 else s0
    i1 = s1 - 0x100000000 if s1 & 0x80000000 else s1
    tot = i0 + i1
    if clamp:
        clamped = max(-0x80000000, min(tot, 0x7FFFFFFF))
        return clamped & MASK32
    return tot & MASK32


def oracle_sub_nc_u32(s0: int, s1: int, clamp: bool = False) -> int:
    s0, s1 = s0 & MASK32, s1 & MASK32
    if clamp:
        return max(0, s0 - s1)
    return (s0 - s1) & MASK32


def oracle_sub_nc_i32(s0: int, s1: int, clamp: bool = False) -> int:
    s0, s1 = s0 & MASK32, s1 & MASK32
    i0 = s0 - 0x100000000 if s0 & 0x80000000 else s0
    i1 = s1 - 0x100000000 if s1 & 0x80000000 else s1
    diff = i0 - i1
    if clamp:
        clamped = max(-0x80000000, min(diff, 0x7FFFFFFF))
        return clamped & MASK32
    return diff & MASK32


def oracle_subrev_nc_u32(s0: int, s1: int, clamp: bool = False) -> int:
    s0, s1 = s0 & MASK32, s1 & MASK32
    if clamp:
        return max(0, s1 - s0)
    return (s1 - s0) & MASK32


# -----------------------------------------------------------------------------
# 1. v_add_nc_u32
# -----------------------------------------------------------------------------

class TestVAddNcU32Oracle(unittest.TestCase):
    def test_basic_addition(self) -> None:
        self.assertEqual(oracle_add_nc_u32(100, 25), 125)
        self.assertEqual(oracle_add_nc_u32(0, 0), 0)
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 25",
            "v_add_nc_u32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_add_u32_basic")
        self.assertEqual(words[0], 125)

    def test_wrapping_behavior(self) -> None:
        self.assertEqual(oracle_add_nc_u32(MASK32, 1), 0)
        self.assertEqual(oracle_add_nc_u32(MASK32, 5), 4)
        lines = [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_mov_b32 v2, 5",
            "v_add_nc_u32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_add_u32_wrap")
        self.assertEqual(words[0], 4)

    def test_clamp_saturation(self) -> None:
        self.assertEqual(oracle_add_nc_u32(MASK32, 1, clamp=True), MASK32)
        self.assertEqual(oracle_add_nc_u32(100, 20, clamp=True), 120)
        lines = [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_mov_b32 v2, 1",
            "v_add_nc_u32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(lines, "ps_add_u32_clamp")
        self.assertEqual(words[0], MASK32)


class TestVAddNcU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x1000",
            "v_mov_b32 v2, 0x2000",
            "v_add_nc_u32 v0, v1, v2",
        ]
        spv = support.translate_lines("ps_ast_add_u32", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpIAdd", spv)
        words = run_bundle_ps(lines, "ps_ast_add_u32_exec")
        self.assertEqual(words[0], 0x3000)

    def test_ast_clamp_evaluation(self) -> None:
        lines = [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_mov_b32 v2, 10",
            "v_add_nc_u32 v0, v1, v2 clamp",
        ]
        spv = support.translate_lines("ps_ast_add_u32_cl", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpIAddCarry", spv)
        words = run_bundle_ps(lines, "ps_ast_add_u32_cl_exec")
        self.assertEqual(words[0], MASK32)


class TestVAddNcU32WrappingAndSaturation(unittest.TestCase):
    def test_modular_wrapping_patterns(self) -> None:
        for offset in (0, 1, 2, 0xFFFF, 0x10000):
            res = oracle_add_nc_u32(MASK32 - offset, offset + 1)
            self.assertEqual(res, 0)
            lines = [
                f"v_mov_b32 v1, {MASK32 - offset:#x}",
                f"v_mov_b32 v2, {offset + 1:#x}",
                "v_add_nc_u32 v0, v1, v2",
            ]
            words = run_bundle_ps(lines, f"ps_add_u32_pat_{offset}")
            self.assertEqual(words[0], 0)

    def test_saturation_extremes(self) -> None:
        self.assertEqual(oracle_add_nc_u32(0x80000000, 0x80000000, clamp=True), MASK32)
        self.assertEqual(oracle_add_nc_u32(0x80000000, 0x80000000, clamp=False), 0)
        lines_cl = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0x80000000",
            "v_add_nc_u32 v0, v1, v2 clamp",
        ]
        words_cl = run_bundle_ps(lines_cl, "ps_add_u32_sat_ext")
        self.assertEqual(words_cl[0], MASK32)

        lines_wrap = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0x80000000",
            "v_add_nc_u32 v0, v1, v2",
        ]
        words_wrap = run_bundle_ps(lines_wrap, "ps_add_u32_wrap_ext")
        self.assertEqual(words_wrap[0], 0)


class TestVAddNcU32SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        for l in (0, 1, 7, 15, 16, 31):
            s0 = l * 100
            s1 = l * 20
            expected = oracle_add_nc_u32(s0, s1)
            lines = [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_add_nc_u32 v0, v1, v2",
            ]
            words = run_bundle_ps(lines, f"ps_simd_add_u32_{l}")
            self.assertEqual(words[0], expected)


class TestVAddNcU32TranslatorAndControls(unittest.TestCase):
    def test_translator_v_add_nc_u32_execution(self) -> None:
        body = [
            "v_mov_b32 v1, 1000",
            "v_mov_b32 v2, 337",
            "v_add_nc_u32_e32 v0, v1, v2",
        ]
        words = run_bundle_ps(body, "ps_v_add_nc_u32_e32")
        self.assertEqual(words[0], 1337)

    def test_translator_v_add_nc_u32_clamp(self) -> None:
        body = [
            "v_mov_b32 v1, 0xffffffff",
            "v_mov_b32 v2, 50",
            "v_add_nc_u32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(body, "ps_v_add_nc_u32_cl")
        self.assertEqual(words[0], 0xFFFFFFFF)

    def test_v_add_nc_u32_negative_controls(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_add_nc_u32", ["v_add_nc_u32", "s_endpgm"])

    def test_v_add_nc_u32_mutation_contrast(self) -> None:
        lines = [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_mov_b32 v2, 10",
            "v_add_nc_u32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(lines, "ps_mut_add_u32")
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertNotEqual(words[0], 9)  # without clamp would wrap to 9


# -----------------------------------------------------------------------------
# 2. v_add_nc_i32
# -----------------------------------------------------------------------------

class TestVAddNcI32Oracle(unittest.TestCase):
    def test_basic_addition(self) -> None:
        self.assertEqual(oracle_add_nc_i32(50, -20), 30)
        self.assertEqual(oracle_add_nc_i32(-50, -20), (-70) & MASK32)
        lines = [
            "v_mov_b32 v1, 50",
            f"v_mov_b32 v2, {(-20) & MASK32:#x}",
            "v_add_nc_i32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_add_i32_basic")
        self.assertEqual(words[0], 30)

    def test_wrapping_signed(self) -> None:
        self.assertEqual(oracle_add_nc_i32(0x7FFFFFFF, 1), 0x80000000)
        lines = [
            "v_mov_b32 v1, 0x7FFFFFFF",
            "v_mov_b32 v2, 1",
            "v_add_nc_i32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_add_i32_wrap")
        self.assertEqual(words[0], 0x80000000)

    def test_clamp_signed(self) -> None:
        self.assertEqual(oracle_add_nc_i32(0x7FFFFFFF, 10, clamp=True), 0x7FFFFFFF)
        self.assertEqual(oracle_add_nc_i32(0x80000000, -10, clamp=True), 0x80000000)
        lines_pos = [
            "v_mov_b32 v1, 0x7FFFFFFF",
            "v_mov_b32 v2, 10",
            "v_add_nc_i32 v0, v1, v2 clamp",
        ]
        words_pos = run_bundle_ps(lines_pos, "ps_add_i32_cl_pos")
        self.assertEqual(words_pos[0], 0x7FFFFFFF)

        lines_neg = [
            "v_mov_b32 v1, 0x80000000",
            f"v_mov_b32 v2, {(-10) & MASK32:#x}",
            "v_add_nc_i32 v0, v1, v2 clamp",
        ]
        words_neg = run_bundle_ps(lines_neg, "ps_add_i32_cl_neg")
        self.assertEqual(words_neg[0], 0x80000000)


class TestVAddNcI32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        lines = [
            "v_mov_b32 v1, 500",
            f"v_mov_b32 v2, {(-200) & MASK32:#x}",
            "v_add_nc_i32 v0, v1, v2",
        ]
        spv = support.translate_lines("ps_ast_add_i32", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpIAdd", spv)
        words = run_bundle_ps(lines, "ps_ast_add_i32_exec")
        self.assertEqual(words[0], 300)

    def test_ast_clamp_evaluation(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x7FFFFFFF",
            "v_mov_b32 v2, 1",
            "v_add_nc_i32 v0, v1, v2 clamp",
        ]
        spv = support.translate_lines("ps_ast_add_i32_cl", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpSLessThan", spv)
        words = run_bundle_ps(lines, "ps_ast_add_i32_cl_exec")
        self.assertEqual(words[0], 0x7FFFFFFF)


class TestVAddNcI32WrappingAndSaturation(unittest.TestCase):
    def test_positive_and_negative_extremes(self) -> None:
        # Max positive + 1 without clamp wraps to min negative
        self.assertEqual(oracle_add_nc_i32(0x7FFFFFFF, 1, clamp=False), 0x80000000)
        lines_pos = [
            "v_mov_b32 v1, 0x7FFFFFFF",
            "v_mov_b32 v2, 1",
            "v_add_nc_i32 v0, v1, v2",
        ]
        words_pos = run_bundle_ps(lines_pos, "ps_add_i32_pos_wrap")
        self.assertEqual(words_pos[0], 0x80000000)

        # Min negative + (-1) without clamp wraps to max positive
        self.assertEqual(oracle_add_nc_i32(0x80000000, 0xFFFFFFFF, clamp=False), 0x7FFFFFFF)
        lines_neg = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_add_nc_i32 v0, v1, v2",
        ]
        words_neg = run_bundle_ps(lines_neg, "ps_add_i32_neg_wrap")
        self.assertEqual(words_neg[0], 0x7FFFFFFF)


class TestVAddNcI32SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        for l in (0, 1, 7, 15, 16, 31):
            s0 = (l - 16) * 10
            s1 = 50
            expected = oracle_add_nc_i32(s0, s1)
            lines = [
                f"v_mov_b32 v1, {s0 & MASK32:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_add_nc_i32 v0, v1, v2",
            ]
            words = run_bundle_ps(lines, f"ps_simd_add_i32_{l}")
            self.assertEqual(words[0], expected)


class TestVAddNcI32TranslatorAndControls(unittest.TestCase):
    def test_translator_v_add_nc_i32_execution(self) -> None:
        body = [
            "v_mov_b32 v1, 0x7fffffff",
            "v_mov_b32 v2, 1",
            "v_add_nc_i32 v0, v1, v2",
        ]
        words = run_bundle_ps(body, "ps_v_add_nc_i32")
        self.assertEqual(words[0], 0x80000000)

    def test_translator_v_add_nc_i32_clamp(self) -> None:
        body = [
            "v_mov_b32 v1, 0x7fffffff",
            "v_mov_b32 v2, 10",
            "v_add_nc_i32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(body, "ps_v_add_nc_i32_clamp")
        self.assertEqual(words[0], 0x7FFFFFFF)

    def test_v_add_nc_i32_negative_controls(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_add_nc_i32", ["v_add_nc_i32", "s_endpgm"])

    def test_v_add_nc_i32_mutation_contrast(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x7FFFFFFF",
            "v_mov_b32 v2, 10",
            "v_add_nc_i32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(lines, "ps_mut_add_i32")
        self.assertEqual(words[0], 0x7FFFFFFF)
        self.assertNotEqual(words[0], 0x80000009)  # without clamp would wrap to 0x80000009


# -----------------------------------------------------------------------------
# 3. v_sub_nc_u32
# -----------------------------------------------------------------------------

class TestVSubNcU32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        self.assertEqual(oracle_sub_nc_u32(100, 30), 70)
        self.assertEqual(oracle_sub_nc_u32(50, 50), 0)
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 30",
            "v_sub_nc_u32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_sub_u32_basic")
        self.assertEqual(words[0], 70)

    def test_wrapping(self) -> None:
        self.assertEqual(oracle_sub_nc_u32(0, 1), MASK32)
        self.assertEqual(oracle_sub_nc_u32(10, 20), (MASK32 - 9))
        lines = [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 1",
            "v_sub_nc_u32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_sub_u32_wrap")
        self.assertEqual(words[0], MASK32)

    def test_clamp_saturation(self) -> None:
        self.assertEqual(oracle_sub_nc_u32(10, 20, clamp=True), 0)
        self.assertEqual(oracle_sub_nc_u32(0, 1, clamp=True), 0)
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_sub_nc_u32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(lines, "ps_sub_u32_clamp")
        self.assertEqual(words[0], 0)


class TestVSubNcU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        lines = [
            "v_mov_b32 v1, 1000",
            "v_mov_b32 v2, 250",
            "v_sub_nc_u32 v0, v1, v2",
        ]
        spv = support.translate_lines("ps_ast_sub_u32", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpISub", spv)
        words = run_bundle_ps(lines, "ps_ast_sub_u32_exec")
        self.assertEqual(words[0], 750)

    def test_ast_clamp_evaluation(self) -> None:
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 100",
            "v_sub_nc_u32 v0, v1, v2 clamp",
        ]
        spv = support.translate_lines("ps_ast_sub_u32_cl", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpISubBorrow", spv)
        words = run_bundle_ps(lines, "ps_ast_sub_u32_cl_exec")
        self.assertEqual(words[0], 0)


class TestVSubNcU32WrappingAndSaturation(unittest.TestCase):
    def test_unsigned_underflow(self) -> None:
        self.assertEqual(oracle_sub_nc_u32(0, MASK32, clamp=False), 1)
        self.assertEqual(oracle_sub_nc_u32(0, MASK32, clamp=True), 0)
        lines_wrap = [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_sub_nc_u32 v0, v1, v2",
        ]
        words_wrap = run_bundle_ps(lines_wrap, "ps_sub_u32_unflow_wrap")
        self.assertEqual(words_wrap[0], 1)

        lines_cl = [
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_sub_nc_u32 v0, v1, v2 clamp",
        ]
        words_cl = run_bundle_ps(lines_cl, "ps_sub_u32_unflow_cl")
        self.assertEqual(words_cl[0], 0)


class TestVSubNcU32SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        for l in (0, 1, 7, 15, 16, 31):
            s0 = l * 50
            s1 = 100
            expected = oracle_sub_nc_u32(s0, s1, clamp=True)
            lines = [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_sub_nc_u32 v0, v1, v2 clamp",
            ]
            words = run_bundle_ps(lines, f"ps_simd_sub_u32_{l}")
            self.assertEqual(words[0], expected)


class TestVSubNcU32TranslatorAndControls(unittest.TestCase):
    def test_translator_v_sub_nc_u32_execution(self) -> None:
        body = [
            "v_mov_b32 v1, 200",
            "v_mov_b32 v2, 75",
            "v_sub_nc_u32_e32 v0, v1, v2",
        ]
        words = run_bundle_ps(body, "ps_v_sub_nc_u32_e32")
        self.assertEqual(words[0], 125)

    def test_translator_v_sub_nc_u32_clamp(self) -> None:
        body = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 50",
            "v_sub_nc_u32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(body, "ps_v_sub_nc_u32_clamp")
        self.assertEqual(words[0], 0)

    def test_v_sub_nc_u32_negative_controls(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_sub_nc_u32", ["v_sub_nc_u32", "s_endpgm"])

    def test_v_sub_nc_u32_mutation_contrast(self) -> None:
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 50",
            "v_sub_nc_u32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(lines, "ps_mut_sub_u32")
        self.assertEqual(words[0], 0)
        self.assertNotEqual(words[0], (10 - 50) & MASK32)  # clamp to 0 instead of underflow


# -----------------------------------------------------------------------------
# 4. v_sub_nc_i32
# -----------------------------------------------------------------------------

class TestVSubNcI32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        self.assertEqual(oracle_sub_nc_i32(100, 40), 60)
        self.assertEqual(oracle_sub_nc_i32(-20, -10), (-10) & MASK32)
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 40",
            "v_sub_nc_i32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_sub_i32_basic")
        self.assertEqual(words[0], 60)

    def test_wrapping_signed(self) -> None:
        self.assertEqual(oracle_sub_nc_i32(0x80000000, 1), 0x7FFFFFFF)
        lines = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 1",
            "v_sub_nc_i32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_sub_i32_wrap")
        self.assertEqual(words[0], 0x7FFFFFFF)

    def test_clamp_signed(self) -> None:
        self.assertEqual(oracle_sub_nc_i32(0x80000000, 10, clamp=True), 0x80000000)
        self.assertEqual(oracle_sub_nc_i32(0x7FFFFFFF, -10, clamp=True), 0x7FFFFFFF)
        lines_neg = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 10",
            "v_sub_nc_i32 v0, v1, v2 clamp",
        ]
        words_neg = run_bundle_ps(lines_neg, "ps_sub_i32_cl_neg")
        self.assertEqual(words_neg[0], 0x80000000)

        lines_pos = [
            "v_mov_b32 v1, 0x7FFFFFFF",
            f"v_mov_b32 v2, {(-10) & MASK32:#x}",
            "v_sub_nc_i32 v0, v1, v2 clamp",
        ]
        words_pos = run_bundle_ps(lines_pos, "ps_sub_i32_cl_pos")
        self.assertEqual(words_pos[0], 0x7FFFFFFF)


class TestVSubNcI32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        lines = [
            "v_mov_b32 v1, 400",
            "v_mov_b32 v2, 150",
            "v_sub_nc_i32 v0, v1, v2",
        ]
        spv = support.translate_lines("ps_ast_sub_i32", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpISub", spv)
        words = run_bundle_ps(lines, "ps_ast_sub_i32_exec")
        self.assertEqual(words[0], 250)

    def test_ast_clamp_evaluation(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 5",
            "v_sub_nc_i32 v0, v1, v2 clamp",
        ]
        spv = support.translate_lines("ps_ast_sub_i32_cl", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpSLessThan", spv)
        words = run_bundle_ps(lines, "ps_ast_sub_i32_cl_exec")
        self.assertEqual(words[0], 0x80000000)


class TestVSubNcI32WrappingAndSaturation(unittest.TestCase):
    def test_signed_sub_extremes(self) -> None:
        # Min negative - 1 without clamp wraps to max positive
        self.assertEqual(oracle_sub_nc_i32(0x80000000, 1, clamp=False), 0x7FFFFFFF)
        lines_neg = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 1",
            "v_sub_nc_i32 v0, v1, v2",
        ]
        words_neg = run_bundle_ps(lines_neg, "ps_sub_i32_neg_wrap")
        self.assertEqual(words_neg[0], 0x7FFFFFFF)

        # Max positive - (-1) without clamp wraps to min negative
        self.assertEqual(oracle_sub_nc_i32(0x7FFFFFFF, 0xFFFFFFFF, clamp=False), 0x80000000)
        lines_pos = [
            "v_mov_b32 v1, 0x7FFFFFFF",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_sub_nc_i32 v0, v1, v2",
        ]
        words_pos = run_bundle_ps(lines_pos, "ps_sub_i32_pos_wrap")
        self.assertEqual(words_pos[0], 0x80000000)


class TestVSubNcI32SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        for l in (0, 1, 7, 15, 16, 31):
            s0 = l * 10
            s1 = 50
            expected = oracle_sub_nc_i32(s0, s1)
            lines = [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_sub_nc_i32 v0, v1, v2",
            ]
            words = run_bundle_ps(lines, f"ps_simd_sub_i32_{l}")
            self.assertEqual(words[0], expected)


class TestVSubNcI32TranslatorAndControls(unittest.TestCase):
    def test_translator_v_sub_nc_i32_execution(self) -> None:
        body = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 1",
            "v_sub_nc_i32 v0, v1, v2",
        ]
        words = run_bundle_ps(body, "ps_v_sub_nc_i32")
        self.assertEqual(words[0], 0x7FFFFFFF)

    def test_translator_v_sub_nc_i32_clamp(self) -> None:
        body = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 10",
            "v_sub_nc_i32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(body, "ps_v_sub_nc_i32_clamp")
        self.assertEqual(words[0], 0x80000000)

    def test_v_sub_nc_i32_negative_controls(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_sub_nc_i32", ["v_sub_nc_i32", "s_endpgm"])

    def test_v_sub_nc_i32_mutation_contrast(self) -> None:
        lines = [
            "v_mov_b32 v1, 0x80000000",
            "v_mov_b32 v2, 10",
            "v_sub_nc_i32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(lines, "ps_mut_sub_i32")
        self.assertEqual(words[0], 0x80000000)
        self.assertNotEqual(words[0], 0x7FFFFFF6)  # without clamp would wrap to 0x7FFFFFF6


# -----------------------------------------------------------------------------
# 5. v_subrev_nc_u32
# -----------------------------------------------------------------------------

class TestVSubrevNcU32Oracle(unittest.TestCase):
    def test_reverse_subtraction(self) -> None:
        # src0 = 30, src1 = 100 -> 100 - 30 = 70
        self.assertEqual(oracle_subrev_nc_u32(30, 100), 70)
        # src0 = 100, src1 = 30 -> 30 - 100 = -70 -> 0xFFFFFFBA
        self.assertEqual(oracle_subrev_nc_u32(100, 30), (MASK32 - 69))
        lines = [
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 100",
            "v_subrev_nc_u32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_subrev_u32_basic")
        self.assertEqual(words[0], 70)

    def test_clamp_saturation(self) -> None:
        self.assertEqual(oracle_subrev_nc_u32(100, 30, clamp=True), 0)
        self.assertEqual(oracle_subrev_nc_u32(30, 100, clamp=True), 70)
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 30",
            "v_subrev_nc_u32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(lines, "ps_subrev_u32_clamp")
        self.assertEqual(words[0], 0)


class TestVSubrevNcU32AST(unittest.TestCase):
    def test_ast_evaluation(self) -> None:
        lines = [
            "v_mov_b32 v1, 20",
            "v_mov_b32 v2, 80",
            "v_subrev_nc_u32 v0, v1, v2",
        ]
        spv = support.translate_lines("ps_ast_subrev_u32", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpISub", spv)
        words = run_bundle_ps(lines, "ps_ast_subrev_u32_exec")
        self.assertEqual(words[0], 60)

    def test_ast_clamp_evaluation(self) -> None:
        lines = [
            "v_mov_b32 v1, 80",
            "v_mov_b32 v2, 20",
            "v_subrev_nc_u32 v0, v1, v2 clamp",
        ]
        spv = support.translate_lines("ps_ast_subrev_u32_cl", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpISubBorrow", spv)
        words = run_bundle_ps(lines, "ps_ast_subrev_u32_cl_exec")
        self.assertEqual(words[0], 0)


class TestVSubrevNcU32SIMD(unittest.TestCase):
    def test_simd_lane_divergence(self) -> None:
        for l in (0, 1, 7, 15, 16, 31):
            s0 = l * 20
            s1 = 1000
            expected = oracle_subrev_nc_u32(s0, s1)
            lines = [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_subrev_nc_u32 v0, v1, v2",
            ]
            words = run_bundle_ps(lines, f"ps_simd_subrev_u32_{l}")
            self.assertEqual(words[0], expected)


class TestVSubrevNcU32TranslatorAndControls(unittest.TestCase):
    def test_translator_v_subrev_nc_u32_execution(self) -> None:
        body = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 30",
            "v_subrev_nc_u32_e32 v0, v1, v2",
        ]
        words = run_bundle_ps(body, "ps_v_subrev_nc_u32_e32")
        self.assertEqual(words[0], 0xFFFFFFBA)

    def test_translator_v_subrev_nc_u32_clamp(self) -> None:
        body = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 30",
            "v_subrev_nc_u32 v0, v1, v2 clamp",
        ]
        words = run_bundle_ps(body, "ps_v_subrev_nc_u32_clamp")
        self.assertEqual(words[0], 0)

    def test_v_subrev_nc_u32_negative_controls(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_subrev_nc_u32", ["v_subrev_nc_u32", "s_endpgm"])

    def test_v_subrev_nc_u32_mutation_contrast(self) -> None:
        lines = [
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 100",
            "v_subrev_nc_u32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mut_subrev_u32")
        self.assertEqual(words[0], 70)  # 100 - 30 = 70
        self.assertNotEqual(words[0], (30 - 100) & MASK32)  # must NOT be non-reversed 30 - 100


if __name__ == "__main__":
    unittest.main()
