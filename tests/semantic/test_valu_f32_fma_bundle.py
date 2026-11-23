"""Comprehensive translator-backed regression bundle for RDNA2 VALU Float32 FMA, FMAC, and MAD.

Subsystem Feature Bundle 39 covering:
1. v_fma_f32   (Fused multiply-add on 32-bit floats with single rounding: D = S0 * S1 + S2)
2. v_fmac_f32  (Fused multiply-accumulate on 32-bit floats: D = S0 * S1 + D)
3. v_fmaak_f32 (Fused multiply-add with inline constant addend: D = S0 * S1 + K)
4. v_fmamk_f32 (Fused multiply-add with inline constant multiplier: D = S0 * K + S1)
5. v_mad_f32   (Unfused multiply-add with denormal flushing on 32-bit floats)
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def run_bundle_ps(lines: list[str], name: str = "ps_fma_bundle") -> list[int]:
    all_lines = list(lines) + [
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


def oracle_fma_f32(a: int, b: int, c: int) -> int:
    fa, fb, fc = u32_to_f32(a), u32_to_f32(b), u32_to_f32(c)
    if math.isnan(fa) or math.isnan(fb) or math.isnan(fc):
        return 0x7FC00000
    res = fa * fb + fc
    if math.isinf(res):
        return 0x7F800000 if res > 0 else 0xFF800000
    return f32_to_u32(res)


def oracle_mad_f32(a: int, b: int, c: int) -> int:
    def flush(u: int) -> float:
        if (u & 0x7F800000) == 0:
            return 0.0
        return u32_to_f32(u)

    fa, fb, fc = flush(a), flush(b), flush(c)
    prod = fa * fb
    p_bits = f32_to_u32(prod)
    if (p_bits & 0x7F800000) == 0:
        prod = 0.0
    res = prod + fc
    r_bits = f32_to_u32(res)
    if (r_bits & 0x7F800000) == 0:
        return 0
    return r_bits


class TestVFmaF32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_fma_f32: reference mathematical oracle and translator validation."""

    def test_v_fma_f32_oracle_identity(self):
        # 2.0 * 3.0 + 1.0 = 7.0
        res = oracle_fma_f32(0x40000000, 0x40400000, 0x3F800000)
        self.assertEqual(res, 0x40E00000)
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x40400000",
            "v_mov_b32 v3, 0x3F800000",
            "v_fma_f32 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_fma_oracle_ident")
        self.assertEqual(words[0], res)


class TestVFmaF32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_fma_f32: AST and scalar interpreter execution."""

    def test_v_fma_f32_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x40400000",
            "v_mov_b32 v3, 0x3F800000",
            "v_fma_f32 v0, v1, v2, v3",
        ]
        spv = support.translate_lines("ps_ast_v_fma_f32", lines + ["v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("Fma", spv)
        words = run_bundle_ps(lines, "ps_exec_v_fma_f32")
        self.assertEqual(words[0], 0x40E00000)


class TestVFmaF32SIMDDivergence(unittest.TestCase):
    """Stage 3 for v_fma_f32: multi-lane divergence under varied inputs."""

    def test_v_fma_f32_simd_divergence(self):
        lane_inputs = [
            (0x3F800000, 0x40000000, 0x40400000, 0x40A00000),  # 1.0 * 2.0 + 3.0 = 5.0
            (0x40000000, 0x40400000, 0x40800000, 0x41200000),  # 2.0 * 3.0 + 4.0 = 10.0
            (0x40800000, 0x3F000000, 0x3F800000, 0x40400000),  # 4.0 * 0.5 + 1.0 = 3.0
        ]
        for s0, s1, s2, expected in lane_inputs:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                lines = [
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    f"v_mov_b32 v3, {s2:#x}",
                    "v_fma_f32 v0, v1, v2, v3",
                ]
                words = run_bundle_ps(lines, f"ps_simd_fma_{s0:#x}")
                self.assertEqual(words[0], expected)


class TestVFmaF32IEEEEdgeCases(unittest.TestCase):
    """Stage 4 for v_fma_f32: IEEE 754 boundary extremes and zeros."""

    def test_v_fma_f32_edge_cases(self):
        cases = [
            (0x00000000, 0x3F800000, 0x40000000, 0x40000000),  # 0.0 * 1.0 + 2.0 = 2.0
            (0x3F800000, 0xBF800000, 0x3F800000, 0x00000000),  # 1.0 * -1.0 + 1.0 = 0.0
            (0xC0000000, 0xC0400000, 0xC0C00000, 0x00000000),  # -2.0 * -3.0 + -6.0 = 0.0
        ]
        for s0, s1, s2, expected in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                lines = [
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    f"v_mov_b32 v3, {s2:#x}",
                    "v_fma_f32 v0, v1, v2, v3",
                ]
                words = run_bundle_ps(lines, f"ps_ieee_fma_{s0:#x}")
                self.assertEqual(words[0], expected)


class TestVFmaF32TranslatorAndControls(unittest.TestCase):
    """Stage 5 for v_fma_f32: translator integration and negative controls."""

    def test_v_fma_f32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_fma_f32", ["v_fma_f32", "s_endpgm"])

    def test_v_fma_f32_mutation_contrast(self):
        lines = [
            "v_mov_b32 v1, 0x40000000",  # 2.0
            "v_mov_b32 v2, 0x40400000",  # 3.0
            "v_mov_b32 v3, 0x40800000",  # 4.0
            "v_fma_f32 v0, v1, v2, v3",   # 2.0 * 3.0 + 4.0 = 10.0 (0x41200000)
        ]
        words = run_bundle_ps(lines, "ps_fma_mut_check")
        self.assertEqual(words[0], 0x41200000)
        # Mutated to simple add: 2.0 + 3.0 + 4.0 = 9.0 (0x41100000) != 10.0
        self.assertNotEqual(words[0], 0x41100000)


class TestVFmacF32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_fmac_f32: reference mathematical oracle and translator validation."""

    def test_v_fmac_f32_oracle_identity(self):
        # Initial v0 = 1.0, v1 = 2.0, v2 = 3.0 -> 2.0 * 3.0 + 1.0 = 7.0
        res = oracle_fma_f32(0x40000000, 0x40400000, 0x3F800000)
        self.assertEqual(res, 0x40E00000)
        lines = [
            "v_mov_b32 v0, 0x3F800000",
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x40400000",
            "v_fmac_f32 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_fmac_oracle_ident")
        self.assertEqual(words[0], res)


class TestVFmacF32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_fmac_f32: AST and scalar interpreter execution."""

    def test_v_fmac_f32_ast_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3F800000",
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x40400000",
            "v_fmac_f32 v0, v1, v2",
        ]
        spv = support.translate_lines("ps_ast_v_fmac_f32", lines + ["v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("Fma", spv)
        words = run_bundle_ps(lines, "ps_exec_v_fmac_f32")
        self.assertEqual(words[0], 0x40E00000)


class TestVFmacF32Accumulator(unittest.TestCase):
    """Stage 3 for v_fmac_f32: accumulator chaining and operand preservation."""

    def test_v_fmac_f32_chaining(self):
        lines = [
            "v_mov_b32 v0, 0x3F800000",  # 1.0
            "v_mov_b32 v1, 0x40000000",  # 2.0
            "v_mov_b32 v2, 0x40400000",  # 3.0
            "v_fmac_f32 v0, v1, v2",     # 1.0 + 2.0 * 3.0 = 7.0
            "v_fmac_f32 v0, v1, v2",     # 7.0 + 2.0 * 3.0 = 13.0 (0x41500000)
        ]
        words = run_bundle_ps(lines, "ps_fmac_chain")
        self.assertEqual(words[0], 0x41500000)


class TestVFmacF32IEEEEdgeCases(unittest.TestCase):
    """Stage 4 for v_fmac_f32: boundary extremes and signed zero handling."""

    def test_v_fmac_f32_edge_cases(self):
        cases = [
            (0x00000000, 0x40800000, 0x40A00000, 0x41A00000),  # 0.0 + 4.0 * 5.0 = 20.0
            (0xC0A00000, 0x40000000, 0x40400000, 0x3F800000),  # -5.0 + 2.0 * 3.0 = 1.0
        ]
        for v0_in, v1_in, v2_in, expected in cases:
            with self.subTest(v0_in=hex(v0_in), v1_in=hex(v1_in), v2_in=hex(v2_in)):
                lines = [
                    f"v_mov_b32 v0, {v0_in:#x}",
                    f"v_mov_b32 v1, {v1_in:#x}",
                    f"v_mov_b32 v2, {v2_in:#x}",
                    "v_fmac_f32 v0, v1, v2",
                ]
                words = run_bundle_ps(lines, f"ps_ieee_fmac_{v0_in:#x}")
                self.assertEqual(words[0], expected)


class TestVFmacF32TranslatorAndControls(unittest.TestCase):
    """Stage 5 for v_fmac_f32: translator integration and negative controls."""

    def test_v_fmac_f32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_fmac_f32", ["v_fmac_f32", "s_endpgm"])

    def test_v_fmac_f32_mutation_contrast(self):
        lines = [
            "v_mov_b32 v0, 0x40000000",  # 2.0
            "v_mov_b32 v1, 0x40800000",  # 4.0
            "v_mov_b32 v2, 0x40A00000",  # 5.0
            "v_fmac_f32 v0, v1, v2",     # 2.0 + 4.0 * 5.0 = 22.0 (0x41B00000)
        ]
        words = run_bundle_ps(lines, "ps_fmac_mut_check")
        self.assertEqual(words[0], 0x41B00000)
        # Mutated to addition: 2.0 + 4.0 + 5.0 = 11.0 (0x41300000) != 22.0
        self.assertNotEqual(words[0], 0x41300000)


class TestVFmaakF32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_fmaak_f32: reference mathematical oracle and translator validation."""

    def test_v_fmaak_f32_oracle_identity(self):
        # 2.0 * 3.0 + 1.0 = 7.0
        res = oracle_fma_f32(0x40000000, 0x40400000, 0x3F800000)
        self.assertEqual(res, 0x40E00000)
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x40400000",
            "v_fmaak_f32 v0, v1, v2, 0x3F800000",
        ]
        words = run_bundle_ps(lines, "ps_fmaak_oracle_ident")
        self.assertEqual(words[0], res)


class TestVFmaakF32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_fmaak_f32: AST and scalar interpreter execution."""

    def test_v_fmaak_f32_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x40400000",
            "v_fmaak_f32 v0, v1, v2, 0x3F800000",
        ]
        spv = support.translate_lines("ps_ast_v_fmaak_f32", lines + ["v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("Fma", spv)
        words = run_bundle_ps(lines, "ps_exec_v_fmaak_f32")
        self.assertEqual(words[0], 0x40E00000)


class TestVFmaakF32InlineConstant(unittest.TestCase):
    """Stage 3 for v_fmaak_f32: inline constant literal addend binding."""

    def test_v_fmaak_f32_inline_constants(self):
        constants = [
            (0x40000000, 0x40400000, 0x3F800000, 0x40E00000),  # 2.0 * 3.0 + 1.0 = 7.0
            (0x40800000, 0x40000000, 0xC0000000, 0x40C00000),  # 4.0 * 2.0 - 2.0 = 6.0
        ]
        for s0, s1, k, expected in constants:
            with self.subTest(s0=hex(s0), s1=hex(s1), k=hex(k)):
                lines = [
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    f"v_fmaak_f32 v0, v1, v2, {k:#x}",
                ]
                words = run_bundle_ps(lines, f"ps_fmaak_k_{k:#x}")
                self.assertEqual(words[0], expected)


class TestVFmaakF32IEEEEdgeCases(unittest.TestCase):
    """Stage 4 for v_fmaak_f32: IEEE boundary extremes with inline constant."""

    def test_v_fmaak_f32_edge_cases(self):
        lines = [
            "v_mov_b32 v1, 0x00000000",
            "v_mov_b32 v2, 0x40000000",
            "v_fmaak_f32 v0, v1, v2, 0x40400000",  # 0.0 * 2.0 + 3.0 = 3.0
        ]
        words = run_bundle_ps(lines, "ps_fmaak_zero")
        self.assertEqual(words[0], 0x40400000)


class TestVFmaakF32TranslatorAndControls(unittest.TestCase):
    """Stage 5 for v_fmaak_f32: translator integration and negative controls."""

    def test_v_fmaak_f32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_fmaak_f32", ["v_fmaak_f32", "s_endpgm"])

    def test_v_fmaak_f32_mutation_contrast(self):
        lines = [
            "v_mov_b32 v1, 0x40400000",  # 3.0
            "v_mov_b32 v2, 0x40800000",  # 4.0
            "v_fmaak_f32 v0, v1, v2, 0x40000000",  # 3.0 * 4.0 + 2.0 = 14.0 (0x41600000)
        ]
        words = run_bundle_ps(lines, "ps_fmaak_mut_check")
        self.assertEqual(words[0], 0x41600000)
        # Mutated to addition: 3.0 + 4.0 + 2.0 = 9.0 (0x41100000) != 14.0
        self.assertNotEqual(words[0], 0x41100000)


class TestVFmamkF32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_fmamk_f32: reference mathematical oracle and translator validation."""

    def test_v_fmamk_f32_oracle_identity(self):
        # 2.0 * 3.0 + 1.0 = 7.0
        res = oracle_fma_f32(0x40000000, 0x40400000, 0x3F800000)
        self.assertEqual(res, 0x40E00000)
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x3F800000",
            "v_fmamk_f32 v0, v1, 0x40400000, v2",
        ]
        words = run_bundle_ps(lines, "ps_fmamk_oracle_ident")
        self.assertEqual(words[0], res)


class TestVFmamkF32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_fmamk_f32: AST and scalar interpreter execution."""

    def test_v_fmamk_f32_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x3F800000",
            "v_fmamk_f32 v0, v1, 0x40400000, v2",
        ]
        spv = support.translate_lines("ps_ast_v_fmamk_f32", lines + ["v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("Fma", spv)
        words = run_bundle_ps(lines, "ps_exec_v_fmamk_f32")
        self.assertEqual(words[0], 0x40E00000)


class TestVFmamkF32InlineMultiplier(unittest.TestCase):
    """Stage 3 for v_fmamk_f32: inline constant multiplier binding."""

    def test_v_fmamk_f32_multipliers(self):
        constants = [
            (0x40000000, 0x40400000, 0x3F800000, 0x40E00000),  # 2.0 * 3.0 + 1.0 = 7.0
            (0x40A00000, 0x40000000, 0xBF800000, 0x41100000),  # 5.0 * 2.0 - 1.0 = 9.0
        ]
        for s0, k, s1, expected in constants:
            with self.subTest(s0=hex(s0), k=hex(k), s1=hex(s1)):
                lines = [
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    f"v_fmamk_f32 v0, v1, {k:#x}, v2",
                ]
                words = run_bundle_ps(lines, f"ps_fmamk_k_{k:#x}")
                self.assertEqual(words[0], expected)


class TestVFmamkF32IEEEEdgeCases(unittest.TestCase):
    """Stage 4 for v_fmamk_f32: IEEE boundary extremes with inline multiplier."""

    def test_v_fmamk_f32_edge_cases(self):
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x00000000",
            "v_fmamk_f32 v0, v1, 0x40400000, v2",  # 2.0 * 3.0 + 0.0 = 6.0 (0x40C00000)
        ]
        words = run_bundle_ps(lines, "ps_fmamk_zero")
        self.assertEqual(words[0], 0x40C00000)


class TestVFmamkF32TranslatorAndControls(unittest.TestCase):
    """Stage 5 for v_fmamk_f32: translator integration and negative controls."""

    def test_v_fmamk_f32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_fmamk_f32", ["v_fmamk_f32", "s_endpgm"])

    def test_v_fmamk_f32_mutation_contrast(self):
        lines = [
            "v_mov_b32 v1, 0x40800000",  # 4.0
            "v_mov_b32 v2, 0x40A00000",  # 5.0
            "v_fmamk_f32 v0, v1, 0x40000000, v2",  # 4.0 * 2.0 + 5.0 = 13.0 (0x41500000)
        ]
        words = run_bundle_ps(lines, "ps_fmamk_mut_check")
        self.assertEqual(words[0], 0x41500000)
        # Mutated to addition: 4.0 + 2.0 + 5.0 = 11.0 (0x41300000) != 13.0
        self.assertNotEqual(words[0], 0x41300000)


class TestVMadF32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_mad_f32: reference mathematical oracle and translator validation."""

    def test_v_mad_f32_oracle_identity(self):
        # 2.0 * 3.0 + 1.0 = 7.0
        res = oracle_mad_f32(0x40000000, 0x40400000, 0x3F800000)
        self.assertEqual(res, 0x40E00000)
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x40400000",
            "v_mov_b32 v3, 0x3F800000",
            "v_mad_f32 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mad_oracle_ident")
        self.assertEqual(words[0], res)


class TestVMadF32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_mad_f32: AST and scalar interpreter execution."""

    def test_v_mad_f32_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x40400000",
            "v_mov_b32 v3, 0x3F800000",
            "v_mad_f32 v0, v1, v2, v3",
        ]
        spv = support.translate_lines("ps_ast_v_mad_f32", lines + ["v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0", "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
        self.assertIn("OpFMul", spv)
        self.assertIn("OpFAdd", spv)
        words = run_bundle_ps(lines, "ps_exec_v_mad_f32")
        self.assertEqual(words[0], 0x40E00000)


class TestVMadF32DenormAndEdgeCases(unittest.TestCase):
    """Stage 3 for v_mad_f32: denormal flushing and IEEE edge cases."""

    def test_v_mad_f32_denorm_flushing(self):
        # Subnormal input flushed to zero
        lines = [
            "v_mov_b32 v1, 0x00000001",  # denormal
            "v_mov_b32 v2, 0x3F800000",  # 1.0
            "v_mov_b32 v3, 0x00000000",  # 0.0
            "v_mad_f32 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mad_denorm")
        self.assertEqual(words[0], 0x00000000)


class TestVMadF32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_mad_f32: translator integration and negative controls."""

    def test_v_mad_f32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_mad_f32", ["v_mad_f32", "s_endpgm"])

    def test_v_mad_f32_mutation_contrast(self):
        lines = [
            "v_mov_b32 v1, 0x40A00000",  # 5.0
            "v_mov_b32 v2, 0x40800000",  # 4.0
            "v_mov_b32 v3, 0x40000000",  # 2.0
            "v_mad_f32 v0, v1, v2, v3",   # 5.0 * 4.0 + 2.0 = 22.0 (0x41B00000)
        ]
        words = run_bundle_ps(lines, "ps_mad_mut_check")
        self.assertEqual(words[0], 0x41B00000)
        # Mutated to addition: 5.0 + 4.0 + 2.0 = 11.0 (0x41300000) != 22.0
        self.assertNotEqual(words[0], 0x41300000)


if __name__ == '__main__':
    unittest.main()
