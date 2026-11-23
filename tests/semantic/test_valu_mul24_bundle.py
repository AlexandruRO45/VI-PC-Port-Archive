"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU 24-bit Multiply Suite.

Subsystem Feature Bundle 38 covers 6 VALU 24-bit integer multiply and multiply-add instructions:
  1. v_mul_u32_u24    - Low 32 bits of 24-bit unsigned product: D = (S0[23:0] * S1[23:0])[31:0]
  2. v_mul_i32_i24    - Low 32 bits of 24-bit signed product: D = (sext24(S0) * sext24(S1))[31:0]
  3. v_mul_hi_u32_u24 - High 32 bits of 48-bit unsigned product: D = (S0[23:0] * S1[23:0])[47:32] (zero-extended)
  4. v_mul_hi_i32_i24 - High 32 bits of 48-bit signed product: D = (sext24(S0) * sext24(S1))[47:32] (sign-extended)
  5. v_mad_u32_u24    - 24-bit unsigned multiply-add: D = (S0[23:0] * S1[23:0])[31:0] + S2
  6. v_mad_i32_i24    - 24-bit signed multiply-add: D = (sext24(S0) * sext24(S1))[31:0] + S2

RDNA2 24-bit Arithmetic Architecture:
  - Operates on the lower 24 bits of source operands S0 and S1.
  - Bits [31:24] of source operands are ignored; for signed operations, sign extension originates from bit 23.
  - 48-bit products are partitioned into low 32 bits (v_mul_*) and high 32 bits (v_mul_hi_*).
  - 100% of tests translate synthetic assembly via tools/rdna2spirv.py and execute via spirv_mini_interp.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK24 = 0x00FFFFFF


def sext24(val: int) -> int:
    val &= MASK24
    return val if val < 0x00800000 else val - 0x01000000


def u32(val: int) -> int:
    return val & MASK32


def run_bundle_ps(lines: list[str], name: str = "ps_mul24_bundle") -> list[int]:
    """Helper to translate and run a pixel shader; returns MRT0 four 32-bit words."""
    all_lines = list(lines) + [
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(name, all_lines)
    return mrt0_words(spv)


# -----------------------------------------------------------------------------
# Opcode 1: v_mul_u32_u24
# -----------------------------------------------------------------------------
def oracle_mul_u32_u24(s0: int, s1: int) -> int:
    """Reference mathematical oracle for v_mul_u32_u24."""
    u0 = s0 & MASK24
    u1 = s1 & MASK24
    return (u0 * u1) & MASK32


class VMulU32U24OracleTests(unittest.TestCase):
    def test_mul_u24_basic(self):
        expected = oracle_mul_u32_u24(100, 200)
        self.assertEqual(expected, 20000)
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_mul_u32_u24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_u24_basic")
        self.assertEqual(words[0], expected)

    def test_mul_u24_ignores_high_byte(self):
        # 0xFF000064 & MASK24 is 100, 0xAA0000C8 & MASK24 is 200
        expected = oracle_mul_u32_u24(0xFF000064, 0xAA0000C8)
        self.assertEqual(expected, 20000)
        lines = [
            "v_mov_b32 v1, 0xFF000064",
            "v_mov_b32 v2, 0xAA0000C8",
            "v_mul_u32_u24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_u24_high_byte")
        self.assertEqual(words[0], expected)

    def test_mul_u24_full_24bit_range(self):
        # 0xFFFFFF * 0xFFFFFF = 0xFFFFFE000001 -> low 32 bits is 0xFE000001
        expected = oracle_mul_u32_u24(0xFFFFFF, 0xFFFFFF)
        self.assertEqual(expected, 0xFE000001)
        lines = [
            "v_mov_b32 v1, 0xFFFFFF",
            "v_mov_b32 v2, 0xFFFFFF",
            "v_mul_u32_u24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_u24_full_range")
        self.assertEqual(words[0], expected)


class VMulU32U24AstTests(unittest.TestCase):
    def test_mul_u24_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 40",
            "v_mul_u32_u24 v0, v1, v2",
        ]
        spv = translate_lines("ps_ast_mul_u24", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpIMul", spv)
        self.assertIn("OpBitwiseAnd", spv)
        words = run_bundle_ps(lines, "ps_ast_mul_u24_exec")
        self.assertEqual(words[0], 1200)


class VMulU32U24SimdDivergenceTests(unittest.TestCase):
    def test_mul_u24_simd_divergence(self):
        for l in (0, 1, 7, 15, 16, 31):
            s0 = l * 10
            s1 = 20
            expected = oracle_mul_u32_u24(s0, s1)
            lines = [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_mul_u32_u24 v0, v1, v2",
            ]
            words = run_bundle_ps(lines, f"ps_simd_mul_u24_{l}")
            self.assertEqual(words[0], expected)


class VMulU32U24TranslatorTests(unittest.TestCase):
    def test_mul_u24_translation_and_execution(self):
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_mul_u32_u24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_u24_exec")
        self.assertEqual(words[0], 20000)

    def test_mul_u24_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_mul_u24", ["v_mul_u32_u24"])

    def test_mul_u24_mutation_contrast(self):
        # 0x01000002 has high bit set. 24-bit mul ignores it -> 2 * 3 = 6.
        # Direct 32-bit mul would yield 0x03000006 != 6.
        lines = [
            "v_mov_b32 v1, 0x01000002",
            "v_mov_b32 v2, 3",
            "v_mul_u32_u24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mut_mul_u24")
        self.assertEqual(words[0], 6)
        self.assertNotEqual(words[0], 0x03000006)


# -----------------------------------------------------------------------------
# Opcode 2: v_mul_i32_i24
# -----------------------------------------------------------------------------
def oracle_mul_i32_i24(s0: int, s1: int) -> int:
    """Reference mathematical oracle for v_mul_i32_i24."""
    i0 = sext24(s0)
    i1 = sext24(s1)
    return (i0 * i1) & MASK32


class VMulI32I24OracleTests(unittest.TestCase):
    def test_mul_i24_basic(self):
        expected = oracle_mul_i32_i24(10, 20)
        self.assertEqual(expected, 200)
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mul_i32_i24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_i24_basic")
        self.assertEqual(words[0], expected)

    def test_mul_i24_negative(self):
        # -10 * 20 = -200 (0xFFFFFF38)
        expected_neg = oracle_mul_i32_i24(-10, 20)
        self.assertEqual(expected_neg, u32(-200))
        lines_neg = [
            f"v_mov_b32 v1, {u32(-10):#x}",
            "v_mov_b32 v2, 20",
            "v_mul_i32_i24 v0, v1, v2",
        ]
        words_neg = run_bundle_ps(lines_neg, "ps_mul_i24_neg")
        self.assertEqual(words_neg[0], expected_neg)

        # -10 * -20 = 200
        expected_both = oracle_mul_i32_i24(-10, -20)
        self.assertEqual(expected_both, 200)
        lines_both = [
            f"v_mov_b32 v1, {u32(-10):#x}",
            f"v_mov_b32 v2, {u32(-20):#x}",
            "v_mul_i32_i24 v0, v1, v2",
        ]
        words_both = run_bundle_ps(lines_both, "ps_mul_i24_both_neg")
        self.assertEqual(words_both[0], expected_both)

    def test_mul_i24_sign_extension_from_bit23(self):
        # 0x00800000 is -8388608 in 24-bit two's complement. -8388608 * 2 = -16777216
        expected = oracle_mul_i32_i24(0x00800000, 2)
        self.assertEqual(expected, u32(-16777216))
        lines = [
            "v_mov_b32 v1, 0x00800000",
            "v_mov_b32 v2, 2",
            "v_mul_i32_i24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_i24_sext23")
        self.assertEqual(words[0], expected)


class VMulI32I24AstTests(unittest.TestCase):
    def test_mul_i24_ast_and_execution(self):
        lines = [
            f"v_mov_b32 v1, {u32(-15):#x}",
            "v_mov_b32 v2, 3",
            "v_mul_i32_i24 v0, v1, v2",
        ]
        spv = translate_lines("ps_ast_mul_i24", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpIMul", spv)
        self.assertIn("OpBitFieldSExtract", spv)
        words = run_bundle_ps(lines, "ps_ast_mul_i24_exec")
        self.assertEqual(words[0], u32(-45))


class VMulI32I24SimdDivergenceTests(unittest.TestCase):
    def test_mul_i24_simd_divergence(self):
        for l in (0, 1, 7, 15, 16, 31):
            s0 = -l
            s1 = 5
            expected = oracle_mul_i32_i24(s0, s1)
            lines = [
                f"v_mov_b32 v1, {u32(s0):#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_mul_i32_i24 v0, v1, v2",
            ]
            words = run_bundle_ps(lines, f"ps_simd_mul_i24_{l}")
            self.assertEqual(words[0], expected)


class VMulI32I24TranslatorTests(unittest.TestCase):
    def test_mul_i24_translation_and_execution(self):
        lines = [
            f"v_mov_b32 v1, {u32(-10):#x}",
            "v_mov_b32 v2, 20",
            "v_mul_i32_i24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_i24_exec")
        self.assertEqual(words[0], u32(-200))

    def test_mul_i24_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_mul_i24", ["v_mul_i32_i24"])

    def test_mul_i24_mutation_contrast(self):
        # 0x00800000 signed is -8388608 -> -8388608 * 2 = -16777216 (0xFF000000).
        # If treated as unsigned, 0x800000 * 2 = 0x01000000.
        lines = [
            "v_mov_b32 v1, 0x00800000",
            "v_mov_b32 v2, 2",
            "v_mul_i32_i24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mut_mul_i24")
        self.assertEqual(words[0], 0xFF000000)
        self.assertNotEqual(words[0], 0x01000000)


# -----------------------------------------------------------------------------
# Opcode 3: v_mul_hi_u32_u24
# -----------------------------------------------------------------------------
def oracle_mul_hi_u32_u24(s0: int, s1: int) -> int:
    """Reference mathematical oracle for v_mul_hi_u32_u24."""
    u0 = s0 & MASK24
    u1 = s1 & MASK24
    prod = u0 * u1
    return (prod >> 32) & MASK32


class VMulHiU32U24OracleTests(unittest.TestCase):
    def test_mul_hi_u24_small_product_zero(self):
        expected = oracle_mul_hi_u32_u24(100, 200)
        self.assertEqual(expected, 0)
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 200",
            "v_mul_hi_u32_u24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_hi_u24_small")
        self.assertEqual(words[0], 0)

    def test_mul_hi_u24_large_product(self):
        # 0xFFFFFF * 0xFFFFFF = 0xFFFFFE000001 -> high 32 bits is 0x0000FFFF
        expected = oracle_mul_hi_u32_u24(0xFFFFFF, 0xFFFFFF)
        self.assertEqual(expected, 0x0000FFFF)
        lines = [
            "v_mov_b32 v1, 0xFFFFFF",
            "v_mov_b32 v2, 0xFFFFFF",
            "v_mul_hi_u32_u24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_hi_u24_large")
        self.assertEqual(words[0], expected)


class VMulHiU32U24AstTests(unittest.TestCase):
    def test_mul_hi_u24_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0x800000",
            "v_mov_b32 v2, 0x800000",
            "v_mul_hi_u32_u24 v0, v1, v2",
        ]
        spv = translate_lines("ps_ast_mul_hi_u24", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpBitFieldUExtract", spv)
        words = run_bundle_ps(lines, "ps_ast_mul_hi_u24_exec")
        self.assertEqual(words[0], 0x4000)


class VMulHiU32U24SimdDivergenceTests(unittest.TestCase):
    def test_mul_hi_u24_simd_divergence(self):
        for l in (0, 1, 7, 15, 16, 31):
            s0 = 0xFFFFFF
            s1 = l * 0x10000
            expected = oracle_mul_hi_u32_u24(s0, s1)
            lines = [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_mul_hi_u32_u24 v0, v1, v2",
            ]
            words = run_bundle_ps(lines, f"ps_simd_mul_hi_u24_{l}")
            self.assertEqual(words[0], expected)


class VMulHiU32U24TranslatorTests(unittest.TestCase):
    def test_mul_hi_u24_translation_and_execution(self):
        lines = [
            "v_mov_b32 v1, 0xFFFFFF",
            "v_mov_b32 v2, 0xFFFFFF",
            "v_mul_hi_u32_u24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_hi_u24_exec")
        self.assertEqual(words[0], 0x0000FFFF)

    def test_mul_hi_u24_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_mul_hi_u24", ["v_mul_hi_u32_u24"])

    def test_mul_hi_u24_product_reconstruction(self):
        # (hi << 32) | lo == 0xFFFFFF * 0xFFFFFF
        lines_lo = [
            "v_mov_b32 v1, 0xFFFFFF",
            "v_mov_b32 v2, 0xFFFFFF",
            "v_mul_u32_u24 v0, v1, v2",
        ]
        lines_hi = [
            "v_mov_b32 v1, 0xFFFFFF",
            "v_mov_b32 v2, 0xFFFFFF",
            "v_mul_hi_u32_u24 v0, v1, v2",
        ]
        lo = run_bundle_ps(lines_lo, "ps_rec_lo")[0]
        hi = run_bundle_ps(lines_hi, "ps_rec_hi")[0]
        full_product = (hi << 32) | lo
        self.assertEqual(full_product, 0xFFFFFF * 0xFFFFFF)


# -----------------------------------------------------------------------------
# Opcode 4: v_mul_hi_i32_i24
# -----------------------------------------------------------------------------
def oracle_mul_hi_i32_i24(s0: int, s1: int) -> int:
    """Reference mathematical oracle for v_mul_hi_i32_i24."""
    i0 = sext24(s0)
    i1 = sext24(s1)
    prod = i0 * i1
    return (prod >> 32) & MASK32


class VMulHiI32I24OracleTests(unittest.TestCase):
    def test_mul_hi_i24_positive_small(self):
        expected = oracle_mul_hi_i32_i24(10, 20)
        self.assertEqual(expected, 0)
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mul_hi_i32_i24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_hi_i24_small")
        self.assertEqual(words[0], 0)

    def test_mul_hi_i24_negative_small(self):
        # -100 * 200 = -20000 -> sign-extended high 32 bits is 0xFFFFFFFF
        expected = oracle_mul_hi_i32_i24(-100, 200)
        self.assertEqual(expected, 0xFFFFFFFF)
        lines = [
            f"v_mov_b32 v1, {u32(-100):#x}",
            "v_mov_b32 v2, 200",
            "v_mul_hi_i32_i24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_hi_i24_neg_small")
        self.assertEqual(words[0], expected)

    def test_mul_hi_i24_extreme_signs(self):
        # -8388608 * -8388608 = 70368744177664 (0x400000000000) -> high 32 bits is 0x4000
        expected = oracle_mul_hi_i32_i24(0x00800000, 0x00800000)
        self.assertEqual(expected, 0x00004000)
        lines = [
            "v_mov_b32 v1, 0x00800000",
            "v_mov_b32 v2, 0x00800000",
            "v_mul_hi_i32_i24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_hi_i24_extreme")
        self.assertEqual(words[0], expected)


class VMulHiI32I24AstTests(unittest.TestCase):
    def test_mul_hi_i24_ast_and_execution(self):
        lines = [
            f"v_mov_b32 v1, {u32(-50):#x}",
            "v_mov_b32 v2, 100",
            "v_mul_hi_i32_i24 v0, v1, v2",
        ]
        spv = translate_lines("ps_ast_mul_hi_i24", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpBitFieldSExtract", spv)
        words = run_bundle_ps(lines, "ps_ast_mul_hi_i24_exec")
        self.assertEqual(words[0], 0xFFFFFFFF)


class VMulHiI32I24SimdDivergenceTests(unittest.TestCase):
    def test_mul_hi_i24_simd_divergence(self):
        for l in (0, 1, 7, 15, 16, 31):
            s0 = -l * 100
            s1 = 1000
            expected = oracle_mul_hi_i32_i24(s0, s1)
            lines = [
                f"v_mov_b32 v1, {u32(s0):#x}",
                f"v_mov_b32 v2, {s1:#x}",
                "v_mul_hi_i32_i24 v0, v1, v2",
            ]
            words = run_bundle_ps(lines, f"ps_simd_mul_hi_i24_{l}")
            self.assertEqual(words[0], expected)


class VMulHiI32I24TranslatorTests(unittest.TestCase):
    def test_mul_hi_i24_translation_and_execution(self):
        lines = [
            f"v_mov_b32 v1, {u32(-100):#x}",
            "v_mov_b32 v2, 200",
            "v_mul_hi_i32_i24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mul_hi_i24_exec")
        self.assertEqual(words[0], 0xFFFFFFFF)

    def test_mul_hi_i24_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_mul_hi_i24", ["v_mul_hi_i32_i24"])

    def test_mul_hi_i24_mutation_contrast(self):
        # -100 * 200 is negative -> signed hi word is 0xFFFFFFFF.
        # Unsigned hi word would be 0x00000000.
        lines = [
            f"v_mov_b32 v1, {u32(-100):#x}",
            "v_mov_b32 v2, 200",
            "v_mul_hi_i32_i24 v0, v1, v2",
        ]
        words = run_bundle_ps(lines, "ps_mut_mul_hi_i24")
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertNotEqual(words[0], 0)


# -----------------------------------------------------------------------------
# Opcode 5: v_mad_u32_u24
# -----------------------------------------------------------------------------
def oracle_mad_u32_u24(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for v_mad_u32_u24: D = S0[23:0] * S1[23:0] + S2."""
    u0 = s0 & MASK24
    u1 = s1 & MASK24
    return (u0 * u1 + (s2 & MASK32)) & MASK32


class VMadU32U24OracleTests(unittest.TestCase):
    def test_mad_u24_basic(self):
        # 10 * 20 + 5 = 205
        expected = oracle_mad_u32_u24(10, 20, 5)
        self.assertEqual(expected, 205)
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 5",
            "v_mad_u32_u24 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mad_u24_basic")
        self.assertEqual(words[0], expected)

    def test_mad_u24_ignores_high_bytes_in_factors(self):
        expected = oracle_mad_u32_u24(0xFF00000A, 0xEE000014, 5)
        self.assertEqual(expected, 205)
        lines = [
            "v_mov_b32 v1, 0xFF00000A",
            "v_mov_b32 v2, 0xEE000014",
            "v_mov_b32 v3, 5",
            "v_mad_u32_u24 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mad_u24_high_bytes")
        self.assertEqual(words[0], expected)

    def test_mad_u24_wrapping(self):
        expected = oracle_mad_u32_u24(0xFFFFFF, 0xFFFFFF, 0xFFFFFFFF)
        self.assertEqual(expected, 0xFE000000)
        lines = [
            "v_mov_b32 v1, 0xFFFFFF",
            "v_mov_b32 v2, 0xFFFFFF",
            "v_mov_b32 v3, 0xFFFFFFFF",
            "v_mad_u32_u24 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mad_u24_wrap")
        self.assertEqual(words[0], expected)


class VMadU32U24AstTests(unittest.TestCase):
    def test_mad_u24_ast_and_execution(self):
        lines = [
            "v_mov_b32 v1, 7",
            "v_mov_b32 v2, 8",
            "v_mov_b32 v3, 9",
            "v_mad_u32_u24 v0, v1, v2, v3",
        ]
        spv = translate_lines("ps_ast_mad_u24", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpIMul", spv)
        self.assertIn("OpIAdd", spv)
        words = run_bundle_ps(lines, "ps_ast_mad_u24_exec")
        self.assertEqual(words[0], 65)


class VMadU32U24SimdDivergenceTests(unittest.TestCase):
    def test_mad_u24_simd_divergence(self):
        for l in (0, 1, 7, 15, 16, 31):
            s0 = l
            s1 = 2
            s2 = 1
            expected = oracle_mad_u32_u24(s0, s1, s2)
            lines = [
                f"v_mov_b32 v1, {s0:#x}",
                f"v_mov_b32 v2, {s1:#x}",
                f"v_mov_b32 v3, {s2:#x}",
                "v_mad_u32_u24 v0, v1, v2, v3",
            ]
            words = run_bundle_ps(lines, f"ps_simd_mad_u24_{l}")
            self.assertEqual(words[0], expected)


class VMadU32U24TranslatorTests(unittest.TestCase):
    def test_mad_u24_translation_and_execution(self):
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 5",
            "v_mad_u32_u24 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mad_u24_exec")
        self.assertEqual(words[0], 205)

    def test_mad_u24_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_mad_u24", ["v_mad_u32_u24"])

    def test_mad_u24_mutation_contrast(self):
        # 10 * 20 + 5 = 205. If addend was subtracted, result would be 195.
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 5",
            "v_mad_u32_u24 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mut_mad_u24")
        self.assertEqual(words[0], 205)
        self.assertNotEqual(words[0], 195)
        self.assertNotEqual(words[0], 200)  # addend ignored


# -----------------------------------------------------------------------------
# Opcode 6: v_mad_i32_i24
# -----------------------------------------------------------------------------
def oracle_mad_i32_i24(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for v_mad_i32_i24: D = sext24(S0) * sext24(S1) + S2."""
    i0 = sext24(s0)
    i1 = sext24(s1)
    s2_val = s2 & MASK32
    prod = i0 * i1
    return (prod + s2_val) & MASK32


class VMadI32I24OracleTests(unittest.TestCase):
    def test_mad_i24_basic(self):
        # -10 * 20 + 5 = -195 (0xFFFFFF3D)
        expected = oracle_mad_i32_i24(-10, 20, 5)
        self.assertEqual(expected, u32(-195))
        lines = [
            f"v_mov_b32 v1, {u32(-10):#x}",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 5",
            "v_mad_i32_i24 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mad_i24_basic")
        self.assertEqual(words[0], expected)

    def test_mad_i24_negative_factors(self):
        # -5 * -10 + 2 = 52
        expected = oracle_mad_i32_i24(-5, -10, 2)
        self.assertEqual(expected, 52)
        lines = [
            f"v_mov_b32 v1, {u32(-5):#x}",
            f"v_mov_b32 v2, {u32(-10):#x}",
            "v_mov_b32 v3, 2",
            "v_mad_i32_i24 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mad_i24_neg_factors")
        self.assertEqual(words[0], expected)


class VMadI32I24AstTests(unittest.TestCase):
    def test_mad_i24_ast_and_execution(self):
        lines = [
            f"v_mov_b32 v1, {u32(-4):#x}",
            "v_mov_b32 v2, 5",
            "v_mov_b32 v3, 1",
            "v_mad_i32_i24 v0, v1, v2, v3",
        ]
        spv = translate_lines("ps_ast_mad_i24", lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        self.assertIn("OpBitFieldSExtract", spv)
        self.assertIn("OpIMul", spv)
        self.assertIn("OpIAdd", spv)
        words = run_bundle_ps(lines, "ps_ast_mad_i24_exec")
        self.assertEqual(words[0], u32(-19))


class VMadI32I24SimdDivergenceTests(unittest.TestCase):
    def test_mad_i24_simd_divergence(self):
        for l in (0, 1, 7, 15, 16, 31):
            s0 = -l
            s1 = 2
            s2 = 10
            expected = oracle_mad_i32_i24(s0, s1, s2)
            lines = [
                f"v_mov_b32 v1, {u32(s0):#x}",
                f"v_mov_b32 v2, {s1:#x}",
                f"v_mov_b32 v3, {s2:#x}",
                "v_mad_i32_i24 v0, v1, v2, v3",
            ]
            words = run_bundle_ps(lines, f"ps_simd_mad_i24_{l}")
            self.assertEqual(words[0], expected)


class VMadI32I24TranslatorTests(unittest.TestCase):
    def test_mad_i24_translation_and_execution(self):
        lines = [
            f"v_mov_b32 v1, {u32(-10):#x}",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 5",
            "v_mad_i32_i24 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mad_i24_exec")
        self.assertEqual(words[0], u32(-195))

    def test_mad_i24_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_mad_i24", ["v_mad_i32_i24"])

    def test_mad_i24_mutation_contrast(self):
        # -10 * 20 + 5 = -195 (0xFFFFFF3D).
        # Unsigned mul would give (0x00FFFFF6 * 20) + 5.
        lines = [
            f"v_mov_b32 v1, {u32(-10):#x}",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 5",
            "v_mad_i32_i24 v0, v1, v2, v3",
        ]
        words = run_bundle_ps(lines, "ps_mut_mad_i24")
        self.assertEqual(words[0], u32(-195))
        self.assertNotEqual(words[0], u32(-200))  # without addend


if __name__ == "__main__":
    unittest.main()
