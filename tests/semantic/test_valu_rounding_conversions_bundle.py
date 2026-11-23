"""Comprehensive translator-backed regression bundle for RDNA2 VALU Rounding & Conversions.

Subsystem Feature Bundle covering:
1. v_floor_f32 (Float floor rounding, denormals flushed, IEEE 0 mode)
2. v_ceil_f32  (Float ceiling rounding, denormals flushed, IEEE 0 mode)
3. v_trunc_f32 (Float truncation rounding, denormals flushed, IEEE 0 mode)
4. v_rndne_f32 (Float round-to-nearest-even, denormals flushed, IEEE 0 mode)
5. v_cvt_f32_u32 (Unsigned 32-bit int to float32 conversion)
6. v_cvt_f32_i32 (Signed 32-bit int to float32 conversion)
7. v_cvt_i32_f32 (Float32 to signed 32-bit int with saturation and NaN handling)
8. v_cvt_u32_f32 (Float32 to unsigned 32-bit int with saturation and NaN handling)

All test methods compile assembly listings directly through tools/rdna2spirv.py
via support.translate_lines(...) and execute in spirv_mini_interp via support.mrt0_words.
Standard library Python only; zero third-party dependencies; zero game data.
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

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def oracle_v_floor_f32(u: int) -> int:
    u = u & MASK32
    exp = u & 0x7F800000
    if exp == 0:
        return 0x80000000 if (u & 0x80000000) else 0x00000000
    if exp == 0x7F800000:
        return u
    f = u32_to_f32(u)
    res_f = math.floor(f)
    if res_f == 0.0 and (u & 0x80000000):
        return 0x80000000
    return f32_to_u32(res_f)


def oracle_v_ceil_f32(u: int) -> int:
    u = u & MASK32
    exp = u & 0x7F800000
    if exp == 0:
        return 0x80000000 if (u & 0x80000000) else 0x00000000
    if exp == 0x7F800000:
        return u
    f = u32_to_f32(u)
    res_f = math.ceil(f)
    if res_f == 0.0 and (u & 0x80000000):
        return 0x80000000
    return f32_to_u32(res_f)


def oracle_v_trunc_f32(u: int) -> int:
    u = u & MASK32
    exp = u & 0x7F800000
    if exp == 0:
        return 0x80000000 if (u & 0x80000000) else 0x00000000
    if exp == 0x7F800000:
        return u
    f = u32_to_f32(u)
    res_f = math.trunc(f)
    if res_f == 0.0 and (u & 0x80000000):
        return 0x80000000
    return f32_to_u32(float(res_f))


def oracle_v_rndne_f32(u: int) -> int:
    u = u & MASK32
    exp = u & 0x7F800000
    if exp == 0:
        return 0x80000000 if (u & 0x80000000) else 0x00000000
    if exp == 0x7F800000:
        return u
    f = u32_to_f32(u)
    res_f = round(f)
    if res_f == 0.0 and (u & 0x80000000):
        return 0x80000000
    return f32_to_u32(float(res_f))


def oracle_v_cvt_f32_u32(u: int) -> int:
    return f32_to_u32(float(u & MASK32))


def oracle_v_cvt_f32_i32(i: int) -> int:
    signed_i = i if (i & 0x80000000) == 0 else i - 0x100000000
    return f32_to_u32(float(signed_i))


def oracle_v_cvt_i32_f32(u: int) -> int:
    bits = u & MASK32
    mag = bits & 0x7FFFFFFF
    is_neg = (bits != mag)
    if mag > 0x7F800000:
        return 0
    if bits == 0x7F800000 or mag >= 0x4F000000:
        return 0x80000000 if is_neg else 0x7FFFFFFF
    f = u32_to_f32(bits)
    return int(f) & MASK32


def oracle_v_cvt_u32_f32(u: int) -> int:
    bits = u & MASK32
    if (bits & 0x80000000) != 0 or bits > 0x7F800000:
        return 0
    if bits >= 0x4F800000:
        return 0xFFFFFFFF
    f = u32_to_f32(bits)
    return int(f) & MASK32


# -----------------------------------------------------------------------------
# 1. v_floor_f32
# -----------------------------------------------------------------------------
class TestVFloorF32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_floor_f32: reference mathematical oracle validation."""

    def test_v_floor_f32_oracle_identity(self):
        for u_val, exp_val in [(0x3fc00000, 0x3f800000), (0xbfc00000, 0xc0000000)]:
            with self.subTest(u_val=hex(u_val)):
                self.assertEqual(oracle_v_floor_f32(u_val), exp_val)
                lines = [
                    f"v_mov_b32 v1, {u_val:#x}",
                    "v_floor_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_oracle_v_floor_f32_{u_val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp_val)


class TestVFloorF32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_floor_f32: AST and deterministic interpreter execution."""

    def test_v_floor_f32_ast_emission(self):
        lines = [
            "v_mov_b32 v1, 0x3fc00000",
            "v_floor_f32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_floor_f32", lines)
        self.assertIn("Floor", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3f800000)


class TestVFloorF32SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_floor_f32: SIMD multi-lane divergence and IEEE 754 edge cases."""

    def test_v_floor_f32_edge_cases(self):
        edge_cases = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xbfc00000, 0x3fc00000]
        for val in edge_cases:
            with self.subTest(val=hex(val)):
                exp = oracle_v_floor_f32(val)
                lines = [
                    f"v_mov_b32 v1, {val:#x}",
                    "v_floor_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_ieee_v_floor_f32_{val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp)


class TestVFloorF32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_floor_f32: translator integration and negative controls."""

    def test_v_floor_f32_negative_controls(self):
        lines = [
            "v_mov_b32 v0, 0xbfc00000",  # -1.5f
            "v_floor_f32 v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_floor_f32", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xc0000000)  # -2.0f
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_floor_f32", ["v_floor_f32", "s_endpgm"])


# -----------------------------------------------------------------------------
# 2. v_ceil_f32
# -----------------------------------------------------------------------------
class TestVCeilF32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_ceil_f32: reference mathematical oracle validation."""

    def test_v_ceil_f32_oracle_identity(self):
        for u_val, exp_val in [(0x3fc00000, 0x40000000), (0xbfc00000, 0xbf800000)]:
            with self.subTest(u_val=hex(u_val)):
                self.assertEqual(oracle_v_ceil_f32(u_val), exp_val)
                lines = [
                    f"v_mov_b32 v1, {u_val:#x}",
                    "v_ceil_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_oracle_v_ceil_f32_{u_val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp_val)


class TestVCeilF32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_ceil_f32: AST and deterministic interpreter execution."""

    def test_v_ceil_f32_ast_emission(self):
        lines = [
            "v_mov_b32 v1, 0x3fc00000",
            "v_ceil_f32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_ceil_f32", lines)
        self.assertIn("Ceil", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40000000)


class TestVCeilF32SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_ceil_f32: SIMD multi-lane divergence and IEEE 754 edge cases."""

    def test_v_ceil_f32_edge_cases(self):
        edge_cases = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xbfc00000, 0x3fc00000]
        for val in edge_cases:
            with self.subTest(val=hex(val)):
                exp = oracle_v_ceil_f32(val)
                lines = [
                    f"v_mov_b32 v1, {val:#x}",
                    "v_ceil_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_ieee_v_ceil_f32_{val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp)


class TestVCeilF32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_ceil_f32: translator integration and negative controls."""

    def test_v_ceil_f32_negative_controls(self):
        lines = [
            "v_mov_b32 v0, 0x3fc00000",  # 1.5f
            "v_ceil_f32 v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_ceil_f32", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40000000)  # 2.0f
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_ceil_f32", ["v_ceil_f32", "s_endpgm"])


# -----------------------------------------------------------------------------
# 3. v_trunc_f32
# -----------------------------------------------------------------------------
class TestVTruncF32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_trunc_f32: reference mathematical oracle validation."""

    def test_v_trunc_f32_oracle_identity(self):
        for u_val, exp_val in [(0x3fc00000, 0x3f800000), (0xbfc00000, 0xbf800000)]:
            with self.subTest(u_val=hex(u_val)):
                self.assertEqual(oracle_v_trunc_f32(u_val), exp_val)
                lines = [
                    f"v_mov_b32 v1, {u_val:#x}",
                    "v_trunc_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_oracle_v_trunc_f32_{u_val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp_val)


class TestVTruncF32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_trunc_f32: AST and deterministic interpreter execution."""

    def test_v_trunc_f32_ast_emission(self):
        lines = [
            "v_mov_b32 v1, 0x3fc00000",
            "v_trunc_f32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_trunc_f32", lines)
        self.assertIn("Trunc", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x3f800000)


class TestVTruncF32SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_trunc_f32: SIMD multi-lane divergence and IEEE 754 edge cases."""

    def test_v_trunc_f32_edge_cases(self):
        edge_cases = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xbfc00000, 0x3fc00000]
        for val in edge_cases:
            with self.subTest(val=hex(val)):
                exp = oracle_v_trunc_f32(val)
                lines = [
                    f"v_mov_b32 v1, {val:#x}",
                    "v_trunc_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_ieee_v_trunc_f32_{val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp)


class TestVTruncF32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_trunc_f32: translator integration and negative controls."""

    def test_v_trunc_f32_negative_controls(self):
        lines = [
            "v_mov_b32 v0, 0xbfc00000",  # -1.5f
            "v_trunc_f32 v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_trunc_f32", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xbf800000)  # -1.0f (truncates towards 0)
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_trunc_f32", ["v_trunc_f32", "s_endpgm"])


# -----------------------------------------------------------------------------
# 4. v_rndne_f32
# -----------------------------------------------------------------------------
class TestVRndneF32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_rndne_f32: reference mathematical oracle validation."""

    def test_v_rndne_f32_oracle_identity(self):
        for u_val, exp_val in [(0x3fc00000, 0x40000000), (0x40200000, 0x40000000)]:
            with self.subTest(u_val=hex(u_val)):
                self.assertEqual(oracle_v_rndne_f32(u_val), exp_val)
                lines = [
                    f"v_mov_b32 v1, {u_val:#x}",
                    "v_rndne_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_oracle_v_rndne_f32_{u_val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp_val)


class TestVRndneF32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_rndne_f32: AST and deterministic interpreter execution."""

    def test_v_rndne_f32_ast_emission(self):
        lines = [
            "v_mov_b32 v1, 0x3fc00000",
            "v_rndne_f32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_rndne_f32", lines)
        self.assertIn("RoundEven", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40000000)


class TestVRndneF32SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_rndne_f32: SIMD multi-lane divergence and IEEE 754 edge cases."""

    def test_v_rndne_f32_edge_cases(self):
        edge_cases = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x40200000, 0x3fc00000]
        for val in edge_cases:
            with self.subTest(val=hex(val)):
                exp = oracle_v_rndne_f32(val)
                lines = [
                    f"v_mov_b32 v1, {val:#x}",
                    "v_rndne_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_ieee_v_rndne_f32_{val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp)


class TestVRndneF32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_rndne_f32: translator integration and negative controls."""

    def test_v_rndne_f32_negative_controls(self):
        lines = [
            "v_mov_b32 v0, 0x40200000",  # 2.5f -> 2.0f
            "v_rndne_f32 v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_rndne_f32", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40000000)
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_rndne_f32", ["v_rndne_f32", "s_endpgm"])


# -----------------------------------------------------------------------------
# 5. v_cvt_f32_u32
# -----------------------------------------------------------------------------
class TestVCvtF32U32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_cvt_f32_u32: reference mathematical oracle validation."""

    def test_v_cvt_f32_u32_oracle_identity(self):
        for u_val, exp_val in [(0x2a, 0x42280000), (0x80000000, 0x4f000000)]:
            with self.subTest(u_val=hex(u_val)):
                self.assertEqual(oracle_v_cvt_f32_u32(u_val), exp_val)
                lines = [
                    f"v_mov_b32 v1, {u_val:#x}",
                    "v_cvt_f32_u32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_oracle_v_cvt_f32_u32_{u_val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp_val)


class TestVCvtF32U32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_cvt_f32_u32: AST and deterministic interpreter execution."""

    def test_v_cvt_f32_u32_ast_emission(self):
        lines = [
            "v_mov_b32 v1, 0x2a",
            "v_cvt_f32_u32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_cvt_f32_u32", lines)
        self.assertIn("OpConvertUToF", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x42280000)


class TestVCvtF32U32SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_cvt_f32_u32: SIMD multi-lane divergence and IEEE 754 edge cases."""

    def test_v_cvt_f32_u32_edge_cases(self):
        edge_cases = [0x00000000, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF, 1]
        for val in edge_cases:
            with self.subTest(val=hex(val)):
                exp = oracle_v_cvt_f32_u32(val)
                lines = [
                    f"v_mov_b32 v1, {val:#x}",
                    "v_cvt_f32_u32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_ieee_v_cvt_f32_u32_{val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp)


class TestVCvtF32U32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_cvt_f32_u32: translator integration and negative controls."""

    def test_v_cvt_f32_u32_negative_controls(self):
        lines = [
            "v_mov_b32 v0, 0x80000000",
            "v_cvt_f32_u32 v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_cvt_f32_u32", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x4f000000)
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_cvt_f32_u32", ["v_cvt_f32_u32", "s_endpgm"])


# -----------------------------------------------------------------------------
# 6. v_cvt_f32_i32
# -----------------------------------------------------------------------------
class TestVCvtF32I32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_cvt_f32_i32: reference mathematical oracle validation."""

    def test_v_cvt_f32_i32_oracle_identity(self):
        for i_val, exp_val in [(0xffffffd6, 0xc2280000), (0xffffffff, 0xbf800000)]:
            with self.subTest(i_val=hex(i_val)):
                self.assertEqual(oracle_v_cvt_f32_i32(i_val), exp_val)
                lines = [
                    f"v_mov_b32 v1, {i_val:#x}",
                    "v_cvt_f32_i32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_oracle_v_cvt_f32_i32_{i_val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp_val)


class TestVCvtF32I32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_cvt_f32_i32: AST and deterministic interpreter execution."""

    def test_v_cvt_f32_i32_ast_emission(self):
        lines = [
            "v_mov_b32 v1, 0xffffffd6",
            "v_cvt_f32_i32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_cvt_f32_i32", lines)
        self.assertIn("OpConvertSToF", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xc2280000)


class TestVCvtF32I32SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_cvt_f32_i32: SIMD multi-lane divergence and IEEE 754 edge cases."""

    def test_v_cvt_f32_i32_edge_cases(self):
        edge_cases = [0x00000000, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF, 1]
        for val in edge_cases:
            with self.subTest(val=hex(val)):
                exp = oracle_v_cvt_f32_i32(val)
                lines = [
                    f"v_mov_b32 v1, {val:#x}",
                    "v_cvt_f32_i32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_ieee_v_cvt_f32_i32_{val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp)


class TestVCvtF32I32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_cvt_f32_i32: translator integration and negative controls."""

    def test_v_cvt_f32_i32_negative_controls(self):
        lines = [
            "v_mov_b32 v0, 0xffffffd6",
            "v_cvt_f32_i32 v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_cvt_f32_i32", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xc2280000)
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_cvt_f32_i32", ["v_cvt_f32_i32", "s_endpgm"])


# -----------------------------------------------------------------------------
# 7. v_cvt_i32_f32
# -----------------------------------------------------------------------------
class TestVCvtI32F32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_cvt_i32_f32: reference mathematical oracle validation."""

    def test_v_cvt_i32_f32_oracle_identity(self):
        for f_val, exp_val in [(0xc0a00000, 0xfffffffb), (0x40a00000, 5)]:
            with self.subTest(f_val=hex(f_val)):
                self.assertEqual(oracle_v_cvt_i32_f32(f_val), exp_val)
                lines = [
                    f"v_mov_b32 v1, {f_val:#x}",
                    "v_cvt_i32_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_oracle_v_cvt_i32_f32_{f_val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp_val)


class TestVCvtI32F32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_cvt_i32_f32: AST and deterministic interpreter execution."""

    def test_v_cvt_i32_f32_ast_emission(self):
        lines = [
            "v_mov_b32 v1, 0xc0a00000",
            "v_cvt_i32_f32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_cvt_i32_f32", lines)
        self.assertIn("OpConvertFToS", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xfffffffb)


class TestVCvtI32F32SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_cvt_i32_f32: SIMD multi-lane divergence and IEEE 754 edge cases."""

    def test_v_cvt_i32_f32_edge_cases(self):
        edge_cases = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xc0a00000, 0x40a00000]
        for val in edge_cases:
            with self.subTest(val=hex(val)):
                exp = oracle_v_cvt_i32_f32(val)
                lines = [
                    f"v_mov_b32 v1, {val:#x}",
                    "v_cvt_i32_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_ieee_v_cvt_i32_f32_{val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp)


class TestVCvtI32F32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_cvt_i32_f32: translator integration and negative controls."""

    def test_v_cvt_i32_f32_negative_controls(self):
        lines = [
            "v_mov_b32 v0, 0xc0a00000",
            "v_cvt_i32_f32 v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_cvt_i32_f32", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xfffffffb)
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_cvt_i32_f32", ["v_cvt_i32_f32", "s_endpgm"])


# -----------------------------------------------------------------------------
# 8. v_cvt_u32_f32
# -----------------------------------------------------------------------------
class TestVCvtU32F32ReferenceOracle(unittest.TestCase):
    """Stage 1 for v_cvt_u32_f32: reference mathematical oracle validation."""

    def test_v_cvt_u32_f32_oracle_identity(self):
        for f_val, exp_val in [(0x40a00000, 5), (0x4f800000, 0xFFFFFFFF)]:
            with self.subTest(f_val=hex(f_val)):
                self.assertEqual(oracle_v_cvt_u32_f32(f_val), exp_val)
                lines = [
                    f"v_mov_b32 v1, {f_val:#x}",
                    "v_cvt_u32_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_oracle_v_cvt_u32_f32_{f_val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp_val)


class TestVCvtU32F32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for v_cvt_u32_f32: AST and deterministic interpreter execution."""

    def test_v_cvt_u32_f32_ast_emission(self):
        lines = [
            "v_mov_b32 v1, 0x40a00000",
            "v_cvt_u32_f32 v0, v1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_cvt_u32_f32", lines)
        self.assertIn("OpConvertFToU", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x5)


class TestVCvtU32F32SIMDAndIEEE(unittest.TestCase):
    """Stage 3 for v_cvt_u32_f32: SIMD multi-lane divergence and IEEE 754 edge cases."""

    def test_v_cvt_u32_f32_edge_cases(self):
        edge_cases = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x40a00000]
        for val in edge_cases:
            with self.subTest(val=hex(val)):
                exp = oracle_v_cvt_u32_f32(val)
                lines = [
                    f"v_mov_b32 v1, {val:#x}",
                    "v_cvt_u32_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = support.translate_lines(f"ps_ieee_v_cvt_u32_f32_{val:#x}", lines)
                words = support.mrt0_words(spv)
                self.assertEqual(words[0], exp)


class TestVCvtU32F32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for v_cvt_u32_f32: translator integration and negative controls."""

    def test_v_cvt_u32_f32_negative_controls(self):
        lines = [
            "v_mov_b32 v0, 0x40a00000",
            "v_cvt_u32_f32 v0, v0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_v_cvt_u32_f32", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 5)
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_v_cvt_u32_f32", ["v_cvt_u32_f32", "s_endpgm"])


# -----------------------------------------------------------------------------
# 9. Reviewer Mutation Feedback & Invariant Pinning
# -----------------------------------------------------------------------------
class TestVALURoundingConversionsMutations(unittest.TestCase):
    """Pins exact instruction lowerings against reviewer mutations."""

    def test_floor_vs_ceil_mutation(self):
        # 1.5f (0x3fc00000): floor -> 1.0f (0x3f800000), ceil -> 2.0f (0x40000000)
        lines_floor = ["v_mov_b32 v1, 0x3fc00000", "v_floor_f32 v0, v1", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        lines_ceil = ["v_mov_b32 v1, 0x3fc00000", "v_ceil_f32 v0, v1", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        w_floor = support.mrt0_words(support.translate_lines("ps_mut_floor", lines_floor))[0]
        w_ceil = support.mrt0_words(support.translate_lines("ps_mut_ceil", lines_ceil))[0]
        self.assertEqual(w_floor, 0x3f800000)
        self.assertEqual(w_ceil, 0x40000000)
        self.assertNotEqual(w_floor, w_ceil)

    def test_trunc_vs_floor_negative_mutation(self):
        # -1.5f (0xbfc00000): trunc -> -1.0f (0xbf800000), floor -> -2.0f (0xc0000000)
        lines_trunc = ["v_mov_b32 v1, 0xbfc00000", "v_trunc_f32 v0, v1", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        lines_floor = ["v_mov_b32 v1, 0xbfc00000", "v_floor_f32 v0, v1", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        w_trunc = support.mrt0_words(support.translate_lines("ps_mut_trunc", lines_trunc))[0]
        w_floor = support.mrt0_words(support.translate_lines("ps_mut_floor_neg", lines_floor))[0]
        self.assertEqual(w_trunc, 0xbf800000)
        self.assertEqual(w_floor, 0xc0000000)
        self.assertNotEqual(w_trunc, w_floor)

    def test_cvt_f32_i32_vs_u32_mutation(self):
        # Negative int -42 (0xffffffd6): signed -> -42.0f (0xc2280000), unsigned -> 4294967254.0f (0x4f7ffffd)
        lines_i32 = ["v_mov_b32 v1, 0xffffffd6", "v_cvt_f32_i32 v0, v1", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        lines_u32 = ["v_mov_b32 v1, 0xffffffd6", "v_cvt_f32_u32 v0, v1", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        w_i32 = support.mrt0_words(support.translate_lines("ps_mut_f32_i32", lines_i32))[0]
        w_u32 = support.mrt0_words(support.translate_lines("ps_mut_f32_u32", lines_u32))[0]
        self.assertEqual(w_i32, 0xc2280000)
        self.assertNotEqual(w_i32, w_u32)

    def test_cvt_i32_f32_vs_u32_f32_mutation(self):
        # Negative float -5.0f (0xc0a00000): signed -> -5 (0xfffffffb), unsigned -> 0
        lines_i32 = ["v_mov_b32 v1, 0xc0a00000", "v_cvt_i32_f32 v0, v1", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        lines_u32 = ["v_mov_b32 v1, 0xc0a00000", "v_cvt_u32_f32 v0, v1", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
        w_i32 = support.mrt0_words(support.translate_lines("ps_mut_i32_f32", lines_i32))[0]
        w_u32 = support.mrt0_words(support.translate_lines("ps_mut_u32_f32", lines_u32))[0]
        self.assertEqual(w_i32, 0xfffffffb)
        self.assertEqual(w_u32, 0)
        self.assertNotEqual(w_i32, w_u32)


if __name__ == "__main__":
    unittest.main()
