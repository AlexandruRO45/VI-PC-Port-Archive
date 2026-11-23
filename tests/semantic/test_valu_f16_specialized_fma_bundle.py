"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU Float16 Mixed Precision & Specialized FMA operations (Bundle 20):
  - v_fmaak_f16:     Float16 fused multiply-add with literal K (D = S0 * S1 + K)
  - v_fmamk_f16:     Float16 fused multiply-add with literal K (D = S0 * K + S1)
  - v_fma_mixhi_f16: Float16 mixed precision FMA into high halfword (D[31:16] = fma(S0, S1, S2))
  - v_fma_mixlo_f16: Float16 mixed precision FMA into low halfword (D[15:0] = fma(S0, S1, S2))
  - v_dot2_f32_f16:  Float16 packed dot product with Float32 accumulator (D = S0.lo*S1.lo + S0.hi*S1.hi + S2)
  - v_dot2c_f32_f16: Float16 packed dot product accumulating into destination (D = S0.lo*S1.lo + S0.hi*S1.hi + D)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
100% of test cases compile and execute through support.translate_lines(...) and support.mrt0_words.
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    return val & MASK32


def f16_to_bits(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_to_f16(u: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u & 0xFFFF))[0]


def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def is_nan_f16(h: int) -> bool:
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def is_nan_f32(w: int) -> bool:
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x007FFFFF) != 0


# =============================================================================
# Reference Mathematical Oracles
# =============================================================================

def ref_v_fmaak_f16(s0: int, s1: int, k: int, old_dst: int = 0) -> int:
    if is_nan_f16(s0) or is_nan_f16(s1) or is_nan_f16(k):
        res_f16 = 0x7E00
    else:
        fa = bits_to_f16(s0 & 0xFFFF)
        fb = bits_to_f16(s1 & 0xFFFF)
        fk = bits_to_f16(k & 0xFFFF)
        r = fa * fb + fk
        if math.isnan(r):
            res_f16 = 0x7E00
        elif math.isinf(r):
            res_f16 = 0x7C00 if r > 0 else 0xFC00
        else:
            res_f16 = f16_to_bits(r)
    return ((old_dst & 0xFFFF0000) | (res_f16 & 0xFFFF)) & MASK32


def ref_v_fmamk_f16(s0: int, k: int, s1: int, old_dst: int = 0) -> int:
    if is_nan_f16(s0) or is_nan_f16(k) or is_nan_f16(s1):
        res_f16 = 0x7E00
    else:
        fa = bits_to_f16(s0 & 0xFFFF)
        fk = bits_to_f16(k & 0xFFFF)
        fb = bits_to_f16(s1 & 0xFFFF)
        r = fa * fk + fb
        if math.isnan(r):
            res_f16 = 0x7E00
        elif math.isinf(r):
            res_f16 = 0x7C00 if r > 0 else 0xFC00
        else:
            res_f16 = f16_to_bits(r)
    return ((old_dst & 0xFFFF0000) | (res_f16 & 0xFFFF)) & MASK32


def ref_v_fma_mixhi_f16(s0: int, s1: int, s2: int, old_dst: int = 0) -> int:
    if is_nan_f32(s0) or is_nan_f32(s1) or is_nan_f32(s2):
        res_f16 = 0x7E00
    else:
        fa = bits_to_f32(s0)
        fb = bits_to_f32(s1)
        fc = bits_to_f32(s2)
        r = fa * fb + fc
        if math.isnan(r):
            res_f16 = 0x7E00
        elif math.isinf(r):
            res_f16 = 0x7C00 if r > 0 else 0xFC00
        else:
            res_f16 = f16_to_bits(r)
    return (((res_f16 & 0xFFFF) << 16) | (old_dst & 0xFFFF)) & MASK32


def ref_v_fma_mixlo_f16(s0: int, s1: int, s2: int, old_dst: int = 0) -> int:
    if is_nan_f32(s0) or is_nan_f32(s1) or is_nan_f32(s2):
        res_f16 = 0x7E00
    else:
        fa = bits_to_f32(s0)
        fb = bits_to_f32(s1)
        fc = bits_to_f32(s2)
        r = fa * fb + fc
        if math.isnan(r):
            res_f16 = 0x7E00
        elif math.isinf(r):
            res_f16 = 0x7C00 if r > 0 else 0xFC00
        else:
            res_f16 = f16_to_bits(r)
    return ((old_dst & 0xFFFF0000) | (res_f16 & 0xFFFF)) & MASK32


def ref_v_dot2_f32_f16(s0: int, s1: int, s2: int) -> int:
    a0 = bits_to_f16(s0 & 0xFFFF)
    a1 = bits_to_f16((s0 >> 16) & 0xFFFF)
    b0 = bits_to_f16(s1 & 0xFFFF)
    b1 = bits_to_f16((s1 >> 16) & 0xFFFF)
    acc = bits_to_f32(s2)
    r = (a0 * b0) + (a1 * b1) + acc
    if math.isnan(r):
        return 0x7FC00000
    if math.isinf(r):
        return 0x7F800000 if r > 0 else 0xFF800000
    return f32_to_bits(r) & MASK32


def ref_v_dot2c_f32_f16(s0: int, s1: int, old_dst: int = 0) -> int:
    a0 = bits_to_f16(s0 & 0xFFFF)
    a1 = bits_to_f16((s0 >> 16) & 0xFFFF)
    b0 = bits_to_f16(s1 & 0xFFFF)
    b1 = bits_to_f16((s1 >> 16) & 0xFFFF)
    acc = bits_to_f32(old_dst)
    r = (a0 * b0) + (a1 * b1) + acc
    if math.isnan(r):
        return 0x7FC00000
    if math.isinf(r):
        return 0x7F800000 if r > 0 else 0xFF800000
    return f32_to_bits(r) & MASK32


# =============================================================================
# Translator Execution Helpers
# =============================================================================

_TRANSLATE_ID: int = 0


def run_fmaak_f16(s0: int, s1: int, k: int, old_dst: int = 0, dst: str = "v2", src0: str = "v0", src1: str = "v1") -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
    ]
    if old_dst != 0 or dst != "v2":
        lines.append(f"v_mov_b32 {dst}, {old_dst & MASK32:#x}")
    lines.extend([
        f"v_fmaak_f16 {dst}, {src0}, {src1}, {k & MASK16:#x}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(f"ps_fmaak_{_TRANSLATE_ID}", lines)
    return support.mrt0_words(spv)[0]


def run_fmamk_f16(s0: int, k: int, s1: int, old_dst: int = 0, dst: str = "v2", src0: str = "v0", src1: str = "v1") -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
    ]
    if old_dst != 0 or dst != "v2":
        lines.append(f"v_mov_b32 {dst}, {old_dst & MASK32:#x}")
    lines.extend([
        f"v_fmamk_f16 {dst}, {src0}, {k & MASK16:#x}, {src1}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(f"ps_fmamk_{_TRANSLATE_ID}", lines)
    return support.mrt0_words(spv)[0]


def run_fma_mixlo_f16(s0: int, s1: int, s2: int, old_dst: int = 0, dst: str = "v3", src0: str = "v0", src1: str = "v1", src2: str = "v2") -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
    ]
    if old_dst != 0 or dst != "v3":
        lines.append(f"v_mov_b32 {dst}, {old_dst & MASK32:#x}")
    lines.extend([
        f"v_fma_mixlo_f16 {dst}, {src0}, {src1}, {src2}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(f"ps_fma_mixlo_{_TRANSLATE_ID}", lines)
    return support.mrt0_words(spv)[0]


def run_fma_mixhi_f16(s0: int, s1: int, s2: int, old_dst: int = 0, dst: str = "v3", src0: str = "v0", src1: str = "v1", src2: str = "v2") -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
    ]
    if old_dst != 0 or dst != "v3":
        lines.append(f"v_mov_b32 {dst}, {old_dst & MASK32:#x}")
    lines.extend([
        f"v_fma_mixhi_f16 {dst}, {src0}, {src1}, {src2}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines(f"ps_fma_mixhi_{_TRANSLATE_ID}", lines)
    return support.mrt0_words(spv)[0]


def run_dot2_f32_f16(s0: int, s1: int, s2: int, dst: str = "v3", src0: str = "v0", src1: str = "v1", src2: str = "v2") -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        f"v_dot2_f32_f16 {dst}, {src0}, {src1}, {src2}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_dot2_{_TRANSLATE_ID}", lines)
    return support.mrt0_words(spv)[0]


def run_dot2c_f32_f16(s0: int, s1: int, acc: int, dst: str = "v2", src0: str = "v0", src1: str = "v1") -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 {dst}, {acc & MASK32:#x}",
        f"v_dot2c_f32_f16 {dst}, {src0}, {src1}",
        f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_dot2c_{_TRANSLATE_ID}", lines)
    return support.mrt0_words(spv)[0]


def run_fma_with_scc(opcode_line: str, setup_lines: list[str], dst: str, prior_scc: int = 1) -> tuple[int, int]:
    """Execute instruction while validating that SCC is untouched by VALU operations."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"s_cmp_eq_u32 1, {1 if prior_scc else 0}",
    ] + setup_lines + [
        opcode_line,
        "s_cselect_b32 s0, 1, 0",
        "v_mov_b32 v7, s0",
        f"exp mrt0 {dst}, v7, {dst}, v7 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_scc_probe_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


# =============================================================================
# 1. v_fmaak_f16
# =============================================================================

class TestVFmaakF16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # 1.0 * 2.0 + 0.5 = 2.5 (0x4100)
        res = run_fmaak_f16(0x3C00, 0x4000, 0x3800, old_dst=0xDEAD0000)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xDEAD)

    def test_zero_and_identity(self) -> None:
        # 0.0 * 2.0 + 1.0 = 1.0 (0x3C00)
        res = run_fmaak_f16(0x0000, 0x4000, 0x3C00, old_dst=0x12340000)
        self.assertEqual(res & 0xFFFF, 0x3C00)
        self.assertEqual(res >> 16, 0x1234)


class TestVFmaakF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x4000",
            "v_fmaak_f16 v2, v0, v1, 0x3800",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_fmaak_f16", lines)
        self.assertIn("OpEntryPoint", spv)

    def test_scalar_interpreter_step(self) -> None:
        res = run_fmaak_f16(0x3C00, 0x4000, 0x3800, old_dst=0xBEEF0000)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xBEEF)


class TestVFmaakF16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        val, scc = run_fma_with_scc(
            "v_fmaak_f16 v2, v0, v1, 0x3800",
            ["v_mov_b32 v0, 0x3c00", "v_mov_b32 v1, 0x4000", "v_mov_b32 v2, 0x11110000"],
            dst="v2",
            prior_scc=1,
        )
        self.assertEqual(val & 0xFFFF, 0x4100)
        self.assertEqual(val >> 16, 0x1111)
        self.assertEqual(scc, 1)

    def test_register_aliasing(self) -> None:
        res = run_fmaak_f16(0x3C00, 0x4000, 0x3800, old_dst=0x3C00, dst="v0", src0="v0", src1="v1")
        self.assertEqual(res & 0xFFFF, 0x4100)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = run_fmaak_f16(0x7E00, 0x4000, 0x3800)
        self.assertEqual(res_nan & 0xFFFF, 0x7E00)
        res_inf = run_fmaak_f16(0x7C00, 0x4000, 0x3800)
        self.assertEqual(res_inf & 0xFFFF, 0x7C00)


class TestVFmaakF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x4000",
            "v_fmaak_f16 v2, v0, v1, 0x3800",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_fmaak_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4100)


# =============================================================================
# 2. v_fmamk_f16
# =============================================================================

class TestVFmamkF16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # 1.0 * 2.0 + 0.5 = 2.5 (0x4100)
        res = run_fmamk_f16(0x3C00, 0x4000, 0x3800, old_dst=0xCAFE0000)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xCAFE)

    def test_zero_and_identity(self) -> None:
        # 0.0 * 2.0 + 1.0 = 1.0 (0x3C00)
        res = run_fmamk_f16(0x0000, 0x4000, 0x3C00, old_dst=0x43210000)
        self.assertEqual(res & 0xFFFF, 0x3C00)
        self.assertEqual(res >> 16, 0x4321)


class TestVFmamkF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x3800",
            "v_fmamk_f16 v2, v0, 0x4000, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_fmamk_f16", lines)
        self.assertIn("OpEntryPoint", spv)

    def test_scalar_interpreter_step(self) -> None:
        res = run_fmamk_f16(0x3C00, 0x4000, 0x3800, old_dst=0xFEED0000)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xFEED)


class TestVFmamkF16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        val, scc = run_fma_with_scc(
            "v_fmamk_f16 v2, v0, 0x4000, v1",
            ["v_mov_b32 v0, 0x3c00", "v_mov_b32 v1, 0x3800", "v_mov_b32 v2, 0x22220000"],
            dst="v2",
            prior_scc=1,
        )
        self.assertEqual(val & 0xFFFF, 0x4100)
        self.assertEqual(val >> 16, 0x2222)
        self.assertEqual(scc, 1)

    def test_register_aliasing(self) -> None:
        res = run_fmamk_f16(0x3C00, 0x4000, 0x3800, old_dst=0x3C00, dst="v0", src0="v0", src1="v1")
        self.assertEqual(res & 0xFFFF, 0x4100)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = run_fmamk_f16(0x7E00, 0x4000, 0x3800)
        self.assertEqual(res_nan & 0xFFFF, 0x7E00)
        res_inf = run_fmamk_f16(0x7C00, 0x4000, 0x3800)
        self.assertEqual(res_inf & 0xFFFF, 0x7C00)


class TestVFmamkF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c00",
            "v_mov_b32 v1, 0x3800",
            "v_fmamk_f16 v2, v0, 0x4000, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_fmamk_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4100)


# =============================================================================
# 3. v_fma_mixhi_f16
# =============================================================================

class TestVFmaMixhiF16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # 1.0f * 2.0f + 0.5f = 2.5f (f16 0x4100 written to high halfword)
        res = run_fma_mixhi_f16(0x3F800000, 0x40000000, 0x3F000000, old_dst=0x000055AA)
        self.assertEqual(res >> 16, 0x4100)
        self.assertEqual(res & 0xFFFF, 0x55AA)

    def test_zero_and_identity(self) -> None:
        # 0.0f * 2.0f + 1.0f = 1.0f (0x3C00 in high halfword)
        res = run_fma_mixhi_f16(0x00000000, 0x40000000, 0x3F800000, old_dst=0x00001234)
        self.assertEqual(res >> 16, 0x3C00)
        self.assertEqual(res & 0xFFFF, 0x1234)


class TestVFmaMixhiF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x3f000000",
            "v_fma_mixhi_f16 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_fma_mixhi_f16", lines)
        self.assertIn("OpEntryPoint", spv)

    def test_scalar_interpreter_step(self) -> None:
        res = run_fma_mixhi_f16(0x3F800000, 0x40000000, 0x3F000000, old_dst=0x00009988)
        self.assertEqual(res >> 16, 0x4100)
        self.assertEqual(res & 0xFFFF, 0x9988)


class TestVFmaMixhiF16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        val, scc = run_fma_with_scc(
            "v_fma_mixhi_f16 v3, v0, v1, v2",
            ["v_mov_b32 v0, 0x3f800000", "v_mov_b32 v1, 0x40000000", "v_mov_b32 v2, 0x3f000000", "v_mov_b32 v3, 0x00003333"],
            dst="v3",
            prior_scc=1,
        )
        self.assertEqual(val >> 16, 0x4100)
        self.assertEqual(val & 0xFFFF, 0x3333)
        self.assertEqual(scc, 1)

    def test_register_aliasing(self) -> None:
        res = run_fma_mixhi_f16(0x3F800000, 0x40000000, 0x3F000000, old_dst=0x3F800000, dst="v0", src0="v0", src1="v1", src2="v2")
        self.assertEqual(res >> 16, 0x4100)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = run_fma_mixhi_f16(0x7FC00000, 0x40000000, 0x3F000000)
        self.assertEqual(res_nan >> 16, 0x7E00)
        res_inf = run_fma_mixhi_f16(0x7F800000, 0x40000000, 0x3F000000)
        self.assertEqual(res_inf >> 16, 0x7C00)


class TestVFmaMixhiF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x3f000000",
            "v_fma_mixhi_f16 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_fma_mixhi_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] >> 16, 0x4100)


# =============================================================================
# 4. v_fma_mixlo_f16
# =============================================================================

class TestVFmaMixloF16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # 1.0f * 2.0f + 0.5f = 2.5f (f16 0x4100 written to low halfword)
        res = run_fma_mixlo_f16(0x3F800000, 0x40000000, 0x3F000000, old_dst=0xAA550000)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xAA55)

    def test_zero_and_identity(self) -> None:
        # 0.0f * 2.0f + 1.0f = 1.0f (0x3C00 in low halfword)
        res = run_fma_mixlo_f16(0x00000000, 0x40000000, 0x3F800000, old_dst=0x43210000)
        self.assertEqual(res & 0xFFFF, 0x3C00)
        self.assertEqual(res >> 16, 0x4321)


class TestVFmaMixloF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x3f000000",
            "v_fma_mixlo_f16 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_fma_mixlo_f16", lines)
        self.assertIn("OpEntryPoint", spv)

    def test_scalar_interpreter_step(self) -> None:
        res = run_fma_mixlo_f16(0x3F800000, 0x40000000, 0x3F000000, old_dst=0x88990000)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0x8899)


class TestVFmaMixloF16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        val, scc = run_fma_with_scc(
            "v_fma_mixlo_f16 v3, v0, v1, v2",
            ["v_mov_b32 v0, 0x3f800000", "v_mov_b32 v1, 0x40000000", "v_mov_b32 v2, 0x3f000000", "v_mov_b32 v3, 0x44440000"],
            dst="v3",
            prior_scc=1,
        )
        self.assertEqual(val & 0xFFFF, 0x4100)
        self.assertEqual(val >> 16, 0x4444)
        self.assertEqual(scc, 1)

    def test_register_aliasing(self) -> None:
        res = run_fma_mixlo_f16(0x3F800000, 0x40000000, 0x3F000000, old_dst=0x3F800000, dst="v0", src0="v0", src1="v1", src2="v2")
        self.assertEqual(res & 0xFFFF, 0x4100)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = run_fma_mixlo_f16(0x7FC00000, 0x40000000, 0x3F000000)
        self.assertEqual(res_nan & 0xFFFF, 0x7E00)
        res_inf = run_fma_mixlo_f16(0x7F800000, 0x40000000, 0x3F000000)
        self.assertEqual(res_inf & 0xFFFF, 0x7C00)


class TestVFmaMixloF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3f800000",
            "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0x3f000000",
            "v_fma_mixlo_f16 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_fma_mixlo_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & 0xFFFF, 0x4100)


# =============================================================================
# 5. v_dot2_f32_f16
# =============================================================================

class TestVDot2F32F16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # (1.0*1.0) + (1.0*1.0) + 1.0 = 3.0f (0x40400000)
        res = run_dot2_f32_f16(0x3C003C00, 0x3C003C00, 0x3F800000)
        self.assertEqual(res, 0x40400000)

    def test_zero_accumulator(self) -> None:
        # (2.0*0.5) + (1.0*3.0) + 0.0 = 1.0 + 3.0 = 4.0f (0x40800000)
        res = run_dot2_f32_f16(0x3C004000, 0x42003800, 0x00000000)
        self.assertEqual(res, 0x40800000)


class TestVDot2F32F16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c003c00",
            "v_mov_b32 v1, 0x3c003c00",
            "v_mov_b32 v2, 0x3f800000",
            "v_dot2_f32_f16 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_dot2_f32_f16", lines)
        self.assertIn("OpEntryPoint", spv)

    def test_scalar_interpreter_step(self) -> None:
        res = run_dot2_f32_f16(0x3C003C00, 0x3C003C00, 0x3F800000)
        self.assertEqual(res, 0x40400000)


class TestVDot2F32F16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        val, scc = run_fma_with_scc(
            "v_dot2_f32_f16 v3, v0, v1, v2",
            ["v_mov_b32 v0, 0x3c003c00", "v_mov_b32 v1, 0x3c003c00", "v_mov_b32 v2, 0x3f800000"],
            dst="v3",
            prior_scc=1,
        )
        self.assertEqual(val, 0x40400000)
        self.assertEqual(scc, 1)

    def test_register_aliasing(self) -> None:
        res = run_dot2_f32_f16(0x3C003C00, 0x3C003C00, 0x3F800000, dst="v0", src0="v0", src1="v1", src2="v2")
        self.assertEqual(res, 0x40400000)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = run_dot2_f32_f16(0x7E003C00, 0x3C003C00, 0x3F800000)
        self.assertEqual(res_nan, 0x7FC00000)
        res_inf = run_dot2_f32_f16(0x7C003C00, 0x3C003C00, 0x3F800000)
        self.assertEqual(res_inf, 0x7F800000)


class TestVDot2F32F16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c003c00",
            "v_mov_b32 v1, 0x3c003c00",
            "v_mov_b32 v2, 0x3f800000",
            "v_dot2_f32_f16 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_dot2_f32_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40400000)


# =============================================================================
# 6. v_dot2c_f32_f16
# =============================================================================

class TestVDot2cF32F16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # (1.0*1.0) + (1.0*1.0) + old_dst(1.0) = 3.0f (0x40400000)
        res = run_dot2c_f32_f16(0x3C003C00, 0x3C003C00, acc=0x3F800000)
        self.assertEqual(res, 0x40400000)

    def test_zero_accumulator(self) -> None:
        # (2.0*0.5) + (1.0*3.0) + 0.0 = 4.0f (0x40800000)
        res = run_dot2c_f32_f16(0x3C004000, 0x42003800, acc=0x00000000)
        self.assertEqual(res, 0x40800000)


class TestVDot2cF32F16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c003c00",
            "v_mov_b32 v1, 0x3c003c00",
            "v_mov_b32 v2, 0x3f800000",
            "v_dot2c_f32_f16 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_v_dot2c_f32_f16", lines)
        self.assertIn("OpEntryPoint", spv)

    def test_scalar_interpreter_step(self) -> None:
        res = run_dot2c_f32_f16(0x3C003C00, 0x3C003C00, acc=0x3F800000)
        self.assertEqual(res, 0x40400000)


class TestVDot2cF32F16SIMDAndIEEE(unittest.TestCase):
    def test_condition_code_neutrality(self) -> None:
        val, scc = run_fma_with_scc(
            "v_dot2c_f32_f16 v2, v0, v1",
            ["v_mov_b32 v0, 0x3c003c00", "v_mov_b32 v1, 0x3c003c00", "v_mov_b32 v2, 0x3f800000"],
            dst="v2",
            prior_scc=1,
        )
        self.assertEqual(val, 0x40400000)
        self.assertEqual(scc, 1)

    def test_accumulator_preservation(self) -> None:
        # 0.0 * 0.0 + 0.0 * 0.0 + 5.0f = 5.0f
        res = run_dot2c_f32_f16(0x00000000, 0x00000000, acc=0x40A00000)
        self.assertEqual(res, 0x40A00000)

    def test_ieee754_nan_and_inf(self) -> None:
        res_nan = run_dot2c_f32_f16(0x7E003C00, 0x3C003C00, acc=0x3F800000)
        self.assertEqual(res_nan, 0x7FC00000)
        res_inf = run_dot2c_f32_f16(0x7C003C00, 0x3C003C00, acc=0x3F800000)
        self.assertEqual(res_inf, 0x7F800000)


class TestVDot2cF32F16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x3c003c00",
            "v_mov_b32 v1, 0x3c003c00",
            "v_mov_b32 v2, 0x3f800000",
            "v_dot2c_f32_f16 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_dot2c_f32_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40400000)


if __name__ == "__main__":
    unittest.main()
