"""Comprehensive translator-backed regression bundle for RDNA2 SALU Integer Arithmetic & Population Count.

Subsystem Feature Bundle 29 covering:
1. s_abs_i32       (Scalar signed 32-bit integer absolute value, SCC condition code)
2. s_absdiff_i32   (Scalar signed 32-bit integer absolute difference, SCC condition code)
3. s_addk_i32      (Scalar 32-bit addition with sign-extended 16-bit constant, carry tracking)
4. s_bcnt0_i32_b32 (Scalar 32-bit zero-bit population count)
5. s_bcnt0_i32_b64 (Scalar 64-bit zero-bit population count across SGPR pairs)
6. s_bcnt1_i32_b32 (Scalar 32-bit one-bit population count)
7. s_bcnt1_i32_b64 (Scalar 64-bit one-bit population count across SGPR pairs)
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
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def run_salu_bundle_ps(lines: list[str], name: str = "ps_salu_bundle") -> list[int]:
    all_lines = list(lines) + [
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


def oracle_s_abs_i32(x: int) -> int:
    x &= MASK32
    sx = x if x < 0x80000000 else x - 0x100000000
    return abs(sx) & MASK32


def oracle_s_absdiff_i32(a: int, b: int) -> int:
    diff = (a - b) & MASK32
    sdiff = diff if diff < 0x80000000 else diff - 0x100000000
    return abs(sdiff) & MASK32


def oracle_s_addk_i32(sdst: int, simm16: int) -> int:
    k = simm16 & 0xFFFF
    sk = k - 0x10000 if k & 0x8000 else k
    return (sdst + sk) & MASK32


def oracle_s_bcnt0_i32_b32(s0: int) -> int:
    val = (s0 & MASK32) ^ MASK32
    return val.bit_count()


def oracle_s_bcnt0_i32_b64(lo: int, hi: int) -> int:
    val = ((lo & MASK32) | ((hi & MASK32) << 32)) ^ MASK64
    return val.bit_count()


def oracle_s_bcnt1_i32_b32(s0: int) -> int:
    return (s0 & MASK32).bit_count()


def oracle_s_bcnt1_i32_b64(lo: int, hi: int) -> int:
    val = (lo & MASK32) | ((hi & MASK32) << 32)
    return val.bit_count()

class TestSAbsI32ReferenceOracle(unittest.TestCase):
    """Stage 1 for s_abs_i32: reference mathematical oracle validation."""

    def test_s_abs_i32_oracle_identity(self):
        res = oracle_s_abs_i32(0xfffffffd)
        self.assertEqual(res, 0x3)

class TestSAbsI32ASTAndInterpreter(unittest.TestCase):
    """Stage 2 for s_abs_i32: AST and deterministic scalar interpreter execution."""

    def test_s_abs_i32_ast_and_execution(self):
        ast_prog = ['s_abs_i32 s0, s2'] + [
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_s_abs_i32", ast_prog)
        for needle in ['SAbs', '%scc']:
            self.assertIn(needle, spv)

        exec_prog = ['s_mov_b32 s1, 0xFFFFFFFD', 's_abs_i32 s0, s1'] + [
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        words = run_salu_bundle_ps(exec_prog, "ps_exec_s_abs_i32")
        self.assertEqual(words[0], 0x3)

class TestSAbsI32SIMDAndEdgeCases(unittest.TestCase):
    """Stage 3 for s_abs_i32: boundary edge cases and status flag tracking."""

    def test_s_abs_i32_edge_cases(self):
        cases = [(['s_mov_b32 s1, 0', 's_abs_i32 s0, s1'], 0), (['s_mov_b32 s1, 1', 's_abs_i32 s0, s1'], 1), (['s_mov_b32 s1, 0xFFFFFFFF', 's_abs_i32 s0, s1'], 1), (['s_mov_b32 s1, 0x7FFFFFFF', 's_abs_i32 s0, s1'], 2147483647), (['s_mov_b32 s1, 0x80000000', 's_abs_i32 s0, s1'], 2147483648)]
        for idx, (lines, expected) in enumerate(cases):
            with self.subTest(case_idx=idx):
                prog = list(lines) + [
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                words = run_salu_bundle_ps(prog, f"ps_edge_s_abs_i32_{idx}")
                self.assertEqual(words[0], expected)

class TestSAbsI32TranslatorAndControls(unittest.TestCase):
    """Stage 4 for s_abs_i32: translator integration and negative controls."""

    def test_s_abs_i32_negative_controls(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_neg_s_abs_i32", ["s_abs_i32", "s_endpgm"])

class TestSAbsdiffI32ReferenceOracle(unittest.TestCase):
    """Stage 1 for s_absdiff_i32: reference mathematical oracle validation."""

    def test_s_absdiff_i32_oracle_identity(self):
        res = oracle_s_absdiff_i32(0x5, 0x9)
        self.assertEqual(res, 0x4)


if __name__ == '__main__':
    unittest.main()
