"""Hardware-measured and translator-backed semantic tests for v_mul_i32_i24.

RDNA2 VALU instruction v_mul_i32_i24:
Multiplies the low 24 bits of two inputs, treating each as a 24-bit signed integer
(sign-extended from bit 23), and returns the low 32 bits of the 48-bit product:
    v[d] = (sext24(src0) * sext24(src1)) & 0xFFFFFFFF
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def sext24(v: int) -> int:
    """Sign-extend a 24-bit integer to signed Python integer."""
    u24 = v & 0xFFFFFF
    return u24 - 0x1000000 if (u24 & 0x800000) else u24


def v_mul_i32_i24_oracle(x: int, y: int) -> int:
    """Exact bit-level mathematical reference oracle for v_mul_i32_i24."""
    sx = sext24(x)
    sy = sext24(y)
    prod = sx * sy
    return prod & MASK32


def run_mul24_ps(lines: list[str], name: str = "ps_valu_mul_i32_i24") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class Mul24ReferenceModel:
    """Architectural model tracking vector registers and v_mul_i32_i24 execution."""

    def __init__(self, num_vgprs: int = 16) -> None:
        self.vgprs = [0] * num_vgprs

    def set_vgpr(self, reg: int, val: int) -> None:
        self.vgprs[reg] = val & MASK32

    def get_vgpr(self, reg: int) -> int:
        return self.vgprs[reg]

    def v_mul_i32_i24(self, dst: int, src0: int, src1: int) -> int:
        res = v_mul_i32_i24_oracle(self.vgprs[src0], self.vgprs[src1])
        self.vgprs[dst] = res
        return res


class TestMul24ReferenceOracle(unittest.TestCase):
    """Commit 1: reference mathematical oracle and architectural model for v_mul_i32_i24."""

    def test_oracle_basic_multiplication(self):
        # 10 * 20 = 200
        self.assertEqual(v_mul_i32_i24_oracle(10, 20), 200)
        # 0xFFFFFF (-1) * 5 = -5
        self.assertEqual(v_mul_i32_i24_oracle(0xFFFFFF, 5), (-5) & MASK32)
        # 0xFFFFFF (-1) * 0xFFFFFF (-1) = 1
        self.assertEqual(v_mul_i32_i24_oracle(0xFFFFFF, 0xFFFFFF), 1)
        # 0x7FFFFF (max pos 24-bit: 8388607) * 2 = 16777214
        self.assertEqual(v_mul_i32_i24_oracle(0x7FFFFF, 2), 16777214)
        # 0x800000 (min neg 24-bit: -8388608) * 2 = -16777216
        self.assertEqual(v_mul_i32_i24_oracle(0x800000, 2), (-16777216) & MASK32)

    def test_translator_execution_matches_oracle(self):
        cases = [
            (10, 20),
            (0xFFFFFF, 5),
            (0xFFFFFF, 0xFFFFFF),
            (1000, -3),
            (0x7FFFFF, 2),
            (0x800000, 2),
            (0x123456, 0x10),
            (0, 0x123456),
        ]
        for idx, (x, y) in enumerate(cases):
            with self.subTest(x=hex(x), y=hex(y)):
                expected = v_mul_i32_i24_oracle(x, y)
                lines = [
                    f"v_mov_b32 v1, {x:#x}",
                    f"v_mov_b32 v2, {y:#x}",
                    "v_mul_i32_i24 v0, v1, v2",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_mul24_ps(lines, f"ps_mul24_c1_{idx}")
                self.assertEqual(words[0], expected)


if __name__ == '__main__':
    unittest.main()
