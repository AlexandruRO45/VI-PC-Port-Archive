"""Hardware-measured and translator-backed semantic tests for s_mulk_i32.

RDNA2 SALU instruction s_mulk_i32:
Multiplies a 32-bit scalar general-purpose register (sD) by a sign-extended 16-bit
signed immediate (simm16) and stores the 32-bit result back in sD:
    sD = (sD * sign_extend_16(simm16)) & 0xFFFFFFFF
Condition codes: does not modify SCC (Scalar Condition Code).
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


def s_mulk_i32_oracle(s_val: int, simm16: int) -> int:
    """Exact bit-level mathematical reference oracle for s_mulk_i32.

    Takes a 32-bit integer s_val and 16-bit integer simm16.
    Sign extends simm16 to a signed 32-bit value and performs two's-complement
    32-bit multiplication.
    """
    s_val = s_val & MASK32
    u16 = simm16 & 0xFFFF
    k = u16 - 0x10000 if (u16 & 0x8000) else u16
    signed_s = s_val - 0x100000000 if (s_val & 0x80000000) else s_val
    prod = signed_s * k
    return prod & MASK32


def run_mulk_ps(lines: list[str], name: str = "ps_salu_mulk_i32") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class MulkReferenceModel:
    """Architectural model tracking scalar registers and s_mulk_i32 execution."""

    def __init__(self, num_sgprs: int = 16) -> None:
        self.sgprs = [0] * num_sgprs

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg]

    def s_mulk_i32(self, reg: int, simm16: int) -> int:
        res = s_mulk_i32_oracle(self.sgprs[reg], simm16)
        self.sgprs[reg] = res
        return res


class TestMulkReferenceOracle(unittest.TestCase):
    """Commit 1: reference mathematical oracle and architectural model for s_mulk_i32."""

    def test_oracle_basic_multiplication(self):
        self.assertEqual(s_mulk_i32_oracle(7, 5), 35)
        self.assertEqual(s_mulk_i32_oracle(10, 0), 0)
        self.assertEqual(s_mulk_i32_oracle(12345, 1), 12345)
        self.assertEqual(s_mulk_i32_oracle(12345, -1), (-12345) & MASK32)
        self.assertEqual(s_mulk_i32_oracle(7, -3), (-21) & MASK32)

    def test_oracle_sign_extension(self):
        self.assertEqual(s_mulk_i32_oracle(42, 0xFFFF), (-42) & MASK32)
        self.assertEqual(s_mulk_i32_oracle(2, 0x7FFF), 65534)
        self.assertEqual(s_mulk_i32_oracle(2, 0x8000), (-65536) & MASK32)

    def test_translator_execution_matches_oracle(self):
        test_cases = [
            (7, 5),
            (100, -3),
            (0, 100),
            (42, 1),
            (42, -1),
            (1000, 32),
            (0x12345678, 2),
            (0x7FFFFFFF, 2),
            (0x80000000, 2),
            (0x1000, 0x7FFF),
            (0x1000, -32768),
        ]
        for s_init, k in test_cases:
            with self.subTest(s_init=hex(s_init), k=k):
                expected = s_mulk_i32_oracle(s_init, k)
                lines = [
                    f"s_mov_b32 s0, {s_init:#x}",
                    f"s_mulk_i32 s0, {k}",
                    "v_mov_b32 v0, s0",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_mulk_ps(lines, f"ps_mulk_c1_{s_init & 0xFFFF}_{abs(k)}")
                self.assertEqual(words[0], expected)


if __name__ == '__main__':
    unittest.main()
