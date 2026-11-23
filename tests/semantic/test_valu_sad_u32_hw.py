"""Hardware-measured and translator-backed semantic tests for v_sad_u32.

RDNA2 VALU instruction v_sad_u32:
Computes the sum of unsigned absolute difference between 32-bit unsigned inputs
plus accumulator:
    v[d] = (|src0 - src1| + src2) & 0xFFFFFFFF
All operations are 32-bit unsigned wrapping arithmetic.
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


def v_sad_u32_oracle(s0: int, s1: int, s2: int) -> int:
    """Exact bit-level mathematical reference oracle for v_sad_u32.

    Computes unsigned absolute difference |s0 - s1| + s2 with 32-bit unsigned wrapping.
    """
    s0 = s0 & MASK32
    s1 = s1 & MASK32
    s2 = s2 & MASK32
    diff = s0 - s1 if s0 >= s1 else s1 - s0
    return (diff + s2) & MASK32


def run_sad_ps(lines: list[str], name: str = "ps_valu_sad_u32") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class SadReferenceModel:
    """Architectural model tracking vector registers and v_sad_u32 execution."""

    def __init__(self, num_vgprs: int = 16) -> None:
        self.vgprs = [0] * num_vgprs

    def set_vgpr(self, reg: int, val: int) -> None:
        self.vgprs[reg] = val & MASK32

    def get_vgpr(self, reg: int) -> int:
        return self.vgprs[reg]

    def v_sad_u32(self, dst: int, src0: int, src1: int, src2: int) -> int:
        res = v_sad_u32_oracle(self.vgprs[src0], self.vgprs[src1], self.vgprs[src2])
        self.vgprs[dst] = res
        return res


class TestSadReferenceOracle(unittest.TestCase):
    """Commit 1: reference mathematical oracle and architectural model for v_sad_u32."""

    def test_oracle_basic_differences(self):
        # |100 - 35| + 10 = 65 + 10 = 75
        self.assertEqual(v_sad_u32_oracle(100, 35, 10), 75)
        # |35 - 100| + 10 = 65 + 10 = 75 (symmetric difference)
        self.assertEqual(v_sad_u32_oracle(35, 100, 10), 75)
        # |42 - 42| + 0 = 0
        self.assertEqual(v_sad_u32_oracle(42, 42, 0), 0)
        # |0 - 0xFFFFFFFF| + 1 = 0xFFFFFFFF + 1 = 0 (wrap)
        self.assertEqual(v_sad_u32_oracle(0, 0xFFFFFFFF, 1), 0)

    def test_translator_execution_matches_oracle(self):
        cases = [
            (100, 35, 10),
            (35, 100, 10),
            (50, 50, 25),
            (0, 0, 0),
            (0xFFFFFFFF, 0, 0),
            (0, 0xFFFFFFFF, 1),
            (0x80000000, 0x10000000, 0x05),
            (0x7FFFFFFF, 0x7FFFFFFF, 0x1234),
        ]
        for idx, (s0, s1, s2) in enumerate(cases):
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                expected = v_sad_u32_oracle(s0, s1, s2)
                lines = [
                    f"v_mov_b32 v1, {s0:#x}",
                    f"v_mov_b32 v2, {s1:#x}",
                    f"v_mov_b32 v3, {s2:#x}",
                    "v_sad_u32 v0, v1, v2, v3",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_sad_ps(lines, f"ps_sad_c1_{idx}")
                self.assertEqual(words[0], expected)


if __name__ == '__main__':
    unittest.main()
