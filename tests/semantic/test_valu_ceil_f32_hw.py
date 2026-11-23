"""Hardware-measured and translator-backed semantic tests for v_ceil_f32.

RDNA2 VALU instruction v_ceil_f32:
Computes the ceiling of a 32-bit floating-point input in the MODE inferred for the
game (IEEE 0, f32 denormals flushed):
    - Denormal operands are flushed to zero with preserving sign:
      ceil(+1e-45) = +0.0, ceil(-1e-45) = -0.0
    - NaNs and Infinities pass through unchanged (signalling NaNs preserved under IEEE 0)
    - Fractional values rounded upwards to smallest integer not less than input
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


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def v_ceil_f32_oracle(u: int) -> int:
    """Exact bit-level mathematical reference oracle for v_ceil_f32.

    Matches RDNA2 hardware execution in game mode (f32 denormals flushed to zero of same sign,
    NaNs preserved unchanged, IEEE 0 mode).
    """
    u = u & MASK32
    exp = u & 0x7F800000
    if exp == 0:
        return 0x80000000 if (u & 0x80000000) else 0x00000000
    if exp == 0x7F800000:
        return u
    f = u32_to_f32(u)
    res_f = math.ceil(f)
    if res_f == 0.0:
        if f < 0.0 or (u & 0x80000000):
            return 0x80000000
        return 0x00000000
    return f32_to_u32(res_f)


def run_ceil_ps(lines: list[str], name: str = "ps_valu_ceil_f32") -> list[int]:
    """Helper to assemble, translate, and run a pixel shader ending with mrt0 export."""
    all_lines = list(lines) + [
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, all_lines)
    return support.mrt0_words(spv)


class CeilReferenceModel:
    """Architectural model tracking vector registers and v_ceil_f32 execution."""

    def __init__(self, num_vgprs: int = 16) -> None:
        self.vgprs = [0] * num_vgprs

    def set_vgpr(self, reg: int, val: int) -> None:
        self.vgprs[reg] = val & MASK32

    def get_vgpr(self, reg: int) -> int:
        return self.vgprs[reg]

    def v_ceil_f32(self, dst: int, src: int) -> int:
        res = v_ceil_f32_oracle(self.vgprs[src])
        self.vgprs[dst] = res
        return res


class TestCeilReferenceOracle(unittest.TestCase):
    """Commit 1: reference mathematical oracle and architectural model for v_ceil_f32."""

    def test_oracle_basic_rounding(self):
        # 1.5 -> 2.0
        self.assertEqual(v_ceil_f32_oracle(0x3FC00000), 0x40000000)
        # -0.5 -> -0.0
        self.assertEqual(v_ceil_f32_oracle(0xBF000000), 0x80000000)
        # 0.0 -> 0.0
        self.assertEqual(v_ceil_f32_oracle(0x00000000), 0x00000000)
        # -0.0 -> -0.0
        self.assertEqual(v_ceil_f32_oracle(0x80000000), 0x80000000)
        # 0.99999994 -> 1.0
        self.assertEqual(v_ceil_f32_oracle(0x3F7FFFFF), 0x3F800000)

    def test_oracle_denormals_and_special_values(self):
        # +denormal -> +0.0
        self.assertEqual(v_ceil_f32_oracle(0x00000001), 0x00000000)
        # -denormal -> -0.0
        self.assertEqual(v_ceil_f32_oracle(0x80000001), 0x80000000)
        # +inf -> +inf
        self.assertEqual(v_ceil_f32_oracle(0x7F800000), 0x7F800000)
        # -inf -> -inf
        self.assertEqual(v_ceil_f32_oracle(0xFF800000), 0xFF800000)
        # sNaN -> sNaN
        self.assertEqual(v_ceil_f32_oracle(0x7FA00000), 0x7FA00000)
        # qNaN -> qNaN
        self.assertEqual(v_ceil_f32_oracle(0x7FC00000), 0x7FC00000)

    def test_translator_execution_matches_oracle(self):
        cases = [
            0x3FC00000,  # 1.5 -> 2.0
            0xBF000000,  # -0.5 -> -0.0
            0x3F7FFFFF,  # 0.99999994 -> 1.0
            0xBF7FFFFF,  # -0.99999994 -> -0.0
            0x00000001,  # +denorm -> +0.0
            0x80000001,  # -denorm -> -0.0
            0x00800000,  # smallest pos normal -> 1.0
            0x80800000,  # smallest neg normal -> -0.0
            0x7F800000,  # +inf
            0xFF800000,  # -inf
            0x7FC00000,  # qNaN
        ]
        for idx, u in enumerate(cases):
            with self.subTest(u=hex(u)):
                expected = v_ceil_f32_oracle(u)
                lines = [
                    f"v_mov_b32 v1, {u:#x}",
                    "v_ceil_f32 v0, v1",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words = run_ceil_ps(lines, f"ps_ceil_c1_{idx}")
                self.assertEqual(words[0], expected)


if __name__ == '__main__':
    unittest.main()
