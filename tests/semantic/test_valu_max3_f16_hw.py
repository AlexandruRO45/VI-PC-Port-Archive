"""Semantic tests for RDNA2 v_max3_f16 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_max3_f16 dst, src0, src1, src2
Computes the 16-bit floating-point maximum of three binary16 operands:
    dst[15:0] = max(src0[15:0], src1[15:0], src2[15:0])
Follows IEEE 754-2008 maxNum / GLSL NMax rules:
- Skips NaN operands: if 1 or 2 operands are NaN, returns the maximum of the non-NaN operands.
- If all 3 operands are NaN, returns quiet NaN (0x7E00).
- Signed zeros: -0.0 is strictly less than +0.0 (+0.0 wins over -0.0).
- Destination preservation: the unwritten high half dst[31:16] retains its previous value.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def f16_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def float_to_f16_bits(val: float) -> int:
    return struct.unpack("<H", struct.pack("<e", val))[0]


def ref_max3_f16(a_bits: int, b_bits: int, c_bits: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_max3_f16."""
    vals = [f16_to_float(h) for h in (a_bits, b_bits, c_bits)]
    non_nan = [v for v in vals if not math.isnan(v)]
    if not non_nan:
        return 0x7E00  # quiet NaN
    # Sort with secondary key copysign so +0.0 beats -0.0
    chosen = max(non_nan, key=lambda v: (v, math.copysign(1.0, v)))
    return float_to_f16_bits(chosen)


def run_hw_max3_f16(a_bits: int, b_bits: int, c_bits: int, old_dst: int = 0x12340000) -> int:
    """Execute v_max3_f16 through the rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v0, {old_dst:#x}",
        f"v_mov_b32 v1, {a_bits & MASK16:#x}",
        f"v_mov_b32 v2, {b_bits & MASK16:#x}",
        f"v_mov_b32 v3, {c_bits & MASK16:#x}",
        "v_max3_f16 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_max3_f16_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVMax3F16Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_numerical_cases(self):
        # 1.0 (0x3C00), 2.0 (0x4000), 0.5 (0x3800) -> 2.0 (0x4000)
        self.assertEqual(ref_max3_f16(0x3C00, 0x4000, 0x3800), 0x4000)
        # -1.0 (0xBC00), -2.0 (0xC000), -0.5 (0xB800) -> -0.5 (0xB800)
        self.assertEqual(ref_max3_f16(0xBC00, 0xC000, 0xB800), 0xB800)

    def test_reference_signed_zero(self):
        # +0.0 (0x0000) vs -0.0 (0x8000) vs -1.0 (0xBC00) -> +0.0 wins
        self.assertEqual(ref_max3_f16(0xBC00, 0x0000, 0x8000), 0x0000)
        self.assertEqual(ref_max3_f16(0x8000, 0x8000, 0x0000), 0x0000)

    def test_reference_nan_skipping(self):
        qnan = 0x7E00
        # Single NaN skipped
        self.assertEqual(ref_max3_f16(qnan, 0x3C00, 0x3800), 0x3C00)
        # Two NaNs skipped
        self.assertEqual(ref_max3_f16(qnan, qnan, 0x4000), 0x4000)
        # All NaNs returns NaN
        self.assertEqual(ref_max3_f16(qnan, qnan, qnan), 0x7E00)

    def test_translator_hw_max3(self):
        cases = [
            (0x3C00, 0x4000, 0x3800),  # 1.0, 2.0, 0.5 -> 2.0
            (0xBC00, 0x0000, 0x8000),  # -1.0, +0.0, -0.0 -> +0.0
            (0x7E00, 0x3C00, 0x3800),  # NaN, 1.0, 0.5 -> 1.0
            (0x7C00, 0x4000, 0x3C00),  # +inf, 2.0, 1.0 -> +inf
            (0xFC00, 0xBC00, 0xC000),  # -inf, -1.0, -2.0 -> -1.0
        ]
        for a, b, c in cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                hw_res = run_hw_max3_f16(a, b, c, old_dst=0x12340000)
                expected_lo = ref_max3_f16(a, b, c)
                expected_word = 0x12340000 | expected_lo
                self.assertEqual(hw_res, expected_word)


if __name__ == "__main__":
    unittest.main()
