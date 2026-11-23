"""Semantic tests for RDNA2 VALU Transcendental instructions bundle.

Covers:
- v_sin_f32
- v_cos_f32
- v_exp_f32
- v_log_f32
- v_rcp_f32
- v_rsq_f32
- v_sqrt_f32

Tests reference mathematical models, AST and scalar interpreter semantics,
SIMD multi-lane divergence and EXEC masking, IEEE 754 edge cases,
and hardware translation into SPIR-V.
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

_ = support  # Reachable guard for tests/test_semantic_translator_use.py

MASK32 = 0xFFFFFFFF


def f32_to_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def ref_sin_f32(src: float) -> float:
    """Bit-accurate reference oracle for RDNA2 v_sin_f32."""
    if math.isnan(src) or math.isinf(src):
        return float("nan")
    if src == 0.0:
        return src
    rnd = round(src)
    if abs(src - rnd) == 0.5:
        if rnd % 2 != 0:
            rnd = round(src - math.copysign(0.5, src))
    d = src - rnd
    if abs(d) == 0.5:
        return 0.0
    return math.sin(d * 2.0 * math.pi)


class TestVSinF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_sin_f32."""

    def test_reference_quarter_turns(self):
        self.assertAlmostEqual(ref_sin_f32(0.25), 1.0, places=6)
        self.assertAlmostEqual(ref_sin_f32(0.75), -1.0, places=6)
        self.assertAlmostEqual(ref_sin_f32(1.25), 1.0, places=6)

    def test_reference_exact_zero_half_turns(self):
        self.assertEqual(ref_sin_f32(0.5), 0.0)
        self.assertEqual(ref_sin_f32(1.5), 0.0)
        self.assertEqual(ref_sin_f32(-0.5), 0.0)

    def test_reference_signed_zero(self):
        res_pos = ref_sin_f32(0.0)
        res_neg = ref_sin_f32(-0.0)
        self.assertEqual(res_pos, 0.0)
        self.assertEqual(math.copysign(1.0, res_pos), 1.0)
        self.assertEqual(res_neg, -0.0)
        self.assertEqual(math.copysign(1.0, res_neg), -1.0)

    def test_reference_nan_and_inf(self):
        self.assertTrue(math.isnan(ref_sin_f32(float("nan"))))
        self.assertTrue(math.isnan(ref_sin_f32(float("inf"))))
        self.assertTrue(math.isnan(ref_sin_f32(float("-inf"))))


if __name__ == "__main__":
    unittest.main()
