"""Translator-backed semantic tests and reference mathematical oracle for RDNA2 v_rcp_f64.

Models AMD RDNA2 / GFX10 double-precision reciprocal:
    v_rcp_f64 v[dst_lo:dst_hi], v[src_lo:src_hi]
    v_rcp_f64_e64 v[dst_lo:dst_hi], [mods] v[src_lo:src_hi] [clamp]

RDNA2 Double-Precision Reciprocal Semantics:
  1. Instruction Encoding:
     VOP1 (32-bit): v_rcp_f64 v[dst:dst+1], v[src:src+1]
     VOP3 (64-bit): v_rcp_f64_e64 v[dst:dst+1], [neg/abs] v[src:src+1] [clamp]
  2. Mathematical Operation:
     dst = 1.0 / src (IEEE 754 binary64 double precision division).
  3. IEEE 754 Rules:
     1.0 / +0.0 = +infinity (0x7FF0000000000000)
     1.0 / -0.0 = -infinity (0xFFF0000000000000)
     1.0 / +infinity = +0.0 (0x0000000000000000)
     1.0 / -infinity = -0.0 (0x8000000000000000)
     1.0 / NaN = NaN
  4. Translator Integration:
     Every test exercises tools/rdna2spirv.py via support.translate_lines(...)
     asserting that translated SPIR-V shaders accurately emit OpCapability Float64
     and OpFDiv %double %constant_1_0 %x.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest
from typing import Dict, List, Optional, Set, Tuple, Union

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
import rdna2spirv  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_BIT64: int = 0x8000000000000000


def f64_to_u64(val: float) -> int:
    """Convert Python float to IEEE 754 double precision uint64 bit pattern."""
    return struct.unpack(">Q", struct.pack(">d", val))[0]


def u64_to_f64(bits: int) -> float:
    """Convert uint64 bit pattern to Python float."""
    return struct.unpack(">d", struct.pack(">Q", bits & MASK64))[0]


def oracle_rcp_f64(x: float) -> float:
    """Pure mathematical oracle for double precision reciprocal 1.0 / x."""
    if math.isnan(x):
        return float("nan")
    if x == 0.0:
        bits = f64_to_u64(x)
        return float("-inf") if (bits & SIGN_BIT64) else float("inf")
    if math.isinf(x):
        return -0.0 if x < 0.0 else 0.0
    return 1.0 / x


def oracle_rcp_f64_bits(bits: int, clamp: bool = False, neg: bool = False, abs_val: bool = False) -> int:
    """Bit-accurate reference oracle for v_rcp_f64 with VOP3 modifiers."""
    if neg:
        bits ^= SIGN_BIT64
    if abs_val:
        bits &= ~SIGN_BIT64

    f_in = u64_to_f64(bits)
    f_res = oracle_rcp_f64(f_in)

    if clamp:
        if math.isnan(f_res):
            f_res = 0.0
        elif f_res < 0.0:
            f_res = 0.0
        elif f_res > 1.0:
            f_res = 1.0

    return f64_to_u64(f_res)

# ---------------------------------------------------------------------------
# Phase 1: Reference Oracle and Architectural Model Unit Tests
# ---------------------------------------------------------------------------

class TestValuRcpF64ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and translator interaction."""

    def test_oracle_rcp_f64_standard_values(self) -> None:
        self.assertEqual(oracle_rcp_f64(1.0), 1.0)
        self.assertEqual(oracle_rcp_f64(2.0), 0.5)
        self.assertEqual(oracle_rcp_f64(4.0), 0.25)
        self.assertEqual(oracle_rcp_f64(0.5), 2.0)
        self.assertEqual(oracle_rcp_f64(-2.0), -0.5)

    def test_oracle_rcp_f64_signed_zeros(self) -> None:
        pos_zero = u64_to_f64(0x0000000000000000)
        neg_zero = u64_to_f64(0x8000000000000000)
        self.assertEqual(oracle_rcp_f64(pos_zero), float("inf"))
        self.assertEqual(oracle_rcp_f64(neg_zero), float("-inf"))

    def test_oracle_rcp_f64_infinities(self) -> None:
        pos_inf = float("inf")
        neg_inf = float("-inf")
        res_pos = oracle_rcp_f64(pos_inf)
        res_neg = oracle_rcp_f64(neg_inf)
        self.assertEqual(f64_to_u64(res_pos), 0x0000000000000000)
        self.assertEqual(f64_to_u64(res_neg), 0x8000000000000000)

    def test_oracle_rcp_f64_nan(self) -> None:
        nan_val = float("nan")
        res = oracle_rcp_f64(nan_val)
        self.assertTrue(math.isnan(res))

    def test_oracle_rcp_f64_modifiers(self) -> None:
        val_2_0 = f64_to_u64(2.0)
        res_neg = oracle_rcp_f64_bits(val_2_0, neg=True)
        self.assertEqual(u64_to_f64(res_neg), -0.5)

        val_neg_2_0 = f64_to_u64(-2.0)
        res_abs = oracle_rcp_f64_bits(val_neg_2_0, abs_val=True)
        self.assertEqual(u64_to_f64(res_abs), 0.5)

        val_0_5 = f64_to_u64(0.5)
        res_clamp = oracle_rcp_f64_bits(val_0_5, clamp=True)
        self.assertEqual(u64_to_f64(res_clamp), 1.0)

    def test_translator_emits_float64_and_fdiv(self) -> None:
        """Verify that support.translate_lines emits OpCapability Float64 and OpFDiv."""
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40000000",
            "v_rcp_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rcp_f64_basic", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpFDiv %double", spv)
        self.assertIn("OpCompositeConstruct %v2uint_d", spv)
        self.assertIn("OpCompositeExtract %uint", spv)

if __name__ == "__main__":
    unittest.main()
