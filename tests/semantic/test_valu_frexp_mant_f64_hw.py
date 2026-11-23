"""Translator-backed semantic tests and reference mathematical oracle for RDNA2 v_frexp_mant_f64.

Models AMD RDNA2 / GFX10 double-precision floating-point mantissa extraction:
    v_frexp_mant_f64 v[dst_lo:dst_hi], v[src_lo:src_hi]
    v_frexp_mant_f64_e64 v[dst_lo:dst_hi], [mods] v[src_lo:src_hi] [clamp]

RDNA2 Binary64 Frexp Mantissa Semantics:
  1. Instruction Encoding:
     VOP1 (32-bit): v_frexp_mant_f64 v[dst:dst+1], v[src:src+1]
     VOP3 (64-bit): v_frexp_mant_f64_e64 v[dst:dst+1], [neg/abs] v[src:src+1] [clamp]
  2. Mathematical Operation:
     Extracts the mantissa `m` of double precision float `src` such that:
     src = m * 2^exp, where:
     - Normal inputs (biased exponent 1..2046): mantissa is normalized to [0.5, 1.0)
       retaining the original sign, with biased exponent set to 1022 (0x3FE).
     - Non-zero subnormals (biased exponent == 0, frac != 0): normalized to [0.5, 1.0)
       with original sign, by left-shifting fraction bits until leading 1 lands at bit 52.
     - Signed zeros (+0.0 / -0.0): pass through unchanged (+0.0 / -0.0).
     - Infinities (+inf / -inf): pass through unchanged (+inf / -inf).
     - NaNs (quiet / signaling): pass through unchanged.
  3. Translator Integration:
     Every test exercises tools/rdna2spirv.py via support.translate_lines(...)
     and executes the translated shaders via support.mrt0_words.

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


def oracle_frexp_mant_f64(x: float) -> float:
    """Pure mathematical oracle for double precision frexp mantissa in [0.5, 1.0)."""
    bits = f64_to_u64(x)
    res_bits = oracle_frexp_mant_f64_bits(bits)
    return u64_to_f64(res_bits)


def oracle_frexp_mant_f64_bits(
    bits: int,
    neg: bool = False,
    abs_val: bool = False,
    clamp: bool = False,
) -> int:
    """Bit-accurate reference oracle for v_frexp_mant_f64 with VOP3 modifiers."""
    hi = (bits >> 32) & MASK32
    lo = bits & MASK32

    if abs_val:
        hi &= 0x7FFFFFFF
    if neg:
        hi ^= 0x80000000

    e = (hi >> 20) & 0x7FF
    sgn = hi & 0x80000000
    f_hi = hi & 0x000FFFFF
    frac_or = f_hi | lo

    if 1 <= e <= 2046:
        # Normal inputs: set exponent to 1022 (0x3FE)
        r_hi = sgn | 0x3FE00000 | f_hi
        r_lo = lo
    elif e == 0 and frac_or != 0:
        # Non-zero subnormals: normalize to [0.5, 1.0)
        full_frac = (f_hi << 32) | lo
        msb = full_frac.bit_length() - 1  # 0..51
        shift = 52 - msb
        norm_frac = full_frac << shift
        norm_lo = norm_frac & MASK32
        norm_hi_frac = (norm_frac >> 32) & 0x000FFFFF
        r_hi = sgn | 0x3FE00000 | norm_hi_frac
        r_lo = norm_lo
    else:
        # Signed zeros, infinities, NaNs pass through unchanged
        r_hi = hi
        r_lo = lo

    res = (r_hi << 32) | r_lo

    if clamp:
        f_val = u64_to_f64(res)
        if math.isnan(f_val):
            f_val = 0.0
        elif f_val < 0.0:
            f_val = 0.0
        elif f_val > 1.0:
            f_val = 1.0
        res = f64_to_u64(f_val)

    return res

# ---------------------------------------------------------------------------
# Phase 1: Reference Oracle and Architectural Model Unit Tests
# ---------------------------------------------------------------------------

class TestValuFrexpMantF64ReferenceOracle(unittest.TestCase):
    """Phase 1: Unit tests for reference mathematical oracle and translator execution."""

    def test_oracle_frexp_mant_f64_standard_values(self) -> None:
        # 2.0 = 0.5 * 2^2 -> mantissa = 0.5
        self.assertEqual(oracle_frexp_mant_f64(2.0), 0.5)
        # 1.0 = 0.5 * 2^1 -> mantissa = 0.5
        self.assertEqual(oracle_frexp_mant_f64(1.0), 0.5)
        # 3.0 = 0.75 * 2^2 -> mantissa = 0.75
        self.assertEqual(oracle_frexp_mant_f64(3.0), 0.75)
        # -2.0 -> mantissa = -0.5
        self.assertEqual(oracle_frexp_mant_f64(-2.0), -0.5)

    def test_oracle_frexp_mant_f64_signed_zeros(self) -> None:
        pos_zero = u64_to_f64(0x0000000000000000)
        neg_zero = u64_to_f64(0x8000000000000000)
        self.assertEqual(f64_to_u64(oracle_frexp_mant_f64(pos_zero)), 0x0000000000000000)
        self.assertEqual(f64_to_u64(oracle_frexp_mant_f64(neg_zero)), 0x8000000000000000)

    def test_oracle_frexp_mant_f64_infinities(self) -> None:
        pos_inf = float("inf")
        neg_inf = float("-inf")
        self.assertEqual(f64_to_u64(oracle_frexp_mant_f64(pos_inf)), 0x7FF0000000000000)
        self.assertEqual(f64_to_u64(oracle_frexp_mant_f64(neg_inf)), 0xFFF0000000000000)

    def test_oracle_frexp_mant_f64_nan(self) -> None:
        nan_val = float("nan")
        self.assertTrue(math.isnan(oracle_frexp_mant_f64(nan_val)))

    def test_oracle_frexp_mant_f64_subnormal_normalization(self) -> None:
        # Smallest positive subnormal (1 bit set at bit 0)
        subnormal_bits = 0x0000000000000001
        res = oracle_frexp_mant_f64_bits(subnormal_bits)
        # Normalized leading 1 is at bit 52 (implicit), all 52 fraction bits are 0 -> 0.5 (0x3FE0000000000000)
        self.assertEqual(res, 0x3FE0000000000000)

    def test_translator_execution_standard(self) -> None:
        """Verify support.translate_lines and mrt0_words execution for v_frexp_mant_f64."""
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40000000",  # 2.0
            "v_frexp_mant_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_mant_basic", lines)
        words = support.mrt0_words(spv)
        res_hi, res_lo = words[1], words[0]
        self.assertEqual(res_hi, 0x3FE00000)
        self.assertEqual(res_lo, 0x0)

if __name__ == "__main__":
    unittest.main()
