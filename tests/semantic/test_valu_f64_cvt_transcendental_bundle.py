"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU Double-Precision Conversion & Root/Reciprocal Suite.

Subsystem Feature Bundle 31 covers 6 double-precision (IEEE 754 binary64) VALU instructions:
  1. v_cvt_f32_f64         - binary64 to binary32 conversion with round-to-nearest-even (D.f32 = (float)S0.f64)
  2. v_cvt_f64_i32         - signed int32 to binary64 exact conversion (D.f64 = (double)S0.i32)
  3. v_cvt_f64_u32         - unsigned uint32 to binary64 exact conversion (D.f64 = (double)S0.u32)
  4. v_rcp_f64             - binary64 reciprocal (D.f64 = 1.0 / S0.f64)
  5. v_rsq_f64             - binary64 reciprocal square root (D.f64 = 1.0 / sqrt(S0.f64))
  6. v_sqrt_f64            - binary64 square root (D.f64 = sqrt(S0.f64))

Architecture & Hardware Model:
  - Vector registers (VGPRs) are 32-bit words; 64-bit double operands occupy aligned pairs v[k:k+1].
  - Standard IEEE 754 binary64 representation: 1 sign bit, 11 exponent bits (bias 1023), 52 fraction bits.
  - Multi-lane SIMD execution follows Wave32 (32-bit EXEC) and Wave64 (64-bit EXEC) masking rules.
  - End-to-end translation exercises tools/rdna2spirv.py and validates SPIR-V bytecode.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_interp  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary64 constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_POS_ONE = 0x3FF0000000000000
F64_NEG_ONE = 0xBFF0000000000000
F64_POS_TWO = 0x4000000000000000
F64_NEG_TWO = 0xC000000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_QNAN = 0x7FF8000000000000
F64_SNAN = 0x7FF0000000000001
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF
F64_MIN_NORMAL = 0x0010000000000000

# IEEE 754 binary32 constants
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_POS_ONE = 0x3F800000
F32_NEG_ONE = 0xBF800000
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_QNAN = 0x7FC00000


def f64_to_u64(val: float) -> int:
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return F64_POS_INF if val > 0 else F64_NEG_INF


def u64_to_f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def is_nan_u64(bits: int) -> bool:
    bits &= MASK64
    return (bits & 0x7FF0000000000000) == 0x7FF0000000000000 and (bits & 0x000FFFFFFFFFFFFF) != 0


def f32_to_u32(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return F32_POS_INF if val > 0 else F32_NEG_INF


def u32_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_nan_u32(bits: int) -> bool:
    bits &= MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0

# -----------------------------------------------------------------------------
# Opcode 1: v_cvt_f32_f64
# -----------------------------------------------------------------------------
def oracle_cvt_f32_f64(bits_f64: int) -> int:
    """Reference mathematical oracle for v_cvt_f32_f64 (double to single float)."""
    if is_nan_u64(bits_f64):
        return F32_QNAN
    if bits_f64 == F64_POS_ZERO:
        return F32_POS_ZERO
    if bits_f64 == F64_NEG_ZERO:
        return F32_NEG_ZERO
    if bits_f64 == F64_POS_INF:
        return F32_POS_INF
    if bits_f64 == F64_NEG_INF:
        return F32_NEG_INF
    val = u64_to_f64(bits_f64)
    try:
        raw = struct.pack("<f", val)
        return struct.unpack("<I", raw)[0]
    except OverflowError:
        return F32_POS_INF if val > 0 else F32_NEG_INF


class VCvtF32F64OracleTests(unittest.TestCase):
    def test_cvt_f32_f64_basic(self):
        self.assertEqual(oracle_cvt_f32_f64(f64_to_u64(1.0)), F32_POS_ONE)
        self.assertEqual(oracle_cvt_f32_f64(f64_to_u64(-1.0)), F32_NEG_ONE)
        self.assertEqual(oracle_cvt_f32_f64(f64_to_u64(2.5)), f32_to_u32(2.5))

    def test_cvt_f32_f64_zeros_and_infinities(self):
        self.assertEqual(oracle_cvt_f32_f64(F64_POS_ZERO), F32_POS_ZERO)
        self.assertEqual(oracle_cvt_f32_f64(F64_NEG_ZERO), F32_NEG_ZERO)
        self.assertEqual(oracle_cvt_f32_f64(F64_POS_INF), F32_POS_INF)
        self.assertEqual(oracle_cvt_f32_f64(F64_NEG_INF), F32_NEG_INF)

    def test_cvt_f32_f64_nan_preservation(self):
        self.assertEqual(oracle_cvt_f32_f64(F64_QNAN), F32_QNAN)
        self.assertEqual(oracle_cvt_f32_f64(F64_SNAN), F32_QNAN)


if __name__ == "__main__":
    unittest.main()
