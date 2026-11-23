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
  - End-to-end translation exercises tools/rdna2spirv.py via support.translate_lines(...).
  - 100% translator-backed: zero in-file mock models.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

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

from support import translate_lines

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary64 constants
F64_POS_ZERO: int = 0x0000000000000000
F64_NEG_ZERO: int = 0x8000000000000000
F64_POS_ONE: int = 0x3FF0000000000000
F64_NEG_ONE: int = 0xBFF0000000000000
F64_POS_TWO: int = 0x4000000000000000
F64_NEG_TWO: int = 0xC000000000000000
F64_POS_INF: int = 0x7FF0000000000000
F64_NEG_INF: int = 0xFFF0000000000000
F64_QNAN: int = 0x7FF8000000000000
F64_SNAN: int = 0x7FF0000000000001
F64_MAX_NORMAL: int = 0x7FEFFFFFFFFFFFFF
F64_MIN_NORMAL: int = 0x0010000000000000

# IEEE 754 binary32 constants
F32_POS_ZERO: int = 0x00000000
F32_NEG_ZERO: int = 0x80000000
F32_POS_ONE: int = 0x3F800000
F32_NEG_ONE: int = 0xBF800000
F32_POS_INF: int = 0x7F800000
F32_NEG_INF: int = 0xFF800000
F32_QNAN: int = 0x7FC00000


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
# Opcode 1: v_cvt_f32_f64 Reference Oracle & Translator Tests
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


class VCvtF32F64TranslatorTests(unittest.TestCase):
    """100% translator-backed tests for v_cvt_f32_f64."""

    def test_cvt_f32_f64_translation(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(3.141592653589793) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(3.141592653589793) >> 32) & MASK32:#010x}",
            "v_cvt_f32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_f32_f64_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpFConvert %float", spv)

    def test_cvt_f32_f64_aligned_registers(self):
        lines = [
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0x3FF00000",
            "v_cvt_f32_f64 v10, v[4:5]",
            "exp mrt0 v10, v10, v10, v10 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_f32_f64_regs", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpFConvert %float", spv)

    def test_cvt_f32_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_cvt_f32_f64", ["v_cvt_f32_f64"])


# -----------------------------------------------------------------------------
# Opcode 2: v_cvt_f64_i32 Reference Oracle & Translator Tests
# -----------------------------------------------------------------------------

def oracle_cvt_f64_i32(i32_bits: int) -> int:
    """Reference mathematical oracle for v_cvt_f64_i32 (signed int32 to double)."""
    i32_bits &= MASK32
    if i32_bits & 0x80000000:
        s_val = i32_bits - 0x100000000
    else:
        s_val = i32_bits
    return f64_to_u64(float(s_val))


class VCvtF64I32OracleTests(unittest.TestCase):
    def test_cvt_f64_i32_basic(self):
        self.assertEqual(oracle_cvt_f64_i32(0), F64_POS_ZERO)
        self.assertEqual(oracle_cvt_f64_i32(1), F64_POS_ONE)
        self.assertEqual(oracle_cvt_f64_i32(0xFFFFFFFF), F64_NEG_ONE)
        self.assertEqual(oracle_cvt_f64_i32(42), f64_to_u64(42.0))

    def test_cvt_f64_i32_extremes(self):
        self.assertEqual(oracle_cvt_f64_i32(0x80000000), f64_to_u64(-2147483648.0))
        self.assertEqual(oracle_cvt_f64_i32(0x7FFFFFFF), f64_to_u64(2147483647.0))


class VCvtF64I32TranslatorTests(unittest.TestCase):
    """100% translator-backed tests for v_cvt_f64_i32."""

    def test_cvt_f64_i32_translation(self):
        lines = [
            "v_mov_b32 v2, 42",
            "v_cvt_f64_i32 v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_f64_i32_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpConvertSToF %double", spv)

    def test_cvt_f64_i32_register_pairs(self):
        lines = [
            "v_mov_b32 v6, -100",
            "v_cvt_f64_i32 v[4:5], v6",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_f64_i32_pairs", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpConvertSToF %double", spv)

    def test_cvt_f64_i32_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_cvt_f64_i32", ["v_cvt_f64_i32"])


# -----------------------------------------------------------------------------
# Opcode 3: v_cvt_f64_u32 Reference Oracle & Translator Tests
# -----------------------------------------------------------------------------

def oracle_cvt_f64_u32(u32_bits: int) -> int:
    """Reference mathematical oracle for v_cvt_f64_u32 (unsigned uint32 to double)."""
    u_val = u32_bits & MASK32
    return f64_to_u64(float(u_val))


class VCvtF64U32OracleTests(unittest.TestCase):
    def test_cvt_f64_u32_basic(self):
        self.assertEqual(oracle_cvt_f64_u32(0), F64_POS_ZERO)
        self.assertEqual(oracle_cvt_f64_u32(1), F64_POS_ONE)
        self.assertEqual(oracle_cvt_f64_u32(100), f64_to_u64(100.0))

    def test_cvt_f64_u32_large_unsigned(self):
        self.assertEqual(oracle_cvt_f64_u32(0xFFFFFFFF), f64_to_u64(4294967295.0))
        self.assertEqual(oracle_cvt_f64_u32(0x80000000), f64_to_u64(2147483648.0))


class VCvtF64U32TranslatorTests(unittest.TestCase):
    """100% translator-backed tests for v_cvt_f64_u32."""

    def test_cvt_f64_u32_translation(self):
        lines = [
            "v_mov_b32 v2, 100",
            "v_cvt_f64_u32 v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_f64_u32_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpConvertUToF %double", spv)

    def test_cvt_f64_u32_register_pairs(self):
        lines = [
            "v_mov_b32 v10, 0xFFFFFFFF",
            "v_cvt_f64_u32 v[8:9], v10",
            "exp mrt0 v8, v9, v8, v9 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_f64_u32_pairs", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpConvertUToF %double", spv)

    def test_cvt_f64_u32_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_cvt_f64_u32", ["v_cvt_f64_u32"])


# -----------------------------------------------------------------------------
# Opcode 4: v_rcp_f64 Reference Oracle & Translator Tests
# -----------------------------------------------------------------------------

def oracle_rcp_f64(bits: int) -> int:
    """Reference mathematical oracle for v_rcp_f64 (double reciprocal 1.0 / x)."""
    if is_nan_u64(bits):
        return F64_QNAN
    if bits == F64_POS_ZERO:
        return F64_POS_INF
    if bits == F64_NEG_ZERO:
        return F64_NEG_INF
    if bits == F64_POS_INF:
        return F64_POS_ZERO
    if bits == F64_NEG_INF:
        return F64_NEG_ZERO
    val = u64_to_f64(bits)
    try:
        res = 1.0 / val
        return f64_to_u64(res)
    except ZeroDivisionError:
        return F64_POS_INF if val >= 0 else F64_NEG_INF
    except OverflowError:
        return F64_POS_INF if val > 0 else F64_NEG_INF


class VRcpF64OracleTests(unittest.TestCase):
    def test_rcp_f64_basic(self):
        self.assertEqual(oracle_rcp_f64(F64_POS_ONE), F64_POS_ONE)
        self.assertEqual(oracle_rcp_f64(f64_to_u64(2.0)), f64_to_u64(0.5))
        self.assertEqual(oracle_rcp_f64(f64_to_u64(-2.0)), f64_to_u64(-0.5))
        self.assertEqual(oracle_rcp_f64(f64_to_u64(4.0)), f64_to_u64(0.25))

    def test_rcp_f64_zeros_and_infinities(self):
        self.assertEqual(oracle_rcp_f64(F64_POS_ZERO), F64_POS_INF)
        self.assertEqual(oracle_rcp_f64(F64_NEG_ZERO), F64_NEG_INF)
        self.assertEqual(oracle_rcp_f64(F64_POS_INF), F64_POS_ZERO)
        self.assertEqual(oracle_rcp_f64(F64_NEG_INF), F64_NEG_ZERO)

    def test_rcp_f64_nan_preservation(self):
        self.assertTrue(is_nan_u64(oracle_rcp_f64(F64_QNAN)))
        self.assertTrue(is_nan_u64(oracle_rcp_f64(F64_SNAN)))


class VRcpF64TranslatorTests(unittest.TestCase):
    """100% translator-backed tests for v_rcp_f64."""

    def test_rcp_f64_translation(self):
        lines = [
            f"v_mov_b32 v2, {F64_POS_TWO & MASK32:#010x}",
            f"v_mov_b32 v3, {(F64_POS_TWO >> 32) & MASK32:#010x}",
            "v_rcp_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_rcp_f64_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpFDiv %double", spv)

    def test_rcp_f64_registers(self):
        lines = [
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0x40100000",
            "v_rcp_f64 v[6:7], v[4:5]",
            "exp mrt0 v6, v7, v6, v7 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_rcp_f64_regs", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpFDiv %double", spv)

    def test_rcp_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_rcp_f64", ["v_rcp_f64"])


# -----------------------------------------------------------------------------
# Opcode 5: v_rsq_f64 Reference Oracle & Translator Tests
# -----------------------------------------------------------------------------

def oracle_rsq_f64(bits: int) -> int:
    """Reference mathematical oracle for v_rsq_f64 (double reciprocal square root 1.0 / sqrt(x))."""
    if is_nan_u64(bits):
        return F64_QNAN
    if bits == F64_POS_ZERO:
        return F64_POS_INF
    if bits == F64_NEG_ZERO:
        return F64_NEG_INF
    if bits == F64_POS_INF:
        return F64_POS_ZERO
    val = u64_to_f64(bits)
    if val < 0.0:
        return F64_QNAN
    try:
        res = 1.0 / math.sqrt(val)
        return f64_to_u64(res)
    except ZeroDivisionError:
        return F64_POS_INF
    except OverflowError:
        return F64_POS_INF


class VRsqF64OracleTests(unittest.TestCase):
    def test_rsq_f64_basic(self):
        self.assertEqual(oracle_rsq_f64(F64_POS_ONE), F64_POS_ONE)
        self.assertEqual(oracle_rsq_f64(f64_to_u64(4.0)), f64_to_u64(0.5))
        self.assertEqual(oracle_rsq_f64(f64_to_u64(16.0)), f64_to_u64(0.25))
        self.assertEqual(oracle_rsq_f64(f64_to_u64(0.25)), f64_to_u64(2.0))

    def test_rsq_f64_zeros_and_inf(self):
        self.assertEqual(oracle_rsq_f64(F64_POS_ZERO), F64_POS_INF)
        self.assertEqual(oracle_rsq_f64(F64_NEG_ZERO), F64_NEG_INF)
        self.assertEqual(oracle_rsq_f64(F64_POS_INF), F64_POS_ZERO)

    def test_rsq_f64_negative_nan(self):
        self.assertTrue(is_nan_u64(oracle_rsq_f64(f64_to_u64(-4.0))))
        self.assertTrue(is_nan_u64(oracle_rsq_f64(F64_NEG_INF)))
        self.assertTrue(is_nan_u64(oracle_rsq_f64(F64_QNAN)))


class VRsqF64TranslatorTests(unittest.TestCase):
    """100% translator-backed tests for v_rsq_f64."""

    def test_rsq_f64_translation(self):
        lines = [
            f"v_mov_b32 v2, {f64_to_u64(4.0) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(4.0) >> 32) & MASK32:#010x}",
            "v_rsq_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_rsq_f64_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("InverseSqrt", spv)

    def test_rsq_f64_registers(self):
        lines = [
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0x40300000",
            "v_rsq_f64 v[2:3], v[4:5]",
            "exp mrt0 v2, v3, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_rsq_f64_regs", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("InverseSqrt", spv)

    def test_rsq_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_rsq_f64", ["v_rsq_f64"])


# -----------------------------------------------------------------------------
# Opcode 6: v_sqrt_f64 Reference Oracle & Translator Tests
# -----------------------------------------------------------------------------

def oracle_sqrt_f64(bits: int) -> int:
    """Reference mathematical oracle for v_sqrt_f64 (double square root sqrt(x))."""
    if is_nan_u64(bits):
        return F64_QNAN
    if bits == F64_POS_ZERO:
        return F64_POS_ZERO
    if bits == F64_NEG_ZERO:
        return F64_NEG_ZERO
    if bits == F64_POS_INF:
        return F64_POS_INF
    val = u64_to_f64(bits)
    if val < 0.0:
        return F64_QNAN
    try:
        res = math.sqrt(val)
        return f64_to_u64(res)
    except ValueError:
        return F64_QNAN


class VSqrtF64OracleTests(unittest.TestCase):
    def test_sqrt_f64_basic(self):
        self.assertEqual(oracle_sqrt_f64(F64_POS_ZERO), F64_POS_ZERO)
        self.assertEqual(oracle_sqrt_f64(F64_NEG_ZERO), F64_NEG_ZERO)
        self.assertEqual(oracle_sqrt_f64(F64_POS_ONE), F64_POS_ONE)
        self.assertEqual(oracle_sqrt_f64(f64_to_u64(4.0)), f64_to_u64(2.0))
        self.assertEqual(oracle_sqrt_f64(f64_to_u64(9.0)), f64_to_u64(3.0))
        self.assertEqual(oracle_sqrt_f64(f64_to_u64(2.25)), f64_to_u64(1.5))

    def test_sqrt_f64_inf_and_negative(self):
        self.assertEqual(oracle_sqrt_f64(F64_POS_INF), F64_POS_INF)
        self.assertTrue(is_nan_u64(oracle_sqrt_f64(f64_to_u64(-1.0))))
        self.assertTrue(is_nan_u64(oracle_sqrt_f64(F64_NEG_INF)))
        self.assertTrue(is_nan_u64(oracle_sqrt_f64(F64_QNAN)))


class VSqrtF64TranslatorTests(unittest.TestCase):
    """100% translator-backed tests for v_sqrt_f64."""

    def test_sqrt_f64_translation(self):
        lines = [
            f"v_mov_b32 v2, {f64_to_u64(9.0) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(9.0) >> 32) & MASK32:#010x}",
            "v_sqrt_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_sqrt_f64_test", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("Sqrt", spv)

    def test_sqrt_f64_registers(self):
        lines = [
            "v_mov_b32 v6, 0",
            "v_mov_b32 v7, 0x40300000",
            "v_sqrt_f64 v[4:5], v[6:7]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_sqrt_f64_regs", lines)
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("Sqrt", spv)

    def test_sqrt_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_sqrt_f64", ["v_sqrt_f64"])


# -----------------------------------------------------------------------------
# Mutation Detection & Cross-Opcode Discriminators
# -----------------------------------------------------------------------------

class VF64MutationDetectionTests(unittest.TestCase):
    """Mutation guards to prevent incorrect opcode lowering."""

    def test_mutation_detection_signed_vs_unsigned_cvt(self):
        # 0xFFFFFFFF converts to -1.0 in signed, 4294967295.0 in unsigned
        lines_s = [
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_cvt_f64_i32 v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        lines_u = [
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_cvt_f64_u32 v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv_s = translate_lines("ps_mut_cvt_s", lines_s)
        spv_u = translate_lines("ps_mut_cvt_u", lines_u)
        self.assertIn("OpConvertSToF", spv_s)
        self.assertIn("OpConvertUToF", spv_u)
        self.assertNotIn("OpConvertUToF", spv_s)
        self.assertNotIn("OpConvertSToF", spv_u)

    def test_mutation_detection_sqrt_vs_rsq(self):
        lines_sqrt = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40100000",
            "v_sqrt_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        lines_rsq = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40100000",
            "v_rsq_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv_sqrt = translate_lines("ps_mut_sqrt", lines_sqrt)
        spv_rsq = translate_lines("ps_mut_rsq", lines_rsq)
        self.assertIn("Sqrt", spv_sqrt)
        self.assertIn("InverseSqrt", spv_rsq)
        self.assertNotIn("InverseSqrt", spv_sqrt)

    def test_mutation_detection_rcp_vs_sqrt(self):
        lines_rcp = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x40100000",
            "v_rcp_f64 v[0:1], v[2:3]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv_rcp = translate_lines("ps_mut_rcp", lines_rcp)
        self.assertIn("OpFDiv %double", spv_rcp)
        self.assertNotIn("Sqrt", spv_rcp)


if __name__ == "__main__":
    unittest.main()
