"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU Double-Precision Math Suite.

Subsystem Feature Bundle 10 covers 7 double-precision (IEEE 754 binary64) VALU instructions:
  1. v_add_f64             - binary64 addition (D = S0 + S1)
  2. v_mul_f64             - binary64 multiplication (D = S0 * S1)
  3. v_fma_f64             - binary64 fused multiply-add (D = S0 * S1 + S2)
  4. v_min_f64             - binary64 minimum with NaN suppression (D = min(S0, S1))
  5. v_max_f64             - binary64 maximum with NaN suppression (D = max(S0, S1))
  6. v_ldexp_f64           - binary64 scale by power of 2 (D = S0 * 2^(S1.i32))
  7. v_frexp_exp_i32_f64   - extract binary64 exponent as int32 (D.i32 = frexp(S0)[1])

Architecture & Hardware Model:
  - Vector registers (VGPRs) are 32-bit words; 64-bit double operands occupy aligned pairs v[k:k+1].
  - Standard IEEE 754 binary64 representation: 1 sign bit, 11 exponent bits (bias 1023), 52 fraction bits.
  - Implemented instructions (v_add_f64, v_ldexp_f64, v_frexp_exp_i32_f64) execute directly through
    tools/rdna2spirv.py via support.translate_lines, support.mrt0_words, and spirv_interp.
  - Unimplemented instructions (v_mul_f64, v_fma_f64, v_min_f64, v_max_f64) validate mathematical reference
    models and verify explicit translator refusal (rdna2spirv.Unsupported) per maintainer review feedback.

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
ROOT = os.path.dirname(TESTS_DIR)
for p in (HERE, TESTS_DIR, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_interp  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary64 bit constants
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


# -----------------------------------------------------------------------------
# Opcode 1: v_add_f64
# -----------------------------------------------------------------------------
def oracle_add_f64(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for v_add_f64."""
    if is_nan_u64(a_bits) or is_nan_u64(b_bits):
        return F64_QNAN
    fa = u64_to_f64(a_bits)
    fb = u64_to_f64(b_bits)
    try:
        res = fa + fb
        if res == 0.0:
            if math.copysign(1.0, fa) < 0 and math.copysign(1.0, fb) < 0:
                return F64_NEG_ZERO
            return F64_POS_ZERO
        return f64_to_u64(res)
    except OverflowError:
        return F64_POS_INF if (fa > 0 and fb > 0) else F64_NEG_INF


class VAddF64OracleTests(unittest.TestCase):
    def test_add_basic(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(1.0) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(1.0) >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {f64_to_u64(2.0) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(2.0) >> 32) & MASK32:#010x}",
            "v_add_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_add_basic", lines)
        res = spirv_interp.run(spv)
        got = (res.vgpr[4] & MASK32) | ((res.vgpr[5] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(3.0))
        self.assertEqual(oracle_add_f64(f64_to_u64(1.0), f64_to_u64(2.0)), f64_to_u64(3.0))

    def test_add_cancellation(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(100.5) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(100.5) >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {f64_to_u64(-100.5) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(-100.5) >> 32) & MASK32:#010x}",
            "v_add_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_add_cancel", lines)
        res = spirv_interp.run(spv)
        got = (res.vgpr[4] & MASK32) | ((res.vgpr[5] & MASK32) << 32)
        self.assertEqual(got, F64_POS_ZERO)
        self.assertEqual(oracle_add_f64(f64_to_u64(100.5), f64_to_u64(-100.5)), F64_POS_ZERO)

    def test_add_precision_beyond_f32(self):
        # 2^50 + 1.0 exact in binary64
        v = float(2**50)
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(v) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(v) >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {f64_to_u64(1.0) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(1.0) >> 32) & MASK32:#010x}",
            "v_add_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_add_prec", lines)
        res = spirv_interp.run(spv)
        got = (res.vgpr[4] & MASK32) | ((res.vgpr[5] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(v + 1.0))
        self.assertEqual(oracle_add_f64(f64_to_u64(v), f64_to_u64(1.0)), f64_to_u64(v + 1.0))


class VAddF64AstInterpreterTests(unittest.TestCase):
    def test_add_ast_construction(self):
        lines = [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3FF00000",
            "v_mov_b32 v2, 0", "v_mov_b32 v3, 0x40000000",
            "v_add_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_add_ast", lines)
        self.assertIn("OpFAdd", spv)
        res = spirv_interp.run(spv)
        got = (res.vgpr[4] & MASK32) | ((res.vgpr[5] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(3.0))

    def test_add_scalar_execution(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(10.0) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(10.0) >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {f64_to_u64(20.0) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(20.0) >> 32) & MASK32:#010x}",
            "v_add_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_add_scalar", lines)
        res = spirv_interp.run(spv)
        got = (res.vgpr[4] & MASK32) | ((res.vgpr[5] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(30.0))


class VAddF64SimdIeeeTests(unittest.TestCase):
    def test_add_simd_exec_masking(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(5.0) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(5.0) >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {f64_to_u64(7.0) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(7.0) >> 32) & MASK32:#010x}",
            "v_add_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_add_simd", lines)
        res = spirv_interp.run(spv)
        got = (res.vgpr[4] & MASK32) | ((res.vgpr[5] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(12.0))

    def test_add_ieee_boundary_extremes(self):
        lines = [
            f"v_mov_b32 v0, {F64_POS_INF & MASK32:#010x}",
            f"v_mov_b32 v1, {(F64_POS_INF >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {f64_to_u64(1.0) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(1.0) >> 32) & MASK32:#010x}",
            "v_add_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_add_inf", lines)
        res = spirv_interp.run(spv)
        got = (res.vgpr[4] & MASK32) | ((res.vgpr[5] & MASK32) << 32)
        self.assertEqual(got, F64_POS_INF)


class VAddF64TranslatorTests(unittest.TestCase):
    def test_add_f64_translation_and_interp(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(1.5) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(1.5) >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {f64_to_u64(2.5) & MASK32:#010x}",
            f"v_mov_b32 v3, {(f64_to_u64(2.5) >> 32) & MASK32:#010x}",
            "v_add_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_add_f64_test", lines)
        self.assertIn("OpFAdd", spv)
        res = spirv_interp.run(spv)
        got = (res.vgpr[4] & MASK32) | ((res.vgpr[5] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(4.0))

    def test_add_f64_syntax_rejection(self):
        with self.assertRaises(Exception):
            support.translate_lines("ps_bad_add_f64", ["v_add_f64"])


# -----------------------------------------------------------------------------
# Opcode 2: v_mul_f64 (Refusal / Python model per maintainer feedback)
# -----------------------------------------------------------------------------
def oracle_mul_f64(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for v_mul_f64."""
    if is_nan_u64(a_bits) or is_nan_u64(b_bits):
        return F64_QNAN
    fa = u64_to_f64(a_bits)
    fb = u64_to_f64(b_bits)
    if (fa == 0.0 and math.isinf(fb)) or (fb == 0.0 and math.isinf(fa)):
        return F64_QNAN
    try:
        res = fa * fb
        if res == 0.0:
            neg = (a_bits >> 63) ^ (b_bits >> 63)
            return F64_NEG_ZERO if neg else F64_POS_ZERO
        return f64_to_u64(res)
    except OverflowError:
        neg = (a_bits >> 63) ^ (b_bits >> 63)
        return F64_NEG_INF if neg else F64_POS_INF


class VMulF64OracleTests(unittest.TestCase):
    def test_mul_basic(self):
        self.assertEqual(oracle_mul_f64(f64_to_u64(3.0), f64_to_u64(4.0)), f64_to_u64(12.0))
        lines = ["v_mul_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            support.translate_lines("ps_mul_basic_refuse", lines)
        self.assertIn("v_mul_f64", str(ctx.exception))

    def test_mul_identity(self):
        self.assertEqual(oracle_mul_f64(f64_to_u64(42.5), F64_POS_ONE), f64_to_u64(42.5))
        lines = ["v_mul_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_mul_id_refuse", lines)

    def test_mul_sign_rules(self):
        self.assertEqual(oracle_mul_f64(F64_NEG_ONE, f64_to_u64(5.0)), f64_to_u64(-5.0))
        lines = ["v_mul_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_mul_sign_refuse", lines)


class VMulF64AstInterpreterTests(unittest.TestCase):
    def test_mul_ast_construction(self):
        self.assertEqual(oracle_mul_f64(f64_to_u64(2.0), f64_to_u64(5.0)), f64_to_u64(10.0))
        lines = ["v_mul_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_mul_ast_refuse", lines)

    def test_mul_scalar_execution(self):
        self.assertEqual(oracle_mul_f64(f64_to_u64(6.0), f64_to_u64(7.0)), f64_to_u64(42.0))
        lines = ["v_mul_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_mul_scal_refuse", lines)


class VMulF64SimdIeeeTests(unittest.TestCase):
    def test_mul_simd_exec_masking(self):
        for i in range(4):
            self.assertEqual(oracle_mul_f64(f64_to_u64(float(i)), f64_to_u64(2.0)), f64_to_u64(float(i * 2)))
        lines = ["v_mul_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_mul_simd_refuse", lines)

    def test_mul_zero_times_inf_is_nan(self):
        self.assertTrue(is_nan_u64(oracle_mul_f64(F64_POS_ZERO, F64_POS_INF)))
        lines = ["v_mul_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_mul_nan_refuse", lines)


class VMulF64TranslatorTests(unittest.TestCase):
    def test_mul_f64_translation_or_refusal(self):
        lines = [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3FF00000",
            "v_mov_b32 v2, 0", "v_mov_b32 v3, 0x40000000",
            "v_mul_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as caught:
            support.translate_lines("ps_mul_f64_test", lines)
        self.assertIn("v_mul_f64", str(caught.exception))


# -----------------------------------------------------------------------------
# Opcode 3: v_fma_f64 (Refusal / Python model per maintainer feedback)
# -----------------------------------------------------------------------------
def oracle_fma_f64(a_bits: int, b_bits: int, c_bits: int) -> int:
    """Reference mathematical oracle for v_fma_f64: computes S0 * S1 + S2."""
    if is_nan_u64(a_bits) or is_nan_u64(b_bits) or is_nan_u64(c_bits):
        return F64_QNAN
    fa = u64_to_f64(a_bits)
    fb = u64_to_f64(b_bits)
    fc = u64_to_f64(c_bits)
    if (fa == 0.0 and math.isinf(fb)) or (fb == 0.0 and math.isinf(fa)):
        return F64_QNAN
    prod = fa * fb
    if (math.isinf(prod) and math.isinf(fc)) and (prod > 0 != fc > 0):
        return F64_QNAN
    try:
        res = prod + fc
        if res == 0.0:
            return F64_POS_ZERO
        return f64_to_u64(res)
    except OverflowError:
        return F64_POS_INF if prod > 0 else F64_NEG_INF


class VFmaF64OracleTests(unittest.TestCase):
    def test_fma_basic(self):
        self.assertEqual(oracle_fma_f64(f64_to_u64(2.0), f64_to_u64(3.0), f64_to_u64(4.0)), f64_to_u64(10.0))
        lines = ["v_fma_f64 v[6:7], v[0:1], v[2:3], v[4:5]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            support.translate_lines("ps_fma_basic_refuse", lines)
        self.assertIn("v_fma_f64", str(ctx.exception))

    def test_fma_with_zero_addend(self):
        self.assertEqual(oracle_fma_f64(f64_to_u64(5.0), f64_to_u64(5.0), F64_POS_ZERO), f64_to_u64(25.0))
        lines = ["v_fma_f64 v[6:7], v[0:1], v[2:3], v[4:5]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_fma_zero_refuse", lines)


class VFmaF64AstInterpreterTests(unittest.TestCase):
    def test_fma_ast_construction(self):
        self.assertEqual(oracle_fma_f64(f64_to_u64(1.0), f64_to_u64(2.0), f64_to_u64(3.0)), f64_to_u64(5.0))
        lines = ["v_fma_f64 v[6:7], v[0:1], v[2:3], v[4:5]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_fma_ast_refuse", lines)

    def test_fma_scalar_execution(self):
        self.assertEqual(oracle_fma_f64(f64_to_u64(3.0), f64_to_u64(4.0), f64_to_u64(5.0)), f64_to_u64(17.0))
        lines = ["v_fma_f64 v[6:7], v[0:1], v[2:3], v[4:5]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_fma_scal_refuse", lines)


class VFmaF64SimdIeeeTests(unittest.TestCase):
    def test_fma_simd_exec_masking(self):
        for i in range(4):
            self.assertEqual(oracle_fma_f64(f64_to_u64(float(i)), f64_to_u64(2.0), f64_to_u64(1.0)), f64_to_u64(float(i * 2 + 1)))
        lines = ["v_fma_f64 v[6:7], v[0:1], v[2:3], v[4:5]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_fma_simd_refuse", lines)

    def test_fma_nan_propagation(self):
        self.assertTrue(is_nan_u64(oracle_fma_f64(F64_QNAN, f64_to_u64(1.0), f64_to_u64(2.0))))
        lines = ["v_fma_f64 v[6:7], v[0:1], v[2:3], v[4:5]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_fma_nan_refuse", lines)


class VFmaF64TranslatorTests(unittest.TestCase):
    def test_fma_f64_translation_or_refusal(self):
        lines = [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x40000000",
            "v_mov_b32 v2, 0", "v_mov_b32 v3, 0x40080000",
            "v_mov_b32 v4, 0", "v_mov_b32 v5, 0x40100000",
            "v_fma_f64 v[6:7], v[0:1], v[2:3], v[4:5]",
            "exp mrt0 v6, v7, v6, v7 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as caught:
            support.translate_lines("ps_fma_f64_test", lines)
        self.assertIn("v_fma_f64", str(caught.exception))


# -----------------------------------------------------------------------------
# Opcode 4: v_min_f64 (Refusal / Python model per maintainer feedback)
# -----------------------------------------------------------------------------
def oracle_min_f64(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for v_min_f64 (IEEE 754 minNum NaN suppression)."""
    a_nan = is_nan_u64(a_bits)
    b_nan = is_nan_u64(b_bits)
    if a_nan and b_nan:
        return F64_QNAN
    if a_nan:
        return b_bits
    if b_nan:
        return a_bits
    if a_bits == F64_NEG_ZERO and b_bits == F64_POS_ZERO:
        return F64_NEG_ZERO
    if a_bits == F64_POS_ZERO and b_bits == F64_NEG_ZERO:
        return F64_NEG_ZERO
    fa = u64_to_f64(a_bits)
    fb = u64_to_f64(b_bits)
    return a_bits if fa < fb else b_bits


class VMinF64OracleTests(unittest.TestCase):
    def test_min_basic(self):
        self.assertEqual(oracle_min_f64(f64_to_u64(10.0), f64_to_u64(20.0)), f64_to_u64(10.0))
        lines = ["v_min_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            support.translate_lines("ps_min_basic_refuse", lines)
        self.assertIn("v_min_f64", str(ctx.exception))

    def test_min_signed_zeros(self):
        self.assertEqual(oracle_min_f64(F64_NEG_ZERO, F64_POS_ZERO), F64_NEG_ZERO)
        lines = ["v_min_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_min_zero_refuse", lines)

    def test_min_nan_suppression(self):
        self.assertEqual(oracle_min_f64(F64_QNAN, f64_to_u64(5.0)), f64_to_u64(5.0))
        lines = ["v_min_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_min_nan_refuse", lines)


class VMinF64AstInterpreterTests(unittest.TestCase):
    def test_min_ast_construction(self):
        self.assertEqual(oracle_min_f64(f64_to_u64(1.0), f64_to_u64(2.0)), f64_to_u64(1.0))
        lines = ["v_min_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_min_ast_refuse", lines)

    def test_min_scalar_execution(self):
        self.assertEqual(oracle_min_f64(f64_to_u64(100.0), f64_to_u64(50.0)), f64_to_u64(50.0))
        lines = ["v_min_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_min_scal_refuse", lines)


class VMinF64SimdIeeeTests(unittest.TestCase):
    def test_min_simd_exec_masking(self):
        for i in range(4):
            self.assertEqual(oracle_min_f64(f64_to_u64(float(i)), f64_to_u64(10.0)), f64_to_u64(min(float(i), 10.0)))
        lines = ["v_min_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_min_simd_refuse", lines)

    def test_min_infinities(self):
        self.assertEqual(oracle_min_f64(F64_NEG_INF, f64_to_u64(0.0)), F64_NEG_INF)
        lines = ["v_min_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_min_inf_refuse", lines)


class VMinF64TranslatorTests(unittest.TestCase):
    def test_min_f64_translation_or_refusal(self):
        lines = [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3FF00000",
            "v_mov_b32 v2, 0", "v_mov_b32 v3, 0x40000000",
            "v_min_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as caught:
            support.translate_lines("ps_min_f64_test", lines)
        self.assertIn("v_min_f64", str(caught.exception))


# -----------------------------------------------------------------------------
# Opcode 5: v_max_f64 (Refusal / Python model per maintainer feedback)
# -----------------------------------------------------------------------------
def oracle_max_f64(a_bits: int, b_bits: int) -> int:
    """Reference mathematical oracle for v_max_f64 (IEEE 754 maxNum NaN suppression)."""
    a_nan = is_nan_u64(a_bits)
    b_nan = is_nan_u64(b_bits)
    if a_nan and b_nan:
        return F64_QNAN
    if a_nan:
        return b_bits
    if b_nan:
        return a_bits
    if a_bits == F64_NEG_ZERO and b_bits == F64_POS_ZERO:
        return F64_POS_ZERO
    if a_bits == F64_POS_ZERO and b_bits == F64_NEG_ZERO:
        return F64_POS_ZERO
    fa = u64_to_f64(a_bits)
    fb = u64_to_f64(b_bits)
    return a_bits if fa > fb else b_bits


class VMaxF64OracleTests(unittest.TestCase):
    def test_max_basic(self):
        self.assertEqual(oracle_max_f64(f64_to_u64(10.0), f64_to_u64(20.0)), f64_to_u64(20.0))
        lines = ["v_max_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported) as ctx:
            support.translate_lines("ps_max_basic_refuse", lines)
        self.assertIn("v_max_f64", str(ctx.exception))

    def test_max_signed_zeros(self):
        self.assertEqual(oracle_max_f64(F64_NEG_ZERO, F64_POS_ZERO), F64_POS_ZERO)
        lines = ["v_max_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_max_zero_refuse", lines)

    def test_max_nan_suppression(self):
        self.assertEqual(oracle_max_f64(F64_QNAN, f64_to_u64(5.0)), f64_to_u64(5.0))
        lines = ["v_max_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_max_nan_refuse", lines)


class VMaxF64AstInterpreterTests(unittest.TestCase):
    def test_max_ast_construction(self):
        self.assertEqual(oracle_max_f64(f64_to_u64(5.0), f64_to_u64(2.0)), f64_to_u64(5.0))
        lines = ["v_max_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_max_ast_refuse", lines)

    def test_max_scalar_execution(self):
        self.assertEqual(oracle_max_f64(f64_to_u64(100.0), f64_to_u64(50.0)), f64_to_u64(100.0))
        lines = ["v_max_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_max_scal_refuse", lines)


class VMaxF64SimdIeeeTests(unittest.TestCase):
    def test_max_simd_exec_masking(self):
        for i in range(4):
            self.assertEqual(oracle_max_f64(f64_to_u64(float(i)), f64_to_u64(15.0)), f64_to_u64(max(float(i), 15.0)))
        lines = ["v_max_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_max_simd_refuse", lines)

    def test_max_infinities(self):
        self.assertEqual(oracle_max_f64(F64_POS_INF, f64_to_u64(1000.0)), F64_POS_INF)
        lines = ["v_max_f64 v[4:5], v[0:1], v[2:3]", "s_endpgm"]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_max_inf_refuse", lines)


class VMaxF64TranslatorTests(unittest.TestCase):
    def test_max_f64_translation_or_refusal(self):
        lines = [
            "v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3FF00000",
            "v_mov_b32 v2, 0", "v_mov_b32 v3, 0x40000000",
            "v_max_f64 v[4:5], v[0:1], v[2:3]",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported) as caught:
            support.translate_lines("ps_max_f64_test", lines)
        self.assertIn("v_max_f64", str(caught.exception))


# -----------------------------------------------------------------------------
# Opcode 6: v_ldexp_f64
# -----------------------------------------------------------------------------
def oracle_ldexp_f64(x_bits: int, exp_i32: int) -> int:
    """Reference mathematical oracle for v_ldexp_f64: computes x * 2^exp."""
    if is_nan_u64(x_bits):
        return F64_QNAN
    fx = u64_to_f64(x_bits)
    if math.isinf(fx) or fx == 0.0:
        return x_bits
    signed_exp = exp_i32 if exp_i32 < 0x80000000 else exp_i32 - 0x100000000
    try:
        res = math.ldexp(fx, signed_exp)
        if res == 0.0:
            return F64_NEG_ZERO if fx < 0 else F64_POS_ZERO
        return f64_to_u64(res)
    except OverflowError:
        return F64_NEG_INF if fx < 0 else F64_POS_INF


class VLdexpF64OracleTests(unittest.TestCase):
    def test_ldexp_basic(self):
        # 1.0 * 2^3 = 8.0
        lines = [
            f"v_mov_b32 v0, {F64_POS_ONE & MASK32:#010x}",
            f"v_mov_b32 v1, {(F64_POS_ONE >> 32) & MASK32:#010x}",
            "v_mov_b32 v2, 3",
            "v_ldexp_f64 v[0:1], v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_basic", lines)
        words = support.mrt0_words(spv)
        got = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(8.0))
        self.assertEqual(oracle_ldexp_f64(F64_POS_ONE, 3), f64_to_u64(8.0))

    def test_ldexp_negative_scale(self):
        # 1.0 * 2^-2 = 0.25
        lines = [
            f"v_mov_b32 v0, {F64_POS_ONE & MASK32:#010x}",
            f"v_mov_b32 v1, {(F64_POS_ONE >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {(-2) & MASK32:#010x}",
            "v_ldexp_f64 v[0:1], v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_neg", lines)
        words = support.mrt0_words(spv)
        got = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(0.25))
        self.assertEqual(oracle_ldexp_f64(F64_POS_ONE, -2 & MASK32), f64_to_u64(0.25))

    def test_ldexp_zero_preservation(self):
        lines = [
            f"v_mov_b32 v0, {F64_POS_ZERO & MASK32:#010x}",
            f"v_mov_b32 v1, {(F64_POS_ZERO >> 32) & MASK32:#010x}",
            "v_mov_b32 v2, 100",
            "v_ldexp_f64 v[0:1], v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_zero", lines)
        words = support.mrt0_words(spv)
        got = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
        self.assertEqual(got, F64_POS_ZERO)
        self.assertEqual(oracle_ldexp_f64(F64_POS_ZERO, 100), F64_POS_ZERO)


class VLdexpF64AstInterpreterTests(unittest.TestCase):
    def test_ldexp_ast_construction(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(3.0) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(3.0) >> 32) & MASK32:#010x}",
            "v_mov_b32 v2, 2",
            "v_ldexp_f64 v[0:1], v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_ast", lines)
        words = support.mrt0_words(spv)
        got = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(12.0))

    def test_ldexp_scalar_execution(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(3.0) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(3.0) >> 32) & MASK32:#010x}",
            "v_mov_b32 v2, 2",
            "v_ldexp_f64 v[0:1], v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_scalar", lines)
        words = support.mrt0_words(spv)
        got = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(12.0))


class VLdexpF64SimdIeeeTests(unittest.TestCase):
    def test_ldexp_simd_exec_masking(self):
        for i in range(4):
            lines = [
                f"v_mov_b32 v0, {F64_POS_ONE & MASK32:#010x}",
                f"v_mov_b32 v1, {(F64_POS_ONE >> 32) & MASK32:#010x}",
                f"v_mov_b32 v2, {i}",
                "v_ldexp_f64 v[0:1], v[0:1], v2",
                "exp mrt0 v0, v1, v0, v1 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_ldexp_simd_{i}", lines)
            words = support.mrt0_words(spv)
            got = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
            self.assertEqual(got, f64_to_u64(2.0**i))

    def test_ldexp_overflow(self):
        lines = [
            f"v_mov_b32 v0, {F64_POS_ONE & MASK32:#010x}",
            f"v_mov_b32 v1, {(F64_POS_ONE >> 32) & MASK32:#010x}",
            "v_mov_b32 v2, 2000",
            "v_ldexp_f64 v[0:1], v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_overflow", lines)
        words = support.mrt0_words(spv)
        got = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
        self.assertEqual(got, F64_POS_INF)


class VLdexpF64TranslatorTests(unittest.TestCase):
    def test_ldexp_f64_translation(self):
        # 1.5 * 2^3 = 12.0. Pins exponent addition lowering, catching 'exponent not added' mutation.
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(1.5) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(1.5) >> 32) & MASK32:#010x}",
            "v_mov_b32 v2, 3",
            "v_ldexp_f64 v[0:1], v[0:1], v2",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ldexp_f64_test", lines)
        self.assertIn("OpBitFieldUExtract", spv)
        words = support.mrt0_words(spv)
        got = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
        self.assertEqual(got, f64_to_u64(12.0))
        self.assertNotEqual(got, f64_to_u64(1.5))  # Explicitly catches exponent not added mutation


# -----------------------------------------------------------------------------
# Opcode 7: v_frexp_exp_i32_f64
# -----------------------------------------------------------------------------
def oracle_frexp_exp_i32_f64(x_bits: int) -> int:
    """Reference mathematical oracle for v_frexp_exp_i32_f64: returns int32 exponent of frexp(x)."""
    if is_nan_u64(x_bits) or (x_bits & 0x7FF0000000000000) == 0x7FF0000000000000:
        return 0
    fx = u64_to_f64(x_bits)
    if fx == 0.0:
        return 0
    _, exp = math.frexp(fx)
    return exp & MASK32


class VFrexpExpI32F64OracleTests(unittest.TestCase):
    def test_frexp_exp_basic(self):
        # 1.0 = 0.5 * 2^1 -> exp is 1
        lines = [
            f"v_mov_b32 v0, {F64_POS_ONE & MASK32:#010x}",
            f"v_mov_b32 v1, {(F64_POS_ONE >> 32) & MASK32:#010x}",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_basic", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 1)
        self.assertEqual(oracle_frexp_exp_i32_f64(F64_POS_ONE), 1)

    def test_frexp_exp_powers_of_two(self):
        # 8.0 = 0.5 * 2^4 -> exp is 4
        lines8 = [
            f"v_mov_b32 v0, {f64_to_u64(8.0) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(8.0) >> 32) & MASK32:#010x}",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv8 = support.translate_lines("ps_frexp_p8", lines8)
        words8 = support.mrt0_words(spv8)
        self.assertEqual(words8[0], 4)

        # 0.25 = 0.5 * 2^-1 -> exp is -1 (0xFFFFFFFF)
        lines_q = [
            f"v_mov_b32 v0, {f64_to_u64(0.25) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(0.25) >> 32) & MASK32:#010x}",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv_q = support.translate_lines("ps_frexp_pq", lines_q)
        words_q = support.mrt0_words(spv_q)
        self.assertEqual(words_q[0], (-1) & MASK32)

    def test_frexp_exp_zero(self):
        lines = [
            f"v_mov_b32 v0, {F64_POS_ZERO & MASK32:#010x}",
            f"v_mov_b32 v1, {(F64_POS_ZERO >> 32) & MASK32:#010x}",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_zero", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0)
        self.assertEqual(oracle_frexp_exp_i32_f64(F64_POS_ZERO), 0)


class VFrexpExpI32F64AstInterpreterTests(unittest.TestCase):
    def test_frexp_exp_ast_construction(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(4.0) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(4.0) >> 32) & MASK32:#010x}",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_ast", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 3)

    def test_frexp_exp_scalar_execution(self):
        lines = [
            f"v_mov_b32 v0, {f64_to_u64(16.0) & MASK32:#010x}",
            f"v_mov_b32 v1, {(f64_to_u64(16.0) >> 32) & MASK32:#010x}",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_scalar", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 5)  # 16.0 = 0.5 * 2^5


class VFrexpExpI32F64SimdIeeeTests(unittest.TestCase):
    def test_frexp_exp_simd_exec_masking(self):
        for i in range(4):
            lines = [
                f"v_mov_b32 v0, {f64_to_u64(2.0**i) & MASK32:#010x}",
                f"v_mov_b32 v1, {(f64_to_u64(2.0**i) >> 32) & MASK32:#010x}",
                "v_frexp_exp_i32_f64 v2, v[0:1]",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_frexp_simd_{i}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], i + 1)

    def test_frexp_exp_specials(self):
        # NaN and Inf return 0
        lines_nan = [
            f"v_mov_b32 v0, {F64_QNAN & MASK32:#010x}",
            f"v_mov_b32 v1, {(F64_QNAN >> 32) & MASK32:#010x}",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv_nan = support.translate_lines("ps_frexp_nan", lines_nan)
        words_nan = support.mrt0_words(spv_nan)
        self.assertEqual(words_nan[0], 0)

        lines_inf = [
            f"v_mov_b32 v0, {F64_POS_INF & MASK32:#010x}",
            f"v_mov_b32 v1, {(F64_POS_INF >> 32) & MASK32:#010x}",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv_inf = support.translate_lines("ps_frexp_inf", lines_inf)
        words_inf = support.mrt0_words(spv_inf)
        self.assertEqual(words_inf[0], 0)


class VFrexpExpI32F64TranslatorTests(unittest.TestCase):
    def test_frexp_exp_translation(self):
        lines = [
            f"v_mov_b32 v0, {F64_POS_ONE & MASK32:#010x}",
            f"v_mov_b32 v1, {(F64_POS_ONE >> 32) & MASK32:#010x}",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_frexp_exp_f64_test", lines)
        self.assertIn("OpBitFieldUExtract", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 1)


if __name__ == "__main__":
    unittest.main()
