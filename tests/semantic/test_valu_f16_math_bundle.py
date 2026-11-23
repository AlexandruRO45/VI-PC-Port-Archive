"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU Float16 Math Suite.

Subsystem Feature Bundle 23 covers 4 half-precision (IEEE 754 binary16) exponential/mantissa/scale instructions:
  1. v_fract_f16          - fractional part computation (x - floor(x)) with 0xfe00 on infinities
  2. v_frexp_exp_i16_f16  - integer exponent extraction as 16-bit signed integer
  3. v_frexp_mant_f16     - normalized significand extraction in [0.5, 1.0) with source sign
  4. v_ldexp_f16          - scale by power of 2 with RNE rounding across subnormals

RDNA2 Float16 Architecture:
  - Operates on 16-bit half-precision floats stored in the lower half of 32-bit VGPRs.
  - Standard IEEE 754 binary16 format: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits.
  - Denormals are preserved and normalized.
  - 100% translator-backed execution: every test compiles assembly through tools/rdna2spirv.py
    via support.translate_lines and verifies computed results through support.mrt0_words.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import os
import struct
import sys
from typing import Optional
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (TOOLS_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

# Float16 constants
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_POS_ONE = 0x3C00   # 1.0
F16_NEG_ONE = 0xBC00   # -1.0
F16_POS_HALF = 0x3800  # 0.5
F16_NEG_HALF = 0xB800  # -0.5
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_QNAN = 0x7E00
F16_MAX_NORMAL = 0x7BFF
F16_MIN_NORMAL = 0x0400


def f16_to_u16(f: float) -> int:
    return struct.unpack(">H", struct.pack(">e", f))[0]


def u16_to_f16(u: int) -> float:
    return struct.unpack(">e", struct.pack(">H", u & MASK16))[0]


def is_nan_f16(u: int) -> bool:
    return (u & 0x7C00) == 0x7C00 and (u & 0x03FF) != 0


def is_inf_f16(u: int) -> bool:
    return (u & 0x7FFF) == 0x7C00


_RUN_ID = 0


def run_f16_unary(opcode: str, src0: int, name: Optional[str] = None) -> int:
    """Translate and run a pixel shader executing single unary float16 opcode."""
    global _RUN_ID
    _RUN_ID += 1
    prog = name or f"ps_{opcode}_{_RUN_ID}"
    lines = [
        f"v_mov_b32 v0, {src0 & MASK32:#x}",
        f"{opcode} v1, v0",
        "exp mrt0 v1, v1, v1, v1 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(prog, lines)
    words = mrt0_words(spv)
    return words[0] & MASK16


def run_f16_binary(opcode: str, src0: int, src1: int, name: Optional[str] = None) -> int:
    """Translate and run a pixel shader executing binary float16 opcode (e.g. v_ldexp_f16)."""
    global _RUN_ID
    _RUN_ID += 1
    prog = name or f"ps_{opcode}_{_RUN_ID}"
    lines = [
        f"v_mov_b32 v0, {src0 & MASK32:#x}",
        f"v_mov_b32 v1, {src1 & MASK32:#x}",
        f"{opcode} v2, v0, v1",
        "exp mrt0 v2, v2, v2, v2 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(prog, lines)
    words = mrt0_words(spv)
    return words[0] & MASK16


def run_f16_exec_masked(opcode: str, src0: int, active: bool, sentinel: int = 0xEEEE) -> int:
    """Verify execution masking via EXEC register: inactive lanes preserve destination."""
    global _RUN_ID
    _RUN_ID += 1
    prog = f"ps_mask_{opcode}_{_RUN_ID}"
    lines = [
        f"v_mov_b32 v0, {src0 & MASK32:#x}",
        f"v_mov_b32 v1, {sentinel & MASK32:#x}",
        *(["s_mov_b32 exec_lo, 0"] if not active else []),
        f"{opcode} v1, v0",
        *(["s_mov_b32 exec_lo, -1"] if not active else []),
        "exp mrt0 v1, v1, v1, v1 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(prog, lines)
    words = mrt0_words(spv)
    return words[0] & MASK16


def run_ldexp_exec_masked(src0: int, src1: int, active: bool, sentinel: int = 0xAAAA) -> int:
    """Verify ldexp execution masking via EXEC register."""
    global _RUN_ID
    _RUN_ID += 1
    prog = f"ps_mask_ldexp_{_RUN_ID}"
    lines = [
        f"v_mov_b32 v0, {src0 & MASK32:#x}",
        f"v_mov_b32 v1, {src1 & MASK32:#x}",
        f"v_mov_b32 v2, {sentinel & MASK32:#x}",
        *(["s_mov_b32 exec_lo, 0"] if not active else []),
        "v_ldexp_f16 v2, v0, v1",
        *(["s_mov_b32 exec_lo, -1"] if not active else []),
        "exp mrt0 v2, v2, v2, v2 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(prog, lines)
    words = mrt0_words(spv)
    return words[0] & MASK16


# -----------------------------------------------------------------------------
# Opcode 1: v_fract_f16
# -----------------------------------------------------------------------------

class VFractF16OracleTests(unittest.TestCase):
    def test_fract_basic(self):
        # fract(1.75) = 0.75 (0x3a00)
        res = run_f16_unary("v_fract_f16", f16_to_u16(1.75))
        self.assertEqual(res, 0x3A00)

    def test_fract_negative(self):
        # fract(-1.25) = -1.25 - (-2.0) = 0.75 (0x3a00)
        res = run_f16_unary("v_fract_f16", f16_to_u16(-1.25))
        self.assertEqual(res, 0x3A00)

    def test_fract_zeros(self):
        self.assertEqual(run_f16_unary("v_fract_f16", F16_POS_ZERO), F16_POS_ZERO)
        self.assertEqual(run_f16_unary("v_fract_f16", F16_NEG_ZERO), F16_POS_ZERO)


class VFractF16AstTests(unittest.TestCase):
    def test_fract_ast_construction(self):
        lines = [
            "v_mov_b32 v0, 0x3800",  # 0.5
            "v_fract_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_fract_f16_ast", lines)
        self.assertIn("Fract", spv)
        res = mrt0_words(spv)[0] & MASK16
        self.assertEqual(res, F16_POS_HALF)


class VFractF16ScalarInterpreterTests(unittest.TestCase):
    def test_fract_scalar_execution(self):
        # fract(2.5) = 0.5 (0x3800)
        res = run_f16_unary("v_fract_f16", f16_to_u16(2.5))
        self.assertEqual(res, F16_POS_HALF)


class VFractF16SimdDivergenceTests(unittest.TestCase):
    def test_fract_simd_exec_masking(self):
        # Active lane executes fract(2.25) = 0.25 (0x3400)
        val_active = run_f16_exec_masked("v_fract_f16", f16_to_u16(2.25), active=True)
        self.assertEqual(val_active, f16_to_u16(0.25))

        # Inactive lane preserves sentinel
        val_inactive = run_f16_exec_masked("v_fract_f16", f16_to_u16(2.25), active=False, sentinel=0xEEEE)
        self.assertEqual(val_inactive, 0xEEEE)


class VFractF16EdgeCasesTests(unittest.TestCase):
    def test_fract_clamp_below_one(self):
        # 0x3BFF (0.9995) remains 0x3BFF
        res = run_f16_unary("v_fract_f16", 0x3BFF)
        self.assertEqual(res, 0x3BFF)

    def test_fract_infinity_and_nan(self):
        # Hardware & translator rule: infinities return 0xFE00
        self.assertEqual(run_f16_unary("v_fract_f16", F16_POS_INF), 0xFE00)
        self.assertEqual(run_f16_unary("v_fract_f16", F16_NEG_INF), 0xFE00)
        # NaN returns NaN
        res_nan = run_f16_unary("v_fract_f16", F16_QNAN)
        self.assertTrue(is_nan_f16(res_nan))


class VFractF16TranslatorTests(unittest.TestCase):
    def test_fract_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3f00",  # 1.75 in f16
            "v_fract_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_fract_f16_exec", lines)
        val = mrt0_words(spv)[0] & MASK16
        self.assertEqual(val, 0x3A00)  # 0.75 in f16

    def test_fract_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_fract_f16", ["v_fract_f16"])


# -----------------------------------------------------------------------------
# Opcode 2: v_frexp_exp_i16_f16
# -----------------------------------------------------------------------------

class VFrexpExpI16F16OracleTests(unittest.TestCase):
    def test_frexp_exp_basic(self):
        # 1.0 (0x3c00): 1.0 = 0.5 * 2^1 -> exp is 1
        self.assertEqual(run_f16_unary("v_frexp_exp_i16_f16", F16_POS_ONE), 1)
        # 4.0: 4.0 = 0.5 * 2^3 -> exp is 3
        self.assertEqual(run_f16_unary("v_frexp_exp_i16_f16", f16_to_u16(4.0)), 3)
        # 0.25: 0.25 = 0.5 * 2^-1 -> exp is -1 (0xFFFF)
        self.assertEqual(run_f16_unary("v_frexp_exp_i16_f16", f16_to_u16(0.25)), (-1) & MASK16)


class VFrexpExpI16F16AstTests(unittest.TestCase):
    def test_frexp_exp_ast_construction(self):
        lines = [
            "v_mov_b32 v0, 0x4000",  # 2.0
            "v_frexp_exp_i16_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_exp_f16_ast", lines)
        self.assertIsInstance(spv, str)
        res = mrt0_words(spv)[0] & MASK16
        self.assertEqual(res, 2)


class VFrexpExpI16F16ScalarInterpreterTests(unittest.TestCase):
    def test_frexp_exp_scalar_execution(self):
        # 2.0 = 0.5 * 2^2 -> exp is 2
        res = run_f16_unary("v_frexp_exp_i16_f16", f16_to_u16(2.0))
        self.assertEqual(res, 2)


class VFrexpExpI16F16SimdDivergenceTests(unittest.TestCase):
    def test_frexp_exp_simd_exec_masking(self):
        # Active lane computes 8.0 = 0.5 * 2^4 -> exp is 4
        res_active = run_f16_exec_masked("v_frexp_exp_i16_f16", f16_to_u16(8.0), active=True)
        self.assertEqual(res_active, 4)

        # Inactive lane preserves sentinel
        res_inactive = run_f16_exec_masked("v_frexp_exp_i16_f16", f16_to_u16(8.0), active=False, sentinel=0x9999)
        self.assertEqual(res_inactive, 0x9999)


class VFrexpExpI16F16EdgeCasesTests(unittest.TestCase):
    def test_frexp_exp_zeros_and_specials(self):
        self.assertEqual(run_f16_unary("v_frexp_exp_i16_f16", F16_POS_ZERO), 0)
        self.assertEqual(run_f16_unary("v_frexp_exp_i16_f16", F16_NEG_ZERO), 0)
        self.assertEqual(run_f16_unary("v_frexp_exp_i16_f16", F16_POS_INF), 0)
        self.assertEqual(run_f16_unary("v_frexp_exp_i16_f16", F16_QNAN), 0)


class VFrexpExpI16F16TranslatorTests(unittest.TestCase):
    def test_frexp_exp_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3c00",  # 1.0 in f16
            "v_frexp_exp_i16_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_exp_f16_exec", lines)
        val = mrt0_words(spv)[0] & MASK16
        self.assertEqual(val, 1)


# -----------------------------------------------------------------------------
# Opcode 3: v_frexp_mant_f16
# -----------------------------------------------------------------------------

class VFrexpMantF16OracleTests(unittest.TestCase):
    def test_frexp_mant_basic(self):
        # 1.0 (0x3c00): mantissa is 0.5 (0x3800)
        self.assertEqual(run_f16_unary("v_frexp_mant_f16", F16_POS_ONE), F16_POS_HALF)
        # -1.0 (0xbc00): mantissa is -0.5 (0xb800)
        self.assertEqual(run_f16_unary("v_frexp_mant_f16", F16_NEG_ONE), F16_NEG_HALF)
        # 3.0: 3.0 = 0.75 * 2^2 -> mantissa is 0.75 (0x3a00)
        self.assertEqual(run_f16_unary("v_frexp_mant_f16", f16_to_u16(3.0)), 0x3A00)


class VFrexpMantF16AstTests(unittest.TestCase):
    def test_frexp_mant_ast_construction(self):
        lines = [
            "v_mov_b32 v0, 0x4200",  # 3.0
            "v_frexp_mant_f16 v2, v0",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_mant_f16_ast", lines)
        self.assertIsInstance(spv, str)
        res = mrt0_words(spv)[0] & MASK16
        self.assertEqual(res, 0x3A00)


class VFrexpMantF16ScalarInterpreterTests(unittest.TestCase):
    def test_frexp_mant_scalar_execution(self):
        # 6.0: 6.0 = 0.75 * 2^3 -> mantissa is 0.75 (0x3a00)
        res = run_f16_unary("v_frexp_mant_f16", f16_to_u16(6.0))
        self.assertEqual(res, 0x3A00)


class VFrexpMantF16SimdDivergenceTests(unittest.TestCase):
    def test_frexp_mant_simd_exec_masking(self):
        # Active lane computes mantissa of 1.0 -> 0.5 (0x3800)
        res_active = run_f16_exec_masked("v_frexp_mant_f16", F16_POS_ONE, active=True)
        self.assertEqual(res_active, F16_POS_HALF)

        # Inactive lane preserves sentinel
        res_inactive = run_f16_exec_masked("v_frexp_mant_f16", F16_POS_ONE, active=False, sentinel=0x7777)
        self.assertEqual(res_inactive, 0x7777)


class VFrexpMantF16EdgeCasesTests(unittest.TestCase):
    def test_frexp_mant_specials_passthrough(self):
        self.assertEqual(run_f16_unary("v_frexp_mant_f16", F16_POS_ZERO), F16_POS_ZERO)
        self.assertEqual(run_f16_unary("v_frexp_mant_f16", F16_NEG_ZERO), F16_NEG_ZERO)
        self.assertEqual(run_f16_unary("v_frexp_mant_f16", F16_POS_INF), F16_POS_INF)
        self.assertEqual(run_f16_unary("v_frexp_mant_f16", F16_QNAN), F16_QNAN)


class VFrexpMantF16TranslatorTests(unittest.TestCase):
    def test_frexp_mant_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3c00",  # 1.0 in f16
            "v_frexp_mant_f16 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_mant_f16_exec", lines)
        val = mrt0_words(spv)[0] & MASK16
        self.assertEqual(val, 0x3800)  # 0.5 in f16


# -----------------------------------------------------------------------------
# Opcode 4: v_ldexp_f16
# -----------------------------------------------------------------------------

class VLdexpF16OracleTests(unittest.TestCase):
    def test_ldexp_basic(self):
        # 1.0 * 2^3 = 8.0 (0x4800)
        self.assertEqual(run_f16_binary("v_ldexp_f16", F16_POS_ONE, 3), 0x4800)
        # 1.0 * 2^-2 = 0.25 (0x3400)
        self.assertEqual(run_f16_binary("v_ldexp_f16", F16_POS_ONE, -2 & MASK16), 0x3400)


class VLdexpF16AstTests(unittest.TestCase):
    def test_ldexp_ast_construction(self):
        lines = [
            "v_mov_b32 v0, 0x3c00",  # 1.0
            "v_mov_b32 v1, 2",       # scale 2 -> 4.0 (0x4400)
            "v_ldexp_f16 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_ldexp_f16_ast", lines)
        self.assertIsInstance(spv, str)
        res = mrt0_words(spv)[0] & MASK16
        self.assertEqual(res, 0x4400)


class VLdexpF16ScalarInterpreterTests(unittest.TestCase):
    def test_ldexp_scalar_execution(self):
        # 1.5 (0x3e00) * 2^1 = 3.0 (0x4200)
        res = run_f16_binary("v_ldexp_f16", f16_to_u16(1.5), 1)
        self.assertEqual(res, f16_to_u16(3.0))


class VLdexpF16SimdDivergenceTests(unittest.TestCase):
    def test_ldexp_simd_exec_masking(self):
        # Active lane computes 1.0 * 2^2 = 4.0 (0x4400)
        res_active = run_ldexp_exec_masked(F16_POS_ONE, 2, active=True)
        self.assertEqual(res_active, 0x4400)

        # Inactive lane preserves sentinel
        res_inactive = run_ldexp_exec_masked(F16_POS_ONE, 2, active=False, sentinel=0xAAAA)
        self.assertEqual(res_inactive, 0xAAAA)


class VLdexpF16EdgeCasesTests(unittest.TestCase):
    def test_ldexp_overflow(self):
        # 1.0 * 2^50 -> overflow to +inf
        self.assertEqual(run_f16_binary("v_ldexp_f16", F16_POS_ONE, 50), F16_POS_INF)
        # -1.0 * 2^50 -> overflow to -inf
        self.assertEqual(run_f16_binary("v_ldexp_f16", F16_NEG_ONE, 50), F16_NEG_INF)

    def test_ldexp_underflow(self):
        # 1.0 * 2^-50 -> underflow to zero
        self.assertEqual(run_f16_binary("v_ldexp_f16", F16_POS_ONE, -50 & MASK16), F16_POS_ZERO)


class VLdexpF16TranslatorTests(unittest.TestCase):
    def test_ldexp_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3c00",  # 1.0 in f16
            "v_mov_b32 v1, 3",
            "v_ldexp_f16 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_ldexp_f16_exec", lines)
        val = mrt0_words(spv)[0] & MASK16
        self.assertEqual(val, 0x4800)  # 8.0 in f16


if __name__ == "__main__":
    unittest.main()
