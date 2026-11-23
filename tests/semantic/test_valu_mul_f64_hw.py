"""Hardware and translator regression tests for RDNA2 double-precision v_mul_f64.

Tests the translation of v_mul_f64 instructions through tools/rdna2spirv.py and validates
the bit-exact execution of emitted SPIR-V via support.mrt0_words (spirv_mini_interp).

Standard library only; zero external dependencies, zero game data.
    python -I tests/semantic/test_valu_mul_f64_hw.py
    python -I tests/test_semantic_translator_use.py
"""
from __future__ import annotations

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def f64_to_u64(val: float) -> int:
    """Pack Python float (IEEE 754 binary64) into unsigned 64-bit integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def u64_to_f64(bits: int) -> float:
    """Unpack unsigned 64-bit integer into Python float (IEEE 754 binary64)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def u64_to_words(val: int) -> tuple[int, int]:
    """Split 64-bit word into (low 32-bit dword, high 32-bit dword)."""
    return val & MASK32, (val >> 32) & MASK32


def words_to_u64(lo: int, hi: int) -> int:
    """Combine (low 32-bit dword, high 32-bit dword) into 64-bit word."""
    return (lo & MASK32) | ((hi & MASK32) << 32)


def run_mul_f64(
    src0_bits: int,
    src1_bits: int,
    *,
    instruction: str = "v_mul_f64 v[0:1], v[2:3], v[4:5]",
    test_name: str = "ps_mul_f64",
) -> int:
    """Translate and run a pixel shader computing v_mul_f64, returning the 64-bit result bits."""
    s0_lo, s0_hi = u64_to_words(src0_bits)
    s1_lo, s1_hi = u64_to_words(src1_bits)

    lines = [
        f"v_mov_b32 v2, {s0_lo:#x}",
        f"v_mov_b32 v3, {s0_hi:#x}",
        f"v_mov_b32 v4, {s1_lo:#x}",
        f"v_mov_b32 v5, {s1_hi:#x}",
        instruction,
        "exp mrt0 v0, v1, v0, v1 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(test_name, lines)
    words = support.mrt0_words(spv)
    return words_to_u64(words[0], words[1])


class TestValuMulF64Translator(unittest.TestCase):
    """Verifies that rdna2spirv emits the required SPIR-V structure for v_mul_f64."""

    def test_translator_emits_opfmul_double(self) -> None:
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x3ff00000",
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0x40000000",
            "v_mul_f64 v[0:1], v[2:3], v[4:5]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mul_spirv_check", lines)
        self.assertIn("OpFMul %double", spv)
        self.assertIn("OpTypeFloat 64", spv)
        self.assertIn("%v2uint_d = OpTypeVector %uint 2", spv)

    def test_translator_emits_source_modifiers(self) -> None:
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x3ff00000",
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0x40000000",
            "v_mul_f64 v[0:1], -v[2:3], |v[4:5]|",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mul_mods_check", lines)
        self.assertIn("OpFNegate %double", spv)
        self.assertIn("OpExtInst %double %glsl FAbs", spv)
        self.assertIn("OpFMul %double", spv)

    def test_translator_emits_clamp(self) -> None:
        lines = [
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0x3ff00000",
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0x40000000",
            "v_mul_f64 v[0:1], v[2:3], v[4:5] clamp",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mul_clamp_check", lines)
        self.assertIn("OpExtInst %double %glsl FClamp", spv)


class TestValuMulF64Execution(unittest.TestCase):
    """Verifies that translated v_mul_f64 programs execute with bit-exact results."""

    def test_execution_basic_multiplication(self) -> None:
        # 3.0 * 4.0 = 12.0
        res = run_mul_f64(
            f64_to_u64(3.0),
            f64_to_u64(4.0),
            test_name="ps_mul_basic_1",
        )
        self.assertEqual(u64_to_f64(res), 12.0)

        # 1.5 * (-2.0) = -3.0
        res = run_mul_f64(
            f64_to_u64(1.5),
            f64_to_u64(-2.0),
            test_name="ps_mul_basic_2",
        )
        self.assertEqual(u64_to_f64(res), -3.0)

        # 0.25 * 0.5 = 0.125
        res = run_mul_f64(
            f64_to_u64(0.25),
            f64_to_u64(0.5),
            test_name="ps_mul_basic_3",
        )
        self.assertEqual(u64_to_f64(res), 0.125)

    def test_execution_source_modifiers(self) -> None:
        # -3.0 * 4.0 = -12.0
        res = run_mul_f64(
            f64_to_u64(3.0),
            f64_to_u64(4.0),
            instruction="v_mul_f64 v[0:1], -v[2:3], v[4:5]",
            test_name="ps_mul_neg",
        )
        self.assertEqual(u64_to_f64(res), -12.0)

        # |-5.0| * 2.0 = 10.0
        res = run_mul_f64(
            f64_to_u64(-5.0),
            f64_to_u64(2.0),
            instruction="v_mul_f64 v[0:1], |v[2:3]|, v[4:5]",
            test_name="ps_mul_abs",
        )
        self.assertEqual(u64_to_f64(res), 10.0)

    def test_execution_clamp_modifier(self) -> None:
        # 2.0 * 3.0 = 6.0 -> clamped to 1.0
        res = run_mul_f64(
            f64_to_u64(2.0),
            f64_to_u64(3.0),
            instruction="v_mul_f64 v[0:1], v[2:3], v[4:5] clamp",
            test_name="ps_mul_clamp_high",
        )
        self.assertEqual(u64_to_f64(res), 1.0)

        # -2.0 * 3.0 = -6.0 -> clamped to 0.0
        res = run_mul_f64(
            f64_to_u64(-2.0),
            f64_to_u64(3.0),
            instruction="v_mul_f64 v[0:1], v[2:3], v[4:5] clamp",
            test_name="ps_mul_clamp_low",
        )
        self.assertEqual(u64_to_f64(res), 0.0)

    def test_execution_signed_zeros(self) -> None:
        # (+0.0) * (+0.0) = +0.0
        res = run_mul_f64(
            f64_to_u64(0.0),
            f64_to_u64(0.0),
            test_name="ps_mul_sz_pos_pos",
        )
        self.assertEqual(res, 0)

        # (-0.0) * (+0.0) = -0.0
        res = run_mul_f64(
            f64_to_u64(-0.0),
            f64_to_u64(0.0),
            test_name="ps_mul_sz_neg_pos",
        )
        self.assertEqual(res, 0x8000000000000000)

        # (-0.0) * (-0.0) = +0.0
        res = run_mul_f64(
            f64_to_u64(-0.0),
            f64_to_u64(-0.0),
            test_name="ps_mul_sz_neg_neg",
        )
        self.assertEqual(res, 0)

    def test_execution_infinity_arithmetic(self) -> None:
        # inf * 2.0 = inf
        res = run_mul_f64(
            f64_to_u64(math.inf),
            f64_to_u64(2.0),
            test_name="ps_mul_inf_pos",
        )
        self.assertTrue(math.isinf(u64_to_f64(res)))
        self.assertGreater(u64_to_f64(res), 0)

        # -inf * 2.0 = -inf
        res = run_mul_f64(
            f64_to_u64(-math.inf),
            f64_to_u64(2.0),
            test_name="ps_mul_inf_neg",
        )
        self.assertTrue(math.isinf(u64_to_f64(res)))
        self.assertLess(u64_to_f64(res), 0)

        # inf * 0.0 = nan
        res = run_mul_f64(
            f64_to_u64(math.inf),
            f64_to_u64(0.0),
            test_name="ps_mul_inf_zero",
        )
        self.assertTrue(math.isnan(u64_to_f64(res)))

    def test_execution_nan_propagation(self) -> None:
        # nan * 5.0 = nan
        res = run_mul_f64(
            f64_to_u64(math.nan),
            f64_to_u64(5.0),
            test_name="ps_mul_nan",
        )
        self.assertTrue(math.isnan(u64_to_f64(res)))

    def test_rejection_invalid_operands(self) -> None:
        lines = [
            "v_mul_f64 v0, v[2:3], v[4:5]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_mul_bad_dst", lines)


if __name__ == "__main__":
    unittest.main()
