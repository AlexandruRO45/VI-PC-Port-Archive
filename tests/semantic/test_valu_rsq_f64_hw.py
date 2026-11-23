"""Translator tests for v_rsq_f64 (IEEE-754 binary64 reciprocal square root, VOP3 modifiers).

  v_rsq_f64  D = 1.0 / sqrt(S0)

Each case translates a hand-written pixel-shader listing with tools/rdna2spirv.py
and verifies the resulting SPIR-V and execution via spirv_interp / support.mrt0_words.
Standard library only.
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

import spirv_interp  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
POS_INF = 0x7FF0000000000000
NEG_INF = 0xFFF0000000000000


def f64_to_bits(x: float) -> int:
    """Pack a Python float into its IEEE-754 binary64 64-bit unsigned integer word."""
    try:
        return struct.unpack("<Q", struct.pack("<d", x))[0]
    except OverflowError:
        return POS_INF if x > 0 else NEG_INF


def bits_to_f64(bits: int) -> float:
    """Unpack a 64-bit unsigned integer word as an IEEE-754 binary64 Python float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def run_rsq_f64(
    a_bits: int,
    *,
    setup: list[str] | None = None,
    operands: str = "v[0:1]",
) -> tuple[int, str]:
    """Translate and run `v_rsq_f64 v[2:3], <operands>` via spirv_interp; return (u64_bits, spv)."""
    if setup is None:
        setup = [
            f"v_mov_b32 v0, {a_bits & MASK32:#010x}",
            f"v_mov_b32 v1, {(a_bits >> 32) & MASK32:#010x}",
        ]
    lines = list(setup) + [
        f"v_rsq_f64 v[2:3], {operands}",
        "exp mrt0 v2, v3, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_rsq_f64", lines)
    res = spirv_interp.run(spv)
    got = (res.vgpr[2] & MASK32) | ((res.vgpr[3] & MASK32) << 32)
    return got, spv


class RsqF64PrecisionAndPowersTests(unittest.TestCase):
    def test_exact_powers_of_four_and_two(self):
        cases = [
            (1.0, 1.0),
            (4.0, 0.5),
            (16.0, 0.25),
            (64.0, 0.125),
            (256.0, 0.0625),
            (0.25, 2.0),
            (0.0625, 4.0),
            (0.015625, 8.0),
        ]
        for val, expected in cases:
            with self.subTest(val=val):
                got_bits, _ = run_rsq_f64(f64_to_bits(val))
                self.assertEqual(got_bits, f64_to_bits(expected))

    def test_precision_beyond_f32_range(self):
        cases = [
            (float(2**50), float(2**-25)),
            (float(2**-50), float(2**25)),
            (float(2**100), float(2**-50)),
            (float(2**-100), float(2**50)),
            (float(2**200), float(2**-100)),
            (float(2**-200), float(2**100)),
            (float(2**500), float(2**-250)),
            (float(2**-500), float(2**250)),
        ]
        for val, expected in cases:
            with self.subTest(val=val):
                got_bits, _ = run_rsq_f64(f64_to_bits(val))
                self.assertEqual(got_bits, f64_to_bits(expected))

    def test_fractional_non_powers_precision(self):
        cases = [9.0, 100.0, 144.0, 0.01, 12345.6789]
        for val in cases:
            with self.subTest(val=val):
                got_bits, _ = run_rsq_f64(f64_to_bits(val))
                expected = 1.0 / math.sqrt(val)
                self.assertAlmostEqual(bits_to_f64(got_bits), expected, places=14)


class RsqF64ModifiersTests(unittest.TestCase):
    def test_abs_modifier_with_negative_inputs(self):
        cases = [-1.0, -4.0, -16.0, -64.0, -0.25, -0.0625, -float(2**50)]
        for val in cases:
            with self.subTest(val=val):
                got_bits, _ = run_rsq_f64(f64_to_bits(val), operands="|v[0:1]|")
                expected = 1.0 / math.sqrt(abs(val))
                self.assertEqual(got_bits, f64_to_bits(expected))

    def test_output_modifiers(self):
        cases = [
            (4.0, "v[0:1] mul:2", 1.0),
            (4.0, "v[0:1] mul:4", 2.0),
            (4.0, "v[0:1] div:2", 0.25),
            (16.0, "v[0:1] mul:2", 0.5),
            (16.0, "v[0:1] mul:4", 1.0),
            (16.0, "v[0:1] div:2", 0.125),
        ]
        for val, ops, exp in cases:
            with self.subTest(ops=ops):
                got_bits, _ = run_rsq_f64(f64_to_bits(val), operands=ops)
                self.assertEqual(got_bits, f64_to_bits(exp))

    def test_clamp_modifier(self):
        cases = [
            (0.25, "v[0:1] clamp", 1.0),
            (0.0625, "v[0:1] clamp", 1.0),
            (4.0, "v[0:1] clamp", 0.5),
            (16.0, "v[0:1] clamp", 0.25),
            (0.25, "v[0:1] div:2 clamp", 1.0),
            (4.0, "v[0:1] mul:2 clamp", 1.0),
            (4.0, "v[0:1] div:2 clamp", 0.25),
        ]
        for val, ops, exp in cases:
            with self.subTest(ops=ops):
                got_bits, _ = run_rsq_f64(f64_to_bits(val), operands=ops)
                self.assertEqual(got_bits, f64_to_bits(exp))


class RsqF64OperandsAndAliasingTests(unittest.TestCase):
    def test_sgpr_pair_source(self):
        val = 16.0
        val_bits = f64_to_bits(val)
        setup = [
            f"s_mov_b32 s0, {val_bits & MASK32:#010x}",
            f"s_mov_b32 s1, {(val_bits >> 32) & MASK32:#010x}",
        ]
        got_bits, _ = run_rsq_f64(0, setup=setup, operands="s[0:1]")
        self.assertEqual(got_bits, f64_to_bits(0.25))

    def test_inline_float_constants(self):
        cases = [
            ("1.0", 1.0),
            ("4.0", 0.5),
            ("0.5", 1.0 / math.sqrt(0.5)),
        ]
        for const_str, exp in cases:
            with self.subTest(const=const_str):
                lines = [
                    f"v_rsq_f64 v[2:3], {const_str}",
                    "exp mrt0 v2, v3, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_rsq_c_{const_str.replace('.', '_')}", lines)
                res = spirv_interp.run(spv)
                got = (res.vgpr[2] & MASK32) | ((res.vgpr[3] & MASK32) << 32)
                self.assertAlmostEqual(bits_to_f64(got), exp, places=14)

    def test_register_aliasing_in_place(self):
        val = 64.0
        val_bits = f64_to_bits(val)
        lines = [
            f"v_mov_b32 v0, {val_bits & MASK32:#010x}",
            f"v_mov_b32 v1, {(val_bits >> 32) & MASK32:#010x}",
            "v_rsq_f64 v[0:1], v[0:1]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_rsq_alias", lines)
        res = spirv_interp.run(spv)
        got = (res.vgpr[0] & MASK32) | ((res.vgpr[1] & MASK32) << 32)
        self.assertEqual(got, f64_to_bits(0.125))

    def test_mrt0_words_pipeline_execution(self):
        val = 4.0
        val_bits = f64_to_bits(val)
        lines = [
            f"v_mov_b32 v0, {val_bits & MASK32:#010x}",
            f"v_mov_b32 v1, {(val_bits >> 32) & MASK32:#010x}",
            "v_rsq_f64 v[2:3], v[0:1]",
            "exp mrt0 v2, v3, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_rsq_mrt0", lines)
        words = mrt0_words(spv)
        lo = f64_to_bits(0.5) & MASK32
        hi = (f64_to_bits(0.5) >> 32) & MASK32
        self.assertEqual(words, [lo, hi, lo, hi])


class RsqF64InfinitiesAndSpirvTests(unittest.TestCase):
    def test_positive_infinity(self):
        got_bits, _ = run_rsq_f64(POS_INF)
        self.assertEqual(got_bits, 0x0000000000000000)

    def test_spirv_instruction_emission(self):
        _, spv = run_rsq_f64(f64_to_bits(4.0))
        self.assertIn("OpCapability Float64", spv)
        self.assertIn("OpTypeFloat 64", spv)
        self.assertIn("InverseSqrt", spv)
        self.assertIn("OpBitcast %v2uint_d", spv)


if __name__ == "__main__":
    unittest.main()
