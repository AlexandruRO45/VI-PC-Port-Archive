"""Translator tests for v_frexp_exp_i32_f64 (IEEE-754 binary64 frexp exponent component).

  v_frexp_exp_i32_f64  D.i = exp(S0.d)

Computes the 32-bit signed integer exponent component of binary64 double-precision float:
  x = mant * 2^exp, where mant in [0.5, 1.0)
  Normal: exp = biased_exp - 1022 (e.g., 1.0 -> 1, 2.0 -> 2, 0.5 -> 0, 0.25 -> -1)
  Subnormals: exp in [-1073, -1022] (e.g., 2^-1074 -> -1073)
  Zeros (+-0), Infinities (+-inf), NaNs -> exp = 0

Each case translates a hand-written pixel-shader listing with tools/rdna2spirv.py
and verifies the resulting SPIR-V and execution via support.mrt0_words.
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

import rdna2spirv  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
POS_INF = 0x7FF0000000000000
NEG_INF = 0xFFF0000000000000
QNAN = 0x7FF8000000000000
SNAN = 0x7FF0000000000001


def f64_to_bits(x: float) -> int:
    try:
        return struct.unpack("<Q", struct.pack("<d", x))[0]
    except OverflowError:
        return POS_INF if x > 0 else NEG_INF


def s32(u: int) -> int:
    u &= MASK32
    return u if u < 0x80000000 else u - 0x100000000


def frexp_exp_f64_oracle(bits: int) -> int:
    hi = (bits >> 32) & MASK32
    lo = bits & MASK32
    e = (hi >> 20) & 0x7FF
    f_hi = hi & 0x000FFFFF
    if 1 <= e <= 2046:
        return e - 1022
    if e == 0:
        frac = (f_hi << 32) | lo
        if frac == 0:
            return 0
        msb = frac.bit_length() - 1
        return msb - 1073
    return 0


def run_frexp_exp(bits: int, operands: str = "v[0:1]", setup: list[str] | None = None) -> int:
    if setup is None:
        setup = [
            f"v_mov_b32 v0, {bits & MASK32:#010x}",
            f"v_mov_b32 v1, {(bits >> 32) & MASK32:#010x}",
        ]
    lines = list(setup) + [
        f"v_frexp_exp_i32_f64 v2, {operands}",
        "exp mrt0 v2, v2, v2, v2 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_frexp_exp", lines)
    words = mrt0_words(spv)
    return s32(words[0])


class TestFrexpExpF64OracleAndNormals(unittest.TestCase):
    def test_oracle_matches_math_frexp(self):
        cases = [1.0, 2.0, 4.0, 8.0, 16.0, 0.5, 0.25, 0.125, 0.0625, 3.0, 5.0, 100.0, 12345.6789]
        for val in cases:
            with self.subTest(val=val):
                expected = math.frexp(val)[1]
                bits = f64_to_bits(val)
                self.assertEqual(frexp_exp_f64_oracle(bits), expected)

    def test_translated_powers_of_two(self):
        cases = [
            (1.0, 1),
            (2.0, 2),
            (4.0, 3),
            (8.0, 4),
            (16.0, 5),
            (0.5, 0),
            (0.25, -1),
            (0.125, -2),
            (0.0625, -3),
        ]
        for val, expected in cases:
            with self.subTest(val=val):
                got = run_frexp_exp(f64_to_bits(val))
                self.assertEqual(got, expected)

    def test_translated_non_powers_of_two(self):
        cases = [
            (3.0, 2),
            (5.0, 3),
            (7.5, 3),
            (9.0, 4),
            (100.0, 7),
            (1024.0, 11),
        ]
        for val, expected in cases:
            with self.subTest(val=val):
                got = run_frexp_exp(f64_to_bits(val))
                self.assertEqual(got, expected)


class TestFrexpExpF64AstAndTypes(unittest.TestCase):
    def test_spirv_emission_structure(self):
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x3ff00000",
            "v_frexp_exp_i32_f64 v2, v[0:1]",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_ast", lines)
        self.assertIn("OpBitFieldUExtract", spv)
        self.assertIn("OpISub", spv)
        self.assertIn("OpSelect", spv)

    def test_single_vgpr_destination_type(self):
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x40100000",
            "v_frexp_exp_i32_f64 v5, v[0:1]",
            "exp mrt0 v5, v5, v5, v5 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_frexp_dst", lines)
        words = mrt0_words(spv)
        self.assertEqual(s32(words[0]), 3)


if __name__ == "__main__":
    unittest.main()
