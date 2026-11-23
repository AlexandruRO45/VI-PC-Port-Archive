"""Semantic tests for RDNA2 int -> float conversion: v_cvt_f32_u32 / v_cvt_f32_i32.

v_cvt_f32_u32: D = float32(S0) where S0 is a 32-bit unsigned integer.
v_cvt_f32_i32: D = float32(S0) where S0 is a 32-bit signed integer.
Both round to nearest even when the integer is not exactly representable in binary32.

Each case translates a hand-written pixel-shader listing with tools/rdna2spirv.py
and executes the resulting SPIR-V via tests/semantic/spirv_mini_interp.py. The expected
word is a plain-Python reference (Python's correctly-rounded float() conversion).

Standard library only: no GPU, no LLVM, no game data.
"""
import math
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines  # noqa: E402  (support puts tools/ on sys.path)

MASK32 = 0xFFFFFFFF
U31 = 1 << 31
U32 = 1 << 32


def f32_bits(x):
    """Pack a Python float into its IEEE-754 binary32 32-bit unsigned word."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def s32(bits):
    """Interpret a 32-bit word as a signed int32."""
    return bits - U32 if bits >= U31 else bits


def ref_cvt_f32_u32(bits):
    """Plain-Python reference for v_cvt_f32_u32 given an unsigned int32 operand."""
    return f32_bits(float(bits))


def ref_cvt_f32_i32(bits):
    """Plain-Python reference for v_cvt_f32_i32 given a signed int32 operand."""
    return f32_bits(float(s32(bits)))


def run_cvt(op, src_bits):
    """Translate and run `op v0, v1` on a pixel shader; return the uniform mrt0 word."""
    setup = [f"v_mov_b32 v1, {src_bits & MASK32:#010x}"]
    lines = list(setup) + [
        f"{op} v0, v1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_{op}", lines)
    words = mrt0_words(spv)
    if len(set(words)) != 1:
        raise AssertionError(f"{op}: exported mrt0 words differ: {[hex(w) for w in words]}")
    return words[0]


class CvtF32U32(unittest.TestCase):
    HAND = [
        0, 1, 2, 3, 15, 16, 127, 128, 255, 256,
        16777214, 16777215, 16777216, 16777217, 16777218, 16777219,
        0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, U32 - 1,
    ]

    def test_hand(self):
        for bits in self.HAND:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(run_cvt("v_cvt_f32_u32", bits), ref_cvt_f32_u32(bits))

    def test_random(self):
        rng = random.Random(0xC0DE)
        for _ in range(200):
            bits = rng.getrandbits(32)
            with self.subTest(bits=hex(bits)):
                self.assertEqual(run_cvt("v_cvt_f32_u32", bits), ref_cvt_f32_u32(bits))


class CvtF32I32(unittest.TestCase):
    HAND = [
        0, 1, 2, -1, -2, 15, -15, 127, -128, 128, 255, -256,
        16777214, 16777215, 16777216, -16777216,
        16777217, -16777217, 16777218, -16777218,
        0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF,
    ]

    def test_hand(self):
        for bits in self.HAND:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(run_cvt("v_cvt_f32_i32", bits), ref_cvt_f32_i32(bits))

    def test_random(self):
        rng = random.Random(0xF00D)
        for _ in range(200):
            bits = rng.getrandbits(32)
            with self.subTest(bits=hex(bits)):
                self.assertEqual(run_cvt("v_cvt_f32_i32", bits), ref_cvt_f32_i32(bits))


if __name__ == "__main__":
    unittest.main()
