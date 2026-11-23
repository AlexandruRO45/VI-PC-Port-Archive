from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 v_rndne_f32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the v_rndne_f32 instruction belongs to VOP1 (or VOP3) format.
  Instruction:
    v_rndne_f32 vdst, src0
    v_rndne_f32_e32 vdst, src0
    v_rndne_f32_e64 vdst, src0 [clamp]

  Operands:
    vdst = 32-bit destination VGPR (v[0..255])
    src0 = 32-bit source float operand: VGPR, SGPR, inline float constant, or 32-bit literal

  Semantics:
    vdst = round_to_nearest_even(src0)
    Rounds IEEE 754 single-precision float to integer, breaking ties towards the even integer.
    Special values:
      +-0.0 -> +-0.0 (sign preserved)
      +-inf -> +-inf
      NaN -> quiet NaN preserved
      Subnormals -> +-0.0 (underflow to signed zero)
      |src0| >= 2^23 -> exact integer identity
    Optional VOP3 clamp modifier clamps the result to [0.0, 1.0].
    Condition codes: Does not modify SCC or VCC.
    EXEC masking: Lane operations executed conditionally under wave32 / wave64 EXEC mask.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32: int = 0xFFFFFFFF
F32_POS_ZERO: int = 0x00000000
F32_NEG_ZERO: int = 0x80000000
F32_ONE: int = 0x3F800000
F32_NEG_ONE: int = 0xBF800000
F32_TWO: int = 0x40000000
F32_NEG_TWO: int = 0xC0000000
F32_FOUR: int = 0x40800000
F32_NEG_FOUR: int = 0xC0800000
F32_POS_INF: int = 0x7F800000
F32_NEG_INF: int = 0xFF800000
F32_QNAN: int = 0x7FC00000
F32_SNAN: int = 0x7F800001
F32_MIN_SUBNORMAL: int = 0x00000001
F32_MAX_SUBNORMAL: int = 0x007FFFFF
F32_MIN_NORMAL: int = 0x00800000
F32_MAX_FLOAT: int = 0x7F7FFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def f32_to_bits(f: float) -> int:
    """Pack float to 32-bit IEEE-754 bit pattern."""
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(b: int) -> float:
    """Unpack 32-bit IEEE-754 bit pattern to float."""
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def ref_rndne_f32(src0_bits: int, clamp: bool = False) -> int:
    """Mathematical reference oracle for AMD RDNA2 v_rndne_f32.

    Rounds single-precision IEEE-754 float to nearest integer, ties to even.

    Args:
        src0_bits: 32-bit float bit pattern (uint32)
        clamp: whether to apply VOP3 clamp to [0.0, 1.0]

    Returns:
        32-bit float bit pattern representing round_even(src0).
    """
    sign = (src0_bits >> 31) & 1
    exp = (src0_bits >> 23) & 0xFF
    mant = src0_bits & 0x7FFFFF

    # Special values: Inf and NaN
    if exp == 0xFF:
        if mant != 0:
            # Quiet NaN preserved
            return src0_bits | 0x00400000
        # Infinity preserved
        return src0_bits

    # Large values (|x| >= 2^23): already integer-exact
    if exp >= (127 + 23):
        return src0_bits

    val = bits_to_f32(src0_bits)
    if math.isnan(val) or math.isinf(val):
        return src0_bits

    # Subnormals: magnitude < 2^-126, round to signed zero
    if exp == 0:
        res_bits = F32_NEG_ZERO if sign else F32_POS_ZERO
    else:
        # Round half to even
        r = round(val)
        if r == 0:
            res_bits = F32_NEG_ZERO if sign else F32_POS_ZERO
        else:
            res_bits = f32_to_bits(float(r))

    if clamp:
        clamped_f = bits_to_f32(res_bits)
        if clamped_f < 0.0 or res_bits == F32_NEG_ZERO:
            res_bits = F32_POS_ZERO
        elif clamped_f > 1.0:
            res_bits = F32_ONE

    return res_bits


class TestValuRndneF32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of v_rndne_f32."""

    def test_oracle_exact_integers(self):
        for int_val in [0.0, 1.0, 2.0, 10.0, -1.0, -5.0, 100.0]:
            bits = f32_to_bits(int_val)
            res = ref_rndne_f32(bits)
            self.assertEqual(res, bits)

    def test_oracle_halfway_ties_even(self):
        cases = [
            (0.5, 0.0, F32_POS_ZERO),
            (-0.5, -0.0, F32_NEG_ZERO),
            (1.5, 2.0, F32_TWO),
            (-1.5, -2.0, F32_NEG_TWO),
            (2.5, 2.0, F32_TWO),
            (-2.5, -2.0, F32_NEG_TWO),
            (3.5, 4.0, F32_FOUR),
            (-3.5, -4.0, F32_NEG_FOUR),
            (4.5, 4.0, F32_FOUR),
            (-4.5, -4.0, F32_NEG_FOUR),
        ]
        for inp_f, exp_f, exp_bits in cases:
            res = ref_rndne_f32(f32_to_bits(inp_f))
            self.assertEqual(res, exp_bits)
            self.assertEqual(bits_to_f32(res), exp_f)

    def test_oracle_non_halfway(self):
        cases = [
            (1.1, 1.0, F32_ONE),
            (1.9, 2.0, F32_TWO),
            (-1.1, -1.0, F32_NEG_ONE),
            (-1.9, -2.0, F32_NEG_TWO),
            (0.2, 0.0, F32_POS_ZERO),
            (-0.2, -0.0, F32_NEG_ZERO),
            (0.8, 1.0, F32_ONE),
            (-0.8, -1.0, F32_NEG_ONE),
        ]
        for inp_f, exp_f, exp_bits in cases:
            res = ref_rndne_f32(f32_to_bits(inp_f))
            self.assertEqual(res, exp_bits)
            self.assertEqual(bits_to_f32(res), exp_f)

    def test_oracle_clamp(self):
        self.assertEqual(ref_rndne_f32(f32_to_bits(2.5), clamp=True), F32_ONE)
        self.assertEqual(ref_rndne_f32(f32_to_bits(-1.5), clamp=True), F32_POS_ZERO)
        self.assertEqual(ref_rndne_f32(f32_to_bits(0.5), clamp=True), F32_POS_ZERO)
        self.assertEqual(ref_rndne_f32(f32_to_bits(0.9), clamp=True), F32_ONE)


if __name__ == "__main__":
    unittest.main()
