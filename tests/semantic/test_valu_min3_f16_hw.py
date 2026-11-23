"""Semantic hardware and translator verification suite for RDNA2 v_min3_f16:
v_min3_f16 (VOP3 3-source minimum of half-precision floating-point numbers).

Features & specifications:
- IEEE 754-2008 / RDNA2 rules: NaN suppression (return non-NaN operand; return quiet NaN if all 3 are NaN).
- Signed zero ordering: -0.0 < +0.0.
- Destination register half preservation: keeps unwritten 16-bit half (low or high based on op_sel dst bit).
- VOP3 op_sel support: select low or high 16 bits of each source and destination.
- Input modifiers: abs, neg, -abs.
- Output clamp: clamp to [0.0, 1.0], negative values / -0.0 become +0.0, NaNs become +0.0.
- Standard library Python only; zero game data.
"""
from __future__ import annotations

import math
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
OLD_VAL = 0xA55AA55A

# IEEE 754 binary16 bit constants
POS_ZERO = 0x0000
NEG_ZERO = 0x8000
POS_ONE = 0x3C00
NEG_ONE = 0xBC00
POS_HALF = 0x3800
NEG_HALF = 0xB800
POS_TWO = 0x4000
NEG_TWO = 0xC000
POS_THREE = 0x4200
NEG_THREE = 0xC200
POS_FOUR = 0x4400
NEG_FOUR = 0xC400
POS_INF = 0x7C00
NEG_INF = 0xFC00
QNAN_CANONICAL = 0x7E00
QNAN_ALT = 0x7E55
SNAN = 0x7C01
NEG_QNAN = 0xFE00

POS_MIN_SUBNORM = 0x0001
POS_MAX_SUBNORM = 0x03FF
NEG_MIN_SUBNORM = 0x8001
NEG_MAX_SUBNORM = 0x83FF
POS_MIN_NORMAL = 0x0400
POS_MAX_NORMAL = 0x7BFF
NEG_MAX_NORMAL = 0xFBFF


def f16_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def float_to_f16_bits(f: float) -> int:
    if math.isnan(f):
        return QNAN_CANONICAL
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return POS_INF if f > 0 else NEG_INF


def is_nan16(bits: int) -> bool:
    return (bits & 0x7C00) == 0x7C00 and (bits & 0x03FF) != 0


def ref_min2_f16(a_bits: int, b_bits: int) -> int:
    a_nan = is_nan16(a_bits)
    b_nan = is_nan16(b_bits)
    if a_nan and b_nan:
        return QNAN_CANONICAL
    if a_nan:
        return b_bits & MASK16
    if b_nan:
        return a_bits & MASK16

    is_a_neg_zero = (a_bits & MASK16) == NEG_ZERO
    is_a_pos_zero = (a_bits & MASK16) == POS_ZERO
    is_b_neg_zero = (b_bits & MASK16) == NEG_ZERO
    is_b_pos_zero = (b_bits & MASK16) == POS_ZERO

    if (is_a_neg_zero and is_b_pos_zero) or (is_a_pos_zero and is_b_neg_zero):
        return NEG_ZERO
    if is_a_neg_zero and is_b_neg_zero:
        return NEG_ZERO
    if is_a_pos_zero and is_b_pos_zero:
        return POS_ZERO

    fa = f16_to_float(a_bits)
    fb = f16_to_float(b_bits)
    key_a = (fa, math.copysign(1.0, fa))
    key_b = (fb, math.copysign(1.0, fb))
    return (a_bits & MASK16) if key_a <= key_b else (b_bits & MASK16)


def ref_min3_f16(s0: int, s1: int, s2: int, *, clamp: bool = False) -> int:
    m = ref_min2_f16(s0, s1)
    res = ref_min2_f16(m, s2)
    if clamp:
        if is_nan16(res):
            return POS_ZERO
        cf = f16_to_float(res)
        if cf <= 0.0 or (res & MASK16) == NEG_ZERO:
            return POS_ZERO
        if cf >= 1.0:
            return POS_ONE
    return res & MASK16


def apply_f16_input_mod(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    b = bits & MASK16
    if abs_val:
        b &= 0x7FFF
    if neg:
        b ^= 0x8000
    return b


class TestValuMin3F16Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model."""

    def test_basic_orderings(self):
        self.assertEqual(ref_min3_f16(POS_ONE, POS_TWO, POS_HALF), POS_HALF)
        self.assertEqual(ref_min3_f16(POS_TWO, POS_HALF, POS_ONE), POS_HALF)
        self.assertEqual(ref_min3_f16(POS_HALF, POS_ONE, POS_TWO), POS_HALF)
        self.assertEqual(ref_min3_f16(NEG_ONE, NEG_TWO, POS_HALF), NEG_TWO)

    def test_signed_zeros(self):
        self.assertEqual(ref_min3_f16(POS_ZERO, NEG_ZERO, POS_ONE), NEG_ZERO)
        self.assertEqual(ref_min3_f16(POS_ZERO, POS_ZERO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_min3_f16(POS_ZERO, POS_ZERO, POS_ZERO), POS_ZERO)
        self.assertEqual(ref_min3_f16(NEG_ZERO, NEG_ZERO, NEG_ZERO), NEG_ZERO)

    def test_nan_suppression(self):
        self.assertEqual(ref_min3_f16(QNAN_CANONICAL, POS_ONE, POS_TWO), POS_ONE)
        self.assertEqual(ref_min3_f16(POS_TWO, QNAN_CANONICAL, POS_ONE), POS_ONE)
        self.assertEqual(ref_min3_f16(QNAN_CANONICAL, QNAN_ALT, POS_THREE), POS_THREE)
        self.assertTrue(is_nan16(ref_min3_f16(QNAN_CANONICAL, QNAN_ALT, SNAN)))

    def test_clamping(self):
        self.assertEqual(ref_min3_f16(POS_TWO, POS_THREE, POS_FOUR, clamp=True), POS_ONE)
        self.assertEqual(ref_min3_f16(NEG_ONE, POS_HALF, POS_TWO, clamp=True), POS_ZERO)
        self.assertEqual(ref_min3_f16(QNAN_CANONICAL, QNAN_ALT, SNAN, clamp=True), POS_ZERO)
        self.assertEqual(ref_min3_f16(POS_HALF, POS_ONE, POS_TWO, clamp=True), POS_HALF)

if __name__ == '__main__':
    unittest.main()
