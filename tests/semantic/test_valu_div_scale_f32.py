from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_div_scale_f32.

Tests the single-precision floating-point division pre-scaling instruction:
    v_div_scale_f32 D, sdst, S0, S1, S2

Hardware Specification (RDNA2 / GFX10 / GFX1030 VOP3b dual-output):
    D    = 32-bit vector destination register v[d] receiving scaled S0
    sdst = scalar destination (VCC or SGPR) receiving scale condition bit mask
    S0   = operand to scale (32-bit float, typically denominator S1 or numerator S2)
    S1   = denominator (32-bit float)
    S2   = numerator (32-bit float)

Operation:
    v_div_scale_f32 examines the denominator S1 and numerator S2 to check whether
    intermediate Newton-Raphson division approximation will overflow or underflow.
    If scaling is necessary:
      - If 0.0 < |S1| <= 2^-88: D = S0 * 2^64, sdst = 1
      - Else if |S1| >= 2^62 and finite: D = S0 * 2^-64, sdst = 1
      - Else if 0.0 < |S2| <= 2^-88: D = S0 * 2^64, sdst = 1
      - Else if |S2| >= 2^62 and finite: D = S0 * 2^-64, sdst = 1
      - Else: D = S0, sdst = 0

Special Cases:
    - If S0 is NaN: D = quiet(S0), sdst = 0.
    - If S1 or S2 is NaN: D = S0, sdst = 0.
    - Zero/Infinity: if not matching threshold conditions, unscaled, sdst = 0.

Modifiers:
    - Input modifiers: abs (|src|: clear sign bit), neg (-src: invert sign bit).
    - Output modifiers:
        - clamp: saturates result to [0.0, 1.0], with NaN clamped to +0.0.
        - omod: mul:2, mul:4, div:2.

Architectural Model:
    32-bit registers (VGPR / SGPR) with IEEE 754 binary32 representation.
"""

import math
import struct
import unittest

# IEEE 754 single precision constants
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_ONE = 0x3F800000
F32_NEG_ONE = 0xBF800000
F32_TWO = 0x40000000
F32_HALF = 0x3F000000
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_CANONICAL_QNAN = 0x7FC00000
F32_SNAN = 0x7F800001
F32_SUB_MIN = 0x00000001
F32_SUB_MAX = 0x007FFFFF
F32_MIN_NORMAL = 0x00800000
F32_MAX_NORMAL = 0x7F7FFFFF

MASK32 = 0xFFFFFFFF

# Scaling constants for f32
SCALE64_POS = 2.0 ** 64
SCALE64_NEG = 2.0 ** -64

THRESH_F32_SMALL = 2.0 ** -88
THRESH_F32_LARGE = 2.0 ** 62


def f32_to_u32(val: float) -> int:
    """Pack Python float into unsigned 32-bit integer (IEEE 754 binary32)."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def u32_to_f32(bits: int) -> float:
    """Unpack unsigned 32-bit integer into Python float (IEEE 754 binary32)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def is_f32_nan(bits: int) -> bool:
    return ((bits & 0x7F800000) == 0x7F800000) and ((bits & 0x007FFFFF) != 0)


def is_f32_snan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) == 0)


def is_f32_qnan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) != 0)


def quiet_f32(bits: int) -> int:
    return (bits & MASK32) | 0x00400000


def apply_src_mods_u32(val_u32: int, *, neg: bool = False, abs_: bool = False) -> int:
    """Apply bit-exact RDNA2 source modifiers (|src|, -src) to a 32-bit float bit pattern."""
    bits = val_u32 & MASK32
    if abs_:
        bits &= ~F32_NEG_ZERO
    if neg:
        bits ^= F32_NEG_ZERO
    return bits


def apply_clamp_f32(val: float) -> float:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to float32, mapping NaNs to +0.0."""
    if math.isnan(val):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def apply_omod_f32(val: float, omod: int) -> float:
    """Apply RDNA2 omod modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0:
        return val
    elif omod == 1:
        return val * 2.0
    elif omod == 2:
        return val * 4.0
    elif omod == 3:
        return val * 0.5
    else:
        raise ValueError(f"Invalid omod value: {omod}")


def ref_div_scale_f32(
    s0: float,
    s1: float,
    s2: float,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> tuple[float, int]:
    """Reference mathematical oracle for v_div_scale_f32.

    Returns (scaled_result, vcc_bit).
    """
    u0 = apply_src_mods_u32(f32_to_u32(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u32(f32_to_u32(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u32(f32_to_u32(s2), neg=neg2, abs_=abs2)

    if is_f32_nan(u0):
        res = u32_to_f32(quiet_f32(u0))
        if clamp:
            res = apply_clamp_f32(res)
        return res, 0

    if is_f32_nan(u1) or is_f32_nan(u2):
        res = u32_to_f32(u0)
        if clamp:
            res = apply_clamp_f32(res)
        return res, 0

    v0 = u32_to_f32(u0)
    v1 = u32_to_f32(u1)
    v2 = u32_to_f32(u2)

    abs_y = abs(v1)
    abs_x = abs(v2)

    scaled = False
    scale_factor = 1.0

    # Denominator check has priority
    if 0.0 < abs_y <= THRESH_F32_SMALL:
        scaled = True
        scale_factor = SCALE64_POS
    elif abs_y >= THRESH_F32_LARGE and math.isfinite(abs_y):
        scaled = True
        scale_factor = SCALE64_NEG
    elif 0.0 < abs_x <= THRESH_F32_SMALL:
        scaled = True
        scale_factor = SCALE64_POS
    elif abs_x >= THRESH_F32_LARGE and math.isfinite(abs_x):
        scaled = True
        scale_factor = SCALE64_NEG

    if scaled:
        res = v0 * scale_factor
        vcc = 1
    else:
        res = v0
        vcc = 0

    if omod != 0:
        res = apply_omod_f32(res, omod)
    if clamp:
        res = apply_clamp_f32(res)

    return res, vcc


def ref_div_scale_f32_u32(
    u0: int,
    u1: int,
    u2: int,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> tuple[int, int]:
    """Bit-exact reference mathematical oracle for v_div_scale_f32 using unsigned 32-bit words."""
    res_f, vcc = ref_div_scale_f32(
        u32_to_f32(u0),
        u32_to_f32(u1),
        u32_to_f32(u2),
        neg0=neg0,
        neg1=neg1,
        neg2=neg2,
        abs0=abs0,
        abs1=abs1,
        abs2=abs2,
        clamp=clamp,
        omod=omod,
    )
    return f32_to_u32(res_f), vcc


class ScalarRegister:
    """Architectural model of a 32-bit vector or scalar register."""

    def __init__(self, raw: int = 0) -> None:
        self.raw = raw & MASK32

    @classmethod
    def from_u32(cls, val: int) -> ScalarRegister:
        return cls(val)

    @classmethod
    def from_f32(cls, val: float) -> ScalarRegister:
        return cls(f32_to_u32(val))

    @property
    def as_u32(self) -> int:
        return self.raw

    @property
    def as_f32(self) -> float:
        return u32_to_f32(self.raw)

    def set_u32(self, val: int) -> None:
        self.raw = val & MASK32

    def set_f32(self, val: float) -> None:
        self.raw = f32_to_u32(val)

    def __repr__(self) -> str:
        return f"ScalarRegister(raw={self.raw:#010x}, f32={self.as_f32})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScalarRegister):
            return NotImplemented
        return self.raw == other.raw


class TestDivScaleF32MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_normal_no_scale(self):
        # 10.0 / 2.0 with s0 = 2.0 (denominator) -> no scale required
        res, vcc = ref_div_scale_f32(2.0, 2.0, 10.0)
        self.assertEqual(res, 2.0)
        self.assertEqual(vcc, 0)

        # 10.0 / 2.0 with s0 = 10.0 (numerator) -> no scale required
        res_num, vcc_num = ref_div_scale_f32(10.0, 2.0, 10.0)
        self.assertEqual(res_num, 10.0)
        self.assertEqual(vcc_num, 0)

    def test_scale_up_small_denominator(self):
        # Small denominator (e.g. 2^-100): scale up by 2^64
        small_denom = 2.0 ** -100
        res, vcc = ref_div_scale_f32(small_denom, small_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, small_denom * SCALE64_POS)

    def test_scale_down_large_denominator(self):
        # Large denominator (e.g. 2^70): scale down by 2^-64
        large_denom = 2.0 ** 70
        res, vcc = ref_div_scale_f32(large_denom, large_denom, 1.0)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, large_denom * SCALE64_NEG)

    def test_scale_up_small_numerator(self):
        # Small numerator (e.g. 2^-100): scale up by 2^64
        small_num = 2.0 ** -100
        res, vcc = ref_div_scale_f32(small_num, 1.0, small_num)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, small_num * SCALE64_POS)

    def test_scale_down_large_numerator(self):
        # Large numerator (e.g. 2^70): scale down by 2^-64
        large_num = 2.0 ** 70
        res, vcc = ref_div_scale_f32(large_num, 1.0, large_num)
        self.assertEqual(vcc, 1)
        self.assertAlmostEqual(res, large_num * SCALE64_NEG)

    def test_source_modifiers_neg_abs(self):
        # abs clears sign bit
        res_abs, _ = ref_div_scale_f32(-2.5, 2.0, 1.0, abs0=True)
        self.assertEqual(res_abs, 2.5)

        # neg inverts sign bit
        res_neg, _ = ref_div_scale_f32(2.5, 2.0, 1.0, neg0=True)
        self.assertEqual(res_neg, -2.5)

        # combined neg and abs: -|src|
        res_neg_abs, _ = ref_div_scale_f32(2.5, 2.0, 1.0, neg0=True, abs0=True)
        self.assertEqual(res_neg_abs, -2.5)

    def test_output_modifiers_clamp_omod(self):
        # clamp saturates to [0.0, 1.0]
        res_clamp_hi, _ = ref_div_scale_f32(2.5, 2.0, 1.0, clamp=True)
        self.assertEqual(res_clamp_hi, 1.0)

        res_clamp_lo, _ = ref_div_scale_f32(-0.5, 2.0, 1.0, clamp=True)
        self.assertEqual(res_clamp_lo, 0.0)

        # omod: mul:2, mul:4, div:2
        res_mul2, _ = ref_div_scale_f32(1.5, 2.0, 1.0, omod=1)
        self.assertEqual(res_mul2, 3.0)

        res_mul4, _ = ref_div_scale_f32(1.5, 2.0, 1.0, omod=2)
        self.assertEqual(res_mul4, 6.0)

        res_div2, _ = ref_div_scale_f32(1.5, 2.0, 1.0, omod=3)
        self.assertEqual(res_div2, 0.75)

    def test_architectural_register_model(self):
        reg = ScalarRegister.from_f32(1.5)
        self.assertEqual(reg.as_f32, 1.5)
        self.assertEqual(reg.as_u32, 0x3FC00000)

        reg.set_f32(-1.0)
        self.assertEqual(reg.as_f32, -1.0)
        self.assertEqual(reg.as_u32, F32_NEG_ONE)


if __name__ == "__main__":
    unittest.main()
