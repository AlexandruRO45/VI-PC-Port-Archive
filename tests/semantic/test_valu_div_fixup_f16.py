from __future__ import annotations

"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_div_fixup_f16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, 16-bit half-precision floating-point operations
  operate on 16-bit values (IEEE 754 binary16): vd from source operands src0, src1, and src2.

  Instruction:
    v_div_fixup_f16 vd, src0, src1, src2 [clamp] [omod]

  Operands:
    src0 = provisional quotient from previous reciprocal/FMAS step
    src1 = denominator
    src2 = numerator

  IEEE-754 binary16 encoding:
    - 1 sign bit (bit 15)
    - 5 exponent bits with bias 15 (bits 14:10)
    - 10 mantissa / fraction bits (bits 9:0)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

import math
import struct
import unittest

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

# IEEE-754 binary16 bit constants
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_CANONICAL_QNAN = 0x7E00
F16_INDEFINITE_QNAN = 0xFE00  # Default indefinite NaN returned by AMD GPU for 0/0 and inf/inf
F16_QUIET_BIT = 0x0200

F16_EXP_MASK = 0x7C00
F16_MANT_MASK = 0x03FF
F16_SIGN_MASK = 0x8000

F16_ONE = 0x3C00            # 1.0
F16_NEG_ONE = 0xBC00        # -1.0
F16_TWO = 0x4000            # 2.0
F16_NEG_TWO = 0xC000        # -2.0
F16_HALF = 0x3800           # 0.5
F16_FOUR = 0x4400           # 4.0
F16_QUARTER = 0x3400        # 0.25
F16_MIN_NORMAL = 0x0400     # 2^-14
F16_MAX_NORMAL = 0x7BFF     # (2 - 2^-10) * 2^15 = 65504.0
F16_MIN_SUBNORMAL = 0x0001  # 2^-24
F16_MAX_SUBNORMAL = 0x03FF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range [0, 2^16 - 1]."""
    return val & MASK16


def f16_to_u16(val: float) -> int:
    """Pack an IEEE-754 half precision float into an unsigned 16-bit integer."""
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return F16_NEG_INF if val < 0 else F16_POS_INF


def u16_to_f16(bits: int) -> float:
    """Unpack an unsigned 16-bit integer into an IEEE-754 half precision float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def is_f16_nan(bits: int) -> bool:
    """Return True if the 16-bit pattern represents an IEEE-754 NaN."""
    return ((bits & F16_EXP_MASK) == F16_EXP_MASK) and ((bits & F16_MANT_MASK) != 0)


def is_f16_snan(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a signaling NaN."""
    return is_f16_nan(bits) and ((bits & F16_QUIET_BIT) == 0)


def is_f16_qnan(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a quiet NaN."""
    return is_f16_nan(bits) and ((bits & F16_QUIET_BIT) != 0)


def is_f16_zero(bits: int) -> bool:
    """Return True if the 16-bit pattern represents +0.0 or -0.0."""
    return (bits & (F16_EXP_MASK | F16_MANT_MASK)) == 0


def is_f16_inf(bits: int) -> bool:
    """Return True if the 16-bit pattern represents +inf or -inf."""
    return ((bits & F16_EXP_MASK) == F16_EXP_MASK) and ((bits & F16_MANT_MASK) == 0)


def is_f16_subnormal(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a subnormal/denormal float."""
    return ((bits & F16_EXP_MASK) == 0) and ((bits & F16_MANT_MASK) != 0)


def is_f16_normal(bits: int) -> bool:
    """Return True if the 16-bit pattern represents a normalized finite float."""
    exp = (bits >> 10) & 0x1F
    return 0 < exp < 0x1F


def f16_sign(bits: int) -> int:
    """Return the sign bit (0 or 1)."""
    return (bits >> 15) & 1


def f16_unbiased_exponent(bits: int) -> int:
    """Extract true mathematical binary exponent for finite non-zero value."""
    val = abs(u16_to_f16(bits))
    if val == 0.0 or math.isnan(val) or math.isinf(val):
        return 0
    _, e = math.frexp(val)
    return e - 1


def quiet_f16(bits: int) -> int:
    """Force an IEEE-754 half NaN to quiet by setting the quiet bit."""
    return (bits & MASK16) | F16_QUIET_BIT


def apply_src_mods_f16(bits: int, *, neg: bool = False, abs_val: bool = False) -> int:
    """Apply VOP3 input modifiers (abs, neg)."""
    res = bits & MASK16
    if abs_val:
        res &= ~F16_SIGN_MASK
    if neg:
        res ^= F16_SIGN_MASK
    return res


def apply_clamp_f16(bits: int) -> int:
    """Apply VOP3 clamp modifier to half precision float."""
    if is_f16_nan(bits) or bits == F16_NEG_ZERO:
        return F16_POS_ZERO
    val = u16_to_f16(bits)
    if val < 0.0:
        return F16_POS_ZERO
    elif val > 1.0:
        return F16_ONE
    return bits


def apply_omod_f16(bits: int, omod: int) -> int:
    """Apply RDNA2 output modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0 or is_f16_nan(bits) or is_f16_inf(bits) or is_f16_zero(bits):
        return bits
    val = u16_to_f16(bits)
    if omod == 1:
        val *= 2.0
    elif omod == 2:
        val *= 4.0
    elif omod == 3:
        val *= 0.5
    else:
        raise ValueError(f"Invalid omod value: {omod}")
    return f16_to_u16(val)


def ref_div_fixup_f16(
    s0_bits: int,
    s1_bits: int,
    s2_bits: int,
    *,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
    src2_neg: bool = False,
    src2_abs: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_div_fixup_f16 instruction."""
    s0 = apply_src_mods_f16(s0_bits, neg=src0_neg, abs_val=src0_abs)
    s1 = apply_src_mods_f16(s1_bits, neg=src1_neg, abs_val=src1_abs)
    s2 = apply_src_mods_f16(s2_bits, neg=src2_neg, abs_val=src2_abs)

    sign1 = f16_sign(s1)
    sign2 = f16_sign(s2)
    sign_out = sign1 ^ sign2

    # Decision tree according to AMD RDNA2 ISA specification
    if is_f16_nan(s2):
        res = quiet_f16(s2)
    elif is_f16_nan(s1):
        res = quiet_f16(s1)
    elif is_f16_zero(s1) and is_f16_zero(s2):
        # 0.0 / 0.0 -> Indefinite QNaN
        res = F16_INDEFINITE_QNAN
    elif is_f16_inf(s1) and is_f16_inf(s2):
        # inf / inf -> Indefinite QNaN
        res = F16_INDEFINITE_QNAN
    elif is_f16_zero(s1) or is_f16_inf(s2):
        # x / 0 or inf / y -> signed infinity
        res = F16_NEG_INF if sign_out else F16_POS_INF
    elif is_f16_inf(s1) or is_f16_zero(s2):
        # x / inf or 0 / y -> signed zero
        res = F16_NEG_ZERO if sign_out else F16_POS_ZERO
    elif is_f16_nan(s0):
        # Intermediate quotient overflowed to NaN -> signed infinity
        res = F16_NEG_INF if sign_out else F16_POS_INF
    elif (f16_unbiased_exponent(s2) - f16_unbiased_exponent(s1)) < -25:
        # Exponent underflow threshold (-25 for binary16 subnormal limit)
        res = F16_NEG_ZERO if sign_out else F16_POS_ZERO
    else:
        # Standard case: sign_out applied to absolute magnitude of s0
        res = (s0 & ~F16_SIGN_MASK) | (F16_SIGN_MASK if sign_out else 0)

    # Output modifiers
    if omod != 0:
        res = apply_omod_f16(res, omod)

    if clamp:
        res = apply_clamp_f16(res)

    return res & MASK16


def ref_div_fixup_f16_float(s0: float, s1: float, s2: float, **kwargs) -> float:
    """Float wrapper for ref_div_fixup_f16."""
    b0 = f16_to_u16(s0)
    b1 = f16_to_u16(s1)
    b2 = f16_to_u16(s2)
    res_bits = ref_div_fixup_f16(b0, b1, b2, **kwargs)
    return u16_to_f16(res_bits)


class HalfRegister:
    """Architectural model of a 16-bit register value."""

    def __init__(self, val_u16: int = 0) -> None:
        self._val = val_u16 & MASK16

    @property
    def as_u16(self) -> int:
        return self._val

    @as_u16.setter
    def as_u16(self, val: int) -> None:
        self._val = val & MASK16

    @property
    def as_f16(self) -> float:
        return u16_to_f16(self._val)

    @as_f16.setter
    def as_f16(self, val: float) -> None:
        self._val = f16_to_u16(val)

    @classmethod
    def from_u16(cls, val: int) -> HalfRegister:
        return cls(val)

    @classmethod
    def from_f16(cls, val: float) -> HalfRegister:
        return cls(f16_to_u16(val))

    def __repr__(self) -> str:
        return f"HalfRegister({self._val:#06x}, f16={self.as_f16})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HalfRegister):
            return NotImplemented
        return self._val == other._val


class TestDivFixupF16MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_normal_quotient_fixup(self):
        # 6.0 / 2.0 = 3.0, s0 is estimated 3.0
        s0 = f16_to_u16(3.0)
        s1 = f16_to_u16(2.0)
        s2 = f16_to_u16(6.0)
        res = ref_div_fixup_f16(s0, s1, s2)
        self.assertEqual(res, f16_to_u16(3.0))

        # Sign correction
        s2_neg = f16_to_u16(-6.0)
        res_neg = ref_div_fixup_f16(s0, s1, s2_neg)
        self.assertEqual(res_neg, f16_to_u16(-3.0))

    def test_division_by_zero(self):
        s0 = f16_to_u16(1.0)
        res_pz = ref_div_fixup_f16(s0, F16_POS_ZERO, f16_to_u16(1.0))
        self.assertEqual(res_pz, F16_POS_INF)

        res_nz = ref_div_fixup_f16(s0, F16_NEG_ZERO, f16_to_u16(1.0))
        self.assertEqual(res_nz, F16_NEG_INF)

    def test_zero_numerator(self):
        s0 = f16_to_u16(0.0)
        res = ref_div_fixup_f16(s0, f16_to_u16(4.0), F16_POS_ZERO)
        self.assertEqual(res, F16_POS_ZERO)

        res_nz = ref_div_fixup_f16(s0, f16_to_u16(4.0), F16_NEG_ZERO)
        self.assertEqual(res_nz, F16_NEG_ZERO)

    def test_zero_divided_by_zero(self):
        res = ref_div_fixup_f16(F16_POS_ZERO, F16_POS_ZERO, F16_POS_ZERO)
        self.assertEqual(res, F16_INDEFINITE_QNAN)

    def test_infinity_divided_by_infinity(self):
        res = ref_div_fixup_f16(F16_POS_INF, F16_POS_INF, F16_POS_INF)
        self.assertEqual(res, F16_INDEFINITE_QNAN)

    def test_finite_divided_by_infinity(self):
        res = ref_div_fixup_f16(f16_to_u16(0.0), F16_POS_INF, f16_to_u16(3.0))
        self.assertEqual(res, F16_POS_ZERO)

    def test_infinity_divided_by_finite(self):
        res = ref_div_fixup_f16(f16_to_u16(1.0), f16_to_u16(3.0), F16_POS_INF)
        self.assertEqual(res, F16_POS_INF)

    def test_nan_numerator_and_denominator_priority(self):
        nan2 = 0x7E20
        nan1 = 0x7E10
        res = ref_div_fixup_f16(F16_ONE, nan1, nan2)
        self.assertEqual(res, quiet_f16(nan2))

        res1 = ref_div_fixup_f16(F16_ONE, nan1, F16_ONE)
        self.assertEqual(res1, quiet_f16(nan1))

    def test_s0_nan_overflow(self):
        res = ref_div_fixup_f16(F16_CANONICAL_QNAN, F16_ONE, F16_ONE)
        self.assertEqual(res, F16_POS_INF)

        res_neg = ref_div_fixup_f16(F16_CANONICAL_QNAN, F16_ONE, F16_NEG_ONE)
        self.assertEqual(res_neg, F16_NEG_INF)

    def test_exponent_underflow(self):
        # Exponent diff < -25 produces signed zero
        s2 = f16_to_u16(2.0 ** -14)
        s1 = f16_to_u16(2.0 ** 14)
        res = ref_div_fixup_f16(F16_ONE, s1, s2)
        self.assertEqual(res, F16_POS_ZERO)

    def test_source_and_output_modifiers(self):
        s0 = f16_to_u16(2.0)
        s1 = f16_to_u16(1.0)
        s2 = f16_to_u16(2.0)

        # src2 neg
        res_neg2 = ref_div_fixup_f16(s0, s1, s2, src2_neg=True)
        self.assertEqual(res_neg2, f16_to_u16(-2.0))

        # omod: mul:2
        res_omod = ref_div_fixup_f16(s0, s1, s2, omod=1)
        self.assertEqual(res_omod, f16_to_u16(4.0))

        # clamp
        res_clamp = ref_div_fixup_f16(f16_to_u16(5.0), s1, s2, clamp=True)
        self.assertEqual(res_clamp, F16_ONE)

    def test_half_register_model(self):
        reg = HalfRegister.from_f16(2.5)
        self.assertEqual(reg.as_f16, 2.5)
        self.assertEqual(reg.as_u16, f16_to_u16(2.5))

        reg.as_u16 = F16_ONE
        self.assertEqual(reg.as_f16, 1.0)


if __name__ == "__main__":
    unittest.main()
