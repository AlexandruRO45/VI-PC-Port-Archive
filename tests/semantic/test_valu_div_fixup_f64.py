from __future__ import annotations

"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_div_fixup_f64.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector general-purpose registers (VGPRs) are 32 bits wide.
  Operations on 64-bit double-precision floating-point numbers (binary64, IEEE 754)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source operands
  v[s0:s0+1], v[s1:s1+1], and v[s2:s2+1] (or scalar SGPR pairs s[k:k+1], literal constants).

  Instruction:
    v_div_fixup_f64 v[d:d+1], src0, src1, src2 [clamp] [omod]

  Operands:
    src0 = provisional quotient from previous reciprocal/FMAS step
    src1 = denominator
    src2 = numerator

  IEEE-754 binary64 encoding:
    - 1 sign bit (bit 63)
    - 11 exponent bits with bias 1023 (bits 62:52)
    - 52 mantissa / fraction bits (bits 51:0)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

from dataclasses import dataclass
import math
import struct
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE-754 binary64 bit constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_CANONICAL_QNAN = 0x7FF8000000000000
F64_INDEFINITE_QNAN = 0xFFF8000000000000  # Default indefinite NaN returned by AMD GPU for 0/0 and inf/inf
F64_QUIET_BIT = 0x0008000000000000

F64_EXP_MASK = 0x7FF0000000000000
F64_MANT_MASK = 0x000FFFFFFFFFFFFF
F64_SIGN_MASK = 0x8000000000000000

F64_ONE = 0x3FF0000000000000            # 1.0
F64_NEG_ONE = 0xBFF0000000000000        # -1.0
F64_TWO = 0x4000000000000000            # 2.0
F64_NEG_TWO = 0xC000000000000000        # -2.0
F64_HALF = 0x3FE0000000000000           # 0.5
F64_FOUR = 0x4010000000000000           # 4.0
F64_QUARTER = 0x3FD0000000000000        # 0.25
F64_MIN_NORMAL = 0x0010000000000000     # 2^-1022
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF     # (2 - 2^-52) * 2^1023
F64_MIN_SUBNORMAL = 0x0000000000000001    # 2^-1074
F64_MAX_SUBNORMAL = 0x000FFFFFFFFFFFFF

OLD_VGPR_SENTINEL = 0xDEADBEEF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_to_u64(val: float) -> int:
    """Pack an IEEE-754 double precision float into an unsigned 64-bit integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return F64_NEG_INF if val < 0 else F64_POS_INF


def u64_to_f64(bits: int) -> float:
    """Unpack an unsigned 64-bit integer into an IEEE-754 double precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def is_f64_nan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents an IEEE-754 NaN."""
    return ((bits & F64_EXP_MASK) == F64_EXP_MASK) and ((bits & F64_MANT_MASK) != 0)


def is_f64_snan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a signaling NaN."""
    return is_f64_nan(bits) and ((bits & F64_QUIET_BIT) == 0)


def is_f64_qnan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a quiet NaN."""
    return is_f64_nan(bits) and ((bits & F64_QUIET_BIT) != 0)


def is_f64_zero(bits: int) -> bool:
    """Return True if the 64-bit pattern represents +0.0 or -0.0."""
    return (bits & (F64_EXP_MASK | F64_MANT_MASK)) == 0


def is_f64_inf(bits: int) -> bool:
    """Return True if the 64-bit pattern represents +inf or -inf."""
    return ((bits & F64_EXP_MASK) == F64_EXP_MASK) and ((bits & F64_MANT_MASK) == 0)


def is_f64_subnormal(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a subnormal/denormal float."""
    return ((bits & F64_EXP_MASK) == 0) and ((bits & F64_MANT_MASK) != 0)


def is_f64_normal(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a normalized finite float."""
    exp = (bits >> 52) & 0x7FF
    return 0 < exp < 0x7FF


def f64_sign(bits: int) -> int:
    """Return the sign bit (0 or 1)."""
    return (bits >> 63) & 1


def f64_unbiased_exponent(bits: int) -> int:
    """Extract true mathematical binary exponent for finite non-zero value."""
    val = abs(u64_to_f64(bits))
    if val == 0.0 or math.isnan(val) or math.isinf(val):
        return 0
    _, e = math.frexp(val)
    return e - 1


def quiet_f64(bits: int) -> int:
    """Force an IEEE-754 double NaN to quiet by setting the quiet bit."""
    return (bits & MASK64) | F64_QUIET_BIT


def apply_src_mods_f64(bits: int, *, neg: bool = False, abs_val: bool = False) -> int:
    """Apply VOP3 input modifiers (abs, neg).

    In RDNA2 ISA:
      When both abs and neg are set, abs is evaluated before neg:
      i.e. neg(abs(x)) forces the sign bit to 1 (negative).
    """
    res = bits & MASK64
    if abs_val:
        res &= ~F64_SIGN_MASK
    if neg:
        res ^= F64_SIGN_MASK
    return res


def apply_clamp_f64(bits: int) -> int:
    """Apply VOP3 clamp modifier to double precision float.

    Saturates result to [+0.0, 1.0].
    NaN is converted to +0.0.
    -0.0 is clamped to +0.0.
    Values < 0.0 are clamped to +0.0.
    Values > 1.0 are clamped to 1.0.
    """
    if is_f64_nan(bits) or bits == F64_NEG_ZERO:
        return F64_POS_ZERO
    val = u64_to_f64(bits)
    if val < 0.0:
        return F64_POS_ZERO
    elif val > 1.0:
        return F64_ONE
    return bits


def ref_div_fixup_f64(
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
    omod: float = 1.0,
) -> int:
    """Reference mathematical oracle for AMD RDNA2 v_div_fixup_f64 instruction.

    Args:
        s0_bits: Provisional quotient (64-bit uint)
        s1_bits: Denominator (64-bit uint)
        s2_bits: Numerator (64-bit uint)
        src0_neg / src0_abs: Modifiers for src0
        src1_neg / src1_abs: Modifiers for src1
        src2_neg / src2_abs: Modifiers for src2
        clamp: If True, saturates to [0.0, 1.0] (NaN -> +0.0)
        omod: Output modifier (1.0, 2.0, 4.0, 0.5)

    Returns:
        Fixed-up quotient as 64-bit uint.
    """
    s0 = apply_src_mods_f64(s0_bits, neg=src0_neg, abs_val=src0_abs)
    s1 = apply_src_mods_f64(s1_bits, neg=src1_neg, abs_val=src1_abs)
    s2 = apply_src_mods_f64(s2_bits, neg=src2_neg, abs_val=src2_abs)

    sign1 = f64_sign(s1)
    sign2 = f64_sign(s2)
    sign_out = sign1 ^ sign2

    # Decision tree according to AMD RDNA2 ISA specification
    if is_f64_nan(s2):
        res = quiet_f64(s2)
    elif is_f64_nan(s1):
        res = quiet_f64(s1)
    elif is_f64_zero(s1) and is_f64_zero(s2):
        # 0.0 / 0.0 -> Indefinite QNaN
        res = F64_INDEFINITE_QNAN
    elif is_f64_inf(s1) and is_f64_inf(s2):
        # inf / inf -> Indefinite QNaN
        res = F64_INDEFINITE_QNAN
    elif is_f64_zero(s1) or is_f64_inf(s2):
        # x / 0 or inf / y -> signed infinity
        res = F64_NEG_INF if sign_out else F64_POS_INF
    elif is_f64_inf(s1) or is_f64_zero(s2):
        # x / inf or 0 / y -> signed zero
        res = F64_NEG_ZERO if sign_out else F64_POS_ZERO
    elif is_f64_nan(s0):
        # Intermediate quotient overflowed to NaN -> signed infinity
        res = F64_NEG_INF if sign_out else F64_POS_INF
    elif (f64_unbiased_exponent(s2) - f64_unbiased_exponent(s1)) < -1075:
        # Exponent underflow threshold (-1075 for binary64 subnormal limit)
        res = F64_NEG_ZERO if sign_out else F64_POS_ZERO
    else:
        # Standard case: sign_out applied to absolute magnitude of s0
        res = (s0 & ~F64_SIGN_MASK) | (F64_SIGN_MASK if sign_out else 0)

    # Output modifiers
    if omod != 1.0 and not is_f64_nan(res) and not is_f64_inf(res) and not is_f64_zero(res):
        fval = u64_to_f64(res) * omod
        res = f64_to_u64(fval)

    if clamp:
        res = apply_clamp_f64(res)

    return res & MASK64


def ref_div_fixup_f64_dwords(
    s0_lo: int,
    s0_hi: int,
    s1_lo: int,
    s1_hi: int,
    s2_lo: int,
    s2_hi: int,
    **kwargs,
) -> tuple[int, int]:
    """Dword-level wrapper for ref_div_fixup_f64 operating on 32-bit register halves."""
    s0 = join_dwords(s0_lo, s0_hi)
    s1 = join_dwords(s1_lo, s1_hi)
    s2 = join_dwords(s2_lo, s2_hi)
    res = ref_div_fixup_f64(s0, s1, s2, **kwargs)
    return split_dwords(res)


class VALURegisterFile:
    """Architectural register model for RDNA2 VALU VGPR file.

    Models 32-bit physical VGPRs with 64-bit aligned pair access.
    """

    def __init__(self, num_vgprs: int = 256) -> None:
        if not (1 <= num_vgprs <= 256):
            raise ValueError(f"num_vgprs must be in range [1, 256], got {num_vgprs}")
        self.num_vgprs = num_vgprs
        self.vgprs = [0] * num_vgprs

    def get_vgpr(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: {idx}")
        return self.vgprs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: {idx}")
        self.vgprs[idx] = val & MASK32

    def get_vgpr_pair(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"VGPR pair index out of range: v[{idx}:{idx+1}]")
        lo = self.get_vgpr(idx)
        hi = self.get_vgpr(idx + 1)
        return join_dwords(lo, hi)

    def set_vgpr_pair(self, idx: int, val64: int) -> None:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"VGPR pair index out of range: v[{idx}:{idx+1}]")
        lo, hi = split_dwords(val64)
        self.set_vgpr(idx, lo)
        self.set_vgpr(idx + 1, hi)


class TestVDivFixupF64ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_basic_quotient_fixup(self) -> None:
        # Normal quotient: 10.0 / 2.0 = 5.0
        s1 = f64_to_u64(2.0)
        s2 = f64_to_u64(10.0)
        s0 = f64_to_u64(5.0)

        # Both positive (+ / + -> +)
        r = ref_div_fixup_f64(s0, s1, s2)
        self.assertEqual(u64_to_f64(r), 5.0)

        # Numerator negative (- / + -> -)
        r = ref_div_fixup_f64(s0, s1, f64_to_u64(-10.0))
        self.assertEqual(u64_to_f64(r), -5.0)

        # Denominator negative (+ / - -> -)
        r = ref_div_fixup_f64(s0, f64_to_u64(-2.0), s2)
        self.assertEqual(u64_to_f64(r), -5.0)

        # Both negative (- / - -> +)
        r = ref_div_fixup_f64(s0, f64_to_u64(-2.0), f64_to_u64(-10.0))
        self.assertEqual(u64_to_f64(r), 5.0)

    def test_zero_divided_by_zero(self) -> None:
        # 0.0 / 0.0 -> Indefinite QNaN (0xFFF8_0000_0000_0000)
        r = ref_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_POS_ZERO)
        self.assertEqual(r, F64_INDEFINITE_QNAN)

        r = ref_div_fixup_f64(F64_ONE, F64_NEG_ZERO, F64_POS_ZERO)
        self.assertEqual(r, F64_INDEFINITE_QNAN)

    def test_inf_divided_by_inf(self) -> None:
        # Inf / Inf -> Indefinite QNaN (0xFFF8_0000_0000_0000)
        r = ref_div_fixup_f64(F64_ONE, F64_POS_INF, F64_POS_INF)
        self.assertEqual(r, F64_INDEFINITE_QNAN)

        r = ref_div_fixup_f64(F64_ONE, F64_NEG_INF, F64_POS_INF)
        self.assertEqual(r, F64_INDEFINITE_QNAN)

    def test_division_by_zero(self) -> None:
        # x / +0.0 -> +Inf (for x > 0)
        r = ref_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_ONE)
        self.assertEqual(r, F64_POS_INF)

        # -x / +0.0 -> -Inf
        r = ref_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_NEG_ONE)
        self.assertEqual(r, F64_NEG_INF)

        # x / -0.0 -> -Inf
        r = ref_div_fixup_f64(F64_ONE, F64_NEG_ZERO, F64_ONE)
        self.assertEqual(r, F64_NEG_INF)

        # -x / -0.0 -> +Inf
        r = ref_div_fixup_f64(F64_ONE, F64_NEG_ZERO, F64_NEG_ONE)
        self.assertEqual(r, F64_POS_INF)

    def test_numerator_infinity(self) -> None:
        # +Inf / x -> +Inf
        r = ref_div_fixup_f64(F64_ONE, F64_TWO, F64_POS_INF)
        self.assertEqual(r, F64_POS_INF)

        # -Inf / x -> -Inf
        r = ref_div_fixup_f64(F64_ONE, F64_TWO, F64_NEG_INF)
        self.assertEqual(r, F64_NEG_INF)

    def test_denominator_infinity(self) -> None:
        # x / +Inf -> +0.0
        r = ref_div_fixup_f64(F64_ONE, F64_POS_INF, F64_TWO)
        self.assertEqual(r, F64_POS_ZERO)

        # -x / +Inf -> -0.0
        r = ref_div_fixup_f64(F64_ONE, F64_POS_INF, F64_NEG_TWO)
        self.assertEqual(r, F64_NEG_ZERO)

    def test_zero_numerator(self) -> None:
        # +0.0 / x -> +0.0
        r = ref_div_fixup_f64(F64_ONE, F64_TWO, F64_POS_ZERO)
        self.assertEqual(r, F64_POS_ZERO)

        # -0.0 / x -> -0.0
        r = ref_div_fixup_f64(F64_ONE, F64_TWO, F64_NEG_ZERO)
        self.assertEqual(r, F64_NEG_ZERO)

    def test_nan_propagation(self) -> None:
        # S2 (numerator) is NaN -> quieted S2 returned
        snan = 0x7FF0000000000001
        r = ref_div_fixup_f64(F64_ONE, F64_TWO, snan)
        self.assertTrue(is_f64_qnan(r))
        self.assertEqual(r, snan | F64_QUIET_BIT)

        # S1 (denominator) is NaN -> quieted S1 returned
        r2 = ref_div_fixup_f64(F64_ONE, snan, F64_TWO)
        self.assertTrue(is_f64_qnan(r2))
        self.assertEqual(r2, snan | F64_QUIET_BIT)

    def test_s0_nan_recovery(self) -> None:
        # Intermediate S0 is NaN while S1 and S2 are finite non-zero -> recovers to signed Inf
        r = ref_div_fixup_f64(F64_CANONICAL_QNAN, F64_TWO, F64_FOUR)
        self.assertEqual(r, F64_POS_INF)

        r_neg = ref_div_fixup_f64(F64_CANONICAL_QNAN, F64_TWO, F64_NEG_ONE)
        self.assertEqual(r_neg, F64_NEG_INF)

    def test_exponent_underflow(self) -> None:
        # Exponent diff < -1075 -> underflow to signed zero
        # Numerator = 2^-1074 (min subnormal), Denominator = 2^10 (exp diff = -1084 < -1075)
        s2 = F64_MIN_SUBNORMAL
        s1 = f64_to_u64(1024.0)
        r = ref_div_fixup_f64(F64_ONE, s1, s2)
        self.assertEqual(r, F64_POS_ZERO)

        # Negative sign preserved
        r_neg = ref_div_fixup_f64(F64_ONE, s1, s2, src2_neg=True)
        self.assertEqual(r_neg, F64_NEG_ZERO)

    def test_input_modifiers(self) -> None:
        s1 = f64_to_u64(-2.0)
        s2 = f64_to_u64(-10.0)
        s0 = f64_to_u64(5.0)

        # abs on s1: -10 / 2 = -5.0
        r = ref_div_fixup_f64(s0, s1, s2, src1_abs=True)
        self.assertEqual(u64_to_f64(r), -5.0)

        # neg on s0: s0 sign is ignored in magnitude, sign_out dictates final sign
        r2 = ref_div_fixup_f64(s0, s1, s2, src0_neg=True)
        self.assertEqual(u64_to_f64(r2), 5.0)

    def test_clamp_modifier(self) -> None:
        s1 = f64_to_u64(2.0)
        s2 = f64_to_u64(1.0)
        s0 = f64_to_u64(0.5)

        # 0.5 in [0.0, 1.0] -> 0.5
        r = ref_div_fixup_f64(s0, s1, s2, clamp=True)
        self.assertEqual(u64_to_f64(r), 0.5)

        # Negative clamped to +0.0
        r_neg = ref_div_fixup_f64(s0, s1, f64_to_u64(-1.0), clamp=True)
        self.assertEqual(r_neg, F64_POS_ZERO)

        # > 1.0 clamped to 1.0
        r_large = ref_div_fixup_f64(f64_to_u64(5.0), s1, f64_to_u64(10.0), clamp=True)
        self.assertEqual(r_large, F64_ONE)

        # NaN clamped to +0.0
        r_nan = ref_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_POS_ZERO, clamp=True)
        self.assertEqual(r_nan, F64_POS_ZERO)

    def test_dword_decomposition(self) -> None:
        s0 = f64_to_u64(5.0)
        s1 = f64_to_u64(2.0)
        s2 = f64_to_u64(10.0)

        s0_lo, s0_hi = split_dwords(s0)
        s1_lo, s1_hi = split_dwords(s1)
        s2_lo, s2_hi = split_dwords(s2)

        r_lo, r_hi = ref_div_fixup_f64_dwords(s0_lo, s0_hi, s1_lo, s1_hi, s2_lo, s2_hi)
        res = join_dwords(r_lo, r_hi)
        self.assertEqual(u64_to_f64(res), 5.0)

    def test_register_file_pair_access(self) -> None:
        rf = VALURegisterFile(num_vgprs=16)
        val = 0x01234567_89ABCDEF
        rf.set_vgpr_pair(4, val)

        self.assertEqual(rf.get_vgpr(4), 0x89ABCDEF)
        self.assertEqual(rf.get_vgpr(5), 0x01234567)
        self.assertEqual(rf.get_vgpr_pair(4), val)

        with self.assertRaises(IndexError):
            rf.get_vgpr_pair(15)


if __name__ == "__main__":
    unittest.main()
