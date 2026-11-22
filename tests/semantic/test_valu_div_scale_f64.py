from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_div_scale_f64.

Tests the double-precision floating-point division pre-scaling instruction:
    v_div_scale_f64 D, sdst, S0, S1, S2

Hardware Specification (RDNA2 / GFX10 / GFX1030 VOP3b dual-output):
    D    = 64-bit vector destination register pair v[d:d+1] receiving scaled S0
    sdst = scalar destination (VCC or SGPR pair/single) receiving scale condition bit mask
    S0   = operand to scale (64-bit float, typically S1 or S2)
    S1   = denominator (64-bit float)
    S2   = numerator (64-bit float)

Operation:
    Given numerator S2 and denominator S1, v_div_scale_f64 determines whether intermediate
    Newton-Raphson division approximation will underflow or overflow, and conditionally scales
    S0 by 2^128 or 2^-128 while recording whether scaling took place in sdst (VCC).
    
    Based on measured RDNA2 hardware semantics (docs/design.md):
    With e1, e2 the exponent fields of S1 (denom) and S2 (num), the first matching rule wins:
      1. e2 - e1 >= 768:
         sdst = 1, D = S0 * 2^128 if S0 == S1 else S0
      2. e1 == 0 (zero or denormal denominator):
         sdst = 0, D = S0 * 2^128
      3. e1 >= 2045 and e1 - e2 >= 768:
         sdst = 1, D = S0 * 2^-128 if S0 == S1 else S0
      4. e1 >= 2045:
         sdst = 0, D = S0 * 2^-128
      5. e1 - e2 >= 768:
         sdst = 1, D = S0 if S0 == S1 else S0 * 2^128
      6. e2 <= 53:
         sdst = 0, D = S0 * 2^128
      7. otherwise:
         sdst = 0, D = S0

    Zero behavior:
      A zero S1 or S2 makes D = 0xFFF8000000000000 (NaN), while sdst retains its table bit.
    
    Equality:
      S0 == S1 compares 64-bit values after neg modifier is applied.
    
    Modifiers:
      - neg(-src): inverts sign bit of S0, S1, S2.
      - clamp: maps NaNs, zeros, and negative values to +0.0; values >= 1.0 to 1.0.
      - omod: accepted by assembler and ignored by hardware.

Architectural model:
    64-bit floating point registers mapped onto aligned VGPR/SGPR pairs:
    [lo: 32-bit dword, hi: 32-bit dword].
"""

import math
import struct
import unittest

# IEEE 754 double precision bitwise constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_ONE = 0x3FF0000000000000
F64_NEG_ONE = 0xBFF0000000000000
F64_TWO = 0x4000000000000000
F64_HALF = 0x3FE0000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_CANONICAL_QNAN = 0x7FF8000000000000
F64_SNAN = 0x7FF0000000000001
F64_SUB_MIN = 0x0000000000000001
F64_SUB_MAX = 0x000FFFFFFFFFFFFF
F64_MIN_NORMAL = 0x0010000000000000
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def f64_to_u64(val: float) -> int:
    """Pack Python float (IEEE 754 binary64) into unsigned 64-bit integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def u64_to_f64(bits: int) -> float:
    """Unpack unsigned 64-bit integer into Python float (IEEE 754 binary64)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def u64_to_dwords(val: int) -> tuple[int, int]:
    """Split 64-bit word into (low 32-bit dword, high 32-bit dword)."""
    return val & MASK32, (val >> 32) & MASK32


def dwords_to_u64(lo: int, hi: int) -> int:
    """Combine (low 32-bit dword, high 32-bit dword) into 64-bit word."""
    return (lo & MASK32) | ((hi & MASK32) << 32)


def apply_src_mods_u64(val_u64: int, *, neg: bool = False, abs_: bool = False) -> int:
    """Apply bit-exact source modifiers (|src|, -src) to a 64-bit float bit pattern."""
    bits = val_u64 & MASK64
    if abs_:
        bits &= ~F64_NEG_ZERO
    if neg:
        bits ^= F64_NEG_ZERO
    return bits


def apply_clamp_f64(val_u64: int) -> int:
    """Apply RDNA2 clamp modifier to double precision result."""
    x = u64_to_f64(val_u64)
    if x > 0.0:
        return f64_to_u64(min(x, 1.0))
    return 0


def apply_omod_f64(val_u64: int, omod: int) -> int:
    """Apply RDNA2 omod modifier (measured: ignored on hardware for v_div_scale_f64)."""
    return val_u64


def ex_f64(b: int) -> int:
    """Extract 11-bit IEEE 754 exponent field from 64-bit float pattern."""
    return (b >> 52) & 0x7FF


def scale_f64_bits(b: int, n: int) -> int:
    """Scale 64-bit float bit pattern by 2^n per RDNA2 hardware semantics."""
    if ex_f64(b) == 0x7FF or not (b & ~(1 << 63)):
        return b  # NaN (payload preserved), inf, zero
    if n < 0 and ex_f64(b) <= 116:
        return (b & (1 << 63)) | 0x7FF0000000000000  # measured: inf with sign of S0
    x = u64_to_f64(b)
    try:
        res = math.ldexp(x, n)
        return f64_to_u64(res)
    except OverflowError:
        return (b & (1 << 63)) | 0x7FF0000000000000


def ref_div_scale_f64_u64(
    u0: int,
    u1: int,
    u2: int,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> tuple[int, int]:
    """Bit-exact reference mathematical oracle for v_div_scale_f64 using 64-bit integer words.

    Returns (d_u64, sdst_bit).
    """
    s0 = (u0 & MASK64) ^ (F64_NEG_ZERO if neg0 else 0)
    s1 = (u1 & MASK64) ^ (F64_NEG_ZERO if neg1 else 0)
    s2 = (u2 & MASK64) ^ (F64_NEG_ZERO if neg2 else 0)

    e1, e2 = ex_f64(s1), ex_f64(s2)
    vcc, d = 0, s0

    if e2 - e1 >= 768:
        vcc, d = 1, scale_f64_bits(s0, 128) if s0 == s1 else s0
    elif e1 == 0:
        d = scale_f64_bits(s0, 128)
    elif e1 >= 2045 and e1 - e2 >= 768:
        vcc, d = 1, scale_f64_bits(s0, -128) if s0 == s1 else s0
    elif e1 >= 2045:
        d = scale_f64_bits(s0, -128)
    elif e1 - e2 >= 768:
        vcc, d = 1, s0 if s0 == s1 else scale_f64_bits(s0, 128)
    elif e2 <= 53:
        d = scale_f64_bits(s0, 128)

    # A zero S1 or S2 makes D = 0xFFF8000000000000 (NaN), while sdst retains its table bit
    if not (s1 & ~(1 << 63)) or not (s2 & ~(1 << 63)):
        d = 0xFFF8000000000000

    if clamp:
        d = apply_clamp_f64(d)

    return d, vcc


def ref_div_scale_f64(
    s0: float,
    s1: float,
    s2: float,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> tuple[float, int]:
    """Float wrapper for ref_div_scale_f64_u64. Returns (d_float, sdst_bit)."""
    d_u64, vcc = ref_div_scale_f64_u64(
        f64_to_u64(s0),
        f64_to_u64(s1),
        f64_to_u64(s2),
        neg0=neg0,
        neg1=neg1,
        neg2=neg2,
        clamp=clamp,
        omod=omod,
    )
    return u64_to_f64(d_u64), vcc


class RegisterPair:
    """Architectural model of an aligned 64-bit register pair (e.g. v[0:1] or s[0:1])."""

    def __init__(self, lo: int = 0, hi: int = 0) -> None:
        self.lo = lo & MASK32
        self.hi = hi & MASK32

    @classmethod
    def from_u64(cls, val: int) -> RegisterPair:
        lo, hi = u64_to_dwords(val)
        return cls(lo, hi)

    @classmethod
    def from_f64(cls, val: float) -> RegisterPair:
        return cls.from_u64(f64_to_u64(val))

    @property
    def as_u64(self) -> int:
        return dwords_to_u64(self.lo, self.hi)

    @property
    def as_f64(self) -> float:
        return u64_to_f64(self.as_u64)

    def set_u64(self, val: int) -> None:
        self.lo, self.hi = u64_to_dwords(val)

    def set_f64(self, val: float) -> None:
        self.set_u64(f64_to_u64(val))

    def __repr__(self) -> str:
        return f"RegisterPair(lo={self.lo:#010x}, hi={self.hi:#010x}, f64={self.as_f64})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisterPair):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi


class TestDivScaleF64MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_basic_normal_no_scale(self):
        # 1.0 / 2.0 with s0 = 2.0: exponents 1023 and 1024, no scaling needed
        d, vcc = ref_div_scale_f64(2.0, 2.0, 1.0)
        self.assertEqual(vcc, 0)
        self.assertEqual(d, 2.0)

    def test_rdna2_measured_cases(self):
        # Hardware measured 13 cases from gfx1036 (docs/design.md)
        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000
        tiny = 0x0DF0000000000003

        cases = [
            (one5, one5, big, False),
            (big, one5, big, False),
            (tiny, one5, tiny, False),
            (one5, one5, tiny, False),
            (0x7FE0000000000000, 0x7FE0000000000000, 0x3FF0000000000000, False),
            (0x7F80000000000009, 0x7FE0000000000000, 0x7F80000000000009, False),
            (7, 7, 0x2580000000000000, False),
            (0x0140000000000001, 0x0010000000000000, 0x0140000000000001, False),
            (0, 0, 0x3FF0000000000000, False),
            (0x7FF0000000000001, 0x3FF0000000000000, 0x7FF0000000000001, False),
            (one5 | (1 << 63), one5, big, False),
            (0, 0, 0x3FF0000000000000, True),
            (0x3FD0000000000000, 0x3FD0000000000000, 0x3FE0000000000000, False),
        ]
        want = [
            0x00000005, 0x47F00000, 1,
            0x00000000, 0x71F00000, 1,
            0x00000003, 0x15F00000, 1,
            0x00000005, 0x3FF00000, 1,
            0x00000000, 0x77E00000, 1,
            0x00000009, 0x77800000, 0,
            0x00000000, 0x04FC0000, 0,
            0x00000001, 0x09400000, 0,
            0x00000000, 0xFFF80000, 1,
            0x00000001, 0x7FF00000, 1,
            0x00000005, 0xBFF00000, 1,
            0x00000000, 0x00000000, 1,
            0x00000000, 0x3FD00000, 0,
        ]

        got = []
        for s0, s1, s2, clamp in cases:
            d_u, vcc = ref_div_scale_f64_u64(s0, s1, s2, clamp=clamp)
            got.extend([d_u & MASK32, (d_u >> 32) & MASK32, vcc])

        self.assertEqual(got, want)

    def test_source_modifiers(self):
        # neg modifier flips sign bit
        one5 = 0x3FF0000000000005
        big = 0x71F0000000000000

        # When both s0 and s1 have neg applied, s0 == s1, so both scale and have inverted sign
        d_pos, vcc_p = ref_div_scale_f64_u64(one5, one5, big)
        d_neg, vcc_n = ref_div_scale_f64_u64(one5, one5, big, neg0=True, neg1=True)
        self.assertEqual(vcc_p, 1)
        self.assertEqual(vcc_n, 1)
        self.assertEqual(d_pos ^ (1 << 63), d_neg)

        # Equality S0 == S1 is checked after neg
        # When only s0 has neg, s0 != s1, so case 1 (e2 - e1 >= 768) does not scale s0
        d_noteq, vcc_noteq = ref_div_scale_f64_u64(one5, one5, big, neg0=True)
        self.assertEqual(vcc_noteq, 1)
        self.assertEqual(d_noteq, one5 ^ (1 << 63))  # unscaled s0 with negative sign

    def test_output_modifiers_clamp(self):
        # clamp saturates to [0.0, 1.0], NaNs and negatives map to +0.0
        d_pos, _ = ref_div_scale_f64_u64(F64_TWO, F64_ONE, F64_ONE, clamp=True)
        self.assertEqual(d_pos, F64_ONE)

        d_neg, _ = ref_div_scale_f64_u64(F64_NEG_ONE, F64_ONE, F64_ONE, clamp=True)
        self.assertEqual(d_neg, F64_POS_ZERO)

        # Zero denominator produces NaN which clamps to +0.0
        # For s1 = +0.0, s2 = 1.0, e2 - e1 >= 768 is met, so vcc is 1
        d_zero_denom, vcc_z = ref_div_scale_f64_u64(F64_ONE, F64_POS_ZERO, F64_ONE, clamp=True)
        self.assertEqual(d_zero_denom, F64_POS_ZERO)
        self.assertEqual(vcc_z, 1)

    def test_register_pair_model(self):
        rp = RegisterPair.from_f64(1.5)
        self.assertEqual(rp.as_f64, 1.5)
        self.assertEqual(rp.lo, 0x00000000)
        self.assertEqual(rp.hi, 0x3FF80000)

        rp.set_u64(F64_NEG_ONE)
        self.assertEqual(rp.as_f64, -1.0)
        self.assertEqual(rp.hi, 0xBFF00000)


if __name__ == "__main__":
    unittest.main()
