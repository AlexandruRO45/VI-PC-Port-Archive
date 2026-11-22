from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_pknorm_u16_f16.

Models IEEE 754-2008 binary16 half-precision floating-point to unsigned normalized 16-bit
integer (UNORM16) conversions and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU)
architectural semantics:
  v_cvt_pknorm_u16_f16 D, S0, S1 [op_sel:[s0, s1]]

Lowered Architectural Semantics:
  D[15:0]  = f16_to_unorm16(select_half(S0, op_sel[0]))
  D[31:16] = f16_to_unorm16(select_half(S1, op_sel[1]))

RDNA2 VALU Specification Rules:
  1. Packing & Destination:
     - Destination is a single 32-bit VGPR (v[d]).
     - Overwrites all 32 bits of the destination register atomically.
     - D[15:0] is derived exclusively from S0; D[31:16] is derived exclusively from S1.
  2. Operand Selection (op_sel):
     - op_sel[0] selects the low 16 bits (0) or high 16 bits (1) of S0.
     - op_sel[1] selects the low 16 bits (0) or high 16 bits (1) of S1.
  3. Conversion Mathematics:
     - Input S0 and S1 half-words are IEEE 754-2008 binary16 floats.
     - Clamped to [0.0, 1.0]:
         * Values <= 0.0 (including -0.0 and negative numbers) clamp to 0.0 -> 0x0000.
         * Values >= 1.0 clamp to 1.0 -> 0xFFFF (65535).
         * +Infinity clamps to 1.0 -> 0xFFFF.
         * -Infinity clamps to 0.0 -> 0x0000.
     - Normalized scaling:
         * Product = clamped * 65535.0
         * Rounded to nearest integer with tie-breaking to even (IEEE 754 roundTiesToEven).
  4. NaN Semantics:
     - Any NaN input (Quiet NaN or Signaling NaN, positive or negative sign, any payload)
       evaluates to 0x0000 per AMD VALU hardware specifications.
  5. VOP3 Source Modifiers:
     - abs (|x|): Clears sign bit (bit 15) of f16 input before conversion.
     - neg (-x): Inverts sign bit (bit 15) of f16 input before conversion.
     - neg(abs(x)): Inverts sign bit after abs, forcing negative sign -> clamps to 0x0000.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""

import math
import struct
import unittest
from typing import Tuple

# Bitwise Masks & Architectural Constants
MASK32: int = 0xFFFFFFFF
MASK16: int = 0xFFFF
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
MANT_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200
CANONICAL_QNAN_F16: int = 0x7E00
CANONICAL_SNAN_F16: int = 0x7C01

UNORM16_MIN: int = 0x0000
UNORM16_MAX: int = 0xFFFF
SCALE_U16: float = 65535.0

# IEEE 754 binary16 Bit Patterns
POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_ONE_F16: int = 0x3C00              # +1.0f16
NEG_ONE_F16: int = 0xBC00              # -1.0f16
POS_HALF_F16: int = 0x3800             # +0.5f16
NEG_HALF_F16: int = 0xB800             # -0.5f16
POS_TWO_F16: int = 0x4000              # +2.0f16
NEG_TWO_F16: int = 0xC000              # -2.0f16
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
MIN_SUBNORMAL_F16: int = 0x0001        # 2^-24
MAX_SUBNORMAL_F16: int = 0x03FF
MIN_NORMAL_F16: int = 0x0400           # 2^-14
MAX_NORMAL_F16: int = 0x7BFF           # 65504.0


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit range [0, 65535]."""
    return val & MASK16


def f16_to_bits(val: float) -> int:
    """Encode float to IEEE 754 half-precision (binary16) unsigned 16-bit integer."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Decode 16-bit unsigned integer to IEEE 754 half-precision float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def is_nan_f16(bits: int) -> bool:
    """True if 16-bit word represents an IEEE 754 half-precision NaN."""
    bits &= MASK16
    return (bits & EXP_MASK_F16) == EXP_MASK_F16 and (bits & MANT_MASK_F16) != 0


def is_qnan_f16(bits: int) -> bool:
    """True if bits represent a quiet half-precision NaN (bit 9 is 1)."""
    bits &= MASK16
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """True if bits represent a signaling half-precision NaN (bit 9 is 0)."""
    bits &= MASK16
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """True if bits represent positive or negative infinity."""
    bits &= MASK16
    return (bits & EXP_MASK_F16) == EXP_MASK_F16 and (bits & MANT_MASK_F16) == 0


def is_subnormal_f16(bits: int) -> bool:
    """True if bits represent an IEEE 754 half-precision subnormal number."""
    bits &= MASK16
    return (bits & EXP_MASK_F16) == 0 and (bits & MANT_MASK_F16) != 0


def apply_f16_modifiers(bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Applies VOP3 input modifiers to 16-bit float: abs (|x|) first, then neg (-x)."""
    bits &= MASK16
    if abs_mod:
        bits &= ~SIGN_MASK_F16
    if neg_mod:
        bits ^= SIGN_MASK_F16
    return bits & MASK16


def select_f16_half(src32: int, sel_high: int) -> int:
    """Select low (0) or high (1) 16-bit half of a 32-bit register."""
    src = src32 & MASK32
    return ((src >> 16) & MASK16) if sel_high else (src & MASK16)


def round_ties_to_even(val: float) -> int:
    """Round floating-point value to nearest integer using IEEE 754 roundTiesToEven."""
    return int(round(val))


def f16_to_unorm16(bits16: int) -> int:
    """Convert an IEEE 754 binary16 float to UNORM16 uint16.

    Semantics:
      - NaN converts to 0 (0x0000).
      - Clamped to [0.0, 1.0].
      - Scaled by 65535.0.
      - Rounded with roundTiesToEven.
      - Representable range: [0, 65535].
    """
    bits16 &= MASK16
    if is_nan_f16(bits16):
        return 0

    val = bits_to_f16(bits16)
    if math.isinf(val):
        val = 1.0 if val > 0 else 0.0
    else:
        val = max(0.0, min(1.0, val))

    scaled = val * SCALE_U16
    return round_ties_to_even(scaled) & MASK16


def unpack_pknorm_u16(packed_b32: int) -> Tuple[int, int]:
    """Unpack 32-bit packed word into (ch0_u16, ch1_u16) unsigned integers."""
    packed = packed_b32 & MASK32
    ch0 = packed & MASK16
    ch1 = (packed >> 16) & MASK16
    return ch0, ch1


def unorm16_to_f16(val16: int) -> float:
    """Convert unsigned 16-bit normalized integer back to float in range [0.0, 1.0]."""
    u_val = val16 & MASK16
    return max(0.0, min(1.0, u_val / SCALE_U16))


def oracle_v_cvt_pknorm_u16_f16(
    src0_bits: int,
    src1_bits: int,
    op_sel: Tuple[int, int] = (0, 0),
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_pknorm_u16_f16.

    Packs two half-precision float sources into a single 32-bit destination:
      dst[15:0]  = unorm16(src0_half)
      dst[31:16] = unorm16(src1_half)
    """
    h0 = select_f16_half(src0_bits, op_sel[0])
    h1 = select_f16_half(src1_bits, op_sel[1])

    h0_mod = apply_f16_modifiers(h0, abs_mod=abs0, neg_mod=neg0)
    h1_mod = apply_f16_modifiers(h1, abs_mod=abs1, neg_mod=neg1)

    u0 = f16_to_unorm16(h0_mod)
    u1 = f16_to_unorm16(h1_mod)

    return ((u1 & MASK16) << 16) | (u0 & MASK16)


# ==============================================================================
# Commit 1: Reference Mathematical Oracle and Architectural Model Tests
# ==============================================================================

class TestValuCvtPknormU16F16ModelAndOracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_normalized_boundaries(self) -> None:
        """Verify conversions at normalized boundaries: +1.0, +0.0, -0.0, -1.0."""
        # +1.0 in both channels -> 0xFFFFFFFF
        res = oracle_v_cvt_pknorm_u16_f16(POS_ONE_F16, POS_ONE_F16)
        self.assertEqual(res, 0xFFFFFFFF)
        ch0, ch1 = unpack_pknorm_u16(res)
        self.assertEqual(ch0, 65535)
        self.assertEqual(ch1, 65535)

        # 0.0 in both channels -> 0x00000000
        res = oracle_v_cvt_pknorm_u16_f16(POS_ZERO_F16, NEG_ZERO_F16)
        self.assertEqual(res, 0x00000000)
        ch0, ch1 = unpack_pknorm_u16(res)
        self.assertEqual(ch0, 0)
        self.assertEqual(ch1, 0)

        # Negative values clamp to 0x0000
        res = oracle_v_cvt_pknorm_u16_f16(NEG_ONE_F16, NEG_ONE_F16)
        self.assertEqual(res, 0x00000000)
        ch0, ch1 = unpack_pknorm_u16(res)
        self.assertEqual(ch0, 0)
        self.assertEqual(ch1, 0)

        # Asymmetric pairing: (+1.0, 0.0) -> 0x0000FFFF
        res = oracle_v_cvt_pknorm_u16_f16(POS_ONE_F16, POS_ZERO_F16)
        self.assertEqual(res, 0x0000FFFF)
        ch0, ch1 = unpack_pknorm_u16(res)
        self.assertEqual(ch0, 65535)
        self.assertEqual(ch1, 0)

    def test_fractional_values(self) -> None:
        """Verify conversions for fractional values: +0.5, +0.25, +0.75."""
        # +0.5: 0.5 * 65535.0 = 32767.5 -> roundTiesToEven -> 32768 (0x8000)
        res = oracle_v_cvt_pknorm_u16_f16(POS_HALF_F16, POS_HALF_F16)
        self.assertEqual(res, 0x80008000)
        ch0, ch1 = unpack_pknorm_u16(res)
        self.assertEqual(ch0, 32768)
        self.assertEqual(ch1, 32768)

        # +0.25: 0.25 * 65535.0 = 16383.75 -> 16384 (0x4000)
        q25 = f16_to_bits(0.25)
        res_q = oracle_v_cvt_pknorm_u16_f16(q25, q25)
        self.assertEqual(res_q, 0x40004000)

        # +0.75: 0.75 * 65535.0 = 49151.25 -> 49151 (0xBFFF)
        q75 = f16_to_bits(0.75)
        res_75 = oracle_v_cvt_pknorm_u16_f16(q75, q75)
        self.assertEqual(res_75, 0xBFFFBFFF)

    def test_round_ties_to_even(self) -> None:
        """Verify exact tie-breaking to nearest even integer."""
        self.assertEqual(round_ties_to_even(0.5), 0)
        self.assertEqual(round_ties_to_even(1.5), 2)
        self.assertEqual(round_ties_to_even(2.5), 2)
        self.assertEqual(round_ties_to_even(3.5), 4)

        # In f16, 0.5 is exact: 0.5 * 65535.0 = 32767.5 -> nearest even is 32768 (0x8000)
        self.assertEqual(f16_to_unorm16(POS_HALF_F16), 32768)

    def test_out_of_range_clamping(self) -> None:
        """Verify inputs outside [0.0, 1.0] are clamped to boundaries."""
        # +2.0 clamps to 1.0 -> 0xFFFF
        res_pos = oracle_v_cvt_pknorm_u16_f16(POS_TWO_F16, f16_to_bits(10.0))
        self.assertEqual(res_pos, 0xFFFFFFFF)

        # Negative numbers clamp to 0.0 -> 0x0000
        res_neg = oracle_v_cvt_pknorm_u16_f16(NEG_TWO_F16, f16_to_bits(-10.0))
        self.assertEqual(res_neg, 0x00000000)

    def test_nan_and_inf_handling(self) -> None:
        """Verify NaN converts to 0x0000 and infinities clamp to boundaries."""
        # NaNs convert to 0x0000
        res_nan = oracle_v_cvt_pknorm_u16_f16(CANONICAL_QNAN_F16, CANONICAL_SNAN_F16)
        self.assertEqual(res_nan, 0x00000000)

        # +inf clamps to 0xFFFF, -inf clamps to 0x0000
        res_inf = oracle_v_cvt_pknorm_u16_f16(POS_INF_F16, NEG_INF_F16)
        self.assertEqual(res_inf, 0x0000FFFF)

    def test_op_sel_operand_selection(self) -> None:
        """Verify op_sel selects low or high 16-bit half from 32-bit dwords."""
        # reg0: low = +1.0 (0x3C00), high = 0.0 (0x0000) -> 0x00003C00
        reg0 = (POS_ZERO_F16 << 16) | POS_ONE_F16
        # reg1: low = 0.0 (0x0000), high = +0.5 (0x3800) -> 0x38000000
        reg1 = (POS_HALF_F16 << 16) | POS_ZERO_F16

        # op_sel=[0, 0]: select low from reg0 (+1.0) and low from reg1 (0.0) -> 0x0000FFFF
        res00 = oracle_v_cvt_pknorm_u16_f16(reg0, reg1, op_sel=(0, 0))
        self.assertEqual(res00, 0x0000FFFF)

        # op_sel=[1, 1]: select high from reg0 (0.0) and high from reg1 (+0.5) -> 0x80000000
        res11 = oracle_v_cvt_pknorm_u16_f16(reg0, reg1, op_sel=(1, 1))
        self.assertEqual(res11, 0x80000000)

        # op_sel=[0, 1]: select low from reg0 (+1.0) and high from reg1 (+0.5) -> 0x8000FFFF
        res01 = oracle_v_cvt_pknorm_u16_f16(reg0, reg1, op_sel=(0, 1))
        self.assertEqual(res01, 0x8000FFFF)

    def test_input_modifiers(self) -> None:
        """Verify abs and neg modifiers applied per channel."""
        # abs on negative input: abs(-1.0) = +1.0 -> 0xFFFF
        res_abs = oracle_v_cvt_pknorm_u16_f16(NEG_ONE_F16, NEG_HALF_F16, abs0=True, abs1=True)
        self.assertEqual(res_abs, 0x8000FFFF)

        # neg on positive input: neg(+1.0) = -1.0 -> clamps to 0x0000
        res_neg = oracle_v_cvt_pknorm_u16_f16(POS_ONE_F16, POS_HALF_F16, neg0=True, neg1=True)
        self.assertEqual(res_neg, 0x00000000)

        # abs+neg guarantees negative -> clamps to 0x0000
        res_absneg = oracle_v_cvt_pknorm_u16_f16(NEG_ONE_F16, POS_HALF_F16, abs0=True, neg0=True, abs1=True, neg1=True)
        self.assertEqual(res_absneg, 0x00000000)

    def test_unpack_and_roundtrip(self) -> None:
        """Verify unpacking and precision round-trip within 1 ULP of UNORM16."""
        for val in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
            f_bits = f16_to_bits(val)
            packed = oracle_v_cvt_pknorm_u16_f16(f_bits, f_bits)
            ch0, ch1 = unpack_pknorm_u16(packed)
            self.assertEqual(ch0, ch1)
            reconstructed = unorm16_to_f16(ch0)
            self.assertAlmostEqual(val, reconstructed, delta=1.0 / 65535.0)


if __name__ == "__main__":
    unittest.main()
