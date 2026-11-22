from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_u32_f16.

Instruction:
  v_cvt_norm_u32_f16 dst, src0 (VOP1 / VOP3)
  D.u32 = f16_to_unorm(S0.f16)

Architecture:
  - RDNA2 Vector ALU (GFX10) half-precision float to unsigned normalized 32-bit integer conversion.
  - S0: 16-bit IEEE 754 binary16 float (from VGPR halfword, SGPR halfword, or literal/inline const).
  - D:  32-bit unsigned normalized integer in [0, 4294967295] (full 32-bit VGPR).
  - Normalization:
      * Maps floating-point range [0.0, 1.0] to unsigned integer range [0, 4294967295].
      * Scaling factor is 4294967295.0 (2^32 - 1).
      * Rounding mode: round-to-nearest-even (IEEE 754-2008 roundTiesToEven).
      * Clamping: values <= 0.0 (or -inf) clamp to 0 (0x00000000).
                  values >= 1.0 (or +inf) clamp to 4294967295 (0xFFFFFFFF).
      * NaNs (both QNaN and SNaN) convert to 0 (0x00000000).
      * Signed zeros (+0.0, -0.0) convert to 0 (0x00000000).
  - Register model:
      * 32-bit vector registers (VGPRs: v0-v255) and scalar registers (SGPRs: s0-s105).
      * Overwrites the full 32-bit destination register.
      * Source can read either low half (.l [15:0]) or high half (.h [31:16]) via VOP3 op_sel.
  - Modifiers:
      * Input modifiers: abs (|S0|), neg (-S0).
      * Order of application: abs clears sign bit first, then neg inverts sign bit.
"""

from fractions import Fraction
import math
import struct
import unittest

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF

# IEEE 754 binary16 architectural bit layout
SIGN_MASK_F16: int = 0x8000
EXP_MASK_F16: int = 0x7C00
MANT_MASK_F16: int = 0x03FF
QNAN_BIT_F16: int = 0x0200
BIAS_F16: int = 15

# Special half-precision bit patterns
POS_ZERO_F16: int = 0x0000
NEG_ZERO_F16: int = 0x8000
POS_INF_F16: int = 0x7C00
NEG_INF_F16: int = 0xFC00
QNAN_F16: int = 0x7E00
SNAN_F16: int = 0x7C01

# Standard reference floating-point constants in binary16
ONE_F16: int = 0x3C00            # 1.0
NEG_ONE_F16: int = 0xBC00        # -1.0
TWO_F16: int = 0x4000            # 2.0
NEG_TWO_F16: int = 0xC000        # -2.0
HALF_F16: int = 0x3800           # 0.5
NEG_HALF_F16: int = 0xB800       # -0.5
QUARTER_F16: int = 0x3400        # 0.25
NEG_QUARTER_F16: int = 0xB400    # -0.25
THREE_QUARTERS_F16: int = 0x3A00 # 0.75
FOUR_F16: int = 0x4400           # 4.0
NEG_FOUR_F16: int = 0xC400       # -4.0

MAX_FINITE_F16: int = 0x7BFF     # 65504.0
MIN_FINITE_NEG_F16: int = 0xFBFF # -65504.0
MIN_NORMAL_F16: int = 0x0400     # 2^-14 (~6.103515625e-5)
MAX_SUBNORMAL_F16: int = 0x03FF  # ~6.09755516e-5
MIN_SUBNORMAL_F16: int = 0x0001  # 2^-24 (~5.9604644775e-8)

# 32-bit unsigned normalized integer range constants
UNORM32_MAX: int = 4294967295    # 0xFFFFFFFF (+1.0)
UNORM32_MIN: int = 0             # 0x00000000 (0.0)
UNORM32_SCALE: float = 4294967295.0


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit halfword."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def bits_to_f16(bits: int) -> float:
    """Convert unsigned 16-bit bits into Python float via struct IEEE 754 half."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f16_to_bits(val: float) -> int:
    """Convert Python float into unsigned 16-bit IEEE 754 half bits."""
    if math.isnan(val):
        return QNAN_F16
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return NEG_INF_F16 if val < 0.0 else POS_INF_F16


def is_nan_f16(bits: int) -> bool:
    """Check if binary16 bits represent any NaN."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) != 0)


def is_qnan_f16(bits: int) -> bool:
    """Check if binary16 bits represent a Quiet NaN."""
    return is_nan_f16(bits) and bool(bits & QNAN_BIT_F16)


def is_snan_f16(bits: int) -> bool:
    """Check if binary16 bits represent a Signaling NaN."""
    return is_nan_f16(bits) and not bool(bits & QNAN_BIT_F16)


def is_inf_f16(bits: int) -> bool:
    """Check if binary16 bits represent +/- infinity."""
    return ((bits & EXP_MASK_F16) == EXP_MASK_F16) and ((bits & MANT_MASK_F16) == 0)


def is_subnormal_f16(bits: int) -> bool:
    """Check if binary16 bits represent a subnormal number."""
    return ((bits & EXP_MASK_F16) == 0) and ((bits & MANT_MASK_F16) != 0)


def is_zero_f16(bits: int) -> bool:
    """Check if binary16 bits represent +0.0 or -0.0."""
    return (bits & ~SIGN_MASK_F16) == 0


def apply_f16_modifiers(bits16: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply RDNA2 source modifiers (|src|, -src) to a 16-bit half float word.
    
    Order of operations: abs clears sign bit (bit 15) first, then neg inverts sign bit (bit 15).
    """
    val = bits16 & MASK16
    if abs_mod:
        val &= ~SIGN_MASK_F16
    if neg_mod:
        val ^= SIGN_MASK_F16
    return val


def f16_to_unorm32(src0_bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 f16_to_unorm32 conversion.
    
    Converts 16-bit float bits to unsigned normalized 32-bit integer bits.
    - NaNs convert to 0.
    - Negative values (<= 0.0) clamp to 0.
    - Values >= 1.0 (and +inf) clamp to 4294967295.
    - Scaled by 4294967295.0 and rounded with round-to-nearest-even.
    Returns unsigned 32-bit integer.
    """
    val_bits = apply_f16_modifiers(src0_bits, abs_mod, neg_mod)
    if is_nan_f16(val_bits):
        return 0
    if is_zero_f16(val_bits):
        return 0
    if is_inf_f16(val_bits):
        return 0 if (val_bits & SIGN_MASK_F16) else UNORM32_MAX

    val_float = bits_to_f16(val_bits)
    if val_float <= 0.0:
        return 0
    if val_float >= 1.0:
        return UNORM32_MAX

    scaled = val_float * UNORM32_SCALE
    rounded = int(round(scaled))
    if rounded > UNORM32_MAX:
        rounded = UNORM32_MAX
    elif rounded < 0:
        rounded = 0
    return u32(rounded)


def oracle_v_cvt_norm_u32_f16(
    src0_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
) -> int:
    """Bit-exact reference oracle for v_cvt_norm_u32_f16."""
    return f16_to_unorm32(src0_bits, abs_mod, neg_mod)


class GpuRegisterState:
    """Architectural register file state model for RDNA2 VALU execution."""

    def __init__(self) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs[idx & 0xFF] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx & 0xFF] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs[idx % 106] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx % 106] = val & MASK32


class TestValuCvtNormU32F16Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_oracle_zeros(self) -> None:
        self.assertEqual(oracle_v_cvt_norm_u32_f16(POS_ZERO_F16), 0)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_ZERO_F16), 0)

    def test_oracle_unit_boundaries(self) -> None:
        # +1.0 -> 4294967295 (0xFFFFFFFF)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(ONE_F16), 4294967295)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(ONE_F16), 0xFFFFFFFF)
        # -1.0 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_ONE_F16), 0)

    def test_oracle_half_values(self) -> None:
        # +0.5 * 4294967295 = 2147483647.5 -> rounds to 2147483648 (even) = 0x80000000
        self.assertEqual(oracle_v_cvt_norm_u32_f16(HALF_F16), 2147483648)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(HALF_F16), 0x80000000)
        # +0.25 * 4294967295 = 1073741823.75 -> rounds to 1073741824 = 0x40000000
        self.assertEqual(oracle_v_cvt_norm_u32_f16(QUARTER_F16), 1073741824)
        # -0.5 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_HALF_F16), 0)

    def test_oracle_clamping_extremes(self) -> None:
        # +2.0 clamps to +1.0 -> 4294967295
        self.assertEqual(oracle_v_cvt_norm_u32_f16(TWO_F16), 4294967295)
        # -2.0 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_TWO_F16), 0)
        # +4.0 clamps to 4294967295, -4.0 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(FOUR_F16), 4294967295)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_FOUR_F16), 0)
        # Infinities clamp to extremes
        self.assertEqual(oracle_v_cvt_norm_u32_f16(POS_INF_F16), 4294967295)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_INF_F16), 0)

    def test_oracle_nans(self) -> None:
        # Both QNaN and SNaN convert to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(QNAN_F16), 0)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(SNAN_F16), 0)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(0x7E01), 0)
        self.assertEqual(oracle_v_cvt_norm_u32_f16(0xFE00), 0)

    def test_oracle_modifiers(self) -> None:
        # abs(-1.0) = 1.0 -> 4294967295
        self.assertEqual(oracle_v_cvt_norm_u32_f16(NEG_ONE_F16, abs_mod=True), 4294967295)
        # neg(1.0) = -1.0 -> clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(ONE_F16, neg_mod=True), 0)
        # abs and neg: -|1.0| = -1.0 -> clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u32_f16(ONE_F16, abs_mod=True, neg_mod=True), 0)

    def test_architectural_register_state(self) -> None:
        state = GpuRegisterState()
        state.set_vgpr(0, 0x12345678)
        self.assertEqual(state.get_vgpr(0), 0x12345678)
        state.set_sgpr(5, 0x9ABCDEF0)
        self.assertEqual(state.get_sgpr(5), 0x9ABCDEF0)


if __name__ == "__main__":
    unittest.main()
