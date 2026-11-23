from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_cvt_norm_u16_f32.

Instruction:
  v_cvt_norm_u16_f32 dst, src0 (VOP1 / VOP3)
  D.u16 = f32_to_unorm16(S0.f32)

Architecture:
  - RDNA2 Vector ALU (GFX10) single-precision float to unsigned normalized 16-bit integer conversion.
  - S0: 32-bit IEEE 754 binary32 float (from VGPR, SGPR, or literal/inline const).
  - D:  16-bit unsigned normalized integer in [0, 65535] (halfword in VGPR).
  - Normalization:
      * Maps floating-point range [0.0, 1.0] to unsigned integer range [0, 65535].
      * Scaling factor is 65535.0 (2^16 - 1).
      * Rounding mode: round-to-nearest-even (IEEE 754-2008 roundTiesToEven).
      * Clamping: values <= 0.0 (or -inf) clamp to 0 (0x0000).
                  values >= 1.0 (or +inf) clamp to 65535 (0xFFFF).
      * NaNs (both QNaN and SNaN) convert to 0 (0x0000).
      * Signed zeros (+0.0, -0.0) convert to 0 (0x0000).
  - Register model:
      * 32-bit vector registers (VGPRs: v0-v255) and scalar registers (SGPRs: s0-s105).
      * Destination writes to 16-bit low half (.l [15:0]) or high half (.h [31:16]), preserving
        the untouched 16-bit half of the 32-bit VGPR.
      * Source reads full 32-bit dword float from VGPR or SGPR.
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

# IEEE 754 binary32 architectural bit layout
SIGN_MASK_F32: int = 0x80000000
EXP_MASK_F32: int = 0x7F800000
MANT_MASK_F32: int = 0x007FFFFF
QNAN_BIT_F32: int = 0x00400000
BIAS_F32: int = 127

# Special single-precision bit patterns
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000
SNAN_F32: int = 0x7F800001

# Standard reference floating-point constants in binary32
ONE_F32: int = 0x3F800000            # 1.0f
NEG_ONE_F32: int = 0xBF800000        # -1.0f
TWO_F32: int = 0x40000000            # 2.0f
NEG_TWO_F32: int = 0xC0000000        # -2.0f
HALF_F32: int = 0x3F000000           # 0.5f
NEG_HALF_F32: int = 0xBF000000       # -0.5f
QUARTER_F32: int = 0x3E800000        # 0.25f
NEG_QUARTER_F32: int = 0xBE800000    # -0.25f
FOUR_F32: int = 0x40800000           # 4.0f
NEG_FOUR_F32: int = 0xC0800000       # -4.0f

MAX_FINITE_F32: int = 0x7F7FFFFF     # ~3.402823466e+38
MIN_FINITE_NEG_F32: int = 0xFF7FFFFF # ~-3.402823466e+38
MIN_NORMAL_F32: int = 0x00800000     # 2^-126 (~1.17549435e-38)
MIN_SUBNORMAL_F32: int = 0x00000001  # 2^-149 (~1.40129846e-45)
MAX_SUBNORMAL_F32: int = 0x007FFFFF  # ~1.17549421e-38

# 16-bit unsigned normalized integer range constants
UNORM16_MAX: int = 65535             # 0xFFFF (+1.0)
UNORM16_MIN: int = 0                 # 0x0000 (0.0)
UNORM16_SCALE: float = 65535.0


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


def bits_to_f32(bits: int) -> float:
    """Convert unsigned 32-bit bits into Python float via struct IEEE 754 single."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Convert Python float into unsigned 32-bit IEEE 754 single bits."""
    if math.isnan(val):
        return QNAN_F32
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return NEG_INF_F32 if val < 0.0 else POS_INF_F32


def is_nan_f32(bits: int) -> bool:
    """Check if binary32 bits represent any NaN."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) != 0)


def is_qnan_f32(bits: int) -> bool:
    """Check if binary32 bits represent a Quiet NaN."""
    return is_nan_f32(bits) and bool(bits & QNAN_BIT_F32)


def is_snan_f32(bits: int) -> bool:
    """Check if binary32 bits represent a Signaling NaN."""
    return is_nan_f32(bits) and not bool(bits & QNAN_BIT_F32)


def is_inf_f32(bits: int) -> bool:
    """Check if binary32 bits represent +/- infinity."""
    return ((bits & EXP_MASK_F32) == EXP_MASK_F32) and ((bits & MANT_MASK_F32) == 0)


def is_subnormal_f32(bits: int) -> bool:
    """Check if binary32 bits represent a subnormal number."""
    return ((bits & EXP_MASK_F32) == 0) and ((bits & MANT_MASK_F32) != 0)


def is_zero_f32(bits: int) -> bool:
    """Check if binary32 bits represent +0.0 or -0.0."""
    return (bits & ~SIGN_MASK_F32) == 0


def apply_f32_modifiers(bits32: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply RDNA2 source modifiers (|src|, -src) to a 32-bit float word.
    
    Order of operations: abs clears sign bit (bit 31) first, then neg inverts sign bit (bit 31).
    """
    val = bits32 & MASK32
    if abs_mod:
        val &= ~SIGN_MASK_F32
    if neg_mod:
        val ^= SIGN_MASK_F32
    return val


def f32_to_unorm16(src0_bits: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Reference mathematical oracle for RDNA2 f32_to_unorm16 conversion.
    
    Converts 32-bit float bits to unsigned normalized 16-bit integer bits.
    - NaNs convert to 0.
    - Negative values (<= 0.0) clamp to 0.
    - Values >= 1.0 (and +inf) clamp to 65535.
    - Scaled by 65535.0 and rounded with round-to-nearest-even.
    Returns unsigned 16-bit integer.
    """
    val_bits = apply_f32_modifiers(src0_bits, abs_mod, neg_mod)
    if is_nan_f32(val_bits):
        return 0
    if is_zero_f32(val_bits):
        return 0
    if is_inf_f32(val_bits):
        return 0 if (val_bits & SIGN_MASK_F32) else UNORM16_MAX

    val_float = bits_to_f32(val_bits)
    if val_float <= 0.0:
        return 0
    if val_float >= 1.0:
        return UNORM16_MAX

    scaled = val_float * UNORM16_SCALE
    rounded = int(round(scaled))
    if rounded > UNORM16_MAX:
        rounded = UNORM16_MAX
    elif rounded < 0:
        rounded = 0
    return u16(rounded)


def oracle_v_cvt_norm_u16_f32(
    src0_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
) -> int:
    """Bit-exact reference oracle for v_cvt_norm_u16_f32."""
    return f32_to_unorm16(src0_bits, abs_mod, neg_mod)


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


class TestValuCvtNormU16F32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_oracle_zeros(self) -> None:
        self.assertEqual(oracle_v_cvt_norm_u16_f32(POS_ZERO_F32), 0)
        self.assertEqual(oracle_v_cvt_norm_u16_f32(NEG_ZERO_F32), 0)

    def test_oracle_unit_boundaries(self) -> None:
        # +1.0 -> 65535 (0xFFFF)
        self.assertEqual(oracle_v_cvt_norm_u16_f32(ONE_F32), 65535)
        self.assertEqual(oracle_v_cvt_norm_u16_f32(ONE_F32), 0xFFFF)
        # -1.0 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u16_f32(NEG_ONE_F32), 0)

    def test_oracle_half_values(self) -> None:
        # +0.5 * 65535 = 32767.5 -> rounds to 32768 (even) = 0x8000
        self.assertEqual(oracle_v_cvt_norm_u16_f32(HALF_F32), 32768)
        self.assertEqual(oracle_v_cvt_norm_u16_f32(HALF_F32), 0x8000)
        # +0.25 * 65535 = 16383.75 -> rounds to 16384 = 0x4000
        self.assertEqual(oracle_v_cvt_norm_u16_f32(QUARTER_F32), 16384)
        # -0.5 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u16_f32(NEG_HALF_F32), 0)

    def test_oracle_clamping_extremes(self) -> None:
        # +2.0 clamps to +1.0 -> 65535
        self.assertEqual(oracle_v_cvt_norm_u16_f32(TWO_F32), 65535)
        # -2.0 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u16_f32(NEG_TWO_F32), 0)
        # +4.0 clamps to 65535, -4.0 clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u16_f32(FOUR_F32), 65535)
        self.assertEqual(oracle_v_cvt_norm_u16_f32(NEG_FOUR_F32), 0)
        # Infinities clamp to extremes
        self.assertEqual(oracle_v_cvt_norm_u16_f32(POS_INF_F32), 65535)
        self.assertEqual(oracle_v_cvt_norm_u16_f32(NEG_INF_F32), 0)

    def test_oracle_nans(self) -> None:
        # Both QNaN and SNaN convert to 0
        self.assertEqual(oracle_v_cvt_norm_u16_f32(QNAN_F32), 0)
        self.assertEqual(oracle_v_cvt_norm_u16_f32(SNAN_F32), 0)
        self.assertEqual(oracle_v_cvt_norm_u16_f32(0x7FC00001), 0)
        self.assertEqual(oracle_v_cvt_norm_u16_f32(0xFFC00000), 0)

    def test_oracle_modifiers(self) -> None:
        # abs(-1.0) = 1.0 -> 65535
        self.assertEqual(oracle_v_cvt_norm_u16_f32(NEG_ONE_F32, abs_mod=True), 65535)
        # neg(1.0) = -1.0 -> clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u16_f32(ONE_F32, neg_mod=True), 0)
        # abs and neg: -|1.0| = -1.0 -> clamps to 0
        self.assertEqual(oracle_v_cvt_norm_u16_f32(ONE_F32, abs_mod=True, neg_mod=True), 0)

    def test_architectural_register_state(self) -> None:
        state = GpuRegisterState()
        state.set_vgpr(0, 0x12345678)
        self.assertEqual(state.get_vgpr(0), 0x12345678)
        state.set_sgpr(5, 0x9ABCDEF0)
        self.assertEqual(state.get_sgpr(5), 0x9ABCDEF0)


if __name__ == "__main__":
    unittest.main()
