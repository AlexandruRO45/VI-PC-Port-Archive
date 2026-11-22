from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_perm_b32_f16.

Tests vector 32-bit permutation of 16-bit float elements:
    v_perm_b32_f16 D, S0, S1, S2
Lowered semantics:
    Destination is a 32-bit register holding two 16-bit half-precision floating-point
    values (D[15:0] as Half 0, D[31:16] as Half 1).
    Source pool is the 64-bit concatenated value {S0, S1}:
    - Half 0: S1[15:0] (low half of S1)
    - Half 1: S1[31:16] (high half of S1)
    - Half 2: S0[15:0] (low half of S0)
    - Half 3: S0[31:16] (high half of S0)

    Selector S2 provides selection control codes for each destination half:
    - S2[7:0]: selector code for destination Half 0 (D[15:0])
    - S2[15:8]: selector code for destination Half 1 (D[31:16])

    Selector codes:
    - 0: selects Half 0 (S1[15:0])
    - 1: selects Half 1 (S1[31:16])
    - 2: selects Half 2 (S0[15:0])
    - 3: selects Half 3 (S0[31:16])
    - 4: sign-extension of S1 bit 31 (0xFFFF if negative, else 0x0000)
    - 5: sign-extension of S0 bit 15 (0xFFFF if negative, else 0x0000)
    - 6: sign-extension of S0 bit 31 (0xFFFF if negative, else 0x0000)
    - 7: sign-extension of S1 bit 15 (0xFFFF if negative, else 0x0000)
    - 8: constant +0.0 (0x0000)
    - 9: constant -0.0 (0x8000)
    - 10: constant +1.0 (0x3C00)
    - 11: constant -1.0 (0xBC00)
    - 12: constant canonical Quiet NaN (0x7E00)
    - >= 13: constant all-ones (0xFFFF)

    Supports source modifiers per operand (abs, neg) and destination clamp [0.0, 1.0].

Architectural model:
    32-bit vector and scalar registers (VGPRs/SGPRs) with per-half lane slicing,
    selector decoding, IEEE 754-2008 binary16 bit-exact execution.
    Standard library only; zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
MASK16: int = 0xFFFF
SIGN16: int = 0x8000
ABS16: int = 0x7FFF
EXP_MASK16: int = 0x7C00
FRAC_MASK16: int = 0x03FF
QNAN_BIT16: int = 0x0200
CANONICAL_QNAN16: int = 0x7E00

POS_ZERO: int = 0x0000              # +0.0
NEG_ZERO: int = 0x8000              # -0.0
POS_ONE: int = 0x3C00               # +1.0
NEG_ONE: int = 0xBC00               # -1.0
POS_TWO: int = 0x4000               # +2.0
NEG_TWO: int = 0xC000               # -2.0
POS_HALF: int = 0x3800              # +0.5
NEG_HALF: int = 0xB800              # -0.5
MIN_NORMAL: int = 0x0400            # 2^-14
MAX_FINITE: int = 0x7BFF            # 65504.0
MIN_SUBNORMAL: int = 0x0001         # Smallest positive subnormal
MAX_SUBNORMAL: int = 0x03FF         # Largest positive subnormal
MIN_SUBNORMAL_NEG: int = 0x8001     # Smallest negative subnormal
MAX_SUBNORMAL_NEG: int = 0x83FF     # Largest negative subnormal
POS_INF: int = 0x7C00               # +Infinity
NEG_INF: int = 0xFC00               # -Infinity
POS_QNAN: int = 0x7E00              # Canonical Quiet NaN
NEG_QNAN: int = 0xFE00              # Negative Quiet NaN
POS_SNAN: int = 0x7C01              # Signaling NaN
ALL_ONES16: int = 0xFFFF            # 0xFFFF

# Selector codes
SEL_SRC1_LO: int = 0                # S1[15:0]
SEL_SRC1_HI: int = 1                # S1[31:16]
SEL_SRC0_LO: int = 2                # S0[15:0]
SEL_SRC0_HI: int = 3                # S0[31:16]

SEL_SIGN_EXT_SRC1_HI: int = 4       # S1[31] replicated over 16 bits
SEL_SIGN_EXT_SRC0_LO: int = 5       # S0[15] replicated over 16 bits
SEL_SIGN_EXT_SRC0_HI: int = 6       # S0[31] replicated over 16 bits
SEL_SIGN_EXT_SRC1_LO: int = 7       # S1[15] replicated over 16 bits

SEL_POS_ZERO: int = 8               # Constant +0.0 (0x0000)
SEL_NEG_ZERO: int = 9               # Constant -0.0 (0x8000)
SEL_POS_ONE: int = 10               # Constant +1.0 (0x3C00)
SEL_NEG_ONE: int = 11               # Constant -1.0 (0xBC00)
SEL_CANONICAL_QNAN: int = 12        # Constant Canonical Quiet NaN (0x7E00)
SEL_ONES_MIN: int = 13              # Constant 0xFFFF (>= 13)


# ==============================================================================
# Bit Manipulation and Half-Float Extraction Utilities
# ==============================================================================

def f16_to_bits(val: float) -> int:
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def extract_half(val32: int, idx: int) -> int:
    if idx not in (0, 1):
        raise ValueError(f"Half index out of range [0, 1]: {idx}")
    return (val32 >> (idx * 16)) & MASK16


def pack_halves(h0: int, h1: int) -> int:
    return ((h0 & MASK16) | ((h1 & MASK16) << 16)) & MASK32


def make_selector(sel0: int, sel1: int) -> int:
    return ((sel0 & 0xFF) | ((sel1 & 0xFF) << 8)) & MASK32


def apply_f16_modifiers(bits16: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    res = bits16 & MASK16
    if abs_mod:
        res &= ABS16
    if neg_mod:
        res ^= SIGN16
    return res


def clamp_f16(bits16: int) -> int:
    b = bits16 & MASK16
    if (b & EXP_MASK16) == EXP_MASK16 and (b & FRAC_MASK16) != 0:
        return POS_ZERO
    val = bits_to_f16(b)
    if val < 0.0 or (b == NEG_ZERO):
        return POS_ZERO
    elif val > 1.0:
        return POS_ONE
    return b


def permute_single_half(
    src0: int,
    src1: int,
    sel_code: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
) -> int:
    sel = sel_code & 0xFF
    if sel == SEL_SRC1_LO:
        return apply_f16_modifiers(extract_half(src1, 0), abs1, neg1)
    elif sel == SEL_SRC1_HI:
        return apply_f16_modifiers(extract_half(src1, 1), abs1, neg1)
    elif sel == SEL_SRC0_LO:
        return apply_f16_modifiers(extract_half(src0, 0), abs0, neg0)
    elif sel == SEL_SRC0_HI:
        return apply_f16_modifiers(extract_half(src0, 1), abs0, neg0)
    elif sel == SEL_SIGN_EXT_SRC1_HI:
        return 0xFFFF if bool(src1 & 0x8000_0000) else 0x0000
    elif sel == SEL_SIGN_EXT_SRC0_LO:
        return 0xFFFF if bool(src0 & 0x0000_8000) else 0x0000
    elif sel == SEL_SIGN_EXT_SRC0_HI:
        return 0xFFFF if bool(src0 & 0x8000_0000) else 0x0000
    elif sel == SEL_SIGN_EXT_SRC1_LO:
        return 0xFFFF if bool(src1 & 0x0000_8000) else 0x0000
    elif sel == SEL_POS_ZERO:
        return POS_ZERO
    elif sel == SEL_NEG_ZERO:
        return NEG_ZERO
    elif sel == SEL_POS_ONE:
        return POS_ONE
    elif sel == SEL_NEG_ONE:
        return NEG_ONE
    elif sel == SEL_CANONICAL_QNAN:
        return CANONICAL_QNAN16
    else:
        return ALL_ONES16


def oracle_v_perm_b32_f16(
    src0: int,
    src1: int,
    selector: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    sel = selector & MASK32

    sel0 = sel & 0xFF
    sel1 = (sel >> 8) & 0xFF

    h0 = permute_single_half(s0, s1, sel0, abs0, neg0, abs1, neg1)
    h1 = permute_single_half(s0, s1, sel1, abs0, neg0, abs1, neg1)

    if clamp:
        h0 = clamp_f16(h0)
        h1 = clamp_f16(h1)

    return pack_halves(h0, h1)


@dataclass
class VGPRWord:
    raw: int = 0

    @property
    def u32(self) -> int:
        return self.raw & MASK32

    @property
    def lo_half(self) -> int:
        return self.raw & MASK16

    @property
    def hi_half(self) -> int:
        return (self.raw >> 16) & MASK16

    def set_half(self, idx: int, val16: int) -> None:
        if idx == 0:
            self.raw = ((self.raw & 0xFFFF_0000) | (val16 & MASK16)) & MASK32
        elif idx == 1:
            self.raw = ((self.raw & 0x0000_FFFF) | ((val16 & MASK16) << 16)) & MASK32
        else:
            raise ValueError(f"Invalid half index: {idx}")


@dataclass
class PermSelectorWord:
    sel0: int = 0
    sel1: int = 0

    @property
    def raw(self) -> int:
        return make_selector(self.sel0, self.sel1)

    @classmethod
    def identity_src1(cls) -> PermSelectorWord:
        return cls(sel0=SEL_SRC1_LO, sel1=SEL_SRC1_HI)

    @classmethod
    def identity_src0(cls) -> PermSelectorWord:
        return cls(sel0=SEL_SRC0_LO, sel1=SEL_SRC0_HI)

    @classmethod
    def swap_halves_src1(cls) -> PermSelectorWord:
        return cls(sel0=SEL_SRC1_HI, sel1=SEL_SRC1_LO)

    @classmethod
    def swap_halves_src0(cls) -> PermSelectorWord:
        return cls(sel0=SEL_SRC0_HI, sel1=SEL_SRC0_LO)

    @classmethod
    def broadcast_half(cls, half_idx: int) -> PermSelectorWord:
        return cls(sel0=half_idx, sel1=half_idx)


class TestPermB32F16ReferenceOracle(unittest.TestCase):
    def test_identity_src1_selection(self):
        src0 = pack_halves(0x1111, 0x2222)
        src1 = pack_halves(0x3333, 0x4444)
        sel = PermSelectorWord.identity_src1().raw
        self.assertEqual(oracle_v_perm_b32_f16(src0, src1, sel), src1)

    def test_identity_src0_selection(self):
        src0 = pack_halves(0x1111, 0x2222)
        src1 = pack_halves(0x3333, 0x4444)
        sel = PermSelectorWord.identity_src0().raw
        self.assertEqual(oracle_v_perm_b32_f16(src0, src1, sel), src0)

    def test_cross_source_swizzle(self):
        src0 = pack_halves(0xCCCC, 0xDDDD)
        src1 = pack_halves(0xAAAA, 0xBBBB)
        sel = make_selector(SEL_SRC1_LO, SEL_SRC0_HI)
        self.assertEqual(oracle_v_perm_b32_f16(src0, src1, sel), 0xDDDD_AAAA)

    def test_half_broadcast(self):
        src1 = pack_halves(0x1234, 0x5678)
        sel0 = PermSelectorWord.broadcast_half(0).raw
        self.assertEqual(oracle_v_perm_b32_f16(0, src1, sel0), 0x1234_1234)

    def test_constant_zeros_and_ones(self):
        src0 = 0xFFFF_FFFF
        src1 = 0xFFFF_FFFF
        sel_zeros = make_selector(SEL_POS_ZERO, SEL_NEG_ZERO)
        self.assertEqual(oracle_v_perm_b32_f16(src0, src1, sel_zeros), 0x8000_0000)

        sel_ones = make_selector(13, 255)
        self.assertEqual(oracle_v_perm_b32_f16(0, 0, sel_ones), 0xFFFF_FFFF)

    def test_constant_ones_f16(self):
        sel = make_selector(SEL_POS_ONE, SEL_NEG_ONE)
        self.assertEqual(oracle_v_perm_b32_f16(0, 0, sel), 0xBC00_3C00)

    def test_canonical_qnan_fill(self):
        sel = make_selector(SEL_CANONICAL_QNAN, SEL_CANONICAL_QNAN)
        self.assertEqual(oracle_v_perm_b32_f16(0, 0, sel), 0x7E00_7E00)

    def test_sign_extension_modes(self):
        s1_pos = 0x7FFF_FFFF
        s1_neg = 0x8000_0000
        sel4 = make_selector(SEL_SIGN_EXT_SRC1_HI, SEL_SIGN_EXT_SRC1_HI)
        self.assertEqual(oracle_v_perm_b32_f16(0, s1_pos, sel4), 0x0000_0000)
        self.assertEqual(oracle_v_perm_b32_f16(0, s1_neg, sel4), 0xFFFF_FFFF)

    def test_source_modifiers_abs_neg(self):
        s1 = pack_halves(NEG_ONE, NEG_TWO)
        sel = PermSelectorWord.identity_src1().raw
        res_abs = oracle_v_perm_b32_f16(0, s1, sel, abs1=True)
        self.assertEqual(res_abs, pack_halves(POS_ONE, POS_TWO))

    def test_clamp_modifier(self):
        s1 = pack_halves(POS_TWO, NEG_ONE)
        sel = PermSelectorWord.identity_src1().raw
        res_clamped = oracle_v_perm_b32_f16(0, s1, sel, clamp=True)
        self.assertEqual(res_clamped, pack_halves(POS_ONE, POS_ZERO))

if __name__ == "__main__":
    unittest.main()
