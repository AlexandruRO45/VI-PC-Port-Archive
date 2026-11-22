from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_perm_b32_i16.

Tests vector 32-bit permutation of 16-bit signed integer elements:
    v_perm_b32_i16 D, S0, S1, S2
Lowered semantics:
    Destination is a 32-bit register holding two 16-bit signed integers (D[15:0] as i16 lo,
    D[31:16] as i16 hi).
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
    - 8: constant 0x0000 (0)
    - 9: constant 0x0001 (+1)
    - 10: constant 0xFFFF (-1)
    - 11: constant 0x8000 (MIN_INT16: -32768)
    - 12: constant 0x7FFF (MAX_INT16: +32767)
    - >= 13: constant 0xFFFF (-1)

Architectural model:
    32-bit vector and scalar registers (VGPRs/SGPRs) with per-half lane slicing,
    selector decoding, signed 16-bit two's complement integer execution.
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

INT16_ZERO: int = 0x0000
INT16_ONE: int = 0x0001
INT16_NEG_ONE: int = 0xFFFF
INT16_MIN: int = 0x8000             # -32768
INT16_MAX: int = 0x7FFF             # +32767

# Selector codes
SEL_SRC1_LO: int = 0                # S1[15:0]
SEL_SRC1_HI: int = 1                # S1[31:16]
SEL_SRC0_LO: int = 2                # S0[15:0]
SEL_SRC0_HI: int = 3                # S0[31:16]

SEL_SIGN_EXT_SRC1_HI: int = 4       # S1[31] replicated over 16 bits
SEL_SIGN_EXT_SRC0_LO: int = 5       # S0[15] replicated over 16 bits
SEL_SIGN_EXT_SRC0_HI: int = 6       # S0[31] replicated over 16 bits
SEL_SIGN_EXT_SRC1_LO: int = 7       # S1[15] replicated over 16 bits

SEL_CONST_ZERO: int = 8             # Constant 0x0000
SEL_CONST_ONE: int = 9              # Constant 0x0001
SEL_CONST_NEG_ONE: int = 10         # Constant 0xFFFF (-1)
SEL_CONST_MIN: int = 11             # Constant 0x8000 (-32768)
SEL_CONST_MAX: int = 12             # Constant 0x7FFF (+32767)
SEL_ONES_MIN: int = 13              # Constant 0xFFFF (>= 13)


# ==============================================================================
# Bit Manipulation and Half-Word Extraction Utilities
# ==============================================================================

def extract_half(val32: int, idx: int) -> int:
    if idx not in (0, 1):
        raise ValueError(f"Half index out of range [0, 1]: {idx}")
    return (val32 >> (idx * 16)) & MASK16


def pack_halves(h0: int, h1: int) -> int:
    return ((h0 & MASK16) | ((h1 & MASK16) << 16)) & MASK32


def to_signed_i16(u16: int) -> int:
    val = u16 & MASK16
    return val - 0x10000 if val & SIGN16 else val


def from_signed_i16(i16: int) -> int:
    return i16 & MASK16


def make_selector(sel0: int, sel1: int) -> int:
    return ((sel0 & 0xFF) | ((sel1 & 0xFF) << 8)) & MASK32


def permute_single_i16(src0: int, src1: int, sel_code: int) -> int:
    sel = sel_code & 0xFF
    if sel == SEL_SRC1_LO:
        return extract_half(src1, 0)
    elif sel == SEL_SRC1_HI:
        return extract_half(src1, 1)
    elif sel == SEL_SRC0_LO:
        return extract_half(src0, 0)
    elif sel == SEL_SRC0_HI:
        return extract_half(src0, 1)
    elif sel == SEL_SIGN_EXT_SRC1_HI:
        return 0xFFFF if bool(src1 & 0x8000_0000) else 0x0000
    elif sel == SEL_SIGN_EXT_SRC0_LO:
        return 0xFFFF if bool(src0 & 0x0000_8000) else 0x0000
    elif sel == SEL_SIGN_EXT_SRC0_HI:
        return 0xFFFF if bool(src0 & 0x8000_0000) else 0x0000
    elif sel == SEL_SIGN_EXT_SRC1_LO:
        return 0xFFFF if bool(src1 & 0x0000_8000) else 0x0000
    elif sel == SEL_CONST_ZERO:
        return INT16_ZERO
    elif sel == SEL_CONST_ONE:
        return INT16_ONE
    elif sel == SEL_CONST_NEG_ONE:
        return INT16_NEG_ONE
    elif sel == SEL_CONST_MIN:
        return INT16_MIN
    elif sel == SEL_CONST_MAX:
        return INT16_MAX
    else:
        return 0xFFFF


def oracle_v_perm_b32_i16(src0: int, src1: int, selector: int) -> int:
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    sel = selector & MASK32

    sel0 = sel & 0xFF
    sel1 = (sel >> 8) & 0xFF

    h0 = permute_single_i16(s0, s1, sel0)
    h1 = permute_single_i16(s0, s1, sel1)

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


class TestPermB32I16ReferenceOracle(unittest.TestCase):
    def test_identity_src1_selection(self):
        src0 = pack_halves(0x1111, 0x2222)
        src1 = pack_halves(0x3333, 0x4444)
        sel = PermSelectorWord.identity_src1().raw
        self.assertEqual(oracle_v_perm_b32_i16(src0, src1, sel), src1)

    def test_identity_src0_selection(self):
        src0 = pack_halves(0x1111, 0x2222)
        src1 = pack_halves(0x3333, 0x4444)
        sel = PermSelectorWord.identity_src0().raw
        self.assertEqual(oracle_v_perm_b32_i16(src0, src1, sel), src0)

    def test_cross_source_swizzle(self):
        src0 = pack_halves(0xCCCC, 0xDDDD)
        src1 = pack_halves(0xAAAA, 0xBBBB)
        sel = make_selector(SEL_SRC1_LO, SEL_SRC0_HI)
        self.assertEqual(oracle_v_perm_b32_i16(src0, src1, sel), 0xDDDD_AAAA)

    def test_constants_fill(self):
        sel_const = make_selector(SEL_CONST_MIN, SEL_CONST_MAX)
        res = oracle_v_perm_b32_i16(0, 0, sel_const)
        self.assertEqual(res, pack_halves(INT16_MIN, INT16_MAX))

    def test_sign_extension_modes(self):
        s1_neg = 0x8000_0000
        sel4 = make_selector(SEL_SIGN_EXT_SRC1_HI, SEL_SIGN_EXT_SRC1_HI)
        self.assertEqual(oracle_v_perm_b32_i16(0, s1_neg, sel4), 0xFFFF_FFFF)

if __name__ == "__main__":
    unittest.main()
