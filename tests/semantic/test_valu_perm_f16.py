from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_perm_f16.

Tests the vector 16-bit half-precision float permutation instruction:
    v_perm_f16 D, S0, S1, S2
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
import math
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

# IEEE 754-2008 binary16 architectural float constants
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
    """Pack Python float into IEEE 754 binary16 unsigned 16-bit integer."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def bits_to_f16(bits: int) -> float:
    """Unpack unsigned 16-bit integer into Python float (IEEE 754 binary16)."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def extract_half(val32: int, idx: int) -> int:
    """Extract 16-bit half idx (0 for low [15:0], 1 for high [31:16])."""
    if idx not in (0, 1):
        raise ValueError(f"Half index out of range [0, 1]: {idx}")
    return (val32 >> (idx * 16)) & MASK16


def pack_halves(h0: int, h1: int) -> int:
    """Pack two 16-bit halves (h0 low, h1 high) into an unsigned 32-bit dword."""
    return ((h0 & MASK16) | ((h1 & MASK16) << 16)) & MASK32


def make_selector(sel0: int, sel1: int) -> int:
    """Construct a 32-bit selector word from 2 individual half selector codes."""
    return ((sel0 & 0xFF) | ((sel1 & 0xFF) << 8)) & MASK32


def apply_f16_modifiers(bits16: int, abs_mod: bool = False, neg_mod: bool = False) -> int:
    """Apply RDNA2 source modifiers (|src|, -src) to a 16-bit half float word.
    
    In RDNA2 ISA, abs clears sign bit (bit 15) first, then neg inverts sign bit (bit 15).
    """
    res = bits16 & MASK16
    if abs_mod:
        res &= ABS16
    if neg_mod:
        res ^= SIGN16
    return res


def clamp_f16(bits16: int) -> int:
    """Apply RDNA2 VALU clamp [0.0, 1.0] to a 16-bit float word.
    
    IEEE 754 NaN clamps to +0.0. Values < 0.0 clamp to +0.0. Values > 1.0 clamp to 1.0.
    """
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
    """Evaluate a single 16-bit half according to RDNA2 v_perm_f16 hardware semantics."""
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
    else:  # sel >= SEL_ONES_MIN (13..255)
        return ALL_ONES16


def oracle_v_perm_f16(
    src0: int,
    src1: int,
    selector: int,
    abs0: bool = False,
    neg0: bool = False,
    abs1: bool = False,
    neg1: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_perm_f16.
    
    src0: upper pool providing Half 2 (bits 0..15) and Half 3 (bits 16..31)
    src1: lower pool providing Half 0 (bits 0..15) and Half 1 (bits 16..31)
    selector: control word where byte 0 selects Half 0, byte 1 selects Half 1
    """
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


# ==============================================================================
# Architectural Register and Word Models
# ==============================================================================

@dataclass
class VGPRWord:
    """Architectural representation of a 32-bit Vector General Purpose Register (VGPR)."""
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
class PermSelectorF16Word:
    """Structured representation of a 32-bit permutation selector word for v_perm_f16."""
    sel0: int = 0  # S2[7:0]
    sel1: int = 0  # S2[15:8]

    @property
    def raw(self) -> int:
        return make_selector(self.sel0, self.sel1)

    @classmethod
    def identity_src1(cls) -> PermSelectorF16Word:
        """Selects Half 0 to low, Half 1 to high (passes S1 through)."""
        return cls(sel0=SEL_SRC1_LO, sel1=SEL_SRC1_HI)

    @classmethod
    def identity_src0(cls) -> PermSelectorF16Word:
        """Selects Half 2 to low, Half 3 to high (passes S0 through)."""
        return cls(sel0=SEL_SRC0_LO, sel1=SEL_SRC0_HI)

    @classmethod
    def swap_halves_src1(cls) -> PermSelectorF16Word:
        """Swaps low and high halves of S1."""
        return cls(sel0=SEL_SRC1_HI, sel1=SEL_SRC1_LO)

    @classmethod
    def swap_halves_src0(cls) -> PermSelectorF16Word:
        """Swaps low and high halves of S0."""
        return cls(sel0=SEL_SRC0_HI, sel1=SEL_SRC0_LO)

    @classmethod
    def broadcast_half(cls, half_idx: int) -> PermSelectorF16Word:
        """Replicates a single half index (0..3) across both destination halves."""
        return cls(sel0=half_idx, sel1=half_idx)


# ==============================================================================
# Unit Tests: Commit 1 (Reference Oracle & Architectural Model)
# ==============================================================================

class TestPermF16ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_identity_src1_selection(self):
        src0 = pack_halves(0x1111, 0x2222)
        src1 = pack_halves(0x3333, 0x4444)
        sel = PermSelectorF16Word.identity_src1().raw
        self.assertEqual(oracle_v_perm_f16(src0, src1, sel), src1)

    def test_identity_src0_selection(self):
        src0 = pack_halves(0x1111, 0x2222)
        src1 = pack_halves(0x3333, 0x4444)
        sel = PermSelectorF16Word.identity_src0().raw
        self.assertEqual(oracle_v_perm_f16(src0, src1, sel), src0)

    def test_cross_source_swizzle(self):
        # Pick S1.lo (0xAAAA) and S0.hi (0xDDDD) -> 0xDDDD_AAAA
        src0 = pack_halves(0xCCCC, 0xDDDD)
        src1 = pack_halves(0xAAAA, 0xBBBB)
        sel = make_selector(SEL_SRC1_LO, SEL_SRC0_HI)
        self.assertEqual(oracle_v_perm_f16(src0, src1, sel), 0xDDDD_AAAA)

    def test_half_broadcast(self):
        src1 = pack_halves(0x1234, 0x5678)
        # Broadcast Half 0 (0x1234)
        sel0 = PermSelectorF16Word.broadcast_half(0).raw
        self.assertEqual(oracle_v_perm_f16(0, src1, sel0), 0x1234_1234)

        # Broadcast Half 3 (S0.hi = 0x9ABC)
        src0 = pack_halves(0x5678, 0x9ABC)
        sel3 = PermSelectorF16Word.broadcast_half(3).raw
        self.assertEqual(oracle_v_perm_f16(src0, 0, sel3), 0x9ABC_9ABC)

    def test_constant_zeros_and_ones(self):
        src0 = 0xFFFF_FFFF
        src1 = 0xFFFF_FFFF
        # Code 8 is +0.0 (0x0000), code 9 is -0.0 (0x8000)
        sel_zeros = make_selector(SEL_POS_ZERO, SEL_NEG_ZERO)
        self.assertEqual(oracle_v_perm_f16(src0, src1, sel_zeros), 0x8000_0000)

        # Code 13+ is all ones (0xFFFF)
        sel_ones = make_selector(13, 255)
        self.assertEqual(oracle_v_perm_f16(0, 0, sel_ones), 0xFFFF_FFFF)

    def test_constant_ones_f16(self):
        # Code 10 is +1.0 (0x3C00), code 11 is -1.0 (0xBC00)
        sel = make_selector(SEL_POS_ONE, SEL_NEG_ONE)
        self.assertEqual(oracle_v_perm_f16(0, 0, sel), 0xBC00_3C00)

    def test_canonical_qnan_fill(self):
        # Code 12 is canonical Quiet NaN (0x7E00)
        sel = make_selector(SEL_CANONICAL_QNAN, SEL_CANONICAL_QNAN)
        self.assertEqual(oracle_v_perm_f16(0, 0, sel), 0x7E00_7E00)

    def test_sign_extension_modes(self):
        # Code 4: sign of S1 bit 31
        s1_pos = 0x7FFF_FFFF
        s1_neg = 0x8000_0000
        sel4 = make_selector(SEL_SIGN_EXT_SRC1_HI, SEL_SIGN_EXT_SRC1_HI)
        self.assertEqual(oracle_v_perm_f16(0, s1_pos, sel4), 0x0000_0000)
        self.assertEqual(oracle_v_perm_f16(0, s1_neg, sel4), 0xFFFF_FFFF)

        # Code 5: sign of S0 bit 15
        s0_pos = 0x0000_7FFF
        s0_neg = 0x0000_8000
        sel5 = make_selector(SEL_SIGN_EXT_SRC0_LO, SEL_SIGN_EXT_SRC0_LO)
        self.assertEqual(oracle_v_perm_f16(s0_pos, 0, sel5), 0x0000_0000)
        self.assertEqual(oracle_v_perm_f16(s0_neg, 0, sel5), 0xFFFF_FFFF)

        # Code 6: sign of S0 bit 31
        s0_pos31 = 0x7FFF_0000
        s0_neg31 = 0x8000_0000
        sel6 = make_selector(SEL_SIGN_EXT_SRC0_HI, SEL_SIGN_EXT_SRC0_HI)
        self.assertEqual(oracle_v_perm_f16(s0_pos31, 0, sel6), 0x0000_0000)
        self.assertEqual(oracle_v_perm_f16(s0_neg31, 0, sel6), 0xFFFF_FFFF)

        # Code 7: sign of S1 bit 15
        s1_pos15 = 0x0000_7FFF
        s1_neg15 = 0x0000_8000
        sel7 = make_selector(SEL_SIGN_EXT_SRC1_LO, SEL_SIGN_EXT_SRC1_LO)
        self.assertEqual(oracle_v_perm_f16(0, s1_pos15, sel7), 0x0000_0000)
        self.assertEqual(oracle_v_perm_f16(0, s1_neg15, sel7), 0xFFFF_FFFF)

    def test_source_modifiers_abs_neg(self):
        # S1 has negative numbers: 0xBC00 (-1.0) and 0xC000 (-2.0)
        s1 = pack_halves(NEG_ONE, NEG_TWO)
        sel = PermSelectorF16Word.identity_src1().raw

        # abs1 clears sign bits -> +1.0 (0x3C00) and +2.0 (0x4000)
        res_abs = oracle_v_perm_f16(0, s1, sel, abs1=True)
        self.assertEqual(res_abs, pack_halves(POS_ONE, POS_TWO))

        # neg1 inverts sign bits on positive numbers
        s1_pos = pack_halves(POS_ONE, POS_TWO)
        res_neg = oracle_v_perm_f16(0, s1_pos, sel, neg1=True)
        self.assertEqual(res_neg, pack_halves(NEG_ONE, NEG_TWO))

    def test_clamp_modifier(self):
        # Values outside [0.0, 1.0]: 0x4000 (+2.0) and 0xBC00 (-1.0)
        s1 = pack_halves(POS_TWO, NEG_ONE)
        sel = PermSelectorF16Word.identity_src1().raw

        # Clamped: +2.0 -> +1.0 (0x3C00), -1.0 -> +0.0 (0x0000)
        res_clamped = oracle_v_perm_f16(0, s1, sel, clamp=True)
        self.assertEqual(res_clamped, pack_halves(POS_ONE, POS_ZERO))

    def test_vgpr_word_f16_model(self):
        vw = VGPRWord(0x1234_5678)
        self.assertEqual(vw.u32, 0x1234_5678)
        self.assertEqual(vw.lo_half, 0x5678)
        self.assertEqual(vw.hi_half, 0x1234)

        vw.set_half(0, 0xAAAA)
        self.assertEqual(vw.u32, 0x1234_AAAA)
        vw.set_half(1, 0xBBBB)
        self.assertEqual(vw.u32, 0xBBBB_AAAA)

    def test_selector_word_f16_model(self):
        sw = PermSelectorF16Word(sel0=2, sel1=3)
        self.assertEqual(sw.sel0, 2)
        self.assertEqual(sw.sel1, 3)
        self.assertEqual(sw.raw, 0x0302)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class Opcode(Enum):
    """VALU instruction opcodes modeled for semantic analysis."""
    V_PERM_F16 = auto()
    V_NOP = auto()


class OperandKind(Enum):
    """Hardware operand categories for RDNA2 instructions."""
    VGPR = auto()
    SGPR = auto()
    LITERAL_CONST = auto()
    INLINE_CONST = auto()


class DataType(Enum):
    """Architectural data types for operands and intermediate evaluations."""
    F16 = auto()
    V2F16 = auto()
    U32 = auto()


@dataclass
class ASTOperand:
    """Symbolic AST node representing an instruction operand with modifiers."""
    kind: OperandKind
    reg_num: int = 0
    constant_val: int = 0
    data_type: DataType = DataType.V2F16
    abs_mod: bool = False
    neg_mod: bool = False

    @classmethod
    def vgpr(cls, num: int, abs_mod: bool = False, neg_mod: bool = False, dtype: DataType = DataType.V2F16) -> ASTOperand:
        if not 0 <= num <= 255:
            raise IndexError(f"VGPR index out of range [0, 255]: {num}")
        return cls(kind=OperandKind.VGPR, reg_num=num, abs_mod=abs_mod, neg_mod=neg_mod, data_type=dtype)

    @classmethod
    def sgpr(cls, num: int, abs_mod: bool = False, neg_mod: bool = False, dtype: DataType = DataType.V2F16) -> ASTOperand:
        if not 0 <= num <= 105:
            raise IndexError(f"SGPR index out of range [0, 105]: {num}")
        return cls(kind=OperandKind.SGPR, reg_num=num, abs_mod=abs_mod, neg_mod=neg_mod, data_type=dtype)

    @classmethod
    def literal(cls, val: int, dtype: DataType = DataType.U32) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL_CONST, constant_val=val & MASK32, data_type=dtype)

    @classmethod
    def inline_const(cls, val: int, dtype: DataType = DataType.U32) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, constant_val=val & MASK32, data_type=dtype)

    def validate(self) -> None:
        if self.kind == OperandKind.VGPR and not (0 <= self.reg_num <= 255):
            raise IndexError(f"VGPR out of bounds: {self.reg_num}")
        if self.kind == OperandKind.SGPR and not (0 <= self.reg_num <= 105):
            raise IndexError(f"SGPR out of bounds: {self.reg_num}")


@dataclass
class ASTInstruction:
    """Symbolic AST node representing a complete v_perm_f16 instruction."""
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    src2: ASTOperand
    clamp: bool = False

    def validate(self) -> None:
        """Validate RDNA2 architectural encoding rules."""
        if self.dst.kind != OperandKind.VGPR:
            raise ValueError(f"VALU instruction destination must be a VGPR, got {self.dst.kind}")
        self.dst.validate()
        self.src0.validate()
        self.src1.validate()
        self.src2.validate()

        # Enforce max one 32-bit literal constant in instruction word
        literals = sum(1 for src in (self.src0, self.src1, self.src2) if src.kind == OperandKind.LITERAL_CONST)
        if literals > 1:
            raise ValueError(f"Hardware encoding permits at most one literal constant, got {literals}")


@dataclass
class ASTExpression:
    """Symbolic typed expression tree representing the operation semantics."""
    opcode: Opcode
    operands: List[Union[ASTOperand, ASTExpression]]
    clamp: bool = False

    def to_symbolic_string(self) -> str:
        ops_str = ", ".join(
            op.to_symbolic_string() if isinstance(op, ASTExpression)
            else f"{'|' if getattr(op, 'abs_mod', False) else ''}{'-' if getattr(op, 'neg_mod', False) else ''}{op.kind.name}{op.reg_num if op.kind in (OperandKind.VGPR, OperandKind.SGPR) else f'({hex(op.constant_val)})'}{'|' if getattr(op, 'abs_mod', False) else ''}"
            for op in self.operands
        )
        clamp_str = " [clamp]" if self.clamp else ""
        return f"{self.opcode.name}({ops_str}){clamp_str}"


# ==============================================================================
# Unit Tests: Commit 2 (Symbolic AST & Typed Expressions)
# ==============================================================================

class TestPermF16SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instruction_creation(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.sgpr(0),
            clamp=True,
        )
        self.assertEqual(instr.opcode, Opcode.V_PERM_F16)
        self.assertEqual(instr.dst.reg_num, 0)
        self.assertEqual(instr.src0.reg_num, 1)
        self.assertEqual(instr.src1.reg_num, 2)
        self.assertEqual(instr.src2.reg_num, 0)
        self.assertTrue(instr.clamp)

    def test_ast_operand_factories(self):
        v = ASTOperand.vgpr(42, abs_mod=True, neg_mod=False)
        self.assertEqual(v.kind, OperandKind.VGPR)
        self.assertEqual(v.reg_num, 42)
        self.assertTrue(v.abs_mod)
        self.assertFalse(v.neg_mod)

        s = ASTOperand.sgpr(10, abs_mod=False, neg_mod=True)
        self.assertEqual(s.kind, OperandKind.SGPR)
        self.assertEqual(s.reg_num, 10)
        self.assertTrue(s.neg_mod)

        lit = ASTOperand.literal(0x1234)
        self.assertEqual(lit.kind, OperandKind.LITERAL_CONST)
        self.assertEqual(lit.constant_val, 0x1234)

        inline = ASTOperand.inline_const(0)
        self.assertEqual(inline.kind, OperandKind.INLINE_CONST)
        self.assertEqual(inline.constant_val, 0)

    def test_ast_instruction_validation_valid(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(5),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.sgpr(2),
            src2=ASTOperand.literal(0x0100),
        )
        # Should not raise
        instr.validate()

    def test_ast_instruction_validation_rejects_sgpr_dst(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.sgpr(5),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(0x0100),
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_instruction_validation_rejects_multiple_literals(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(5),
            src0=ASTOperand.literal(0x1111_1111),
            src1=ASTOperand.literal(0x2222_2222),
            src2=ASTOperand.vgpr(0),
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_expression_tree_representation(self):
        expr = ASTExpression(
            opcode=Opcode.V_PERM_F16,
            operands=[
                ASTOperand.vgpr(1, abs_mod=True),
                ASTOperand.vgpr(2, neg_mod=True),
                ASTOperand.literal(0x0302),
            ],
            clamp=True,
        )
        sym_str = expr.to_symbolic_string()
        self.assertIn("V_PERM_F16", sym_str)
        self.assertIn("[clamp]", sym_str)
        self.assertIn("VGPR1", sym_str)
        self.assertIn("0x302", sym_str)

    def test_ast_modifiers_representation(self):
        op = ASTOperand.vgpr(7, abs_mod=True, neg_mod=True)
        self.assertTrue(op.abs_mod)
        self.assertTrue(op.neg_mod)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarRegisterFile:
    """Architectural register file for deterministic scalar simulation."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    scc: int = 0
    vcc: int = 0
    exec_mask: int = MASK32
    status: int = 0
    mode: int = 0

    def get_vgpr(self, idx: int) -> int:
        if not 0 <= idx <= 255:
            raise IndexError(f"VGPR index out of range [0, 255]: {idx}")
        return self.vgprs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        if not 0 <= idx <= 255:
            raise IndexError(f"VGPR index out of range [0, 255]: {idx}")
        self.vgprs[idx] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        if not 0 <= idx <= 105:
            raise IndexError(f"SGPR index out of range [0, 105]: {idx}")
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        if not 0 <= idx <= 105:
            raise IndexError(f"SGPR index out of range [0, 105]: {idx}")
        self.sgprs[idx] = val & MASK32

    def read_operand(self, op: ASTOperand) -> Tuple[int, bool, bool]:
        """Resolve operand to its 32-bit integer value and associated modifiers."""
        if op.kind == OperandKind.VGPR:
            val = self.get_vgpr(op.reg_num)
        elif op.kind == OperandKind.SGPR:
            val = self.get_sgpr(op.reg_num)
        elif op.kind in (OperandKind.LITERAL_CONST, OperandKind.INLINE_CONST):
            val = op.constant_val & MASK32
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")
        return val, op.abs_mod, op.neg_mod


class ScalarInterpreter:
    """Deterministic instruction interpreter for single-lane scalar execution."""

    def __init__(self, reg_file: Optional[ScalarRegisterFile] = None):
        self.rf = reg_file if reg_file is not None else ScalarRegisterFile()

    def execute(self, instr: ASTInstruction) -> None:
        """Execute a single AST instruction deterministically."""
        instr.validate()
        if instr.opcode == Opcode.V_PERM_F16:
            s0_val, abs0, neg0 = self.rf.read_operand(instr.src0)
            s1_val, abs1, neg1 = self.rf.read_operand(instr.src1)
            sel_val, _, _ = self.rf.read_operand(instr.src2)

            res = oracle_v_perm_f16(
                src0=s0_val,
                src1=s1_val,
                selector=sel_val,
                abs0=abs0,
                neg0=neg0,
                abs1=abs1,
                neg1=neg1,
                clamp=instr.clamp,
            )
            self.rf.set_vgpr(instr.dst.reg_num, res)
        elif instr.opcode == Opcode.V_NOP:
            pass
        else:
            raise NotImplementedError(f"Opcode not implemented: {instr.opcode}")


# ==============================================================================
# Unit Tests: Commit 3 (Deterministic Scalar Interpreter)
# ==============================================================================

class TestPermF16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_scalar_execute_basic_perm(self):
        self.rf.set_vgpr(1, pack_halves(0x1111, 0x2222))  # S0
        self.rf.set_vgpr(2, pack_halves(0x3333, 0x4444))  # S1
        self.rf.set_vgpr(3, make_selector(SEL_SRC1_LO, SEL_SRC0_HI))  # S2: pick S1.lo and S0.hi

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(0x3333, 0x2222))

    def test_scalar_execute_with_sgpr_sources(self):
        self.rf.set_sgpr(0, pack_halves(POS_ONE, POS_TWO))
        self.rf.set_sgpr(1, pack_halves(POS_HALF, MIN_NORMAL))
        self.rf.set_vgpr(5, make_selector(SEL_SRC1_HI, SEL_SRC0_LO))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(10),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.sgpr(1),
            src2=ASTOperand.vgpr(5),
        )
        self.interp.execute(instr)
        # S1.hi is MIN_NORMAL, S0.lo is POS_ONE
        self.assertEqual(self.rf.get_vgpr(10), pack_halves(MIN_NORMAL, POS_ONE))

    def test_scalar_execute_with_literal_selector(self):
        self.rf.set_vgpr(1, pack_halves(0xAAAA, 0xBBBB))
        self.rf.set_vgpr(2, pack_halves(0xCCCC, 0xDDDD))

        # Identity S1 via literal
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(make_selector(SEL_SRC1_LO, SEL_SRC1_HI)),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(0xCCCC, 0xDDDD))

    def test_scalar_execute_with_modifiers(self):
        self.rf.set_vgpr(1, pack_halves(NEG_ONE, NEG_TWO))
        self.rf.set_vgpr(2, pack_halves(POS_ONE, POS_TWO))
        self.rf.set_vgpr(3, make_selector(SEL_SRC0_LO, SEL_SRC1_LO))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1, abs_mod=True),  # abs(NEG_ONE) -> POS_ONE
            src1=ASTOperand.vgpr(2, neg_mod=True),  # neg(POS_ONE) -> NEG_ONE
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(instr)
        # Low half gets S0.lo with abs -> POS_ONE
        # High half gets S1.lo with neg -> NEG_ONE
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(POS_ONE, NEG_ONE))

    def test_scalar_execute_with_clamp(self):
        self.rf.set_vgpr(1, pack_halves(POS_TWO, NEG_ONE))
        self.rf.set_vgpr(2, pack_halves(POS_HALF, POS_ONE))
        self.rf.set_vgpr(3, make_selector(SEL_SRC0_LO, SEL_SRC0_HI))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
            clamp=True,
        )
        self.interp.execute(instr)
        # S0.lo POS_TWO clamps to POS_ONE; S0.hi NEG_ONE clamps to POS_ZERO
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(POS_ONE, POS_ZERO))

    def test_scalar_execute_sign_extensions(self):
        self.rf.set_vgpr(1, 0x8000_7FFF)  # S0: bit 31=1, bit 15=0
        self.rf.set_vgpr(2, 0x7FFF_8000)  # S1: bit 31=0, bit 15=1
        self.rf.set_vgpr(3, make_selector(SEL_SIGN_EXT_SRC0_HI, SEL_SIGN_EXT_SRC1_LO))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(instr)
        # Half 0 gets S0 bit 31 sign-ext (0xFFFF)
        # Half 1 gets S1 bit 15 sign-ext (0xFFFF)
        self.assertEqual(self.rf.get_vgpr(0), 0xFFFF_FFFF)

    def test_scalar_execute_constants(self):
        self.rf.set_vgpr(3, make_selector(SEL_POS_ZERO, SEL_CANONICAL_QNAN))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.inline_const(0),
            src1=ASTOperand.inline_const(0),
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(POS_ZERO, CANONICAL_QNAN16))


# ==============================================================================
# SIMD Multi-Lane Wave State and 32/64-bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDWaveState:
    """Architectural model of a SIMD Wavefront (Wave32 or Wave64)."""
    num_lanes: int = 32
    exec_mask: Optional[int] = None
    lane_vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def __post_init__(self):
        if self.num_lanes not in (32, 64):
            raise ValueError(f"Wavefront size must be 32 or 64, got {self.num_lanes}")
        if not self.lane_vgprs:
            self.lane_vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = 0xFFFF_FFFF_FFFF_FFFF if self.num_lanes == 64 else MASK32

    def is_lane_active(self, lane: int) -> bool:
        if not 0 <= lane < self.num_lanes:
            raise IndexError(f"Lane out of bounds [0, {self.num_lanes - 1}]: {lane}")
        return bool((self.exec_mask >> lane) & 1)

    def get_lane_vgpr(self, reg: int, lane: int) -> int:
        return self.lane_vgprs[lane][reg] & MASK32

    def set_lane_vgpr(self, reg: int, lane: int, val: int) -> None:
        self.lane_vgprs[lane][reg] = val & MASK32

    def set_wave_vgpr(self, reg: int, vals: List[int]) -> None:
        if len(vals) != self.num_lanes:
            raise ValueError(f"Expected {self.num_lanes} values, got {len(vals)}")
        for lane in range(self.num_lanes):
            self.set_lane_vgpr(reg, lane, vals[lane])

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg] & MASK32

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def read_operand_for_lane(self, op: ASTOperand, lane: int) -> Tuple[int, bool, bool]:
        if op.kind == OperandKind.VGPR:
            val = self.get_lane_vgpr(op.reg_num, lane)
        elif op.kind == OperandKind.SGPR:
            val = self.get_sgpr(op.reg_num)
        elif op.kind in (OperandKind.LITERAL_CONST, OperandKind.INLINE_CONST):
            val = op.constant_val & MASK32
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")
        return val, op.abs_mod, op.neg_mod

    def execute_instruction(self, instr: ASTInstruction) -> None:
        """Execute AST instruction across all active SIMD lanes."""
        instr.validate()
        for lane in range(self.num_lanes):
            if not self.is_lane_active(lane):
                continue

            s0_val, abs0, neg0 = self.read_operand_for_lane(instr.src0, lane)
            s1_val, abs1, neg1 = self.read_operand_for_lane(instr.src1, lane)
            sel_val, _, _ = self.read_operand_for_lane(instr.src2, lane)

            res = oracle_v_perm_f16(
                src0=s0_val,
                src1=s1_val,
                selector=sel_val,
                abs0=abs0,
                neg0=neg0,
                abs1=abs1,
                neg1=neg1,
                clamp=instr.clamp,
            )
            self.set_lane_vgpr(instr.dst.reg_num, lane, res)


# ==============================================================================
# Unit Tests: Commit 4 (SIMD Multi-Lane & Divergence)
# ==============================================================================

class TestPermF16SIMDDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_wave32_all_lanes_active(self):
        wave = SIMDWaveState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        # S0: lane id packed as halves
        wave.set_wave_vgpr(1, [pack_halves(i, i * 2) for i in range(32)])
        wave.set_wave_vgpr(2, [pack_halves(0x100 + i, 0x200 + i) for i in range(32)])
        # S2: identity S1
        sel = PermSelectorF16Word.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            expected = pack_halves(0x100 + lane, 0x200 + lane)
            self.assertEqual(wave.get_lane_vgpr(0, lane), expected)

    def test_simd_wave64_all_lanes_active(self):
        wave = SIMDWaveState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        wave.set_wave_vgpr(1, [pack_halves(i, 0x4000) for i in range(64)])
        wave.set_wave_vgpr(2, [pack_halves(0x3C00, i) for i in range(64)])
        # Swap halves of S0
        sel = PermSelectorF16Word.swap_halves_src0().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(64):
            expected = pack_halves(0x4000, lane)
            self.assertEqual(wave.get_lane_vgpr(0, lane), expected)

    def test_simd_wave32_divergent_even_odd_lanes(self):
        # Even lanes active (0x55555555)
        wave = SIMDWaveState(num_lanes=32, exec_mask=0x5555_5555)
        # Pre-fill destination with poison value
        wave.set_wave_vgpr(0, [0xDEAD_BEEF] * 32)
        wave.set_wave_vgpr(1, [pack_halves(POS_ONE, POS_TWO)] * 32)
        wave.set_wave_vgpr(2, [pack_halves(POS_HALF, MIN_NORMAL)] * 32)
        sel = PermSelectorF16Word.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.get_lane_vgpr(0, lane), pack_halves(POS_HALF, MIN_NORMAL))
            else:
                self.assertEqual(wave.get_lane_vgpr(0, lane), 0xDEAD_BEEF)

    def test_simd_wave64_divergent_high_low_half(self):
        # Lower 32 lanes active, upper 32 inactive
        wave = SIMDWaveState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)
        wave.set_wave_vgpr(0, [0xCAFE_BABE] * 64)
        wave.set_wave_vgpr(1, [pack_halves(POS_TWO, POS_TWO)] * 64)
        wave.set_wave_vgpr(2, [pack_halves(POS_ONE, POS_ONE)] * 64)
        sel = PermSelectorF16Word.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(64):
            if lane < 32:
                self.assertEqual(wave.get_lane_vgpr(0, lane), pack_halves(POS_ONE, POS_ONE))
            else:
                self.assertEqual(wave.get_lane_vgpr(0, lane), 0xCAFE_BABE)

    def test_simd_single_active_lane(self):
        # Only lane 17 active
        wave = SIMDWaveState(num_lanes=32, exec_mask=(1 << 17))
        wave.set_wave_vgpr(0, [0x0000_0000] * 32)
        wave.set_wave_vgpr(1, [pack_halves(0x1111, 0x2222)] * 32)
        wave.set_wave_vgpr(2, [pack_halves(0x3333, 0x4444)] * 32)
        sel = PermSelectorF16Word.identity_src0().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            if lane == 17:
                self.assertEqual(wave.get_lane_vgpr(0, lane), pack_halves(0x1111, 0x2222))
            else:
                self.assertEqual(wave.get_lane_vgpr(0, lane), 0x0000_0000)

    def test_simd_zero_exec_mask_no_modification(self):
        wave = SIMDWaveState(num_lanes=32, exec_mask=0x0000_0000)
        initial_vgpr0 = [0x55AA_55AA + i for i in range(32)]
        wave.set_wave_vgpr(0, initial_vgpr0)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(0),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr(0, lane), initial_vgpr0[lane])

    def test_simd_per_lane_divergent_selectors(self):
        wave = SIMDWaveState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        wave.set_wave_vgpr(1, [pack_halves(0xAAAA, 0xBBBB)] * 32)
        wave.set_wave_vgpr(2, [pack_halves(0xCCCC, 0xDDDD)] * 32)
        # Even lanes pick S1, odd lanes pick S0
        sel_even = PermSelectorF16Word.identity_src1().raw
        sel_odd = PermSelectorF16Word.identity_src0().raw
        wave.set_wave_vgpr(3, [sel_even if i % 2 == 0 else sel_odd for i in range(32)])

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            expected = pack_halves(0xCCCC, 0xDDDD) if lane % 2 == 0 else pack_halves(0xAAAA, 0xBBBB)
            self.assertEqual(wave.get_lane_vgpr(0, lane), expected)


# ==============================================================================
# Unit Tests: Commit 5 (Register Aliasing & In-Place Operand Preservation)
# ==============================================================================

class TestPermF16RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing hazards and in-place operand preservation."""

    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_dst_aliases_src0(self):
        # D == src0: v0 = perm(v0, v1, sel)
        # v0 = [0x1111, 0x2222], v1 = [0x3333, 0x4444]
        # sel picks S1.lo (v1.lo = 0x3333) and S0.hi (v0.hi = 0x2222) -> [0x3333, 0x2222]
        self.rf.set_vgpr(0, pack_halves(0x1111, 0x2222))
        self.rf.set_vgpr(1, pack_halves(0x3333, 0x4444))
        sel = make_selector(SEL_SRC1_LO, SEL_SRC0_HI)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(0x3333, 0x2222))

    def test_dst_aliases_src1(self):
        # D == src1: v0 = perm(v1, v0, sel)
        # v1 = [0xAAAA, 0xBBBB], v0 = [0xCCCC, 0xDDDD]
        # sel picks S1.hi (v0.hi = 0xDDDD) and S0.lo (v1.lo = 0xAAAA) -> [0xDDDD, 0xAAAA]
        self.rf.set_vgpr(1, pack_halves(0xAAAA, 0xBBBB))
        self.rf.set_vgpr(0, pack_halves(0xCCCC, 0xDDDD))
        sel = make_selector(SEL_SRC1_HI, SEL_SRC0_LO)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(0xDDDD, 0xAAAA))

    def test_dst_aliases_src2_selector(self):
        # D == src2: selector is in v0 and destination is v0
        # v1 = [0x1234, 0x5678], v2 = [0x9ABC, 0xDEF0]
        # v0 contains selector: pick S1.lo (v2.lo = 0x9ABC) and S0.hi (v1.hi = 0x5678)
        sel = make_selector(SEL_SRC1_LO, SEL_SRC0_HI)
        self.rf.set_vgpr(1, pack_halves(0x1234, 0x5678))
        self.rf.set_vgpr(2, pack_halves(0x9ABC, 0xDEF0))
        self.rf.set_vgpr(0, sel)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(0),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(0x9ABC, 0x5678))

    def test_dst_aliases_src0_and_src1(self):
        # D == src0 == src1: in-place half-swap
        # v0 = [POS_ONE, POS_TWO]
        # selector swaps halves of S1 -> [POS_TWO, POS_ONE]
        self.rf.set_vgpr(0, pack_halves(POS_ONE, POS_TWO))
        sel = PermSelectorF16Word.swap_halves_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(POS_TWO, POS_ONE))

    def test_dst_aliases_all_sources(self):
        # D == src0 == src1 == src2: single register v0 for all operands
        # To be valid, v0's bytes must decode into a valid selector.
        # Let v0 = make_selector(SEL_SRC1_LO, SEL_SRC1_LO) = 0x0000 -> broadcast Half 0!
        # Half 0 of 0x0000_0000 is 0x0000 (+0.0)
        self.rf.set_vgpr(0, 0x0000_0000)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(0),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), 0x0000_0000)

    def test_inplace_swizzle_half_words(self):
        # Swapping high and low halves in-place within v3
        self.rf.set_vgpr(3, pack_halves(0xCAFE, 0xBABE))
        sel = PermSelectorF16Word.swap_halves_src0().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(3),
            src0=ASTOperand.vgpr(3),
            src1=ASTOperand.vgpr(3),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(3), pack_halves(0xBABE, 0xCAFE))

    def test_inplace_broadcast_and_sign_ext(self):
        # S0 = [0x7FFF, 0x8000] (bit 31 is 1, bit 15 is 0)
        # S2 picks S0 bit 31 sign-ext and S0.lo broadcast
        self.rf.set_vgpr(7, 0x8000_7FFF)
        sel = make_selector(SEL_SIGN_EXT_SRC0_HI, SEL_SRC0_LO)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(7),
            src0=ASTOperand.vgpr(7),
            src1=ASTOperand.vgpr(7),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        # Low half gets sign-ext of bit 31 (0xFFFF), high half gets S0.lo (0x7FFF)
        self.assertEqual(self.rf.get_vgpr(7), pack_halves(0xFFFF, 0x7FFF))

    def test_simd_multi_lane_aliasing(self):
        wave = SIMDWaveState(num_lanes=32)
        # In-place half reverse on v0 across all 32 lanes
        wave.set_wave_vgpr(0, [pack_halves(i, i * 10) for i in range(32)])
        sel = PermSelectorF16Word.swap_halves_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr(0, lane), pack_halves(lane * 10, lane))


# ==============================================================================
# Condition Codes, Architectural Status Flags, and FP Classifications
# ==============================================================================

@dataclass
class FPStatusFlags:
    """Architectural floating-point status flags produced by VALU execution."""
    zero: bool = False
    nan: bool = False
    subnormal: bool = False
    negative: bool = False
    infinity: bool = False


def is_nan16(bits: int) -> bool:
    """Return True if 16-bit word represents an IEEE 754 binary16 NaN."""
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & FRAC_MASK16) != 0)


def is_inf16(bits: int) -> bool:
    """Return True if 16-bit word represents positive or negative infinity."""
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & FRAC_MASK16) == 0)


def is_subnormal16(bits: int) -> bool:
    """Return True if 16-bit word represents a subnormal (denormal) half float."""
    return ((bits & EXP_MASK16) == 0) and ((bits & FRAC_MASK16) != 0)


def is_zero16(bits: int) -> bool:
    """Return True if 16-bit word represents +0.0 or -0.0."""
    return (bits & ~SIGN16) == 0


def eval_fp_status(val32: int) -> FPStatusFlags:
    """Evaluate floating-point condition flags for a packed pair of f16 values."""
    h0 = val32 & MASK16
    h1 = (val32 >> 16) & MASK16

    both_zero = is_zero16(h0) and is_zero16(h1)
    has_nan = is_nan16(h0) or is_nan16(h1)
    has_subnormal = is_subnormal16(h0) or is_subnormal16(h1)
    has_inf = is_inf16(h0) or is_inf16(h1)
    is_neg = bool(h0 & SIGN16) or bool(h1 & SIGN16)

    return FPStatusFlags(
        zero=both_zero,
        nan=has_nan,
        subnormal=has_subnormal,
        negative=is_neg,
        infinity=has_inf,
    )


# ==============================================================================
# Unit Tests: Commit 6 (Condition Codes & Status Flags)
# ==============================================================================

class TestPermF16ConditionCodesAndStatus(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow generation."""

    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_status_flags_zero_result(self):
        # Permute resulting in [+0.0, -0.0]
        sel = make_selector(SEL_POS_ZERO, SEL_NEG_ZERO)
        res = oracle_v_perm_f16(0, 0, sel)
        flags = eval_fp_status(res)
        self.assertTrue(flags.zero)
        self.assertTrue(flags.negative)  # -0.0 has sign bit set
        self.assertFalse(flags.nan)
        self.assertFalse(flags.subnormal)
        self.assertFalse(flags.infinity)

    def test_status_flags_nan_detection(self):
        # Permute injecting canonical Quiet NaN into high half
        sel = make_selector(SEL_POS_ONE, SEL_CANONICAL_QNAN)
        res = oracle_v_perm_f16(0, 0, sel)
        flags = eval_fp_status(res)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.nan)
        self.assertFalse(flags.infinity)

    def test_status_flags_subnormal_detection(self):
        # S1 has subnormal in low half
        s1 = pack_halves(MIN_SUBNORMAL, POS_ONE)
        sel = PermSelectorF16Word.identity_src1().raw
        res = oracle_v_perm_f16(0, s1, sel)
        flags = eval_fp_status(res)
        self.assertTrue(flags.subnormal)
        self.assertFalse(flags.nan)
        self.assertFalse(flags.zero)

    def test_status_flags_negative_sign(self):
        s1 = pack_halves(NEG_ONE, POS_ONE)
        sel = PermSelectorF16Word.identity_src1().raw
        res = oracle_v_perm_f16(0, s1, sel)
        flags = eval_fp_status(res)
        self.assertTrue(flags.negative)

    def test_status_flags_infinity_detection(self):
        s1 = pack_halves(POS_INF, NEG_INF)
        sel = PermSelectorF16Word.identity_src1().raw
        res = oracle_v_perm_f16(0, s1, sel)
        flags = eval_fp_status(res)
        self.assertTrue(flags.infinity)
        self.assertFalse(flags.nan)

    def test_scc_preserved_by_valu_perm(self):
        # In RDNA2 VALU semantics, v_perm_f16 never touches SCC (Scalar Condition Code)
        self.rf.scc = 1  # Set SCC
        self.rf.set_vgpr(1, pack_halves(POS_ONE, POS_ONE))
        self.rf.set_vgpr(2, pack_halves(POS_TWO, POS_TWO))
        sel = PermSelectorF16Word.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        # SCC must remain 1
        self.assertEqual(self.rf.scc, 1)

    def test_vcc_condition_code_comparison_after_perm(self):
        # Simulate downstream v_cmp_gt_f16 testing if high half > low half
        s1 = pack_halves(POS_ONE, POS_TWO)
        sel_swap = PermSelectorF16Word.swap_halves_src1().raw
        res = oracle_v_perm_f16(0, s1, sel_swap)

        h0 = bits_to_f16(extract_half(res, 0))  # 2.0
        h1 = bits_to_f16(extract_half(res, 1))  # 1.0

        vcc_bit = 1 if h1 > h0 else 0
        self.assertEqual(vcc_bit, 0)

    def test_simd_lane_divergent_vcc_generation(self):
        # Generate wave VCC mask from permuted lane results
        wave = SIMDWaveState(num_lanes=32)
        wave.set_wave_vgpr(1, [pack_halves(i, i * 2) for i in range(32)])
        sel = PermSelectorF16Word.identity_src0().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.inline_const(0),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        # Compute VCC mask: lane is 1 if hi half > lo half (which is true for i > 0)
        vcc_mask = 0
        for lane in range(32):
            val = wave.get_lane_vgpr(0, lane)
            lo = extract_half(val, 0)
            hi = extract_half(val, 1)
            if hi > lo:
                vcc_mask |= (1 << lane)

        # For i=0: 0 > 0 is False. For i in 1..31: 2*i > i is True.
        expected_vcc = 0xFFFF_FFFE
        self.assertEqual(vcc_mask, expected_vcc)


# ==============================================================================
# Unit Tests: Commit 7 (IEEE 754 Boundaries, Subnormals, and NaN Handling)
# ==============================================================================

class TestPermF16IEEE754Boundaries(unittest.TestCase):
    """Test suite for IEEE 754 boundary extremes, subnormals, and NaN preservation."""

    def test_subnormal_half_preservation(self):
        # Permute must preserve subnormal bit patterns exactly
        src1 = pack_halves(MIN_SUBNORMAL, MAX_SUBNORMAL)
        src0 = pack_halves(MIN_SUBNORMAL_NEG, MAX_SUBNORMAL_NEG)
        sel = PermSelectorF16Word.identity_src1().raw
        res = oracle_v_perm_f16(src0, src1, sel)
        self.assertEqual(extract_half(res, 0), MIN_SUBNORMAL)
        self.assertEqual(extract_half(res, 1), MAX_SUBNORMAL)

        # Cross-select negative subnormals
        sel_neg = make_selector(SEL_SRC0_LO, SEL_SRC0_HI)
        res_neg = oracle_v_perm_f16(src0, src1, sel_neg)
        self.assertEqual(extract_half(res_neg, 0), MIN_SUBNORMAL_NEG)
        self.assertEqual(extract_half(res_neg, 1), MAX_SUBNORMAL_NEG)

    def test_infinities_preservation(self):
        src1 = pack_halves(POS_INF, NEG_INF)
        sel = PermSelectorF16Word.swap_halves_src1().raw
        res = oracle_v_perm_f16(0, src1, sel)
        self.assertEqual(extract_half(res, 0), NEG_INF)
        self.assertEqual(extract_half(res, 1), POS_INF)

    def test_quiet_nan_payload_preservation(self):
        # Custom NaN payloads: quiet NaN with payload bits
        qnan1 = 0x7E55  # Exponent all 1s, quiet bit set, payload 0x55
        qnan2 = 0x7EFF  # Exponent all 1s, quiet bit set, payload 0xFF
        src0 = pack_halves(qnan1, qnan2)
        sel = PermSelectorF16Word.identity_src0().raw
        res = oracle_v_perm_f16(src0, 0, sel)
        self.assertEqual(extract_half(res, 0), qnan1)
        self.assertEqual(extract_half(res, 1), qnan2)

    def test_signaling_nan_payload_preservation(self):
        # Signaling NaNs: quiet bit 9 is 0, payload non-zero
        snan1 = 0x7C01  # Smallest signaling NaN
        snan2 = 0x7DF0  # Signaling NaN with custom payload
        src1 = pack_halves(snan1, snan2)
        sel = PermSelectorF16Word.identity_src1().raw
        res = oracle_v_perm_f16(0, src1, sel)
        # Permutation does not quiet signaling NaNs
        self.assertEqual(extract_half(res, 0), snan1)
        self.assertEqual(extract_half(res, 1), snan2)

    def test_signed_zero_sign_preservation(self):
        src1 = pack_halves(POS_ZERO, NEG_ZERO)
        sel = PermSelectorF16Word.identity_src1().raw
        res = oracle_v_perm_f16(0, src1, sel)
        self.assertEqual(extract_half(res, 0), POS_ZERO)
        self.assertEqual(extract_half(res, 1), NEG_ZERO)
        self.assertEqual(extract_half(res, 0) & SIGN16, 0)
        self.assertEqual(extract_half(res, 1) & SIGN16, SIGN16)

    def test_max_min_finite_boundaries(self):
        src0 = pack_halves(MAX_FINITE, MIN_NORMAL)
        sel = PermSelectorF16Word.identity_src0().raw
        res = oracle_v_perm_f16(src0, 0, sel)
        self.assertEqual(extract_half(res, 0), MAX_FINITE)
        self.assertEqual(extract_half(res, 1), MIN_NORMAL)

    def test_ieee754_modifiers_on_nan_and_inf(self):
        # abs modifier clears sign bit even on NaNs and -Inf
        src1 = pack_halves(NEG_INF, NEG_QNAN)
        sel = PermSelectorF16Word.identity_src1().raw
        res = oracle_v_perm_f16(0, src1, sel, abs1=True)
        self.assertEqual(extract_half(res, 0), POS_INF)
        self.assertEqual(extract_half(res, 1), POS_QNAN)

        # neg modifier inverts sign bit of +Inf to -Inf
        src0 = pack_halves(POS_INF, POS_QNAN)
        sel0 = PermSelectorF16Word.identity_src0().raw
        res_neg = oracle_v_perm_f16(src0, 0, sel0, neg0=True)
        self.assertEqual(extract_half(res_neg, 0), NEG_INF)
        self.assertEqual(extract_half(res_neg, 1), NEG_QNAN)

    def test_clamp_behavior_on_nans_and_infinities(self):
        # In RDNA2, clamping NaN yields +0.0; clamping +Inf yields 1.0; clamping -Inf yields +0.0
        src1 = pack_halves(POS_QNAN, POS_INF)
        src0 = pack_halves(NEG_INF, NEG_TWO)
        sel = make_selector(SEL_SRC1_LO, SEL_SRC1_HI)
        res1 = oracle_v_perm_f16(src0, src1, sel, clamp=True)
        self.assertEqual(extract_half(res1, 0), POS_ZERO)  # NaN -> +0.0
        self.assertEqual(extract_half(res1, 1), POS_ONE)   # +Inf -> 1.0

        sel2 = make_selector(SEL_SRC0_LO, SEL_SRC0_HI)
        res2 = oracle_v_perm_f16(src0, src1, sel2, clamp=True)
        self.assertEqual(extract_half(res2, 0), POS_ZERO)  # -Inf -> +0.0
        self.assertEqual(extract_half(res2, 1), POS_ZERO)  # -2.0 -> +0.0


if __name__ == "__main__":
    unittest.main()






