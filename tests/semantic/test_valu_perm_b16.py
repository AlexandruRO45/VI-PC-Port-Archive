from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_perm_b16.

Tests the vector 16-bit byte permutation instruction:
    v_perm_b16 D, S0, S1, S2
Lowered semantics:
    Destination is a 16-bit integer holding two bytes (D[7:0] as Byte 0, D[15:8] as Byte 1).
    The source pool is formed by the two 16-bit inputs {S0[15:0], S1[15:0]}:
    - Byte 0: S1[7:0]
    - Byte 1: S1[15:8]
    - Byte 2: S0[7:0]
    - Byte 3: S0[15:8]

    Selector S2 provides selection control codes for each destination byte:
    - S2[7:0]: selector code for destination Byte 0 (D[7:0])
    - S2[15:8]: selector code for destination Byte 1 (D[15:8])

    Selector codes:
    - 0: selects Byte 0 (S1[7:0])
    - 1: selects Byte 1 (S1[15:8])
    - 2: selects Byte 2 (S0[7:0])
    - 3: selects Byte 3 (S0[15:8])
    - 4: sign-extension of S1 bit 15 (0xFF if negative, else 0x00)
    - 5: sign-extension of S0 bit 7 (0xFF if negative, else 0x00)
    - 6: sign-extension of S0 bit 15 (0xFF if negative, else 0x00)
    - 7: sign-extension of S1 bit 7 (0xFF if negative, else 0x00)
    - 8: constant 0x00
    - >= 9: constant 0xFF

Architectural model:
    16-bit destination and 16-bit operands with sub-dword packing,
    selector decoding, and bit-exact execution. Standard library only; zero game data.
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
BYTE_MASK: int = 0xFF
BYTE_WIDTH: int = 8

# Selector codes
SEL_SRC1_B0: int = 0            # S1[7:0]
SEL_SRC1_B1: int = 1            # S1[15:8]
SEL_SRC0_B0: int = 2            # S0[7:0]
SEL_SRC0_B1: int = 3            # S0[15:8]

SEL_SIGN_EXT_SRC1_B1: int = 4   # S1[15] replicated over byte
SEL_SIGN_EXT_SRC0_B0: int = 5   # S0[7] replicated over byte
SEL_SIGN_EXT_SRC0_B1: int = 6   # S0[15] replicated over byte
SEL_SIGN_EXT_SRC1_B0: int = 7   # S1[7] replicated over byte

SEL_ZERO: int = 8               # Constant 0x00
SEL_ONES_MIN: int = 9           # Constant 0xFF (>= 9)


# ==============================================================================
# Bit Manipulation and Byte Extraction Utilities
# ==============================================================================

def extract_byte_16(val: int, idx: int) -> int:
    """Extract byte idx (0 for [7:0], 1 for [15:8]) from an unsigned 16-bit integer."""
    if idx not in (0, 1):
        raise ValueError(f"Byte index out of range [0, 1]: {idx}")
    return (val >> (idx * BYTE_WIDTH)) & BYTE_MASK


def pack_bytes_16(b0: int, b1: int) -> int:
    """Pack two unsigned bytes (b0 low, b1 high) into an unsigned 16-bit integer."""
    return ((b0 & BYTE_MASK) | ((b1 & BYTE_MASK) << BYTE_WIDTH)) & MASK16


def make_selector_16(b0: int, b1: int) -> int:
    """Construct a 16-bit selector word from 2 individual byte selector values."""
    return pack_bytes_16(b0, b1)


def permute_single_byte_16(src0: int, src1: int, sel_code: int) -> int:
    """Evaluate a single byte according to RDNA2 v_perm_b16 hardware semantics."""
    sel = sel_code & BYTE_MASK
    if sel == SEL_SRC1_B0:
        return src1 & BYTE_MASK
    elif sel == SEL_SRC1_B1:
        return (src1 >> 8) & BYTE_MASK
    elif sel == SEL_SRC0_B0:
        return src0 & BYTE_MASK
    elif sel == SEL_SRC0_B1:
        return (src0 >> 8) & BYTE_MASK
    elif sel == SEL_SIGN_EXT_SRC1_B1:
        return 0xFF if bool(src1 & 0x8000) else 0x00
    elif sel == SEL_SIGN_EXT_SRC0_B0:
        return 0xFF if bool(src0 & 0x0080) else 0x00
    elif sel == SEL_SIGN_EXT_SRC0_B1:
        return 0xFF if bool(src0 & 0x8000) else 0x00
    elif sel == SEL_SIGN_EXT_SRC1_B0:
        return 0xFF if bool(src1 & 0x0080) else 0x00
    elif sel == SEL_ZERO:
        return 0x00
    else:  # sel >= SEL_ONES_MIN (9..255)
        return 0xFF


def oracle_v_perm_b16(src0: int, src1: int, selector: int) -> int:
    """Reference mathematical oracle for RDNA2 v_perm_b16.
    
    src0: upper pool providing Byte 2 (bits 0..7) and Byte 3 (bits 8..15)
    src1: lower pool providing Byte 0 (bits 0..7) and Byte 1 (bits 8..15)
    selector: control word where byte 0 selects Byte 0, byte 1 selects Byte 1
    """
    s0 = src0 & MASK16
    s1 = src1 & MASK16
    sel = selector & MASK16

    sel0 = sel & BYTE_MASK
    sel1 = (sel >> BYTE_WIDTH) & BYTE_MASK

    b0 = permute_single_byte_16(s0, s1, sel0)
    b1 = permute_single_byte_16(s0, s1, sel1)

    return pack_bytes_16(b0, b1)


# ==============================================================================
# Architectural Register and Word Models
# ==============================================================================

@dataclass
class VGPRWord16:
    """Architectural representation of a 16-bit halfword within a VGPR."""
    raw: int = 0

    @property
    def u16(self) -> int:
        return self.raw & MASK16

    @property
    def byte0(self) -> int:
        return self.raw & BYTE_MASK

    @property
    def byte1(self) -> int:
        return (self.raw >> 8) & BYTE_MASK

    def set_byte(self, idx: int, val8: int) -> None:
        if idx == 0:
            self.raw = ((self.raw & 0xFF00) | (val8 & BYTE_MASK)) & MASK16
        elif idx == 1:
            self.raw = ((self.raw & 0x00FF) | ((val8 & BYTE_MASK) << 8)) & MASK16
        else:
            raise ValueError(f"Invalid byte index: {idx}")


@dataclass
class PermSelectorB16Word:
    """Structured representation of a 16-bit permutation selector word for v_perm_b16."""
    sel0: int = 0  # S2[7:0]
    sel1: int = 0  # S2[15:8]

    @property
    def raw(self) -> int:
        return make_selector_16(self.sel0, self.sel1)

    @classmethod
    def identity_src1(cls) -> PermSelectorB16Word:
        """Selects Byte 0 to low, Byte 1 to high (passes S1 through)."""
        return cls(sel0=SEL_SRC1_B0, sel1=SEL_SRC1_B1)

    @classmethod
    def identity_src0(cls) -> PermSelectorB16Word:
        """Selects Byte 2 to low, Byte 3 to high (passes S0 through)."""
        return cls(sel0=SEL_SRC0_B0, sel1=SEL_SRC0_B1)

    @classmethod
    def bswap_src1(cls) -> PermSelectorB16Word:
        """Reverses byte order of S1."""
        return cls(sel0=SEL_SRC1_B1, sel1=SEL_SRC1_B0)

    @classmethod
    def bswap_src0(cls) -> PermSelectorB16Word:
        """Reverses byte order of S0."""
        return cls(sel0=SEL_SRC0_B1, sel1=SEL_SRC0_B0)

    @classmethod
    def broadcast_byte(cls, byte_idx: int) -> PermSelectorB16Word:
        """Replicates a single byte index (0..3) across both destination bytes."""
        return cls(sel0=byte_idx, sel1=byte_idx)


# ==============================================================================
# Unit Tests: Commit 1 (Reference Oracle & Architectural Model)
# ==============================================================================

class TestPermB16ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural models."""

    def test_identity_src1_selection(self):
        src0 = 0xAABB
        src1 = 0x1122
        sel = PermSelectorB16Word.identity_src1().raw
        self.assertEqual(oracle_v_perm_b16(src0, src1, sel), 0x1122)

    def test_identity_src0_selection(self):
        src0 = 0xAABB
        src1 = 0x1122
        sel = PermSelectorB16Word.identity_src0().raw
        self.assertEqual(oracle_v_perm_b16(src0, src1, sel), 0xAABB)

    def test_cross_source_interleaving(self):
        # Pick Byte 0 of S1 (0x22) and Byte 1 of S0 (0xAA) -> 0xAA22
        src0 = 0xAABB
        src1 = 0x1122
        sel = make_selector_16(SEL_SRC1_B0, SEL_SRC0_B1)
        self.assertEqual(oracle_v_perm_b16(src0, src1, sel), 0xAA22)

    def test_endianness_bswap(self):
        # Reverse S1: 0x1234 -> 0x3412
        src1 = 0x1234
        sel_rev1 = PermSelectorB16Word.bswap_src1().raw
        self.assertEqual(oracle_v_perm_b16(0, src1, sel_rev1), 0x3412)

        # Reverse S0: 0x5678 -> 0x7856
        src0 = 0x5678
        sel_rev0 = PermSelectorB16Word.bswap_src0().raw
        self.assertEqual(oracle_v_perm_b16(src0, 0, sel_rev0), 0x7856)

    def test_byte_broadcast(self):
        # Broadcast Byte 0 of S1 (0x44)
        src1 = 0x3344
        sel0 = PermSelectorB16Word.broadcast_byte(SEL_SRC1_B0).raw
        self.assertEqual(oracle_v_perm_b16(0, src1, sel0), 0x4444)

        # Broadcast Byte 3 of pool (Byte 1 of S0 = 0x88)
        src0 = 0x8899
        sel3 = PermSelectorB16Word.broadcast_byte(SEL_SRC0_B1).raw
        self.assertEqual(oracle_v_perm_b16(src0, 0, sel3), 0x8888)

    def test_constant_zero_fill(self):
        src0 = 0xFFFF
        src1 = 0xFFFF
        sel_zero = make_selector_16(SEL_ZERO, SEL_ZERO)
        self.assertEqual(oracle_v_perm_b16(src0, src1, sel_zero), 0x0000)

        # Mix zero fill with selected byte
        sel_mix = make_selector_16(SEL_SRC1_B0, SEL_ZERO)
        self.assertEqual(oracle_v_perm_b16(0, 0x1234, sel_mix), 0x0034)

    def test_constant_ones_fill(self):
        src0 = 0x0000
        src1 = 0x0000
        sel_ones = make_selector_16(9, 255)
        self.assertEqual(oracle_v_perm_b16(src0, src1, sel_ones), 0xFFFF)

    def test_sign_extension_modes(self):
        # Code 4: sign of S1 bit 15
        s1_pos = 0x7FFF
        s1_neg = 0x8000
        sel4 = make_selector_16(SEL_SIGN_EXT_SRC1_B1, SEL_SIGN_EXT_SRC1_B1)
        self.assertEqual(oracle_v_perm_b16(0, s1_pos, sel4), 0x0000)
        self.assertEqual(oracle_v_perm_b16(0, s1_neg, sel4), 0xFFFF)

        # Code 5: sign of S0 bit 7
        s0_pos7 = 0x007F
        s0_neg7 = 0x0080
        sel5 = make_selector_16(SEL_SIGN_EXT_SRC0_B0, SEL_SIGN_EXT_SRC0_B0)
        self.assertEqual(oracle_v_perm_b16(s0_pos7, 0, sel5), 0x0000)
        self.assertEqual(oracle_v_perm_b16(s0_neg7, 0, sel5), 0xFFFF)

        # Code 6: sign of S0 bit 15
        s0_pos15 = 0x7FFF
        s0_neg15 = 0x8000
        sel6 = make_selector_16(SEL_SIGN_EXT_SRC0_B1, SEL_SIGN_EXT_SRC0_B1)
        self.assertEqual(oracle_v_perm_b16(s0_pos15, 0, sel6), 0x0000)
        self.assertEqual(oracle_v_perm_b16(s0_neg15, 0, sel6), 0xFFFF)

        # Code 7: sign of S1 bit 7
        s1_pos7 = 0x007F
        s1_neg7 = 0x0080
        sel7 = make_selector_16(SEL_SIGN_EXT_SRC1_B0, SEL_SIGN_EXT_SRC1_B0)
        self.assertEqual(oracle_v_perm_b16(0, s1_pos7, sel7), 0x0000)
        self.assertEqual(oracle_v_perm_b16(0, s1_neg7, sel7), 0xFFFF)

    def test_vgpr_word_16_model(self):
        vw = VGPRWord16(0x1234)
        self.assertEqual(vw.u16, 0x1234)
        self.assertEqual(vw.byte0, 0x34)
        self.assertEqual(vw.byte1, 0x12)

        vw.set_byte(0, 0xAA)
        self.assertEqual(vw.u16, 0x12AA)
        vw.set_byte(1, 0xFF)
        self.assertEqual(vw.u16, 0xFFAA)

    def test_selector_word_16_model(self):
        sw = PermSelectorB16Word(sel0=2, sel1=3)
        self.assertEqual(sw.sel0, 2)
        self.assertEqual(sw.sel1, 3)
        self.assertEqual(sw.raw, 0x0302)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class Opcode(Enum):
    """VALU instruction opcodes modeled for semantic analysis."""
    V_PERM_B16 = auto()
    V_NOP = auto()


class OperandKind(Enum):
    """Hardware operand categories for RDNA2 instructions."""
    VGPR = auto()
    SGPR = auto()
    LITERAL_CONST = auto()
    INLINE_CONST = auto()


class DataType(Enum):
    """Architectural data types for operands and intermediate evaluations."""
    U16 = auto()
    B16 = auto()
    U32 = auto()


@dataclass
class ASTOperand:
    """Symbolic AST node representing an instruction operand."""
    kind: OperandKind
    reg_num: int = 0
    constant_val: int = 0
    data_type: DataType = DataType.B16

    @classmethod
    def vgpr(cls, num: int, dtype: DataType = DataType.B16) -> ASTOperand:
        if not 0 <= num <= 255:
            raise IndexError(f"VGPR index out of range [0, 255]: {num}")
        return cls(kind=OperandKind.VGPR, reg_num=num, data_type=dtype)

    @classmethod
    def sgpr(cls, num: int, dtype: DataType = DataType.B16) -> ASTOperand:
        if not 0 <= num <= 105:
            raise IndexError(f"SGPR index out of range [0, 105]: {num}")
        return cls(kind=OperandKind.SGPR, reg_num=num, data_type=dtype)

    @classmethod
    def literal(cls, val: int, dtype: DataType = DataType.U16) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL_CONST, constant_val=val & MASK16, data_type=dtype)

    @classmethod
    def inline_const(cls, val: int, dtype: DataType = DataType.U16) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, constant_val=val & MASK16, data_type=dtype)

    def validate(self) -> None:
        if self.kind == OperandKind.VGPR and not (0 <= self.reg_num <= 255):
            raise IndexError(f"VGPR out of bounds: {self.reg_num}")
        if self.kind == OperandKind.SGPR and not (0 <= self.reg_num <= 105):
            raise IndexError(f"SGPR out of bounds: {self.reg_num}")


@dataclass
class ASTInstruction:
    """Symbolic AST node representing a complete v_perm_b16 instruction."""
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    src2: ASTOperand

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

    def to_symbolic_string(self) -> str:
        ops_str = ", ".join(
            op.to_symbolic_string() if isinstance(op, ASTExpression)
            else f"{op.kind.name}{op.reg_num if op.kind in (OperandKind.VGPR, OperandKind.SGPR) else f'({hex(op.constant_val)})'}"
            for op in self.operands
        )
        return f"{self.opcode.name}({ops_str})"


# ==============================================================================
# Unit Tests: Commit 2 (Symbolic AST & Typed Expressions)
# ==============================================================================

class TestPermB16SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_instruction_creation(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.sgpr(0),
        )
        self.assertEqual(instr.opcode, Opcode.V_PERM_B16)
        self.assertEqual(instr.dst.reg_num, 0)
        self.assertEqual(instr.src0.reg_num, 1)
        self.assertEqual(instr.src1.reg_num, 2)
        self.assertEqual(instr.src2.reg_num, 0)

    def test_ast_operand_factories(self):
        v = ASTOperand.vgpr(42)
        self.assertEqual(v.kind, OperandKind.VGPR)
        self.assertEqual(v.reg_num, 42)

        s = ASTOperand.sgpr(10)
        self.assertEqual(s.kind, OperandKind.SGPR)
        self.assertEqual(s.reg_num, 10)

        lit = ASTOperand.literal(0x1234)
        self.assertEqual(lit.kind, OperandKind.LITERAL_CONST)
        self.assertEqual(lit.constant_val, 0x1234)

        inline = ASTOperand.inline_const(0)
        self.assertEqual(inline.kind, OperandKind.INLINE_CONST)
        self.assertEqual(inline.constant_val, 0)

    def test_ast_instruction_validation_valid(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(5),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.sgpr(2),
            src2=ASTOperand.literal(0x0100),
        )
        instr.validate()

    def test_ast_instruction_validation_rejects_sgpr_dst(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.sgpr(5),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(0x0100),
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_instruction_validation_rejects_multiple_literals(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(5),
            src0=ASTOperand.literal(0x1111),
            src1=ASTOperand.literal(0x2222),
            src2=ASTOperand.vgpr(0),
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_expression_tree_representation(self):
        expr = ASTExpression(
            opcode=Opcode.V_PERM_B16,
            operands=[
                ASTOperand.vgpr(1),
                ASTOperand.vgpr(2),
                ASTOperand.literal(0x0302),
            ],
        )
        sym_str = expr.to_symbolic_string()
        self.assertIn("V_PERM_B16", sym_str)
        self.assertIn("VGPR1", sym_str)
        self.assertIn("0x302", sym_str)


if __name__ == "__main__":
    unittest.main()

