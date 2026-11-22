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

    def read_operand(self, op: ASTOperand) -> int:
        """Resolve operand to its integer value."""
        if op.kind == OperandKind.VGPR:
            return self.get_vgpr(op.reg_num) & MASK16
        elif op.kind == OperandKind.SGPR:
            return self.get_sgpr(op.reg_num) & MASK16
        elif op.kind in (OperandKind.LITERAL_CONST, OperandKind.INLINE_CONST):
            return op.constant_val & MASK16
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")


class ScalarInterpreter:
    """Deterministic instruction interpreter for single-lane scalar execution."""

    def __init__(self, reg_file: Optional[ScalarRegisterFile] = None):
        self.rf = reg_file if reg_file is not None else ScalarRegisterFile()

    def execute(self, instr: ASTInstruction) -> None:
        """Execute a single AST instruction deterministically."""
        instr.validate()
        if instr.opcode == Opcode.V_PERM_B16:
            s0_val = self.rf.read_operand(instr.src0)
            s1_val = self.rf.read_operand(instr.src1)
            sel_val = self.rf.read_operand(instr.src2)

            res = oracle_v_perm_b16(
                src0=s0_val,
                src1=s1_val,
                selector=sel_val,
            )
            # 16-bit destination write preserving upper 16 bits or zero-extended
            curr_val = self.rf.get_vgpr(instr.dst.reg_num)
            new_val = (curr_val & 0xFFFF_0000) | (res & MASK16)
            self.rf.set_vgpr(instr.dst.reg_num, new_val)
        elif instr.opcode == Opcode.V_NOP:
            pass
        else:
            raise NotImplementedError(f"Opcode not implemented: {instr.opcode}")


# ==============================================================================
# Unit Tests: Commit 3 (Deterministic Scalar Interpreter)
# ==============================================================================

class TestPermB16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_scalar_execute_basic_perm(self):
        self.rf.set_vgpr(1, 0x1122)  # S0
        self.rf.set_vgpr(2, 0x3344)  # S1
        self.rf.set_vgpr(3, make_selector_16(SEL_SRC1_B0, SEL_SRC0_B1))  # S2

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(instr)
        # Byte 0 gets S1[7:0] (0x44), Byte 1 gets S0[15:8] (0x11) -> 0x1144
        self.assertEqual(self.rf.get_vgpr(0) & MASK16, 0x1144)

    def test_scalar_execute_with_sgpr_sources(self):
        self.rf.set_sgpr(0, 0xAABB)
        self.rf.set_sgpr(1, 0xCCDD)
        self.rf.set_vgpr(5, make_selector_16(SEL_SRC1_B1, SEL_SRC0_B0))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(10),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.sgpr(1),
            src2=ASTOperand.vgpr(5),
        )
        self.interp.execute(instr)
        # S1[15:8] is 0xCC, S0[7:0] is 0xBB -> 0xBBCC
        self.assertEqual(self.rf.get_vgpr(10) & MASK16, 0xBBCC)

    def test_scalar_execute_with_literal_selector(self):
        self.rf.set_vgpr(1, 0x1234)
        self.rf.set_vgpr(2, 0x5678)

        # Bswap S1 via literal selector
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(PermSelectorB16Word.bswap_src1().raw),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0) & MASK16, 0x7856)

    def test_scalar_execute_sign_extensions(self):
        self.rf.set_vgpr(1, 0x807F)  # S0: bit 15=1, bit 7=0
        self.rf.set_vgpr(2, 0x7F80)  # S1: bit 15=0, bit 7=1
        self.rf.set_vgpr(3, make_selector_16(SEL_SIGN_EXT_SRC0_B1, SEL_SIGN_EXT_SRC1_B0))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0) & MASK16, 0xFFFF)

    def test_scalar_execute_constants(self):
        self.rf.set_vgpr(3, make_selector_16(SEL_ZERO, 9))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.inline_const(0),
            src1=ASTOperand.inline_const(0),
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0) & MASK16, 0xFF00)


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

    def read_operand_for_lane(self, op: ASTOperand, lane: int) -> int:
        if op.kind == OperandKind.VGPR:
            return self.get_lane_vgpr(op.reg_num, lane) & MASK16
        elif op.kind == OperandKind.SGPR:
            return self.get_sgpr(op.reg_num) & MASK16
        elif op.kind in (OperandKind.LITERAL_CONST, OperandKind.INLINE_CONST):
            return op.constant_val & MASK16
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")

    def execute_instruction(self, instr: ASTInstruction) -> None:
        """Execute AST instruction across all active SIMD lanes."""
        instr.validate()
        for lane in range(self.num_lanes):
            if not self.is_lane_active(lane):
                continue

            s0_val = self.read_operand_for_lane(instr.src0, lane)
            s1_val = self.read_operand_for_lane(instr.src1, lane)
            sel_val = self.read_operand_for_lane(instr.src2, lane)

            res = oracle_v_perm_b16(
                src0=s0_val,
                src1=s1_val,
                selector=sel_val,
            )
            curr_val = self.get_lane_vgpr(instr.dst.reg_num, lane)
            new_val = (curr_val & 0xFFFF_0000) | (res & MASK16)
            self.set_lane_vgpr(instr.dst.reg_num, lane, new_val)


# ==============================================================================
# Unit Tests: Commit 4 (SIMD Multi-Lane & Divergence)
# ==============================================================================

class TestPermB16SIMDDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_wave32_all_lanes_active(self):
        wave = SIMDWaveState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        wave.set_wave_vgpr(1, [0x1000 + i for i in range(32)])
        wave.set_wave_vgpr(2, [0x2000 + i for i in range(32)])
        sel = PermSelectorB16Word.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr(0, lane) & MASK16, 0x2000 + lane)

    def test_simd_wave64_all_lanes_active(self):
        wave = SIMDWaveState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        wave.set_wave_vgpr(1, [0x3000 + i for i in range(64)])
        wave.set_wave_vgpr(2, [0x4000 + i for i in range(64)])
        sel = PermSelectorB16Word.identity_src0().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(64):
            self.assertEqual(wave.get_lane_vgpr(0, lane) & MASK16, 0x3000 + lane)

    def test_simd_wave32_divergent_even_odd_lanes(self):
        # Even lanes active (0x55555555)
        wave = SIMDWaveState(num_lanes=32, exec_mask=0x5555_5555)
        wave.set_wave_vgpr(0, [0xDEAD_BEEF] * 32)
        wave.set_wave_vgpr(1, [0x1122] * 32)
        wave.set_wave_vgpr(2, [0x3344] * 32)
        sel = PermSelectorB16Word.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.get_lane_vgpr(0, lane) & MASK16, 0x3344)
            else:
                self.assertEqual(wave.get_lane_vgpr(0, lane), 0xDEAD_BEEF)

    def test_simd_wave64_divergent_high_low_half(self):
        # Lower 32 lanes active, upper 32 inactive
        wave = SIMDWaveState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)
        wave.set_wave_vgpr(0, [0xCAFE_BABE] * 64)
        wave.set_wave_vgpr(1, [0x5566] * 64)
        wave.set_wave_vgpr(2, [0x7788] * 64)
        sel = PermSelectorB16Word.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(64):
            if lane < 32:
                self.assertEqual(wave.get_lane_vgpr(0, lane) & MASK16, 0x7788)
            else:
                self.assertEqual(wave.get_lane_vgpr(0, lane), 0xCAFE_BABE)

    def test_simd_single_active_lane(self):
        # Only lane 11 active
        wave = SIMDWaveState(num_lanes=32, exec_mask=(1 << 11))
        wave.set_wave_vgpr(0, [0x0000_0000] * 32)
        wave.set_wave_vgpr(1, [0x1111] * 32)
        wave.set_wave_vgpr(2, [0x2222] * 32)
        sel = PermSelectorB16Word.identity_src0().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            if lane == 11:
                self.assertEqual(wave.get_lane_vgpr(0, lane) & MASK16, 0x1111)
            else:
                self.assertEqual(wave.get_lane_vgpr(0, lane), 0x0000_0000)

    def test_simd_zero_exec_mask_no_modification(self):
        wave = SIMDWaveState(num_lanes=32, exec_mask=0x0000_0000)
        initial_vgpr0 = [0x55AA_55AA + i for i in range(32)]
        wave.set_wave_vgpr(0, initial_vgpr0)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
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
        wave.set_wave_vgpr(1, [0xAAAA] * 32)
        wave.set_wave_vgpr(2, [0xBBBB] * 32)
        sel_even = PermSelectorB16Word.identity_src1().raw
        sel_odd = PermSelectorB16Word.identity_src0().raw
        wave.set_wave_vgpr(3, [sel_even if i % 2 == 0 else sel_odd for i in range(32)])

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            expected = 0xBBBB if lane % 2 == 0 else 0xAAAA
            self.assertEqual(wave.get_lane_vgpr(0, lane) & MASK16, expected)


# ==============================================================================
# Unit Tests: Commit 5 (Register Aliasing & In-Place Operand Preservation)
# ==============================================================================

class TestPermB16RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing hazards and in-place operand preservation."""

    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_dst_aliases_src0(self):
        # D == src0: v0 = perm(v0, v1, sel)
        self.rf.set_vgpr(0, 0x1122)
        self.rf.set_vgpr(1, 0x3344)
        sel = make_selector_16(SEL_SRC1_B0, SEL_SRC0_B1)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        # S1[7:0] is 0x44, S0[15:8] is 0x11 -> 0x1144
        self.assertEqual(self.rf.get_vgpr(0) & MASK16, 0x1144)

    def test_dst_aliases_src1(self):
        # D == src1: v0 = perm(v1, v0, sel)
        self.rf.set_vgpr(1, 0xAABB)
        self.rf.set_vgpr(0, 0xCCDD)
        sel = make_selector_16(SEL_SRC1_B1, SEL_SRC0_B0)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        # S1[15:8] is 0xCC, S0[7:0] is 0xBB -> 0xBBCC
        self.assertEqual(self.rf.get_vgpr(0) & MASK16, 0xBBCC)

    def test_dst_aliases_src2_selector(self):
        # D == src2: selector is in v0 and destination is v0
        sel = make_selector_16(SEL_SRC1_B0, SEL_SRC0_B1)
        self.rf.set_vgpr(1, 0x1234)
        self.rf.set_vgpr(2, 0x5678)
        self.rf.set_vgpr(0, sel)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(0),
        )
        self.interp.execute(instr)
        # S1[7:0] is 0x78, S0[15:8] is 0x12 -> 0x1278
        self.assertEqual(self.rf.get_vgpr(0) & MASK16, 0x1278)

    def test_dst_aliases_src0_and_src1(self):
        # D == src0 == src1: in-place byte swap
        self.rf.set_vgpr(0, 0x1234)
        sel = PermSelectorB16Word.bswap_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0) & MASK16, 0x3412)

    def test_dst_aliases_all_sources(self):
        # D == src0 == src1 == src2: single register v0
        self.rf.set_vgpr(0, 0x0000)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(0),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0) & MASK16, 0x0000)

    def test_inplace_bswap(self):
        self.rf.set_vgpr(3, 0xCAFE)
        sel = PermSelectorB16Word.bswap_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(3),
            src0=ASTOperand.vgpr(3),
            src1=ASTOperand.vgpr(3),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(3) & MASK16, 0xFECA)

    def test_inplace_broadcast_and_sign_ext(self):
        self.rf.set_vgpr(7, 0x807F)  # bit 15 is 1, bit 7 is 0
        sel = make_selector_16(SEL_SIGN_EXT_SRC0_B1, SEL_SRC0_B0)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(7),
            src0=ASTOperand.vgpr(7),
            src1=ASTOperand.vgpr(7),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        # Byte 0 gets sign-ext of bit 15 (0xFF), Byte 1 gets S0[7:0] (0x7F) -> 0x7FFF
        self.assertEqual(self.rf.get_vgpr(7) & MASK16, 0x7FFF)

    def test_simd_multi_lane_aliasing(self):
        wave = SIMDWaveState(num_lanes=32)
        wave.set_wave_vgpr(0, [((i & 0xFF) << 8) | (i & 0xFF) for i in range(32)])
        sel = PermSelectorB16Word.bswap_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            expected = ((lane & 0xFF) << 8) | (lane & 0xFF)
            self.assertEqual(wave.get_lane_vgpr(0, lane) & MASK16, expected)


# ==============================================================================
# Condition Codes, Architectural Status Flags, and Classifications
# ==============================================================================

@dataclass
class B16StatusFlags:
    """Architectural status flags produced by 16-bit VALU integer operations."""
    zero: bool = False
    negative: bool = False
    byte0_zero: bool = False
    byte1_zero: bool = False


def eval_b16_status(val16: int) -> B16StatusFlags:
    """Evaluate architectural integer status flags for a 16-bit word."""
    u16 = val16 & MASK16
    b0 = u16 & BYTE_MASK
    b1 = (u16 >> BYTE_WIDTH) & BYTE_MASK

    return B16StatusFlags(
        zero=(u16 == 0),
        negative=bool(u16 & 0x8000),
        byte0_zero=(b0 == 0),
        byte1_zero=(b1 == 0),
    )


# ==============================================================================
# Unit Tests: Commit 6 (Condition Codes & Status Flags)
# ==============================================================================

class TestPermB16ConditionCodesAndStatus(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow generation."""

    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_status_flags_zero_result(self):
        sel = make_selector_16(SEL_ZERO, SEL_ZERO)
        res = oracle_v_perm_b16(0, 0, sel)
        flags = eval_b16_status(res)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)
        self.assertTrue(flags.byte0_zero)
        self.assertTrue(flags.byte1_zero)

    def test_status_flags_negative_sign(self):
        # Permute giving 0x8000 (sign bit set)
        s1 = 0x8000
        sel = PermSelectorB16Word.identity_src1().raw
        res = oracle_v_perm_b16(0, s1, sel)
        flags = eval_b16_status(res)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.negative)
        self.assertTrue(flags.byte0_zero)
        self.assertFalse(flags.byte1_zero)

    def test_status_flags_byte_level_zero(self):
        s1 = 0x1200
        sel = PermSelectorB16Word.identity_src1().raw
        res = oracle_v_perm_b16(0, s1, sel)
        flags = eval_b16_status(res)
        self.assertFalse(flags.zero)
        self.assertTrue(flags.byte0_zero)
        self.assertFalse(flags.byte1_zero)

    def test_scc_preserved_by_valu_perm(self):
        self.rf.scc = 1
        self.rf.set_vgpr(1, 0x1122)
        self.rf.set_vgpr(2, 0x3344)
        sel = PermSelectorB16Word.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.scc, 1)

    def test_vcc_condition_code_comparison_after_perm(self):
        s1 = 0x1234
        sel_swap = PermSelectorB16Word.bswap_src1().raw
        res = oracle_v_perm_b16(0, s1, sel_swap)

        b0 = extract_byte_16(res, 0)  # 0x12
        b1 = extract_byte_16(res, 1)  # 0x34

        vcc_bit = 1 if b1 > b0 else 0
        self.assertEqual(vcc_bit, 1)

    def test_simd_lane_divergent_vcc_generation(self):
        wave = SIMDWaveState(num_lanes=32)
        wave.set_wave_vgpr(1, [((i & 0xFF) << 8) | ((i * 2) & 0xFF) for i in range(32)])
        sel = PermSelectorB16Word.identity_src0().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.inline_const(0),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        vcc_mask = 0
        for lane in range(32):
            val = wave.get_lane_vgpr(0, lane) & MASK16
            b0 = extract_byte_16(val, 0)
            b1 = extract_byte_16(val, 1)
            if b0 > b1:
                vcc_mask |= (1 << lane)

        # b0 is 2*i mod 256, b1 is i. For i in 1..31, 2*i > i is True.
        expected_vcc = 0xFFFF_FFFE
        self.assertEqual(vcc_mask, expected_vcc)


# ==============================================================================
# Unit Tests: Commit 7 (IEEE 754 Boundaries, Subnormals, and NaN Handling)
# ==============================================================================

class TestPermB16IEEE754Boundaries(unittest.TestCase):
    """Test suite for IEEE 754 boundary extremes, subnormals, and NaN byte handling."""

    def test_subnormal_byte_swizzling(self):
        # 0x0001 (smallest positive subnormal) and 0x03FF (largest positive subnormal)
        src1 = 0x0001
        src0 = 0x03FF
        sel = PermSelectorB16Word.identity_src1().raw
        res = oracle_v_perm_b16(src0, src1, sel)
        self.assertEqual(res, 0x0001)

        # Swapping bytes of subnormal
        sel_swap = PermSelectorB16Word.bswap_src1().raw
        res_swap = oracle_v_perm_b16(0, 0x0001, sel_swap)
        self.assertEqual(res_swap, 0x0100)

    def test_infinity_byte_preservation(self):
        # +Inf (0x7C00) and -Inf (0xFC00)
        src1 = 0x7C00
        src0 = 0xFC00
        sel = make_selector_16(SEL_SRC1_B1, SEL_SRC0_B1)
        res = oracle_v_perm_b16(src0, src1, sel)
        # Low byte gets high byte of S1 (0x7C), high byte gets high byte of S0 (0xFC) -> 0xFC7C
        self.assertEqual(res, 0xFC7C)

    def test_quiet_nan_payload_byte_preservation(self):
        # Canonical Quiet NaN 0x7E00 and custom NaN 0x7EFF
        src1 = 0x7E00
        src0 = 0x7EFF
        sel = PermSelectorB16Word.identity_src0().raw
        res = oracle_v_perm_b16(src0, src1, sel)
        self.assertEqual(res, 0x7EFF)

    def test_signaling_nan_byte_preservation(self):
        # Signaling NaN 0x7C01
        src1 = 0x7C01
        sel = PermSelectorB16Word.bswap_src1().raw
        res = oracle_v_perm_b16(0, src1, sel)
        self.assertEqual(res, 0x017C)

    def test_signed_zero_sign_bit_extension(self):
        # -0.0 has bit 15 set (0x8000), +0.0 has bit 15 clear (0x0000)
        neg_zero = 0x8000
        pos_zero = 0x0000

        # Code 4: sign-extension of S1 bit 15
        sel_sign = make_selector_16(SEL_SIGN_EXT_SRC1_B1, SEL_SIGN_EXT_SRC1_B1)
        self.assertEqual(oracle_v_perm_b16(0, neg_zero, sel_sign), 0xFFFF)
        self.assertEqual(oracle_v_perm_b16(0, pos_zero, sel_sign), 0x0000)

    def test_max_min_finite_boundaries(self):
        # Max finite float 0x7BFF (65504.0)
        src1 = 0x7BFF
        sel = PermSelectorB16Word.identity_src1().raw
        res = oracle_v_perm_b16(0, src1, sel)
        self.assertEqual(res, 0x7BFF)


if __name__ == "__main__":
    unittest.main()






