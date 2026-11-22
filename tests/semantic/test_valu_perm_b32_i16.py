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

# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class Opcode(Enum):
    V_PERM_B32_I16 = auto()
    V_NOP = auto()


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL_CONST = auto()
    INLINE_CONST = auto()


class DataType(Enum):
    I16 = auto()
    V2I16 = auto()
    U32 = auto()


@dataclass
class ASTOperand:
    kind: OperandKind
    reg_num: int = 0
    constant_val: int = 0
    data_type: DataType = DataType.V2I16

    @classmethod
    def vgpr(cls, num: int, dtype: DataType = DataType.V2I16) -> ASTOperand:
        if not 0 <= num <= 255:
            raise IndexError(f"VGPR index out of range: {num}")
        return cls(kind=OperandKind.VGPR, reg_num=num, data_type=dtype)

    @classmethod
    def sgpr(cls, num: int, dtype: DataType = DataType.V2I16) -> ASTOperand:
        if not 0 <= num <= 105:
            raise IndexError(f"SGPR index out of range: {num}")
        return cls(kind=OperandKind.SGPR, reg_num=num, data_type=dtype)

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
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    src2: ASTOperand

    def validate(self) -> None:
        if self.dst.kind != OperandKind.VGPR:
            raise ValueError(f"Destination must be a VGPR, got {self.dst.kind}")
        self.dst.validate()
        self.src0.validate()
        self.src1.validate()
        self.src2.validate()

        literals = sum(1 for src in (self.src0, self.src1, self.src2) if src.kind == OperandKind.LITERAL_CONST)
        if literals > 1:
            raise ValueError(f"Hardware encoding permits at most one literal constant, got {literals}")


class TestPermB32I16SymbolicAST(unittest.TestCase):
    def test_ast_instruction_creation(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_I16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.sgpr(0),
        )
        self.assertEqual(instr.opcode, Opcode.V_PERM_B32_I16)
        instr.validate()

    def test_ast_instruction_validation_rejects_sgpr_dst(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_I16,
            dst=ASTOperand.sgpr(5),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(0x0100),
        )
        with self.assertRaises(ValueError):
            instr.validate()

# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarRegisterFile:
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    scc: int = 0
    vcc: int = 0
    exec_mask: int = MASK32

    def get_vgpr(self, idx: int) -> int:
        if not 0 <= idx <= 255:
            raise IndexError(f"VGPR index out of range: {idx}")
        return self.vgprs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        if not 0 <= idx <= 255:
            raise IndexError(f"VGPR index out of range: {idx}")
        self.vgprs[idx] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        if not 0 <= idx <= 105:
            raise IndexError(f"SGPR index out of range: {idx}")
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        if not 0 <= idx <= 105:
            raise IndexError(f"SGPR index out of range: {idx}")
        self.sgprs[idx] = val & MASK32

    def read_operand(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.get_vgpr(op.reg_num)
        elif op.kind == OperandKind.SGPR:
            return self.get_sgpr(op.reg_num)
        elif op.kind in (OperandKind.LITERAL_CONST, OperandKind.INLINE_CONST):
            return op.constant_val & MASK32
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")


class ScalarInterpreter:
    def __init__(self, reg_file: Optional[ScalarRegisterFile] = None):
        self.rf = reg_file if reg_file is not None else ScalarRegisterFile()

    def execute(self, instr: ASTInstruction) -> None:
        instr.validate()
        if instr.opcode == Opcode.V_PERM_B32_I16:
            s0_val = self.rf.read_operand(instr.src0)
            s1_val = self.rf.read_operand(instr.src1)
            sel_val = self.rf.read_operand(instr.src2)

            res = oracle_v_perm_b32_i16(src0=s0_val, src1=s1_val, selector=sel_val)
            self.rf.set_vgpr(instr.dst.reg_num, res)


class TestPermB32I16ScalarInterpreter(unittest.TestCase):
    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_scalar_execute_basic_perm(self):
        self.rf.set_vgpr(1, pack_halves(0x1111, 0x2222))
        self.rf.set_vgpr(2, pack_halves(0x3333, 0x4444))
        self.rf.set_vgpr(3, make_selector(SEL_SRC1_LO, SEL_SRC0_HI))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_I16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(0x3333, 0x2222))

# ==============================================================================
# SIMD Multi-Lane Wave State and 32/64-bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDWaveState:
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
        return bool((self.exec_mask >> lane) & 1)

    def get_lane_vgpr(self, reg: int, lane: int) -> int:
        return self.lane_vgprs[lane][reg] & MASK32

    def set_lane_vgpr(self, reg: int, lane: int, val: int) -> None:
        self.lane_vgprs[lane][reg] = val & MASK32

    def set_wave_vgpr(self, reg: int, vals: List[int]) -> None:
        for lane in range(self.num_lanes):
            self.set_lane_vgpr(reg, lane, vals[lane])

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg] & MASK32

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def read_operand_for_lane(self, op: ASTOperand, lane: int) -> int:
        if op.kind == OperandKind.VGPR:
            return self.get_lane_vgpr(op.reg_num, lane)
        elif op.kind == OperandKind.SGPR:
            return self.get_sgpr(op.reg_num)
        elif op.kind in (OperandKind.LITERAL_CONST, OperandKind.INLINE_CONST):
            return op.constant_val & MASK32
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")

    def execute_instruction(self, instr: ASTInstruction) -> None:
        instr.validate()
        for lane in range(self.num_lanes):
            if not self.is_lane_active(lane):
                continue

            s0_val = self.read_operand_for_lane(instr.src0, lane)
            s1_val = self.read_operand_for_lane(instr.src1, lane)
            sel_val = self.read_operand_for_lane(instr.src2, lane)

            res = oracle_v_perm_b32_i16(src0=s0_val, src1=s1_val, selector=sel_val)
            self.set_lane_vgpr(instr.dst.reg_num, lane, res)


class TestPermB32I16SIMDDivergence(unittest.TestCase):
    def test_simd_wave32_divergence(self):
        wave = SIMDWaveState(num_lanes=32, exec_mask=0x5555_5555)
        wave.set_wave_vgpr(0, [0xDEAD_BEEF] * 32)
        wave.set_wave_vgpr(1, [pack_halves(INT16_ONE, INT16_NEG_ONE)] * 32)
        wave.set_wave_vgpr(2, [pack_halves(INT16_MIN, INT16_MAX)] * 32)
        sel = PermSelectorWord.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_I16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        wave.execute_instruction(instr)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.get_lane_vgpr(0, lane), pack_halves(INT16_MIN, INT16_MAX))
            else:
                self.assertEqual(wave.get_lane_vgpr(0, lane), 0xDEAD_BEEF)

# ==============================================================================
# Register Aliasing & In-Place Operand Preservation
# ==============================================================================

class TestPermB32I16RegisterAliasing(unittest.TestCase):
    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_dst_aliases_src0(self):
        self.rf.set_vgpr(0, pack_halves(0x1111, 0x2222))
        self.rf.set_vgpr(1, pack_halves(0x3333, 0x4444))
        sel = make_selector(SEL_SRC1_LO, SEL_SRC0_HI)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_I16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(0x3333, 0x2222))

# ==============================================================================
# Condition Codes, Status Flags, and Classifications
# ==============================================================================

@dataclass
class I16StatusFlags:
    zero: bool = False
    negative: bool = False
    has_min: bool = False
    has_max: bool = False


def eval_i16_status(val32: int) -> I16StatusFlags:
    h0 = val32 & MASK16
    h1 = (val32 >> 16) & MASK16

    both_zero = (h0 == 0) and (h1 == 0)
    is_neg = bool(h0 & SIGN16) or bool(h1 & SIGN16)
    has_min = (h0 == INT16_MIN) or (h1 == INT16_MIN)
    has_max = (h0 == INT16_MAX) or (h1 == INT16_MAX)

    return I16StatusFlags(
        zero=both_zero,
        negative=is_neg,
        has_min=has_min,
        has_max=has_max,
    )


class TestPermB32I16ConditionCodesAndStatus(unittest.TestCase):
    def test_status_flags_zero_result(self):
        sel = make_selector(SEL_CONST_ZERO, SEL_CONST_ZERO)
        res = oracle_v_perm_b32_i16(0, 0, sel)
        flags = eval_i16_status(res)
        self.assertTrue(flags.zero)
        self.assertFalse(flags.negative)

    def test_status_flags_min_max(self):
        sel = make_selector(SEL_CONST_MIN, SEL_CONST_MAX)
        res = oracle_v_perm_b32_i16(0, 0, sel)
        flags = eval_i16_status(res)
        self.assertTrue(flags.has_min)
        self.assertTrue(flags.has_max)

if __name__ == "__main__":
    unittest.main()
