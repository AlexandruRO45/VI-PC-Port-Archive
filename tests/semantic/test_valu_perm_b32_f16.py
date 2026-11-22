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

# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class Opcode(Enum):
    V_PERM_B32_F16 = auto()
    V_NOP = auto()


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL_CONST = auto()
    INLINE_CONST = auto()


class DataType(Enum):
    F16 = auto()
    V2F16 = auto()
    U32 = auto()


@dataclass
class ASTOperand:
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
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    src2: ASTOperand
    clamp: bool = False

    def validate(self) -> None:
        if self.dst.kind != OperandKind.VGPR:
            raise ValueError(f"VALU instruction destination must be a VGPR, got {self.dst.kind}")
        self.dst.validate()
        self.src0.validate()
        self.src1.validate()
        self.src2.validate()

        literals = sum(1 for src in (self.src0, self.src1, self.src2) if src.kind == OperandKind.LITERAL_CONST)
        if literals > 1:
            raise ValueError(f"Hardware encoding permits at most one literal constant, got {literals}")


@dataclass
class ASTExpression:
    opcode: Opcode
    operands: List[Union[ASTOperand, ASTExpression]]
    clamp: bool = False

    def to_symbolic_string(self) -> str:
        ops_str = ", ".join(
            op.to_symbolic_string() if isinstance(op, ASTExpression)
            else f"{op.kind.name}{op.reg_num if op.kind in (OperandKind.VGPR, OperandKind.SGPR) else f'({hex(op.constant_val)})'}"
            for op in self.operands
        )
        clamp_str = " [clamp]" if self.clamp else ""
        return f"{self.opcode.name}({ops_str}){clamp_str}"


class TestPermB32F16SymbolicAST(unittest.TestCase):
    def test_ast_instruction_creation(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.sgpr(0),
            clamp=True,
        )
        self.assertEqual(instr.opcode, Opcode.V_PERM_B32_F16)
        self.assertTrue(instr.clamp)

    def test_ast_operand_factories(self):
        v = ASTOperand.vgpr(42, abs_mod=True)
        self.assertEqual(v.kind, OperandKind.VGPR)
        self.assertTrue(v.abs_mod)

    def test_ast_instruction_validation_rejects_sgpr_dst(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
            dst=ASTOperand.sgpr(5),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(0x0100),
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_instruction_validation_rejects_multiple_literals(self):
        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
            dst=ASTOperand.vgpr(5),
            src0=ASTOperand.literal(0x1111_1111),
            src1=ASTOperand.literal(0x2222_2222),
            src2=ASTOperand.vgpr(0),
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
    status: int = 0
    mode: int = 0

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

    def read_operand(self, op: ASTOperand) -> Tuple[int, bool, bool]:
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
    def __init__(self, reg_file: Optional[ScalarRegisterFile] = None):
        self.rf = reg_file if reg_file is not None else ScalarRegisterFile()

    def execute(self, instr: ASTInstruction) -> None:
        instr.validate()
        if instr.opcode == Opcode.V_PERM_B32_F16:
            s0_val, abs0, neg0 = self.rf.read_operand(instr.src0)
            s1_val, abs1, neg1 = self.rf.read_operand(instr.src1)
            sel_val, _, _ = self.rf.read_operand(instr.src2)

            res = oracle_v_perm_b32_f16(
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


class TestPermB32F16ScalarInterpreter(unittest.TestCase):
    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_scalar_execute_basic_perm(self):
        self.rf.set_vgpr(1, pack_halves(0x1111, 0x2222))
        self.rf.set_vgpr(2, pack_halves(0x3333, 0x4444))
        self.rf.set_vgpr(3, make_selector(SEL_SRC1_LO, SEL_SRC0_HI))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(0x3333, 0x2222))

    def test_scalar_execute_with_clamp(self):
        self.rf.set_vgpr(1, pack_halves(POS_TWO, NEG_ONE))
        self.rf.set_vgpr(2, pack_halves(POS_HALF, POS_ONE))
        self.rf.set_vgpr(3, make_selector(SEL_SRC0_LO, SEL_SRC0_HI))

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(3),
            clamp=True,
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(POS_ONE, POS_ZERO))

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
        instr.validate()
        for lane in range(self.num_lanes):
            if not self.is_lane_active(lane):
                continue

            s0_val, abs0, neg0 = self.read_operand_for_lane(instr.src0, lane)
            s1_val, abs1, neg1 = self.read_operand_for_lane(instr.src1, lane)
            sel_val, _, _ = self.read_operand_for_lane(instr.src2, lane)

            res = oracle_v_perm_b32_f16(
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


class TestPermB32F16SIMDDivergence(unittest.TestCase):
    def test_simd_wave32_divergent_even_odd_lanes(self):
        wave = SIMDWaveState(num_lanes=32, exec_mask=0x5555_5555)
        wave.set_wave_vgpr(0, [0xDEAD_BEEF] * 32)
        wave.set_wave_vgpr(1, [pack_halves(POS_ONE, POS_TWO)] * 32)
        wave.set_wave_vgpr(2, [pack_halves(POS_HALF, MIN_NORMAL)] * 32)
        sel = PermSelectorWord.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
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
        wave = SIMDWaveState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)
        wave.set_wave_vgpr(0, [0xCAFE_BABE] * 64)
        wave.set_wave_vgpr(1, [pack_halves(POS_TWO, POS_TWO)] * 64)
        wave.set_wave_vgpr(2, [pack_halves(POS_ONE, POS_ONE)] * 64)
        sel = PermSelectorWord.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
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

# ==============================================================================
# Register Aliasing & In-Place Operand Preservation
# ==============================================================================

class TestPermB32F16RegisterAliasing(unittest.TestCase):
    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_dst_aliases_src0(self):
        self.rf.set_vgpr(0, pack_halves(0x1111, 0x2222))
        self.rf.set_vgpr(1, pack_halves(0x3333, 0x4444))
        sel = make_selector(SEL_SRC1_LO, SEL_SRC0_HI)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(1),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(0x3333, 0x2222))

    def test_dst_aliases_src1(self):
        self.rf.set_vgpr(1, pack_halves(0xAAAA, 0xBBBB))
        self.rf.set_vgpr(0, pack_halves(0xCCCC, 0xDDDD))
        sel = make_selector(SEL_SRC1_HI, SEL_SRC0_LO)

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(0xDDDD, 0xAAAA))

    def test_dst_aliases_src0_and_src1(self):
        self.rf.set_vgpr(0, pack_halves(POS_ONE, POS_TWO))
        sel = PermSelectorWord.swap_halves_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.get_vgpr(0), pack_halves(POS_TWO, POS_ONE))

# ==============================================================================
# Condition Codes, Status Flags, and FP Classifications
# ==============================================================================

@dataclass
class FPStatusFlags:
    zero: bool = False
    nan: bool = False
    subnormal: bool = False
    negative: bool = False
    infinity: bool = False


def is_nan16(bits: int) -> bool:
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & FRAC_MASK16) != 0)


def is_inf16(bits: int) -> bool:
    return ((bits & EXP_MASK16) == EXP_MASK16) and ((bits & FRAC_MASK16) == 0)


def is_subnormal16(bits: int) -> bool:
    return ((bits & EXP_MASK16) == 0) and ((bits & FRAC_MASK16) != 0)


def is_zero16(bits: int) -> bool:
    return (bits & ~SIGN16) == 0


def eval_fp_status(val32: int) -> FPStatusFlags:
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


class TestPermB32F16ConditionCodesAndStatus(unittest.TestCase):
    def setUp(self):
        self.rf = ScalarRegisterFile()
        self.interp = ScalarInterpreter(self.rf)

    def test_status_flags_zero_result(self):
        sel = make_selector(SEL_POS_ZERO, SEL_NEG_ZERO)
        res = oracle_v_perm_b32_f16(0, 0, sel)
        flags = eval_fp_status(res)
        self.assertTrue(flags.zero)
        self.assertTrue(flags.negative)

    def test_scc_preserved_by_valu_perm(self):
        self.rf.scc = 1
        self.rf.set_vgpr(1, pack_halves(POS_ONE, POS_ONE))
        self.rf.set_vgpr(2, pack_halves(POS_TWO, POS_TWO))
        sel = PermSelectorWord.identity_src1().raw

        instr = ASTInstruction(
            opcode=Opcode.V_PERM_B32_F16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.literal(sel),
        )
        self.interp.execute(instr)
        self.assertEqual(self.rf.scc, 1)

if __name__ == "__main__":
    unittest.main()
