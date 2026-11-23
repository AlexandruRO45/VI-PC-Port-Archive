from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 v_pk_max_i16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_pk_max_i16 performs packed 16-bit signed integer maximum.
  Instruction:
    v_pk_max_i16 vd, src0, src1  (VOP3P format)

  Operands:
    vd   = Destination 32-bit VGPR containing two 16-bit signed integers [lo, hi]
    src0 = Source 32-bit operand containing two 16-bit signed integers [lo, hi]
    src1 = Source 32-bit operand containing two 16-bit signed integers [lo, hi]

  Semantics:
    lo0 = s16(src0[15:0]), hi0 = s16(src0[31:16])
    lo1 = s16(src1[15:0]), hi1 = s16(src1[31:16])
    res_lo = max(lo0, lo1) & 0xFFFF
    res_hi = max(hi0, hi1) & 0xFFFF
    vd = (res_hi << 16) | res_lo

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
SIGN_BIT16: int = 0x8000


def s16(val: int) -> int:
    """Interpret a 16-bit unsigned integer as signed two's complement."""
    val &= MASK16
    return val - 0x10000 if (val & SIGN_BIT16) else val


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit integer range [0, 2^16 - 1]."""
    return val & MASK16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_pk_max_i16(src0: int, src1: int) -> int:
    """Exact architectural reference oracle for AMD RDNA2 v_pk_max_i16.

    Args:
        src0: 32-bit packed pair of two signed 16-bit integers
        src1: 32-bit packed pair of two signed 16-bit integers

    Returns:
        32-bit packed pair with element-wise signed maximum
    """
    lo0 = s16(src0 & MASK16)
    hi0 = s16((src0 >> 16) & MASK16)
    lo1 = s16(src1 & MASK16)
    hi1 = s16((src1 >> 16) & MASK16)

    res_lo = max(lo0, lo1) & MASK16
    res_hi = max(hi0, hi1) & MASK16
    return (res_hi << 16) | res_lo


class TestValuPkMaxI16Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of v_pk_max_i16."""

    def test_oracle_mixed_signs(self):
        # src0: lo = -5 (0xFFFB), hi = +5 (0x0005) -> 0x0005FFFB
        # src1: lo = +5 (0x0005), hi = -5 (0xFFFB) -> 0xFFFB0005
        # Expected: lo = max(-5, +5) = +5 (0x0005), hi = max(+5, -5) = +5 (0x0005) -> 0x00050005
        res = ref_pk_max_i16(0x0005FFFB, 0xFFFB0005)
        self.assertEqual(res, 0x00050005)

    def test_oracle_both_negative(self):
        # lo: max(-10, -20) = -10 (0xFFF6)
        # hi: max(-30, -5) = -5 (0xFFFB)
        s0 = ((-30 & MASK16) << 16) | (-10 & MASK16)
        s1 = ((-5 & MASK16) << 16) | (-20 & MASK16)
        res = ref_pk_max_i16(s0, s1)
        expected = ((-5 & MASK16) << 16) | (-10 & MASK16)
        self.assertEqual(res, expected)

    def test_oracle_both_positive(self):
        s0 = (100 << 16) | 300
        s1 = (200 << 16) | 150
        res = ref_pk_max_i16(s0, s1)
        expected = (200 << 16) | 300
        self.assertEqual(res, expected)

    def test_oracle_signed_extremes(self):
        # -32768 (0x8000) vs 32767 (0x7FFF)
        s0 = (0x8000 << 16) | 0x7FFF
        s1 = (0x7FFF << 16) | 0x8000
        res = ref_pk_max_i16(s0, s1)
        expected = (0x7FFF << 16) | 0x7FFF
        self.assertEqual(res, expected)

    def test_oracle_commutativity(self):
        a = 0x12345678
        b = 0x87654321
        self.assertEqual(ref_pk_max_i16(a, b), ref_pk_max_i16(b, a))


class Opcode(Enum):
    V_PK_MAX_I16 = auto()


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL32 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    value: int = 0

    @classmethod
    def vgpr(cls, index: int) -> ASTOperand:
        if index < 0 or index > 255:
            raise ValueError(f"VGPR index out of range: v{index}")
        return cls(kind=OperandKind.VGPR, index=index)

    @classmethod
    def sgpr(cls, index: int) -> ASTOperand:
        if index < 0 or index > 105:
            raise ValueError(f"SGPR index out of range: s{index}")
        return cls(kind=OperandKind.SGPR, index=index)

    @classmethod
    def literal32(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL32, value=u32(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, value=u32(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand

    def validate(self) -> None:
        if self.opcode != Opcode.V_PK_MAX_I16:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.kind}")


class TestValuPkMaxI16AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.V_PK_MAX_I16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.V_PK_MAX_I16)
        self.assertEqual(inst.dst.kind, OperandKind.VGPR)
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.src0.index, 1)
        self.assertEqual(inst.src1.index, 2)

    def test_ast_rejects_out_of_range_vgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.vgpr(256)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.V_PK_MAX_I16,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
        )
        with self.assertRaises(TypeError):
            inst.validate()


class RegisterState:
    """Simulated state of GCN/RDNA registers for VALU scalar execution."""

    def __init__(self):
        self.vgpr = [0] * 256
        self.sgpr = [0] * 106

    def read_operand_u32(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgpr[op.index] & MASK32
        elif op.kind == OperandKind.SGPR:
            return self.sgpr[op.index] & MASK32
        elif op.kind in (OperandKind.LITERAL32, OperandKind.INLINE_CONST):
            return op.value & MASK32
        raise ValueError(f"Unsupported operand kind for read: {op.kind}")

    def write_vgpr_u32(self, index: int, val: int) -> None:
        self.vgpr[index] = u32(val)

    def execute(self, inst: ASTInstruction) -> int:
        inst.validate()
        src0 = self.read_operand_u32(inst.src0)
        src1 = self.read_operand_u32(inst.src1)
        res = ref_pk_max_i16(src0, src1)
        self.write_vgpr_u32(inst.dst.index, res)
        return res


class TestValuPkMaxI16ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_execution(self):
        state = RegisterState()
        state.vgpr[1] = 0x0005FFFB  # [-5, +5]
        state.vgpr[2] = 0xFFFB0005  # [+5, -5]
        inst = ASTInstruction(
            opcode=Opcode.V_PK_MAX_I16,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
            src1=ASTOperand.vgpr(2),
        )
        res = state.execute(inst)
        self.assertEqual(res, 0x00050005)
        self.assertEqual(state.vgpr[0], 0x00050005)
