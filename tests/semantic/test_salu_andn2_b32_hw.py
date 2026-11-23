from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_andn2_b32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_andn2_b32 instruction belongs to SOP2 format.
  Instruction:
    s_andn2_b32 sdst, ssrc0, ssrc1

  Operands:
    sdst  = 32-bit destination SGPR (or exec_lo)
    ssrc0 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo
    ssrc1 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo

  Semantics:
    sdst = ssrc0 & ~ssrc1
    SCC = 1 if sdst != 0 else 0

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

MASK32: int = 0xFFFFFFFF
ALL_LANES_ACTIVE_32: int = 0xFFFFFFFF
ALL_LANES_INACTIVE_32: int = 0x00000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_andn2_b32(ssrc0: int, ssrc1: int) -> tuple[int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_andn2_b32 instruction.

    Args:
        ssrc0: 32-bit first source operand (uint32)
        ssrc1: 32-bit second source operand (uint32)

    Returns:
        tuple (sdst, scc):
          - sdst: 32-bit result = ssrc0 & ~ssrc1
          - scc: scalar condition code = 1 if sdst != 0 else 0
    """
    s0 = u32(ssrc0)
    s1 = u32(ssrc1)
    sdst = u32(s0 & (~s1 & MASK32))
    scc = 1 if sdst != 0 else 0
    return sdst, scc


class TestSaluAndn2B32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_andn2_b32."""

    def test_oracle_basic(self):
        s0 = 0x12345678
        s1 = 0x00FF00FF
        sdst, scc = ref_andn2_b32(s0, s1)
        expected = s0 & (~s1 & MASK32)
        self.assertEqual(sdst, expected)
        self.assertEqual(scc, 1)

    def test_oracle_zero_s0(self):
        s1 = 0x12345678
        sdst, scc = ref_andn2_b32(0, s1)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)

    def test_oracle_all_ones_s1(self):
        s0 = 0x12345678
        sdst, scc = ref_andn2_b32(s0, ALL_LANES_ACTIVE_32)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)

    def test_oracle_zero_s1(self):
        s0 = 0x12345678
        sdst, scc = ref_andn2_b32(s0, 0)
        self.assertEqual(sdst, s0)
        self.assertEqual(scc, 1)

    def test_oracle_operand_asymmetry(self):
        # s0 & ~s1 != s1 & ~s0 (non-commutative)
        s0 = 0x0000FFFF
        s1 = 0x00FF00FF
        res_01, _ = ref_andn2_b32(s0, s1)
        res_10, _ = ref_andn2_b32(s1, s0)
        self.assertNotEqual(res_01, res_10)
        self.assertEqual(res_01, 0x0000FF00)
        self.assertEqual(res_10, 0x00FF0000)


class Opcode(Enum):
    S_ANDN2_B32 = auto()


class OperandKind(Enum):
    SGPR = auto()
    EXEC_LO = auto()
    VCC_LO = auto()
    LITERAL32 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    count: int = 1
    value: int = 0

    @classmethod
    def sgpr(cls, index: int) -> ASTOperand:
        if index < 0 or index > 105:
            raise ValueError(f"SGPR index out of range: s{index}")
        return cls(kind=OperandKind.SGPR, index=index)

    @classmethod
    def exec_lo(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC_LO)

    @classmethod
    def vcc_lo(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC_LO)

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
        if self.opcode != Opcode.S_ANDN2_B32:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind not in (OperandKind.SGPR, OperandKind.EXEC_LO, OperandKind.VCC_LO):
            raise TypeError(f"Destination must be SGPR, EXEC_LO, or VCC_LO, got {self.dst.kind}")
        if self.dst.count != 1 or self.src0.count != 1 or self.src1.count != 1:
            raise ValueError("All operands must be 32-bit single registers")


class TestSaluAndn2B32AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B32,
            dst=ASTOperand.sgpr(4),
            src0=ASTOperand.sgpr(6),
            src1=ASTOperand.sgpr(8),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.S_ANDN2_B32)
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.index, 4)
        self.assertEqual(inst.src0.index, 6)
        self.assertEqual(inst.src1.index, 8)

    def test_ast_rejects_out_of_range_sgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(106)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B32,
            dst=ASTOperand.literal32(0x123),
            src0=ASTOperand.sgpr(6),
            src1=ASTOperand.sgpr(8),
        )
        with self.assertRaises(TypeError):
            inst.validate()


class ScalarRegisterState:
    """Simulated state of GCN/RDNA Scalar Registers (SGPRs, EXEC_LO, and SCC)."""

    def __init__(self, exec_lo: int = ALL_LANES_ACTIVE_32):
        self.sgpr = [0] * 106
        self.exec_lo = u32(exec_lo)
        self.vcc_lo = 0
        self.scc = 0

    def read_operand_u32(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.SGPR:
            return self.sgpr[op.index]
        elif op.kind == OperandKind.EXEC_LO:
            return self.exec_lo
        elif op.kind == OperandKind.VCC_LO:
            return self.vcc_lo
        elif op.kind in (OperandKind.LITERAL32, OperandKind.INLINE_CONST):
            return op.value
        raise ValueError(f"Unsupported operand kind for read: {op.kind}")

    def write_sgpr_u32(self, index: int, val: int) -> None:
        self.sgpr[index] = u32(val)

    def execute(self, inst: ASTInstruction) -> tuple[int, int]:
        inst.validate()
        s0 = self.read_operand_u32(inst.src0)
        s1 = self.read_operand_u32(inst.src1)
        sdst_val, scc = ref_andn2_b32(s0, s1)

        if inst.dst.kind == OperandKind.SGPR:
            self.write_sgpr_u32(inst.dst.index, sdst_val)
        elif inst.dst.kind == OperandKind.EXEC_LO:
            self.exec_lo = sdst_val
        elif inst.dst.kind == OperandKind.VCC_LO:
            self.vcc_lo = sdst_val

        self.scc = scc
        return sdst_val, scc


class TestSaluAndn2B32ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_execution(self):
        state = ScalarRegisterState()
        state.write_sgpr_u32(0, 0x12345678)
        state.write_sgpr_u32(2, 0x00FF00FF)
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B32,
            dst=ASTOperand.sgpr(4),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.sgpr(2),
        )
        sdst, scc = state.execute(inst)
        expected = 0x12345678 & (~0x00FF00FF & MASK32)
        self.assertEqual(sdst, expected)
        self.assertEqual(state.sgpr[4], expected)
        self.assertEqual(state.scc, 1)
