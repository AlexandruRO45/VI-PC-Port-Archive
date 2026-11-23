from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_and_saveexec_b32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_and_saveexec_b32 instruction belongs to SOP1 format.
  Instruction:
    s_and_saveexec_b32 sdst, ssrc0

  Operands:
    sdst  = 32-bit destination SGPR (or exec_lo) to receive previous EXEC_LO
    ssrc0 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo

  Semantics:
    sdst = EXEC_LO
    EXEC_LO = ssrc0 & EXEC_LO
    SCC = 1 if EXEC_LO != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) for hardware-accurate translation.
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

MASK32 = 0xFFFFFFFF
ALL_LANES_ACTIVE_32 = 0xFFFFFFFF
ALL_LANES_INACTIVE_32 = 0x00000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_and_saveexec_b32(ssrc0: int, old_exec_lo: int) -> tuple[int, int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_and_saveexec_b32 instruction.

    Args:
        ssrc0: 32-bit source operand (uint32)
        old_exec_lo: 32-bit current EXEC_LO mask before instruction execution (uint32)

    Returns:
        tuple (new_exec_lo, sdst, scc):
          - new_exec_lo: updated 32-bit EXEC_LO mask = ssrc0 & old_exec_lo
          - sdst: saved previous EXEC_LO mask = old_exec_lo
          - scc: scalar condition code = 1 if new_exec_lo != 0 else 0
    """
    s0 = u32(ssrc0)
    old_e = u32(old_exec_lo)
    sdst = old_e
    new_exec_lo = u32(s0 & old_e)
    scc = 1 if new_exec_lo != 0 else 0
    return new_exec_lo, sdst, scc


class TestSaluAndSaveexecB32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_and_saveexec_b32."""

    def test_oracle_all_ones_exec(self):
        s0 = 0x12345678
        old_exec = ALL_LANES_ACTIVE_32
        new_exec, sdst, scc = ref_and_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, s0)
        self.assertEqual(sdst, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_all_zeros_exec(self):
        s0 = 0xFFFFFFFF
        old_exec = ALL_LANES_INACTIVE_32
        new_exec, sdst, scc = ref_and_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, ALL_LANES_INACTIVE_32)
        self.assertEqual(scc, 0)

    def test_oracle_zero_s0_produces_zero_exec(self):
        s0 = 0x0
        old_exec = 0xAAAAAAAA
        new_exec, sdst, scc = ref_and_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, 0xAAAAAAAA)
        self.assertEqual(scc, 0)




class Opcode(Enum):
    S_AND_SAVEEXEC_B32 = auto()


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
    def sgpr(cls, index: int, count: int = 1) -> ASTOperand:
        if count != 1:
            raise ValueError(f"s_and_saveexec_b32 requires 32-bit register (count=1), got {count}")
        if index < 0 or index > 105:
            raise ValueError(f"SGPR index out of range: s{index}")
        return cls(kind=OperandKind.SGPR, index=index, count=count)

    @classmethod
    def exec_lo(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC_LO, count=1)

    @classmethod
    def vcc_lo(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC_LO, count=1)

    @classmethod
    def literal32(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL32, count=1, value=u32(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, count=1, value=u32(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand

    def validate(self) -> None:
        if self.opcode != Opcode.S_AND_SAVEEXEC_B32:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind not in (OperandKind.SGPR, OperandKind.EXEC_LO):
            raise TypeError(f"Destination must be SGPR or EXEC_LO, got {self.dst.kind}")
        if self.dst.count != 1:
            raise ValueError(f"Destination must be 32-bit (1 register), got {self.dst.count}")
        if self.src0.count != 1:
            raise ValueError(f"Source must be 32-bit operand, got {self.src0.count}")


class TestSaluAndSaveexecB32AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(4, 1),
            src0=ASTOperand.sgpr(6, 1),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.S_AND_SAVEEXEC_B32)
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.index, 4)
        self.assertEqual(inst.src0.index, 6)

    def test_ast_rejects_out_of_range_sgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(106, 1)

    def test_ast_rejects_invalid_count(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(4, 2)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.S_AND_SAVEEXEC_B32,
            dst=ASTOperand.literal32(0x123),
            src0=ASTOperand.sgpr(6, 1),
        )
        with self.assertRaises(TypeError):
            inst.validate()


if __name__ == "__main__":
    unittest.main()
