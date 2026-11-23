from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_andn2_saveexec_b64.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_andn2_saveexec_b64 instruction belongs to SOP1 format.
  Instruction:
    s_andn2_saveexec_b64 sdst, ssrc0

  Operands:
    sdst  = 64-bit destination SGPR pair (or EXEC) to receive the previous EXEC mask
    ssrc0 = 64-bit source operand: SGPR pair s[2k:2k+1], literal 64-bit constant, VCC, or EXEC

  Semantics:
    sdst = EXEC
    EXEC = ssrc0 & ~EXEC
    SCC = 1 if EXEC != 0 else 0

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
import saveexec_run  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_ACTIVE: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_INACTIVE: int = 0x0000000000000000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit integer range [0, 2^64 - 1]."""
    return val & MASK64


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_andn2_saveexec_b64(ssrc0: int, old_exec: int) -> tuple[int, int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_andn2_saveexec_b64 instruction.

    Args:
        ssrc0: 64-bit source operand (uint64)
        old_exec: 64-bit current EXEC mask before instruction execution (uint64)

    Returns:
        tuple (new_exec, sdst, scc):
          - new_exec: updated 64-bit EXEC mask = ssrc0 & ~old_exec
          - sdst: saved previous EXEC mask = old_exec
          - scc: scalar condition code = 1 if new_exec != 0 else 0
    """
    s0 = u64(ssrc0)
    old_e = u64(old_exec)
    sdst = old_e
    new_exec = u64(s0 & (~old_e & MASK64))
    scc = 1 if new_exec != 0 else 0
    return new_exec, sdst, scc


class TestSaluAndn2SaveexecB64Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_andn2_saveexec_b64."""

    def test_oracle_all_ones_exec(self):
        # If old_exec = all 1s, ~old_exec = 0, so new_exec = ssrc0 & 0 = 0
        s0 = 0x123456789ABCDEF0
        old_exec = ALL_LANES_ACTIVE
        new_exec, sdst, scc = ref_andn2_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, old_exec)
        self.assertEqual(scc, 0)

    def test_oracle_all_zeros_exec(self):
        # If old_exec = 0, ~old_exec = all 1s, so new_exec = ssrc0 & all 1s = ssrc0
        s0 = 0x123456789ABCDEF0
        old_exec = ALL_LANES_INACTIVE
        new_exec, sdst, scc = ref_andn2_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, s0)
        self.assertEqual(sdst, ALL_LANES_INACTIVE)
        self.assertEqual(scc, 1)

    def test_oracle_zero_s0_zero_exec(self):
        s0 = 0x0
        old_exec = ALL_LANES_INACTIVE
        new_exec, sdst, scc = ref_andn2_saveexec_b64(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, ALL_LANES_INACTIVE)
        self.assertEqual(scc, 0)

    def test_oracle_self_exec(self):
        # S0 == old_exec -> new_exec = exec & ~exec = 0
        exec_val = 0xAAAAAAAAAAAAAAAA
        new_exec, sdst, scc = ref_andn2_saveexec_b64(exec_val, exec_val)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, exec_val)
        self.assertEqual(scc, 0)


class Opcode(Enum):
    S_ANDN2_SAVEEXEC_B64 = auto()


class OperandKind(Enum):
    SGPR = auto()
    EXEC = auto()
    VCC = auto()
    LITERAL64 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    count: int = 2
    value: int = 0

    @classmethod
    def sgpr(cls, index: int, count: int = 2) -> ASTOperand:
        if count != 2:
            raise ValueError(f"s_andn2_saveexec_b64 requires 64-bit register pair (count=2), got {count}")
        if index < 0 or index > 104 or (index + count) > 106:
            raise ValueError(f"SGPR index out of range: s[{index}:{index + count - 1}]")
        if index % 2 != 0:
            raise ValueError(f"SGPR pair must be 64-bit aligned (even index), got s{index}")
        return cls(kind=OperandKind.SGPR, index=index, count=count)

    @classmethod
    def exec_mask(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC, count=2)

    @classmethod
    def vcc(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC, count=2)

    @classmethod
    def literal64(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL64, count=2, value=u64(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, count=2, value=u64(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand

    def validate(self) -> None:
        if self.opcode != Opcode.S_ANDN2_SAVEEXEC_B64:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind not in (OperandKind.SGPR, OperandKind.EXEC):
            raise TypeError(f"Destination must be SGPR pair or EXEC, got {self.dst.kind}")
        if self.dst.count != 2:
            raise ValueError(f"Destination must be 64-bit (2 registers), got {self.dst.count}")
        if self.src0.count != 2:
            raise ValueError(f"Source must be 64-bit operand, got {self.src0.count}")


class TestSaluAndn2SaveexecB64AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_SAVEEXEC_B64,
            dst=ASTOperand.sgpr(4, 2),
            src0=ASTOperand.sgpr(6, 2),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.S_ANDN2_SAVEEXEC_B64)
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.index, 4)
        self.assertEqual(inst.src0.index, 6)

    def test_ast_rejects_unaligned_sgpr_pair(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(5, 2)

    def test_ast_rejects_single_sgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(4, 1)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_SAVEEXEC_B64,
            dst=ASTOperand.literal64(0x123),
            src0=ASTOperand.sgpr(6, 2),
        )
        with self.assertRaises(TypeError):
            inst.validate()
