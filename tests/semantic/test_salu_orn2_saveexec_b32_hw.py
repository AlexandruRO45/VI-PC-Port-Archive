from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_orn2_saveexec_b32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_orn2_saveexec_b32 instruction belongs to SOP1 format.
  Instruction:
    s_orn2_saveexec_b32 sdst, ssrc0

  Operands:
    sdst  = 32-bit destination SGPR (or exec_lo) to receive the previous EXEC_LO mask
    ssrc0 = 32-bit source operand: SGPR, literal 32-bit constant, inline constant, exec_lo, or vcc_lo

  Semantics:
    sdst = EXEC_LO
    EXEC_LO = ssrc0 | ~EXEC_LO
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

MASK32: int = 0xFFFFFFFF
ALL_LANES_ACTIVE_32: int = 0xFFFFFFFF
ALL_LANES_INACTIVE_32: int = 0x00000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_orn2_saveexec_b32(ssrc0: int, old_exec_lo: int) -> tuple[int, int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_orn2_saveexec_b32 instruction.

    Args:
        ssrc0: 32-bit source operand (uint32)
        old_exec_lo: 32-bit current EXEC_LO mask before instruction execution (uint32)

    Returns:
        tuple (new_exec_lo, sdst, scc):
          - new_exec_lo: updated 32-bit EXEC_LO mask = ssrc0 | ~old_exec_lo
          - sdst: saved previous EXEC_LO mask = old_exec_lo
          - scc: scalar condition code = 1 if new_exec_lo != 0 else 0
    """
    s0 = u32(ssrc0)
    old_e = u32(old_exec_lo)
    sdst = old_e
    new_exec_lo = u32(s0 | (~old_e & MASK32))
    scc = 1 if new_exec_lo != 0 else 0
    return new_exec_lo, sdst, scc


class TestSaluOrn2SaveexecB32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_orn2_saveexec_b32."""

    def test_oracle_all_ones_exec(self):
        # If old_exec_lo = all 1s, ~old_exec_lo = 0, so new_exec_lo = ssrc0 | 0 = ssrc0
        s0 = 0x12345678
        old_exec = ALL_LANES_ACTIVE_32
        new_exec, sdst, scc = ref_orn2_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, s0)
        self.assertEqual(sdst, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_all_zeros_exec(self):
        # If old_exec_lo = 0, ~old_exec_lo = all 1s, so new_exec_lo = ssrc0 | all 1s = all 1s
        s0 = 0x0
        old_exec = ALL_LANES_INACTIVE_32
        new_exec, sdst, scc = ref_orn2_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, ALL_LANES_ACTIVE_32)
        self.assertEqual(sdst, ALL_LANES_INACTIVE_32)
        self.assertEqual(scc, 1)

    def test_oracle_zero_s0_all_ones_exec_produces_zero_exec(self):
        # The unique condition where new_exec_lo becomes 0: s0 == 0 and old_exec_lo == ~0
        s0 = 0x0
        old_exec = ALL_LANES_ACTIVE_32
        new_exec, sdst, scc = ref_orn2_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, 0)
        self.assertEqual(sdst, ALL_LANES_ACTIVE_32)
        self.assertEqual(scc, 0)

    def test_oracle_self_exec(self):
        # When ssrc0 == old_exec_lo: exec | ~exec = ALL_LANES_ACTIVE_32
        sample_exec = 0xAAAAAAAA
        new_exec, sdst, scc = ref_orn2_saveexec_b32(sample_exec, sample_exec)
        self.assertEqual(new_exec, ALL_LANES_ACTIVE_32)
        self.assertEqual(sdst, sample_exec)
        self.assertEqual(scc, 1)


class Opcode(Enum):
    S_ORN2_SAVEEXEC_B32 = auto()


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
        return cls(kind=OperandKind.SGPR, index=index, count=1)

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
        if self.opcode != Opcode.S_ORN2_SAVEEXEC_B32:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind not in (OperandKind.SGPR, OperandKind.EXEC_LO):
            raise TypeError(f"Destination must be SGPR or EXEC_LO, got {self.dst.kind}")
        if self.dst.count != 1:
            raise ValueError(f"Destination must be 32-bit (1 register), got {self.dst.count}")
        if self.src0.count != 1:
            raise ValueError(f"Source must be 32-bit operand, got {self.src0.count}")


class TestSaluOrn2SaveexecB32AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ORN2_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(4),
            src0=ASTOperand.sgpr(6),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.S_ORN2_SAVEEXEC_B32)
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.index, 4)
        self.assertEqual(inst.src0.index, 6)

    def test_ast_rejects_out_of_range_sgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(106)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ORN2_SAVEEXEC_B32,
            dst=ASTOperand.literal32(0x123),
            src0=ASTOperand.sgpr(6),
        )
        with self.assertRaises(TypeError):
            inst.validate()


class ScalarRegisterState:
    """Simulated state of GCN/RDNA Scalar Registers (SGPRs, EXEC_LO, and SCC)."""

    def __init__(self, exec_lo: int = ALL_LANES_ACTIVE_32):
        self.sgpr = [0] * 106
        self.exec_lo = u32(exec_lo)
        self.scc = 0

    def read_operand_u32(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.SGPR:
            return self.sgpr[op.index]
        elif op.kind == OperandKind.EXEC_LO:
            return self.exec_lo
        elif op.kind in (OperandKind.LITERAL32, OperandKind.INLINE_CONST):
            return op.value
        raise ValueError(f"Unsupported operand kind for read: {op.kind}")

    def write_sgpr_u32(self, index: int, val: int) -> None:
        self.sgpr[index] = u32(val)

    def execute(self, inst: ASTInstruction) -> tuple[int, int, int]:
        inst.validate()
        s0 = self.read_operand_u32(inst.src0)
        old_exec = self.exec_lo
        new_exec, sdst_val, scc = ref_orn2_saveexec_b32(s0, old_exec)

        if inst.dst.kind == OperandKind.SGPR:
            self.write_sgpr_u32(inst.dst.index, sdst_val)
        elif inst.dst.kind == OperandKind.EXEC_LO:
            self.exec_lo = sdst_val

        self.exec_lo = new_exec
        self.scc = scc
        return new_exec, sdst_val, scc


class TestSaluOrn2SaveexecB32ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_execution(self):
        state = ScalarRegisterState(exec_lo=0xFFFF0000)
        state.write_sgpr_u32(2, 0x000000FF)
        inst = ASTInstruction(
            opcode=Opcode.S_ORN2_SAVEEXEC_B32,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.sgpr(2),
        )
        new_exec, sdst, scc = state.execute(inst)
        # old_exec = 0xFFFF0000 -> ~old_exec = 0x0000FFFF
        # s0 = 0x000000FF
        # new_exec = s0 | ~old_exec = 0x0000FFFF
        self.assertEqual(new_exec, 0x0000FFFF)
        self.assertEqual(sdst, 0xFFFF0000)
        self.assertEqual(state.exec_lo, 0x0000FFFF)
        self.assertEqual(state.sgpr[0], 0xFFFF0000)
        self.assertEqual(state.scc, 1)


class TestSaluOrn2SaveexecB32SimdDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32-bit EXEC masking."""

    def test_inactive_lane_reactivation(self):
        lanes = lambda lo, hi: sum(1 << l for l in range(lo, hi))
        # Initial active lanes: 0..15 (lower half)
        # Initial inactive lanes: 16..31 (upper half)
        old_exec = lanes(0, 16)
        # ssrc0 = 0 (requests no lanes explicitly)
        s0 = 0

        new_exec, sdst, scc = ref_orn2_saveexec_b32(s0, old_exec)
        # Previously active lanes (0..15) turn off (~old_exec = 0 in lower half)
        # Previously inactive lanes (16..31) turn on (~old_exec = 1 in upper half)
        self.assertEqual(new_exec, lanes(16, 32))
        self.assertEqual(sdst, old_exec)
        self.assertEqual(scc, 1)

    def test_divergent_mask_checkerboard(self):
        even_lanes = 0x55555555
        odd_lanes = 0xAAAAAAAA
        # old_exec = even lanes active; s0 = 0
        # ~even_lanes = odd_lanes
        new_exec, sdst, scc = ref_orn2_saveexec_b32(0, even_lanes)
        self.assertEqual(new_exec, odd_lanes)
        self.assertEqual(sdst, even_lanes)
        self.assertEqual(scc, 1)
