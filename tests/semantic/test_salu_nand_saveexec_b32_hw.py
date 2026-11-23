from __future__ import annotations

import enum
import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

"""
RDNA2 SALU instruction semantic tests: s_nand_saveexec_b32.

Exercises s_nand_saveexec_b32 through tools/rdna2spirv.py and spirv_mini_interp.
RDNA2 ISA, SOP1:
    s_nand_saveexec_b32 sD, sS0
Semantics:
    D = EXEC_LO
    EXEC_LO = ~(S0 & EXEC_LO) & 0xFFFFFFFF
    SCC = (EXEC_LO != 0)

In Wave64 mode:
    EXEC_HI (lanes 32..63) remains untouched.
    D receives the 32-bit old EXEC_LO.
    SCC reflects (new EXEC_LO != 0).
"""

MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def ref_nand_saveexec_b32(s0: int, old_exec_lo: int) -> Tuple[int, int, int]:
    """
    Reference mathematical model for s_nand_saveexec_b32.
    Returns (new_exec_lo, saved_d, scc).
    """
    s0_32 = s0 & MASK32
    exec_lo = old_exec_lo & MASK32
    new_exec_lo = (~(s0_32 & exec_lo)) & MASK32
    saved_d = exec_lo
    scc = 1 if new_exec_lo != 0 else 0
    return new_exec_lo, saved_d, scc


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle
# ==============================================================================

class TestSaluNandSaveexecB32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical model of s_nand_saveexec_b32."""

    def test_both_zero(self) -> None:
        new_exec, saved_d, scc = ref_nand_saveexec_b32(0, 0)
        self.assertEqual(new_exec, MASK32)
        self.assertEqual(saved_d, 0)
        self.assertEqual(scc, 1)

    def test_all_ones_self_cancellation(self) -> None:
        new_exec, saved_d, scc = ref_nand_saveexec_b32(MASK32, MASK32)
        self.assertEqual(new_exec, 0)
        self.assertEqual(saved_d, MASK32)
        self.assertEqual(scc, 0)

    def test_orthogonal_masks(self) -> None:
        # S0 and EXEC have no overlapping bits -> S0 & EXEC == 0 -> NAND is ~0
        s0 = 0x5555_5555
        old_exec = 0xAAAA_AAAA
        new_exec, saved_d, scc = ref_nand_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, MASK32)
        self.assertEqual(saved_d, old_exec)
        self.assertEqual(scc, 1)

    def test_partial_overlap(self) -> None:
        s0 = 0x0000_FFFF
        old_exec = 0x00FF_00FF
        # s0 & old_exec = 0x0000_00FF
        # ~(0x0000_00FF) = 0xFFFF_FF00
        new_exec, saved_d, scc = ref_nand_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, 0xFFFF_FF00)
        self.assertEqual(saved_d, old_exec)
        self.assertEqual(scc, 1)

    def test_single_bit_patterns(self) -> None:
        for bit in range(32):
            val = 1 << bit
            new_exec, saved_d, scc = ref_nand_saveexec_b32(val, val)
            self.assertEqual(new_exec, (~val) & MASK32)
            self.assertEqual(saved_d, val)
            self.assertEqual(scc, 1)

# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegKind(enum.Enum):
    SGPR = "s"
    EXEC_LO = "exec_lo"
    VCC_LO = "vcc_lo"


@dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int = 0

    def __str__(self) -> str:
        if self.kind == RegKind.SGPR:
            return f"s{self.index}"
        return self.kind.value


@dataclass(frozen=True)
class Immediate:
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value & MASK32)

    def __str__(self) -> str:
        return f"0x{self.value:08X}"


Operand = Union[Register, Immediate]


class Opcode(enum.Enum):
    S_NAND_SAVEEXEC_B32 = "s_nand_saveexec_b32"
    S_MOV_B32 = "s_mov_b32"


@dataclass
class Instruction:
    opcode: Opcode
    dst: Register
    src0: Operand

    def __str__(self) -> str:
        return f"{self.opcode.value} {self.dst}, {self.src0}"


class ASTNode:
    def eval(self, env: Dict[str, int]) -> int:
        raise NotImplementedError


@dataclass
class LiteralNode(ASTNode):
    val: int

    def eval(self, env: Dict[str, int]) -> int:
        return self.val & MASK32


@dataclass
class RegReadNode(ASTNode):
    name: str

    def eval(self, env: Dict[str, int]) -> int:
        return env.get(self.name, 0) & MASK32


@dataclass
class NandSaveExecB32Node(ASTNode):
    src0: ASTNode
    exec_lo: ASTNode

    def eval(self, env: Dict[str, int]) -> Tuple[int, int, int]:
        s0 = self.src0.eval(env)
        exec_val = self.exec_lo.eval(env)
        return ref_nand_saveexec_b32(s0, exec_val)


# ==============================================================================
# Unit Tests: Symbolic AST Opcode Definitions
# ==============================================================================

class TestSaluNandSaveexecB32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_creation_and_string(self) -> None:
        inst = Instruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B32,
            dst=Register(RegKind.SGPR, 4),
            src0=Register(RegKind.SGPR, 6),
        )
        self.assertEqual(inst.opcode, Opcode.S_NAND_SAVEEXEC_B32)
        self.assertEqual(str(inst), "s_nand_saveexec_b32 s4, s6")

    def test_instruction_with_immediate(self) -> None:
        inst = Instruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B32,
            dst=Register(RegKind.SGPR, 2),
            src0=Immediate(0x1234_5678),
        )
        self.assertEqual(str(inst), "s_nand_saveexec_b32 s2, 0x12345678")

    def test_ast_evaluation_matches_oracle(self) -> None:
        env = {"s0": 0x00FF_00FF, "exec_lo": 0x0F0F_0F0F}
        node = NandSaveExecB32Node(RegReadNode("s0"), RegReadNode("exec_lo"))
        result = node.eval(env)
        expected = ref_nand_saveexec_b32(0x00FF_00FF, 0x0F0F_0F0F)
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
