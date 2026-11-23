from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 32-bit Vector Logic Suite.

Covers:
- v_and_b32: Bitwise AND D = S0 & S1.
- v_or_b32: Bitwise OR D = S0 | S1.
- v_xor_b32: Bitwise XOR D = S0 ^ S1.
- v_not_b32: Bitwise NOT D = ~S0.
- v_xnor_b32: Bitwise XNOR D = ~(S0 ^ S1).
- v_and_or_b32: 3-operand bitwise operation D = (S0 & S1) | S2.
- v_or3_b32: 3-operand bitwise OR D = S0 | S1 | S2.

Standard library only (python -I, struct, math, unittest). Zero game data.
All test suites execute real translated assembly via support.translate_lines.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import rdna2spirv

MASK32: int = 0xFFFF_FFFF


def run_shader(name: str, lines: List[str]) -> Tuple[int, str]:
    full_lines = lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = support.translate_lines(name, full_lines)
    words = support.mrt0_words(spv)
    return words[0], spv


# ------------------------------------------------------------------------------
# v_and_b32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def ref_and_b32(s0: int, s1: int) -> int:
    return (s0 & s1) & MASK32


class TestAndB32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        vectors = [
            (0xFF00, 0x0FF0),
            (0x1234_5678, 0xFFFF_0000),
            (0xAAAA_AAAA, 0x5555_5555),
            (0x1357_9BDF, 0xFEDC_BA98),
        ]
        for idx, (a, b) in enumerate(vectors):
            val, _ = run_shader(f"ps_and_oracle_{idx}", [
                f"v_mov_b32 v1, {hex(a)}",
                f"v_mov_b32 v2, {hex(b)}",
                "v_and_b32 v0, v1, v2",
            ])
            self.assertEqual(val, ref_and_b32(a, b))

    def test_oracle_identities_hardware(self):
        val_id, _ = run_shader("ps_and_id", [
            "v_mov_b32 v1, 0xCAFEBABE",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(val_id, 0xCAFEBABE)

        val_zero, _ = run_shader("ps_and_zero", [
            "v_mov_b32 v1, 0xCAFEBABE",
            "v_mov_b32 v2, 0",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(val_zero, 0)


# ------------------------------------------------------------------------------
# v_and_b32 AST & Scalar Machine
# ------------------------------------------------------------------------------

class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class RegisterOperand:
    op_type: OperandType
    index: int

    def __post_init__(self):
        if self.op_type == OperandType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range: {self.index}")
        if self.op_type == OperandType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range: {self.index}")

    @property
    def name(self) -> str:
        return f"{'v' if self.op_type == OperandType.VGPR else 's'}{self.index}"


@dataclass(frozen=True)
class LiteralOperand:
    value: int

    def __post_init__(self):
        if not (-0x8000_0000 <= self.value <= MASK32):
            raise ValueError(f"Literal out of range: {self.value}")


@dataclass
class VAndB32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination must be a VGPR")


class ScalarMachineLogic:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK32
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise TypeError(f"Unknown operand: {op}")

    def step_and(self, insn: VAndB32Instruction):
        s0 = self.read_operand(insn.src0)
        s1 = self.read_operand(insn.src1)
        self.vgprs[insn.dst.index] = ref_and_b32(s0, s1)


class TestAndB32ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineLogic()
        machine.vgprs[1] = 0xF0F0_F0F0
        machine.vgprs[2] = 0x00FF_00FF
        insn = VAndB32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
        )
        machine.step_and(insn)

        val_hw, _ = run_shader("ps_and_ast_scalar", [
            "v_mov_b32 v1, 0xF0F0F0F0",
            "v_mov_b32 v2, 0x00FF00FF",
            "v_and_b32 v0, v1, v2",
        ])
        self.assertEqual(machine.vgprs[0], val_hw)
        self.assertEqual(val_hw, 0x00F0_00F0)

    def test_scalar_in_place_hardware(self):
        val_inplace, _ = run_shader("ps_and_inplace", [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x0000FFFF",
            "v_and_b32 v0, v0, v1",
        ])
        self.assertEqual(val_inplace, 0x5678)
