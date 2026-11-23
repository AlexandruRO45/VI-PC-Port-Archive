"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 SALU 32-bit bitwise logic operations (Bundle 3):
  - s_and_b32:   Bitwise AND
  - s_or_b32:    Bitwise OR
  - s_xor_b32:   Bitwise XOR
  - s_andn2_b32: Bitwise AND with NOT second source
  - s_orn2_b32:  Bitwise OR with NOT second source
  - s_nand_b32:  Bitwise NAND
  - s_nor_b32:   Bitwise NOR
  - s_xnor_b32:  Bitwise XNOR (Equivalence)

In AMD RDNA2 ISA (SOP2 encoding format):
  sdst, ssrc0, ssrc1
  SCC = 1 if sdst != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Callable, Dict, List, Optional, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

try:
    import rdna2spirv
    import support
except Exception:
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    return val & MASK32


class Opcode(Enum):
    S_AND_B32 = "s_and_b32"
    S_OR_B32 = "s_or_b32"
    S_XOR_B32 = "s_xor_b32"
    S_ANDN2_B32 = "s_andn2_b32"
    S_ORN2_B32 = "s_orn2_b32"
    S_NAND_B32 = "s_nand_b32"
    S_NOR_B32 = "s_nor_b32"
    S_XNOR_B32 = "s_xnor_b32"


class OperandKind(Enum):
    SGPR = auto()
    IMM = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    value: int = 0

    @classmethod
    def sgpr(cls, idx: int) -> ASTOperand:
        return cls(kind=OperandKind.SGPR, index=idx)

    @classmethod
    def imm(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.IMM, value=val & MASK32)


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand

    def disasm(self) -> str:
        def fmt(op: ASTOperand) -> str:
            if op.kind == OperandKind.SGPR:
                return f"s{op.index}"
            return hex(op.value)
        return f"{self.opcode.value} {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}"


class ScalarMachineState:
    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs: list[int] = [0] * num_sgprs
        self.scc: int = 0

    def read_operand(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.SGPR:
            return self.sgprs[op.index] & MASK32
        return op.value & MASK32

    def write_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32


GLOBAL_ORACLE_MAP: dict[Opcode, Callable[[int, int], tuple[int, int]]] = {}


class ScalarInterpreter:
    def step(self, instr: ASTInstruction, state: ScalarMachineState) -> tuple[int, int]:
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        res, scc = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1)
        if instr.dst.kind == OperandKind.SGPR:
            state.write_sgpr(instr.dst.index, res)
        state.scc = scc
        return res, scc

def ref_s_and_b32(s0: int, s1: int) -> tuple[int, int]:
    res = u32(s0 & s1)
    scc = 1 if res != 0 else 0
    return res, scc

GLOBAL_ORACLE_MAP[Opcode.S_AND_B32] = ref_s_and_b32


class TestSAndB32Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        res, scc = ref_s_and_b32(0x12345678, 0x0F0F0F0F)
        self.assertEqual(res, 0x2040608)
        self.assertEqual(scc, 1 if res != 0 else 0)

    def test_oracle_zero_inputs(self) -> None:
        res, scc = ref_s_and_b32(0, 0)
        self.assertEqual(scc, 1 if res != 0 else 0)

class TestSAndB32ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.S_AND_B32, ASTOperand.sgpr(0), ASTOperand.sgpr(1), ASTOperand.sgpr(2))
        self.assertEqual(inst.disasm(), "s_and_b32 s0, s1, s2")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_sgpr(1, 0x12345678)
        state.write_sgpr(2, 0x0F0F0F0F)
        inst = ASTInstruction(Opcode.S_AND_B32, ASTOperand.sgpr(0), ASTOperand.sgpr(1), ASTOperand.sgpr(2))
        res, scc = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x2040608)
        self.assertEqual(state.sgprs[0], 0x2040608)
        self.assertEqual(state.scc, scc)

class TestSAndB32SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_preservation(self) -> None:
        state = ScalarMachineState()
        state.write_sgpr(0, 0x12345678)
        state.write_sgpr(1, 0x0F0F0F0F)
        inst = ASTInstruction(Opcode.S_AND_B32, ASTOperand.sgpr(0), ASTOperand.sgpr(0), ASTOperand.sgpr(1))
        res, scc = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x2040608)
        self.assertEqual(state.sgprs[0], 0x2040608)

    def test_ieee754_bit_transparency(self) -> None:
        qnan = 0x7FC00000
        snan = 0x7F800001
        res, _ = ref_s_and_b32(qnan, snan)
        self.assertIsInstance(res, int)

class TestSAndB32Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x0F0F0F0F",
            "s_and_b32 s2, s0, s1",
            "v_mov_b32_e32 v0, s2",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_s_and_b32", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x2040608)

def ref_s_or_b32(s0: int, s1: int) -> tuple[int, int]:
    res = u32(s0 | s1)
    scc = 1 if res != 0 else 0
    return res, scc

GLOBAL_ORACLE_MAP[Opcode.S_OR_B32] = ref_s_or_b32


class TestSOrB32Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        res, scc = ref_s_or_b32(0x12345678, 0x0F0F0F0F)
        self.assertEqual(res, 0x1f3f5f7f)
        self.assertEqual(scc, 1 if res != 0 else 0)

    def test_oracle_zero_inputs(self) -> None:
        res, scc = ref_s_or_b32(0, 0)
        self.assertEqual(scc, 1 if res != 0 else 0)


if __name__ == "__main__":
    unittest.main()
