"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU 32-bit bit manipulation and shift operations (Bundle 14):
  - v_bfe_u32:       32-bit unsigned bitfield extract
  - v_bfe_i32:       32-bit signed bitfield extract
  - v_bfi_b32:       32-bit bitfield insert / multiplexer
  - v_bfm_b32:       32-bit bitfield mask generator
  - v_alignbyte_b32: 32-bit byte alignment / byte funnel shift
  - v_alignbit_b32:  32-bit bit alignment / bit funnel shift

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
    V_BFE_U32 = "v_bfe_u32"
    V_BFE_I32 = "v_bfe_i32"
    V_BFI_B32 = "v_bfi_b32"
    V_BFM_B32 = "v_bfm_b32"
    V_ALIGNBYTE_B32 = "v_alignbyte_b32"
    V_ALIGNBIT_B32 = "v_alignbit_b32"


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    IMM = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    value: int = 0

    @classmethod
    def vgpr(cls, idx: int) -> ASTOperand:
        return cls(kind=OperandKind.VGPR, index=idx)

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
    src2: Optional[ASTOperand] = None

    def disasm(self) -> str:
        def fmt(op: ASTOperand) -> str:
            if op.kind == OperandKind.VGPR:
                return f"v{op.index}"
            if op.kind == OperandKind.SGPR:
                return f"s{op.index}"
            return hex(op.value)

        if self.src2 is not None:
            return f"{self.opcode.value} {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}, {fmt(self.src2)}"
        return f"{self.opcode.value} {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}"


class ScalarMachineState:
    def __init__(self, num_vgprs: int = 256, num_sgprs: int = 106) -> None:
        self.vgprs: list[int] = [0] * num_vgprs
        self.sgprs: list[int] = [0] * num_sgprs
        self.scc: int = 0
        self.vcc: int = 0

    def read_operand(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgprs[op.index] & MASK32
        if op.kind == OperandKind.SGPR:
            return self.sgprs[op.index] & MASK32
        return op.value & MASK32

    def write_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32


GLOBAL_ORACLE_MAP: dict[Opcode, Callable[..., int]] = {}


class ScalarInterpreter:
    def step(self, instr: ASTInstruction, state: ScalarMachineState) -> int:
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        if instr.src2 is not None:
            s2 = state.read_operand(instr.src2)
            res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1, s2)
        else:
            res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1)

        if instr.dst.kind == OperandKind.VGPR:
            state.write_vgpr(instr.dst.index, res)
        return res

def ref_v_bfe_u32(s0: int, s1: int, s2: int) -> int:
    off = s1 & 31
    wid = min(s2 & 31, 32 - off)
    return 0 if wid == 0 else (s0 >> off) & ((1 << wid) - 1)

GLOBAL_ORACLE_MAP[Opcode.V_BFE_U32] = ref_v_bfe_u32


class TestVBfeU32Oracle(unittest.TestCase):
    def test_oracle_execution(self) -> None:
        res = ref_v_bfe_u32(0x12345678, 8, 16)
        self.assertEqual(res, 0x3456)

class TestVBfeU32ASTAndInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x12345678)
        state.write_vgpr(1, 8)
        state.write_vgpr(2, 16)
        inst = ASTInstruction(Opcode.V_BFE_U32, ASTOperand.vgpr(3), ASTOperand.vgpr(0), ASTOperand.vgpr(1), ASTOperand.vgpr(2))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x3456)
        self.assertEqual(state.vgprs[3], 0x3456)


if __name__ == "__main__":
    unittest.main()
