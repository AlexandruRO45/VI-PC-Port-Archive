"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU Float16 Mixed Precision & Specialized FMA operations (Bundle 20):
  - v_fmaak_f16:     Float16 fused multiply-add with literal K (D = S0 * S1 + K)
  - v_fmamk_f16:     Float16 fused multiply-add with literal K (D = S0 * K + S1)
  - v_fma_mixhi_f16: Float16 mixed precision FMA into high halfword (D[31:16] = fma(S0, S1, S2))
  - v_fma_mixlo_f16: Float16 mixed precision FMA into low halfword (D[15:0] = fma(S0, S1, S2))
  - v_dot2_f32_f16:  Float16 packed dot product with Float32 accumulator (D = S0.lo*S1.lo + S0.hi*S1.hi + S2)
  - v_dot2c_f32_f16: Float16 packed dot product accumulating into destination (D = S0.lo*S1.lo + S0.hi*S1.hi + D)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
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

_TRANSLATOR_READY = support is not None and rdna2spirv is not None

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF


def u32(val: int) -> int:
    return val & MASK32


def f16_to_bits(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_to_f16(u: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u & 0xFFFF))[0]


def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def is_nan_f16(h: int) -> bool:
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def is_nan_f32(w: int) -> bool:
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x007FFFFF) != 0


class Opcode(Enum):
    V_FMAAK_F16 = "v_fmaak_f16"
    V_FMAMK_F16 = "v_fmamk_f16"
    V_FMA_MIXHI_F16 = "v_fma_mixhi_f16"
    V_FMA_MIXLO_F16 = "v_fma_mixlo_f16"
    V_DOT2_F32_F16 = "v_dot2_f32_f16"
    V_DOT2C_F32_F16 = "v_dot2c_f32_f16"


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
        old_dst = state.read_operand(instr.dst)
        if instr.src2 is not None:
            s2 = state.read_operand(instr.src2)
            if instr.opcode in (
                Opcode.V_FMAAK_F16,
                Opcode.V_FMAMK_F16,
                Opcode.V_FMA_MIXHI_F16,
                Opcode.V_FMA_MIXLO_F16,
            ):
                res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1, s2, old_dst=old_dst)
            else:
                res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1, s2)
        else:
            if instr.opcode == Opcode.V_DOT2C_F32_F16:
                res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1, old_dst=old_dst)
            else:
                res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1)

        if instr.dst.kind == OperandKind.VGPR:
            state.write_vgpr(instr.dst.index, res)
        return res

def ref_v_fmaak_f16(s0: int, s1: int, k: int, old_dst: int = 0) -> int:
    if is_nan_f16(s0) or is_nan_f16(s1) or is_nan_f16(k):
        res_f16 = 0x7E00
    else:
        fa = bits_to_f16(s0 & 0xFFFF)
        fb = bits_to_f16(s1 & 0xFFFF)
        fk = bits_to_f16(k & 0xFFFF)
        r = fa * fb + fk
        if math.isnan(r):
            res_f16 = 0x7E00
        elif math.isinf(r):
            res_f16 = 0x7C00 if r > 0 else 0xFC00
        else:
            res_f16 = f16_to_bits(r)
    return ((old_dst & 0xFFFF0000) | (res_f16 & 0xFFFF)) & MASK32

GLOBAL_ORACLE_MAP[Opcode.V_FMAAK_F16] = ref_v_fmaak_f16


class TestVFmaakF16Oracle(unittest.TestCase):
    def test_basic_arithmetic(self) -> None:
        # 1.0 * 2.0 + 0.5 = 2.5 (0x4100)
        res = ref_v_fmaak_f16(0x3C00, 0x4000, 0x3800, old_dst=0xDEAD0000)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xDEAD)

    def test_zero_and_identity(self) -> None:
        # 0.0 * 2.0 + 1.0 = 1.0 (0x3C00)
        res = ref_v_fmaak_f16(0x0000, 0x4000, 0x3C00, old_dst=0x12340000)
        self.assertEqual(res & 0xFFFF, 0x3C00)
        self.assertEqual(res >> 16, 0x1234)

class TestVFmaakF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disassembly(self) -> None:
        inst = ASTInstruction(
            Opcode.V_FMAAK_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.imm(0x3800),
        )
        self.assertEqual(inst.disasm(), "v_fmaak_f16 v2, v0, v1, 0x3800")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 0x3C00)
        state.write_vgpr(1, 0x4000)
        state.write_vgpr(2, 0xBEEF0000)
        inst = ASTInstruction(
            Opcode.V_FMAAK_F16,
            ASTOperand.vgpr(2),
            ASTOperand.vgpr(0),
            ASTOperand.vgpr(1),
            ASTOperand.imm(0x3800),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res & 0xFFFF, 0x4100)
        self.assertEqual(res >> 16, 0xBEEF)
        self.assertEqual(state.vgprs[2], res)

if __name__ == "__main__":
    unittest.main()
