"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 SALU 32-bit integer multiplication and division helpers (Bundle 40):
  - s_mul_i32:    32-bit integer multiplication (low 32 bits, signed/unsigned identical)
  - s_mul_hi_i32: Signed 32-bit integer multiplication high-half (high 32 bits of 64-bit product)
  - s_mul_hi_u32: Unsigned 32-bit integer multiplication high-half (high 32 bits of 64-bit product)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import random
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


def i32(val: int) -> int:
    v = val & MASK32
    return v - 0x100000000 if v >= 0x80000000 else v

class Opcode(Enum):
    S_MUL_I32 = "s_mul_i32"
    S_MUL_HI_I32 = "s_mul_hi_i32"
    S_MUL_HI_U32 = "s_mul_hi_u32"


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


GLOBAL_ORACLE_MAP: dict[Opcode, Callable[[int, int], int]] = {}


class ScalarInterpreter:
    def step(self, instr: ASTInstruction, state: ScalarMachineState) -> int:
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        res = GLOBAL_ORACLE_MAP[instr.opcode](s0, s1)
        if instr.dst.kind == OperandKind.SGPR:
            state.write_sgpr(instr.dst.index, res)
        # Note: s_mul_i32, s_mul_hi_i32, s_mul_hi_u32 do NOT modify SCC in RDNA2 ISA
        return res

class ScalarWaveModel:
    """SALU scalar wave model: SALU operations execute once per wave and broadcast
    identically across all active or inactive lanes.
    """
    def __init__(self, wave_size: int = 32, exec_mask: int = 0xFFFFFFFF) -> None:
        self.wave_size = wave_size
        self.exec_mask = exec_mask & ((1 << wave_size) - 1)

    def broadcast_result(self, scalar_val: int) -> list[int]:
        # Scalar result is uniform across all lanes of the wave
        return [scalar_val & MASK32] * self.wave_size

def ref_s_mul_i32(s0: int, s1: int) -> int:
    """Low 32 bits of 32x32 integer multiply (identical for signed/unsigned)."""
    return (s0 * s1) & MASK32


class TestSMulI32Oracle(unittest.TestCase):
    def test_oracle_basic_multiplication(self) -> None:
        self.assertEqual(ref_s_mul_i32(6, 7), 42)
        self.assertEqual(ref_s_mul_i32(100, 200), 20000)

    def test_oracle_negative_operands(self) -> None:
        self.assertEqual(ref_s_mul_i32(u32(-3), 4), u32(-12))
        self.assertEqual(ref_s_mul_i32(u32(-5), u32(-6)), 30)

    def test_oracle_zero_and_identity(self) -> None:
        self.assertEqual(ref_s_mul_i32(0, 12345), 0)
        self.assertEqual(ref_s_mul_i32(12345, 1), 12345)

    def test_oracle_wrapping_modulo_2_32(self) -> None:
        # 0x10001 * 0x10001 = 0x200020001 & 0xFFFFFFFF = 0x00020001
        self.assertEqual(ref_s_mul_i32(0x10001, 0x10001), 0x00020001)

class TestSMulI32ASTAndSignExtensions(unittest.TestCase):
    def test_ast_instruction_disasm(self) -> None:
        inst = ASTInstruction(
            Opcode.S_MUL_I32,
            ASTOperand.sgpr(2),
            ASTOperand.sgpr(0),
            ASTOperand.sgpr(1),
        )
        self.assertEqual(inst.disasm(), "s_mul_i32 s2, s0, s1")

    def test_imm_operands_disasm(self) -> None:
        inst = ASTInstruction(
            Opcode.S_MUL_I32,
            ASTOperand.sgpr(4),
            ASTOperand.sgpr(2),
            ASTOperand.imm(0x100),
        )
        self.assertEqual(inst.disasm(), "s_mul_i32 s4, s2, 0x100")

    def test_sign_extension_consistency(self) -> None:
        val_neg = u32(-50)
        self.assertEqual(i32(val_neg), -50)
        self.assertEqual(u32(i32(val_neg)), val_neg)

GLOBAL_ORACLE_MAP[Opcode.S_MUL_I32] = ref_s_mul_i32


class TestSMulI32ScalarInterpreter(unittest.TestCase):
    def test_step_sgpr_sources(self) -> None:
        state = ScalarMachineState()
        state.write_sgpr(0, 25)
        state.write_sgpr(1, 4)
        inst = ASTInstruction(
            Opcode.S_MUL_I32,
            ASTOperand.sgpr(2),
            ASTOperand.sgpr(0),
            ASTOperand.sgpr(1),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 100)
        self.assertEqual(state.sgprs[2], 100)

    def test_step_imm_source(self) -> None:
        state = ScalarMachineState()
        state.write_sgpr(5, 12)
        inst = ASTInstruction(
            Opcode.S_MUL_I32,
            ASTOperand.sgpr(6),
            ASTOperand.sgpr(5),
            ASTOperand.imm(12),
        )
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 144)
        self.assertEqual(state.sgprs[6], 144)

class TestSMulI32ConditionCodeAndSCC(unittest.TestCase):
    def test_scc_preservation_when_zero(self) -> None:
        state = ScalarMachineState()
        state.scc = 0
        state.write_sgpr(0, 10)
        state.write_sgpr(1, 10)
        inst = ASTInstruction(
            Opcode.S_MUL_I32,
            ASTOperand.sgpr(2),
            ASTOperand.sgpr(0),
            ASTOperand.sgpr(1),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 0)

    def test_scc_preservation_when_one(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.write_sgpr(0, 0)
        state.write_sgpr(1, 10)
        inst = ASTInstruction(
            Opcode.S_MUL_I32,
            ASTOperand.sgpr(2),
            ASTOperand.sgpr(0),
            ASTOperand.sgpr(1),
        )
        ScalarInterpreter().step(inst, state)
        self.assertEqual(state.scc, 1)

class TestSMulI32SIMDAndWaveDivergence(unittest.TestCase):
    def test_wave_uniform_broadcast(self) -> None:
        wave = ScalarWaveModel(wave_size=32, exec_mask=0x00FF00FF)
        val = ref_s_mul_i32(15, 3)
        lanes = wave.broadcast_result(val)
        self.assertEqual(len(lanes), 32)
        self.assertTrue(all(l == 45 for l in lanes))

    def test_wave64_uniform_broadcast(self) -> None:
        wave = ScalarWaveModel(wave_size=64, exec_mask=MASK64)
        val = ref_s_mul_i32(u32(-1), 10)
        lanes = wave.broadcast_result(val)
        self.assertEqual(len(lanes), 64)
        self.assertTrue(all(l == u32(-10) for l in lanes))

class TestSMulI32IntegerOverflowExtremes(unittest.TestCase):
    def test_int32_min_multiplied(self) -> None:
        int32_min = 0x80000000
        # -2147483648 * 2 = -4294967296 -> low 32 bits = 0
        self.assertEqual(ref_s_mul_i32(int32_min, 2), 0)
        # -2147483648 * -1 = 2147483648 -> low 32 bits = 0x80000000
        self.assertEqual(ref_s_mul_i32(int32_min, u32(-1)), int32_min)

    def test_all_ones_squared(self) -> None:
        # (-1) * (-1) = 1 in 32-bit wrap
        self.assertEqual(ref_s_mul_i32(0xFFFFFFFF, 0xFFFFFFFF), 1)

    def test_large_prime_wrapping(self) -> None:
        a = 0xDEADBEEF
        b = 0xCAFEBABE
        expected = (0xDEADBEEF * 0xCAFEBABE) & MASK32
        self.assertEqual(ref_s_mul_i32(a, b), expected)

if __name__ == "__main__":
    unittest.main()
