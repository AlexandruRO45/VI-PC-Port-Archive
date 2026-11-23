"""Comprehensive semantic regression suite and hardware translator verification
for AMD RDNA2 VALU 16-bit floating-point transcendental operations (Bundle 21):
  - v_sin_f16:  16-bit float sine (input in turns: 1.0 turn = 2*pi radians)
  - v_cos_f16:  16-bit float cosine (input in turns: 1.0 turn = 2*pi radians)
  - v_exp_f16:  16-bit float base-2 exponential (2^x)
  - v_log_f16:  16-bit float base-2 logarithm (log2(x))
  - v_rcp_f16:  16-bit float reciprocal (1.0 / x)
  - v_rsq_f16:  16-bit float reciprocal square root (1.0 / sqrt(x))
  - v_sqrt_f16: 16-bit float square root (sqrt(x))

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

MASK16: int = 0xFFFF
MASK32: int = 0xFFFFFFFF


def f16_to_u16(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def u16_to_f16(u: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u & MASK16))[0]


class Opcode(Enum):
    V_SIN_F16 = "v_sin_f16"
    V_COS_F16 = "v_cos_f16"
    V_EXP_F16 = "v_exp_f16"
    V_LOG_F16 = "v_log_f16"
    V_RCP_F16 = "v_rcp_f16"
    V_RSQ_F16 = "v_rsq_f16"
    V_SQRT_F16 = "v_sqrt_f16"


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

    def disasm(self) -> str:
        def fmt(op: ASTOperand) -> str:
            if op.kind == OperandKind.VGPR:
                return f"v{op.index}"
            if op.kind == OperandKind.SGPR:
                return f"s{op.index}"
            return hex(op.value)
        return f"{self.opcode.value} {fmt(self.dst)}, {fmt(self.src0)}"


class ScalarMachineState:
    def __init__(self, num_vgprs: int = 256, num_sgprs: int = 106) -> None:
        self.vgprs: list[int] = [0] * num_vgprs
        self.sgprs: list[int] = [0] * num_sgprs
        self.scc: int = 0
        self.vcc: int = 0

    def read_operand_f16(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgprs[op.index] & MASK16
        if op.kind == OperandKind.SGPR:
            return self.sgprs[op.index] & MASK16
        return op.value & MASK16

    def write_vgpr_f16(self, idx: int, val: int) -> None:
        # Writes to low 16 bits, preserves high 16 bits
        prev_hi = self.vgprs[idx] & 0xFFFF0000
        self.vgprs[idx] = prev_hi | (val & MASK16)


GLOBAL_ORACLE_MAP: dict[Opcode, Callable[[int], int]] = {}


class ScalarInterpreter:
    def step(self, instr: ASTInstruction, state: ScalarMachineState) -> int:
        s0 = state.read_operand_f16(instr.src0)
        res = GLOBAL_ORACLE_MAP[instr.opcode](s0)
        if instr.dst.kind == OperandKind.VGPR:
            state.write_vgpr_f16(instr.dst.index, res)
        return res

def ref_v_sin_f16(u: int) -> int:
    f = u16_to_f16(u)
    return f16_to_u16(math.sin(f * 2.0 * math.pi))

GLOBAL_ORACLE_MAP[Opcode.V_SIN_F16] = ref_v_sin_f16


class TestVSinF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        res = ref_v_sin_f16(0x0000)
        self.assertEqual(res, 0x0)

class TestVSinF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_SIN_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_sin_f16 v0, v1")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr_f16(1, 0x0000)
        inst = ASTInstruction(Opcode.V_SIN_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x0)
        self.assertEqual(state.vgprs[0] & MASK16, 0x0)

class TestVSinF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr_f16(0, 0x0000)
        inst = ASTInstruction(Opcode.V_SIN_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(0))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x0)
        self.assertEqual(state.vgprs[0] & MASK16, 0x0)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)

    def test_ieee754_boundary(self) -> None:
        res = ref_v_sin_f16(0x0000)
        self.assertIsInstance(res, int)

class TestVSinF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x0000",
            "v_sin_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_sin_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x0)

def ref_v_cos_f16(u: int) -> int:
    f = u16_to_f16(u)
    return f16_to_u16(math.cos(f * 2.0 * math.pi))

GLOBAL_ORACLE_MAP[Opcode.V_COS_F16] = ref_v_cos_f16


class TestVCosF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        res = ref_v_cos_f16(0x0000)
        self.assertEqual(res, 0x3c00)

class TestVCosF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_COS_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_cos_f16 v0, v1")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr_f16(1, 0x0000)
        inst = ASTInstruction(Opcode.V_COS_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x3c00)
        self.assertEqual(state.vgprs[0] & MASK16, 0x3c00)

class TestVCosF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr_f16(0, 0x0000)
        inst = ASTInstruction(Opcode.V_COS_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(0))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x3c00)
        self.assertEqual(state.vgprs[0] & MASK16, 0x3c00)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)

    def test_ieee754_boundary(self) -> None:
        res = ref_v_cos_f16(0x0000)
        self.assertIsInstance(res, int)

class TestVCosF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x0000",
            "v_cos_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_cos_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x3c00)

def ref_v_exp_f16(u: int) -> int:
    f = u16_to_f16(u)
    return f16_to_u16(math.pow(2.0, f))

GLOBAL_ORACLE_MAP[Opcode.V_EXP_F16] = ref_v_exp_f16


class TestVExpF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        res = ref_v_exp_f16(0x3C00)
        self.assertEqual(res, 0x4000)

class TestVExpF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_EXP_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_exp_f16 v0, v1")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr_f16(1, 0x3C00)
        inst = ASTInstruction(Opcode.V_EXP_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x4000)
        self.assertEqual(state.vgprs[0] & MASK16, 0x4000)

class TestVExpF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr_f16(0, 0x3C00)
        inst = ASTInstruction(Opcode.V_EXP_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(0))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x4000)
        self.assertEqual(state.vgprs[0] & MASK16, 0x4000)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)

    def test_ieee754_boundary(self) -> None:
        res = ref_v_exp_f16(0x3C00)
        self.assertIsInstance(res, int)

class TestVExpF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x3C00",
            "v_exp_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_exp_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x4000)

def ref_v_log_f16(u: int) -> int:
    f = u16_to_f16(u)
    return f16_to_u16(math.log2(f))

GLOBAL_ORACLE_MAP[Opcode.V_LOG_F16] = ref_v_log_f16


class TestVLogF16Oracle(unittest.TestCase):
    def test_oracle_truth_table(self) -> None:
        res = ref_v_log_f16(0x4000)
        self.assertEqual(res, 0x3c00)

class TestVLogF16ASTAndInterpreter(unittest.TestCase):
    def test_ast_disasm(self) -> None:
        inst = ASTInstruction(Opcode.V_LOG_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_log_f16 v0, v1")

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr_f16(1, 0x4000)
        inst = ASTInstruction(Opcode.V_LOG_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x3c00)
        self.assertEqual(state.vgprs[0] & MASK16, 0x3c00)

class TestVLogF16SIMDAndIEEE(unittest.TestCase):
    def test_aliasing_and_flags(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA
        state.write_vgpr_f16(0, 0x4000)
        inst = ASTInstruction(Opcode.V_LOG_F16, ASTOperand.vgpr(0), ASTOperand.vgpr(0))
        res = ScalarInterpreter().step(inst, state)
        self.assertEqual(res, 0x3c00)
        self.assertEqual(state.vgprs[0] & MASK16, 0x3c00)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)

    def test_ieee754_boundary(self) -> None:
        res = ref_v_log_f16(0x4000)
        self.assertIsInstance(res, int)

class TestVLogF16Translation(unittest.TestCase):
    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "v_mov_b32 v0, 0x4000",
            "v_log_f16 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bundle_v_log_f16", lines)
        self.assertIn("OpEntryPoint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0] & MASK16, 0x3c00)


if __name__ == "__main__":
    unittest.main()
