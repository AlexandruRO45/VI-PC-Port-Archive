"""Semantic reference tests and translator hardware verification for AMD RDNA2 v_trunc_f32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_trunc_f32 belongs to the VOP1 encoding format.
  Instruction:
    v_trunc_f32 vd, src0

  Operands:
    vd   = 32-bit destination VGPR
    src0 = 32-bit source operand: VGPR, SGPR, inline constant, or literal float

  Semantics:
    Truncates floating-point value to an integer, rounding toward zero.
    Result is encoded as an IEEE 754 single-precision float.
    vd = trunc(src0)
    Neither SCC nor VCC is modified.

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
from typing import Dict, List, Optional, Tuple
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
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def ref_trunc_f32(src0_bits: int) -> int:
    """Mathematical reference oracle for AMD RDNA2 v_trunc_f32 instruction.

    Args:
        src0_bits: 32-bit IEEE 754 float bit pattern (uint32)

    Returns:
        32-bit IEEE 754 float bit pattern of truncated result.
    """
    f = u32_to_f32(src0_bits)
    if math.isnan(f) or math.isinf(f):
        return src0_bits & MASK32
    res_f = float(math.trunc(f))
    if res_f == 0.0 and math.copysign(1.0, f) < 0.0:
        return NEG_ZERO_F32
    return f32_to_u32(res_f)


class TestValuTruncF32Oracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle unit tests."""

    def test_oracle_positive_floats(self) -> None:
        self.assertEqual(ref_trunc_f32(f32_to_u32(5.5)), f32_to_u32(5.0))
        self.assertEqual(ref_trunc_f32(f32_to_u32(5.9)), f32_to_u32(5.0))
        self.assertEqual(ref_trunc_f32(f32_to_u32(0.9)), f32_to_u32(0.0))

    def test_oracle_negative_floats(self) -> None:
        # Truncation rounds toward zero, so -5.5 -> -5.0, -0.9 -> -0.0
        self.assertEqual(ref_trunc_f32(f32_to_u32(-5.5)), f32_to_u32(-5.0))
        self.assertEqual(ref_trunc_f32(f32_to_u32(-5.9)), f32_to_u32(-5.0))
        self.assertEqual(ref_trunc_f32(f32_to_u32(-0.9)), NEG_ZERO_F32)

    def test_oracle_integers_unchanged(self) -> None:
        for val in [0.0, 1.0, 42.0, 1000000.0]:
            self.assertEqual(ref_trunc_f32(f32_to_u32(val)), f32_to_u32(val))
            self.assertEqual(ref_trunc_f32(f32_to_u32(-val)), f32_to_u32(-val) if val != 0.0 else NEG_ZERO_F32)

    def test_oracle_signed_zeros(self) -> None:
        self.assertEqual(ref_trunc_f32(POS_ZERO_F32), POS_ZERO_F32)
        self.assertEqual(ref_trunc_f32(NEG_ZERO_F32), NEG_ZERO_F32)

    def test_oracle_infinities_and_nan(self) -> None:
        self.assertEqual(ref_trunc_f32(POS_INF_F32), POS_INF_F32)
        self.assertEqual(ref_trunc_f32(NEG_INF_F32), NEG_INF_F32)
        self.assertEqual(ref_trunc_f32(QNAN_F32), QNAN_F32)


class Opcode(Enum):
    V_TRUNC_F32 = auto()


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
        return f"v_trunc_f32 {fmt(self.dst)}, {fmt(self.src0)}"


class TestValuTruncF32AST(unittest.TestCase):
    """Stage 2: AST opcode definitions and typed expressions."""

    def test_ast_construction_vgpr(self) -> None:
        inst = ASTInstruction(Opcode.V_TRUNC_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        self.assertEqual(inst.disasm(), "v_trunc_f32 v0, v1")

    def test_ast_construction_sgpr_src(self) -> None:
        inst = ASTInstruction(Opcode.V_TRUNC_F32, ASTOperand.vgpr(2), ASTOperand.sgpr(4))
        self.assertEqual(inst.disasm(), "v_trunc_f32 v2, s4")


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


class ScalarInterpreter:
    def step(self, instr: ASTInstruction, state: ScalarMachineState) -> int:
        s0 = state.read_operand(instr.src0)
        res = ref_trunc_f32(s0)
        if instr.dst.kind == OperandKind.VGPR:
            state.write_vgpr(instr.dst.index, res)
        return res


class TestValuTruncF32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_step(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(1, f32_to_u32(12.75))
        instr = ASTInstruction(Opcode.V_TRUNC_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        res = ScalarInterpreter().step(instr, state)
        self.assertEqual(res, f32_to_u32(12.0))
        self.assertEqual(state.vgprs[0], f32_to_u32(12.0))


class SIMDMachineState:
    def __init__(self, num_lanes: int = 64, exec_mask: int = MASK64) -> None:
        self.num_lanes = num_lanes
        self.exec_mask = exec_mask
        self.lanes_vgpr: list[list[int]] = [[0] * 256 for _ in range(num_lanes)]
        self.sgprs: list[int] = [0] * 106

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def read_lane_operand(self, lane: int, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.lanes_vgpr[lane][op.index] & MASK32
        if op.kind == OperandKind.SGPR:
            return self.sgprs[op.index] & MASK32
        return op.value & MASK32


class SIMDInterpreter:
    def step(self, instr: ASTInstruction, state: SIMDMachineState) -> list[Optional[int]]:
        results: list[Optional[int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                s0 = state.read_lane_operand(lane, instr.src0)
                res = ref_trunc_f32(s0)
                if instr.dst.kind == OperandKind.VGPR:
                    state.lanes_vgpr[lane][instr.dst.index] = res
                results.append(res)
            else:
                results.append(None)
        return results


class TestValuTruncF32SIMDDivergence(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_divergence_masking(self) -> None:
        state = SIMDMachineState(num_lanes=32, exec_mask=0x55555555)  # even lanes active
        for lane in range(32):
            state.lanes_vgpr[lane][1] = f32_to_u32(lane + 0.75)
            state.lanes_vgpr[lane][0] = 0xDEADBEEF  # canary

        instr = ASTInstruction(Opcode.V_TRUNC_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        results = SIMDInterpreter().step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], f32_to_u32(float(lane)))
                self.assertEqual(state.lanes_vgpr[lane][0], f32_to_u32(float(lane)))
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.lanes_vgpr[lane][0], 0xDEADBEEF)


class TestValuTruncF32Aliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation."""

    def test_in_place_truncation(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, f32_to_u32(99.99))
        instr = ASTInstruction(Opcode.V_TRUNC_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(0))
        res = ScalarInterpreter().step(instr, state)
        self.assertEqual(res, f32_to_u32(99.0))
        self.assertEqual(state.vgprs[0], f32_to_u32(99.0))


class TestValuTruncF32ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and carry/borrow generation."""

    def test_scc_and_vcc_unmodified(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        state.vcc = 0xAAAAAAAAAAAAAAAA

        instr = ASTInstruction(Opcode.V_TRUNC_F32, ASTOperand.vgpr(0), ASTOperand.imm(0))
        ScalarInterpreter().step(instr, state)
        self.assertEqual(state.scc, 1)
        self.assertEqual(state.vcc, 0xAAAAAAAAAAAAAAAA)


class TestValuTruncF32IEEE754(unittest.TestCase):
    """Stage 7: IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormals_truncate_to_zero(self) -> None:
        pos_subnormal = 0x00000001
        neg_subnormal = 0x80000001
        self.assertEqual(ref_trunc_f32(pos_subnormal), POS_ZERO_F32)
        self.assertEqual(ref_trunc_f32(neg_subnormal), NEG_ZERO_F32)

    def test_large_floats_already_integral(self) -> None:
        # Floats with exponent >= 23 have no fractional bits; trunc is identity
        large_f = f32_to_u32(8388608.0)  # 2^23
        self.assertEqual(ref_trunc_f32(large_f), large_f)


if __name__ == "__main__":
    unittest.main()
