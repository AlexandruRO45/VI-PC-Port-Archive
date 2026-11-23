"""Semantic reference oracle and regression tests for RDNA2 v_dot4_i32_i8.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_dot4_i32_i8 is a VOP3P instruction computing
  a four-component dot product of 8-bit signed integers with a 32-bit integer addend:
      D.i32 = (S0.b0 * S1.b0) + (S0.b1 * S1.b1) + (S0.b2 * S1.b2) + (S0.b3 * S1.b3) + S2.i32   (mod 2^32)

  - S0 and S1 each hold four packed 8-bit signed integer components (range -128..127).
  - Each byte is sign-extended to 32 bits before multiplication.
  - S2 is a 32-bit signed integer addend/accumulator.
  - The sum of products lies in [-65024, 65536]. Added to S2, it wraps modulo 2^32.
  - With optional clamp modifier, the result saturates to signed 32-bit range [-2^31, 2^31-1].
  - Condition codes: Neither SCC nor VCC is modified.
  - Floating-point representations (NaN, Inf, denormals) are treated strictly as raw bit patterns;
    no flushing or quietening occurs.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import support  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

MASK32: int = 0xFFFFFFFF
INT32_MIN: int = -0x80000000
INT32_MAX: int = 0x7FFFFFFF


def u32(val: int) -> int:
    return val & MASK32


def s32(val: int) -> int:
    v = val & MASK32
    return v - 0x100000000 if v & 0x80000000 else v


def s8(val: int) -> int:
    v = val & 0xFF
    return v - 256 if v & 0x80 else v


def pack_i8(b0: int, b1: int, b2: int, b3: int) -> int:
    return ((b0 & 0xFF) |
            ((b1 & 0xFF) << 8) |
            ((b2 & 0xFF) << 16) |
            ((b3 & 0xFF) << 24)) & MASK32


def ref_dot4_i32_i8(s0: int, s1: int, s2: int, clamp: bool = False) -> int:
    """RDNA2 v_dot4_i32_i8 reference mathematical oracle."""
    b0_0, b0_1, b0_2, b0_3 = (s8(s0 >> (8 * i)) for i in range(4))
    b1_0, b1_1, b1_2, b1_3 = (s8(s1 >> (8 * i)) for i in range(4))
    dot = (b0_0 * b1_0) + (b0_1 * b1_1) + (b0_2 * b1_2) + (b0_3 * b1_3)
    acc = s32(s2)
    total = acc + dot
    if clamp:
        total = max(INT32_MIN, min(INT32_MAX, total))
    return total & MASK32

class TestValuDot4I32I8Oracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle unit tests."""

    def test_oracle_basic_products(self) -> None:
        # 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        s0 = pack_i8(1, 2, 3, 4)
        s1 = pack_i8(5, 6, 7, 8)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 70)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 1000), 1070)

    def test_oracle_commutativity_of_multiplicands(self) -> None:
        s0 = pack_i8(10, -20, 30, -40)
        s1 = pack_i8(-2, 3, -4, 5)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 123), ref_dot4_i32_i8(s1, s0, 123))

    def test_oracle_signed_byte_sign_extension(self) -> None:
        # 0xFF is -1, 0x80 is -128, 0x7F is 127
        s0 = pack_i8(0xFF, 0x80, 0x7F, 0x01)  # -1, -128, 127, 1
        s1 = pack_i8(0x01, 0x01, 0x01, 0x01)  # 1, 1, 1, 1
        # sum = (-1)*1 + (-128)*1 + 127*1 + 1*1 = -1 - 128 + 127 + 1 = -1
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 0xFFFFFFFF)

    def test_oracle_extreme_positive_dot(self) -> None:
        # Max positive product per byte: -128 * -128 = 16384. Four bytes = 65536.
        s0 = pack_i8(0x80, 0x80, 0x80, 0x80)
        s1 = pack_i8(0x80, 0x80, 0x80, 0x80)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 65536)

    def test_oracle_extreme_negative_dot(self) -> None:
        # Max negative product per byte: -128 * 127 = -16256. Four bytes = -65024.
        s0 = pack_i8(0x80, 0x80, 0x80, 0x80)
        s1 = pack_i8(0x7F, 0x7F, 0x7F, 0x7F)
        # -65024 in two's complement 32-bit: 0xFFFF0200
        self.assertEqual(ref_dot4_i32_i8(s0, s1, 0), 0xFFFF0200)

    def test_oracle_overflow_wrapping_without_clamp(self) -> None:
        # INT32_MAX + 65536 wraps around into negative
        s0 = pack_i8(0x80, 0x80, 0x80, 0x80)
        s1 = pack_i8(0x80, 0x80, 0x80, 0x80)
        self.assertEqual(ref_dot4_i32_i8(s0, s1, INT32_MAX, clamp=False), 0x8000FFFF)

    def test_oracle_clamp_saturation(self) -> None:
        s0_pos = pack_i8(0x80, 0x80, 0x80, 0x80)
        s1_pos = pack_i8(0x80, 0x80, 0x80, 0x80)
        # Overflow saturates to INT32_MAX
        self.assertEqual(ref_dot4_i32_i8(s0_pos, s1_pos, INT32_MAX, clamp=True), INT32_MAX)

        s0_neg = pack_i8(0x80, 0x80, 0x80, 0x80)
        s1_neg = pack_i8(0x7F, 0x7F, 0x7F, 0x7F)
        # Underflow saturates to INT32_MIN
        self.assertEqual(ref_dot4_i32_i8(s0_neg, s1_neg, 0x80000000, clamp=True), 0x80000000)

    def test_oracle_zero_multiplicands(self) -> None:
        self.assertEqual(ref_dot4_i32_i8(0, 0x12345678, 0xDEADBEEF), 0xDEADBEEF)
        self.assertEqual(ref_dot4_i32_i8(0x12345678, 0, 0xCAFEBABE), 0xCAFEBABE)

class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    IMM = auto()


@dataclass(frozen=True)
class Operand:
    kind: OperandKind
    index: int = 0
    value: int = 0

    @staticmethod
    def vgpr(idx: int) -> Operand:
        return Operand(OperandKind.VGPR, index=idx)

    @staticmethod
    def sgpr(idx: int) -> Operand:
        return Operand(OperandKind.SGPR, index=idx)

    @staticmethod
    def imm(val: int) -> Operand:
        return Operand(OperandKind.IMM, value=val & MASK32)


@dataclass(frozen=True)
class VDot4I32I8Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def disasm(self) -> str:
        def fmt(op: Operand) -> str:
            if op.kind == OperandKind.VGPR:
                return f"v{op.index}"
            if op.kind == OperandKind.SGPR:
                return f"s{op.index}"
            return f"{op.value:#x}"
        c = " clamp" if self.clamp else ""
        return f"v_dot4_i32_i8 {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}, {fmt(self.src2)}{c}"


class TestValuDot4I32I8AST(unittest.TestCase):
    """Stage 2: AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        instr = VDot4I32I8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        self.assertEqual(instr.disasm(), "v_dot4_i32_i8 v0, v1, v2, v3")

    def test_ast_with_clamp_modifier(self) -> None:
        instr = VDot4I32I8Instruction(
            dst=Operand.vgpr(4),
            src0=Operand.sgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.imm(0x100),
            clamp=True,
        )
        self.assertEqual(instr.disasm(), "v_dot4_i32_i8 v4, s0, v1, 0x100 clamp")

class ScalarMachineState:
    def __init__(self, num_vgprs: int = 256, num_sgprs: int = 106) -> None:
        self.vgprs: list[int] = [0] * num_vgprs
        self.sgprs: list[int] = [0] * num_sgprs
        self.scc: int = 0
        self.vcc: int = 0

    def read_operand(self, op: Operand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgprs[op.index] & MASK32
        if op.kind == OperandKind.SGPR:
            return self.sgprs[op.index] & MASK32
        return op.value & MASK32

    def write_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32


class ScalarInterpreter:
    def step(self, instr: VDot4I32I8Instruction, state: ScalarMachineState) -> int:
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        s2 = state.read_operand(instr.src2)
        res = ref_dot4_i32_i8(s0, s1, s2, clamp=instr.clamp)
        if instr.dst.kind == OperandKind.VGPR:
            state.write_vgpr(instr.dst.index, res)
        return res


class TestValuDot4I32I8ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter."""

    def test_scalar_step_all_vgprs(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(1, pack_i8(1, 2, 3, 4))
        state.write_vgpr(2, pack_i8(5, 6, 7, 8))
        state.write_vgpr(3, 500)
        instr = VDot4I32I8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        res = ScalarInterpreter().step(instr, state)
        self.assertEqual(res, 570)
        self.assertEqual(state.vgprs[0], 570)

    def test_scalar_step_sgpr_and_imm_operands(self) -> None:
        state = ScalarMachineState()
        state.sgprs[0] = pack_i8(-1, 2, -3, -4)
        state.write_vgpr(1, pack_i8(5, -6, -7, 8))
        instr = VDot4I32I8Instruction(
            dst=Operand.vgpr(2),
            src0=Operand.sgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.imm(100),
        )
        # -1*5 + 2*(-6) + (-3)*(-7) + (-4)*8 = -5 - 12 + 21 - 32 = -28. + 100 = 72.
        res = ScalarInterpreter().step(instr, state)
        self.assertEqual(res, 72)
        self.assertEqual(state.vgprs[2], 72)

MASK64: int = 0xFFFFFFFFFFFFFFFF


class SIMDMachineState:
    def __init__(self, num_lanes: int = 64, exec_mask: int = MASK64) -> None:
        self.num_lanes = num_lanes
        self.exec_mask = exec_mask
        self.lanes_vgpr: list[list[int]] = [[0] * 256 for _ in range(num_lanes)]
        self.sgprs: list[int] = [0] * 106
        self.scc: int = 0

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def read_lane_operand(self, lane: int, op: Operand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.lanes_vgpr[lane][op.index] & MASK32
        if op.kind == OperandKind.SGPR:
            return self.sgprs[op.index] & MASK32
        return op.value & MASK32


class SIMDInterpreter:
    def step(self, instr: VDot4I32I8Instruction, state: SIMDMachineState) -> list[Optional[int]]:
        results: list[Optional[int]] = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                s0 = state.read_lane_operand(lane, instr.src0)
                s1 = state.read_lane_operand(lane, instr.src1)
                s2 = state.read_lane_operand(lane, instr.src2)
                res = ref_dot4_i32_i8(s0, s1, s2, clamp=instr.clamp)
                if instr.dst.kind == OperandKind.VGPR:
                    state.lanes_vgpr[lane][instr.dst.index] = res
                results.append(res)
            else:
                results.append(None)
        return results


class TestValuDot4I32I8SIMDDivergence(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_divergence_even_odd_mask(self) -> None:
        state = SIMDMachineState(num_lanes=32, exec_mask=0x55555555)
        for lane in range(32):
            state.lanes_vgpr[lane][1] = pack_i8(1, 1, 1, 1)
            state.lanes_vgpr[lane][2] = pack_i8(lane, 0, 0, 0)
            state.lanes_vgpr[lane][3] = 10
            state.lanes_vgpr[lane][0] = 0xDEADBEEF

        instr = VDot4I32I8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(3),
        )
        results = SIMDInterpreter().step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                expected = lane + 10
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.lanes_vgpr[lane][0], expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.lanes_vgpr[lane][0], 0xDEADBEEF)

    def test_wave64_zero_mask_preservation(self) -> None:
        state = SIMDMachineState(num_lanes=64, exec_mask=0)
        for lane in range(64):
            state.lanes_vgpr[lane][0] = 0xCAFE0000 | lane

        instr = VDot4I32I8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.imm(pack_i8(1, 2, 3, 4)),
            src1=Operand.imm(pack_i8(5, 6, 7, 8)),
            src2=Operand.imm(0),
        )
        results = SIMDInterpreter().step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(64):
            self.assertEqual(state.lanes_vgpr[lane][0], 0xCAFE0000 | lane)

class TestValuDot4I32I8Aliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation."""

    def test_aliasing_dst_src0(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, pack_i8(1, 2, 3, 4))
        state.write_vgpr(1, pack_i8(5, 6, 7, 8))
        state.write_vgpr(2, 10)
        instr = VDot4I32I8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        ScalarInterpreter().step(instr, state)
        self.assertEqual(state.vgprs[0], 70 + 10)

    def test_aliasing_dst_src1(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, pack_i8(5, 6, 7, 8))
        state.write_vgpr(1, pack_i8(1, 2, 3, 4))
        state.write_vgpr(2, 20)
        instr = VDot4I32I8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(0),
            src2=Operand.vgpr(2),
        )
        ScalarInterpreter().step(instr, state)
        self.assertEqual(state.vgprs[0], 70 + 20)

    def test_aliasing_dst_src2_accumulator(self) -> None:
        state = ScalarMachineState()
        state.write_vgpr(0, 100)
        state.write_vgpr(1, pack_i8(1, 2, 3, 4))
        state.write_vgpr(2, pack_i8(5, 6, 7, 8))
        instr = VDot4I32I8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
            src1=Operand.vgpr(2),
            src2=Operand.vgpr(0),
        )
        ScalarInterpreter().step(instr, state)
        self.assertEqual(state.vgprs[0], 70 + 100)

    def test_aliasing_all_operands_same_vgpr(self) -> None:
        state = ScalarMachineState()
        val = pack_i8(2, 2, 2, 2)
        state.write_vgpr(0, val)
        instr = VDot4I32I8Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(0),
            src2=Operand.vgpr(0),
        )
        # dot = 4 * (2 * 2) = 16. acc = 0x02020202. total = val + 16
        ScalarInterpreter().step(instr, state)
        self.assertEqual(state.vgprs[0], val + 16)

if __name__ == "__main__":
    unittest.main()
