from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for v_min3_i32.

RDNA2 Instruction Set Architecture, VOP3:
    v_min3_i32 D, S0, S1, S2

Semantic operation:
    D.i32 = min(to_s32(S0), to_s32(S1), to_s32(S2))
    D.u32 = to_u32(D.i32)

Outputs:
    D: 32-bit destination VGPR storing the signed 32-bit minimum of the three input sources.

Standard library Python only (unittest, struct, math, dataclasses, enum, typing).
Strictly zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Dict, List, Optional, Set, Tuple, Union
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000
MIN_INT32: int = -0x8000_0000
MAX_INT32: int = 0x7FFF_FFFF

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def to_s32(val: int) -> int:
    """Convert 32-bit unsigned integer to signed two's-complement integer."""
    val = val & MASK32
    return val - 0x1_0000_0000 if (val & SIGN_BIT32) else val


def to_u32(val: int) -> int:
    """Convert signed integer to 32-bit unsigned integer representation."""
    return val & MASK32


def v_min3_i32_oracle(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical model for v_min3_i32.

    Computes the minimum of three signed 32-bit integers.
    """
    v0 = to_s32(s0)
    v1 = to_s32(s1)
    v2 = to_s32(s2)
    return to_u32(min(v0, v1, v2))


class VMin3I32Oracle:
    """Architectural model and hardware-rule validator for v_min3_i32."""

    @staticmethod
    def eval(s0: int, s1: int, s2: int) -> int:
        return v_min3_i32_oracle(s0, s1, s2)

    @staticmethod
    def build_compute_shader(s0: int, s1: int, s2: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test v_min3_i32."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_min3_i32_hw"
        lines = [
            "s_mov_b64 exec, -1",
            f"v_mov_b32_e32 v1, {to_s32(s0)}",
            f"v_mov_b32_e32 v2, {to_s32(s1)}",
            f"v_mov_b32_e32 v3, {to_s32(s2)}",
            "v_min3_i32 v4, v1, v2, v3",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, s1: int, s2: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = VMin3I32Oracle.build_compute_shader(s0, s1, s2)
        return support.translate_lines(name, lines, fssm=fssm)


class TestVMin3I32Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_signed_negatives_vs_positives(self):
        # In signed comparison, -100 is less than 50 and 100
        # -100 is 0xFFFFFF9C
        res = VMin3I32Oracle.eval(100, 50, to_u32(-100))
        self.assertEqual(to_s32(res), -100)

    def test_oracle_signed_negative_one_less_than_zero(self):
        # 0xFFFFFFFF (-1) vs 0 vs 1
        res = VMin3I32Oracle.eval(0, 0xFFFF_FFFF, 1)
        self.assertEqual(to_s32(res), -1)
        self.assertEqual(res, 0xFFFF_FFFF)

    def test_oracle_extreme_min_int32(self):
        # -2147483648 (0x80000000) is the absolute minimum signed 32-bit integer
        res = VMin3I32Oracle.eval(0x8000_0000, 0, 0x7FFF_FFFF)
        self.assertEqual(res, 0x8000_0000)
        self.assertEqual(to_s32(res), MIN_INT32)

    def test_oracle_identical_values(self):
        val = to_u32(-42)
        self.assertEqual(VMin3I32Oracle.eval(val, val, val), val)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = VMin3I32Oracle.translate_and_validate(10, to_u32(-20), 30)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpFunction", spv)
        self.assertIn("OpReturn", spv)

# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------

class RegisterKind(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL32 = auto()


@dataclass(frozen=True)
class RegOperand:
    kind: RegisterKind
    index: int = 0

    def format(self) -> str:
        if self.kind == RegisterKind.VGPR:
            return f"v{self.index}"
        elif self.kind == RegisterKind.SGPR:
            return f"s{self.index}"
        elif self.kind == RegisterKind.LITERAL32:
            return f"{self.index:#x}"
        return f"reg_{self.index}"


@dataclass
class VMin3I32Instruction:
    dst: RegOperand
    src0: RegOperand
    src1: RegOperand
    src2: RegOperand
    opcode: str = "v_min3_i32"

    def validate(self) -> None:
        if self.opcode != "v_min3_i32":
            raise ValueError(f"Invalid opcode {self.opcode}")
        if self.dst.kind != RegisterKind.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.kind}")
        if self.dst.index > 255:
            raise ValueError(f"Destination VGPR index out of range: v{self.dst.index}")
        for src, name in ((self.src0, "src0"), (self.src1, "src1"), (self.src2, "src2")):
            if src.kind == RegisterKind.VGPR and src.index > 255:
                raise ValueError(f"{name} VGPR index out of range: v{src.index}")
            elif src.kind == RegisterKind.SGPR and src.index > 105:
                raise ValueError(f"{name} SGPR index out of range: s{src.index}")

    def emit_assembly(self) -> str:
        self.validate()
        return f"{self.opcode} {self.dst.format()}, {self.src0.format()}, {self.src1.format()}, {self.src2.format()}"


class TestVMin3I32AST(unittest.TestCase):
    """Unit tests for symbolic AST representation and validation."""

    def test_ast_valid_vgpr_operands(self):
        instr = VMin3I32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=4),
            src0=RegOperand(RegisterKind.VGPR, index=1),
            src1=RegOperand(RegisterKind.VGPR, index=2),
            src2=RegOperand(RegisterKind.VGPR, index=3)
        )
        self.assertEqual(instr.emit_assembly(), "v_min3_i32 v4, v1, v2, v3")

    def test_ast_valid_sgpr_and_literal_operands(self):
        instr = VMin3I32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=5),
            src0=RegOperand(RegisterKind.SGPR, index=8),
            src1=RegOperand(RegisterKind.VGPR, index=2),
            src2=RegOperand(RegisterKind.LITERAL32, index=0x100)
        )
        self.assertEqual(instr.emit_assembly(), "v_min3_i32 v5, s8, v2, 0x100")

    def test_ast_invalid_sgpr_destination(self):
        instr = VMin3I32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=0),
            src0=RegOperand(RegisterKind.VGPR, index=1),
            src1=RegOperand(RegisterKind.VGPR, index=2),
            src2=RegOperand(RegisterKind.VGPR, index=3)
        )
        with self.assertRaises(TypeError):
            instr.validate()

    def test_ast_vgpr_out_of_range(self):
        instr = VMin3I32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=256),
            src0=RegOperand(RegisterKind.VGPR, index=1),
            src1=RegOperand(RegisterKind.VGPR, index=2),
            src2=RegOperand(RegisterKind.VGPR, index=3)
        )
        with self.assertRaises(ValueError):
            instr.validate()

# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter
# -----------------------------------------------------------------------------

@dataclass
class LaneRegisterState:
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def read_operand(self, op: RegOperand) -> int:
        if op.kind == RegisterKind.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.kind == RegisterKind.SGPR:
            return self.sgprs[op.index] & MASK32
        elif op.kind == RegisterKind.LITERAL32:
            return op.index & MASK32
        raise NotImplementedError(f"Unsupported operand kind: {op.kind}")

    def write_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter executing v_min3_i32 for a single lane."""

    @staticmethod
    def step(instr: VMin3I32Instruction, state: LaneRegisterState) -> None:
        instr.validate()
        s0_val = state.read_operand(instr.src0)
        s1_val = state.read_operand(instr.src1)
        s2_val = state.read_operand(instr.src2)

        res = v_min3_i32_oracle(s0_val, s1_val, s2_val)
        state.write_vgpr(instr.dst.index, res)


class TestVMin3I32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter execution."""

    def setUp(self):
        self.state = LaneRegisterState()

    def test_interpreter_step_basic(self):
        self.state.vgprs[1] = to_u32(-500)
        self.state.vgprs[2] = to_u32(200)
        self.state.vgprs[3] = to_u32(-100)

        instr = VMin3I32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=4),
            src0=RegOperand(RegisterKind.VGPR, index=1),
            src1=RegOperand(RegisterKind.VGPR, index=2),
            src2=RegOperand(RegisterKind.VGPR, index=3)
        )
        ScalarInterpreter.step(instr, self.state)
        self.assertEqual(to_s32(self.state.vgprs[4]), -500)

    def test_interpreter_step_with_sgpr_and_literal(self):
        self.state.sgprs[10] = to_u32(-10)
        self.state.vgprs[2] = to_u32(0)

        instr = VMin3I32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=5),
            src0=RegOperand(RegisterKind.SGPR, index=10),
            src1=RegOperand(RegisterKind.VGPR, index=2),
            src2=RegOperand(RegisterKind.LITERAL32, index=to_u32(-100))
        )
        ScalarInterpreter.step(instr, self.state)
        self.assertEqual(to_s32(self.state.vgprs[5]), -100)

# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# -----------------------------------------------------------------------------

class SIMDMultiLaneModel:
    """Models SIMD multi-lane execution of v_min3_i32 under EXEC masking."""

    @staticmethod
    def execute_wave(
        instr: VMin3I32Instruction,
        exec_mask: int,
        lane_states: List[LaneRegisterState],
        wave_size: int = 64
    ) -> None:
        instr.validate()
        for lane in range(wave_size):
            if (exec_mask & (1 << lane)) != 0:
                s0 = lane_states[lane].read_operand(instr.src0)
                s1 = lane_states[lane].read_operand(instr.src1)
                s2 = lane_states[lane].read_operand(instr.src2)
                res = v_min3_i32_oracle(s0, s1, s2)
                lane_states[lane].write_vgpr(instr.dst.index, res)


class TestVMin3I32SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD divergence under Wave64 / Wave32 EXEC masking."""

    def test_divergent_lanes_preserve_inactive_state(self):
        # 64 lanes, only even lanes active
        exec_mask = 0x5555_5555_5555_5555
        lanes = [LaneRegisterState() for _ in range(64)]

        for i, lane in enumerate(lanes):
            lane.vgprs[1] = to_u32(-i)
            lane.vgprs[2] = to_u32(i)
            lane.vgprs[3] = to_u32(0)
            lane.vgprs[4] = 0xDEAD_BEEF  # sentinel

        instr = VMin3I32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=4),
            src0=RegOperand(RegisterKind.VGPR, index=1),
            src1=RegOperand(RegisterKind.VGPR, index=2),
            src2=RegOperand(RegisterKind.VGPR, index=3)
        )
        SIMDMultiLaneModel.execute_wave(instr, exec_mask, lanes)

        for i, lane in enumerate(lanes):
            if i % 2 == 0:
                # Active: min(-i, i, 0) = -i
                self.assertEqual(to_s32(lane.vgprs[4]), -i)
            else:
                # Inactive: untouched sentinel
                self.assertEqual(lane.vgprs[4], 0xDEAD_BEEF)

    def test_all_lanes_active(self):
        exec_mask = 0xFFFF_FFFF_FFFF_FFFF
        lanes = [LaneRegisterState() for _ in range(64)]
        for i, lane in enumerate(lanes):
            lane.vgprs[1] = to_u32(i * 10 - 300)
            lane.vgprs[2] = to_u32(0)
            lane.vgprs[3] = to_u32(100)

        instr = VMin3I32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=4),
            src0=RegOperand(RegisterKind.VGPR, index=1),
            src1=RegOperand(RegisterKind.VGPR, index=2),
            src2=RegOperand(RegisterKind.VGPR, index=3)
        )
        SIMDMultiLaneModel.execute_wave(instr, exec_mask, lanes)

        for i, lane in enumerate(lanes):
            expected = min(i * 10 - 300, 0, 100)
            self.assertEqual(to_s32(lane.vgprs[4]), expected)

if __name__ == "__main__":
    unittest.main()
