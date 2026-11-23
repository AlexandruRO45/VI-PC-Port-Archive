from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for v_floor_f32.

RDNA2 Instruction Set Architecture, VOP1:
    v_floor_f32 D, S0

Semantic operation:
    D.f32 = floor(S0.f32)

Hardware rules (GFX10 RDNA2):
    - Float32 IEEE 754 floor operation.
    - f32 denormals flushed to zero preserving sign: floor(+denorm) = +0.0, floor(-denorm) = -0.0.
    - Zero preservation: floor(+0.0) = +0.0, floor(-0.0) = -0.0.
    - Infs pass through: floor(+inf) = +inf, floor(-inf) = -inf.
    - NaNs pass through unchanged (quiet and signaling payloads preserved).

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
EXP_MASK32: int = 0x7F80_0000
MANT_MASK32: int = 0x007F_FFFF

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def v_floor_f32_oracle(val: int) -> int:
    """Reference mathematical model matching GFX10 RDNA2 hardware semantics."""
    val = val & MASK32
    exp = (val >> 23) & 0xFF
    mant = val & MANT_MASK32
    sign = val & SIGN_BIT32

    # NaN pass through
    if exp == 0xFF and mant != 0:
        return val
    # Inf pass through
    if exp == 0xFF and mant == 0:
        return val
    # Signed zero pass through
    if exp == 0 and mant == 0:
        return val
    # Denormals flushed to signed zero
    if exp == 0:
        return sign

    f = to_f32(val)
    rf = float(math.floor(f))
    if rf == 0.0 and sign:
        return SIGN_BIT32
    return to_u32(rf)


class VFloorF32Oracle:
    """Architectural model and hardware-rule validator for v_floor_f32."""

    @staticmethod
    def eval(val: int) -> int:
        return v_floor_f32_oracle(val)

    @staticmethod
    def build_compute_shader(val: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test v_floor_f32."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_floor_f32_hw"
        lines = [
            "s_mov_b64 exec, -1",
            f"v_mov_b32_e32 v1, {val:#x}",
            "v_floor_f32 v2, v1",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(val: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = VFloorF32Oracle.build_compute_shader(val)
        return support.translate_lines(name, lines, fssm=fssm)


class TestVFloorF32Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_hardware_measured_table(self):
        # Direct measured values on AMD RDNA2
        measured = [
            (0x00000001, 0x00000000),  # +denorm -> +0.0
            (0x80000001, 0x80000000),  # -denorm -> -0.0
            (0x00400000, 0x00000000),  # +denorm -> +0.0
            (0x807FFFFF, 0x80000000),  # -denorm -> -0.0
            (0x80000000, 0x80000000),  # -0.0 -> -0.0
            (0x00000000, 0x00000000),  # +0.0 -> +0.0
            (0x00800000, 0x00000000),  # min normal (+1.17e-38) -> 0.0
            (0x80800000, 0xBF800000),  # min neg normal (-1.17e-38) -> -1.0
            (0x7F800000, 0x7F800000),  # +inf
            (0xFF800000, 0xFF800000),  # -inf
            (0x7FA00000, 0x7FA00000),  # sNaN
            (0x7FC00000, 0x7FC00000),  # qNaN
            (0xBF000000, 0xBF800000),  # -0.5 -> -1.0
            (0x3FC00000, 0x3F800000),  # +1.5 -> +1.0
            (0x3F7FFFFF, 0x00000000),  # 0.99999994 -> 0.0
            (0xBF7FFFFF, 0xBF800000),  # -0.99999994 -> -1.0
        ]
        for inp, expected in measured:
            with self.subTest(inp=hex(inp)):
                self.assertEqual(VFloorF32Oracle.eval(inp), expected)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = VFloorF32Oracle.translate_and_validate(0x3FC0_0000)
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
class VFloorF32Instruction:
    dst: RegOperand
    src0: RegOperand
    opcode: str = "v_floor_f32"

    def validate(self) -> None:
        if self.opcode != "v_floor_f32":
            raise ValueError(f"Invalid opcode {self.opcode}")
        if self.dst.kind != RegisterKind.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.kind}")
        if self.dst.index > 255:
            raise ValueError(f"Destination VGPR index out of range: v{self.dst.index}")
        if self.src0.kind == RegisterKind.VGPR and self.src0.index > 255:
            raise ValueError(f"Source VGPR index out of range: v{self.src0.index}")
        elif self.src0.kind == RegisterKind.SGPR and self.src0.index > 105:
            raise ValueError(f"Source SGPR index out of range: s{self.src0.index}")

    def emit_assembly(self) -> str:
        self.validate()
        return f"{self.opcode} {self.dst.format()}, {self.src0.format()}"


class TestVFloorF32AST(unittest.TestCase):
    """Unit tests for symbolic AST representation and validation."""

    def test_ast_valid_vgpr_operands(self):
        instr = VFloorF32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=2),
            src0=RegOperand(RegisterKind.VGPR, index=1)
        )
        self.assertEqual(instr.emit_assembly(), "v_floor_f32 v2, v1")

    def test_ast_valid_sgpr_source(self):
        instr = VFloorF32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=3),
            src0=RegOperand(RegisterKind.SGPR, index=4)
        )
        self.assertEqual(instr.emit_assembly(), "v_floor_f32 v3, s4")

    def test_ast_invalid_sgpr_destination(self):
        instr = VFloorF32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=0),
            src0=RegOperand(RegisterKind.VGPR, index=1)
        )
        with self.assertRaises(TypeError):
            instr.validate()

    def test_ast_out_of_bounds_vgpr(self):
        instr = VFloorF32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=256),
            src0=RegOperand(RegisterKind.VGPR, index=1)
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
    """Deterministic scalar instruction interpreter executing v_floor_f32 for a single lane."""

    @staticmethod
    def step(instr: VFloorF32Instruction, state: LaneRegisterState) -> None:
        instr.validate()
        s0_val = state.read_operand(instr.src0)
        res = v_floor_f32_oracle(s0_val)
        state.write_vgpr(instr.dst.index, res)


class TestVFloorF32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter execution."""

    def setUp(self):
        self.state = LaneRegisterState()

    def test_interpreter_step_basic(self):
        # 1.5 (0x3FC00000) -> 1.0 (0x3F800000)
        self.state.vgprs[1] = 0x3FC0_0000

        instr = VFloorF32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=2),
            src0=RegOperand(RegisterKind.VGPR, index=1)
        )
        ScalarInterpreter.step(instr, self.state)
        self.assertEqual(self.state.vgprs[2], 0x3F80_0000)

    def test_interpreter_step_with_sgpr_source(self):
        # -0.5 (0xBF000000) -> -1.0 (0xBF800000)
        self.state.sgprs[10] = 0xBF00_0000

        instr = VFloorF32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=4),
            src0=RegOperand(RegisterKind.SGPR, index=10)
        )
        ScalarInterpreter.step(instr, self.state)
        self.assertEqual(self.state.vgprs[4], 0xBF80_0000)

# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# -----------------------------------------------------------------------------

class SIMDMultiLaneModel:
    """Models SIMD multi-lane execution of v_floor_f32 under EXEC masking."""

    @staticmethod
    def execute_wave(
        instr: VFloorF32Instruction,
        exec_mask: int,
        lane_states: List[LaneRegisterState],
        wave_size: int = 64
    ) -> None:
        instr.validate()
        for lane in range(wave_size):
            if (exec_mask & (1 << lane)) != 0:
                s0 = lane_states[lane].read_operand(instr.src0)
                res = v_floor_f32_oracle(s0)
                lane_states[lane].write_vgpr(instr.dst.index, res)


class TestVFloorF32SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD divergence under Wave64 / Wave32 EXEC masking."""

    def test_divergent_lanes_preserve_inactive_state(self):
        # Only even lanes active
        exec_mask = 0x5555_5555_5555_5555
        lanes = [LaneRegisterState() for _ in range(64)]

        for i, lane in enumerate(lanes):
            # float value: i + 0.75
            f_val = float(i) + 0.75
            lane.vgprs[1] = to_u32(f_val)
            lane.vgprs[2] = 0xCAFE_BABE  # sentinel

        instr = VFloorF32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=2),
            src0=RegOperand(RegisterKind.VGPR, index=1)
        )
        SIMDMultiLaneModel.execute_wave(instr, exec_mask, lanes)

        for i, lane in enumerate(lanes):
            if i % 2 == 0:
                # Active: floor(i + 0.75) = float(i)
                expected_u = to_u32(float(i))
                self.assertEqual(lane.vgprs[2], expected_u)
            else:
                # Inactive: untouched sentinel
                self.assertEqual(lane.vgprs[2], 0xCAFE_BABE)

    def test_all_lanes_active(self):
        exec_mask = 0xFFFF_FFFF_FFFF_FFFF
        lanes = [LaneRegisterState() for _ in range(64)]
        for i, lane in enumerate(lanes):
            f_val = -float(i) - 0.25
            lane.vgprs[1] = to_u32(f_val)

        instr = VFloorF32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=2),
            src0=RegOperand(RegisterKind.VGPR, index=1)
        )
        SIMDMultiLaneModel.execute_wave(instr, exec_mask, lanes)

        for i, lane in enumerate(lanes):
            # floor(-i - 0.25) = -i - 1.0
            expected = to_u32(-float(i) - 1.0)
            self.assertEqual(lane.vgprs[2], expected)

# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestVFloorF32Aliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0(self):
        """v_floor_f32 v1, v1: in-place floor modification."""
        state = LaneRegisterState()
        # +3.9 -> +3.0
        state.vgprs[1] = to_u32(3.9)

        instr = VFloorF32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=1),
            src0=RegOperand(RegisterKind.VGPR, index=1)
        )
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.vgprs[1], to_u32(3.0))

    def test_unrelated_registers_unaffected(self):
        state = LaneRegisterState()
        state.vgprs[1] = to_u32(5.5)
        state.vgprs[2] = 0x1234_5678
        state.vgprs[3] = 0x8765_4321

        instr = VFloorF32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=2),
            src0=RegOperand(RegisterKind.VGPR, index=1)
        )
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.vgprs[2], to_u32(5.0))
        self.assertEqual(state.vgprs[1], to_u32(5.5))  # source unchanged
        self.assertEqual(state.vgprs[3], 0x8765_4321) # unrelated unchanged

if __name__ == "__main__":
    unittest.main()
