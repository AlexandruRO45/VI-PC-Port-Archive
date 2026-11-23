from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for v_ldexp_f32.

RDNA2 Instruction Set Architecture, VOP2/VOP3:
    v_ldexp_f32 D, S0, S1

Semantic operation:
    D.f32 = S0.f32 * 2^(S1.i32)

Bit-exact binary32 semantics:
    - Normal inputs (biased exponent 1..254): exponent field is incremented by clamped S1 in [-512, 512].
    - Zero and denormals (biased exponent 0): flushed to signed zero (+0.0 / -0.0).
    - Infinities and NaNs (biased exponent 255): preserved bit-exact without modification.
    - Exponent overflow (> 254): saturates to +/-infinity with original sign.
    - Exponent underflow (<= 0): saturates to +/-zero with original sign.

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
SIGN_MASK: int = 0x8000_0000
EXP_MASK: int = 0x7F80_0000
MANT_MASK: int = 0x007F_FFFF
POS_INF: int = 0x7F80_0000
NEG_INF: int = 0xFF80_0000

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def ref_ldexp_f32(s0_u32: int, s1_i32: int) -> int:
    """Reference mathematical model for v_ldexp_f32 matching RDNA2 hardware semantics."""
    x = s0_u32 & MASK32
    sgn = x & SIGN_MASK
    e = (x >> 23) & 0xFF

    # Zero and denormals flush to signed zero (+-0.0)
    if e == 0:
        return sgn
    # Infinities and NaNs pass through unchanged
    if e == 255:
        return x

    # Normal input: clamp s1 to [-512, 512]
    if s1_i32 >= 0x8000_0000:
        s1_val = s1_i32 - 0x1_0000_0000
    else:
        s1_val = s1_i32

    clamped_n = max(-512, min(512, s1_val))
    new_e = e + clamped_n

    if 1 <= new_e <= 254:
        return sgn | (new_e << 23) | (x & MANT_MASK)
    elif new_e > 254:
        return sgn | POS_INF
    else:
        return sgn


class VLdexpF32Oracle:
    """Architectural model and hardware-rule validator for v_ldexp_f32."""

    @staticmethod
    def eval(s0_u32: int, s1_i32: int) -> int:
        return ref_ldexp_f32(s0_u32, s1_i32)

    @staticmethod
    def build_shader(s0_u32: int, s1_i32: int) -> Tuple[str, List[str]]:
        name = "ps_ldexp_f32_hw"
        lines = [
            f"v_mov_b32_e32 v0, {s0_u32:#x}",
            f"s_mov_b32 s0, {s1_i32:#x}",
            "v_ldexp_f32 v2, v0, s0",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0 v2, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        return name, lines

    @staticmethod
    def translate_and_validate(s0_u32: int, s1_i32: int) -> str:
        name, lines = VLdexpF32Oracle.build_shader(s0_u32, s1_i32)
        return support.translate_lines(name, lines)

    @staticmethod
    def execute_and_verify(s0_u32: int, s1_i32: int) -> int:
        spv = VLdexpF32Oracle.translate_and_validate(s0_u32, s1_i32)
        out = support.mrt0_words(spv)
        return out[0]


class TestVLdexpF32Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_identity_scale_zero(self):
        # 1.0f * 2^0 = 1.0f (0x3F80_0000)
        one = 0x3F80_0000
        res = VLdexpF32Oracle.eval(one, 0)
        self.assertEqual(res, one)

    def test_oracle_positive_scale(self):
        # 1.0f * 2^2 = 4.0f (0x4080_0000)
        one = 0x3F80_0000
        res = VLdexpF32Oracle.eval(one, 2)
        self.assertEqual(res, 0x4080_0000)

    def test_oracle_negative_scale(self):
        # 1.0f * 2^(-2) = 0.25f (0x3E80_0000)
        one = 0x3F80_0000
        res = VLdexpF32Oracle.eval(one, -2)
        self.assertEqual(res, 0x3E80_0000)

    def test_oracle_overflow_to_infinity(self):
        one = 0x3F80_0000
        res = VLdexpF32Oracle.eval(one, 150)
        self.assertEqual(res, POS_INF)

    def test_oracle_underflow_to_zero(self):
        one = 0x3F80_0000
        res = VLdexpF32Oracle.eval(one, -150)
        self.assertEqual(res, 0)

    def test_translator_and_execution_pipeline(self):
        """Verify translation and execution of 1.0f * 2^2 = 4.0f through rdna2spirv and spirv_mini_interp."""
        one = 0x3F80_0000
        got = VLdexpF32Oracle.execute_and_verify(one, 2)
        self.assertEqual(got, 0x4080_0000)

# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------

class RegKind(Enum):
    VGPR = auto()
    SGPR = auto()
    IMM = auto()


@dataclass(frozen=True)
class Operand:
    kind: RegKind
    index: int = 0
    imm_val: int = 0

    def format(self) -> str:
        if self.kind == RegKind.VGPR:
            return f"v{self.index}"
        elif self.kind == RegKind.SGPR:
            return f"s{self.index}"
        elif self.kind == RegKind.IMM:
            return f"{self.imm_val:#x}"
        return f"reg_{self.index}"


@dataclass
class VLdexpF32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    opcode: str = "v_ldexp_f32"

    def validate(self) -> None:
        if self.opcode != "v_ldexp_f32":
            raise ValueError(f"Invalid opcode: {self.opcode}")
        if self.dst.kind != RegKind.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.kind}")
        if not (0 <= self.dst.index <= 255):
            raise ValueError(f"Destination VGPR out of bounds: v{self.dst.index}")
        for src, name in ((self.src0, "src0"), (self.src1, "src1")):
            if src.kind == RegKind.VGPR and not (0 <= src.index <= 255):
                raise ValueError(f"{name} VGPR out of bounds: v{src.index}")
            elif src.kind == RegKind.SGPR and not (0 <= src.index <= 105):
                raise ValueError(f"{name} SGPR out of bounds: s{src.index}")

    def emit_assembly(self) -> str:
        self.validate()
        return f"{self.opcode} {self.dst.format()}, {self.src0.format()}, {self.src1.format()}"


class TestVLdexpF32AST(unittest.TestCase):
    """Unit tests for symbolic AST representation and validation."""

    def test_ast_valid_vgpr_operands(self):
        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.VGPR, index=1),
            src1=Operand(RegKind.VGPR, index=2),
        )
        self.assertEqual(instr.emit_assembly(), "v_ldexp_f32 v0, v1, v2")

    def test_ast_valid_sgpr_scale(self):
        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.VGPR, index=1),
            src1=Operand(RegKind.SGPR, index=4),
        )
        self.assertEqual(instr.emit_assembly(), "v_ldexp_f32 v0, v1, s4")

    def test_ast_reject_sgpr_destination(self):
        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.SGPR, index=0),
            src0=Operand(RegKind.VGPR, index=1),
            src1=Operand(RegKind.VGPR, index=2),
        )
        with self.assertRaises(TypeError):
            instr.validate()

    def test_ast_reject_out_of_bounds_vgpr(self):
        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.VGPR, index=256),
            src0=Operand(RegKind.VGPR, index=1),
            src1=Operand(RegKind.VGPR, index=2),
        )
        with self.assertRaises(ValueError):
            instr.validate()

# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter
# -----------------------------------------------------------------------------

@dataclass
class ScalarVALUState:
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    exec_mask: int = MASK32

    def read_operand(self, op: Operand) -> int:
        if op.kind == RegKind.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.kind == RegKind.SGPR:
            return self.sgprs[op.index] & MASK32
        elif op.kind == RegKind.IMM:
            return op.imm_val & MASK32
        raise NotImplementedError(f"Unsupported operand kind: {op.kind}")

    def write_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32


class ScalarVALUInterpreter:
    """Deterministic single-lane VALU interpreter for v_ldexp_f32."""

    @staticmethod
    def step(instr: VLdexpF32Instruction, state: ScalarVALUState) -> None:
        instr.validate()
        s0_val = state.read_operand(instr.src0)
        s1_val = state.read_operand(instr.src1)
        res = ref_ldexp_f32(s0_val, s1_val)
        state.write_vgpr(instr.dst.index, res)


class TestVLdexpF32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter execution."""

    def setUp(self):
        self.state = ScalarVALUState()

    def test_interpreter_step_basic(self):
        # v1 = 1.0f (0x3F80_0000), s2 = 3 -> v0 = 8.0f (0x4100_0000)
        self.state.vgprs[1] = 0x3F80_0000
        self.state.sgprs[2] = 3

        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.VGPR, index=1),
            src1=Operand(RegKind.SGPR, index=2),
        )
        ScalarVALUInterpreter.step(instr, self.state)
        self.assertEqual(self.state.vgprs[0], 0x4100_0000)

    def test_interpreter_step_negative_scale(self):
        # v1 = 8.0f (0x4100_0000), v2 = -3 (0xFFFF_FFFD) -> v0 = 1.0f (0x3F80_0000)
        self.state.vgprs[1] = 0x4100_0000
        self.state.vgprs[2] = (-3) & MASK32

        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.VGPR, index=1),
            src1=Operand(RegKind.VGPR, index=2),
        )
        ScalarVALUInterpreter.step(instr, self.state)
        self.assertEqual(self.state.vgprs[0], 0x3F80_0000)

# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# -----------------------------------------------------------------------------

class SIMDWavefrontState:
    """Models SIMD multi-lane register states across Wave32 and Wave64."""

    def __init__(self, lanes: int = 64):
        self.lanes = lanes
        self.vgprs: List[List[int]] = [[0] * lanes for _ in range(256)]
        self.exec_mask: int = (1 << lanes) - 1

    def step(self, instr: VLdexpF32Instruction, s1_scale: int) -> None:
        instr.validate()
        d_idx = instr.dst.index
        s0_idx = instr.src0.index
        for lane in range(self.lanes):
            if (self.exec_mask & (1 << lane)) != 0:
                s0_val = self.vgprs[s0_idx][lane]
                self.vgprs[d_idx][lane] = ref_ldexp_f32(s0_val, s1_scale)


class TestVLdexpF32SIMDDivergence(unittest.TestCase):
    """Unit tests for multi-lane divergence and active/inactive lane masking."""

    def test_wave32_even_odd_mask_divergence(self):
        simd = SIMDWavefrontState(lanes=32)
        # Initialize v1 with 1.0f in all lanes, canary in v0
        one = 0x3F80_0000
        canary = 0xDEAD_BEEF
        for lane in range(32):
            simd.vgprs[1][lane] = one
            simd.vgprs[0][lane] = canary

        # Active mask: even lanes only (0x5555_5555)
        simd.exec_mask = 0x5555_5555
        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.VGPR, index=1),
            src1=Operand(RegKind.SGPR, index=0),
        )
        simd.step(instr, s1_scale=1)  # 1.0f * 2^1 = 2.0f (0x4000_0000)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(simd.vgprs[0][lane], 0x4000_0000)
            else:
                self.assertEqual(simd.vgprs[0][lane], canary)

    def test_wave64_divergence(self):
        simd = SIMDWavefrontState(lanes=64)
        one = 0x3F80_0000
        canary = 0xCAFE_BABE
        for lane in range(64):
            simd.vgprs[1][lane] = one
            simd.vgprs[0][lane] = canary

        # Active mask: lower 32 lanes only
        simd.exec_mask = 0x0000_0000_FFFF_FFFF
        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.VGPR, index=1),
            src1=Operand(RegKind.SGPR, index=0),
        )
        simd.step(instr, s1_scale=2)  # 1.0f * 2^2 = 4.0f (0x4080_0000)

        for lane in range(64):
            if lane < 32:
                self.assertEqual(simd.vgprs[0][lane], 0x4080_0000)
            else:
                self.assertEqual(simd.vgprs[0][lane], canary)

# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestVLdexpF32Aliasing(unittest.TestCase):
    """Unit tests for in-place register aliasing."""

    def test_alias_dst_equals_src0(self):
        """v0 = v_ldexp_f32 v0, s0: in-place scale."""
        state = ScalarVALUState()
        state.vgprs[0] = 0x3F80_0000  # 1.0f
        state.sgprs[0] = 3

        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.VGPR, index=0),
            src1=Operand(RegKind.SGPR, index=0),
        )
        ScalarVALUInterpreter.step(instr, state)
        self.assertEqual(state.vgprs[0], 0x4100_0000)  # 8.0f

    def test_alias_dst_equals_src1(self):
        """v1 = v_ldexp_f32 v0, v1: scale register overwritten."""
        state = ScalarVALUState()
        state.vgprs[0] = 0x3F80_0000  # 1.0f
        state.vgprs[1] = 2

        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.VGPR, index=1),
            src0=Operand(RegKind.VGPR, index=0),
            src1=Operand(RegKind.VGPR, index=1),
        )
        ScalarVALUInterpreter.step(instr, state)
        self.assertEqual(state.vgprs[1], 0x4080_0000)  # 4.0f
        self.assertEqual(state.vgprs[0], 0x3F80_0000)  # src0 untouched

    def test_all_operands_aliased(self):
        """v0 = v_ldexp_f32 v0, v0."""
        state = ScalarVALUState()
        # 2.0f is 0x4000_0000 (interpreted as int: 1073741824) -> overflows to +inf
        state.vgprs[0] = 0x4000_0000

        instr = VLdexpF32Instruction(
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.VGPR, index=0),
            src1=Operand(RegKind.VGPR, index=0),
        )
        ScalarVALUInterpreter.step(instr, state)
        self.assertEqual(state.vgprs[0], POS_INF)

if __name__ == "__main__":
    unittest.main()
