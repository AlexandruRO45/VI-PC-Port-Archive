from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_andn2_b32.

RDNA2 Instruction Set Architecture, SOP2:
    s_andn2_b32 D, S0, S1

Semantic operation:
    D.u32 = S0.u32 & (~S1.u32 & 0xFFFFFFFF)
    SCC = (D != 0)

Outputs:
    D: 32-bit destination SGPR (or EXEC_LO, VCC_LO, NULL) storing the bitwise AND of S0 and bitwise NOT of S1.
    SCC: 1 if destination is non-zero, 0 if all zeros.

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
ALL32: int = 0xFFFF_FFFF

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def s_andn2_b32_oracle(s0: int, s1: int) -> Tuple[int, int]:
    """Reference mathematical model for s_andn2_b32.

    Args:
        s0: 32-bit unsigned integer source 0.
        s1: 32-bit unsigned integer source 1.

    Returns:
        (d, scc) tuple where:
            d: (s0 & (~s1 & MASK32)) & MASK32
            scc: 1 if d != 0 else 0
    """
    s0_u32 = s0 & MASK32
    s1_u32 = s1 & MASK32
    d = (s0_u32 & (~s1_u32 & MASK32)) & MASK32
    scc = 1 if d != 0 else 0
    return d, scc


class SAndn2B32Oracle:
    """Architectural model and hardware-rule validator for s_andn2_b32."""

    @staticmethod
    def eval(s0: int, s1: int) -> Tuple[int, int]:
        return s_andn2_b32_oracle(s0, s1)

    @staticmethod
    def build_pixel_shader(s0: int, s1: int) -> Tuple[str, List[str]]:
        """Generate synthetic pixel shader lines to test s_andn2_b32."""
        name = "ps_andn2_b32_test"
        lines = [
            f"s_mov_b32 s0, {s0 & MASK32:#x}",
            f"s_mov_b32 s1, {s1 & MASK32:#x}",
            "s_andn2_b32 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        return name, lines

    @staticmethod
    def translate_and_validate(s0: int, s1: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines = SAndn2B32Oracle.build_pixel_shader(s0, s1)
        return support.translate_lines(name, lines)


class TestSAndn2B32Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_truth_table_bit_combinations(self):
        cases = [
            (0, 0, 0),
            (0, 1, 0),
            (1, 0, 1),
            (1, 1, 0),
        ]
        for b0, b1, expected in cases:
            with self.subTest(b0=b0, b1=b1):
                d, scc = SAndn2B32Oracle.eval(b0, b1)
                self.assertEqual(d & 1, expected)
                self.assertEqual(scc, expected)

    def test_oracle_identical_operands_produce_zero(self):
        for val in (0, ALL32, 0x1234_5678, 0x5555_5555, 0xAAAA_AAAA, 0x8000_0000):
            with self.subTest(val=hex(val)):
                d, scc = SAndn2B32Oracle.eval(val, val)
                self.assertEqual(d, 0)
                self.assertEqual(scc, 0)

    def test_oracle_s1_zero_produces_s0(self):
        for s0 in (0, ALL32, 0xCAFE_BABE, 0x0000_0001, 0x8000_0000):
            with self.subTest(s0=hex(s0)):
                d, scc = SAndn2B32Oracle.eval(s0, 0)
                self.assertEqual(d, s0)
                self.assertEqual(scc, 1 if s0 != 0 else 0)

    def test_oracle_s1_all_ones_produces_zero(self):
        for s0 in (0, ALL32, 0x1234_5678, 0xDEAD_BEEF):
            with self.subTest(s0=hex(s0)):
                d, scc = SAndn2B32Oracle.eval(s0, ALL32)
                self.assertEqual(d, 0)
                self.assertEqual(scc, 0)

    def test_oracle_s0_zero_produces_zero(self):
        for s1 in (0, ALL32, 0x5555_5555, 0x1234_5678):
            with self.subTest(s1=hex(s1)):
                d, scc = SAndn2B32Oracle.eval(0, s1)
                self.assertEqual(d, 0)
                self.assertEqual(scc, 0)

    def test_oracle_non_commutative_asymmetry(self):
        a = 0xF0F0_0F0F
        b = 0x3333_CCCC
        d_ab, _ = SAndn2B32Oracle.eval(a, b)
        d_ba, _ = SAndn2B32Oracle.eval(b, a)
        self.assertNotEqual(d_ab, d_ba)
        self.assertEqual(d_ab, a & (~b & MASK32))
        self.assertEqual(d_ba, b & (~a & MASK32))

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SAndn2B32Oracle.translate_and_validate(0x1234_5678, 0x00FF_00FF)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)

# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------

class RegisterKind(Enum):
    SGPR = auto()
    VGPR = auto()
    SPECIAL = auto()
    IMM = auto()


@dataclass(frozen=True)
class RegOperand:
    kind: RegisterKind
    index: int = 0
    name: str = ""
    size_bits: int = 32

    def __post_init__(self):
        if self.kind == RegisterKind.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of bounds [0, 105]: {self.index}")
        if self.kind == RegisterKind.VGPR:
            if not (0 <= self.index <= 255):
                raise ValueError(f"VGPR index out of bounds [0, 255]: {self.index}")


@dataclass(frozen=True)
class ImmOperand:
    value: int
    size_bits: int = 32

    def __post_init__(self):
        if not (-0x8000_0000 <= self.value <= 0xFFFF_FFFF):
            raise ValueError(f"Immediate value out of 32-bit range: {self.value}")


Operand = Union[RegOperand, ImmOperand]


@dataclass
class SAndn2B32Instruction:
    dst: RegOperand
    src0: Operand
    src1: Operand

    def validate(self) -> None:
        if self.dst.kind == RegisterKind.VGPR:
            raise TypeError("SALU SOP2 instruction destination cannot be VGPR")
        if self.dst.kind not in (RegisterKind.SGPR, RegisterKind.SPECIAL):
            raise TypeError(f"Invalid destination kind: {self.dst.kind}")
        if self.dst.kind == RegisterKind.SGPR and not (0 <= self.dst.index <= 105):
            raise ValueError(f"Destination SGPR index out of range: {self.dst.index}")


class TestSAndn2B32AST(unittest.TestCase):
    """Unit tests for symbolic AST representation and validation."""

    def test_ast_valid_sgpr_operands(self):
        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=2),
            src0=RegOperand(RegisterKind.SGPR, index=0),
            src1=RegOperand(RegisterKind.SGPR, index=1),
        )
        instr.validate()
        self.assertEqual(instr.dst.index, 2)
        self.assertEqual(instr.src0.index, 0)
        self.assertEqual(instr.src1.index, 1)

    def test_ast_valid_immediate_operands(self):
        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=4),
            src0=RegOperand(RegisterKind.SGPR, index=0),
            src1=ImmOperand(value=0x00FF_00FF),
        )
        instr.validate()
        self.assertIsInstance(instr.src1, ImmOperand)
        self.assertEqual(instr.src1.value, 0x00FF_00FF)

    def test_ast_valid_special_registers(self):
        for name in ("exec_lo", "vcc_lo", "null"):
            with self.subTest(special_name=name):
                instr = SAndn2B32Instruction(
                    dst=RegOperand(RegisterKind.SPECIAL, name=name),
                    src0=RegOperand(RegisterKind.SPECIAL, name=name),
                    src1=ImmOperand(value=0),
                )
                instr.validate()
                self.assertEqual(instr.dst.name, name)

    def test_ast_reject_invalid_sgpr_index(self):
        for invalid_idx in (106, 128, 256):
            with self.subTest(invalid_idx=invalid_idx):
                with self.assertRaises(ValueError):
                    RegOperand(RegisterKind.SGPR, index=invalid_idx)

    def test_ast_reject_vgpr_destination(self):
        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=0),
            src0=RegOperand(RegisterKind.SGPR, index=1),
            src1=RegOperand(RegisterKind.SGPR, index=2),
        )
        with self.assertRaises(TypeError):
            instr.validate()

# -----------------------------------------------------------------------------
# Commit 3: Deterministic Single-Lane Scalar Interpreter
# -----------------------------------------------------------------------------

class ScalarState:
    """Models 32-bit RDNA2 scalar execution state."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs = [0] * num_sgprs
        self.exec_lo: int = ALL32
        self.vcc_lo: int = 0
        self.scc: int = 0

    def read_operand(self, op: Operand) -> int:
        if isinstance(op, ImmOperand):
            return op.value & MASK32
        if isinstance(op, RegOperand):
            if op.kind == RegisterKind.SGPR:
                return self.sgprs[op.index] & MASK32
            if op.kind == RegisterKind.SPECIAL:
                if op.name == "exec_lo":
                    return self.exec_lo & MASK32
                if op.name == "vcc_lo":
                    return self.vcc_lo & MASK32
                if op.name == "null":
                    return 0
                raise ValueError(f"Unknown special register: {op.name}")
        raise TypeError(f"Unsupported operand type: {type(op)}")

    def write_operand(self, op: RegOperand, val: int) -> None:
        val = val & MASK32
        if op.kind == RegisterKind.SGPR:
            self.sgprs[op.index] = val
        elif op.kind == RegisterKind.SPECIAL:
            if op.name == "exec_lo":
                self.exec_lo = val
            elif op.name == "vcc_lo":
                self.vcc_lo = val
            elif op.name == "null":
                pass
            else:
                raise ValueError(f"Unknown special register: {op.name}")
        else:
            raise TypeError(f"Cannot write to operand kind: {op.kind}")


class ScalarInterpreter:
    """Executes s_andn2_b32 instructions and manages architectural register state."""

    @staticmethod
    def step(instr: SAndn2B32Instruction, state: ScalarState) -> int:
        instr.validate()
        s0 = state.read_operand(instr.src0)
        s1 = state.read_operand(instr.src1)
        d, scc = s_andn2_b32_oracle(s0, s1)
        state.write_operand(instr.dst, d)
        state.scc = scc
        return d


class TestSAndn2B32ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter execution."""

    def test_interpreter_sgpr_to_sgpr(self):
        state = ScalarState()
        state.sgprs[0] = 0x1234_5678
        state.sgprs[1] = 0x00FF_00FF
        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=2),
            src0=RegOperand(RegisterKind.SGPR, index=0),
            src1=RegOperand(RegisterKind.SGPR, index=1),
        )
        res = ScalarInterpreter.step(instr, state)
        expected = (0x1234_5678 & ~0x00FF_00FF) & MASK32
        self.assertEqual(res, expected)
        self.assertEqual(state.sgprs[2], expected)
        self.assertEqual(state.scc, 1)

    def test_interpreter_immediate_source(self):
        state = ScalarState()
        state.sgprs[0] = 0xFFFF_0000
        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=1),
            src0=RegOperand(RegisterKind.SGPR, index=0),
            src1=ImmOperand(value=0xFF00_0000),
        )
        res = ScalarInterpreter.step(instr, state)
        self.assertEqual(res, 0x00FF_0000)
        self.assertEqual(state.sgprs[1], 0x00FF_0000)
        self.assertEqual(state.scc, 1)

    def test_interpreter_special_destinations(self):
        state = ScalarState()
        state.sgprs[0] = 0xAAAA_AAAA
        state.sgprs[1] = 0x5555_5555
        instr_null = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SPECIAL, name="null"),
            src0=RegOperand(RegisterKind.SGPR, index=0),
            src1=RegOperand(RegisterKind.SGPR, index=1),
        )
        res = ScalarInterpreter.step(instr_null, state)
        self.assertEqual(res, 0xAAAA_AAAA)
        self.assertEqual(state.scc, 1)

    def test_interpreter_scc_tracking(self):
        state = ScalarState()
        state.sgprs[0] = 0x1111_1111
        instr_zero = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=1),
            src0=RegOperand(RegisterKind.SGPR, index=0),
            src1=RegOperand(RegisterKind.SGPR, index=0),
        )
        res = ScalarInterpreter.step(instr_zero, state)
        self.assertEqual(res, 0)
        self.assertEqual(state.scc, 0)

# -----------------------------------------------------------------------------
# Commit 4: SIMD Divergence and Wave32 Execution Masking
# -----------------------------------------------------------------------------

class TestSAndn2B32Divergence(unittest.TestCase):
    """Unit tests for SIMD divergence, branch management, and Wave32 EXEC masks."""

    def test_wave32_branch_divergence_exec_pruning(self):
        state = ScalarState()
        state.exec_lo = ALL32
        state.vcc_lo = 0x5555_5555

        then_mask = state.vcc_lo & state.exec_lo
        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
            src0=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
            src1=RegOperand(RegisterKind.SPECIAL, name="vcc_lo"),
        )
        ScalarInterpreter.step(instr, state)
        else_mask = state.exec_lo

        self.assertEqual(then_mask & else_mask, 0)
        self.assertEqual(then_mask | else_mask, ALL32)
        self.assertEqual(else_mask, 0xAAAA_AAAA)
        self.assertEqual(state.scc, 1)

    def test_wave32_all_lanes_disabled(self):
        state = ScalarState()
        state.exec_lo = ALL32
        state.vcc_lo = ALL32

        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
            src0=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
            src1=RegOperand(RegisterKind.SPECIAL, name="vcc_lo"),
        )
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.exec_lo, 0)
        self.assertEqual(state.scc, 0)

    def test_wave32_single_lane_toggles(self):
        for lane in range(32):
            with self.subTest(lane=lane):
                state = ScalarState()
                state.exec_lo = ALL32
                instr = SAndn2B32Instruction(
                    dst=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
                    src0=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
                    src1=ImmOperand(value=1 << lane),
                )
                ScalarInterpreter.step(instr, state)
                expected = ALL32 & ~(1 << lane)
                self.assertEqual(state.exec_lo, expected)
                self.assertEqual(state.scc, 1)

    def test_wave32_inactive_lane_preservation(self):
        state = ScalarState()
        state.exec_lo = 0x0000_FFFF
        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
            src0=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
            src1=ImmOperand(value=0x0000_00FF),
        )
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.exec_lo, 0x0000_FF00)
        self.assertEqual(state.scc, 1)

# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestSAndn2B32RegisterAliasing(unittest.TestCase):
    """Unit tests verifying in-place operand aliasing and register preservation."""

    def test_aliasing_dst_equals_src0(self):
        state = ScalarState()
        s0_val = 0x1234_5678
        s1_val = 0x00FF_00FF
        state.sgprs[0] = s0_val
        state.sgprs[1] = s1_val

        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=0),
            src0=RegOperand(RegisterKind.SGPR, index=0),
            src1=RegOperand(RegisterKind.SGPR, index=1),
        )
        res = ScalarInterpreter.step(instr, state)
        expected = s0_val & (~s1_val & MASK32)
        self.assertEqual(res, expected)
        self.assertEqual(state.sgprs[0], expected)
        self.assertEqual(state.sgprs[1], s1_val)

    def test_aliasing_dst_equals_src1(self):
        state = ScalarState()
        s0_val = 0x1234_5678
        s1_val = 0x00FF_00FF
        state.sgprs[0] = s0_val
        state.sgprs[1] = s1_val

        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=1),
            src0=RegOperand(RegisterKind.SGPR, index=0),
            src1=RegOperand(RegisterKind.SGPR, index=1),
        )
        res = ScalarInterpreter.step(instr, state)
        expected = s0_val & (~s1_val & MASK32)
        self.assertEqual(res, expected)
        self.assertEqual(state.sgprs[0], s0_val)
        self.assertEqual(state.sgprs[1], expected)

    def test_aliasing_dst_equals_src0_and_src1(self):
        state = ScalarState()
        state.sgprs[5] = 0xDEAD_BEEF
        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=5),
            src0=RegOperand(RegisterKind.SGPR, index=5),
            src1=RegOperand(RegisterKind.SGPR, index=5),
        )
        res = ScalarInterpreter.step(instr, state)
        self.assertEqual(res, 0)
        self.assertEqual(state.sgprs[5], 0)
        self.assertEqual(state.scc, 0)

    def test_aliasing_special_exec_lo_self(self):
        state = ScalarState()
        state.exec_lo = 0x5555_5555
        instr = SAndn2B32Instruction(
            dst=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
            src0=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
            src1=RegOperand(RegisterKind.SPECIAL, name="exec_lo"),
        )
        res = ScalarInterpreter.step(instr, state)
        self.assertEqual(res, 0)
        self.assertEqual(state.exec_lo, 0)
        self.assertEqual(state.scc, 0)

if __name__ == "__main__":
    unittest.main()
