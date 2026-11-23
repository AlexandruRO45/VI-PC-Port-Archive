from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_nor_b64.

RDNA2 Instruction Set Architecture, SOP2:
    s_nor_b64 D, S0, S1

Semantic operation:
    D.u64 = ~(S0.u64 | S1.u64) & 0xFFFFFFFFFFFFFFFF
    SCC = (D != 0)

Outputs:
    D: 64-bit destination SGPR pair (or EXEC/VCC) storing the bitwise NOR of S0 and S1.
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
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
ALL64: int = 0xFFFF_FFFF_FFFF_FFFF

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def s_nor_b64_oracle(s0: int, s1: int) -> Tuple[int, int]:
    """Reference mathematical model for s_nor_b64.

    Args:
        s0: 64-bit unsigned integer source 0.
        s1: 64-bit unsigned integer source 1.

    Returns:
        (d, scc) tuple where:
            d: (~(s0 | s1)) & MASK64
            scc: 1 if d != 0 else 0
    """
    s0_u64 = s0 & MASK64
    s1_u64 = s1 & MASK64
    d = (~(s0_u64 | s1_u64)) & MASK64
    scc = 1 if d != 0 else 0
    return d, scc


class SNorB64Oracle:
    """Architectural model and hardware-rule validator for s_nor_b64."""

    @staticmethod
    def eval(s0: int, s1: int) -> Tuple[int, int]:
        return s_nor_b64_oracle(s0, s1)

    @staticmethod
    def build_compute_shader(s0: int, s1: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test s_nor_b64."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_nor_b64_hw"
        lines = [
            "s_mov_b64 exec, -1",
            f"s_mov_b32 s2, {s0 & MASK32:#x}",
            f"s_mov_b32 s3, {(s0 >> 32) & MASK32:#x}",
            f"s_mov_b32 s4, {s1 & MASK32:#x}",
            f"s_mov_b32 s5, {(s1 >> 32) & MASK32:#x}",
            "s_nor_b64 s[6:7], s[2:3], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, s1: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = SNorB64Oracle.build_compute_shader(s0, s1)
        return support.translate_lines(name, lines, fssm=fssm)


class TestSNorB64Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_truth_table_bit_combinations(self):
        # NOR truth table: (0,0)->1, (0,1)->0, (1,0)->0, (1,1)->0
        cases = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (1, 1, 0),
        ]
        for b0, b1, expected in cases:
            with self.subTest(b0=b0, b1=b1):
                d, scc = SNorB64Oracle.eval(b0, b1)
                self.assertEqual(d & 1, expected)

    def test_oracle_both_zeros_produce_all_ones(self):
        # NOR(0, 0) = ALL64
        d, scc = SNorB64Oracle.eval(0, 0)
        self.assertEqual(d, ALL64)
        self.assertEqual(scc, 1)

    def test_oracle_all_ones_produces_zero(self):
        # NOR(ALL64, x) = 0
        d, scc = SNorB64Oracle.eval(ALL64, 0x1234_5678_9ABC_DEF0)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)

    def test_oracle_self_nor_is_bitwise_not(self):
        # NOR(x, x) = ~x
        val = 0x5555_5555_5555_5555
        d, scc = SNorB64Oracle.eval(val, val)
        self.assertEqual(d, 0xAAAA_AAAA_AAAA_AAAA)
        self.assertEqual(scc, 1)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SNorB64Oracle.translate_and_validate(0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA)
        self.assertIsInstance(spv, str)
        self.assertIn("OpCapability Shader", spv)
        self.assertIn("OpFunction", spv)
        self.assertIn("OpReturn", spv)

# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------

class RegisterKind(Enum):
    SGPR = auto()
    VGPR = auto()
    EXEC = auto()
    VCC = auto()
    SCC = auto()
    LITERAL64 = auto()


@dataclass(frozen=True)
class RegOperand:
    kind: RegisterKind
    index: int = 0
    size_bits: int = 64

    def format(self) -> str:
        if self.kind == RegisterKind.SGPR:
            if self.size_bits == 64:
                return f"s[{self.index}:{self.index + 1}]"
            return f"s{self.index}"
        elif self.kind == RegisterKind.VGPR:
            if self.size_bits == 64:
                return f"v[{self.index}:{self.index + 1}]"
            return f"v{self.index}"
        elif self.kind == RegisterKind.EXEC:
            return "exec" if self.size_bits == 64 else "exec_lo"
        elif self.kind == RegisterKind.VCC:
            return "vcc" if self.size_bits == 64 else "vcc_lo"
        elif self.kind == RegisterKind.SCC:
            return "scc"
        elif self.kind == RegisterKind.LITERAL64:
            return f"{self.index:#x}"
        return f"reg_{self.index}"


@dataclass
class SNorB64Instruction:
    dst: RegOperand
    src0: RegOperand
    src1: RegOperand
    opcode: str = "s_nor_b64"

    def validate(self) -> None:
        if self.opcode != "s_nor_b64":
            raise ValueError(f"Invalid opcode {self.opcode}")
        if self.dst.kind not in (RegisterKind.SGPR, RegisterKind.EXEC, RegisterKind.VCC):
            raise TypeError(f"Invalid destination register kind: {self.dst.kind}")
        if self.dst.kind == RegisterKind.SGPR:
            if self.dst.size_bits != 64 or (self.dst.index % 2 != 0):
                raise ValueError(f"Destination SGPR must be an even-aligned pair, got s[{self.dst.index}:{self.dst.index+1}]")
            if self.dst.index > 102:
                raise ValueError(f"Destination SGPR index out of range: s{self.dst.index}")
        for src, name in ((self.src0, "src0"), (self.src1, "src1")):
            if src.kind == RegisterKind.SGPR:
                if src.size_bits != 64 or (src.index % 2 != 0):
                    raise ValueError(f"{name} SGPR must be an even-aligned pair, got s[{src.index}:{src.index+1}]")
                if src.index > 104:
                    raise ValueError(f"{name} SGPR index out of range: s{src.index}")

    def emit_assembly(self) -> str:
        self.validate()
        return f"{self.opcode} {self.dst.format()}, {self.src0.format()}, {self.src1.format()}"


class TestSNorB64AST(unittest.TestCase):
    """Unit tests for symbolic AST representation and validation."""

    def test_ast_valid_sgpr_operands(self):
        instr = SNorB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_nor_b64 s[6:7], s[2:3], s[4:5]")

    def test_ast_valid_exec_destination(self):
        instr = SNorB64Instruction(
            dst=RegOperand(RegisterKind.EXEC, size_bits=64),
            src0=RegOperand(RegisterKind.EXEC, size_bits=64),
            src1=RegOperand(RegisterKind.VCC, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_nor_b64 exec, exec, vcc")

    def test_ast_misaligned_destination(self):
        instr = SNorB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=5, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_invalid_vgpr_destination(self):
        instr = SNorB64Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=0, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        with self.assertRaises(TypeError):
            instr.validate()

# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter
# -----------------------------------------------------------------------------

@dataclass
class ScalarHardwareState:
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    exec_mask: int = ALL64
    scc: int = 0
    vcc: int = 0

    def get_sgpr_pair(self, index: int) -> int:
        if index % 2 != 0 or index > 104:
            raise ValueError(f"Invalid SGPR pair index: {index}")
        lo = self.sgprs[index] & MASK32
        hi = self.sgprs[index + 1] & MASK32
        return (hi << 32) | lo

    def set_sgpr_pair(self, index: int, val: int) -> None:
        if index % 2 != 0 or index > 104:
            raise ValueError(f"Invalid SGPR pair index: {index}")
        self.sgprs[index] = val & MASK32
        self.sgprs[index + 1] = (val >> 32) & MASK32

    def read_operand(self, op: RegOperand) -> int:
        if op.kind == RegisterKind.SGPR:
            return self.get_sgpr_pair(op.index)
        elif op.kind == RegisterKind.EXEC:
            return self.exec_mask
        elif op.kind == RegisterKind.VCC:
            return self.vcc
        elif op.kind == RegisterKind.LITERAL64:
            return op.index & MASK64
        raise NotImplementedError(f"Unsupported operand kind: {op.kind}")

    def write_destination(self, op: RegOperand, val: int) -> None:
        val_u64 = val & MASK64
        if op.kind == RegisterKind.SGPR:
            self.set_sgpr_pair(op.index, val_u64)
        elif op.kind == RegisterKind.EXEC:
            self.exec_mask = val_u64
        elif op.kind == RegisterKind.VCC:
            self.vcc = val_u64
        else:
            raise NotImplementedError(f"Unsupported destination kind: {op.kind}")


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter executing s_nor_b64."""

    @staticmethod
    def step(instr: SNorB64Instruction, state: ScalarHardwareState) -> None:
        instr.validate()
        s0_val = state.read_operand(instr.src0)
        s1_val = state.read_operand(instr.src1)

        d, scc = s_nor_b64_oracle(s0_val, s1_val)
        state.write_destination(instr.dst, d)
        state.scc = scc


class TestSNorB64ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter execution."""

    def setUp(self):
        self.state = ScalarHardwareState()

    def test_interpreter_step_basic(self):
        self.state.set_sgpr_pair(2, 0x00FF_0000_00FF_0000)
        self.state.set_sgpr_pair(4, 0x0000_00FF_0000_00FF)
        self.state.scc = 0

        instr = SNorB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)

        expected = (~(0x00FF_0000_00FF_0000 | 0x0000_00FF_0000_00FF)) & MASK64
        self.assertEqual(self.state.get_sgpr_pair(6), expected)
        self.assertEqual(self.state.scc, 1)

    def test_interpreter_step_zero_output(self):
        # S0 | S1 = ALL64 gives NOR = 0
        self.state.set_sgpr_pair(2, 0x5555_5555_5555_5555)
        self.state.set_sgpr_pair(4, 0xAAAA_AAAA_AAAA_AAAA)
        self.state.scc = 1

        instr = SNorB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)

        self.assertEqual(self.state.get_sgpr_pair(6), 0)
        self.assertEqual(self.state.scc, 0)

    def test_interpreter_step_with_vcc_and_exec(self):
        self.state.exec_mask = 0x0F0F_0F0F_0F0F_0F0F
        self.state.vcc = 0x0000_0000_0000_0000

        instr = SNorB64Instruction(
            dst=RegOperand(RegisterKind.VCC, size_bits=64),
            src0=RegOperand(RegisterKind.EXEC, size_bits=64),
            src1=RegOperand(RegisterKind.VCC, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)

        expected = (~0x0F0F_0F0F_0F0F_0F0F) & MASK64
        self.assertEqual(self.state.vcc, expected)
        self.assertEqual(self.state.scc, 1)

# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 64-Bit Lane Masking
# -----------------------------------------------------------------------------

class SIMDMultiLaneDivergenceModel:
    """Models multi-lane execution in Wave64 environments using s_nor_b64."""

    @staticmethod
    def active_lanes_from_mask(mask: int) -> Set[int]:
        return {lane for lane in range(64) if (mask & (1 << lane)) != 0}

    @staticmethod
    def nor_lane_mask(s0_mask: int, s1_mask: int) -> Tuple[int, int]:
        """In Wave64 flow control, s_nor_b64 computes lane activation masks:

        Lanes are ON ONLY if they are inactive in BOTH S0 and S1.
        """
        return s_nor_b64_oracle(s0_mask, s1_mask)


class TestSNorB64SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD Wave64 multi-lane divergence and mask calculation."""

    def test_nor_lane_activation(self):
        # S0 covers lanes 0..15
        s0 = 0x0000_0000_0000_FFFF
        # S1 covers lanes 16..31
        s1 = 0x0000_0000_FFFF_0000

        # NOR activates lanes 32..63
        d, scc = SIMDMultiLaneDivergenceModel.nor_lane_mask(s0, s1)
        self.assertEqual(d, 0xFFFF_FFFF_0000_0000)
        self.assertEqual(scc, 1)

        active = SIMDMultiLaneDivergenceModel.active_lanes_from_mask(d)
        self.assertEqual(active, set(range(32, 64)))

    def test_complete_deactivation(self):
        # S0 and S1 together cover all lanes
        s0 = 0x5555_5555_5555_5555
        s1 = 0xAAAA_AAAA_AAAA_AAAA
        d, scc = s_nor_b64_oracle(s0, s1)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(len(SIMDMultiLaneDivergenceModel.active_lanes_from_mask(d)), 0)

    def test_boundary_lane_nor(self):
        # S0 has lane 0, S1 has lane 63
        s0 = 1 << 0
        s1 = 1 << 63
        d, scc = s_nor_b64_oracle(s0, s1)
        expected = ALL64 & ~((1 << 0) | (1 << 63))
        self.assertEqual(d, expected)
        self.assertEqual(scc, 1)

# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestSNorB64Aliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0(self):
        """s_nor_b64 s[2:3], s[2:3], s[4:5]: D aliases S0."""
        state = ScalarHardwareState()
        s0_val = 0x1111_1111_1111_1111
        s1_val = 0x3333_3333_3333_3333
        state.set_sgpr_pair(2, s0_val)
        state.set_sgpr_pair(4, s1_val)

        instr = SNorB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, state)

        expected = (~(s0_val | s1_val)) & MASK64
        self.assertEqual(state.get_sgpr_pair(2), expected)
        self.assertEqual(state.get_sgpr_pair(4), s1_val)  # S1 untouched
        self.assertEqual(state.scc, 1)

    def test_alias_dst_equals_src1(self):
        """s_nor_b64 s[4:5], s[2:3], s[4:5]: D aliases S1."""
        state = ScalarHardwareState()
        s0_val = 0x00FF_0000_0000_0000
        s1_val = 0x0000_0000_FFFF_0000
        state.set_sgpr_pair(2, s0_val)
        state.set_sgpr_pair(4, s1_val)

        instr = SNorB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=4, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, state)

        expected = (~(s0_val | s1_val)) & MASK64
        self.assertEqual(state.get_sgpr_pair(4), expected)
        self.assertEqual(state.get_sgpr_pair(2), s0_val)  # S0 untouched
        self.assertEqual(state.scc, 1)

    def test_all_operands_aliased_inverts(self):
        """s_nor_b64 s[2:3], s[2:3], s[2:3]: D, S0, and S1 all share same pair.

        NOR(x, x) = ~x (acts as bitwise NOT).
        """
        state = ScalarHardwareState()
        val = 0x0123_4567_89AB_CDEF
        state.set_sgpr_pair(2, val)

        instr = SNorB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=2, size_bits=64)
        )
        ScalarInterpreter.step(instr, state)

        self.assertEqual(state.get_sgpr_pair(2), (~val) & MASK64)
        self.assertEqual(state.scc, 1)

if __name__ == "__main__":
    unittest.main()
