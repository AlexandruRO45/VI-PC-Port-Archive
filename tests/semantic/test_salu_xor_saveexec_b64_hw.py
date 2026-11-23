from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_xor_saveexec_b64.

RDNA2 Instruction Set Architecture, SOP1:
    s_xor_saveexec_b64 D, S0

Semantic operation:
    D.u64 = EXEC
    EXEC.u64 = S0.u64 ^ EXEC
    SCC = (EXEC != 0)

Outputs:
    D: 64-bit destination SGPR pair storing the previous EXEC mask.
    EXEC: Updated 64-bit execution mask.
    SCC: 1 if the updated EXEC mask is non-zero, 0 if it is all zeros.

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

def s_xor_saveexec_b64_oracle(s0: int, old_exec: int) -> Tuple[int, int, int]:
    """Reference mathematical model for s_xor_saveexec_b64.

    Args:
        s0: 64-bit unsigned integer source mask.
        old_exec: 64-bit unsigned integer execution mask before execution.

    Returns:
        (new_exec, d, scc) tuple where:
            new_exec: (s0 ^ old_exec) & MASK64
            d: old_exec (64-bit unsigned)
            scc: 1 if new_exec != 0 else 0
    """
    s0_u64 = s0 & MASK64
    old_exec_u64 = old_exec & MASK64
    new_exec = (s0_u64 ^ old_exec_u64) & MASK64
    d = old_exec_u64
    scc = 1 if new_exec != 0 else 0
    return new_exec, d, scc


class SXorSaveExecB64Oracle:
    """Architectural model and hardware-rule validator for s_xor_saveexec_b64."""

    @staticmethod
    def eval(s0: int, old_exec: int) -> Tuple[int, int, int]:
        return s_xor_saveexec_b64_oracle(s0, old_exec)

    @staticmethod
    def build_compute_shader(s0: int, old_exec: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test s_xor_saveexec_b64."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_xor_saveexec_b64_hw"
        lines = [
            f"s_mov_b32 exec_lo, {old_exec & MASK32:#x}",
            f"s_mov_b32 exec_hi, {(old_exec >> 32) & MASK32:#x}",
            f"s_mov_b32 s4, {s0 & MASK32:#x}",
            f"s_mov_b32 s5, {(s0 >> 32) & MASK32:#x}",
            "s_xor_saveexec_b64 s[6:7], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, old_exec: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = SXorSaveExecB64Oracle.build_compute_shader(s0, old_exec)
        return support.translate_lines(name, lines, fssm=fssm)


class TestSXorSaveExecOracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_basic_xor(self):
        s0 = 0x0000_FFFF_0000_FFFF
        old_exec = 0x00FF_00FF_00FF_00FF
        new_exec, d, scc = SXorSaveExecB64Oracle.eval(s0, old_exec)
        self.assertEqual(new_exec, 0x00FF_FF00_00FF_FF00)
        self.assertEqual(d, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_identical_operands_produce_zero(self):
        val = 0x1234_5678_9ABC_DEF0
        new_exec, d, scc = SXorSaveExecB64Oracle.eval(val, val)
        self.assertEqual(new_exec, 0)
        self.assertEqual(d, val)
        self.assertEqual(scc, 0)

    def test_oracle_zero_s0(self):
        old_exec = 0x8000_0000_0000_0001
        new_exec, d, scc = SXorSaveExecB64Oracle.eval(0, old_exec)
        self.assertEqual(new_exec, old_exec)
        self.assertEqual(d, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_zero_exec(self):
        s0 = 0xDEAD_BEEF_CAFE_BABE
        new_exec, d, scc = SXorSaveExecB64Oracle.eval(s0, 0)
        self.assertEqual(new_exec, s0)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 1)

    def test_oracle_all_ones_inversion(self):
        mask = 0x5555_5555_5555_5555
        new_exec, d, scc = SXorSaveExecB64Oracle.eval(ALL64, mask)
        self.assertEqual(new_exec, 0xAAAA_AAAA_AAAA_AAAA)
        self.assertEqual(d, mask)
        self.assertEqual(scc, 1)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SXorSaveExecB64Oracle.translate_and_validate(0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA)
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
class SXorSaveExecB64Instruction:
    dst: RegOperand
    src0: RegOperand
    opcode: str = "s_xor_saveexec_b64"

    def validate(self) -> None:
        if self.opcode != "s_xor_saveexec_b64":
            raise ValueError(f"Invalid opcode {self.opcode}")
        if self.dst.kind != RegisterKind.SGPR:
            raise TypeError(f"Destination must be SGPR pair, got {self.dst.kind}")
        if self.dst.size_bits != 64 or (self.dst.index % 2 != 0):
            raise ValueError(f"Destination SGPR must be an even-aligned pair, got s[{self.dst.index}:{self.dst.index+1}]")
        if self.dst.index > 102:
            raise ValueError(f"Destination SGPR index out of range: s{self.dst.index}")
        if self.src0.kind == RegisterKind.SGPR:
            if self.src0.size_bits != 64 or (self.src0.index % 2 != 0):
                raise ValueError(f"Source SGPR must be an even-aligned pair, got s[{self.src0.index}:{self.src0.index+1}]")

    def emit_assembly(self) -> str:
        self.validate()
        return f"{self.opcode} {self.dst.format()}, {self.src0.format()}"


class TestSXorSaveExecAST(unittest.TestCase):
    """Unit tests for symbolic AST representation and validation."""

    def test_ast_valid_sgpr_source(self):
        instr = SXorSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_xor_saveexec_b64 s[6:7], s[4:5]")

    def test_ast_valid_vcc_source(self):
        instr = SXorSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=10, size_bits=64),
            src0=RegOperand(RegisterKind.VCC, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_xor_saveexec_b64 s[10:11], vcc")

    def test_ast_valid_exec_source(self):
        instr = SXorSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=8, size_bits=64),
            src0=RegOperand(RegisterKind.EXEC, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_xor_saveexec_b64 s[8:9], exec")

    def test_ast_misaligned_destination(self):
        instr = SXorSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=3, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_invalid_destination_kind(self):
        instr = SXorSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=0, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
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


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter executing s_xor_saveexec_b64."""

    @staticmethod
    def step(instr: SXorSaveExecB64Instruction, state: ScalarHardwareState) -> None:
        instr.validate()
        # Read source operand
        if instr.src0.kind == RegisterKind.SGPR:
            s0_val = state.get_sgpr_pair(instr.src0.index)
        elif instr.src0.kind == RegisterKind.EXEC:
            s0_val = state.exec_mask
        elif instr.src0.kind == RegisterKind.VCC:
            s0_val = state.vcc
        elif instr.src0.kind == RegisterKind.LITERAL64:
            s0_val = instr.src0.index
        else:
            raise NotImplementedError(f"Unsupported source kind: {instr.src0.kind}")

        new_exec, d, scc = s_xor_saveexec_b64_oracle(s0_val, state.exec_mask)
        # Update destination SGPR pair with old EXEC
        state.set_sgpr_pair(instr.dst.index, d)
        # Update EXEC and SCC
        state.exec_mask = new_exec
        state.scc = scc


class TestSXorSaveExecScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter state transitions."""

    def setUp(self):
        self.state = ScalarHardwareState()

    def test_interpreter_step_basic(self):
        self.state.set_sgpr_pair(4, 0x00FF_0000_00FF_0000)
        self.state.exec_mask = 0x00FF_00FF_00FF_00FF
        self.state.scc = 0

        instr = SXorSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)

        # Destination must hold old EXEC
        self.assertEqual(self.state.get_sgpr_pair(6), 0x00FF_00FF_00FF_00FF)
        # EXEC must be bitwise XOR of s0 and old EXEC
        self.assertEqual(self.state.exec_mask, 0x0000_00FF_0000_00FF)
        # SCC must be 1 because new EXEC != 0
        self.assertEqual(self.state.scc, 1)

    def test_interpreter_step_zero_result(self):
        val = 0x1111_2222_3333_4444
        self.state.set_sgpr_pair(4, val)
        self.state.exec_mask = val
        self.state.scc = 1  # Previous SCC was 1

        instr = SXorSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)

        self.assertEqual(self.state.get_sgpr_pair(6), val)
        self.assertEqual(self.state.exec_mask, 0)
        self.assertEqual(self.state.scc, 0)  # SCC cleared to 0

    def test_interpreter_step_from_vcc(self):
        self.state.vcc = 0x1111_2222_3333_4444
        self.state.exec_mask = 0x5555_6666_7777_8888
        instr = SXorSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=12, size_bits=64),
            src0=RegOperand(RegisterKind.VCC, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)
        self.assertEqual(self.state.get_sgpr_pair(12), 0x5555_6666_7777_8888)
        self.assertEqual(self.state.exec_mask, 0x1111_2222_3333_4444 ^ 0x5555_6666_7777_8888)
        self.assertEqual(self.state.scc, 1)

if __name__ == "__main__":
    unittest.main()
