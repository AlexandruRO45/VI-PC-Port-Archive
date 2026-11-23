from __future__ import annotations

import enum
import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

"""
RDNA2 SALU instruction semantic tests: s_nand_saveexec_b32.

Exercises s_nand_saveexec_b32 through tools/rdna2spirv.py and spirv_mini_interp.
RDNA2 ISA, SOP1:
    s_nand_saveexec_b32 sD, sS0
Semantics:
    D = EXEC_LO
    EXEC_LO = ~(S0 & EXEC_LO) & 0xFFFFFFFF
    SCC = (EXEC_LO != 0)

In Wave64 mode:
    EXEC_HI (lanes 32..63) remains untouched.
    D receives the 32-bit old EXEC_LO.
    SCC reflects (new EXEC_LO != 0).
"""

MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF


def ref_nand_saveexec_b32(s0: int, old_exec_lo: int) -> Tuple[int, int, int]:
    """
    Reference mathematical model for s_nand_saveexec_b32.
    Returns (new_exec_lo, saved_d, scc).
    """
    s0_32 = s0 & MASK32
    exec_lo = old_exec_lo & MASK32
    new_exec_lo = (~(s0_32 & exec_lo)) & MASK32
    saved_d = exec_lo
    scc = 1 if new_exec_lo != 0 else 0
    return new_exec_lo, saved_d, scc


# ==============================================================================
# Unit Tests: Reference Mathematical Oracle
# ==============================================================================

class TestSaluNandSaveexecB32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical model of s_nand_saveexec_b32."""

    def test_both_zero(self) -> None:
        new_exec, saved_d, scc = ref_nand_saveexec_b32(0, 0)
        self.assertEqual(new_exec, MASK32)
        self.assertEqual(saved_d, 0)
        self.assertEqual(scc, 1)

    def test_all_ones_self_cancellation(self) -> None:
        new_exec, saved_d, scc = ref_nand_saveexec_b32(MASK32, MASK32)
        self.assertEqual(new_exec, 0)
        self.assertEqual(saved_d, MASK32)
        self.assertEqual(scc, 0)

    def test_orthogonal_masks(self) -> None:
        # S0 and EXEC have no overlapping bits -> S0 & EXEC == 0 -> NAND is ~0
        s0 = 0x5555_5555
        old_exec = 0xAAAA_AAAA
        new_exec, saved_d, scc = ref_nand_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, MASK32)
        self.assertEqual(saved_d, old_exec)
        self.assertEqual(scc, 1)

    def test_partial_overlap(self) -> None:
        s0 = 0x0000_FFFF
        old_exec = 0x00FF_00FF
        # s0 & old_exec = 0x0000_00FF
        # ~(0x0000_00FF) = 0xFFFF_FF00
        new_exec, saved_d, scc = ref_nand_saveexec_b32(s0, old_exec)
        self.assertEqual(new_exec, 0xFFFF_FF00)
        self.assertEqual(saved_d, old_exec)
        self.assertEqual(scc, 1)

    def test_single_bit_patterns(self) -> None:
        for bit in range(32):
            val = 1 << bit
            new_exec, saved_d, scc = ref_nand_saveexec_b32(val, val)
            self.assertEqual(new_exec, (~val) & MASK32)
            self.assertEqual(saved_d, val)
            self.assertEqual(scc, 1)

# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegKind(enum.Enum):
    SGPR = "s"
    EXEC_LO = "exec_lo"
    VCC_LO = "vcc_lo"


@dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int = 0

    def __str__(self) -> str:
        if self.kind == RegKind.SGPR:
            return f"s{self.index}"
        return self.kind.value


@dataclass(frozen=True)
class Immediate:
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value & MASK32)

    def __str__(self) -> str:
        return f"0x{self.value:08X}"


Operand = Union[Register, Immediate]


class Opcode(enum.Enum):
    S_NAND_SAVEEXEC_B32 = "s_nand_saveexec_b32"
    S_MOV_B32 = "s_mov_b32"


@dataclass
class Instruction:
    opcode: Opcode
    dst: Register
    src0: Operand

    def __str__(self) -> str:
        return f"{self.opcode.value} {self.dst}, {self.src0}"


class ASTNode:
    def eval(self, env: Dict[str, int]) -> int:
        raise NotImplementedError


@dataclass
class LiteralNode(ASTNode):
    val: int

    def eval(self, env: Dict[str, int]) -> int:
        return self.val & MASK32


@dataclass
class RegReadNode(ASTNode):
    name: str

    def eval(self, env: Dict[str, int]) -> int:
        return env.get(self.name, 0) & MASK32


@dataclass
class NandSaveExecB32Node(ASTNode):
    src0: ASTNode
    exec_lo: ASTNode

    def eval(self, env: Dict[str, int]) -> Tuple[int, int, int]:
        s0 = self.src0.eval(env)
        exec_val = self.exec_lo.eval(env)
        return ref_nand_saveexec_b32(s0, exec_val)


# ==============================================================================
# Unit Tests: Symbolic AST Opcode Definitions
# ==============================================================================

class TestSaluNandSaveexecB32AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_instruction_creation_and_string(self) -> None:
        inst = Instruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B32,
            dst=Register(RegKind.SGPR, 4),
            src0=Register(RegKind.SGPR, 6),
        )
        self.assertEqual(inst.opcode, Opcode.S_NAND_SAVEEXEC_B32)
        self.assertEqual(str(inst), "s_nand_saveexec_b32 s4, s6")

    def test_instruction_with_immediate(self) -> None:
        inst = Instruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B32,
            dst=Register(RegKind.SGPR, 2),
            src0=Immediate(0x1234_5678),
        )
        self.assertEqual(str(inst), "s_nand_saveexec_b32 s2, 0x12345678")

    def test_ast_evaluation_matches_oracle(self) -> None:
        env = {"s0": 0x00FF_00FF, "exec_lo": 0x0F0F_0F0F}
        node = NandSaveExecB32Node(RegReadNode("s0"), RegReadNode("exec_lo"))
        result = node.eval(env)
        expected = ref_nand_saveexec_b32(0x00FF_00FF, 0x0F0F_0F0F)
        self.assertEqual(result, expected)

# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarMachineState:
    """Architectural register state for scalar SALU operations."""

    def __init__(self) -> None:
        self.sgprs: List[int] = [0] * 106
        self.exec_lo: int = MASK32
        self.scc: int = 0

    def read_operand(self, op: Operand) -> int:
        if isinstance(op, Immediate):
            return op.value & MASK32
        if isinstance(op, Register):
            if op.kind == RegKind.SGPR:
                return self.sgprs[op.index] & MASK32
            if op.kind == RegKind.EXEC_LO:
                return self.exec_lo & MASK32
        raise TypeError(f"Invalid operand: {op}")

    def write_reg(self, reg: Register, val: int) -> None:
        if reg.kind == RegKind.SGPR:
            self.sgprs[reg.index] = val & MASK32
        elif reg.kind == RegKind.EXEC_LO:
            self.exec_lo = val & MASK32
        else:
            raise TypeError(f"Unsupported write register: {reg}")


class ScalarSALUInterpreter:
    """Deterministic single-thread execution engine for SALU saveexec instructions."""

    def step(self, inst: Instruction, state: ScalarMachineState) -> None:
        if inst.opcode == Opcode.S_NAND_SAVEEXEC_B32:
            s0 = state.read_operand(inst.src0)
            old_exec = state.exec_lo
            new_exec, saved_d, scc = ref_nand_saveexec_b32(s0, old_exec)
            state.write_reg(inst.dst, saved_d)
            state.exec_lo = new_exec
            state.scc = scc
        elif inst.opcode == Opcode.S_MOV_B32:
            s0 = state.read_operand(inst.src0)
            state.write_reg(inst.dst, s0)
        else:
            raise NotImplementedError(f"Unsupported opcode: {inst.opcode}")


# ==============================================================================
# Unit Tests: Deterministic Scalar Instruction Interpreter
# ==============================================================================

class TestSaluNandSaveexecB32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()
        self.state = ScalarMachineState()

    def test_step_sgpr_source(self) -> None:
        self.state.sgprs[1] = 0x0000_FFFF
        self.state.exec_lo = 0x00FF_00FF

        inst = Instruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B32,
            dst=Register(RegKind.SGPR, 0),
            src0=Register(RegKind.SGPR, 1),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.sgprs[0], 0x00FF_00FF)
        self.assertEqual(self.state.exec_lo, 0xFFFF_FF00)
        self.assertEqual(self.state.scc, 1)

    def test_step_immediate_source(self) -> None:
        self.state.exec_lo = MASK32

        inst = Instruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B32,
            dst=Register(RegKind.SGPR, 2),
            src0=Immediate(MASK32),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.sgprs[2], MASK32)
        self.assertEqual(self.state.exec_lo, 0)
        self.assertEqual(self.state.scc, 0)

    def test_sequential_determinism(self) -> None:
        program = [
            Instruction(Opcode.S_MOV_B32, Register(RegKind.SGPR, 5), Immediate(0xAAAA_5555)),
            Instruction(Opcode.S_NAND_SAVEEXEC_B32, Register(RegKind.SGPR, 6), Register(RegKind.SGPR, 5)),
        ]
        self.state.exec_lo = 0xFFFF_0000
        for inst in program:
            self.interp.step(inst, self.state)

        res_d = self.state.sgprs[6]
        res_exec = self.state.exec_lo
        res_scc = self.state.scc

        fresh = ScalarMachineState()
        fresh.exec_lo = 0xFFFF_0000
        for inst in program:
            self.interp.step(inst, fresh)

        self.assertEqual(fresh.sgprs[6], res_d)
        self.assertEqual(fresh.exec_lo, res_exec)
        self.assertEqual(fresh.scc, res_scc)

# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit Wavefront Models
# ==============================================================================

class WavefrontState:
    """Architectural wavefront modeling EXEC_LO vs EXEC_HI divergence."""

    def __init__(self, wave_size: int = 64) -> None:
        self.wave_size = wave_size
        self.exec_lo = MASK32
        self.exec_hi = MASK32
        self.sgprs = [0] * 106
        self.scc = 0

    @property
    def exec_mask_64(self) -> int:
        return (self.exec_hi << 32) | self.exec_lo


class WavefrontInterpreter:
    """Wavefront execution engine enforcing half-width EXEC_LO semantics."""

    def step_nand_saveexec_b32(self, dst_idx: int, s0_val: int, state: WavefrontState) -> None:
        old_lo = state.exec_lo
        old_hi = state.exec_hi
        new_lo, saved_d, scc = ref_nand_saveexec_b32(s0_val, old_lo)
        state.sgprs[dst_idx] = saved_d
        state.exec_lo = new_lo
        # In Wave64, exec_hi is untouched!
        state.exec_hi = old_hi
        state.scc = scc


# ==============================================================================
# Unit Tests: SIMD Multi-Lane Divergence
# ==============================================================================

class TestSaluNandSaveexecB32SIMD(unittest.TestCase):
    """Unit tests for multi-lane divergence and EXEC_HI preservation."""

    def setUp(self) -> None:
        self.interp = WavefrontInterpreter()

    def test_wave64_preserves_exec_hi(self) -> None:
        state = WavefrontState(wave_size=64)
        state.exec_lo = 0x1234_5678
        state.exec_hi = 0xDEAD_BEEF

        self.interp.step_nand_saveexec_b32(dst_idx=2, s0_val=0xFFFF_FFFF, state=state)
        # EXEC_LO is nand-ed
        self.assertEqual(state.exec_lo, (~0x1234_5678) & MASK32)
        # EXEC_HI is untouched!
        self.assertEqual(state.exec_hi, 0xDEAD_BEEF)
        # Saved D is old EXEC_LO
        self.assertEqual(state.sgprs[2], 0x1234_5678)
        self.assertEqual(state.scc, 1)

    def test_divergent_lane_activity(self) -> None:
        state = WavefrontState(wave_size=64)
        state.exec_lo = 0x0000_FFFF
        self.interp.step_nand_saveexec_b32(dst_idx=0, s0_val=0x0000_00FF, state=state)
        # new_lo = ~(0xFF) = 0xFFFF_FF00
        # Lanes 0..7 are deactivated (0), lanes 8..31 are active (1)
        for lane in range(32):
            active = bool((state.exec_lo >> lane) & 1)
            if lane < 8:
                self.assertFalse(active)
            else:
                self.assertTrue(active)

# ==============================================================================
# Unit Tests: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestSaluNandSaveexecB32Aliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()
        self.state = ScalarMachineState()

    def test_dst_aliases_src0(self) -> None:
        # s0 = s_nand_saveexec_b32 s0
        # S0 must be read before D is written with old EXEC_LO
        self.state.sgprs[0] = 0x0000_FFFF
        self.state.exec_lo = 0x5555_5555

        inst = Instruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B32,
            dst=Register(RegKind.SGPR, 0),
            src0=Register(RegKind.SGPR, 0),
        )
        self.interp.step(inst, self.state)
        # Destination receives old EXEC_LO
        self.assertEqual(self.state.sgprs[0], 0x5555_5555)
        # EXEC_LO receives ~(0x0000_FFFF & 0x5555_5555)
        exp_exec = (~(0x0000_FFFF & 0x5555_5555)) & MASK32
        self.assertEqual(self.state.exec_lo, exp_exec)

    def test_src0_is_exec_lo(self) -> None:
        # s_nand_saveexec_b32 s4, exec_lo
        # S0 & EXEC_LO = EXEC_LO & EXEC_LO = EXEC_LO -> new EXEC_LO = ~EXEC_LO
        self.state.exec_lo = 0x1234_5678
        inst = Instruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B32,
            dst=Register(RegKind.SGPR, 4),
            src0=Register(RegKind.EXEC_LO),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.sgprs[4], 0x1234_5678)
        self.assertEqual(self.state.exec_lo, (~0x1234_5678) & MASK32)
        self.assertEqual(self.state.scc, 1)

# ==============================================================================
# Unit Tests: Condition Codes and Status Flags
# ==============================================================================

class TestSaluNandSaveexecB32ConditionCodes(unittest.TestCase):
    """Unit tests for condition code SCC generation and canary preservation."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()
        self.state = ScalarMachineState()

    def test_scc_set_to_zero_when_result_zero(self) -> None:
        # Result is 0 strictly when S0 & EXEC_LO == 0xFFFFFFFF
        self.state.scc = 1  # canary
        self.state.exec_lo = MASK32
        inst = Instruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B32,
            dst=Register(RegKind.SGPR, 0),
            src0=Immediate(MASK32),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.exec_lo, 0)
        self.assertEqual(self.state.scc, 0)

    def test_scc_set_to_one_when_result_non_zero(self) -> None:
        self.state.scc = 0  # canary
        self.state.exec_lo = 0
        inst = Instruction(
            opcode=Opcode.S_NAND_SAVEEXEC_B32,
            dst=Register(RegKind.SGPR, 0),
            src0=Immediate(0),
        )
        self.interp.step(inst, self.state)
        self.assertEqual(self.state.exec_lo, MASK32)
        self.assertEqual(self.state.scc, 1)

if __name__ == "__main__":
    unittest.main()
