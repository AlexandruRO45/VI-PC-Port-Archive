from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_nor_saveexec_b32.

RDNA2 Instruction Set Architecture, SOP1:
    s_nor_saveexec_b32 D, S0

Semantic operation:
    D.u32 = EXEC_LO
    EXEC_LO.u32 = ~(S0.u32 | EXEC_LO.u32) & 0xFFFFFFFF
    SCC = (EXEC_LO != 0)

Outputs:
    D: 32-bit destination SGPR storing the previous EXEC_LO mask.
    EXEC_LO: Updated 32-bit execution mask.
    SCC: 1 if the updated EXEC_LO mask is non-zero, 0 if all zeros.

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

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------

def s_nor_saveexec_b32_oracle(s0: int, old_exec: int) -> Tuple[int, int, int]:
    """Reference mathematical model for s_nor_saveexec_b32.

    Args:
        s0: 32-bit unsigned integer source mask.
        old_exec: 32-bit unsigned integer execution mask before execution.

    Returns:
        (new_exec, d, scc) tuple where:
            new_exec: ~(s0 | old_exec) & 0xFFFFFFFF (32-bit unsigned)
            d: old_exec & 0xFFFFFFFF (32-bit unsigned)
            scc: 1 if new_exec != 0 else 0
    """
    s0_u32 = s0 & MASK32
    old_exec_u32 = old_exec & MASK32
    new_exec = (~(s0_u32 | old_exec_u32)) & MASK32
    d = old_exec_u32
    scc = 1 if new_exec != 0 else 0
    return new_exec, d, scc


class SNorSaveExecB32Oracle:
    """Architectural model and hardware-rule validator for s_nor_saveexec_b32."""

    @staticmethod
    def eval(s0: int, old_exec: int) -> Tuple[int, int, int]:
        return s_nor_saveexec_b32_oracle(s0, old_exec)

    @staticmethod
    def build_compute_shader(s0: int, old_exec: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test s_nor_saveexec_b32."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 32, 1, 1)
        name = "cs_nor_saveexec_b32_hw"
        lines = [
            f"s_mov_b32 exec_lo, {old_exec & MASK32:#x}",
            f"s_mov_b32 s4, {s0 & MASK32:#x}",
            "s_nor_saveexec_b32 s6, s4",
            "s_cselect_b32 s8, 1, 0",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, old_exec: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = SNorSaveExecB32Oracle.build_compute_shader(s0, old_exec)
        return support.translate_lines(name, lines, fssm=fssm)


class TestSNorSaveExecOracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_basic_nor(self):
        s0 = 0x0000_FFFF
        old_exec = 0x00FF_00FF
        # s0 | old_exec = 0x00FF_FFFF; ~0x00FF_FFFF = 0xFF00_0000
        new_exec, d, scc = SNorSaveExecB32Oracle.eval(s0, old_exec)
        self.assertEqual(new_exec, 0xFF00_0000)
        self.assertEqual(d, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_both_zeros_produce_all_ones(self):
        # NOR(0, 0) = 0xFFFFFFFF: all lanes activate
        new_exec, d, scc = SNorSaveExecB32Oracle.eval(0, 0)
        self.assertEqual(new_exec, MASK32)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 1)

    def test_oracle_all_ones_produces_zero(self):
        # NOR(0xFFFFFFFF, x) = 0: all lanes turn off
        new_exec, d, scc = SNorSaveExecB32Oracle.eval(MASK32, 0x1234_5678)
        self.assertEqual(new_exec, 0)
        self.assertEqual(d, 0x1234_5678)
        self.assertEqual(scc, 0)

    def test_oracle_zero_s0_inverts_exec(self):
        old_exec = 0x5555_5555
        new_exec, d, scc = SNorSaveExecB32Oracle.eval(0, old_exec)
        self.assertEqual(new_exec, 0xAAAA_AAAA)
        self.assertEqual(d, old_exec)
        self.assertEqual(scc, 1)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SNorSaveExecB32Oracle.translate_and_validate(0x5555_5555, 0xAAAA_AAAA)
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
    EXEC_LO = auto()
    VCC_LO = auto()
    SCC = auto()
    LITERAL32 = auto()


@dataclass(frozen=True)
class RegOperand:
    kind: RegisterKind
    index: int = 0
    size_bits: int = 32

    def format(self) -> str:
        if self.kind == RegisterKind.SGPR:
            return f"s{self.index}"
        elif self.kind == RegisterKind.VGPR:
            return f"v{self.index}"
        elif self.kind == RegisterKind.EXEC_LO:
            return "exec_lo"
        elif self.kind == RegisterKind.VCC_LO:
            return "vcc_lo"
        elif self.kind == RegisterKind.SCC:
            return "scc"
        elif self.kind == RegisterKind.LITERAL32:
            return f"{self.index:#x}"
        return f"reg_{self.index}"


@dataclass
class SNorSaveExecB32Instruction:
    dst: RegOperand
    src0: RegOperand
    opcode: str = "s_nor_saveexec_b32"

    def validate(self) -> None:
        if self.opcode != "s_nor_saveexec_b32":
            raise ValueError(f"Invalid opcode {self.opcode}")
        if self.dst.kind != RegisterKind.SGPR:
            raise TypeError(f"Destination must be SGPR, got {self.dst.kind}")
        if self.dst.size_bits != 32:
            raise ValueError(f"Destination SGPR must be 32-bit, got {self.dst.size_bits}")
        if self.dst.index > 105:
            raise ValueError(f"Destination SGPR index out of range: s{self.dst.index}")
        if self.src0.kind == RegisterKind.SGPR:
            if self.src0.size_bits != 32 or self.src0.index > 105:
                raise ValueError(f"Source SGPR must be a valid 32-bit SGPR: s{self.src0.index}")

    def emit_assembly(self) -> str:
        self.validate()
        return f"{self.opcode} {self.dst.format()}, {self.src0.format()}"


class TestSNorSaveExecAST(unittest.TestCase):
    """Unit tests for symbolic AST representation and validation."""

    def test_ast_valid_sgpr_source(self):
        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=32),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=32)
        )
        self.assertEqual(instr.emit_assembly(), "s_nor_saveexec_b32 s6, s4")

    def test_ast_valid_vcc_lo_source(self):
        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=10, size_bits=32),
            src0=RegOperand(RegisterKind.VCC_LO, size_bits=32)
        )
        self.assertEqual(instr.emit_assembly(), "s_nor_saveexec_b32 s10, vcc_lo")

    def test_ast_valid_exec_lo_source(self):
        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=8, size_bits=32),
            src0=RegOperand(RegisterKind.EXEC_LO, size_bits=32)
        )
        self.assertEqual(instr.emit_assembly(), "s_nor_saveexec_b32 s8, exec_lo")

    def test_ast_invalid_destination_size(self):
        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=32)
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_invalid_destination_kind(self):
        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=0, size_bits=32),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=32)
        )
        with self.assertRaises(TypeError):
            instr.validate()

# -----------------------------------------------------------------------------
# Commit 3: Deterministic Scalar Instruction Interpreter
# -----------------------------------------------------------------------------

@dataclass
class ScalarHardwareState32:
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    exec_lo: int = MASK32
    scc: int = 0
    vcc_lo: int = 0

    def get_sgpr(self, index: int) -> int:
        if index < 0 or index > 105:
            raise ValueError(f"Invalid SGPR index: {index}")
        return self.sgprs[index] & MASK32

    def set_sgpr(self, index: int, val: int) -> None:
        if index < 0 or index > 105:
            raise ValueError(f"Invalid SGPR index: {index}")
        self.sgprs[index] = val & MASK32


class ScalarInterpreter32:
    """Deterministic scalar instruction interpreter executing s_nor_saveexec_b32."""

    @staticmethod
    def step(instr: SNorSaveExecB32Instruction, state: ScalarHardwareState32) -> None:
        instr.validate()
        # Read source operand
        if instr.src0.kind == RegisterKind.SGPR:
            s0_val = state.get_sgpr(instr.src0.index)
        elif instr.src0.kind == RegisterKind.EXEC_LO:
            s0_val = state.exec_lo
        elif instr.src0.kind == RegisterKind.VCC_LO:
            s0_val = state.vcc_lo
        elif instr.src0.kind == RegisterKind.LITERAL32:
            s0_val = instr.src0.index & MASK32
        else:
            raise NotImplementedError(f"Unsupported source kind: {instr.src0.kind}")

        new_exec, d, scc = s_nor_saveexec_b32_oracle(s0_val, state.exec_lo)
        # Update destination SGPR with old EXEC_LO
        state.set_sgpr(instr.dst.index, d)
        # Update EXEC_LO and SCC
        state.exec_lo = new_exec
        state.scc = scc


class TestSNorSaveExecScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter state transitions."""

    def setUp(self):
        self.state = ScalarHardwareState32()

    def test_interpreter_step_basic(self):
        self.state.set_sgpr(4, 0x0000_FFFF)
        self.state.exec_lo = 0x00FF_00FF
        self.state.scc = 0

        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=32),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=32)
        )
        ScalarInterpreter32.step(instr, self.state)

        # Destination must hold old EXEC_LO
        self.assertEqual(self.state.get_sgpr(6), 0x00FF_00FF)
        # EXEC_LO must be ~(0x0000_FFFF | 0x00FF_00FF) = 0xFF00_0000
        self.assertEqual(self.state.exec_lo, 0xFF00_0000)
        # SCC must be 1 because new EXEC_LO != 0
        self.assertEqual(self.state.scc, 1)

    def test_interpreter_step_zero_result(self):
        self.state.set_sgpr(4, MASK32)
        self.state.exec_lo = 0
        self.state.scc = 1

        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=32),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=32)
        )
        ScalarInterpreter32.step(instr, self.state)

        self.assertEqual(self.state.get_sgpr(6), 0)
        self.assertEqual(self.state.exec_lo, 0)
        self.assertEqual(self.state.scc, 0)  # SCC cleared to 0

    def test_interpreter_step_from_vcc_lo(self):
        self.state.vcc_lo = 0x0F0F_0F0F
        self.state.exec_lo = 0xF0F0_F0F0
        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=12, size_bits=32),
            src0=RegOperand(RegisterKind.VCC_LO, size_bits=32)
        )
        ScalarInterpreter32.step(instr, self.state)
        self.assertEqual(self.state.get_sgpr(12), 0xF0F0_F0F0)
        # ~(0x0F0F0F0F | 0xF0F0F0F0) = ~0xFFFFFFFF = 0
        self.assertEqual(self.state.exec_lo, 0)
        self.assertEqual(self.state.scc, 0)

# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32-Bit EXEC Masking
# -----------------------------------------------------------------------------

class SIMDMultiLaneDivergenceModel32:
    """Models multi-lane execution in Wave32 environments for NOR logic."""

    @staticmethod
    def active_lanes_from_mask(mask: int) -> Set[int]:
        return {lane for lane in range(32) if (mask & (1 << lane)) != 0}

    @staticmethod
    def mask_from_active_lanes(lanes: Set[int]) -> int:
        return sum(1 << l for l in lanes if 0 <= l < 32)

    @staticmethod
    def nor_divergence_transition(current_exec: int, s0_mask: int) -> Tuple[int, int, int]:
        """In Wave32 control flow, s_nor_saveexec_b32 turns on ONLY lanes

        where NEITHER S0 nor previous EXEC had active threads.
        """
        new_exec, saved_exec, scc = s_nor_saveexec_b32_oracle(s0_mask, current_exec)
        return new_exec, saved_exec, scc


class TestSNorSaveExecSIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD Wave32 multi-lane divergence and NOR lane activation."""

    def test_inactive_lane_resurrection(self):
        # Current active: lanes 0..15 (0x0000_FFFF)
        current_exec = 0x0000_FFFF
        # S0: lanes 16..23 (0x00FF_0000)
        s0_mask = 0x00FF_0000

        # NOR activates lanes 24..31 (0xFF00_0000)
        new_exec, saved_exec, scc = SIMDMultiLaneDivergenceModel32.nor_divergence_transition(
            current_exec, s0_mask
        )
        self.assertEqual(saved_exec, current_exec)
        self.assertEqual(new_exec, 0xFF00_0000)
        self.assertEqual(scc, 1)

        active_lanes = SIMDMultiLaneDivergenceModel32.active_lanes_from_mask(new_exec)
        self.assertEqual(active_lanes, set(range(24, 32)))

    def test_complete_wave_deactivation(self):
        # If S0 covers all remaining lanes, new EXEC becomes empty (wave sleeps)
        current_exec = 0x5555_5555
        s0_mask = 0xAAAA_AAAA
        new_exec, saved_exec, scc = SIMDMultiLaneDivergenceModel32.nor_divergence_transition(
            current_exec, s0_mask
        )
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(len(SIMDMultiLaneDivergenceModel32.active_lanes_from_mask(new_exec)), 0)

    def test_single_lane_divergence(self):
        # Lane 0 and Lane 31
        current_exec = 1 << 0
        s0_mask = 1 << 31
        new_exec, d, scc = s_nor_saveexec_b32_oracle(s0_mask, current_exec)
        # All lanes except 0 and 31 become active
        expected = MASK32 & ~((1 << 0) | (1 << 31))
        self.assertEqual(new_exec, expected)
        self.assertEqual(scc, 1)

# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestSNorSaveExecAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0(self):
        """s_nor_saveexec_b32 s4, s4: D and S0 share the same SGPR.

        In RDNA2 hardware, S0 is sampled into the ALU before destination D
        is updated with the old EXEC_LO.
        """
        state = ScalarHardwareState32()
        initial_s4 = 0x1234_5678
        initial_exec = 0xF0E1_D2C3
        state.set_sgpr(4, initial_s4)
        state.exec_lo = initial_exec

        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=4, size_bits=32),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=32)
        )
        ScalarInterpreter32.step(instr, state)

        # D receives old EXEC_LO
        self.assertEqual(state.get_sgpr(4), initial_exec)
        # EXEC_LO receives ~(initial_s4 | initial_exec)
        expected_exec = (~(initial_s4 | initial_exec)) & MASK32
        self.assertEqual(state.exec_lo, expected_exec)
        self.assertEqual(state.scc, 1)

    def test_src0_is_exec_lo(self):
        """s_nor_saveexec_b32 s6, exec_lo: S0 is EXEC_LO itself.

        ~(EXEC_LO | EXEC_LO) = ~EXEC_LO. D receives old EXEC_LO.
        """
        state = ScalarHardwareState32()
        exec_val = 0xCAFE_BABE
        state.exec_lo = exec_val

        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=32),
            src0=RegOperand(RegisterKind.EXEC_LO, size_bits=32)
        )
        ScalarInterpreter32.step(instr, state)

        self.assertEqual(state.get_sgpr(6), exec_val)
        self.assertEqual(state.exec_lo, (~exec_val) & MASK32)
        self.assertEqual(state.scc, 1)

    def test_multiple_sgpr_independence(self):
        """Ensure un-aliased SGPRs remain untouched."""
        state = ScalarHardwareState32()
        state.set_sgpr(4, 0x1111_1111)
        state.set_sgpr(5, 0x2222_2222)
        state.set_sgpr(6, 0x3333_3333)
        state.exec_lo = 0x0F0F_0F0F

        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=4, size_bits=32),
            src0=RegOperand(RegisterKind.SGPR, index=5, size_bits=32)
        )
        ScalarInterpreter32.step(instr, state)

        # s4 updated with old EXEC_LO
        self.assertEqual(state.get_sgpr(4), 0x0F0F_0F0F)
        # s5 (src0) preserved
        self.assertEqual(state.get_sgpr(5), 0x2222_2222)
        # s6 untouched
        self.assertEqual(state.get_sgpr(6), 0x3333_3333)

# -----------------------------------------------------------------------------
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# -----------------------------------------------------------------------------

class TestSNorSaveExecConditionCodes(unittest.TestCase):
    """Unit tests for SCC flag generation under NOR logic."""

    def test_scc_cleared_when_nor_all_zeros(self):
        """SCC must be 0 when NOR produces 0."""
        for full_mask in (MASK32, 0xFFFF_FFFF):
            with self.subTest(mask=hex(full_mask)):
                new_exec, _, scc = s_nor_saveexec_b32_oracle(full_mask, 0)
                self.assertEqual(new_exec, 0)
                self.assertEqual(scc, 0)

    def test_scc_set_when_single_bit_active(self):
        """SCC must be 1 if at least one bit in new EXEC_LO is set."""
        for bit in (0, 1, 15, 16, 30, 31):
            with self.subTest(bit=bit):
                # An input mask that has 1s everywhere EXCEPT `bit` will leave `bit` as 1 after NOR(mask, 0)
                mask = MASK32 & ~(1 << bit)
                new_exec, _, scc = s_nor_saveexec_b32_oracle(mask, 0)
                self.assertEqual(new_exec, 1 << bit)
                self.assertEqual(scc, 1)

    def test_scc_transitions_state_machine(self):
        """Test all 4 state transitions for SCC: 0->0, 0->1, 1->0, 1->1."""
        state = ScalarHardwareState32()
        instr = SNorSaveExecB32Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=32),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=32)
        )

        # 1. 0 -> 0 transition (NOR(MASK32, 0) = 0)
        state.scc = 0
        state.set_sgpr(4, MASK32)
        state.exec_lo = 0
        ScalarInterpreter32.step(instr, state)
        self.assertEqual(state.scc, 0)

        # 2. 0 -> 1 transition (NOR(0, 0) = MASK32)
        state.set_sgpr(4, 0)
        ScalarInterpreter32.step(instr, state)
        self.assertEqual(state.scc, 1)

        # 3. 1 -> 1 transition (NOR(0, 1) = MASK32 - 1)
        state.set_sgpr(4, 0)
        state.exec_lo = 1
        ScalarInterpreter32.step(instr, state)
        self.assertEqual(state.scc, 1)

        # 4. 1 -> 0 transition (NOR(MASK32, 0) = 0)
        state.set_sgpr(4, MASK32)
        ScalarInterpreter32.step(instr, state)
        self.assertEqual(state.scc, 0)

# -----------------------------------------------------------------------------
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# -----------------------------------------------------------------------------

class TestSNorSaveExecIEEE754Bitwise(unittest.TestCase):
    """Unit tests verifying that IEEE 754 float32 bit patterns are treated as raw bitmasks."""

    FLOAT32_PATTERNS = {
        "pos_zero": 0x0000_0000,
        "neg_zero": 0x8000_0000,
        "pos_inf": 0x7F80_0000,
        "neg_inf": 0xFF80_0000,
        "qnan": 0x7FC0_0000,
        "snan": 0x7F80_0001,
        "min_subnormal": 0x0000_0001,
        "max_subnormal": 0x007F_FFFF,
        "min_normal": 0x0080_0000,
        "max_normal": 0x7F7F_FFFF,
        "one_point_zero": 0x3F80_0000,
    }

    def test_float32_bitwise_nor_inversion(self):
        """Verify SALU bitwise NOR inverts float32 bit patterns bit-exact."""
        for name, pattern in self.FLOAT32_PATTERNS.items():
            with self.subTest(pattern_name=name):
                # NOR(pattern, 0) = ~pattern & 0xFFFFFFFF
                new_exec, d, scc = s_nor_saveexec_b32_oracle(pattern, 0)
                expected_exec = (~pattern) & MASK32
                self.assertEqual(new_exec, expected_exec)
                self.assertEqual(d, 0)
                expected_scc = 1 if expected_exec != 0 else 0
                self.assertEqual(scc, expected_scc)

    def test_float32_zeros_nor(self):
        """Verify NOR of pos_zero and neg_zero."""
        pos_zero = self.FLOAT32_PATTERNS["pos_zero"]
        neg_zero = self.FLOAT32_PATTERNS["neg_zero"]
        # NOR(0, 0x80000000) = ~0x80000000 = 0x7FFFFFFF
        new_exec, _, scc = s_nor_saveexec_b32_oracle(pos_zero, neg_zero)
        self.assertEqual(new_exec, 0x7FFF_FFFF)
        self.assertEqual(scc, 1)

if __name__ == "__main__":
    unittest.main()
