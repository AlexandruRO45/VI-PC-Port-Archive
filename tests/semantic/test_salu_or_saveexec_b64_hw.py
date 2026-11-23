from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_or_saveexec_b64.

RDNA2 Instruction Set Architecture, SOP1:
    s_or_saveexec_b64 D, S0

Semantic operation:
    D.u64 = EXEC
    EXEC.u64 = S0.u64 | EXEC
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

def s_or_saveexec_b64_oracle(s0: int, old_exec: int) -> Tuple[int, int, int]:
    """Reference mathematical model for s_or_saveexec_b64.

    Args:
        s0: 64-bit unsigned integer source mask.
        old_exec: 64-bit unsigned integer execution mask before execution.

    Returns:
        (new_exec, d, scc) tuple where:
            new_exec: s0 | old_exec (64-bit unsigned)
            d: old_exec (64-bit unsigned)
            scc: 1 if new_exec != 0 else 0
    """
    s0_u64 = s0 & MASK64
    old_exec_u64 = old_exec & MASK64
    new_exec = (s0_u64 | old_exec_u64) & MASK64
    d = old_exec_u64
    scc = 1 if new_exec != 0 else 0
    return new_exec, d, scc


class SOrSaveExecB64Oracle:
    """Architectural model and hardware-rule validator for s_or_saveexec_b64."""

    @staticmethod
    def eval(s0: int, old_exec: int) -> Tuple[int, int, int]:
        return s_or_saveexec_b64_oracle(s0, old_exec)

    @staticmethod
    def build_compute_shader(s0: int, old_exec: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test s_or_saveexec_b64."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_or_saveexec_b64_hw"
        lines = [
            f"s_mov_b32 exec_lo, {old_exec & MASK32:#x}",
            f"s_mov_b32 exec_hi, {(old_exec >> 32) & MASK32:#x}",
            f"s_mov_b32 s4, {s0 & MASK32:#x}",
            f"s_mov_b32 s5, {(s0 >> 32) & MASK32:#x}",
            "s_or_saveexec_b64 s[6:7], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, old_exec: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = SOrSaveExecB64Oracle.build_compute_shader(s0, old_exec)
        return support.translate_lines(name, lines, fssm=fssm)


class TestSOrSaveExecOracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_basic_overlap(self):
        s0 = 0x0000_FFFF_0000_FFFF
        old_exec = 0x00FF_00FF_00FF_00FF
        new_exec, d, scc = SOrSaveExecB64Oracle.eval(s0, old_exec)
        self.assertEqual(new_exec, 0x00FF_FFFF_00FF_FFFF)
        self.assertEqual(d, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_zero_s0(self):
        old_exec = 0x1234_5678_9ABC_DEF0
        new_exec, d, scc = SOrSaveExecB64Oracle.eval(0, old_exec)
        self.assertEqual(new_exec, old_exec)
        self.assertEqual(d, old_exec)
        self.assertEqual(scc, 1)

    def test_oracle_zero_exec(self):
        s0 = 0x8000_0000_0000_0001
        new_exec, d, scc = SOrSaveExecB64Oracle.eval(s0, 0)
        self.assertEqual(new_exec, s0)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 1)

    def test_oracle_all_zeros(self):
        new_exec, d, scc = SOrSaveExecB64Oracle.eval(0, 0)
        self.assertEqual(new_exec, 0)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)

    def test_oracle_all_ones(self):
        new_exec, d, scc = SOrSaveExecB64Oracle.eval(ALL64, ALL64)
        self.assertEqual(new_exec, ALL64)
        self.assertEqual(d, ALL64)
        self.assertEqual(scc, 1)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SOrSaveExecB64Oracle.translate_and_validate(0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA)
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
class SOrSaveExecB64Instruction:
    dst: RegOperand
    src0: RegOperand
    opcode: str = "s_or_saveexec_b64"

    def validate(self) -> None:
        if self.opcode != "s_or_saveexec_b64":
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


class TestSOrSaveExecAST(unittest.TestCase):
    """Unit tests for symbolic AST representation and validation."""

    def test_ast_valid_sgpr_source(self):
        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_or_saveexec_b64 s[6:7], s[4:5]")

    def test_ast_valid_vcc_source(self):
        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=10, size_bits=64),
            src0=RegOperand(RegisterKind.VCC, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_or_saveexec_b64 s[10:11], vcc")

    def test_ast_valid_exec_source(self):
        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=8, size_bits=64),
            src0=RegOperand(RegisterKind.EXEC, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_or_saveexec_b64 s[8:9], exec")

    def test_ast_misaligned_destination(self):
        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=5, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=6, size_bits=64)
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_invalid_destination_kind(self):
        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=6, size_bits=64),
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
    """Deterministic scalar instruction interpreter executing s_or_saveexec_b64."""

    @staticmethod
    def step(instr: SOrSaveExecB64Instruction, state: ScalarHardwareState) -> None:
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

        new_exec, d, scc = s_or_saveexec_b64_oracle(s0_val, state.exec_mask)
        # Update destination SGPR pair with old EXEC
        state.set_sgpr_pair(instr.dst.index, d)
        # Update EXEC and SCC
        state.exec_mask = new_exec
        state.scc = scc


class TestSOrSaveExecScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter state transitions."""

    def setUp(self):
        self.state = ScalarHardwareState()

    def test_interpreter_step_basic(self):
        self.state.set_sgpr_pair(4, 0x00FF_0000_00FF_0000)
        self.state.exec_mask = 0x0000_FF00_0000_FF00
        self.state.scc = 0

        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)

        # Destination must hold old EXEC
        self.assertEqual(self.state.get_sgpr_pair(6), 0x0000_FF00_0000_FF00)
        # EXEC must be bitwise OR of s0 and old EXEC
        self.assertEqual(self.state.exec_mask, 0x00FF_FF00_00FF_FF00)
        # SCC must be 1 because new EXEC != 0
        self.assertEqual(self.state.scc, 1)

    def test_interpreter_step_zero_result(self):
        self.state.set_sgpr_pair(4, 0)
        self.state.exec_mask = 0
        self.state.scc = 1  # Previous SCC was 1

        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)

        self.assertEqual(self.state.get_sgpr_pair(6), 0)
        self.assertEqual(self.state.exec_mask, 0)
        self.assertEqual(self.state.scc, 0)  # SCC cleared to 0

    def test_interpreter_step_from_vcc(self):
        self.state.vcc = 0x1111_2222_3333_4444
        self.state.exec_mask = 0x5555_6666_7777_8888
        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=12, size_bits=64),
            src0=RegOperand(RegisterKind.VCC, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)
        self.assertEqual(self.state.get_sgpr_pair(12), 0x5555_6666_7777_8888)
        self.assertEqual(self.state.exec_mask, 0x1111_2222_3333_4444 | 0x5555_6666_7777_8888)
        self.assertEqual(self.state.scc, 1)

# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# -----------------------------------------------------------------------------

class SIMDMultiLaneDivergenceModel:
    """Models multi-lane execution in Wave64 / Wave32 environments."""

    @staticmethod
    def active_lanes_from_mask(mask: int, wave_size: int = 64) -> Set[int]:
        return {lane for lane in range(wave_size) if (mask & (1 << lane)) != 0}

    @staticmethod
    def mask_from_active_lanes(lanes: Set[int]) -> int:
        return sum(1 << l for l in lanes if 0 <= l < 64)

    @staticmethod
    def simulate_divergent_branch_reconvergence(
        initial_exec: int,
        then_branch_mask: int,
        else_branch_mask: int
    ) -> Tuple[int, int, int]:
        """Simulate conditional branch divergence where inactive lanes re-activate.

        In RDNA2 control flow, s_or_saveexec_b64 is used to re-enable lanes
        that were previously turned off by a divergent branch.
        """
        new_exec, saved_exec, scc = s_or_saveexec_b64_oracle(else_branch_mask, then_branch_mask)
        return new_exec, saved_exec, scc


class TestSOrSaveExecSIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and lane-re-activation."""

    def test_divergent_lane_re_activation(self):
        # Initial: only even lanes active (0, 2, 4, ... 62)
        even_mask = 0x5555_5555_5555_5555
        # Else branch: odd lanes (1, 3, 5, ... 63)
        odd_mask = 0xAAAA_AAAA_AAAA_AAAA

        new_exec, saved_exec, scc = SIMDMultiLaneDivergenceModel.simulate_divergent_branch_reconvergence(
            ALL64, even_mask, odd_mask
        )
        self.assertEqual(saved_exec, even_mask)
        self.assertEqual(new_exec, ALL64)
        self.assertEqual(scc, 1)

        active_lanes = SIMDMultiLaneDivergenceModel.active_lanes_from_mask(new_exec)
        self.assertEqual(len(active_lanes), 64)

    def test_wave32_compatibility_half_mask(self):
        # Wave32 uses lower 32 lanes; upper 32 bits remain 0
        w32_s0 = 0x0000_0000_0000_F0F0
        w32_exec = 0x0000_0000_0000_0F0F
        new_exec, d, scc = s_or_saveexec_b64_oracle(w32_s0, w32_exec)
        self.assertEqual(new_exec, 0x0000_0000_0000_FFFF)
        self.assertEqual(d, w32_exec)
        self.assertEqual(scc, 1)
        # Ensure high 32 bits remain zero
        self.assertEqual(new_exec >> 32, 0)

    def test_sparse_multi_lane_patterns(self):
        # Only lanes 0, 31, 32, 63
        boundary_s0 = (1 << 0) | (1 << 31)
        boundary_exec = (1 << 32) | (1 << 63)
        new_exec, d, scc = s_or_saveexec_b64_oracle(boundary_s0, boundary_exec)
        self.assertEqual(new_exec, (1 << 0) | (1 << 31) | (1 << 32) | (1 << 63))
        self.assertEqual(d, boundary_exec)
        self.assertEqual(scc, 1)

# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestSOrSaveExecAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0(self):
        """s_or_saveexec_b64 s[4:5], s[4:5]: D and S0 share the same SGPR pair.

        In RDNA2 hardware, S0 is sampled into the ALU before destination D
        is updated with the old EXEC.
        """
        state = ScalarHardwareState()
        initial_s4_5 = 0x1234_5678_9ABC_DEF0
        initial_exec = 0xF0E1_D2C3_B4A5_9687
        state.set_sgpr_pair(4, initial_s4_5)
        state.exec_mask = initial_exec

        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=4, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, state)

        # D receives old EXEC
        self.assertEqual(state.get_sgpr_pair(4), initial_exec)
        # EXEC receives S0 | old EXEC
        self.assertEqual(state.exec_mask, initial_s4_5 | initial_exec)
        self.assertEqual(state.scc, 1)

    def test_src0_is_exec(self):
        """s_or_saveexec_b64 s[6:7], exec: S0 is the EXEC mask itself.

        EXEC | EXEC = EXEC, so new EXEC equals old EXEC. D receives old EXEC.
        """
        state = ScalarHardwareState()
        exec_val = 0xCAFE_BABE_DEAD_BEEF
        state.exec_mask = exec_val

        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.EXEC, size_bits=64)
        )
        ScalarInterpreter.step(instr, state)

        self.assertEqual(state.get_sgpr_pair(6), exec_val)
        self.assertEqual(state.exec_mask, exec_val)
        self.assertEqual(state.scc, 1)

    def test_multiple_sgpr_pair_independence(self):
        """Ensure un-aliased SGPR pairs are completely untouched."""
        state = ScalarHardwareState()
        state.set_sgpr_pair(4, 0x1111_1111_1111_1111)
        state.set_sgpr_pair(8, 0x8888_8888_8888_8888)
        state.set_sgpr_pair(10, 0xAAAA_AAAA_AAAA_AAAA)
        state.exec_mask = 0x0F0F_0F0F_0F0F_0F0F

        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=4, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=8, size_bits=64)
        )
        ScalarInterpreter.step(instr, state)

        # s[4:5] updated with old EXEC
        self.assertEqual(state.get_sgpr_pair(4), 0x0F0F_0F0F_0F0F_0F0F)
        # s[8:9] (src0) preserved
        self.assertEqual(state.get_sgpr_pair(8), 0x8888_8888_8888_8888)
        # s[10:11] untouched
        self.assertEqual(state.get_sgpr_pair(10), 0xAAAA_AAAA_AAAA_AAAA)

# -----------------------------------------------------------------------------
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# -----------------------------------------------------------------------------

class TestSOrSaveExecConditionCodes(unittest.TestCase):
    """Unit tests for SCC (Scalar Condition Code) flag generation and transitions."""

    def test_scc_set_when_single_bit_nonzero(self):
        """SCC must be set to 1 if any bit in new EXEC is set."""
        for bit in (0, 1, 15, 16, 31, 32, 47, 48, 63):
            with self.subTest(bit=bit):
                mask = 1 << bit
                new_exec, _, scc = s_or_saveexec_b64_oracle(mask, 0)
                self.assertEqual(scc, 1)
                new_exec, _, scc = s_or_saveexec_b64_oracle(0, mask)
                self.assertEqual(scc, 1)

    def test_scc_cleared_when_both_zero(self):
        """SCC must be strictly 0 when new EXEC is completely 0."""
        new_exec, _, scc = s_or_saveexec_b64_oracle(0, 0)
        self.assertEqual(new_exec, 0)
        self.assertEqual(scc, 0)

    def test_scc_transitions_state_machine(self):
        """Test all 4 state transitions for SCC: 0->0, 0->1, 1->0, 1->1."""
        state = ScalarHardwareState()
        instr = SOrSaveExecB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )

        # 1. 0 -> 0 transition
        state.scc = 0
        state.set_sgpr_pair(4, 0)
        state.exec_mask = 0
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.scc, 0)

        # 2. 0 -> 1 transition
        state.set_sgpr_pair(4, 1)
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.scc, 1)

        # 3. 1 -> 1 transition
        state.set_sgpr_pair(4, 2)
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.scc, 1)

        # 4. 1 -> 0 transition
        state.set_sgpr_pair(4, 0)
        state.exec_mask = 0
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.scc, 0)

# -----------------------------------------------------------------------------
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# -----------------------------------------------------------------------------

class TestSOrSaveExecIEEE754Bitwise(unittest.TestCase):
    """Unit tests verifying that IEEE 754 float bit patterns are preserved as pure raw integers."""

    FLOAT64_PATTERNS = {
        "pos_zero": 0x0000_0000_0000_0000,
        "neg_zero": 0x8000_0000_0000_0000,
        "pos_inf": 0x7FF0_0000_0000_0000,
        "neg_inf": 0xFFF0_0000_0000_0000,
        "qnan": 0x7FF8_0000_0000_0000,
        "snan": 0x7FF0_0000_0000_0001,
        "min_subnormal": 0x0000_0000_0000_0001,
        "max_subnormal": 0x000F_FFFF_FFFF_FFFF,
        "min_normal": 0x0010_0000_0000_0000,
        "max_normal": 0x7FEF_FFFF_FFFF_FFFF,
        "one_point_zero": 0x3FF0_0000_0000_0000,
    }

    def test_float64_bit_exact_pass_through(self):
        """Verify SALU bitwise OR never canonicalizes or flushes float64 NaN or subnormals."""
        for name, pattern in self.FLOAT64_PATTERNS.items():
            with self.subTest(pattern_name=name):
                # OR with zero leaves bit pattern bit-exact
                new_exec, d, scc = s_or_saveexec_b64_oracle(pattern, 0)
                self.assertEqual(new_exec, pattern)
                self.assertEqual(d, 0)
                expected_scc = 1 if pattern != 0 else 0
                self.assertEqual(scc, expected_scc)

    def test_float64_nan_and_sign_bit_merging(self):
        """Verify combining sign bit with infinity produces negative infinity bit pattern."""
        neg_zero = self.FLOAT64_PATTERNS["neg_zero"]
        pos_inf = self.FLOAT64_PATTERNS["pos_inf"]
        new_exec, _, scc = s_or_saveexec_b64_oracle(neg_zero, pos_inf)
        self.assertEqual(new_exec, self.FLOAT64_PATTERNS["neg_inf"])
        self.assertEqual(scc, 1)

# -----------------------------------------------------------------------------
# Commit 8: Negative Controls, Syntax Rejection, and Mutation Detection
# -----------------------------------------------------------------------------

class TestSOrSaveExecNegativeAndMutation(unittest.TestCase):
    """Unit tests for negative syntax controls, bounds checking, and mutation detection."""

    def test_reject_odd_aligned_sgpr_destination(self):
        for odd_idx in (1, 3, 5, 7, 99, 101):
            with self.subTest(odd_idx=odd_idx):
                instr = SOrSaveExecB64Instruction(
                    dst=RegOperand(RegisterKind.SGPR, index=odd_idx, size_bits=64),
                    src0=RegOperand(RegisterKind.SGPR, index=0, size_bits=64)
                )
                with self.assertRaises(ValueError):
                    instr.validate()

    def test_reject_out_of_bounds_sgpr_destination(self):
        for oob_idx in (104, 106, 120, 256):
            with self.subTest(oob_idx=oob_idx):
                instr = SOrSaveExecB64Instruction(
                    dst=RegOperand(RegisterKind.SGPR, index=oob_idx, size_bits=64),
                    src0=RegOperand(RegisterKind.SGPR, index=0, size_bits=64)
                )
                with self.assertRaises(ValueError):
                    instr.validate()

    def test_reject_invalid_destination_register_kind(self):
        for bad_kind in (RegisterKind.VGPR, RegisterKind.EXEC, RegisterKind.VCC, RegisterKind.SCC):
            with self.subTest(bad_kind=bad_kind):
                instr = SOrSaveExecB64Instruction(
                    dst=RegOperand(bad_kind, index=0, size_bits=64),
                    src0=RegOperand(RegisterKind.SGPR, index=0, size_bits=64)
                )
                with self.assertRaises(TypeError):
                    instr.validate()

    def test_mutation_detector_logic(self):
        """Verify detection of mutant logic implementations."""
        def mutant_and(s0: int, e: int):
            return s0 & e, e, int((s0 & e) != 0)

        def mutant_overwrite_d(s0: int, e: int):
            new_e = s0 | e
            return new_e, new_e, int(new_e != 0)

        s0 = 0x00FF
        e = 0xFF00
        # Correct oracle
        correct_new, correct_d, correct_scc = s_or_saveexec_b64_oracle(s0, e)
        self.assertEqual(correct_new, 0xFFFF)
        self.assertEqual(correct_d, 0xFF00)
        self.assertEqual(correct_scc, 1)

        # Mutant AND fails
        m_new, m_d, m_scc = mutant_and(s0, e)
        self.assertNotEqual(m_new, correct_new)

        # Mutant overwrite D fails
        m_new2, m_d2, m_scc2 = mutant_overwrite_d(s0, e)
        self.assertNotEqual(m_d2, correct_d)


if __name__ == "__main__":
    unittest.main()
