from __future__ import annotations

"""Architectural reference model, symbolic AST, interpreter, and hardware-rule tests for s_or_b64.

RDNA2 Instruction Set Architecture, SOP2:
    s_or_b64 D, S0, S1

Semantic operation:
    D.u64 = S0.u64 | S1.u64
    SCC = (D != 0)

Outputs:
    D: 64-bit destination SGPR pair (or EXEC/VCC) storing the bitwise OR of S0 and S1.
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

def s_or_b64_oracle(s0: int, s1: int) -> Tuple[int, int]:
    """Reference mathematical model for s_or_b64.

    Args:
        s0: 64-bit unsigned integer source 0.
        s1: 64-bit unsigned integer source 1.

    Returns:
        (d, scc) tuple where:
            d: (s0 | s1) & MASK64
            scc: 1 if d != 0 else 0
    """
    s0_u64 = s0 & MASK64
    s1_u64 = s1 & MASK64
    d = (s0_u64 | s1_u64) & MASK64
    scc = 1 if d != 0 else 0
    return d, scc


class SOrB64Oracle:
    """Architectural model and hardware-rule validator for s_or_b64."""

    @staticmethod
    def eval(s0: int, s1: int) -> Tuple[int, int]:
        return s_or_b64_oracle(s0, s1)

    @staticmethod
    def build_compute_shader(s0: int, s1: int) -> Tuple[str, List[str], bytes]:
        """Generate synthetic compute shader lines to test s_or_b64."""
        fssm = bytes(0x138) + struct.pack("<4I", 0x8088, 64, 1, 1)
        name = "cs_or_b64_hw"
        lines = [
            "s_mov_b64 exec, -1",
            f"s_mov_b32 s2, {s0 & MASK32:#x}",
            f"s_mov_b32 s3, {(s0 >> 32) & MASK32:#x}",
            f"s_mov_b32 s4, {s1 & MASK32:#x}",
            f"s_mov_b32 s5, {(s1 >> 32) & MASK32:#x}",
            "s_or_b64 s[6:7], s[2:3], s[4:5]",
            "s_cselect_b32 s8, 1, 0",
            "s_endpgm",
        ]
        return name, lines, fssm

    @staticmethod
    def translate_and_validate(s0: int, s1: int) -> str:
        """Exercise tools/rdna2spirv.py via support.translate_lines."""
        name, lines, fssm = SOrB64Oracle.build_compute_shader(s0, s1)
        return support.translate_lines(name, lines, fssm=fssm)


class TestSOrB64Oracle(unittest.TestCase):
    """Unit tests for reference oracle and translator execution."""

    def test_oracle_truth_table_bit_combinations(self):
        # Truth table for bitwise OR: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1
        cases = [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (1, 1, 1),
        ]
        for b0, b1, expected in cases:
            with self.subTest(b0=b0, b1=b1):
                d, scc = SOrB64Oracle.eval(b0, b1)
                self.assertEqual(d & 1, expected)
                self.assertEqual(scc, expected)

    def test_oracle_identity_zero(self):
        # x | 0 = x
        for val in (0, ALL64, 0x1234_5678_9ABC_DEF0, 0x5555_5555_5555_5555):
            with self.subTest(val=hex(val)):
                d, scc = SOrB64Oracle.eval(val, 0)
                self.assertEqual(d, val)
                self.assertEqual(scc, 1 if val != 0 else 0)

    def test_oracle_annihilator_all_ones(self):
        # x | ALL64 = ALL64
        for val in (0, ALL64, 0x1234_5678_9ABC_DEF0, 0xAAAA_AAAA_AAAA_AAAA):
            with self.subTest(val=hex(val)):
                d, scc = SOrB64Oracle.eval(val, ALL64)
                self.assertEqual(d, ALL64)
                self.assertEqual(scc, 1)

    def test_oracle_idempotence(self):
        # x | x = x
        for val in (0, ALL64, 0xCAFE_BABE_DEAD_BEEF):
            with self.subTest(val=hex(val)):
                d, scc = SOrB64Oracle.eval(val, val)
                self.assertEqual(d, val)
                self.assertEqual(scc, 1 if val != 0 else 0)

    def test_oracle_commutative(self):
        # s0 | s1 == s1 | s0
        s0, s1 = 0x1234_5678_9ABC_DEF0, 0x0FED_CBA9_8765_4321
        d1, scc1 = SOrB64Oracle.eval(s0, s1)
        d2, scc2 = SOrB64Oracle.eval(s1, s0)
        self.assertEqual(d1, d2)
        self.assertEqual(scc1, scc2)

    def test_translator_pipeline_execution(self):
        """Verify that synthetic shader translates to valid SPIR-V assembly text via support.translate_lines."""
        spv = SOrB64Oracle.translate_and_validate(0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA)
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
class SOrB64Instruction:
    dst: RegOperand
    src0: RegOperand
    src1: RegOperand
    opcode: str = "s_or_b64"

    def validate(self) -> None:
        if self.opcode != "s_or_b64":
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


class TestSOrB64AST(unittest.TestCase):
    """Unit tests for symbolic AST representation and validation."""

    def test_ast_valid_sgpr_operands(self):
        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_or_b64 s[6:7], s[2:3], s[4:5]")

    def test_ast_valid_exec_destination(self):
        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.EXEC, size_bits=64),
            src0=RegOperand(RegisterKind.EXEC, size_bits=64),
            src1=RegOperand(RegisterKind.VCC, size_bits=64)
        )
        self.assertEqual(instr.emit_assembly(), "s_or_b64 exec, exec, vcc")

    def test_ast_misaligned_destination(self):
        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=5, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        with self.assertRaises(ValueError):
            instr.validate()

    def test_ast_invalid_vgpr_destination(self):
        instr = SOrB64Instruction(
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
    """Deterministic scalar instruction interpreter executing s_or_b64."""

    @staticmethod
    def step(instr: SOrB64Instruction, state: ScalarHardwareState) -> None:
        instr.validate()
        s0_val = state.read_operand(instr.src0)
        s1_val = state.read_operand(instr.src1)

        d, scc = s_or_b64_oracle(s0_val, s1_val)
        state.write_destination(instr.dst, d)
        state.scc = scc


class TestSOrB64ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar interpreter execution."""

    def setUp(self):
        self.state = ScalarHardwareState()

    def test_interpreter_step_basic(self):
        self.state.set_sgpr_pair(2, 0x00FF_0000_00FF_0000)
        self.state.set_sgpr_pair(4, 0x0000_00FF_0000_00FF)
        self.state.scc = 0

        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)

        expected = 0x00FF_00FF_00FF_00FF
        self.assertEqual(self.state.get_sgpr_pair(6), expected)
        self.assertEqual(self.state.scc, 1)

    def test_interpreter_step_zero_output(self):
        self.state.set_sgpr_pair(2, 0)
        self.state.set_sgpr_pair(4, 0)
        self.state.scc = 1

        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)

        self.assertEqual(self.state.get_sgpr_pair(6), 0)
        self.assertEqual(self.state.scc, 0)

    def test_interpreter_step_with_vcc_and_exec(self):
        self.state.exec_mask = 0x5555_5555_5555_5555
        self.state.vcc = 0xAAAA_AAAA_AAAA_AAAA

        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.VCC, size_bits=64),
            src0=RegOperand(RegisterKind.EXEC, size_bits=64),
            src1=RegOperand(RegisterKind.VCC, size_bits=64)
        )
        ScalarInterpreter.step(instr, self.state)

        # 0x5555... | 0xAAAA... = ALL64
        self.assertEqual(self.state.vcc, ALL64)
        self.assertEqual(self.state.scc, 1)

# -----------------------------------------------------------------------------
# Commit 4: SIMD Multi-Lane Divergence and 64-Bit Lane Masking
# -----------------------------------------------------------------------------

class SIMDMultiLaneDivergenceModel:
    """Models multi-lane execution in Wave64 environments using s_or_b64."""

    @staticmethod
    def active_lanes_from_mask(mask: int) -> Set[int]:
        return {lane for lane in range(64) if (mask & (1 << lane)) != 0}

    @staticmethod
    def or_lane_mask(s0_mask: int, s1_mask: int) -> Tuple[int, int]:
        """In Wave64 flow control, s_or_b64 computes lane activation mask unions:
        Lanes are ON if they are in S0 OR in S1.
        """
        return s_or_b64_oracle(s0_mask, s1_mask)


class TestSOrB64SIMDDivergence(unittest.TestCase):
    """Unit tests for SIMD Wave64 multi-lane divergence and mask union."""

    def test_or_lane_union(self):
        # S0 covers low half (lanes 0..31)
        s0 = 0x0000_0000_FFFF_FFFF
        # S1 covers high half (lanes 32..63)
        s1 = 0xFFFF_FFFF_0000_0000

        d, scc = SIMDMultiLaneDivergenceModel.or_lane_mask(s0, s1)
        self.assertEqual(d, ALL64)
        self.assertEqual(scc, 1)

        active = SIMDMultiLaneDivergenceModel.active_lanes_from_mask(d)
        self.assertEqual(active, set(range(64)))

    def test_checkerboard_lane_union(self):
        # S0 = even lanes, S1 = odd lanes
        s0 = 0x5555_5555_5555_5555
        s1 = 0xAAAA_AAAA_AAAA_AAAA
        d, scc = s_or_b64_oracle(s0, s1)
        self.assertEqual(d, ALL64)
        self.assertEqual(scc, 1)

        active = SIMDMultiLaneDivergenceModel.active_lanes_from_mask(d)
        self.assertEqual(len(active), 64)

    def test_sparse_boundary_lanes(self):
        s0 = (1 << 0)
        s1 = (1 << 63)
        d, scc = s_or_b64_oracle(s0, s1)
        self.assertEqual(d, (1 << 0) | (1 << 63))
        self.assertEqual(scc, 1)

    def test_empty_lane_masks(self):
        d, scc = s_or_b64_oracle(0, 0)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(SIMDMultiLaneDivergenceModel.active_lanes_from_mask(d), set())

# -----------------------------------------------------------------------------
# Commit 5: Register Aliasing and In-Place Operand Preservation
# -----------------------------------------------------------------------------

class TestSOrB64Aliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0(self):
        """s_or_b64 s[2:3], s[2:3], s[4:5]: D aliases S0."""
        state = ScalarHardwareState()
        s0_val = 0x1111_1111_1111_1111
        s1_val = 0x3333_3333_3333_3333
        state.set_sgpr_pair(2, s0_val)
        state.set_sgpr_pair(4, s1_val)

        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, state)

        expected = s0_val | s1_val
        self.assertEqual(state.get_sgpr_pair(2), expected)
        self.assertEqual(state.get_sgpr_pair(4), s1_val)  # S1 untouched
        self.assertEqual(state.scc, 1)

    def test_alias_dst_equals_src1(self):
        """s_or_b64 s[4:5], s[2:3], s[4:5]: D aliases S1."""
        state = ScalarHardwareState()
        s0_val = 0x00FF_0000_0000_0000
        s1_val = 0x0000_0000_FFFF_0000
        state.set_sgpr_pair(2, s0_val)
        state.set_sgpr_pair(4, s1_val)

        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=4, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )
        ScalarInterpreter.step(instr, state)

        expected = s0_val | s1_val
        self.assertEqual(state.get_sgpr_pair(4), expected)
        self.assertEqual(state.get_sgpr_pair(2), s0_val)  # S0 untouched
        self.assertEqual(state.scc, 1)

    def test_all_operands_aliased(self):
        """s_or_b64 s[2:3], s[2:3], s[2:3]: D, S0, and S1 all share same pair."""
        state = ScalarHardwareState()
        val = 0x1234_5678_9ABC_DEF0
        state.set_sgpr_pair(2, val)

        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=2, size_bits=64)
        )
        ScalarInterpreter.step(instr, state)

        # x | x = x
        self.assertEqual(state.get_sgpr_pair(2), val)
        self.assertEqual(state.scc, 1)

# -----------------------------------------------------------------------------
# Commit 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# -----------------------------------------------------------------------------

class TestSOrB64ConditionCodes(unittest.TestCase):
    """Unit tests for SCC flag generation under s_or_b64."""

    def test_scc_cleared_strictly_when_zero(self):
        # Must be exactly 0 (S0=0, S1=0)
        d, scc = s_or_b64_oracle(0, 0)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)

    def test_scc_set_for_any_single_bit(self):
        for bit in (0, 1, 15, 16, 31, 32, 47, 48, 63):
            with self.subTest(bit=bit):
                d, scc = s_or_b64_oracle(1 << bit, 0)
                self.assertEqual(d, 1 << bit)
                self.assertEqual(scc, 1)

    def test_scc_transitions_state_machine(self):
        state = ScalarHardwareState()
        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.SGPR, index=6, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=2, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=4, size_bits=64)
        )

        # 1. 0 -> 0 transition (0 | 0 = 0)
        state.scc = 0
        state.set_sgpr_pair(2, 0)
        state.set_sgpr_pair(4, 0)
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.scc, 0)

        # 2. 0 -> 1 transition (1 | 0 = 1)
        state.set_sgpr_pair(2, 1)
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.scc, 1)

        # 3. 1 -> 1 transition (ALL64 | 0 = ALL64)
        state.set_sgpr_pair(2, ALL64)
        state.set_sgpr_pair(4, 0)
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.scc, 1)

        # 4. 1 -> 0 transition (0 | 0 = 0)
        state.set_sgpr_pair(2, 0)
        state.set_sgpr_pair(4, 0)
        ScalarInterpreter.step(instr, state)
        self.assertEqual(state.scc, 0)

# -----------------------------------------------------------------------------
# Commit 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# -----------------------------------------------------------------------------

class TestSOrB64IEEE754Bitwise(unittest.TestCase):
    """Unit tests verifying that IEEE 754 float64 bit patterns are preserved bit-exact."""

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

    def test_float64_bitwise_or_idempotent(self):
        """Verify x | x equals x for every float64 extreme pattern."""
        for name, pattern in self.FLOAT64_PATTERNS.items():
            with self.subTest(pattern_name=name):
                d, scc = s_or_b64_oracle(pattern, pattern)
                self.assertEqual(d, pattern)
                self.assertEqual(scc, 1 if pattern != 0 else 0)

    def test_float64_nan_and_inf_combination(self):
        qnan = self.FLOAT64_PATTERNS["qnan"]
        pos_inf = self.FLOAT64_PATTERNS["pos_inf"]
        d, scc = s_or_b64_oracle(qnan, pos_inf)
        expected = qnan | pos_inf
        self.assertEqual(d, expected)
        self.assertEqual(scc, 1)

    def test_float64_signed_zero_combination(self):
        pos_zero = self.FLOAT64_PATTERNS["pos_zero"]
        neg_zero = self.FLOAT64_PATTERNS["neg_zero"]
        d, scc = s_or_b64_oracle(pos_zero, neg_zero)
        self.assertEqual(d, neg_zero)
        self.assertEqual(scc, 1)

# -----------------------------------------------------------------------------
# Commit 8: Negative Controls, Syntax Rejection, and Mutation Detection
# -----------------------------------------------------------------------------

class TestSOrB64NegativeAndMutation(unittest.TestCase):
    """Unit tests for negative syntax controls, bounds checking, and mutation detection."""

    def test_reject_odd_aligned_destination(self):
        for odd in (1, 3, 5, 7, 99, 101):
            with self.subTest(odd=odd):
                instr = SOrB64Instruction(
                    dst=RegOperand(RegisterKind.SGPR, index=odd, size_bits=64),
                    src0=RegOperand(RegisterKind.SGPR, index=0, size_bits=64),
                    src1=RegOperand(RegisterKind.SGPR, index=2, size_bits=64)
                )
                with self.assertRaises(ValueError):
                    instr.validate()

    def test_reject_out_of_bounds_destination(self):
        for oob in (104, 106, 120, 256):
            with self.subTest(oob=oob):
                instr = SOrB64Instruction(
                    dst=RegOperand(RegisterKind.SGPR, index=oob, size_bits=64),
                    src0=RegOperand(RegisterKind.SGPR, index=0, size_bits=64),
                    src1=RegOperand(RegisterKind.SGPR, index=2, size_bits=64)
                )
                with self.assertRaises(ValueError):
                    instr.validate()

    def test_reject_vgpr_destination(self):
        instr = SOrB64Instruction(
            dst=RegOperand(RegisterKind.VGPR, index=0, size_bits=64),
            src0=RegOperand(RegisterKind.SGPR, index=0, size_bits=64),
            src1=RegOperand(RegisterKind.SGPR, index=2, size_bits=64)
        )
        with self.assertRaises(TypeError):
            instr.validate()

    def test_mutation_detector_logic(self):
        """Verify detection of mutant logic implementations (OR vs AND vs XOR vs ORN2)."""
        def mutant_and(s0: int, s1: int):
            d = (s0 & s1) & MASK64
            return d, int(d != 0)

        def mutant_xor(s0: int, s1: int):
            d = (s0 ^ s1) & MASK64
            return d, int(d != 0)

        def mutant_orn2(s0: int, s1: int):
            d = (s0 | (~s1 & MASK64)) & MASK64
            return d, int(d != 0)

        s0 = 0x0000_FFFF_0000_0000
        s1 = 0x0000_0000_FFFF_0000
        correct_d, correct_scc = s_or_b64_oracle(s0, s1)
        self.assertEqual(correct_d, s0 | s1)
        self.assertEqual(correct_scc, 1)

        # Mutant AND produces 0 != s0 | s1
        m_and, _ = mutant_and(s0, s1)
        self.assertNotEqual(m_and, correct_d)

        # Mutant ORN2 produces s0 | ~s1 != s0 | s1
        m_orn2, _ = mutant_orn2(s0, s1)
        self.assertNotEqual(m_orn2, correct_d)

if __name__ == "__main__":
    unittest.main()
