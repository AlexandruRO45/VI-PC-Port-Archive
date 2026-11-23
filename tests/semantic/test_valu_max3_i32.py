from __future__ import annotations

"""Semantic regression test suite and reference verification for v_max3_i32.

RDNA2 Vector ALU instruction:
    v_max3_i32 vdst, vsrc0, vsrc1, vsrc2

Semantics:
    Signed two's complement 32-bit maximum across three operands:
        vdst = max(int32(vsrc0), int32(vsrc1), int32(vsrc2))

Reference mathematical oracle, properties, signed vs unsigned divergence,
AST definitions, deterministic scalar interpreter, SIMD multi-lane divergence,
register aliasing, condition codes invariance, two's complement and IEEE 754
bit pattern edge cases, and translator-backed SPIR-V execution.

Standard library Python only (python -I), zero game data.
"""

import itertools
import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32 = 0xFFFFFFFF
SIGN_BIT32 = 0x80000000
INT32_MIN = -0x80000000
INT32_MAX = 0x7FFFFFFF
UINT32_MAX = 0xFFFFFFFF


def to_signed_i32(x: int) -> int:
    """Interpret raw 32-bit integer as two's complement signed 32-bit integer."""
    x &= MASK32
    return x - (1 << 32) if (x & SIGN_BIT32) else x


def to_unsigned_u32(x: int) -> int:
    """Mask integer to 32-bit unsigned integer range [0, 2^32 - 1]."""
    return x & MASK32


def ref_v_max3_i32(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for v_max3_i32: signed maximum of 3 operands."""
    v0 = to_signed_i32(s0)
    v1 = to_signed_i32(s1)
    v2 = to_signed_i32(s2)
    return max(v0, v1, v2) & MASK32


def ref_pairwise_max3_i32(s0: int, s1: int, s2: int) -> int:
    """Cascaded pairwise signed maximum: max(max(s0, s1), s2)."""
    step1 = max(to_signed_i32(s0), to_signed_i32(s1))
    return max(step1, to_signed_i32(s2)) & MASK32

class TestVMax3I32ReferenceOracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and core properties for v_max3_i32."""

    def test_basic_signed_max(self) -> None:
        """Verify standard signed maximum calculations with positive and negative inputs."""
        self.assertEqual(ref_v_max3_i32(10, 20, 30), 30)
        self.assertEqual(ref_v_max3_i32(-10, -20, -5), (-5) & MASK32)
        self.assertEqual(ref_v_max3_i32(-1, 0, 1), 1)
        self.assertEqual(ref_v_max3_i32(-100, 50, -200), 50)
        self.assertEqual(ref_v_max3_i32(0, 0, 0), 0)

    def test_signed_vs_unsigned_divergence(self) -> None:
        """Verify divergence between signed maximum and unsigned maximum on identical bit patterns."""
        # 0xFFFFFFFF is -1 as i32, UINT32_MAX as u32.
        # With 0 and 10: signed max is 10 (-1 < 0 < 10); unsigned max is 0xFFFFFFFF.
        self.assertEqual(ref_v_max3_i32(0, 0xFFFFFFFF, 10), 10)
        self.assertNotEqual(ref_v_max3_i32(0, 0xFFFFFFFF, 10), max(0, 0xFFFFFFFF, 10))

        # 0x80000000 is INT32_MIN as i32, 2^31 as u32.
        # With 0x7FFFFFFF (INT32_MAX) and 0:
        self.assertEqual(ref_v_max3_i32(0x7FFFFFFF, 0x80000000, 0), 0x7FFFFFFF)
        # Unsigned max would pick 0x80000000 (2147483648 > 2147483647).
        self.assertNotEqual(ref_v_max3_i32(0x7FFFFFFF, 0x80000000, 0), max(0x7FFFFFFF, 0x80000000, 0))

    def test_pairwise_reduction_equivalence(self) -> None:
        """Verify 3-way signed max identically matches chained pairwise signed max."""
        samples = [
            (0, 0, 0),
            (1, 2, 3),
            (-5, 10, -20),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
            (0x80000000, 0x80000001, 0x7FFFFFFE),
            (0x12345678, 0x87654321, 0x00000000),
            (0xFFFFFFFE, 0xFFFFFFFF, 0x00000001),
        ]
        for s0, s1, s2 in samples:
            with self.subTest(s0=hex(s0 & MASK32), s1=hex(s1 & MASK32), s2=hex(s2 & MASK32)):
                self.assertEqual(ref_v_max3_i32(s0, s1, s2), ref_pairwise_max3_i32(s0, s1, s2))

    def test_idempotent_and_identity_properties(self) -> None:
        """Verify idempotent max(x, x, x) == x and neutral/absorbing element laws."""
        test_values = [0, 1, -1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 42, -9999]
        for val in test_values:
            raw = val & MASK32
            # Idempotence: max(x, x, x) == x
            self.assertEqual(ref_v_max3_i32(val, val, val), raw)
            # Neutral element: INT32_MIN (0x80000000)
            self.assertEqual(ref_v_max3_i32(val, 0x80000000, 0x80000000), raw)
            # Absorbing element: INT32_MAX (0x7FFFFFFF)
            self.assertEqual(ref_v_max3_i32(val, 0x7FFFFFFF, 0), 0x7FFFFFFF)

    def test_permutation_invariance(self) -> None:
        """Verify that all 3! = 6 permutations yield identical results."""
        triples = [
            (0x80000000, 0, 0x7FFFFFFF),
            (0xFFFFFFFF, 1, 0x80000000),
            (42, 42, 42),
            (0x12345678, 0x9ABCDEF0, 0x00000000),
            (0x80000000, 0x80000001, 0x80000002),
            (0xFFFFFFFD, 0xFFFFFFFE, 0xFFFFFFFF),
        ]
        for triple in triples:
            expected = ref_v_max3_i32(*triple)
            for perm in itertools.permutations(triple):
                self.assertEqual(ref_v_max3_i32(*perm), expected)

# ==============================================================================
# SECTION 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    IMMEDIATE = auto()


@dataclass(frozen=True)
class Operand:
    kind: OperandKind
    value: int

    def __post_init__(self) -> None:
        if self.kind == OperandKind.VGPR:
            if not (0 <= self.value <= 255):
                raise ValueError(f"VGPR index out of range [0, 255]: {self.value}")
        elif self.kind == OperandKind.SGPR:
            if not (0 <= self.value <= 105):
                raise ValueError(f"SGPR index out of range [0, 105]: {self.value}")
        elif self.kind == OperandKind.IMMEDIATE:
            if not (-0x80000000 <= self.value <= 0xFFFFFFFF):
                raise ValueError(f"Immediate value out of 32-bit range: {self.value}")


def VGPR(idx: int) -> Operand:
    return Operand(OperandKind.VGPR, idx)


def SGPR(idx: int) -> Operand:
    return Operand(OperandKind.SGPR, idx)


def Immediate(val: int) -> Operand:
    return Operand(OperandKind.IMMEDIATE, val)


@dataclass(frozen=True)
class VMax3I32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self) -> None:
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.kind}")

    def format_operand(self, op: Operand) -> str:
        if op.kind == OperandKind.VGPR:
            return f"v{op.value}"
        elif op.kind == OperandKind.SGPR:
            return f"s{op.value}"
        else:
            return f"{op.value}"

    def to_assembly(self) -> str:
        d = self.format_operand(self.dst)
        s0 = self.format_operand(self.src0)
        s1 = self.format_operand(self.src1)
        s2 = self.format_operand(self.src2)
        return f"v_max3_i32 {d}, {s0}, {s1}, {s2}"


class TestVMax3I32AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        """Verify AST correctly formats to valid RDNA2 assembly text."""
        inst = VMax3I32Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=SGPR(4),
            src2=Immediate(-16),
        )
        asm = inst.to_assembly()
        self.assertEqual(asm, "v_max3_i32 v0, v1, s4, -16")

    def test_ast_operand_bounds_validation(self) -> None:
        """Verify AST rejects out-of-range registers and invalid destinations."""
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            VGPR(-1)
        with self.assertRaises(ValueError):
            SGPR(106)
        with self.assertRaises(ValueError):
            SGPR(-1)
        with self.assertRaises(ValueError):
            Immediate(0x1_0000_0000)
        with self.assertRaises(TypeError):
            VMax3I32Instruction(dst=SGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))

    def test_ast_assembly_execution_structure(self) -> None:
        """Verify assembly generated from AST has the expected syntax."""
        inst = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))
        self.assertTrue(inst.to_assembly().startswith("v_max3_i32 v0, v1, v2, v3"))

# ==============================================================================
# SECTION 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    vgprs: Dict[int, int] = field(default_factory=dict)
    sgprs: Dict[int, int] = field(default_factory=dict)

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs.get(idx, 0) & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0) & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32


class ScalarInterpreter:
    def resolve_operand(self, op: Operand, state: ScalarMachineState) -> int:
        if op.kind == OperandKind.VGPR:
            return state.get_vgpr(op.value)
        elif op.kind == OperandKind.SGPR:
            return state.get_sgpr(op.value)
        elif op.kind == OperandKind.IMMEDIATE:
            return op.value & MASK32
        raise ValueError(f"Unknown operand kind: {op.kind}")

    def step(self, instr: VMax3I32Instruction, state: ScalarMachineState) -> int:
        s0 = self.resolve_operand(instr.src0, state)
        s1 = self.resolve_operand(instr.src1, state)
        s2 = self.resolve_operand(instr.src2, state)
        res = ref_v_max3_i32(s0, s1, s2)
        state.set_vgpr(instr.dst.value, res)
        return res


class TestVMax3I32ScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_basic_step(self) -> None:
        """Verify scalar step evaluates signed maximum across VGPRs, SGPRs, and immediates."""
        state = ScalarMachineState()
        state.set_vgpr(1, (-50) & MASK32)
        state.set_sgpr(0, 100)

        interp = ScalarInterpreter()
        instr = VMax3I32Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=SGPR(0),
            src2=Immediate(25),
        )
        res = interp.step(instr, state)
        self.assertEqual(res, 100)
        self.assertEqual(state.get_vgpr(0), 100)

    def test_scalar_interpreter_negative_maximum(self) -> None:
        """Verify scalar step correctly identifies maximum when all operands are negative."""
        state = ScalarMachineState()
        state.set_vgpr(1, (-100) & MASK32)
        state.set_vgpr(2, (-20) & MASK32)
        state.set_vgpr(3, (-50) & MASK32)

        interp = ScalarInterpreter()
        instr = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))
        res = interp.step(instr, state)
        self.assertEqual(res, (-20) & MASK32)
        self.assertEqual(state.get_vgpr(0), (-20) & MASK32)

    def test_scalar_interpreter_state_isolation(self) -> None:
        """Verify only the destination register is modified by step."""
        state = ScalarMachineState()
        state.set_vgpr(0, 0xDEADBEEF)
        state.set_vgpr(1, 10)
        state.set_vgpr(2, 20)
        state.set_vgpr(3, 30)
        state.set_vgpr(4, 0xCAFEBABE)

        interp = ScalarInterpreter()
        instr = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))
        interp.step(instr, state)

        self.assertEqual(state.get_vgpr(0), 30)
        self.assertEqual(state.get_vgpr(1), 10)
        self.assertEqual(state.get_vgpr(2), 20)
        self.assertEqual(state.get_vgpr(3), 30)
        self.assertEqual(state.get_vgpr(4), 0xCAFEBABE)

# ==============================================================================
# SECTION 4: SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    num_lanes: int = 64
    exec_mask: int = 0xFFFFFFFFFFFFFFFF
    vgprs: List[Dict[int, int]] = field(default_factory=list)
    sgprs: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [dict() for _ in range(self.num_lanes)]

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, idx: int) -> int:
        return self.vgprs[lane].get(idx, 0) & MASK32

    def set_vgpr(self, lane: int, idx: int, val: int) -> None:
        self.vgprs[lane][idx] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0) & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32


class SIMDInterpreter:
    def resolve_lane_operand(self, op: Operand, lane: int, state: SIMDState) -> int:
        if op.kind == OperandKind.VGPR:
            return state.get_vgpr(lane, op.value)
        elif op.kind == OperandKind.SGPR:
            return state.get_sgpr(op.value)
        elif op.kind == OperandKind.IMMEDIATE:
            return op.value & MASK32
        raise ValueError(f"Unknown operand kind: {op.kind}")

    def step(self, instr: VMax3I32Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = []
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                results.append(None)
                continue
            s0 = self.resolve_lane_operand(instr.src0, lane, state)
            s1 = self.resolve_lane_operand(instr.src1, lane, state)
            s2 = self.resolve_lane_operand(instr.src2, lane, state)
            res = ref_v_max3_i32(s0, s1, s2)
            state.set_vgpr(lane, instr.dst.value, res)
            results.append(res)
        return results


class TestVMax3I32SIMD(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_alternating_exec_mask(self) -> None:
        """Verify Wave32 execution: inactive lanes strictly preserve prior contents."""
        even_mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=even_mask)
        sentinel = 0xDEADBEEF

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, (-10 + lane) & MASK32)
            state.set_vgpr(lane, 2, 0)
            state.set_vgpr(lane, 3, (10 - lane) & MASK32)

        simd = SIMDInterpreter()
        instr = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))
        results = simd.step(instr, state)

        for lane in range(32):
            if lane % 2 == 0:
                expected = max(-10 + lane, 0, 10 - lane) & MASK32
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_wave64_divergence_and_scalar_broadcast(self) -> None:
        """Verify Wave64 with SGPR uniform broadcast across 64 divergent lanes."""
        state = SIMDState(num_lanes=64, exec_mask=0xFFFFFFFFFFFFFFFF)
        state.set_sgpr(0, 0)  # Uniform threshold 0

        for lane in range(64):
            state.set_vgpr(lane, 1, (lane - 32) & MASK32)
            state.set_vgpr(lane, 2, (16 - lane) & MASK32)

        simd = SIMDInterpreter()
        instr = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=SGPR(0))
        results = simd.step(instr, state)

        for lane in range(64):
            s1 = lane - 32
            s2 = 16 - lane
            expected = max(s1, s2, 0) & MASK32
            self.assertEqual(results[lane], expected)
            self.assertEqual(state.get_vgpr(lane, 0), expected)

    def test_sparse_exec_mask_boundary_lanes(self) -> None:
        """Verify execution on sparse EXEC mask containing first, middle, and last lanes."""
        mask = (1 << 0) | (1 << 31) | (1 << 32) | (1 << 63)
        state = SIMDState(num_lanes=64, exec_mask=mask)
        sentinel = 0xCAFEBABE

        for lane in range(64):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, 100)
            state.set_vgpr(lane, 2, 200)
            state.set_vgpr(lane, 3, 300)

        simd = SIMDInterpreter()
        instr = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))
        results = simd.step(instr, state)

        for lane in range(64):
            if lane in (0, 31, 32, 63):
                self.assertEqual(results[lane], 300)
                self.assertEqual(state.get_vgpr(lane, 0), 300)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

# ==============================================================================
# SECTION 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestVMax3I32RegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0(self) -> None:
        """Verify in-place update when dst is src0: v0 = max3(v0, v1, v2)."""
        state = ScalarMachineState()
        state.set_vgpr(0, (-50) & MASK32)
        state.set_vgpr(1, 10)
        state.set_vgpr(2, 5)

        interp = ScalarInterpreter()
        instr = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(0), src1=VGPR(1), src2=VGPR(2))
        res = interp.step(instr, state)

        self.assertEqual(res, 10)
        self.assertEqual(state.get_vgpr(0), 10)

    def test_alias_dst_equals_src1(self) -> None:
        """Verify in-place update when dst is src1: v0 = max3(v1, v0, v2)."""
        state = ScalarMachineState()
        state.set_vgpr(0, 100)
        state.set_vgpr(1, (-10) & MASK32)
        state.set_vgpr(2, 50)

        interp = ScalarInterpreter()
        instr = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(0), src2=VGPR(2))
        res = interp.step(instr, state)

        self.assertEqual(res, 100)
        self.assertEqual(state.get_vgpr(0), 100)

    def test_alias_dst_equals_src2(self) -> None:
        """Verify in-place update when dst is src2: v0 = max3(v1, v2, v0)."""
        state = ScalarMachineState()
        state.set_vgpr(0, 75)
        state.set_vgpr(1, 10)
        state.set_vgpr(2, 20)

        interp = ScalarInterpreter()
        instr = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(0))
        res = interp.step(instr, state)

        self.assertEqual(res, 75)
        self.assertEqual(state.get_vgpr(0), 75)

    def test_alias_all_registers_same(self) -> None:
        """Verify idempotent in-place update when dst and all sources share one register: v0 = max3(v0, v0, v0)."""
        state = ScalarMachineState()
        state.set_vgpr(0, (-999) & MASK32)

        interp = ScalarInterpreter()
        instr = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(0), src1=VGPR(0), src2=VGPR(0))
        res = interp.step(instr, state)

        self.assertEqual(res, (-999) & MASK32)
        self.assertEqual(state.get_vgpr(0), (-999) & MASK32)

    def test_alias_two_sources_same(self) -> None:
        """Verify behavior when two sources share a register: v0 = max3(v1, v1, v2)."""
        state = ScalarMachineState()
        state.set_vgpr(1, (-20) & MASK32)
        state.set_vgpr(2, (-5) & MASK32)

        interp = ScalarInterpreter()
        instr = VMax3I32Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(1), src2=VGPR(2))
        res = interp.step(instr, state)

        self.assertEqual(res, (-5) & MASK32)
        self.assertEqual(state.get_vgpr(0), (-5) & MASK32)

# ==============================================================================
# SECTION 6: Condition Codes, Status Flags, and Carry/Borrow Invariance
# ==============================================================================

class TestVMax3I32ConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow invariance."""

    def test_vcc_and_scc_invariance(self) -> None:
        """Verify v_max3_i32 does not write or modify VCC or SCC condition codes."""
        lines = [
            "v_cmp_eq_u32 vcc, 42, 42",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 30",
            "v_max3_i32 v0, v1, v2, v3",
            "v_cndmask_b32 v4, 0x11111111, 0x22222222, vcc",
            "exp mrt0 v4, v0, v4, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_i32_vcc_invar", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x22222222)
        self.assertEqual(words[1], 30)

    def test_no_carry_borrow_generation(self) -> None:
        """Verify integer comparison/selection has no arithmetic carry/borrow side-effects."""
        lines = [
            "v_mov_b32 v1, 0x7FFFFFFF",
            "v_mov_b32 v2, 0x80000000",
            "v_mov_b32 v3, 0",
            "v_max3_i32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_i32_nocarry", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x7FFFFFFF)

# ==============================================================================
# SECTION 7: IEEE 754 Boundary Extremes and Two's Complement Edge Cases
# ==============================================================================

class TestVMax3I32IEEE754EdgeCases(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes and two's complement edge cases."""

    def test_twos_complement_extrema_matrix(self) -> None:
        """Verify behavior at and around two's complement signed 32-bit boundaries."""
        extrema = (
            0x80000000,  # INT32_MIN (-2147483648)
            0x80000001,  # INT32_MIN + 1
            0xFFFFFFFF,  # -1 as i32
            0x00000000,  # 0
            0x00000001,  # 1
            0x7FFFFFFE,  # INT32_MAX - 1
            0x7FFFFFFF,  # INT32_MAX (2147483647)
        )
        for x in extrema:
            for y in extrema:
                self.assertEqual(ref_v_max3_i32(0x7FFFFFFF, x, y), 0x7FFFFFFF)

        self.assertEqual(ref_v_max3_i32(0x80000000, 0x80000000, 0x80000000), 0x80000000)
        self.assertEqual(ref_v_max3_i32(0x80000000, 0x80000001, 0xFFFFFFFF), 0xFFFFFFFF)

    def test_ieee754_reinterpreted_as_signed_i32(self) -> None:
        """Verify raw IEEE 754 float bit patterns are evaluated strictly as two's complement integers."""
        # IEEE 754 +0.0 is 0x00000000; -0.0 is 0x80000000 (INT32_MIN)
        self.assertEqual(ref_v_max3_i32(0x00000000, 0x80000000, 0x80000000), 0x00000000)

        # IEEE 754 +Inf (0x7F800000) vs -Inf (0xFF800000)
        self.assertEqual(ref_v_max3_i32(0x7F800000, 0xFF800000, 0), 0x7F800000)

        # IEEE 754 qNaN (0x7FC00000) vs -qNaN (0xFFC00000)
        self.assertEqual(ref_v_max3_i32(0x7FC00000, 0xFFC00000, 0), 0x7FC00000)

    def test_distinct_signed_unsigned_quad_triple(self) -> None:
        """On (0x80000000, 0x00000000, 0xFFFFFFFF), signed max is 0, unsigned max is 0xFFFFFFFF."""
        s0, s1, s2 = 0x80000000, 0x00000000, 0xFFFFFFFF
        signed_max = ref_v_max3_i32(s0, s1, s2)
        unsigned_max = max(s0, s1, s2)
        self.assertEqual(signed_max, 0x00000000)
        self.assertEqual(unsigned_max, 0xFFFFFFFF)
        self.assertNotEqual(signed_max, unsigned_max)

if __name__ == "__main__":
    unittest.main()
