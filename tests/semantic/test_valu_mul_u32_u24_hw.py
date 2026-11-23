from __future__ import annotations

"""Semantic reference oracle and architectural model for v_mul_u32_u24.

RDNA2 / GFX10 VOP2 24-bit Unsigned Integer Multiply:
Multiplies the low 24 bits of two unsigned integer source operands (bits [31:24] ignored)
and returns the lower 32 bits of the 48-bit product into a 32-bit VGPR.

Syntax:
    v_mul_u32_u24 vdst, vsrc0, vsrc1

Semantics:
    s0_u24 = src0 & 0x00FF_FFFF
    s1_u24 = src1 & 0x00FF_FFFF
    product_48 = s0_u24 * s1_u24
    vdst = product_48 & 0xFFFF_FFFF
"""

import os
import struct
import sys
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si
import support

# 24-bit and 32-bit Constants
MASK_24: int = 0x00FF_FFFF
MASK_32: int = 0xFFFF_FFFF
MAX_U24: int = 0x00FF_FFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def u24(val: int) -> int:
    """Normalize integer to unsigned 24-bit integer."""
    return val & MASK_24


def oracle_mul_u32_u24(src0: int, src1: int) -> int:
    """Reference mathematical oracle for v_mul_u32_u24."""
    s0 = u24(src0)
    s1 = u24(src1)
    return u32(s0 * s1)


def run_translator_mul_u24(
    src0: int,
    src1: int,
    name: str = "ps_mul_u24",
) -> int:
    """Translate and run a minimal shader through tools/rdna2spirv and return result word."""
    lines = [
        f"v_mov_b32 v1, {u32(src0):#x}",
        f"v_mov_b32 v2, {u32(src1):#x}",
        "v_mul_u32_u24 v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestOracleAndHardwareModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_basic_multiplication(self) -> None:
        """Verify 100 * 250 = 25000."""
        got = oracle_mul_u32_u24(100, 250)
        self.assertEqual(got, 25000)

    def test_high_byte_masking(self) -> None:
        """Verify bits [31:24] are strictly ignored on both operands."""
        # 0xFF000002 low 24 bits = 2, 0xAA000003 low 24 bits = 3 -> 6
        got = oracle_mul_u32_u24(0xFF00_0002, 0xAA00_0003)
        self.assertEqual(got, 6)

    def test_max_u24_product(self) -> None:
        """Verify (2^24 - 1) * (2^24 - 1) low 32 bits equals 0xFE000001."""
        # (0xFFFFFF * 0xFFFFFF) = 0xFFFFFE000001; low 32 bits = 0xFE000001
        got = oracle_mul_u32_u24(MAX_U24, MAX_U24)
        self.assertEqual(got, 0xFE00_0001)

    def test_oracle_matches_translator(self) -> None:
        """Verify oracle output directly against translator output."""
        src0 = 0x1234_5678
        src1 = 0xABCD_EF02
        expected = oracle_mul_u32_u24(src0, src1)
        translated = run_translator_mul_u24(src0, src1, name="ps_mul_oracle")
        self.assertEqual(translated, expected)


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
                raise ValueError(f"VGPR index out of range: {self.value}")
        elif self.kind == OperandKind.SGPR:
            if not (0 <= self.value <= 105):
                raise ValueError(f"SGPR index out of range: {self.value}")
        elif self.kind == OperandKind.IMMEDIATE:
            if not (-0x8000_0000 <= self.value <= 0xFFFF_FFFF):
                raise ValueError(f"Immediate value out of 32-bit range: {self.value}")


def VGPR(idx: int) -> Operand:
    return Operand(OperandKind.VGPR, idx)


def SGPR(idx: int) -> Operand:
    return Operand(OperandKind.SGPR, idx)


def Immediate(val: int) -> Operand:
    return Operand(OperandKind.IMMEDIATE, u32(val))


@dataclass(frozen=True)
class VMulU24Instruction:
    dst: Operand
    src0: Operand
    src1: Operand

    def __post_init__(self) -> None:
        if self.dst.kind != OperandKind.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.kind}")

    def format_operand(self, op: Operand) -> str:
        if op.kind == OperandKind.VGPR:
            return f"v{op.value}"
        elif op.kind == OperandKind.SGPR:
            return f"s{op.value}"
        else:
            return f"{op.value:#x}"

    def to_assembly(self) -> str:
        d = self.format_operand(self.dst)
        s0 = self.format_operand(self.src0)
        s1 = self.format_operand(self.src1)
        return f"v_mul_u32_u24 {d}, {s0}, {s1}"


class TestASTDefinitions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        """Verify AST correctly serializes to valid RDNA2 assembly text."""
        instr = VMulU24Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
        )
        asm = instr.to_assembly()
        self.assertEqual(asm, "v_mul_u32_u24 v0, v1, v2")

    def test_ast_translation_validity(self) -> None:
        """Verify that assembly generated by AST translates cleanly."""
        instr = VMulU24Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
        )
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 250",
            instr.to_assembly(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_mul_valid", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 25000)


# ==============================================================================
# SECTION 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    vgprs: Dict[int, int] = field(default_factory=dict)
    sgprs: Dict[int, int] = field(default_factory=dict)

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs.get(idx, 0)

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = u32(val)

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0)

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = u32(val)


class ScalarInterpreter:
    def resolve_operand(self, op: Operand, state: ScalarMachineState) -> int:
        if op.kind == OperandKind.VGPR:
            return state.get_vgpr(op.value)
        elif op.kind == OperandKind.SGPR:
            return state.get_sgpr(op.value)
        elif op.kind == OperandKind.IMMEDIATE:
            return u32(op.value)
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")

    def step(self, instr: VMulU24Instruction, state: ScalarMachineState) -> int:
        val0 = self.resolve_operand(instr.src0, state)
        val1 = self.resolve_operand(instr.src1, state)

        res = oracle_mul_u32_u24(val0, val1)
        state.set_vgpr(instr.dst.value, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_execution(self) -> None:
        """Verify interpreter correctly executes across VGPR, SGPR, and Immediate inputs."""
        state = ScalarMachineState()
        state.set_vgpr(1, 0x1200_0005)                         # low 24 = 5
        state.set_sgpr(4, 0xAB00_0007)                         # low 24 = 7

        interp = ScalarInterpreter()
        instr = VMulU24Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=SGPR(4),
        )
        res = interp.step(instr, state)
        self.assertEqual(res, 35)
        self.assertEqual(state.get_vgpr(0), 35)

    def test_scalar_interpreter_cross_validation_with_translator(self) -> None:
        """Verify scalar interpreter exactly matches translated SPIR-V execution."""
        state = ScalarMachineState()
        src0 = 0x0012_3456
        src1 = 0x0000_0008
        state.set_vgpr(1, src0)
        state.set_vgpr(2, src1)

        interp = ScalarInterpreter()
        instr = VMulU24Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
        )
        interp_res = interp.step(instr, state)
        spv_res = run_translator_mul_u24(src0, src1, name="ps_scalar_mul_cross")
        self.assertEqual(interp_res, spv_res)


# ==============================================================================
# SECTION 4: SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    num_lanes: int = 32
    exec_mask: int = MASK_32
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
        return self.vgprs[lane].get(idx, 0)

    def set_vgpr(self, lane: int, idx: int, val: int) -> None:
        self.vgprs[lane][idx] = u32(val)

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0)

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = u32(val)


class SIMDInterpreter:
    def resolve_lane_operand(self, op: Operand, lane: int, state: SIMDState) -> int:
        if op.kind == OperandKind.VGPR:
            return state.get_vgpr(lane, op.value)
        elif op.kind == OperandKind.SGPR:
            return state.get_sgpr(op.value)
        elif op.kind == OperandKind.IMMEDIATE:
            return u32(op.value)
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")

    def step(self, instr: VMulU24Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = []
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                results.append(None)
                continue

            v0 = self.resolve_lane_operand(instr.src0, lane, state)
            v1 = self.resolve_lane_operand(instr.src1, lane, state)

            res = oracle_mul_u32_u24(v0, v1)
            state.set_vgpr(lane, instr.dst.value, res)
            results.append(res)
        return results


class TestSIMDDivergence(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_alternating_lanes(self) -> None:
        """Verify Wave32 alternating even/odd lane execution and inactive preservation."""
        mask = 0x5555_5555  # Even lanes active
        state = SIMDState(num_lanes=32, exec_mask=mask)
        sentinel = 0xDEAD_BEEF

        for lane in range(32):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, lane + 1)
            state.set_vgpr(lane, 2, 10)

        interp = SIMDInterpreter()
        instr = VMulU24Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2))
        results = interp.step(instr, state)

        for lane in range(32):
            if lane % 2 == 0:
                expected = (lane + 1) * 10
                self.assertEqual(results[lane], expected)
                self.assertEqual(state.get_vgpr(lane, 0), expected)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_wave64_divergence(self) -> None:
        """Verify Wave64 boundary execution across low and high 32-lane words."""
        mask = (1 << 0) | (1 << 31) | (1 << 32) | (1 << 63)
        state = SIMDState(num_lanes=64, exec_mask=mask)
        sentinel = 0xBAAD_F00D

        for lane in range(64):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, 100)
            state.set_vgpr(lane, 2, 20)

        interp = SIMDInterpreter()
        instr = VMulU24Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2))
        results = interp.step(instr, state)

        for lane in range(64):
            if lane in (0, 31, 32, 63):
                self.assertEqual(results[lane], 2000)
                self.assertEqual(state.get_vgpr(lane, 0), 2000)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)


# ==============================================================================
# SECTION 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0_in_place(self) -> None:
        """Verify v0 = v0 * v1 reads initial v0 before writeback."""
        state = ScalarMachineState()
        init_v0 = 0xAA00_0007  # low 24 = 7
        v1_val = 0xBB00_0009   # low 24 = 9

        state.set_vgpr(0, init_v0)
        state.set_vgpr(1, v1_val)

        interp = ScalarInterpreter()
        instr = VMulU24Instruction(dst=VGPR(0), src0=VGPR(0), src1=VGPR(1))
        res = interp.step(instr, state)

        self.assertEqual(res, 63)
        self.assertEqual(state.get_vgpr(0), 63)

        # Cross check with translated SPIR-V
        spv_res = run_translator_mul_u24(init_v0, v1_val, name="ps_alias_dst_src0")
        self.assertEqual(res, spv_res)

    def test_alias_all_same_register(self) -> None:
        """Verify v0 = v0 * v0 squaring."""
        state = ScalarMachineState()
        state.set_vgpr(0, 12)

        interp = ScalarInterpreter()
        instr = VMulU24Instruction(dst=VGPR(0), src0=VGPR(0), src1=VGPR(0))
        res = interp.step(instr, state)

        self.assertEqual(res, 144)
        self.assertEqual(state.get_vgpr(0), 144)
