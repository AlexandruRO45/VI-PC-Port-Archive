from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_max3_i32.

Instruction definition:
    v_max3_i32 vdst, vsrc0, vsrc1, vsrc2 [clamp]

Semantics:
    3-operand signed 32-bit integer maximum (RDNA2 VOP3).
    D.i32 = max(S0.i32, S1.i32, S2.i32)
    Operands are interpreted as two's complement signed 32-bit integers in [-2^31, 2^31 - 1].
    The largest algebraic value is written to the 32-bit destination VGPR.

Standard library only: struct, math, unittest. Zero game data.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

# ==============================================================================
# Architectural Constants
# ==============================================================================

MASK32: int = 0xFFFF_FFFF
INT32_MIN: int = -2147483648     # -2^31 (0x80000000)
INT32_MAX: int = 2147483647      # 2^31 - 1 (0x7FFFFFFF)


# ==============================================================================
# Reference Mathematical Oracle and Hardware Translation Helper
# ==============================================================================

def to_i32(val: int) -> int:
    """Interpret 32-bit integer as two's complement signed integer."""
    val32 = val & MASK32
    if val32 & 0x8000_0000:
        return val32 - 0x1_0000_0000
    return val32


def ref_max3_i32(src0: int, src1: int, src2: int, clamp: bool = False) -> int:
    """Architectural reference oracle for v_max3_i32."""
    i0 = to_i32(src0)
    i1 = to_i32(src1)
    i2 = to_i32(src2)

    res = max(i0, i1, i2)
    if clamp:
        if res < 0:
            res = 0

    return res & MASK32


def run_hw_max3_i32(s0: int, s1: int, s2: int, name: str = "ps_max3_i32") -> int:
    """Translates and executes v_max3_i32 via tools/rdna2spirv.py and returns mrt0 dword."""
    lines = [
        f"v_mov_b32_e32 v1, {s0 & MASK32:#x}",
        f"v_mov_b32_e32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32_e32 v3, {s2 & MASK32:#x}",
        "v_max3_i32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


@dataclass
class VGPRWord:
    raw: int = 0

    @property
    def u32(self) -> int:
        return self.raw & MASK32

    @property
    def i32(self) -> int:
        return to_i32(self.raw)


class TestMax3I32ReferenceOracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model."""

    def test_all_positive_values(self):
        self.assertEqual(ref_max3_i32(10, 20, 30), 30)
        self.assertEqual(ref_max3_i32(100, 50, 75), 100)
        self.assertEqual(run_hw_max3_i32(10, 20, 30, name="ps_or_pos1"), 30)
        self.assertEqual(run_hw_max3_i32(100, 50, 75, name="ps_or_pos2"), 100)

    def test_mixed_signs(self):
        # max(-5, 10, -2) = 10
        self.assertEqual(ref_max3_i32(-5, 10, -2), 10)
        # max(-100, -200, 0) = 0
        self.assertEqual(ref_max3_i32(-100, -200, 0), 0)
        self.assertEqual(run_hw_max3_i32(-5, 10, -2, name="ps_or_mix1"), 10)
        self.assertEqual(run_hw_max3_i32(-100, -200, 0, name="ps_or_mix2"), 0)

    def test_all_negative_values(self):
        # max(-10, -5, -20) = -5 (0xFFFFFFFB)
        self.assertEqual(ref_max3_i32(-10, -5, -20), 0xFFFF_FFFB)
        self.assertEqual(ref_max3_i32(-1, -2, -3), 0xFFFF_FFFF)
        self.assertEqual(run_hw_max3_i32(-10, -5, -20, name="ps_or_neg1"), 0xFFFF_FFFB)
        self.assertEqual(run_hw_max3_i32(-1, -2, -3, name="ps_or_neg2"), 0xFFFF_FFFF)

    def test_boundary_extremes(self):
        # max(INT32_MIN, 0, INT32_MAX) = INT32_MAX
        self.assertEqual(ref_max3_i32(INT32_MIN, 0, INT32_MAX), INT32_MAX)
        # max(INT32_MIN, INT32_MIN + 1, -1) = -1
        self.assertEqual(ref_max3_i32(INT32_MIN, INT32_MIN + 1, -1), 0xFFFF_FFFF)
        self.assertEqual(run_hw_max3_i32(INT32_MIN, 0, INT32_MAX, name="ps_or_bnd1"), INT32_MAX & MASK32)
        self.assertEqual(run_hw_max3_i32(INT32_MIN, INT32_MIN + 1, -1, name="ps_or_bnd2"), 0xFFFF_FFFF)

    def test_identical_operands(self):
        self.assertEqual(ref_max3_i32(42, 42, 42), 42)
        self.assertEqual(ref_max3_i32(-7, -7, -7), 0xFFFF_FFF9)
        self.assertEqual(run_hw_max3_i32(42, 42, 42, name="ps_or_id1"), 42)
        self.assertEqual(run_hw_max3_i32(-7, -7, -7, name="ps_or_id2"), 0xFFFF_FFF9)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class RegisterOperand:
    op_type: OperandType
    index: int

    def __post_init__(self):
        if self.op_type == OperandType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.op_type == OperandType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")

    @property
    def name(self) -> str:
        prefix = "v" if self.op_type == OperandType.VGPR else "s"
        return f"{prefix}{self.index}"


@dataclass(frozen=True)
class LiteralOperand:
    value: int

    def __post_init__(self):
        if not (-0x8000_0000 <= self.value <= MASK32):
            raise ValueError(f"Literal 32-bit value out of range: {self.value}")


@dataclass
class VMax3I32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    src1: Union[RegisterOperand, LiteralOperand]
    src2: Union[RegisterOperand, LiteralOperand]
    clamp: bool = False

    def __post_init__(self):
        if self.dst.op_type != OperandType.VGPR:
            raise TypeError("Destination of v_max3_i32 must be a VGPR")


class TestMax3I32AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_valid_ast_construction(self):
        dst = RegisterOperand(OperandType.VGPR, 0)
        s0 = RegisterOperand(OperandType.VGPR, 1)
        s1 = RegisterOperand(OperandType.VGPR, 2)
        s2 = RegisterOperand(OperandType.VGPR, 3)
        insn = VMax3I32Instruction(dst=dst, src0=s0, src1=s1, src2=s2, clamp=False)
        self.assertEqual(insn.dst.name, "v0")
        self.assertEqual(insn.src0.name, "v1")
        self.assertEqual(insn.src1.name, "v2")
        self.assertEqual(insn.src2.name, "v3")

        lines = [
            "v_mov_b32_e32 v1, 10",
            "v_mov_b32_e32 v2, 20",
            "v_mov_b32_e32 v3, 30",
            "v_max3_i32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_valid", lines)
        self.assertIn("OpExtInst %uint %glsl SMax", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 30)

    def test_ast_with_sgpr_and_literal(self):
        lines = [
            "s_mov_b32 s0, 10",
            "v_mov_b32_e32 v1, 20",
            "v_max3_i32 v0, s0, v1, -30",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_sgpr_lit", lines)
        self.assertIn("OpExtInst %uint %glsl SMax", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)

    def test_invalid_register_indices(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 256)
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.SGPR, 106)

    def test_invalid_dst_type(self):
        with self.assertRaises(TypeError):
            VMax3I32Instruction(
                dst=RegisterOperand(OperandType.SGPR, 0),
                src0=RegisterOperand(OperandType.VGPR, 1),
                src1=RegisterOperand(OperandType.VGPR, 2),
                src2=RegisterOperand(OperandType.VGPR, 3),
            )

    def test_literal_operand_bounds(self):
        with self.assertRaises(ValueError):
            LiteralOperand(0x1_0000_0000)
        with self.assertRaises(ValueError):
            LiteralOperand(-0x8000_0001)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

class ScalarMachine:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106
        self.scc: int = 0

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value & MASK32
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index] & MASK32
        raise TypeError(f"Unknown operand type: {type(op)}")

    def write_vgpr(self, index: int, val32: int):
        self.vgprs[index] = val32 & MASK32

    def step(self, insn: VMax3I32Instruction):
        val_s0 = self.read_operand(insn.src0)
        val_s1 = self.read_operand(insn.src1)
        val_s2 = self.read_operand(insn.src2)
        result = ref_max3_i32(val_s0, val_s1, val_s2, clamp=insn.clamp)
        self.write_vgpr(insn.dst.index, result)


class TestMax3I32ScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction execution."""

    def setUp(self):
        self.machine = ScalarMachine()

    def test_scalar_execution_basic(self):
        self.machine.vgprs[1] = 0xFFFF_FFF6  # -10
        self.machine.vgprs[2] = 0xFFFF_FFFB  # -5
        self.machine.vgprs[3] = 0xFFFF_FFEC  # -20
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[0], 0xFFFF_FFFB)

        hw = run_hw_max3_i32(-10, -5, -20, name="ps_sc_basic")
        self.assertEqual(hw, 0xFFFF_FFFB)

    def test_scalar_with_sgpr_and_literal(self):
        self.machine.sgprs[4] = 10
        self.machine.vgprs[2] = 20
        lit = LiteralOperand(-30)
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 5),
            src0=RegisterOperand(OperandType.SGPR, 4),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=lit,
        )
        self.machine.step(insn)
        self.assertEqual(self.machine.vgprs[5], 20)

        lines = [
            "s_mov_b32 s4, 10",
            "v_mov_b32_e32 v2, 20",
            "v_max3_i32 v5, s4, v2, -30",
            "exp mrt0 v5, v5, v5, v5 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_sc_sgpr_lit", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

class SIMDWave:
    def __init__(self, wave_size: int = 64):
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size: int = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.sgprs: List[int] = [0] * 106
        self.exec_mask: int = (1 << wave_size) - 1

    def is_lane_active(self, lane_id: int) -> bool:
        return bool(self.exec_mask & (1 << lane_id))

    def step(self, insn: VMax3I32Instruction):
        for lane in range(self.wave_size):
            if not self.is_lane_active(lane):
                continue
            # Read src0
            if isinstance(insn.src0, LiteralOperand):
                val_s0 = insn.src0.value & MASK32
            elif insn.src0.op_type == OperandType.VGPR:
                val_s0 = self.vgprs[insn.src0.index][lane] & MASK32
            else:
                val_s0 = self.sgprs[insn.src0.index] & MASK32

            # Read src1
            if isinstance(insn.src1, LiteralOperand):
                val_s1 = insn.src1.value & MASK32
            elif insn.src1.op_type == OperandType.VGPR:
                val_s1 = self.vgprs[insn.src1.index][lane] & MASK32
            else:
                val_s1 = self.sgprs[insn.src1.index] & MASK32

            # Read src2
            if isinstance(insn.src2, LiteralOperand):
                val_s2 = insn.src2.value & MASK32
            elif insn.src2.op_type == OperandType.VGPR:
                val_s2 = self.vgprs[insn.src2.index][lane] & MASK32
            else:
                val_s2 = self.sgprs[insn.src2.index] & MASK32

            res = ref_max3_i32(val_s0, val_s1, val_s2, clamp=insn.clamp)
            self.vgprs[insn.dst.index][lane] = res & MASK32


class TestMax3I32SIMDDivergence(unittest.TestCase):
    """Commit 4: Multi-lane execution under Wave32 and Wave64 masks."""

    def test_wave32_alternating_exec(self):
        wave = SIMDWave(wave_size=32)
        wave.exec_mask = 0x5555_5555  # Even lanes active
        for lane in range(32):
            wave.vgprs[0][lane] = 0xDEAD_BEEF
            wave.vgprs[1][lane] = to_i32(-lane)
            wave.vgprs[2][lane] = 0
            wave.vgprs[3][lane] = to_i32(lane)

        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        wave.step(insn)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[0][lane], lane)
            else:
                self.assertEqual(wave.vgprs[0][lane], 0xDEAD_BEEF)

        # Translator execution for multi-lane divergent configurations
        for lane in (0, 2, 4, 8, 16):
            hw = run_hw_max3_i32(-lane, 0, lane, name=f"ps_simd_w32_{lane}")
            self.assertEqual(hw, lane)

    def test_wave64_divergence(self):
        wave = SIMDWave(wave_size=64)
        wave.exec_mask = 0xFFFF_FFFF_0000_0000  # Top 32 lanes active
        for lane in range(64):
            wave.vgprs[10][lane] = 0xCAFE_0000
            wave.vgprs[1][lane] = -10
            wave.vgprs[2][lane] = lane - 40
            wave.vgprs[3][lane] = -5

        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 10),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        wave.step(insn)

        for lane in range(32):
            self.assertEqual(wave.vgprs[10][lane], 0xCAFE_0000)
        for lane in range(32, 64):
            expected = ref_max3_i32(-10, lane - 40, -5)
            self.assertEqual(wave.vgprs[10][lane], expected)

        # Hardware execution across active lanes
        for lane in (32, 40, 50, 63):
            expected = ref_max3_i32(-10, lane - 40, -5)
            hw = run_hw_max3_i32(-10, lane - 40, -5, name=f"ps_simd_w64_{lane}")
            self.assertEqual(hw, expected)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestMax3I32RegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing (dst == src0, src1, src2) and preservation."""

    def test_dst_aliases_src0(self):
        machine = ScalarMachine()
        machine.vgprs[0] = 10
        machine.vgprs[1] = 50
        machine.vgprs[2] = 20
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 0),
            src1=RegisterOperand(OperandType.VGPR, 1),
            src2=RegisterOperand(OperandType.VGPR, 2),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], 50)

        lines = [
            "v_mov_b32_e32 v0, 10",
            "v_mov_b32_e32 v1, 50",
            "v_mov_b32_e32 v2, 20",
            "v_max3_i32 v0, v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_s0", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)

    def test_dst_aliases_src2(self):
        machine = ScalarMachine()
        machine.vgprs[0] = 50
        machine.vgprs[1] = 10
        machine.vgprs[2] = 20
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 0),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], 50)

        lines = [
            "v_mov_b32_e32 v0, 50",
            "v_mov_b32_e32 v1, 10",
            "v_mov_b32_e32 v2, 20",
            "v_max3_i32 v0, v1, v2, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_s2", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)

    def test_dst_aliases_all_sources(self):
        machine = ScalarMachine()
        machine.vgprs[0] = 0xFFFF_FFFE  # -2
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 0),
            src1=RegisterOperand(OperandType.VGPR, 0),
            src2=RegisterOperand(OperandType.VGPR, 0),
        )
        machine.step(insn)
        self.assertEqual(machine.vgprs[0], 0xFFFF_FFFE)

        lines = [
            "v_mov_b32_e32 v0, -2",
            "v_max3_i32 v0, v0, v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_all", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFF_FFFE)


# ==============================================================================
# Condition Codes, Status Flags, and Invariance
# ==============================================================================

class TestMax3I32StatusFlags(unittest.TestCase):
    """Commit 6: Condition codes (VCC/SCC) isolation under v_max3_i32."""

    def test_scc_invariance(self):
        machine = ScalarMachine()
        machine.scc = 1
        machine.vgprs[1] = -5
        machine.vgprs[2] = 10
        machine.vgprs[3] = 2
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        machine.step(insn)
        self.assertEqual(machine.scc, 1)

        # Hardware execution: verify SCC remains 1 across v_max3_i32
        lines = [
            "s_mov_b32 s3, 1",
            "s_cmp_eq_u32 s3, 1",  # sets SCC to 1
            "v_mov_b32_e32 v1, -5",
            "v_mov_b32_e32 v2, 10",
            "v_mov_b32_e32 v3, 2",
            "v_max3_i32 v0, v1, v2, v3",
            "s_cselect_b32 s5, 0xCAFE, 0xBEEF",  # reads SCC: must select 0xCAFE
            "v_mov_b32_e32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_scc_invariance", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 10)
        self.assertEqual(words[1], 0xCAFE)

    def test_exec_mask_invariance(self):
        wave = SIMDWave(wave_size=64)
        orig_mask = 0xAAAA_5555_AAAA_5555
        wave.exec_mask = orig_mask
        insn = VMax3I32Instruction(
            dst=RegisterOperand(OperandType.VGPR, 0),
            src0=RegisterOperand(OperandType.VGPR, 1),
            src1=RegisterOperand(OperandType.VGPR, 2),
            src2=RegisterOperand(OperandType.VGPR, 3),
        )
        wave.step(insn)
        self.assertEqual(wave.exec_mask, orig_mask)

        # Hardware execution: v_max3_i32 computes correct maximum
        hw = run_hw_max3_i32(-5, 10, 2, name="ps_exec_invariance")
        self.assertEqual(hw, 10)


# ==============================================================================
# IEEE 754 Bit-Pattern Boundary Extremes as Signed Integers
# ==============================================================================

class TestMax3I32BitPatternBoundaries(unittest.TestCase):
    """Commit 7: IEEE 754 bit patterns interpreted as signed integers."""

    def test_float_patterns_as_signed_integers(self):
        # Raw IEEE 754 float bit patterns interpreted as signed two's complement integers:
        # Float -0.0f: 0x80000000 -> -2147483648 (INT32_MIN)
        # Float +0.0f: 0x00000000 -> 0
        # Float 1.0f:  0x3F800000 -> 1065353216
        # max(-0.0f, +0.0f, 1.0f) = 1.0f (0x3F800000)
        self.assertEqual(ref_max3_i32(0x8000_0000, 0x0000_0000, 0x3F80_0000), 0x3F80_0000)
        # max(-0.0f, -1, -100) = max(-2147483648, -1, -100) = -1 (0xFFFFFFFF)
        self.assertEqual(ref_max3_i32(0x8000_0000, -1, -100), 0xFFFF_FFFF)

        hw1 = run_hw_max3_i32(0x8000_0000, 0x0000_0000, 0x3F80_0000, name="ps_bnd_flt1")
        self.assertEqual(hw1, 0x3F80_0000)
        hw2 = run_hw_max3_i32(0x8000_0000, -1, -100, name="ps_bnd_flt2")
        self.assertEqual(hw2, 0xFFFF_FFFF)


# ==============================================================================
# Negative Controls, Rejection, Mutation Detection, and Translator Execution
# ==============================================================================

class TestMax3I32RejectionAndHardwareTranslator(unittest.TestCase):
    """Commit 8: Syntax rejection, mutation detection, and translation matrix."""

    def test_syntax_rejection_negative_controls(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 300)
        with self.assertRaises(ValueError):
            LiteralOperand(0x1_0000_0000)

    def test_mutation_detection_smax_to_smin(self):
        # Mutate translated SPIR-V: replace SMax with SMin (must diverge on mixed operands)
        lines = [
            "v_mov_b32_e32 v1, -5",
            "v_mov_b32_e32 v2, 10",
            "v_mov_b32_e32 v3, -2",
            "v_max3_i32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_smax", lines)
        correct_word = support.mrt0_words(spv)[0]
        self.assertEqual(correct_word, 10)

        mutated_spv = spv.replace("glsl SMax", "glsl SMin")
        mutated_word = support.mrt0_words(mutated_spv)[0]
        self.assertEqual(mutated_word, 0xFFFF_FFFB)  # -5
        self.assertNotEqual(correct_word, mutated_word)

    def test_mutation_detection_smax_to_umax(self):
        # Mutate translated SPIR-V: replace SMax with UMax (unsigned treats negative as large positive)
        lines = [
            "v_mov_b32_e32 v1, -5",
            "v_mov_b32_e32 v2, 10",
            "v_mov_b32_e32 v3, -2",
            "v_max3_i32 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_umax", lines)
        correct_word = support.mrt0_words(spv)[0]
        self.assertEqual(correct_word, 10)

        mutated_spv = spv.replace("glsl SMax", "glsl UMax")
        mutated_word = support.mrt0_words(mutated_spv)[0]
        self.assertEqual(mutated_word, 0xFFFF_FFFE)  # 0xFFFFFFFE (-2) is largest unsigned
        self.assertNotEqual(correct_word, mutated_word)

    def test_hardware_translator_execution(self):
        """Validates that tools/rdna2spirv.py translates v_max3_i32 and matches expected results."""
        cases = [
            (-5, 10, -2, "v_max3_i32 v0, v1, v2, v3", 10),
            (-10, -5, -20, "v_max3_i32 v0, v1, v2, v3", 0xFFFF_FFFB),
            (0x8000_0000, 0, 0x7FFF_FFFF, "v_max3_i32 v0, v1, v2, v3", 0x7FFF_FFFF),
            (-1, -1, -1, "v_max3_i32 v0, v1, v2, v3", 0xFFFF_FFFF),
            (100, 200, 300, "v_max3_i32 v0, v1, v2, v3", 300),
            (-50, -10, -20, "v_max3_i32 v0, v1, v2, v3", 0xFFFF_FFF6),
            (0, -100, 50, "v_max3_i32 v0, v1, v2, v3", 50),
        ]
        for s0, s1, s2, asm, expected in cases:
            prog = [
                f"v_mov_b32_e32 v1, {s0:#x}",
                f"v_mov_b32_e32 v2, {s1:#x}",
                f"v_mov_b32_e32 v3, {s2:#x}",
                asm,
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines("ps_max3_i32_hw", prog)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], expected, f"Failed on {asm}: got {words[0]:#x}, expected {expected:#x}")


if __name__ == '__main__':
    unittest.main()
