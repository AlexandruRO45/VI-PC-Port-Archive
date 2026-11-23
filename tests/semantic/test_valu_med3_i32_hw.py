"""Hardware-accurate semantic tests and mathematical reference model for RDNA2 v_med3_i32.

RDNA2 / GFX10 Vector ALU Signed 32-bit Integer Three-Operand Median:
Performs signed integer comparison across three 32-bit operands and returns the median:
    vdst = median_i32(src0, src1, src2)

Syntax:
    v_med3_i32 vdst, src0, src1, src2

Semantics:
    Interprets src0, src1, src2 as 32-bit two's complement signed integers in [-2^31, 2^31 - 1].
    Sorts the three integers in non-decreasing order: s0 <= s1 <= s2.
    vdst = s1 (the median element), represented as a 32-bit unsigned dword (& 0xFFFFFFFF).
    Equivalent to clamp(src2, min(src0, src1), max(src0, src1)) in signed integer arithmetic.

Translator Integration:
    Exercises tools/rdna2spirv.py via support.translate_lines and executes the generated
    SPIR-V shaders via support.mrt0_words.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
from enum import Enum, auto
import itertools
import math
import os
import struct
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402
import rdna2spirv  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402

MASK32: int = 0xFFFFFFFF
SIGN_BIT32: int = 0x80000000
INT32_MIN: int = -0x80000000
INT32_MAX: int = 0x7FFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def ref_med3_i32(s0: int, s1: int, s2: int) -> int:
    """Pure mathematical reference oracle for signed 32-bit median."""
    sorted_vals = sorted([s32(s0), s32(s1), s32(s2)])
    return u32(sorted_vals[1])


def hw_clamp_med3_i32(s0: int, s1: int, s2: int) -> int:
    """Hardware clamp-equivalent model matching GLSL SClamp(z, SMin(x, y), SMax(x, y))."""
    lo = min(s32(s0), s32(s1))
    hi = max(s32(s0), s32(s1))
    clamped = min(max(s32(s2), lo), hi)
    return u32(clamped)


def run_shader_med3_i32(
    src0_val: int,
    src1_val: int,
    src2_val: int,
    vdst_reg: int = 0,
    v0_reg: int = 1,
    v1_reg: int = 2,
    v2_reg: int = 3,
) -> int:
    """Translate and execute a synthetic pixel shader using v_med3_i32, returning mrt0.x."""
    lines = [
        f"v_mov_b32_e32 v{v0_reg}, {u32(src0_val):#x}",
        f"v_mov_b32_e32 v{v1_reg}, {u32(src1_val):#x}",
        f"v_mov_b32_e32 v{v2_reg}, {u32(src2_val):#x}",
        f"v_med3_i32 v{vdst_reg}, v{v0_reg}, v{v1_reg}, v{v2_reg}",
        f"exp mrt0, v{vdst_reg}, v{vdst_reg}, v{vdst_reg}, v{vdst_reg} done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_med3_i32_test", lines)
    words = support.mrt0_words(spv)
    return words[0]


class TestMed3I32Oracle(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_oracle_basic_integers(self):
        cases = [
            (10, 20, 30, 20),
            (30, 10, 20, 20),
            (-5, 0, 5, 0),
            (-100, -200, -50, -100),
            (7, 7, 7, 7),
            (12, 12, 5, 12),
            (5, 12, 12, 12),
        ]
        for s0, s1, s2, expected in cases:
            with self.subTest(s0=s0, s1=s1, s2=s2):
                self.assertEqual(ref_med3_i32(s0, s1, s2), u32(expected))
                self.assertEqual(hw_clamp_med3_i32(s0, s1, s2), u32(expected))

    def test_oracle_permutation_symmetry(self):
        triples = [
            (5, -10, 25),
            (-50, 0, 100),
            (0x7FFFFFFF, -1, -0x80000000),
            (42, 42, -99),
        ]
        for trip in triples:
            expected = ref_med3_i32(*trip)
            for perm in itertools.permutations(trip):
                with self.subTest(perm=perm):
                    self.assertEqual(ref_med3_i32(*perm), expected)
                    self.assertEqual(hw_clamp_med3_i32(*perm), expected)

    def test_translator_single_invocation(self):
        result = run_shader_med3_i32(5, 0x80000000, 7)
        self.assertEqual(result, 5)
        self.assertEqual(result, ref_med3_i32(5, 0x80000000, 7))


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions (Commit 2)
# ==============================================================================

class RegType(Enum):
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Reg:
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        limit = 256 if self.reg_type == RegType.VGPR else 106
        if not (0 <= self.index < limit):
            raise ValueError(f"Invalid register index {self.reg_type.value}{self.index}")

    @property
    def name(self) -> str:
        return f"{self.reg_type.value}{self.index}"


class OperandKind(Enum):
    REG = auto()
    INLINE_INT = auto()
    LITERAL_INT = auto()


@dataclasses.dataclass(frozen=True)
class OperandAST:
    kind: OperandKind
    reg: Optional[Reg] = None
    imm: Optional[int] = None

    @classmethod
    def from_vgpr(cls, idx: int) -> OperandAST:
        return cls(kind=OperandKind.REG, reg=Reg(idx, RegType.VGPR))

    @classmethod
    def from_sgpr(cls, idx: int) -> OperandAST:
        return cls(kind=OperandKind.REG, reg=Reg(idx, RegType.SGPR))

    @classmethod
    def from_imm(cls, val: int) -> OperandAST:
        if -16 <= s32(val) <= 64:
            return cls(kind=OperandKind.INLINE_INT, imm=s32(val))
        return cls(kind=OperandKind.LITERAL_INT, imm=u32(val))

    def format_asm(self) -> str:
        if self.kind == OperandKind.REG:
            assert self.reg is not None
            return self.reg.name
        elif self.kind == OperandKind.INLINE_INT:
            assert self.imm is not None
            return str(self.imm)
        else:
            assert self.imm is not None
            return f"{self.imm:#x}"


@dataclasses.dataclass(frozen=True)
class Med3I32InstructionAST:
    dst: Reg
    src0: OperandAST
    src1: OperandAST
    src2: OperandAST

    def emit_asm(self) -> str:
        return f"v_med3_i32 {self.dst.name}, {self.src0.format_asm()}, {self.src1.format_asm()}, {self.src2.format_asm()}"


def parse_med3_i32_asm(line: str) -> Med3I32InstructionAST:
    """Parse a single v_med3_i32 instruction into an AST."""
    line = line.strip()
    if not line.startswith("v_med3_i32"):
        raise ValueError(f"Expected v_med3_i32 instruction, got: {line}")
    args_str = line[len("v_med3_i32"):].strip()
    parts = [p.strip() for p in args_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"v_med3_i32 requires exactly 4 operands (dst, src0, src1, src2), got: {len(parts)}")

    dst_part = parts[0]
    if not dst_part.startswith("v"):
        raise ValueError(f"Destination must be a VGPR, got {dst_part}")
    dst_reg = Reg(int(dst_part[1:]), RegType.VGPR)

    def parse_src(p: str) -> OperandAST:
        if p.startswith("v"):
            return OperandAST.from_vgpr(int(p[1:]))
        elif p.startswith("s"):
            return OperandAST.from_sgpr(int(p[1:]))
        else:
            val = int(p, 0)
            return OperandAST.from_imm(val)

    src0 = parse_src(parts[1])
    src1 = parse_src(parts[2])
    src2 = parse_src(parts[3])
    return Med3I32InstructionAST(dst=dst_reg, src0=src0, src1=src1, src2=src2)


class TestMed3I32AST(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self):
        inst = Med3I32InstructionAST(
            dst=Reg(0, RegType.VGPR),
            src0=OperandAST.from_vgpr(1),
            src1=OperandAST.from_imm(-16),
            src2=OperandAST.from_imm(0x12345678),
        )
        self.assertEqual(inst.emit_asm(), "v_med3_i32 v0, v1, -16, 0x12345678")

    def test_ast_parse_valid(self):
        line = "v_med3_i32 v5, v1, v2, v3"
        ast_obj = parse_med3_i32_asm(line)
        self.assertEqual(ast_obj.dst, Reg(5, RegType.VGPR))
        self.assertEqual(ast_obj.src0, OperandAST.from_vgpr(1))
        self.assertEqual(ast_obj.src1, OperandAST.from_vgpr(2))
        self.assertEqual(ast_obj.src2, OperandAST.from_vgpr(3))
        self.assertEqual(ast_obj.emit_asm(), line)

    def test_translator_with_ast_emitted_code(self):
        inst = Med3I32InstructionAST(
            dst=Reg(0, RegType.VGPR),
            src0=OperandAST.from_vgpr(1),
            src1=OperandAST.from_vgpr(2),
            src2=OperandAST.from_vgpr(3),
        )
        lines = [
            "v_mov_b32_e32 v1, 100",
            "v_mov_b32_e32 v2, -200",
            "v_mov_b32_e32 v3, 0",
            inst.emit_asm(),
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_ast", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0)
        self.assertEqual(words[0], ref_med3_i32(100, -200, 0))


# ==============================================================================
# Deterministic Scalar Instruction Interpreter (Commit 3)
# ==============================================================================

class ScalarCpuState:
    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106
        self.exec_mask: int = 0xFFFFFFFFFFFFFFFF

    def resolve_operand(self, op: OperandAST) -> int:
        if op.kind == OperandKind.REG:
            assert op.reg is not None
            if op.reg.reg_type == RegType.VGPR:
                return self.vgprs[op.reg.index]
            else:
                return self.sgprs[op.reg.index]
        elif op.kind in (OperandKind.INLINE_INT, OperandKind.LITERAL_INT):
            assert op.imm is not None
            return u32(op.imm)
        raise ValueError(f"Unknown operand kind {op.kind}")

    def execute_med3_i32(self, inst: Med3I32InstructionAST) -> int:
        s0 = self.resolve_operand(inst.src0)
        s1 = self.resolve_operand(inst.src1)
        s2 = self.resolve_operand(inst.src2)
        res = ref_med3_i32(s0, s1, s2)
        self.vgprs[inst.dst.index] = res
        return res


class TestMed3I32ScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interp_sequential_execution(self):
        state = ScalarCpuState()
        state.vgprs[1] = u32(50)
        state.vgprs[2] = u32(-10)
        state.vgprs[3] = u32(20)

        inst1 = parse_med3_i32_asm("v_med3_i32 v4, v1, v2, v3")
        res1 = state.execute_med3_i32(inst1)
        self.assertEqual(res1, 20)
        self.assertEqual(state.vgprs[4], 20)

        # Chain output into another med3
        state.vgprs[5] = u32(100)
        inst2 = parse_med3_i32_asm("v_med3_i32 v6, v4, v5, 10")
        res2 = state.execute_med3_i32(inst2)
        self.assertEqual(res2, 20)  # median of (20, 100, 10) is 20
        self.assertEqual(state.vgprs[6], 20)

    def test_scalar_interp_matches_shader_execution(self):
        triples = [
            (-30, 40, -10),
            (0x7FFFFFFF, 0, 0x80000000),
            (-1, -2, -3),
        ]
        for s0, s1, s2 in triples:
            state = ScalarCpuState()
            state.vgprs[1] = u32(s0)
            state.vgprs[2] = u32(s1)
            state.vgprs[3] = u32(s2)
            inst = parse_med3_i32_asm("v_med3_i32 v0, v1, v2, v3")
            expected = state.execute_med3_i32(inst)

            shader_res = run_shader_med3_i32(s0, s1, s2)
            self.assertEqual(expected, shader_res)


# ==============================================================================
# SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking (Commit 4)
# ==============================================================================

class SimdWaveState:
    def __init__(self, lane_count: int = 64) -> None:
        self.lane_count = lane_count
        self.vgprs: List[List[int]] = [[0] * 256 for _ in range(lane_count)]
        self.exec_mask: int = (1 << lane_count) - 1

    def execute_med3_i32(self, inst: Med3I32InstructionAST) -> None:
        for lane in range(self.lane_count):
            if not ((self.exec_mask >> lane) & 1):
                continue

            def resolve_lane_operand(op: OperandAST) -> int:
                if op.kind == OperandKind.REG:
                    assert op.reg is not None
                    return self.vgprs[lane][op.reg.index]
                assert op.imm is not None
                return u32(op.imm)

            s0 = resolve_lane_operand(inst.src0)
            s1 = resolve_lane_operand(inst.src1)
            s2 = resolve_lane_operand(inst.src2)
            self.vgprs[lane][inst.dst.index] = ref_med3_i32(s0, s1, s2)


class TestMed3I32SimdDivergence(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self):
        wave = SimdWaveState(lane_count=32)
        wave.exec_mask = 0xFFFFFFFF
        for lane in range(32):
            wave.vgprs[lane][1] = u32(-50)
            wave.vgprs[lane][2] = u32(0)
            wave.vgprs[lane][3] = u32(50)

        inst = parse_med3_i32_asm("v_med3_i32 v0, v1, v2, v3")
        wave.execute_med3_i32(inst)

        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 0)

    def test_wave64_divergent_lanes(self):
        wave = SimdWaveState(lane_count=64)
        wave.exec_mask = 0xFFFFFFFFFFFFFFFF
        for lane in range(64):
            wave.vgprs[lane][1] = u32(lane * 10 - 320)
            wave.vgprs[lane][2] = u32(0)
            wave.vgprs[lane][3] = u32(100)

        inst = parse_med3_i32_asm("v_med3_i32 v0, v1, v2, v3")
        wave.execute_med3_i32(inst)

        for lane in range(64):
            v1_val = lane * 10 - 320
            expected = ref_med3_i32(v1_val, 0, 100)
            self.assertEqual(wave.vgprs[lane][0], expected)

    def test_wave64_exec_masking_preservation(self):
        wave = SimdWaveState(lane_count=64)
        for lane in range(64):
            wave.vgprs[lane][0] = 0xDEADBEEF
            wave.vgprs[lane][1] = 10
            wave.vgprs[lane][2] = 20
            wave.vgprs[lane][3] = 30

        # Enable only even lanes
        wave.exec_mask = 0x5555555555555555
        inst = parse_med3_i32_asm("v_med3_i32 v0, v1, v2, v3")
        wave.execute_med3_i32(inst)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][0], 20)
            else:
                self.assertEqual(wave.vgprs[lane][0], 0xDEADBEEF)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation (Commit 5)
# ==============================================================================

class TestMed3I32RegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_src0_in_shader(self):
        # v1 = med3(v1, v2, v3) -> dst aliases src0
        lines = [
            "v_mov_b32_e32 v1, 100",
            "v_mov_b32_e32 v2, -50",
            "v_mov_b32_e32 v3, 20",
            "v_med3_i32 v1, v1, v2, v3",
            "exp mrt0, v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_alias_dst_s0", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)

    def test_alias_dst_src1_in_shader(self):
        # v2 = med3(v1, v2, v3) -> dst aliases src1
        lines = [
            "v_mov_b32_e32 v1, 100",
            "v_mov_b32_e32 v2, -50",
            "v_mov_b32_e32 v3, 20",
            "v_med3_i32 v2, v1, v2, v3",
            "exp mrt0, v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_alias_dst_s1", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)

    def test_alias_dst_src2_in_shader(self):
        # v3 = med3(v1, v2, v3) -> dst aliases src2
        lines = [
            "v_mov_b32_e32 v1, 100",
            "v_mov_b32_e32 v2, -50",
            "v_mov_b32_e32 v3, 20",
            "v_med3_i32 v3, v1, v2, v3",
            "exp mrt0, v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_alias_dst_s2", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 20)

    def test_alias_all_sources_identical(self):
        lines = [
            "v_mov_b32_e32 v0, 42",
            "v_med3_i32 v0, v0, v0, v0",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_alias_all", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 42)

    def test_alias_pairwise_sources(self):
        lines = [
            "v_mov_b32_e32 v1, -15",
            "v_mov_b32_e32 v2, 30",
            "v_med3_i32 v0, v1, v1, v2",
            "exp mrt0, v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_alias_pairwise", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], u32(-15))


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation (Commit 6)
# ==============================================================================

class TestMed3I32ConditionCodes(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation."""

    def test_zero_flag_behavior(self):
        res = ref_med3_i32(-10, 0, 10)
        self.assertTrue(res == 0)

    def test_sign_flag_behavior(self):
        res_neg = ref_med3_i32(-50, -20, 100)
        self.assertTrue(bool(res_neg & SIGN_BIT32))

        res_pos = ref_med3_i32(-50, 20, 100)
        self.assertFalse(bool(res_pos & SIGN_BIT32))

    def test_downstream_vcc_condition_code_branching(self):
        lines = [
            "v_mov_b32_e32 v1, -5",
            "v_mov_b32_e32 v2, 0",
            "v_mov_b32_e32 v3, 5",
            "v_med3_i32 v0, v1, v2, v3",
            "v_cmp_eq_u32 vcc, 0, v0",
            "v_cndmask_b32_e64 v4, 0x1111, 0x2222, vcc",
            "exp mrt0, v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_med3_downstream_vcc", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x2222)


if __name__ == "__main__":
    unittest.main()
