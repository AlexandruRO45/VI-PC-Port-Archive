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


if __name__ == "__main__":
    unittest.main()
