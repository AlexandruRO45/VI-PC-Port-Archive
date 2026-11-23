"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_bfe_i32.

Models AMD RDNA2 / GFX10 Vector ALU (VALU) v_bfe_i32 signed bitfield extract semantics:
  - Computes signed bitfield extract: D = (S0.i >> (S1.u & 31)) with bitfield length (S2.u & 31)
  - Extracted field is sign-extended to 32-bit two's-complement integer.
  - If width is 0, the result is 0.
  - VOP3 encoding with VGPR destination and VGPR/SGPR/literal sources.
  - VALU operation: scalar condition code (SCC) and vector condition code (VCC) remain untouched.

Standard library Python only (unittest, struct, math, dataclasses, enum), strictly zero game data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Optional, Sequence, Tuple
import unittest

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

MASK32 = 0xFFFFFFFF


def ref_bfe_i32(base: int, offset: int, width: int) -> int:
    """Architectural reference oracle for RDNA2 v_bfe_i32 signed bitfield extract."""
    base &= MASK32
    offset &= 31
    width &= 31
    if width == 0:
        return 0
    signed = base - (1 << 32) if (base >> 31) else base
    field = (signed >> offset) & ((1 << width) - 1)
    if field & (1 << (width - 1)):
        field -= 1 << width
    return field & MASK32


def run_ps_bfe_i32(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_bfe_i32_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_basic_signed_bitfield_extract(self):
        self.assertEqual(ref_bfe_i32(0x12345678, 4, 8), 0x67)
        self.assertEqual(ref_bfe_i32(0xFFFFFFF0, 2, 4), 0xFFFFFFFC)
        self.assertEqual(ref_bfe_i32(0x80000000, 31, 1), 0xFFFFFFFF)
        self.assertEqual(ref_bfe_i32(0x7FFFFFFF, 31, 1), 0)

    def test_width_zero_yields_zero(self):
        self.assertEqual(ref_bfe_i32(0xFFFFFFFF, 0, 0), 0)
        self.assertEqual(ref_bfe_i32(0x12345678, 12, 0), 0)
        self.assertEqual(ref_bfe_i32(0x80000000, 31, 0), 0)

    def test_translator_basic_execution(self):
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 4",
            "v_mov_b32 v2, 8",
            "v_bfe_i32 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], 0x67)

    def test_translator_sign_extended_negative(self):
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xFFFFFFF0",
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 4",
            "v_bfe_i32 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], 0xFFFFFFFC)


class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    literal_val: int = 0

    def __post_init__(self):
        if self.reg_type == RegType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of bounds [0, 255]")
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of bounds [0, 105]")

    @classmethod
    def vgpr(cls, idx: int) -> Operand:
        return cls(RegType.VGPR, index=idx)

    @classmethod
    def sgpr(cls, idx: int) -> Operand:
        return cls(RegType.SGPR, index=idx)

    @classmethod
    def literal(cls, val: int) -> Operand:
        return cls(RegType.LITERAL, literal_val=val & MASK32)

    def to_asm(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v{self.index}"
        elif self.reg_type == RegType.SGPR:
            return f"s{self.index}"
        else:
            return f"{self.literal_val:#x}"


@dataclass(frozen=True)
class VBfeI32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination must be a VGPR for VALU instruction v_bfe_i32")

    def to_asm(self) -> str:
        return f"v_bfe_i32 {self.dst.to_asm()}, {self.src0.to_asm()}, {self.src1.to_asm()}, {self.src2.to_asm()}"


class TestSymbolicAST(unittest.TestCase):
    def test_ast_construction_and_formatting(self):
        insn = VBfeI32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.sgpr(4),
            src2=Operand.literal(8),
        )
        self.assertEqual(insn.to_asm(), "v_bfe_i32 v3, v0, s4, 0x8")

    def test_ast_immutability(self):
        insn = VBfeI32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        with self.assertRaises(Exception):
            insn.dst = Operand.vgpr(4)  # type: ignore

    def test_ast_validation_errors(self):
        with self.assertRaises(TypeError):
            VBfeI32Instruction(
                dst=Operand.sgpr(3),
                src0=Operand.vgpr(0),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(2),
            )
        with self.assertRaises(ValueError):
            Operand.vgpr(256)


class ScalarRegisterState:
    def __init__(self):
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106
        self.vcc = 0
        self.scc = 0

    def read_operand(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index]
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index]
        else:
            return op.literal_val

    def write_vgpr(self, idx: int, val: int):
        self.vgprs[idx] = val & MASK32


class ScalarInterpreter:
    def __init__(self, state: Optional[ScalarRegisterState] = None):
        self.state = state if state is not None else ScalarRegisterState()

    def step(self, insn: VBfeI32Instruction) -> int:
        u0 = self.state.read_operand(insn.src0)
        u1 = self.state.read_operand(insn.src1)
        u2 = self.state.read_operand(insn.src2)
        res = ref_bfe_i32(u0, u1, u2)
        self.state.write_vgpr(insn.dst.index, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step_and_chaining(self):
        interp = ScalarInterpreter()
        interp.state.vgprs[0] = 0x12345678
        interp.state.vgprs[1] = 4
        interp.state.vgprs[2] = 8
        interp.state.vgprs[4] = 0
        interp.state.vgprs[5] = 4

        # Step 1: v3 = v_bfe_i32 v0, v1, v2 -> (0x12345678 >> 4) & 0xFF = 0x67
        insn1 = VBfeI32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        res1 = interp.step(insn1)
        self.assertEqual(res1, 0x67)
        self.assertEqual(interp.state.vgprs[3], 0x67)

        # Step 2: v6 = v_bfe_i32 v3, v4, v5 -> (0x67 >> 0) & 0xF = 0x7
        insn2 = VBfeI32Instruction(
            dst=Operand.vgpr(6),
            src0=Operand.vgpr(3),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr(5),
        )
        res2 = interp.step(insn2)
        self.assertEqual(res2, 0x7)
        self.assertEqual(interp.state.vgprs[6], 0x7)

    def test_scalar_interpreter_matches_translator_chain(self):
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 4",
            "v_mov_b32 v2, 8",
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 4",
            "v_bfe_i32 v3, v0, v1, v2",
            "v_bfe_i32 v3, v3, v4, v5",
        ]
        words = run_ps_bfe_i32(lines)
        self.assertEqual(words[0], 0x7)


if __name__ == "__main__":
    unittest.main()
