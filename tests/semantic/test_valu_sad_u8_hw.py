"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_sad_u8.

Models AMD RDNA2 / GFX10 Vector ALU (VALU) v_sad_u8 sum of absolute differences semantics:
  - Computes unsigned byte-level sum of absolute differences plus accumulator:
      D = S2 + |S0.byte0 - S1.byte0| + |S0.byte1 - S1.byte1| + |S0.byte2 - S1.byte2| + |S0.byte3 - S1.byte3|
  - Each byte difference is unsigned (the absolute diff of two 8-bit unsigned values).
  - The accumulator S2 and the final result are 32-bit unsigned (wrapping).
  - VOP3 encoding with 3 input operands (VGPR, SGPR, inline constants, 32-bit literals).
  - VALU operation: does not modify SCC or VCC.

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
MASK8 = 0xFF


def ref_sad_u8(s0: int, s1: int, s2: int) -> int:
    """Architectural reference oracle for RDNA2 v_sad_u8.

    Returns s2 + sum(|s0.byte_i - s1.byte_i| for i in 0..3), 32-bit unsigned wrapping.
    """
    s0 &= MASK32
    s1 &= MASK32
    s2 &= MASK32
    total = s2
    for shift in (0, 8, 16, 24):
        b0 = (s0 >> shift) & MASK8
        b1 = (s1 >> shift) & MASK8
        total += abs(b0 - b1)
    return total & MASK32


def run_ps_sad_u8(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_sad_u8_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_basic_sad_all_zero_diff(self):
        self.assertEqual(ref_sad_u8(0x01020304, 0x01020304, 0), 0)
        self.assertEqual(ref_sad_u8(0x01020304, 0x01020304, 100), 100)

    def test_basic_sad_known_values(self):
        self.assertEqual(ref_sad_u8(0x04030201, 0x00000000, 0), 10)
        self.assertEqual(ref_sad_u8(0x00000000, 0xFFFFFFFF, 0), 1020)

    def test_accumulator_addition(self):
        self.assertEqual(ref_sad_u8(0x04030201, 0x00000000, 50), 60)

    def test_translator_basic_execution(self):
        words = run_ps_sad_u8([
            "v_mov_b32 v0, 0x04030201",
            "v_mov_b32 v1, 0x00000000",
            "v_mov_b32 v2, 0",
            "v_sad_u8 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], 10)

    def test_translator_with_accumulator(self):
        words = run_ps_sad_u8([
            "v_mov_b32 v0, 0x04030201",
            "v_mov_b32 v1, 0x00000000",
            "v_mov_b32 v2, 50",
            "v_sad_u8 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], 60)


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
class VSadU8Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination must be a VGPR for VALU instruction v_sad_u8")

    def to_asm(self) -> str:
        return f"v_sad_u8 {self.dst.to_asm()}, {self.src0.to_asm()}, {self.src1.to_asm()}, {self.src2.to_asm()}"


class TestSymbolicAST(unittest.TestCase):
    def test_ast_construction_and_formatting(self):
        insn = VSadU8Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.sgpr(4),
            src2=Operand.literal(0),
        )
        self.assertEqual(insn.to_asm(), "v_sad_u8 v3, v0, s4, 0x0")

    def test_ast_immutability(self):
        insn = VSadU8Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        with self.assertRaises(Exception):
            insn.dst = Operand.vgpr(4)  # type: ignore

    def test_ast_validation_errors(self):
        with self.assertRaises(TypeError):
            VSadU8Instruction(
                dst=Operand.sgpr(3),
                src0=Operand.vgpr(0),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(2),
            )
        with self.assertRaises(ValueError):
            Operand.vgpr(256)


if __name__ == "__main__":
    unittest.main()
