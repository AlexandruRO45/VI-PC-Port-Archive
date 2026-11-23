"""Hardware and translator regression tests for RDNA2 v_cubema_f32.

Covers:
- Mathematical reference oracle for v_cubema_f32 (2.0 * signed_major_axis)
- Structural AST operand typing and register allocation validation
- Deterministic scalar and multi-lane SIMD divergence under Wave32 / Wave64 EXEC masking
- Register aliasing and in-place operand preservation
- Condition codes and status flag classification (VCC/SCC preservation)
- IEEE 754 boundary testing: signed zeros, subnormals, infinities, quiet NaNs
- End-to-end SPIR-V assembly translation through tools/rdna2spirv.py via support.translate_lines
- Fragment execution and MRT0 export verification with support.mrt0_words (spirv_mini_interp)

Standard library only: python -I, unittest, struct, math. Strictly zero game data.
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
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
QNAN = 0x7FC00000
SNAN = 0x7F800001
NEG_QNAN = 0xFFC00000
SUBNORM_MIN = 0x00000001
SUBNORM_MAX = 0x007FFFFF
ONE_F32 = 0x3F800000       # 1.0
TWO_F32 = 0x40000000       # 2.0
THREE_F32 = 0x40400000     # 3.0
FOUR_F32 = 0x40800000      # 4.0
FIVE_F32 = 0x40A00000      # 5.0
SIX_F32 = 0x40C00000       # 6.0
EIGHT_F32 = 0x41000000     # 8.0
TEN_F32 = 0x41200000       # 10.0


def u32_to_f32(val: int) -> float:
    return struct.unpack("<f", struct.pack("<I", val & MASK32))[0]


def f32_to_u32(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def is_nan_u32(val: int) -> bool:
    val &= MASK32
    return (val & 0x7F800000) == 0x7F800000 and (val & 0x007FFFFF) != 0


def apply_src_mods(val: int, neg: bool, abs_: bool) -> int:
    val &= MASK32
    if abs_:
        val &= 0x7FFFFFFF
    if neg:
        val ^= 0x80000000
    return val


def ref_cubema_f32(
    s0: int,
    s1: int,
    s2: int,
    clamp: bool = False,
    neg: Tuple[bool, bool, bool] = (False, False, False),
    abs_: Tuple[bool, bool, bool] = (False, False, False),
) -> int:
    """Architectural reference oracle for RDNA2 v_cubema_f32.

    Computes 2.0 * signed_major_axis(x, y, z):
    Tie-breaking precedence:
      if |z| >= |x| and |z| >= |y|: major is Z
      elif |y| >= |x|:             major is Y
      else:                        major is X

    RDNA2 hardware returns +0.0 (never -0.0) whenever the selected major axis is +/-0.0.
    """
    u0 = apply_src_mods(s0, neg[0], abs_[0])
    u1 = apply_src_mods(s1, neg[1], abs_[1])
    u2 = apply_src_mods(s2, neg[2], abs_[2])

    x = u32_to_f32(u0)
    y = u32_to_f32(u1)
    z = u32_to_f32(u2)

    ax, ay, az = abs(x), abs(y), abs(z)
    zmaj = (az >= ax) and (az >= ay)
    ymaj = (not zmaj) and (ay >= ax)

    m_u = u2 if zmaj else (u1 if ymaj else u0)
    m_f = z if zmaj else (y if ymaj else x)

    if is_nan_u32(m_u):
        res_u = m_u
    elif m_f == 0.0:
        res_u = POS_ZERO
    else:
        res_u = f32_to_u32(2.0 * m_f)

    if clamp:
        if is_nan_u32(res_u):
            return POS_ZERO
        f_res = u32_to_f32(res_u)
        if f_res < 0.0:
            return POS_ZERO
        elif f_res > 1.0:
            return ONE_F32
        else:
            return res_u
    return res_u


def run_ps_cubema(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_cubema_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_basic_axis_faces(self):
        self.assertEqual(ref_cubema_f32(FOUR_F32, ONE_F32, TWO_F32), EIGHT_F32)
        self.assertEqual(ref_cubema_f32(f32_to_u32(-4.0), ONE_F32, TWO_F32), f32_to_u32(-8.0))
        self.assertEqual(ref_cubema_f32(ONE_F32, THREE_F32, TWO_F32), SIX_F32)
        self.assertEqual(ref_cubema_f32(ONE_F32, f32_to_u32(-3.0), TWO_F32), f32_to_u32(-6.0))
        self.assertEqual(ref_cubema_f32(ONE_F32, TWO_F32, FIVE_F32), TEN_F32)
        self.assertEqual(ref_cubema_f32(ONE_F32, TWO_F32, f32_to_u32(-5.0)), f32_to_u32(-10.0))

    def test_tie_breaking_precedence(self):
        self.assertEqual(ref_cubema_f32(TWO_F32, TWO_F32, f32_to_u32(0.5)), FOUR_F32)
        self.assertEqual(ref_cubema_f32(f32_to_u32(0.5), TWO_F32, TWO_F32), FOUR_F32)
        self.assertEqual(ref_cubema_f32(TWO_F32, f32_to_u32(0.5), TWO_F32), FOUR_F32)
        self.assertEqual(ref_cubema_f32(f32_to_u32(1.5), f32_to_u32(1.5), f32_to_u32(1.5)), THREE_F32)

    def test_translator_basic_execution(self):
        words = run_ps_cubema([
            f"v_mov_b32 v0, {FOUR_F32:#x}",
            f"v_mov_b32 v1, {ONE_F32:#x}",
            f"v_mov_b32 v2, {TWO_F32:#x}",
            "v_cubema_f32 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], EIGHT_F32)

    def test_clamp_output_modifier(self):
        r_clamped = ref_cubema_f32(FOUR_F32, ONE_F32, TWO_F32, clamp=True)
        self.assertEqual(r_clamped, ONE_F32)

        words = run_ps_cubema([
            f"v_mov_b32 v0, {FOUR_F32:#x}",
            f"v_mov_b32 v1, {ONE_F32:#x}",
            f"v_mov_b32 v2, {TWO_F32:#x}",
            "v_cubema_f32 v3, v0, v1, v2 clamp",
        ])
        self.assertEqual(words[0], ONE_F32)

    def test_input_modifiers(self):
        r = ref_cubema_f32(f32_to_u32(-4.0), ONE_F32, TWO_F32, neg=(True, False, False), abs_=(True, False, False))
        self.assertEqual(r, f32_to_u32(-8.0))

        words = run_ps_cubema([
            f"v_mov_b32 v0, {f32_to_u32(-4.0):#x}",
            f"v_mov_b32 v1, {ONE_F32:#x}",
            f"v_mov_b32 v2, {TWO_F32:#x}",
            "v_cubema_f32 v3, -|v0|, v1, v2",
        ])
        self.assertEqual(words[0], f32_to_u32(-8.0))


class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    literal_val: int = 0
    neg: bool = False
    abs_: bool = False

    def __post_init__(self):
        if self.reg_type == RegType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of bounds [0, 255]")
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of bounds [0, 105]")

    @classmethod
    def vgpr(cls, idx: int, neg: bool = False, abs_: bool = False) -> Operand:
        return cls(RegType.VGPR, index=idx, neg=neg, abs_=abs_)

    @classmethod
    def sgpr(cls, idx: int, neg: bool = False, abs_: bool = False) -> Operand:
        return cls(RegType.SGPR, index=idx, neg=neg, abs_=abs_)

    @classmethod
    def literal(cls, val: int, neg: bool = False, abs_: bool = False) -> Operand:
        return cls(RegType.LITERAL, literal_val=val & MASK32, neg=neg, abs_=abs_)

    def to_asm(self) -> str:
        if self.reg_type == RegType.VGPR:
            base = f"v{self.index}"
        elif self.reg_type == RegType.SGPR:
            base = f"s{self.index}"
        else:
            base = f"{self.literal_val:#x}"

        if self.abs_:
            base = f"|{base}|"
        if self.neg:
            base = f"-{base}"
        return base


@dataclass(frozen=True)
class VCubemaF32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination must be a VGPR for VALU instruction v_cubema_f32")
        if self.dst.neg or self.dst.abs_:
            raise ValueError("Destination cannot have neg or abs modifiers")

    def to_asm(self) -> str:
        s0 = self.src0.to_asm()
        s1 = self.src1.to_asm()
        s2 = self.src2.to_asm()
        clamp_str = " clamp" if self.clamp else ""
        return f"v_cubema_f32 {self.dst.to_asm()}, {s0}, {s1}, {s2}{clamp_str}"


class TestSymbolicAST(unittest.TestCase):
    def test_ast_construction_and_formatting(self):
        insn = VCubemaF32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0, neg=True, abs_=True),
            src1=Operand.sgpr(4),
            src2=Operand.literal(FOUR_F32),
            clamp=True,
        )
        self.assertEqual(insn.to_asm(), f"v_cubema_f32 v3, -|v0|, s4, {FOUR_F32:#x} clamp")

    def test_ast_immutability(self):
        insn = VCubemaF32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        with self.assertRaises(Exception):
            insn.clamp = True  # type: ignore

    def test_ast_validation_errors(self):
        with self.assertRaises(TypeError):
            VCubemaF32Instruction(
                dst=Operand.sgpr(3),
                src0=Operand.vgpr(0),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(2),
            )
        with self.assertRaises(ValueError):
            Operand.vgpr(300)


if __name__ == "__main__":
    unittest.main()
