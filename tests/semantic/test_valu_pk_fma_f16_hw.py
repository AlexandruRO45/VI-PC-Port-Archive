from __future__ import annotations

"""Semantic reference oracle and architectural model for v_pk_fma_f16.

RDNA2 / GFX10 VOP3P Packed 16-bit Floating-Point Fused Multiply-Add:
Performs dual half-precision (IEEE 754-2008 binary16) fused multiply-accumulate
operations in parallel on the lower [15:0] and upper [31:16] halves of 32-bit VGPRs.

Syntax:
    v_pk_fma_f16 vdst, vsrc0, vsrc1, vsrc2 [op_sel:[s0,s1,s2]] [op_sel_hi:[s0,s1,s2]] [neg_lo:[n0,n1,n2]] [neg_hi:[n0,n1,n2]] [clamp]

Semantics:
    lo = fma(s0_lo, s1_lo, s2_lo)
    hi = fma(s0_hi, s1_hi, s2_hi)
    if clamp: lo = clamp01(lo); hi = clamp01(hi)
    vdst = (hi << 16) | lo
"""

import math
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

# IEEE 754-2008 Binary16 Constants
F16_POS_ZERO: int = 0x0000
F16_NEG_ZERO: int = 0x8000
F16_ONE: int = 0x3C00
F16_TWO: int = 0x4000
F16_HALF: int = 0x3800
F16_MAX_NORMAL: int = 0x7BFF
F16_MIN_NORMAL: int = 0x0400
F16_MIN_SUBNORMAL: int = 0x0001
F16_MAX_SUBNORMAL: int = 0x03FF
F16_POS_INF: int = 0x7C00
F16_NEG_INF: int = 0xFC00
F16_QNAN: int = 0x7E00
F16_SNAN: int = 0x7C01

MASK_16: int = 0xFFFF
MASK_32: int = 0xFFFF_FFFF


def u16(val: int) -> int:
    """Normalize integer to unsigned 16-bit word."""
    return val & MASK_16


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK_32


def u16_to_f16(val: int) -> float:
    """Unpack 16-bit unsigned integer to Python float."""
    return struct.unpack("<e", struct.pack("<H", u16(val)))[0]


def f16_to_u16(val: float) -> int:
    """Pack Python float to IEEE 754 binary16 16-bit unsigned integer."""
    if math.isnan(val):
        return F16_QNAN
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return F16_POS_INF if val > 0 else F16_NEG_INF


def unpack_f16x2(word: int) -> Tuple[int, int]:
    """Split 32-bit dword into (lo, hi) 16-bit halves."""
    word = u32(word)
    return word & MASK_16, (word >> 16) & MASK_16


def pack_f16x2(lo: int, hi: int) -> int:
    """Pack (lo, hi) 16-bit halves into a 32-bit dword."""
    return (u16(hi) << 16) | u16(lo)


def fma_f16(s0: int, s1: int, s2: int, clamp: bool = False) -> int:
    """Bit-accurate IEEE 754 half-precision FMA matching RDNA2 hardware semantics."""
    s0, s1, s2 = u16(s0), u16(s1), u16(s2)

    s0_exp, s0_mant = (s0 >> 10) & 0x1F, s0 & 0x3FF
    s1_exp, s1_mant = (s1 >> 10) & 0x1F, s1 & 0x3FF
    s2_exp, s2_mant = (s2 >> 10) & 0x1F, s2 & 0x3FF

    s0_is_nan = (s0_exp == 0x1F and s0_mant != 0)
    s1_is_nan = (s1_exp == 0x1F and s1_mant != 0)
    s2_is_nan = (s2_exp == 0x1F and s2_mant != 0)

    if s0_is_nan or s1_is_nan or s2_is_nan:
        return F16_QNAN

    s0_is_zero = ((s0 & 0x7FFF) == 0)
    s1_is_zero = ((s1 & 0x7FFF) == 0)
    s0_is_inf = (s0_exp == 0x1F and s0_mant == 0)
    s1_is_inf = (s1_exp == 0x1F and s1_mant == 0)
    s2_is_inf = (s2_exp == 0x1F and s2_mant == 0)

    # 0 * Inf or Inf * 0 is invalid -> NaN
    if (s0_is_zero and s1_is_inf) or (s0_is_inf and s1_is_zero):
        return F16_QNAN

    prod_sign = ((s0 >> 15) ^ (s1 >> 15)) & 1
    prod_is_inf = s0_is_inf or s1_is_inf

    # (+Inf) + (-Inf) is invalid -> NaN
    if prod_is_inf and s2_is_inf:
        s2_sign = (s2 >> 15) & 1
        if prod_sign != s2_sign:
            return F16_QNAN

    # Pure floating point arithmetic
    x = u16_to_f16(s0)
    y = u16_to_f16(s1)
    z = u16_to_f16(s2)

    res = x * y + z

    if clamp:
        if math.isnan(res):
            return F16_POS_ZERO
        res = max(0.0, min(1.0, res))

    return f16_to_u16(res)


def oracle_pk_fma_f16(
    src0: int,
    src1: int,
    src2: int,
    *,
    op_sel: Tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: Tuple[int, int, int] = (1, 1, 1),
    neg_lo: Tuple[int, int, int] = (0, 0, 0),
    neg_hi: Tuple[int, int, int] = (0, 0, 0),
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for v_pk_fma_f16 with all VOP3P modifiers."""
    srcs = [u32(src0), u32(src1), u32(src2)]

    # Lower 16-bit half
    lo_operands: List[int] = []
    for i in range(3):
        h = (srcs[i] >> 16) & MASK_16 if op_sel[i] else (srcs[i] & MASK_16)
        if neg_lo[i]:
            h ^= 0x8000
        lo_operands.append(h)
    lo_res = fma_f16(lo_operands[0], lo_operands[1], lo_operands[2], clamp=clamp)

    # Upper 16-bit half
    hi_operands: List[int] = []
    for i in range(3):
        h = (srcs[i] >> 16) & MASK_16 if op_sel_hi[i] else (srcs[i] & MASK_16)
        if neg_hi[i]:
            h ^= 0x8000
        hi_operands.append(h)
    hi_res = fma_f16(hi_operands[0], hi_operands[1], hi_operands[2], clamp=clamp)

    return pack_f16x2(lo_res, hi_res)


def run_translator_pk_fma(
    src0: int,
    src1: int,
    src2: int,
    mods: str = "",
    name: str = "ps_pk_fma",
) -> int:
    """Translate and run a minimal shader through tools/rdna2spirv and return result word."""
    lines = [
        f"v_mov_b32 v1, {src0:#x}",
        f"v_mov_b32 v2, {src1:#x}",
        f"v_mov_b32 v3, {src2:#x}",
        f"v_pk_fma_f16 v0, v1, v2, v3 {mods}".strip(),
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestOracleAndHardwareModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_basic_fma_arithmetic(self) -> None:
        """Verify 1.0 * 2.0 + 0.5 = 2.5 (0x4100)."""
        res = fma_f16(F16_ONE, F16_TWO, F16_HALF)
        self.assertEqual(res, 0x4100)

    def test_packed_dual_channel(self) -> None:
        """Verify simultaneous computation in both 16-bit halves."""
        p0 = pack_f16x2(F16_ONE, F16_TWO)       # lo=1.0, hi=2.0
        p1 = pack_f16x2(F16_TWO, F16_ONE)       # lo=2.0, hi=1.0
        p2 = pack_f16x2(F16_HALF, F16_HALF)     # lo=0.5, hi=0.5
        # lo: 1.0 * 2.0 + 0.5 = 2.5 (0x4100)
        # hi: 2.0 * 1.0 + 0.5 = 2.5 (0x4100)
        got = oracle_pk_fma_f16(p0, p1, p2)
        self.assertEqual(got, 0x41004100)

    def test_clamp_modifier(self) -> None:
        """Verify clamp restricts values to [0.0, 1.0]."""
        # 2.0 * 2.0 + 0.0 = 4.0 -> clamped to 1.0 (0x3C00)
        # -1.0 * 2.0 + 0.0 = -2.0 -> clamped to 0.0 (0x0000)
        p0 = pack_f16x2(F16_TWO, 0xBC00)        # lo=2.0, hi=-1.0
        p1 = pack_f16x2(F16_TWO, F16_TWO)       # lo=2.0, hi=2.0
        p2 = pack_f16x2(F16_POS_ZERO, F16_POS_ZERO)
        got = oracle_pk_fma_f16(p0, p1, p2, clamp=True)
        self.assertEqual(got, 0x00003C00)

    def test_oracle_matches_translator(self) -> None:
        """Verify oracle output directly against translator output."""
        p0 = pack_f16x2(F16_ONE, F16_TWO)
        p1 = pack_f16x2(F16_TWO, F16_ONE)
        p2 = pack_f16x2(F16_HALF, F16_HALF)
        expected = oracle_pk_fma_f16(p0, p1, p2)
        translated = run_translator_pk_fma(p0, p1, p2, name="ps_pk_fma_oracle")
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
class VOP3PModifiers:
    op_sel: Tuple[int, int, int] = (0, 0, 0)
    op_sel_hi: Tuple[int, int, int] = (1, 1, 1)
    neg_lo: Tuple[int, int, int] = (0, 0, 0)
    neg_hi: Tuple[int, int, int] = (0, 0, 0)
    clamp: bool = False

    def to_assembly(self) -> str:
        parts: List[str] = []
        if self.op_sel != (0, 0, 0):
            parts.append(f"op_sel:[{self.op_sel[0]},{self.op_sel[1]},{self.op_sel[2]}]")
        if self.op_sel_hi != (1, 1, 1):
            parts.append(f"op_sel_hi:[{self.op_sel_hi[0]},{self.op_sel_hi[1]},{self.op_sel_hi[2]}]")
        if self.neg_lo != (0, 0, 0):
            parts.append(f"neg_lo:[{self.neg_lo[0]},{self.neg_lo[1]},{self.neg_lo[2]}]")
        if self.neg_hi != (0, 0, 0):
            parts.append(f"neg_hi:[{self.neg_hi[0]},{self.neg_hi[1]},{self.neg_hi[2]}]")
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)


@dataclass(frozen=True)
class VPKFMA16Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    mods: VOP3PModifiers = field(default_factory=VOP3PModifiers)

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
        s2 = self.format_operand(self.src2)
        m = self.mods.to_assembly()
        suffix = f" {m}" if m else ""
        return f"v_pk_fma_f16 {d}, {s0}, {s1}, {s2}{suffix}"


class TestASTDefinitions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        """Verify AST correctly serializes to valid RDNA2 assembly text."""
        instr = VPKFMA16Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
            src2=VGPR(3),
            mods=VOP3PModifiers(clamp=True, op_sel=(1, 0, 0), op_sel_hi=(0, 1, 0)),
        )
        asm = instr.to_assembly()
        self.assertIn("v_pk_fma_f16 v0, v1, v2, v3", asm)
        self.assertIn("clamp", asm)
        self.assertIn("op_sel:[1,0,0]", asm)
        self.assertIn("op_sel_hi:[0,1,0]", asm)

    def test_ast_translation_validity(self) -> None:
        """Verify that assembly generated by AST translates cleanly."""
        instr = VPKFMA16Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
            src2=VGPR(3),
        )
        lines = [
            "v_mov_b32 v1, 0x3c003c00",
            "v_mov_b32 v2, 0x40004000",
            "v_mov_b32 v3, 0x38003800",
            instr.to_assembly(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_valid", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x41004100)
