from __future__ import annotations

"""Semantic reference oracle and architectural model for v_fmac_f16.

RDNA2 / GFX10 VOP2 / VOP3 Half-Precision Floating-Point Fused Multiply-Accumulate:
Computes dst = fma(src0, src1, dst) = (src0 * src1) + dst on 16-bit floats.
The accumulator is dst itself. Under GFX10 rules, 16-bit scalar float operations
write one 16-bit half (low half by default, or high half when op_sel specifies dst_hi)
and strictly PRESERVE the untouched half of the 32-bit destination VGPR.

Syntax:
    v_fmac_f16 vdst, vsrc0, vsrc1 [op_sel:[s0,s1,s2,dhi]] [clamp]

Semantics:
    s0 = (src0 >> 16) if op_sel[0] else src0
    s1 = (src1 >> 16) if op_sel[1] else src1
    acc = (vdst >> 16) if op_sel[2] else vdst
    res = fma(s0, s1, acc)
    if clamp: res = clamp01(res)
    vdst = (vdst & 0xFFFF) | (res << 16) if op_sel[3] else (vdst & 0xFFFF0000) | res
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
F16_THREE: int = 0x4200
F16_FOUR: int = 0x4400
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

    if (s0_is_zero and s1_is_inf) or (s0_is_inf and s1_is_zero):
        return F16_QNAN

    prod_sign = ((s0 >> 15) ^ (s1 >> 15)) & 1
    prod_is_inf = s0_is_inf or s1_is_inf

    if prod_is_inf and s2_is_inf:
        s2_sign = (s2 >> 15) & 1
        if prod_sign != s2_sign:
            return F16_QNAN

    x = u16_to_f16(s0)
    y = u16_to_f16(s1)
    z = u16_to_f16(s2)

    res = x * y + z

    if clamp:
        if math.isnan(res):
            return F16_POS_ZERO
        res = max(0.0, min(1.0, res))

    return f16_to_u16(res)


def oracle_fmac_f16(
    dst: int,
    src0: int,
    src1: int,
    *,
    op_sel: Tuple[int, int, int, int] = (0, 0, 0, 0),
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for v_fmac_f16."""
    dst = u32(dst)
    src0 = u32(src0)
    src1 = u32(src1)

    s0_val = (src0 >> 16) & MASK_16 if op_sel[0] else (src0 & MASK_16)
    s1_val = (src1 >> 16) & MASK_16 if op_sel[1] else (src1 & MASK_16)
    acc_val = (dst >> 16) & MASK_16 if op_sel[2] else (dst & MASK_16)

    res_f16 = fma_f16(s0_val, s1_val, acc_val, clamp=clamp)

    dhi = op_sel[3] if len(op_sel) > 3 else 0
    if dhi:
        # Write high half [31:16], preserve low half [15:0]
        return (res_f16 << 16) | (dst & MASK_16)
    else:
        # Write low half [15:0], preserve high half [31:16]
        return (dst & 0xFFFF_0000) | res_f16


def run_translator_fmac(
    dst: int,
    src0: int,
    src1: int,
    mods: str = "",
    name: str = "ps_fmac_f16",
) -> int:
    """Translate and run a minimal shader through tools/rdna2spirv and return result word."""
    lines = [
        f"v_mov_b32 v0, {dst:#x}",
        f"v_mov_b32 v1, {src0:#x}",
        f"v_mov_b32 v2, {src1:#x}",
        f"v_fmac_f16 v0, v1, v2 {mods}".strip(),
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestOracleAndHardwareModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_basic_fmac_arithmetic(self) -> None:
        """Verify dst = 1.0 * 2.0 + 0.5 = 2.5 (0x4100)."""
        dst = F16_HALF
        src0 = F16_ONE
        src1 = F16_TWO
        got = oracle_fmac_f16(dst, src0, src1)
        self.assertEqual(got, 0x4100)

    def test_destination_high_half_preservation(self) -> None:
        """Verify low-half write strictly preserves high half 0xBEEF."""
        dst = 0xBEEF_3800  # high=0xBEEF, low=0.5
        src0 = F16_ONE
        src1 = F16_TWO
        got = oracle_fmac_f16(dst, src0, src1)
        self.assertEqual(got, 0xBEEF_4100)

    def test_clamp_modifier(self) -> None:
        """Verify clamp restricts output to [0.0, 1.0]."""
        dst = F16_POS_ZERO
        src0 = F16_TWO
        src1 = F16_TWO
        got = oracle_fmac_f16(dst, src0, src1, clamp=True)
        self.assertEqual(got, F16_ONE)

    def test_oracle_matches_translator(self) -> None:
        """Verify oracle output directly matches translated SPIR-V."""
        dst = 0xCAFE_3800
        src0 = F16_ONE
        src1 = F16_TWO
        expected = oracle_fmac_f16(dst, src0, src1)
        translated = run_translator_fmac(dst, src0, src1, name="ps_fmac_oracle")
        self.assertEqual(translated, expected)

    def test_fused_rounding_discriminators(self) -> None:
        """Verify single-rounding vs intermediate rounding:
        dst = 0xA55ABC00 (-1.0), src0 = 0x3C01 (1025/1024), src1 = 0x3BFE (1023/1024).
        Single-rounded FMA yields -2^(-20) = 0x8010 with high half 0xA55A preserved (0xA55A8010).
        If product was rounded first to 1.0, 1.0 + (-1.0) would incorrectly yield 0xA55A0000.
        """
        dst = 0xA55A_BC00
        src0 = 0x3C01
        src1 = 0x3BFE
        expected = 0xA55A_8010
        got = oracle_fmac_f16(dst, src0, src1)
        self.assertEqual(got, expected)

        # Second discriminator: sentinel 0x5AA5
        dst2 = 0x5AA5_BC00
        got2 = oracle_fmac_f16(dst2, src0, src1)
        self.assertEqual(got2, 0x5AA5_8010)


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
class VOP3Modifiers:
    op_sel: Tuple[int, int, int, int] = (0, 0, 0, 0)
    clamp: bool = False

    def to_assembly(self) -> str:
        parts: List[str] = []
        if self.op_sel != (0, 0, 0, 0):
            parts.append(f"op_sel:[{self.op_sel[0]},{self.op_sel[1]},{self.op_sel[2]},{self.op_sel[3]}]")
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)


@dataclass(frozen=True)
class VFMAC16Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    mods: VOP3Modifiers = field(default_factory=VOP3Modifiers)

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
        m = self.mods.to_assembly()
        suffix = f" {m}" if m else ""
        return f"v_fmac_f16 {d}, {s0}, {s1}{suffix}"


class TestASTDefinitions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        """Verify AST correctly serializes to valid RDNA2 assembly text."""
        instr = VFMAC16Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
            mods=VOP3Modifiers(clamp=True, op_sel=(1, 1, 0, 1)),
        )
        asm = instr.to_assembly()
        self.assertIn("v_fmac_f16 v0, v1, v2", asm)
        self.assertIn("clamp", asm)
        self.assertIn("op_sel:[1,1,0,1]", asm)

    def test_ast_translation_validity(self) -> None:
        """Verify that assembly generated by AST translates cleanly."""
        instr = VFMAC16Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
        )
        lines = [
            "v_mov_b32 v0, 0x00003800",
            "v_mov_b32 v1, 0x00003c00",
            "v_mov_b32 v2, 0x00004000",
            instr.to_assembly(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_fmac_valid", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x4100)
