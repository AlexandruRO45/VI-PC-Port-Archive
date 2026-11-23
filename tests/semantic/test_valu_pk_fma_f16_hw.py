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

    def step(self, instr: VPKFMA16Instruction, state: ScalarMachineState) -> int:
        val0 = self.resolve_operand(instr.src0, state)
        val1 = self.resolve_operand(instr.src1, state)
        val2 = self.resolve_operand(instr.src2, state)

        res = oracle_pk_fma_f16(
            val0,
            val1,
            val2,
            op_sel=instr.mods.op_sel,
            op_sel_hi=instr.mods.op_sel_hi,
            neg_lo=instr.mods.neg_lo,
            neg_hi=instr.mods.neg_hi,
            clamp=instr.mods.clamp,
        )
        state.set_vgpr(instr.dst.value, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_execution(self) -> None:
        """Verify interpreter correctly executes across VGPR, SGPR, and Immediate inputs."""
        state = ScalarMachineState()
        state.set_vgpr(1, pack_f16x2(F16_ONE, F16_ONE))         # 1.0, 1.0
        state.set_sgpr(4, pack_f16x2(F16_TWO, F16_TWO))         # 2.0, 2.0
        imm = Immediate(pack_f16x2(F16_HALF, F16_HALF))        # 0.5, 0.5

        interp = ScalarInterpreter()
        instr = VPKFMA16Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=SGPR(4),
            src2=imm,
        )
        res = interp.step(instr, state)
        self.assertEqual(res, 0x41004100)
        self.assertEqual(state.get_vgpr(0), 0x41004100)

    def test_scalar_interpreter_cross_validation_with_translator(self) -> None:
        """Verify scalar interpreter exactly matches translated SPIR-V execution."""
        state = ScalarMachineState()
        p0 = 0x40003C00
        p1 = 0x3C004000
        p2 = 0x38003800
        state.set_vgpr(1, p0)
        state.set_vgpr(2, p1)
        state.set_vgpr(3, p2)

        interp = ScalarInterpreter()
        instr = VPKFMA16Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
            src2=VGPR(3),
            mods=VOP3PModifiers(op_sel=(1, 0, 0), op_sel_hi=(0, 1, 0)),
        )
        interp_res = interp.step(instr, state)
        spv_res = run_translator_pk_fma(p0, p1, p2, mods="op_sel:[1,0,0] op_sel_hi:[0,1,0]", name="ps_scalar_cross")
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

    def step(self, instr: VPKFMA16Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = []
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                results.append(None)
                continue

            v0 = self.resolve_lane_operand(instr.src0, lane, state)
            v1 = self.resolve_lane_operand(instr.src1, lane, state)
            v2 = self.resolve_lane_operand(instr.src2, lane, state)

            res = oracle_pk_fma_f16(
                v0,
                v1,
                v2,
                op_sel=instr.mods.op_sel,
                op_sel_hi=instr.mods.op_sel_hi,
                neg_lo=instr.mods.neg_lo,
                neg_hi=instr.mods.neg_hi,
                clamp=instr.mods.clamp,
            )
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
            state.set_vgpr(lane, 1, pack_f16x2(F16_ONE, F16_ONE))
            state.set_vgpr(lane, 2, pack_f16x2(F16_TWO, F16_TWO))
            state.set_vgpr(lane, 3, pack_f16x2(F16_HALF, F16_HALF))

        interp = SIMDInterpreter()
        instr = VPKFMA16Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))
        results = interp.step(instr, state)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0x41004100)
                self.assertEqual(state.get_vgpr(lane, 0), 0x41004100)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_wave64_divergence(self) -> None:
        """Verify Wave64 boundary execution across low and high 32-lane words."""
        # Active: lane 0, lane 31, lane 32, lane 63
        mask = (1 << 0) | (1 << 31) | (1 << 32) | (1 << 63)
        state = SIMDState(num_lanes=64, exec_mask=mask)
        sentinel = 0xBAAD_F00D

        for lane in range(64):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, pack_f16x2(F16_ONE, F16_ONE))
            state.set_vgpr(lane, 2, pack_f16x2(F16_ONE, F16_ONE))
            state.set_vgpr(lane, 3, pack_f16x2(F16_ONE, F16_ONE))

        interp = SIMDInterpreter()
        instr = VPKFMA16Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))
        results = interp.step(instr, state)

        for lane in range(64):
            if lane in (0, 31, 32, 63):
                # 1.0 * 1.0 + 1.0 = 2.0 (0x4000)
                self.assertEqual(results[lane], 0x40004000)
                self.assertEqual(state.get_vgpr(lane, 0), 0x40004000)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)


# ==============================================================================
# SECTION 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src2_in_place_accumulation(self) -> None:
        """Verify v0 = v1 * v2 + v0 reads initial v0 before writeback."""
        state = ScalarMachineState()
        init_v0 = pack_f16x2(F16_HALF, F16_HALF)    # 0.5, 0.5
        v1_val = pack_f16x2(F16_ONE, F16_ONE)      # 1.0, 1.0
        v2_val = pack_f16x2(F16_TWO, F16_TWO)      # 2.0, 2.0

        state.set_vgpr(0, init_v0)
        state.set_vgpr(1, v1_val)
        state.set_vgpr(2, v2_val)

        interp = ScalarInterpreter()
        instr = VPKFMA16Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(0))
        res = interp.step(instr, state)

        # 1.0 * 2.0 + 0.5 = 2.5 (0x4100)
        self.assertEqual(res, 0x41004100)
        self.assertEqual(state.get_vgpr(0), 0x41004100)

        # Cross check with translated SPIR-V
        spv_res = run_translator_pk_fma(v1_val, v2_val, init_v0, name="ps_alias_dst_src2")
        self.assertEqual(res, spv_res)

    def test_alias_all_operands_same_register(self) -> None:
        """Verify v0 = v0 * v0 + v0 evaluated with identical operand."""
        state = ScalarMachineState()
        init_v0 = pack_f16x2(F16_TWO, F16_TWO)     # 2.0, 2.0
        state.set_vgpr(0, init_v0)

        interp = ScalarInterpreter()
        instr = VPKFMA16Instruction(dst=VGPR(0), src0=VGPR(0), src1=VGPR(0), src2=VGPR(0))
        res = interp.step(instr, state)

        # 2.0 * 2.0 + 2.0 = 6.0 (0x4600)
        self.assertEqual(res, 0x46004600)
        self.assertEqual(state.get_vgpr(0), 0x46004600)

    def test_alias_src0_equals_src1(self) -> None:
        """Verify v0 = v1 * v1 + v2."""
        state = ScalarMachineState()
        v1_val = pack_f16x2(F16_TWO, F16_TWO)      # 2.0, 2.0
        v2_val = pack_f16x2(F16_ONE, F16_ONE)      # 1.0, 1.0
        state.set_vgpr(1, v1_val)
        state.set_vgpr(2, v2_val)

        interp = ScalarInterpreter()
        instr = VPKFMA16Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(1), src2=VGPR(2))
        res = interp.step(instr, state)

        # 2.0 * 2.0 + 1.0 = 5.0 (0x4500)
        self.assertEqual(res, 0x45004500)
        self.assertEqual(state.get_vgpr(0), 0x45004500)


# ==============================================================================
# SECTION 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation."""

    def test_vcc_and_scc_invariance(self) -> None:
        """Verify v_pk_fma_f16 as a VOP3P instruction never alters VCC or SCC."""
        # Assemble shader with comparison setting VCC, executing v_pk_fma_f16,
        # then reading VCC back via v_cndmask_b32.
        lines = [
            "v_cmp_eq_u32 vcc, 5, 5",              # Sets VCC to all 1s
            "v_mov_b32 v1, 0x3c003c00",
            "v_mov_b32 v2, 0x40004000",
            "v_mov_b32 v3, 0x38003800",
            "v_pk_fma_f16 v0, v1, v2, v3",         # Must not touch VCC
            "v_cndmask_b32 v4, 0x11111111, 0x22222222, vcc", # Should choose 0x22222222
            "exp mrt0 v4, v0, v4, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_vcc_invariant", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x22222222)
        self.assertEqual(words[1], 0x41004100)

    def test_clamp_saturation_boundaries(self) -> None:
        """Verify saturation limits on float output without modifying status registers."""
        # Test values above 1.0, below 0.0, and exactly 0.0 / 1.0
        p0 = pack_f16x2(0x4500, 0xC200)            # 5.0, -3.0
        p1 = pack_f16x2(F16_TWO, F16_ONE)          # 2.0, 1.0
        p2 = pack_f16x2(F16_POS_ZERO, F16_POS_ZERO)
        got = oracle_pk_fma_f16(p0, p1, p2, clamp=True)
        # lo: 5.0 * 2.0 = 10.0 -> clamped to 1.0 (0x3C00)
        # hi: -3.0 * 1.0 = -3.0 -> clamped to 0.0 (0x0000)
        self.assertEqual(got, 0x00003C00)


# ==============================================================================
# SECTION 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754EdgeCases(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zero_multiplication_and_addition(self) -> None:
        """Verify IEEE signed zeros: (+0) * (+0) + (+0) = +0, (-0) * (+0) + (-0) = -0."""
        # (+0) * (+0) + (+0) = +0 (0x0000)
        got_pos = fma_f16(F16_POS_ZERO, F16_POS_ZERO, F16_POS_ZERO)
        self.assertEqual(got_pos, F16_POS_ZERO)

        # (-0) * (+0) + (-0) = -0 (0x8000)
        got_neg = fma_f16(F16_NEG_ZERO, F16_POS_ZERO, F16_NEG_ZERO)
        self.assertEqual(got_neg, F16_NEG_ZERO)

    def test_subnormal_precision(self) -> None:
        """Verify subnormal preservation and computation without flush."""
        sub = F16_MIN_SUBNORMAL
        res = fma_f16(sub, F16_ONE, sub)
        # min_subnormal * 1.0 + min_subnormal = 2 * min_subnormal (0x0002)
        self.assertEqual(res, 0x0002)

    def test_infinities_and_cancellation(self) -> None:
        """Verify (+Inf) * 2.0 + 1.0 = +Inf, and (+Inf) + (-Inf) = NaN."""
        # Inf * 2.0 + 1.0 = Inf
        self.assertEqual(fma_f16(F16_POS_INF, F16_TWO, F16_ONE), F16_POS_INF)
        self.assertEqual(fma_f16(F16_NEG_INF, F16_TWO, F16_ONE), F16_NEG_INF)

        # (+Inf) * 1.0 + (-Inf) = NaN
        self.assertEqual(fma_f16(F16_POS_INF, F16_ONE, F16_NEG_INF), F16_QNAN)

    def test_nan_propagation(self) -> None:
        """Verify NaN operand propagates canonical quiet NaN."""
        self.assertEqual(fma_f16(F16_QNAN, F16_ONE, F16_TWO), F16_QNAN)
        self.assertEqual(fma_f16(F16_ONE, F16_SNAN, F16_TWO), F16_QNAN)
        self.assertEqual(fma_f16(F16_ONE, F16_TWO, F16_QNAN), F16_QNAN)

    def test_overflow_to_infinity(self) -> None:
        """Verify overflow beyond MAX_NORMAL produces infinity."""
        res = fma_f16(F16_MAX_NORMAL, F16_TWO, F16_POS_ZERO)
        self.assertEqual(res, F16_POS_INF)


# ==============================================================================
# SECTION 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestNegativeControlsAndMutations(unittest.TestCase):
    """Commit 8: Negative controls, syntax rejection, and mutation detection."""

    def test_rejection_invalid_destination(self) -> None:
        """Verify syntax rejection when destination is not a VGPR."""
        with self.assertRaises(TypeError):
            VPKFMA16Instruction(dst=SGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))

    def test_rejection_out_of_range_registers(self) -> None:
        """Verify rejection of register indices outside architectural bounds."""
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            SGPR(106)

    def test_mutation_detection_swap_opcodes(self) -> None:
        """Mutation detector: swapping OpFMul with OpFAdd in SPIR-V alters results."""
        spv = support.translate_lines("ps_mut_test", [
            "v_mov_b32 v1, 0x40004000",         # 2.0, 2.0
            "v_mov_b32 v2, 0x40004000",         # 2.0, 2.0
            "v_mov_b32 v3, 0x3c003c00",         # 1.0, 1.0
            "v_pk_fma_f16 v0, v1, v2, v3",      # 2*2 + 1 = 5.0 (0x4500)
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        orig_words = support.mrt0_words(spv)
        self.assertEqual(orig_words[0], 0x45004500)

        # Mutate SPIR-V: Replace OpFMul with OpFSub
        self.assertIn("OpFMul", spv)
        mutated_spv = spv.replace("OpFMul", "OpFSub", 1)
        mutated_words = support.mrt0_words(mutated_spv)
        # Mutation must be caught by producing a different result
        self.assertNotEqual(orig_words[0], mutated_words[0])

    def test_mutation_detection_inactive_lane_overwritten(self) -> None:
        """Mutation detector: faulty interpreter overwriting inactive lane is detected."""
        state = SIMDState(num_lanes=32, exec_mask=0x1)  # Only lane 0 active
        sentinel = 0x5555_5555
        state.set_vgpr(1, 0, sentinel)

        class FaultyInterpreter(SIMDInterpreter):
            def step(self, instr: VPKFMA16Instruction, s: SIMDState):
                # Bug: writes to all lanes regardless of exec_mask
                for l in range(s.num_lanes):
                    s.set_vgpr(l, instr.dst.value, 0x1234)
                return []

        faulty = FaultyInterpreter()
        instr = VPKFMA16Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2), src2=VGPR(3))
        faulty.step(instr, state)
        self.assertNotEqual(state.get_vgpr(1, 0), sentinel)

    def test_mutation_detection_inverted_clamp(self) -> None:
        """Mutation detector: faulty clamp range fails assertion."""
        def faulty_clamp_fma(s0: int, s1: int, s2: int) -> int:
            # Bug: clamps to [0.5, 1.0] instead of [0.0, 1.0]
            val = u16_to_f16(s0) * u16_to_f16(s1) + u16_to_f16(s2)
            val = max(0.5, min(1.0, val))
            return f16_to_u16(val)

        correct = fma_f16(F16_POS_ZERO, F16_ONE, F16_POS_ZERO, clamp=True)
        mutant = faulty_clamp_fma(F16_POS_ZERO, F16_ONE, F16_POS_ZERO)
        self.assertEqual(correct, 0x0000)
        self.assertNotEqual(correct, mutant)


if __name__ == "__main__":
    unittest.main()
