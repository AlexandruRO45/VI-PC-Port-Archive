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

    def step(self, instr: VFMAC16Instruction, state: ScalarMachineState) -> int:
        dst_val = state.get_vgpr(instr.dst.value)
        val0 = self.resolve_operand(instr.src0, state)
        val1 = self.resolve_operand(instr.src1, state)

        res = oracle_fmac_f16(
            dst_val,
            val0,
            val1,
            op_sel=instr.mods.op_sel,
            clamp=instr.mods.clamp,
        )
        state.set_vgpr(instr.dst.value, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_execution(self) -> None:
        """Verify interpreter correctly executes across VGPR, SGPR, and Immediate inputs."""
        state = ScalarMachineState()
        state.set_vgpr(0, 0x1234_3800)                          # acc = 0.5, hi = 0x1234
        state.set_vgpr(1, F16_ONE)                              # 1.0
        state.set_sgpr(4, F16_TWO)                              # 2.0

        interp = ScalarInterpreter()
        instr = VFMAC16Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=SGPR(4),
        )
        res = interp.step(instr, state)
        # 1.0 * 2.0 + 0.5 = 2.5 (0x4100), keeping hi 0x1234
        self.assertEqual(res, 0x1234_4100)
        self.assertEqual(state.get_vgpr(0), 0x1234_4100)

    def test_scalar_interpreter_cross_validation_with_translator(self) -> None:
        """Verify scalar interpreter exactly matches translated SPIR-V execution."""
        state = ScalarMachineState()
        dst = 0x5678_3800
        src0 = F16_ONE
        src1 = F16_TWO
        state.set_vgpr(0, dst)
        state.set_vgpr(1, src0)
        state.set_vgpr(2, src1)

        interp = ScalarInterpreter()
        instr = VFMAC16Instruction(
            dst=VGPR(0),
            src0=VGPR(1),
            src1=VGPR(2),
        )
        interp_res = interp.step(instr, state)
        spv_res = run_translator_fmac(dst, src0, src1, name="ps_scalar_fmac_cross")
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

    def step(self, instr: VFMAC16Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = []
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                results.append(None)
                continue

            dst_val = state.get_vgpr(lane, instr.dst.value)
            v0 = self.resolve_lane_operand(instr.src0, lane, state)
            v1 = self.resolve_lane_operand(instr.src1, lane, state)

            res = oracle_fmac_f16(
                dst_val,
                v0,
                v1,
                op_sel=instr.mods.op_sel,
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
            state.set_vgpr(lane, 0, 0x1111_3800 if lane % 2 == 0 else sentinel)
            state.set_vgpr(lane, 1, F16_ONE)
            state.set_vgpr(lane, 2, F16_TWO)

        interp = SIMDInterpreter()
        instr = VFMAC16Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2))
        results = interp.step(instr, state)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], 0x1111_4100)
                self.assertEqual(state.get_vgpr(lane, 0), 0x1111_4100)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_wave64_divergence(self) -> None:
        """Verify Wave64 boundary execution across low and high 32-lane words."""
        mask = (1 << 0) | (1 << 31) | (1 << 32) | (1 << 63)
        state = SIMDState(num_lanes=64, exec_mask=mask)
        sentinel = 0xBAAD_F00D

        for lane in range(64):
            state.set_vgpr(lane, 0, 0x2222_3800 if ((mask >> lane) & 1) else sentinel)
            state.set_vgpr(lane, 1, F16_ONE)
            state.set_vgpr(lane, 2, F16_ONE)

        interp = SIMDInterpreter()
        instr = VFMAC16Instruction(dst=VGPR(0), src0=VGPR(1), src1=VGPR(2))
        results = interp.step(instr, state)

        for lane in range(64):
            if lane in (0, 31, 32, 63):
                # 1.0 * 1.0 + 0.5 = 1.5 (0x3E00)
                self.assertEqual(results[lane], 0x2222_3E00)
                self.assertEqual(state.get_vgpr(lane, 0), 0x2222_3E00)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)


# ==============================================================================
# SECTION 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0_in_place(self) -> None:
        """Verify v0 = fmac(v0, v1) evaluated with v0 as both multiplier and accumulator."""
        state = ScalarMachineState()
        init_v0 = 0x4444_4000  # 2.0, hi=0x4444
        v1_val = F16_TWO       # 2.0

        state.set_vgpr(0, init_v0)
        state.set_vgpr(1, v1_val)

        interp = ScalarInterpreter()
        instr = VFMAC16Instruction(dst=VGPR(0), src0=VGPR(0), src1=VGPR(1))
        res = interp.step(instr, state)

        # 2.0 * 2.0 + 2.0 = 6.0 (0x4600), hi preserved (0x4444)
        self.assertEqual(res, 0x4444_4600)
        self.assertEqual(state.get_vgpr(0), 0x4444_4600)

        # Cross check with translated SPIR-V
        spv_res = run_translator_fmac(init_v0, init_v0, v1_val, name="ps_alias_dst_src0")
        self.assertEqual(res, spv_res)

    def test_alias_dst_equals_src0_equals_src1(self) -> None:
        """Verify v0 = v0 * v0 + v0."""
        state = ScalarMachineState()
        init_v0 = 0x5555_4000  # 2.0, hi=0x5555
        state.set_vgpr(0, init_v0)

        interp = ScalarInterpreter()
        instr = VFMAC16Instruction(dst=VGPR(0), src0=VGPR(0), src1=VGPR(0))
        res = interp.step(instr, state)

        self.assertEqual(res, 0x5555_4600)
        self.assertEqual(state.get_vgpr(0), 0x5555_4600)


# ==============================================================================
# SECTION 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation."""

    def test_vcc_and_scc_invariance(self) -> None:
        """Verify v_fmac_f16 never alters VCC or SCC."""
        lines = [
            "v_cmp_eq_u32 vcc, 15, 15",            # Sets VCC to all 1s
            "v_mov_b32 v0, 0x00003800",
            "v_mov_b32 v1, 0x00003c00",
            "v_mov_b32 v2, 0x00004000",
            "v_fmac_f16 v0, v1, v2",               # Must not touch VCC
            "v_cndmask_b32 v4, 0x11111111, 0x22222222, vcc", # Should choose 0x22222222
            "exp mrt0 v4, v0, v4, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_fmac_vcc_invariant", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x22222222)
        self.assertEqual(words[1], 0x4100)

    def test_clamp_saturation_boundaries(self) -> None:
        """Verify saturation limits on float output without modifying status registers."""
        # 3.0 * 2.0 + 0.0 = 6.0 -> clamped to 1.0 (0x3C00)
        # -2.0 * 2.0 + 0.0 = -4.0 -> clamped to 0.0 (0x0000)
        got_pos = oracle_fmac_f16(F16_POS_ZERO, F16_THREE, F16_TWO, clamp=True)
        self.assertEqual(got_pos, F16_ONE)

        got_neg = oracle_fmac_f16(F16_POS_ZERO, 0xC000, F16_TWO, clamp=True)
        self.assertEqual(got_neg, F16_POS_ZERO)


# ==============================================================================
# SECTION 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754EdgeCases(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zeros(self) -> None:
        """Verify IEEE signed zeros in FMA."""
        self.assertEqual(fma_f16(F16_POS_ZERO, F16_POS_ZERO, F16_POS_ZERO), F16_POS_ZERO)
        self.assertEqual(fma_f16(F16_NEG_ZERO, F16_POS_ZERO, F16_NEG_ZERO), F16_NEG_ZERO)

    def test_subnormal_accumulation(self) -> None:
        """Verify subnormal preservation."""
        sub = F16_MIN_SUBNORMAL
        res = fma_f16(sub, F16_ONE, sub)
        self.assertEqual(res, 0x0002)

    def test_infinities_and_cancellation(self) -> None:
        """Verify Inf * 2.0 + 1.0 = Inf, and (+Inf) + (-Inf) = NaN."""
        self.assertEqual(fma_f16(F16_POS_INF, F16_TWO, F16_ONE), F16_POS_INF)
        self.assertEqual(fma_f16(F16_NEG_INF, F16_TWO, F16_ONE), F16_NEG_INF)
        self.assertEqual(fma_f16(F16_POS_INF, F16_ONE, F16_NEG_INF), F16_QNAN)

    def test_nan_propagation(self) -> None:
        """Verify NaN operand propagates quiet NaN."""
        self.assertEqual(fma_f16(F16_QNAN, F16_ONE, F16_TWO), F16_QNAN)
        self.assertEqual(fma_f16(F16_ONE, F16_QNAN, F16_TWO), F16_QNAN)
        self.assertEqual(fma_f16(F16_ONE, F16_TWO, F16_QNAN), F16_QNAN)
