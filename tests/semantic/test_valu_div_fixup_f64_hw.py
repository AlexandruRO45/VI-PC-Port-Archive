from __future__ import annotations

"""Semantic reference oracle and architectural model for v_div_fixup_f64.

RDNA2 / GFX10 VOP3 Double-Precision Floating-Point Division Fixup:
Performs the final IEEE-754 special-case resolution and sign determination for 64-bit float division.
Operands:
  S0 = quot (estimated quotient)
  S1 = denom (denominator)
  S2 = num (numerator)

Syntax:
    v_div_fixup_f64 v[dst:dst+1], v[src0:src0+1], v[src1:src1+1], v[src2:src2+1] [clamp]

Semantics:
    sign_out = sign(S1) ^ sign(S2)
    NaN(S2) -> quiet(S2)
    NaN(S1) -> quiet(S1)
    (0/0 or inf/inf) -> -qNaN (0xFFF80000_00000000)
    (x/inf or 0/y or underflow) -> +-0 (sign_out)
    (x/0 or inf/y or NaN(S0)) -> +-inf (sign_out | 0x7FF00000_00000000)
    otherwise -> copysign(S0, sign_out)
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

# IEEE 754 Binary64 Constants
F64_POS_ZERO: int = 0x00000000_00000000
F64_NEG_ZERO: int = 0x80000000_00000000
F64_ONE: int = 0x3FF00000_00000000
F64_TWO: int = 0x40000000_00000000
F64_FOUR: int = 0x40100000_00000000
F64_HALF: int = 0x3FE00000_00000000
F64_POS_INF: int = 0x7FF00000_00000000
F64_NEG_INF: int = 0xFFF00000_00000000
F64_QNAN: int = 0x7FF80000_00000000
F64_NEG_QNAN: int = 0xFFF80000_00000000

MASK_32: int = 0xFFFF_FFFF
MASK_64: int = 0xFFFF_FFFF_FFFF_FFFF


def u32(val: int) -> int:
    return val & MASK_32


def u64(val: int) -> int:
    return val & MASK_64


def oracle_div_fixup_f64(s0_u64: int, s1_u64: int, s2_u64: int, clamp: bool = False) -> int:
    """Bit-accurate IEEE 754 double-precision division fixup matching RDNA2 hardware semantics."""
    s0_u64, s1_u64, s2_u64 = u64(s0_u64), u64(s1_u64), u64(s2_u64)

    lo0, hi0 = s0_u64 & MASK_32, (s0_u64 >> 32) & MASK_32
    lo1, hi1 = s1_u64 & MASK_32, (s1_u64 >> 32) & MASK_32
    lo2, hi2 = s2_u64 & MASK_32, (s2_u64 >> 32) & MASK_32

    def cls64(lo: int, hi: int) -> Tuple[int, bool, bool, bool]:
        mag = hi & 0x7FFF_FFFF
        iz = ((mag | lo) == 0)
        exp_all = ((mag & 0x7FF0_0000) == 0x7FF0_0000)
        iinf = exp_all and (((mag & 0x000F_FFFF) | lo) == 0)
        inan = exp_all and (((mag & 0x000F_FFFF) | lo) != 0)
        return mag, iz, iinf, inan

    m0, z0, inf0, nan0 = cls64(lo0, hi0)
    _, z1, inf1, nan1 = cls64(lo1, hi1)
    _, z2, inf2, nan2 = cls64(lo2, hi2)

    e1 = (hi1 >> 20) & 0x7FF
    e2 = (hi2 >> 20) & 0x7FF
    ediff = (e2 - e1) & MASK_32
    ediff_i = ediff - 0x1_0000_0000 if (ediff & 0x8000_0000) else ediff
    uf = (ediff_i <= -1076)

    sgn = (hi1 ^ hi2) & 0x8000_0000
    inf_hi = sgn | 0x7FF0_0000
    norm_hi = sgn | m0

    b_zz = z1 and z2
    b_ii = inf1 and inf2
    b_ind = b_zz or b_ii
    b_zero = inf1 or z2 or uf
    b_inf = z1 or inf2 or nan0

    if nan2:
        r_hi, r_lo = hi2, lo2
    elif nan1:
        r_hi, r_lo = hi1, lo1
    elif b_ind:
        r_hi, r_lo = 0xFFF8_0000, 0
    elif b_zero:
        r_hi, r_lo = sgn, 0
    elif b_inf:
        r_hi, r_lo = inf_hi, 0
    else:
        r_hi, r_lo = norm_hi, lo0

    if clamp:
        val_f = struct.unpack("<d", struct.pack("<II", r_lo, r_hi))[0]
        if math.isnan(val_f):
            r_lo, r_hi = 0, 0
        else:
            val_f = max(0.0, min(1.0, val_f))
            r_lo, r_hi = struct.unpack("<II", struct.pack("<d", val_f))

    return u64((r_hi << 32) | r_lo)


def run_translator_div_fixup(
    s0: int,
    s1: int,
    s2: int,
    mods: str = "",
    name: str = "ps_div_fixup",
) -> int:
    """Translate and run a minimal shader through tools/rdna2spirv and return result 64-bit integer."""
    lo0, hi0 = s0 & MASK_32, (s0 >> 32) & MASK_32
    lo1, hi1 = s1 & MASK_32, (s1 >> 32) & MASK_32
    lo2, hi2 = s2 & MASK_32, (s2 >> 32) & MASK_32
    lines = [
        f"v_mov_b32 v2, {lo0:#x}",
        f"v_mov_b32 v3, {hi0:#x}",
        f"v_mov_b32 v4, {lo1:#x}",
        f"v_mov_b32 v5, {hi1:#x}",
        f"v_mov_b32 v6, {lo2:#x}",
        f"v_mov_b32 v7, {hi2:#x}",
        f"v_div_fixup_f64 v[0:1], v[2:3], v[4:5], v[6:7] {mods}".strip(),
        "exp mrt0 v0, v1, v0, v1 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    words = support.mrt0_words(spv)
    return u64((words[1] << 32) | words[0])


class TestOracleAndHardwareModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_basic_normal_quotient(self) -> None:
        """Verify normal division fixup passes quotient with correct sign."""
        got = oracle_div_fixup_f64(F64_ONE, F64_TWO, F64_TWO)
        self.assertEqual(got, F64_ONE)

    def test_division_by_zero_produces_infinity(self) -> None:
        """Verify x / 0 produces signed infinity."""
        got = oracle_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_TWO)
        self.assertEqual(got, F64_POS_INF)

    def test_zero_divided_by_y_produces_zero(self) -> None:
        """Verify 0 / y produces signed zero."""
        got = oracle_div_fixup_f64(F64_ONE, F64_TWO, F64_POS_ZERO)
        self.assertEqual(got, F64_POS_ZERO)

    def test_indeterminate_0_div_0_produces_neg_qnan(self) -> None:
        """Verify 0 / 0 produces canonical negative quiet NaN."""
        got = oracle_div_fixup_f64(F64_ONE, F64_POS_ZERO, F64_POS_ZERO)
        self.assertEqual(got, F64_NEG_QNAN)

    def test_oracle_matches_translator(self) -> None:
        """Verify oracle output directly against translator output."""
        expected = oracle_div_fixup_f64(F64_ONE, F64_TWO, F64_TWO)
        translated = run_translator_div_fixup(F64_ONE, F64_TWO, F64_TWO, name="ps_div_oracle")
        self.assertEqual(translated, expected)


# ==============================================================================
# SECTION 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class OperandKind(Enum):
    VGPR_PAIR = auto()
    SGPR_PAIR = auto()
    IMMEDIATE_F64 = auto()


@dataclass(frozen=True)
class Operand64:
    kind: OperandKind
    base_reg: int = 0
    imm_val: int = 0

    def __post_init__(self) -> None:
        if self.kind == OperandKind.VGPR_PAIR:
            if not (0 <= self.base_reg <= 254):
                raise ValueError(f"VGPR pair base index out of range: {self.base_reg}")
        elif self.kind == OperandKind.SGPR_PAIR:
            if not (0 <= self.base_reg <= 104):
                raise ValueError(f"SGPR pair base index out of range: {self.base_reg}")


def VGPRPair(base_reg: int) -> Operand64:
    return Operand64(OperandKind.VGPR_PAIR, base_reg=base_reg)


def SGPRPair(base_reg: int) -> Operand64:
    return Operand64(OperandKind.SGPR_PAIR, base_reg=base_reg)


def ImmediateF64(val: int) -> Operand64:
    return Operand64(OperandKind.IMMEDIATE_F64, imm_val=u64(val))


@dataclass(frozen=True)
class VDivFixup64Modifiers:
    clamp: bool = False

    def to_assembly(self) -> str:
        return "clamp" if self.clamp else ""


@dataclass(frozen=True)
class VDivFixup64Instruction:
    dst: Operand64
    src0: Operand64
    src1: Operand64
    src2: Operand64
    mods: VDivFixup64Modifiers = field(default_factory=VDivFixup64Modifiers)

    def __post_init__(self) -> None:
        if self.dst.kind != OperandKind.VGPR_PAIR:
            raise TypeError(f"Destination must be a VGPR pair, got {self.dst.kind}")

    def format_operand(self, op: Operand64) -> str:
        if op.kind == OperandKind.VGPR_PAIR:
            return f"v[{op.base_reg}:{op.base_reg + 1}]"
        elif op.kind == OperandKind.SGPR_PAIR:
            return f"s[{op.base_reg}:{op.base_reg + 1}]"
        else:
            return f"{op.imm_val:#x}"

    def to_assembly(self) -> str:
        d = self.format_operand(self.dst)
        s0 = self.format_operand(self.src0)
        s1 = self.format_operand(self.src1)
        s2 = self.format_operand(self.src2)
        m = self.mods.to_assembly()
        suffix = f" {m}" if m else ""
        return f"v_div_fixup_f64 {d}, {s0}, {s1}, {s2}{suffix}"


class TestASTDefinitions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        """Verify AST correctly serializes to valid RDNA2 assembly text."""
        instr = VDivFixup64Instruction(
            dst=VGPRPair(0),
            src0=VGPRPair(2),
            src1=VGPRPair(4),
            src2=VGPRPair(6),
            mods=VDivFixup64Modifiers(clamp=True),
        )
        asm = instr.to_assembly()
        self.assertIn("v_div_fixup_f64 v[0:1], v[2:3], v[4:5], v[6:7] clamp", asm)

    def test_ast_translation_validity(self) -> None:
        """Verify that assembly generated by AST translates cleanly."""
        instr = VDivFixup64Instruction(
            dst=VGPRPair(0),
            src0=VGPRPair(2),
            src1=VGPRPair(4),
            src2=VGPRPair(6),
        )
        lines = [
            "v_mov_b32 v2, 0", "v_mov_b32 v3, 0x3ff00000",
            "v_mov_b32 v4, 0", "v_mov_b32 v5, 0x40000000",
            "v_mov_b32 v6, 0", "v_mov_b32 v7, 0x40000000",
            instr.to_assembly(),
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_div_valid", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(u64((words[1] << 32) | words[0]), F64_ONE)


# ==============================================================================
# SECTION 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState64:
    vgprs: Dict[int, int] = field(default_factory=dict)
    sgprs: Dict[int, int] = field(default_factory=dict)

    def get_vgpr64(self, base_idx: int) -> int:
        lo = self.vgprs.get(base_idx, 0)
        hi = self.vgprs.get(base_idx + 1, 0)
        return u64((hi << 32) | lo)

    def set_vgpr64(self, base_idx: int, val: int) -> None:
        val = u64(val)
        self.vgprs[base_idx] = val & MASK_32
        self.vgprs[base_idx + 1] = (val >> 32) & MASK_32

    def get_sgpr64(self, base_idx: int) -> int:
        lo = self.sgprs.get(base_idx, 0)
        hi = self.sgprs.get(base_idx + 1, 0)
        return u64((hi << 32) | lo)

    def set_sgpr64(self, base_idx: int, val: int) -> None:
        val = u64(val)
        self.sgprs[base_idx] = val & MASK_32
        self.sgprs[base_idx + 1] = (val >> 32) & MASK_32


class ScalarInterpreter64:
    def resolve_operand64(self, op: Operand64, state: ScalarMachineState64) -> int:
        if op.kind == OperandKind.VGPR_PAIR:
            return state.get_vgpr64(op.base_reg)
        elif op.kind == OperandKind.SGPR_PAIR:
            return state.get_sgpr64(op.base_reg)
        elif op.kind == OperandKind.IMMEDIATE_F64:
            return u64(op.imm_val)
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")

    def step(self, instr: VDivFixup64Instruction, state: ScalarMachineState64) -> int:
        v0 = self.resolve_operand64(instr.src0, state)
        v1 = self.resolve_operand64(instr.src1, state)
        v2 = self.resolve_operand64(instr.src2, state)

        res = oracle_div_fixup_f64(v0, v1, v2, clamp=instr.mods.clamp)
        state.set_vgpr64(instr.dst.base_reg, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_execution(self) -> None:
        """Verify interpreter correctly executes across VGPR, SGPR, and Immediate inputs."""
        state = ScalarMachineState64()
        state.set_vgpr64(2, F64_ONE)
        state.set_sgpr64(4, F64_TWO)
        imm = ImmediateF64(F64_TWO)

        interp = ScalarInterpreter64()
        instr = VDivFixup64Instruction(
            dst=VGPRPair(0),
            src0=VGPRPair(2),
            src1=SGPRPair(4),
            src2=imm,
        )
        res = interp.step(instr, state)
        self.assertEqual(res, F64_ONE)
        self.assertEqual(state.get_vgpr64(0), F64_ONE)

    def test_scalar_interpreter_cross_validation_with_translator(self) -> None:
        """Verify scalar interpreter exactly matches translated SPIR-V execution."""
        state = ScalarMachineState64()
        state.set_vgpr64(2, F64_FOUR)
        state.set_vgpr64(4, F64_TWO)
        state.set_vgpr64(6, F64_TWO)

        interp = ScalarInterpreter64()
        instr = VDivFixup64Instruction(
            dst=VGPRPair(0),
            src0=VGPRPair(2),
            src1=VGPRPair(4),
            src2=VGPRPair(6),
            mods=VDivFixup64Modifiers(clamp=True),
        )
        interp_res = interp.step(instr, state)
        spv_res = run_translator_div_fixup(F64_FOUR, F64_TWO, F64_TWO, mods="clamp", name="ps_scalar_div_cross")
        self.assertEqual(interp_res, spv_res)


# ==============================================================================
# SECTION 4: SIMD Multi-Lane Divergence and 32/64-bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDState64:
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

    def get_vgpr64(self, lane: int, base_idx: int) -> int:
        lo = self.vgprs[lane].get(base_idx, 0)
        hi = self.vgprs[lane].get(base_idx + 1, 0)
        return u64((hi << 32) | lo)

    def set_vgpr64(self, lane: int, base_idx: int, val: int) -> None:
        val = u64(val)
        self.vgprs[lane][base_idx] = val & MASK_32
        self.vgprs[lane][base_idx + 1] = (val >> 32) & MASK_32

    def get_sgpr64(self, base_idx: int) -> int:
        lo = self.sgprs.get(base_idx, 0)
        hi = self.sgprs.get(base_idx + 1, 0)
        return u64((hi << 32) | lo)

    def set_sgpr64(self, base_idx: int, val: int) -> None:
        val = u64(val)
        self.sgprs[base_idx] = val & MASK_32
        self.sgprs[base_idx + 1] = (val >> 32) & MASK_32


class SIMDInterpreter64:
    def resolve_lane_operand64(self, op: Operand64, lane: int, state: SIMDState64) -> int:
        if op.kind == OperandKind.VGPR_PAIR:
            return state.get_vgpr64(lane, op.base_reg)
        elif op.kind == OperandKind.SGPR_PAIR:
            return state.get_sgpr64(op.base_reg)
        elif op.kind == OperandKind.IMMEDIATE_F64:
            return u64(op.imm_val)
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")

    def step(self, instr: VDivFixup64Instruction, state: SIMDState64) -> List[Optional[int]]:
        results: List[Optional[int]] = []
        for lane in range(state.num_lanes):
            if not state.is_lane_active(lane):
                results.append(None)
                continue

            v0 = self.resolve_lane_operand64(instr.src0, lane, state)
            v1 = self.resolve_lane_operand64(instr.src1, lane, state)
            v2 = self.resolve_lane_operand64(instr.src2, lane, state)

            res = oracle_div_fixup_f64(v0, v1, v2, clamp=instr.mods.clamp)
            state.set_vgpr64(lane, instr.dst.base_reg, res)
            results.append(res)
        return results


class TestSIMDDivergence(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_alternating_lanes(self) -> None:
        """Verify Wave32 alternating even/odd lane execution and inactive preservation."""
        mask = 0x5555_5555
        state = SIMDState64(num_lanes=32, exec_mask=mask)
        sentinel = 0xDEAD_BEEF_BAAD_F00D

        for lane in range(32):
            state.set_vgpr64(lane, 0, sentinel)
            state.set_vgpr64(lane, 2, F64_ONE)
            state.set_vgpr64(lane, 4, F64_TWO)
            state.set_vgpr64(lane, 6, F64_TWO)

        interp = SIMDInterpreter64()
        instr = VDivFixup64Instruction(dst=VGPRPair(0), src0=VGPRPair(2), src1=VGPRPair(4), src2=VGPRPair(6))
        results = interp.step(instr, state)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(results[lane], F64_ONE)
                self.assertEqual(state.get_vgpr64(lane, 0), F64_ONE)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr64(lane, 0), sentinel)

    def test_wave64_divergence(self) -> None:
        """Verify Wave64 boundary execution across low and high 32-lane words."""
        mask = (1 << 0) | (1 << 31) | (1 << 32) | (1 << 63)
        state = SIMDState64(num_lanes=64, exec_mask=mask)
        sentinel = 0x1234_5678_9ABC_DEF0

        for lane in range(64):
            state.set_vgpr64(lane, 0, sentinel)
            state.set_vgpr64(lane, 2, F64_HALF)
            state.set_vgpr64(lane, 4, F64_TWO)
            state.set_vgpr64(lane, 6, F64_ONE)

        interp = SIMDInterpreter64()
        instr = VDivFixup64Instruction(dst=VGPRPair(0), src0=VGPRPair(2), src1=VGPRPair(4), src2=VGPRPair(6))
        results = interp.step(instr, state)

        for lane in range(64):
            if lane in (0, 31, 32, 63):
                self.assertEqual(results[lane], F64_HALF)
                self.assertEqual(state.get_vgpr64(lane, 0), F64_HALF)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr64(lane, 0), sentinel)


# ==============================================================================
# SECTION 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src0_in_place(self) -> None:
        """Verify v[0:1] = div_fixup(v[0:1], v[2:3], v[4:5]) evaluates correctly."""
        state = ScalarMachineState64()
        state.set_vgpr64(0, F64_ONE)
        state.set_vgpr64(2, F64_TWO)
        state.set_vgpr64(4, F64_TWO)

        interp = ScalarInterpreter64()
        instr = VDivFixup64Instruction(dst=VGPRPair(0), src0=VGPRPair(0), src1=VGPRPair(2), src2=VGPRPair(4))
        res = interp.step(instr, state)

        self.assertEqual(res, F64_ONE)
        self.assertEqual(state.get_vgpr64(0), F64_ONE)

        # Cross check with translated SPIR-V
        spv_res = run_translator_div_fixup(F64_ONE, F64_TWO, F64_TWO, name="ps_alias_dst_src0")
        self.assertEqual(res, spv_res)

    def test_alias_dst_equals_src1(self) -> None:
        """Verify v[0:1] = div_fixup(v[2:3], v[0:1], v[4:5])."""
        state = ScalarMachineState64()
        state.set_vgpr64(0, F64_TWO)
        state.set_vgpr64(2, F64_ONE)
        state.set_vgpr64(4, F64_TWO)

        interp = ScalarInterpreter64()
        instr = VDivFixup64Instruction(dst=VGPRPair(0), src0=VGPRPair(2), src1=VGPRPair(0), src2=VGPRPair(4))
        res = interp.step(instr, state)

        self.assertEqual(res, F64_ONE)
        self.assertEqual(state.get_vgpr64(0), F64_ONE)
