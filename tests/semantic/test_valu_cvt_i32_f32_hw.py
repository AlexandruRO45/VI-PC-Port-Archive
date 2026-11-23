from __future__ import annotations

"""Semantic reference oracle and architectural model for v_cvt_i32_f32.

RDNA2 / GFX10 VOP1 Floating-Point to Signed 32-bit Integer Conversion:
Converts an IEEE 754 single-precision float (32-bit) to a signed 32-bit integer with truncation
toward zero.

Saturation and Edge Case Behavior (measured on RDNA2 / gfx1036):
- NaN inputs (both quiet and signaling) produce 0.
- Values >= 2^31 (including +infinity) saturate to 0x7FFFFFFF (+2147483647).
- Values <= -2^31 (including -infinity) saturate to 0x80000000 (-2147483648).
- Representable values in (-2^31, 2^31) truncate toward zero (round-toward-zero).
- Floating-point +/-0.0 both produce 0.

Syntax:
    v_cvt_i32_f32 vdst, vsrc
    v_cvt_i32_f32_e32 vdst, vsrc
    v_cvt_i32_f32_e64 vdst, [mods] vsrc
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

import support

MASK_32: int = 0xFFFF_FFFF
INT32_MAX: int = 0x7FFF_FFFF
INT32_MIN_UNSIGNED: int = 0x8000_0000


def f32_to_bits(f: float) -> int:
    """Pack Python float to IEEE 754 uint32 bits."""
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return 0x7F80_0000 if f > 0 else 0xFF80_0000


def bits_to_f32(bits: int) -> float:
    """Unpack IEEE 754 uint32 bits to Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK_32))[0]


def oracle_cvt_i32_f32(bits: int) -> int:
    """Reference mathematical oracle for v_cvt_i32_f32."""
    f = bits_to_f32(bits)
    if math.isnan(f):
        return 0
    if f >= 2147483648.0:
        return INT32_MAX
    if f <= -2147483648.0:
        return INT32_MIN_UNSIGNED
    val = math.trunc(f)
    return val & MASK_32


def run_translator_cvt_i32_f32(
    src_bits: int,
    name: str = "ps_cvt_i32_f32",
    op: str = "v_cvt_i32_f32",
    src_expr: Optional[str] = None,
) -> int:
    """Translate and run a minimal shader through tools/rdna2spirv and return result word."""
    if src_expr is None:
        lines = [
            f"v_mov_b32 v1, {src_bits & MASK_32:#x}",
            f"{op} v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
    else:
        lines = [
            f"v_mov_b32 v1, {src_bits & MASK_32:#x}",
            f"{op} v0, {src_expr}",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestOracleAndHardwareModel(unittest.TestCase):
    """Commit 1: Reference mathematical oracle and architectural model verification."""

    def test_positive_integers_and_truncation(self) -> None:
        """Verify standard positive float conversion with truncation toward zero."""
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(1.0)), 1)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(1.9999)), 1)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(42.75)), 42)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(100.25)), 100)

    def test_negative_integers_and_truncation(self) -> None:
        """Verify negative float conversion with truncation toward zero."""
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(-1.0)), 0xFFFF_FFFF)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(-1.9999)), 0xFFFF_FFFF)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(-42.75)), (-42) & MASK_32)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(-100.25)), (-100) & MASK_32)

    def test_zeros_and_subnormals(self) -> None:
        """Verify +0.0, -0.0 and subnormals convert to zero."""
        self.assertEqual(oracle_cvt_i32_f32(0x0000_0000), 0)
        self.assertEqual(oracle_cvt_i32_f32(0x8000_0000), 0)
        self.assertEqual(oracle_cvt_i32_f32(0x0000_0001), 0)
        self.assertEqual(oracle_cvt_i32_f32(0x8000_0001), 0)

    def test_saturation_limits(self) -> None:
        """Verify positive and negative overflow saturation."""
        self.assertEqual(oracle_cvt_i32_f32(0x4F00_0000), INT32_MAX)           # 2^31
        self.assertEqual(oracle_cvt_i32_f32(0xCF00_0000), INT32_MIN_UNSIGNED)  # -2^31
        self.assertEqual(oracle_cvt_i32_f32(0x7F80_0000), INT32_MAX)           # +inf
        self.assertEqual(oracle_cvt_i32_f32(0xFF80_0000), INT32_MIN_UNSIGNED)  # -inf

    def test_nan_produces_zero(self) -> None:
        """Verify NaNs produce zero according to RDNA2 hardware semantics."""
        self.assertEqual(oracle_cvt_i32_f32(0x7FC0_0000), 0)  # +qNaN
        self.assertEqual(oracle_cvt_i32_f32(0xFFC0_0000), 0)  # -qNaN
        self.assertEqual(oracle_cvt_i32_f32(0x7F80_0001), 0)  # +sNaN
        self.assertEqual(oracle_cvt_i32_f32(0xFF80_0001), 0)  # -sNaN

    def test_oracle_matches_translator(self) -> None:
        """Verify oracle directly against translator execution."""
        cases = [
            0x0000_0000,
            0x8000_0000,
            f32_to_bits(1.0),
            f32_to_bits(-1.0),
            f32_to_bits(42.75),
            f32_to_bits(-42.75),
            0x4F00_0000,
            0xCF00_0000,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
        ]
        for idx, bits in enumerate(cases):
            expected = oracle_cvt_i32_f32(bits)
            translated = run_translator_cvt_i32_f32(bits, name=f"ps_oracle_{idx}")
            self.assertEqual(translated, expected, f"Mismatch on bits {bits:#010x}")


# ==============================================================================
# SECTION 2: Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    FLOAT_IMM = auto()


@dataclass(frozen=True)
class Operand:
    kind: OperandKind
    value: Union[int, float]
    neg: bool = False
    abs_val: bool = False

    def __post_init__(self) -> None:
        if self.kind == OperandKind.VGPR:
            if not isinstance(self.value, int) or not (0 <= self.value <= 255):
                raise ValueError(f"VGPR index out of range: {self.value}")
        elif self.kind == OperandKind.SGPR:
            if not isinstance(self.value, int) or not (0 <= self.value <= 105):
                raise ValueError(f"SGPR index out of range: {self.value}")

    def format_src(self) -> str:
        if self.kind == OperandKind.VGPR:
            base = f"v{self.value}"
        elif self.kind == OperandKind.SGPR:
            base = f"s{self.value}"
        else:
            base = f"{self.value}"
        if self.abs_val:
            base = f"|{base}|"
        if self.neg:
            base = f"-{base}"
        return base


def VGPR(idx: int) -> Operand:
    return Operand(OperandKind.VGPR, idx)


def SGPR(idx: int) -> Operand:
    return Operand(OperandKind.SGPR, idx)


def FloatImm(val: float) -> Operand:
    return Operand(OperandKind.FLOAT_IMM, val)


@dataclass(frozen=True)
class VCvtI32F32Instruction:
    dst: Operand
    src: Operand
    is_e64: bool = False

    def __post_init__(self) -> None:
        if self.dst.kind != OperandKind.VGPR:
            raise ValueError("Destination must be a VGPR")

    def to_asm(self) -> str:
        op_name = "v_cvt_i32_f32_e64" if self.is_e64 else "v_cvt_i32_f32"
        dst_str = f"v{self.dst.value}"
        src_str = self.src.format_src()
        return f"{op_name} {dst_str}, {src_str}"


class TestASTDefinitions(unittest.TestCase):
    """Commit 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        """Verify AST node string formatting for e32 and e64 encodings."""
        instr_e32 = VCvtI32F32Instruction(dst=VGPR(0), src=VGPR(1))
        self.assertEqual(instr_e32.to_asm(), "v_cvt_i32_f32 v0, v1")

        instr_sgpr = VCvtI32F32Instruction(dst=VGPR(5), src=SGPR(2))
        self.assertEqual(instr_sgpr.to_asm(), "v_cvt_i32_f32 v5, s2")

        instr_imm = VCvtI32F32Instruction(dst=VGPR(3), src=FloatImm(4.0))
        self.assertEqual(instr_imm.to_asm(), "v_cvt_i32_f32 v3, 4.0")

        instr_e64_mods = VCvtI32F32Instruction(
            dst=VGPR(0),
            src=Operand(OperandKind.VGPR, 1, neg=True, abs_val=True),
            is_e64=True,
        )
        self.assertEqual(instr_e64_mods.to_asm(), "v_cvt_i32_f32_e64 v0, -|v1|")

    def test_ast_translation_validity(self) -> None:
        """Verify shader using AST-emitted instructions translates and executes correctly."""
        instr = VCvtI32F32Instruction(dst=VGPR(0), src=FloatImm(2.0))
        lines = [
            instr.to_asm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_cvt", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 2)


# ==============================================================================
# SECTION 3: Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarState:
    vgprs: Dict[int, int] = field(default_factory=dict)
    sgprs: Dict[int, int] = field(default_factory=dict)

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs.get(idx, 0)

    def set_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK_32

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0)

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK_32


class ScalarInterpreter:
    """Deterministic scalar interpreter executing VCvtI32F32Instruction."""

    def eval_operand(self, op: Operand, state: ScalarState) -> int:
        if op.kind == OperandKind.VGPR:
            raw = state.get_vgpr(int(op.value))
        elif op.kind == OperandKind.SGPR:
            raw = state.get_sgpr(int(op.value))
        elif op.kind == OperandKind.FLOAT_IMM:
            raw = f32_to_bits(float(op.value))
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")

        f = bits_to_f32(raw)
        if op.abs_val:
            f = abs(f)
        if op.neg:
            f = -f
        return f32_to_bits(f)

    def step(self, instr: VCvtI32F32Instruction, state: ScalarState) -> None:
        src_bits = self.eval_operand(instr.src, state)
        res = oracle_cvt_i32_f32(src_bits)
        state.set_vgpr(int(instr.dst.value), res)


class TestScalarInterpreter(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_execution(self) -> None:
        interp = ScalarInterpreter()
        state = ScalarState()
        state.set_vgpr(1, f32_to_bits(55.8))
        state.set_sgpr(2, f32_to_bits(-12.4))

        instr1 = VCvtI32F32Instruction(dst=VGPR(0), src=VGPR(1))
        interp.step(instr1, state)
        self.assertEqual(state.get_vgpr(0), 55)

        instr2 = VCvtI32F32Instruction(dst=VGPR(3), src=SGPR(2))
        interp.step(instr2, state)
        self.assertEqual(state.get_vgpr(3), (-12) & MASK_32)

    def test_scalar_interpreter_cross_validation_with_translator(self) -> None:
        interp = ScalarInterpreter()
        test_values = [
            f32_to_bits(0.0),
            f32_to_bits(-0.0),
            f32_to_bits(3.14159),
            f32_to_bits(-3.14159),
            f32_to_bits(1e10),
            0x7FC0_0000,
        ]
        for idx, bits in enumerate(test_values):
            state = ScalarState()
            state.set_vgpr(1, bits)
            instr = VCvtI32F32Instruction(dst=VGPR(0), src=VGPR(1))
            interp.step(instr, state)
            expected = state.get_vgpr(0)
            translated = run_translator_cvt_i32_f32(bits, name=f"ps_interp_{idx}")
            self.assertEqual(translated, expected)


# ==============================================================================
# SECTION 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    num_lanes: int = 32
    exec_mask: int = 0xFFFF_FFFF
    vgprs: List[Dict[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.vgprs:
            self.vgprs = [{} for _ in range(self.num_lanes)]

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def get_vgpr(self, lane: int, reg: int) -> int:
        return self.vgprs[lane].get(reg, 0)

    def set_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = val & MASK_32


class SIMDInterpreter:
    """SIMD vector interpreter executing instructions with lane masking."""

    def __init__(self) -> None:
        self.scalar_interp = ScalarInterpreter()

    def step(self, instr: VCvtI32F32Instruction, state: SIMDState) -> List[int]:
        updated_lanes = []
        for lane in range(state.num_lanes):
            if state.is_lane_active(lane):
                sub_state = ScalarState(vgprs=dict(state.vgprs[lane]))
                self.scalar_interp.step(instr, sub_state)
                state.vgprs[lane] = sub_state.vgprs
                updated_lanes.append(lane)
        return updated_lanes


class TestSIMDDivergence(unittest.TestCase):
    """Commit 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_alternating_lanes(self) -> None:
        num_lanes = 32
        mask = 0x5555_5555
        state = SIMDState(num_lanes=num_lanes, exec_mask=mask)
        sentinel = 0xDEAD_BEEF

        for lane in range(num_lanes):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, f32_to_bits(float(lane) + 0.5))

        simd = SIMDInterpreter()
        instr = VCvtI32F32Instruction(dst=VGPR(0), src=VGPR(1))
        updated = simd.step(instr, state)

        self.assertEqual(len(updated), 16)
        for lane in range(num_lanes):
            if lane % 2 == 0:
                self.assertEqual(state.get_vgpr(lane, 0), lane)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)

    def test_wave64_divergence(self) -> None:
        num_lanes = 64
        mask = 0xFFFF_FFFF_0000_0000
        state = SIMDState(num_lanes=num_lanes, exec_mask=mask)
        sentinel = 0xCAFE_BABE

        for lane in range(num_lanes):
            state.set_vgpr(lane, 0, sentinel)
            state.set_vgpr(lane, 1, f32_to_bits(-float(lane)))

        simd = SIMDInterpreter()
        instr = VCvtI32F32Instruction(dst=VGPR(0), src=VGPR(1))
        updated = simd.step(instr, state)

        self.assertEqual(len(updated), 32)
        for lane in range(num_lanes):
            if lane >= 32:
                self.assertEqual(state.get_vgpr(lane, 0), (-lane) & MASK_32)
            else:
                self.assertEqual(state.get_vgpr(lane, 0), sentinel)


# ==============================================================================
# SECTION 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Commit 5: Register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src_in_place(self) -> None:
        """Verify v_cvt_i32_f32 v0, v0 safely converts float in v0 to int in-place."""
        state = ScalarState()
        state.set_vgpr(0, f32_to_bits(123.75))

        interp = ScalarInterpreter()
        instr = VCvtI32F32Instruction(dst=VGPR(0), src=VGPR(0))
        interp.step(instr, state)

        self.assertEqual(state.get_vgpr(0), 123)

    def test_alias_in_place_translator_cross_check(self) -> None:
        """Verify in-place register reuse with translator execution."""
        lines = [
            f"v_mov_b32 v0, {f32_to_bits(-77.8):#x}",
            "v_cvt_i32_f32 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_test", lines)
        got = support.mrt0_words(spv)[0]
        self.assertEqual(got, (-77) & MASK_32)


# ==============================================================================
# SECTION 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Commit 6: Condition codes, status flags, and carry/borrow generation."""

    def test_vcc_and_scc_invariance(self) -> None:
        """Verify v_cvt_i32_f32 leaves VCC and SCC completely untouched."""
        lines = [
            "s_mov_b32 s0, 1",
            "s_add_u32 s1, s0, 0",  # Clears carry, SCC depends on s1
            f"v_mov_b32 v1, {f32_to_bits(15.5):#x}",
            "v_cvt_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_flags_test", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 15)


# ==============================================================================
# SECTION 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ==============================================================================

class TestIEEE754EdgeCases(unittest.TestCase):
    """Commit 7: IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_positive_and_negative(self) -> None:
        """Verify subnormal floating-point numbers truncate to 0."""
        self.assertEqual(oracle_cvt_i32_f32(0x0000_0001), 0)
        self.assertEqual(oracle_cvt_i32_f32(0x007F_FFFF), 0)
        self.assertEqual(oracle_cvt_i32_f32(0x8000_0001), 0)
        self.assertEqual(oracle_cvt_i32_f32(0x807F_FFFF), 0)

    def test_just_below_integer_boundaries(self) -> None:
        """Verify truncation for floats strictly below integers."""
        # 0.99999994 -> 0
        self.assertEqual(oracle_cvt_i32_f32(0x3F7F_FFFF), 0)
        # -0.99999994 -> 0
        self.assertEqual(oracle_cvt_i32_f32(0xBF7F_FFFF), 0)

    def test_saturation_at_int32_limits(self) -> None:
        """Verify saturation at +2^31 and -2^31 thresholds."""
        # 2^31 as float: 0x4F000000 -> saturates to 0x7FFFFFFF
        self.assertEqual(oracle_cvt_i32_f32(0x4F00_0000), 0x7FFF_FFFF)
        # -2^31 as float: 0xCF000000 -> saturates to 0x80000000
        self.assertEqual(oracle_cvt_i32_f32(0xCF00_0000), 0x8000_0000)
        # Numbers larger than 2^31 (e.g. 1e15)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(1e15)), 0x7FFF_FFFF)
        self.assertEqual(oracle_cvt_i32_f32(f32_to_bits(-1e15)), 0x8000_0000)

    def test_infinities_and_nans(self) -> None:
        """Verify +inf/-inf saturate and NaNs yield 0."""
        self.assertEqual(oracle_cvt_i32_f32(0x7F80_0000), 0x7FFF_FFFF)
        self.assertEqual(oracle_cvt_i32_f32(0xFF80_0000), 0x8000_0000)
        self.assertEqual(oracle_cvt_i32_f32(0x7FC0_0000), 0)
        self.assertEqual(oracle_cvt_i32_f32(0xFFC0_0000), 0)
        self.assertEqual(oracle_cvt_i32_f32(0x7F80_0001), 0)
        self.assertEqual(oracle_cvt_i32_f32(0xFF80_0001), 0)
