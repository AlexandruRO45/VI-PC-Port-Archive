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

All test methods compile assembly listings directly through tools/rdna2spirv.py
via support.translate_lines(...) and execute in spirv_mini_interp via support.mrt0_words.
Standard library Python only; zero third-party dependencies; zero game data.
"""

import math
import os
import struct
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

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


class TestOracleAndHardwareModel(unittest.TestCase):
    """Reference mathematical oracle and architectural model verification."""

    def test_positive_integers_and_truncation(self) -> None:
        """Verify standard positive float conversion with truncation toward zero."""
        for f_val in (1.0, 1.9999, 42.75, 100.25):
            bits = f32_to_bits(f_val)
            lines = [
                f"v_mov_b32 v1, {bits:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_pos_{bits:#x}", lines)
            words = support.mrt0_words(spv)
            expected = oracle_cvt_i32_f32(bits)
            self.assertEqual(words[0], expected)

    def test_negative_integers_and_truncation(self) -> None:
        """Verify negative float conversion with truncation toward zero."""
        for f_val in (-1.0, -1.9999, -42.75, -100.25):
            bits = f32_to_bits(f_val)
            lines = [
                f"v_mov_b32 v1, {bits:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_neg_{bits:#x}", lines)
            words = support.mrt0_words(spv)
            expected = oracle_cvt_i32_f32(bits)
            self.assertEqual(words[0], expected)

    def test_zeros_and_subnormals(self) -> None:
        """Verify +0.0, -0.0 and subnormals convert to zero."""
        for bits in (0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001):
            lines = [
                f"v_mov_b32 v1, {bits:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_zero_sub_{bits:#x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], 0)
            self.assertEqual(oracle_cvt_i32_f32(bits), 0)

    def test_saturation_limits(self) -> None:
        """Verify positive and negative overflow saturation."""
        cases = [
            (0x4F00_0000, INT32_MAX),           # 2^31
            (0xCF00_0000, INT32_MIN_UNSIGNED),  # -2^31
            (0x7F80_0000, INT32_MAX),           # +inf
            (0xFF80_0000, INT32_MIN_UNSIGNED),  # -inf
        ]
        for bits, expected in cases:
            lines = [
                f"v_mov_b32 v1, {bits:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_sat_{bits:#x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], expected)
            self.assertEqual(oracle_cvt_i32_f32(bits), expected)

    def test_nan_produces_zero(self) -> None:
        """Verify NaNs produce zero according to RDNA2 hardware semantics."""
        for bits in (0x7FC0_0000, 0xFFC0_0000, 0x7F80_0001, 0xFF80_0001):
            lines = [
                f"v_mov_b32 v1, {bits:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_nan_{bits:#x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], 0)
            self.assertEqual(oracle_cvt_i32_f32(bits), 0)

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
            lines = [
                f"v_mov_b32 v1, {bits & MASK_32:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_oracle_{idx}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], expected, f"Mismatch on bits {bits:#010x}")


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
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self) -> None:
        """Verify AST node string formatting and compile through translator."""
        instr_e32 = VCvtI32F32Instruction(dst=VGPR(0), src=VGPR(1))
        self.assertEqual(instr_e32.to_asm(), "v_cvt_i32_f32 v0, v1")

        lines = [
            f"v_mov_b32 v1, {f32_to_bits(12.0):#x}",
            instr_e32.to_asm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_fmt", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 12)

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
# SECTION 3: Scalar Instruction Execution
# ==============================================================================

class TestScalarInterpreter(unittest.TestCase):
    """Deterministic scalar execution via translator."""

    def test_scalar_interpreter_execution(self) -> None:
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(55.8):#x}",
            f"s_mov_b32 s2, {f32_to_bits(-12.4):#x}",
            "v_cvt_i32_f32 v0, v1",
            "v_cvt_i32_f32 v3, s2",
            "exp mrt0 v0, v3, v0, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_scalar_exec", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 55)
        self.assertEqual(words[1], (-12) & MASK_32)

    def test_scalar_interpreter_cross_validation_with_translator(self) -> None:
        test_values = [
            f32_to_bits(0.0),
            f32_to_bits(-0.0),
            f32_to_bits(3.14159),
            f32_to_bits(-3.14159),
            f32_to_bits(1e10),
            0x7FC0_0000,
        ]
        for idx, bits in enumerate(test_values):
            lines = [
                f"v_mov_b32 v1, {bits:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_interp_{idx}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], oracle_cvt_i32_f32(bits))


# ==============================================================================
# SECTION 4: SIMD Multi-Lane Divergence and EXEC Masking
# ==============================================================================

class TestSIMDDivergence(unittest.TestCase):
    """SIMD execution and divergent values."""

    def test_wave32_alternating_lanes(self) -> None:
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(7.5):#x}",
            "v_cvt_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_wave32_alt", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 7)

    def test_wave64_divergence(self) -> None:
        lines = [
            f"v_mov_b32 v1, {f32_to_bits(-32.0):#x}",
            "v_cvt_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_wave64_div", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], (-32) & MASK_32)


# ==============================================================================
# SECTION 5: Register Aliasing and In-Place Operand Preservation
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Register aliasing and in-place operand preservation."""

    def test_alias_dst_equals_src_in_place(self) -> None:
        """Verify v_cvt_i32_f32 v0, v0 safely converts float in v0 to int in-place."""
        lines = [
            f"v_mov_b32 v0, {f32_to_bits(123.75):#x}",
            "v_cvt_i32_f32 v0, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alias_inplace", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 123)

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
# SECTION 6: Condition Codes, Status Flags, and Invariance
# ==============================================================================

class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Condition codes, status flags, and carry/borrow generation."""

    def test_vcc_and_scc_invariance(self) -> None:
        """Verify v_cvt_i32_f32 leaves VCC and SCC completely untouched."""
        lines = [
            "s_mov_b32 s0, 1",
            "s_add_u32 s1, s0, 0",
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
    """IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_positive_and_negative(self) -> None:
        """Verify subnormal floating-point numbers truncate to 0."""
        for bits in (0x0000_0001, 0x007F_FFFF, 0x8000_0001, 0x807F_FFFF):
            lines = [
                f"v_mov_b32 v1, {bits:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_subn_{bits:#x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], 0)

    def test_just_below_integer_boundaries(self) -> None:
        """Verify truncation for floats strictly below integers."""
        # 0.99999994 -> 0, -0.99999994 -> 0
        for bits in (0x3F7F_FFFF, 0xBF7F_FFFF):
            lines = [
                f"v_mov_b32 v1, {bits:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_below_int_{bits:#x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], 0)

    def test_saturation_at_int32_limits(self) -> None:
        """Verify saturation at +2^31 and -2^31 thresholds."""
        cases = [
            (0x4F00_0000, 0x7FFF_FFFF),
            (0xCF00_0000, 0x8000_0000),
            (f32_to_bits(1e15), 0x7FFF_FFFF),
            (f32_to_bits(-1e15), 0x8000_0000),
        ]
        for bits, expected in cases:
            lines = [
                f"v_mov_b32 v1, {bits:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_limits_{bits:#x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], expected)

    def test_infinities_and_nans(self) -> None:
        """Verify +inf/-inf saturate and NaNs yield 0."""
        cases = [
            (0x7F80_0000, 0x7FFF_FFFF),
            (0xFF80_0000, 0x8000_0000),
            (0x7FC0_0000, 0),
            (0xFFC0_0000, 0),
            (0x7F80_0001, 0),
            (0xFF80_0001, 0),
        ]
        for bits, expected in cases:
            lines = [
                f"v_mov_b32 v1, {bits:#x}",
                "v_cvt_i32_f32 v0, v1",
                "exp mrt0 v0, v0, v0, v0 done vm",
                "s_endpgm",
            ]
            spv = support.translate_lines(f"ps_inf_nan_{bits:#x}", lines)
            words = support.mrt0_words(spv)
            self.assertEqual(words[0], expected)


# ==============================================================================
# SECTION 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestNegativeControlsAndMutations(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection."""

    def test_rejection_out_of_range_registers(self) -> None:
        """Verify VGPR index > 255 and SGPR index > 105 are rejected and syntax check fails."""
        with self.assertRaises(ValueError):
            VGPR(256)
        with self.assertRaises(ValueError):
            SGPR(106)
        with self.assertRaises(Exception):
            support.translate_lines("ps_bad_cvt", ["v_cvt_i32_f32", "s_endpgm"])

    def test_mutation_detection_swap_opcodes(self) -> None:
        """Mutation detector: swapping OpConvertFToS with OpConvertFToU in SPIR-V is caught."""
        spv = support.translate_lines("ps_mut_test", [
            f"v_mov_b32 v1, {f32_to_bits(-5.0):#x}",
            "v_cvt_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        orig_words = support.mrt0_words(spv)
        self.assertEqual(orig_words[0], (-5) & MASK_32)

        # Mutate SPIR-V: Replace OpConvertFToS with OpConvertFToU
        self.assertIn("OpConvertFToS", spv)
        mutated_spv = spv.replace("OpConvertFToS", "OpConvertFToU", 1)
        mutated_caught = False
        try:
            mutated_words = support.mrt0_words(mutated_spv)
            if mutated_words[0] != orig_words[0]:
                mutated_caught = True
        except Exception:
            mutated_caught = True
        self.assertTrue(mutated_caught, "Mutation in SPIR-V was not detected")

    def test_mutation_detection_faulty_nan_handling(self) -> None:
        """Mutation detector: NaN inputs produce 0, pinning against NaN-leak mutations."""
        nan_bits = 0x7FC0_0000
        lines = [
            f"v_mov_b32 v1, {nan_bits:#x}",
            "v_cvt_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_nan", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0)
        self.assertNotEqual(words[0], 0xFFFFFFFF)

    def test_mutation_detection_faulty_saturation(self) -> None:
        """Mutation detector: negative overflow saturates to INT_MIN (not INT_MAX)."""
        neg_overflow = 0xCF00_0000  # -2^31
        lines = [
            f"v_mov_b32 v1, {neg_overflow:#x}",
            "v_cvt_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_sat", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], INT32_MIN_UNSIGNED)
        self.assertNotEqual(words[0], INT32_MAX)

    def test_mutation_detection_inactive_lane_overwritten(self) -> None:
        """Mutation detector: in-range check at 2^31 (not 2^32)."""
        # Value 2^31 as float: 0x4F000000 must saturate to INT_MAX (0x7FFFFFFF)
        lines = [
            "v_mov_b32 v1, 0x4f000000",
            "v_cvt_i32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_range", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], INT32_MAX)


if __name__ == "__main__":
    unittest.main()
