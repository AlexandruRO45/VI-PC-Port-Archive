from __future__ import annotations

"""v_cvt_u32_f32 (RDNA2 VOP1) hardware-referenced semantic regression tests.

Covers:
- Float-to-unsigned-integer conversion with round-to-zero (truncation):
    D.u32 = convert_f32_to_u32(S0.f32)
- Positive overflow saturation: values >= 2^32 and +Infinity saturate to 0xFFFFFFFF
- Negative value saturation: all negative values (< 0.0) and -Infinity saturate to 0
- NaN handling: all quiet and signaling NaNs evaluate to 0
- Positive and negative subnormals truncate to 0
- Register operand variations: VGPRs, SGPRs, literals, and in-place register aliasing
- Full translation to SPIR-V via tools/rdna2spirv.py and execution via spirv_mini_interp
- Negative controls, syntax rejection, and mutation detection

Standard library only: python -I, unittest, struct, math. Strictly zero game data.
"""

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402


def f32_to_bits(f: float) -> int:
    """Convert float to IEEE 754 32-bit uint bit representation."""
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(bits: int) -> float:
    """Convert 32-bit uint bit pattern to IEEE 754 float."""
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def ref_cvt_u32_f32(bits: int) -> int:
    """Reference mathematical oracle for RDNA2 v_cvt_u32_f32."""
    bits &= 0xFFFFFFFF
    # Check for NaN: exponent is 0xFF and mantissa != 0
    if (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0:
        return 0
    # Negative floats (including -0.0, negative subnormals, and -inf) saturate to 0
    if bits >= 0x80000000:
        return 0
    # +Infinity saturates to 0xFFFFFFFF
    if bits == 0x7F800000:
        return 0xFFFFFFFF
    f = bits_to_f32(bits)
    # Positive overflow: values >= 2^32 saturate to 0xFFFFFFFF
    if f >= 4294967296.0:
        return 0xFFFFFFFF
    # In range: round towards zero (truncate)
    return int(f)


def run_cvt_u32_f32(instruction: str, seeds: dict[str, int], dst_reg: str = "v0") -> tuple[int, str]:
    """Translate and run a minimal pixel shader exercising `v_cvt_u32_f32`, returning dst word and spv."""
    prog = []
    for reg, val in sorted(seeds.items()):
        prefix = "s" if reg.startswith("s") else "v"
        prog.append(f"{prefix}_mov_b32 {reg}, {val:#x}")
    prog.append(instruction)
    prog.append(f"exp mrt0 {dst_reg}, {dst_reg}, {dst_reg}, {dst_reg} done vm")
    prog.append("s_endpgm")
    spv = support.translate_lines(f"ps_cvt_{dst_reg}_{abs(hash(instruction)) % 10000:04d}", prog)
    return support.mrt0_words(spv)[0], spv


class TestValuCvtU32F32OracleAndExecution(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and hardware conversion execution."""

    def test_oracle_small_fractions_truncate_to_zero(self) -> None:
        """Floats in [0.0, 1.0) truncate to uint32 0."""
        cases = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9999, 0.9999999]
        for fval in cases:
            bits = f32_to_bits(fval)
            with self.subTest(fval=fval):
                word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": bits})
                self.assertEqual(word, 0)
                self.assertEqual(word, ref_cvt_u32_f32(bits))

    def test_oracle_exact_integers(self) -> None:
        """Exact integer floats convert directly to corresponding uint32 value."""
        cases = [
            0.0, 1.0, 2.0, 3.0, 10.0, 42.0, 100.0, 255.0, 65535.0,
            16777215.0,  # 2^24 - 1
            16777216.0,  # 2^24
            33554432.0,  # 2^25
        ]
        for fval in cases:
            bits = f32_to_bits(fval)
            expected = int(fval)
            with self.subTest(fval=fval):
                word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": bits})
                self.assertEqual(word, expected)
                self.assertEqual(word, ref_cvt_u32_f32(bits))

    def test_oracle_large_uint32_range(self) -> None:
        """Large positive floats within [2^24, 2^32) convert correctly."""
        cases = [
            1073741824.0,  # 2^30
            2147483648.0,  # 2^31
            2147484672.0,  # 2^31 + 1024
            3221225472.0,  # 3 * 2^30
            4294967040.0,  # largest float strictly < 2^32 (0x4F7FFFFF)
        ]
        for fval in cases:
            bits = f32_to_bits(fval)
            expected = int(fval)
            with self.subTest(fval=fval):
                word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": bits})
                self.assertEqual(word, expected)
                self.assertEqual(word, ref_cvt_u32_f32(bits))

    def test_truncation_rounding_behavior(self) -> None:
        """v_cvt_u32_f32 truncates towards zero (not round-to-nearest)."""
        cases = [
            (1.1, 1),
            (1.5, 1),
            (1.99, 1),
            (42.8, 42),
            (100.999, 100),
            (65535.75, 65535),
            (16777216.5, 16777216),
            (2147483648.5, 2147483648),
        ]
        for fval, exp in cases:
            bits = f32_to_bits(fval)
            with self.subTest(fval=fval):
                word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": bits})
                self.assertEqual(word, exp)
                self.assertEqual(word, ref_cvt_u32_f32(bits))


class TestValuCvtU32F32SaturationAndBoundaries(unittest.TestCase):
    """Stage 2: Positive overflow saturation and negative clamping boundaries."""

    def test_positive_overflow_saturation(self) -> None:
        """Values >= 2^32 and +Infinity saturate to 0xFFFFFFFF."""
        cases = [
            0x4F800000,  # 2^32 exactly (4294967296.0)
            0x4F800001,  # 4294967296.0 + 1024.0
            0x50000000,  # 2^33
            0x5F800000,  # 2^52
            0x7F000000,  # near float max
            0x7F7FFFFF,  # float max
            0x7F800000,  # +Infinity
        ]
        for bits in cases:
            with self.subTest(bits=hex(bits)):
                word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": bits})
                self.assertEqual(word, 0xFFFFFFFF)
                self.assertEqual(word, ref_cvt_u32_f32(bits))

    def test_saturation_threshold_exact_boundary(self) -> None:
        """Check exact threshold: 0x4F7FFFFF (< 2^32) converts; 0x4F800000 (== 2^32) saturates."""
        just_below = 0x4F7FFFFF  # 4294967040.0
        word_below, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": just_below})
        self.assertEqual(word_below, 4294967040)
        self.assertNotEqual(word_below, 0xFFFFFFFF)

        exactly_at = 0x4F800000  # 4294967296.0
        word_at, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": exactly_at})
        self.assertEqual(word_at, 0xFFFFFFFF)

    def test_negative_values_clamp_to_zero(self) -> None:
        """All negative floats and -Infinity clamp to 0."""
        cases = [
            0x80000000,  # -0.0
            0x80000001,  # smallest negative subnormal
            0x807FFFFF,  # largest negative subnormal
            0xBDCCCCCD,  # -0.1
            0xBF000000,  # -0.5
            0xBF800000,  # -1.0
            0xC0000000,  # -2.0
            0xC2C80000,  # -100.0
            0xDF800000,  # large negative float
            0xFF7FFFFF,  # -float max
            0xFF800000,  # -Infinity
        ]
        for bits in cases:
            with self.subTest(bits=hex(bits)):
                word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": bits})
                self.assertEqual(word, 0)
                self.assertEqual(word, ref_cvt_u32_f32(bits))

    def test_nan_evaluates_to_zero(self) -> None:
        """All NaN operands (quiet, signaling, positive, negative) yield 0."""
        cases = [
            0x7FC00000,  # Quiet NaN
            0x7F800001,  # Signaling NaN
            0x7FA00000,  # Payload NaN
            0x7FFFFFFF,  # Max positive NaN
            0xFFC00000,  # Negative Quiet NaN
            0xFF800001,  # Negative Signaling NaN
            0xFFA00000,  # Negative Payload NaN
            0xFFFFFFFF,  # Max negative NaN
        ]
        for bits in cases:
            with self.subTest(bits=hex(bits)):
                word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": bits})
                self.assertEqual(word, 0)
                self.assertEqual(word, ref_cvt_u32_f32(bits))

    def test_subnormals_truncate_to_zero(self) -> None:
        """Subnormal floats truncate to 0."""
        cases = [
            0x00000001,  # Smallest positive subnormal
            0x00000042,
            0x007FFFFF,  # Largest positive subnormal
            0x80000001,  # Smallest negative subnormal
            0x807FFFFF,  # Largest negative subnormal
        ]
        for bits in cases:
            with self.subTest(bits=hex(bits)):
                word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": bits})
                self.assertEqual(word, 0)
                self.assertEqual(word, ref_cvt_u32_f32(bits))


class TestValuCvtU32F32RegisterOperandsAndForms(unittest.TestCase):
    """Stage 3: Register operand variations, literals, and aliasing."""

    def test_vgpr_source(self) -> None:
        """Standard VGPR source conversion."""
        bits = f32_to_bits(12345.0)
        word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": bits})
        self.assertEqual(word, 12345)

    def test_sgpr_source(self) -> None:
        """Scalar SGPR source conversion."""
        bits = f32_to_bits(54321.0)
        word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, s2", {"s2": bits})
        self.assertEqual(word, 54321)

    def test_register_aliasing_in_place(self) -> None:
        """In-place register aliasing: v0 = v_cvt_u32_f32 v0."""
        bits = f32_to_bits(98765.0)
        word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v0", {"v0": bits})
        self.assertEqual(word, 98765)

    def test_non_zero_destination_register(self) -> None:
        """Conversion writing to a non-zero destination VGPR."""
        bits = f32_to_bits(31415.0)
        word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v3, v1", {"v1": bits}, dst_reg="v3")
        self.assertEqual(word, 31415)

    def test_literal_constant_source(self) -> None:
        """Literal constant 32-bit float source."""
        word, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, 0x42280000", {})  # 42.0
        self.assertEqual(word, 42)


class TestValuCvtU32F32SPIRVAndNegativeControls(unittest.TestCase):
    """Stage 4: SPIR-V structure emission, syntax rejection, and mutation detection."""

    def test_spirv_instruction_structure(self) -> None:
        """Verify that translation produces proper OpConvertFToU %uint."""
        prog = [
            f"v_mov_b32 v1, {f32_to_bits(500.5):#x}",
            "v_cvt_u32_f32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_cvt_u32_f32_struct", prog)
        self.assertIn("OpConvertFToU %uint", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 500)

    def test_syntax_rejection_insufficient_operands(self) -> None:
        """Missing source operand must be rejected during translation."""
        if support is not None and getattr(support, "translate_lines", None) is not None:
            with self.assertRaises(Exception):
                support.translate_lines("ps_invalid", ["v_cvt_u32_f32 v0", "s_endpgm"])

    def test_syntax_tolerance_extra_operands(self) -> None:
        """Surplus operands in VOP1 assembly are permissively accepted by the translator."""
        if support is not None and getattr(support, "translate_lines", None) is not None:
            spv = support.translate_lines("ps_extra", ["v_cvt_u32_f32 v0, v1, v2", "s_endpgm"])
            self.assertIn("OpConvertFToU %uint", spv)

    def test_mutation_positive_overflow_guard(self) -> None:
        """Mutant producing 0xFFFFFFFE on overflow instead of 0xFFFFFFFF is detected."""
        overflow_val = 0x4F800000  # 2^32
        translated, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": overflow_val})
        self.assertEqual(translated, 0xFFFFFFFF)
        # Verify mutant value fails equality
        mutant_val = 0xFFFFFFFE
        self.assertNotEqual(translated, mutant_val)

    def test_mutation_negative_clamp_guard(self) -> None:
        """Mutant failing to clamp negative values to 0 is detected."""
        neg_val = 0xBF800000  # -1.0
        translated, _ = run_cvt_u32_f32("v_cvt_u32_f32 v0, v1", {"v1": neg_val})
        self.assertEqual(translated, 0)
        # If signed conversion were used, -1.0 -> 0xFFFFFFFF (or negative)
        signed_mutant = 0xFFFFFFFF
        self.assertNotEqual(translated, signed_mutant)


if __name__ == "__main__":
    unittest.main()
