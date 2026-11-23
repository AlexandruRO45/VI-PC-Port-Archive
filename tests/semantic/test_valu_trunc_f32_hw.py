from __future__ import annotations

"""Semantic translator hardware verification for AMD RDNA2 v_trunc_f32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_trunc_f32 belongs to the VOP1 encoding format.
  Instruction:
    v_trunc_f32 vd, src0

  Operands:
    vd   = 32-bit destination VGPR
    src0 = 32-bit source operand: VGPR, SGPR, inline constant, or literal float

  Semantics:
    Truncates floating-point value to an integer, rounding toward zero.
    Result is encoded as an IEEE 754 single-precision float.
    vd = trunc(src0)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""

import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFFFFFF
POS_ZERO_F32: int = 0x00000000
NEG_ZERO_F32: int = 0x80000000
POS_INF_F32: int = 0x7F800000
NEG_INF_F32: int = 0xFF800000
QNAN_F32: int = 0x7FC00000


def u32_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f32_to_u32(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


class TestValuTruncF32HardwareExecution(unittest.TestCase):
    """End-to-end SPIR-V translation and fragment execution for v_trunc_f32."""

    def _exec(self, src0_hex: int, clamp: bool = False, src_sgpr: bool = False, alias: bool = False) -> tuple[int, str]:
        clamp_str = " clamp" if clamp else ""
        if src_sgpr:
            lines = [
                f"s_mov_b32 s4, {src0_hex:#x}",
                f"v_trunc_f32 v1, s4{clamp_str}",
                "exp mrt0 v1, v1, v1, v1 done vm",
                "s_endpgm",
            ]
        elif alias:
            lines = [
                f"v_mov_b32 v1, {src0_hex:#x}",
                f"v_trunc_f32 v1, v1{clamp_str}",
                "exp mrt0 v1, v1, v1, v1 done vm",
                "s_endpgm",
            ]
        else:
            lines = [
                f"v_mov_b32 v0, {src0_hex:#x}",
                f"v_trunc_f32 v1, v0{clamp_str}",
                "exp mrt0 v1, v1, v1, v1 done vm",
                "s_endpgm",
            ]
        spv = support.translate_lines(f"ps_trunc_{src0_hex & 0xFFFFFFFF:08x}{'_c' if clamp else ''}", lines)
        words = support.mrt0_words(spv)
        return words[0], spv

    def test_spirv_emission(self) -> None:
        lines = [
            "v_mov_b32 v0, 0x40b00000",
            "v_trunc_f32 v1, v0",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_trunc_f32_emit", lines)
        self.assertIn("OpExtInst %float %glsl Trunc", spv)
        self.assertNotIn("Floor", spv)
        self.assertNotIn("Ceil", spv)

    def test_hardware_execution_positive(self) -> None:
        word, _ = self._exec(0x40B00000)  # +5.5
        self.assertEqual(u32_to_f32(word), 5.0)

    def test_hardware_execution_negative(self) -> None:
        word, _ = self._exec(0xC0B00000)  # -5.5
        self.assertEqual(u32_to_f32(word), -5.0)

    def test_translator_positive_fractionals(self) -> None:
        cases = [
            (f32_to_u32(1.5), f32_to_u32(1.0)),
            (f32_to_u32(0.75), f32_to_u32(0.0)),
            (f32_to_u32(5.9), f32_to_u32(5.0)),
            (f32_to_u32(0.25), f32_to_u32(0.0)),
            (f32_to_u32(99.99), f32_to_u32(99.0)),
            (f32_to_u32(0.999), f32_to_u32(0.0)),
        ]
        for src, expected in cases:
            with self.subTest(src=hex(src)):
                word, _ = self._exec(src)
                self.assertEqual(word, expected)

    def test_translator_negative_fractionals_catches_floor_mutation(self) -> None:
        # Negative fractionals are the critical differentiator between Trunc and Floor
        cases = [
            (f32_to_u32(-1.5), f32_to_u32(-1.0)),   # floor gives -2.0
            (f32_to_u32(-5.5), f32_to_u32(-5.0)),   # floor gives -6.0
            (f32_to_u32(-0.75), NEG_ZERO_F32),      # floor gives -1.0
            (f32_to_u32(-5.9), f32_to_u32(-5.0)),   # floor gives -6.0
            (f32_to_u32(-0.1), NEG_ZERO_F32),       # floor gives -1.0
            (f32_to_u32(-99.99), f32_to_u32(-99.0)),
        ]
        for src, expected in cases:
            with self.subTest(src=hex(src)):
                word, _ = self._exec(src)
                self.assertEqual(word, expected)

    def test_translator_integral_values_preserved(self) -> None:
        cases = [0.0, 1.0, -1.0, 42.0, 1000000.0, 8388608.0]
        for val in cases:
            u = f32_to_u32(val)
            with self.subTest(val=val):
                word, _ = self._exec(u)
                self.assertEqual(word, u)

    def test_translator_special_values_passthrough(self) -> None:
        for val in [POS_INF_F32, NEG_INF_F32, QNAN_F32]:
            with self.subTest(val=hex(val)):
                word, _ = self._exec(val)
                self.assertEqual(word, val)

    def test_translator_signed_zeros(self) -> None:
        word_pos, _ = self._exec(POS_ZERO_F32)
        word_neg, _ = self._exec(NEG_ZERO_F32)
        self.assertEqual(word_pos, POS_ZERO_F32)
        self.assertEqual(word_neg, NEG_ZERO_F32)

    def test_translator_subnormals_truncate_to_zero(self) -> None:
        # Denormals truncate to signed zeros
        word_pos, _ = self._exec(0x00000001)
        word_neg, _ = self._exec(0x80000001)
        self.assertEqual(word_pos, POS_ZERO_F32)
        self.assertEqual(word_neg, NEG_ZERO_F32)

    def test_translator_sgpr_source(self) -> None:
        word, _ = self._exec(0x40B00000, src_sgpr=True)  # 5.5 in s4
        self.assertEqual(u32_to_f32(word), 5.0)

    def test_translator_register_aliasing(self) -> None:
        word, _ = self._exec(0x40B00000, alias=True)  # v_trunc_f32 v1, v1
        self.assertEqual(u32_to_f32(word), 5.0)

    def test_translator_clamp_modifier(self) -> None:
        # +5.5 truncates to +5.0, clamp saturates to 1.0 (0x3F800000)
        word_pos, spv_pos = self._exec(0x40B00000, clamp=True)
        self.assertEqual(word_pos, 0x3F800000)
        self.assertIn("OpExtInst %float %glsl FClamp", spv_pos)

        # -5.5 truncates to -5.0, clamp saturates to 0.0 (0x00000000)
        word_neg, spv_neg = self._exec(0xC0B00000, clamp=True)
        self.assertEqual(word_neg, 0x00000000)
        self.assertIn("OpExtInst %float %glsl FClamp", spv_neg)

    def test_translator_mutation_trunc_vs_floor_ceil_distinguisher(self) -> None:
        """Explicitly proves that the translator output distinguishes Trunc from Floor and Ceil."""
        # For -5.5: trunc is -5.0, floor is -6.0, ceil is -5.0
        neg_val = f32_to_u32(-5.5)
        translated_neg, _ = self._exec(neg_val)
        floor_neg = f32_to_u32(float(math.floor(-5.5)))
        self.assertEqual(translated_neg, f32_to_u32(-5.0))
        self.assertNotEqual(translated_neg, floor_neg)

        # For +5.5: trunc is 5.0, floor is 5.0, ceil is 6.0
        pos_val = f32_to_u32(5.5)
        translated_pos, _ = self._exec(pos_val)
        ceil_pos = f32_to_u32(float(math.ceil(5.5)))
        self.assertEqual(translated_pos, f32_to_u32(5.0))
        self.assertNotEqual(translated_pos, ceil_pos)

    def test_translator_syntax_rejection_insufficient_operands(self) -> None:
        with self.assertRaises(Exception):
            support.translate_lines("ps_invalid", ["v_trunc_f32 v0", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
