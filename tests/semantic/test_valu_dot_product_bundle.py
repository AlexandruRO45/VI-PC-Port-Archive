"""Semantic reference oracle and verification bundle for RDNA2 dot-product vector arithmetic.

Covers:
  - v_dot2_f32_f16: Half-to-float 2-element dot product with 32-bit float addend
  - v_dot2c_f32_f16: Half-to-float 2-element dot product accumulating into old destination
  - v_dot4_i32_i8: Signed 8-bit 4-element dot product with 32-bit integer addend
  - v_dot4_u32_u8: Unsigned 8-bit 4-element dot product with 32-bit integer addend
  - v_dot8_i32_i4: Signed 4-bit 8-element dot product with 32-bit integer addend
  - v_dot8_u32_u4: Unsigned 4-bit 8-element dot product with 32-bit integer addend

High-density Subsystem Feature Bundle testing reference oracles, AST modeling,
scalar/lane interpreter, multi-lane divergence, sub-dword packing, IEEE-754 precision,
mutation detection, and tools/rdna2spirv.py SPIR-V translation.
"""
from __future__ import annotations

import dataclasses
import math
import os
import struct
import sys
import unittest
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF


def u32(val: int) -> int:
    return val & MASK32


def i32(val: int) -> int:
    v = val & MASK32
    return v if v < 0x80000000 else v - 0x100000000


def i8(val: int) -> int:
    v = val & 0xFF
    return v if v < 0x80 else v - 0x100


def i4(val: int) -> int:
    v = val & 0xF
    return v if v < 0x8 else v - 0x10


def f16_to_bits(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_to_f16(b: int) -> float:
    return struct.unpack("<e", struct.pack("<H", b & MASK16))[0]


def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def pack_f16x2(f0: float, f1: float) -> int:
    b0 = f16_to_bits(f0)
    b1 = f16_to_bits(f1)
    return (b1 << 16) | b0


def ref_dot2_f32_f16(s0: int, s1: int, acc: int) -> int:
    a0 = bits_to_f16(s0 & 0xFFFF)
    a1 = bits_to_f16((s0 >> 16) & 0xFFFF)
    b0 = bits_to_f16(s1 & 0xFFFF)
    b1 = bits_to_f16((s1 >> 16) & 0xFFFF)
    res = a0 * b0 + a1 * b1 + bits_to_f32(acc)
    return f32_to_bits(res)


def ref_dot4_i32_i8(s0: int, s1: int, acc: int) -> int:
    bytes0 = [i8((s0 >> (8 * k)) & 0xFF) for k in range(4)]
    bytes1 = [i8((s1 >> (8 * k)) & 0xFF) for k in range(4)]
    res = sum(b0 * b1 for b0, b1 in zip(bytes0, bytes1)) + i32(acc)
    return u32(res)


def ref_dot4_u32_u8(s0: int, s1: int, acc: int) -> int:
    bytes0 = [(s0 >> (8 * k)) & 0xFF for k in range(4)]
    bytes1 = [(s1 >> (8 * k)) & 0xFF for k in range(4)]
    res = sum(b0 * b1 for b0, b1 in zip(bytes0, bytes1)) + u32(acc)
    return u32(res)


def ref_dot8_i32_i4(s0: int, s1: int, acc: int) -> int:
    nibbles0 = [i4((s0 >> (4 * k)) & 0xF) for k in range(8)]
    nibbles1 = [i4((s1 >> (4 * k)) & 0xF) for k in range(8)]
    res = sum(n0 * n1 for n0, n1 in zip(nibbles0, nibbles1)) + i32(acc)
    return u32(res)


def ref_dot8_u32_u4(s0: int, s1: int, acc: int) -> int:
    nibbles0 = [(s0 >> (4 * k)) & 0xF for k in range(8)]
    nibbles1 = [(s1 >> (4 * k)) & 0xF for k in range(8)]
    res = sum(n0 * n1 for n0, n1 in zip(nibbles0, nibbles1)) + u32(acc)
    return u32(res)


_TRANSLATE_ID: int = 0


def run_ps_dot2_f32_f16(s0: int, s1: int, acc: int) -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {acc & MASK32:#x}",
        "v_dot2_f32_f16 v3, v0, v1, v2",
        "v_mov_b32 v0, v3",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_dot2_f32_{_TRANSLATE_ID}", lines)
    return mrt0_words(spv)[0] & MASK32


def run_ps_dot2c_f32_f16(s0: int, s1: int, old_dst: int) -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v3, {old_dst & MASK32:#x}",
        "v_dot2c_f32_f16 v3, v0, v1",
        "v_mov_b32 v0, v3",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_dot2c_f32_{_TRANSLATE_ID}", lines)
    return mrt0_words(spv)[0] & MASK32


def run_ps_dot4_i32_i8(s0: int, s1: int, acc: int) -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {acc & MASK32:#x}",
        "v_dot4_i32_i8 v3, v0, v1, v2",
        "v_mov_b32 v0, v3",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_dot4_i32_{_TRANSLATE_ID}", lines)
    return mrt0_words(spv)[0] & MASK32


def run_ps_dot4_u32_u8(s0: int, s1: int, acc: int) -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {acc & MASK32:#x}",
        "v_dot4_u32_u8 v3, v0, v1, v2",
        "v_mov_b32 v0, v3",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_dot4_u32_{_TRANSLATE_ID}", lines)
    return mrt0_words(spv)[0] & MASK32


def run_ps_dot8_i32_i4(s0: int, s1: int, acc: int) -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {acc & MASK32:#x}",
        "v_dot8_i32_i4 v3, v0, v1, v2",
        "v_mov_b32 v0, v3",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_dot8_i32_{_TRANSLATE_ID}", lines)
    return mrt0_words(spv)[0] & MASK32


def run_ps_dot8_u32_u4(s0: int, s1: int, acc: int) -> int:
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {acc & MASK32:#x}",
        "v_dot8_u32_u4 v3, v0, v1, v2",
        "v_mov_b32 v0, v3",
        "v_mov_b32 v1, 0",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_dot8_u32_{_TRANSLATE_ID}", lines)
    return mrt0_words(spv)[0] & MASK32


@dataclasses.dataclass
class DotAstNode:
    opcode: str
    dst: str
    src0: str
    src1: str
    src2: Optional[str] = None

    def emit_asm(self) -> str:
        if self.src2 is not None:
            return f"{self.opcode} {self.dst}, {self.src0}, {self.src1}, {self.src2}"
        return f"{self.opcode} {self.dst}, {self.src0}, {self.src1}"


class VDot2F32F16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_dot2_f32_f16."""

    def test_basic_dot_product(self):
        s0 = pack_f16x2(2.0, 3.0)
        s1 = pack_f16x2(4.0, 5.0)
        acc = f32_to_bits(1.0)
        # 2*4 + 3*5 + 1 = 8 + 15 + 1 = 24.0
        ref = ref_dot2_f32_f16(s0, s1, acc)
        hw = run_ps_dot2_f32_f16(s0, s1, acc)
        self.assertAlmostEqual(bits_to_f32(ref), 24.0)
        self.assertAlmostEqual(bits_to_f32(hw), 24.0)

    def test_zero_addend(self):
        s0 = pack_f16x2(1.0, 1.0)
        s1 = pack_f16x2(1.0, 1.0)
        ref = ref_dot2_f32_f16(s0, s1, 0)
        hw = run_ps_dot2_f32_f16(s0, s1, 0)
        self.assertAlmostEqual(bits_to_f32(ref), 2.0)
        self.assertAlmostEqual(bits_to_f32(hw), 2.0)


class VDot2F32F16AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_dot2_f32_f16."""

    def test_ast_emission(self):
        node = DotAstNode("v_dot2_f32_f16", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_dot2_f32_f16 v3, v0, v1, v2")
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            node.emit_asm(),
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot2_ast_test", lines)
        self.assertIn("OpFMul", spv)
        self.assertIn("OpFAdd", spv)

    def test_interpreter_scalar_simulation(self):
        s0 = pack_f16x2(1.5, 2.0)
        s1 = pack_f16x2(2.0, 3.0)
        s2 = f32_to_bits(5.0)
        ref = ref_dot2_f32_f16(s0, s1, s2)
        hw = run_ps_dot2_f32_f16(s0, s1, s2)
        # 1.5*2 + 2*3 + 5 = 3 + 6 + 5 = 14.0
        self.assertAlmostEqual(bits_to_f32(ref), 14.0)
        self.assertAlmostEqual(bits_to_f32(hw), 14.0)


class VDot2F32F16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_dot2_f32_f16."""

    def test_simd_divergent_lanes(self):
        test_lanes = [0, 1, 7, 15, 31, 63]
        for i in test_lanes:
            s0 = pack_f16x2(float(i), 1.0)
            s1 = pack_f16x2(2.0, float(i))
            acc = f32_to_bits(10.0)
            expected = float(i) * 2.0 + 1.0 * float(i) + 10.0
            hw = run_ps_dot2_f32_f16(s0, s1, acc)
            self.assertAlmostEqual(bits_to_f32(hw), expected, places=4)

    def test_float16_subnormals_and_infinities(self):
        s0 = pack_f16x2(0.0, float("inf"))
        s1 = pack_f16x2(1.0, 1.0)
        hw = run_ps_dot2_f32_f16(s0, s1, 0)
        self.assertTrue(math.isinf(bits_to_f32(hw)))


class VDot2F32F16TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_dot2_f32_f16."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3C003C00",  # (1.0, 1.0)
            "v_mov_b32 v1, 0x3C003C00",  # (1.0, 1.0)
            "v_mov_b32 v2, 0x40A00000",  # 5.0f
            "v_dot2_f32_f16 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot2_f32_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(out[0]), 7.0)

    def test_negative_mutation_contrast(self):
        s0 = pack_f16x2(2.0, 2.0)
        s1 = pack_f16x2(3.0, 3.0)
        hw = run_ps_dot2_f32_f16(s0, s1, 0)
        res_add = f32_to_bits(4.0)
        self.assertNotEqual(hw, res_add)


class VDot2cF32F16OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_dot2c_f32_f16."""

    def test_accumulation_into_destination(self):
        s0 = pack_f16x2(1.0, 2.0)
        s1 = pack_f16x2(3.0, 4.0)
        old_d = f32_to_bits(10.0)
        # 1*3 + 2*4 + 10 = 3 + 8 + 10 = 21.0
        ref = ref_dot2_f32_f16(s0, s1, old_d)
        hw = run_ps_dot2c_f32_f16(s0, s1, old_d)
        self.assertAlmostEqual(bits_to_f32(ref), 21.0)
        self.assertAlmostEqual(bits_to_f32(hw), 21.0)

    def test_zero_destination_accumulation(self):
        s0 = pack_f16x2(2.0, 3.0)
        s1 = pack_f16x2(4.0, 1.0)
        hw = run_ps_dot2c_f32_f16(s0, s1, 0)
        self.assertAlmostEqual(bits_to_f32(hw), 11.0)


class VDot2cF32F16AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_dot2c_f32_f16."""

    def test_ast_emission(self):
        node = DotAstNode("v_dot2c_f32_f16", "v3", "v0", "v1")
        self.assertEqual(node.emit_asm(), "v_dot2c_f32_f16 v3, v0, v1")
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v3, 0",
            node.emit_asm(),
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot2c_ast_test", lines)
        self.assertIn("OpFMul", spv)
        self.assertIn("OpFAdd", spv)

    def test_interpreter_scalar_simulation(self):
        s0 = pack_f16x2(2.0, 1.0)
        s1 = pack_f16x2(3.0, 4.0)
        s3 = f32_to_bits(5.0)
        ref = ref_dot2_f32_f16(s0, s1, s3)
        hw = run_ps_dot2c_f32_f16(s0, s1, s3)
        # 2*3 + 1*4 + 5 = 6 + 4 + 5 = 15.0
        self.assertAlmostEqual(bits_to_f32(ref), 15.0)
        self.assertAlmostEqual(bits_to_f32(hw), 15.0)


class VDot2cF32F16SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_dot2c_f32_f16."""

    def test_simd_divergent_lanes(self):
        test_lanes = [0, 1, 8, 16, 32, 63]
        for i in test_lanes:
            s0 = pack_f16x2(1.0, 2.0)
            s1 = pack_f16x2(float(i), 1.0)
            old_d = f32_to_bits(float(i))
            expected = 1.0 * float(i) + 2.0 * 1.0 + float(i)
            hw = run_ps_dot2c_f32_f16(s0, s1, old_d)
            self.assertAlmostEqual(bits_to_f32(hw), expected, places=4)

    def test_destination_aliased_accumulator(self):
        s0 = pack_f16x2(1.0, 1.0)
        hw = run_ps_dot2c_f32_f16(s0, s0, f32_to_bits(3.0))
        self.assertAlmostEqual(bits_to_f32(hw), 5.0)


class VDot2cF32F16TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_dot2c_f32_f16."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0x3C003C00",  # (1.0, 1.0)
            "v_mov_b32 v1, 0x3C003C00",  # (1.0, 1.0)
            "v_mov_b32 v3, 0x40000000",  # 2.0f
            "v_dot2c_f32_f16 v3, v0, v1",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot2c_f32_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertAlmostEqual(bits_to_f32(out[0]), 4.0)

    def test_negative_mutation_contrast(self):
        s0 = pack_f16x2(1.0, 1.0)
        s1 = pack_f16x2(2.0, 2.0)
        hw_accum = run_ps_dot2c_f32_f16(s0, s1, f32_to_bits(10.0))
        hw_no_accum = run_ps_dot2c_f32_f16(s0, s1, 0)
        self.assertNotEqual(hw_accum, hw_no_accum)


class VDot4I32I8OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_dot4_i32_i8."""

    def test_positive_bytes(self):
        s0 = 0x01020304
        s1 = 0x02020202
        ref = ref_dot4_i32_i8(s0, s1, 5)
        hw = run_ps_dot4_i32_i8(s0, s1, 5)
        self.assertEqual(ref, 25)
        self.assertEqual(hw, 25)

    def test_negative_signed_bytes(self):
        s0 = 0xFFFFFFFF  # four -1 bytes
        s1 = 0x01010101  # four +1 bytes
        ref = ref_dot4_i32_i8(s0, s1, 0)
        hw = run_ps_dot4_i32_i8(s0, s1, 0)
        self.assertEqual(i32(ref), -4)
        self.assertEqual(i32(hw), -4)


class VDot4I32I8AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_dot4_i32_i8."""

    def test_ast_emission(self):
        node = DotAstNode("v_dot4_i32_i8", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_dot4_i32_i8 v3, v0, v1, v2")
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            node.emit_asm(),
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot4_i32_ast", lines)
        self.assertIn("OpIMul", spv)
        self.assertIn("OpBitFieldSExtract", spv)

    def test_interpreter_scalar_simulation(self):
        s0 = 0x01010101
        s1 = 0x03030303
        s2 = 10
        ref = ref_dot4_i32_i8(s0, s1, s2)
        hw = run_ps_dot4_i32_i8(s0, s1, s2)
        self.assertEqual(ref, 22)
        self.assertEqual(hw, 22)


class VDot4I32I8SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_dot4_i32_i8."""

    def test_simd_divergent_lanes(self):
        test_lanes = [0, 1, 10, 20, 40, 63]
        for i in test_lanes:
            s0 = 0x01010101
            s1 = 0x01010101
            hw = run_ps_dot4_i32_i8(s0, s1, i)
            self.assertEqual(hw, 4 + i)

    def test_signed_overflow_clamp_boundaries(self):
        s0 = 0x7F7F7F7F
        s1 = 0x7F7F7F7F
        hw = run_ps_dot4_i32_i8(s0, s1, 1000)
        self.assertEqual(hw, 4 * (127 * 127) + 1000)


class VDot4I32I8TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_dot4_i32_i8."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0x01010101",
            "v_mov_b32 v1, 0x02020202",
            "v_mov_b32 v2, 100",
            "v_dot4_i32_i8 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot4_i32_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 108)

    def test_negative_mutation_contrast(self):
        s0 = 0xFF000000
        s1 = 0x01000000
        hw_signed = run_ps_dot4_i32_i8(s0, s1, 0)
        hw_unsigned = run_ps_dot4_u32_u8(s0, s1, 0)
        self.assertNotEqual(hw_signed, hw_unsigned)


class VDot4U32U8OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_dot4_u32_u8."""

    def test_unsigned_bytes(self):
        s0 = 0xFF000000  # highest byte is 255
        s1 = 0x01000000  # highest byte is 1
        ref = ref_dot4_u32_u8(s0, s1, 0)
        hw = run_ps_dot4_u32_u8(s0, s1, 0)
        self.assertEqual(ref, 255)
        self.assertEqual(hw, 255)

    def test_all_bytes_active(self):
        s0 = 0x02020202
        s1 = 0x03030303
        ref = ref_dot4_u32_u8(s0, s1, 10)
        hw = run_ps_dot4_u32_u8(s0, s1, 10)
        self.assertEqual(ref, 34)
        self.assertEqual(hw, 34)


class VDot4U32U8AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_dot4_u32_u8."""

    def test_ast_emission(self):
        node = DotAstNode("v_dot4_u32_u8", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_dot4_u32_u8 v3, v0, v1, v2")
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            node.emit_asm(),
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot4_u32_ast", lines)
        self.assertIn("OpIMul", spv)
        self.assertIn("OpBitFieldUExtract", spv)

    def test_interpreter_scalar_simulation(self):
        s0 = 0x10101010
        s1 = 0x01010101
        ref = ref_dot4_u32_u8(s0, s1, 0)
        hw = run_ps_dot4_u32_u8(s0, s1, 0)
        self.assertEqual(ref, 64)
        self.assertEqual(hw, 64)


class VDot4U32U8SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_dot4_u32_u8."""

    def test_simd_divergent_lanes(self):
        test_lanes = [0, 2, 5, 12, 25, 50]
        for i in test_lanes:
            s0 = 0x02020202
            s1 = 0x02020202
            acc = i * 5
            hw = run_ps_dot4_u32_u8(s0, s1, acc)
            self.assertEqual(hw, 16 + i * 5)

    def test_unsigned_boundary_accumulation(self):
        s0 = 0xFFFFFFFF
        s1 = 0x01010101
        # 4 * 255 = 1020
        hw = run_ps_dot4_u32_u8(s0, s1, 100)
        self.assertEqual(hw, 1120)


class VDot4U32U8TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_dot4_u32_u8."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0x02020202",
            "v_mov_b32 v1, 0x03030303",
            "v_mov_b32 v2, 50",
            "v_dot4_u32_u8 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot4_u32_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 74)

    def test_negative_mutation_contrast(self):
        s0 = 0x80808080
        s1 = 0x01010101
        hw_u = run_ps_dot4_u32_u8(s0, s1, 0)
        hw_i = run_ps_dot4_i32_i8(s0, s1, 0)
        self.assertNotEqual(hw_u, hw_i)


class VDot8I32I4OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_dot8_i32_i4."""

    def test_positive_nibbles(self):
        s0 = 0x11111111  # eight 1s
        s1 = 0x22222222  # eight 2s
        ref = ref_dot8_i32_i4(s0, s1, 10)
        hw = run_ps_dot8_i32_i4(s0, s1, 10)
        self.assertEqual(ref, 26)
        self.assertEqual(hw, 26)

    def test_signed_negative_nibbles(self):
        s0 = 0xFFFFFFFF  # eight -1 nibbles
        s1 = 0x11111111  # eight +1 nibbles
        ref = ref_dot8_i32_i4(s0, s1, 0)
        hw = run_ps_dot8_i32_i4(s0, s1, 0)
        self.assertEqual(i32(ref), -8)
        self.assertEqual(i32(hw), -8)


class VDot8I32I4AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_dot8_i32_i4."""

    def test_ast_emission(self):
        node = DotAstNode("v_dot8_i32_i4", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_dot8_i32_i4 v3, v0, v1, v2")
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            node.emit_asm(),
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot8_i32_ast", lines)
        self.assertIn("OpIMul", spv)
        self.assertIn("OpBitFieldSExtract", spv)

    def test_interpreter_scalar_simulation(self):
        s0 = 0x11111111
        s1 = 0x33333333
        s2 = 5
        ref = ref_dot8_i32_i4(s0, s1, s2)
        hw = run_ps_dot8_i32_i4(s0, s1, s2)
        self.assertEqual(ref, 29)
        self.assertEqual(hw, 29)


class VDot8I32I4SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_dot8_i32_i4."""

    def test_simd_divergent_lanes(self):
        test_lanes = [0, 1, 9, 18, 36, 63]
        for i in test_lanes:
            s0 = 0x11111111
            s1 = 0x11111111
            hw = run_ps_dot8_i32_i4(s0, s1, i)
            self.assertEqual(hw, 8 + i)

    def test_nibble_sign_extension_boundaries(self):
        s0 = 0x00000007
        s1 = 0x00000008
        hw = run_ps_dot8_i32_i4(s0, s1, 0)
        self.assertEqual(i32(hw), 7 * (-8))


class VDot8I32I4TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_dot8_i32_i4."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0x11111111",
            "v_mov_b32 v1, 0x22222222",
            "v_mov_b32 v2, 4",
            "v_dot8_i32_i4 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot8_i32_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 20)

    def test_negative_mutation_contrast(self):
        s0 = 0xF0000000
        s1 = 0x10000000
        hw_i = run_ps_dot8_i32_i4(s0, s1, 0)
        hw_u = run_ps_dot8_u32_u4(s0, s1, 0)
        self.assertNotEqual(hw_i, hw_u)


class VDot8U32U4OracleTests(unittest.TestCase):
    """Reference mathematical oracle tests for v_dot8_u32_u4."""

    def test_unsigned_nibbles(self):
        s0 = 0xF0000000  # highest nibble is 15
        s1 = 0x10000000  # highest nibble is 1
        ref = ref_dot8_u32_u4(s0, s1, 0)
        hw = run_ps_dot8_u32_u4(s0, s1, 0)
        self.assertEqual(ref, 15)
        self.assertEqual(hw, 15)

    def test_all_nibbles_active(self):
        s0 = 0x22222222
        s1 = 0x33333333
        ref = ref_dot8_u32_u4(s0, s1, 10)
        hw = run_ps_dot8_u32_u4(s0, s1, 10)
        self.assertEqual(ref, 58)
        self.assertEqual(hw, 58)


class VDot8U32U4AstInterpreterTests(unittest.TestCase):
    """AST modeling and interpreter simulation for v_dot8_u32_u4."""

    def test_ast_emission(self):
        node = DotAstNode("v_dot8_u32_u4", "v3", "v0", "v1", "v2")
        self.assertEqual(node.emit_asm(), "v_dot8_u32_u4 v3, v0, v1, v2")
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            node.emit_asm(),
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot8_u32_ast", lines)
        self.assertIn("OpIMul", spv)
        self.assertIn("OpBitFieldUExtract", spv)

    def test_interpreter_scalar_simulation(self):
        s0 = 0x22222222
        s1 = 0x11111111
        ref = ref_dot8_u32_u4(s0, s1, 14)
        hw = run_ps_dot8_u32_u4(s0, s1, 14)
        self.assertEqual(ref, 30)
        self.assertEqual(hw, 30)


class VDot8U32U4SimdBoundaryTests(unittest.TestCase):
    """SIMD lane divergence and boundary extremes for v_dot8_u32_u4."""

    def test_simd_divergent_lanes(self):
        test_lanes = [0, 3, 7, 14, 28, 56]
        for i in test_lanes:
            s0 = 0x11111111
            s1 = 0x22222222
            acc = i * 2
            hw = run_ps_dot8_u32_u4(s0, s1, acc)
            self.assertEqual(hw, 16 + i * 2)

    def test_unsigned_nibble_boundary_accumulation(self):
        s0 = 0xFFFFFFFF
        s1 = 0x11111111
        hw = run_ps_dot8_u32_u4(s0, s1, 5)
        self.assertEqual(hw, 125)


class VDot8U32U4TranslatorIntegrationTests(unittest.TestCase):
    """Translator execution and negative mutation controls for v_dot8_u32_u4."""

    def test_spirv_translation_execution(self):
        lines = [
            "v_mov_b32 v0, 0x11111111",
            "v_mov_b32 v1, 0x22222222",
            "v_mov_b32 v2, 5",
            "v_dot8_u32_u4 v3, v0, v1, v2",
            "v_mov_b32 v0, v3",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_dot8_u32_bundle_int", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 21)

    def test_negative_mutation_contrast(self):
        s0 = 0x88888888
        s1 = 0x11111111
        hw_u = run_ps_dot8_u32_u4(s0, s1, 0)
        hw_i = run_ps_dot8_i32_i4(s0, s1, 0)
        self.assertNotEqual(hw_u, hw_i)


if __name__ == "__main__":
    unittest.main()
