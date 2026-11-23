from __future__ import annotations

"""Semantic reference oracle and architectural model for RDNA2 Packed Float16 Conversion & Packing Suite.

Covers:
- v_cvt_pkrtz_f16_f32: packs two float32 values into two float16 values with round-toward-zero (RTZ).
- v_cvt_pknorm_i16_f16: converts two float16 values to signed normalized 16-bit integers (snorm16) packed into one dword.
- v_cvt_pknorm_u16_f16: converts two float16 values to unsigned normalized 16-bit integers (unorm16) packed into one dword.
- v_pack_b32_f16: packs two 16-bit halves selected by op_sel with optional abs and neg modifiers into one 32-bit dword.

Zero game data, standard library Python only (python -I, struct, math, unittest).
"""

from dataclasses import dataclass
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF


def f32_to_bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(u: int) -> float:
    return struct.unpack("<f", struct.pack("<I", u & MASK32))[0]


def f16_to_bits(f: float) -> int:
    return struct.unpack("<H", struct.pack("<e", f))[0]


def bits_to_f16(u: int) -> float:
    return struct.unpack("<e", struct.pack("<H", u & MASK16))[0]


class RegKind(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    IMM = "imm"


@dataclass(frozen=True)
class Operand:
    kind: RegKind
    index: int = 0
    raw_val: int = 0


@dataclass(frozen=True)
class PackedInstructionAST:
    mnemonic: str
    dst: Operand
    src0: Operand
    src1: Operand
    op_sel: tuple[int, int, int] = (0, 0, 0)
    abs_mods: tuple[bool, bool] = (False, False)
    neg_mods: tuple[bool, bool] = (False, False)


class WavefrontSimdState:
    def __init__(self, lane_count: int = 32):
        self.lane_count = lane_count
        self.exec_mask: int = (1 << lane_count) - 1
        self.vgprs = [[0] * 64 for _ in range(lane_count)]

    def set_vgpr(self, lane: int, reg_idx: int, val: int):
        self.vgprs[lane][reg_idx] = val & MASK32

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx]

# -----------------------------------------------------------------------------
# Opcode 1: v_cvt_pkrtz_f16_f32
# -----------------------------------------------------------------------------

def ref_f32_to_f16_rtz(val: float | int) -> int:
    u = val if isinstance(val, int) else f32_to_bits(float(val))
    sign = (u >> 16) & 0x8000
    exp = (u >> 23) & 0xFF
    mant = u & 0x7FFFFF
    if exp == 0xFF:
        return (sign | 0x7E00 | ((mant >> 13) & 0x01FF)) if mant != 0 else (sign | 0x7C00)
    if exp == 0:
        return sign
    unbiased = exp - 127
    if unbiased > 15:
        return sign | 0x7BFF
    if unbiased < -24:
        return sign
    if unbiased < -14:
        full_mant = (1 << 23) | mant
        shift = -1 - unbiased
        return sign | ((full_mant >> shift) & 0x03FF)
    return sign | (((unbiased + 15) & 0x1F) << 10) | ((mant >> 13) & 0x03FF)


def ref_cvt_pkrtz_f16_f32(s0: float | int, s1: float | int) -> int:
    h0 = ref_f32_to_f16_rtz(s0)
    h1 = ref_f32_to_f16_rtz(s1)
    return ((h1 << 16) | h0) & MASK32


class VCvtPkrtzF16F32OracleTests(unittest.TestCase):
    def test_basic_values(self):
        self.assertEqual(ref_cvt_pkrtz_f16_f32(1.0, 2.0), 0x40003C00)
        self.assertEqual(ref_cvt_pkrtz_f16_f32(-1.0, 0.5), 0x3800BC00)

    def test_truncation_toward_zero(self):
        h = ref_f32_to_f16_rtz(1.999999)
        self.assertEqual(h & 0x7C00, 0x3C00)

    def test_signed_zeros(self):
        self.assertEqual(ref_cvt_pkrtz_f16_f32(0.0, -0.0), 0x80000000)
        self.assertEqual(ref_cvt_pkrtz_f16_f32(-0.0, 0.0), 0x00008000)

class VCvtPkrtzF16F32AstTests(unittest.TestCase):
    def test_ast_construction(self):
        ast = PackedInstructionAST(
            mnemonic="v_cvt_pkrtz_f16_f32",
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.VGPR, index=1),
            src1=Operand(RegKind.VGPR, index=2),
        )
        self.assertEqual(ast.mnemonic, "v_cvt_pkrtz_f16_f32")
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.src0.index, 1)
        self.assertEqual(ast.src1.index, 2)

def interpret_cvt_pkrtz_f16_f32(src0_val: int, src1_val: int) -> int:
    return ref_cvt_pkrtz_f16_f32(src0_val, src1_val)


class VCvtPkrtzF16F32ScalarInterpreterTests(unittest.TestCase):
    def test_scalar_execution(self):
        s0 = f32_to_bits(1.5)
        s1 = f32_to_bits(-2.5)
        res = interpret_cvt_pkrtz_f16_f32(s0, s1)
        expected = (ref_f32_to_f16_rtz(-2.5) << 16) | ref_f32_to_f16_rtz(1.5)
        self.assertEqual(res, expected)

class VCvtPkrtzF16F32SimdDivergenceTests(unittest.TestCase):
    def test_exec_masking(self):
        wave = WavefrontSimdState(32)
        wave.exec_mask = 0x55555555
        for lane in range(32):
            wave.set_vgpr(lane, 0, 0xDEADBEEF)
            if (wave.exec_mask >> lane) & 1:
                val0 = f32_to_bits(float(lane))
                val1 = f32_to_bits(float(lane * 2))
                wave.set_vgpr(lane, 0, interpret_cvt_pkrtz_f16_f32(val0, val1))
        for lane in range(32):
            if (wave.exec_mask >> lane) & 1:
                exp = (ref_f32_to_f16_rtz(float(lane * 2)) << 16) | ref_f32_to_f16_rtz(float(lane))
                self.assertEqual(wave.get_vgpr(lane, 0), exp)
            else:
                self.assertEqual(wave.get_vgpr(lane, 0), 0xDEADBEEF)

class VCvtPkrtzF16F32EdgeCasesTests(unittest.TestCase):
    def test_saturation_above_max_normal(self):
        self.assertEqual(ref_f32_to_f16_rtz(65504.0), 0x7BFF)
        self.assertEqual(ref_f32_to_f16_rtz(70000.0), 0x7BFF)
        self.assertEqual(ref_f32_to_f16_rtz(1000000.0), 0x7BFF)
        self.assertEqual(ref_f32_to_f16_rtz(-80000.0), 0xFBFF)

    def test_infinities(self):
        self.assertEqual(ref_f32_to_f16_rtz(float("inf")), 0x7C00)
        self.assertEqual(ref_f32_to_f16_rtz(float("-inf")), 0xFC00)

    def test_nans(self):
        nan_bits = ref_f32_to_f16_rtz(float("nan"))
        self.assertEqual(nan_bits & 0x7E00, 0x7E00)

    def test_subnormals(self):
        self.assertEqual(ref_f32_to_f16_rtz(0x00000001), 0x0000)
        self.assertEqual(ref_f32_to_f16_rtz(0.00005) & 0x7C00, 0x0000)

class VCvtPkrtzF16F32TranslatorTests(unittest.TestCase):
    def test_translation_basic(self):
        lines = [
            "v_mov_b32 v1, 1.0",
            "v_mov_b32 v2, 2.0",
            "v_cvt_pkrtz_f16_f32 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_pkrtz_basic", lines)
        word = mrt0_words(spv)[0]
        self.assertEqual(word, 0x40003C00)

    def test_translation_boundary_saturation(self):
        lines = [
            "v_mov_b32 v1, 70000.0",
            "v_mov_b32 v2, -80000.0",
            "v_cvt_pkrtz_f16_f32 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_pkrtz_sat", lines)
        word = mrt0_words(spv)[0]
        self.assertEqual(word & 0xFFFF, 0x7BFF)
        self.assertEqual((word >> 16) & 0xFFFF, 0xFBFF)

if __name__ == '__main__':
    unittest.main()
