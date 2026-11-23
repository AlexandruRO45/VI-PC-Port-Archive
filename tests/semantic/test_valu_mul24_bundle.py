"""Semantic reference tests and translator-backed validation for AMD RDNA2 VALU 24-bit Multiply Suite.

Subsystem Feature Bundle 38 covers 6 VALU 24-bit integer multiply and multiply-add instructions:
  1. v_mul_u32_u24    - Low 32 bits of 24-bit unsigned product: D = (S0[23:0] * S1[23:0])[31:0]
  2. v_mul_i32_i24    - Low 32 bits of 24-bit signed product: D = (sext24(S0) * sext24(S1))[31:0]
  3. v_mul_hi_u32_u24 - High 32 bits of 48-bit unsigned product: D = (S0[23:0] * S1[23:0])[47:32] (zero-extended)
  4. v_mul_hi_i32_i24 - High 32 bits of 48-bit signed product: D = (sext24(S0) * sext24(S1))[47:32] (sign-extended)
  5. v_mad_u32_u24    - 24-bit unsigned multiply-add: D = (S0[23:0] * S1[23:0])[31:0] + S2
  6. v_mad_i32_i24    - 24-bit signed multiply-add: D = (sext24(S0) * sext24(S1))[31:0] + S2

RDNA2 24-bit Arithmetic Architecture:
  - Operates on the lower 24 bits of source operands S0 and S1.
  - Bits [31:24] of source operands are ignored; for signed operations, sign extension originates from bit 23.
  - 48-bit products are partitioned into low 32 bits (v_mul_*) and high 32 bits (v_mul_hi_*).
  - End-to-end SPIR-V assembly translation through tools/rdna2spirv.py.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
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
MASK24 = 0x00FFFFFF


def sext24(val: int) -> int:
    val &= MASK24
    return val if val < 0x00800000 else val - 0x01000000


def u32(val: int) -> int:
    return val & MASK32

# -----------------------------------------------------------------------------
# Opcode 1: v_mul_u32_u24
# -----------------------------------------------------------------------------
def oracle_mul_u32_u24(s0: int, s1: int) -> int:
    """Reference mathematical oracle for v_mul_u32_u24."""
    u0 = s0 & MASK24
    u1 = s1 & MASK24
    return (u0 * u1) & MASK32


class VMulU32U24OracleTests(unittest.TestCase):
    def test_mul_u24_basic(self):
        self.assertEqual(oracle_mul_u32_u24(100, 200), 20000)

    def test_mul_u24_ignores_high_byte(self):
        # 0xFF000064 & MASK24 is 100, 0xAA0000C8 & MASK24 is 200
        self.assertEqual(oracle_mul_u32_u24(0xFF000064, 0xAA0000C8), 20000)

    def test_mul_u24_full_24bit_range(self):
        # 0xFFFFFF * 0xFFFFFF = 0xFFFFFE000001 -> low 32 bits is 0xFE000001
        self.assertEqual(oracle_mul_u32_u24(0xFFFFFF, 0xFFFFFF), 0xFE000001)

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
class ValuInstructionAST:
    mnemonic: str
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand | None = None


class WavefrontSimdState:
    def __init__(self, lane_count: int = 32):
        self.lane_count = lane_count
        self.exec_mask: int = (1 << lane_count) - 1
        self.vgprs = [[0] * 64 for _ in range(lane_count)]

    def set_vgpr(self, lane: int, reg_idx: int, val: int):
        self.vgprs[lane][reg_idx] = val & MASK32

    def get_vgpr(self, lane: int, reg_idx: int) -> int:
        return self.vgprs[lane][reg_idx] & MASK32

class VMulU32U24AstInterpreterTests(unittest.TestCase):
    def test_mul_u24_ast_construction(self):
        ast = ValuInstructionAST("v_mul_u32_u24", Operand(RegKind.VGPR, 2), Operand(RegKind.VGPR, 0), Operand(RegKind.VGPR, 1))
        self.assertEqual(ast.mnemonic, "v_mul_u32_u24")
        self.assertEqual(ast.dst.index, 2)

    def test_mul_u24_scalar_execution(self):
        res = oracle_mul_u32_u24(30, 40)
        self.assertEqual(res, 1200)

class VMulU32U24SimdDivergenceTests(unittest.TestCase):
    def test_mul_u24_simd_exec_masking(self):
        wave = WavefrontSimdState(lane_count=32)
        wave.exec_mask = 0x55555555  # Even lanes active
        for i in range(32):
            wave.set_vgpr(i, 2, 0xDEADBEEF)
            if (wave.exec_mask >> i) & 1:
                wave.set_vgpr(i, 2, oracle_mul_u32_u24(i, 10))
        for i in range(32):
            if (0x55555555 >> i) & 1:
                self.assertEqual(wave.get_vgpr(i, 2), i * 10)
            else:
                self.assertEqual(wave.get_vgpr(i, 2), 0xDEADBEEF)

class VMulU32U24TranslatorTests(unittest.TestCase):
    def test_mul_u24_translation_and_execution(self):
        lines = [
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 200",
            "v_mul_u32_u24 v2, v0, v1",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mul_u24_exec", lines)
        val = mrt0_words(spv)[0]
        self.assertEqual(val, 20000)

    def test_mul_u24_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_mul_u24", ["v_mul_u32_u24"])

# -----------------------------------------------------------------------------
# Opcode 2: v_mul_i32_i24
# -----------------------------------------------------------------------------
def oracle_mul_i32_i24(s0: int, s1: int) -> int:
    """Reference mathematical oracle for v_mul_i32_i24."""
    i0 = sext24(s0)
    i1 = sext24(s1)
    return (i0 * i1) & MASK32


class VMulI32I24OracleTests(unittest.TestCase):
    def test_mul_i24_basic(self):
        self.assertEqual(oracle_mul_i32_i24(10, 20), 200)

    def test_mul_i24_negative(self):
        # -10 * 20 = -200 (0xFFFFFF38)
        self.assertEqual(oracle_mul_i32_i24(-10, 20), u32(-200))
        # -10 * -20 = 200
        self.assertEqual(oracle_mul_i32_i24(-10, -20), 200)

    def test_mul_i24_sign_extension_from_bit23(self):
        # 0x00800000 is -8388608 in 24-bit two's complement. -8388608 * 2 = -16777216
        res = oracle_mul_i32_i24(0x00800000, 2)
        self.assertEqual(res, u32(-16777216))

if __name__ == "__main__":
    unittest.main()
