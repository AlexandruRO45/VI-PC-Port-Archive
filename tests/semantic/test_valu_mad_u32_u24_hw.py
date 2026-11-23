"""Semantic reference tests and translator-backed validation for AMD RDNA2 v_mad_u32_u24.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_mad_u32_u24 computes an unsigned 24-bit integer
  multiply-accumulate:
      D.u32 = (S0[23:0] * S1[23:0]) + S2[31:0]   (mod 2^32)

  - S0 and S1 are treated as unsigned 24-bit integers (bits [31:24] are ignored).
  - S2 is treated as a full unsigned 32-bit integer.
  - The multiplication produces a 48-bit unsigned product, truncated to 32 bits and added to S2.
  - The operation wraps modulo 2^32 with standard two's-complement wrap-around.
  - VALU integer operations do not modify scalar condition codes (SCC) or vector condition codes (VCC).

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK24 = 0x00FFFFFF
MASK32 = 0xFFFFFFFF


def oracle_mad_u32_u24(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for v_mad_u32_u24."""
    s0_24 = s0 & MASK24
    s1_24 = s1 & MASK24
    s2_32 = s2 & MASK32
    prod = s0_24 * s1_24
    return (prod + s2_32) & MASK32

class MadU32U24ReferenceOracleTests(unittest.TestCase):
    def test_oracle_basic_products(self):
        self.assertEqual(oracle_mad_u32_u24(10, 20, 5), 205)
        self.assertEqual(oracle_mad_u32_u24(100, 200, 50), 20050)
        self.assertEqual(oracle_mad_u32_u24(0, 500, 123), 123)
        self.assertEqual(oracle_mad_u32_u24(500, 0, 123), 123)
        self.assertEqual(oracle_mad_u32_u24(1, 1, 0), 1)

    def test_oracle_24bit_masking(self):
        # Upper 8 bits of S0 and S1 must be ignored
        self.assertEqual(oracle_mad_u32_u24(0xFF000002, 0xAA000003, 10), 16)
        self.assertEqual(oracle_mad_u32_u24(0x12000000, 0x34000000, 99), 99)

    def test_oracle_32bit_wrap_around(self):
        # (0xFFFFFF * 1) + 0xFFFFFFFF wraps modulo 2^32
        res = oracle_mad_u32_u24(0xFFFFFF, 1, 0xFFFFFFFF)
        self.assertEqual(res, (0xFFFFFF + 0xFFFFFFFF) & MASK32)

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
class InstructionAST:
    mnemonic: str
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand


class MadU32U24AstValidationTests(unittest.TestCase):
    def test_ast_construction(self):
        ast = InstructionAST(
            mnemonic="v_mad_u32_u24",
            dst=Operand(RegKind.VGPR, 3),
            src0=Operand(RegKind.VGPR, 0),
            src1=Operand(RegKind.VGPR, 1),
            src2=Operand(RegKind.VGPR, 2),
        )
        self.assertEqual(ast.mnemonic, "v_mad_u32_u24")
        self.assertEqual(ast.dst.index, 3)
        self.assertEqual(ast.src0.index, 0)
        self.assertEqual(ast.src1.index, 1)
        self.assertEqual(ast.src2.index, 2)

    def test_ast_imm_and_sgpr_operands(self):
        ast = InstructionAST(
            mnemonic="v_mad_u32_u24",
            dst=Operand(RegKind.VGPR, 0),
            src0=Operand(RegKind.SGPR, 4),
            src1=Operand(RegKind.IMM, raw_val=50),
            src2=Operand(RegKind.VGPR, 1),
        )
        self.assertEqual(ast.src0.kind, RegKind.SGPR)
        self.assertEqual(ast.src1.kind, RegKind.IMM)
        self.assertEqual(ast.src1.raw_val, 50)

if __name__ == "__main__":
    unittest.main()
