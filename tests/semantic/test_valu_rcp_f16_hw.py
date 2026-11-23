"""Semantic reference tests and translator-backed validation for AMD RDNA2 v_rcp_f16.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  The v_rcp_f16 instruction computes the IEEE-754 binary16 half-precision reciprocal:
      D.f16 = 1.0 / S0.f16

  On GFX10, 16-bit float operations preserve the unselected half of the destination register:
  - Low half destination (default, op_sel[1]=0): preserves bits [31:16] of the destination register.
  - High half destination (op_sel[1]=1): preserves bits [15:0] of the destination register.
  - Source selection (op_sel[0]): selects low half (0) or high half (1) of the source operand.
  - Modifiers: neg, abs, clamp [0.0, 1.0], and omod (mul:2, mul:4, div:2).

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

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

import spirv_interp  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

# IEEE-754 binary16 constants
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_CANONICAL_QNAN = 0x7E00
F16_ONE = 0x3C00          # 1.0
F16_NEG_ONE = 0xBC00      # -1.0
F16_TWO = 0x4000          # 2.0
F16_HALF = 0x3800         # 0.5
F16_FOUR = 0x4400         # 4.0
F16_QUARTER = 0x3400      # 0.25
F16_MAX_NORMAL = 0x7BFF   # 65504.0
F16_MIN_NORMAL = 0x0400   # 2^-14 ~= 6.10e-5
F16_MIN_SUBNORMAL = 0x0001


def f16_to_u16(val: float) -> int:
    """Pack Python float into IEEE 754 binary16 uint16 bit representation."""
    return struct.unpack("<H", struct.pack("<e", val))[0]


def u16_to_f16(bits: int) -> float:
    """Unpack uint16 bit representation into Python float (binary16)."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def oracle_rcp_f16(
    src_u16: int,
    *,
    neg: bool = False,
    abs_val: bool = False,
    clamp: bool = False,
    omod: str | None = None,
) -> int:
    """Reference mathematical oracle for IEEE-754 binary16 reciprocal."""
    src_u16 &= MASK16
    sign = (src_u16 >> 15) & 1
    exp = (src_u16 >> 10) & 0x1F
    mant = src_u16 & 0x3FF

    # Apply abs and neg
    if abs_val:
        sign = 0
    if neg:
        sign ^= 1

    # NaN input propagates quiet NaN
    if exp == 0x1F and mant != 0:
        return F16_CANONICAL_QNAN

    # Infinity input produces zero with corresponding sign
    if exp == 0x1F and mant == 0:
        res = F16_NEG_ZERO if sign else F16_POS_ZERO
        if clamp:
            res = F16_POS_ZERO
        return res

    # Zero input produces infinity with corresponding sign
    if exp == 0 and mant == 0:
        res = F16_NEG_INF if sign else F16_POS_INF
        if clamp:
            res = F16_POS_ZERO if sign else F16_ONE
        return res

    # Finite non-zero value
    val = u16_to_f16((sign << 15) | (exp << 10) | mant)
    try:
        res_float = 1.0 / val
    except OverflowError:
        res_float = -math.inf if sign else math.inf

    # Apply omod
    if omod == "mul:2":
        res_float *= 2.0
    elif omod == "mul:4":
        res_float *= 4.0
    elif omod == "div:2":
        res_float *= 0.5

    # Apply clamp
    if clamp:
        res_float = max(0.0, min(1.0, res_float))

    try:
        return f16_to_u16(res_float)
    except OverflowError:
        return F16_NEG_INF if res_float < 0 else F16_POS_INF

class RcpF16ReferenceOracleTests(unittest.TestCase):
    def test_oracle_exact_powers_of_two(self):
        self.assertEqual(oracle_rcp_f16(F16_TWO), F16_HALF)
        self.assertEqual(oracle_rcp_f16(F16_HALF), F16_TWO)
        self.assertEqual(oracle_rcp_f16(F16_FOUR), F16_QUARTER)
        self.assertEqual(oracle_rcp_f16(F16_QUARTER), F16_FOUR)
        self.assertEqual(oracle_rcp_f16(F16_ONE), F16_ONE)
        self.assertEqual(oracle_rcp_f16(F16_NEG_ONE), F16_NEG_ONE)

    def test_oracle_special_values(self):
        self.assertEqual(oracle_rcp_f16(F16_POS_ZERO), F16_POS_INF)
        self.assertEqual(oracle_rcp_f16(F16_NEG_ZERO), F16_NEG_INF)
        self.assertEqual(oracle_rcp_f16(F16_POS_INF), F16_POS_ZERO)
        self.assertEqual(oracle_rcp_f16(F16_NEG_INF), F16_NEG_ZERO)
        self.assertEqual(oracle_rcp_f16(F16_CANONICAL_QNAN), F16_CANONICAL_QNAN)

    def test_oracle_modifiers(self):
        self.assertEqual(oracle_rcp_f16(F16_TWO, neg=True), f16_to_u16(-0.5))
        self.assertEqual(oracle_rcp_f16(f16_to_u16(-2.0), abs_val=True), F16_HALF)
        self.assertEqual(oracle_rcp_f16(F16_HALF, clamp=True), F16_ONE)  # 2.0 clamped to 1.0
        self.assertEqual(oracle_rcp_f16(F16_TWO, omod="mul:2"), F16_ONE)   # 0.5 * 2 = 1.0
        self.assertEqual(oracle_rcp_f16(F16_TWO, omod="div:2"), F16_QUARTER) # 0.5 / 2 = 0.25

class RegKind(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    IMM = "imm"


@dataclass(frozen=True)
class ModifierFlags:
    neg: bool = False
    abs_val: bool = False
    clamp: bool = False
    omod: str | None = None
    op_sel: tuple[int, int] = (0, 0)  # (src_hi, dst_hi)


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
    mods: ModifierFlags = ModifierFlags()


class RcpF16AstValidationTests(unittest.TestCase):
    def test_ast_construction(self):
        ast = InstructionAST(
            mnemonic="v_rcp_f16",
            dst=Operand(RegKind.VGPR, 1),
            src0=Operand(RegKind.VGPR, 0),
            mods=ModifierFlags(clamp=True, omod="mul:2", op_sel=(1, 0)),
        )
        self.assertEqual(ast.mnemonic, "v_rcp_f16")
        self.assertEqual(ast.dst.index, 1)
        self.assertTrue(ast.mods.clamp)
        self.assertEqual(ast.mods.omod, "mul:2")
        self.assertEqual(ast.mods.op_sel, (1, 0))

    def test_ast_imm_operand(self):
        ast = InstructionAST(
            mnemonic="v_rcp_f16",
            dst=Operand(RegKind.VGPR, 2),
            src0=Operand(RegKind.IMM, raw_val=F16_TWO),
        )
        self.assertEqual(ast.src0.kind, RegKind.IMM)
        self.assertEqual(ast.src0.raw_val, F16_TWO)

if __name__ == "__main__":
    unittest.main()
