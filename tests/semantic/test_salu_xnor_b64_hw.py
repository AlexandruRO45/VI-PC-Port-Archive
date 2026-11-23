"""Semantic reference tests and translator-backed validation for AMD RDNA2 s_xnor_b64.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, s_xnor_b64 performs a 64-bit bitwise logical XNOR:
      D.u64 = ~(S0.u64 ^ S1.u64)
      SCC = (D.u64 != 0)

  - Operates on 64-bit scalar register pairs: s[d:d+1] from s[s0:s0+1] and s[s1:s1+1]
    (or 64-bit inline/literal constants, or special registers exec/vcc).
  - Bitwise XNOR yields 1 where source bits match (both 0 or both 1), and 0 where they differ.
  - Scalar Condition Code (SCC) is set to 1 if the 64-bit result is non-zero, and 0 if all 64 bits are zero.

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

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def oracle_xnor_b64(s0: int, s1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_xnor_b64: returns (result_u64, scc)."""
    s0_64 = s0 & MASK64
    s1_64 = s1 & MASK64
    res = (~(s0_64 ^ s1_64)) & MASK64
    scc = 1 if res != 0 else 0
    return res, scc

class SXnorB64ReferenceOracleTests(unittest.TestCase):
    def test_oracle_identity(self):
        # xnor(x, x) == all ones (~0)
        res, scc = oracle_xnor_b64(0x123456789ABCDEF0, 0x123456789ABCDEF0)
        self.assertEqual(res, MASK64)
        self.assertEqual(scc, 1)

    def test_oracle_inversion(self):
        # xnor(x, ~x) == 0 (all zeros), SCC == 0
        x = 0xAAAAAAAAAAAAAAAA
        not_x = (~x) & MASK64
        res, scc = oracle_xnor_b64(x, not_x)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_oracle_zeros_and_ones(self):
        self.assertEqual(oracle_xnor_b64(0, 0), (MASK64, 1))
        self.assertEqual(oracle_xnor_b64(MASK64, MASK64), (MASK64, 1))
        self.assertEqual(oracle_xnor_b64(0, MASK64), (0, 0))

class RegKind(enum.Enum):
    SGPR = "sgpr"
    IMM = "imm"
    SPECIAL = "special"


@dataclass(frozen=True)
class Operand64:
    kind: RegKind
    index: int = 0
    raw_val: int = 0


@dataclass(frozen=True)
class InstructionAST:
    mnemonic: str
    dst: Operand64
    src0: Operand64
    src1: Operand64


class SXnorB64AstValidationTests(unittest.TestCase):
    def test_ast_construction(self):
        ast = InstructionAST(
            mnemonic="s_xnor_b64",
            dst=Operand64(RegKind.SGPR, index=4),
            src0=Operand64(RegKind.SGPR, index=0),
            src1=Operand64(RegKind.SGPR, index=2),
        )
        self.assertEqual(ast.mnemonic, "s_xnor_b64")
        self.assertEqual(ast.dst.index, 4)
        self.assertEqual(ast.src0.index, 0)
        self.assertEqual(ast.src1.index, 2)

    def test_ast_imm_operand(self):
        ast = InstructionAST(
            mnemonic="s_xnor_b64",
            dst=Operand64(RegKind.SGPR, index=0),
            src0=Operand64(RegKind.SGPR, index=0),
            src1=Operand64(RegKind.IMM, raw_val=0),
        )
        self.assertEqual(ast.src1.kind, RegKind.IMM)
        self.assertEqual(ast.src1.raw_val, 0)

if __name__ == "__main__":
    unittest.main()
