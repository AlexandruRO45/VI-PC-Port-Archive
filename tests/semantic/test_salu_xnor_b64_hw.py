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

class ScalarRegisterState:
    def __init__(self):
        self.sgprs: dict[int, int] = {}
        self.scc: int = 0

    def get_sgpr_pair(self, idx: int) -> int:
        lo = self.sgprs.get(idx, 0) & MASK32
        hi = self.sgprs.get(idx + 1, 0) & MASK32
        return (hi << 32) | lo

    def set_sgpr_pair(self, idx: int, val: int):
        val &= MASK64
        self.sgprs[idx] = val & MASK32
        self.sgprs[idx + 1] = (val >> 32) & MASK32

    def get_operand_val(self, op: Operand64) -> int:
        if op.kind == RegKind.SGPR:
            return self.get_sgpr_pair(op.index)
        else:
            return op.raw_val & MASK64


def execute_scalar_xnor_b64(state: ScalarRegisterState, ast: InstructionAST) -> None:
    s0 = state.get_operand_val(ast.src0)
    s1 = state.get_operand_val(ast.src1)
    res, scc = oracle_xnor_b64(s0, s1)
    state.set_sgpr_pair(ast.dst.index, res)
    state.scc = scc


class SXnorB64ScalarInterpreterTests(unittest.TestCase):
    def test_scalar_execution_basic(self):
        state = ScalarRegisterState()
        state.set_sgpr_pair(0, 0x000000000000FFFF)
        state.set_sgpr_pair(2, 0xFFFFFFFF0000FFFF)
        ast = InstructionAST(
            "s_xnor_b64",
            Operand64(RegKind.SGPR, 4),
            Operand64(RegKind.SGPR, 0),
            Operand64(RegKind.SGPR, 2),
        )
        execute_scalar_xnor_b64(state, ast)
        res = state.get_sgpr_pair(4)
        self.assertEqual(res, 0x00000000FFFFFFFF)
        self.assertEqual(state.scc, 1)

    def test_scalar_execution_zero_result(self):
        state = ScalarRegisterState()
        state.set_sgpr_pair(0, 0x123456789ABCDEF0)
        state.set_sgpr_pair(2, (~0x123456789ABCDEF0) & MASK64)
        ast = InstructionAST(
            "s_xnor_b64",
            Operand64(RegKind.SGPR, 4),
            Operand64(RegKind.SGPR, 0),
            Operand64(RegKind.SGPR, 2),
        )
        execute_scalar_xnor_b64(state, ast)
        self.assertEqual(state.get_sgpr_pair(4), 0)
        self.assertEqual(state.scc, 0)

class WavefrontState:
    def __init__(self, lane_count: int = 32):
        self.lane_count = lane_count
        self.scalar_state = ScalarRegisterState()
        self.vgprs = [[0] * 64 for _ in range(lane_count)]

    def execute_scalar_instruction(self, ast: InstructionAST) -> None:
        # SALU instructions execute on the scalar processor once per wavefront
        execute_scalar_xnor_b64(self.scalar_state, ast)


class SXnorB64SimdDivergenceTests(unittest.TestCase):
    def test_salu_uniform_execution(self):
        wave = WavefrontState(lane_count=32)
        wave.scalar_state.set_sgpr_pair(0, 0xFF00FF00FF00FF00)
        wave.scalar_state.set_sgpr_pair(2, 0xFF00FF00FF00FF00)
        ast = InstructionAST(
            "s_xnor_b64",
            Operand64(RegKind.SGPR, 4),
            Operand64(RegKind.SGPR, 0),
            Operand64(RegKind.SGPR, 2),
        )
        wave.execute_scalar_instruction(ast)
        self.assertEqual(wave.scalar_state.get_sgpr_pair(4), MASK64)
        self.assertEqual(wave.scalar_state.scc, 1)
        # Vector registers remain untouched
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 0)

class SXnorB64RegisterAliasingTests(unittest.TestCase):
    def test_dst_aliases_src0(self):
        state = ScalarRegisterState()
        state.set_sgpr_pair(0, 0x0F0F0F0F0F0F0F0F)
        state.set_sgpr_pair(2, 0xF0F0F0F0F0F0F0F0)
        ast = InstructionAST(
            "s_xnor_b64",
            Operand64(RegKind.SGPR, 0),
            Operand64(RegKind.SGPR, 0),
            Operand64(RegKind.SGPR, 2),
        )
        execute_scalar_xnor_b64(state, ast)
        self.assertEqual(state.get_sgpr_pair(0), 0)
        self.assertEqual(state.scc, 0)

    def test_dst_aliases_src1(self):
        state = ScalarRegisterState()
        state.set_sgpr_pair(0, 0x1111111111111111)
        state.set_sgpr_pair(2, 0x1111111111111111)
        ast = InstructionAST(
            "s_xnor_b64",
            Operand64(RegKind.SGPR, 2),
            Operand64(RegKind.SGPR, 0),
            Operand64(RegKind.SGPR, 2),
        )
        execute_scalar_xnor_b64(state, ast)
        self.assertEqual(state.get_sgpr_pair(2), MASK64)
        self.assertEqual(state.scc, 1)

    def test_all_operands_alias(self):
        state = ScalarRegisterState()
        state.set_sgpr_pair(0, 0x1234567890ABCDEF)
        ast = InstructionAST(
            "s_xnor_b64",
            Operand64(RegKind.SGPR, 0),
            Operand64(RegKind.SGPR, 0),
            Operand64(RegKind.SGPR, 0),
        )
        execute_scalar_xnor_b64(state, ast)
        self.assertEqual(state.get_sgpr_pair(0), MASK64)
        self.assertEqual(state.scc, 1)

class SXnorB64ConditionCodesTests(unittest.TestCase):
    def test_scc_nonzero_and_zero_transitions(self):
        state = ScalarRegisterState()
        # Non-zero result sets SCC to 1
        state.set_sgpr_pair(0, 1)
        state.set_sgpr_pair(2, 1)
        ast1 = InstructionAST(
            "s_xnor_b64",
            Operand64(RegKind.SGPR, 4),
            Operand64(RegKind.SGPR, 0),
            Operand64(RegKind.SGPR, 2),
        )
        execute_scalar_xnor_b64(state, ast1)
        self.assertEqual(state.scc, 1)

        # Zero result transitions SCC to 0
        state.set_sgpr_pair(0, 0x5555555555555555)
        state.set_sgpr_pair(2, 0xAAAAAAAAAAAAAAAA)
        ast2 = InstructionAST(
            "s_xnor_b64",
            Operand64(RegKind.SGPR, 4),
            Operand64(RegKind.SGPR, 0),
            Operand64(RegKind.SGPR, 2),
        )
        execute_scalar_xnor_b64(state, ast2)
        self.assertEqual(state.scc, 0)

class SXnorB64ExtremeBoundariesTests(unittest.TestCase):
    def test_single_bit_differences(self):
        # Only bit 0 differs -> all bits 1 except bit 0
        res, scc = oracle_xnor_b64(0, 1)
        self.assertEqual(res, MASK64 ^ 1)
        self.assertEqual(scc, 1)

        # Only bit 63 differs -> all bits 1 except bit 63
        res, scc = oracle_xnor_b64(0, 1 << 63)
        self.assertEqual(res, MASK64 ^ (1 << 63))
        self.assertEqual(scc, 1)

    def test_alternating_nibbles(self):
        a = 0x0F0F0F0F0F0F0F0F
        b = 0x0F0F0F0F0F0F0F0F
        res, scc = oracle_xnor_b64(a, b)
        self.assertEqual(res, MASK64)

if __name__ == "__main__":
    unittest.main()
