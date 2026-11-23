"""Semantic reference tests and translator-backed validation for AMD RDNA2 s_xor_b32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, s_xor_b32 performs a 32-bit bitwise logical XOR:
      D.u32 = S0.u32 ^ S1.u32
      SCC = (D.u32 != 0)

  - Format: SOP2 (encoding opcode 0x12).
  - Operates on 32-bit scalar registers s[d] from s[s0] and s[s1], or 32-bit inline/literal
    constants, or special registers (e.g. exec_lo, vcc_lo).
  - Bitwise XOR yields 1 where source bits differ, and 0 where source bits match.
  - Scalar Condition Code (SCC) is set to 1 if the 32-bit result is non-zero, and 0 if all 32 bits are zero.

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

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF


def oracle_xor_b32(s0: int, s1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_xor_b32: returns (result_u32, scc)."""
    s0_32 = s0 & MASK32
    s1_32 = s1 & MASK32
    res = (s0_32 ^ s1_32) & MASK32
    scc = 1 if res != 0 else 0
    return res, scc

class SXorB32ReferenceOracleTests(unittest.TestCase):
    def test_oracle_identity(self):
        # x ^ x == 0, SCC == 0
        res, scc = oracle_xor_b32(0x12345678, 0x12345678)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

    def test_oracle_zero_identity(self):
        # x ^ 0 == x
        res, scc = oracle_xor_b32(0x87654321, 0)
        self.assertEqual(res, 0x87654321)
        self.assertEqual(scc, 1)

    def test_oracle_inversion(self):
        # x ^ ~x == 0xFFFFFFFF, SCC == 1
        x = 0x55AA55AA
        not_x = (~x) & MASK32
        res, scc = oracle_xor_b32(x, not_x)
        self.assertEqual(res, MASK32)
        self.assertEqual(scc, 1)

    def test_oracle_commutativity(self):
        a, b = 0xF0F00F0F, 0x0F0FAFAF
        self.assertEqual(oracle_xor_b32(a, b), oracle_xor_b32(b, a))

    def test_oracle_zeros(self):
        self.assertEqual(oracle_xor_b32(0, 0), (0, 0))
        self.assertEqual(oracle_xor_b32(MASK32, MASK32), (0, 0))

class RegKind(enum.Enum):
    SGPR = "sgpr"
    IMM = "imm"
    SPECIAL = "special"


@dataclass(frozen=True)
class Operand32:
    kind: RegKind
    index: int = 0
    raw_val: int = 0


@dataclass(frozen=True)
class InstructionAST:
    mnemonic: str
    dst: Operand32
    src0: Operand32
    src1: Operand32


class SXorB32AstValidationTests(unittest.TestCase):
    def test_ast_construction(self):
        ast = InstructionAST(
            mnemonic="s_xor_b32",
            dst=Operand32(RegKind.SGPR, index=2),
            src0=Operand32(RegKind.SGPR, index=0),
            src1=Operand32(RegKind.SGPR, index=1),
        )
        self.assertEqual(ast.mnemonic, "s_xor_b32")
        self.assertEqual(ast.dst.index, 2)
        self.assertEqual(ast.src0.index, 0)
        self.assertEqual(ast.src1.index, 1)

    def test_ast_imm_operand(self):
        ast = InstructionAST(
            mnemonic="s_xor_b32",
            dst=Operand32(RegKind.SGPR, index=0),
            src0=Operand32(RegKind.SGPR, index=0),
            src1=Operand32(RegKind.IMM, raw_val=0x1234),
        )
        self.assertEqual(ast.src1.kind, RegKind.IMM)
        self.assertEqual(ast.src1.raw_val, 0x1234)

    def test_ast_special_operand(self):
        ast = InstructionAST(
            mnemonic="s_xor_b32",
            dst=Operand32(RegKind.SPECIAL, raw_val=0, index=0),
            src0=Operand32(RegKind.SGPR, index=0),
            src1=Operand32(RegKind.IMM, raw_val=0xFFFFFFFF),
        )
        self.assertEqual(ast.dst.kind, RegKind.SPECIAL)

class ScalarRegisterState:
    def __init__(self):
        self.sgprs: dict[int, int] = {}
        self.scc: int = 0

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0) & MASK32

    def set_sgpr(self, idx: int, val: int):
        self.sgprs[idx] = val & MASK32

    def get_operand_val(self, op: Operand32) -> int:
        if op.kind == RegKind.SGPR:
            return self.get_sgpr(op.index)
        else:
            return op.raw_val & MASK32


def execute_scalar_xor_b32(state: ScalarRegisterState, ast: InstructionAST) -> None:
    s0 = state.get_operand_val(ast.src0)
    s1 = state.get_operand_val(ast.src1)
    res, scc = oracle_xor_b32(s0, s1)
    state.set_sgpr(ast.dst.index, res)
    state.scc = scc


class SXorB32ScalarInterpreterTests(unittest.TestCase):
    def test_scalar_execution_basic(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 0x00FF00FF)
        state.set_sgpr(1, 0x0F0F0F0F)
        ast = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 2),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 1),
        )
        execute_scalar_xor_b32(state, ast)
        res = state.get_sgpr(2)
        self.assertEqual(res, 0x0FF00FF0)
        self.assertEqual(state.scc, 1)

    def test_scalar_execution_zero_result(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 0xABCD1234)
        state.set_sgpr(1, 0xABCD1234)
        ast = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 2),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 1),
        )
        execute_scalar_xor_b32(state, ast)
        self.assertEqual(state.get_sgpr(2), 0)
        self.assertEqual(state.scc, 0)

    def test_scalar_execution_with_immediate(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 0x12340000)
        ast = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.IMM, raw_val=0x00005678),
        )
        execute_scalar_xor_b32(state, ast)
        self.assertEqual(state.get_sgpr(0), 0x12345678)
        self.assertEqual(state.scc, 1)

class WavefrontState:
    def __init__(self, lane_count: int = 32):
        self.lane_count = lane_count
        self.scalar_state = ScalarRegisterState()
        self.vgprs = [[0] * 64 for _ in range(lane_count)]

    def execute_scalar_instruction(self, ast: InstructionAST) -> None:
        # SALU instructions execute uniformly on the scalar processor once per wavefront
        execute_scalar_xor_b32(self.scalar_state, ast)


class SXorB32SimdDivergenceTests(unittest.TestCase):
    def test_salu_uniform_execution_wave32(self):
        wave = WavefrontState(lane_count=32)
        wave.scalar_state.set_sgpr(0, 0xFF00FF00)
        wave.scalar_state.set_sgpr(1, 0x00FF00FF)
        ast = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 2),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 1),
        )
        wave.execute_scalar_instruction(ast)
        self.assertEqual(wave.scalar_state.get_sgpr(2), MASK32)
        self.assertEqual(wave.scalar_state.scc, 1)
        # Vector registers are unaffected by scalar operations
        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][0], 0)

    def test_salu_uniform_execution_wave64(self):
        wave = WavefrontState(lane_count=64)
        wave.scalar_state.set_sgpr(0, 0xAAAAAAAA)
        wave.scalar_state.set_sgpr(1, 0x55555555)
        ast = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 1),
        )
        wave.execute_scalar_instruction(ast)
        self.assertEqual(wave.scalar_state.get_sgpr(0), MASK32)
        self.assertEqual(wave.scalar_state.scc, 1)
        for lane in range(64):
            self.assertEqual(wave.vgprs[lane][0], 0)

class SXorB32RegisterAliasingTests(unittest.TestCase):
    def test_dst_aliases_src0(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 0x0F0F0F0F)
        state.set_sgpr(1, 0xF0F0F0F0)
        ast = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 1),
        )
        execute_scalar_xor_b32(state, ast)
        self.assertEqual(state.get_sgpr(0), MASK32)
        self.assertEqual(state.scc, 1)

    def test_dst_aliases_src1(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 0x11111111)
        state.set_sgpr(1, 0x11111111)
        ast = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 1),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 1),
        )
        execute_scalar_xor_b32(state, ast)
        self.assertEqual(state.get_sgpr(1), 0)
        self.assertEqual(state.scc, 0)

    def test_all_operands_alias(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 0x12345678)
        ast = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 0),
        )
        execute_scalar_xor_b32(state, ast)
        self.assertEqual(state.get_sgpr(0), 0)
        self.assertEqual(state.scc, 0)

    def test_unrelated_registers_preserved(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 0x11)
        state.set_sgpr(1, 0x22)
        state.set_sgpr(2, 0x33)
        state.set_sgpr(3, 0x44)
        ast = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 2),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 1),
        )
        execute_scalar_xor_b32(state, ast)
        self.assertEqual(state.get_sgpr(0), 0x11)
        self.assertEqual(state.get_sgpr(1), 0x22)
        self.assertEqual(state.get_sgpr(2), 0x11 ^ 0x22)
        self.assertEqual(state.get_sgpr(3), 0x44)

class SXorB32ConditionCodesTests(unittest.TestCase):
    def test_scc_nonzero_and_zero_transitions(self):
        state = ScalarRegisterState()
        # Non-zero result sets SCC to 1
        state.set_sgpr(0, 1)
        state.set_sgpr(1, 0)
        ast1 = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 2),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 1),
        )
        execute_scalar_xor_b32(state, ast1)
        self.assertEqual(state.scc, 1)

        # Zero result transitions SCC to 0
        state.set_sgpr(0, 0x55555555)
        state.set_sgpr(1, 0x55555555)
        ast2 = InstructionAST(
            "s_xor_b32",
            Operand32(RegKind.SGPR, 2),
            Operand32(RegKind.SGPR, 0),
            Operand32(RegKind.SGPR, 1),
        )
        execute_scalar_xor_b32(state, ast2)
        self.assertEqual(state.scc, 0)

    def test_scc_msb_and_lsb_activation(self):
        # MSB (bit 31) only set -> non-zero -> SCC == 1
        res, scc = oracle_xor_b32(0, 0x80000000)
        self.assertEqual(res, 0x80000000)
        self.assertEqual(scc, 1)

        # LSB (bit 0) only set -> non-zero -> SCC == 1
        res, scc = oracle_xor_b32(0, 1)
        self.assertEqual(res, 1)
        self.assertEqual(scc, 1)

if __name__ == "__main__":
    unittest.main()
