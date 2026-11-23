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

class ScalarRegisterState:
    def __init__(self):
        self.vgprs: dict[int, int] = {}
        self.sgprs: dict[int, int] = {}

    def get_vgpr(self, idx: int) -> int:
        return self.vgprs.get(idx, 0) & MASK32

    def set_vgpr(self, idx: int, val: int):
        self.vgprs[idx] = val & MASK32

    def get_operand_val(self, op: Operand) -> int:
        if op.kind == RegKind.VGPR:
            return self.get_vgpr(op.index)
        elif op.kind == RegKind.SGPR:
            return self.sgprs.get(op.index, 0) & MASK32
        else:
            return op.raw_val & MASK32


def execute_scalar_mad_u32_u24(state: ScalarRegisterState, ast: InstructionAST) -> None:
    s0 = state.get_operand_val(ast.src0)
    s1 = state.get_operand_val(ast.src1)
    s2 = state.get_operand_val(ast.src2)
    res = oracle_mad_u32_u24(s0, s1, s2)
    state.set_vgpr(ast.dst.index, res)


class MadU32U24ScalarInterpreterTests(unittest.TestCase):
    def test_scalar_execution_basic(self):
        state = ScalarRegisterState()
        state.set_vgpr(0, 15)
        state.set_vgpr(1, 4)
        state.set_vgpr(2, 7)
        ast = InstructionAST(
            "v_mad_u32_u24",
            Operand(RegKind.VGPR, 3),
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
            Operand(RegKind.VGPR, 2),
        )
        execute_scalar_mad_u32_u24(state, ast)
        self.assertEqual(state.get_vgpr(3), (15 * 4) + 7)

    def test_scalar_execution_with_sgpr_and_imm(self):
        state = ScalarRegisterState()
        state.sgprs[0] = 100
        state.set_vgpr(1, 25)
        ast = InstructionAST(
            "v_mad_u32_u24",
            Operand(RegKind.VGPR, 2),
            Operand(RegKind.SGPR, 0),
            Operand(RegKind.IMM, raw_val=3),
            Operand(RegKind.VGPR, 1),
        )
        execute_scalar_mad_u32_u24(state, ast)
        self.assertEqual(state.get_vgpr(2), (100 * 3) + 25)

class WavefrontState:
    def __init__(self, lane_count: int = 32):
        self.lane_count = lane_count
        self.exec_mask = (1 << lane_count) - 1
        self.lanes = [ScalarRegisterState() for _ in range(lane_count)]

    def execute_instruction(self, ast: InstructionAST) -> None:
        for lane in range(self.lane_count):
            if (self.exec_mask >> lane) & 1:
                execute_scalar_mad_u32_u24(self.lanes[lane], ast)


class MadU32U24SimdDivergenceTests(unittest.TestCase):
    def test_simd_wave32_all_active(self):
        wave = WavefrontState(lane_count=32)
        for i in range(32):
            wave.lanes[i].set_vgpr(0, i + 1)
            wave.lanes[i].set_vgpr(1, 2)
            wave.lanes[i].set_vgpr(2, 10)
        ast = InstructionAST(
            "v_mad_u32_u24",
            Operand(RegKind.VGPR, 3),
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
            Operand(RegKind.VGPR, 2),
        )
        wave.execute_instruction(ast)
        for i in range(32):
            self.assertEqual(wave.lanes[i].get_vgpr(3), ((i + 1) * 2) + 10)

    def test_simd_exec_divergence_mask(self):
        wave = WavefrontState(lane_count=32)
        wave.exec_mask = 0x0000FFFF  # Lower 16 lanes active only
        for i in range(32):
            wave.lanes[i].set_vgpr(0, 10)
            wave.lanes[i].set_vgpr(1, 20)
            wave.lanes[i].set_vgpr(2, 5)
            wave.lanes[i].set_vgpr(3, 0xDEADBEEF)
        ast = InstructionAST(
            "v_mad_u32_u24",
            Operand(RegKind.VGPR, 3),
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
            Operand(RegKind.VGPR, 2),
        )
        wave.execute_instruction(ast)
        for i in range(32):
            if i < 16:
                self.assertEqual(wave.lanes[i].get_vgpr(3), 205)
            else:
                self.assertEqual(wave.lanes[i].get_vgpr(3), 0xDEADBEEF)  # Inactive lanes unchanged

class MadU32U24RegisterAliasingTests(unittest.TestCase):
    def test_dst_aliases_src0(self):
        state = ScalarRegisterState()
        state.set_vgpr(0, 7)
        state.set_vgpr(1, 6)
        state.set_vgpr(2, 2)
        ast = InstructionAST(
            "v_mad_u32_u24",
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
            Operand(RegKind.VGPR, 2),
        )
        execute_scalar_mad_u32_u24(state, ast)
        self.assertEqual(state.get_vgpr(0), (7 * 6) + 2)

    def test_dst_aliases_src2(self):
        state = ScalarRegisterState()
        state.set_vgpr(0, 5)
        state.set_vgpr(1, 4)
        state.set_vgpr(2, 100)
        ast = InstructionAST(
            "v_mad_u32_u24",
            Operand(RegKind.VGPR, 2),
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
            Operand(RegKind.VGPR, 2),
        )
        execute_scalar_mad_u32_u24(state, ast)
        self.assertEqual(state.get_vgpr(2), (5 * 4) + 100)

    def test_all_operands_alias(self):
        state = ScalarRegisterState()
        state.set_vgpr(0, 3)
        ast = InstructionAST(
            "v_mad_u32_u24",
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 0),
        )
        execute_scalar_mad_u32_u24(state, ast)
        self.assertEqual(state.get_vgpr(0), (3 * 3) + 3)

class MadU32U24ConditionCodesTests(unittest.TestCase):
    def test_scc_and_vcc_isolation(self):
        initial_scc = 1
        initial_vcc = 0xDEADBEEFCAFEBABE
        state = ScalarRegisterState()
        state.set_vgpr(0, 10)
        state.set_vgpr(1, 10)
        state.set_vgpr(2, 10)
        ast = InstructionAST(
            "v_mad_u32_u24",
            Operand(RegKind.VGPR, 3),
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
            Operand(RegKind.VGPR, 2),
        )
        execute_scalar_mad_u32_u24(state, ast)
        self.assertEqual(initial_scc, 1)
        self.assertEqual(initial_vcc, 0xDEADBEEFCAFEBABE)

class MadU32U24ExtremeBoundariesTests(unittest.TestCase):
    def test_max_24bit_values(self):
        # 0xFFFFFF * 0xFFFFFF = 0xFFFFFE000001 -> low 32 bits is 0xFE000001
        res = oracle_mad_u32_u24(0xFFFFFF, 0xFFFFFF, 0)
        self.assertEqual(res, 0xFE000001)

    def test_max_24bit_with_max_accumulator(self):
        # 0xFE000001 + 0xFFFFFFFF = 0xFE000000 (modulo 2^32)
        res = oracle_mad_u32_u24(0xFFFFFF, 0xFFFFFF, 0xFFFFFFFF)
        self.assertEqual(res, 0xFE000000)

    def test_zero_product_with_extremes(self):
        self.assertEqual(oracle_mad_u32_u24(0, 0xFFFFFF, 0x7FFFFFFF), 0x7FFFFFFF)
        self.assertEqual(oracle_mad_u32_u24(0xFFFFFF, 0, 0x80000000), 0x80000000)

class MadU32U24TranslatorIntegrationTests(unittest.TestCase):
    def _run_ps(self, lines: list[str]) -> int:
        lines = list(lines) + [
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mad_u32_u24", lines)
        words = mrt0_words(spv)
        return words[0]

    def test_translator_basic_execution(self):
        lines = [
            "v_mov_b32 v0, 100",
            "v_mov_b32 v1, 200",
            "v_mov_b32 v2, 50",
            "v_mad_u32_u24 v3, v0, v1, v2",
        ]
        res = self._run_ps(lines)
        self.assertEqual(res, (100 * 200) + 50)

    def test_translator_24bit_masking(self):
        lines = [
            "v_mov_b32 v0, 0xFF000002",
            "v_mov_b32 v1, 0xAA000003",
            "v_mov_b32 v2, 10",
            "v_mad_u32_u24 v3, v0, v1, v2",
        ]
        res = self._run_ps(lines)
        self.assertEqual(res, (2 * 3) + 10)

    def test_translator_with_sgpr_and_inline(self):
        lines = [
            "s_mov_b32 s0, 25",
            "v_mov_b32 v1, 4",
            "v_mov_b32 v2, 12",
            "v_mad_u32_u24 v3, s0, v1, v2",
        ]
        res = self._run_ps(lines)
        self.assertEqual(res, (25 * 4) + 12)

    def test_translator_wrap_around(self):
        lines = [
            "v_mov_b32 v0, 0xFFFFFF",
            "v_mov_b32 v1, 1",
            "v_mov_b32 v2, 0xFFFFFFFF",
            "v_mad_u32_u24 v3, v0, v1, v2",
        ]
        res = self._run_ps(lines)
        self.assertEqual(res, (0xFFFFFF + 0xFFFFFFFF) & MASK32)

    def test_translator_in_place_aliasing(self):
        lines = [
            "v_mov_b32 v3, 3",
            "v_mad_u32_u24 v3, v3, v3, v3",
        ]
        res = self._run_ps(lines)
        self.assertEqual(res, (3 * 3) + 3)


if __name__ == "__main__":
    unittest.main()
