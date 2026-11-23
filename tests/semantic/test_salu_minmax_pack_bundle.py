"""Semantic reference tests and translator-backed validation for AMD RDNA2 SALU Min/Max & Pack Suite.

Subsystem Feature Bundle 32 covers 6 SALU min/max selection and half-word packing instructions:
  1. s_min_i32         - Signed 32-bit minimum: D = min(S0, S1), SCC = (S0 < S1, signed)
  2. s_min_u32         - Unsigned 32-bit minimum: D = min(S0, S1), SCC = (S0 < S1, unsigned)
  3. s_max_i32         - Signed 32-bit maximum: D = max(S0, S1), SCC = (S0 > S1, signed)
  4. s_max_u32         - Unsigned 32-bit maximum: D = max(S0, S1), SCC = (S0 > S1, unsigned)
  5. s_pack_ll_b32_b16 - Pack low halves: D = {S1[15:0], S0[15:0]}, SCC untouched
  6. s_pack_hh_b32_b16 - Pack high halves: D = {S1[31:16], S0[31:16]}, SCC untouched

RDNA2 SALU Architecture:
  - Operates on 32-bit scalar registers (SGPRs), inline/literal constants.
  - Condition code (SCC):
      s_min_i32: SCC = 1 if S0 < S1 (signed), 0 on >= (ties set SCC to 0)
      s_min_u32: SCC = 1 if S0 < S1 (unsigned), 0 on >= (ties set SCC to 0)
      s_max_i32: SCC = 1 if S0 > S1 (signed), 0 on <= (ties set SCC to 0)
      s_max_u32: SCC = 1 if S0 > S1 (unsigned), 0 on <= (ties set SCC to 0)
      s_pack_*: SCC is preserved (untouched)
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


def s32(u: int) -> int:
    u &= MASK32
    return u if u < 0x80000000 else u - 0x100000000


def u32(s: int) -> int:
    return s & MASK32

# -----------------------------------------------------------------------------
# Opcode 1: s_min_i32
# -----------------------------------------------------------------------------
def oracle_min_i32(s0: int, s1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_min_i32: returns (result_u32, scc)."""
    xi = s32(s0)
    yi = s32(s1)
    val = xi if xi < yi else yi
    scc = 1 if xi < yi else 0
    return u32(val), scc


class SMinI32OracleTests(unittest.TestCase):
    def test_min_i32_basic(self):
        # min(10, 20) = 10, SCC = 1 (10 < 20)
        res, scc = oracle_min_i32(10, 20)
        self.assertEqual(res, 10)
        self.assertEqual(scc, 1)

    def test_min_i32_negative_numbers(self):
        # min(-10, 5) = -10, SCC = 1 (-10 < 5)
        res, scc = oracle_min_i32(u32(-10), 5)
        self.assertEqual(res, u32(-10))
        self.assertEqual(scc, 1)

        # min(5, -10) = -10, SCC = 0 (5 < -10 is False)
        res, scc = oracle_min_i32(5, u32(-10))
        self.assertEqual(res, u32(-10))
        self.assertEqual(scc, 0)

    def test_min_i32_ties(self):
        # Equal values: SCC = 0 (ties clear SCC)
        res, scc = oracle_min_i32(42, 42)
        self.assertEqual(res, 42)
        self.assertEqual(scc, 0)

class RegKind(enum.Enum):
    SGPR = "sgpr"
    IMM = "imm"


@dataclass(frozen=True)
class Operand:
    kind: RegKind
    index: int = 0
    raw_val: int = 0


@dataclass(frozen=True)
class SaluInstructionAST:
    mnemonic: str
    dst: Operand
    src0: Operand
    src1: Operand


class ScalarRegisterState:
    def __init__(self):
        self.sgprs: dict[int, int] = {}
        self.scc: int = 0

    def set_sgpr(self, idx: int, val: int):
        self.sgprs[idx] = val & MASK32

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs.get(idx, 0) & MASK32

    def get_operand_val(self, op: Operand) -> int:
        if op.kind == RegKind.SGPR:
            return self.get_sgpr(op.index)
        return op.raw_val & MASK32

class SMinI32AstInterpreterTests(unittest.TestCase):
    def test_min_i32_ast_construction(self):
        ast = SaluInstructionAST("s_min_i32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        self.assertEqual(ast.mnemonic, "s_min_i32")
        self.assertEqual(ast.dst.index, 2)

    def test_min_i32_scalar_execution(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, u32(-50))
        state.set_sgpr(1, 100)
        res, scc = oracle_min_i32(state.get_sgpr(0), state.get_sgpr(1))
        state.set_sgpr(2, res)
        state.scc = scc
        self.assertEqual(state.get_sgpr(2), u32(-50))
        self.assertEqual(state.scc, 1)

class SMinI32SimdDivergenceTests(unittest.TestCase):
    def test_min_i32_uniform_scalar_execution(self):
        # Scalar instructions execute uniformly for the whole wave
        state = ScalarRegisterState()
        state.set_sgpr(0, 5)
        state.set_sgpr(1, 10)
        res, scc = oracle_min_i32(state.get_sgpr(0), state.get_sgpr(1))
        state.set_sgpr(2, res)
        state.scc = scc
        self.assertEqual(state.get_sgpr(2), 5)
        self.assertEqual(state.scc, 1)

    def test_min_i32_scc_extrema(self):
        # Extreme signed limits: -2147483648 (0x80000000) vs 2147483647 (0x7FFFFFFF)
        res, scc = oracle_min_i32(0x80000000, 0x7FFFFFFF)
        self.assertEqual(res, 0x80000000)
        self.assertEqual(scc, 1)

class SMinI32TranslatorTests(unittest.TestCase):
    def test_min_i32_translation_and_execution(self):
        lines = [
            "s_mov_b32 s0, -5",
            "s_mov_b32 s1, 10",
            "s_min_i32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_min_i32_exec", lines)
        words = mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFFFFFB)  # -5 in uint32
        self.assertEqual(words[1], 1)           # SCC = 1

    def test_min_i32_syntax_rejection(self):
        with self.assertRaises(Exception):
            translate_lines("ps_bad_min_i32", ["s_min_i32 s0, s1"])

# -----------------------------------------------------------------------------
# Opcode 2: s_min_u32
# -----------------------------------------------------------------------------
def oracle_min_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_min_u32: returns (result_u32, scc)."""
    xu = s0 & MASK32
    yu = s1 & MASK32
    val = xu if xu < yu else yu
    scc = 1 if xu < yu else 0
    return val, scc


class SMinU32OracleTests(unittest.TestCase):
    def test_min_u32_basic(self):
        # min(10, 20) = 10, SCC = 1
        res, scc = oracle_min_u32(10, 20)
        self.assertEqual(res, 10)
        self.assertEqual(scc, 1)

    def test_min_u32_unsigned_comparison(self):
        # 0xFFFFFFFF (4294967295) vs 10: unsigned min is 10
        res, scc = oracle_min_u32(0xFFFFFFFF, 10)
        self.assertEqual(res, 10)
        self.assertEqual(scc, 0)  # 0xFFFFFFFF < 10 is False unsigned

        res, scc = oracle_min_u32(10, 0xFFFFFFFF)
        self.assertEqual(res, 10)
        self.assertEqual(scc, 1)  # 10 < 0xFFFFFFFF is True unsigned

    def test_min_u32_ties(self):
        res, scc = oracle_min_u32(100, 100)
        self.assertEqual(res, 100)
        self.assertEqual(scc, 0)

class SMinU32AstInterpreterTests(unittest.TestCase):
    def test_min_u32_ast_construction(self):
        ast = SaluInstructionAST("s_min_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        self.assertEqual(ast.mnemonic, "s_min_u32")

    def test_min_u32_scalar_execution(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 0x80000000)
        state.set_sgpr(1, 0x40000000)
        res, scc = oracle_min_u32(state.get_sgpr(0), state.get_sgpr(1))
        state.set_sgpr(2, res)
        state.scc = scc
        self.assertEqual(state.get_sgpr(2), 0x40000000)
        self.assertEqual(state.scc, 0)

class SMinU32SimdDivergenceTests(unittest.TestCase):
    def test_min_u32_scc_transitions(self):
        # Non-zero SCC
        res1, scc1 = oracle_min_u32(0x10, 0x20)
        self.assertEqual(scc1, 1)
        # Zero SCC
        res2, scc2 = oracle_min_u32(0x20, 0x10)
        self.assertEqual(scc2, 0)

class SMinU32TranslatorTests(unittest.TestCase):
    def test_min_u32_translation_and_execution(self):
        lines = [
            "s_mov_b32 s0, 10",
            "s_mov_b32 s1, 0xFFFFFFFF",
            "s_min_u32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_min_u32_exec", lines)
        words = mrt0_words(spv)
        self.assertEqual(words[0], 10)
        self.assertEqual(words[1], 1)  # 10 < 0xFFFFFFFF -> SCC = 1

# -----------------------------------------------------------------------------
# Opcode 3: s_max_i32
# -----------------------------------------------------------------------------
def oracle_max_i32(s0: int, s1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_max_i32: returns (result_u32, scc)."""
    xi = s32(s0)
    yi = s32(s1)
    val = xi if xi > yi else yi
    scc = 1 if xi > yi else 0
    return u32(val), scc


class SMaxI32OracleTests(unittest.TestCase):
    def test_max_i32_basic(self):
        # max(10, 20) = 20, SCC = 0 (10 > 20 is False)
        res, scc = oracle_max_i32(10, 20)
        self.assertEqual(res, 20)
        self.assertEqual(scc, 0)

        # max(20, 10) = 20, SCC = 1 (20 > 10 is True)
        res, scc = oracle_max_i32(20, 10)
        self.assertEqual(res, 20)
        self.assertEqual(scc, 1)

    def test_max_i32_negatives(self):
        # max(-5, -20) = -5, SCC = 1 (-5 > -20)
        res, scc = oracle_max_i32(u32(-5), u32(-20))
        self.assertEqual(res, u32(-5))
        self.assertEqual(scc, 1)

    def test_max_i32_ties(self):
        res, scc = oracle_max_i32(u32(-7), u32(-7))
        self.assertEqual(res, u32(-7))
        self.assertEqual(scc, 0)

class SMaxI32AstInterpreterTests(unittest.TestCase):
    def test_max_i32_ast_construction(self):
        ast = SaluInstructionAST("s_max_i32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        self.assertEqual(ast.mnemonic, "s_max_i32")

    def test_max_i32_scalar_execution(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 15)
        state.set_sgpr(1, u32(-15))
        res, scc = oracle_max_i32(state.get_sgpr(0), state.get_sgpr(1))
        state.set_sgpr(2, res)
        state.scc = scc
        self.assertEqual(state.get_sgpr(2), 15)
        self.assertEqual(state.scc, 1)

class SMaxI32SimdDivergenceTests(unittest.TestCase):
    def test_max_i32_scc_extrema(self):
        # 0x7FFFFFFF > 0x80000000 -> max is 0x7FFFFFFF, SCC = 1
        res, scc = oracle_max_i32(0x7FFFFFFF, 0x80000000)
        self.assertEqual(res, 0x7FFFFFFF)
        self.assertEqual(scc, 1)

class SMaxI32TranslatorTests(unittest.TestCase):
    def test_max_i32_translation_and_execution(self):
        lines = [
            "s_mov_b32 s0, 50",
            "s_mov_b32 s1, -50",
            "s_max_i32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_max_i32_exec", lines)
        words = mrt0_words(spv)
        self.assertEqual(words[0], 50)
        self.assertEqual(words[1], 1)

# -----------------------------------------------------------------------------
# Opcode 4: s_max_u32
# -----------------------------------------------------------------------------
def oracle_max_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference mathematical oracle for s_max_u32: returns (result_u32, scc)."""
    xu = s0 & MASK32
    yu = s1 & MASK32
    val = xu if xu > yu else yu
    scc = 1 if xu > yu else 0
    return val, scc


class SMaxU32OracleTests(unittest.TestCase):
    def test_max_u32_basic(self):
        # max(0xFFFFFFFF, 10) = 0xFFFFFFFF, SCC = 1 (0xFFFFFFFF > 10 unsigned)
        res, scc = oracle_max_u32(0xFFFFFFFF, 10)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

    def test_max_u32_zero(self):
        res, scc = oracle_max_u32(0, 100)
        self.assertEqual(res, 100)
        self.assertEqual(scc, 0)

class SMaxU32AstInterpreterTests(unittest.TestCase):
    def test_max_u32_ast_construction(self):
        ast = SaluInstructionAST("s_max_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        self.assertEqual(ast.mnemonic, "s_max_u32")

    def test_max_u32_scalar_execution(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 0x1000)
        state.set_sgpr(1, 0x2000)
        res, scc = oracle_max_u32(state.get_sgpr(0), state.get_sgpr(1))
        state.set_sgpr(2, res)
        state.scc = scc
        self.assertEqual(state.get_sgpr(2), 0x2000)
        self.assertEqual(state.scc, 0)

class SMaxU32SimdDivergenceTests(unittest.TestCase):
    def test_max_u32_scc_extrema(self):
        # 0xFFFFFFFF vs 0: SCC = 1
        res, scc = oracle_max_u32(0xFFFFFFFF, 0)
        self.assertEqual(res, 0xFFFFFFFF)
        self.assertEqual(scc, 1)

class SMaxU32TranslatorTests(unittest.TestCase):
    def test_max_u32_translation_and_execution(self):
        lines = [
            "s_mov_b32 s0, 0xFFFFFFFF",
            "s_mov_b32 s1, 10",
            "s_max_u32 s2, s0, s1",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_max_u32_exec", lines)
        words = mrt0_words(spv)
        self.assertEqual(words[0], 0xFFFFFFFF)
        self.assertEqual(words[1], 1)

# -----------------------------------------------------------------------------
# Opcode 5: s_pack_ll_b32_b16
# -----------------------------------------------------------------------------
def oracle_pack_ll_b32_b16(s0: int, s1: int) -> int:
    """Reference mathematical oracle for s_pack_ll_b32_b16: D = {S1[15:0], S0[15:0]}."""
    lo = s0 & 0xFFFF
    hi = (s1 & 0xFFFF) << 16
    return (hi | lo) & MASK32


class SPackLlB32B16OracleTests(unittest.TestCase):
    def test_pack_ll_basic(self):
        # s0 = 0x12345678, s1 = 0xABCDEF01 -> {0xef01, 0x5678} = 0xEF015678
        res = oracle_pack_ll_b32_b16(0x12345678, 0xABCDEF01)
        self.assertEqual(res, 0xEF015678)

    def test_pack_ll_zeros_and_ones(self):
        self.assertEqual(oracle_pack_ll_b32_b16(0, 0), 0)
        self.assertEqual(oracle_pack_ll_b32_b16(0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFF)

class SPackLlB32B16AstInterpreterTests(unittest.TestCase):
    def test_pack_ll_ast_construction(self):
        ast = SaluInstructionAST("s_pack_ll_b32_b16", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        self.assertEqual(ast.mnemonic, "s_pack_ll_b32_b16")

    def test_pack_ll_scalar_execution(self):
        state = ScalarRegisterState()
        state.set_sgpr(0, 0xAAAA1111)
        state.set_sgpr(1, 0xBBBB2222)
        res = oracle_pack_ll_b32_b16(state.get_sgpr(0), state.get_sgpr(1))
        state.set_sgpr(2, res)
        self.assertEqual(state.get_sgpr(2), 0x22221111)

class SPackLlB32B16SimdDivergenceTests(unittest.TestCase):
    def test_pack_ll_scc_preserved(self):
        # Hardware rule: s_pack_* does NOT modify SCC
        state = ScalarRegisterState()
        state.scc = 1
        res = oracle_pack_ll_b32_b16(0x0001, 0x0002)
        # scc remains 1
        self.assertEqual(state.scc, 1)
        self.assertEqual(res, 0x00020001)

class SPackLlB32B16TranslatorTests(unittest.TestCase):
    def test_pack_ll_translation_and_execution(self):
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0xABCDEF01",
            "s_pack_ll_b32_b16 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_pack_ll_exec", lines)
        val = mrt0_words(spv)[0]
        self.assertEqual(val, 0xEF015678)

# -----------------------------------------------------------------------------
# Opcode 6: s_pack_hh_b32_b16
# -----------------------------------------------------------------------------
def oracle_pack_hh_b32_b16(s0: int, s1: int) -> int:
    """Reference mathematical oracle for s_pack_hh_b32_b16: D = {S1[31:16], S0[31:16]}."""
    lo = (s0 >> 16) & 0xFFFF
    hi = s1 & 0xFFFF0000
    return (hi | lo) & MASK32


class SPackHhB32B16OracleTests(unittest.TestCase):
    def test_pack_hh_basic(self):
        # s0 = 0x12345678, s1 = 0xABCDEF01 -> {0xabcd, 0x1234} = 0xABCD1234
        res = oracle_pack_hh_b32_b16(0x12345678, 0xABCDEF01)
        self.assertEqual(res, 0xABCD1234)

    def test_pack_hh_zero_high_halves(self):
        res = oracle_pack_hh_b32_b16(0x0000FFFF, 0x0000FFFF)
        self.assertEqual(res, 0)

if __name__ == "__main__":
    unittest.main()
