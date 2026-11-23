"""Semantic reference oracle and architectural model for RDNA2 s_or_b32.

AMD RDNA2 / GFX10 Scalar ALU (SALU) Architecture:
  - 32-bit Scalar Bitwise OR:
    - RDNA2 ISA: D.u32 = S0.u32 | S1.u32
    - Destination: SGPR (s0..s105) or special registers (exec_lo, exec_hi, vcc_lo, vcc_hi, m0, null)
    - Sources: SGPRs, inline constants (0..64, -1..-16), 32-bit literals, special registers
    - Condition code SCC:
        SCC = 1 if (D != 0) else 0
  - Mathematical and algebraic properties:
    - Identity element:       A | 0 == A
    - Annihilator / Dominance: A | 0xFFFFFFFF == 0xFFFFFFFF
    - Idempotence:            A | A == A
    - Commutativity:          A | B == B | A
    - Associativity:          (A | B) | C == A | (B | C)
    - De Morgan duality:      ~(A | B) == (~A) & (~B)
"""
from __future__ import annotations

import ast
import dataclasses
import enum
import math
import os
import random
import struct
import sys
import unittest
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from support import translate_lines, mrt0_words  # noqa: E402

MASK32: int = 0xFFFFFFFF


def ref_s_or_b32(s0: int, s1: int) -> int:
    """Pure reference oracle for RDNA2 32-bit scalar bitwise OR: (s0 | s1) & 0xFFFFFFFF."""
    return ((s0 & MASK32) | (s1 & MASK32)) & MASK32


def ref_scc(val: int) -> int:
    """RDNA2 Scalar Condition Code for bitwise SALU operations: 1 if val != 0 else 0."""
    return 1 if (val & MASK32) != 0 else 0


class SaluOrB32OracleTests(unittest.TestCase):
    """Reference mathematical oracle and architectural model tests for s_or_b32."""

    def test_truth_table_exhaustive(self):
        """Verify full 2-input 1-bit truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1."""
        truth_table = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 0): 1,
            (1, 1): 1,
        }
        for (b0, b1), expected in truth_table.items():
            with self.subTest(b0=b0, b1=b1):
                s0 = MASK32 if b0 else 0
                s1 = MASK32 if b1 else 0
                want = MASK32 if expected else 0
                got = ref_s_or_b32(s0, s1)
                self.assertEqual(got, want)
                self.assertEqual(ref_scc(got), expected)

    def test_algebraic_identities(self):
        """Test algebraic laws: identity, annihilator, idempotence, commutativity, associativity."""
        test_values = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0x87654321,
            0xAAAAAAAA,
            0x55555555,
            0x80000000,
            0x00000001,
        ]
        for a in test_values:
            with self.subTest(identity=hex(a)):
                self.assertEqual(ref_s_or_b32(a, 0), a)
                self.assertEqual(ref_s_or_b32(0, a), a)
                self.assertEqual(ref_s_or_b32(a, MASK32), MASK32)
                self.assertEqual(ref_s_or_b32(MASK32, a), MASK32)
                self.assertEqual(ref_s_or_b32(a, a), a)

        for a in test_values:
            for b in test_values:
                with self.subTest(commutative=(hex(a), hex(b))):
                    self.assertEqual(ref_s_or_b32(a, b), ref_s_or_b32(b, a))

        for a in test_values[:4]:
            for b in test_values[:4]:
                for c in test_values[:4]:
                    with self.subTest(associative=(hex(a), hex(b), hex(c))):
                        left = ref_s_or_b32(ref_s_or_b32(a, b), c)
                        right = ref_s_or_b32(a, ref_s_or_b32(b, c))
                        self.assertEqual(left, right)

    def test_de_morgan_duality(self):
        """Test De Morgan duality: ~(A | B) == (~A) & (~B)."""
        rng = random.Random(0x028332)
        for _ in range(64):
            a = rng.getrandbits(32)
            b = rng.getrandbits(32)
            or_ab = ref_s_or_b32(a, b)
            not_or_ab = (~or_ab) & MASK32
            not_a = (~a) & MASK32
            not_b = (~b) & MASK32
            and_not_a_not_b = (not_a & not_b) & MASK32
            self.assertEqual(not_or_ab, and_not_a_not_b)

    def test_boundary_bit_patterns(self):
        """Test single-bit walking, byte boundaries, and alternating bit patterns."""
        for i in range(32):
            bit = 1 << i
            self.assertEqual(ref_s_or_b32(bit, 0), bit)
            self.assertEqual(ref_s_or_b32(bit, bit), bit)
            self.assertEqual(ref_scc(ref_s_or_b32(bit, 0)), 1)

        for i in range(31):
            b0 = 1 << i
            b1 = 1 << (i + 1)
            combined = ref_s_or_b32(b0, b1)
            self.assertEqual(combined, b0 | b1)
            self.assertEqual(ref_scc(combined), 1)

    def test_scc_oracle_mapping(self):
        """Verify SCC mapping across exact zero and non-zero boundary transitions."""
        self.assertEqual(ref_scc(0x00000000), 0)
        self.assertEqual(ref_scc(0x00000001), 1)
        self.assertEqual(ref_scc(0x80000000), 1)
        self.assertEqual(ref_scc(0x7FFFFFFF), 1)
        self.assertEqual(ref_scc(0xFFFFFFFF), 1)
        self.assertEqual(ref_scc(ref_s_or_b32(0, 0)), 0)

    def test_translator_smoke(self):
        """Smoke test verifying tools/rdna2spirv.py translates s_or_b32 correctly."""
        lines = [
            "s_mov_b32 s0, 0x12",
            "s_mov_b32 s1, 0x20",
            "s_or_b32 s2, s0, s1",
            "v_mov_b32 v0, s2",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_salu_or_smoke", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x32)
        self.assertEqual(out[1], 1)


class OperandType(enum.Enum):
    SGPR = "sgpr"
    IMM_INT = "imm_int"
    IMM_HEX = "imm_hex"
    SPECIAL = "special"


@dataclasses.dataclass(frozen=True)
class Operand:
    name: str
    op_type: OperandType
    value: Optional[int] = None

    @classmethod
    def sgpr(cls, idx: int) -> Operand:
        if not (0 <= idx <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: s{idx}")
        return cls(name=f"s{idx}", op_type=OperandType.SGPR, value=idx)

    @classmethod
    def imm_int(cls, val: int) -> Operand:
        return cls(name=str(val), op_type=OperandType.IMM_INT, value=val)

    @classmethod
    def imm_hex(cls, val: int) -> Operand:
        return cls(name=f"{val:#x}", op_type=OperandType.IMM_HEX, value=val & MASK32)

    @classmethod
    def special(cls, name: str) -> Operand:
        valid_specials = {"exec_lo", "exec_hi", "vcc_lo", "vcc_hi", "m0", "scc", "null"}
        if name not in valid_specials:
            raise ValueError(f"Invalid special register: {name}")
        return cls(name=name, op_type=OperandType.SPECIAL)


@dataclasses.dataclass
class AstInstruction:
    opcode: str
    dst: Operand
    src0: Operand
    src1: Operand

    def emit_asm(self) -> str:
        return f"{self.opcode} {self.dst.name}, {self.src0.name}, {self.src1.name}"

    def type_check(self) -> bool:
        if self.opcode != "s_or_b32":
            return False
        if self.dst.op_type not in (OperandType.SGPR, OperandType.SPECIAL):
            return False
        return True


class Expr:
    def evaluate(self, env: Dict[str, int]) -> int:
        raise NotImplementedError


@dataclasses.dataclass
class ConstExpr(Expr):
    val: int

    def evaluate(self, env: Dict[str, int]) -> int:
        return self.val & MASK32


@dataclasses.dataclass
class RegExpr(Expr):
    name: str

    def evaluate(self, env: Dict[str, int]) -> int:
        return env.get(self.name, 0) & MASK32


@dataclasses.dataclass
class BitwiseOrExpr(Expr):
    left: Expr
    right: Expr

    def evaluate(self, env: Dict[str, int]) -> int:
        l = self.left.evaluate(env)
        r = self.right.evaluate(env)
        return ref_s_or_b32(l, r)


class SaluOrB32AstTests(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions for s_or_b32."""

    def test_ast_construction_and_formatting(self):
        """Construct AST nodes and format into assembly lines."""
        inst = AstInstruction(
            opcode="s_or_b32",
            dst=Operand.sgpr(2),
            src0=Operand.sgpr(0),
            src1=Operand.sgpr(1),
        )
        self.assertTrue(inst.type_check())
        self.assertEqual(inst.emit_asm(), "s_or_b32 s2, s0, s1")

        inst_imm = AstInstruction(
            opcode="s_or_b32",
            dst=Operand.sgpr(4),
            src0=Operand.sgpr(4),
            src1=Operand.imm_hex(0x80000000),
        )
        self.assertTrue(inst_imm.type_check())
        self.assertEqual(inst_imm.emit_asm(), "s_or_b32 s4, s4, 0x80000000")

    def test_ast_type_validation(self):
        """Validate destination constraints and register index bounds."""
        with self.assertRaises(ValueError):
            Operand.sgpr(106)

        with self.assertRaises(ValueError):
            Operand.special("invalid_reg")

        bad_dst_inst = AstInstruction(
            opcode="s_or_b32",
            dst=Operand.imm_int(42),
            src0=Operand.sgpr(0),
            src1=Operand.sgpr(1),
        )
        self.assertFalse(bad_dst_inst.type_check())

    def test_ast_expression_evaluation(self):
        """Evaluate symbolic AST expression trees against reference oracle."""
        env = {"s0": 0x00FF0000, "s1": 0x0000FF00}
        expr = BitwiseOrExpr(RegExpr("s0"), RegExpr("s1"))
        self.assertEqual(expr.evaluate(env), 0x00FFFF00)

        nested = BitwiseOrExpr(expr, ConstExpr(0x000000FF))
        self.assertEqual(nested.evaluate(env), 0x00FFFFFF)

    def test_ast_assembly_translation(self):
        """Generate assembly from AST and translate with support.translate_lines."""
        inst = AstInstruction(
            opcode="s_or_b32",
            dst=Operand.sgpr(2),
            src0=Operand.sgpr(0),
            src1=Operand.imm_hex(0xFF),
        )
        lines = [
            "s_mov_b32 s0, 0x1000",
            inst.emit_asm(),
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_ast_or_emit", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x10FF)

if __name__ == "__main__":
    unittest.main()
