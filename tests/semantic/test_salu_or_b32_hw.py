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


def run_salu_or_b32(s0: int, s1: int) -> tuple[int, int]:
    """Execute s_or_b32 s2, s0, s1 through translate_lines and mrt0_words, returning (s2, scc)."""
    lines = [
        f"s_mov_b32 s0, {s0 & MASK32:#x}",
        f"s_mov_b32 s1, {s1 & MASK32:#x}",
        "s_or_b32 s2, s0, s1",
        "v_mov_b32 v0, s2",
        "s_cselect_b32 s3, 1, 0",
        "v_mov_b32 v1, s3",
        "v_mov_b32 v2, 0",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_or_{s0 & MASK32:08x}_{s1 & MASK32:08x}", lines)
    out = mrt0_words(spv)
    return out[0], out[1]


class SaluOrB32OracleTests(unittest.TestCase):
    """Reference mathematical oracle and architectural model tests for s_or_b32."""

    def test_truth_table_exhaustive(self):
        """Verify full 2-input 1-bit truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->1 via translator."""
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
                got, scc = run_salu_or_b32(s0, s1)
                self.assertEqual(got, want)
                self.assertEqual(scc, expected)

    def test_algebraic_identities(self):
        """Test algebraic laws: identity, annihilator, idempotence, commutativity via translator."""
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
                res_id, scc_id = run_salu_or_b32(a, 0)
                self.assertEqual(res_id, a)
                self.assertEqual(scc_id, 1 if a != 0 else 0)

                res_ann, scc_ann = run_salu_or_b32(a, MASK32)
                self.assertEqual(res_ann, MASK32)
                self.assertEqual(scc_ann, 1)

                res_idem, scc_idem = run_salu_or_b32(a, a)
                self.assertEqual(res_idem, a)
                self.assertEqual(scc_idem, 1 if a != 0 else 0)

        for a in test_values[:4]:
            for b in test_values[:4]:
                with self.subTest(commutative=(hex(a), hex(b))):
                    res_ab, _ = run_salu_or_b32(a, b)
                    res_ba, _ = run_salu_or_b32(b, a)
                    self.assertEqual(res_ab, res_ba)

    def test_de_morgan_duality(self):
        """Test De Morgan duality: ~(A | B) == (~A) & (~B) via translated shader."""
        test_pairs = [
            (0x12345678, 0x87654321),
            (0xAAAAAAAA, 0x55555555),
            (0xFF00FF00, 0x00FF00FF),
            (0x00000000, 0xFFFFFFFF),
        ]
        for idx, (a, b) in enumerate(test_pairs):
            with self.subTest(a=hex(a), b=hex(b)):
                lines = [
                    f"s_mov_b32 s0, {a:#x}",
                    f"s_mov_b32 s1, {b:#x}",
                    "s_or_b32 s2, s0, s1",
                    "s_not_b32 s3, s2",
                    "s_not_b32 s4, s0",
                    "s_not_b32 s5, s1",
                    "s_and_b32 s6, s4, s5",
                    "v_mov_b32 v0, s3",
                    "v_mov_b32 v1, s6",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_demorgan_{idx}", lines)
                out = mrt0_words(spv)
                self.assertEqual(out[0], out[1])
                self.assertEqual(out[0], (~(a | b)) & MASK32)

    def test_boundary_bit_patterns(self):
        """Test single-bit walking, byte boundaries, and alternating bit patterns via translator."""
        for i in [0, 1, 7, 8, 15, 16, 23, 24, 30, 31]:
            bit = 1 << i
            res, scc = run_salu_or_b32(bit, 0)
            self.assertEqual(res, bit)
            self.assertEqual(scc, 1)

            res_self, scc_self = run_salu_or_b32(bit, bit)
            self.assertEqual(res_self, bit)
            self.assertEqual(scc_self, 1)

        for i in [0, 7, 15, 23, 30]:
            b0 = 1 << i
            b1 = 1 << (i + 1)
            combined, scc_comb = run_salu_or_b32(b0, b1)
            self.assertEqual(combined, b0 | b1)
            self.assertEqual(scc_comb, 1)

    def test_scc_oracle_mapping(self):
        """Verify SCC mapping across exact zero and non-zero boundary transitions via translator."""
        cases = [
            (0, 0, 0),
            (0, 1, 1),
            (0x80000000, 0, 1),
            (0x7FFFFFFF, 0, 1),
            (0xFFFFFFFF, 0, 1),
        ]
        for a, b, expected_scc in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                res, scc = run_salu_or_b32(a, b)
                self.assertEqual(res, (a | b) & MASK32)
                self.assertEqual(scc, expected_scc)

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


class ScalarRegisterState:
    """Architectural state machine modeling RDNA2 scalar execution."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs: List[int] = [0] * num_sgprs
        self.exec_lo: int = MASK32
        self.exec_hi: int = MASK32
        self.vcc_lo: int = 0
        self.vcc_hi: int = 0
        self.m0: int = 0
        self.scc: int = 0

    def read_reg(self, name: str) -> int:
        if name.startswith("s") and name[1:].isdigit():
            idx = int(name[1:])
            if not (0 <= idx < len(self.sgprs)):
                raise ValueError(f"SGPR index out of range: {name}")
            return self.sgprs[idx] & MASK32
        if name == "exec_lo":
            return self.exec_lo & MASK32
        if name == "exec_hi":
            return self.exec_hi & MASK32
        if name == "vcc_lo":
            return self.vcc_lo & MASK32
        if name == "vcc_hi":
            return self.vcc_hi & MASK32
        if name == "m0":
            return self.m0 & MASK32
        if name == "scc":
            return self.scc & 1
        if name == "null":
            return 0
        raise ValueError(f"Unknown register: {name}")

    def write_reg(self, name: str, val: int) -> None:
        v = val & MASK32
        if name.startswith("s") and name[1:].isdigit():
            idx = int(name[1:])
            if not (0 <= idx < len(self.sgprs)):
                raise ValueError(f"SGPR index out of range: {name}")
            self.sgprs[idx] = v
        elif name == "exec_lo":
            self.exec_lo = v
        elif name == "exec_hi":
            self.exec_hi = v
        elif name == "vcc_lo":
            self.vcc_lo = v
        elif name == "vcc_hi":
            self.vcc_hi = v
        elif name == "m0":
            self.m0 = v
        elif name == "scc":
            self.scc = 1 if v != 0 else 0
        elif name == "null":
            pass
        else:
            raise ValueError(f"Unknown register: {name}")

    def resolve_src(self, src: str | int) -> int:
        if isinstance(src, int):
            return src & MASK32
        s = src.strip()
        if s.startswith("0x") or s.startswith("-0x"):
            return int(s, 16) & MASK32
        try:
            return int(s) & MASK32
        except ValueError:
            return self.read_reg(s)

    def execute_s_or_b32(self, dst: str, src0: str | int, src1: str | int) -> int:
        v0 = self.resolve_src(src0)
        v1 = self.resolve_src(src1)
        res = ref_s_or_b32(v0, v1)
        self.write_reg(dst, res)
        self.scc = ref_scc(res)
        return res

    def run_program(self, lines: List[str]) -> None:
        for raw in lines:
            ln = raw.split(";")[0].strip()
            if not ln:
                continue
            parts = ln.replace(",", " ").split()
            op = parts[0]
            if op == "s_mov_b32":
                dst = parts[1]
                src = parts[2]
                self.write_reg(dst, self.resolve_src(src))
            elif op == "s_or_b32":
                dst = parts[1]
                s0 = parts[2]
                s1 = parts[3]
                self.execute_s_or_b32(dst, s0, s1)
            elif op == "s_cselect_b32":
                dst = parts[1]
                s0 = self.resolve_src(parts[2])
                s1 = self.resolve_src(parts[3])
                self.write_reg(dst, s0 if self.scc != 0 else s1)


class SaluOrB32InterpreterTests(unittest.TestCase):
    """Deterministic scalar instruction interpreter tests for s_or_b32."""

    def test_interpreter_basic_or(self):
        """Test standard scalar OR execution on SGPRs via interpreter and hardware translator."""
        state = ScalarRegisterState()
        state.write_reg("s0", 0x12340000)
        state.write_reg("s1", 0x00005678)
        res = state.execute_s_or_b32("s2", "s0", "s1")
        self.assertEqual(res, 0x12345678)
        self.assertEqual(state.read_reg("s2"), 0x12345678)
        self.assertEqual(state.scc, 1)

        hw_res, hw_scc = run_salu_or_b32(0x12340000, 0x00005678)
        self.assertEqual(hw_res, 0x12345678)
        self.assertEqual(hw_scc, 1)

    def test_interpreter_scc_updates(self):
        """Test that SCC is cleared strictly on zero results and set otherwise via interpreter and translator."""
        state = ScalarRegisterState()
        state.execute_s_or_b32("s0", 0, 0)
        self.assertEqual(state.read_reg("s0"), 0)
        self.assertEqual(state.scc, 0)

        hw_res0, hw_scc0 = run_salu_or_b32(0, 0)
        self.assertEqual(hw_res0, 0)
        self.assertEqual(hw_scc0, 0)

        state.execute_s_or_b32("s1", 0, 1)
        self.assertEqual(state.read_reg("s1"), 1)
        self.assertEqual(state.scc, 1)

        hw_res1, hw_scc1 = run_salu_or_b32(0, 1)
        self.assertEqual(hw_res1, 1)
        self.assertEqual(hw_scc1, 1)

    def test_interpreter_special_registers(self):
        """Test OR operations reading and writing m0, vcc_lo, exec_lo, and null via translator."""
        state = ScalarRegisterState()
        state.write_reg("m0", 0x000000F0)
        state.execute_s_or_b32("m0", "m0", 0x00000F00)
        self.assertEqual(state.read_reg("m0"), 0x00000FF0)

        state.execute_s_or_b32("null", 0x1234, 0x5678)
        self.assertEqual(state.read_reg("null"), 0)
        self.assertEqual(state.scc, 1)

        lines = [
            "s_mov_b32 m0, 0x000000F0",
            "s_or_b32 m0, m0, 0x00000F00",
            "s_or_b32 null, 0x1234, 0x5678",
            "s_cselect_b32 s3, 1, 0",
            "v_mov_b32 v0, m0",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_salu_special_regs", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x00000FF0)
        self.assertEqual(out[1], 1)

    def test_translator_interpreter_equivalence(self):
        """Validate exact equivalence between Python interpreter and translated SPIR-V."""
        test_pairs = [
            (0x00000000, 0x00000000),
            (0x12345678, 0x00000000),
            (0x00000000, 0x87654321),
            (0xAAAAAAAA, 0x55555555),
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0x80000000, 0x00000001),
        ]
        for idx, (v0, v1) in enumerate(test_pairs):
            with self.subTest(v0=hex(v0), v1=hex(v1)):
                sim = ScalarRegisterState()
                sim.write_reg("s0", v0)
                sim.write_reg("s1", v1)
                sim.execute_s_or_b32("s2", "s0", "s1")

                lines = [
                    f"s_mov_b32 s0, {v0:#x}",
                    f"s_mov_b32 s1, {v1:#x}",
                    "s_or_b32 s2, s0, s1",
                    "v_mov_b32 v0, s2",
                    "s_cselect_b32 s3, 1, 0",
                    "v_mov_b32 v1, s3",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_salu_interp_eq_{idx}", lines)
                out = mrt0_words(spv)
                self.assertEqual(out[0], sim.read_reg("s2"))
                self.assertEqual(out[1], sim.scc)


class SaluOrB32SimdDivergenceTests(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking for s_or_b32."""

    def test_wave32_divergence_reconvergence(self):
        """Simulate divergent branch merging in Wave32 using s_or_b32 via translator."""
        even_lanes = 0x55555555  # lanes 0, 2, 4, ... 30
        odd_lanes = 0xAAAAAAAA   # lanes 1, 3, 5, ... 31

        state = ScalarRegisterState()
        state.write_reg("s0", even_lanes)
        state.write_reg("s1", odd_lanes)
        state.execute_s_or_b32("exec_lo", "s0", "s1")
        self.assertEqual(state.read_reg("exec_lo"), MASK32)
        self.assertEqual(state.scc, 1)

        out, scc = run_salu_or_b32(even_lanes, odd_lanes)
        self.assertEqual(out, MASK32)
        self.assertEqual(scc, 1)

    def test_wave64_dual_half_masks(self):
        """Manage 64-bit wavefront execution masks across dual halves via translator."""
        state = ScalarRegisterState()
        state.write_reg("s0", 0x0000FFFF)
        state.write_reg("s1", 0xFFFF0000)
        state.write_reg("s2", 0x000000FF)
        state.write_reg("s3", 0xFFFFFF00)

        state.execute_s_or_b32("exec_lo", "s0", "s1")
        state.execute_s_or_b32("exec_hi", "s2", "s3")

        self.assertEqual(state.read_reg("exec_lo"), MASK32)
        self.assertEqual(state.read_reg("exec_hi"), MASK32)

        out_lo, _ = run_salu_or_b32(0x0000FFFF, 0xFFFF0000)
        out_hi, _ = run_salu_or_b32(0x000000FF, 0xFFFFFF00)
        self.assertEqual(out_lo, MASK32)
        self.assertEqual(out_hi, MASK32)

    def test_vcc_condition_accumulation(self):
        """Accumulate vector condition outcomes via s_or_b32 via translator."""
        state = ScalarRegisterState()
        state.write_reg("vcc_lo", 0x00000000)
        state.execute_s_or_b32("vcc_lo", "vcc_lo", 0x0000000F)
        state.execute_s_or_b32("vcc_lo", "vcc_lo", 0x000000F0)
        state.execute_s_or_b32("vcc_lo", "vcc_lo", 0x00000F00)
        self.assertEqual(state.read_reg("vcc_lo"), 0x00000FFF)

        lines = [
            "s_mov_b32 s2, 0x00000000",
            "s_or_b32 s2, s2, 0x0000000F",
            "s_or_b32 s2, s2, 0x000000F0",
            "s_or_b32 s2, s2, 0x00000F00",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_vcc_accum", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x00000FFF)

    def test_translated_exec_mask_manipulation(self):
        """Translate shader performing s_or_b32 on exec_lo and verify SPIR-V output."""
        lines = [
            "s_mov_b32 s0, 0x0000FFFF",
            "s_mov_b32 s1, 0xFFFF0000",
            "s_or_b32 exec_lo, s0, s1",
            "v_mov_b32 v0, exec_lo",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_exec_or_mask", lines)
        self.assertIn("OpBitwiseOr", spv)


class SaluOrB32AliasingTests(unittest.TestCase):
    """Register aliasing and in-place operand preservation for s_or_b32."""

    def test_self_aliasing_all_operands_identical(self):
        """s_or_b32 s0, s0, s0: in-place idempotent self-OR via translator."""
        val = 0xFEEDBEEF
        state = ScalarRegisterState()
        state.write_reg("s0", val)
        res = state.execute_s_or_b32("s0", "s0", "s0")
        self.assertEqual(res, val)
        self.assertEqual(state.read_reg("s0"), val)
        self.assertEqual(state.scc, 1)

        lines = [
            f"s_mov_b32 s0, {val:#x}",
            "s_or_b32 s0, s0, s0",
            "v_mov_b32 v0, s0",
            "s_cselect_b32 s1, 1, 0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_self_alias", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], val)
        self.assertEqual(out[1], 1)

    def test_in_place_dst_alias_src0(self):
        """s_or_b32 s0, s0, s1: accumulate into src0 while preserving src1 via translator."""
        state = ScalarRegisterState()
        state.write_reg("s0", 0x0F0F0F0F)
        state.write_reg("s1", 0xF0F0F0F0)
        state.execute_s_or_b32("s0", "s0", "s1")
        self.assertEqual(state.read_reg("s0"), MASK32)
        self.assertEqual(state.read_reg("s1"), 0xF0F0F0F0)

        lines = [
            "s_mov_b32 s0, 0x0F0F0F0F",
            "s_mov_b32 s1, 0xF0F0F0F0",
            "s_or_b32 s0, s0, s1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_alias_src0_test", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], MASK32)
        self.assertEqual(out[1], 0xF0F0F0F0)

    def test_in_place_dst_alias_src1(self):
        """s_or_b32 s1, s0, s1: accumulate into src1 while preserving src0 via translator."""
        state = ScalarRegisterState()
        state.write_reg("s0", 0x11111111)
        state.write_reg("s1", 0x22222222)
        state.execute_s_or_b32("s1", "s0", "s1")
        self.assertEqual(state.read_reg("s0"), 0x11111111)
        self.assertEqual(state.read_reg("s1"), 0x33333333)

        lines = [
            "s_mov_b32 s0, 0x11111111",
            "s_mov_b32 s1, 0x22222222",
            "s_or_b32 s1, s0, s1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_alias_src1_test", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x11111111)
        self.assertEqual(out[1], 0x33333333)

    def test_adjacent_register_isolation(self):
        """Modifying destination register must not touch adjacent SGPRs via translator."""
        state = ScalarRegisterState()
        state.write_reg("s0", 0x11111111)
        state.write_reg("s1", 0x22222222)
        state.write_reg("s2", 0x33333333)
        state.write_reg("s3", 0x44444444)

        state.execute_s_or_b32("s1", "s1", "s2")
        self.assertEqual(state.read_reg("s0"), 0x11111111)
        self.assertEqual(state.read_reg("s1"), 0x33333333)
        self.assertEqual(state.read_reg("s2"), 0x33333333)
        self.assertEqual(state.read_reg("s3"), 0x44444444)

        lines = [
            "s_mov_b32 s0, 0x11111111",
            "s_mov_b32 s1, 0x22222222",
            "s_mov_b32 s2, 0x33333333",
            "s_mov_b32 s3, 0x44444444",
            "s_or_b32 s1, s1, s2",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s2",
            "v_mov_b32 v3, s3",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_adjacent_isolation_test", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], 0x11111111)
        self.assertEqual(out[1], 0x33333333)
        self.assertEqual(out[2], 0x33333333)
        self.assertEqual(out[3], 0x44444444)

    def test_translated_aliasing_shader(self):
        """Translate shader with dst aliasing src0 and execute via mrt0_words."""
        lines = [
            "s_mov_b32 s0, 0x0F0F0F0F",
            "s_mov_b32 s1, 0xF0F0F0F0",
            "s_or_b32 s0, s0, s1",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_salu_or_alias", lines)
        out = mrt0_words(spv)
        self.assertEqual(out[0], MASK32)
        self.assertEqual(out[1], 0xF0F0F0F0)


class SaluOrB32ConditionCodeTests(unittest.TestCase):
    """Condition codes, status flags, and carry/borrow generation for s_or_b32."""

    def test_scc_zero_flag_exhaustion(self):
        """SCC must be 0 if and only if the bitwise result is exactly zero via translator."""
        res0, scc0 = run_salu_or_b32(0, 0)
        self.assertEqual(res0, 0)
        self.assertEqual(scc0, 0)

        for bit in [0, 1, 7, 15, 16, 24, 30, 31]:
            val = 1 << bit
            with self.subTest(bit=bit):
                res, scc = run_salu_or_b32(val, 0)
                self.assertEqual(res, val)
                self.assertEqual(scc, 1)

    def test_scc_preservation_across_mov(self):
        """s_mov_b32 does NOT alter the SCC flag set by s_or_b32 via translator."""
        lines1 = [
            "s_or_b32 s0, 0, 0",
            "s_mov_b32 s1, 0x12345678",
            "s_cselect_b32 s2, 100, 200",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv1 = translate_lines("ps_scc_pres_0", lines1)
        self.assertEqual(mrt0_words(spv1)[0], 200)

        lines2 = [
            "s_or_b32 s0, 1, 0",
            "s_mov_b32 s1, 0x00000000",
            "s_cselect_b32 s2, 100, 200",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv2 = translate_lines("ps_scc_pres_1", lines2)
        self.assertEqual(mrt0_words(spv2)[0], 100)

    def test_scc_conditional_branching_shader(self):
        """Translate shader utilizing SCC from s_or_b32 to drive s_cselect_b32."""
        cases = [
            (0, 0, 0, 555),
            (0, 1, 1, 777),
            (0x80000000, 0, 1, 777),
        ]
        for idx, (a, b, expected_scc, expected_sel) in enumerate(cases):
            with self.subTest(a=hex(a), b=hex(b)):
                lines = [
                    f"s_mov_b32 s0, {a:#x}",
                    f"s_mov_b32 s1, {b:#x}",
                    "s_or_b32 s2, s0, s1",
                    "s_cselect_b32 s3, 777, 555",
                    "v_mov_b32 v0, s2",
                    "v_mov_b32 v1, s3",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                    "exp mrt0 v0, v1, v2, v3 done vm",
                    "s_endpgm",
                ]
                spv = translate_lines(f"ps_scc_cselect_{idx}", lines)
                out = mrt0_words(spv)
                self.assertEqual(out[1], expected_sel)


class SaluOrB32FloatBitPatternTests(unittest.TestCase):
    """IEEE 754 boundary extremes, subnormals, and NaN handling for s_or_b32."""

    def test_ieee754_sign_bit_manipulation(self):
        """Inject sign bit (0x80000000) into positive floats via translator."""
        pos_zero = 0x00000000
        neg_zero = 0x80000000
        res, scc = run_salu_or_b32(pos_zero, 0x80000000)
        self.assertEqual(res, neg_zero)
        self.assertEqual(scc, 1)

        pos_one_bits = struct.unpack("<I", struct.pack("<f", 1.0))[0]
        neg_one_bits = struct.unpack("<I", struct.pack("<f", -1.0))[0]
        res_neg_one, scc_neg = run_salu_or_b32(pos_one_bits, 0x80000000)
        self.assertEqual(res_neg_one, neg_one_bits)
        self.assertEqual(scc_neg, 1)

    def test_nan_quieting_and_payload_synthesis(self):
        """Set quiet bit (bit 22, 0x00400000) to convert signaling NaN to quiet NaN via translator."""
        snan_bits = 0x7F800001
        qnan_expected = 0x7FC00001
        res, scc = run_salu_or_b32(snan_bits, 0x00400000)
        self.assertEqual(res, qnan_expected)
        self.assertEqual(scc, 1)

    def test_subnormal_to_infinity_promotion(self):
        """Promote subnormal float mantissa to infinity exponent using bitwise OR via translator."""
        subnorm = 0x000000FF
        inf_exp = 0x7F800000
        res, scc = run_salu_or_b32(subnorm, inf_exp)
        self.assertEqual(res, inf_exp | subnorm)
        self.assertEqual(scc, 1)
        self.assertTrue(math.isnan(struct.unpack("<f", struct.pack("<I", res))[0]))

    def test_translated_float_bitwise_shader(self):
        """Verify shader synthesizing float NaN and sign bit manipulation translates."""
        lines = [
            "s_mov_b32 s0, 0x3F800000",   # +1.0f
            "s_or_b32 s1, s0, 0x80000000", # -1.0f
            "v_mov_b32 v0, s1",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_float_or_shader", lines)
        out = mrt0_words(spv)
        expected_neg_one = struct.unpack("<I", struct.pack("<f", -1.0))[0]
        self.assertEqual(out[0], expected_neg_one)


class SaluOrB32NegativeControlsTests(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection for s_or_b32."""

    def test_mutation_discrimination_matrix(self):
        """Ensure s_or_b32 is strictly distinguished from AND, XOR, ADD, and NOR via translator."""
        pairs = [
            (0x12345678, 0x87654321),
            (0xFF00FF00, 0x00FF00FF),
            (0xAAAAAAAA, 0x55555555),
            (0x0000FFFF, 0xFFFF0000),
            (0x00000001, 0x00000002),
        ]
        for a, b in pairs:
            or_res, _ = run_salu_or_b32(a, b)
            and_res = (a & b) & MASK32
            xor_res = (a ^ b) & MASK32
            nor_res = (~(a | b)) & MASK32
            add_res = (a + b) & MASK32

            with self.subTest(a=hex(a), b=hex(b)):
                self.assertNotEqual(or_res, and_res)
                self.assertNotEqual(or_res, nor_res)
                if (a & b) != 0:
                    self.assertNotEqual(or_res, xor_res)
                    self.assertNotEqual(or_res, add_res)

    def test_interpreter_invalid_register_rejection(self):
        """Interpreter must reject invalid register access."""
        state = ScalarRegisterState()
        with self.assertRaises(ValueError):
            state.read_reg("v0")  # Vector register not allowed in SALU scalar state
        with self.assertRaises(ValueError):
            state.write_reg("s999", 1)
        with self.assertRaises(ValueError):
            state.resolve_src("unknown_symbol")

    def test_syntax_rejection_malformed_instruction(self):
        """RDNA2 parser / translator must reject malformed scalar instruction syntax."""
        bad_syntax_programs = [
            ["s_or_b32 s0, s1"],  # Missing second source operand
            ["s_or_b32"],          # Missing all operands
        ]
        for idx, lines in enumerate(bad_syntax_programs):
            with self.subTest(idx=idx):
                with self.assertRaises(Exception):
                    translate_lines(f"ps_bad_syntax_{idx}", lines)


if __name__ == "__main__":
    unittest.main()
