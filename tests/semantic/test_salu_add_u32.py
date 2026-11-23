from __future__ import annotations

"""Semantic reference tests and translator-backed validation for AMD RDNA2 s_add_u32.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, s_add_u32 computes the 32-bit unsigned addition
  of two scalar source operands with carry-out generation:
      D.u = (S0.u + S1.u) mod 2^32
      SCC = (S0.u + S1.u >= 2^32) ? 1 : 0

  - Format: SOP2 (Scalar Operation with 2 inputs, 1 destination).
  - Operands:
      S0 and S1 can be scalar general-purpose registers (SGPR), inline constants
      (-16..64, float constants), or 32-bit literal constants.
      D is a 32-bit destination SGPR.
  - Condition code register (SCC):
      Updated with the 1-bit unsigned carry-out: 1 if unsigned sum overflows 2^32,
      0 otherwise. Prior SCC state does not affect the calculation.
  - SPIR-V translation (tools/rdna2spirv.py):
      Lowers to OpIAddCarry %umul %s0 %s1.
      Member 0: 32-bit wrapped sum stored to destination SGPR.
      Member 1: 32-bit carry-out flag stored to %scc variable.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""

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


# --- Section 1: Mathematical Reference Oracle ---

def ref_s_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Pure mathematical reference oracle for RDNA2 s_add_u32.

    Returns (wrapped_sum, scc_carry).
    """
    a = s0 & MASK32
    b = s1 & MASK32
    full = a + b
    wrapped = full & MASK32
    carry = 1 if full >= (1 << 32) else 0
    return wrapped, carry


def oracle_add_u32_analysis(s0: int, s1: int) -> dict[str, int | bool]:
    """Detailed architectural breakdown of s_add_u32 execution."""
    a = s0 & MASK32
    b = s1 & MASK32
    full = a + b
    has_carry = full >= (1 << 32)
    return {
        "s0": a,
        "s1": b,
        "full_sum": full,
        "carry": has_carry,
        "scc": 1 if has_carry else 0,
        "wrapped_result": full & MASK32,
    }


class SaluAddU32ReferenceOracleTests(unittest.TestCase):
    """Commit 1: Reference oracle tests for RDNA2 s_add_u32."""

    def test_oracle_zero_identity(self) -> None:
        """Adding zero to any 32-bit value preserves the value and sets SCC=0."""
        cases = [0, 1, 42, 0x12345678, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                res0, scc0 = ref_s_add_u32(val, 0)
                self.assertEqual(res0, val)
                self.assertEqual(scc0, 0)

                res1, scc1 = ref_s_add_u32(0, val)
                self.assertEqual(res1, val)
                self.assertEqual(scc1, 0)

    def test_oracle_basic_addition_no_carry(self) -> None:
        """Sums strictly below 2^32 must produce exact integer sum with SCC=0."""
        cases = [
            (10, 20, 30),
            (0x1000, 0x2000, 0x3000),
            (0x12345678, 0x01020304, 0x1336597C),
            (0x7FFFFFFE, 1, 0x7FFFFFFF),
            (0x40000000, 0x3FFFFFFF, 0x7FFFFFFF),
            (0x7FFFFFFF, 0x7FFFFFFF, 0xFFFFFFFE),
            (0x7FFFFFFF, 0x80000000, 0xFFFFFFFF),
        ]
        for a, b, expected_sum in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                res, scc = ref_s_add_u32(a, b)
                self.assertEqual(res, expected_sum)
                self.assertEqual(scc, 0)

    def test_oracle_carry_generation(self) -> None:
        """Sums at or above 2^32 must set SCC=1 and wrap modulo 2^32."""
        cases = [
            (0xFFFFFFFF, 1, 0, 1),
            (0xFFFFFFFF, 2, 1, 1),
            (0xFFFFFFFE, 2, 0, 1),
            (0x80000000, 0x80000000, 0, 1),
            (0x80000000, 0x80000001, 1, 1),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 1),
            (0xC0000000, 0x50000000, 0x10000000, 1),
        ]
        for a, b, exp_res, exp_scc in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                res, scc = ref_s_add_u32(a, b)
                self.assertEqual(res, exp_res)
                self.assertEqual(scc, exp_scc)

    def test_oracle_commutativity(self) -> None:
        """s_add_u32 must be strictly commutative across all operand ranges."""
        pairs = [
            (0, 0),
            (0, 0xFFFFFFFF),
            (1, 0xFFFFFFFF),
            (0x12345678, 0x87654321),
            (0x80000000, 0x80000000),
            (0x7FFFFFFF, 0x80000000),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0xCAFEBABE, 0xDEADBEEF),
            (0xAAAAAAAA, 0x55555555),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(a), b=hex(b)):
                r1, s1 = ref_s_add_u32(a, b)
                r2, s2 = ref_s_add_u32(b, a)
                self.assertEqual(r1, r2)
                self.assertEqual(s1, s2)

    def test_oracle_analysis_breakdown(self) -> None:
        """Detailed analysis oracle must compute full sums and carry flags."""
        info = oracle_add_u32_analysis(0xFFFFFFFF, 1)
        self.assertEqual(info["s0"], 0xFFFFFFFF)
        self.assertEqual(info["s1"], 1)
        self.assertEqual(info["full_sum"], 0x100000000)
        self.assertTrue(info["carry"])
        self.assertEqual(info["scc"], 1)
        self.assertEqual(info["wrapped_result"], 0)

        info2 = oracle_add_u32_analysis(0x1234, 0x5678)
        self.assertFalse(info2["carry"])
        self.assertEqual(info2["scc"], 0)
        self.assertEqual(info2["wrapped_result"], 0x68AC)



# --- Section 2: Symbolic AST Opcode Definitions ---

class RegKind(enum.Enum):
    SGPR = "sgpr"
    VGPR = "vgpr"
    IMM = "imm"
    LITERAL = "literal"


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


def parse_salu_instruction(line: str) -> InstructionAST:
    """Parse a single SALU SOP2 assembly line into an InstructionAST."""
    line = line.split(";")[0].strip()
    if not line:
        raise ValueError("Empty instruction line")
    parts = line.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError(f"SOP2 instruction requires 3 operands, got: {line}")
    mnemonic, d_str, s0_str, s1_str = parts
    if mnemonic != "s_add_u32":
        raise ValueError(f"Expected s_add_u32 mnemonic, got {mnemonic}")

    def parse_op(tok: str) -> Operand:
        tok = tok.strip()
        if tok.startswith("s"):
            return Operand(RegKind.SGPR, index=int(tok[1:]))
        elif tok.startswith("v"):
            return Operand(RegKind.VGPR, index=int(tok[1:]))
        else:
            val = int(tok, 0)
            if -16 <= val <= 64:
                return Operand(RegKind.IMM, raw_val=val & MASK32)
            return Operand(RegKind.LITERAL, raw_val=val & MASK32)

    dst = parse_op(d_str)
    if dst.kind != RegKind.SGPR:
        raise ValueError(f"Destination must be an SGPR, got: {d_str}")

    return InstructionAST(mnemonic, dst, parse_op(s0_str), parse_op(s1_str))


class SaluAddU32AstValidationTests(unittest.TestCase):
    """Commit 2: AST definition and parsing tests for s_add_u32."""

    def test_ast_construction_sgpr_operands(self) -> None:
        ast = InstructionAST(
            mnemonic="s_add_u32",
            dst=Operand(RegKind.SGPR, index=4),
            src0=Operand(RegKind.SGPR, index=2),
            src1=Operand(RegKind.SGPR, index=3),
        )
        self.assertEqual(ast.mnemonic, "s_add_u32")
        self.assertEqual(ast.dst.kind, RegKind.SGPR)
        self.assertEqual(ast.dst.index, 4)
        self.assertEqual(ast.src0.index, 2)
        self.assertEqual(ast.src1.index, 3)

    def test_ast_imm_and_literal_operands(self) -> None:
        ast = InstructionAST(
            mnemonic="s_add_u32",
            dst=Operand(RegKind.SGPR, index=0),
            src0=Operand(RegKind.SGPR, index=1),
            src1=Operand(RegKind.IMM, raw_val=42),
        )
        self.assertEqual(ast.src1.kind, RegKind.IMM)
        self.assertEqual(ast.src1.raw_val, 42)

    def test_parser_valid_sop2_instructions(self) -> None:
        cases = [
            ("s_add_u32 s4, s2, s3", 4, RegKind.SGPR, 2, RegKind.SGPR, 3),
            ("s_add_u32 s0, s1, 1", 0, RegKind.SGPR, 1, RegKind.IMM, 1),
            ("s_add_u32 s10, 0x1000, s5", 10, RegKind.LITERAL, 0x1000, RegKind.SGPR, 5),
        ]
        for line, dst_idx, s0_k, s0_val, s1_k, s1_val in cases:
            ast = parse_salu_instruction(line)
            self.assertEqual(ast.mnemonic, "s_add_u32")
            self.assertEqual(ast.dst.index, dst_idx)
            self.assertEqual(ast.src0.kind, s0_k)
            self.assertEqual(ast.src1.kind, s1_k)

    def test_parser_syntax_rejection(self) -> None:
        invalid = [
            "s_add_u32",
            "s_add_u32 s0",
            "s_add_u32 s0, s1",
            "s_add_u32 s0, s1, s2, s3",
            "s_add_u32 v0, s1, s2",  # dst must be SGPR
            "v_add_u32 s0, s1, s2",  # wrong mnemonic
        ]
        for line in invalid:
            with self.subTest(line=line):
                with self.assertRaises(ValueError):
                    parse_salu_instruction(line)


# --- Section 3: Deterministic Scalar Interpreter ---

class ScalarRegisterState:
    """Simulates scalar register file (SGPRs) and condition code (SCC)."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs = [0] * num_sgprs
        self.scc: int = 0

    def get_sgpr(self, idx: int) -> int:
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def get_operand_val(self, op: Operand) -> int:
        if op.kind == RegKind.SGPR:
            return self.get_sgpr(op.index)
        return op.raw_val & MASK32

    def step(self, ast: InstructionAST) -> tuple[int, int]:
        """Execute a single SALU instruction, returning (dst_val, new_scc)."""
        s0 = self.get_operand_val(ast.src0)
        s1 = self.get_operand_val(ast.src1)
        if ast.mnemonic == "s_add_u32":
            res, scc = ref_s_add_u32(s0, s1)
            self.scc = scc
            self.set_sgpr(ast.dst.index, res)
            return res, scc
        raise NotImplementedError(f"Unsupported mnemonic: {ast.mnemonic}")


class SaluAddU32ScalarInterpreterTests(unittest.TestCase):
    """Commit 3: Deterministic scalar instruction interpreter tests."""

    def test_scalar_step_basic_add(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(2, 0x100)
        state.set_sgpr(3, 0x200)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 4), Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 3))
        res, scc = state.step(ast)
        self.assertEqual(res, 0x300)
        self.assertEqual(scc, 0)
        self.assertEqual(state.get_sgpr(4), 0x300)
        self.assertEqual(state.scc, 0)

    def test_scalar_step_carry_update(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(0, 0xFFFFFFFF)
        state.set_sgpr(1, 5)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 2), Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 1))
        res, scc = state.step(ast)
        self.assertEqual(res, 4)
        self.assertEqual(scc, 1)
        self.assertEqual(state.get_sgpr(2), 4)
        self.assertEqual(state.scc, 1)

    def test_scalar_chain_accumulations(self) -> None:
        state = ScalarRegisterState()
        state.set_sgpr(0, 10)
        ast = InstructionAST("s_add_u32", Operand(RegKind.SGPR, 0), Operand(RegKind.SGPR, 0), Operand(RegKind.IMM, raw_val=5))
        for expected in [15, 20, 25, 30]:
            res, scc = state.step(ast)
            self.assertEqual(res, expected)
            self.assertEqual(scc, 0)
            self.assertEqual(state.get_sgpr(0), expected)

if __name__ == "__main__":
    unittest.main()
