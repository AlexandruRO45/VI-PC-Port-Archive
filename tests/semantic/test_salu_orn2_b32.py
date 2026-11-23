"""Semantic interpreter tests and reference hardware-matching oracle for RDNA2 s_orn2_b32.

Models AMD RDNA2 / GFX10 Scalar ALU (SALU) SOP2 32-bit bitwise OR-NOT2 operation:
  s_orn2_b32 sdst, ssrc0, ssrc1

RDNA2 SALU Bitwise Architecture & Semantics:
  1. Instruction Operands:
     - sdst: Destination SGPR receiving the 32-bit bitwise result.
     - ssrc0: First 32-bit scalar source operand (SGPR, literal, or inline constant).
     - ssrc1: Second 32-bit scalar source operand (SGPR, literal, or inline constant).
  2. Bitwise Logic:
     sdst = (ssrc0 | (~ssrc1)) & 0xFFFFFFFF
     Note that ssrc1 is bitwise inverted (NOT) prior to the bitwise OR with ssrc0.
  3. Scalar Condition Code (SCC):
     SCC is updated based on the 32-bit result:
     SCC = 1 if (sdst != 0) else 0.
  4. Non-commutativity & Algebraic Properties:
     - Non-commutative: (A | ~B) != (B | ~A) in general.
     - Self-identity: (A | ~A) == 0xFFFFFFFF, with SCC = 1 for all A.
     - Zero-absorption: (0 | ~B) == ~B.
     - Maximum element: (A | ~0) == 0xFFFFFFFF; (A | ~0xFFFFFFFF) == A.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import random
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv
    import spirv_mini_interp as si
    import support
    TRANSLATOR_ERROR = None
except Exception as _err:
    rdna2spirv = None  # type: ignore
    si = None  # type: ignore
    support = None  # type: ignore
    TRANSLATOR_ERROR = str(_err)

MASK32: int = 0xFFFFFFFF

def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def ref_orn2_b32(s0: int, s1: int) -> tuple[int, int]:
    """Pure bitwise reference oracle for s_orn2_b32: D = S0 | ~S1, SCC = (D != 0).

    Returns tuple (sdst, scc) normalized to 32-bit unsigned integers.
    """
    s0_u = u32(s0)
    s1_u = u32(s1)
    res = u32(s0_u | (~s1_u))
    scc = 1 if res != 0 else 0
    return res, scc


class TestSaluOrn2B32Oracle(unittest.TestCase):
    """Unit tests verifying the reference mathematical oracle against RDNA2 hardware semantics."""

    def test_oracle_truth_table_basic(self) -> None:
        # 0 | ~0 = 0xFFFFFFFF, scc = 1
        self.assertEqual(ref_orn2_b32(0x00000000, 0x00000000), (0xFFFFFFFF, 1))
        # 0 | ~0xFFFFFFFF = 0x00000000, scc = 0 (only way to produce 0)
        self.assertEqual(ref_orn2_b32(0x00000000, 0xFFFFFFFF), (0x00000000, 0))
        # 0xFFFFFFFF | ~0 = 0xFFFFFFFF, scc = 1
        self.assertEqual(ref_orn2_b32(0xFFFFFFFF, 0x00000000), (0xFFFFFFFF, 1))
        # 0xFFFFFFFF | ~0xFFFFFFFF = 0xFFFFFFFF, scc = 1
        self.assertEqual(ref_orn2_b32(0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 1))

    def test_oracle_self_identity(self) -> None:
        # For any word X, X | ~X == 0xFFFFFFFF, SCC == 1
        test_vectors = [
            0x00000000, 0xFFFFFFFF, 0x12345678, 0xAAAAAAAA,
            0x55555555, 0x80000000, 0x7FFFFFFF, 0x00000001,
            0xCAFEBABE, 0xDEADBEEF,
        ]
        for val in test_vectors:
            res, scc = ref_orn2_b32(val, val)
            self.assertEqual(res, 0xFFFFFFFF, f"Self-identity failed for {val:#x}")
            self.assertEqual(scc, 1, f"SCC failed for {val:#x}")

    def test_oracle_non_commutativity(self) -> None:
        # A | ~B != B | ~A when A != B
        a = 0x0F0F0F0F
        b = 0x33333333
        res_ab, _ = ref_orn2_b32(a, b)  # 0x0F0F0F0F | 0xCCCCCCCC = 0xCFCFCFCF
        res_ba, _ = ref_orn2_b32(b, a)  # 0x33333333 | 0xF0F0F0F0 = 0xF3F3F3F3
        self.assertEqual(res_ab, 0xCFCFCFCF)
        self.assertEqual(res_ba, 0xF3F3F3F3)
        self.assertNotEqual(res_ab, res_ba)

    def test_oracle_bit_decomposition(self) -> None:
        # Check all 32 independent bit positions
        for bit in range(32):
            mask = 1 << bit
            # s0=0, s1=0: bit in ~s1 is 1 -> res has 1
            res, _ = ref_orn2_b32(0, 0)
            self.assertTrue(bool(res & mask))
            # s0=0, s1=mask: bit in ~s1 is 0 -> res has 0
            res, _ = ref_orn2_b32(0, mask)
            self.assertFalse(bool(res & mask))
            # s0=mask, s1=mask: bit in s0 is 1 -> res has 1
            res, _ = ref_orn2_b32(mask, mask)
            self.assertTrue(bool(res & mask))
            # s0=mask, s1=0: both are 1 -> res has 1
            res, _ = ref_orn2_b32(mask, 0)
            self.assertTrue(bool(res & mask))

    def test_oracle_exhaustive_nibbles(self) -> None:
        # Exhaustive verification across all 16x16 4-bit nibbles
        for n0 in range(16):
            for n1 in range(16):
                expected_nibble = (n0 | (~n1 & 0xF)) & 0xF
                expected_res = sum(expected_nibble << (shift * 4) for shift in range(8))
                res, _ = ref_orn2_b32(
                    sum(n0 << (shift * 4) for shift in range(8)),
                    sum(n1 << (shift * 4) for shift in range(8)),
                )
                self.assertEqual(res, expected_res)

class RegKind(enum.Enum):
    SGPR = "sgpr"
    VGPR = "vgpr"
    LITERAL = "literal"
    INLINE_INT = "inline_int"


@dataclasses.dataclass(frozen=True)
class RegOperand:
    """Representation of an RDNA2 register or constant operand."""
    kind: RegKind
    index_or_val: int

    def __post_init__(self) -> None:
        if self.kind == RegKind.SGPR:
            if not (0 <= self.index_or_val <= 105):
                raise ValueError(f"SGPR index out of range [0, 105]: {self.index_or_val}")
        elif self.kind == RegKind.VGPR:
            if not (0 <= self.index_or_val <= 255):
                raise ValueError(f"VGPR index out of range [0, 255]: {self.index_or_val}")

    def format_asm(self) -> str:
        if self.kind == RegKind.SGPR:
            return f"s{self.index_or_val}"
        elif self.kind == RegKind.VGPR:
            return f"v{self.index_or_val}"
        elif self.kind == RegKind.LITERAL:
            return f"{u32(self.index_or_val):#x}"
        elif self.kind == RegKind.INLINE_INT:
            return str(self.index_or_val)
        return str(self.index_or_val)


class Opcode(enum.Enum):
    S_ORN2_B32 = "s_orn2_b32"
    S_MOV_B32 = "s_mov_b32"


@dataclasses.dataclass(frozen=True)
class SALUInstruction:
    """AST node representing a Scalar ALU instruction."""
    opcode: Opcode
    sdst: RegOperand
    ssrc0: RegOperand
    ssrc1: RegOperand | None = None

    def __post_init__(self) -> None:
        if self.sdst.kind != RegKind.SGPR:
            raise ValueError("Destination of SALU instruction must be an SGPR")
        if self.opcode == Opcode.S_ORN2_B32:
            if self.ssrc1 is None:
                raise ValueError("s_orn2_b32 requires two source operands")
            literals = sum(1 for op in (self.ssrc0, self.ssrc1) if op is not None and op.kind == RegKind.LITERAL)
            if literals > 1:
                raise ValueError("SOP2 encoding cannot have more than one literal pool")

    def format_asm(self) -> str:
        if self.opcode == Opcode.S_ORN2_B32:
            assert self.ssrc1 is not None
            return f"{self.opcode.value} {self.sdst.format_asm()}, {self.ssrc0.format_asm()}, {self.ssrc1.format_asm()}"
        elif self.opcode == Opcode.S_MOV_B32:
            return f"{self.opcode.value} {self.sdst.format_asm()}, {self.ssrc0.format_asm()}"
        return f"{self.opcode.value} {self.sdst.format_asm()}"


class TestSaluOrn2B32AST(unittest.TestCase):
    """Unit tests for AST definitions and instruction formatting."""

    def test_ast_reg_formatting(self) -> None:
        self.assertEqual(RegOperand(RegKind.SGPR, 5).format_asm(), "s5")
        self.assertEqual(RegOperand(RegKind.VGPR, 12).format_asm(), "v12")
        self.assertEqual(RegOperand(RegKind.LITERAL, 0x12345678).format_asm(), "0x12345678")
        self.assertEqual(RegOperand(RegKind.INLINE_INT, 0).format_asm(), "0")

    def test_ast_instruction_formatting(self) -> None:
        inst = SALUInstruction(
            opcode=Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        self.assertEqual(inst.format_asm(), "s_orn2_b32 s0, s1, s2")

    def test_ast_instruction_with_literal(self) -> None:
        inst = SALUInstruction(
            opcode=Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 4),
            ssrc0=RegOperand(RegKind.SGPR, 5),
            ssrc1=RegOperand(RegKind.LITERAL, 0xCAFEBABE),
        )
        self.assertEqual(inst.format_asm(), "s_orn2_b32 s4, s5, 0xcafebabe")

    def test_ast_validation_errors(self) -> None:
        # Invalid SGPR index
        with self.assertRaises(ValueError):
            RegOperand(RegKind.SGPR, 106)
        with self.assertRaises(ValueError):
            RegOperand(RegKind.SGPR, -1)
        # SALU destination cannot be VGPR
        with self.assertRaises(ValueError):
            SALUInstruction(
                opcode=Opcode.S_ORN2_B32,
                sdst=RegOperand(RegKind.VGPR, 0),
                ssrc0=RegOperand(RegKind.SGPR, 1),
                ssrc1=RegOperand(RegKind.SGPR, 2),
            )
        # Two literals in one SOP2 instruction rejected
        with self.assertRaises(ValueError):
            SALUInstruction(
                opcode=Opcode.S_ORN2_B32,
                sdst=RegOperand(RegKind.SGPR, 0),
                ssrc0=RegOperand(RegKind.LITERAL, 0x1234),
                ssrc1=RegOperand(RegKind.LITERAL, 0x5678),
            )

class ScalarSALUInterpreter:
    """Deterministic architectural scalar instruction interpreter for SALU state."""

    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs: list[int] = [0] * num_sgprs
        self.scc: int = 0

    def read_operand(self, op: RegOperand) -> int:
        if op.kind == RegKind.SGPR:
            return self.sgprs[op.index_or_val]
        elif op.kind in (RegKind.LITERAL, RegKind.INLINE_INT):
            return u32(op.index_or_val)
        raise ValueError(f"Cannot read operand kind {op.kind} as scalar")

    def write_sgpr(self, reg_idx: int, val: int) -> None:
        self.sgprs[reg_idx] = u32(val)

    def execute(self, inst: SALUInstruction) -> tuple[int, int]:
        """Execute a single SALU instruction, updating state and returning (result, scc)."""
        if inst.opcode == Opcode.S_MOV_B32:
            val = self.read_operand(inst.ssrc0)
            self.write_sgpr(inst.sdst.index_or_val, val)
            return val, self.scc
        elif inst.opcode == Opcode.S_ORN2_B32:
            assert inst.ssrc1 is not None
            s0 = self.read_operand(inst.ssrc0)
            s1 = self.read_operand(inst.ssrc1)
            res, scc = ref_orn2_b32(s0, s1)
            self.write_sgpr(inst.sdst.index_or_val, res)
            self.scc = scc
            return res, scc
        raise NotImplementedError(f"Unsupported opcode: {inst.opcode}")


class TestSaluOrn2B32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarSALUInterpreter()

    def test_interp_single_execution(self) -> None:
        self.interp.write_sgpr(1, 0x11110000)
        self.interp.write_sgpr(2, 0x00FF00FF)
        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        res, scc = self.interp.execute(inst)
        expected_res = 0x11110000 | (~0x00FF00FF & MASK32)
        self.assertEqual(res, expected_res)
        self.assertEqual(self.interp.sgprs[0], expected_res)
        self.assertEqual(scc, 1)
        self.assertEqual(self.interp.scc, 1)

    def test_interp_sequential_chain(self) -> None:
        self.interp.write_sgpr(1, 0xAAAAAAAA)
        self.interp.write_sgpr(2, 0x55555555)
        # s3 = s1 | ~s2
        inst1 = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 3),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        self.interp.execute(inst1)
        self.assertEqual(self.interp.sgprs[3], 0xAAAAAAAA)
        # s4 = s3 | ~s1 (0xAAAAAAAA | ~0xAAAAAAAA = 0xFFFFFFFF)
        inst2 = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 4),
            ssrc0=RegOperand(RegKind.SGPR, 3),
            ssrc1=RegOperand(RegKind.SGPR, 1),
        )
        self.interp.execute(inst2)
        self.assertEqual(self.interp.sgprs[4], 0xFFFFFFFF)
        self.assertEqual(self.interp.scc, 1)

    def test_interp_unmodified_registers(self) -> None:
        self.interp.write_sgpr(10, 0x12345678)
        self.interp.write_sgpr(11, 0x9ABCDEF0)
        inst = SALUInstruction(
            Opcode.S_ORN2_B32,
            sdst=RegOperand(RegKind.SGPR, 0),
            ssrc0=RegOperand(RegKind.SGPR, 1),
            ssrc1=RegOperand(RegKind.SGPR, 2),
        )
        self.interp.execute(inst)
        self.assertEqual(self.interp.sgprs[10], 0x12345678)
        self.assertEqual(self.interp.sgprs[11], 0x9ABCDEF0)

if __name__ == "__main__":
    unittest.main()
