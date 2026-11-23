from __future__ import annotations

"""Hardware-verified semantic reference and translator regression suite for RDNA2 v_add3_u32.

Tests v_add3_u32 (three-operand 32-bit unsigned addition) against mathematical reference semantics
and translated SPIR-V shader executions via rdna2spirv and spirv_mini_interp.
"""

import dataclasses
import enum
import os
import struct
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def oracle_add3_u32(a: int, b: int, c: int) -> int:
    """Reference mathematical oracle for RDNA2 v_add3_u32: (a + b + c) mod 2^32."""
    return (u32(a) + u32(b) + u32(c)) & MASK32


_TRANSLATE_ID: int = 0


def run_translated_add3_u32(src0: int, src1: int, src2: int) -> int:
    """Execute v_add3_u32 through rdna2spirv translation and spirv_mini_interp execution."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_add3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_add3_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


class TestValuAdd3U32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_add3_u32."""

    def test_oracle_basic_addition(self) -> None:
        self.assertEqual(oracle_add3_u32(10, 20, 30), 60)
        self.assertEqual(oracle_add3_u32(0, 0, 0), 0)
        self.assertEqual(oracle_add3_u32(1, 2, 3), 6)
        self.assertEqual(oracle_add3_u32(100, 200, 300), 600)

    def test_oracle_wrapping(self) -> None:
        self.assertEqual(oracle_add3_u32(MASK32, 1, 0), 0)
        self.assertEqual(oracle_add3_u32(MASK32, 1, 5), 5)
        self.assertEqual(oracle_add3_u32(MASK32, MASK32, MASK32), MASK32 - 2)

    def test_oracle_matches_translated_execution(self) -> None:
        triples = [
            (0, 0, 0),
            (10, 20, 30),
            (1000, 2000, 3000),
            (0x12345678, 0x11111111, 0x22222222),
            (MASK32, 1, 0),
            (MASK32, MASK32, 1),
            (0x7FFFFFFF, 1, 0x80000000),
            (0xAAAAAAAA, 0x55555555, 1),
        ]
        for a, b, c in triples:
            expected = oracle_add3_u32(a, b, c)
            actual = run_translated_add3_u32(a, b, c)
            self.assertEqual(actual, expected)

    def test_oracle_commutativity_and_associativity(self) -> None:
        a, b, c = 0x1234, 0x5678, 0x9ABC
        res1 = oracle_add3_u32(a, b, c)
        res2 = oracle_add3_u32(c, a, b)
        res3 = oracle_add3_u32(b, c, a)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)


# ---------------------------------------------------------------------------
# Stage 2: Symbolic AST definitions and typed expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    VGPR = enum.auto()
    SGPR = enum.auto()


@dataclasses.dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of range [0, 255]")
        if self.kind == RegKind.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of range [0, 105]")

    def __str__(self) -> str:
        prefix = "v" if self.kind == RegKind.VGPR else "s"
        return f"{prefix}{self.index}"


class Opcode(enum.Enum):
    V_ADD3_U32 = "v_add3_u32"


@dataclasses.dataclass
class VOP3Instruction:
    opcode: Opcode
    dst: Register
    src0: Union[Register, int]
    src1: Union[Register, int]
    src2: Union[Register, int]

    def __post_init__(self) -> None:
        if self.dst.kind != RegKind.VGPR:
            raise ValueError("VOP3 destination must be VGPR")
        if self.opcode != Opcode.V_ADD3_U32:
            raise ValueError(f"Invalid opcode {self.opcode}")

    def to_asm(self) -> str:
        s0 = f"{self.src0:#x}" if isinstance(self.src0, int) else str(self.src0)
        s1 = f"{self.src1:#x}" if isinstance(self.src1, int) else str(self.src1)
        s2 = f"{self.src2:#x}" if isinstance(self.src2, int) else str(self.src2)
        return f"{self.opcode.value} {self.dst}, {s0}, {s1}, {s2}"


class TestValuAdd3U32AST(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_add3_u32."""

    def test_instruction_creation_and_disasm(self) -> None:
        dst = Register(RegKind.VGPR, 0)
        s0 = Register(RegKind.VGPR, 1)
        s1 = Register(RegKind.VGPR, 2)
        s2 = Register(RegKind.VGPR, 3)
        inst = VOP3Instruction(Opcode.V_ADD3_U32, dst, s0, s1, s2)
        self.assertEqual(inst.to_asm(), "v_add3_u32 v0, v1, v2, v3")

    def test_instruction_with_literals_and_sgprs(self) -> None:
        dst = Register(RegKind.VGPR, 4)
        s0 = Register(RegKind.SGPR, 0)
        s1 = 0x100
        s2 = Register(RegKind.VGPR, 5)
        inst = VOP3Instruction(Opcode.V_ADD3_U32, dst, s0, s1, s2)
        self.assertEqual(inst.to_asm(), "v_add3_u32 v4, s0, 0x100, v5")

    def test_ast_emits_valid_spirv(self) -> None:
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        lines = [
            "v_mov_b32 v1, 10",
            "v_mov_b32 v2, 20",
            "v_mov_b32 v3, 30",
            inst.to_asm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_add3", lines)
        self.assertIn("OpIAdd", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 60)


# ---------------------------------------------------------------------------
# Stage 3: Deterministic scalar instruction interpreter
# ---------------------------------------------------------------------------

class ScalarMachineState:
    """Architectural state for executing VALU scalar instructions."""

    def __init__(self) -> None:
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106

    def read_operand(self, op: Union[Register, int]) -> int:
        if isinstance(op, int):
            return op & MASK32
        if op.kind == RegKind.VGPR:
            return self.vgprs[op.index]
        return self.sgprs[op.index]

    def write_reg(self, reg: Register, val: int) -> None:
        val &= MASK32
        if reg.kind == RegKind.VGPR:
            self.vgprs[reg.index] = val
        else:
            self.sgprs[reg.index] = val

    def step(self, inst: VOP3Instruction) -> int:
        val0 = self.read_operand(inst.src0)
        val1 = self.read_operand(inst.src1)
        val2 = self.read_operand(inst.src2)
        res = oracle_add3_u32(val0, val1, val2)
        self.write_reg(inst.dst, res)
        return res


class TestValuAdd3U32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter for v_add3_u32."""

    def setUp(self) -> None:
        self.state = ScalarMachineState()

    def test_scalar_step_basic(self) -> None:
        self.state.vgprs[1] = 5
        self.state.vgprs[2] = 10
        self.state.vgprs[3] = 15
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 30)
        self.assertEqual(self.state.vgprs[0], 30)

    def test_scalar_step_sgpr_and_imm(self) -> None:
        self.state.sgprs[4] = 100
        self.state.vgprs[1] = 50
        inst = VOP3Instruction(
            Opcode.V_ADD3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SGPR, 4),
            25,
            Register(RegKind.VGPR, 1),
        )
        res = self.state.step(inst)
        self.assertEqual(res, 175)
        self.assertEqual(self.state.vgprs[0], 175)

    def test_scalar_matches_translated_shader(self) -> None:
        test_cases = [
            (1, 2, 3),
            (100, 200, 300),
            (0xFFFF_0000, 0x0000_FFFF, 1),
            (0x8000_0000, 0x4000_0000, 0x4000_0000),
        ]
        for a, b, c in test_cases:
            self.state.vgprs[1] = a
            self.state.vgprs[2] = b
            self.state.vgprs[3] = c
            inst = VOP3Instruction(
                Opcode.V_ADD3_U32,
                Register(RegKind.VGPR, 0),
                Register(RegKind.VGPR, 1),
                Register(RegKind.VGPR, 2),
                Register(RegKind.VGPR, 3),
            )
            scalar_res = self.state.step(inst)
            trans_res = run_translated_add3_u32(a, b, c)
            self.assertEqual(scalar_res, trans_res)


if __name__ == "__main__":
    unittest.main()
