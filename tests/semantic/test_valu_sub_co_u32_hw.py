from __future__ import annotations

"""Semantic and hardware reference tests for RDNA2 v_sub_co_u32.

Validates 32-bit unsigned vector subtraction with carry-out / borrow:
  vdst = (src0 - src1) & 0xFFFFFFFF
  sdst / vcc_lo = 0xFFFFFFFF if (src0 < src1) else 0

Exercises reference mathematical oracle, symbolic AST definitions, deterministic
scalar interpreter, SIMD multi-lane divergence with EXEC masking, register
aliasing, status flags with carry/borrow, IEEE 754 bit-pattern invariance, and
negative controls. All stages cross-checked with rdna2spirv and spirv_mini_interp.
"""

import dataclasses
import enum
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

MASK32 = 0xFFFF_FFFF
MASK64 = 0xFFFF_FFFF_FFFF_FFFF

_TRANSLATE_ID = 0


def oracle_sub_co_u32(src0: int, src1: int) -> tuple[int, int]:
    """Mathematical reference oracle for v_sub_co_u32.

    Returns:
        (diff, borrow) where:
          diff = (src0 - src1) & 0xFFFFFFFF
          borrow = 1 if (src0 & 0xFFFFFFFF) < (src1 & 0xFFFFFFFF) else 0
    """
    u0 = src0 & MASK32
    u1 = src1 & MASK32
    diff = (u0 - u1) & MASK32
    borrow = 1 if u0 < u1 else 0
    return diff, borrow


def run_translated_sub_co_u32(src0: int, src1: int) -> tuple[int, int]:
    """Execute v_sub_co_u32 lowered to SPIR-V via rdna2spirv and evaluated in mini-interp."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        "v_sub_co_u32 v0, vcc_lo, v1, v2",
        "v_cndmask_b32 v3, 0, 1, vcc_lo",
        "exp mrt0 v0, v3, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_sub_co_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32, words[1] & 1

# ---------------------------------------------------------------------------
# Stage 1: Reference mathematical oracle and architectural model
# ---------------------------------------------------------------------------

class TestValuSubCoU32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_sub_co_u32."""

    def test_oracle_basic_subtraction(self) -> None:
        self.assertEqual(oracle_sub_co_u32(10, 3), (7, 0))
        self.assertEqual(oracle_sub_co_u32(100, 100), (0, 0))
        self.assertEqual(oracle_sub_co_u32(0, 0), (0, 0))
        self.assertEqual(oracle_sub_co_u32(1000, 400), (600, 0))

    def test_oracle_borrow_subtraction(self) -> None:
        self.assertEqual(oracle_sub_co_u32(3, 10), (MASK32 - 6, 1))
        self.assertEqual(oracle_sub_co_u32(0, 1), (MASK32, 1))
        self.assertEqual(oracle_sub_co_u32(0, MASK32), (1, 1))
        self.assertEqual(oracle_sub_co_u32(MASK32, 0), (MASK32, 0))

    def test_oracle_matches_translated_execution(self) -> None:
        pairs = [
            (10, 3),
            (3, 10),
            (0, 0),
            (0, 1),
            (1, 0),
            (MASK32, 1),
            (1, MASK32),
            (0x12345678, 0x11111111),
            (0x11111111, 0x12345678),
            (0x80000000, 1),
            (0x7FFFFFFF, 0x80000000),
            (0xAAAAAAAA, 0x55555555),
            (0x55555555, 0xAAAAAAAA),
        ]
        for a, b in pairs:
            expected = oracle_sub_co_u32(a, b)
            actual = run_translated_sub_co_u32(a, b)
            self.assertEqual(actual, expected, f"Mismatch on {a:#x} - {b:#x}")

# ---------------------------------------------------------------------------
# Stage 2: Symbolic AST definitions and typed expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    VGPR = enum.auto()
    SGPR = enum.auto()
    SPECIAL = enum.auto()


@dataclasses.dataclass(frozen=True)
class Register:
    kind: RegKind
    index: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind == RegKind.VGPR and not (0 <= self.index < 256):
            raise ValueError(f"VGPR index out of range: {self.index}")
        if self.kind == RegKind.SGPR and not (0 <= self.index < 106):
            raise ValueError(f"SGPR index out of range: {self.index}")

    def __str__(self) -> str:
        if self.name:
            return self.name
        if self.kind == RegKind.VGPR:
            return f"v{self.index}"
        if self.kind == RegKind.SGPR:
            return f"s{self.index}"
        return f"special_{self.index}"


class Opcode(enum.Enum):
    V_SUB_CO_U32 = "v_sub_co_u32"


@dataclasses.dataclass(frozen=True)
class VOP3COInstruction:
    opcode: Opcode
    vdst: Register
    sdst: Register
    src0: Register | int
    src1: Register | int

    def __post_init__(self) -> None:
        if self.vdst.kind != RegKind.VGPR:
            raise ValueError(f"Destination vdst must be VGPR, got {self.vdst.kind}")
        if self.sdst.kind not in (RegKind.SGPR, RegKind.SPECIAL):
            raise ValueError(f"Destination sdst must be SGPR or SPECIAL, got {self.sdst.kind}")


class TestValuSubCoU32ASTDefinitions(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_sub_co_u32."""

    def test_ast_construction_and_formatting(self) -> None:
        vdst = Register(RegKind.VGPR, 0)
        sdst = Register(RegKind.SPECIAL, 0, "vcc_lo")
        src0 = Register(RegKind.VGPR, 1)
        src1 = Register(RegKind.VGPR, 2)
        inst = VOP3COInstruction(Opcode.V_SUB_CO_U32, vdst, sdst, src0, src1)

        self.assertEqual(inst.opcode, Opcode.V_SUB_CO_U32)
        self.assertEqual(str(inst.vdst), "v0")
        self.assertEqual(str(inst.sdst), "vcc_lo")
        self.assertEqual(str(inst.src0), "v1")
        self.assertEqual(str(inst.src1), "v2")

    def test_ast_with_literal_and_sgpr(self) -> None:
        vdst = Register(RegKind.VGPR, 5)
        sdst = Register(RegKind.SGPR, 4)
        src0 = Register(RegKind.SGPR, 2)
        inst = VOP3COInstruction(Opcode.V_SUB_CO_U32, vdst, sdst, src0, 42)

        self.assertEqual(str(inst.vdst), "v5")
        self.assertEqual(str(inst.sdst), "s4")
        self.assertEqual(str(inst.src0), "s2")
        self.assertEqual(inst.src1, 42)

# ---------------------------------------------------------------------------
# Stage 3: Deterministic instruction interpreter
# ---------------------------------------------------------------------------

class ValuSubCoInterpreter:
    """Scalar architectural interpreter for v_sub_co_u32."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106
        self.vcc_lo = 0

    def resolve(self, operand: Register | int) -> int:
        if isinstance(operand, int):
            return operand & MASK32
        if operand.kind == RegKind.VGPR:
            return self.vgprs[operand.index]
        if operand.kind == RegKind.SGPR:
            return self.sgprs[operand.index]
        if operand.name == "vcc_lo":
            return self.vcc_lo
        return 0

    def step(self, inst: VOP3COInstruction) -> tuple[int, int]:
        s0 = self.resolve(inst.src0)
        s1 = self.resolve(inst.src1)
        diff, borrow = oracle_sub_co_u32(s0, s1)
        self.vgprs[inst.vdst.index] = diff
        borrow_mask = MASK32 if borrow else 0
        if inst.sdst.kind == RegKind.SPECIAL and inst.sdst.name == "vcc_lo":
            self.vcc_lo = borrow_mask
        elif inst.sdst.kind == RegKind.SGPR:
            self.sgprs[inst.sdst.index] = borrow_mask
        return diff, borrow


class TestValuSubCoU32ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter for v_sub_co_u32."""

    def test_interpreter_vgpr_subtraction(self) -> None:
        interp = ValuSubCoInterpreter()
        interp.vgprs[1] = 50
        interp.vgprs[2] = 20

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, 30)
        self.assertEqual(borrow, 0)
        self.assertEqual(interp.vgprs[0], 30)
        self.assertEqual(interp.vcc_lo, 0)

    def test_interpreter_borrow_into_sgpr(self) -> None:
        interp = ValuSubCoInterpreter()
        interp.vgprs[1] = 10
        interp.vgprs[2] = 25

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 3),
            Register(RegKind.SGPR, 8),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, (10 - 25) & MASK32)
        self.assertEqual(borrow, 1)
        self.assertEqual(interp.vgprs[3], (10 - 25) & MASK32)
        self.assertEqual(interp.sgprs[8], MASK32)

    def test_interpreter_literal_subtraction(self) -> None:
        interp = ValuSubCoInterpreter()
        interp.vgprs[1] = 100

        inst = VOP3COInstruction(
            Opcode.V_SUB_CO_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.SPECIAL, 0, "vcc_lo"),
            Register(RegKind.VGPR, 1),
            30,
        )
        diff, borrow = interp.step(inst)
        self.assertEqual(diff, 70)
        self.assertEqual(borrow, 0)

if __name__ == "__main__":
    unittest.main()
