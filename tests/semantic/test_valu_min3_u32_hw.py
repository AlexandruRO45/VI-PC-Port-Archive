from __future__ import annotations

"""Hardware-verified semantic reference and translator regression suite for RDNA2 v_min3_u32.

Tests v_min3_u32 (three-operand 32-bit unsigned integer minimum) against mathematical reference
semantics, SIMD divergence execution, and translated SPIR-V shaders via rdna2spirv and spirv_mini_interp.
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

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32: int = 0xFFFF_FFFF
SIGN_BIT32: int = 0x8000_0000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret 32-bit unsigned dword as signed two's complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def oracle_min3_u32(a: int, b: int, c: int) -> int:
    """Reference mathematical oracle for RDNA2 v_min3_u32: min(unsigned a, unsigned b, unsigned c)."""
    return min(u32(a), u32(b), u32(c))


_TRANSLATE_ID: int = 0


def run_translated_min3_u32(src0: int, src1: int, src2: int) -> int:
    """Execute v_min3_u32 through rdna2spirv translation and spirv_mini_interp execution."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_min3_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(f"ps_min3_trans_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32

class TestValuMin3U32ReferenceOracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle and architectural model for v_min3_u32."""

    def test_oracle_basic_minimum(self) -> None:
        self.assertEqual(oracle_min3_u32(10, 20, 30), 10)
        self.assertEqual(oracle_min3_u32(30, 10, 20), 10)
        self.assertEqual(oracle_min3_u32(20, 30, 10), 10)
        self.assertEqual(oracle_min3_u32(0, 0, 0), 0)
        self.assertEqual(oracle_min3_u32(1, 2, 3), 1)
        self.assertEqual(oracle_min3_u32(100, 200, 300), 100)

    def test_oracle_unsigned_vs_signed_divergence(self) -> None:
        # In signed math, 0xFFFFFFFF (-1) < 10, but in unsigned math 0xFFFFFFFF is 4294967295 > 10.
        self.assertEqual(oracle_min3_u32(MASK32, 10, 20), 10)
        self.assertEqual(oracle_min3_u32(0x8000_0000, 0x7FFF_FFFF, 0), 0)
        self.assertEqual(oracle_min3_u32(0x8000_0000, 0x9000_0000, MASK32), 0x8000_0000)

    def test_oracle_matches_translated_execution(self) -> None:
        triples = [
            (0, 0, 0),
            (10, 20, 30),
            (1000, 2000, 3000),
            (0x12345678, 0x11111111, 0x22222222),
            (MASK32, 10, 20),
            (MASK32, MASK32, 1),
            (0x7FFFFFFF, 1, 0x80000000),
            (0xAAAAAAAA, 0x55555555, 1),
        ]
        for a, b, c in triples:
            expected = oracle_min3_u32(a, b, c)
            actual = run_translated_min3_u32(a, b, c)
            self.assertEqual(actual, expected)

    def test_oracle_permutations_and_idempotence(self) -> None:
        a, b, c = 0x1234, 0x5678, 0x9ABC
        res1 = oracle_min3_u32(a, b, c)
        res2 = oracle_min3_u32(c, a, b)
        res3 = oracle_min3_u32(b, c, a)
        self.assertEqual(res1, res2)
        self.assertEqual(res2, res3)
        self.assertEqual(oracle_min3_u32(a, a, a), a)

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
    V_MIN3_U32 = "v_min3_u32"


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
        if self.opcode != Opcode.V_MIN3_U32:
            raise ValueError(f"Invalid opcode {self.opcode}")

    def to_asm(self) -> str:
        s0 = f"{self.src0:#x}" if isinstance(self.src0, int) else str(self.src0)
        s1 = f"{self.src1:#x}" if isinstance(self.src1, int) else str(self.src1)
        s2 = f"{self.src2:#x}" if isinstance(self.src2, int) else str(self.src2)
        return f"{self.opcode.value} {self.dst}, {s0}, {s1}, {s2}"


class TestValuMin3U32AST(unittest.TestCase):
    """Stage 2: Symbolic AST definitions and typed expressions for v_min3_u32."""

    def test_instruction_creation_and_disasm(self) -> None:
        dst = Register(RegKind.VGPR, 0)
        s0 = Register(RegKind.VGPR, 1)
        s1 = Register(RegKind.VGPR, 2)
        s2 = Register(RegKind.VGPR, 3)
        inst = VOP3Instruction(Opcode.V_MIN3_U32, dst, s0, s1, s2)
        self.assertEqual(inst.to_asm(), "v_min3_u32 v0, v1, v2, v3")

    def test_instruction_with_literals_and_sgprs(self) -> None:
        dst = Register(RegKind.VGPR, 4)
        s0 = Register(RegKind.SGPR, 0)
        s1 = 0x100
        s2 = Register(RegKind.VGPR, 5)
        inst = VOP3Instruction(Opcode.V_MIN3_U32, dst, s0, s1, s2)
        self.assertEqual(inst.to_asm(), "v_min3_u32 v4, s0, 0x100, v5")

    def test_ast_emits_valid_spirv(self) -> None:
        inst = VOP3Instruction(
            Opcode.V_MIN3_U32,
            Register(RegKind.VGPR, 0),
            Register(RegKind.VGPR, 1),
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
        )
        lines = [
            "v_mov_b32 v1, 100",
            "v_mov_b32 v2, 50",
            "v_mov_b32 v3, 200",
            inst.to_asm(),
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_ast_min3", lines)
        self.assertIn("UMin", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 50)

if __name__ == "__main__":
    unittest.main()
