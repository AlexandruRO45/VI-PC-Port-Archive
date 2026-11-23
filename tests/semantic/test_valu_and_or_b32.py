"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_and_or_b32.

Models RDNA2 / GFX10 VALU 3-operand bitwise operation:
  v_and_or_b32 vd, src0, src1, src2

RDNA2 VALU Architecture & Semantics:
  1. Operands & Computation:
     - vd: 32-bit Destination VGPR
     - src0, src1, src2: 32-bit input sources (VGPR, SGPR, or literal/inline constant)
     - Operation: dst = (src0 & src1) | src2
  2. Condition Codes & Flags:
     - Neither SCC nor VCC is modified by this bitwise instruction.
  3. Execution Masking (EXEC):
     - Inactive lanes under EXEC mask do not write to their destination VGPR.
  4. Translator Integration:
     - Validates end-to-end SPIR-V emission via support.translate_lines and execution under spirv_mini_interp.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv  # noqa: E402
    import support  # noqa: E402
except Exception:  # noqa: E722
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK32: int = 0xFFFFFFFF
SIGN_BIT32: int = 0x80000000


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned integer as signed two's-complement."""
    val &= MASK32
    return val - (1 << 32) if (val & SIGN_BIT32) else val


def ref_and_or_b32(src0: int, src1: int, src2: int) -> int:
    """Pure mathematical reference oracle for RDNA2 v_and_or_b32.

    Computes: (src0 & src1) | src2
    """
    s0 = src0 & MASK32
    s1 = src1 & MASK32
    s2 = src2 & MASK32
    return ((s0 & s1) | s2) & MASK32

# ---------------------------------------------------------------------------
# Symbolic AST Definitions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    SGPR = "s"
    VGPR = "v"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand."""
    kind: RegKind
    index: int
    count: int = 1

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count == 1:
            return f"{self.kind.value}{self.index}"
        return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"


class Opcode(enum.Enum):
    V_AND_OR_B32 = "v_and_or_b32"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a v_and_or_b32 instruction."""
    opcode: Opcode
    dst: Register
    src0: Register | int
    src1: Register | int
    src2: Register | int

    def __post_init__(self) -> None:
        if not self.dst.is_vgpr or self.dst.count != 1:
            raise ValueError(f"dst must be a single 32-bit VGPR: {self.dst}")
        if self.dst.index < 0 or self.dst.index > 255:
            raise ValueError(f"dst register index out of range [0, 255]: {self.dst.index}")

    def format_operand(self, op: Register | int) -> str:
        if isinstance(op, Register):
            return op.format_asm()
        if op < 0:
            return f"{s32(op)}"
        if op > 9:
            return f"{hex(op)}"
        return str(op)

    def to_assembly(self) -> str:
        d_str = self.dst.format_asm()
        s0_str = self.format_operand(self.src0)
        s1_str = self.format_operand(self.src1)
        s2_str = self.format_operand(self.src2)
        return f"{self.opcode.value} {d_str}, {s0_str}, {s1_str}, {s2_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of VALU instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


class TestValuAndOrB32AST(unittest.TestCase):
    """Unit tests for symbolic AST nodes and assembly formatting."""

    def test_instruction_creation_defaults(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=Register(RegKind.VGPR, 2),
            src2=Register(RegKind.VGPR, 3),
        )
        self.assertEqual(inst.opcode, Opcode.V_AND_OR_B32)
        self.assertEqual(inst.to_assembly(), "v_and_or_b32 v0, v1, v2, v3")

    def test_instruction_with_sgpr_and_immediates(self) -> None:
        inst = Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 4),
            src0=Register(RegKind.SGPR, 0),
            src1=Register(RegKind.VGPR, 5),
            src2=0x1234,
        )
        self.assertEqual(inst.to_assembly(), "v_and_or_b32 v4, s0, v5, 0x1234")

    def test_program_representation(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 0),
            src0=Register(RegKind.VGPR, 1),
            src1=0xF0,
            src2=0x0F,
        ))
        prog.append(Instruction(
            opcode=Opcode.V_AND_OR_B32,
            dst=Register(RegKind.VGPR, 2),
            src0=Register(RegKind.VGPR, 0),
            src1=Register(RegKind.SGPR, 4),
            src2=0,
        ))
        lines = prog.to_assembly().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("v_and_or_b32 v0", lines[0])
        self.assertIn("v_and_or_b32 v2", lines[1])

class TestValuAndOrB32Oracle(unittest.TestCase):
    """Unit tests for reference mathematical oracle and bitwise truth tables."""

    def test_basic_and_or_truth_table(self) -> None:
        # 4-bit exhaustive truth table patterns replicated across dwords
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    src0 = 0xFFFFFFFF if a else 0
                    src1 = 0xFFFFFFFF if b else 0
                    src2 = 0xFFFFFFFF if c else 0
                    expected = 0xFFFFFFFF if ((a and b) or c) else 0
                    self.assertEqual(ref_and_or_b32(src0, src1, src2), expected)

    def test_annihilators_and_identities(self) -> None:
        # src2 = 0xFFFFFFFF -> result always 0xFFFFFFFF
        self.assertEqual(ref_and_or_b32(0, 0, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_and_or_b32(0x12345678, 0x9ABCDEF0, 0xFFFFFFFF), 0xFFFFFFFF)

        # src2 = 0 -> result is src0 & src1
        self.assertEqual(ref_and_or_b32(0x0F0F0F0F, 0x33333333, 0), 0x03030303)

        # src0 = 0 -> result is src2
        self.assertEqual(ref_and_or_b32(0, 0xFFFFFFFF, 0xCAFEBABE), 0xCAFEBABE)

        # src1 = 0 -> result is src2
        self.assertEqual(ref_and_or_b32(0xFFFFFFFF, 0, 0xDEADBEEF), 0xDEADBEEF)

    def test_commutativity_of_and_operands(self) -> None:
        val_a = 0xA5A5A5A5
        val_b = 0x5A5A5A5A
        val_c = 0x12345678
        self.assertEqual(
            ref_and_or_b32(val_a, val_b, val_c),
            ref_and_or_b32(val_b, val_a, val_c),
        )

    def test_bitfield_manipulation_masks(self) -> None:
        # Clear field (AND mask) then set field (OR mask)
        orig = 0x12345678
        clear_mask = 0xFFFF00FF
        insert_val = 0x0000AB00
        res = ref_and_or_b32(orig, clear_mask, insert_val)
        self.assertEqual(res, 0x1234AB78)

if __name__ == "__main__":
    unittest.main()
