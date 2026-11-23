"""Semantic tests for RDNA2 v_subrev_co_u32 (vector reverse subtract with carry out).

RDNA2 VALU architecture contract:
- Opcode: v_subrev_co_u32 vdst, sdst, src0, src1 [clamp]
- Operation: D = src1 - src0 (unsigned 32-bit integer subtraction).
- Reverse operand order: computes src1 minus src0, contrasting with v_sub_co_u32 (src0 minus src1).
- Borrow out: borrow occurs if src1 < src0 (unsigned), writing 1 to sdst mask for active lanes.
- VOP3 clamp modifier: hardware saturates D to 0 upon borrow (underflow) while still reporting
  the real borrow bit in sdst. If no borrow occurs, D receives (src1 - src0) & 0xFFFFFFFF.
- EXEC masking: inactive lanes preserve their destination VGPR and contribute 0 to the sdst mask.
- Condition codes: VALU operations do not modify SCC.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support

# -----------------------------------------------------------------------------
# Commit 1: Reference Mathematical Oracle and Architectural Model
# -----------------------------------------------------------------------------
M32 = 0xFFFFFFFF


def ref_subrev_co_u32(src0: int, src1: int, clamp: bool = False) -> tuple[int, int]:
    """Pure mathematical oracle matching RDNA2 hardware semantics for v_subrev_co_u32.

    Returns (dst, borrow) where:
    - dst is (src1 - src0) & 0xFFFFFFFF (or 0 if clamp and borrow occurred)
    - borrow is 1 if (src1 & 0xFFFFFFFF) < (src0 & 0xFFFFFFFF) else 0
    """
    s0 = src0 & M32
    s1 = src1 & M32
    diff = (s1 - s0) & M32
    borrow = 1 if s1 < s0 else 0
    dst = 0 if (clamp and borrow) else diff
    return dst, borrow


class TestSubrevCoU32Oracle(unittest.TestCase):
    def test_basic_subtraction(self) -> None:
        dst, borrow = ref_subrev_co_u32(10, 25)
        self.assertEqual(dst, 15)
        self.assertEqual(borrow, 0)

    def test_borrow_underflow(self) -> None:
        dst, borrow = ref_subrev_co_u32(25, 10)
        self.assertEqual(dst, 0xFFFFFFF1)
        self.assertEqual(borrow, 1)

    def test_zero_identity(self) -> None:
        dst, borrow = ref_subrev_co_u32(0, 0)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 0)

        dst, borrow = ref_subrev_co_u32(12345, 12345)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 0)

    def test_boundary_opposites(self) -> None:
        # src1 = 0, src0 = 0xFFFFFFFF -> 0 - 0xFFFFFFFF = 1 with borrow
        dst, borrow = ref_subrev_co_u32(0xFFFFFFFF, 0)
        self.assertEqual(dst, 1)
        self.assertEqual(borrow, 1)

        # src1 = 0xFFFFFFFF, src0 = 0 -> 0xFFFFFFFF - 0 = 0xFFFFFFFF without borrow
        dst, borrow = ref_subrev_co_u32(0, 0xFFFFFFFF)
        self.assertEqual(dst, 0xFFFFFFFF)
        self.assertEqual(borrow, 0)

    def test_clamp_saturation(self) -> None:
        # Underflow with clamp saturates dst to 0, borrow remains 1
        dst, borrow = ref_subrev_co_u32(50, 20, clamp=True)
        self.assertEqual(dst, 0)
        self.assertEqual(borrow, 1)

        # Normal subtraction with clamp remains diff, borrow is 0
        dst, borrow = ref_subrev_co_u32(20, 50, clamp=True)
        self.assertEqual(dst, 30)
        self.assertEqual(borrow, 0)

    def test_subtraction_reversibility_asymmetry(self) -> None:
        # v_subrev_co_u32 computes src1 - src0, which is anti-symmetric to src0 - src1
        d1, b1 = ref_subrev_co_u32(100, 40)
        self.assertEqual(d1, (40 - 100) & M32)
        self.assertEqual(b1, 1)

# -----------------------------------------------------------------------------
# Commit 2: Symbolic AST Opcode Definitions and Typed Expressions
# -----------------------------------------------------------------------------
class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    VCC = auto()
    SPECIAL = auto()
    IMM = auto()


@dataclass(frozen=True)
class TypedOperand:
    kind: OperandKind
    index: int = 0
    name: str = ""
    value: int = 0

    @classmethod
    def vgpr(cls, idx: int) -> TypedOperand:
        return cls(kind=OperandKind.VGPR, index=idx, name=f"v{idx}")

    @classmethod
    def sgpr(cls, idx: int) -> TypedOperand:
        return cls(kind=OperandKind.SGPR, index=idx, name=f"s{idx}")

    @classmethod
    def vcc_lo(cls) -> TypedOperand:
        return cls(kind=OperandKind.VCC, name="vcc_lo")

    @classmethod
    def null(cls) -> TypedOperand:
        return cls(kind=OperandKind.SPECIAL, name="null")

    @classmethod
    def imm(cls, val: int) -> TypedOperand:
        return cls(kind=OperandKind.IMM, value=val, name=str(val))


@dataclass(frozen=True)
class VSubrevCoU32AST:
    vdst: TypedOperand
    sdst: TypedOperand
    src0: TypedOperand
    src1: TypedOperand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.vdst.kind != OperandKind.VGPR:
            raise TypeError(f"vdst must be VGPR, got {self.vdst.kind}")
        if self.sdst.kind not in (OperandKind.SGPR, OperandKind.VCC, OperandKind.SPECIAL):
            raise TypeError(f"sdst must be SGPR, VCC, or null, got {self.sdst.kind}")

    def to_assembly(self) -> str:
        mod = " clamp" if self.clamp else ""
        return f"v_subrev_co_u32 {self.vdst.name}, {self.sdst.name}, {self.src0.name}, {self.src1.name}{mod}"


class TestSubrevCoU32AST(unittest.TestCase):
    def test_ast_construction_valid(self) -> None:
        ast = VSubrevCoU32AST(
            vdst=TypedOperand.vgpr(0),
            sdst=TypedOperand.vcc_lo(),
            src0=TypedOperand.vgpr(1),
            src1=TypedOperand.vgpr(2),
        )
        self.assertEqual(ast.to_assembly(), "v_subrev_co_u32 v0, vcc_lo, v1, v2")

    def test_ast_with_clamp(self) -> None:
        ast = VSubrevCoU32AST(
            vdst=TypedOperand.vgpr(5),
            sdst=TypedOperand.sgpr(2),
            src0=TypedOperand.vgpr(3),
            src1=TypedOperand.imm(100),
            clamp=True,
        )
        self.assertEqual(ast.to_assembly(), "v_subrev_co_u32 v5, s2, v3, 100 clamp")

    def test_ast_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            VSubrevCoU32AST(
                vdst=TypedOperand.sgpr(0),  # Invalid: vdst must be VGPR
                sdst=TypedOperand.vcc_lo(),
                src0=TypedOperand.vgpr(1),
                src1=TypedOperand.vgpr(2),
            )
