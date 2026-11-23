from __future__ import annotations

"""Semantic reference oracle, AST definitions, interpreter models, and translator tests for s_add_i32.

AMD RDNA2 / GFX10 SOP2 (Scalar ALU) 32-bit Signed Integer Addition:
    s_add_i32 sdst, ssrc0, ssrc1

Hardware Semantics:
    sdst = (ssrc0 + ssrc1) & 0xFFFFFFFF
    SCC  = 1 if signed 32-bit integer overflow occurs, else 0

RDNA2 Condition Code (SCC) Behavior for s_add_i32:
    - ssrc0 and ssrc1 are interpreted as signed two's-complement 32-bit integers in [-2^31, 2^31 - 1].
    - Signed overflow occurs when:
        * both operands are positive (bit 31 == 0) and the sum is negative (bit 31 == 1)
        * both operands are negative (bit 31 == 1) and the sum is positive (bit 31 == 0)
    - Bitwise formulation: ((ssrc0 ^ sum) & (ssrc1 ^ sum)) >> 31 == 1
    - Note that unsigned wrap (carry-out) without signed overflow (such as (-1) + 1 = 0)
      clears SCC to 0 on s_add_i32, in direct contrast to s_add_u32 which sets SCC to 1.

Standard library Python only; zero third-party dependencies; zero game data.
"""

import dataclasses
import enum
import math
import os
import random
import struct
import sys
import unittest
from typing import List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

MASK32: int = 0xFFFFFFFF
INT32_MIN: int = -0x80000000       # -2147483648
INT32_MAX: int = 0x7FFFFFFF        # 2147483647
UINT32_INT_MIN: int = 0x80000000   # 2147483648 as unsigned

def to_u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit word [0, 0xFFFFFFFF]."""
    return val & MASK32


def to_s32(val: int) -> int:
    """Interpret 32-bit unsigned word as signed two's-complement integer in [-2^31, 2^31 - 1]."""
    u = val & MASK32
    return u - (1 << 32) if (u >> 31) else u


def ref_s_add_i32(s0: int, s1: int) -> tuple[int, int]:
    """Pure mathematical reference oracle for RDNA2 s_add_i32.

    Returns (dst_u32, scc_overflow).
    """
    v0 = to_s32(s0)
    v1 = to_s32(s1)
    tot = v0 + v1
    dst = tot & MASK32
    scc = 1 if (tot < INT32_MIN or tot > INT32_MAX) else 0
    return dst, scc


def ref_s_add_u32(s0: int, s1: int) -> tuple[int, int]:
    """Reference oracle for RDNA2 s_add_u32 for direct contrast (carry out of bit 31)."""
    u0 = to_u32(s0)
    u1 = to_u32(s1)
    tot = u0 + u1
    dst = tot & MASK32
    scc = 1 if tot > MASK32 else 0
    return dst, scc


class TestSAddI32ReferenceOracle(unittest.TestCase):
    """Mathematical reference oracle tests for RDNA2 s_add_i32."""

    def test_identity_element(self) -> None:
        for x in (0, 1, -1, INT32_MAX, INT32_MIN, 0x12345678, 0xEDCBA987):
            with self.subTest(x=x):
                dst, scc = ref_s_add_i32(x, 0)
                self.assertEqual(dst, to_u32(x))
                self.assertEqual(scc, 0)
                dst_rev, scc_rev = ref_s_add_i32(0, x)
                self.assertEqual(dst_rev, to_u32(x))
                self.assertEqual(scc_rev, 0)

    def test_commutativity(self) -> None:
        pairs = [
            (0, 0),
            (100, 200),
            (INT32_MAX, 1),
            (INT32_MIN, -1),
            (0xFFFFFFFF, 1),
            (0x7FFFFFFF, 0x7FFFFFFF),
            (0x80000000, 0x80000000),
            (0x12345678, 0x87654321),
        ]
        for a, b in pairs:
            with self.subTest(a=hex(to_u32(a)), b=hex(to_u32(b))):
                self.assertEqual(ref_s_add_i32(a, b), ref_s_add_i32(b, a))

    def test_positive_overflow(self) -> None:
        # INT32_MAX + 1 -> 0x80000000 (signed negative), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MAX, 1)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 1)

        # INT32_MAX + INT32_MAX -> 0xFFFFFFFE (-2), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MAX, INT32_MAX)
        self.assertEqual(dst, 0xFFFFFFFE)
        self.assertEqual(scc, 1)

        # 2^30 + 2^30 -> 2^31 (0x80000000), overflow = 1
        dst, scc = ref_s_add_i32(0x40000000, 0x40000000)
        self.assertEqual(dst, UINT32_INT_MIN)
        self.assertEqual(scc, 1)

    def test_negative_overflow(self) -> None:
        # INT32_MIN + (-1) -> 0x7FFFFFFF (positive), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MIN, -1)
        self.assertEqual(dst, INT32_MAX)
        self.assertEqual(scc, 1)

        # INT32_MIN + INT32_MIN -> 0x00000000 (0), overflow = 1
        dst, scc = ref_s_add_i32(INT32_MIN, INT32_MIN)
        self.assertEqual(dst, 0)
        self.assertEqual(scc, 1)

        # -2^30 + (-2^30 - 1) -> 0x7FFFFFFF, overflow = 1
        dst, scc = ref_s_add_i32(-0x40000000, -0x40000001)
        self.assertEqual(dst, INT32_MAX)
        self.assertEqual(scc, 1)

    def test_unsigned_wrap_without_signed_overflow(self) -> None:
        # (-1) + 1 = 0: unsigned carry-out occurs, but signed -1 + 1 = 0 (no signed overflow)
        dst_i, scc_i = ref_s_add_i32(0xFFFFFFFF, 1)
        self.assertEqual(dst_i, 0)
        self.assertEqual(scc_i, 0)

        # Contrast with s_add_u32 which sets SCC on unsigned carry:
        dst_u, scc_u = ref_s_add_u32(0xFFFFFFFF, 1)
        self.assertEqual(dst_u, 0)
        self.assertEqual(scc_u, 1)

        # (-1) + (-1) = -2: unsigned wrap occurs, signed -1 + -1 = -2 (no signed overflow)
        dst_i2, scc_i2 = ref_s_add_i32(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(dst_i2, 0xFFFFFFFE)
        self.assertEqual(scc_i2, 0)

        dst_u2, scc_u2 = ref_s_add_u32(0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(dst_u2, 0xFFFFFFFE)
        self.assertEqual(scc_u2, 1)

    def test_opposite_signs_no_overflow(self) -> None:
        # Opposite signs can never overflow 32-bit signed range
        dst, scc = ref_s_add_i32(INT32_MIN, INT32_MAX)
        self.assertEqual(dst, 0xFFFFFFFF)
        self.assertEqual(scc, 0)

        dst, scc = ref_s_add_i32(0x12345678, to_u32(-0x12345678))
        self.assertEqual(dst, 0)
        self.assertEqual(scc, 0)

class OperandKind(enum.Enum):
    SGPR = "sgpr"
    VGPR = "vgpr"
    LITERAL = "literal"
    INLINE_INT = "inline_int"


@dataclasses.dataclass(frozen=True)
class Reg:
    kind: OperandKind
    index: int

    def __post_init__(self) -> None:
        if self.kind == OperandKind.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of range [0, 105]")
        if self.kind == OperandKind.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of range [0, 255]")

    def format_asm(self) -> str:
        prefix = "s" if self.kind == OperandKind.SGPR else "v"
        return f"{prefix}{self.index}"


@dataclasses.dataclass(frozen=True)
class Imm:
    value: int
    is_inline: bool = False

    def __post_init__(self) -> None:
        if self.is_inline and not (-16 <= self.value <= 64):
            raise ValueError(f"Inline integer {self.value} out of range [-16, 64]")

    def format_asm(self) -> str:
        if self.is_inline:
            return str(self.value)
        return hex(to_u32(self.value))


class Opcode(enum.Enum):
    S_ADD_I32 = "s_add_i32"
    S_ADD_U32 = "s_add_u32"
    S_SUB_I32 = "s_sub_i32"
    S_MOV_B32 = "s_mov_b32"
    S_CSELECT_B32 = "s_cselect_b32"


@dataclasses.dataclass(frozen=True)
class SOP2Inst:
    opcode: Opcode
    sdst: Reg
    ssrc0: Reg | Imm
    ssrc1: Reg | Imm

    def __post_init__(self) -> None:
        if self.sdst.kind != OperandKind.SGPR:
            raise ValueError("sdst must be an SGPR")
        lit_count = 0
        if isinstance(self.ssrc0, Imm) and not self.ssrc0.is_inline:
            lit_count += 1
        if isinstance(self.ssrc1, Imm) and not self.ssrc1.is_inline:
            lit_count += 1
        if lit_count > 1:
            raise ValueError("SOP2 cannot encode more than one 32-bit literal constant")

    def format_asm(self) -> str:
        return f"{self.opcode.value} {self.sdst.format_asm()}, {self.ssrc0.format_asm()}, {self.ssrc1.format_asm()}"


class TestSAddI32AST(unittest.TestCase):
    """Unit tests for symbolic AST representation and serialization."""

    def test_reg_creation_and_formatting(self) -> None:
        r0 = Reg(OperandKind.SGPR, 0)
        r105 = Reg(OperandKind.SGPR, 105)
        v5 = Reg(OperandKind.VGPR, 5)
        self.assertEqual(r0.format_asm(), "s0")
        self.assertEqual(r105.format_asm(), "s105")
        self.assertEqual(v5.format_asm(), "v5")

    def test_imm_creation_and_formatting(self) -> None:
        imm_inline = Imm(1, is_inline=True)
        imm_inline_neg = Imm(-16, is_inline=True)
        imm_lit = Imm(0x12345678, is_inline=False)
        self.assertEqual(imm_inline.format_asm(), "1")
        self.assertEqual(imm_inline_neg.format_asm(), "-16")
        self.assertEqual(imm_lit.format_asm(), "0x12345678")

    def test_instruction_formatting(self) -> None:
        inst_rr = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Reg(OperandKind.SGPR, 2),
        )
        self.assertEqual(inst_rr.format_asm(), "s_add_i32 s0, s1, s2")

        inst_ri = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Imm(42, is_inline=True),
        )
        self.assertEqual(inst_ri.format_asm(), "s_add_i32 s0, s1, 42")

        inst_rl = SOP2Inst(
            opcode=Opcode.S_ADD_I32,
            sdst=Reg(OperandKind.SGPR, 0),
            ssrc0=Reg(OperandKind.SGPR, 1),
            ssrc1=Imm(0x7FFFFFFF, is_inline=False),
        )
        self.assertEqual(inst_rl.format_asm(), "s_add_i32 s0, s1, 0x7fffffff")

    def test_ast_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            Reg(OperandKind.SGPR, 106)
        with self.assertRaises(ValueError):
            Reg(OperandKind.SGPR, -1)
        with self.assertRaises(ValueError):
            Imm(65, is_inline=True)
        with self.assertRaises(ValueError):
            Imm(-17, is_inline=True)
        with self.assertRaises(ValueError):
            SOP2Inst(
                opcode=Opcode.S_ADD_I32,
                sdst=Reg(OperandKind.VGPR, 0),
                ssrc0=Reg(OperandKind.SGPR, 1),
                ssrc1=Reg(OperandKind.SGPR, 2),
            )
        with self.assertRaises(ValueError):
            SOP2Inst(
                opcode=Opcode.S_ADD_I32,
                sdst=Reg(OperandKind.SGPR, 0),
                ssrc0=Imm(0x1000, is_inline=False),
                ssrc1=Imm(0x2000, is_inline=False),
            )

if __name__ == "__main__":
    unittest.main()
