"""Semantic tests for RDNA2 v_pk_max_u16 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_pk_max_u16 dst, src0, src1 (VOP3P)
Computes packed 16-bit unsigned maximum:
    dst.u16[0] = max(src0.u16[op_sel[0]], src1.u16[op_sel[1]])
    dst.u16[1] = max(src0.u16[op_sel_hi[0]], src1.u16[op_sel_hi[1]])
Default modifiers:
    op_sel: [0, 0]
    op_sel_hi: [1, 1]
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

try:
    import support  # noqa: E402
except Exception:
    support = None

MASK32 = 0xFFFFFFFF
MASK16 = 0xFFFF


def P(lo: int, hi: int) -> int:
    """Pack two 16-bit integers into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def ref_pk_max_u16(
    s0: int,
    s1: int,
    op_sel: tuple[int, int] = (0, 0),
    op_sel_hi: tuple[int, int] = (1, 1),
) -> int:
    """Bit-accurate reference oracle for RDNA2 v_pk_max_u16."""
    s0_halves = (s0 & MASK16, (s0 >> 16) & MASK16)
    s1_halves = (s1 & MASK16, (s1 >> 16) & MASK16)
    lo = max(s0_halves[op_sel[0]], s1_halves[op_sel[1]])
    hi = max(s0_halves[op_sel_hi[0]], s1_halves[op_sel_hi[1]])
    return P(lo, hi)


def run_hw_pk_max_u16(instruction: str, seeds: dict[str, int]) -> int:
    """Execute v_pk_max_u16 through rdna2spirv translator and spirv_mini_interp."""
    lines = ["v_mov_b32 v0, 0xdeadbeef"]
    for reg, val in sorted(seeds.items()):
        prefix = "s" if reg.startswith("s") else "v"
        lines.append(f"{prefix}_mov_b32 {reg}, {val & MASK32:#x}")
    lines.append(instruction)
    lines.append("exp mrt0 v0, v0, v0, v0 done vm")
    lines.append("s_endpgm")
    spv = support.translate_lines("ps_pk_max_u16_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVPkMaxU16Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_basic_lanes(self):
        cases = [
            # ((lo0, hi0), (lo1, hi1), (exp_lo, exp_hi))
            ((10, 20), (5, 50), (10, 50)),
            ((500, 100), (200, 600), (500, 600)),
            ((0, 0), (0, 0), (0, 0)),
            ((65535, 1), (1, 65535), (65535, 65535)),
        ]
        for (lo0, hi0), (lo1, hi1), (exp_lo, exp_hi) in cases:
            s0, s1 = P(lo0, hi0), P(lo1, hi1)
            expected = P(exp_lo, exp_hi)
            self.assertEqual(ref_pk_max_u16(s0, s1), expected)

    def test_reference_unsigned_extremes(self):
        # 0x8000 (32768) > 0x7FFF (32767) in unsigned comparison
        s0 = P(0x8000, 0x0000)
        s1 = P(0x7FFF, 0xFFFF)
        self.assertEqual(ref_pk_max_u16(s0, s1), P(0x8000, 0xFFFF))

    def test_reference_op_sel_modifiers(self):
        s0 = P(10, 20)
        s1 = P(30, 40)
        # op_sel:[1, 0] selects high half of s0 (20) and low half of s1 (30) for low result
        # op_sel_hi:[0, 1] selects low half of s0 (10) and high half of s1 (40) for high result
        got = ref_pk_max_u16(s0, s1, op_sel=(1, 0), op_sel_hi=(0, 1))
        self.assertEqual(got, P(max(20, 30), max(10, 40)))

    def test_translator_hw_pk_max_u16(self):
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        cases = [
            (P(10, 20), P(5, 50)),
            (P(0x8000, 1), P(0x7FFF, 0xFFFF)),
            (P(0, 0), P(100, 200)),
        ]
        for s0, s1 in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                got = run_hw_pk_max_u16("v_pk_max_u16 v0, v1, v2", {"v1": s0, "v2": s1})
                self.assertEqual(got, ref_pk_max_u16(s0, s1))


class RegType(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    IMM = "imm"


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    val: int = 0

    def format(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v{self.index}"
        elif self.reg_type == RegType.SGPR:
            return f"s{self.index}"
        return f"{self.val:#x}"


@dataclasses.dataclass(frozen=True)
class VPkMaxU16Inst:
    dst: Operand
    src0: Operand
    src1: Operand
    op_sel: tuple[int, int] = (0, 0)
    op_sel_hi: tuple[int, int] = (1, 1)

    def emit_asm(self) -> str:
        mods = ""
        if self.op_sel != (0, 0):
            mods += f" op_sel:[{self.op_sel[0]},{self.op_sel[1]}]"
        if self.op_sel_hi != (1, 1):
            mods += f" op_sel_hi:[{self.op_sel_hi[0]},{self.op_sel_hi[1]}]"
        return f"v_pk_max_u16 {self.dst.format()}, {self.src0.format()}, {self.src1.format()}{mods}"


class TestVPkMaxU16AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self):
        inst = VPkMaxU16Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
        )
        self.assertEqual(inst.emit_asm(), "v_pk_max_u16 v0, v1, v2")

    def test_ast_with_modifiers(self):
        inst = VPkMaxU16Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            op_sel=(1, 0),
            op_sel_hi=(0, 1),
        )
        self.assertEqual(inst.emit_asm(), "v_pk_max_u16 v0, v1, v2 op_sel:[1,0] op_sel_hi:[0,1]")

    def test_ast_immutability(self):
        inst = VPkMaxU16Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.dst = Operand(RegType.VGPR, 3)  # type: ignore


if __name__ == '__main__':
    unittest.main()
