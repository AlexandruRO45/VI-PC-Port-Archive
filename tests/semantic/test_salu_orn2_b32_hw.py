"""Semantic tests for RDNA2 s_orn2_b32 opcode with hardware translator execution.

RDNA2 Instruction Set Architecture, SOP2:
    s_orn2_b32 D, S0, S1

Semantic operation:
    D = (S0 | (~S1 & 0xFFFFFFFF)) & 0xFFFFFFFF
    SCC = (D != 0)
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


def ref_orn2_b32(s0: int, s1: int) -> tuple[int, int]:
    """Bit-accurate reference oracle for RDNA2 s_orn2_b32."""
    s0_u32 = s0 & MASK32
    s1_u32 = s1 & MASK32
    d = (s0_u32 | (~s1_u32 & MASK32)) & MASK32
    scc = 1 if d != 0 else 0
    return d, scc


def run_hw_orn2_b32(s0: int, s1: int) -> tuple[int, int]:
    """Execute s_orn2_b32 through rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"s_mov_b32 s2, {s0 & MASK32:#x}",
        f"s_mov_b32 s3, {s1 & MASK32:#x}",
        "s_orn2_b32 s4, s2, s3",
        "s_cselect_b32 s5, 1, 0",
        "v_mov_b32 v0, s4",
        "v_mov_b32 v1, s5",
        "exp mrt0 v0, v1, v0, v1 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_orn2_b32_hw", lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


class TestSOrn2B32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_zeros_and_ones(self):
        # s0=0, s1=0 -> ~s1 = 0xFFFFFFFF -> d = 0xFFFFFFFF, scc = 1
        d, scc = ref_orn2_b32(0, 0)
        self.assertEqual(d, MASK32)
        self.assertEqual(scc, 1)

        # s0=0, s1=0xFFFFFFFF -> ~s1 = 0 -> d = 0, scc = 0 (the unique zero result)
        d, scc = ref_orn2_b32(0, MASK32)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)

        # s0=0xFFFFFFFF, s1=0xFFFFFFFF -> d = 0xFFFFFFFF, scc = 1
        d, scc = ref_orn2_b32(MASK32, MASK32)
        self.assertEqual(d, MASK32)
        self.assertEqual(scc, 1)

    def test_reference_alternating_patterns(self):
        s0 = 0xAAAAAAAA
        s1 = 0x55555555
        # ~s1 is 0xAAAAAAAA. 0xAAAAAAAA | 0xAAAAAAAA = 0xAAAAAAAA
        d, scc = ref_orn2_b32(s0, s1)
        self.assertEqual(d, 0xAAAAAAAA)
        self.assertEqual(scc, 1)

        # Inverted: s0=0x55555555, s1=0x55555555 -> ~s1 = 0xAAAAAAAA -> d = 0xFFFFFFFF
        d, scc = ref_orn2_b32(0x55555555, 0x55555555)
        self.assertEqual(d, MASK32)
        self.assertEqual(scc, 1)

    def test_translator_hw_orn2_b32(self):
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        cases = [
            (0, 0),
            (0, MASK32),
            (MASK32, MASK32),
            (0x12345678, 0x87654321),
            (0xAAAAAAAA, 0x55555555),
        ]
        for s0, s1 in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = run_hw_orn2_b32(s0, s1)
                exp_d, exp_scc = ref_orn2_b32(s0, s1)
                self.assertEqual(d, exp_d)
                self.assertEqual(scc, exp_scc)


class RegType(enum.Enum):
    SGPR = "sgpr"
    IMM = "imm"


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    val: int = 0

    def format(self) -> str:
        if self.reg_type == RegType.SGPR:
            return f"s{self.index}"
        return f"{self.val:#x}"


@dataclasses.dataclass(frozen=True)
class SOrn2B32Inst:
    dst: Operand
    src0: Operand
    src1: Operand

    def emit_asm(self) -> str:
        return f"s_orn2_b32 {self.dst.format()}, {self.src0.format()}, {self.src1.format()}"


class TestSOrn2B32AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self):
        inst = SOrn2B32Inst(
            dst=Operand(RegType.SGPR, 4),
            src0=Operand(RegType.SGPR, 2),
            src1=Operand(RegType.SGPR, 3),
        )
        self.assertEqual(inst.emit_asm(), "s_orn2_b32 s4, s2, s3")

    def test_ast_with_literal(self):
        inst = SOrn2B32Inst(
            dst=Operand(RegType.SGPR, 4),
            src0=Operand(RegType.SGPR, 2),
            src1=Operand(RegType.IMM, 0, 0x1234),
        )
        self.assertEqual(inst.emit_asm(), "s_orn2_b32 s4, s2, 0x1234")

    def test_ast_immutability(self):
        inst = SOrn2B32Inst(
            dst=Operand(RegType.SGPR, 0),
            src0=Operand(RegType.SGPR, 1),
            src1=Operand(RegType.SGPR, 2),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.dst = Operand(RegType.SGPR, 5)  # type: ignore


if __name__ == '__main__':
    unittest.main()
