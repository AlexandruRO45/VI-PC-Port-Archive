"""Semantic tests for RDNA2 v_cvt_f32_u32 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_cvt_f32_u32 dst, src0 (VOP1/VOP3)
Converts a 32-bit unsigned integer to an IEEE 754 single-precision float (binary32).
Rounding follows round-to-nearest-even (RNE).
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


def ref_cvt_f32_u32(val: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_cvt_f32_u32: u32 to binary32 bits."""
    val &= MASK32
    f = float(val)
    return struct.unpack("<I", struct.pack("<f", f))[0]


def run_hw_cvt_f32_u32(val: int) -> int:
    """Execute v_cvt_f32_u32 through rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v1, {val & MASK32:#x}",
        "v_cvt_f32_u32 v0, v1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_cvt_f32_u32_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVCvtF32U32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_exact_integers(self):
        cases = [
            (0, 0x00000000),         # +0.0
            (1, 0x3F800000),         # 1.0
            (2, 0x40000000),         # 2.0
            (4, 0x40800000),         # 4.0
            (10, 0x41200000),        # 10.0
            (16777215, 0x4B7FFFFF),  # 2^24 - 1
            (16777216, 0x4B800000),  # 2^24
        ]
        for u, exp_bits in cases:
            with self.subTest(u=u):
                self.assertEqual(ref_cvt_f32_u32(u), exp_bits)

    def test_reference_rounding_mode_rne(self):
        # 16777217 (2^24 + 1): midway between 16777216 and 16777218 -> rounds to even 16777216.0
        self.assertEqual(ref_cvt_f32_u32(16777217), 0x4B800000)
        # 16777219 (2^24 + 3): midway between 16777218 and 16777220 -> rounds to even 16777220.0
        f_val = struct.unpack("<f", struct.pack("<I", ref_cvt_f32_u32(16777219)))[0]
        self.assertEqual(f_val, 16777220.0)
        # 0xFFFFFFFF (4294967295): rounds to 4294967296.0 (2^32) = 0x4F800000
        self.assertEqual(ref_cvt_f32_u32(0xFFFFFFFF), 0x4F800000)

    def test_translator_hw_cvt_f32_u32(self):
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        cases = [0, 1, 100, 16777216, 0x80000000, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                got = run_hw_cvt_f32_u32(val)
                self.assertEqual(got, ref_cvt_f32_u32(val))


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
class VCvtF32U32Inst:
    dst: Operand
    src0: Operand

    def emit_asm(self) -> str:
        return f"v_cvt_f32_u32 {self.dst.format()}, {self.src0.format()}"


class TestVCvtF32U32AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self):
        inst = VCvtF32U32Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
        )
        self.assertEqual(inst.emit_asm(), "v_cvt_f32_u32 v0, v1")

    def test_ast_with_literal(self):
        inst = VCvtF32U32Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.IMM, 0, 100),
        )
        self.assertEqual(inst.emit_asm(), "v_cvt_f32_u32 v0, 0x64")

    def test_ast_immutability(self):
        inst = VCvtF32U32Inst(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.dst = Operand(RegType.VGPR, 2)  # type: ignore


if __name__ == '__main__':
    unittest.main()
