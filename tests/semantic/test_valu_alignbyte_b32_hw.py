"""Semantic tests for RDNA2 v_alignbyte_b32 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_alignbyte_b32 dst, src0, src1, src2
    dst = ({src0, src1} >> (8 * (src2 & 3))) & 0xFFFFFFFF
where src1 provides the low 32 bits and src0 provides the high 32 bits of the 64-bit concatenated value.
src2 is the byte shift selector, masked to its lowest 2 bits (shifts of 0, 8, 16, or 24 bits).
"""
from __future__ import annotations

import dataclasses
import enum
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

import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def ref_alignbyte(src0: int, src1: int, sel: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_alignbyte_b32.
    dst = ({src0, src1} >> (8 * (sel & 3))) & 0xFFFFFFFF
    """
    shift = (sel & 3) * 8
    concat = ((src0 & MASK32) << 32) | (src1 & MASK32)
    return (concat >> shift) & MASK32


def run_hw_alignbyte(src0: int, src1: int, sel: int, inline: bool = False) -> int:
    """Execute v_alignbyte_b32 through the rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
    ]
    if inline:
        lines.append(f"v_alignbyte_b32 v0, v1, v2, {sel & 3}")
    else:
        lines.append(f"v_mov_b32 v3, {sel & MASK32:#x}")
        lines.append("v_alignbyte_b32 v0, v1, v2, v3")
    lines.extend([
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines("ps_alignbyte_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVAlignbyteB32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_manual_vectors(self):
        s0 = 0x8899AABB
        s1 = 0x11223344
        self.assertEqual(ref_alignbyte(s0, s1, 0), 0x11223344)
        self.assertEqual(ref_alignbyte(s0, s1, 1), 0xBB112233)
        self.assertEqual(ref_alignbyte(s0, s1, 2), 0xAABB1122)
        self.assertEqual(ref_alignbyte(s0, s1, 3), 0x99AABB11)
        self.assertEqual(ref_alignbyte(s0, s1, 4), 0x11223344)

    def test_translator_hw_every_byte_shift(self):
        test_pairs = [
            (0x8899AABB, 0x11223344),
            (0xFFFFFFFF, 0x00000000),
            (0x00000000, 0xFFFFFFFF),
            (0x12345678, 0x9ABCDEF0),
        ]
        for s0, s1 in test_pairs:
            for sel in range(4):
                with self.subTest(s0=hex(s0), s1=hex(s1), sel=sel):
                    self.assertEqual(run_hw_alignbyte(s0, s1, sel), ref_alignbyte(s0, s1, sel))

    def test_translator_hw_selector_masking(self):
        s0, s1 = 0xDEADBEEF, 0xCAFEBABE
        for sel in (0, 4, 8, 12, 0x100, 0xFFFFFFFC):
            with self.subTest(sel=hex(sel)):
                self.assertEqual(run_hw_alignbyte(s0, s1, sel), ref_alignbyte(s0, s1, sel))
                self.assertEqual(run_hw_alignbyte(s0, s1, sel), s1)

        for sel in (5, 6, 7, 0x101, 0xFFFFFFFF):
            with self.subTest(sel=hex(sel)):
                self.assertEqual(run_hw_alignbyte(s0, s1, sel), ref_alignbyte(s0, s1, sel))

    def test_translator_hw_inline_literal(self):
        s0, s1 = 0xA1B2C3D4, 0xE5F60718
        for sel in range(4):
            with self.subTest(sel=sel):
                self.assertEqual(run_hw_alignbyte(s0, s1, sel, inline=True), ref_alignbyte(s0, s1, sel))


class RegType(enum.Enum):
    VGPR = 1
    SGPR = 2
    IMM = 3


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    value: int = 0

    def __post_init__(self):
        if self.reg_type == RegType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")


@dataclasses.dataclass(frozen=True)
class VAlignbyteB32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")


class TestVAlignbyteB32AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_valid(self):
        inst = VAlignbyteB32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.SGPR, 2),
            src2=Operand(RegType.IMM, 0, value=3),
        )
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.src0.index, 1)
        self.assertEqual(inst.src1.index, 2)
        self.assertEqual(inst.src2.value, 3)

    def test_ast_validation_rejects_invalid_dst(self):
        with self.assertRaises(TypeError):
            VAlignbyteB32Instruction(
                dst=Operand(RegType.SGPR, 0),
                src0=Operand(RegType.VGPR, 1),
                src1=Operand(RegType.VGPR, 2),
                src2=Operand(RegType.VGPR, 3),
            )
        with self.assertRaises(TypeError):
            VAlignbyteB32Instruction(
                dst=Operand(RegType.IMM, 0, value=0),
                src0=Operand(RegType.VGPR, 1),
                src1=Operand(RegType.VGPR, 2),
                src2=Operand(RegType.VGPR, 3),
            )

    def test_ast_validation_rejects_out_of_range_index(self):
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, 256)
        with self.assertRaises(ValueError):
            Operand(RegType.SGPR, 106)
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, -1)


class ScalarRegisterFile:
    def __init__(self):
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def read_operand(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        elif op.reg_type == RegType.IMM:
            return op.value & MASK32
        raise ValueError(f"Unknown reg_type: {op.reg_type}")

    def write_vgpr(self, idx: int, value: int) -> None:
        if not (0 <= idx <= 255):
            raise ValueError(f"VGPR index out of range: {idx}")
        self.vgprs[idx] = value & MASK32


class ScalarVALUInterpreter:
    @staticmethod
    def execute(rf: ScalarRegisterFile, inst: VAlignbyteB32Instruction) -> int:
        s0 = rf.read_operand(inst.src0)
        s1 = rf.read_operand(inst.src1)
        s2 = rf.read_operand(inst.src2)
        res = ref_alignbyte(s0, s1, s2)
        rf.write_vgpr(inst.dst.index, res)
        return res


class TestVAlignbyteB32ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter and execution model."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_scalar_execution_all_vgpr(self):
        self.rf.write_vgpr(1, 0x11223344)
        self.rf.write_vgpr(2, 0x55667788)
        self.rf.write_vgpr(3, 2)
        inst = VAlignbyteB32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0x33445566)
        self.assertEqual(self.rf.vgprs[0], 0x33445566)

    def test_scalar_execution_sgpr_and_imm(self):
        self.rf.vgprs[1] = 0xAABBCCDD
        self.rf.sgprs[5] = 0xEEFF0011
        inst = VAlignbyteB32Instruction(
            dst=Operand(RegType.VGPR, 4),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.SGPR, 5),
            src2=Operand(RegType.IMM, 0, value=1),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0xDDEEFF00)
        self.assertEqual(self.rf.vgprs[4], 0xDDEEFF00)


if __name__ == "__main__":
    unittest.main()
