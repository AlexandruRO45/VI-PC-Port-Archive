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


class SIMDState:
    def __init__(self, num_lanes: int = 32, exec_mask: int = 0xFFFFFFFF):
        if num_lanes not in (32, 64):
            raise ValueError(f"Unsupported lane count {num_lanes}, must be 32 or 64")
        self.num_lanes = num_lanes
        self.exec_mask = exec_mask
        self.vgprs = [[0] * num_lanes for _ in range(256)]
        self.sgprs = [0] * 106

    def read_lane(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index][lane] & MASK32
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        elif op.reg_type == RegType.IMM:
            return op.value & MASK32
        raise ValueError(f"Unknown reg_type: {op.reg_type}")


class SIMDInterpreter:
    @staticmethod
    def execute(state: SIMDState, inst: VAlignbyteB32Instruction) -> list[int]:
        results = []
        for lane in range(state.num_lanes):
            lane_active = (state.exec_mask >> lane) & 1
            if lane_active:
                s0 = state.read_lane(inst.src0, lane)
                s1 = state.read_lane(inst.src1, lane)
                s2 = state.read_lane(inst.src2, lane)
                res = ref_alignbyte(s0, s1, s2)
                state.vgprs[inst.dst.index][lane] = res
                results.append(res)
            else:
                results.append(state.vgprs[inst.dst.index][lane])
        return results


class TestVAlignbyteB32SIMDDivergence(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_wave32_full_exec(self):
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        for i in range(32):
            state.vgprs[1][i] = 0x12345678
            state.vgprs[2][i] = 0x9ABCDEF0
            state.vgprs[3][i] = i % 4
        inst = VAlignbyteB32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        for i in range(32):
            expected = ref_alignbyte(0x12345678, 0x9ABCDEF0, i % 4)
            self.assertEqual(res[i], expected)
            self.assertEqual(state.vgprs[0][i], expected)

    def test_simd_wave32_divergent_mask(self):
        mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=mask)
        for i in range(32):
            state.vgprs[0][i] = 0xDEADBEEF
            state.vgprs[1][i] = 0x8899AABB
            state.vgprs[2][i] = 0x11223344
            state.vgprs[3][i] = 1
        inst = VAlignbyteB32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        expected_active = ref_alignbyte(0x8899AABB, 0x11223344, 1)
        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(res[i], expected_active)
                self.assertEqual(state.vgprs[0][i], expected_active)
            else:
                self.assertEqual(res[i], 0xDEADBEEF)
                self.assertEqual(state.vgprs[0][i], 0xDEADBEEF)

    def test_simd_wave64_partial_exec(self):
        mask = 0x00000000FFFFFFFF
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for i in range(64):
            state.vgprs[0][i] = 0xBAADF00D
            state.vgprs[1][i] = 0xA1B2C3D4
            state.vgprs[2][i] = 0xE5F60718
            state.vgprs[3][i] = 2
        inst = VAlignbyteB32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        expected_active = ref_alignbyte(0xA1B2C3D4, 0xE5F60718, 2)
        for i in range(64):
            if i < 32:
                self.assertEqual(res[i], expected_active)
            else:
                self.assertEqual(res[i], 0xBAADF00D)


class TestVAlignbyteB32RegisterAliasing(unittest.TestCase):
    """Register aliasing hazards and in-place operand preservation."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_dst_aliases_src0(self):
        self.rf.write_vgpr(1, 0x8899AABB)
        self.rf.write_vgpr(2, 0x11223344)
        self.rf.write_vgpr(3, 1)
        inst = VAlignbyteB32Instruction(
            dst=Operand(RegType.VGPR, 1),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0xBB112233)
        self.assertEqual(self.rf.vgprs[1], 0xBB112233)

    def test_dst_aliases_src1(self):
        self.rf.write_vgpr(1, 0x8899AABB)
        self.rf.write_vgpr(2, 0x11223344)
        self.rf.write_vgpr(3, 2)
        inst = VAlignbyteB32Instruction(
            dst=Operand(RegType.VGPR, 2),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0xAABB1122)
        self.assertEqual(self.rf.vgprs[2], 0xAABB1122)

    def test_dst_aliases_src2(self):
        self.rf.write_vgpr(1, 0x8899AABB)
        self.rf.write_vgpr(2, 0x11223344)
        self.rf.write_vgpr(3, 3)
        inst = VAlignbyteB32Instruction(
            dst=Operand(RegType.VGPR, 3),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0x99AABB11)
        self.assertEqual(self.rf.vgprs[3], 0x99AABB11)

    def test_dst_aliases_all_operands(self):
        self.rf.write_vgpr(1, 0x11223344)
        inst = VAlignbyteB32Instruction(
            dst=Operand(RegType.VGPR, 1),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 1),
            src2=Operand(RegType.VGPR, 1),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0x11223344)
        self.assertEqual(self.rf.vgprs[1], 0x11223344)

    def test_hw_translator_aliasing(self):
        lines = [
            "v_mov_b32 v1, 0x8899aabb",
            "v_mov_b32 v2, 0x11223344",
            "v_mov_b32 v3, 1",
            "v_alignbyte_b32 v1, v1, v2, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_alignbyte_alias", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0xBB112233)


@dataclasses.dataclass(frozen=True)
class AlignbyteStatusFlags:
    is_zero: bool
    is_negative: bool
    shift_bytes: int
    high_bit: int
    all_from_src1: bool
    mixed_bytes: bool


def compute_status_flags(res: int, sel: int) -> AlignbyteStatusFlags:
    shift = sel & 3
    return AlignbyteStatusFlags(
        is_zero=(res == 0),
        is_negative=((res & 0x80000000) != 0),
        shift_bytes=shift,
        high_bit=(res >> 31) & 1,
        all_from_src1=(shift == 0),
        mixed_bytes=(shift in (1, 2, 3)),
    )


class TestVAlignbyteB32ConditionCodes(unittest.TestCase):
    """Condition codes, status flags, and classification."""

    def test_zero_result_flags(self):
        flags = compute_status_flags(0, sel=0)
        self.assertTrue(flags.is_zero)
        self.assertFalse(flags.is_negative)
        self.assertEqual(flags.shift_bytes, 0)
        self.assertTrue(flags.all_from_src1)
        self.assertFalse(flags.mixed_bytes)

    def test_negative_result_flags(self):
        flags = compute_status_flags(0x80000000, sel=2)
        self.assertFalse(flags.is_zero)
        self.assertTrue(flags.is_negative)
        self.assertEqual(flags.shift_bytes, 2)
        self.assertEqual(flags.high_bit, 1)
        self.assertFalse(flags.all_from_src1)
        self.assertTrue(flags.mixed_bytes)

    def test_mixed_shifts(self):
        for shift in range(4):
            flags = compute_status_flags(0x12345678, sel=shift)
            self.assertEqual(flags.shift_bytes, shift)
            if shift == 0:
                self.assertTrue(flags.all_from_src1)
                self.assertFalse(flags.mixed_bytes)
            else:
                self.assertFalse(flags.all_from_src1)
                self.assertTrue(flags.mixed_bytes)


if __name__ == "__main__":
    unittest.main()
