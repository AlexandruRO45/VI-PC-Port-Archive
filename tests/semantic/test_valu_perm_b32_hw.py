"""Semantic tests for RDNA2 v_perm_b32 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_perm_b32 dst, src0, src1, src2
Source container is 64-bit packed {src0, src1}, where src1 provides low bytes 0..3
and src0 provides high bytes 4..7.
For each byte i (0..3) of dst:
    sel = (src2 >> (8 * i)) & 0xFF
    0..7: select byte sel from {src0, src1}
    8..11: sign extend 16-bit halfword (sel - 8): bit 15, 31, 47, or 63 replicated across the byte
    12: 0x00
    >= 13: 0xFF
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


def ref_perm_b32(s0: int, s1: int, sel: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_perm_b32."""
    s0 &= MASK32
    s1 &= MASK32
    sel &= MASK32
    packed = (s0 << 32) | s1
    out = 0
    for i in range(4):
        sb = (sel >> (8 * i)) & 0xFF
        if sb < 8:
            byte_val = (packed >> (8 * sb)) & 0xFF
        elif sb < 12:
            sign_bit = (packed >> (16 * (sb - 8) + 15)) & 1
            byte_val = 0xFF if sign_bit else 0x00
        elif sb == 12:
            byte_val = 0x00
        else:
            byte_val = 0xFF
        out |= byte_val << (8 * i)
    return out & MASK32


def run_hw_perm_b32(src0: int, src1: int, sel: int, inline: bool = False) -> int:
    """Execute v_perm_b32 through the rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
    ]
    if inline:
        lines.append(f"v_perm_b32 v0, v1, v2, {sel & MASK32:#x}")
    else:
        lines.append(f"v_mov_b32 v3, {sel & MASK32:#x}")
        lines.append("v_perm_b32 v0, v1, v2, v3")
    lines.extend([
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines("ps_perm_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVPermB32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_byte_selection(self):
        s0 = 0x07060504
        s1 = 0x03020100
        # Identity permute: bytes 0, 1, 2, 3
        self.assertEqual(ref_perm_b32(s0, s1, 0x03020100), 0x03020100)
        # Reverse bytes in src1
        self.assertEqual(ref_perm_b32(s0, s1, 0x00010203), 0x00010203)
        # Select all from high word src0
        self.assertEqual(ref_perm_b32(s0, s1, 0x07060504), 0x07060504)
        # Interleave src1 and src0
        self.assertEqual(ref_perm_b32(s0, s1, 0x06020501), 0x06020501)

    def test_reference_constants_and_sign_extension(self):
        s0 = 0x80007FFF
        s1 = 0x00018000
        # 12 -> 0x00, 13 -> 0xFF
        self.assertEqual(ref_perm_b32(s0, s1, 0x0D0C0D0C), 0xFF00FF00)
        # Sign extension:
        # halfword 0 (s1[15:0] = 0x8000, bit 15 is 1 -> 0xFF)
        # halfword 1 (s1[31:16] = 0x0001, bit 31 is 0 -> 0x00)
        # halfword 2 (s0[15:0] = 0x7FFF, bit 15 is 0 -> 0x00)
        # halfword 3 (s0[31:16] = 0x8000, bit 31 is 1 -> 0xFF)
        sel = 0x0B0A0908
        self.assertEqual(ref_perm_b32(s0, s1, sel), 0xFF0000FF)

    def test_translator_hw_byte_selection(self):
        s0 = 0xAABBCCDD
        s1 = 0x11223344
        selectors = [
            0x03020100,  # identity src1
            0x07060504,  # identity src0
            0x00010203,  # byte reversal
            0x04000501,  # mixed words
            0x0C0C0C0C,  # zero fill
            0x0D0D0D0D,  # 0xFF fill
            0x0B0A0908,  # sign extension
        ]
        for sel in selectors:
            with self.subTest(sel=hex(sel)):
                hw_res = run_hw_perm_b32(s0, s1, sel)
                ref_res = ref_perm_b32(s0, s1, sel)
                self.assertEqual(hw_res, ref_res)


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
class VPermB32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")


class TestVPermB32AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_valid(self):
        inst = VPermB32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.SGPR, 2),
            src2=Operand(RegType.IMM, 0, value=0x03020100),
        )
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.src0.index, 1)
        self.assertEqual(inst.src1.index, 2)
        self.assertEqual(inst.src2.value, 0x03020100)

    def test_ast_validation_rejects_invalid_dst(self):
        with self.assertRaises(TypeError):
            VPermB32Instruction(
                dst=Operand(RegType.SGPR, 0),
                src0=Operand(RegType.VGPR, 1),
                src1=Operand(RegType.VGPR, 2),
                src2=Operand(RegType.VGPR, 3),
            )
        with self.assertRaises(TypeError):
            VPermB32Instruction(
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
    def execute(rf: ScalarRegisterFile, inst: VPermB32Instruction) -> int:
        s0 = rf.read_operand(inst.src0)
        s1 = rf.read_operand(inst.src1)
        s2 = rf.read_operand(inst.src2)
        res = ref_perm_b32(s0, s1, s2)
        rf.write_vgpr(inst.dst.index, res)
        return res


class TestVPermB32ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter and execution model."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_scalar_execution_all_vgpr(self):
        self.rf.write_vgpr(1, 0x07060504)
        self.rf.write_vgpr(2, 0x03020100)
        self.rf.write_vgpr(3, 0x00010203)
        inst = VPermB32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0x00010203)
        self.assertEqual(self.rf.vgprs[0], 0x00010203)

    def test_scalar_execution_sgpr_and_imm(self):
        self.rf.vgprs[1] = 0x8899AABB
        self.rf.sgprs[5] = 0x11223344
        inst = VPermB32Instruction(
            dst=Operand(RegType.VGPR, 4),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.SGPR, 5),
            src2=Operand(RegType.IMM, 0, value=0x07060100),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        expected = ref_perm_b32(0x8899AABB, 0x11223344, 0x07060100)
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[4], expected)


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
    def execute(state: SIMDState, inst: VPermB32Instruction) -> list[int]:
        results = []
        for lane in range(state.num_lanes):
            lane_active = (state.exec_mask >> lane) & 1
            if lane_active:
                s0 = state.read_lane(inst.src0, lane)
                s1 = state.read_lane(inst.src1, lane)
                s2 = state.read_lane(inst.src2, lane)
                res = ref_perm_b32(s0, s1, s2)
                state.vgprs[inst.dst.index][lane] = res
                results.append(res)
            else:
                results.append(state.vgprs[inst.dst.index][lane])
        return results


class TestVPermB32SIMDDivergence(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_wave32_full_exec(self):
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        selectors = [0x03020100, 0x00010203, 0x07060504, 0x0C0C0C0C]
        for i in range(32):
            state.vgprs[1][i] = 0xAABBCCDD
            state.vgprs[2][i] = 0x11223344
            state.vgprs[3][i] = selectors[i % len(selectors)]
        inst = VPermB32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        for i in range(32):
            expected = ref_perm_b32(0xAABBCCDD, 0x11223344, selectors[i % len(selectors)])
            self.assertEqual(res[i], expected)
            self.assertEqual(state.vgprs[0][i], expected)

    def test_simd_wave32_divergent_mask(self):
        mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=mask)
        for i in range(32):
            state.vgprs[0][i] = 0xDEADBEEF
            state.vgprs[1][i] = 0x07060504
            state.vgprs[2][i] = 0x03020100
            state.vgprs[3][i] = 0x00010203
        inst = VPermB32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        expected_active = ref_perm_b32(0x07060504, 0x03020100, 0x00010203)
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
            state.vgprs[0][i] = 0xCAFEBABE
            state.vgprs[1][i] = 0xFFFFFFFF
            state.vgprs[2][i] = 0x00000000
            state.vgprs[3][i] = 0x0D0C0D0C
        inst = VPermB32Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        expected_active = ref_perm_b32(0xFFFFFFFF, 0x00000000, 0x0D0C0D0C)
        for i in range(64):
            if i < 32:
                self.assertEqual(res[i], expected_active)
            else:
                self.assertEqual(res[i], 0xCAFEBABE)


class TestVPermB32RegisterAliasing(unittest.TestCase):
    """Register aliasing hazards and in-place operand preservation."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_dst_aliases_src0(self):
        self.rf.write_vgpr(1, 0x07060504)
        self.rf.write_vgpr(2, 0x03020100)
        self.rf.write_vgpr(3, 0x00010203)
        inst = VPermB32Instruction(
            dst=Operand(RegType.VGPR, 1),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0x00010203)
        self.assertEqual(self.rf.vgprs[1], 0x00010203)

    def test_dst_aliases_src1(self):
        self.rf.write_vgpr(1, 0x07060504)
        self.rf.write_vgpr(2, 0x03020100)
        self.rf.write_vgpr(3, 0x07060504)
        inst = VPermB32Instruction(
            dst=Operand(RegType.VGPR, 2),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0x07060504)
        self.assertEqual(self.rf.vgprs[2], 0x07060504)

    def test_dst_aliases_src2(self):
        self.rf.write_vgpr(1, 0x07060504)
        self.rf.write_vgpr(2, 0x03020100)
        self.rf.write_vgpr(3, 0x0C0C0C0C)
        inst = VPermB32Instruction(
            dst=Operand(RegType.VGPR, 3),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0x00000000)
        self.assertEqual(self.rf.vgprs[3], 0x00000000)

    def test_dst_aliases_all_operands(self):
        self.rf.write_vgpr(1, 0x03020100)
        inst = VPermB32Instruction(
            dst=Operand(RegType.VGPR, 1),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 1),
            src2=Operand(RegType.VGPR, 1),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0x03020100)
        self.assertEqual(self.rf.vgprs[1], 0x03020100)

    def test_hw_translator_aliasing(self):
        lines = [
            "v_mov_b32 v1, 0x07060504",
            "v_mov_b32 v2, 0x03020100",
            "v_mov_b32 v3, 0x00010203",
            "v_perm_b32 v1, v1, v2, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_perm_alias", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x00010203)


if __name__ == "__main__":
    unittest.main()
