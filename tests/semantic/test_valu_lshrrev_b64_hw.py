"""Semantic tests for RDNA2 v_lshrrev_b64 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_lshrrev_b64 dst[0:1], src0, src1[0:1]
    dst = (src1 & 0xFFFFFFFFFFFFFFFF) >> (src0 & 63)
where src0 is the shift count (masked to 6 bits, range 0..63),
src1 is the 64-bit source operand in a register pair,
and dst is the 64-bit logical right-shifted result in a register pair with zero-fill.
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
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_lshrrev_b64(shift_cnt: int, val64: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_lshrrev_b64."""
    sh = shift_cnt & 63
    return ((val64 & MASK64) >> sh) & MASK64


def run_hw_lshrrev_b64(shift_cnt: int, val64: int, inline_shift: bool = False) -> int:
    """Execute v_lshrrev_b64 through the rdna2spirv translator and spirv_mini_interp."""
    val_lo = val64 & MASK32
    val_hi = (val64 >> 32) & MASK32
    lines = [
        f"v_mov_b32 v2, {val_lo:#x}",
        f"v_mov_b32 v3, {val_hi:#x}",
    ]
    if inline_shift:
        lines.append(f"v_lshrrev_b64 v[0:1], {shift_cnt & 63}, v[2:3]")
    else:
        lines.append(f"v_mov_b32 v4, {shift_cnt & MASK32:#x}")
        lines.append("v_lshrrev_b64 v[0:1], v4, v[2:3]")
    lines.extend([
        "exp mrt0 v0, v1, v0, v1 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines("ps_lshr64_hw", lines)
    words = support.mrt0_words(spv)
    return (words[0] & MASK32) | ((words[1] & MASK32) << 32)


class TestVLshrrevB64Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_shifts(self):
        val = 0x8899AABBCCDDEEFF
        self.assertEqual(ref_lshrrev_b64(0, val), val)
        self.assertEqual(ref_lshrrev_b64(8, val), 0x008899AABBCCDDEE)
        self.assertEqual(ref_lshrrev_b64(32, val), 0x000000008899AABB)
        self.assertEqual(ref_lshrrev_b64(63, val), 1)

    def test_reference_shift_masking_6_bits(self):
        val = 0xFEDCBA9876543210
        # 64 & 63 == 0 -> shift 0
        self.assertEqual(ref_lshrrev_b64(64, val), val)
        # 65 & 63 == 1 -> shift 1
        self.assertEqual(ref_lshrrev_b64(65, val), val >> 1)
        # 0xFFFFFFFF & 63 == 63 -> shift 63
        self.assertEqual(ref_lshrrev_b64(0xFFFFFFFF, val), val >> 63)

    def test_translator_hw_shifts(self):
        val = 0xABCDEF0112345678
        shifts = [0, 1, 8, 16, 31, 32, 33, 48, 63, 64, 65, 0x100]
        for sh in shifts:
            with self.subTest(sh=sh):
                hw_res = run_hw_lshrrev_b64(sh, val)
                ref_res = ref_lshrrev_b64(sh, val)
                self.assertEqual(hw_res, ref_res)

    def test_translator_hw_inline_shift(self):
        val = 0xFFFFFFFFFFFFFFFF
        for sh in (0, 1, 16, 32, 63):
            with self.subTest(sh=sh):
                hw_res = run_hw_lshrrev_b64(sh, val, inline_shift=True)
                ref_res = ref_lshrrev_b64(sh, val)
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
    size: int = 1

    def __post_init__(self):
        if self.reg_type == RegType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {self.index}")
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range [0, 105]: {self.index}")
        if self.size not in (1, 2):
            raise ValueError(f"Invalid operand size {self.size}, must be 1 or 2")


@dataclasses.dataclass(frozen=True)
class VLshrrevB64Instruction:
    dst: Operand
    shift: Operand
    src: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR or self.dst.size != 2:
            raise TypeError("Destination must be a 64-bit VGPR pair (size=2)")
        if self.src.size != 2:
            raise TypeError("Source operand must be a 64-bit register pair or constant (size=2)")


class TestVLshrrevB64AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_valid(self):
        inst = VLshrrevB64Instruction(
            dst=Operand(RegType.VGPR, 0, size=2),
            shift=Operand(RegType.VGPR, 4, size=1),
            src=Operand(RegType.VGPR, 2, size=2),
        )
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.dst.size, 2)
        self.assertEqual(inst.shift.index, 4)
        self.assertEqual(inst.src.index, 2)
        self.assertEqual(inst.src.size, 2)

    def test_ast_validation_rejects_single_word_dst(self):
        with self.assertRaises(TypeError):
            VLshrrevB64Instruction(
                dst=Operand(RegType.VGPR, 0, size=1),
                shift=Operand(RegType.VGPR, 4, size=1),
                src=Operand(RegType.VGPR, 2, size=2),
            )

    def test_ast_validation_rejects_sgpr_dst(self):
        with self.assertRaises(TypeError):
            VLshrrevB64Instruction(
                dst=Operand(RegType.SGPR, 0, size=2),
                shift=Operand(RegType.VGPR, 4, size=1),
                src=Operand(RegType.VGPR, 2, size=2),
            )

    def test_ast_validation_rejects_out_of_range_index(self):
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, 256, size=2)
        with self.assertRaises(ValueError):
            Operand(RegType.SGPR, 106, size=2)
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, -1, size=2)


class ScalarRegisterFile:
    def __init__(self):
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def read_operand32(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index] & MASK32
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        elif op.reg_type == RegType.IMM:
            return op.value & MASK32
        raise ValueError(f"Unknown reg_type: {op.reg_type}")

    def read_operand64(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            lo = self.vgprs[op.index] & MASK32
            hi = self.vgprs[op.index + 1] & MASK32
            return (hi << 32) | lo
        elif op.reg_type == RegType.SGPR:
            lo = self.sgprs[op.index] & MASK32
            hi = self.sgprs[op.index + 1] & MASK32
            return (hi << 32) | lo
        elif op.reg_type == RegType.IMM:
            return op.value & MASK64
        raise ValueError(f"Unknown reg_type: {op.reg_type}")

    def write_vgpr64(self, idx: int, value64: int) -> None:
        if not (0 <= idx <= 254):
            raise ValueError(f"VGPR 64-bit pair index out of range: {idx}")
        self.vgprs[idx] = value64 & MASK32
        self.vgprs[idx + 1] = (value64 >> 32) & MASK32


class ScalarVALUInterpreter:
    @staticmethod
    def execute(rf: ScalarRegisterFile, inst: VLshrrevB64Instruction) -> int:
        sh = rf.read_operand32(inst.shift)
        val = rf.read_operand64(inst.src)
        res = ref_lshrrev_b64(sh, val)
        rf.write_vgpr64(inst.dst.index, res)
        return res


class TestVLshrrevB64ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter and execution model."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_scalar_execution_all_vgpr(self):
        val = 0xABCDEF0112345678
        self.rf.write_vgpr64(2, val)
        self.rf.vgprs[4] = 8
        inst = VLshrrevB64Instruction(
            dst=Operand(RegType.VGPR, 0, size=2),
            shift=Operand(RegType.VGPR, 4, size=1),
            src=Operand(RegType.VGPR, 2, size=2),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        expected = val >> 8
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[0], expected & MASK32)
        self.assertEqual(self.rf.vgprs[1], (expected >> 32) & MASK32)

    def test_scalar_execution_sgpr_and_imm(self):
        val = 0x8000000000000000
        self.rf.sgprs[2] = val & MASK32
        self.rf.sgprs[3] = (val >> 32) & MASK32
        inst = VLshrrevB64Instruction(
            dst=Operand(RegType.VGPR, 4, size=2),
            shift=Operand(RegType.IMM, 0, value=1, size=1),
            src=Operand(RegType.SGPR, 2, size=2),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        expected = 0x4000000000000000
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[4], expected & MASK32)
        self.assertEqual(self.rf.vgprs[5], (expected >> 32) & MASK32)


class SIMDState:
    def __init__(self, num_lanes: int = 32, exec_mask: int = 0xFFFFFFFF):
        if num_lanes not in (32, 64):
            raise ValueError(f"Unsupported lane count {num_lanes}, must be 32 or 64")
        self.num_lanes = num_lanes
        self.exec_mask = exec_mask
        self.vgprs = [[0] * num_lanes for _ in range(256)]
        self.sgprs = [0] * 106

    def read_lane32(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index][lane] & MASK32
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK32
        elif op.reg_type == RegType.IMM:
            return op.value & MASK32
        raise ValueError(f"Unknown reg_type: {op.reg_type}")

    def read_lane64(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            lo = self.vgprs[op.index][lane] & MASK32
            hi = self.vgprs[op.index + 1][lane] & MASK32
            return (hi << 32) | lo
        elif op.reg_type == RegType.SGPR:
            lo = self.sgprs[op.index] & MASK32
            hi = self.sgprs[op.index + 1] & MASK32
            return (hi << 32) | lo
        elif op.reg_type == RegType.IMM:
            return op.value & MASK64
        raise ValueError(f"Unknown reg_type: {op.reg_type}")


class SIMDInterpreter:
    @staticmethod
    def execute(state: SIMDState, inst: VLshrrevB64Instruction) -> list[int]:
        results = []
        for lane in range(state.num_lanes):
            lane_active = (state.exec_mask >> lane) & 1
            if lane_active:
                sh = state.read_lane32(inst.shift, lane)
                val = state.read_lane64(inst.src, lane)
                res = ref_lshrrev_b64(sh, val)
                state.vgprs[inst.dst.index][lane] = res & MASK32
                state.vgprs[inst.dst.index + 1][lane] = (res >> 32) & MASK32
                results.append(res)
            else:
                lo = state.vgprs[inst.dst.index][lane] & MASK32
                hi = state.vgprs[inst.dst.index + 1][lane] & MASK32
                results.append((hi << 32) | lo)
        return results


class TestVLshrrevB64SIMDDivergence(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_wave32_full_exec(self):
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        val = 0x8000000000000000
        for i in range(32):
            state.vgprs[2][i] = val & MASK32
            state.vgprs[3][i] = (val >> 32) & MASK32
            state.vgprs[4][i] = i
        inst = VLshrrevB64Instruction(
            dst=Operand(RegType.VGPR, 0, size=2),
            shift=Operand(RegType.VGPR, 4, size=1),
            src=Operand(RegType.VGPR, 2, size=2),
        )
        res = SIMDInterpreter.execute(state, inst)
        for i in range(32):
            expected = ref_lshrrev_b64(i, val)
            self.assertEqual(res[i], expected)
            self.assertEqual(state.vgprs[0][i], expected & MASK32)
            self.assertEqual(state.vgprs[1][i], (expected >> 32) & MASK32)

    def test_simd_wave32_divergent_mask(self):
        mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=mask)
        val = 0xFFFFFFFFFFFFFFFF
        for i in range(32):
            state.vgprs[0][i] = 0xDEADBEEF
            state.vgprs[1][i] = 0xCAFEBABE
            state.vgprs[2][i] = val & MASK32
            state.vgprs[3][i] = (val >> 32) & MASK32
            state.vgprs[4][i] = 4
        inst = VLshrrevB64Instruction(
            dst=Operand(RegType.VGPR, 0, size=2),
            shift=Operand(RegType.VGPR, 4, size=1),
            src=Operand(RegType.VGPR, 2, size=2),
        )
        res = SIMDInterpreter.execute(state, inst)
        expected_active = ref_lshrrev_b64(4, val)
        expected_inactive = (0xCAFEBABE << 32) | 0xDEADBEEF
        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(res[i], expected_active)
            else:
                self.assertEqual(res[i], expected_inactive)

    def test_simd_wave64_partial_exec(self):
        mask = 0x00000000FFFFFFFF
        state = SIMDState(num_lanes=64, exec_mask=mask)
        val = 0x123456789ABCDEF0
        for i in range(64):
            state.vgprs[0][i] = 0x11111111
            state.vgprs[1][i] = 0x22222222
            state.vgprs[2][i] = val & MASK32
            state.vgprs[3][i] = (val >> 32) & MASK32
            state.vgprs[4][i] = 8
        inst = VLshrrevB64Instruction(
            dst=Operand(RegType.VGPR, 0, size=2),
            shift=Operand(RegType.VGPR, 4, size=1),
            src=Operand(RegType.VGPR, 2, size=2),
        )
        res = SIMDInterpreter.execute(state, inst)
        expected_active = ref_lshrrev_b64(8, val)
        expected_inactive = (0x22222222 << 32) | 0x11111111
        for i in range(64):
            if i < 32:
                self.assertEqual(res[i], expected_active)
            else:
                self.assertEqual(res[i], expected_inactive)


class TestVLshrrevB64RegisterAliasing(unittest.TestCase):
    """Register aliasing hazards and in-place operand preservation."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_dst_aliases_src(self):
        val = 0xABCDEF0112345678
        self.rf.write_vgpr64(2, val)
        self.rf.vgprs[4] = 8
        inst = VLshrrevB64Instruction(
            dst=Operand(RegType.VGPR, 2, size=2),
            shift=Operand(RegType.VGPR, 4, size=1),
            src=Operand(RegType.VGPR, 2, size=2),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        expected = val >> 8
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[2], expected & MASK32)
        self.assertEqual(self.rf.vgprs[3], (expected >> 32) & MASK32)

    def test_dst_aliases_shift(self):
        val = 0x8899AABBCCDDEEFF
        self.rf.write_vgpr64(4, val)
        self.rf.vgprs[2] = 16
        inst = VLshrrevB64Instruction(
            dst=Operand(RegType.VGPR, 2, size=2),
            shift=Operand(RegType.VGPR, 2, size=1),
            src=Operand(RegType.VGPR, 4, size=2),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        expected = val >> 16
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.vgprs[2], expected & MASK32)
        self.assertEqual(self.rf.vgprs[3], (expected >> 32) & MASK32)

    def test_hw_translator_aliasing(self):
        val = 0xABCDEF0112345678
        lines = [
            f"v_mov_b32 v0, {val & MASK32:#x}",
            f"v_mov_b32 v1, {(val >> 32) & MASK32:#x}",
            "v_mov_b32 v2, 8",
            "v_lshrrev_b64 v[0:1], v2, v[0:1]",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_lshr64_alias", lines)
        words = support.mrt0_words(spv)
        hw_res = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
        self.assertEqual(hw_res, val >> 8)


@dataclasses.dataclass(frozen=True)
class Lshr64StatusFlags:
    is_zero: bool
    high_bit: int
    lost_bits: bool
    shift_count: int


def compute_status_flags(shift_cnt: int, val64: int) -> Lshr64StatusFlags:
    sh = shift_cnt & 63
    res = ref_lshrrev_b64(shift_cnt, val64)
    lost_mask = (1 << sh) - 1 if sh > 0 else 0
    return Lshr64StatusFlags(
        is_zero=(res == 0),
        high_bit=(res >> 63) & 1,
        lost_bits=((val64 & lost_mask) != 0),
        shift_count=sh,
    )


class TestVLshrrevB64ConditionCodes(unittest.TestCase):
    """Condition codes, status flags, and classification."""

    def test_zero_result_flags(self):
        flags = compute_status_flags(32, 0x00000000FFFFFFFF)
        self.assertTrue(flags.is_zero)
        self.assertEqual(flags.high_bit, 0)
        self.assertTrue(flags.lost_bits)

    def test_high_bit_zero_fill(self):
        flags = compute_status_flags(1, 0xFFFFFFFFFFFFFFFF)
        self.assertFalse(flags.is_zero)
        # Logical right shift always zero-fills high bit when shift > 0
        self.assertEqual(flags.high_bit, 0)
        self.assertTrue(flags.lost_bits)

    def test_no_lost_bits(self):
        flags = compute_status_flags(8, 0x123456789ABCDE00)
        self.assertFalse(flags.is_zero)
        self.assertFalse(flags.lost_bits)
        self.assertEqual(flags.shift_count, 8)


if __name__ == "__main__":
    unittest.main()
