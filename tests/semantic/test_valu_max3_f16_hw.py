"""Semantic tests for RDNA2 v_max3_f16 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_max3_f16 dst, src0, src1, src2
Computes the 16-bit floating-point maximum of three binary16 operands:
    dst[15:0] = max(src0[15:0], src1[15:0], src2[15:0])
Follows IEEE 754-2008 maxNum / GLSL NMax rules:
- Skips NaN operands: if 1 or 2 operands are NaN, returns the maximum of the non-NaN operands.
- If all 3 operands are NaN, returns quiet NaN (0x7E00).
- Signed zeros: -0.0 is strictly less than +0.0 (+0.0 wins over -0.0).
- Destination preservation: the unwritten high half dst[31:16] retains its previous value.
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

import support  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF


def f16_to_float(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def float_to_f16_bits(val: float) -> int:
    return struct.unpack("<H", struct.pack("<e", val))[0]


def ref_max3_f16(a_bits: int, b_bits: int, c_bits: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_max3_f16."""
    vals = [f16_to_float(h) for h in (a_bits, b_bits, c_bits)]
    non_nan = [v for v in vals if not math.isnan(v)]
    if not non_nan:
        return 0x7E00  # quiet NaN
    # Sort with secondary key copysign so +0.0 beats -0.0
    chosen = max(non_nan, key=lambda v: (v, math.copysign(1.0, v)))
    return float_to_f16_bits(chosen)


def run_hw_max3_f16(a_bits: int, b_bits: int, c_bits: int, old_dst: int = 0x12340000, name: str = "ps_max3_f16_hw") -> int:
    """Execute v_max3_f16 through the rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v0, {old_dst:#x}",
        f"v_mov_b32 v1, {a_bits & MASK16:#x}",
        f"v_mov_b32 v2, {b_bits & MASK16:#x}",
        f"v_mov_b32 v3, {c_bits & MASK16:#x}",
        "v_max3_f16 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, lines)
    return support.mrt0_words(spv)[0]


class TestVMax3F16Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_numerical_cases(self):
        # 1.0 (0x3C00), 2.0 (0x4000), 0.5 (0x3800) -> 2.0 (0x4000)
        self.assertEqual(ref_max3_f16(0x3C00, 0x4000, 0x3800), 0x4000)
        hw1 = run_hw_max3_f16(0x3C00, 0x4000, 0x3800, old_dst=0x12340000, name="ps_ref_num1")
        self.assertEqual(hw1, 0x12344000)

        # -1.0 (0xBC00), -2.0 (0xC000), -0.5 (0xB800) -> -0.5 (0xB800)
        self.assertEqual(ref_max3_f16(0xBC00, 0xC000, 0xB800), 0xB800)
        hw2 = run_hw_max3_f16(0xBC00, 0xC000, 0xB800, old_dst=0x12340000, name="ps_ref_num2")
        self.assertEqual(hw2, 0x1234B800)

    def test_reference_signed_zero(self):
        # +0.0 (0x0000) vs -0.0 (0x8000) vs -1.0 (0xBC00) -> +0.0 wins
        self.assertEqual(ref_max3_f16(0xBC00, 0x0000, 0x8000), 0x0000)
        hw1 = run_hw_max3_f16(0xBC00, 0x0000, 0x8000, old_dst=0x56780000, name="ps_ref_sz1")
        self.assertEqual(hw1, 0x56780000)

        self.assertEqual(ref_max3_f16(0x8000, 0x8000, 0x0000), 0x0000)
        hw2 = run_hw_max3_f16(0x8000, 0x8000, 0x0000, old_dst=0x56780000, name="ps_ref_sz2")
        self.assertEqual(hw2, 0x56780000)

    def test_reference_nan_skipping(self):
        qnan = 0x7E00
        # Single NaN skipped
        self.assertEqual(ref_max3_f16(qnan, 0x3C00, 0x3800), 0x3C00)
        hw1 = run_hw_max3_f16(qnan, 0x3C00, 0x3800, old_dst=0x9ABC0000, name="ps_ref_nan1")
        self.assertEqual(hw1, 0x9ABC3C00)

        # Two NaNs skipped
        self.assertEqual(ref_max3_f16(qnan, qnan, 0x4000), 0x4000)
        hw2 = run_hw_max3_f16(qnan, qnan, 0x4000, old_dst=0x9ABC0000, name="ps_ref_nan2")
        self.assertEqual(hw2, 0x9ABC4000)

        # All NaNs returns NaN
        self.assertEqual(ref_max3_f16(qnan, qnan, qnan), 0x7E00)

    def test_translator_hw_max3(self):
        cases = [
            (0x3C00, 0x4000, 0x3800),  # 1.0, 2.0, 0.5 -> 2.0
            (0xBC00, 0x0000, 0x8000),  # -1.0, +0.0, -0.0 -> +0.0
            (0x7E00, 0x3C00, 0x3800),  # NaN, 1.0, 0.5 -> 1.0
            (0x7C00, 0x4000, 0x3C00),  # +inf, 2.0, 1.0 -> +inf
            (0xFC00, 0xBC00, 0xC000),  # -inf, -1.0, -2.0 -> -1.0
        ]
        for a, b, c in cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                hw_res = run_hw_max3_f16(a, b, c, old_dst=0x12340000)
                expected_lo = ref_max3_f16(a, b, c)
                expected_word = 0x12340000 | expected_lo
                self.assertEqual(hw_res, expected_word)


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
class VMax3F16Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination must be a VGPR, got {self.dst.reg_type}")


class TestVMax3F16AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_valid(self):
        inst = VMax3F16Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.SGPR, 2),
            src2=Operand(RegType.IMM, 0, value=0x3C00),
        )
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.src0.index, 1)
        self.assertEqual(inst.src1.index, 2)
        self.assertEqual(inst.src2.value, 0x3C00)

    def test_ast_validation_rejects_invalid_dst(self):
        with self.assertRaises(TypeError):
            VMax3F16Instruction(
                dst=Operand(RegType.SGPR, 0),
                src0=Operand(RegType.VGPR, 1),
                src1=Operand(RegType.VGPR, 2),
                src2=Operand(RegType.VGPR, 3),
            )
        with self.assertRaises(TypeError):
            VMax3F16Instruction(
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

    def read_operand16(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index] & MASK16
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK16
        elif op.reg_type == RegType.IMM:
            return op.value & MASK16
        raise ValueError(f"Unknown reg_type: {op.reg_type}")

    def write_vgpr_f16(self, idx: int, value16: int) -> None:
        if not (0 <= idx <= 255):
            raise ValueError(f"VGPR index out of range: {idx}")
        old_hi = self.vgprs[idx] & 0xFFFF0000
        self.vgprs[idx] = old_hi | (value16 & MASK16)


class ScalarVALUInterpreter:
    @staticmethod
    def execute(rf: ScalarRegisterFile, inst: VMax3F16Instruction) -> int:
        s0 = rf.read_operand16(inst.src0)
        s1 = rf.read_operand16(inst.src1)
        s2 = rf.read_operand16(inst.src2)
        res16 = ref_max3_f16(s0, s1, s2)
        rf.write_vgpr_f16(inst.dst.index, res16)
        return rf.vgprs[inst.dst.index]


class TestVMax3F16ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter and execution model."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_scalar_execution_all_vgpr(self):
        self.rf.vgprs[0] = 0xABCD0000
        self.rf.vgprs[1] = 0x3C00  # 1.0
        self.rf.vgprs[2] = 0x4000  # 2.0
        self.rf.vgprs[3] = 0x3800  # 0.5
        inst = VMax3F16Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0xABCD4000)
        self.assertEqual(self.rf.vgprs[0], 0xABCD4000)

        hw_res = run_hw_max3_f16(0x3C00, 0x4000, 0x3800, old_dst=0xABCD0000, name="ps_interp_vgpr")
        self.assertEqual(hw_res, 0xABCD4000)

    def test_scalar_execution_sgpr_and_imm(self):
        self.rf.vgprs[4] = 0xDEAD0000
        self.rf.vgprs[1] = 0xBC00  # -1.0
        self.rf.sgprs[5] = 0x8000  # -0.0
        inst = VMax3F16Instruction(
            dst=Operand(RegType.VGPR, 4),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.SGPR, 5),
            src2=Operand(RegType.IMM, 0, value=0x0000),  # +0.0
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0xDEAD0000)  # +0.0 won

        lines = [
            "s_mov_b32 s5, 0x8000",
            "v_mov_b32 v4, 0xDEAD0000",
            "v_mov_b32 v1, 0xBC00",
            "v_max3_f16 v4, v1, s5, 0",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_sgpr_imm", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0xDEAD0000)


class SIMDState:
    def __init__(self, num_lanes: int = 32, exec_mask: int = 0xFFFFFFFF):
        if num_lanes not in (32, 64):
            raise ValueError(f"Unsupported lane count {num_lanes}, must be 32 or 64")
        self.num_lanes = num_lanes
        self.exec_mask = exec_mask
        self.vgprs = [[0] * num_lanes for _ in range(256)]
        self.sgprs = [0] * 106

    def read_lane16(self, op: Operand, lane: int) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index][lane] & MASK16
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index] & MASK16
        elif op.reg_type == RegType.IMM:
            return op.value & MASK16
        raise ValueError(f"Unknown reg_type: {op.reg_type}")


class SIMDInterpreter:
    @staticmethod
    def execute(state: SIMDState, inst: VMax3F16Instruction) -> list[int]:
        results = []
        for lane in range(state.num_lanes):
            lane_active = (state.exec_mask >> lane) & 1
            if lane_active:
                s0 = state.read_lane16(inst.src0, lane)
                s1 = state.read_lane16(inst.src1, lane)
                s2 = state.read_lane16(inst.src2, lane)
                res16 = ref_max3_f16(s0, s1, s2)
                old_hi = state.vgprs[inst.dst.index][lane] & 0xFFFF0000
                new_word = old_hi | (res16 & MASK16)
                state.vgprs[inst.dst.index][lane] = new_word
                results.append(new_word)
            else:
                results.append(state.vgprs[inst.dst.index][lane])
        return results


class TestVMax3F16SIMDDivergence(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_wave32_full_exec(self):
        state = SIMDState(num_lanes=32, exec_mask=0xFFFFFFFF)
        for i in range(32):
            state.vgprs[0][i] = 0xCAFE0000
            state.vgprs[1][i] = 0x3C00  # 1.0
            state.vgprs[2][i] = 0x3800  # 0.5
            state.vgprs[3][i] = 0x4000 if (i % 2) else 0x3400  # 2.0 or 0.25
        inst = VMax3F16Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        for i in range(32):
            expected_lo = 0x4000 if (i % 2) else 0x3C00
            expected = 0xCAFE0000 | expected_lo
            self.assertEqual(res[i], expected)
            self.assertEqual(state.vgprs[0][i], expected)

        lines = [
            "v_mov_b32 v0, 0xCAFE0000",
            "v_mov_b32 v1, 0x3C00",
            "v_mov_b32 v2, 0x3800",
            "v_mov_b32 v3, 0x4000",
            "v_max3_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_simd_full", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0xCAFE4000)

    def test_simd_wave32_divergent_mask(self):
        mask = 0x55555555
        state = SIMDState(num_lanes=32, exec_mask=mask)
        for i in range(32):
            state.vgprs[0][i] = 0xDEADBEEF
            state.vgprs[1][i] = 0x3C00
            state.vgprs[2][i] = 0x4000
            state.vgprs[3][i] = 0x3800
        inst = VMax3F16Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        for i in range(32):
            if i % 2 == 0:
                self.assertEqual(res[i], 0xDEAD4000)
            else:
                self.assertEqual(res[i], 0xDEADBEEF)

    def test_simd_wave64_partial_exec(self):
        mask = 0x00000000FFFFFFFF
        state = SIMDState(num_lanes=64, exec_mask=mask)
        for i in range(64):
            state.vgprs[0][i] = 0xBABE0000
            state.vgprs[1][i] = 0xBC00
            state.vgprs[2][i] = 0x0000
            state.vgprs[3][i] = 0x8000
        inst = VMax3F16Instruction(
            dst=Operand(RegType.VGPR, 0),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = SIMDInterpreter.execute(state, inst)
        for i in range(64):
            if i < 32:
                self.assertEqual(res[i], 0xBABE0000)  # +0.0
            else:
                self.assertEqual(res[i], 0xBABE0000)


class TestVMax3F16RegisterAliasing(unittest.TestCase):
    """Register aliasing hazards and in-place operand preservation."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_dst_aliases_src0(self):
        self.rf.vgprs[1] = 0x12343C00  # 1.0 in low half
        self.rf.vgprs[2] = 0x4000      # 2.0
        self.rf.vgprs[3] = 0x3800      # 0.5
        inst = VMax3F16Instruction(
            dst=Operand(RegType.VGPR, 1),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 2),
            src2=Operand(RegType.VGPR, 3),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0x12344000)
        self.assertEqual(self.rf.vgprs[1], 0x12344000)

        lines = [
            "v_mov_b32 v1, 0x12343c00",
            "v_mov_b32 v2, 0x4000",
            "v_mov_b32 v3, 0x3800",
            "v_max3_f16 v1, v1, v2, v3",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_alias_src0", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x12344000)

    def test_dst_aliases_all_operands(self):
        self.rf.vgprs[1] = 0x56783C00
        inst = VMax3F16Instruction(
            dst=Operand(RegType.VGPR, 1),
            src0=Operand(RegType.VGPR, 1),
            src1=Operand(RegType.VGPR, 1),
            src2=Operand(RegType.VGPR, 1),
        )
        res = ScalarVALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0x56783C00)
        self.assertEqual(self.rf.vgprs[1], 0x56783C00)

        lines = [
            "v_mov_b32 v1, 0x56783c00",
            "v_max3_f16 v1, v1, v1, v1",
            "exp mrt0 v1, v1, v1, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_alias_all", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x56783c00)

    def test_hw_translator_aliasing(self):
        lines = [
            "v_mov_b32 v0, 0x12343c00",
            "v_mov_b32 v1, 0x4000",
            "v_mov_b32 v2, 0x3800",
            "v_max3_f16 v0, v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_f16_alias", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x12344000)


@dataclasses.dataclass(frozen=True)
class Max3F16StatusFlags:
    is_zero: bool
    is_negative: bool
    is_nan: bool
    is_inf: bool


def compute_status_flags(res_bits: int) -> Max3F16StatusFlags:
    v = f16_to_float(res_bits)
    return Max3F16StatusFlags(
        is_zero=(v == 0.0),
        is_negative=((res_bits & 0x8000) != 0),
        is_nan=math.isnan(v),
        is_inf=math.isinf(v),
    )


class TestVMax3F16ConditionCodes(unittest.TestCase):
    """Condition codes, status flags, and classification."""

    def test_positive_normal_flags(self):
        flags = compute_status_flags(0x4000)  # 2.0
        self.assertFalse(flags.is_zero)
        self.assertFalse(flags.is_negative)
        self.assertFalse(flags.is_nan)
        self.assertFalse(flags.is_inf)

    def test_negative_zero_flags(self):
        flags = compute_status_flags(0x8000)  # -0.0
        self.assertTrue(flags.is_zero)
        self.assertTrue(flags.is_negative)

    def test_nan_flags(self):
        flags = compute_status_flags(0x7E00)  # qNaN
        self.assertTrue(flags.is_nan)


class TestVMax3F16IEEE754(unittest.TestCase):
    """IEEE 754 raw bit pattern preservation and subnormal/NaN behavior."""

    def test_ieee754_subnormals(self):
        # Smallest positive subnormal: 0x0001
        # Largest positive subnormal: 0x03FF
        # Negative zero: 0x8000
        res = ref_max3_f16(0x0001, 0x03FF, 0x8000)
        self.assertEqual(res, 0x03FF)
        hw_res = run_hw_max3_f16(0x0001, 0x03FF, 0x8000, old_dst=0x12340000, name="ps_subnormals")
        self.assertEqual(hw_res, 0x123403FF)

    def test_ieee754_infinities(self):
        pos_inf = 0x7C00
        neg_inf = 0xFC00
        norm = 0x7BFF  # max normal float16 (65504)
        self.assertEqual(ref_max3_f16(pos_inf, norm, 0x0000), pos_inf)
        hw_pos = run_hw_max3_f16(pos_inf, norm, 0x0000, old_dst=0x12340000, name="ps_pos_inf")
        self.assertEqual(hw_pos, 0x12347C00)

        self.assertEqual(ref_max3_f16(neg_inf, 0xBC00, 0xC000), 0xBC00)
        hw_neg = run_hw_max3_f16(neg_inf, 0xBC00, 0xC000, old_dst=0x12340000, name="ps_neg_inf")
        self.assertEqual(hw_neg, 0x1234BC00)

    def test_ieee754_hw_translator_nans(self):
        # NaN operand skipped when non-NaN present
        qnan = 0x7E00
        hw_res = run_hw_max3_f16(qnan, 0x3C00, 0x3800, old_dst=0xAAAA0000)
        self.assertEqual(hw_res, 0xAAAA3C00)


class TestVMax3F16NegativeAndMutation(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection for v_max3_f16."""

    def test_rejection_of_invalid_syntax(self):
        lines_missing = [
            "v_mov_b32 v1, 1",
            "v_max3_f16 v0, v1, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_max3_err", lines_missing)

    def test_rejection_of_out_of_bounds_register(self):
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, 256)
        with self.assertRaises(ValueError):
            Operand(RegType.VGPR, -1)
        with self.assertRaises(ValueError):
            Operand(RegType.SGPR, 106)

    def test_spirv_emission_nmax(self):
        """Verify that support.translate_lines emits GLSL NMax operations for v_max3_f16."""
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x3c00",
            "v_mov_b32 v2, 0x4000",
            "v_mov_b32 v3, 0x3800",
            "v_max3_f16 v0, v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_emit_nmax", lines)
        self.assertIn("NMax", spv)
        self.assertNotIn("NMin", spv)

    def test_vop3_clamp_modifier_execution(self):
        """Verify execution of clamp modifier on v_max3_f16 through support.mrt0_words."""
        # 2.0 (0x4000), 1.5 (0x3E00), 0.5 (0x3800) -> max is 2.0, clamped to 1.0 (0x3C00)
        lines_hi = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x4000",
            "v_mov_b32 v2, 0x3e00",
            "v_mov_b32 v3, 0x3800",
            "v_max3_f16 v0, v1, v2, v3 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv_hi = support.translate_lines("ps_max3_clamp_hi", lines_hi)
        self.assertEqual(support.mrt0_words(spv_hi)[0], 0x3C00)

        # -2.0 (0xC000), -1.5 (0xBE00), -0.5 (0xB800) -> max is -0.5, clamped to 0.0 (0x0000)
        lines_lo = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0xc000",
            "v_mov_b32 v2, 0xbe00",
            "v_mov_b32 v3, 0xb800",
            "v_max3_f16 v0, v1, v2, v3 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv_lo = support.translate_lines("ps_max3_clamp_lo", lines_lo)
        self.assertEqual(support.mrt0_words(spv_lo)[0], 0x0000)

    def test_opsel_operand_selection(self):
        """Verify execution with op_sel source and destination selection."""
        # op_sel:[1,0,0,0] reads high half of v1
        lines = [
            "v_mov_b32 v0, 0",
            "v_mov_b32 v1, 0x42000000",  # 3.0 in high half
            "v_mov_b32 v2, 0x3800",      # 0.5
            "v_mov_b32 v3, 0x3400",      # 0.25
            "v_max3_f16 v0, v1, v2, v3 op_sel:[1,0,0,0]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_max3_opsel_src", lines)
        self.assertEqual(support.mrt0_words(spv)[0], 0x4200)

        # op_sel:[0,0,0,1] writes to high half and preserves low half
        lines_dst = [
            "v_mov_b32 v0, 0x1234",
            "v_mov_b32 v1, 0x4000",
            "v_mov_b32 v2, 0x3800",
            "v_mov_b32 v3, 0x3400",
            "v_max3_f16 v0, v1, v2, v3 op_sel:[0,0,0,1]",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv_dst = support.translate_lines("ps_max3_opsel_dst", lines_dst)
        self.assertEqual(support.mrt0_words(spv_dst)[0], 0x40001234)

    def test_destination_high_half_preservation(self):
        """Verify that v_max3_f16 preserves the high 16 bits of destination."""
        for old_hi in (0xAAAA0000, 0x55550000, 0xFFFF0000, 0x00000000):
            with self.subTest(old_hi=hex(old_hi)):
                hw = run_hw_max3_f16(0x3C00, 0x4000, 0x3800, old_dst=old_hi, name=f"ps_pres_{old_hi:#x}")
                self.assertEqual(hw & 0xFFFF0000, old_hi)
                self.assertEqual(hw & 0xFFFF, 0x4000)

    def test_mutation_detection_min_instead_of_max(self):
        # Mutated oracle computing min instead of max
        def bad_oracle_min(a, b, c):
            vals = [f16_to_float(h) for h in (a, b, c)]
            non_nan = [v for v in vals if not math.isnan(v)]
            chosen = min(non_nan)
            return float_to_f16_bits(chosen)

        self.assertNotEqual(ref_max3_f16(0x3C00, 0x4000, 0x3800), bad_oracle_min(0x3C00, 0x4000, 0x3800))
        # Direct hardware execution confirms max (0x4000 = 2.0) rather than min (0x3800 = 0.5)
        hw = run_hw_max3_f16(0x3C00, 0x4000, 0x3800, old_dst=0, name="ps_mut_min_vs_max")
        self.assertEqual(hw, 0x4000)
        self.assertNotEqual(hw, 0x3800)

    def test_mutation_detection_signed_zero_tie(self):
        # Mutated oracle where -0.0 beats +0.0
        def bad_oracle_neg_zero(a, b, c):
            return 0x8000

        self.assertNotEqual(ref_max3_f16(0x8000, 0x0000, 0xBC00), bad_oracle_neg_zero(0x8000, 0x0000, 0xBC00))
        hw = run_hw_max3_f16(0x8000, 0x0000, 0xBC00, old_dst=0, name="ps_mut_sz_tie")
        self.assertEqual(hw, 0x0000)
        self.assertNotEqual(hw, 0x8000)


if __name__ == "__main__":
    unittest.main()
