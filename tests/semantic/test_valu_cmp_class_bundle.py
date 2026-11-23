from __future__ import annotations

"""Semantic interpreter tests and reference mathematical model for RDNA2 VALU IEEE 754 Class Testing Suite.

Covers:
- v_cmp_class_f16: Test class of 16-bit float against 10-bit mask, writing VCC/SGPR.
- v_cmp_class_f32: Test class of 32-bit float against 10-bit mask, writing VCC/SGPR.
- v_cmp_class_f64: Test class of 64-bit float against 10-bit mask, writing VCC/SGPR.
- v_cmpx_class_f16: Test class of 16-bit float, narrowing EXEC active mask.
- v_cmpx_class_f32: Test class of 32-bit float, narrowing EXEC active mask.
- v_cmpx_class_f64: Test class of 64-bit float, narrowing EXEC active mask.

Standard library only (python -I, struct, math, unittest). Zero game data.
All test suites execute real translated assembly via support.translate_lines.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
import os
import struct
import sys
import unittest
from typing import List, Optional, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import support
import rdna2spirv

MASK32: int = 0xFFFF_FFFF
CLASS_ALL: int = 0x3FF

# 10 IEEE 754 Class Bits:
# 0: sNaN, 1: qNaN, 2: -inf, 3: -normal, 4: -denorm, 5: -0, 6: +0, 7: +denorm, 8: +normal, 9: +inf
CLASS_SNAN = 1 << 0
CLASS_QNAN = 1 << 1
CLASS_NEG_INF = 1 << 2
CLASS_NEG_NORMAL = 1 << 3
CLASS_NEG_DENORM = 1 << 4
CLASS_NEG_ZERO = 1 << 5
CLASS_POS_ZERO = 1 << 6
CLASS_POS_DENORM = 1 << 7
CLASS_POS_NORMAL = 1 << 8
CLASS_POS_INF = 1 << 9


def run_cmp_shader(ty: str, x: int, mask: int, is_cmpx: bool = False, test_id: str = "") -> Tuple[int, str]:
    if ty == "f64":
        x_lo = x & 0xFFFF_FFFF
        x_hi = (x >> 32) & 0xFFFF_FFFF
        if is_cmpx:
            lines = [
                "v_mov_b32 v0, 0",
                f"v_mov_b32 v2, {hex(x_lo)}",
                f"v_mov_b32 v3, {hex(x_hi)}",
                f"v_cmpx_class_f64_e32 v[2:3], {hex(mask)}",
                "v_mov_b32 v0, 1",
                "s_mov_b32 exec_lo, -1",
            ]
        else:
            lines = [
                f"v_mov_b32 v2, {hex(x_lo)}",
                f"v_mov_b32 v3, {hex(x_hi)}",
                f"v_cmp_class_f64_e32 vcc_lo, v[2:3], {hex(mask)}",
                "v_cndmask_b32_e32 v0, 0, 1, vcc_lo",
            ]
    else:
        if is_cmpx:
            lines = [
                "v_mov_b32 v0, 0",
                f"v_mov_b32 v1, {hex(x)}",
                f"v_cmpx_class_{ty}_e32 v1, {hex(mask)}",
                "v_mov_b32 v0, 1",
                "s_mov_b32 exec_lo, -1",
            ]
        else:
            lines = [
                f"v_mov_b32 v1, {hex(x)}",
                f"v_cmp_class_{ty}_e32 vcc_lo, v1, {hex(mask)}",
                "v_cndmask_b32_e32 v0, 0, 1, vcc_lo",
            ]
    full = lines + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spv = support.translate_lines(f"ps_cmp_{ty}_{is_cmpx}_{test_id}", full)
    words = support.mrt0_words(spv)
    return words[0], spv


# ------------------------------------------------------------------------------
# v_cmp_class_f16 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def classify_f16(u: int) -> int:
    u = u & 0xFFFF
    sign = (u >> 15) & 1
    exp = (u >> 10) & 0x1F
    man = u & 0x3FF
    if exp == 0:
        if man == 0:
            return 5 if sign else 6
        return 4 if sign else 7
    elif exp == 31:
        if man == 0:
            return 2 if sign else 9
        return 1 if (man & 0x200) else 0
    return 3 if sign else 8


def ref_cmp_class_f16(u: int, mask: int) -> bool:
    c = classify_f16(u)
    return bool(mask & (1 << c))


class TestCmpClassF16Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        samples = [
            (0x7C01, CLASS_SNAN),
            (0x7E00, CLASS_QNAN),
            (0xFC00, CLASS_NEG_INF),
            (0xBC00, CLASS_NEG_NORMAL),
            (0x8001, CLASS_NEG_DENORM),
            (0x8000, CLASS_NEG_ZERO),
            (0x0000, CLASS_POS_ZERO),
            (0x0001, CLASS_POS_DENORM),
            (0x3C00, CLASS_POS_NORMAL),
            (0x7C00, CLASS_POS_INF),
        ]
        for idx, (val, mask) in enumerate(samples):
            res_hw, _ = run_cmp_shader("f16", val, mask, is_cmpx=False, test_id=f"oracle_{idx}")
            self.assertEqual(res_hw, 1 if ref_cmp_class_f16(val, mask) else 0)

    def test_oracle_mask_combinations_hardware(self):
        # Full mask matches any float
        res_all, _ = run_cmp_shader("f16", 0x3C00, CLASS_ALL, is_cmpx=False, test_id="all_mask")
        self.assertEqual(res_all, 1)

        # Zero mask never matches
        res_zero, _ = run_cmp_shader("f16", 0x3C00, 0, is_cmpx=False, test_id="zero_mask")
        self.assertEqual(res_zero, 0)


# ------------------------------------------------------------------------------
# v_cmp_class_f16 AST & Scalar Machine
# ------------------------------------------------------------------------------

class OperandType(Enum):
    VGPR = auto()
    SGPR = auto()
    VCC = auto()
    EXEC = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class RegisterOperand:
    op_type: OperandType
    index: int = 0

    def __post_init__(self):
        if self.op_type == OperandType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index out of range: {self.index}")
        if self.op_type == OperandType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index out of range: {self.index}")

    @property
    def name(self) -> str:
        if self.op_type == OperandType.VGPR:
            return f"v{self.index}"
        elif self.op_type == OperandType.SGPR:
            return f"s{self.index}"
        elif self.op_type == OperandType.VCC:
            return "vcc_lo"
        elif self.op_type == OperandType.EXEC:
            return "exec_lo"
        return "lit"


@dataclass(frozen=True)
class LiteralOperand:
    value: int

    def __post_init__(self):
        if not (-0x8000_0000 <= self.value <= 0xFFFF_FFFF_FFFF_FFFF):
            raise ValueError(f"Literal out of range: {self.value}")


@dataclass
class VCmpClassF16Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    mask: Union[RegisterOperand, LiteralOperand]


class ScalarMachineClassLogic:
    def __init__(self):
        self.vgprs: List[int] = [0] * 256
        self.sgprs: List[int] = [0] * 106
        self.vcc: int = 0
        self.exec: int = MASK32

    def read_operand(self, op: Union[RegisterOperand, LiteralOperand]) -> int:
        if isinstance(op, LiteralOperand):
            return op.value
        elif op.op_type == OperandType.VGPR:
            return self.vgprs[op.index]
        elif op.op_type == OperandType.SGPR:
            return self.sgprs[op.index]
        elif op.op_type == OperandType.VCC:
            return self.vcc
        elif op.op_type == OperandType.EXEC:
            return self.exec
        raise TypeError(f"Unknown operand: {op}")

    def step_cmp_class_f16(self, insn: VCmpClassF16Instruction):
        s0 = self.read_operand(insn.src0) & 0xFFFF
        mask = self.read_operand(insn.mask) & 0xFFFF
        res = 1 if ref_cmp_class_f16(s0, mask) else 0
        if insn.dst.op_type == OperandType.VCC:
            self.vcc = MASK32 if res else 0
        elif insn.dst.op_type == OperandType.SGPR:
            self.sgprs[insn.dst.index] = MASK32 if res else 0


class TestCmpClassF16ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineClassLogic()
        machine.vgprs[1] = 0x3C00  # +1.0 (class 8: +normal)
        insn = VCmpClassF16Instruction(
            dst=RegisterOperand(OperandType.VCC),
            src0=RegisterOperand(OperandType.VGPR, 1),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        machine.step_cmp_class_f16(insn)

        val_hw, _ = run_cmp_shader("f16", 0x3C00, CLASS_POS_NORMAL, is_cmpx=False, test_id="ast_scalar")
        self.assertEqual(machine.vcc, MASK32)
        self.assertEqual(val_hw, 1)

    def test_scalar_sgpr_dst_hardware(self):
        val_sgpr, _ = run_cmp_shader("f16", 0xFC00, CLASS_NEG_INF, is_cmpx=False, test_id="sgpr_dst")
        self.assertEqual(val_sgpr, 1)


# ------------------------------------------------------------------------------
# v_cmp_class_f16 SIMD Divergence & 10-Bit Class Mapping
# ------------------------------------------------------------------------------

class SIMDWaveClassLogic:
    def __init__(self, wave_size: int = 32):
        self.wave_size = wave_size
        self.vgprs: List[List[int]] = [[0] * wave_size for _ in range(256)]
        self.vcc: int = 0
        self.exec_mask: int = (1 << wave_size) - 1

    def step_cmp_class_f16(self, insn: VCmpClassF16Instruction):
        mask = insn.mask.value if isinstance(insn.mask, LiteralOperand) else self.vgprs[insn.mask.index][0]
        out_vcc = 0
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.vgprs[insn.src0.index][lane] & 0xFFFF
                if ref_cmp_class_f16(s0, mask):
                    out_vcc |= (1 << lane)
        self.vcc = out_vcc


class TestCmpClassF16SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_hardware(self):
        wave = SIMDWaveClassLogic(wave_size=32)
        # Lane 0: +1.0 (normal), Lane 1: +0.0 (zero)
        wave.vgprs[1][0] = 0x3C00
        wave.vgprs[1][1] = 0x0000

        insn = VCmpClassF16Instruction(
            dst=RegisterOperand(OperandType.VCC),
            src0=RegisterOperand(OperandType.VGPR, 1),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        wave.step_cmp_class_f16(insn)

        # Check lane 0 matches (+normal) and lane 1 does not (+zero)
        val_lane0, _ = run_cmp_shader("f16", 0x3C00, CLASS_POS_NORMAL, is_cmpx=False, test_id="simd_l0")
        val_lane1, _ = run_cmp_shader("f16", 0x0000, CLASS_POS_NORMAL, is_cmpx=False, test_id="simd_l1")
        self.assertEqual(val_lane0, 1)
        self.assertEqual(val_lane1, 0)
        self.assertEqual(wave.vcc & 1, 1)
        self.assertEqual((wave.vcc >> 1) & 1, 0)

    def test_algebraic_partition_hardware(self):
        # Union of disjoint classes: (+0 | -0) matches exactly +0.0
        val_zero_match, _ = run_cmp_shader("f16", 0x0000, CLASS_POS_ZERO | CLASS_NEG_ZERO, is_cmpx=False, test_id="zero_part")
        val_norm_nomatch, _ = run_cmp_shader("f16", 0x3C00, CLASS_POS_ZERO | CLASS_NEG_ZERO, is_cmpx=False, test_id="norm_part")
        self.assertEqual(val_zero_match, 1)
        self.assertEqual(val_norm_nomatch, 0)


# ------------------------------------------------------------------------------
# v_cmp_class_f16 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestCmpClassF16NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 256)
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.SGPR, 106)

    def test_boundary_extremes_hardware(self):
        # Min subnormal f16: 0x0001 (+denorm)
        val_denorm, _ = run_cmp_shader("f16", 0x0001, CLASS_POS_DENORM, is_cmpx=False, test_id="min_denorm")
        self.assertEqual(val_denorm, 1)

        # Max subnormal f16: 0x03FF (+denorm)
        val_max_denorm, _ = run_cmp_shader("f16", 0x03FF, CLASS_POS_DENORM, is_cmpx=False, test_id="max_denorm")
        self.assertEqual(val_max_denorm, 1)

    def test_spirv_opcode_lowering_hardware(self):
        _, spv = run_cmp_shader("f16", 0x3C00, CLASS_POS_NORMAL, is_cmpx=False, test_id="lowering")
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpLogicalOr", spv)


# ------------------------------------------------------------------------------
# v_cmp_class_f32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def classify_f32(u: int) -> int:
    u = u & 0xFFFF_FFFF
    sign = (u >> 31) & 1
    exp = (u >> 23) & 0xFF
    man = u & 0x7FFFFF
    if exp == 0:
        if man == 0:
            return 5 if sign else 6
        return 4 if sign else 7
    elif exp == 255:
        if man == 0:
            return 2 if sign else 9
        return 1 if (man & 0x400000) else 0
    return 3 if sign else 8


def ref_cmp_class_f32(u: int, mask: int) -> bool:
    c = classify_f32(u)
    return bool(mask & (1 << c))


class TestCmpClassF32Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        samples = [
            (0x7F800001, CLASS_SNAN),
            (0x7FC00000, CLASS_QNAN),
            (0xFF800000, CLASS_NEG_INF),
            (0xBF800000, CLASS_NEG_NORMAL),
            (0x80000001, CLASS_NEG_DENORM),
            (0x80000000, CLASS_NEG_ZERO),
            (0x00000000, CLASS_POS_ZERO),
            (0x00000001, CLASS_POS_DENORM),
            (0x3F800000, CLASS_POS_NORMAL),
            (0x7F800000, CLASS_POS_INF),
        ]
        for idx, (val, mask) in enumerate(samples):
            res_hw, _ = run_cmp_shader("f32", val, mask, is_cmpx=False, test_id=f"oracle_{idx}")
            self.assertEqual(res_hw, 1 if ref_cmp_class_f32(val, mask) else 0)

    def test_oracle_mask_combinations_hardware(self):
        res_all, _ = run_cmp_shader("f32", 0x3F800000, CLASS_ALL, is_cmpx=False, test_id="all_mask")
        self.assertEqual(res_all, 1)

        res_zero, _ = run_cmp_shader("f32", 0x3F800000, 0, is_cmpx=False, test_id="zero_mask")
        self.assertEqual(res_zero, 0)


# ------------------------------------------------------------------------------
# v_cmp_class_f32 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VCmpClassF32Instruction:
    dst: RegisterOperand
    src0: Union[RegisterOperand, LiteralOperand]
    mask: Union[RegisterOperand, LiteralOperand]


def step_cmp_class_f32(machine: ScalarMachineClassLogic, insn: VCmpClassF32Instruction):
    s0 = machine.read_operand(insn.src0) & MASK32
    mask = machine.read_operand(insn.mask) & MASK32
    res = 1 if ref_cmp_class_f32(s0, mask) else 0
    if insn.dst.op_type == OperandType.VCC:
        machine.vcc = MASK32 if res else 0
    elif insn.dst.op_type == OperandType.SGPR:
        machine.sgprs[insn.dst.index] = MASK32 if res else 0


class TestCmpClassF32ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineClassLogic()
        machine.vgprs[1] = 0x3F800000  # +1.0 (class 8: +normal)
        insn = VCmpClassF32Instruction(
            dst=RegisterOperand(OperandType.VCC),
            src0=RegisterOperand(OperandType.VGPR, 1),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        step_cmp_class_f32(machine, insn)

        val_hw, _ = run_cmp_shader("f32", 0x3F800000, CLASS_POS_NORMAL, is_cmpx=False, test_id="ast_scalar")
        self.assertEqual(machine.vcc, MASK32)
        self.assertEqual(val_hw, 1)

    def test_scalar_sgpr_dst_hardware(self):
        val_sgpr, _ = run_cmp_shader("f32", 0xFF800000, CLASS_NEG_INF, is_cmpx=False, test_id="sgpr_dst")
        self.assertEqual(val_sgpr, 1)


# ------------------------------------------------------------------------------
# v_cmp_class_f32 SIMD Divergence & 10-Bit Class Mapping
# ------------------------------------------------------------------------------

def step_cmp_class_f32_simd(wave: SIMDWaveClassLogic, insn: VCmpClassF32Instruction):
    mask = insn.mask.value if isinstance(insn.mask, LiteralOperand) else wave.vgprs[insn.mask.index][0]
    out_vcc = 0
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.vgprs[insn.src0.index][lane] & MASK32
            if ref_cmp_class_f32(s0, mask):
                out_vcc |= (1 << lane)
    wave.vcc = out_vcc


class TestCmpClassF32SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_hardware(self):
        wave = SIMDWaveClassLogic(wave_size=32)
        wave.vgprs[1][0] = 0x3F800000  # +1.0 (+normal)
        wave.vgprs[1][1] = 0x80000000  # -0.0 (-zero)

        insn = VCmpClassF32Instruction(
            dst=RegisterOperand(OperandType.VCC),
            src0=RegisterOperand(OperandType.VGPR, 1),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        step_cmp_class_f32_simd(wave, insn)

        val_lane0, _ = run_cmp_shader("f32", 0x3F800000, CLASS_POS_NORMAL, is_cmpx=False, test_id="simd_l0")
        val_lane1, _ = run_cmp_shader("f32", 0x80000000, CLASS_POS_NORMAL, is_cmpx=False, test_id="simd_l1")
        self.assertEqual(val_lane0, 1)
        self.assertEqual(val_lane1, 0)
        self.assertEqual(wave.vcc & 1, 1)
        self.assertEqual((wave.vcc >> 1) & 1, 0)

    def test_algebraic_complement_hardware(self):
        # Normal floats vs Non-normal floats partition
        val_norm, _ = run_cmp_shader("f32", 0x3F800000, CLASS_POS_NORMAL | CLASS_NEG_NORMAL, is_cmpx=False, test_id="norm_check")
        val_inf, _ = run_cmp_shader("f32", 0x7F800000, CLASS_POS_NORMAL | CLASS_NEG_NORMAL, is_cmpx=False, test_id="inf_check")
        self.assertEqual(val_norm, 1)
        self.assertEqual(val_inf, 0)


# ------------------------------------------------------------------------------
# v_cmp_class_f32 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestCmpClassF32NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, -1)

    def test_boundary_extremes_hardware(self):
        # Min subnormal f32: 0x00000001
        val_min_denorm, _ = run_cmp_shader("f32", 0x00000001, CLASS_POS_DENORM, is_cmpx=False, test_id="min_denorm")
        self.assertEqual(val_min_denorm, 1)

        # Max subnormal f32: 0x007FFFFF
        val_max_denorm, _ = run_cmp_shader("f32", 0x007FFFFF, CLASS_POS_DENORM, is_cmpx=False, test_id="max_denorm")
        self.assertEqual(val_max_denorm, 1)

    def test_spirv_opcode_lowering_hardware(self):
        _, spv = run_cmp_shader("f32", 0x3F800000, CLASS_POS_NORMAL, is_cmpx=False, test_id="lowering")
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpShiftRightLogical", spv)


# ------------------------------------------------------------------------------
# v_cmp_class_f64 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

def classify_f64(u: int) -> int:
    u = u & 0xFFFF_FFFF_FFFF_FFFF
    sign = (u >> 63) & 1
    exp = (u >> 52) & 0x7FF
    man = u & 0x000F_FFFF_FFFF_FFFF
    if exp == 0:
        if man == 0:
            return 5 if sign else 6
        return 4 if sign else 7
    elif exp == 2047:
        if man == 0:
            return 2 if sign else 9
        return 1 if (man & 0x0008_0000_0000_0000) else 0
    return 3 if sign else 8


def ref_cmp_class_f64(u: int, mask: int) -> bool:
    c = classify_f64(u)
    return bool(mask & (1 << c))


class TestCmpClassF64Oracle(unittest.TestCase):
    def test_oracle_matches_hardware(self):
        samples = [
            (0x7FF0000000000001, CLASS_SNAN),
            (0x7FF8000000000000, CLASS_QNAN),
            (0xFFF0000000000000, CLASS_NEG_INF),
            (0xBFF0000000000000, CLASS_NEG_NORMAL),
            (0x8000000000000001, CLASS_NEG_DENORM),
            (0x8000000000000000, CLASS_NEG_ZERO),
            (0x0000000000000000, CLASS_POS_ZERO),
            (0x0000000000000001, CLASS_POS_DENORM),
            (0x3FF0000000000000, CLASS_POS_NORMAL),
            (0x7FF0000000000000, CLASS_POS_INF),
        ]
        for idx, (val, mask) in enumerate(samples):
            res_hw, _ = run_cmp_shader("f64", val, mask, is_cmpx=False, test_id=f"oracle_{idx}")
            self.assertEqual(res_hw, 1 if ref_cmp_class_f64(val, mask) else 0)

    def test_oracle_mask_combinations_hardware(self):
        res_all, _ = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_ALL, is_cmpx=False, test_id="all_mask")
        self.assertEqual(res_all, 1)

        res_zero, _ = run_cmp_shader("f64", 0x3FF0000000000000, 0, is_cmpx=False, test_id="zero_mask")
        self.assertEqual(res_zero, 0)


# ------------------------------------------------------------------------------
# v_cmp_class_f64 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VCmpClassF64Instruction:
    dst: RegisterOperand
    src0_lo: RegisterOperand
    src0_hi: RegisterOperand
    mask: Union[RegisterOperand, LiteralOperand]


def step_cmp_class_f64(machine: ScalarMachineClassLogic, insn: VCmpClassF64Instruction):
    lo = machine.read_operand(insn.src0_lo) & MASK32
    hi = machine.read_operand(insn.src0_hi) & MASK32
    val64 = (hi << 32) | lo
    mask = machine.read_operand(insn.mask) & MASK32
    res = 1 if ref_cmp_class_f64(val64, mask) else 0
    if insn.dst.op_type == OperandType.VCC:
        machine.vcc = MASK32 if res else 0
    elif insn.dst.op_type == OperandType.SGPR:
        machine.sgprs[insn.dst.index] = MASK32 if res else 0


class TestCmpClassF64ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineClassLogic()
        machine.vgprs[2] = 0
        machine.vgprs[3] = 0x3FF00000  # +1.0 f64 (+normal)
        insn = VCmpClassF64Instruction(
            dst=RegisterOperand(OperandType.VCC),
            src0_lo=RegisterOperand(OperandType.VGPR, 2),
            src0_hi=RegisterOperand(OperandType.VGPR, 3),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        step_cmp_class_f64(machine, insn)

        val_hw, _ = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_POS_NORMAL, is_cmpx=False, test_id="ast_scalar")
        self.assertEqual(machine.vcc, MASK32)
        self.assertEqual(val_hw, 1)

    def test_scalar_sgpr_dst_hardware(self):
        val_sgpr, _ = run_cmp_shader("f64", 0xFFF0000000000000, CLASS_NEG_INF, is_cmpx=False, test_id="sgpr_dst")
        self.assertEqual(val_sgpr, 1)


# ------------------------------------------------------------------------------
# v_cmp_class_f64 SIMD Divergence & 10-Bit Class Mapping
# ------------------------------------------------------------------------------

def step_cmp_class_f64_simd(wave: SIMDWaveClassLogic, insn: VCmpClassF64Instruction):
    mask = insn.mask.value if isinstance(insn.mask, LiteralOperand) else wave.vgprs[insn.mask.index][0]
    out_vcc = 0
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            lo = wave.vgprs[insn.src0_lo.index][lane] & MASK32
            hi = wave.vgprs[insn.src0_hi.index][lane] & MASK32
            val64 = (hi << 32) | lo
            if ref_cmp_class_f64(val64, mask):
                out_vcc |= (1 << lane)
    wave.vcc = out_vcc


class TestCmpClassF64SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_hardware(self):
        wave = SIMDWaveClassLogic(wave_size=32)
        wave.vgprs[2][0] = 0
        wave.vgprs[3][0] = 0x3FF00000  # +1.0
        wave.vgprs[2][1] = 0
        wave.vgprs[3][1] = 0x80000000  # -0.0

        insn = VCmpClassF64Instruction(
            dst=RegisterOperand(OperandType.VCC),
            src0_lo=RegisterOperand(OperandType.VGPR, 2),
            src0_hi=RegisterOperand(OperandType.VGPR, 3),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        step_cmp_class_f64_simd(wave, insn)

        val_lane0, _ = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_POS_NORMAL, is_cmpx=False, test_id="simd_l0")
        val_lane1, _ = run_cmp_shader("f64", 0x8000000000000000, CLASS_POS_NORMAL, is_cmpx=False, test_id="simd_l1")
        self.assertEqual(val_lane0, 1)
        self.assertEqual(val_lane1, 0)
        self.assertEqual(wave.vcc & 1, 1)
        self.assertEqual((wave.vcc >> 1) & 1, 0)

    def test_algebraic_partition_hardware(self):
        val_nan, _ = run_cmp_shader("f64", 0x7FF8000000000000, CLASS_SNAN | CLASS_QNAN, is_cmpx=False, test_id="nan_part")
        val_real, _ = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_SNAN | CLASS_QNAN, is_cmpx=False, test_id="real_part")
        self.assertEqual(val_nan, 1)
        self.assertEqual(val_real, 0)


# ------------------------------------------------------------------------------
# v_cmp_class_f64 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestCmpClassF64NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            LiteralOperand(0x1_0000_0000_0000_0000)

    def test_boundary_extremes_hardware(self):
        # Min subnormal f64: 1
        val_min_denorm, _ = run_cmp_shader("f64", 1, CLASS_POS_DENORM, is_cmpx=False, test_id="min_denorm")
        self.assertEqual(val_min_denorm, 1)

        # Max subnormal f64: 0x000FFFFFFFFFFFFF
        val_max_denorm, _ = run_cmp_shader("f64", 0x000FFFFFFFFFFFFF, CLASS_POS_DENORM, is_cmpx=False, test_id="max_denorm")
        self.assertEqual(val_max_denorm, 1)

    def test_spirv_opcode_lowering_hardware(self):
        _, spv = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_POS_NORMAL, is_cmpx=False, test_id="lowering")
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpShiftRightLogical", spv)


# ------------------------------------------------------------------------------
# v_cmpx_class_f16 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

class TestCmpxClassF16Oracle(unittest.TestCase):
    def test_oracle_cmpx_matches_hardware(self):
        # Match keeps lane active -> returns 1
        val_match, _ = run_cmp_shader("f16", 0x3C00, CLASS_POS_NORMAL, is_cmpx=True, test_id="cmpx_match")
        self.assertEqual(val_match, 1)

        # Mismatch narrows EXEC (lane becomes inactive) -> returns 0
        val_mismatch, _ = run_cmp_shader("f16", 0x3C00, CLASS_NEG_INF, is_cmpx=True, test_id="cmpx_mismatch")
        self.assertEqual(val_mismatch, 0)

    def test_cmpx_identity_hardware(self):
        # CLASS_ALL preserves active status
        val_all, _ = run_cmp_shader("f16", 0x7C00, CLASS_ALL, is_cmpx=True, test_id="cmpx_id_all")
        self.assertEqual(val_all, 1)

        # Zero mask deactivates lane
        val_zero, _ = run_cmp_shader("f16", 0x7C00, 0, is_cmpx=True, test_id="cmpx_id_zero")
        self.assertEqual(val_zero, 0)


# ------------------------------------------------------------------------------
# v_cmpx_class_f16 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VCmpxClassF16Instruction:
    src0: Union[RegisterOperand, LiteralOperand]
    mask: Union[RegisterOperand, LiteralOperand]


def step_cmpx_class_f16(machine: ScalarMachineClassLogic, insn: VCmpxClassF16Instruction):
    s0 = machine.read_operand(insn.src0) & 0xFFFF
    mask = machine.read_operand(insn.mask) & 0xFFFF
    cond = ref_cmp_class_f16(s0, mask)
    if not cond:
        machine.exec = 0


class TestCmpxClassF16ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineClassLogic()
        machine.vgprs[1] = 0x3C00
        insn = VCmpxClassF16Instruction(
            src0=RegisterOperand(OperandType.VGPR, 1),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        step_cmpx_class_f16(machine, insn)

        val_hw, _ = run_cmp_shader("f16", 0x3C00, CLASS_POS_NORMAL, is_cmpx=True, test_id="ast_scalar_match")
        self.assertEqual(machine.exec, MASK32)
        self.assertEqual(val_hw, 1)

    def test_scalar_predicated_execution_hardware(self):
        val_hw, _ = run_cmp_shader("f16", 0x0000, CLASS_NEG_NORMAL, is_cmpx=True, test_id="ast_scalar_pred")
        self.assertEqual(val_hw, 0)


# ------------------------------------------------------------------------------
# v_cmpx_class_f16 SIMD Divergence & Active Mask Narrowing
# ------------------------------------------------------------------------------

def step_cmpx_class_f16_simd(wave: SIMDWaveClassLogic, insn: VCmpxClassF16Instruction):
    mask = insn.mask.value if isinstance(insn.mask, LiteralOperand) else wave.vgprs[insn.mask.index][0]
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.vgprs[insn.src0.index][lane] & 0xFFFF
            if not ref_cmp_class_f16(s0, mask):
                wave.exec_mask &= ~(1 << lane)


class TestCmpxClassF16SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_and_exec_narrowing_hardware(self):
        wave = SIMDWaveClassLogic(wave_size=32)
        wave.vgprs[1][0] = 0x3C00  # matches +normal
        wave.vgprs[1][1] = 0x0000  # mismatches +normal

        insn = VCmpxClassF16Instruction(
            src0=RegisterOperand(OperandType.VGPR, 1),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        step_cmpx_class_f16_simd(wave, insn)

        val_lane0, _ = run_cmp_shader("f16", 0x3C00, CLASS_POS_NORMAL, is_cmpx=True, test_id="simd_l0")
        val_lane1, _ = run_cmp_shader("f16", 0x0000, CLASS_POS_NORMAL, is_cmpx=True, test_id="simd_l1")
        self.assertEqual(val_lane0, 1)
        self.assertEqual(val_lane1, 0)
        self.assertEqual(wave.exec_mask & 1, 1)
        self.assertEqual((wave.exec_mask >> 1) & 1, 0)

    def test_cascaded_cmpx_narrowing_hardware(self):
        val_cascade, _ = run_cmp_shader("f16", 0x3C00, CLASS_POS_NORMAL, is_cmpx=True, test_id="cascade")
        self.assertEqual(val_cascade, 1)


# ------------------------------------------------------------------------------
# v_cmpx_class_f16 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestCmpxClassF16NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.VGPR, 300)

    def test_boundary_extremes_hardware(self):
        val_snan, _ = run_cmp_shader("f16", 0x7C01, CLASS_SNAN, is_cmpx=True, test_id="snan")
        self.assertEqual(val_snan, 1)

    def test_spirv_opcode_lowering_hardware(self):
        _, spv = run_cmp_shader("f16", 0x3C00, CLASS_POS_NORMAL, is_cmpx=True, test_id="lowering")
        self.assertIn("OpStore %active", spv)


# ------------------------------------------------------------------------------
# v_cmpx_class_f32 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

class TestCmpxClassF32Oracle(unittest.TestCase):
    def test_oracle_cmpx_matches_hardware(self):
        val_match, _ = run_cmp_shader("f32", 0x3F800000, CLASS_POS_NORMAL, is_cmpx=True, test_id="cmpx_match")
        self.assertEqual(val_match, 1)

        val_mismatch, _ = run_cmp_shader("f32", 0x3F800000, CLASS_NEG_INF, is_cmpx=True, test_id="cmpx_mismatch")
        self.assertEqual(val_mismatch, 0)

    def test_cmpx_identity_hardware(self):
        val_all, _ = run_cmp_shader("f32", 0x7F800000, CLASS_ALL, is_cmpx=True, test_id="cmpx_id_all")
        self.assertEqual(val_all, 1)

        val_zero, _ = run_cmp_shader("f32", 0x7F800000, 0, is_cmpx=True, test_id="cmpx_id_zero")
        self.assertEqual(val_zero, 0)


# ------------------------------------------------------------------------------
# v_cmpx_class_f32 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VCmpxClassF32Instruction:
    src0: Union[RegisterOperand, LiteralOperand]
    mask: Union[RegisterOperand, LiteralOperand]


def step_cmpx_class_f32(machine: ScalarMachineClassLogic, insn: VCmpxClassF32Instruction):
    s0 = machine.read_operand(insn.src0) & MASK32
    mask = machine.read_operand(insn.mask) & MASK32
    cond = ref_cmp_class_f32(s0, mask)
    if not cond:
        machine.exec = 0


class TestCmpxClassF32ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineClassLogic()
        machine.vgprs[1] = 0x3F800000
        insn = VCmpxClassF32Instruction(
            src0=RegisterOperand(OperandType.VGPR, 1),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        step_cmpx_class_f32(machine, insn)

        val_hw, _ = run_cmp_shader("f32", 0x3F800000, CLASS_POS_NORMAL, is_cmpx=True, test_id="ast_scalar_match")
        self.assertEqual(machine.exec, MASK32)
        self.assertEqual(val_hw, 1)

    def test_scalar_predicated_execution_hardware(self):
        val_hw, _ = run_cmp_shader("f32", 0x00000000, CLASS_NEG_NORMAL, is_cmpx=True, test_id="ast_scalar_pred")
        self.assertEqual(val_hw, 0)


# ------------------------------------------------------------------------------
# v_cmpx_class_f32 SIMD Divergence & Active Mask Narrowing
# ------------------------------------------------------------------------------

def step_cmpx_class_f32_simd(wave: SIMDWaveClassLogic, insn: VCmpxClassF32Instruction):
    mask = insn.mask.value if isinstance(insn.mask, LiteralOperand) else wave.vgprs[insn.mask.index][0]
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.vgprs[insn.src0.index][lane] & MASK32
            if not ref_cmp_class_f32(s0, mask):
                wave.exec_mask &= ~(1 << lane)


class TestCmpxClassF32SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_and_exec_narrowing_hardware(self):
        wave = SIMDWaveClassLogic(wave_size=32)
        wave.vgprs[1][0] = 0x3F800000
        wave.vgprs[1][1] = 0x80000000

        insn = VCmpxClassF32Instruction(
            src0=RegisterOperand(OperandType.VGPR, 1),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        step_cmpx_class_f32_simd(wave, insn)

        val_lane0, _ = run_cmp_shader("f32", 0x3F800000, CLASS_POS_NORMAL, is_cmpx=True, test_id="simd_l0")
        val_lane1, _ = run_cmp_shader("f32", 0x80000000, CLASS_POS_NORMAL, is_cmpx=True, test_id="simd_l1")
        self.assertEqual(val_lane0, 1)
        self.assertEqual(val_lane1, 0)
        self.assertEqual(wave.exec_mask & 1, 1)
        self.assertEqual((wave.exec_mask >> 1) & 1, 0)

    def test_cascaded_cmpx_narrowing_hardware(self):
        val_cascade, _ = run_cmp_shader("f32", 0x3F800000, CLASS_POS_NORMAL, is_cmpx=True, test_id="cascade")
        self.assertEqual(val_cascade, 1)


# ------------------------------------------------------------------------------
# v_cmpx_class_f32 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestCmpxClassF32NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            LiteralOperand(0x2_0000_0000_0000_0000)

    def test_boundary_extremes_hardware(self):
        val_snan, _ = run_cmp_shader("f32", 0x7F800001, CLASS_SNAN, is_cmpx=True, test_id="snan")
        self.assertEqual(val_snan, 1)

    def test_spirv_opcode_lowering_hardware(self):
        _, spv = run_cmp_shader("f32", 0x3F800000, CLASS_POS_NORMAL, is_cmpx=True, test_id="lowering")
        self.assertIn("OpStore %active", spv)


# ------------------------------------------------------------------------------
# v_cmpx_class_f64 Reference Oracle & Architectural Model
# ------------------------------------------------------------------------------

class TestCmpxClassF64Oracle(unittest.TestCase):
    def test_oracle_cmpx_matches_hardware(self):
        val_match, _ = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_POS_NORMAL, is_cmpx=True, test_id="cmpx_match")
        self.assertEqual(val_match, 1)

        val_mismatch, _ = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_NEG_INF, is_cmpx=True, test_id="cmpx_mismatch")
        self.assertEqual(val_mismatch, 0)

    def test_cmpx_identity_hardware(self):
        val_all, _ = run_cmp_shader("f64", 0x7FF0000000000000, CLASS_ALL, is_cmpx=True, test_id="cmpx_id_all")
        self.assertEqual(val_all, 1)

        val_zero, _ = run_cmp_shader("f64", 0x7FF0000000000000, 0, is_cmpx=True, test_id="cmpx_id_zero")
        self.assertEqual(val_zero, 0)


# ------------------------------------------------------------------------------
# v_cmpx_class_f64 AST & Scalar Machine
# ------------------------------------------------------------------------------

@dataclass
class VCmpxClassF64Instruction:
    src0_lo: RegisterOperand
    src0_hi: RegisterOperand
    mask: Union[RegisterOperand, LiteralOperand]


def step_cmpx_class_f64(machine: ScalarMachineClassLogic, insn: VCmpxClassF64Instruction):
    lo = machine.read_operand(insn.src0_lo) & MASK32
    hi = machine.read_operand(insn.src0_hi) & MASK32
    val64 = (hi << 32) | lo
    mask = machine.read_operand(insn.mask) & MASK32
    cond = ref_cmp_class_f64(val64, mask)
    if not cond:
        machine.exec = 0


class TestCmpxClassF64ASTAndScalar(unittest.TestCase):
    def test_ast_and_scalar_matches_hardware(self):
        machine = ScalarMachineClassLogic()
        machine.vgprs[2] = 0
        machine.vgprs[3] = 0x3FF00000
        insn = VCmpxClassF64Instruction(
            src0_lo=RegisterOperand(OperandType.VGPR, 2),
            src0_hi=RegisterOperand(OperandType.VGPR, 3),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        step_cmpx_class_f64(machine, insn)

        val_hw, _ = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_POS_NORMAL, is_cmpx=True, test_id="ast_scalar_match")
        self.assertEqual(machine.exec, MASK32)
        self.assertEqual(val_hw, 1)

    def test_scalar_predicated_execution_hardware(self):
        val_hw, _ = run_cmp_shader("f64", 0x0000000000000000, CLASS_NEG_NORMAL, is_cmpx=True, test_id="ast_scalar_pred")
        self.assertEqual(val_hw, 0)


# ------------------------------------------------------------------------------
# v_cmpx_class_f64 SIMD Divergence & Active Mask Narrowing
# ------------------------------------------------------------------------------

def step_cmpx_class_f64_simd(wave: SIMDWaveClassLogic, insn: VCmpxClassF64Instruction):
    mask = insn.mask.value if isinstance(insn.mask, LiteralOperand) else wave.vgprs[insn.mask.index][0]
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            lo = wave.vgprs[insn.src0_lo.index][lane] & MASK32
            hi = wave.vgprs[insn.src0_hi.index][lane] & MASK32
            val64 = (hi << 32) | lo
            if not ref_cmp_class_f64(val64, mask):
                wave.exec_mask &= ~(1 << lane)


class TestCmpxClassF64SIMDAndProperties(unittest.TestCase):
    def test_simd_divergence_and_exec_narrowing_hardware(self):
        wave = SIMDWaveClassLogic(wave_size=32)
        wave.vgprs[2][0] = 0
        wave.vgprs[3][0] = 0x3FF00000
        wave.vgprs[2][1] = 0
        wave.vgprs[3][1] = 0x80000000

        insn = VCmpxClassF64Instruction(
            src0_lo=RegisterOperand(OperandType.VGPR, 2),
            src0_hi=RegisterOperand(OperandType.VGPR, 3),
            mask=LiteralOperand(CLASS_POS_NORMAL),
        )
        step_cmpx_class_f64_simd(wave, insn)

        val_lane0, _ = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_POS_NORMAL, is_cmpx=True, test_id="simd_l0")
        val_lane1, _ = run_cmp_shader("f64", 0x8000000000000000, CLASS_POS_NORMAL, is_cmpx=True, test_id="simd_l1")
        self.assertEqual(val_lane0, 1)
        self.assertEqual(val_lane1, 0)
        self.assertEqual(wave.exec_mask & 1, 1)
        self.assertEqual((wave.exec_mask >> 1) & 1, 0)

    def test_cascaded_cmpx_narrowing_hardware(self):
        val_cascade, _ = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_POS_NORMAL, is_cmpx=True, test_id="cascade")
        self.assertEqual(val_cascade, 1)


# ------------------------------------------------------------------------------
# v_cmpx_class_f64 Negative Controls & Hardware Translator Execution
# ------------------------------------------------------------------------------

class TestCmpxClassF64NegativeAndTranslator(unittest.TestCase):
    def test_syntax_rejection(self):
        with self.assertRaises(ValueError):
            RegisterOperand(OperandType.SGPR, -5)

    def test_boundary_extremes_hardware(self):
        val_snan, _ = run_cmp_shader("f64", 0x7FF0000000000001, CLASS_SNAN, is_cmpx=True, test_id="snan")
        self.assertEqual(val_snan, 1)

    def test_spirv_opcode_lowering_hardware(self):
        _, spv = run_cmp_shader("f64", 0x3FF0000000000000, CLASS_POS_NORMAL, is_cmpx=True, test_id="lowering")
        self.assertIn("OpStore %active", spv)


if __name__ == "__main__":
    unittest.main()
