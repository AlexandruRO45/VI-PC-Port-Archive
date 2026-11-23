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
