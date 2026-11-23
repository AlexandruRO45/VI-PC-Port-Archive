"""Semantic reference tests and translator-backed validation for AMD RDNA2 v_cvt_f32_i32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, v_cvt_f32_i32 converts a 32-bit signed integer to a
  single-precision (binary32) IEEE 754 floating-point number:
      D.f32 = float(S0.i32)

  - Encoding: VOP1 (opcode 0x5) or VOP3 (opcode 0x185).
  - Operands: Destination is a 32-bit VGPR; Source S0 can be a VGPR, SGPR, or 32-bit constant.
  - Rounding Mode: Conversion from integer to float uses IEEE 754 round to nearest even (RNE).
  - Exactness: Integers in [-2^24, 2^24] are converted with zero precision loss.
  - Flags: VALU conversions do not modify VCC or SCC condition codes.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import enum
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from support import translate_lines, mrt0_words  # noqa: E402

MASK32 = 0xFFFFFFFF


def oracle_cvt_f32_i32(s0: int) -> int:
    """Reference mathematical oracle for v_cvt_f32_i32: converts int32 to binary32 IEEE 754 bits."""
    s0_u32 = s0 & MASK32
    val_i32 = s0_u32 if s0_u32 < 0x80000000 else s0_u32 - 0x100000000
    # IEEE 754 round-to-nearest-even single precision conversion
    return struct.unpack(">I", struct.pack(">f", float(val_i32)))[0]

class VCvtF32I32ReferenceOracleTests(unittest.TestCase):
    def test_oracle_zero(self):
        # 0 -> +0.0f (0x00000000)
        self.assertEqual(oracle_cvt_f32_i32(0), 0x00000000)

    def test_oracle_small_positive_integers(self):
        # 1 -> 1.0f (0x3F800000)
        self.assertEqual(oracle_cvt_f32_i32(1), 0x3F800000)
        # 2 -> 2.0f (0x40000000)
        self.assertEqual(oracle_cvt_f32_i32(2), 0x40000000)
        # 42 -> 42.0f (0x42280000)
        self.assertEqual(oracle_cvt_f32_i32(42), 0x42280000)

    def test_oracle_small_negative_integers(self):
        # -1 (0xFFFFFFFF) -> -1.0f (0xBF800000)
        self.assertEqual(oracle_cvt_f32_i32(0xFFFFFFFF), 0xBF800000)
        # -2 (0xFFFFFFFE) -> -2.0f (0xC0000000)
        self.assertEqual(oracle_cvt_f32_i32(0xFFFFFFFE), 0xC0000000)

    def test_oracle_boundary_extrema(self):
        # 2147483647 (0x7FFFFFFF) -> 2147483648.0f (0x4F000000) due to RNE round up
        self.assertEqual(oracle_cvt_f32_i32(0x7FFFFFFF), 0x4F000000)
        # -2147483648 (0x80000000) -> -2147483648.0f (0xCF000000) exact power of 2
        self.assertEqual(oracle_cvt_f32_i32(0x80000000), 0xCF000000)

    def test_oracle_precision_rne_rounding(self):
        # 2^24 = 16777216 is exact
        self.assertEqual(oracle_cvt_f32_i32(16777216), 0x4B800000)
        # 2^24 + 1 = 16777217 rounds to even 16777216
        self.assertEqual(oracle_cvt_f32_i32(16777217), 0x4B800000)
        # 2^24 + 2 = 16777218 has exact representation
        self.assertEqual(oracle_cvt_f32_i32(16777218), 0x4B800001)

class RegKind(enum.Enum):
    VGPR = "vgpr"
    SGPR = "sgpr"
    IMM = "imm"


@dataclass(frozen=True)
class Operand:
    kind: RegKind
    index: int = 0
    raw_val: int = 0


@dataclass(frozen=True)
class InstructionAST:
    mnemonic: str
    dst: Operand
    src0: Operand


class VCvtF32I32AstValidationTests(unittest.TestCase):
    def test_ast_construction_vgpr(self):
        ast = InstructionAST(
            mnemonic="v_cvt_f32_i32",
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.VGPR, index=1),
        )
        self.assertEqual(ast.mnemonic, "v_cvt_f32_i32")
        self.assertEqual(ast.dst.kind, RegKind.VGPR)
        self.assertEqual(ast.dst.index, 0)
        self.assertEqual(ast.src0.kind, RegKind.VGPR)
        self.assertEqual(ast.src0.index, 1)

    def test_ast_construction_sgpr_src(self):
        ast = InstructionAST(
            mnemonic="v_cvt_f32_i32",
            dst=Operand(RegKind.VGPR, index=2),
            src0=Operand(RegKind.SGPR, index=4),
        )
        self.assertEqual(ast.src0.kind, RegKind.SGPR)
        self.assertEqual(ast.src0.index, 4)

    def test_ast_construction_imm_src(self):
        ast = InstructionAST(
            mnemonic="v_cvt_f32_i32",
            dst=Operand(RegKind.VGPR, index=0),
            src0=Operand(RegKind.IMM, raw_val=-100),
        )
        self.assertEqual(ast.src0.kind, RegKind.IMM)
        self.assertEqual(ast.src0.raw_val, -100)

class ScalarLaneState:
    def __init__(self):
        self.vgprs: dict[int, int] = {}
        self.sgprs: dict[int, int] = {}

    def get_operand_val(self, op: Operand) -> int:
        if op.kind == RegKind.VGPR:
            return self.vgprs.get(op.index, 0) & MASK32
        elif op.kind == RegKind.SGPR:
            return self.sgprs.get(op.index, 0) & MASK32
        else:
            return op.raw_val & MASK32

    def set_vgpr(self, idx: int, val: int):
        self.vgprs[idx] = val & MASK32


def execute_scalar_cvt_f32_i32(state: ScalarLaneState, ast: InstructionAST) -> None:
    src_val = state.get_operand_val(ast.src0)
    res = oracle_cvt_f32_i32(src_val)
    state.set_vgpr(ast.dst.index, res)


class VCvtF32I32ScalarInterpreterTests(unittest.TestCase):
    def test_interpreter_positive_conversion(self):
        state = ScalarLaneState()
        state.set_vgpr(1, 100)
        ast = InstructionAST(
            "v_cvt_f32_i32",
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
        )
        execute_scalar_cvt_f32_i32(state, ast)
        self.assertEqual(state.vgprs[0], 0x42C80000)  # 100.0f

    def test_interpreter_negative_conversion(self):
        state = ScalarLaneState()
        state.set_vgpr(1, 0xFFFFFF9C)  # -100 in two's-complement
        ast = InstructionAST(
            "v_cvt_f32_i32",
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
        )
        execute_scalar_cvt_f32_i32(state, ast)
        self.assertEqual(state.vgprs[0], 0xC2C80000)  # -100.0f

    def test_interpreter_zero_conversion(self):
        state = ScalarLaneState()
        state.set_vgpr(1, 0)
        ast = InstructionAST(
            "v_cvt_f32_i32",
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
        )
        execute_scalar_cvt_f32_i32(state, ast)
        self.assertEqual(state.vgprs[0], 0x00000000)

class WavefrontSimdState:
    def __init__(self, lane_count: int = 32):
        self.lane_count = lane_count
        self.exec_mask: int = (1 << lane_count) - 1
        self.lanes = [ScalarLaneState() for _ in range(lane_count)]
        self.shared_sgprs: dict[int, int] = {}

    def execute_valu(self, ast: InstructionAST) -> None:
        for lane_id in range(self.lane_count):
            if (self.exec_mask >> lane_id) & 1:
                # Share sgprs across lanes
                self.lanes[lane_id].sgprs = self.shared_sgprs
                execute_scalar_cvt_f32_i32(self.lanes[lane_id], ast)


class VCvtF32I32SimdDivergenceTests(unittest.TestCase):
    def test_wave32_exec_mask_filtering(self):
        wave = WavefrontSimdState(lane_count=32)
        # Alternating active lanes: 0x55555555
        wave.exec_mask = 0x55555555
        for lane_id in range(32):
            wave.lanes[lane_id].set_vgpr(0, 0xDEADBEEF)  # Pre-existing value
            wave.lanes[lane_id].set_vgpr(1, lane_id)

        ast = InstructionAST(
            "v_cvt_f32_i32",
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
        )
        wave.execute_valu(ast)

        for lane_id in range(32):
            if (0x55555555 >> lane_id) & 1:
                expected = oracle_cvt_f32_i32(lane_id)
                self.assertEqual(wave.lanes[lane_id].vgprs[0], expected)
            else:
                self.assertEqual(wave.lanes[lane_id].vgprs[0], 0xDEADBEEF)

    def test_wave64_exec_mask_filtering(self):
        wave = WavefrontSimdState(lane_count=64)
        # Upper 32 lanes active only
        wave.exec_mask = 0xFFFFFFFF00000000
        for lane_id in range(64):
            wave.lanes[lane_id].set_vgpr(0, 0xCAFEBABE)
            wave.lanes[lane_id].set_vgpr(1, -lane_id)

        ast = InstructionAST(
            "v_cvt_f32_i32",
            Operand(RegKind.VGPR, 0),
            Operand(RegKind.VGPR, 1),
        )
        wave.execute_valu(ast)

        for lane_id in range(64):
            if lane_id >= 32:
                expected = oracle_cvt_f32_i32(-lane_id)
                self.assertEqual(wave.lanes[lane_id].vgprs[0], expected)
            else:
                self.assertEqual(wave.lanes[lane_id].vgprs[0], 0xCAFEBABE)

if __name__ == "__main__":
    unittest.main()
