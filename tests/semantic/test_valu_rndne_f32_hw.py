from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 v_rndne_f32.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the v_rndne_f32 instruction belongs to VOP1 (or VOP3) format.
  Instruction:
    v_rndne_f32 vdst, src0
    v_rndne_f32_e32 vdst, src0
    v_rndne_f32_e64 vdst, src0 [clamp]

  Operands:
    vdst = 32-bit destination VGPR (v[0..255])
    src0 = 32-bit source float operand: VGPR, SGPR, inline float constant, or 32-bit literal

  Semantics:
    vdst = round_to_nearest_even(src0)
    Rounds IEEE 754 single-precision float to integer, breaking ties towards the even integer.
    Special values:
      +-0.0 -> +-0.0 (sign preserved)
      +-inf -> +-inf
      NaN -> quiet NaN preserved
      Subnormals -> +-0.0 (underflow to signed zero)
      |src0| >= 2^23 -> exact integer identity
    Optional VOP3 clamp modifier clamps the result to [0.0, 1.0].
    Condition codes: Does not modify SCC or VCC.
    EXEC masking: Lane operations executed conditionally under wave32 / wave64 EXEC mask.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32: int = 0xFFFFFFFF
F32_POS_ZERO: int = 0x00000000
F32_NEG_ZERO: int = 0x80000000
F32_ONE: int = 0x3F800000
F32_NEG_ONE: int = 0xBF800000
F32_TWO: int = 0x40000000
F32_NEG_TWO: int = 0xC0000000
F32_FOUR: int = 0x40800000
F32_NEG_FOUR: int = 0xC0800000
F32_POS_INF: int = 0x7F800000
F32_NEG_INF: int = 0xFF800000
F32_QNAN: int = 0x7FC00000
F32_SNAN: int = 0x7F800001
F32_MIN_SUBNORMAL: int = 0x00000001
F32_MAX_SUBNORMAL: int = 0x007FFFFF
F32_MIN_NORMAL: int = 0x00800000
F32_MAX_FLOAT: int = 0x7F7FFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def f32_to_bits(f: float) -> int:
    """Pack float to 32-bit IEEE-754 bit pattern."""
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(b: int) -> float:
    """Unpack 32-bit IEEE-754 bit pattern to float."""
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def ref_rndne_f32(src0_bits: int, clamp: bool = False) -> int:
    """Mathematical reference oracle for AMD RDNA2 v_rndne_f32.

    Rounds single-precision IEEE-754 float to nearest integer, ties to even.

    Args:
        src0_bits: 32-bit float bit pattern (uint32)
        clamp: whether to apply VOP3 clamp to [0.0, 1.0]

    Returns:
        32-bit float bit pattern representing round_even(src0).
    """
    sign = (src0_bits >> 31) & 1
    exp = (src0_bits >> 23) & 0xFF
    mant = src0_bits & 0x7FFFFF

    # Special values: Inf and NaN
    if exp == 0xFF:
        if mant != 0:
            # Quiet NaN preserved
            return src0_bits | 0x00400000
        # Infinity preserved
        return src0_bits

    # Large values (|x| >= 2^23): already integer-exact
    if exp >= (127 + 23):
        return src0_bits

    val = bits_to_f32(src0_bits)
    if math.isnan(val) or math.isinf(val):
        return src0_bits

    # Subnormals: magnitude < 2^-126, round to signed zero
    if exp == 0:
        res_bits = F32_NEG_ZERO if sign else F32_POS_ZERO
    else:
        # Round half to even
        r = round(val)
        if r == 0:
            res_bits = F32_NEG_ZERO if sign else F32_POS_ZERO
        else:
            res_bits = f32_to_bits(float(r))

    if clamp:
        clamped_f = bits_to_f32(res_bits)
        if clamped_f < 0.0 or res_bits == F32_NEG_ZERO:
            res_bits = F32_POS_ZERO
        elif clamped_f > 1.0:
            res_bits = F32_ONE

    return res_bits


class TestValuRndneF32Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of v_rndne_f32."""

    def test_oracle_exact_integers(self):
        for int_val in [0.0, 1.0, 2.0, 10.0, -1.0, -5.0, 100.0]:
            bits = f32_to_bits(int_val)
            res = ref_rndne_f32(bits)
            self.assertEqual(res, bits)

    def test_oracle_halfway_ties_even(self):
        cases = [
            (0.5, 0.0, F32_POS_ZERO),
            (-0.5, -0.0, F32_NEG_ZERO),
            (1.5, 2.0, F32_TWO),
            (-1.5, -2.0, F32_NEG_TWO),
            (2.5, 2.0, F32_TWO),
            (-2.5, -2.0, F32_NEG_TWO),
            (3.5, 4.0, F32_FOUR),
            (-3.5, -4.0, F32_NEG_FOUR),
            (4.5, 4.0, F32_FOUR),
            (-4.5, -4.0, F32_NEG_FOUR),
        ]
        for inp_f, exp_f, exp_bits in cases:
            res = ref_rndne_f32(f32_to_bits(inp_f))
            self.assertEqual(res, exp_bits)
            self.assertEqual(bits_to_f32(res), exp_f)

    def test_oracle_non_halfway(self):
        cases = [
            (1.1, 1.0, F32_ONE),
            (1.9, 2.0, F32_TWO),
            (-1.1, -1.0, F32_NEG_ONE),
            (-1.9, -2.0, F32_NEG_TWO),
            (0.2, 0.0, F32_POS_ZERO),
            (-0.2, -0.0, F32_NEG_ZERO),
            (0.8, 1.0, F32_ONE),
            (-0.8, -1.0, F32_NEG_ONE),
        ]
        for inp_f, exp_f, exp_bits in cases:
            res = ref_rndne_f32(f32_to_bits(inp_f))
            self.assertEqual(res, exp_bits)
            self.assertEqual(bits_to_f32(res), exp_f)

    def test_oracle_clamp(self):
        self.assertEqual(ref_rndne_f32(f32_to_bits(2.5), clamp=True), F32_ONE)
        self.assertEqual(ref_rndne_f32(f32_to_bits(-1.5), clamp=True), F32_POS_ZERO)
        self.assertEqual(ref_rndne_f32(f32_to_bits(0.5), clamp=True), F32_POS_ZERO)
        self.assertEqual(ref_rndne_f32(f32_to_bits(0.9), clamp=True), F32_ONE)


class Opcode(Enum):
    V_RNDNE_F32 = auto()


class OperandKind(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL32 = auto()
    INLINE_FLOAT = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    reg_idx: int = 0
    raw_bits: int = 0

    @classmethod
    def vgpr(cls, idx: int) -> ASTOperand:
        if not (0 <= idx <= 255):
            raise ValueError(f"VGPR index out of range: {idx}")
        return cls(kind=OperandKind.VGPR, reg_idx=idx)

    @classmethod
    def sgpr(cls, idx: int) -> ASTOperand:
        if not (0 <= idx <= 105):
            raise ValueError(f"SGPR index out of range: {idx}")
        return cls(kind=OperandKind.SGPR, reg_idx=idx)

    @classmethod
    def literal32(cls, bits: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL32, raw_bits=u32(bits))

    @classmethod
    def inline_float(cls, f: float) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_FLOAT, raw_bits=f32_to_bits(f))


@dataclass
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    clamp: bool = False

    def validate(self) -> None:
        if self.opcode != Opcode.V_RNDNE_F32:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR:
            raise ValueError(f"Destination must be a VGPR, got {self.dst.kind}")
        if self.src0.kind not in (
            OperandKind.VGPR,
            OperandKind.SGPR,
            OperandKind.LITERAL32,
            OperandKind.INLINE_FLOAT,
        ):
            raise ValueError(f"Invalid src0 operand kind: {self.src0.kind}")


class TestValuRndneF32AST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_vgpr_to_vgpr(self):
        inst = ASTInstruction(
            opcode=Opcode.V_RNDNE_F32,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(1),
        )
        inst.validate()
        self.assertEqual(inst.dst.reg_idx, 0)
        self.assertEqual(inst.src0.reg_idx, 1)
        self.assertFalse(inst.clamp)

    def test_ast_construction_sgpr_source(self):
        inst = ASTInstruction(
            opcode=Opcode.V_RNDNE_F32,
            dst=ASTOperand.vgpr(5),
            src0=ASTOperand.sgpr(2),
        )
        inst.validate()
        self.assertEqual(inst.src0.kind, OperandKind.SGPR)
        self.assertEqual(inst.src0.reg_idx, 2)

    def test_ast_construction_literal_source(self):
        bits = f32_to_bits(1.5)
        inst = ASTInstruction(
            opcode=Opcode.V_RNDNE_F32,
            dst=ASTOperand.vgpr(10),
            src0=ASTOperand.literal32(bits),
            clamp=True,
        )
        inst.validate()
        self.assertEqual(inst.src0.raw_bits, bits)
        self.assertTrue(inst.clamp)

    def test_ast_validation_rejects_sgpr_destination(self):
        inst = ASTInstruction(
            opcode=Opcode.V_RNDNE_F32,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(1),
        )
        with self.assertRaises(ValueError):
            inst.validate()


class VALURegisterState:
    """Scalar architectural register state model for single-lane VALU execution."""

    def __init__(self):
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106

    def read_operand(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR:
            return self.vgprs[op.reg_idx]
        elif op.kind == OperandKind.SGPR:
            return self.sgprs[op.reg_idx]
        elif op.kind in (OperandKind.LITERAL32, OperandKind.INLINE_FLOAT):
            return op.raw_bits
        raise ValueError(f"Unknown operand kind: {op.kind}")

    def write_vgpr(self, reg_idx: int, val: int) -> None:
        self.vgprs[reg_idx] = u32(val)

    def execute(self, inst: ASTInstruction) -> int:
        inst.validate()
        src_val = self.read_operand(inst.src0)
        res = ref_rndne_f32(src_val, clamp=inst.clamp)
        self.write_vgpr(inst.dst.reg_idx, res)
        return res


class TestValuRndneF32ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def test_interpreter_vgpr_source(self):
        state = VALURegisterState()
        state.vgprs[1] = f32_to_bits(2.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        res = state.execute(inst)
        self.assertEqual(res, F32_TWO)
        self.assertEqual(state.vgprs[0], F32_TWO)
        self.assertEqual(state.vgprs[1], f32_to_bits(2.5))

    def test_interpreter_sgpr_source(self):
        state = VALURegisterState()
        state.sgprs[4] = f32_to_bits(3.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(2), ASTOperand.sgpr(4))
        res = state.execute(inst)
        self.assertEqual(res, F32_FOUR)
        self.assertEqual(state.vgprs[2], F32_FOUR)

    def test_interpreter_clamp(self):
        state = VALURegisterState()
        state.vgprs[3] = f32_to_bits(-2.5)
        inst = ASTInstruction(
            Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(3), clamp=True
        )
        res = state.execute(inst)
        self.assertEqual(res, F32_POS_ZERO)
        self.assertEqual(state.vgprs[0], F32_POS_ZERO)

    def test_interpreter_unaffected_registers(self):
        state = VALURegisterState()
        state.vgprs[1] = f32_to_bits(1.5)
        state.vgprs[7] = 0xDEADBEEF
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        state.execute(inst)
        self.assertEqual(state.vgprs[7], 0xDEADBEEF)


class WavefrontState:
    """SIMD wavefront state modeling lane divergence and 32/64-bit EXEC masking."""

    def __init__(self, wave_size: int = 32):
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave size: {wave_size}")
        self.wave_size: int = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.lane_vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.sgprs: list[int] = [0] * 106

    def is_lane_active(self, lane: int) -> bool:
        return bool(self.exec_mask & (1 << lane))

    def execute_wave(self, inst: ASTInstruction) -> None:
        inst.validate()
        for lane in range(self.wave_size):
            if self.is_lane_active(lane):
                if inst.src0.kind == OperandKind.VGPR:
                    src_val = self.lane_vgprs[lane][inst.src0.reg_idx]
                elif inst.src0.kind == OperandKind.SGPR:
                    src_val = self.sgprs[inst.src0.reg_idx]
                else:
                    src_val = inst.src0.raw_bits
                res = ref_rndne_f32(src_val, clamp=inst.clamp)
                self.lane_vgprs[lane][inst.dst.reg_idx] = res


class TestValuRndneF32SIMDDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_full_wave32(self):
        wave = WavefrontState(wave_size=32)
        wave.exec_mask = 0xFFFFFFFF
        test_values = [1.5, 2.5, 3.5, 4.5] * 8
        for lane in range(32):
            wave.lane_vgprs[lane][1] = f32_to_bits(test_values[lane])
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        wave.execute_wave(inst)
        for lane in range(32):
            exp = ref_rndne_f32(f32_to_bits(test_values[lane]))
            self.assertEqual(wave.lane_vgprs[lane][0], exp)

    def test_simd_alternating_lanes_mask(self):
        wave = WavefrontState(wave_size=32)
        wave.exec_mask = 0x55555555
        initial_sentinel = 0xBAADF00D
        for lane in range(32):
            wave.lane_vgprs[lane][0] = initial_sentinel
            wave.lane_vgprs[lane][1] = f32_to_bits(1.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        wave.execute_wave(inst)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.lane_vgprs[lane][0], F32_TWO)
            else:
                self.assertEqual(wave.lane_vgprs[lane][0], initial_sentinel)

    def test_simd_single_active_lane(self):
        wave = WavefrontState(wave_size=32)
        wave.exec_mask = 1 << 15
        sentinel = 0x11223344
        for lane in range(32):
            wave.lane_vgprs[lane][0] = sentinel
            wave.lane_vgprs[lane][1] = f32_to_bits(2.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        wave.execute_wave(inst)
        for lane in range(32):
            if lane == 15:
                self.assertEqual(wave.lane_vgprs[lane][0], F32_TWO)
            else:
                self.assertEqual(wave.lane_vgprs[lane][0], sentinel)

    def test_simd_wave64_divergence(self):
        wave = WavefrontState(wave_size=64)
        wave.exec_mask = 0x00000000FFFFFFFF
        sentinel = 0x55667788
        for lane in range(64):
            wave.lane_vgprs[lane][0] = sentinel
            wave.lane_vgprs[lane][1] = f32_to_bits(-1.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        wave.execute_wave(inst)
        for lane in range(64):
            if lane < 32:
                self.assertEqual(wave.lane_vgprs[lane][0], F32_NEG_TWO)
            else:
                self.assertEqual(wave.lane_vgprs[lane][0], sentinel)


class TestValuRndneF32RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_inplace_destination_aliasing(self):
        state = VALURegisterState()
        state.vgprs[0] = f32_to_bits(1.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(0))
        res = state.execute(inst)
        self.assertEqual(res, F32_TWO)
        self.assertEqual(state.vgprs[0], F32_TWO)

    def test_operand_preservation_distinct_registers(self):
        state = VALURegisterState()
        original_src = f32_to_bits(2.5)
        state.vgprs[5] = original_src
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(5))
        state.execute(inst)
        self.assertEqual(state.vgprs[0], F32_TWO)
        self.assertEqual(state.vgprs[5], original_src)

    def test_repeated_in_place_idempotence(self):
        state = VALURegisterState()
        state.vgprs[2] = f32_to_bits(3.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(2), ASTOperand.vgpr(2))
        res1 = state.execute(inst)
        self.assertEqual(res1, F32_FOUR)
        res2 = state.execute(inst)
        self.assertEqual(res2, F32_FOUR)
        self.assertEqual(state.vgprs[2], F32_FOUR)

    def test_multi_lane_aliased_self_update(self):
        wave = WavefrontState(wave_size=32)
        wave.exec_mask = 0xFFFFFFFF
        for lane in range(32):
            wave.lane_vgprs[lane][4] = f32_to_bits(1.5 if lane % 2 == 0 else 2.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(4), ASTOperand.vgpr(4))
        wave.execute_wave(inst)
        for lane in range(32):
            self.assertEqual(wave.lane_vgprs[lane][4], F32_TWO)


if __name__ == "__main__":
    unittest.main()
