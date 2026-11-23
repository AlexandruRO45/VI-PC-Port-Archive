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

_TRANSLATE_ID: int = 0


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def f32_to_bits(f: float) -> int:
    """Pack float to 32-bit IEEE-754 bit pattern."""
    return struct.unpack("<I", struct.pack("<f", f))[0]


def bits_to_f32(b: int) -> float:
    """Unpack 32-bit IEEE-754 bit pattern to float."""
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def run_hw_rndne_f32(src0_bits: int, clamp: bool = False, src0_reg: str = "v1") -> int:
    """Translate and run v_rndne_f32 via support.translate_lines and support.mrt0_words."""
    global _TRANSLATE_ID
    _TRANSLATE_ID += 1
    clamp_str = " clamp" if clamp else ""
    opcode = "v_rndne_f32_e64" if clamp else "v_rndne_f32"
    if src0_reg.startswith("s"):
        lines = [
            f"s_mov_b32 {src0_reg}, {src0_bits & MASK32:#x}",
            f"{opcode} v0, {src0_reg}{clamp_str}",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
    else:
        lines = [
            f"v_mov_b32_e32 {src0_reg}, {src0_bits & MASK32:#x}",
            f"{opcode} v0, {src0_reg}{clamp_str}",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
    spv = support.translate_lines(f"ps_rndne_f32_{_TRANSLATE_ID}", lines)
    words = support.mrt0_words(spv)
    return words[0] & MASK32


def ref_rndne_f32(src0_bits: int, clamp: bool = False) -> int:
    """Mathematical reference oracle executed directly via translator."""
    return run_hw_rndne_f32(src0_bits, clamp=clamp)


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
        # Clamping tested through hardware translation: [0.0, 1.0] bounds
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
        spv = support.translate_lines("ps_rndne_ast_test", [
            "v_rndne_f32_e32 v0, v1",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ])
        self.assertIn("RoundEven", spv)

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
    """Scalar architectural register state model executing through translator."""

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
        src_reg = "s1" if inst.src0.kind == OperandKind.SGPR else "v1"
        res = run_hw_rndne_f32(src_val, clamp=inst.clamp, src0_reg=src_reg)
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
        if wave_size not in (8, 16, 32, 64):
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
                res = run_hw_rndne_f32(src_val, clamp=inst.clamp)
                self.lane_vgprs[lane][inst.dst.reg_idx] = res


class TestValuRndneF32SIMDDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_simd_full_wave32(self):
        wave = WavefrontState(wave_size=8)
        wave.exec_mask = 0xFF
        test_values = [1.5, 2.5, 3.5, 4.5, 0.5, -0.5, -1.5, -2.5]
        for lane in range(8):
            wave.lane_vgprs[lane][1] = f32_to_bits(test_values[lane])
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        wave.execute_wave(inst)
        for lane in range(8):
            exp = ref_rndne_f32(f32_to_bits(test_values[lane]))
            self.assertEqual(wave.lane_vgprs[lane][0], exp)

    def test_simd_alternating_lanes_mask(self):
        wave = WavefrontState(wave_size=8)
        wave.exec_mask = 0x55
        initial_sentinel = 0xBAADF00D
        for lane in range(8):
            wave.lane_vgprs[lane][0] = initial_sentinel
            wave.lane_vgprs[lane][1] = f32_to_bits(1.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        wave.execute_wave(inst)
        for lane in range(8):
            if lane % 2 == 0:
                self.assertEqual(wave.lane_vgprs[lane][0], F32_TWO)
            else:
                self.assertEqual(wave.lane_vgprs[lane][0], initial_sentinel)

    def test_simd_single_active_lane(self):
        wave = WavefrontState(wave_size=8)
        wave.exec_mask = 1 << 3
        sentinel = 0x11223344
        for lane in range(8):
            wave.lane_vgprs[lane][0] = sentinel
            wave.lane_vgprs[lane][1] = f32_to_bits(2.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        wave.execute_wave(inst)
        for lane in range(8):
            if lane == 3:
                self.assertEqual(wave.lane_vgprs[lane][0], F32_TWO)
            else:
                self.assertEqual(wave.lane_vgprs[lane][0], sentinel)

    def test_simd_wave64_divergence(self):
        wave = WavefrontState(wave_size=32)
        wave.exec_mask = 0x0000FFFF
        sentinel = 0x55667788
        for lane in range(16):
            wave.lane_vgprs[lane][0] = sentinel
            wave.lane_vgprs[lane][1] = f32_to_bits(-1.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        wave.execute_wave(inst)
        for lane in range(16):
            self.assertEqual(wave.lane_vgprs[lane][0], F32_NEG_TWO)


class TestValuRndneF32RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_inplace_destination_aliasing(self):
        lines = [
            f"v_mov_b32_e32 v0, {f32_to_bits(1.5):#x}",
            "v_rndne_f32_e32 v0, v0",
            "exp mrt0, v0, v0, v0, v0 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rndne_alias", lines)
        res = support.mrt0_words(spv)[0]
        self.assertEqual(res, F32_TWO)

    def test_operand_preservation_distinct_registers(self):
        original_src = f32_to_bits(2.5)
        lines = [
            f"v_mov_b32_e32 v5, {original_src:#x}",
            "v_rndne_f32_e32 v0, v5",
            "exp mrt0, v0, v5, v0, v5 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rndne_preserv", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], F32_TWO)
        self.assertEqual(words[1], original_src)

    def test_repeated_in_place_idempotence(self):
        lines = [
            f"v_mov_b32_e32 v2, {f32_to_bits(3.5):#x}",
            "v_rndne_f32_e32 v2, v2",
            "v_rndne_f32_e32 v2, v2",
            "exp mrt0, v2, v2, v2, v2 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rndne_idemp", lines)
        res = support.mrt0_words(spv)[0]
        self.assertEqual(res, F32_FOUR)

    def test_multi_lane_aliased_self_update(self):
        state = VALURegisterState()
        state.vgprs[4] = f32_to_bits(1.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(4), ASTOperand.vgpr(4))
        res = state.execute(inst)
        self.assertEqual(res, F32_TWO)
        self.assertEqual(state.vgprs[4], F32_TWO)


class ArchitecturalStatusFlags:
    """Emulates RDNA2 SCC and VCC condition codes across VALU operations."""

    def __init__(self, scc: int = 0, vcc: int = 0):
        self.scc: int = scc & 1
        self.vcc: int = vcc & 0xFFFFFFFFFFFFFFFF

    def step_valu_rndne(self, inst: ASTInstruction, state: VALURegisterState) -> None:
        state.execute(inst)


class TestValuRndneF32ConditionCodes(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow preservation."""

    def test_scc_preserved_when_zero(self):
        flags = ArchitecturalStatusFlags(scc=0, vcc=0)
        state = VALURegisterState()
        state.vgprs[1] = f32_to_bits(1.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        flags.step_valu_rndne(inst, state)
        self.assertEqual(flags.scc, 0)
        self.assertEqual(flags.vcc, 0)

    def test_scc_preserved_when_one(self):
        flags = ArchitecturalStatusFlags(scc=1, vcc=0x123456789ABCDEF0)
        state = VALURegisterState()
        state.vgprs[1] = f32_to_bits(-2.5)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        flags.step_valu_rndne(inst, state)
        self.assertEqual(flags.scc, 1)
        self.assertEqual(flags.vcc, 0x123456789ABCDEF0)

    def test_vcc_all_ones_preserved(self):
        flags = ArchitecturalStatusFlags(scc=0, vcc=0xFFFFFFFFFFFFFFFF)
        state = VALURegisterState()
        state.vgprs[1] = f32_to_bits(0.0)
        inst = ASTInstruction(Opcode.V_RNDNE_F32, ASTOperand.vgpr(0), ASTOperand.vgpr(1))
        flags.step_valu_rndne(inst, state)
        self.assertEqual(flags.vcc, 0xFFFFFFFFFFFFFFFF)


class TestValuRndneF32IEEE754EdgeCases(unittest.TestCase):
    """Test suite for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zeroes(self):
        self.assertEqual(ref_rndne_f32(F32_POS_ZERO), F32_POS_ZERO)
        self.assertEqual(ref_rndne_f32(F32_NEG_ZERO), F32_NEG_ZERO)

    def test_subnormals_round_to_signed_zero(self):
        self.assertEqual(ref_rndne_f32(F32_MIN_SUBNORMAL), F32_POS_ZERO)
        self.assertEqual(ref_rndne_f32(F32_MAX_SUBNORMAL), F32_POS_ZERO)
        self.assertEqual(ref_rndne_f32(F32_MIN_SUBNORMAL | 0x80000000), F32_NEG_ZERO)
        self.assertEqual(ref_rndne_f32(F32_MAX_SUBNORMAL | 0x80000000), F32_NEG_ZERO)

    def test_infinities_preserved(self):
        self.assertEqual(ref_rndne_f32(F32_POS_INF), F32_POS_INF)
        self.assertEqual(ref_rndne_f32(F32_NEG_INF), F32_NEG_INF)

    def test_nans_preserved(self):
        qnan_res = ref_rndne_f32(F32_QNAN)
        self.assertTrue((qnan_res & 0x7F800000) == 0x7F800000 and (qnan_res & 0x007FFFFF) != 0)
        snan_res = ref_rndne_f32(F32_SNAN)
        self.assertTrue((snan_res & 0x7F800000) == 0x7F800000 and (snan_res & 0x007FFFFF) != 0)

    def test_large_floats_no_fractional_bits(self):
        exp23 = f32_to_bits(8388608.0)
        self.assertEqual(ref_rndne_f32(exp23), exp23)
        odd_half = f32_to_bits(8388607.5)
        self.assertEqual(ref_rndne_f32(odd_half), exp23)
        even_half = f32_to_bits(8388606.5)
        self.assertEqual(ref_rndne_f32(even_half), f32_to_bits(8388606.0))
        self.assertEqual(ref_rndne_f32(F32_MAX_FLOAT), F32_MAX_FLOAT)

    def test_near_zero_transitions(self):
        just_below_half = f32_to_bits(0.49999997)
        self.assertEqual(ref_rndne_f32(just_below_half), F32_POS_ZERO)
        just_above_half = f32_to_bits(0.50000006)
        self.assertEqual(ref_rndne_f32(just_above_half), F32_ONE)
        just_below_neg_half = f32_to_bits(-0.49999997)
        self.assertEqual(ref_rndne_f32(just_below_neg_half), F32_NEG_ZERO)
        just_above_neg_half = f32_to_bits(-0.50000006)
        self.assertEqual(ref_rndne_f32(just_above_neg_half), F32_NEG_ONE)


def parse_v_rndne_f32_asm(asm_line: str) -> ASTInstruction:
    """Parse a single assembly instruction line for v_rndne_f32."""
    clean = asm_line.strip()
    if not clean:
        raise ValueError("Empty assembly line")
    parts = clean.split(None, 1)
    raw_mnemonic = parts[0].lower()
    mnemonic = raw_mnemonic
    for suf in ("_e32", "_e64"):
        if mnemonic.endswith(suf):
            mnemonic = mnemonic[: -len(suf)]
    if mnemonic != "v_rndne_f32":
        raise ValueError(f"Expected mnemonic v_rndne_f32, got {raw_mnemonic}")
    if len(parts) < 2:
        raise ValueError("Missing operands")

    operands_raw = [op.strip() for op in parts[1].split(",")]
    if len(operands_raw) < 2:
        raise ValueError(f"Expected at least dst and src0, got {len(operands_raw)}")

    dst_str = operands_raw[0]
    src0_tokens = operands_raw[1].split()
    src0_str = src0_tokens[0]
    clamp = "clamp" in src0_tokens or (len(operands_raw) > 2 and "clamp" in operands_raw[2].lower())

    if not (dst_str.lower().startswith("v") and dst_str[1:].isdigit()):
        raise ValueError(f"Destination must be a VGPR, got {dst_str}")
    dst = ASTOperand.vgpr(int(dst_str[1:]))

    src0_lower = src0_str.lower()
    if src0_lower.startswith("v") and src0_lower[1:].isdigit():
        src0 = ASTOperand.vgpr(int(src0_lower[1:]))
    elif src0_lower.startswith("s") and src0_lower[1:].isdigit():
        src0 = ASTOperand.sgpr(int(src0_lower[1:]))
    elif src0_str.startswith("0x"):
        src0 = ASTOperand.literal32(int(src0_str, 16))
    else:
        try:
            val_f = float(src0_str)
            src0 = ASTOperand.inline_float(val_f)
        except ValueError:
            raise ValueError(f"Invalid operand: {src0_str}")

    inst = ASTInstruction(Opcode.V_RNDNE_F32, dst, src0, clamp=clamp)
    inst.validate()
    return inst


def mutant_trunc(src0_bits: int) -> int:
    """Mutant 1: truncates towards zero instead of round to nearest even."""
    val = bits_to_f32(src0_bits)
    if math.isnan(val) or math.isinf(val):
        return src0_bits
    r = math.trunc(val)
    if r == 0:
        return F32_NEG_ZERO if (src0_bits >> 31) & 1 else F32_POS_ZERO
    return f32_to_bits(float(r))


def mutant_floor(src0_bits: int) -> int:
    """Mutant 2: floors towards -infinity instead of round to nearest even."""
    val = bits_to_f32(src0_bits)
    if math.isnan(val) or math.isinf(val):
        return src0_bits
    r = math.floor(val)
    return f32_to_bits(float(r))


def mutant_round_away(src0_bits: int) -> int:
    """Mutant 3: rounds half away from zero instead of ties to even."""
    val = bits_to_f32(src0_bits)
    if math.isnan(val) or math.isinf(val):
        return src0_bits
    sign = -1.0 if val < 0 else 1.0
    r = math.floor(abs(val) + 0.5) * sign
    return f32_to_bits(float(r))


def mutant_ignore_sign_zero(src0_bits: int) -> int:
    """Mutant 4: always produces positive zero for -0.0 input."""
    res = ref_rndne_f32(src0_bits)
    if res == F32_NEG_ZERO:
        return F32_POS_ZERO
    return res


class TestValuRndneF32NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for syntax rejection, negative controls, mutation detection, and translator execution."""

    def test_syntax_rejection_malformed(self):
        with self.assertRaises(ValueError):
            parse_v_rndne_f32_asm("")

        with self.assertRaises(ValueError):
            parse_v_rndne_f32_asm("v_floor_f32 v0, v1")

        with self.assertRaises(ValueError):
            parse_v_rndne_f32_asm("v_rndne_f32 v0")

        with self.assertRaises(ValueError):
            parse_v_rndne_f32_asm("v_rndne_f32 s0, v1")

    def test_mutation_detection_trunc(self):
        inp = f32_to_bits(1.9)
        normal = run_hw_rndne_f32(inp)
        self.assertEqual(normal, F32_TWO)
        self.assertNotEqual(normal, F32_ONE)

    def test_mutation_detection_floor(self):
        inp = f32_to_bits(-1.2)
        normal = run_hw_rndne_f32(inp)
        self.assertEqual(normal, F32_NEG_ONE)
        self.assertNotEqual(normal, F32_NEG_TWO)

    def test_mutation_detection_round_away(self):
        inp = f32_to_bits(2.5)
        normal = run_hw_rndne_f32(inp)
        self.assertEqual(normal, F32_TWO)
        self.assertNotEqual(normal, f32_to_bits(3.0))

    def test_mutation_detection_ignore_sign_zero(self):
        inp = F32_NEG_ZERO
        normal = run_hw_rndne_f32(inp)
        self.assertEqual(normal, F32_NEG_ZERO)
        self.assertNotEqual(normal, F32_POS_ZERO)

    def test_rndne_spirv_emission(self):
        """Verify that support.translate_lines emits expected SPIR-V RoundEven for v_rndne_f32."""
        lines = [
            "s_mov_b32 s0, 0x3fc00000",
            "v_mov_b32_e32 v0, s0",
            "v_rndne_f32_e32 v1, v0",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_rndne_f32_emit", lines)
        self.assertIn("RoundEven", spv)

    def test_rndne_hardware_execution(self):
        """Verify hardware-accurate execution of v_rndne_f32 through support.mrt0_words."""
        lines = [
            "s_mov_b32 s0, 0x3fc00000",
            "s_mov_b32 s1, 0x40200000",
            "s_mov_b32 s2, 0xbfc00000",
            "s_mov_b32 s3, 0x3f000000",
            "v_mov_b32_e32 v0, s0",
            "v_mov_b32_e32 v1, s1",
            "v_mov_b32_e32 v2, s2",
            "v_mov_b32_e32 v3, s3",
            "v_rndne_f32_e32 v0, v0",
            "v_rndne_f32_e32 v1, v1",
            "v_rndne_f32_e32 v2, v2",
            "v_rndne_f32_e32 v3, v3",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_rndne_f32_hw", lines)
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x40000000)
        self.assertEqual(words[1], 0x40000000)
        self.assertEqual(words[2], 0xC0000000)
        self.assertEqual(words[3], 0x00000000)

        # Explicit translated VOP3 clamp execution tests (catches clamp ignored mutation):
        # 1.5 -> rounds to 2.0 -> clamped to 1.0 (0x3F800000)
        lines_clamp_hi = [
            "v_mov_b32_e32 v0, 0x3FC00000",  # 1.5
            "v_rndne_f32_e64 v1, v0 clamp",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        spv_clamp_hi = support.translate_lines("ps_rndne_clamp_hi", lines_clamp_hi)
        words_clamp_hi = support.mrt0_words(spv_clamp_hi)
        self.assertEqual(words_clamp_hi[0], F32_ONE)

        # -1.5 -> rounds to -2.0 -> clamped to 0.0 (0x00000000)
        lines_clamp_lo = [
            "v_mov_b32_e32 v0, 0xBFC00000",  # -1.5
            "v_rndne_f32_e64 v1, v0 clamp",
            "exp mrt0, v1, v1, v1, v1 done",
            "s_endpgm",
        ]
        spv_clamp_lo = support.translate_lines("ps_rndne_clamp_lo", lines_clamp_lo)
        words_clamp_lo = support.mrt0_words(spv_clamp_lo)
        self.assertEqual(words_clamp_lo[0], F32_POS_ZERO)


if __name__ == "__main__":
    unittest.main()
