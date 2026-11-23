"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_min3_f32.

Models IEEE 754-2008 / IEEE 754-2019 binary32 single-precision 3-way minimum and AMD RDNA2 / GFX10
Vector ALU (VALU) v_min3_f32 semantics:
  - Computes the minimum of three 32-bit IEEE 754 floating-point values: D = min(S0, S1, S2).
  - VOP3 / VOP3A encoding supports input modifiers (negation, absolute value, neg-abs) on each source.
  - Supports output clamp modifier (clamp to [0.0, 1.0]).
  - Evaluates with IEEE 754-2008 minNum / GLSL.std.450 NMin NaN skipping:
      * If one or two operands are NaN, the non-NaN operand is chosen.
      * If all three operands are NaN, NaN is returned.
  - Signed zero ordering: -0.0 is strictly less than +0.0 (0x80000000 < 0x00000000).

Synthesizes test listings, invokes tools/rdna2spirv.py via tests/semantic/support.py,
executes translated SPIR-V shaders in tests/semantic/spirv_mini_interp.py, and compares mrt0 outputs.

Standard library Python only (unittest, struct, math, dataclasses, enum), strictly zero game data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math
import os
import struct
import sys
from typing import Optional, Sequence, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
QNAN = 0x7FC00000
SNAN = 0x7F800001
NEG_QNAN = 0xFFC00000
SUBNORM_MIN = 0x00000001
SUBNORM_MAX = 0x007FFFFF
ONE_F32 = 0x3F800000
TWO_F32 = 0x40000000
THREE_F32 = 0x40400000
HALF_F32 = 0x3F000000


def u32_to_f32(val: int) -> float:
    return struct.unpack("<f", struct.pack("<I", val & MASK32))[0]


def f32_to_u32(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def is_nan_u32(val: int) -> bool:
    val &= MASK32
    return (val & 0x7F800000) == 0x7F800000 and (val & 0x007FFFFF) != 0


def is_quiet_nan_u32(val: int) -> bool:
    val &= MASK32
    return (val & 0x7FC00000) == 0x7FC00000


def apply_src_mods(val: int, neg: bool, abs_: bool) -> int:
    val &= MASK32
    if abs_:
        val &= 0x7FFFFFFF
    if neg:
        val ^= 0x80000000
    return val


def ref_min2_f32(u0: int, u1: int) -> int:
    """RDNA2 NMin binary32 minimum between two IEEE-754 bit representations."""
    nan0 = is_nan_u32(u0)
    nan1 = is_nan_u32(u1)
    if nan0 and nan1:
        return QNAN
    if nan0:
        return u1
    if nan1:
        return u0

    # Signed zero comparison: -0.0 is strictly less than +0.0
    if (u0 & 0x7FFFFFFF) == 0 and (u1 & 0x7FFFFFFF) == 0:
        if (u0 & 0x80000000) != 0 or (u1 & 0x80000000) != 0:
            return NEG_ZERO
        return POS_ZERO

    f0 = u32_to_f32(u0)
    f1 = u32_to_f32(u1)
    if f0 < f1:
        return u0
    elif f1 < f0:
        return u1
    else:
        # Equal non-zero floats: preserve u0
        return u0


def ref_min3_f32(
    s0: int,
    s1: int,
    s2: int,
    clamp: bool = False,
    neg: Tuple[bool, bool, bool] = (False, False, False),
    abs_: Tuple[bool, bool, bool] = (False, False, False),
) -> int:
    """Architectural reference oracle for RDNA2 v_min3_f32."""
    u0 = apply_src_mods(s0, neg[0], abs_[0])
    u1 = apply_src_mods(s1, neg[1], abs_[1])
    u2 = apply_src_mods(s2, neg[2], abs_[2])

    res = ref_min2_f32(ref_min2_f32(u0, u1), u2)

    if clamp:
        if is_nan_u32(res):
            return POS_ZERO
        f_res = u32_to_f32(res)
        if f_res < 0.0:
            return POS_ZERO
        elif f_res > 1.0:
            return ONE_F32
        else:
            return res
    return res


def run_ps_min3(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_min3_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_basic_ordering(self):
        # min(1.0, 2.0, 3.0) -> 1.0
        r = ref_min3_f32(ONE_F32, TWO_F32, THREE_F32)
        self.assertEqual(r, ONE_F32)

        # min(3.0, 1.0, 2.0) -> 1.0
        r2 = ref_min3_f32(THREE_F32, ONE_F32, TWO_F32)
        self.assertEqual(r2, ONE_F32)

        # min(3.0, 2.0, 1.0) -> 1.0
        r3 = ref_min3_f32(THREE_F32, TWO_F32, ONE_F32)
        self.assertEqual(r3, ONE_F32)

    def test_translator_basic_execution(self):
        words = run_ps_min3([
            f"v_mov_b32 v0, {ONE_F32:#x}",
            f"v_mov_b32 v1, {TWO_F32:#x}",
            f"v_mov_b32 v2, {THREE_F32:#x}",
            "v_min3_f32 v3, v0, v1, v2",
        ])
        self.assertEqual(words[0], ONE_F32)

    def test_clamp_output_modifier(self):
        # min(2.0, 3.0, 4.0) with clamp -> 1.0
        r = ref_min3_f32(TWO_F32, THREE_F32, f32_to_u32(4.0), clamp=True)
        self.assertEqual(r, ONE_F32)

        words = run_ps_min3([
            f"v_mov_b32 v0, {TWO_F32:#x}",
            f"v_mov_b32 v1, {THREE_F32:#x}",
            f"v_mov_b32 v2, {f32_to_u32(4.0):#x}",
            "v_min3_f32 v3, v0, v1, v2 clamp",
        ])
        self.assertEqual(words[0], ONE_F32)

    def test_input_modifiers(self):
        # min(-|2.0|, 3.0, 4.0) -> -2.0
        v0 = TWO_F32
        v1 = THREE_F32
        v2 = f32_to_u32(4.0)
        expected = f32_to_u32(-2.0)
        r = ref_min3_f32(v0, v1, v2, neg=(True, False, False), abs_=(True, False, False))
        self.assertEqual(r, expected)

        words = run_ps_min3([
            f"v_mov_b32 v0, {v0:#x}",
            f"v_mov_b32 v1, {v1:#x}",
            f"v_mov_b32 v2, {v2:#x}",
            "v_min3_f32 v3, -|v0|, v1, v2",
        ])
        self.assertEqual(words[0], expected)

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    literal_val: int = 0
    neg: bool = False
    abs_: bool = False

    def __post_init__(self):
        if self.reg_type == RegType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of bounds [0, 255]")
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of bounds [0, 105]")

    @classmethod
    def vgpr(cls, idx: int, neg: bool = False, abs_: bool = False) -> Operand:
        return cls(RegType.VGPR, index=idx, neg=neg, abs_=abs_)

    @classmethod
    def sgpr(cls, idx: int, neg: bool = False, abs_: bool = False) -> Operand:
        return cls(RegType.SGPR, index=idx, neg=neg, abs_=abs_)

    @classmethod
    def literal(cls, val: int, neg: bool = False, abs_: bool = False) -> Operand:
        return cls(RegType.LITERAL, literal_val=val & MASK32, neg=neg, abs_=abs_)

    def to_asm(self) -> str:
        if self.reg_type == RegType.VGPR:
            base = f"v{self.index}"
        elif self.reg_type == RegType.SGPR:
            base = f"s{self.index}"
        else:
            base = f"{self.literal_val:#x}"

        if self.abs_:
            base = f"|{base}|"
        if self.neg:
            base = f"-{base}"
        return base


@dataclass(frozen=True)
class VMin3F32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand
    clamp: bool = False

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination must be a VGPR for VALU instruction v_min3_f32")
        if self.dst.neg or self.dst.abs_:
            raise ValueError("Destination cannot have neg or abs modifiers")

    def to_asm(self) -> str:
        s0 = self.src0.to_asm()
        s1 = self.src1.to_asm()
        s2 = self.src2.to_asm()
        clamp_str = " clamp" if self.clamp else ""
        return f"v_min3_f32 {self.dst.to_asm()}, {s0}, {s1}, {s2}{clamp_str}"


class TestSymbolicAST(unittest.TestCase):
    def test_ast_construction_and_formatting(self):
        insn = VMin3F32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0, neg=True, abs_=True),
            src1=Operand.sgpr(4),
            src2=Operand.literal(ONE_F32),
            clamp=True,
        )
        self.assertEqual(insn.to_asm(), f"v_min3_f32 v3, -|v0|, s4, {ONE_F32:#x} clamp")

    def test_ast_immutability(self):
        insn = VMin3F32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        with self.assertRaises(Exception):
            insn.clamp = True  # type: ignore

    def test_ast_validation_errors(self):
        with self.assertRaises(TypeError):
            VMin3F32Instruction(
                dst=Operand.sgpr(3),  # Invalid dst
                src0=Operand.vgpr(0),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(2),
            )
        with self.assertRaises(ValueError):
            Operand.vgpr(300)  # Out of range

class ScalarRegisterState:
    def __init__(self):
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106
        self.vcc = 0
        self.scc = 0

    def read_operand(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            raw = self.vgprs[op.index]
        elif op.reg_type == RegType.SGPR:
            raw = self.sgprs[op.index]
        else:
            raw = op.literal_val
        return apply_src_mods(raw, op.neg, op.abs_)

    def write_vgpr(self, idx: int, val: int):
        self.vgprs[idx] = val & MASK32


class ScalarInterpreter:
    def __init__(self, state: Optional[ScalarRegisterState] = None):
        self.state = state if state is not None else ScalarRegisterState()

    def step(self, insn: VMin3F32Instruction) -> int:
        u0 = self.state.read_operand(insn.src0)
        u1 = self.state.read_operand(insn.src1)
        u2 = self.state.read_operand(insn.src2)
        res = ref_min2_f32(ref_min2_f32(u0, u1), u2)
        if insn.clamp:
            if is_nan_u32(res):
                res = POS_ZERO
            else:
                f_res = u32_to_f32(res)
                if f_res < 0.0:
                    res = POS_ZERO
                elif f_res > 1.0:
                    res = ONE_F32
        self.state.write_vgpr(insn.dst.index, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step_and_chaining(self):
        interp = ScalarInterpreter()
        interp.state.vgprs[0] = f32_to_u32(10.0)
        interp.state.vgprs[1] = f32_to_u32(5.0)
        interp.state.vgprs[2] = f32_to_u32(8.0)
        interp.state.vgprs[4] = f32_to_u32(2.0)
        interp.state.vgprs[5] = f32_to_u32(7.0)

        # Insn 1: v3 = min(v0, v1, v2) -> min(10, 5, 8) = 5
        insn1 = VMin3F32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        res1 = interp.step(insn1)
        self.assertEqual(res1, f32_to_u32(5.0))
        self.assertEqual(interp.state.vgprs[3], f32_to_u32(5.0))

        # Insn 2: v6 = min(v3, v4, v5) -> min(5, 2, 7) = 2
        insn2 = VMin3F32Instruction(
            dst=Operand.vgpr(6),
            src0=Operand.vgpr(3),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr(5),
        )
        res2 = interp.step(insn2)
        self.assertEqual(res2, f32_to_u32(2.0))
        self.assertEqual(interp.state.vgprs[6], f32_to_u32(2.0))

    def test_scalar_interpreter_matches_translator_chain(self):
        lines = [
            f"v_mov_b32 v0, {f32_to_u32(10.0):#x}",
            f"v_mov_b32 v1, {f32_to_u32(5.0):#x}",
            f"v_mov_b32 v2, {f32_to_u32(8.0):#x}",
            f"v_mov_b32 v4, {f32_to_u32(2.0):#x}",
            f"v_mov_b32 v5, {f32_to_u32(7.0):#x}",
            "v_min3_f32 v3, v0, v1, v2",
            "v_min3_f32 v3, v3, v4, v5",
        ]
        words = run_ps_min3(lines)
        self.assertEqual(words[0], f32_to_u32(2.0))

class WaveSize(Enum):
    WAVE32 = 32
    WAVE64 = 64


class SIMDWaveState:
    def __init__(self, size: WaveSize = WaveSize.WAVE32):
        self.size = size
        self.num_lanes = size.value
        self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        self.sgprs = [0] * 106
        self.exec_mask = (1 << self.num_lanes) - 1

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def read_operand(self, lane: int, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            raw = self.vgprs[lane][op.index]
        elif op.reg_type == RegType.SGPR:
            raw = self.sgprs[op.index]
        else:
            raw = op.literal_val
        return apply_src_mods(raw, op.neg, op.abs_)

    def write_vgpr(self, lane: int, idx: int, val: int):
        self.vgprs[lane][idx] = val & MASK32


class SIMDInterpreter:
    @staticmethod
    def step(insn: VMin3F32Instruction, wave: SIMDWaveState):
        for lane in range(wave.num_lanes):
            if not wave.is_lane_active(lane):
                continue
            u0 = wave.read_operand(lane, insn.src0)
            u1 = wave.read_operand(lane, insn.src1)
            u2 = wave.read_operand(lane, insn.src2)
            res = ref_min2_f32(ref_min2_f32(u0, u1), u2)
            if insn.clamp:
                if is_nan_u32(res):
                    res = POS_ZERO
                else:
                    f_res = u32_to_f32(res)
                    if f_res < 0.0:
                        res = POS_ZERO
                    elif f_res > 1.0:
                        res = ONE_F32
            wave.write_vgpr(lane, insn.dst.index, res)


class TestSIMDDivergence(unittest.TestCase):
    def test_wave32_divergence_and_exec_masking(self):
        wave = SIMDWaveState(WaveSize.WAVE32)
        # Checkered EXEC mask (alternating active/inactive lanes)
        wave.exec_mask = 0x55555555

        # Initialize v3 with sentinel across all lanes
        sentinel = 0xDEADBEEF
        for lane in range(32):
            wave.vgprs[lane][3] = sentinel
            wave.vgprs[lane][0] = f32_to_u32(float(lane + 1))
            wave.vgprs[lane][1] = f32_to_u32(float(lane + 2))
            wave.vgprs[lane][2] = f32_to_u32(float(lane + 3))

        insn = VMin3F32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        SIMDInterpreter.step(insn, wave)

        for lane in range(32):
            if lane % 2 == 0:  # Active lane
                expected = f32_to_u32(float(lane + 1))
                self.assertEqual(wave.vgprs[lane][3], expected)
            else:  # Inactive lane preserved
                self.assertEqual(wave.vgprs[lane][3], sentinel)

    def test_wave64_divergence(self):
        wave = SIMDWaveState(WaveSize.WAVE64)
        # Upper 32 lanes inactive
        wave.exec_mask = 0x00000000FFFFFFFF

        sentinel = 0xCAFEBABE
        for lane in range(64):
            wave.vgprs[lane][3] = sentinel
            wave.vgprs[lane][0] = f32_to_u32(1.0)
            wave.vgprs[lane][1] = f32_to_u32(2.0)
            wave.vgprs[lane][2] = f32_to_u32(3.0)

        insn = VMin3F32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        SIMDInterpreter.step(insn, wave)

        for lane in range(32):
            self.assertEqual(wave.vgprs[lane][3], ONE_F32)
        for lane in range(32, 64):
            self.assertEqual(wave.vgprs[lane][3], sentinel)


if __name__ == "__main__":
    unittest.main()
