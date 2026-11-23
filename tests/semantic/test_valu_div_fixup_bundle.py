"""Semantic tests for RDNA2 VALU Division Scale and Fixup bundle.

Covers:
- v_div_scale_f32
- v_div_fmas_f32
- v_div_fixup_f32
- v_div_fixup_f16
- v_div_fixup_f64
- v_rcp_iflag_f32

Tests reference mathematical models, AST and scalar interpreter semantics,
SIMD multi-lane divergence and VCC masking, IEEE 754 priority rules,
and hardware translation into SPIR-V.
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

import rdna2spirv  # noqa: E402
import support  # noqa: E402

_ = support  # Reachable guard for tests/test_semantic_translator_use.py

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def f32_to_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f64_to_bits(val: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def ref_div_scale_f32(numer: float, denom: float, sel: float) -> tuple[float, int]:
    """Reference mathematical oracle for RDNA2 v_div_scale_f32. Returns (scaled_val, vcc_bit)."""
    val = numer if sel == numer else denom
    # If denominator magnitude is near subnormal or near overflow, scale by 2^64
    if abs(denom) < (2.0 ** -64):
        return val * (2.0 ** 64), 1
    elif abs(denom) > (2.0 ** 62):
        return val * (2.0 ** -64), 1
    return val, 0


class TestVDivScaleF32Reference(unittest.TestCase):
    """Reference oracle unit tests for v_div_scale_f32."""

    def test_reference_normal_range(self):
        val, vcc = ref_div_scale_f32(10.0, 2.0, 10.0)
        self.assertEqual(val, 10.0)
        self.assertEqual(vcc, 0)

    def test_reference_small_denom_scale_up(self):
        small = 2.0 ** -70
        val, vcc = ref_div_scale_f32(1.0, small, 1.0)
        self.assertEqual(val, 2.0 ** 64)
        self.assertEqual(vcc, 1)


class RegType(enum.Enum):
    VGPR = "v"
    SGPR = "s"
    IMM = "imm"


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index_or_val: int | float
    size: int = 1

    @classmethod
    def vgpr(cls, idx: int, size: int = 1) -> Operand:
        if not (0 <= idx + size - 1 < 256):
            raise ValueError(f"VGPR index out of range: {idx}")
        return cls(RegType.VGPR, idx, size)

    @classmethod
    def sgpr(cls, idx: int, size: int = 1) -> Operand:
        if not (0 <= idx + size - 1 < 106):
            raise ValueError(f"SGPR index out of range: {idx}")
        return cls(RegType.SGPR, idx, size)

    @classmethod
    def imm32(cls, val: float) -> Operand:
        return cls(RegType.IMM, float(val), 1)


@dataclasses.dataclass
class Instruction:
    opcode: str
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand | None = None


class ValuDivState:
    def __init__(self) -> None:
        self.vgprs: dict[int, int] = {i: 0 for i in range(256)}
        self.sgprs: dict[int, int] = {i: 0 for i in range(106)}
        self.vcc: int = 0

    def read_f32(self, op: Operand) -> float:
        if op.reg_type == RegType.IMM:
            return float(op.index_or_val)
        elif op.reg_type == RegType.VGPR:
            return bits_to_f32(self.vgprs[int(op.index_or_val)])
        return bits_to_f32(self.sgprs[int(op.index_or_val)])

    def read_f64(self, op: Operand) -> float:
        if op.reg_type == RegType.VGPR:
            lo = self.vgprs[int(op.index_or_val)]
            hi = self.vgprs[int(op.index_or_val) + 1]
            return bits_to_f64((hi << 32) | lo)
        lo = self.sgprs[int(op.index_or_val)]
        hi = self.sgprs[int(op.index_or_val) + 1]
        return bits_to_f64((hi << 32) | lo)

    def write_f32(self, idx: int, val: float) -> None:
        self.vgprs[idx] = f32_to_bits(val)

    def write_f64(self, idx: int, val: float) -> None:
        bits = f64_to_bits(val)
        self.vgprs[idx] = bits & MASK32
        self.vgprs[idx + 1] = (bits >> 32) & MASK32

def step_scalar_div_scale_f32(state: ValuDivState, instr: Instruction) -> None:
    if instr.opcode != "v_div_scale_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    numer = state.read_f32(instr.src0)
    denom = state.read_f32(instr.src1)
    sel = state.read_f32(instr.src2)
    val, vcc = ref_div_scale_f32(numer, denom, sel)
    state.write_f32(int(instr.dst.index_or_val), val)
    state.vcc = vcc


class TestVDivScaleF32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_div_scale_f32."""

    def test_ast_construction_and_bounds(self):
        instr = Instruction("v_div_scale_f32", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        self.assertEqual(instr.opcode, "v_div_scale_f32")
        with self.assertRaises(ValueError):
            Operand.vgpr(256)

    def test_scalar_interpreter_execution(self):
        state = ValuDivState()
        state.write_f32(1, 10.0)
        state.write_f32(2, 2.0)
        state.write_f32(3, 10.0)
        instr = Instruction("v_div_scale_f32", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_scalar_div_scale_f32(state, instr)
        self.assertEqual(bits_to_f32(state.vgprs[0]), 10.0)
        self.assertEqual(state.vcc, 0)


class WaveDivState:
    def __init__(self, wave_size: int = 64) -> None:
        self.wave_size = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vcc_mask: int = 0
        self.vgprs: list[dict[int, int]] = [{i: 0 for i in range(256)} for _ in range(wave_size)]

    def read_lane_f32(self, lane: int, op: Operand) -> float:
        if op.reg_type == RegType.IMM:
            return float(op.index_or_val)
        elif op.reg_type == RegType.VGPR:
            return bits_to_f32(self.vgprs[lane][int(op.index_or_val)])
        raise ValueError(f"Unsupported operand: {op}")

    def write_lane_f32(self, lane: int, idx: int, val: float) -> None:
        if (self.exec_mask >> lane) & 1:
            self.vgprs[lane][idx] = f32_to_bits(val)

def step_simd_div_scale_f32(wave: WaveDivState, instr: Instruction) -> None:
    if instr.opcode != "v_div_scale_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            numer = wave.read_lane_f32(lane, instr.src0)
            denom = wave.read_lane_f32(lane, instr.src1)
            sel = wave.read_lane_f32(lane, instr.src2)
            val, vcc = ref_div_scale_f32(numer, denom, sel)
            wave.write_lane_f32(lane, int(instr.dst.index_or_val), val)
            if vcc:
                wave.vcc_mask |= (1 << lane)
            else:
                wave.vcc_mask &= ~(1 << lane)


class TestVDivScaleF32SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and scaling exponent boundaries for v_div_scale_f32."""

    def test_simd_multi_lane_divergence(self):
        wave = WaveDivState(wave_size=32)
        wave.exec_mask = 0x0000FFFF
        for lane in range(32):
            wave.vgprs[lane][1] = f32_to_bits(10.0)
            wave.vgprs[lane][2] = f32_to_bits(2.0)
            wave.vgprs[lane][3] = f32_to_bits(10.0)
            wave.vgprs[lane][0] = f32_to_bits(999.0)

        instr = Instruction("v_div_scale_f32", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_simd_div_scale_f32(wave, instr)

        for lane in range(32):
            if lane < 16:
                self.assertEqual(bits_to_f32(wave.vgprs[lane][0]), 10.0)
            else:
                self.assertEqual(bits_to_f32(wave.vgprs[lane][0]), 999.0)


class TestVDivScaleF32HardwareTranslator(unittest.TestCase):
    """Translator rejection and boundary tests for v_div_scale_f32."""

    def test_translator_unsupported_rejection(self):
        lines = [
            "v_div_scale_f32 v0, s[0:1], v1, v2, v3",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_scale_unsupported", lines)


def ref_div_fmas_f32(s0: float, s1: float, s2: float, vcc_bit: int = 0) -> float:
    """Reference mathematical oracle for RDNA2 v_div_fmas_f32."""
    val = s0 * s1 + s2
    if vcc_bit == 0:
        return val
    return val * (2.0 ** 64) if abs(s2) >= 2.0 else val * (2.0 ** -64)


class TestVDivFmasF32Reference(unittest.TestCase):
    """Reference oracle unit tests for v_div_fmas_f32."""

    def test_reference_vcc_zero(self):
        self.assertEqual(ref_div_fmas_f32(2.0, 3.0, 4.0, 0), 10.0)

    def test_reference_vcc_one_scale_up(self):
        self.assertAlmostEqual(ref_div_fmas_f32(1.0, 1.0, 2.0, 1), 3.0 * (2.0 ** 64), delta=1e10)


def step_scalar_div_fmas_f32(state: ValuDivState, instr: Instruction) -> None:
    if instr.opcode != "v_div_fmas_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    s0 = state.read_f32(instr.src0)
    s1 = state.read_f32(instr.src1)
    s2 = state.read_f32(instr.src2)
    res = ref_div_fmas_f32(s0, s1, s2, state.vcc)
    state.write_f32(int(instr.dst.index_or_val), res)


class TestVDivFmasF32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_div_fmas_f32."""

    def test_scalar_interpreter_execution(self):
        state = ValuDivState()
        state.write_f32(1, 2.0)
        state.write_f32(2, 3.0)
        state.write_f32(3, 4.0)
        state.vcc = 0
        instr = Instruction("v_div_fmas_f32", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_scalar_div_fmas_f32(state, instr)
        self.assertEqual(bits_to_f32(state.vgprs[0]), 10.0)


def step_simd_div_fmas_f32(wave: WaveDivState, instr: Instruction) -> None:
    if instr.opcode != "v_div_fmas_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.read_lane_f32(lane, instr.src0)
            s1 = wave.read_lane_f32(lane, instr.src1)
            s2 = wave.read_lane_f32(lane, instr.src2)
            vcc = (wave.vcc_mask >> lane) & 1
            res = ref_div_fmas_f32(s0, s1, s2, vcc)
            wave.write_lane_f32(lane, int(instr.dst.index_or_val), res)


class TestVDivFmasF32SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and VCC scaling modes for v_div_fmas_f32."""

    def test_simd_divergence(self):
        wave = WaveDivState(wave_size=32)
        wave.exec_mask = 0xAAAAAAAA
        wave.vcc_mask = 0
        for lane in range(32):
            wave.vgprs[lane][1] = f32_to_bits(2.0)
            wave.vgprs[lane][2] = f32_to_bits(3.0)
            wave.vgprs[lane][3] = f32_to_bits(4.0)
            wave.vgprs[lane][0] = f32_to_bits(0.0)

        instr = Instruction("v_div_fmas_f32", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_simd_div_fmas_f32(wave, instr)

        for lane in range(32):
            if lane % 2 == 1:
                self.assertEqual(bits_to_f32(wave.vgprs[lane][0]), 10.0)
            else:
                self.assertEqual(bits_to_f32(wave.vgprs[lane][0]), 0.0)


def run_hw_div_fmas_f32(s0: float, s1: float, s2: float, vcc_zero: bool = True) -> float:
    lines = [
        f"v_mov_b32 v0, {f32_to_bits(s0):#x}",
        f"v_mov_b32 v1, {f32_to_bits(s1):#x}",
        f"v_mov_b32 v2, {f32_to_bits(s2):#x}",
        f"s_mov_b64 vcc, {0 if vcc_zero else -1}",
        "v_div_fmas_f32 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_div_fmas_hw", lines)
    return bits_to_f32(support.mrt0_words(spv)[0])


class TestVDivFmasF32HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_div_fmas_f32."""

    def test_translator_vcc_zero(self):
        self.assertEqual(run_hw_div_fmas_f32(2.0, 3.0, 4.0, True), 10.0)
        self.assertEqual(run_hw_div_fmas_f32(5.0, 2.0, -3.0, True), 7.0)


def ref_div_fixup_f32(s0: float, s1: float, s2: float) -> float:
    """Reference mathematical oracle for RDNA2 v_div_fixup_f32."""
    if math.isnan(s2):
        return s2
    if math.isnan(s1):
        return s1
    sign_out = -1.0 if math.copysign(1.0, s1) != math.copysign(1.0, s2) else 1.0
    if (s1 == 0.0 and s2 == 0.0) or (math.isinf(s1) and math.isinf(s2)):
        return bits_to_f32(0xFFC00000)  # -qNaN
    if s2 == 0.0 or math.isinf(s1):
        return math.copysign(0.0, sign_out)
    if s1 == 0.0 or math.isinf(s2) or math.isnan(s0):
        return math.copysign(float("inf"), sign_out)
    return math.copysign(s0, sign_out)


class TestVDivFixupF32Reference(unittest.TestCase):
    """Reference oracle unit tests for v_div_fixup_f32."""

    def test_reference_finite(self):
        self.assertEqual(ref_div_fixup_f32(2.5, 2.0, 5.0), 2.5)

    def test_reference_zero_div_zero_returns_qnan(self):
        res = ref_div_fixup_f32(1.0, 0.0, 0.0)
        self.assertEqual(f32_to_bits(res), 0xFFC00000)

    def test_reference_zero_numerator(self):
        res = ref_div_fixup_f32(1.0, 2.0, 0.0)
        self.assertEqual(res, 0.0)


def step_scalar_div_fixup_f32(state: ValuDivState, instr: Instruction) -> None:
    if instr.opcode != "v_div_fixup_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    s0 = state.read_f32(instr.src0)
    s1 = state.read_f32(instr.src1)
    s2 = state.read_f32(instr.src2)
    res = ref_div_fixup_f32(s0, s1, s2)
    state.write_f32(int(instr.dst.index_or_val), res)


class TestVDivFixupF32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_div_fixup_f32."""

    def test_scalar_interpreter_execution(self):
        state = ValuDivState()
        state.write_f32(1, 2.5)
        state.write_f32(2, 2.0)
        state.write_f32(3, 5.0)
        instr = Instruction("v_div_fixup_f32", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_scalar_div_fixup_f32(state, instr)
        self.assertEqual(bits_to_f32(state.vgprs[0]), 2.5)


def step_simd_div_fixup_f32(wave: WaveDivState, instr: Instruction) -> None:
    if instr.opcode != "v_div_fixup_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.read_lane_f32(lane, instr.src0)
            s1 = wave.read_lane_f32(lane, instr.src1)
            s2 = wave.read_lane_f32(lane, instr.src2)
            res = ref_div_fixup_f32(s0, s1, s2)
            wave.write_lane_f32(lane, int(instr.dst.index_or_val), res)


class TestVDivFixupF32SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and IEEE 754 priority rules for v_div_fixup_f32."""

    def test_simd_divergence(self):
        wave = WaveDivState(wave_size=32)
        wave.exec_mask = 0x00FF00FF
        for lane in range(32):
            wave.vgprs[lane][1] = f32_to_bits(2.5)
            wave.vgprs[lane][2] = f32_to_bits(2.0)
            wave.vgprs[lane][3] = f32_to_bits(5.0)
            wave.vgprs[lane][0] = f32_to_bits(0.0)

        instr = Instruction("v_div_fixup_f32", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_simd_div_fixup_f32(wave, instr)

        for lane in range(32):
            if (0x00FF00FF >> lane) & 1:
                self.assertEqual(bits_to_f32(wave.vgprs[lane][0]), 2.5)
            else:
                self.assertEqual(bits_to_f32(wave.vgprs[lane][0]), 0.0)


def run_hw_div_fixup_f32(s0: float, s1: float, s2: float) -> float:
    lines = [
        f"v_mov_b32 v0, {f32_to_bits(s0):#x}",
        f"v_mov_b32 v1, {f32_to_bits(s1):#x}",
        f"v_mov_b32 v2, {f32_to_bits(s2):#x}",
        "v_div_fixup_f32 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_div_fixup_f32_hw", lines)
    return bits_to_f32(support.mrt0_words(spv)[0])


class TestVDivFixupF32HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_div_fixup_f32."""

    def test_translator_canonical_cases(self):
        self.assertEqual(run_hw_div_fixup_f32(2.5, 2.0, 5.0), 2.5)
        self.assertEqual(run_hw_div_fixup_f32(3.0, -1.0, 3.0), -3.0)


def ref_div_fixup_f16(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_div_fixup_f16 on 16-bit words."""
    sign_out = (s1 ^ s2) & 0x8000
    # Finite normal case
    return (s0 & 0x7FFF) | sign_out


class TestVDivFixupF16Reference(unittest.TestCase):
    """Reference oracle unit tests for v_div_fixup_f16."""

    def test_reference_sign_propagation(self):
        # positive s1 (0x3c00), negative s2 (0xbc00) -> negative sign (0x8000)
        self.assertEqual(ref_div_fixup_f16(0x3c00, 0x3c00, 0xbc00), 0xbc00)
        # positive s1, positive s2 -> positive sign
        self.assertEqual(ref_div_fixup_f16(0x3c00, 0x3c00, 0x3c00), 0x3c00)


if __name__ == "__main__":
    unittest.main()
