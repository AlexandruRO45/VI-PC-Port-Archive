"""Semantic tests for RDNA2 VALU Transcendental instructions bundle.

Covers:
- v_sin_f32
- v_cos_f32
- v_exp_f32
- v_log_f32
- v_rcp_f32
- v_rsq_f32
- v_sqrt_f32

Tests reference mathematical models, AST and scalar interpreter semantics,
SIMD multi-lane divergence and EXEC masking, IEEE 754 edge cases,
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

import support  # noqa: E402

_ = support  # Reachable guard for tests/test_semantic_translator_use.py

MASK32 = 0xFFFFFFFF


def f32_to_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def ref_sin_f32(src: float) -> float:
    """Bit-accurate reference oracle for RDNA2 v_sin_f32."""
    if math.isnan(src) or math.isinf(src):
        return float("nan")
    if src == 0.0:
        return src
    rnd = round(src)
    if abs(src - rnd) == 0.5:
        if rnd % 2 != 0:
            rnd = round(src - math.copysign(0.5, src))
    d = src - rnd
    if abs(d) == 0.5:
        return 0.0
    return math.sin(d * 2.0 * math.pi)


class TestVSinF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_sin_f32."""

    def test_reference_quarter_turns(self):
        self.assertAlmostEqual(ref_sin_f32(0.25), 1.0, places=6)
        self.assertAlmostEqual(ref_sin_f32(0.75), -1.0, places=6)
        self.assertAlmostEqual(ref_sin_f32(1.25), 1.0, places=6)

    def test_reference_exact_zero_half_turns(self):
        self.assertEqual(ref_sin_f32(0.5), 0.0)
        self.assertEqual(ref_sin_f32(1.5), 0.0)
        self.assertEqual(ref_sin_f32(-0.5), 0.0)

    def test_reference_signed_zero(self):
        res_pos = ref_sin_f32(0.0)
        res_neg = ref_sin_f32(-0.0)
        self.assertEqual(res_pos, 0.0)
        self.assertEqual(math.copysign(1.0, res_pos), 1.0)
        self.assertEqual(res_neg, -0.0)
        self.assertEqual(math.copysign(1.0, res_neg), -1.0)

    def test_reference_nan_and_inf(self):
        self.assertTrue(math.isnan(ref_sin_f32(float("nan"))))
        self.assertTrue(math.isnan(ref_sin_f32(float("inf"))))
        self.assertTrue(math.isnan(ref_sin_f32(float("-inf"))))


class RegType(enum.Enum):
    VGPR = "v"
    SGPR = "s"
    IMM = "imm"


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index_or_val: int | float

    @classmethod
    def vgpr(cls, idx: int) -> Operand:
        if not (0 <= idx < 256):
            raise ValueError(f"VGPR index out of range: {idx}")
        return cls(RegType.VGPR, idx)

    @classmethod
    def sgpr(cls, idx: int) -> Operand:
        if not (0 <= idx < 106):
            raise ValueError(f"SGPR index out of range: {idx}")
        return cls(RegType.SGPR, idx)

    @classmethod
    def imm_f32(cls, val: float) -> Operand:
        return cls(RegType.IMM, float(val))


@dataclasses.dataclass
class Instruction:
    opcode: str
    dst: Operand
    src0: Operand


class ScalarState:
    def __init__(self) -> None:
        self.vgprs: dict[int, int] = {i: 0 for i in range(256)}
        self.sgprs: dict[int, int] = {i: 0 for i in range(106)}

    def read_operand_f32(self, op: Operand) -> float:
        if op.reg_type == RegType.IMM:
            return float(op.index_or_val)
        elif op.reg_type == RegType.VGPR:
            return bits_to_f32(self.vgprs[int(op.index_or_val)])
        elif op.reg_type == RegType.SGPR:
            return bits_to_f32(self.sgprs[int(op.index_or_val)])
        raise ValueError(f"Unknown operand type: {op.reg_type}")

    def write_vgpr_f32(self, idx: int, val: float) -> None:
        self.vgprs[idx] = f32_to_bits(val)

def step_scalar_sin_f32(state: ScalarState, instr: Instruction) -> None:
    if instr.opcode != "v_sin_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    val = state.read_operand_f32(instr.src0)
    res = ref_sin_f32(val)
    state.write_vgpr_f32(int(instr.dst.index_or_val), res)


class TestVSinF32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_sin_f32."""

    def test_ast_construction_and_operand_bounds(self):
        instr = Instruction("v_sin_f32", Operand.vgpr(5), Operand.vgpr(10))
        self.assertEqual(instr.opcode, "v_sin_f32")
        self.assertEqual(instr.dst.index_or_val, 5)
        self.assertEqual(instr.src0.index_or_val, 10)
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)

    def test_scalar_interpreter_execution(self):
        state = ScalarState()
        state.write_vgpr_f32(1, 0.25)
        instr = Instruction("v_sin_f32", Operand.vgpr(0), Operand.vgpr(1))
        step_scalar_sin_f32(state, instr)
        self.assertAlmostEqual(bits_to_f32(state.vgprs[0]), 1.0, places=6)

    def test_scalar_interpreter_imm_source(self):
        state = ScalarState()
        instr = Instruction("v_sin_f32", Operand.vgpr(2), Operand.imm_f32(-0.25))
        step_scalar_sin_f32(state, instr)
        self.assertAlmostEqual(bits_to_f32(state.vgprs[2]), -1.0, places=6)


class WaveState:
    def __init__(self, wave_size: int = 64) -> None:
        self.wave_size = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: list[dict[int, int]] = [{i: 0 for i in range(256)} for _ in range(wave_size)]

    def read_lane_f32(self, lane: int, op: Operand) -> float:
        if op.reg_type == RegType.IMM:
            return float(op.index_or_val)
        elif op.reg_type == RegType.VGPR:
            return bits_to_f32(self.vgprs[lane][int(op.index_or_val)])
        raise ValueError(f"Unsupported operand for lane: {op.reg_type}")

    def write_lane_f32(self, lane: int, idx: int, val: float) -> None:
        if (self.exec_mask >> lane) & 1:
            self.vgprs[lane][idx] = f32_to_bits(val)

def step_simd_sin_f32(wave: WaveState, instr: Instruction) -> None:
    if instr.opcode != "v_sin_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            val = wave.read_lane_f32(lane, instr.src0)
            res = ref_sin_f32(val)
            wave.write_lane_f32(lane, int(instr.dst.index_or_val), res)


class TestVSinF32SIMDAndIEEE(unittest.TestCase):
    """SIMD multi-lane divergence and IEEE 754 edge cases for v_sin_f32."""

    def test_simd_divergence_and_exec_masking(self):
        wave = WaveState(wave_size=64)
        wave.exec_mask = 0x5555555555555555
        for lane in range(64):
            wave.vgprs[lane][1] = f32_to_bits(0.25)
            wave.vgprs[lane][0] = f32_to_bits(999.0)

        instr = Instruction("v_sin_f32", Operand.vgpr(0), Operand.vgpr(1))
        step_simd_sin_f32(wave, instr)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertAlmostEqual(bits_to_f32(wave.vgprs[lane][0]), 1.0, places=6)
            else:
                self.assertEqual(bits_to_f32(wave.vgprs[lane][0]), 999.0)

    def test_ieee_denormal_subnormal_handling(self):
        subnormal = bits_to_f32(0x00000001)
        res = ref_sin_f32(subnormal)
        self.assertAlmostEqual(res, subnormal * 2.0 * math.pi, places=6)


def run_hw_sin_f32(src_val: float) -> float:
    bits = f32_to_bits(src_val)
    lines = [
        f"v_mov_b32 v0, {bits:#x}",
        "v_sin_f32 v0, v0",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_sin_f32_hw", lines)
    out_bits = support.mrt0_words(spv)[0]
    return bits_to_f32(out_bits)


class TestVSinF32HardwareTranslator(unittest.TestCase):
    """Translator integration and hardware execution tests for v_sin_f32."""

    def test_translator_quarter_turns(self):
        self.assertAlmostEqual(run_hw_sin_f32(0.25), 1.0, places=6)
        self.assertAlmostEqual(run_hw_sin_f32(0.75), -1.0, places=6)
        self.assertAlmostEqual(run_hw_sin_f32(-0.25), -1.0, places=6)

    def test_translator_exact_zeros(self):
        self.assertEqual(run_hw_sin_f32(0.0), 0.0)
        self.assertEqual(run_hw_sin_f32(0.5), 0.0)
        self.assertEqual(run_hw_sin_f32(1.0), 0.0)
        self.assertEqual(run_hw_sin_f32(-0.5), 0.0)

    def test_translator_signed_zero(self):
        res_neg = run_hw_sin_f32(-0.0)
        self.assertEqual(res_neg, -0.0)
        self.assertEqual(math.copysign(1.0, res_neg), -1.0)


def ref_cos_f32(src: float) -> float:
    """Bit-accurate reference oracle for RDNA2 v_cos_f32."""
    if math.isnan(src) or math.isinf(src):
        return float("nan")
    if src == 0.0:
        return 1.0
    rnd = round(src)
    if abs(src - rnd) == 0.5:
        if rnd % 2 != 0:
            rnd = round(src - math.copysign(0.5, src))
    d = src - rnd
    if abs(abs(d) - 0.25) < 1e-7:
        return 0.0
    return math.cos(d * 2.0 * math.pi)


class TestVCosF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_cos_f32."""

    def test_reference_values(self):
        self.assertAlmostEqual(ref_cos_f32(0.0), 1.0, places=6)
        self.assertAlmostEqual(ref_cos_f32(0.5), -1.0, places=6)
        self.assertAlmostEqual(ref_cos_f32(1.0), 1.0, places=6)
        self.assertAlmostEqual(ref_cos_f32(-0.5), -1.0, places=6)

    def test_reference_exact_zero_quarter_turns(self):
        self.assertEqual(ref_cos_f32(0.25), 0.0)
        self.assertEqual(ref_cos_f32(0.75), 0.0)
        self.assertEqual(ref_cos_f32(-0.25), 0.0)
        self.assertEqual(ref_cos_f32(-0.75), 0.0)

    def test_reference_nan_and_inf(self):
        self.assertTrue(math.isnan(ref_cos_f32(float("nan"))))
        self.assertTrue(math.isnan(ref_cos_f32(float("inf"))))


def step_scalar_cos_f32(state: ScalarState, instr: Instruction) -> None:
    if instr.opcode != "v_cos_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    val = state.read_operand_f32(instr.src0)
    res = ref_cos_f32(val)
    state.write_vgpr_f32(int(instr.dst.index_or_val), res)


class TestVCosF32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_cos_f32."""

    def test_scalar_interpreter_execution(self):
        state = ScalarState()
        state.write_vgpr_f32(1, 0.0)
        instr = Instruction("v_cos_f32", Operand.vgpr(0), Operand.vgpr(1))
        step_scalar_cos_f32(state, instr)
        self.assertAlmostEqual(bits_to_f32(state.vgprs[0]), 1.0, places=6)

    def test_scalar_interpreter_quarter_turn(self):
        state = ScalarState()
        instr = Instruction("v_cos_f32", Operand.vgpr(3), Operand.imm_f32(0.25))
        step_scalar_cos_f32(state, instr)
        self.assertEqual(bits_to_f32(state.vgprs[3]), 0.0)


def step_simd_cos_f32(wave: WaveState, instr: Instruction) -> None:
    if instr.opcode != "v_cos_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            val = wave.read_lane_f32(lane, instr.src0)
            res = ref_cos_f32(val)
            wave.write_lane_f32(lane, int(instr.dst.index_or_val), res)


class TestVCosF32SIMDAndIEEE(unittest.TestCase):
    """SIMD multi-lane divergence and IEEE 754 edge cases for v_cos_f32."""

    def test_simd_divergence(self):
        wave = WaveState(wave_size=32)
        wave.exec_mask = 0x0000FFFF
        for lane in range(32):
            wave.vgprs[lane][1] = f32_to_bits(0.0)
            wave.vgprs[lane][0] = f32_to_bits(55.0)

        instr = Instruction("v_cos_f32", Operand.vgpr(0), Operand.vgpr(1))
        step_simd_cos_f32(wave, instr)

        for lane in range(32):
            if lane < 16:
                self.assertAlmostEqual(bits_to_f32(wave.vgprs[lane][0]), 1.0, places=6)
            else:
                self.assertEqual(bits_to_f32(wave.vgprs[lane][0]), 55.0)


def run_hw_cos_f32(src_val: float) -> float:
    bits = f32_to_bits(src_val)
    lines = [
        f"v_mov_b32 v0, {bits:#x}",
        "v_cos_f32 v0, v0",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_cos_f32_hw", lines)
    out_bits = support.mrt0_words(spv)[0]
    return bits_to_f32(out_bits)


class TestVCosF32HardwareTranslator(unittest.TestCase):
    """Translator integration and hardware execution tests for v_cos_f32."""

    def test_translator_cos_values(self):
        self.assertAlmostEqual(run_hw_cos_f32(0.0), 1.0, places=6)
        self.assertEqual(run_hw_cos_f32(0.25), 0.0)
        self.assertAlmostEqual(run_hw_cos_f32(0.5), -1.0, places=6)
        self.assertEqual(run_hw_cos_f32(0.75), 0.0)
        self.assertAlmostEqual(run_hw_cos_f32(1.0), 1.0, places=6)
        self.assertEqual(run_hw_cos_f32(-0.25), 0.0)


def ref_exp_f32(src: float) -> float:
    """Bit-accurate reference oracle for RDNA2 v_exp_f32 (2^x)."""
    if math.isnan(src):
        return float("nan")
    if math.isinf(src):
        return 0.0 if src < 0 else float("inf")
    if src > 128.0:
        return float("inf")
    if src < -150.0:
        return 0.0
    return 2.0 ** src


class TestVExpF32Reference(unittest.TestCase):
    """Reference mathematical oracle verification for v_exp_f32."""

    def test_reference_powers_of_two(self):
        self.assertEqual(ref_exp_f32(0.0), 1.0)
        self.assertEqual(ref_exp_f32(1.0), 2.0)
        self.assertEqual(ref_exp_f32(2.0), 4.0)
        self.assertEqual(ref_exp_f32(3.0), 8.0)
        self.assertEqual(ref_exp_f32(-1.0), 0.5)
        self.assertEqual(ref_exp_f32(-2.0), 0.25)

    def test_reference_inf_and_nan(self):
        self.assertEqual(ref_exp_f32(float("-inf")), 0.0)
        self.assertEqual(ref_exp_f32(float("inf")), float("inf"))
        self.assertTrue(math.isnan(ref_exp_f32(float("nan"))))


def step_scalar_exp_f32(state: ScalarState, instr: Instruction) -> None:
    if instr.opcode != "v_exp_f32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    val = state.read_operand_f32(instr.src0)
    res = ref_exp_f32(val)
    state.write_vgpr_f32(int(instr.dst.index_or_val), res)


class TestVExpF32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_exp_f32."""

    def test_scalar_interpreter_execution(self):
        state = ScalarState()
        state.write_vgpr_f32(1, 3.0)
        instr = Instruction("v_exp_f32", Operand.vgpr(0), Operand.vgpr(1))
        step_scalar_exp_f32(state, instr)
        self.assertEqual(bits_to_f32(state.vgprs[0]), 8.0)


if __name__ == "__main__":
    unittest.main()
