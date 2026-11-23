"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_frexp_mant_f32.

Models AMD RDNA2 / GFX10 Vector ALU (VALU) v_frexp_mant_f32 normalized mantissa extraction semantics:
  - Extracts the normalized floating-point mantissa of a single-precision (binary32) float.
  - Normal input with biased exponent 1..254: mantissa placed in interval [0.5, 1.0)
    by replacing exponent field with 126 (0x3F000000), preserving sign and fractional bits.
  - Zero and subnormals/denormals are flushed to signed zero (+-0.0), preserving sign bit.
  - Infinities and NaNs (biased exponent 255) are passed through unchanged.
  - VOP1 encoding with single source and destination VGPR.
  - VALU operation: scalar condition code (SCC) and vector condition code (VCC) remain untouched.

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


def f32_to_u32(val: float) -> int:
    """Pack Python float as IEEE 754 binary32 uint32 bits."""
    return struct.unpack("<I", struct.pack("<f", val))[0]


def u32_to_f32(bits: int) -> float:
    """Unpack uint32 bits as Python float (IEEE 754 binary32)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def ref_frexp_mant_f32(x: int) -> int:
    """Architectural reference oracle for RDNA2 v_frexp_mant_f32.

    Returns the normalized mantissa in [0.5, 1.0) with biased exponent 126.
    Zero and denormals flush to +-0.0. Infinities and NaNs pass through unchanged.
    """
    x &= MASK32
    e = (x >> 23) & 0xFF
    sgn = x & 0x80000000
    if e == 0:
        return sgn
    elif e == 255:
        return x
    else:
        keep = x & 0x807FFFFF
        return (keep | 0x3F000000) & MASK32


def run_ps_frexp_mant_f32(instructions: Sequence[str]) -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    spvasm = support.translate_lines("ps_mant_f32_test", lines)
    return support.mrt0_words(spvasm)

class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_powers_of_two(self):
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(1.0)), f32_to_u32(0.5))
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(2.0)), f32_to_u32(0.5))
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(4.0)), f32_to_u32(0.5))
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(0.5)), f32_to_u32(0.5))

    def test_fractional_numbers(self):
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(3.0)), f32_to_u32(0.75))
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(1.5)), f32_to_u32(0.75))

    def test_negative_numbers(self):
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(-1.0)), f32_to_u32(-0.5))
        self.assertEqual(ref_frexp_mant_f32(f32_to_u32(-3.0)), f32_to_u32(-0.75))

    def test_translator_basic_execution(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x40000000",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words[0], f32_to_u32(0.5))

    def test_translator_fractional_execution(self):
        words = run_ps_frexp_mant_f32([
            "v_mov_b32 v1, 0x40400000",
            "v_frexp_mant_f32 v0, v1",
        ])
        self.assertEqual(words[0], f32_to_u32(0.75))


class RegType(Enum):
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    literal_val: int = 0

    def __post_init__(self):
        if self.reg_type == RegType.VGPR and not (0 <= self.index <= 255):
            raise ValueError(f"VGPR index {self.index} out of bounds [0, 255]")
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 105):
            raise ValueError(f"SGPR index {self.index} out of bounds [0, 105]")

    @classmethod
    def vgpr(cls, idx: int) -> Operand:
        return cls(RegType.VGPR, index=idx)

    @classmethod
    def sgpr(cls, idx: int) -> Operand:
        return cls(RegType.SGPR, index=idx)

    @classmethod
    def literal(cls, val: int) -> Operand:
        return cls(RegType.LITERAL, literal_val=val & MASK32)

    def to_asm(self) -> str:
        if self.reg_type == RegType.VGPR:
            return f"v{self.index}"
        elif self.reg_type == RegType.SGPR:
            return f"s{self.index}"
        else:
            return f"{self.literal_val:#x}"


@dataclass(frozen=True)
class VFrexpMantF32Instruction:
    dst: Operand
    src0: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination must be a VGPR for VALU instruction v_frexp_mant_f32")

    def to_asm(self) -> str:
        return f"v_frexp_mant_f32 {self.dst.to_asm()}, {self.src0.to_asm()}"


class TestSymbolicAST(unittest.TestCase):
    def test_ast_construction_and_formatting(self):
        insn = VFrexpMantF32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
        )
        self.assertEqual(insn.to_asm(), "v_frexp_mant_f32 v0, v1")

    def test_ast_immutability(self):
        insn = VFrexpMantF32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
        )
        with self.assertRaises(Exception):
            insn.dst = Operand.vgpr(2)  # type: ignore

    def test_ast_validation_errors(self):
        with self.assertRaises(TypeError):
            VFrexpMantF32Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(1),
            )
        with self.assertRaises(ValueError):
            Operand.vgpr(256)


class ScalarRegisterState:
    def __init__(self):
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106
        self.vcc = 0
        self.scc = 0

    def read_operand(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            return self.vgprs[op.index]
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index]
        else:
            return op.literal_val

    def write_vgpr(self, idx: int, val: int):
        self.vgprs[idx] = val & MASK32


class ScalarInterpreter:
    def __init__(self, state: Optional[ScalarRegisterState] = None):
        self.state = state if state is not None else ScalarRegisterState()

    def step(self, insn: VFrexpMantF32Instruction) -> int:
        u0 = self.state.read_operand(insn.src0)
        res = ref_frexp_mant_f32(u0)
        self.state.write_vgpr(insn.dst.index, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step(self):
        interp = ScalarInterpreter()
        interp.state.vgprs[1] = f32_to_u32(4.0)

        insn = VFrexpMantF32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
        )
        res = interp.step(insn)
        self.assertEqual(res, f32_to_u32(0.5))
        self.assertEqual(interp.state.vgprs[0], f32_to_u32(0.5))

    def test_scalar_interpreter_chaining(self):
        interp = ScalarInterpreter()
        interp.state.vgprs[1] = f32_to_u32(8.0)

        # Step 1: 8.0 -> 0.5
        insn1 = VFrexpMantF32Instruction(dst=Operand.vgpr(0), src0=Operand.vgpr(1))
        interp.step(insn1)
        self.assertEqual(interp.state.vgprs[0], f32_to_u32(0.5))

        # Step 2: 0.5 -> 0.5 (idempotent for normalized mantissas in [0.5, 1.0))
        insn2 = VFrexpMantF32Instruction(dst=Operand.vgpr(2), src0=Operand.vgpr(0))
        res2 = interp.step(insn2)
        self.assertEqual(res2, f32_to_u32(0.5))
        self.assertEqual(interp.state.vgprs[2], f32_to_u32(0.5))

    def test_scalar_interpreter_matches_translator(self):
        lines = [
            "v_mov_b32 v1, 0x41000000",  # 8.0
            "v_frexp_mant_f32 v0, v1",
            "v_frexp_mant_f32 v2, v0",
        ]
        words = run_ps_frexp_mant_f32(lines)
        self.assertEqual(words[0], f32_to_u32(0.5))


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
            return self.vgprs[lane][op.index]
        elif op.reg_type == RegType.SGPR:
            return self.sgprs[op.index]
        else:
            return op.literal_val

    def write_vgpr(self, lane: int, idx: int, val: int):
        self.vgprs[lane][idx] = val & MASK32


class SIMDInterpreter:
    @staticmethod
    def step(insn: VFrexpMantF32Instruction, wave: SIMDWaveState):
        for lane in range(wave.num_lanes):
            if not wave.is_lane_active(lane):
                continue
            u0 = wave.read_operand(lane, insn.src0)
            res = ref_frexp_mant_f32(u0)
            wave.write_vgpr(lane, insn.dst.index, res)


class TestSIMDDivergence(unittest.TestCase):
    def test_simd_wave32_even_lanes_active(self):
        wave = SIMDWaveState(WaveSize.WAVE32)
        wave.exec_mask = 0x55555555  # Even lanes active

        for lane in range(32):
            wave.vgprs[lane][1] = f32_to_u32(2.0)
            wave.vgprs[lane][0] = 0xDEADBEEF  # Sentinel

        insn = VFrexpMantF32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
        )
        SIMDInterpreter.step(insn, wave)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][0], f32_to_u32(0.5))
            else:
                self.assertEqual(wave.vgprs[lane][0], 0xDEADBEEF)

    def test_simd_wave64_odd_lanes_active(self):
        wave = SIMDWaveState(WaveSize.WAVE64)
        wave.exec_mask = 0xAAAAAAAAAAAAAAAA  # Odd lanes active

        for lane in range(64):
            wave.vgprs[lane][1] = f32_to_u32(3.0)
            wave.vgprs[lane][0] = 0xCAFEBABE

        insn = VFrexpMantF32Instruction(
            dst=Operand.vgpr(0),
            src0=Operand.vgpr(1),
        )
        SIMDInterpreter.step(insn, wave)

        for lane in range(64):
            if lane % 2 == 1:
                self.assertEqual(wave.vgprs[lane][0], f32_to_u32(0.75))
            else:
                self.assertEqual(wave.vgprs[lane][0], 0xCAFEBABE)


if __name__ == "__main__":
    unittest.main()
