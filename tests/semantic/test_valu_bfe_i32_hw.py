"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_bfe_i32.

Models AMD RDNA2 / GFX10 Vector ALU (VALU) v_bfe_i32 signed bitfield extract semantics:
  - Computes signed bitfield extract: D = (S0.i >> (S1.u & 31)) with bitfield length (S2.u & 31)
  - Extracted field is sign-extended to 32-bit two's-complement integer.
  - If width is 0, the result is 0.
  - VOP3 encoding with VGPR destination and VGPR/SGPR/literal sources.
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
from typing import Optional, Sequence
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


def ref_bfe_i32(base: int, offset: int, width: int) -> int:
    """Architectural reference oracle for RDNA2 v_bfe_i32 signed bitfield extract."""
    base &= MASK32
    offset &= 31
    width &= 31
    if width == 0:
        return 0
    signed = base - (1 << 32) if (base >> 31) else base
    field = (signed >> offset) & ((1 << width) - 1)
    if field & (1 << (width - 1)):
        field -= 1 << width
    return field & MASK32


def run_ps_bfe_i32(instructions: Sequence[str], name: str = "ps_bfe_i32_test") -> list[int]:
    """Translate and execute a synthetic pixel shader, returning mrt0 output words."""
    lines = list(instructions) + ["exp mrt0 v3, v3, v3, v3 done vm", "s_endpgm"]
    spvasm = support.translate_lines(name, lines)
    return support.mrt0_words(spvasm)


class TestReferenceOracleAndTranslator(unittest.TestCase):
    def test_basic_signed_bitfield_extract(self):
        self.assertEqual(ref_bfe_i32(0x12345678, 4, 8), 0x67)
        self.assertEqual(ref_bfe_i32(0xFFFFFFF0, 2, 4), 0xFFFFFFFC)
        self.assertEqual(ref_bfe_i32(0x80000000, 31, 1), 0xFFFFFFFF)
        self.assertEqual(ref_bfe_i32(0x7FFFFFFF, 31, 1), 0)

    def test_width_zero_yields_zero(self):
        self.assertEqual(ref_bfe_i32(0xFFFFFFFF, 0, 0), 0)
        self.assertEqual(ref_bfe_i32(0x12345678, 12, 0), 0)
        self.assertEqual(ref_bfe_i32(0x80000000, 31, 0), 0)

    def test_translator_basic_execution(self):
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 4",
            "v_mov_b32 v2, 8",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_basic")
        self.assertEqual(words[0], 0x67)

    def test_translator_sign_extended_negative(self):
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xFFFFFFF0",
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 4",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_sign_ext_neg")
        self.assertEqual(words[0], 0xFFFFFFFC)

    def test_translator_sign_bit_31_single_bit(self):
        """Extract sign bit from bit 31, sign-extending 1-bit negative to 0xFFFFFFFF."""
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x80000000",
            "v_mov_b32 v1, 31",
            "v_mov_b32 v2, 1",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_bit31_single")
        self.assertEqual(words[0], 0xFFFFFFFF)

    def test_translator_sign_bit_propagation_various_widths(self):
        """Test sign-bit propagation across 4, 8, 16, 17, and 31-bit fields."""
        cases = [
            (0xFFFFFFF8, 0, 4, 0xFFFFFFF8),   # 4-bit field, bit 3 is 1 -> -8
            (0xFFFFFF80, 0, 8, 0xFFFFFF80),   # 8-bit field, bit 7 is 1 -> -128
            (0xFFFF8000, 0, 16, 0xFFFF8000), # 16-bit field, bit 15 is 1 -> -32768
            (0x00010000, 0, 17, 0xFFFF0000), # 17-bit field, bit 16 is 1 -> -65536
            (0x7FFFFFFF, 0, 31, 0xFFFFFFFF), # 31-bit field, bit 30 is 1 -> -1
        ]
        for base, off, wid, expected in cases:
            with self.subTest(base=hex(base), off=off, wid=wid):
                words = run_ps_bfe_i32([
                    f"v_mov_b32 v0, {base:#x}",
                    f"v_mov_b32 v1, {off}",
                    f"v_mov_b32 v2, {wid}",
                    "v_bfe_i32 v3, v0, v1, v2",
                ], name=f"ps_bfe_prop_{off}_{wid}")
                self.assertEqual(words[0], expected)

    def test_translator_width_zero_yields_zero(self):
        """Verify that zero width yields 0 even for negative values."""
        words1 = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xFFFFFFFF",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_w0_all1")
        self.assertEqual(words1[0], 0)

        words2 = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x80000000",
            "v_mov_b32 v1, 31",
            "v_mov_b32 v2, 0",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_w0_b31")
        self.assertEqual(words2[0], 0)

    def test_translator_register_supplied_width_mask_31(self):
        """Verify register-supplied width with 5-bit masking (49 & 31 = 17) and sign extension."""
        # Tyler Straub regression case: width 49 in VGPR masks to 17 bits.
        # Bit 16 of 0x00010000 is 1, so 17-bit extract sign-extends to 0xFFFF0000.
        words_49 = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x00010000",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 49",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_wid49")
        self.assertEqual(words_49[0], 0xFFFF0000)

        # Width 33 in VGPR masks to 1 bit (33 & 31 = 1). Bit 0 is 0 -> 0.
        words_33 = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x00010000",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 33",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_wid33")
        self.assertEqual(words_33[0], 0)

    def test_translator_sgpr_and_imm_operands(self):
        """Verify execution with SGPR and immediate operands for base, offset, and width."""
        words_sgpr = run_ps_bfe_i32([
            "s_mov_b32 s0, 0xFFFFFFF0",
            "s_mov_b32 s1, 2",
            "v_mov_b32 v0, s0",
            "v_bfe_i32 v3, v0, s1, 4",
        ], name="ps_bfe_sgpr_operands")
        self.assertEqual(words_sgpr[0], 0xFFFFFFFC)

        words_lit = run_ps_bfe_i32([
            "s_mov_b32 s2, 4",
            "v_mov_b32 v0, 0xFFFFFFF0",
            "v_bfe_i32 v3, v0, 2, s2",
        ], name="ps_bfe_mixed_imm")
        self.assertEqual(words_lit[0], 0xFFFFFFFC)


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
class VBfeI32Instruction:
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand

    def __post_init__(self):
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError("Destination must be a VGPR for VALU instruction v_bfe_i32")

    def to_asm(self) -> str:
        return f"v_bfe_i32 {self.dst.to_asm()}, {self.src0.to_asm()}, {self.src1.to_asm()}, {self.src2.to_asm()}"


class TestSymbolicAST(unittest.TestCase):
    def test_ast_construction_and_formatting(self):
        insn = VBfeI32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.sgpr(4),
            src2=Operand.literal(8),
        )
        self.assertEqual(insn.to_asm(), "v_bfe_i32 v3, v0, s4, 0x8")

    def test_ast_immutability(self):
        insn = VBfeI32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        with self.assertRaises(Exception):
            insn.dst = Operand.vgpr(4)  # type: ignore

    def test_ast_validation_errors(self):
        with self.assertRaises(TypeError):
            VBfeI32Instruction(
                dst=Operand.sgpr(3),
                src0=Operand.vgpr(0),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(2),
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

    def step(self, insn: VBfeI32Instruction) -> int:
        u0 = self.state.read_operand(insn.src0)
        u1 = self.state.read_operand(insn.src1)
        u2 = self.state.read_operand(insn.src2)
        res = ref_bfe_i32(u0, u1, u2)
        self.state.write_vgpr(insn.dst.index, res)
        return res


class TestScalarInterpreter(unittest.TestCase):
    def test_scalar_interpreter_step_and_chaining(self):
        interp = ScalarInterpreter()
        interp.state.vgprs[0] = 0x12345678
        interp.state.vgprs[1] = 4
        interp.state.vgprs[2] = 8
        interp.state.vgprs[4] = 0
        interp.state.vgprs[5] = 4

        # Step 1: v3 = v_bfe_i32 v0, v1, v2 -> (0x12345678 >> 4) & 0xFF = 0x67
        insn1 = VBfeI32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        res1 = interp.step(insn1)
        self.assertEqual(res1, 0x67)
        self.assertEqual(interp.state.vgprs[3], 0x67)

        # Step 2: v6 = v_bfe_i32 v3, v4, v5 -> (0x67 >> 0) & 0xF = 0x7
        insn2 = VBfeI32Instruction(
            dst=Operand.vgpr(6),
            src0=Operand.vgpr(3),
            src1=Operand.vgpr(4),
            src2=Operand.vgpr(5),
        )
        res2 = interp.step(insn2)
        self.assertEqual(res2, 0x7)
        self.assertEqual(interp.state.vgprs[6], 0x7)

    def test_scalar_interpreter_matches_translator_chain(self):
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 4",
            "v_mov_b32 v2, 8",
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 4",
            "v_bfe_i32 v3, v0, v1, v2",
            "v_bfe_i32 v3, v3, v4, v5",
        ]
        words = run_ps_bfe_i32(lines, name="ps_bfe_chain")
        self.assertEqual(words[0], 0x7)

    def test_scalar_interpreter_signed_negative_chain(self):
        """Cross-check scalar interpreter with translator execution on negative values."""
        interp = ScalarInterpreter()
        interp.state.vgprs[0] = 0xFFFFFFF0
        interp.state.vgprs[1] = 2
        interp.state.vgprs[2] = 4
        insn = VBfeI32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        res = interp.step(insn)
        self.assertEqual(res, 0xFFFFFFFC)

        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xFFFFFFF0",
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 4",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_interp_neg")
        self.assertEqual(words[0], res)


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
    def step(insn: VBfeI32Instruction, wave: SIMDWaveState):
        for lane in range(wave.num_lanes):
            if not wave.is_lane_active(lane):
                continue
            u0 = wave.read_operand(lane, insn.src0)
            u1 = wave.read_operand(lane, insn.src1)
            u2 = wave.read_operand(lane, insn.src2)
            res = ref_bfe_i32(u0, u1, u2)
            wave.write_vgpr(lane, insn.dst.index, res)


class TestSIMDDivergence(unittest.TestCase):
    def test_simd_wave32_divergence_with_exec_mask(self):
        wave = SIMDWaveState(WaveSize.WAVE32)
        wave.exec_mask = 0x55555555  # Even lanes active

        for lane in range(32):
            wave.vgprs[lane][0] = 0xFFFFFFF0
            wave.vgprs[lane][1] = 2
            wave.vgprs[lane][2] = 4
            wave.vgprs[lane][3] = 0xDEADBEEF  # Initial destination

        insn = VBfeI32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        SIMDInterpreter.step(insn, wave)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][3], 0xFFFFFFFC)
            else:
                self.assertEqual(wave.vgprs[lane][3], 0xDEADBEEF)

    def test_simd_wave64_divergence_with_exec_mask(self):
        wave = SIMDWaveState(WaveSize.WAVE64)
        wave.exec_mask = 0xAAAAAAAAAAAAAAAA  # Odd lanes active

        for lane in range(64):
            wave.vgprs[lane][0] = 0x12345678
            wave.vgprs[lane][1] = 4
            wave.vgprs[lane][2] = 8
            wave.vgprs[lane][3] = 0xCAFEBABE

        insn = VBfeI32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        SIMDInterpreter.step(insn, wave)

        for lane in range(64):
            if lane % 2 == 1:
                self.assertEqual(wave.vgprs[lane][3], 0x67)
            else:
                self.assertEqual(wave.vgprs[lane][3], 0xCAFEBABE)


class TestRegisterAliasing(unittest.TestCase):
    def test_alias_dst_equals_src0(self):
        # v0 = v_bfe_i32 v0, v1, v2 (base aliased with dst)
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 4",
            "v_mov_b32 v2, 8",
            "v_bfe_i32 v0, v0, v1, v2",
            "v_mov_b32 v3, v0",
        ], name="ps_alias_dst_s0")
        self.assertEqual(words[0], 0x67)

    def test_alias_dst_equals_src1(self):
        # v1 = v_bfe_i32 v0, v1, v2 (offset aliased with dst)
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 4",
            "v_mov_b32 v2, 8",
            "v_bfe_i32 v1, v0, v1, v2",
            "v_mov_b32 v3, v1",
        ], name="ps_alias_dst_s1")
        self.assertEqual(words[0], 0x67)

    def test_alias_dst_equals_src2(self):
        # v2 = v_bfe_i32 v0, v1, v2 (width aliased with dst)
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 4",
            "v_mov_b32 v2, 8",
            "v_bfe_i32 v2, v0, v1, v2",
            "v_mov_b32 v3, v2",
        ], name="ps_alias_dst_s2")
        self.assertEqual(words[0], 0x67)

    def test_alias_dst_equals_src0_signed_negative(self):
        """v0 = v_bfe_i32 v0, v1, v2 with sign extension into destination."""
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xFFFFFFF0",
            "v_mov_b32 v1, 2",
            "v_mov_b32 v2, 4",
            "v_bfe_i32 v0, v0, v1, v2",
            "v_mov_b32 v3, v0",
        ], name="ps_alias_dst_s0_neg")
        self.assertEqual(words[0], 0xFFFFFFFC)

    def test_alias_dst_equals_src1_signed_negative(self):
        """v1 = v_bfe_i32 v0, v1, v2 with sign extension into offset register."""
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x80000000",
            "v_mov_b32 v1, 31",
            "v_mov_b32 v2, 1",
            "v_bfe_i32 v1, v0, v1, v2",
            "v_mov_b32 v3, v1",
        ], name="ps_alias_dst_s1_neg")
        self.assertEqual(words[0], 0xFFFFFFFF)

    def test_alias_dst_equals_src2_signed_negative(self):
        """v2 = v_bfe_i32 v0, v1, v2 with sign extension into width register."""
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x7FFFFFFF",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 31",
            "v_bfe_i32 v2, v0, v1, v2",
            "v_mov_b32 v3, v2",
        ], name="ps_alias_dst_s2_neg")
        self.assertEqual(words[0], 0xFFFFFFFF)

    def test_alias_quadruple_collision(self):
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 5",
            "v_bfe_i32 v0, v0, v0, v0",
            "v_mov_b32 v3, v0",
        ], name="ps_alias_quad_5")
        self.assertEqual(words[0], ref_bfe_i32(5, 5, 5))

    def test_alias_self_collision_all_negative(self):
        """v0 = v_bfe_i32 v0, v0, v0 where v0 = 0xFFFFFFFF (-1)."""
        # offset = 31 & 31 = 31, width = 31 & 31 = 31.
        # (-1 >> 31) & ((1<<31)-1) = 1. Sign-extended from 31 bits -> 0xFFFFFFFF.
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xFFFFFFFF",
            "v_bfe_i32 v0, v0, v0, v0",
            "v_mov_b32 v3, v0",
        ], name="ps_alias_self_neg")
        self.assertEqual(words[0], 0xFFFFFFFF)


@dataclass
class IntegerStatusFlags:
    zero: bool = False
    negative: bool = False


def classify_integer_status(val: int) -> IntegerStatusFlags:
    val &= MASK32
    return IntegerStatusFlags(
        zero=(val == 0),
        negative=bool(val & 0x80000000),
    )


class TestConditionCodesAndStatus(unittest.TestCase):
    def test_vcc_and_scc_preservation(self):
        interp = ScalarInterpreter()
        interp.state.scc = 1
        interp.state.vcc = 0x123456789ABCDEF0
        interp.state.vgprs[0] = 0x12345678
        interp.state.vgprs[1] = 4
        interp.state.vgprs[2] = 8

        insn = VBfeI32Instruction(
            dst=Operand.vgpr(3),
            src0=Operand.vgpr(0),
            src1=Operand.vgpr(1),
            src2=Operand.vgpr(2),
        )
        interp.step(insn)

        self.assertEqual(interp.state.scc, 1)
        self.assertEqual(interp.state.vcc, 0x123456789ABCDEF0)

    def test_status_flag_classification(self):
        s_zero = classify_integer_status(ref_bfe_i32(0xFFFFFFFF, 0, 0))
        self.assertTrue(s_zero.zero)
        self.assertFalse(s_zero.negative)

        s_neg = classify_integer_status(ref_bfe_i32(0x80000000, 31, 1))
        self.assertFalse(s_neg.zero)
        self.assertTrue(s_neg.negative)

        s_pos = classify_integer_status(ref_bfe_i32(0x12345678, 4, 8))
        self.assertFalse(s_pos.zero)
        self.assertFalse(s_pos.negative)


class TestBoundaryExtremes(unittest.TestCase):
    def test_float_bit_patterns_treated_as_raw_bits(self):
        # IEEE 754 -1.0 is 0xBF800000. Extract sign bit (offset=31, width=1):
        # Bit 31 is 1 -> sign extended -> 0xFFFFFFFF (-1)
        r = ref_bfe_i32(0xBF800000, 31, 1)
        self.assertEqual(r, 0xFFFFFFFF)

        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xBF800000",
            "v_mov_b32 v1, 31",
            "v_mov_b32 v2, 1",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_float_sign")
        self.assertEqual(words[0], 0xFFFFFFFF)

        # Extract exponent of -1.0 (bits 23..30, width=8):
        # 0xBF800000 >> 23 = 0x17F; 0x17F & 0xFF = 0x7F (127). MSB of 8-bit field is 0 -> positive 127
        r_exp = ref_bfe_i32(0xBF800000, 23, 8)
        self.assertEqual(r_exp, 127)

        words_exp = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xBF800000",
            "v_mov_b32 v1, 23",
            "v_mov_b32 v2, 8",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_float_exp")
        self.assertEqual(words_exp[0], 127)

    def test_nan_and_inf_bit_patterns(self):
        # +inf (0x7F800000): sign bit is 0
        self.assertEqual(ref_bfe_i32(0x7F800000, 31, 1), 0)
        words_pinf = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x7F800000",
            "v_mov_b32 v1, 31",
            "v_mov_b32 v2, 1",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_pinf")
        self.assertEqual(words_pinf[0], 0)

        # -inf (0xFF800000): sign bit is 1 -> 0xFFFFFFFF
        self.assertEqual(ref_bfe_i32(0xFF800000, 31, 1), 0xFFFFFFFF)
        words_ninf = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xFF800000",
            "v_mov_b32 v1, 31",
            "v_mov_b32 v2, 1",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_ninf")
        self.assertEqual(words_ninf[0], 0xFFFFFFFF)

        # qNaN (0x7FC00000): sign bit is 0
        self.assertEqual(ref_bfe_i32(0x7FC00000, 31, 1), 0)
        words_qnan = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x7FC00000",
            "v_mov_b32 v1, 31",
            "v_mov_b32 v2, 1",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_qnan")
        self.assertEqual(words_qnan[0], 0)

    def test_integer_overflow_extremes(self):
        # 0x7FFFFFFF: width=31 field bits 0..30 are all 1s.
        # Bit 30 is the sign bit of the 31-bit field: it is 1, so the result is sign-extended to -1.
        self.assertEqual(ref_bfe_i32(0x7FFFFFFF, 0, 31), 0xFFFFFFFF)
        words_7fff = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x7FFFFFFF",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 31",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_7fffffff_w31")
        self.assertEqual(words_7fff[0], 0xFFFFFFFF)

        # 0x3FFFFFFF: bits 0..30, sign bit (bit 30) = 0 -> stays positive
        self.assertEqual(ref_bfe_i32(0x3FFFFFFF, 0, 31), 0x3FFFFFFF)
        words_3fff = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x3FFFFFFF",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 31",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_3fffffff_w31")
        self.assertEqual(words_3fff[0], 0x3FFFFFFF)

        # 0x80000000 (INT_MIN): bit 31 is 1, lower 31 bits are all 0
        self.assertEqual(ref_bfe_i32(0x80000000, 0, 31), 0)
        words_8000 = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x80000000",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 31",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_80000000_w31")
        self.assertEqual(words_8000[0], 0)

        # width=32 is masked mod 32 to give 0, so result is 0
        self.assertEqual(ref_bfe_i32(0x80000000, 0, 32), 0)
        words_w32 = run_ps_bfe_i32([
            "v_mov_b32 v0, 0x80000000",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 32",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_80000000_w32")
        self.assertEqual(words_w32[0], 0)

    def test_translator_width_past_31_cut(self):
        """When offset + width > 32, width cuts to 32 - offset."""
        # offset = 30, width = 16 -> width cut to 32 - 30 = 2 bits.
        # base = 0xC0000000 (bits 30..31 are 11b = 3).
        # Bit 1 of 2-bit field is 1 -> sign-extended to 0xFFFFFFFF (-1).
        words = run_ps_bfe_i32([
            "v_mov_b32 v0, 0xC0000000",
            "v_mov_b32 v1, 30",
            "v_mov_b32 v2, 16",
            "v_bfe_i32 v3, v0, v1, v2",
        ], name="ps_bfe_cut_past31")
        self.assertEqual(words[0], 0xFFFFFFFF)


class TestNegativeControlsAndMutation(unittest.TestCase):
    def test_negative_dst_rejection(self):
        with self.assertRaises(TypeError):
            VBfeI32Instruction(
                dst=Operand.sgpr(0),
                src0=Operand.vgpr(0),
                src1=Operand.vgpr(1),
                src2=Operand.vgpr(2),
            )

    def test_out_of_bounds_register_rejection(self):
        with self.assertRaises(ValueError):
            Operand.vgpr(256)
        with self.assertRaises(ValueError):
            Operand.sgpr(106)

    def test_mutation_unsigned_vs_signed_bfe(self):
        val = 0x80000000
        signed_result = ref_bfe_i32(val, 31, 1)
        unsigned_result = (val >> 31) & 1
        self.assertNotEqual(signed_result, unsigned_result)
        self.assertEqual(signed_result, 0xFFFFFFFF)
        self.assertEqual(unsigned_result, 1)

    def test_mutation_dropped_sign_extension(self):
        raw_field = (0xFFFFFFF0 >> 2) & 0xF
        proper_result = ref_bfe_i32(0xFFFFFFF0, 2, 4)
        self.assertNotEqual(raw_field, proper_result)
        self.assertEqual(raw_field, 0xC)
        self.assertEqual(proper_result, 0xFFFFFFFC)

    def test_mutation_offset_and_width_mask(self):
        res_masked = ref_bfe_i32(0x12345678, 36, 40)
        res_direct = ref_bfe_i32(0x12345678, 4, 8)
        self.assertEqual(res_masked, res_direct)
        self.assertEqual(res_masked, 0x67)

    def test_rejection_invalid_operands_in_translator(self):
        bad_lines = [
            "v_bfe_i32 v3, v0, 1",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        with self.assertRaises(Exception):
            support.translate_lines("ps_bfe_bad_args", bad_lines)

    def test_spirv_emission_opbitfieldsextract(self):
        """Verify that support.translate_lines emits OpBitFieldSExtract and NOT OpBitFieldUExtract."""
        lines = [
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 4",
            "v_mov_b32 v2, 8",
            "v_bfe_i32 v3, v0, v1, v2",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_bfe_spirv_emit", lines)
        self.assertIn("OpBitFieldSExtract", spv)
        self.assertNotIn("OpBitFieldUExtract", spv)

    def test_mutation_detection_signed_vs_unsigned_hardware_execution(self):
        """Verify that hardware execution distinguishes OpBitFieldSExtract from unsigned extract."""
        test_vectors = [
            (0x80000000, 31, 1, 0xFFFFFFFF, 1),
            (0xFFFFFFF0, 2, 4, 0xFFFFFFFC, 12),
            (0x00010000, 0, 17, 0xFFFF0000, 0x00010000),
            (0x7FFFFFFF, 0, 31, 0xFFFFFFFF, 0x7FFFFFFF),
        ]
        for base, off, wid, signed_exp, unsigned_val in test_vectors:
            with self.subTest(base=hex(base), off=off, wid=wid):
                words = run_ps_bfe_i32([
                    f"v_mov_b32 v0, {base:#x}",
                    f"v_mov_b32 v1, {off}",
                    f"v_mov_b32 v2, {wid}",
                    "v_bfe_i32 v3, v0, v1, v2",
                ], name=f"ps_bfe_mut_check_{off}_{wid}")
                self.assertEqual(words[0], signed_exp)
                self.assertNotEqual(words[0], unsigned_val)


if __name__ == "__main__":
    unittest.main()
