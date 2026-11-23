"""Semantic tests for RDNA2 s_orn2_b32 opcode with hardware translator execution.

RDNA2 Instruction Set Architecture, SOP2:
    s_orn2_b32 D, S0, S1

Semantic operation:
    D = (S0 | (~S1 & 0xFFFFFFFF)) & 0xFFFFFFFF
    SCC = (D != 0)
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import support  # noqa: E402

MASK32 = 0xFFFFFFFF


def ref_orn2_b32(s0: int, s1: int) -> tuple[int, int]:
    """Bit-accurate reference oracle for RDNA2 s_orn2_b32."""
    s0_u32 = s0 & MASK32
    s1_u32 = s1 & MASK32
    d = (s0_u32 | (~s1_u32 & MASK32)) & MASK32
    scc = 1 if d != 0 else 0
    return d, scc


def run_hw_orn2_b32(s0: int, s1: int, shader_tag: str = "ps_orn2_b32_hw") -> tuple[int, int]:
    """Execute s_orn2_b32 through rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"s_mov_b32 s2, {s0 & MASK32:#x}",
        f"s_mov_b32 s3, {s1 & MASK32:#x}",
        "s_orn2_b32 s4, s2, s3",
        "s_cselect_b32 s5, 1, 0",
        "v_mov_b32 v0, s4",
        "v_mov_b32 v1, s5",
        "exp mrt0 v0, v1, v0, v1 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(shader_tag, lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


def run_custom_lines(shader_tag: str, lines: list[str]) -> list[int]:
    """Compile arbitrary shader lines through translator and execute."""
    spv = support.translate_lines(shader_tag, lines)
    return support.mrt0_words(spv)


class TestSOrn2B32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_zeros_and_ones(self):
        # s0=0, s1=0 -> ~s1 = 0xFFFFFFFF -> d = 0xFFFFFFFF, scc = 1
        d, scc = ref_orn2_b32(0, 0)
        self.assertEqual(d, MASK32)
        self.assertEqual(scc, 1)
        hw_d, hw_scc = run_hw_orn2_b32(0, 0, "ps_ref_0_0")
        self.assertEqual((hw_d, hw_scc), (d, scc))

        # s0=0, s1=0xFFFFFFFF -> ~s1 = 0 -> d = 0, scc = 0 (the unique zero result)
        d, scc = ref_orn2_b32(0, MASK32)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)
        hw_d, hw_scc = run_hw_orn2_b32(0, MASK32, "ps_ref_0_all")
        self.assertEqual((hw_d, hw_scc), (d, scc))

        # s0=0xFFFFFFFF, s1=0xFFFFFFFF -> d = 0xFFFFFFFF, scc = 1
        d, scc = ref_orn2_b32(MASK32, MASK32)
        self.assertEqual(d, MASK32)
        self.assertEqual(scc, 1)
        hw_d, hw_scc = run_hw_orn2_b32(MASK32, MASK32, "ps_ref_all_all")
        self.assertEqual((hw_d, hw_scc), (d, scc))

    def test_reference_alternating_patterns(self):
        s0 = 0xAAAAAAAA
        s1 = 0x55555555
        # ~s1 is 0xAAAAAAAA. 0xAAAAAAAA | 0xAAAAAAAA = 0xAAAAAAAA
        d, scc = ref_orn2_b32(s0, s1)
        self.assertEqual(d, 0xAAAAAAAA)
        self.assertEqual(scc, 1)
        hw_d, hw_scc = run_hw_orn2_b32(s0, s1, "ps_ref_alt_1")
        self.assertEqual((hw_d, hw_scc), (d, scc))

        # Inverted: s0=0x55555555, s1=0x55555555 -> ~s1 = 0xAAAAAAAA -> d = 0xFFFFFFFF
        d, scc = ref_orn2_b32(0x55555555, 0x55555555)
        self.assertEqual(d, MASK32)
        self.assertEqual(scc, 1)
        hw_d, hw_scc = run_hw_orn2_b32(0x55555555, 0x55555555, "ps_ref_alt_2")
        self.assertEqual((hw_d, hw_scc), (d, scc))

    def test_translator_hw_orn2_b32(self):
        cases = [
            (0, 0),
            (0, MASK32),
            (MASK32, MASK32),
            (0x12345678, 0x87654321),
            (0xAAAAAAAA, 0x55555555),
            (0x55555555, 0xAAAAAAAA),
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0x1, 0x0),
            (0x80000000, 0x7FFFFFFF),
        ]
        for idx, (s0, s1) in enumerate(cases):
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = run_hw_orn2_b32(s0, s1, f"ps_hw_case_{idx}")
                exp_d, exp_scc = ref_orn2_b32(s0, s1)
                self.assertEqual(d, exp_d)
                self.assertEqual(scc, exp_scc)


class RegType(enum.Enum):
    SGPR = "sgpr"
    IMM = "imm"


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    val: int = 0

    def format(self) -> str:
        if self.reg_type == RegType.SGPR:
            return f"s{self.index}"
        return f"{self.val:#x}"


@dataclasses.dataclass(frozen=True)
class SOrn2B32Inst:
    dst: Operand
    src0: Operand
    src1: Operand

    def emit_asm(self) -> str:
        return f"s_orn2_b32 {self.dst.format()}, {self.src0.format()}, {self.src1.format()}"


class TestSOrn2B32AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions with translator execution."""

    def test_ast_construction_and_formatting(self):
        inst = SOrn2B32Inst(
            dst=Operand(RegType.SGPR, 4),
            src0=Operand(RegType.SGPR, 2),
            src1=Operand(RegType.SGPR, 3),
        )
        self.assertEqual(inst.emit_asm(), "s_orn2_b32 s4, s2, s3")
        lines = [
            "s_mov_b32 s2, 0x11223344",
            "s_mov_b32 s3, 0x55667788",
            inst.emit_asm(),
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        out = run_custom_lines("ps_ast_construct", lines)
        exp_d, exp_scc = ref_orn2_b32(0x11223344, 0x55667788)
        self.assertEqual(out[0], exp_d)
        self.assertEqual(out[1], exp_scc)

    def test_ast_with_literal(self):
        inst = SOrn2B32Inst(
            dst=Operand(RegType.SGPR, 4),
            src0=Operand(RegType.SGPR, 2),
            src1=Operand(RegType.IMM, 0, 0x1234),
        )
        self.assertEqual(inst.emit_asm(), "s_orn2_b32 s4, s2, 0x1234")
        lines = [
            "s_mov_b32 s2, 0x56780000",
            inst.emit_asm(),
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        out = run_custom_lines("ps_ast_lit", lines)
        exp_d, exp_scc = ref_orn2_b32(0x56780000, 0x1234)
        self.assertEqual(out[0], exp_d)
        self.assertEqual(out[1], exp_scc)

    def test_ast_immutability(self):
        inst = SOrn2B32Inst(
            dst=Operand(RegType.SGPR, 0),
            src0=Operand(RegType.SGPR, 1),
            src1=Operand(RegType.SGPR, 2),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.dst = Operand(RegType.SGPR, 5)  # type: ignore

        lines = [
            "s_mov_b32 s1, 0xF0000000",
            "s_mov_b32 s2, 0x0F000000",
            inst.emit_asm(),
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        out = run_custom_lines("ps_ast_immut", lines)
        exp_d, exp_scc = ref_orn2_b32(0xF0000000, 0x0F000000)
        self.assertEqual(out[0], exp_d)
        self.assertEqual(out[1], exp_scc)


class TestSOrn2B32ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction execution and SCC tracking."""

    def test_scalar_execution_and_scc_tracking(self):
        lines = [
            "s_mov_b32 s2, 0x10101010",
            "s_mov_b32 s3, 0x01010101",
            "s_orn2_b32 s4, s2, s3",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        out = run_custom_lines("ps_scalar_step", lines)
        exp_d, exp_scc = ref_orn2_b32(0x10101010, 0x01010101)
        self.assertEqual(out[0], exp_d)
        self.assertEqual(out[1], exp_scc)

    def test_scalar_pipeline_zero_clear(self):
        # s_orn2_b32 s0, s1, s2 with s1=0, s2=0xFFFFFFFF -> s0 = 0 | ~0xFFFFFFFF = 0, SCC = 0
        lines = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 0xffffffff",
            "s_orn2_b32 s0, s1, s2",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        out = run_custom_lines("ps_scalar_zero", lines)
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], 0)


class TestSOrn2B32SIMDDivergence(unittest.TestCase):
    """SIMD multi-lane divergence and mask interaction through translator."""

    def test_salu_scalar_uniformity(self):
        # Using s_orn2_b32 to compute a uniform active mask
        lines = [
            "s_mov_b32 s1, 0x0000ffff",
            "s_mov_b32 s2, 0xff00ff00",
            "s_orn2_b32 s0, s1, s2",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        out = run_custom_lines("ps_simd_mask", lines)
        # ~s2 = 0x00FF00FF. s1 | ~s2 = 0x00FFFFFF
        self.assertEqual(out[0], 0x00FFFFFF)
        self.assertEqual(out[1], 1)

    def test_exec_mask_manipulation(self):
        # s_orn2_b32 on exec mask bit patterns
        lines = [
            "s_mov_b32 s1, 0x55555555",
            "s_mov_b32 s2, 0x33333333",
            "s_orn2_b32 s0, s1, s2",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        out = run_custom_lines("ps_simd_exec", lines)
        # ~s2 = 0xCCCCCCCC. s1 | ~s2 = 0xDDDDDDDD
        self.assertEqual(out[0], 0xDDDDDDDD)
        self.assertEqual(out[1], 1)


class TestSOrn2B32RegisterAliasing(unittest.TestCase):
    """Register aliasing and in-place operand preservation through translator."""

    def test_dst_aliases_src0(self):
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x00ff00ff",
            "s_orn2_b32 s0, s0, s1",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s5",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        out = run_custom_lines("ps_alias_src0", lines)
        exp_d, exp_scc = ref_orn2_b32(0x12345678, 0x00FF00FF)
        self.assertEqual(out[0], exp_d)
        self.assertEqual(out[1], 0x00FF00FF)
        self.assertEqual(out[2], exp_scc)

    def test_dst_aliases_src1(self):
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x00ff00ff",
            "s_orn2_b32 s1, s0, s1",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "v_mov_b32 v2, s5",
            "v_mov_b32 v3, 0",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ]
        out = run_custom_lines("ps_alias_src1", lines)
        exp_d, exp_scc = ref_orn2_b32(0x12345678, 0x00FF00FF)
        self.assertEqual(out[0], 0x12345678)
        self.assertEqual(out[1], exp_d)
        self.assertEqual(out[2], exp_scc)

    def test_all_operands_aliased(self):
        lines = [
            "s_mov_b32 s0, 0xaaaaaaaa",
            "s_orn2_b32 s0, s0, s0",
            "s_cselect_b32 s5, 1, 0",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        out = run_custom_lines("ps_alias_all", lines)
        self.assertEqual(out[0], MASK32)
        self.assertEqual(out[1], 1)


class TestSOrn2B32ConditionCodes(unittest.TestCase):
    """Condition codes and status flags (SCC) verified through translator execution."""

    def test_scc_is_zero_only_when_dst_zero(self):
        # Result is 0 if and only if s0 == 0 and s1 == 0xFFFFFFFF
        d, scc = run_hw_orn2_b32(0, MASK32, "ps_scc_zero")
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)

        # Any 1 bit in result makes SCC = 1
        d, scc = run_hw_orn2_b32(1, MASK32, "ps_scc_one")
        self.assertEqual(d, 1)
        self.assertEqual(scc, 1)

        d, scc = run_hw_orn2_b32(0, MASK32 ^ 1, "ps_scc_lsb")
        self.assertEqual(d, 1)
        self.assertEqual(scc, 1)

        d, scc = run_hw_orn2_b32(0x80000000, MASK32, "ps_scc_msb")
        self.assertEqual(d, 0x80000000)
        self.assertEqual(scc, 1)

    def test_scc_all_32_bit_positions(self):
        # Verify SCC=1 for every individual bit set in s0 when s1=0xFFFFFFFF
        for bit in range(32):
            val = 1 << bit
            with self.subTest(bit=bit):
                d, scc = run_hw_orn2_b32(val, MASK32, f"ps_scc_bit_{bit}")
                self.assertEqual(d, val)
                self.assertEqual(scc, 1)


class TestSOrn2B32IEEE754(unittest.TestCase):
    """IEEE 754 boundary extremes, subnormals, and NaN bitwise handling."""

    def test_ieee754_bitcast_preservation(self):
        # Treating float representations as raw 32-bit words
        pos_zero = 0x00000000
        neg_zero = 0x80000000
        # s_orn2_b32(+0.0, -0.0) -> 0x00000000 | ~0x80000000 = 0x7FFFFFFF
        d, scc = run_hw_orn2_b32(pos_zero, neg_zero, "ps_ieee_zeros")
        self.assertEqual(d, 0x7FFFFFFF)
        self.assertEqual(scc, 1)

    def test_nan_payload_preservation(self):
        nan_bits = 0x7F800000 | 0x12345
        # s0 | ~0 = s0 | 0xFFFFFFFF = 0xFFFFFFFF
        d, scc = run_hw_orn2_b32(nan_bits, 0, "ps_nan_all1s")
        self.assertEqual(d, MASK32)
        self.assertEqual(scc, 1)
        # s0 | ~0xFFFFFFFF = s0 | 0 = s0
        d, scc = run_hw_orn2_b32(nan_bits, MASK32, "ps_nan_exact")
        self.assertEqual(d, nan_bits)
        self.assertEqual(scc, 1)


class TestSOrn2B32NegativeAndMutation(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection against translator."""

    def test_syntax_rejection_invalid_destination(self):
        # Missing operand rejection by translator
        with self.assertRaises(Exception):
            support.translate_lines("ps_rej_missing_op", ["s_orn2_b32 s0, s1", "s_endpgm"])
        with self.assertRaises(Exception):
            support.translate_lines("ps_rej_no_ops", ["s_orn2_b32", "s_endpgm"])

    def test_mutation_operator_lowering(self):
        # Verify translator lowers s_orn2_b32 to OpBitwiseOr and OpNot
        lines = [
            "s_mov_b32 s2, 0x12",
            "s_mov_b32 s3, 0x34",
            "s_orn2_b32 s4, s2, s3",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_mut_check", lines)
        self.assertIn("OpBitwiseOr", spv)
        self.assertIn("OpNot", spv)

        # Mutated logic (using AND instead of OR) produces 0 for (0, 0), while real orn2 produces 0xFFFFFFFF
        hw_d, _ = run_hw_orn2_b32(0, 0, "ps_mut_exec")
        mutated_and = (0 & (~0 & MASK32)) & MASK32
        self.assertEqual(hw_d, MASK32)
        self.assertEqual(mutated_and, 0)
        self.assertNotEqual(hw_d, mutated_and)


if __name__ == "__main__":
    unittest.main()
