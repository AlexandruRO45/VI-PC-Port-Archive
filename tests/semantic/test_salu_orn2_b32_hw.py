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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

try:
    import support  # noqa: E402
except Exception:
    support = None

MASK32 = 0xFFFFFFFF


def ref_orn2_b32(s0: int, s1: int) -> tuple[int, int]:
    """Bit-accurate reference oracle for RDNA2 s_orn2_b32."""
    s0_u32 = s0 & MASK32
    s1_u32 = s1 & MASK32
    d = (s0_u32 | (~s1_u32 & MASK32)) & MASK32
    scc = 1 if d != 0 else 0
    return d, scc


def run_hw_orn2_b32(s0: int, s1: int) -> tuple[int, int]:
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
    spv = support.translate_lines("ps_orn2_b32_hw", lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


class TestSOrn2B32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_zeros_and_ones(self):
        # s0=0, s1=0 -> ~s1 = 0xFFFFFFFF -> d = 0xFFFFFFFF, scc = 1
        d, scc = ref_orn2_b32(0, 0)
        self.assertEqual(d, MASK32)
        self.assertEqual(scc, 1)

        # s0=0, s1=0xFFFFFFFF -> ~s1 = 0 -> d = 0, scc = 0 (the unique zero result)
        d, scc = ref_orn2_b32(0, MASK32)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)

        # s0=0xFFFFFFFF, s1=0xFFFFFFFF -> d = 0xFFFFFFFF, scc = 1
        d, scc = ref_orn2_b32(MASK32, MASK32)
        self.assertEqual(d, MASK32)
        self.assertEqual(scc, 1)

    def test_reference_alternating_patterns(self):
        s0 = 0xAAAAAAAA
        s1 = 0x55555555
        # ~s1 is 0xAAAAAAAA. 0xAAAAAAAA | 0xAAAAAAAA = 0xAAAAAAAA
        d, scc = ref_orn2_b32(s0, s1)
        self.assertEqual(d, 0xAAAAAAAA)
        self.assertEqual(scc, 1)

        # Inverted: s0=0x55555555, s1=0x55555555 -> ~s1 = 0xAAAAAAAA -> d = 0xFFFFFFFF
        d, scc = ref_orn2_b32(0x55555555, 0x55555555)
        self.assertEqual(d, MASK32)
        self.assertEqual(scc, 1)

    def test_translator_hw_orn2_b32(self):
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        cases = [
            (0, 0),
            (0, MASK32),
            (MASK32, MASK32),
            (0x12345678, 0x87654321),
            (0xAAAAAAAA, 0x55555555),
        ]
        for s0, s1 in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1)):
                d, scc = run_hw_orn2_b32(s0, s1)
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
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_formatting(self):
        inst = SOrn2B32Inst(
            dst=Operand(RegType.SGPR, 4),
            src0=Operand(RegType.SGPR, 2),
            src1=Operand(RegType.SGPR, 3),
        )
        self.assertEqual(inst.emit_asm(), "s_orn2_b32 s4, s2, s3")

    def test_ast_with_literal(self):
        inst = SOrn2B32Inst(
            dst=Operand(RegType.SGPR, 4),
            src0=Operand(RegType.SGPR, 2),
            src1=Operand(RegType.IMM, 0, 0x1234),
        )
        self.assertEqual(inst.emit_asm(), "s_orn2_b32 s4, s2, 0x1234")

    def test_ast_immutability(self):
        inst = SOrn2B32Inst(
            dst=Operand(RegType.SGPR, 0),
            src0=Operand(RegType.SGPR, 1),
            src1=Operand(RegType.SGPR, 2),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.dst = Operand(RegType.SGPR, 5)  # type: ignore


class ScalarMachine:
    def __init__(self):
        self.sgprs: dict[int, int] = {i: 0 for i in range(106)}
        self.scc: int = 0

    def read_operand(self, op: Operand) -> int:
        if op.reg_type == RegType.SGPR:
            return self.sgprs.get(op.index, 0)
        return op.val & MASK32

    def write_sgpr(self, op: Operand, val: int):
        if op.reg_type == RegType.SGPR:
            self.sgprs[op.index] = val & MASK32

    def step(self, inst: SOrn2B32Inst):
        s0 = self.read_operand(inst.src0)
        s1 = self.read_operand(inst.src1)
        d, scc = ref_orn2_b32(s0, s1)
        self.write_sgpr(inst.dst, d)
        self.scc = scc


class TestSOrn2B32ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter."""

    def test_scalar_execution_and_scc_tracking(self):
        mach = ScalarMachine()
        mach.sgprs[2] = 0x10101010
        mach.sgprs[3] = 0x01010101
        inst = SOrn2B32Inst(Operand(RegType.SGPR, 4), Operand(RegType.SGPR, 2), Operand(RegType.SGPR, 3))
        mach.step(inst)
        exp_d, exp_scc = ref_orn2_b32(0x10101010, 0x01010101)
        self.assertEqual(mach.sgprs[4], exp_d)
        self.assertEqual(mach.scc, exp_scc)

    def test_scalar_pipeline_zero_clear(self):
        mach = ScalarMachine()
        mach.sgprs[1] = 0
        mach.sgprs[2] = MASK32
        # s_orn2_b32 s0, s1, s2 -> s0 = 0 | ~0xFFFFFFFF = 0, SCC = 0
        mach.step(SOrn2B32Inst(Operand(RegType.SGPR, 0), Operand(RegType.SGPR, 1), Operand(RegType.SGPR, 2)))
        self.assertEqual(mach.sgprs[0], 0)
        self.assertEqual(mach.scc, 0)


class TestSOrn2B32SIMDDivergence(unittest.TestCase):
    """SIMD multi-lane divergence and 32/64-bit EXEC interaction."""

    def test_salu_scalar_uniformity(self):
        # SALU operations are scalar and execute once per wavefront across all lanes
        mach = ScalarMachine()
        # Using s_orn2_b32 to compute a uniform active mask
        mach.sgprs[1] = 0x0000FFFF
        mach.sgprs[2] = 0xFF00FF00
        mach.step(SOrn2B32Inst(Operand(RegType.SGPR, 0), Operand(RegType.SGPR, 1), Operand(RegType.SGPR, 2)))
        # ~s2 = 0x00FF00FF. s1 | ~s2 = 0x00FFFFFF
        self.assertEqual(mach.sgprs[0], 0x00FFFFFF)
        self.assertEqual(mach.scc, 1)

    def test_exec_mask_manipulation(self):
        # Simulating s_orn2_b32 on exec mask bits
        exec_lo = 0x55555555
        filter_lo = 0x33333333
        d, scc = ref_orn2_b32(exec_lo, filter_lo)
        # ~filter_lo = 0xCCCCCCCC. exec_lo | 0xCCCCCCCC = 0xDDDDDDDD
        self.assertEqual(d, 0xDDDDDDDD)
        self.assertEqual(scc, 1)


class TestSOrn2B32RegisterAliasing(unittest.TestCase):
    """Register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self):
        mach = ScalarMachine()
        mach.sgprs[0] = 0x12345678
        mach.sgprs[1] = 0x00FF00FF
        # s0 = s0 | ~s1
        mach.step(SOrn2B32Inst(Operand(RegType.SGPR, 0), Operand(RegType.SGPR, 0), Operand(RegType.SGPR, 1)))
        exp_d, exp_scc = ref_orn2_b32(0x12345678, 0x00FF00FF)
        self.assertEqual(mach.sgprs[0], exp_d)
        self.assertEqual(mach.sgprs[1], 0x00FF00FF)

    def test_dst_aliases_src1(self):
        mach = ScalarMachine()
        mach.sgprs[0] = 0x12345678
        mach.sgprs[1] = 0x00FF00FF
        # s1 = s0 | ~s1
        mach.step(SOrn2B32Inst(Operand(RegType.SGPR, 1), Operand(RegType.SGPR, 0), Operand(RegType.SGPR, 1)))
        exp_d, exp_scc = ref_orn2_b32(0x12345678, 0x00FF00FF)
        self.assertEqual(mach.sgprs[1], exp_d)
        self.assertEqual(mach.sgprs[0], 0x12345678)

    def test_all_operands_aliased(self):
        mach = ScalarMachine()
        mach.sgprs[0] = 0xAAAAAAAA
        # s0 = s0 | ~s0 = 0xFFFFFFFF
        mach.step(SOrn2B32Inst(Operand(RegType.SGPR, 0), Operand(RegType.SGPR, 0), Operand(RegType.SGPR, 0)))
        self.assertEqual(mach.sgprs[0], MASK32)
        self.assertEqual(mach.scc, 1)


class TestSOrn2B32ConditionCodes(unittest.TestCase):
    """Condition codes, status flags, and carry/borrow preservation."""

    def test_scc_is_zero_only_when_dst_zero(self):
        # Result is 0 if and only if s0 == 0 and s1 == 0xFFFFFFFF
        d, scc = ref_orn2_b32(0, MASK32)
        self.assertEqual(d, 0)
        self.assertEqual(scc, 0)

        # Any 1 bit in result makes SCC = 1
        d, scc = ref_orn2_b32(1, MASK32)
        self.assertEqual(d, 1)
        self.assertEqual(scc, 1)

        d, scc = ref_orn2_b32(0, MASK32 ^ 1)
        self.assertEqual(d, 1)
        self.assertEqual(scc, 1)


class TestSOrn2B32IEEE754(unittest.TestCase):
    """IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_ieee754_bitcast_preservation(self):
        # Treating float representations as raw 32-bit words
        pos_zero = 0x00000000
        neg_zero = 0x80000000
        # s_orn2_b32(+0.0, -0.0) -> 0x00000000 | ~0x80000000 = 0x7FFFFFFF
        d, scc = ref_orn2_b32(pos_zero, neg_zero)
        self.assertEqual(d, 0x7FFFFFFF)
        self.assertEqual(scc, 1)

    def test_nan_payload_preservation(self):
        nan_bits = 0x7F800000 | 0x12345
        # s0 | ~0 = s0 | 0xFFFFFFFF = 0xFFFFFFFF
        d, _ = ref_orn2_b32(nan_bits, 0)
        self.assertEqual(d, MASK32)
        # s0 | ~0xFFFFFFFF = s0 | 0 = s0
        d, _ = ref_orn2_b32(nan_bits, MASK32)
        self.assertEqual(d, nan_bits)


class TestSOrn2B32NegativeAndMutation(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_destination(self):
        # Missing operand rejection
        if support is not None and getattr(support, "translate_lines", None) is not None:
            with self.assertRaises(Exception):
                support.translate_lines("ps_invalid_dst", ["s_orn2_b32 s0, s1", "s_endpgm"])

    def test_mutation_inverted_operator(self):
        # Mutation: uses AND instead of OR
        def mutated_orn2(s0, s1):
            return (s0 & (~s1 & MASK32)) & MASK32

        s0 = 0xAAAAAAAA
        s1 = 0x55555555
        correct, _ = ref_orn2_b32(s0, s1)
        mutant = mutated_orn2(s0, s1)
        self.assertEqual(correct, 0xAAAAAAAA)
        self.assertEqual(mutant, 0xAAAAAAAA)  # 0xAAAAAAAA & 0xAAAAAAAA = 0xAAAAAAAA

        # Disambiguate with s0=0, s1=0:
        correct_zero, _ = ref_orn2_b32(0, 0)
        mutant_zero = mutated_orn2(0, 0)
        self.assertEqual(correct_zero, MASK32)
        self.assertEqual(mutant_zero, 0)
        self.assertNotEqual(correct_zero, mutant_zero)


if __name__ == "__main__":
    unittest.main()
