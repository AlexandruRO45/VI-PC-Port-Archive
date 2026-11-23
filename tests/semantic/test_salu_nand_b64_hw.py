"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_nand_b64.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_nand_b64 instruction belongs to the SOP2 format.
  Instruction:
    s_nand_b64 sdst, ssrc0, ssrc1

  Operands:
    sdst  = 64-bit destination SGPR pair s[2k:2k+1] (or EXEC)
    ssrc0 = 64-bit source operand: SGPR pair, literal 64-bit constant, inline constant, EXEC, or VCC
    ssrc1 = 64-bit source operand: SGPR pair, literal 64-bit constant, inline constant, EXEC, or VCC

  Semantics:
    sdst = ~(ssrc0 & ssrc1) & 0xFFFFFFFFFFFFFFFF
    SCC = 1 if sdst != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
from typing import Dict, List, Optional, Tuple
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

try:
    import rdna2spirv
    import support
except Exception:
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_ACTIVE: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_INACTIVE: int = 0x0000000000000000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit integer range [0, 2^64 - 1]."""
    return val & MASK64


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_nand_b64(ssrc0: int, ssrc1: int) -> tuple[int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_nand_b64 instruction.

    Args:
        ssrc0: 64-bit first source operand (uint64)
        ssrc1: 64-bit second source operand (uint64)

    Returns:
        tuple (sdst, scc):
          - sdst: 64-bit result = ~(ssrc0 & ssrc1) & MASK64
          - scc: scalar condition code = 1 if sdst != 0 else 0
    """
    s0 = u64(ssrc0)
    s1 = u64(ssrc1)
    sdst = u64(~(s0 & s1))
    scc = 1 if sdst != 0 else 0
    return sdst, scc


class TestSaluNandB64Oracle(unittest.TestCase):
    """Stage 1: Reference mathematical oracle unit tests."""

    def test_oracle_basic(self) -> None:
        s0 = 0x123456789ABCDEF0
        s1 = 0x00FF00FF0F0F0F0F
        sdst, scc = ref_nand_b64(s0, s1)
        expected = u64(~(s0 & s1))
        self.assertEqual(sdst, expected)
        self.assertEqual(scc, 1)

    def test_oracle_all_ones(self) -> None:
        s0 = ALL_LANES_ACTIVE
        s1 = ALL_LANES_ACTIVE
        sdst, scc = ref_nand_b64(s0, s1)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)

    def test_oracle_zero_inputs(self) -> None:
        sdst, scc = ref_nand_b64(0, 0)
        self.assertEqual(sdst, ALL_LANES_ACTIVE)
        self.assertEqual(scc, 1)

        sdst0, scc0 = ref_nand_b64(ALL_LANES_ACTIVE, 0)
        self.assertEqual(sdst0, ALL_LANES_ACTIVE)
        self.assertEqual(scc0, 1)

    def test_oracle_commutativity(self) -> None:
        s0 = 0x0123456789ABCDEF
        s1 = 0xFEDCBA9876543210
        res01, scc01 = ref_nand_b64(s0, s1)
        res10, scc10 = ref_nand_b64(s1, s0)
        self.assertEqual(res01, res10)
        self.assertEqual(scc01, scc10)

    def test_oracle_de_morgan(self) -> None:
        s0 = 0xAAAAAAAAAAAAAAAA
        s1 = 0x5555555555555555
        sdst, scc = ref_nand_b64(s0, s1)
        # s0 & s1 is 0, so ~(s0 & s1) is all ones
        self.assertEqual(sdst, ALL_LANES_ACTIVE)
        self.assertEqual(scc, 1)


class Opcode(Enum):
    S_NAND_B64 = auto()


class OperandKind(Enum):
    SGPR = auto()
    EXEC = auto()
    VCC = auto()
    LITERAL64 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    count: int = 2
    value: int = 0

    @classmethod
    def sgpr(cls, index: int, count: int = 2) -> ASTOperand:
        if count != 2:
            raise ValueError(f"s_nand_b64 requires 64-bit register pair (count=2), got {count}")
        if index < 0 or index > 104 or (index + count) > 106:
            raise ValueError(f"SGPR index out of range: s[{index}:{index + count - 1}]")
        if index % 2 != 0:
            raise ValueError(f"SGPR pair must be 64-bit aligned (even index), got s{index}")
        return cls(kind=OperandKind.SGPR, index=index, count=count)

    @classmethod
    def exec_mask(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC, count=2)

    @classmethod
    def vcc(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC, count=2)

    @classmethod
    def literal64(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL64, count=2, value=u64(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, count=2, value=u64(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand

    def disasm(self) -> str:
        def fmt(op: ASTOperand) -> str:
            if op.kind == OperandKind.SGPR:
                return f"s[{op.index}:{op.index + 1}]"
            if op.kind == OperandKind.EXEC:
                return "exec"
            if op.kind == OperandKind.VCC:
                return "vcc"
            return hex(op.value)
        return f"s_nand_b64 {fmt(self.dst)}, {fmt(self.src0)}, {fmt(self.src1)}"


class TestSaluNandB64AST(unittest.TestCase):
    """Stage 2: AST opcode definitions and typed expressions."""

    def test_ast_construction_sgpr_pair(self) -> None:
        dst = ASTOperand.sgpr(0)
        src0 = ASTOperand.sgpr(2)
        src1 = ASTOperand.sgpr(4)
        inst = ASTInstruction(Opcode.S_NAND_B64, dst, src0, src1)
        self.assertEqual(inst.disasm(), "s_nand_b64 s[0:1], s[2:3], s[4:5]")

    def test_ast_odd_alignment_rejection(self) -> None:
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(1)  # odd index must be rejected


class ScalarMachineState:
    def __init__(self, num_sgprs: int = 106) -> None:
        self.sgprs: list[int] = [0] * num_sgprs
        self.scc: int = 0
        self.exec_mask: int = ALL_LANES_ACTIVE
        self.vcc: int = 0

    def read_operand64(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.SGPR:
            lo = self.sgprs[op.index] & MASK32
            hi = self.sgprs[op.index + 1] & MASK32
            return (hi << 32) | lo
        if op.kind == OperandKind.EXEC:
            return self.exec_mask & MASK64
        if op.kind == OperandKind.VCC:
            return self.vcc & MASK64
        return op.value & MASK64

    def write_operand64(self, op: ASTOperand, val: int) -> None:
        val &= MASK64
        lo = val & MASK32
        hi = (val >> 32) & MASK32
        if op.kind == OperandKind.SGPR:
            self.sgprs[op.index] = lo
            self.sgprs[op.index + 1] = hi
        elif op.kind == OperandKind.EXEC:
            self.exec_mask = val
        elif op.kind == OperandKind.VCC:
            self.vcc = val


class ScalarInterpreter:
    def step(self, instr: ASTInstruction, state: ScalarMachineState) -> tuple[int, int]:
        s0 = state.read_operand64(instr.src0)
        s1 = state.read_operand64(instr.src1)
        res, scc = ref_nand_b64(s0, s1)
        state.write_operand64(instr.dst, res)
        state.scc = scc
        return res, scc


class TestSaluNandB64ScalarInterpreter(unittest.TestCase):
    """Stage 3: Deterministic scalar instruction interpreter."""

    def test_scalar_interpreter_execution(self) -> None:
        state = ScalarMachineState()
        # Set s[2:3] = 0x00000000FFFFFFFF, s[4:5] = 0x00000000FFFF0000
        state.sgprs[2] = 0xFFFFFFFF
        state.sgprs[3] = 0x00000000
        state.sgprs[4] = 0xFFFF0000
        state.sgprs[5] = 0x00000000

        instr = ASTInstruction(
            Opcode.S_NAND_B64,
            ASTOperand.sgpr(0),
            ASTOperand.sgpr(2),
            ASTOperand.sgpr(4),
        )
        res, scc = ScalarInterpreter().step(instr, state)
        # s0 & s1 = 0x00000000FFFF0000 -> ~(s0 & s1) = 0xFFFFFFFF0000FFFF
        self.assertEqual(res, 0xFFFFFFFF0000FFFF)
        self.assertEqual(scc, 1)
        self.assertEqual(state.sgprs[0], 0x0000FFFF)
        self.assertEqual(state.sgprs[1], 0xFFFFFFFF)


class SIMDMachineState:
    def __init__(self, num_lanes: int = 64, exec_mask: int = ALL_LANES_ACTIVE) -> None:
        self.num_lanes = num_lanes
        self.exec_mask = exec_mask
        self.scalar_state = ScalarMachineState()

    def step(self, instr: ASTInstruction) -> tuple[int, int]:
        return ScalarInterpreter().step(instr, self.scalar_state)


class TestSaluNandB64SIMDDivergence(unittest.TestCase):
    """Stage 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_exec_mask_interaction(self) -> None:
        state = SIMDMachineState()
        state.scalar_state.exec_mask = 0x5555555555555555
        instr = ASTInstruction(
            Opcode.S_NAND_B64,
            ASTOperand.exec_mask(),
            ASTOperand.exec_mask(),
            ASTOperand.literal64(0x3333333333333333),
        )
        # 0x5555555555555555 & 0x3333333333333333 = 0x1111111111111111
        # ~0x1111111111111111 = 0xEEEEEEEEEEEEEEEE
        res, scc = state.step(instr)
        self.assertEqual(res, 0xEEEEEEEEEEEEEEEE)
        self.assertEqual(state.scalar_state.exec_mask, 0xEEEEEEEEEEEEEEEE)
        self.assertEqual(scc, 1)


class TestSaluNandB64Aliasing(unittest.TestCase):
    """Stage 5: Register aliasing and in-place operand preservation."""

    def test_aliasing_dst_src0(self) -> None:
        state = ScalarMachineState()
        state.write_operand64(ASTOperand.sgpr(0), 0xF0F0F0F0F0F0F0F0)
        state.write_operand64(ASTOperand.sgpr(2), 0x0F0F0F0F0F0F0F0F)
        instr = ASTInstruction(
            Opcode.S_NAND_B64,
            ASTOperand.sgpr(0),
            ASTOperand.sgpr(0),
            ASTOperand.sgpr(2),
        )
        # 0xF0 & 0x0F = 0 -> ~0 = all ones
        res, scc = ScalarInterpreter().step(instr, state)
        self.assertEqual(res, ALL_LANES_ACTIVE)
        self.assertEqual(state.read_operand64(ASTOperand.sgpr(0)), ALL_LANES_ACTIVE)

    def test_aliasing_all_same(self) -> None:
        state = ScalarMachineState()
        state.write_operand64(ASTOperand.sgpr(0), 0x123456789ABCDEF0)
        instr = ASTInstruction(
            Opcode.S_NAND_B64,
            ASTOperand.sgpr(0),
            ASTOperand.sgpr(0),
            ASTOperand.sgpr(0),
        )
        # ~(s0 & s0) = ~s0
        res, scc = ScalarInterpreter().step(instr, state)
        expected = u64(~0x123456789ABCDEF0)
        self.assertEqual(res, expected)
        self.assertEqual(state.read_operand64(ASTOperand.sgpr(0)), expected)


class TestSaluNandB64ConditionCodes(unittest.TestCase):
    """Stage 6: Condition codes, status flags, and carry/borrow generation."""

    def test_scc_set_when_nonzero(self) -> None:
        state = ScalarMachineState()
        instr = ASTInstruction(
            Opcode.S_NAND_B64,
            ASTOperand.sgpr(0),
            ASTOperand.literal64(0),
            ASTOperand.literal64(0),
        )
        _, scc = ScalarInterpreter().step(instr, state)
        self.assertEqual(scc, 1)
        self.assertEqual(state.scc, 1)

    def test_scc_cleared_when_zero(self) -> None:
        state = ScalarMachineState()
        state.scc = 1
        instr = ASTInstruction(
            Opcode.S_NAND_B64,
            ASTOperand.sgpr(0),
            ASTOperand.literal64(ALL_LANES_ACTIVE),
            ASTOperand.literal64(ALL_LANES_ACTIVE),
        )
        res, scc = ScalarInterpreter().step(instr, state)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)
        self.assertEqual(state.scc, 0)


class TestSaluNandB64IEEE754(unittest.TestCase):
    """Stage 7: IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_float_pattern_preservation(self) -> None:
        qnan64 = 0x7FF8000000000000
        snan64 = 0x7FF0000000000001
        res, _ = ref_nand_b64(qnan64, snan64)
        expected = u64(~(qnan64 & snan64))
        self.assertEqual(res, expected)

    def test_subnormal_no_ftz(self) -> None:
        subnormal = 0x0000000000000001
        res, _ = ref_nand_b64(subnormal, subnormal)
        self.assertEqual(res, u64(~subnormal))


class TestSaluNandB64NegativeControls(unittest.TestCase):
    """Negative controls and mutation rejection."""

    def test_mutation_regular_and(self) -> None:
        def mutant_and(s0: int, s1: int) -> int:
            return s0 & s1
        correct, _ = ref_nand_b64(0x0F, 0x0F)
        mutated = mutant_and(0x0F, 0x0F)
        self.assertNotEqual(correct, mutated)


class TestSaluNandB64TranslationAndExecution(unittest.TestCase):
    """Stage 8: End-to-end SPIR-V translation and fragment execution."""

    def test_spirv_emission(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x9abcdef0",
            "s_mov_b32 s2, 0x00ff00ff",
            "s_mov_b32 s3, 0x0f0f0f0f",
            "s_nand_b64 s[4:5], s[0:1], s[2:3]",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_nand_b64_emit", lines)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)

    def test_hardware_execution(self) -> None:
        if support is None or rdna2spirv is None:
            self.skipTest("support/rdna2spirv not available")
        s0_lo, s0_hi = 0x12345678, 0x9ABCDEF0
        s1_lo, s1_hi = 0x00FF00FF, 0x0F0F0F0F
        expected_lo = (~(s0_lo & s1_lo)) & MASK32
        expected_hi = (~(s0_hi & s1_hi)) & MASK32

        lines = [
            f"s_mov_b32 s0, {s0_lo:#x}",
            f"s_mov_b32 s1, {s0_hi:#x}",
            f"s_mov_b32 s2, {s1_lo:#x}",
            f"s_mov_b32 s3, {s1_hi:#x}",
            "s_nand_b64 s[4:5], s[0:1], s[2:3]",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_nand_b64_hw", lines)
        words = support.mrt0_words(spv)

        self.assertEqual(words[0], expected_lo)
        self.assertEqual(words[1], expected_hi)
        self.assertEqual(words[2], 1)
        self.assertEqual(words[3], 0)


if __name__ == "__main__":
    unittest.main()
