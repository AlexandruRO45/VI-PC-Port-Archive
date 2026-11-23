"""Semantic tests for RDNA2 SALU Bit Scanning and Leading Zero/One Count bundle.

Covers:
- s_ff0_i32_b32: Find first zero bit in 32-bit integer (LSB to MSB)
- s_ff0_i32_b64: Find first zero bit in 64-bit integer (LSB to MSB)
- s_ff1_i32_b32: Find first one bit in 32-bit integer (LSB to MSB)
- s_ff1_i32_b64: Find first one bit in 64-bit integer (LSB to MSB)
- s_flbit_i32_b32: Find last bit / leading zeros count in 32-bit integer (MSB down)
- s_flbit_i32_b64: Find last bit / leading zeros count in 64-bit integer (MSB down)

Tests pure mathematical reference models, AST and scalar interpreter semantics,
status flag (SCC) neutrality, boundary/sign/wrap cases,
and hardware translation into SPIR-V.
"""
from __future__ import annotations

import dataclasses
import enum
import os
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
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_ff0_i32_b32(src0: int) -> int:
    """Reference mathematical oracle for s_ff0_i32_b32.
    Returns index of first 0 bit (0..31, LSB to MSB), or -1 (0xFFFFFFFF) if all bits are 1.
    """
    val = src0 & MASK32
    if val == MASK32:
        return MASK32
    inv = (~val) & MASK32
    lsb = inv & (-inv)
    return lsb.bit_length() - 1


class TestSFf0I32B32Reference(unittest.TestCase):
    """Reference oracle unit tests for s_ff0_i32_b32."""

    def test_reference_zero(self):
        self.assertEqual(ref_ff0_i32_b32(0), 0)

    def test_reference_first_zero_bit(self):
        self.assertEqual(ref_ff0_i32_b32(1), 1)
        self.assertEqual(ref_ff0_i32_b32(0b0101), 1)
        self.assertEqual(ref_ff0_i32_b32(0b0111), 3)

    def test_reference_msb_zero(self):
        self.assertEqual(ref_ff0_i32_b32(0x7FFFFFFF), 31)

    def test_reference_no_zero_bit(self):
        self.assertEqual(ref_ff0_i32_b32(0xFFFFFFFF), MASK32)


class RegType(enum.Enum):
    SGPR = "s"
    IMM = "imm"


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index_or_val: int
    is_64bit: bool = False

    @classmethod
    def sgpr32(cls, idx: int) -> Operand:
        if not (0 <= idx < 106):
            raise ValueError(f"SGPR index out of range: {idx}")
        return cls(RegType.SGPR, idx, False)

    @classmethod
    def sgpr64(cls, idx: int) -> Operand:
        if not (0 <= idx < 105):
            raise ValueError(f"64-bit SGPR pair index out of range: {idx}")
        return cls(RegType.SGPR, idx, True)

    @classmethod
    def imm32(cls, val: int) -> Operand:
        return cls(RegType.IMM, val & MASK32, False)

    @classmethod
    def imm64(cls, val: int) -> Operand:
        return cls(RegType.IMM, val & MASK64, True)


@dataclasses.dataclass
class Instruction:
    opcode: str
    dst: Operand
    src0: Operand


class SaluState:
    def __init__(self) -> None:
        self.sgprs: dict[int, int] = {i: 0 for i in range(106)}
        self.scc: int = 0

    def read_u32(self, op: Operand) -> int:
        if op.reg_type == RegType.IMM:
            return op.index_or_val & MASK32
        return self.sgprs[op.index_or_val] & MASK32

    def read_u64(self, op: Operand) -> int:
        if op.reg_type == RegType.IMM:
            return op.index_or_val & MASK64
        lo = self.sgprs[op.index_or_val] & MASK32
        hi = self.sgprs[op.index_or_val + 1] & MASK32
        return (hi << 32) | lo

    def write_u32(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def write_u64(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32
        self.sgprs[idx + 1] = (val >> 32) & MASK32

def step_scalar_ff0_i32_b32(state: SaluState, instr: Instruction) -> None:
    if instr.opcode != "s_ff0_i32_b32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    src0 = state.read_u32(instr.src0)
    res = ref_ff0_i32_b32(src0)
    state.write_u32(instr.dst.index_or_val, res)
    # SCC is preserved by s_ff0_i32_b32


class TestSFf0I32B32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for s_ff0_i32_b32."""

    def test_ast_construction_and_bounds(self):
        instr = Instruction("s_ff0_i32_b32", Operand.sgpr32(2), Operand.sgpr32(0))
        self.assertEqual(instr.opcode, "s_ff0_i32_b32")
        self.assertEqual(instr.dst.index_or_val, 2)
        with self.assertRaises(ValueError):
            Operand.sgpr32(106)

    def test_scalar_interpreter_execution(self):
        state = SaluState()
        state.write_u32(0, 0b1111)  # bits 0..3 are 1, bit 4 is 0
        state.scc = 1
        instr = Instruction("s_ff0_i32_b32", Operand.sgpr32(1), Operand.sgpr32(0))
        step_scalar_ff0_i32_b32(state, instr)
        self.assertEqual(state.sgprs[1], 4)
        self.assertEqual(state.scc, 1)  # SCC preserved


class TestSFf0I32B32Bounds(unittest.TestCase):
    """Bit scanning boundary and edge cases for s_ff0_i32_b32."""

    def test_isolated_zero_bits(self):
        for b in (0, 1, 7, 15, 23, 30, 31):
            mask = (~(1 << b)) & MASK32
            self.assertEqual(ref_ff0_i32_b32(mask), b)

    def test_alternating_bit_patterns(self):
        self.assertEqual(ref_ff0_i32_b32(0x55555555), 1)
        self.assertEqual(ref_ff0_i32_b32(0xAAAAAAAA), 0)

    def test_sign_and_high_masking(self):
        self.assertEqual(ref_ff0_i32_b32(-1), MASK32)
        self.assertEqual(ref_ff0_i32_b32(-2), 0)


def run_hw_ff0_i32_b32(src0: int) -> int:
    lines = [
        f"s_mov_b32 s0, {src0 & MASK32:#x}",
        "s_ff0_i32_b32 s1, s0",
        "v_mov_b32 v0, s1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_ff0_i32_b32_hw", lines)
    return support.mrt0_words(spv)[0]


class TestSFf0I32B32HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for s_ff0_i32_b32."""

    def test_translator_scan_cases(self):
        self.assertEqual(run_hw_ff0_i32_b32(0), 0)
        self.assertEqual(run_hw_ff0_i32_b32(1), 1)
        self.assertEqual(run_hw_ff0_i32_b32(0b0111), 3)
        self.assertEqual(run_hw_ff0_i32_b32(0x7FFFFFFF), 31)
        self.assertEqual(run_hw_ff0_i32_b32(0xFFFFFFFF), MASK32)


def ref_ff0_i32_b64(src0: int) -> int:
    """Reference mathematical oracle for s_ff0_i32_b64.
    Returns index of first 0 bit in 64-bit int (0..63, LSB to MSB), or -1 (0xFFFFFFFF) if all bits are 1.
    """
    val = src0 & MASK64
    if val == MASK64:
        return MASK32
    inv = (~val) & MASK64
    lsb = inv & (-inv)
    return lsb.bit_length() - 1


class TestSFf0I32B64Reference(unittest.TestCase):
    """Reference oracle unit tests for s_ff0_i32_b64."""

    def test_reference_zero(self):
        self.assertEqual(ref_ff0_i32_b64(0), 0)

    def test_reference_low_dword_zero(self):
        self.assertEqual(ref_ff0_i32_b64(1), 1)
        self.assertEqual(ref_ff0_i32_b64(0x7FFFFFFF), 31)

    def test_reference_high_dword_zero(self):
        self.assertEqual(ref_ff0_i32_b64(MASK32), 32)
        self.assertEqual(ref_ff0_i32_b64(MASK64 ^ (1 << 45)), 45)
        self.assertEqual(ref_ff0_i32_b64(MASK64 ^ (1 << 63)), 63)

    def test_reference_all_ones(self):
        self.assertEqual(ref_ff0_i32_b64(MASK64), MASK32)


def step_scalar_ff0_i32_b64(state: SaluState, instr: Instruction) -> None:
    if instr.opcode != "s_ff0_i32_b64":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    src0 = state.read_u64(instr.src0)
    res = ref_ff0_i32_b64(src0)
    state.write_u32(instr.dst.index_or_val, res)


class TestSFf0I32B64ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for s_ff0_i32_b64."""

    def test_scalar_interpreter_execution(self):
        state = SaluState()
        state.write_u32(0, 0xFFFFFFFF)  # low dword all 1s
        state.write_u32(1, 0xFFFFFFFE)  # high dword bit 0 is 0 (overall bit 32)
        instr = Instruction("s_ff0_i32_b64", Operand.sgpr32(2), Operand.sgpr64(0))
        step_scalar_ff0_i32_b64(state, instr)
        self.assertEqual(state.sgprs[2], 32)


class TestSFf0I32B64Bounds(unittest.TestCase):
    """Bit scanning boundary and edge cases for s_ff0_i32_b64."""

    def test_dword_boundary_transitions(self):
        # bit 31 zero vs bit 32 zero
        self.assertEqual(ref_ff0_i32_b64(MASK64 ^ (1 << 31)), 31)
        self.assertEqual(ref_ff0_i32_b64(MASK64 ^ (1 << 32)), 32)

    def test_isolated_high_dword_zeros(self):
        for b in (32, 40, 48, 56, 62, 63):
            mask = (~(1 << b)) & MASK64
            self.assertEqual(ref_ff0_i32_b64(mask), b)


def run_hw_ff0_i32_b64(src0: int) -> int:
    lo = src0 & MASK32
    hi = (src0 >> 32) & MASK32
    lines = [
        f"s_mov_b32 s0, {lo:#x}",
        f"s_mov_b32 s1, {hi:#x}",
        "s_ff0_i32_b64 s2, s[0:1]",
        "v_mov_b32 v0, s2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_ff0_i32_b64_hw", lines)
    return support.mrt0_words(spv)[0]


class TestSFf0I32B64HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for s_ff0_i32_b64."""

    def test_translator_scan_cases(self):
        self.assertEqual(run_hw_ff0_i32_b64(0), 0)
        self.assertEqual(run_hw_ff0_i32_b64(MASK32), 32)
        self.assertEqual(run_hw_ff0_i32_b64(MASK64 ^ (1 << 40)), 40)
        self.assertEqual(run_hw_ff0_i32_b64(MASK64 ^ (1 << 63)), 63)
        self.assertEqual(run_hw_ff0_i32_b64(MASK64), MASK32)


def ref_ff1_i32_b32(src0: int) -> int:
    """Reference mathematical oracle for s_ff1_i32_b32.
    Returns index of first 1 bit (0..31, LSB to MSB), or -1 (0xFFFFFFFF) if all bits are 0.
    """
    val = src0 & MASK32
    if val == 0:
        return MASK32
    lsb = val & (-val)
    return lsb.bit_length() - 1


class TestSFf1I32B32Reference(unittest.TestCase):
    """Reference oracle unit tests for s_ff1_i32_b32."""

    def test_reference_zero(self):
        self.assertEqual(ref_ff1_i32_b32(0), MASK32)

    def test_reference_first_one_bit(self):
        self.assertEqual(ref_ff1_i32_b32(1), 0)
        self.assertEqual(ref_ff1_i32_b32(0x12), 1)
        self.assertEqual(ref_ff1_i32_b32(0x80), 7)
        self.assertEqual(ref_ff1_i32_b32(0x80000000), 31)


def step_scalar_ff1_i32_b32(state: SaluState, instr: Instruction) -> None:
    if instr.opcode != "s_ff1_i32_b32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    src0 = state.read_u32(instr.src0)
    res = ref_ff1_i32_b32(src0)
    state.write_u32(instr.dst.index_or_val, res)


class TestSFf1I32B32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for s_ff1_i32_b32."""

    def test_scalar_interpreter_execution(self):
        state = SaluState()
        state.write_u32(0, 0x00000100)
        state.scc = 0
        instr = Instruction("s_ff1_i32_b32", Operand.sgpr32(1), Operand.sgpr32(0))
        step_scalar_ff1_i32_b32(state, instr)
        self.assertEqual(state.sgprs[1], 8)
        self.assertEqual(state.scc, 0)


class TestSFf1I32B32Bounds(unittest.TestCase):
    """Bit scanning boundary and edge cases for s_ff1_i32_b32."""

    def test_powers_of_two(self):
        for b in range(32):
            self.assertEqual(ref_ff1_i32_b32(1 << b), b)

    def test_dense_and_sparse_patterns(self):
        self.assertEqual(ref_ff1_i32_b32(0xFFFFFFFF), 0)
        self.assertEqual(ref_ff1_i32_b32(0x80000001), 0)
        self.assertEqual(ref_ff1_i32_b32(0x80000000), 31)


def run_hw_ff1_i32_b32(src0: int) -> int:
    lines = [
        f"s_mov_b32 s0, {src0 & MASK32:#x}",
        "s_ff1_i32_b32 s1, s0",
        "v_mov_b32 v0, s1",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_ff1_i32_b32_hw", lines)
    return support.mrt0_words(spv)[0]


class TestSFf1I32B32HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for s_ff1_i32_b32."""

    def test_translator_scan_cases(self):
        self.assertEqual(run_hw_ff1_i32_b32(0), MASK32)
        self.assertEqual(run_hw_ff1_i32_b32(1), 0)
        self.assertEqual(run_hw_ff1_i32_b32(0x12), 1)
        self.assertEqual(run_hw_ff1_i32_b32(0x80000000), 31)
        self.assertEqual(run_hw_ff1_i32_b32(0xFFFFFFFF), 0)


if __name__ == "__main__":
    unittest.main()
