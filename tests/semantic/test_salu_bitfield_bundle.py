"""Semantic tests for RDNA2 SALU Bitfield Extract and Mask bundle.

Covers:
- s_bfe_u32
- s_bfe_i32
- s_bfe_u64
- s_bfe_i64
- s_bfm_b32
- s_bfm_b64

Tests reference mathematical models, AST and scalar interpreter semantics,
status flags (SCC), boundary and sign-extension handling,
and hardware translation into SPIR-V.
"""
from __future__ import annotations

import dataclasses
import enum
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
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_bfe_u32(src0: int, ctl: int) -> tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_bfe_u32. Returns (dst, scc)."""
    offset = ctl & 31
    width = (ctl >> 16) & 0x7F
    w = min(width, 32 - offset)
    if w == 0:
        val = 0
    else:
        val = ((src0 & MASK32) >> offset) & ((1 << w) - 1)
    scc = 1 if val != 0 else 0
    return val, scc


class TestSBfeU32Reference(unittest.TestCase):
    """Reference oracle unit tests for s_bfe_u32."""

    def test_reference_canonical(self):
        val, scc = ref_bfe_u32(0x12345678, 4 | (8 << 16))
        self.assertEqual(val, 0x67)
        self.assertEqual(scc, 1)

    def test_reference_zero_result(self):
        val, scc = ref_bfe_u32(0x12345608, 4 | (4 << 16))
        self.assertEqual(val, 0)
        self.assertEqual(scc, 0)

    def test_reference_zero_width(self):
        val, scc = ref_bfe_u32(0xFFFFFFFF, 4 | (0 << 16))
        self.assertEqual(val, 0)
        self.assertEqual(scc, 0)


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
    src1: Operand


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

def step_scalar_bfe_u32(state: SaluState, instr: Instruction) -> None:
    if instr.opcode != "s_bfe_u32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    src0 = state.read_u32(instr.src0)
    ctl = state.read_u32(instr.src1)
    val, scc = ref_bfe_u32(src0, ctl)
    state.write_u32(instr.dst.index_or_val, val)
    state.scc = scc


class TestSBfeU32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for s_bfe_u32."""

    def test_ast_construction_and_bounds(self):
        instr = Instruction("s_bfe_u32", Operand.sgpr32(2), Operand.sgpr32(0), Operand.sgpr32(1))
        self.assertEqual(instr.opcode, "s_bfe_u32")
        self.assertEqual(instr.dst.index_or_val, 2)
        with self.assertRaises(ValueError):
            Operand.sgpr32(106)

    def test_scalar_interpreter_execution(self):
        state = SaluState()
        state.write_u32(0, 0xAABBCCDD)
        state.write_u32(1, 8 | (8 << 16))
        instr = Instruction("s_bfe_u32", Operand.sgpr32(2), Operand.sgpr32(0), Operand.sgpr32(1))
        step_scalar_bfe_u32(state, instr)
        self.assertEqual(state.sgprs[2], 0xCC)
        self.assertEqual(state.scc, 1)


class TestSBfeU32Bounds(unittest.TestCase):
    """Bitfield boundary and status flag handling for s_bfe_u32."""

    def test_width_clamping_to_word_edge(self):
        # offset 28, width 10 -> clamped to 4 bits
        val, scc = ref_bfe_u32(0xF0000000, 28 | (10 << 16))
        self.assertEqual(val, 0xF)
        self.assertEqual(scc, 1)

    def test_offset_at_edge(self):
        val, scc = ref_bfe_u32(0x80000000, 31 | (1 << 16))
        self.assertEqual(val, 1)
        self.assertEqual(scc, 1)


def run_hw_bfe_u32(src0: int, ctl: int) -> int:
    lines = [
        f"s_mov_b32 s0, {src0 & MASK32:#x}",
        f"s_mov_b32 s1, {ctl & MASK32:#x}",
        "s_bfe_u32 s2, s0, s1",
        "v_mov_b32 v0, s2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_bfe_u32_hw", lines)
    return support.mrt0_words(spv)[0]


class TestSBfeU32HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for s_bfe_u32."""

    def test_translator_extract_cases(self):
        self.assertEqual(run_hw_bfe_u32(0x12345678, 4 | (8 << 16)), 0x67)
        self.assertEqual(run_hw_bfe_u32(0xDEADBEEF, 0 | (16 << 16)), 0xBEEF)
        self.assertEqual(run_hw_bfe_u32(0xDEADBEEF, 16 | (16 << 16)), 0xDEAD)


def ref_bfe_i32(src0: int, ctl: int) -> tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_bfe_i32. Returns (dst, scc)."""
    offset = ctl & 31
    width = (ctl >> 16) & 0x7F
    w = min(width, 32 - offset)
    if w == 0:
        val = 0
    else:
        raw = ((src0 & MASK32) >> offset) & ((1 << w) - 1)
        sign_bit = 1 << (w - 1)
        val = ((raw ^ sign_bit) - sign_bit) & MASK32
    scc = 1 if val != 0 else 0
    return val, scc


class TestSBfeI32Reference(unittest.TestCase):
    """Reference oracle unit tests for s_bfe_i32."""

    def test_reference_positive_field(self):
        val, scc = ref_bfe_i32(0x00000070, 4 | (4 << 16))
        self.assertEqual(val, 7)
        self.assertEqual(scc, 1)

    def test_reference_negative_field_sign_extended(self):
        val, scc = ref_bfe_i32(0x000000F0, 4 | (4 << 16))
        self.assertEqual(val, MASK32)
        self.assertEqual(scc, 1)


def step_scalar_bfe_i32(state: SaluState, instr: Instruction) -> None:
    if instr.opcode != "s_bfe_i32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    src0 = state.read_u32(instr.src0)
    ctl = state.read_u32(instr.src1)
    val, scc = ref_bfe_i32(src0, ctl)
    state.write_u32(instr.dst.index_or_val, val)
    state.scc = scc


class TestSBfeI32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for s_bfe_i32."""

    def test_scalar_interpreter_execution(self):
        state = SaluState()
        state.write_u32(0, 0x00000F00)
        state.write_u32(1, 8 | (4 << 16))
        instr = Instruction("s_bfe_i32", Operand.sgpr32(2), Operand.sgpr32(0), Operand.sgpr32(1))
        step_scalar_bfe_i32(state, instr)
        self.assertEqual(state.sgprs[2], MASK32)
        self.assertEqual(state.scc, 1)


class TestSBfeI32Bounds(unittest.TestCase):
    """Status flags and sign-extension boundary handling for s_bfe_i32."""

    def test_sign_extension_at_word_top(self):
        # 3-bit negative field at top: bit 31 set
        val, scc = ref_bfe_i32(0x80000000, 31 | (1 << 16))
        self.assertEqual(val, MASK32)
        self.assertEqual(scc, 1)

    def test_zero_field(self):
        val, scc = ref_bfe_i32(0x00000000, 10 | (5 << 16))
        self.assertEqual(val, 0)
        self.assertEqual(scc, 0)


def run_hw_bfe_i32(src0: int, ctl: int) -> int:
    lines = [
        f"s_mov_b32 s0, {src0 & MASK32:#x}",
        f"s_mov_b32 s1, {ctl & MASK32:#x}",
        "s_bfe_i32 s2, s0, s1",
        "v_mov_b32 v0, s2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_bfe_i32_hw", lines)
    return support.mrt0_words(spv)[0]


class TestSBfeI32HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for s_bfe_i32."""

    def test_translator_extract_cases(self):
        self.assertEqual(run_hw_bfe_i32(0x00000070, 4 | (4 << 16)), 7)
        self.assertEqual(run_hw_bfe_i32(0x000000F0, 4 | (4 << 16)), MASK32)


def ref_bfe_u64(src0: int, ctl: int) -> tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_bfe_u64. Returns (dst, scc)."""
    offset = ctl & 63
    width = (ctl >> 16) & 0x7F
    w = min(width, 64 - offset)
    if w == 0:
        val = 0
    else:
        val = ((src0 & MASK64) >> offset) & ((1 << w) - 1)
    scc = 1 if val != 0 else 0
    return val, scc


class TestSBfeU64Reference(unittest.TestCase):
    """Reference oracle unit tests for s_bfe_u64."""

    def test_reference_64bit_field(self):
        val, scc = ref_bfe_u64(0x12345678_9ABCDEF0, 32 | (16 << 16))
        self.assertEqual(val, 0x5678)
        self.assertEqual(scc, 1)

    def test_reference_cross_dword(self):
        val, scc = ref_bfe_u64(0x0000000F_F0000000, 28 | (8 << 16))
        self.assertEqual(val, 0xFF)
        self.assertEqual(scc, 1)


def step_scalar_bfe_u64(state: SaluState, instr: Instruction) -> None:
    if instr.opcode != "s_bfe_u64":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    src0 = state.read_u64(instr.src0)
    ctl = state.read_u32(instr.src1)
    val, scc = ref_bfe_u64(src0, ctl)
    state.write_u64(instr.dst.index_or_val, val)
    state.scc = scc


class TestSBfeU64ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for s_bfe_u64."""

    def test_scalar_interpreter_execution(self):
        state = SaluState()
        state.write_u64(0, 0x12345678_9ABCDEF0)
        state.write_u32(2, 32 | (16 << 16))
        instr = Instruction("s_bfe_u64", Operand.sgpr64(4), Operand.sgpr64(0), Operand.sgpr32(2))
        step_scalar_bfe_u64(state, instr)
        self.assertEqual(state.sgprs[4], 0x5678)
        self.assertEqual(state.sgprs[5], 0x0)
        self.assertEqual(state.scc, 1)


class TestSBfeU64Bounds(unittest.TestCase):
    """Status flags and 64-bit boundary handling for s_bfe_u64."""

    def test_extract_at_bit_63(self):
        val, scc = ref_bfe_u64(0x80000000_00000000, 63 | (1 << 16))
        self.assertEqual(val, 1)
        self.assertEqual(scc, 1)

    def test_zero_width_64bit(self):
        val, scc = ref_bfe_u64(MASK64, 16 | (0 << 16))
        self.assertEqual(val, 0)
        self.assertEqual(scc, 0)


def run_hw_bfe_u64(src0: int, ctl: int) -> tuple[int, int]:
    lo = src0 & MASK32
    hi = (src0 >> 32) & MASK32
    lines = [
        f"s_mov_b32 s0, {lo:#x}",
        f"s_mov_b32 s1, {hi:#x}",
        f"s_mov_b32 s2, {ctl & MASK32:#x}",
        "s_bfe_u64 s[4:5], s[0:1], s2",
        "v_mov_b32 v0, s4",
        "v_mov_b32 v1, s5",
        "exp mrt0 v0, v1, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_bfe_u64_hw", lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


class TestSBfeU64HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for s_bfe_u64."""

    def test_translator_extract_cases(self):
        rlo, rhi = run_hw_bfe_u64(0x12345678_9ABCDEF0, 32 | (16 << 16))
        self.assertEqual(rlo, 0x5678)
        self.assertEqual(rhi, 0x0)


def ref_bfe_i64(src0: int, ctl: int) -> tuple[int, int]:
    """Reference mathematical oracle for RDNA2 s_bfe_i64. Returns (dst, scc)."""
    offset = ctl & 63
    width = (ctl >> 16) & 0x7F
    w = min(width, 64 - offset)
    if w == 0:
        val = 0
    else:
        raw = ((src0 & MASK64) >> offset) & ((1 << w) - 1)
        sign_bit = 1 << (w - 1)
        val = ((raw ^ sign_bit) - sign_bit) & MASK64
    scc = 1 if val != 0 else 0
    return val, scc


class TestSBfeI64Reference(unittest.TestCase):
    """Reference oracle unit tests for s_bfe_i64."""

    def test_reference_positive(self):
        val, scc = ref_bfe_i64(0x00000000_80000000, 31 | (2 << 16))
        self.assertEqual(val, 1)
        self.assertEqual(scc, 1)

    def test_reference_sign_extended(self):
        val, scc = ref_bfe_i64(0x00000001_80000000, 31 | (2 << 16))
        self.assertEqual(val, MASK64)
        self.assertEqual(scc, 1)


def step_scalar_bfe_i64(state: SaluState, instr: Instruction) -> None:
    if instr.opcode != "s_bfe_i64":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    src0 = state.read_u64(instr.src0)
    ctl = state.read_u32(instr.src1)
    val, scc = ref_bfe_i64(src0, ctl)
    state.write_u64(instr.dst.index_or_val, val)
    state.scc = scc


class TestSBfeI64ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for s_bfe_i64."""

    def test_scalar_interpreter_execution(self):
        state = SaluState()
        state.write_u64(0, 0x00000001_80000000)
        state.write_u32(2, 31 | (2 << 16))
        instr = Instruction("s_bfe_i64", Operand.sgpr64(4), Operand.sgpr64(0), Operand.sgpr32(2))
        step_scalar_bfe_i64(state, instr)
        self.assertEqual(state.sgprs[4], MASK32)
        self.assertEqual(state.sgprs[5], MASK32)
        self.assertEqual(state.scc, 1)


class TestSBfeI64Bounds(unittest.TestCase):
    """Status flags and 64-bit sign-extension handling for s_bfe_i64."""

    def test_sign_bit_at_bit_63(self):
        val, scc = ref_bfe_i64(0x80000000_00000000, 63 | (1 << 16))
        self.assertEqual(val, MASK64)
        self.assertEqual(scc, 1)

    def test_positive_bit_at_bit_63(self):
        val, scc = ref_bfe_i64(0x00000000_00000000, 63 | (1 << 16))
        self.assertEqual(val, 0)
        self.assertEqual(scc, 0)


def run_hw_bfe_i64(src0: int, ctl: int) -> tuple[int, int]:
    lo = src0 & MASK32
    hi = (src0 >> 32) & MASK32
    lines = [
        f"s_mov_b32 s0, {lo:#x}",
        f"s_mov_b32 s1, {hi:#x}",
        f"s_mov_b32 s2, {ctl & MASK32:#x}",
        "s_bfe_i64 s[4:5], s[0:1], s2",
        "v_mov_b32 v0, s4",
        "v_mov_b32 v1, s5",
        "exp mrt0 v0, v1, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_bfe_i64_hw", lines)
    words = support.mrt0_words(spv)
    return words[0], words[1]


class TestSBfeI64HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for s_bfe_i64."""

    def test_translator_extract_cases(self):
        rlo, rhi = run_hw_bfe_i64(0x00000000_80000000, 31 | (2 << 16))
        self.assertEqual(rlo, 1)
        self.assertEqual(rhi, 0)


def ref_bfm_b32(width: int, offset: int) -> int:
    """Reference mathematical oracle for RDNA2 s_bfm_b32."""
    w = width & 31
    o = offset & 31
    return (((1 << w) - 1) << o) & MASK32


class TestSBfmB32Reference(unittest.TestCase):
    """Reference oracle unit tests for s_bfm_b32."""

    def test_reference_canonical(self):
        self.assertEqual(ref_bfm_b32(8, 4), 0xFF0)
        self.assertEqual(ref_bfm_b32(4, 0), 0xF)

    def test_reference_zero_width(self):
        self.assertEqual(ref_bfm_b32(0, 5), 0)


def step_scalar_bfm_b32(state: SaluState, instr: Instruction) -> None:
    if instr.opcode != "s_bfm_b32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    w = state.read_u32(instr.src0)
    o = state.read_u32(instr.src1)
    val = ref_bfm_b32(w, o)
    state.write_u32(instr.dst.index_or_val, val)


class TestSBfmB32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for s_bfm_b32."""

    def test_scalar_interpreter_execution(self):
        state = SaluState()
        state.write_u32(0, 16)
        state.write_u32(1, 8)
        instr = Instruction("s_bfm_b32", Operand.sgpr32(2), Operand.sgpr32(0), Operand.sgpr32(1))
        step_scalar_bfm_b32(state, instr)
        self.assertEqual(state.sgprs[2], 0x00FFFF00)


class TestSBfmB32Bounds(unittest.TestCase):
    """Mask wrapping and boundary handling for s_bfm_b32."""

    def test_max_width_and_offset_wrap(self):
        # width 31, offset 0 -> 0x7FFFFFFF
        self.assertEqual(ref_bfm_b32(31, 0), 0x7FFFFFFF)
        # width 32 mod 32 = 0
        self.assertEqual(ref_bfm_b32(32, 0), 0)
        # offset 32 mod 32 = 0
        self.assertEqual(ref_bfm_b32(8, 32), 0xFF)


def run_hw_bfm_b32(width: int, offset: int) -> int:
    lines = [
        f"s_mov_b32 s0, {width & MASK32:#x}",
        f"s_mov_b32 s1, {offset & MASK32:#x}",
        "s_bfm_b32 s2, s0, s1",
        "v_mov_b32 v0, s2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_bfm_b32_hw", lines)
    return support.mrt0_words(spv)[0]


class TestSBfmB32HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for s_bfm_b32."""

    def test_translator_mask_cases(self):
        self.assertEqual(run_hw_bfm_b32(8, 4), 0xFF0)
        self.assertEqual(run_hw_bfm_b32(16, 0), 0xFFFF)


def ref_bfm_b64(width: int, offset: int) -> int:
    """Reference mathematical oracle for RDNA2 s_bfm_b64."""
    w = width & 63
    o = offset & 63
    return (((1 << w) - 1) << o) & MASK64


class TestSBfmB64Reference(unittest.TestCase):
    """Reference oracle unit tests for s_bfm_b64."""

    def test_reference_64bit_masks(self):
        self.assertEqual(ref_bfm_b64(32, 0), 0xFFFFFFFF)
        self.assertEqual(ref_bfm_b64(16, 32), 0x0000FFFF_00000000)

    def test_reference_zero_width(self):
        self.assertEqual(ref_bfm_b64(0, 10), 0)


def step_scalar_bfm_b64(state: SaluState, instr: Instruction) -> None:
    if instr.opcode != "s_bfm_b64":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    w = state.read_u32(instr.src0)
    o = state.read_u32(instr.src1)
    val = ref_bfm_b64(w, o)
    state.write_u64(instr.dst.index_or_val, val)


class TestSBfmB64ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for s_bfm_b64."""

    def test_scalar_interpreter_execution(self):
        state = SaluState()
        state.write_u32(0, 8)
        state.write_u32(1, 32)
        instr = Instruction("s_bfm_b64", Operand.sgpr64(2), Operand.sgpr32(0), Operand.sgpr32(1))
        step_scalar_bfm_b64(state, instr)
        self.assertEqual(state.sgprs[2], 0)
        self.assertEqual(state.sgprs[3], 0xFF)


class TestSBfmB64Bounds(unittest.TestCase):
    """64-bit mask wrapping and boundary handling for s_bfm_b64."""

    def test_width_and_offset_modulo_64(self):
        # width 64 mod 64 = 0
        self.assertEqual(ref_bfm_b64(64, 0), 0)
        # width 63, offset 0 -> 0x7FFFFFFFFFFFFFFF
        self.assertEqual(ref_bfm_b64(63, 0), 0x7FFFFFFF_FFFFFFFF)
        # offset 64 mod 64 = 0
        self.assertEqual(ref_bfm_b64(8, 64), 0xFF)


if __name__ == "__main__":
    unittest.main()
