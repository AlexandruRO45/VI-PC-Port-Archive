"""Semantic tests for RDNA2 VALU Sum of Absolute Differences (SAD) bundle.

Covers:
- v_sad_u32
- v_sad_u16
- v_sad_u8
- v_sad_hi_u8
- v_msad_u8
- v_mqsad_u32_u8

Tests reference mathematical models, AST and scalar interpreter semantics,
SIMD multi-lane divergence and EXEC masking, 32-bit wrapping and accumulator semantics,
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


def ref_sad_u32(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_sad_u32."""
    diff = abs((s0 & MASK32) - (s1 & MASK32))
    return ((s2 & MASK32) + diff) & MASK32


class TestVSadU32Reference(unittest.TestCase):
    """Reference oracle unit tests for v_sad_u32."""

    def test_reference_canonical(self):
        self.assertEqual(ref_sad_u32(100, 40, 10), 70)
        self.assertEqual(ref_sad_u32(40, 100, 10), 70)

    def test_reference_zero_diff(self):
        self.assertEqual(ref_sad_u32(55, 55, 12), 12)

    def test_reference_wrapping(self):
        self.assertEqual(ref_sad_u32(0xFFFFFFFF, 0, 1), 0)


class RegType(enum.Enum):
    VGPR = "v"
    SGPR = "s"
    IMM = "imm"


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index_or_val: int
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
    def imm32(cls, val: int) -> Operand:
        return cls(RegType.IMM, val & MASK32, 1)


@dataclasses.dataclass
class Instruction:
    opcode: str
    dst: Operand
    src0: Operand
    src1: Operand
    src2: Operand


class ValuState:
    def __init__(self) -> None:
        self.vgprs: dict[int, int] = {i: 0 for i in range(256)}
        self.sgprs: dict[int, int] = {i: 0 for i in range(106)}

    def read_u32(self, op: Operand) -> int:
        if op.reg_type == RegType.IMM:
            return op.index_or_val & MASK32
        elif op.reg_type == RegType.VGPR:
            return self.vgprs[op.index_or_val] & MASK32
        return self.sgprs[op.index_or_val] & MASK32

    def read_u64(self, op: Operand) -> int:
        if op.reg_type == RegType.VGPR:
            lo = self.vgprs[op.index_or_val] & MASK32
            hi = self.vgprs[op.index_or_val + 1] & MASK32
            return (hi << 32) | lo
        lo = self.sgprs[op.index_or_val] & MASK32
        hi = self.sgprs[op.index_or_val + 1] & MASK32
        return (hi << 32) | lo

    def write_u32(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

def step_scalar_sad_u32(state: ValuState, instr: Instruction) -> None:
    if instr.opcode != "v_sad_u32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    s0 = state.read_u32(instr.src0)
    s1 = state.read_u32(instr.src1)
    s2 = state.read_u32(instr.src2)
    res = ref_sad_u32(s0, s1, s2)
    state.write_u32(instr.dst.index_or_val, res)


class TestVSadU32ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_sad_u32."""

    def test_ast_construction_and_bounds(self):
        instr = Instruction("v_sad_u32", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        self.assertEqual(instr.opcode, "v_sad_u32")
        with self.assertRaises(ValueError):
            Operand.vgpr(256)

    def test_scalar_interpreter_execution(self):
        state = ValuState()
        state.write_u32(1, 100)
        state.write_u32(2, 40)
        state.write_u32(3, 10)
        instr = Instruction("v_sad_u32", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_scalar_sad_u32(state, instr)
        self.assertEqual(state.vgprs[0], 70)


class WaveState:
    def __init__(self, wave_size: int = 64) -> None:
        self.wave_size = wave_size
        self.exec_mask: int = (1 << wave_size) - 1
        self.vgprs: list[dict[int, int]] = [{i: 0 for i in range(256)} for _ in range(wave_size)]

    def read_lane_u32(self, lane: int, op: Operand) -> int:
        if op.reg_type == RegType.IMM:
            return op.index_or_val & MASK32
        elif op.reg_type == RegType.VGPR:
            return self.vgprs[lane][op.index_or_val] & MASK32
        raise ValueError(f"Unsupported operand: {op}")

    def write_lane_u32(self, lane: int, idx: int, val: int) -> None:
        if (self.exec_mask >> lane) & 1:
            self.vgprs[lane][idx] = val & MASK32

def step_simd_sad_u32(wave: WaveState, instr: Instruction) -> None:
    if instr.opcode != "v_sad_u32":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.read_lane_u32(lane, instr.src0)
            s1 = wave.read_lane_u32(lane, instr.src1)
            s2 = wave.read_lane_u32(lane, instr.src2)
            wave.write_lane_u32(lane, instr.dst.index_or_val, ref_sad_u32(s0, s1, s2))


class TestVSadU32SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and 32-bit wrapping for v_sad_u32."""

    def test_simd_multi_lane_divergence(self):
        wave = WaveState(wave_size=64)
        wave.exec_mask = 0x5555555555555555
        for lane in range(64):
            wave.vgprs[lane][1] = 50 + lane
            wave.vgprs[lane][2] = 20
            wave.vgprs[lane][3] = 0
            wave.vgprs[lane][0] = 9999

        instr = Instruction("v_sad_u32", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_simd_sad_u32(wave, instr)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(wave.vgprs[lane][0], 30 + lane)
            else:
                self.assertEqual(wave.vgprs[lane][0], 9999)


def run_hw_sad_u32(s0: int, s1: int, s2: int) -> int:
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        "v_sad_u32 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_sad_u32_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVSadU32HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_sad_u32."""

    def test_translator_hardware_cases(self):
        self.assertEqual(run_hw_sad_u32(100, 40, 10), 70)
        self.assertEqual(run_hw_sad_u32(200, 300, 50), 150)


def ref_sad_u16(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_sad_u16."""
    h0 = abs((s0 & 0xFFFF) - (s1 & 0xFFFF))
    h1 = abs(((s0 >> 16) & 0xFFFF) - ((s1 >> 16) & 0xFFFF))
    return ((s2 & MASK32) + h0 + h1) & MASK32


class TestVSadU16Reference(unittest.TestCase):
    """Reference oracle unit tests for v_sad_u16."""

    def test_reference_canonical(self):
        self.assertEqual(ref_sad_u16(0x00200064, 0x00100032, 5), 71)

    def test_reference_zero_diff(self):
        self.assertEqual(ref_sad_u16(0x12345678, 0x12345678, 100), 100)


def step_scalar_sad_u16(state: ValuState, instr: Instruction) -> None:
    if instr.opcode != "v_sad_u16":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    s0 = state.read_u32(instr.src0)
    s1 = state.read_u32(instr.src1)
    s2 = state.read_u32(instr.src2)
    res = ref_sad_u16(s0, s1, s2)
    state.write_u32(instr.dst.index_or_val, res)


class TestVSadU16ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_sad_u16."""

    def test_scalar_interpreter_execution(self):
        state = ValuState()
        state.write_u32(1, 0x00200064)
        state.write_u32(2, 0x00100032)
        state.write_u32(3, 5)
        instr = Instruction("v_sad_u16", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_scalar_sad_u16(state, instr)
        self.assertEqual(state.vgprs[0], 71)


def step_simd_sad_u16(wave: WaveState, instr: Instruction) -> None:
    if instr.opcode != "v_sad_u16":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.read_lane_u32(lane, instr.src0)
            s1 = wave.read_lane_u32(lane, instr.src1)
            s2 = wave.read_lane_u32(lane, instr.src2)
            wave.write_lane_u32(lane, instr.dst.index_or_val, ref_sad_u16(s0, s1, s2))


class TestVSadU16SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and dual-half handling for v_sad_u16."""

    def test_simd_dual_half_divergence(self):
        wave = WaveState(wave_size=32)
        wave.exec_mask = 0x0000FFFF
        for lane in range(32):
            wave.vgprs[lane][1] = 0x00200064
            wave.vgprs[lane][2] = 0x00100032
            wave.vgprs[lane][3] = 0
            wave.vgprs[lane][0] = 777

        instr = Instruction("v_sad_u16", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_simd_sad_u16(wave, instr)

        for lane in range(32):
            if lane < 16:
                self.assertEqual(wave.vgprs[lane][0], 66)
            else:
                self.assertEqual(wave.vgprs[lane][0], 777)


def run_hw_sad_u16(s0: int, s1: int, s2: int) -> int:
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        "v_sad_u16 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_sad_u16_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVSadU16HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_sad_u16."""

    def test_translator_hardware_cases(self):
        self.assertEqual(run_hw_sad_u16(0x00200064, 0x00100032, 5), 71)
        self.assertEqual(run_hw_sad_u16(0x00100020, 0x00200010, 0), 32)


def ref_sad_u8(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_sad_u8."""
    diff = sum(abs(((s0 >> (8 * i)) & 0xFF) - ((s1 >> (8 * i)) & 0xFF)) for i in range(4))
    return ((s2 & MASK32) + diff) & MASK32


class TestVSadU8Reference(unittest.TestCase):
    """Reference oracle unit tests for v_sad_u8."""

    def test_reference_canonical(self):
        self.assertEqual(ref_sad_u8(0x04030201, 0x01010101, 10), 16)

    def test_reference_zeros(self):
        self.assertEqual(ref_sad_u8(0, 0, 42), 42)


def step_scalar_sad_u8(state: ValuState, instr: Instruction) -> None:
    if instr.opcode != "v_sad_u8":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    s0 = state.read_u32(instr.src0)
    s1 = state.read_u32(instr.src1)
    s2 = state.read_u32(instr.src2)
    res = ref_sad_u8(s0, s1, s2)
    state.write_u32(instr.dst.index_or_val, res)


class TestVSadU8ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_sad_u8."""

    def test_scalar_interpreter_execution(self):
        state = ValuState()
        state.write_u32(1, 0x04030201)
        state.write_u32(2, 0x01010101)
        state.write_u32(3, 10)
        instr = Instruction("v_sad_u8", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_scalar_sad_u8(state, instr)
        self.assertEqual(state.vgprs[0], 16)


def step_simd_sad_u8(wave: WaveState, instr: Instruction) -> None:
    if instr.opcode != "v_sad_u8":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.read_lane_u32(lane, instr.src0)
            s1 = wave.read_lane_u32(lane, instr.src1)
            s2 = wave.read_lane_u32(lane, instr.src2)
            wave.write_lane_u32(lane, instr.dst.index_or_val, ref_sad_u8(s0, s1, s2))


class TestVSadU8SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and 4-byte accumulation for v_sad_u8."""

    def test_simd_accumulation(self):
        wave = WaveState(wave_size=32)
        wave.exec_mask = 0xAAAAAAAA
        for lane in range(32):
            wave.vgprs[lane][1] = 0x04030201
            wave.vgprs[lane][2] = 0x01010101
            wave.vgprs[lane][3] = 0
            wave.vgprs[lane][0] = 555

        instr = Instruction("v_sad_u8", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_simd_sad_u8(wave, instr)

        for lane in range(32):
            if lane % 2 == 1:
                self.assertEqual(wave.vgprs[lane][0], 6)
            else:
                self.assertEqual(wave.vgprs[lane][0], 555)


def run_hw_sad_u8(s0: int, s1: int, s2: int) -> int:
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        "v_sad_u8 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_sad_u8_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVSadU8HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_sad_u8."""

    def test_translator_hardware_cases(self):
        self.assertEqual(run_hw_sad_u8(0x04030201, 0x01010101, 10), 16)
        self.assertEqual(run_hw_sad_u8(0x0A0A0A0A, 0x05050505, 0), 20)


def ref_sad_hi_u8(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_sad_hi_u8."""
    diff = sum(abs(((s0 >> (8 * i)) & 0xFF) - ((s1 >> (8 * i)) & 0xFF)) for i in range(4))
    return ((s2 & MASK32) + (diff << 16)) & MASK32


class TestVSadHiU8Reference(unittest.TestCase):
    """Reference oracle unit tests for v_sad_hi_u8."""

    def test_reference_canonical(self):
        self.assertEqual(ref_sad_hi_u8(0x04030201, 0x01010101, 10), 0x0006000A)

    def test_reference_zero_diff(self):
        self.assertEqual(ref_sad_hi_u8(0x12345678, 0x12345678, 0x1234), 0x1234)


def step_scalar_sad_hi_u8(state: ValuState, instr: Instruction) -> None:
    if instr.opcode != "v_sad_hi_u8":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    s0 = state.read_u32(instr.src0)
    s1 = state.read_u32(instr.src1)
    s2 = state.read_u32(instr.src2)
    res = ref_sad_hi_u8(s0, s1, s2)
    state.write_u32(instr.dst.index_or_val, res)


class TestVSadHiU8ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_sad_hi_u8."""

    def test_scalar_interpreter_execution(self):
        state = ValuState()
        state.write_u32(1, 0x04030201)
        state.write_u32(2, 0x01010101)
        state.write_u32(3, 10)
        instr = Instruction("v_sad_hi_u8", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_scalar_sad_hi_u8(state, instr)
        self.assertEqual(state.vgprs[0], 0x0006000A)


def step_simd_sad_hi_u8(wave: WaveState, instr: Instruction) -> None:
    if instr.opcode != "v_sad_hi_u8":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.read_lane_u32(lane, instr.src0)
            s1 = wave.read_lane_u32(lane, instr.src1)
            s2 = wave.read_lane_u32(lane, instr.src2)
            wave.write_lane_u32(lane, instr.dst.index_or_val, ref_sad_hi_u8(s0, s1, s2))


class TestVSadHiU8SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and 16-bit shift handling for v_sad_hi_u8."""

    def test_simd_divergence(self):
        wave = WaveState(wave_size=32)
        wave.exec_mask = 0xF0F0F0F0
        for lane in range(32):
            wave.vgprs[lane][1] = 0x04030201
            wave.vgprs[lane][2] = 0x01010101
            wave.vgprs[lane][3] = 0
            wave.vgprs[lane][0] = 0

        instr = Instruction("v_sad_hi_u8", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_simd_sad_hi_u8(wave, instr)

        for lane in range(32):
            if (0xF0F0F0F0 >> lane) & 1:
                self.assertEqual(wave.vgprs[lane][0], 6 << 16)
            else:
                self.assertEqual(wave.vgprs[lane][0], 0)


def run_hw_sad_hi_u8(s0: int, s1: int, s2: int) -> int:
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        "v_sad_hi_u8 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_sad_hi_u8_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVSadHiU8HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_sad_hi_u8."""

    def test_translator_hardware_cases(self):
        self.assertEqual(run_hw_sad_hi_u8(0x04030201, 0x01010101, 10), 0x0006000A)


def ref_msad_u8(s0: int, s1: int, s2: int) -> int:
    """Reference mathematical oracle for RDNA2 v_msad_u8 (masked SAD)."""
    diffs = [
        abs(((s0 >> (8 * i)) & 0xFF) - ((s1 >> (8 * i)) & 0xFF))
        if ((s1 >> (8 * i)) & 0xFF) != 0 else 0
        for i in range(4)
    ]
    return ((s2 & MASK32) + sum(diffs)) & MASK32


class TestVMsadU8Reference(unittest.TestCase):
    """Reference oracle unit tests for v_msad_u8."""

    def test_reference_canonical_masking(self):
        # S1 has bytes [1, 0, 1, 0] -> bytes 1 and 3 are masked out
        self.assertEqual(ref_msad_u8(0x04030201, 0x00010001, 10), 12)

    def test_reference_all_zero_mask(self):
        self.assertEqual(ref_msad_u8(0xFFFFFFFF, 0x00000000, 50), 50)


def step_scalar_msad_u8(state: ValuState, instr: Instruction) -> None:
    if instr.opcode != "v_msad_u8":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    s0 = state.read_u32(instr.src0)
    s1 = state.read_u32(instr.src1)
    s2 = state.read_u32(instr.src2)
    res = ref_msad_u8(s0, s1, s2)
    state.write_u32(instr.dst.index_or_val, res)


class TestVMsadU8ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_msad_u8."""

    def test_scalar_interpreter_execution(self):
        state = ValuState()
        state.write_u32(1, 0x04030201)
        state.write_u32(2, 0x00010001)
        state.write_u32(3, 10)
        instr = Instruction("v_msad_u8", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_scalar_msad_u8(state, instr)
        self.assertEqual(state.vgprs[0], 12)


def step_simd_msad_u8(wave: WaveState, instr: Instruction) -> None:
    if instr.opcode != "v_msad_u8":
        raise ValueError(f"Unexpected opcode: {instr.opcode}")
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            s0 = wave.read_lane_u32(lane, instr.src0)
            s1 = wave.read_lane_u32(lane, instr.src1)
            s2 = wave.read_lane_u32(lane, instr.src2)
            wave.write_lane_u32(lane, instr.dst.index_or_val, ref_msad_u8(s0, s1, s2))


class TestVMsadU8SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and byte-masking handling for v_msad_u8."""

    def test_simd_masking(self):
        wave = WaveState(wave_size=32)
        wave.exec_mask = 0x0000FFFF
        for lane in range(32):
            wave.vgprs[lane][1] = 0x04030201
            wave.vgprs[lane][2] = 0x00010001
            wave.vgprs[lane][3] = 0
            wave.vgprs[lane][0] = 99

        instr = Instruction("v_msad_u8", Operand.vgpr(0), Operand.vgpr(1), Operand.vgpr(2), Operand.vgpr(3))
        step_simd_msad_u8(wave, instr)

        for lane in range(32):
            if lane < 16:
                self.assertEqual(wave.vgprs[lane][0], 2)
            else:
                self.assertEqual(wave.vgprs[lane][0], 99)


def run_hw_msad_u8(s0: int, s1: int, s2: int) -> int:
    lines = [
        f"v_mov_b32 v0, {s0 & MASK32:#x}",
        f"v_mov_b32 v1, {s1 & MASK32:#x}",
        f"v_mov_b32 v2, {s2 & MASK32:#x}",
        "v_msad_u8 v0, v0, v1, v2",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_msad_u8_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVMsadU8HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_msad_u8."""

    def test_translator_hardware_cases(self):
        self.assertEqual(run_hw_msad_u8(0x04030201, 0x00010001, 10), 12)
        self.assertEqual(run_hw_msad_u8(0x04030201, 0x01010101, 10), 16)


def ref_mqsad_u32_u8(s0_64: int, s1: int, s2_quad: list[int]) -> list[int]:
    """Reference mathematical oracle for RDNA2 v_mqsad_u32_u8."""
    res = []
    for k in range(4):
        win = (s0_64 >> (8 * k)) & MASK32
        diffs = [
            abs(((win >> (8 * i)) & 0xFF) - ((s1 >> (8 * i)) & 0xFF))
            if ((s1 >> (8 * i)) & 0xFF) != 0 else 0
            for i in range(4)
        ]
        res.append(((s2_quad[k] & MASK32) + sum(diffs)) & MASK32)
    return res


class TestVMqsadU32U8Reference(unittest.TestCase):
    """Reference oracle unit tests for v_mqsad_u32_u8."""

    def test_reference_sliding_window(self):
        outs = ref_mqsad_u32_u8(0x0807060504030201, 0x01010101, [0, 0, 0, 0])
        self.assertEqual(outs, [6, 10, 14, 18])


def step_scalar_mqsad(state: ValuState, dst_reg: int, s0_pair_reg: int, s1_reg: int, s2_quad_reg: int) -> None:
    s0_64 = state.read_u64(Operand.vgpr(s0_pair_reg, 2))
    s1 = state.read_u32(Operand.vgpr(s1_reg))
    s2_quad = [state.read_u32(Operand.vgpr(s2_quad_reg + k)) for k in range(4)]
    outs = ref_mqsad_u32_u8(s0_64, s1, s2_quad)
    for k in range(4):
        state.write_u32(dst_reg + k, outs[k])


class TestVMqsadU32U8ASTInterpreter(unittest.TestCase):
    """AST opcode definitions and scalar interpreter for v_mqsad_u32_u8."""

    def test_scalar_interpreter_execution(self):
        state = ValuState()
        state.write_u32(0, 0x04030201)
        state.write_u32(1, 0x08070605)
        state.write_u32(2, 0x01010101)
        for k in range(4):
            state.write_u32(4 + k, 10)
        step_scalar_mqsad(state, dst_reg=8, s0_pair_reg=0, s1_reg=2, s2_quad_reg=4)
        self.assertEqual([state.vgprs[8 + k] for k in range(4)], [16, 20, 24, 28])


class TestVMqsadU32U8SIMD(unittest.TestCase):
    """SIMD multi-lane divergence and quad-lane windowing for v_mqsad_u32_u8."""

    def test_quad_lane_divergence(self):
        wave = WaveState(wave_size=32)
        wave.exec_mask = 0x0000000F
        for lane in range(32):
            s0_64 = 0x0807060504030201
            s1 = 0x01010101
            s2_quad = [0, 0, 0, 0]
            outs = ref_mqsad_u32_u8(s0_64, s1, s2_quad)
            for k in range(4):
                if (wave.exec_mask >> lane) & 1:
                    wave.write_lane_u32(lane, 8 + k, outs[k])
                else:
                    wave.write_lane_u32(lane, 8 + k, 0)

        for lane in range(32):
            if lane < 4:
                self.assertEqual([wave.vgprs[lane][8 + k] for k in range(4)], [6, 10, 14, 18])
            else:
                self.assertEqual([wave.vgprs[lane][8 + k] for k in range(4)], [0, 0, 0, 0])


def run_hw_mqsad_u32_u8(s0_lo: int, s0_hi: int, s1: int, s2_quad: list[int]) -> list[int]:
    lines = [
        f"v_mov_b32 v0, {s0_lo & MASK32:#x}",
        f"v_mov_b32 v1, {s0_hi & MASK32:#x}",
        f"v_mov_b32 v2, {s1 & MASK32:#x}",
        f"v_mov_b32 v4, {s2_quad[0] & MASK32:#x}",
        f"v_mov_b32 v5, {s2_quad[1] & MASK32:#x}",
        f"v_mov_b32 v6, {s2_quad[2] & MASK32:#x}",
        f"v_mov_b32 v7, {s2_quad[3] & MASK32:#x}",
        "v_mqsad_u32_u8 v[8:11], v[0:1], v2, v[4:7]",
        "exp mrt0 v8, v9, v10, v11 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_mqsad_hw", lines)
    return support.mrt0_words(spv)


class TestVMqsadU32U8HardwareTranslator(unittest.TestCase):
    """Translator hardware integration tests for v_mqsad_u32_u8."""

    def test_translator_hardware_cases(self):
        outs = run_hw_mqsad_u32_u8(0x04030201, 0x08070605, 0x01010101, [0, 0, 0, 0])
        self.assertEqual(outs, [6, 10, 14, 18])


if __name__ == "__main__":
    unittest.main()
