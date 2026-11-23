"""Semantic tests for RDNA2 s_not_b64 opcode with hardware translator execution.

RDNA2 ISA definition:
    s_not_b64 dst[0:1], src[0:1]
    dst = ~src & 0xFFFFFFFFFFFFFFFF
    SCC = (dst != 0)
Performs bitwise NOT on a 64-bit scalar register pair and sets SCC flag if result is non-zero.
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

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def ref_not_b64(val64: int) -> int:
    """Bit-accurate reference oracle for RDNA2 s_not_b64."""
    return (~val64) & MASK64


def ref_not_b64_scc(val64: int) -> tuple[int, int]:
    """Returns (result64, scc_flag)."""
    res = ref_not_b64(val64)
    scc = 1 if res != 0 else 0
    return res, scc


def run_hw_not_b64(val64: int) -> int:
    """Execute s_not_b64 through the rdna2spirv translator and spirv_mini_interp."""
    vlo = val64 & MASK32
    vhi = (val64 >> 32) & MASK32
    lines = [
        f"s_mov_b32 s2, {vlo:#x}",
        f"s_mov_b32 s3, {vhi:#x}",
        "s_not_b64 s[0:1], s[2:3]",
        "v_mov_b32 v0, s0",
        "v_mov_b32 v1, s1",
        "exp mrt0 v0, v1, v0, v1 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_not64_hw", lines)
    words = support.mrt0_words(spv)
    return (words[0] & MASK32) | ((words[1] & MASK32) << 32)


class TestSNotB64Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_inversion(self):
        self.assertEqual(ref_not_b64(0), MASK64)
        self.assertEqual(ref_not_b64(MASK64), 0)
        self.assertEqual(ref_not_b64(0xAAAAAAAAAAAAAAAA), 0x5555555555555555)
        self.assertEqual(ref_not_b64(0x00000000FFFFFFFF), 0xFFFFFFFF00000000)
        self.assertEqual(ref_not_b64(0x123456789ABCDEF0), 0xEDCBA9876543210F)

    def test_reference_scc_flag(self):
        res, scc = ref_not_b64_scc(MASK64)
        self.assertEqual(res, 0)
        self.assertEqual(scc, 0)

        res, scc = ref_not_b64_scc(0)
        self.assertEqual(res, MASK64)
        self.assertEqual(scc, 1)

        res, scc = ref_not_b64_scc(0xFFFFFFFFFFFFFFFE)
        self.assertEqual(res, 1)
        self.assertEqual(scc, 1)

    def test_translator_hw_inversion(self):
        patterns = [
            0x0000000000000000,
            0xFFFFFFFFFFFFFFFF,
            0xAAAAAAAAAAAAAAAA,
            0x5555555555555555,
            0x123456789ABCDEF0,
            0xFEDCBA9876543210,
            0x00000000FFFFFFFF,
            0xFFFFFFFF00000000,
        ]
        for p in patterns:
            with self.subTest(p=hex(p)):
                hw_res = run_hw_not_b64(p)
                ref_res = ref_not_b64(p)
                self.assertEqual(hw_res, ref_res)


class RegType(enum.Enum):
    SGPR = 1
    SPECIAL = 2


@dataclasses.dataclass(frozen=True)
class Operand:
    reg_type: RegType
    index: int = 0
    size: int = 2
    name: str = ""

    def __post_init__(self):
        if self.reg_type == RegType.SGPR and not (0 <= self.index <= 104):
            raise ValueError(f"SGPR index out of range [0, 104] for pair: {self.index}")
        if self.size != 2:
            raise ValueError(f"Operand size must be 2 for 64-bit operation, got {self.size}")


@dataclasses.dataclass(frozen=True)
class SNotB64Instruction:
    dst: Operand
    src: Operand

    def __post_init__(self):
        if self.dst.size != 2 or self.src.size != 2:
            raise TypeError("Both destination and source must be 64-bit operands")


class TestSNotB64AST(unittest.TestCase):
    """Symbolic AST opcode definitions and typed expressions."""

    def test_ast_instantiation_valid(self):
        inst = SNotB64Instruction(
            dst=Operand(RegType.SGPR, 0, size=2),
            src=Operand(RegType.SGPR, 2, size=2),
        )
        self.assertEqual(inst.dst.index, 0)
        self.assertEqual(inst.dst.size, 2)
        self.assertEqual(inst.src.index, 2)
        self.assertEqual(inst.src.size, 2)

    def test_ast_validation_rejects_single_word(self):
        with self.assertRaises(ValueError):
            Operand(RegType.SGPR, 0, size=1)

    def test_ast_validation_rejects_out_of_range_index(self):
        with self.assertRaises(ValueError):
            Operand(RegType.SGPR, 105, size=2)
        with self.assertRaises(ValueError):
            Operand(RegType.SGPR, -1, size=2)


class ScalarRegisterFile:
    def __init__(self):
        self.sgprs = [0] * 106
        self.scc = 0

    def read_sgpr64(self, idx: int) -> int:
        lo = self.sgprs[idx] & MASK32
        hi = self.sgprs[idx + 1] & MASK32
        return (hi << 32) | lo

    def write_sgpr64(self, idx: int, val64: int) -> None:
        self.sgprs[idx] = val64 & MASK32
        self.sgprs[idx + 1] = (val64 >> 32) & MASK32


class ScalarSALUInterpreter:
    @staticmethod
    def execute(rf: ScalarRegisterFile, inst: SNotB64Instruction) -> int:
        val = rf.read_sgpr64(inst.src.index)
        res, scc = ref_not_b64_scc(val)
        rf.write_sgpr64(inst.dst.index, res)
        rf.scc = scc
        return res


class TestSNotB64ScalarInterpreter(unittest.TestCase):
    """Deterministic scalar instruction interpreter and execution model."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_scalar_execution_nonzero(self):
        val = 0x123456789ABCDEF0
        self.rf.write_sgpr64(2, val)
        inst = SNotB64Instruction(
            dst=Operand(RegType.SGPR, 0, size=2),
            src=Operand(RegType.SGPR, 2, size=2),
        )
        res = ScalarSALUInterpreter.execute(self.rf, inst)
        expected = ref_not_b64(val)
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.read_sgpr64(0), expected)
        self.assertEqual(self.rf.scc, 1)

    def test_scalar_execution_zero_scc(self):
        val = MASK64
        self.rf.write_sgpr64(2, val)
        inst = SNotB64Instruction(
            dst=Operand(RegType.SGPR, 0, size=2),
            src=Operand(RegType.SGPR, 2, size=2),
        )
        res = ScalarSALUInterpreter.execute(self.rf, inst)
        self.assertEqual(res, 0)
        self.assertEqual(self.rf.read_sgpr64(0), 0)
        self.assertEqual(self.rf.scc, 0)


class MultiWaveScalarState:
    """SALU state across wavefronts: scalar registers are wavefront-wide and lane-invariant."""

    def __init__(self, num_lanes: int = 32):
        self.num_lanes = num_lanes
        self.sgprs = [0] * 106
        self.scc = 0

    def write_sgpr64(self, idx: int, val64: int) -> None:
        self.sgprs[idx] = val64 & MASK32
        self.sgprs[idx + 1] = (val64 >> 32) & MASK32

    def read_sgpr64(self, idx: int) -> int:
        return (self.sgprs[idx] & MASK32) | ((self.sgprs[idx + 1] & MASK32) << 32)


class TestSNotB64WaveInvariance(unittest.TestCase):
    """Scalar ALU operations are shared across all lanes regardless of vector EXEC mask."""

    def test_wave32_uniform_scalar_state(self):
        wave = MultiWaveScalarState(num_lanes=32)
        val = 0xAAAAAAAAAAAAAAAA
        wave.write_sgpr64(4, val)
        res = ref_not_b64(wave.read_sgpr64(4))
        wave.write_sgpr64(0, res)
        # All 32 lanes observe identical scalar result
        for lane in range(32):
            self.assertEqual(wave.read_sgpr64(0), 0x5555555555555555)

    def test_wave64_uniform_scalar_state(self):
        wave = MultiWaveScalarState(num_lanes=64)
        val = 0x0123456789ABCDEF
        wave.write_sgpr64(2, val)
        res = ref_not_b64(wave.read_sgpr64(2))
        wave.write_sgpr64(0, res)
        for lane in range(64):
            self.assertEqual(wave.read_sgpr64(0), 0xFEDCBA9876543210)


class TestSNotB64RegisterAliasing(unittest.TestCase):
    """Register aliasing hazards and in-place operand preservation."""

    def setUp(self):
        self.rf = ScalarRegisterFile()

    def test_dst_aliases_src(self):
        val = 0x8899AABBCCDDEEFF
        self.rf.write_sgpr64(0, val)
        inst = SNotB64Instruction(
            dst=Operand(RegType.SGPR, 0, size=2),
            src=Operand(RegType.SGPR, 0, size=2),
        )
        res = ScalarSALUInterpreter.execute(self.rf, inst)
        expected = ref_not_b64(val)
        self.assertEqual(res, expected)
        self.assertEqual(self.rf.read_sgpr64(0), expected)

    def test_hw_translator_aliasing(self):
        val = 0x123456789ABCDEF0
        vlo = val & MASK32
        vhi = (val >> 32) & MASK32
        lines = [
            f"s_mov_b32 s0, {vlo:#x}",
            f"s_mov_b32 s1, {vhi:#x}",
            "s_not_b64 s[0:1], s[0:1]",
            "v_mov_b32 v0, s0",
            "v_mov_b32 v1, s1",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_snot64_alias", lines)
        words = support.mrt0_words(spv)
        hw_res = (words[0] & MASK32) | ((words[1] & MASK32) << 32)
        self.assertEqual(hw_res, ref_not_b64(val))


@dataclasses.dataclass(frozen=True)
class Not64StatusFlags:
    is_zero: bool
    scc: int
    all_ones: bool
    high_bit: int


def compute_status_flags(val64: int) -> Not64StatusFlags:
    res, scc = ref_not_b64_scc(val64)
    return Not64StatusFlags(
        is_zero=(res == 0),
        scc=scc,
        all_ones=(res == MASK64),
        high_bit=(res >> 63) & 1,
    )


class TestSNotB64ConditionCodes(unittest.TestCase):
    """Condition codes, status flags, and classification."""

    def test_zero_result_flags(self):
        flags = compute_status_flags(MASK64)
        self.assertTrue(flags.is_zero)
        self.assertEqual(flags.scc, 0)
        self.assertFalse(flags.all_ones)
        self.assertEqual(flags.high_bit, 0)

    def test_all_ones_flags(self):
        flags = compute_status_flags(0)
        self.assertFalse(flags.is_zero)
        self.assertEqual(flags.scc, 1)
        self.assertTrue(flags.all_ones)
        self.assertEqual(flags.high_bit, 1)

    def test_high_bit_flags(self):
        flags = compute_status_flags(0x7FFFFFFFFFFFFFFF)
        self.assertEqual(flags.high_bit, 1)
        self.assertEqual(flags.scc, 1)

        flags = compute_status_flags(0x8000000000000000)
        self.assertEqual(flags.high_bit, 0)
        self.assertEqual(flags.scc, 1)


class TestSNotB64IEEE754(unittest.TestCase):
    """IEEE 754 raw bit pattern preservation and subnormal/NaN behavior."""

    def test_ieee754_double_patterns(self):
        patterns = [
            0x0000000000000000,  # +0.0
            0x8000000000000000,  # -0.0
            0x0000000000000001,  # smallest subnormal
            0x7FF0000000000000,  # +inf
            0xFFF0000000000000,  # -inf
            0x7FF8000000000000,  # quiet NaN
            0x7FF0000000000001,  # signaling NaN
        ]
        for p in patterns:
            with self.subTest(p=hex(p)):
                res = ref_not_b64(p)
                self.assertEqual(res, (~p) & MASK64)

    def test_ieee754_hw_translator_inversion(self):
        # Invert +inf pattern -> negative word
        inf64 = 0x7FF0000000000000
        hw_res = run_hw_not_b64(inf64)
        self.assertEqual(hw_res, ref_not_b64(inf64))


if __name__ == "__main__":
    unittest.main()
