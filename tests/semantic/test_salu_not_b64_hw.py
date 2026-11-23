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


if __name__ == "__main__":
    unittest.main()
