"""Semantic tests for RDNA2 v_bfe_u32 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_bfe_u32 dst, src0, src1, src2 (VOP3)
Bitfield extract unsigned 32-bit:
    src0 = base 32-bit unsigned integer
    src1 = bit offset (low 5 bits: offset = src1 & 31)
    src2 = bitfield width (low 5 bits: width = src2 & 31)

Semantics:
    if width == 0:
        dst = 0
    else:
        eff_width = min(width, 32 - offset)
        dst = (src0 >> offset) & ((1 << eff_width) - 1)
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


def ref_bfe_u32(src0: int, src1: int, src2: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_bfe_u32."""
    base = src0 & MASK32
    offset = src1 & 31
    width = src2 & 31
    if width == 0:
        return 0
    eff_width = min(width, 32 - offset)
    return ((base >> offset) & ((1 << eff_width) - 1)) & MASK32


def run_hw_bfe_u32(src0: int, src1: int, src2: int) -> int:
    """Execute v_bfe_u32 through rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
        f"v_mov_b32 v3, {src2 & MASK32:#x}",
        "v_bfe_u32 v0, v1, v2, v3",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines("ps_bfe_u32_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVBfeU32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_zero_width(self):
        for off in range(32):
            self.assertEqual(ref_bfe_u32(0xDEADBEEF, off, 0), 0)
            self.assertEqual(ref_bfe_u32(0xFFFFFFFF, off, 32), 0)  # 32 & 31 == 0

    def test_reference_single_bit_extractions(self):
        val = 0b10101010111100001100110001010101
        for off in range(32):
            expected = (val >> off) & 1
            self.assertEqual(ref_bfe_u32(val, off, 1), expected)

    def test_reference_multi_bit_fields(self):
        val = 0x12345678
        self.assertEqual(ref_bfe_u32(val, 4, 4), 0x7)
        self.assertEqual(ref_bfe_u32(val, 8, 4), 0x6)
        self.assertEqual(ref_bfe_u32(val, 16, 8), 0x34)
        self.assertEqual(ref_bfe_u32(val, 0, 16), 0x5678)
        self.assertEqual(ref_bfe_u32(val, 16, 16), 0x1234)

    def test_reference_offset_masking(self):
        val = 0x87654321
        self.assertEqual(ref_bfe_u32(val, 33, 4), ref_bfe_u32(val, 1, 4))
        self.assertEqual(ref_bfe_u32(val, 64, 8), ref_bfe_u32(val, 0, 8))

    def test_reference_width_masking(self):
        val = 0xFFFFFFFF
        self.assertEqual(ref_bfe_u32(val, 0, 33), 1)
        self.assertEqual(ref_bfe_u32(val, 0, 64), 0)

    def test_reference_past_bit31_clamp(self):
        val = 0xF0000000
        self.assertEqual(ref_bfe_u32(val, 28, 8), 0xF)
        self.assertEqual(ref_bfe_u32(val, 31, 10), 1)

    def test_translator_hw_bfe_u32(self):
        if support is None or getattr(support, "translate_lines", None) is None:
            self.skipTest("support/rdna2spirv requires Python 3.10+ (dict | None union syntax)")
        cases = [
            (0x12345678, 4, 4),
            (0xDEADBEEF, 8, 8),
            (0xFFFFFFFF, 0, 16),
            (0xF0000000, 28, 8),
            (0x12345678, 0, 0),
        ]
        for s0, s1, s2 in cases:
            with self.subTest(s0=hex(s0), s1=s1, s2=s2):
                res = run_hw_bfe_u32(s0, s1, s2)
                self.assertEqual(res, ref_bfe_u32(s0, s1, s2))


if __name__ == '__main__':
    unittest.main()
