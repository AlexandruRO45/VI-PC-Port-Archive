"""Semantic tests for RDNA2 v_lshrrev_b64 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_lshrrev_b64 dst[0:1], src0, src1[0:1]
    dst = (src1 & 0xFFFFFFFFFFFFFFFF) >> (src0 & 63)
where src0 is the shift count (masked to 6 bits, range 0..63),
src1 is the 64-bit source operand in a register pair,
and dst is the 64-bit logical right-shifted result in a register pair with zero-fill.
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


def ref_lshrrev_b64(shift_cnt: int, val64: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_lshrrev_b64."""
    sh = shift_cnt & 63
    return ((val64 & MASK64) >> sh) & MASK64


def run_hw_lshrrev_b64(shift_cnt: int, val64: int, inline_shift: bool = False) -> int:
    """Execute v_lshrrev_b64 through the rdna2spirv translator and spirv_mini_interp."""
    val_lo = val64 & MASK32
    val_hi = (val64 >> 32) & MASK32
    lines = [
        f"v_mov_b32 v2, {val_lo:#x}",
        f"v_mov_b32 v3, {val_hi:#x}",
    ]
    if inline_shift:
        lines.append(f"v_lshrrev_b64 v[0:1], {shift_cnt & 63}, v[2:3]")
    else:
        lines.append(f"v_mov_b32 v4, {shift_cnt & MASK32:#x}")
        lines.append("v_lshrrev_b64 v[0:1], v4, v[2:3]")
    lines.extend([
        "exp mrt0 v0, v1, v0, v1 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines("ps_lshr64_hw", lines)
    words = support.mrt0_words(spv)
    return (words[0] & MASK32) | ((words[1] & MASK32) << 32)


class TestVLshrrevB64Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_shifts(self):
        val = 0x8899AABBCCDDEEFF
        self.assertEqual(ref_lshrrev_b64(0, val), val)
        self.assertEqual(ref_lshrrev_b64(8, val), 0x008899AABBCCDDEE)
        self.assertEqual(ref_lshrrev_b64(32, val), 0x000000008899AABB)
        self.assertEqual(ref_lshrrev_b64(63, val), 1)

    def test_reference_shift_masking_6_bits(self):
        val = 0xFEDCBA9876543210
        # 64 & 63 == 0 -> shift 0
        self.assertEqual(ref_lshrrev_b64(64, val), val)
        # 65 & 63 == 1 -> shift 1
        self.assertEqual(ref_lshrrev_b64(65, val), val >> 1)
        # 0xFFFFFFFF & 63 == 63 -> shift 63
        self.assertEqual(ref_lshrrev_b64(0xFFFFFFFF, val), val >> 63)

    def test_translator_hw_shifts(self):
        val = 0xABCDEF0112345678
        shifts = [0, 1, 8, 16, 31, 32, 33, 48, 63, 64, 65, 0x100]
        for sh in shifts:
            with self.subTest(sh=sh):
                hw_res = run_hw_lshrrev_b64(sh, val)
                ref_res = ref_lshrrev_b64(sh, val)
                self.assertEqual(hw_res, ref_res)

    def test_translator_hw_inline_shift(self):
        val = 0xFFFFFFFFFFFFFFFF
        for sh in (0, 1, 16, 32, 63):
            with self.subTest(sh=sh):
                hw_res = run_hw_lshrrev_b64(sh, val, inline_shift=True)
                ref_res = ref_lshrrev_b64(sh, val)
                self.assertEqual(hw_res, ref_res)


if __name__ == "__main__":
    unittest.main()
