"""Semantic tests for RDNA2 v_perm_b32 opcode with hardware translator execution.

RDNA2 ISA definition:
    v_perm_b32 dst, src0, src1, src2
Source container is 64-bit packed {src0, src1}, where src1 provides low bytes 0..3
and src0 provides high bytes 4..7.
For each byte i (0..3) of dst:
    sel = (src2 >> (8 * i)) & 0xFF
    0..7: select byte sel from {src0, src1}
    8..11: sign extend 16-bit halfword (sel - 8): bit 15, 31, 47, or 63 replicated across the byte
    12: 0x00
    >= 13: 0xFF
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


def ref_perm_b32(s0: int, s1: int, sel: int) -> int:
    """Bit-accurate reference oracle for RDNA2 v_perm_b32."""
    s0 &= MASK32
    s1 &= MASK32
    sel &= MASK32
    packed = (s0 << 32) | s1
    out = 0
    for i in range(4):
        sb = (sel >> (8 * i)) & 0xFF
        if sb < 8:
            byte_val = (packed >> (8 * sb)) & 0xFF
        elif sb < 12:
            sign_bit = (packed >> (16 * (sb - 8) + 15)) & 1
            byte_val = 0xFF if sign_bit else 0x00
        elif sb == 12:
            byte_val = 0x00
        else:
            byte_val = 0xFF
        out |= byte_val << (8 * i)
    return out & MASK32


def run_hw_perm_b32(src0: int, src1: int, sel: int, inline: bool = False) -> int:
    """Execute v_perm_b32 through the rdna2spirv translator and spirv_mini_interp."""
    lines = [
        f"v_mov_b32 v1, {src0 & MASK32:#x}",
        f"v_mov_b32 v2, {src1 & MASK32:#x}",
    ]
    if inline:
        lines.append(f"v_perm_b32 v0, v1, v2, {sel & MASK32:#x}")
    else:
        lines.append(f"v_mov_b32 v3, {sel & MASK32:#x}")
        lines.append("v_perm_b32 v0, v1, v2, v3")
    lines.extend([
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ])
    spv = support.translate_lines("ps_perm_hw", lines)
    return support.mrt0_words(spv)[0]


class TestVPermB32Reference(unittest.TestCase):
    """Reference mathematical oracle and translator hardware-level verification."""

    def test_reference_byte_selection(self):
        s0 = 0x07060504
        s1 = 0x03020100
        # Identity permute: bytes 0, 1, 2, 3
        self.assertEqual(ref_perm_b32(s0, s1, 0x03020100), 0x03020100)
        # Reverse bytes in src1
        self.assertEqual(ref_perm_b32(s0, s1, 0x00010203), 0x00010203)
        # Select all from high word src0
        self.assertEqual(ref_perm_b32(s0, s1, 0x07060504), 0x07060504)
        # Interleave src1 and src0
        self.assertEqual(ref_perm_b32(s0, s1, 0x06020501), 0x06020501)

    def test_reference_constants_and_sign_extension(self):
        s0 = 0x80007FFF
        s1 = 0x00018000
        # 12 -> 0x00, 13 -> 0xFF
        self.assertEqual(ref_perm_b32(s0, s1, 0x0D0C0D0C), 0xFF00FF00)
        # Sign extension:
        # halfword 0 (s1[15:0] = 0x8000, bit 15 is 1 -> 0xFF)
        # halfword 1 (s1[31:16] = 0x0001, bit 31 is 0 -> 0x00)
        # halfword 2 (s0[15:0] = 0x7FFF, bit 15 is 0 -> 0x00)
        # halfword 3 (s0[31:16] = 0x8000, bit 31 is 1 -> 0xFF)
        sel = 0x0B0A0908
        self.assertEqual(ref_perm_b32(s0, s1, sel), 0xFF0000FF)

    def test_translator_hw_byte_selection(self):
        s0 = 0xAABBCCDD
        s1 = 0x11223344
        selectors = [
            0x03020100,  # identity src1
            0x07060504,  # identity src0
            0x00010203,  # byte reversal
            0x04000501,  # mixed words
            0x0C0C0C0C,  # zero fill
            0x0D0D0D0D,  # 0xFF fill
            0x0B0A0908,  # sign extension
        ]
        for sel in selectors:
            with self.subTest(sel=hex(sel)):
                hw_res = run_hw_perm_b32(s0, s1, sel)
                ref_res = ref_perm_b32(s0, s1, sel)
                self.assertEqual(hw_res, ref_res)


if __name__ == "__main__":
    unittest.main()
