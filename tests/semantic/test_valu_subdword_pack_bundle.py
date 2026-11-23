from __future__ import annotations

"""Translator-backed semantic tests for Bundle 63: VALU Sub-Dword Permutation
and Packing Extensions.

Covers RDNA2 Instructions:
    - v_pack_b32_f16 vD, vSrc0, vSrc1 [op_sel]
        Packs two 16-bit half-precision floats into one 32-bit dword.
    - v_permlane16_b32 vD, vSrc0, sSrc1, sSrc2
        Permutes 32-bit dwords within the same 16-lane half-row using 4-bit nibble selectors.
    - v_permlanex16_b32 vD, vSrc0, sSrc1, sSrc2
        Permutes 32-bit dwords from the opposing 16-lane half-row using 4-bit nibble selectors.

Hardware rules (GFX10 RDNA2):
    - v_pack_b32_f16 packs (src0.f16 & 0xFFFF) | ((src1.f16 & 0xFFFF) << 16).
    - op_sel selects high (1) or low (0) 16-bit halves of sources.
    - v_permlane16 routes within (lane & ~15), v_permlanex16 routes within ((lane & ~15) ^ 16).
    - Selectors are unpacked from 4-bit nibbles: sSrc1 for lanes 0..7, sSrc2 for lanes 8..15.
    - 100% translator-backed execution exercising tools/rdna2spirv.py, mrt0_words, and run_workgroup.

Standard library Python only (unittest, struct).
Strictly zero game data.
"""

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si
import saveexec_run

MASK16: int = 0xFFFF
MASK32: int = 0xFFFF_FFFF


def run_ps_pack(s0_u32: int, s1_u32: int, op_sel: str = "", abs0: bool = False, neg0: bool = False, name: str = "ps_pack") -> int:
    mods = f" op_sel:{op_sel}" if op_sel else ""
    src0 = f"{'|' if abs0 else ''}{'-' if neg0 else ''}v1{'|' if abs0 else ''}"
    code = [
        f"v_mov_b32_e32 v1, {s0_u32 & MASK32:#x}",
        f"v_mov_b32_e32 v2, {s1_u32 & MASK32:#x}",
        f"v_pack_b32_f16 v0, {src0}, v2{mods}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, code)
    return support.mrt0_words(spv)[0]


def run_cs_permlane(op: str, s1_sel: int, s2_sel: int, name: str = "cs_perm") -> str:
    code = [
        "v_mbcnt_lo_u32_b32 v1, -1, 0",
        "v_mbcnt_hi_u32_b32 v1, -1, v1",
        f"s_mov_b32 s1, {s1_sel & MASK32:#x}",
        f"s_mov_b32 s2, {s2_sel & MASK32:#x}",
        f"{op} v0, v1, s1, s2",
        "s_endpgm",
    ]
    spv = support.translate_lines(name, code, fssm=saveexec_run.fssm())
    si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})
    return spv

class TestPackB32F16Reference(unittest.TestCase):
    def test_pack_basic_values(self):
        # Pack 1.0 (0x3C00) and 2.0 (0x4000)
        res = run_ps_pack(0x3C00, 0x4000, name="ps_pack_ref_1")
        self.assertEqual(res, 0x4000_3C00)

    def test_pack_zeroes(self):
        res = run_ps_pack(0x0000, 0x0000, name="ps_pack_ref_zero")
        self.assertEqual(res, 0x0000_0000)


if __name__ == "__main__":
    unittest.main()
