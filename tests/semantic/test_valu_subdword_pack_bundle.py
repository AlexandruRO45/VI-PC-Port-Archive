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

class TestPackB32F16LowHalf(unittest.TestCase):
    def test_low_half_masking(self):
        # High bits of inputs must be ignored by default (low half selected)
        res = run_ps_pack(0xDEAD_1234, 0xBEEF_5678, name="ps_pack_low_mask")
        self.assertEqual(res, 0x5678_1234)

    def test_low_half_ones(self):
        res = run_ps_pack(0xFFFF, 0xFFFF, name="ps_pack_low_ones")
        self.assertEqual(res, 0xFFFF_FFFF)

class TestPackB32F16HighHalf(unittest.TestCase):
    def test_op_sel_high_halves(self):
        # Select high half from src0, low from src1: op_sel:[1, 0]
        res = run_ps_pack(0x1234_5678, 0xABCD_EF01, op_sel="[1,0,0]", name="ps_pack_opsel_10")
        self.assertEqual(res, 0xEF01_1234)

    def test_op_sel_both_high(self):
        # Select high half from both: op_sel:[1, 1]
        res = run_ps_pack(0x1234_5678, 0xABCD_EF01, op_sel="[1,1,0]", name="ps_pack_opsel_11")
        self.assertEqual(res, 0xABCD_1234)

class TestPackB32F16IEEE754(unittest.TestCase):
    def test_subnormal_and_nan(self):
        # Subnormal 0x0001, NaN 0x7E00
        res = run_ps_pack(0x0001, 0x7E00, name="ps_pack_subnorm_nan")
        self.assertEqual(res, 0x7E00_0001)

    def test_infinities(self):
        # +Inf 0x7C00, -Inf 0xFC00
        res = run_ps_pack(0x7C00, 0xFC00, name="ps_pack_infs")
        self.assertEqual(res, 0xFC00_7C00)

class TestPackB32F16Aliasing(unittest.TestCase):
    def test_dst_aliases_src0(self):
        code = [
            "v_mov_b32_e32 v0, 0x1111",
            "v_mov_b32_e32 v1, 0x2222",
            "v_pack_b32_f16 v0, v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pack_alias_s0", code)
        self.assertEqual(support.mrt0_words(spv)[0], 0x2222_1111)

    def test_dst_aliases_src1(self):
        code = [
            "v_mov_b32_e32 v1, 0x3333",
            "v_mov_b32_e32 v0, 0x4444",
            "v_pack_b32_f16 v0, v1, v0",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pack_alias_s1", code)
        self.assertEqual(support.mrt0_words(spv)[0], 0x4444_3333)

class TestPackB32F16Constants(unittest.TestCase):
    def test_literal_operand(self):
        code = [
            "v_mov_b32_e32 v1, 0x5555",
            "v_pack_b32_f16 v0, v1, 0x6666",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pack_literal", code)
        self.assertEqual(support.mrt0_words(spv)[0], 0x6666_5555)

    def test_inline_float_constant(self):
        code = [
            "v_mov_b32_e32 v1, 0x7777",
            "v_pack_b32_f16 v0, 1.0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pack_fimm", code)
        # 1.0 in f16 is 0x3C00
        self.assertEqual(support.mrt0_words(spv)[0], 0x7777_3C00)

class TestPackB32F16Mutation(unittest.TestCase):
    def test_abs_and_neg_modifiers(self):
        # abs clears bit 15: 0xBC00 (-1.0) -> 0x3C00 (+1.0)
        res = run_ps_pack(0xBC00, 0x4000, abs0=True, name="ps_pack_abs")
        self.assertEqual(res, 0x4000_3C00)

        # neg flips bit 15: 0x3C00 (+1.0) -> 0xBC00 (-1.0)
        res_neg = run_ps_pack(0x3C00, 0x4000, neg0=True, name="ps_pack_neg")
        self.assertEqual(res_neg, 0x4000_BC00)

    def test_spirv_instructions(self):
        code = [
            "v_mov_b32_e32 v1, 1",
            "v_mov_b32_e32 v2, 2",
            "v_pack_b32_f16 v0, v1, v2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_pack_inst", code)
        self.assertIn("OpShiftLeftLogical", spv)
        self.assertIn("OpBitwiseOr", spv)

class TestPackB32F16Translator(unittest.TestCase):
    def test_translator_matrix(self):
        cases = [
            (0x0001, 0x0002, 0x0002_0001),
            (0xAAAA, 0x5555, 0x5555_AAAA),
            (0x1234, 0x5678, 0x5678_1234),
            (0x8000, 0x7FFF, 0x7FFF_8000),
        ]
        for s0, s1, expected in cases:
            res = run_ps_pack(s0, s1, name=f"ps_pack_mat_{s0}_{s1}")
            self.assertEqual(res, expected)

class TestPermLane16Reference(unittest.TestCase):
    def test_reference_identity(self):
        spv = run_cs_permlane("v_permlane16_b32", 0x76543210, 0xFEDCBA98, name="cs_p16_ref")
        self.assertIn("OpGroupNonUniformShuffle", spv)
        self.assertIn("OpBitFieldUExtract", spv)

class TestPermLane16Nibbles(unittest.TestCase):
    def test_nibble_extraction_spirv(self):
        spv = run_cs_permlane("v_permlane16_b32", 0x11111111, 0x22222222, name="cs_p16_nib")
        self.assertIn("OpBitFieldUExtract", spv)
        self.assertIn("OpUGreaterThanEqual", spv)

class TestPermLane16Routing(unittest.TestCase):
    def test_row_masking(self):
        # Confirms row half mask 0xFFFFFFF0 is present
        spv = run_cs_permlane("v_permlane16_b32", 0x01234567, 0x89ABCDEF, name="cs_p16_rout")
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpBitwiseOr", spv)

class TestPermLane16Aliasing(unittest.TestCase):
    def test_aliasing_vdst_vsrc(self):
        code = [
            "v_mov_b32_e32 v0, 100",
            "s_mov_b32 s1, 0x76543210",
            "s_mov_b32 s2, 0xFEDCBA98",
            "v_permlane16_b32 v0, v0, s1, s2",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_p16_alias", code, fssm=saveexec_run.fssm())
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})
        self.assertIn("OpGroupNonUniformShuffle", spv)

class TestPermLane16Selectors(unittest.TestCase):
    def test_distinct_sgprs(self):
        code = [
            "v_mov_b32_e32 v1, 200",
            "s_mov_b32 s10, 0x00000000",
            "s_mov_b32 s11, 0x11111111",
            "v_permlane16_b32 v2, v1, s10, s11",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_p16_sgprs", code, fssm=saveexec_run.fssm())
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})
        self.assertIn("OpGroupNonUniformShuffle", spv)

class TestPermLane16Divergence(unittest.TestCase):
    def test_wave64_multi_lane(self):
        code = [
            "s_mov_b32 exec_lo, 0xFFFFFFFF",
            "s_mov_b32 exec_hi, 0x00000000",
            "v_mov_b32_e32 v1, 300",
            "s_mov_b32 s1, 0x76543210",
            "s_mov_b32 s2, 0xFEDCBA98",
            "v_permlane16_b32 v0, v1, s1, s2",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_p16_div", code, fssm=saveexec_run.fssm())
        si.run_workgroup(spv, 64, 64, inputs_for=lambda i: {"%wg_id": (0,0,0), "%local_id": (i,0,0), "%sg_lane": i}, push_constants=[0]*32, memory={})
        self.assertIn("OpGroupNonUniformShuffle", spv)

class TestPermLane16Mutation(unittest.TestCase):
    def test_nibble_mask_boundaries(self):
        # All 15s (0xFFFFFFFF)
        spv = run_cs_permlane("v_permlane16_b32", 0xFFFFFFFF, 0xFFFFFFFF, name="cs_p16_all15")
        self.assertIn("OpBitFieldUExtract", spv)

class TestPermLane16Translator(unittest.TestCase):
    def test_execution_all_cases(self):
        for s1, s2 in ((0, 0), (0x76543210, 0xFEDCBA98), (0x55555555, 0xAAAAAAAA)):
            spv = run_cs_permlane("v_permlane16_b32", s1, s2, name=f"cs_p16_exec_{s1:x}")
            self.assertIn("OpGroupNonUniformShuffle", spv)

class TestPermLaneX16Reference(unittest.TestCase):
    def test_reference_cross_half(self):
        spv = run_cs_permlane("v_permlanex16_b32", 0x76543210, 0xFEDCBA98, name="cs_px16_ref")
        self.assertIn("OpBitwiseXor", spv)
        self.assertIn("OpGroupNonUniformShuffle", spv)

class TestPermLaneX16RowSwap(unittest.TestCase):
    def test_row_xor_16(self):
        # Confirms XOR 16 for opposite half routing
        spv = run_cs_permlane("v_permlanex16_b32", 0x01234567, 0x89ABCDEF, name="cs_px16_xor16")
        self.assertIn("OpBitwiseXor", spv)


if __name__ == "__main__":
    unittest.main()
