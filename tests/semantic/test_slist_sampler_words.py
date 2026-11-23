"""Tests that VCCTuple / slist sampler descriptors (S#) work in gather4 and texel-offset-fallback paths.

Covers `tools/rdna2spirv.py` `gather_addr_modes`, `gather_border_one`, `clamp_gather_lod`,
`gather_depth_compare_prep`, and `texel_offset_fold` when the sampler operand is a 4-register list
such as `[s104,s105,vcc_lo,vcc_hi]` or `[s20,s21,s22,s23]`. Standard library only; no GPU or game files.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from support import rdna2spirv, translate_lines  # noqa: E402


def _export_v4_lines(body):
    return [
        "s_load_dwordx8 s[12:19], s[0:1], 0x0",
        "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v0, 0x3f000000",
        "v_mov_b32_e32 v1, 0x3f000000",
        "v_mov_b32_e32 v2, 0",
        "v_mov_b32_e32 v3, 0",
        *body,
        "s_waitcnt vmcnt(0)",
        "exp mrt0 v4, v5, v6, v7 done",
        "s_endpgm",
    ]


class SlistSamplerWordTests(unittest.TestCase):
    def test_texel_offset_fallback_with_vcc_tuple_and_slist(self):
        old = rdna2spirv.TEXEL_OFFSET_FALLBACK
        rdna2spirv.TEXEL_OFFSET_FALLBACK = True
        try:
            spv_vcc = translate_lines("ps_slist_off_vcc", _export_v4_lines([
                "s_load_dwordx2 s[104:105], s[0:1], 0x20",
                "s_load_dwordx2 vcc, s[0:1], 0x28",
                "s_waitcnt lgkmcnt(0)",
                "v_mov_b32_e32 v0, 1",
                "v_mov_b32_e32 v1, 0x3f000000",
                "v_mov_b32_e32 v2, 0x3f000000",
                "image_sample_o v[4:7], v[0:2], s[12:19], [s104,s105,vcc_lo,vcc_hi] dmask:0xf dim:SQ_RSRC_IMG_2D",
            ]))
            self.assertIn("OpLoad %uint %vcc", spv_vcc)
            self.assertIn("OpImageSampleExplicitLod", spv_vcc)

            spv_lit = translate_lines("ps_slist_off_lit", _export_v4_lines([
                "s_mov_b32 s20, 0",
                "s_mov_b32 s21, 0",
                "s_mov_b32 s22, 0x08000000",
                "s_mov_b32 s23, 0",
                "v_mov_b32_e32 v0, 1",
                "v_mov_b32_e32 v1, 0x3f000000",
                "v_mov_b32_e32 v2, 0x3f000000",
                "image_sample_o v[4:7], v[0:2], s[12:19], [s20,s21,s22,s23] dmask:0xf dim:SQ_RSRC_IMG_2D",
            ]))
            self.assertIn("OpImageSampleExplicitLod", spv_lit)
        finally:
            rdna2spirv.TEXEL_OFFSET_FALLBACK = old

    def test_gather4_l_with_vcc_tuple_and_literal_slist(self):
        spv_vcc = translate_lines("ps_slist_g4l_vcc", _export_v4_lines([
            "s_load_dwordx2 s[104:105], s[0:1], 0x20",
            "s_load_dwordx2 vcc, s[0:1], 0x28",
            "s_waitcnt lgkmcnt(0)",
            "image_gather4_l v[4:7], v[0:2], s[12:19], [s104,s105,vcc_lo,vcc_hi] dmask:0x1 dim:SQ_RSRC_IMG_2D",
        ]))
        self.assertIn("OpLoad %uint %vcc_hi", spv_vcc)
        self.assertIn("OpImageFetch", spv_vcc)

        # Literal slist: CLAMP_X=6 (border), CLAMP_Y=2 (clamp), MIN/MAX_LOD=[0, 2.0], opaque white border.
        spv_lit = translate_lines("ps_slist_g4l_lit", _export_v4_lines([
            "s_mov_b32 s20, 0x16",
            "s_mov_b32 s21, 0x00200000",
            "s_mov_b32 s22, 0",
            "s_mov_b32 s23, 0x80000000",
            "image_gather4_l v[4:7], v[0:2], s[12:19], [s20,s21,s22,s23] dmask:0x1 dim:SQ_RSRC_IMG_2D",
        ]))
        self.assertIn("FClamp", spv_lit)
        self.assertIn("OpImageFetch", spv_lit)

    def test_gather4_c_l_with_vcc_tuple_and_slist(self):
        spv = translate_lines("ps_slist_g4cl_vcc", _export_v4_lines([
            "s_load_dwordx2 s[104:105], s[0:1], 0x20",
            "s_load_dwordx2 vcc, s[0:1], 0x28",
            "s_waitcnt lgkmcnt(0)",
            "image_gather4_c_l v[4:7], v[0:3], s[12:19], [s104,s105,vcc_lo,vcc_hi] dmask:0x1 dim:SQ_RSRC_IMG_2D",
        ]))
        self.assertIn("OpLoad %uint %vcc_hi", spv)
        self.assertIn("OpImageFetch", spv)

    def test_literal_slist_still_enforces_gather_restrictions(self):
        # FORCE_UNNORMALIZED (bit 15) on literal slist word 0 must still raise Unsupported.
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "FORCE_UNNORMALIZED"):
            translate_lines("ps_slist_g4l_unorm", _export_v4_lines([
                "s_mov_b32 s20, 0x8000",
                "s_mov_b32 s21, 0",
                "s_mov_b32 s22, 0",
                "s_mov_b32 s23, 0",
                "image_gather4_l v[4:7], v[0:2], s[12:19], [s20,s21,s22,s23] dmask:0x1 dim:SQ_RSRC_IMG_2D",
            ]))
        # BORDER_COLOR_TYPE 3 (table border) on literal slist word 3 must still raise Unsupported.
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "BORDER_COLOR_TYPE 3"):
            translate_lines("ps_slist_g4l_bct3", _export_v4_lines([
                "s_mov_b32 s20, 0x36",
                "s_mov_b32 s21, 0",
                "s_mov_b32 s22, 0",
                "s_mov_b32 s23, 0xc0000000",
                "image_gather4_l v[4:7], v[0:2], s[12:19], [s20,s21,s22,s23] dmask:0x1 dim:SQ_RSRC_IMG_2D",
            ]))


if __name__ == "__main__":
    unittest.main()
