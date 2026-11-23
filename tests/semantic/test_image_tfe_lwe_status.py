"""Semantic regression tests for RDNA2 image instructions with tfe/lwe status VGPR (SHADER-1266).

When tfe (texel-fail-enable) or lwe (level-warning-enable) is specified on image instructions,
an extra 32-bit status VGPR is appended to the destination register vector at vdata[data_count].
In SPIR-V translation, the status register is initialized to 0 (%c0) indicating residency / success.
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.dirname(os.path.abspath(__file__))
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import support  # noqa: E402


class ImageTfeLweStatusTest(unittest.TestCase):
    def assert_status_vgpr(self, name: str, lines: list[str], expected_status_reg: str) -> None:
        spv = support.translate_lines(name, lines)
        target = f"OpStore {expected_status_reg} %c0"
        found = any(target in line for line in spv.splitlines())
        self.assertTrue(found, f"Expected '{target}' in translated SPIR-V for {name}")

    def test_image_sample_2d_tfe(self):
        """image_sample 2D with 4 components + tfe writes status to v4."""
        lines = [
            "image_sample v[0:4], v[1:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D tfe",
            "s_endpgm",
        ]
        self.assert_status_vgpr("ps_sample_2d_tfe", lines, "%reg_v4")

    def test_image_sample_2d_d16_tfe(self):
        """image_sample 2D with d16 packs 4 components in 2 VGPRs, status in v2."""
        lines = [
            "image_sample v[0:2], v[1:2], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_2D d16 tfe",
            "s_endpgm",
        ]
        self.assert_status_vgpr("ps_sample_2d_d16_tfe", lines, "%reg_v2")

    def test_image_sample_3d_lwe(self):
        """image_sample 3D with lwe writes status to v4."""
        lines = [
            "image_sample v[0:4], v[1:3], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_3D lwe",
            "s_endpgm",
        ]
        self.assert_status_vgpr("ps_sample_3d_lwe", lines, "%reg_v4")

    def test_image_sample_c_tfe(self):
        """image_sample_c_l scalar depth compare writes result to v0 and status to v1."""
        lines = [
            "image_sample_c_l v[0:1], v[1:4], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D tfe",
            "s_endpgm",
        ]
        self.assert_status_vgpr("ps_sample_c_tfe", lines, "%reg_v1")

    def test_image_load_2d_tfe(self):
        """image_load 2D with tfe writes status to v4."""
        lines = [
            "image_load v[0:4], v[1:2], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D tfe",
            "s_endpgm",
        ]
        self.assert_status_vgpr("ps_load_2d_tfe", lines, "%reg_v4")

    def test_image_gather4_lz_tfe(self):
        """image_gather4_lz 2D writes 4 taps and status to v4."""
        lines = [
            "image_gather4_lz v[0:4], v[1:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D tfe",
            "s_endpgm",
        ]
        self.assert_status_vgpr("ps_gather4_lz_tfe", lines, "%reg_v4")

    def test_image_gather4_lz_d16_tfe(self):
        """image_gather4_lz 2D with d16 packs 4 taps into 2 VGPRs and writes status to v2."""
        lines = [
            "image_gather4_lz v[0:2], v[1:2], s[16:23], s[24:27] dmask:0x1 dim:SQ_RSRC_IMG_2D d16 tfe",
            "s_endpgm",
        ]
        self.assert_status_vgpr("ps_gather4_lz_d16_tfe", lines, "%reg_v2")

    def test_image_load_mip_2d_tfe(self):
        """image_load_mip 2D non-a16 with tfe writes status to v4."""
        lines = [
            "image_load_mip v[0:4], v[1:3], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_2D tfe",
            "s_endpgm",
        ]
        self.assert_status_vgpr("ps_load_mip_2d_tfe", lines, "%reg_v4")

    def test_image_load_mip_3d_tfe(self):
        """image_load_mip 3D non-a16 with tfe writes status to v4."""
        lines = [
            "image_load_mip v[0:4], v[1:4], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_3D tfe",
            "s_endpgm",
        ]
        self.assert_status_vgpr("ps_load_mip_3d_tfe", lines, "%reg_v4")

    def test_unsupported_status_forms(self):
        """Verify guarded unsupported status combinations continue to raise Unsupported."""
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_samp_d_3d_tfe", [
                "image_sample_d v[0:4], v[1:7], s[16:23], s[24:27] dmask:0xf dim:SQ_RSRC_IMG_3D tfe",
                "s_endpgm",
            ])
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_load_mip_3d_d16_tfe", [
                "image_load_mip v[0:2], v[1:4], s[16:23] dmask:0xf dim:SQ_RSRC_IMG_3D d16 tfe",
                "s_endpgm",
            ])


if __name__ == "__main__":
    unittest.main()
