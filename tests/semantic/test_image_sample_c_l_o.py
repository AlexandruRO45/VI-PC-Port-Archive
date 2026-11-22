"""Structural and assembler tests for RDNA2 image_sample_c_l_o.

Verifies:
  - OpImageSampleDrefExplicitLod %float with Lod|Offset {lod} {off} (Lod=0x2 before Offset=0x10)
  - Signed 6-bit texel offset extraction (x=[5:0], y=[13:8]) into %v2int
  - d16 narrowing of the scalar depth compare result
  - a16 keeping offset and z-compare reference as full 32-bit dwords while unpacking XY and LOD
  - Unsupported rejections for non-2D dims, non-scalar dmask, wrong address length, and glc/tfe
  - Every emitted SPIR-V module assembles cleanly with tools/spirv_asm.py

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.
"""
import os
import re
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
from support import translate_lines  # noqa: E402

STUB_FSSM = bytearray(0x148)
struct.pack_into("<4I", STUB_FSSM, 0x138, 0x8088, 64, 1, 1)
FSSM_BYTES = bytes(STUB_FSSM)


def translate_and_assemble(lines) -> str:
    spv = translate_lines("cs_img_sample_c_l_o", list(lines) + ["s_endpgm"], fssm=FSSM_BYTES)
    binary = spirv_asm.assemble(spv)
    magic = struct.unpack_from("<I", binary, 0)[0]
    assert magic == 0x07230203, f"bad SPIR-V magic {magic:#x}"
    return spv


class ImageSampleCLOtests(unittest.TestCase):
    def test_2d_lod_offset_and_sextract(self):
        spv = translate_and_assemble([
            "image_sample_c_l_o v0, v[0:4], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D",
        ])
        self.assertRegex(
            spv,
            r"OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Lod\|Offset %\w+ %\w+",
        )
        # Signed 6-bit extraction for x ([5:0]) and y ([13:8])
        self.assertIn("OpBitFieldSExtract %uint", spv)
        self.assertIn("OpCompositeConstruct %v2int", spv)

    def test_d16_and_a16(self):
        spv_d16 = translate_and_assemble([
            "image_sample_c_l_o v0, v[0:4], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16",
        ])
        self.assertRegex(
            spv_d16,
            r"OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Lod\|Offset %\w+ %\w+",
        )
        # d16 narrows the scalar compare result with integer f16_bits conversion
        self.assertIn("OpBitwiseAnd %uint", spv_d16)

        spv_a16 = translate_and_assemble([
            "image_sample_c_l_o v0, v[0:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D a16",
        ])
        self.assertRegex(
            spv_a16,
            r"OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Lod\|Offset %\w+ %\w+",
        )
        # Exactly 2 UnpackHalf2x16 calls (for XY and LOD), NOT for offset or ref
        self.assertEqual(len(re.findall(r"OpExtInst %v2float %glsl UnpackHalf2x16", spv_a16)), 2)

    def test_unsupported_forms(self):
        bad_lines = [
            "image_sample_c_l_o v0, v[0:2], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_1D",
            "image_sample_c_l_o v0, v[0:5], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY",
            "image_sample_c_l_o v0, v[0:5], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_3D",
            "image_sample_c_l_o v[0:1], v[0:4], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D",
            "image_sample_c_l_o v0, v[0:3], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D",
            "image_sample_c_l_o v0, v[0:4], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D glc",
            "image_sample_c_l_o v0, v[0:4], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D tfe",
        ]
        for line in bad_lines:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("cs_img_sample_c_l_o_bad", [line, "s_endpgm"], fssm=FSSM_BYTES)


if __name__ == "__main__":
    unittest.main()
