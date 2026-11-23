"""image_sample_l takes its LOD from the address word after the coordinates.

GFX10 MIMG address order is offset, bias, z-compare, derivatives, coordinates, LOD, clamp, so an
`image_sample_l` needs one address word more than its coordinate count (a cube's face is the third
coordinate word, the LOD the fourth). A listing whose address range ends at the coordinates used to
be translated as `Lod %f0`: the sample silently read mip 0 and nothing named the listing. It is now
`Unsupported`, like the other short address forms; the hand-written `sample_l_lod.asm` checks that
the LOD word really becomes the `Lod` operand, and a listing without it is refused.

    python -I tests/semantic/test_image_sample_l_lod.py
"""
from pathlib import Path
import re
import unittest

import support
import rdna2spirv


EPILOGUE = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]


def translate(instruction):
    return support.translate_lines("ps_sample_l_lod", [instruction, *EPILOGUE])


class ImageSampleLod(unittest.TestCase):
    def test_the_lod_address_word_is_the_sample_lod(self):
        fixture = Path(__file__).with_name("fixtures") / "sample_l_lod.asm"
        asm = support.translate_lines("ps_sample_l_lod", fixture.read_text(encoding="utf-8").splitlines())
        load = re.search(r"(%\w+) = OpLoad %uint %reg_v6\b", asm)
        self.assertIsNotNone(load, "the fixture's LOD register v6 was not read")
        lod = re.search(r"(%\w+) = OpBitcast %float " + re.escape(load.group(1)) + r"\b", asm)
        self.assertIsNotNone(lod, "v6 was not turned into the Lod operand")
        self.assertIn(f"OpImageSampleExplicitLod %v4float %", asm)
        self.assertIn(f" Lod {lod.group(1)}", asm)
        self.assertNotIn("Lod %f0", asm)

    def test_a_listing_without_the_lod_word_is_refused(self):
        # 2D takes u, v and then the LOD: v[4:5] ends at the coordinates.
        with self.assertRaises(rdna2spirv.Unsupported):
            translate("image_sample_l v[0:3], v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D")

    def test_a_cube_at_three_address_words_is_refused(self):
        # A cube's third coordinate is its face, so v[4:6] is the address without a LOD word.
        with self.assertRaises(rdna2spirv.Unsupported):
            translate("image_sample_l v[0:3], v[4:6], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_CUBE")

    def test_every_dimension_reads_the_word_after_its_coordinates(self):
        for dim, coords in (("1D", "v[4:5]"), ("2D", "v[4:6]"), ("2D_ARRAY", "v[4:7]"), ("3D", "v[4:7]"),
                            ("CUBE", "v[4:7]")):
            with self.subTest(dim=dim):
                asm = translate(f"image_sample_l v[0:3], {coords}, s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_{dim}")
                self.assertIn("OpImageSampleExplicitLod", asm)
                self.assertNotIn("Lod %f0", asm)


if __name__ == "__main__":
    unittest.main()
