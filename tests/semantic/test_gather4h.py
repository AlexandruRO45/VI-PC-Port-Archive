"""The horizontal gather reads one 2D channel through the sampler at the base mip."""
import unittest

from support import rdna2spirv, translate_lines


def translate(sampler, args="v[4:7], v[0:1]", mods="dmask:0x1 dim:SQ_RSRC_IMG_2D"):
    lines = [f"s_mov_b32 s{16 + i}, {word}" for i, word in enumerate(sampler) if word is not None]
    return translate_lines("ps_gather4h", lines + [
        f"image_gather4h {args}, s[8:15], s[16:19] {mods}", "exp mrt0 v4, v5, v6, v7 done vm", "s_endpgm"])


class Gather4H(unittest.TestCase):
    def test_all_channels_and_numeric_classes(self):
        for channel in range(4):
            with self.subTest(channel=channel):
                module = translate((channel | (channel << 3), 0, 0x04000000, 0), mods=f"dmask:{1 << channel:#x} dim:SQ_RSRC_IMG_2D")
                self.assertNotIn("OpImageGather", module)
                for ty, scalar in (("%v4float", "%float"), ("%v4uint_st", "%uint"), ("%v4int_st", "%int")):
                    fetches = [line.split()[0] for line in module.splitlines() if f"= OpImageFetch {ty} " in line]
                    self.assertEqual(len(fetches), 4)
                    for texel in fetches:
                        self.assertRegex(module, rf"OpCompositeExtract {scalar} {texel} {channel}\b")

    def test_mip_filter_none_ignores_lod_bounds(self):
        module = translate((0x92, 0x00200100, 0, 0), "v[4:5], v0", "dmask:0x8 dim:SQ_RSRC_IMG_2D a16 d16")
        self.assertIn("OpImageFetch", module)

    def test_unhandled_forms_do_not_translate_as_base_mip(self):
        for sampler in [(0x92, 0xc00000, 1 << 26, 0), (0x8092, 0, 0, 0), (None, 0, 0, 0),
                        (0x92, None, None, 0), (6, 0, 0, 0)]:
            with self.subTest(sampler=sampler), self.assertRaises(rdna2spirv.Unsupported):
                translate(sampler)
        for mods in ("dmask:0x3 dim:SQ_RSRC_IMG_2D", "dmask:0x1 dim:SQ_RSRC_IMG_2D_ARRAY",
                     "dmask:0x1 dim:SQ_RSRC_IMG_2D tfe", "dmask:0x1 dim:SQ_RSRC_IMG_2D unorm"):
            with self.subTest(mods=mods), self.assertRaises(rdna2spirv.Unsupported):
                translate((0x92, 0, 0, 0), mods=mods)


if __name__ == "__main__":
    unittest.main()
