"""Regression tests for 32-bit image_atomic_* and image_atomic_cmpswap vdata width validation."""
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support  # noqa: E402

CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def translate(line):
    return support.translate_lines(
        "cs_img_atomic_vdata",
        [
            "v_mov_b32 v2, 1",
            "v_mov_b32 v3, 2",
            "v_mov_b32 v4, 0",
            "v_mov_b32 v5, 0",
            "v_mov_b32 v6, 0",
            line,
            "s_endpgm",
        ],
        fssm=CS_FSSM,
    )


class ImageAtomicOperandTests(unittest.TestCase):
    def test_valid_vdata_widths_translate(self):
        for dim, coords in (("SQ_RSRC_IMG_2D", "v[4:5]"), ("SQ_RSRC_IMG_3D", "v[4:6]")):
            with self.subTest(dim=dim):
                spv_add = translate(f"image_atomic_add v2, {coords}, s[8:15] dmask:0x1 dim:{dim} unorm glc")
                self.assertIn("OpAtomicIAdd %uint", spv_add)
                spv_cmp = translate(f"image_atomic_cmpswap v[2:3], {coords}, s[8:15] dmask:0x3 dim:{dim} unorm glc")
                self.assertIn("OpAtomicCompareExchange %uint", spv_cmp)

    def test_wrong_vdata_width_is_rejected(self):
        for dim, coords in (("SQ_RSRC_IMG_2D", "v[4:5]"), ("SQ_RSRC_IMG_3D", "v[4:6]")):
            for bad in (
                f"image_atomic_umin v[2:3], {coords}, s[8:15] dmask:0x3 dim:{dim} unorm",
                f"image_atomic_sub v[2:3], {coords}, s[8:15] dmask:0x3 dim:{dim} unorm",
                f"image_atomic_add v[2:3], {coords}, s[8:15] dmask:0x3 dim:{dim} unorm",
                f"image_atomic_cmpswap v2, {coords}, s[8:15] dmask:0x1 dim:{dim} unorm",
            ):
                with self.subTest(dim=dim, line=bad):
                    with self.assertRaises(support.rdna2spirv.Unsupported):
                        translate(bad)


if __name__ == "__main__":
    unittest.main()
