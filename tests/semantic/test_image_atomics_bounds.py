"""Bounds of the 32-bit MIMG image atomics on r32ui storage images (2D and 3D).

An out-of-range texel pointer is undefined in Vulkan, while RDNA2 leaves memory untouched and, with `glc`, returns 0.
The translator therefore queries the image size, compares each coordinate unsigned (a negative one wraps to a large
value), and runs the texel pointer and the atomic only for lanes that are active and in range; with `glc` an active,
out-of-range lane gets 0 in vdata and an inactive lane keeps it. Structural checks on the emitted SPIR-V (no GPU, no
LLVM, no game files); the CFG of every case is validated with spirv_mini_interp.check_structure.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_image_atomics_32 import check_cfg, translate_cs  # noqa: E402

OPS = (
    ("image_atomic_add", "v2", "0x1"),
    ("image_atomic_umax", "v2", "0x1"),
    ("image_atomic_cmpswap", "v[2:3]", "0x3"),
    ("image_atomic_swap", "v2", "0x1"),
    ("image_atomic_inc", "v2", "0x1"),
    ("image_atomic_fmax", "v2", "0x1"),
    ("image_atomic_fcmpswap", "v[2:3]", "0x3"),
)
DIMS = (
    ("SQ_RSRC_IMG_2D", "v[4:5]", 2, "%v2int"),
    ("SQ_RSRC_IMG_3D", "v[4:6]", 3, "%v3int"),
)


def translate(op, vdata, dmask, dim, addr, glc):
    return translate_cs([
        "v_mov_b32_e32 v2, 0x3f800000",
        "v_mov_b32_e32 v3, 0x40000000",
        "v_mov_b32_e32 v4, 1",
        "v_mov_b32_e32 v5, 2",
        "v_mov_b32_e32 v6, 3",
        f"{op} {vdata}, {addr}, s[4:11] dmask:{dmask} dim:{dim} unorm{' glc' if glc else ''}",
        "s_waitcnt vmcnt(0)",
    ])


class ImageAtomicBoundsTests(unittest.TestCase):
    def test_texel_pointer_is_guarded_by_the_bounds_and_the_active_lanes(self):
        for op, vdata, dmask in OPS:
            for dim, addr, axes, size_ty in DIMS:
                for glc in (False, True):
                    with self.subTest(op=op, dim=dim, glc=glc):
                        spv = translate(op, vdata, dmask, dim, addr, glc)
                        check_cfg(spv)
                        self.assertIn("OpCapability ImageQuery", spv)
                        query = spv.find(f"OpImageQuerySize {size_ty} ")
                        pointer = spv.find("OpImageTexelPointer %_ptr_Image_uint")
                        self.assertGreaterEqual(query, 0)
                        self.assertGreater(pointer, query, "the size is queried before the texel pointer is made")
                        # One unsigned compare per axis: a negative coordinate is a large unsigned value.
                        guard_text = spv[query:pointer]
                        extents = re.findall(r"(%\w+) = OpCompositeExtract %int %\w+ \d", guard_text)
                        self.assertEqual(len(extents), axes)
                        compared = [e for e in extents if re.search(rf"OpULessThan %bool %\w+ {re.escape(e)}\b",
                                                                    guard_text)]
                        self.assertEqual(compared, extents, "every axis is compared with its extent, unsigned")
                        # The branch that leads to the texel pointer tests (active AND in range).
                        guard = re.findall(r"OpBranchConditional (%\w+) ", spv[query:pointer])[-1]
                        self.assertRegex(spv[query:pointer], rf"{re.escape(guard)} = OpLogicalAnd %bool ")

    def test_out_of_range_active_lane_returns_zero_only_with_glc(self):
        for op, vdata, dmask in OPS:
            for dim, addr, _axes, _size_ty in DIMS:
                for glc in (False, True):
                    with self.subTest(op=op, dim=dim, glc=glc):
                        spv = translate(op, vdata, dmask, dim, addr, glc)
                        self.assertEqual(spv.count("_oobend = OpLabel"), 1 if glc else 0)
                        if glc:
                            # The zero write sits behind (active AND NOT in range), after the atomic's own block.
                            self.assertEqual(spv.count("_oob = OpLabel"), 1)
                            self.assertGreater(spv.find("_oob = OpLabel"), spv.find("OpImageTexelPointer"))
                        else:
                            self.assertNotIn("_oob = OpLabel", spv)

    def test_a16_coordinates_are_checked_too(self):
        spv = translate_cs([
            "v_mov_b32_e32 v1, 0x00050003",
            "v_mov_b32_e32 v2, 7",
            "image_atomic_add v2, v1, s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm a16 glc",
        ])
        check_cfg(spv)
        self.assertLess(spv.find("OpImageQuerySize %v2int"), spv.find("OpImageTexelPointer"))
        guard_text = spv[spv.find("OpImageQuerySize %v2int"):spv.find("OpImageTexelPointer")]
        self.assertEqual(len(re.findall(r"= OpCompositeExtract %int %\w+ \d", guard_text)), 2)


if __name__ == "__main__":
    unittest.main()
