"""Wave32 pixel shaders stay isolated when Vulkan runs them in a 64-lane subgroup.

The inputs are deliberately different in the two physical halves. Each translated PS is run both as one 64-lane
subgroup and as two 32-lane subgroups; outputs must match the independent logical-wave32 expectations.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support
import spirv_mini_interp as si


def fbits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def run_ps(spv, subgroup_size, inputs):
    _, outputs = si.run_workgroup(
        spv, 64, subgroup_size,
        inputs_for=lambda i: {
            "%sg_lane": i % subgroup_size,
            "%in_attr0": (inputs[i], 0.0, 0.0, 0.0),
        },
        return_outputs=True,
    )
    return [out["%out_mrt0"] for out in outputs]


class Wave32PixelSubgroupTests(unittest.TestCase):
    def test_lane_reads_and_writes_stay_in_each_half(self):
        spv = support.translate_lines("ps_wave32_halves_lane_ops", [
            "v_interp_mov_f32_e32 v2, p0, attr0.x",
            "v_readfirstlane_b32 s20, v2",
            "v_readlane_b32 s21, v2, 0",
            "v_mov_b32_e32 v6, s20",
            "v_mov_b32_e32 v7, s21",
            "v_mov_b32_e32 v8, 0",
            "v_writelane_b32 v8, s20, 0",
            "v_mov_b32_e32 v3, 0",
            "ds_bpermute_b32 v9, v3, v2",
            "exp mrt0 v6, v7, v8, v9 done vm",
            "s_endpgm",
        ])
        inputs = [float(i + 1) for i in range(64)]
        expected = []
        for i in range(64):
            first = fbits(float((i // 32) * 32 + 1))
            expected.append((first, first, first if i % 32 == 0 else 0, first))
        for subgroup_size in (32, 64):
            with self.subTest(subgroup_size=subgroup_size):
                self.assertEqual(run_ps(spv, subgroup_size, inputs), expected)

    def test_ds_permute_scatter_stays_in_each_half(self):
        spv = support.translate_lines("ps_wave32_halves_ds_permute", [
            "v_interp_mov_f32_e32 v2, p0, attr0.x",
            "v_mbcnt_lo_u32_b32 v5, -1, 0",
            "v_lshlrev_b32_e32 v3, 2, v5",
            "ds_permute_b32 v4, v3, v2",
            "exp mrt0 v4, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        inputs = [float(i + 1) for i in range(64)]
        expected = [(fbits(value),) * 4 for value in inputs]
        for subgroup_size in (32, 64):
            with self.subTest(subgroup_size=subgroup_size):
                self.assertEqual(run_ps(spv, subgroup_size, inputs), expected)

    def test_ballots_and_votes_stay_in_each_half(self):
        spv = support.translate_lines("ps_wave32_halves_ballot", [
            "v_interp_mov_f32_e32 v1, p0, attr0.x",
            "v_cmp_ne_u32_e32 vcc_lo, 0, v1",
            "s_bcnt1_i32_b32 s20, vcc_lo",
            "s_cmp_lg_u32 vcc_lo, 0",
            "s_cselect_b32 s21, 1, 0",
            "s_cmp_eq_u32 vcc_lo, exec_lo",
            "s_cselect_b32 s22, 1, 0",
            "v_mov_b32_e32 v6, s20",
            "v_mov_b32_e32 v7, s21",
            "v_mov_b32_e32 v8, s22",
            "exp mrt0 v6, v7, v8, v8 done vm",
            "s_endpgm",
        ])
        inputs = [0.0] * 32 + [1.0] * 32
        expected = [(0, 0, 0, 0)] * 32 + [(32, 1, 1, 1)] * 32
        for subgroup_size in (32, 64):
            with self.subTest(subgroup_size=subgroup_size):
                self.assertEqual(run_ps(spv, subgroup_size, inputs), expected)

    def test_image_descriptor_comes_from_a_running_lane_of_each_half(self):
        # At a primitive's edge a half can start with uncovered quads: no invocation runs in its first lanes, and a
        # shuffle from one of them is undefined. Each half's descriptor (its own s0) must still pick its own image.
        spv = support.translate_lines("ps_wave32_halves_image", [
            "v_interp_mov_f32_e32 v2, p0, attr0.x",
            "v_readfirstlane_b32 s0, v2",
            "v_mov_b32_e32 v4, 0",
            "v_mov_b32_e32 v5, 0",
            "image_load v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        # The descriptor map: slot k is words 10k + 1 to 10k + 8 (the descriptor) and 10k + 9 (its bindless index);
        # word 0 is the slot mask. Mask 1 gives two slots, both probed whatever the hash: half h's s0 (h + 1.0) finds
        # image h + 1.
        desc_map = []
        for half in range(2):
            desc_map += [1 if half == 0 else 0, fbits(half + 1.0)] + [0] * 7 + [half + 1]
        texels = {1: (fbits(0.25), 0, 0, fbits(1.0)), 2: (0, fbits(0.75), 0, fbits(1.0))}
        absent = set(range(0, 4)) | set(range(32, 36))       # the first quad of each half is not covered
        expected = [None if i in absent else texels[i // 32 + 1] for i in range(64)]
        for subgroup_size in (32, 64):
            with self.subTest(subgroup_size=subgroup_size):
                _, outputs = si.run_workgroup(
                    spv, 64, subgroup_size,
                    inputs_for=lambda i: {
                        "%sg_lane": i % subgroup_size,
                        "%in_attr0": (float(i // 32 + 1), 0.0, 0.0, 0.0),
                    },
                    images={"%images": texels}, buffers={"%desc_map": desc_map}, absent=absent,
                    return_outputs=True,
                )
                self.assertEqual([out and out["%out_mrt0"] for out in outputs], expected)


if __name__ == "__main__":
    unittest.main()
