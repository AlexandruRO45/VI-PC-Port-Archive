"""VGPR read/write and live-in analysis for buffer/image/flat atomics and non-returning ds_* ops.

Covers `vgpr_reads_all`, `vgpr_dest_registers`, `read_before_write_vgprs`, `reg_reads_writes`,
`interp_ij_analysis`, and `interp_copy_sources`, plus end-to-end PS translation and `spirv_mini_interp`
execution. Standard library only; no GPU or game data.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402  (puts tools/ on sys.path)
import shader_ir  # noqa: E402

rdna2spirv = support.rdna2spirv

BARY = (0.25, 0.25, 0.5)
VERTS0 = ((0.0, 0.0, 1.0, 0.0), (2.0, 0.0, 3.0, 0.0), (0.0, 2.0, 5.0, 0.0))


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


WORDS0 = [tuple(f32_bits(x) for x in v) for v in VERTS0]
ATTR0_INTERP_X = VERTS0[0][0] + BARY[1] * (VERTS0[1][0] - VERTS0[0][0]) + BARY[2] * (VERTS0[2][0] - VERTS0[0][0])
# V# with base=0x1000, stride=0, num_records=0x100, raw data format (r32_uint)
VSHARP_SETUP = [
    "s_mov_b32 s4, 0x1000",
    "s_mov_b32 s5, 0",
    "s_mov_b32 s6, 0x100",
    "s_mov_b32 s7, 0x31016fac",
]


def parse(lines):
    return [shader_ir.parse_line("\t" + ln) for ln in lines]


class TestVgprReadWriteClassification(unittest.TestCase):
    def test_buffer_and_image_atomics_without_and_with_glc(self):
        cases = [
            ("buffer_atomic_add v0, v2, s[0:3], 0 offen", {0, 2}, set()),
            ("buffer_atomic_add v0, v2, s[0:3], 0 offen glc", {0, 2}, {0}),
            ("buffer_atomic_cmpswap v[0:1], v2, s[0:3], 0 offen", {0, 1, 2}, set()),
            ("buffer_atomic_cmpswap v[0:1], v2, s[0:3], 0 offen glc", {0, 1, 2}, {0}),
            ("buffer_atomic_fcmpswap v[0:1], v2, s[0:3], 0 offen glc", {0, 1, 2}, {0}),
            ("buffer_atomic_xor_x2 v[0:1], v2, s[0:3], 0 offen", {0, 1, 2}, set()),
            ("buffer_atomic_xor_x2 v[0:1], v2, s[0:3], 0 offen glc", {0, 1, 2}, {0, 1}),
            ("buffer_atomic_cmpswap_x2 v[0:3], v4, s[0:3], 0 offen glc", {0, 1, 2, 3, 4}, {0, 1}),
            ("image_atomic_add v0, v[2:3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm", {0, 2, 3}, set()),
            ("image_atomic_add v0, v[2:3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm glc", {0, 2, 3}, {0}),
            ("image_atomic_cmpswap v[0:1], v[2:3], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm", {0, 1, 2, 3}, set()),
            ("image_atomic_cmpswap v[0:1], v[2:3], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc", {0, 1, 2, 3}, {0}),
            ("image_atomic_fcmpswap v[0:1], v[2:3], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D unorm glc", {0, 1, 2, 3}, {0}),
        ]
        for line, want_r, want_w in cases:
            with self.subTest(line=line):
                insns = parse([line])
                self.assertEqual(rdna2spirv.reg_reads_writes(insns), (want_r, want_w))
                self.assertEqual(rdna2spirv.read_before_write_vgprs(insns), want_r)

    def test_flat_and_global_atomics_without_and_with_glc(self):
        cases = [
            ("flat_atomic_add v[2:3], v0", {0, 2, 3}, set()),
            ("flat_atomic_add v4, v[2:3], v0 glc", {0, 2, 3}, {4}),
            ("flat_atomic_cmpswap v[2:3], v[4:5]", {2, 3, 4, 5}, set()),
            ("flat_atomic_cmpswap v6, v[2:3], v[4:5] glc", {2, 3, 4, 5}, {6}),
            ("global_atomic_add v[2:3], v0, off", {0, 2, 3}, set()),
            ("global_atomic_add v4, v[2:3], v0, off glc", {0, 2, 3}, {4}),
        ]
        for line, want_r, want_w in cases:
            with self.subTest(line=line):
                insns = parse([line])
                self.assertEqual(rdna2spirv.reg_reads_writes(insns), (want_r, want_w))
                self.assertEqual(rdna2spirv.read_before_write_vgprs(insns), want_r)

    def test_non_returning_vs_returning_ds_ops(self):
        non_returning = [
            ("ds_add_u32 v2, v0", {0, 2}),
            ("ds_sub_u32 v2, v0", {0, 2}),
            ("ds_inc_u32 v2, v0", {0, 2}),
            ("ds_dec_u32 v2, v0", {0, 2}),
            ("ds_min_i32 v2, v0", {0, 2}),
            ("ds_max_f32 v2, v0", {0, 2}),
            ("ds_cmpst_b32 v2, v0, v1", {0, 1, 2}),
            ("ds_cmpst_f32 v2, v0, v1", {0, 1, 2}),
            ("ds_mskor_b32 v2, v0, v1", {0, 1, 2}),
            ("ds_write_addtid_b32 v0", {0}),
            ("ds_gws_barrier v0 offset:0 gds", {0}),
        ]
        for line, want_r in non_returning:
            with self.subTest(line=line):
                insns = parse([line])
                self.assertEqual(rdna2spirv.reg_reads_writes(insns), (want_r, set()))
                self.assertEqual(rdna2spirv.read_before_write_vgprs(insns), want_r)

        returning = [
            ("ds_add_rtn_u32 v4, v2, v0", {0, 2}, {4}),
            ("ds_cmpst_rtn_b32 v4, v2, v0, v1", {0, 1, 2}, {4}),
            ("ds_wrxchg_rtn_b32 v4, v2, v0", {0, 2}, {4}),
            ("ds_read_b32 v4, v2", {2}, {4}),
            ("ds_bpermute_b32 v4, v2, v0", {0, 2}, {4}),
            ("ds_permute_b32 v4, v2, v0", {0, 2}, {4}),
            ("ds_swizzle_b32 v4, v0", {0}, {4}),
            ("ds_append v4", set(), {4}),
            ("ds_consume v4", set(), {4}),
            ("ds_ordered_count v4, v0", {0}, {4}),
            ("ds_read_addtid_b32 v4", set(), {4}),
        ]
        for line, want_r, want_w in returning:
            with self.subTest(line=line):
                insns = parse([line])
                self.assertEqual(rdna2spirv.reg_reads_writes(insns), (want_r, want_w))
                self.assertEqual(rdna2spirv.read_before_write_vgprs(insns), want_r)

    def test_read_before_write_tracks_overwrites_after_read(self):
        seq = parse([
            "buffer_atomic_add v0, v2, s[0:3], 0 offen",
            "v_mov_b32_e32 v0, 1",
            "ds_add_u32 v3, v0",
            "v_mov_b32_e32 v3, 2",
            "v_add_u32_e32 v4, v0, v3",
        ])
        self.assertEqual(rdna2spirv.read_before_write_vgprs(seq), {0, 2, 3})

        written_first = parse([
            "v_mov_b32_e32 v0, 1",
            "buffer_atomic_add v0, v2, s[0:3], 0 offen glc",
        ])
        self.assertEqual(rdna2spirv.read_before_write_vgprs(written_first), {2})

    def test_synthetic_insns_without_mods_key(self):
        insns = [{"op": "flat_atomic_add", "args": [{"k": "vgpr", "r": 2, "n": 2}, {"k": "vgpr", "r": 0, "n": 1}]}]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(insns), {0, 2, 3})
        self.assertEqual(rdna2spirv.reg_reads_writes(insns), ({0, 2, 3}, set()))


class TestPixelShaderLiveInAndInterpAnalysis(unittest.TestCase):
    def test_ps_buffer_atomic_no_glc_preserves_bary_livein_end_to_end(self):
        spv = support.translate_lines("ps_buf_atomic_livein", VSHARP_SETUP + [
            "v_mov_b32_e32 v4, 0",
            "buffer_atomic_add v0, v4, s[4:7], 0 offen",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpDecorate %bary BuiltIn BaryCoordKHR", spv)
        mem = {0x1000: 10}
        res = si.run_fragment(spv, inputs={"%bary": BARY}, memory=mem)
        self.assertEqual(res["%out_mrt0"], (0, 0, 0, 0))
        self.assertEqual(mem[0x1000], (10 + f32_bits(BARY[1])) & 0xFFFFFFFF)

    def test_ps_buffer_atomic_glc_reads_vdata_livein_and_returns_old(self):
        spv = support.translate_lines("ps_buf_atomic_glc_livein", VSHARP_SETUP + [
            "v_mov_b32_e32 v4, 0",
            "buffer_atomic_add v0, v4, s[4:7], 0 offen glc",
            "exp mrt0 v0, v4, v4, v4 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpDecorate %bary BuiltIn BaryCoordKHR", spv)
        mem = {0x1000: 10}
        res = si.run_fragment(spv, inputs={"%bary": BARY}, memory=mem)
        self.assertEqual(res["%out_mrt0"][0], 10)
        self.assertEqual(mem[0x1000], (10 + f32_bits(BARY[1])) & 0xFFFFFFFF)

    def test_ps_buffer_atomic_cmpswap_glc_reads_both_and_keeps_cmp(self):
        spv = support.translate_lines("ps_buf_cmpswap_glc_livein", VSHARP_SETUP + [
            "v_mov_b32_e32 v4, 0",
            "buffer_atomic_cmpswap v[0:1], v4, s[4:7], 0 offen glc",
            "exp mrt0 v0, v1, v4, v4 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpDecorate %bary BuiltIn BaryCoordKHR", spv)
        mem = {0x1000: f32_bits(BARY[2])}
        res = si.run_fragment(spv, inputs={"%bary": BARY}, memory=mem)
        self.assertEqual(mem[0x1000], f32_bits(BARY[1]))
        self.assertEqual(res["%out_mrt0"][:2], (f32_bits(BARY[2]), f32_bits(BARY[2])))

    def test_non_returning_atomic_before_interp_keeps_centre_interpolation(self):
        spv = support.translate_lines("ps_buf_before_interp", VSHARP_SETUP + [
            "v_mov_b32_e32 v4, 0",
            "buffer_atomic_add v0, v4, s[4:7], 0 offen",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        self.assertNotIn("PerVertexKHR", spv)
        self.assertIn("OpDecorate %bary BuiltIn BaryCoordKHR", spv)
        mem = {0x1000: 10}
        res = si.run_fragment(spv, inputs=support.varyings(spv, {0: WORDS0}, BARY), memory=mem)
        self.assertEqual(mem[0x1000], (10 + f32_bits(BARY[1])) & 0xFFFFFFFF)
        self.assertEqual(res["%out_mrt0"][0], f32_bits(ATTR0_INTERP_X))

        spv_img = support.translate_lines("ps_img_before_interp", [
            "v_mov_b32_e32 v4, 0",
            "v_mov_b32_e32 v5, 0",
            "image_atomic_add v0, v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        self.assertNotIn("PerVertexKHR", spv_img)
        self.assertIn("OpDecorate %bary BuiltIn BaryCoordKHR", spv_img)

        spv_glc = support.translate_lines("ps_buf_glc_before_interp", VSHARP_SETUP + [
            "v_mov_b32_e32 v4, 0",
            "buffer_atomic_add v0, v4, s[4:7], 0 offen glc",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        self.assertIn("PerVertexKHR", spv_glc)

    def test_interp_copy_sources_ignores_read_only_ops_and_tracks_glc_write(self):
        for ro_op in ("buffer_atomic_add v6, v4, s[4:7], 0 offen", "v_cmpx_ne_u32_e32 v6, 0"):
            with self.subTest(op=ro_op):
                spv = support.translate_lines("ps_copy_ro", VSHARP_SETUP + [
                    "v_mov_b32_e32 v4, 0",
                    "v_mov_b32_e32 v6, v1",
                    ro_op,
                    "v_interp_p1_f32_e32 v2, v0, attr0.x",
                    "v_interp_p2_f32_e32 v2, v6, attr0.x",
                    "exp mrt0 v2, v2, v2, v2 done vm",
                    "s_endpgm",
                ])
                self.assertNotIn("PerVertexKHR", spv)
                mem = {0x1000: 0}
                res = si.run_fragment(spv, inputs=support.varyings(spv, {0: WORDS0}, BARY), memory=mem)
                self.assertEqual(res["%out_mrt0"][0], f32_bits(ATTR0_INTERP_X))

        spv_img = support.translate_lines("ps_copy_img", [
            "v_mov_b32_e32 v4, 0",
            "v_mov_b32_e32 v5, 0",
            "v_mov_b32_e32 v6, v1",
            "image_atomic_add v6, v[4:5], s[4:11] dmask:0x1 dim:SQ_RSRC_IMG_2D unorm",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v6, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        self.assertNotIn("PerVertexKHR", spv_img)

        spv_glc = support.translate_lines("ps_copy_glc", VSHARP_SETUP + [
            "v_mov_b32_e32 v4, 0",
            "v_mov_b32_e32 v6, v1",
            "buffer_atomic_add v6, v4, s[4:7], 0 offen glc",
            "v_interp_p1_f32_e32 v2, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v6, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        self.assertIn("PerVertexKHR", spv_glc)

        for unit_op in (
            "ds_add_u32 v6, v4",
            "flat_atomic_add v[6:7], v4",
            "global_atomic_add v[6:7], v4, off",
            "flat_store_dword v[6:7], v4",
            "v_cmpx_eq_u32_e32 v6, 0",
            "buffer_atomic_cmpswap v[5:6], v4, s[4:7], 0 offen glc",
        ):
            with self.subTest(kept_op=unit_op):
                insns = parse([
                    "v_mov_b32_e32 v6, v1",
                    unit_op,
                    "v_interp_p1_f32_e32 v2, v0, attr0.x",
                    "v_interp_p2_f32_e32 v2, v6, attr0.x",
                    "exp mrt0 v2, v2, v2, v2 done vm",
                    "s_endpgm",
                ])
                for k, ins in enumerate(insns):
                    ins["pc"] = 4 * k
                copies = rdna2spirv.interp_copy_sources(insns)
                self.assertEqual(copies.get(6), (1, 0))

        for write_op in (
            "buffer_atomic_add v6, v4, s[4:7], 0 offen glc",
            "flat_atomic_add v6, v[2:3], v4 glc",
            "ds_add_rtn_u32 v6, v2, v4",
        ):
            with self.subTest(clobber_op=write_op):
                insns = parse([
                    "v_mov_b32_e32 v6, v1",
                    write_op,
                    "v_interp_p1_f32_e32 v2, v0, attr0.x",
                    "v_interp_p2_f32_e32 v2, v6, attr0.x",
                    "exp mrt0 v2, v2, v2, v2 done vm",
                    "s_endpgm",
                ])
                for k, ins in enumerate(insns):
                    ins["pc"] = 4 * k
                copies = rdna2spirv.interp_copy_sources(insns)
                self.assertNotIn(6, copies)

    def test_partial_between_p1_p2_distinguishes_read_from_rewrite(self):
        for between in ("buffer_atomic_add v2, v4, s[4:7], 0 offen", "ds_add_u32 v2, v4"):
            with self.subTest(between=between):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    insns = parse([
                        "v_mov_b32_e32 v0, 0.25",
                        "v_interp_p1_f32_e32 v2, v0, attr0.x",
                        between,
                        "v_interp_p2_f32_e32 v2, v1, attr0.x",
                        "s_endpgm",
                    ])
                    for k, ins in enumerate(insns):
                        ins["pc"] = 4 * k
                    g = type("G", (), {"v_interp_copy": {}, "pixel_interpolants": None, "frag_inputs": {}})()
                    rdna2spirv.interp_ij_analysis(insns, g)
                self.assertIn("is read between the ops", str(cm.exception))

        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_partial_glc", VSHARP_SETUP + [
                "v_mov_b32_e32 v0, 0.25",
                "v_mov_b32_e32 v4, 0",
                "v_interp_p1_f32_e32 v2, v0, attr0.x",
                "buffer_atomic_add v2, v4, s[4:7], 0 offen glc",
                "v_interp_p2_f32_e32 v2, v1, attr0.x",
                "exp mrt0 v2, v2, v2, v2 done vm",
                "s_endpgm",
            ])
        self.assertIn("is rewritten between the ops", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
