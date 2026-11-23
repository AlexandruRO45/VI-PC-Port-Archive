"""Regression tests for unified VGPR read/write operand analysis in tools/rdna2spirv.py.

read_before_write_vgprs, interp_ij_analysis, reg_reads_writes and interp_copy_sources
must agree on which VGPR operands an instruction reads and writes:
  * v_swap_b32 reads and writes both operands;
  * buffer_atomic_* and image_atomic_* read vdata (a[0]), and only write it with glc
    (cmpswap with glc reads both vdata words and writes only the low half);
  * global_atomic_* without glc has no vdst (reads all VGPR operands);
  * non-returning ds_* atomics (ds_add_u32, ds_mskor_b32, ds_cmpst_b32, ...) have no vdst;
  * ds_read_*_d16 and global_load_*_d16 preserve the untouched half of vdst (read + write);
  * v_cmpx_* has no VGPR destination (does not disqualify pristine barycentric copies).

Standard library only; no GPU, no game files.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from shader_ir import parse_line as parse  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

BARY = (0.25, 0.25, 0.5)  # (k, I, J)
I, J = BARY[1], BARY[2]
UV = ((0.0, 0.0, 1.0, 0.0), (2.0, 0.0, 3.0, 0.0), (0.0, 2.0, 5.0, 0.0))
ATTR0 = [tuple(struct.unpack("<I", struct.pack("<f", x))[0] for x in v) for v in UV]


def run_ps(asm, **kw):
    spv = support.translate_lines("ps_vgpr_rw", asm)
    inputs = support.varyings(spv, {0: ATTR0}, BARY)
    words = support.mrt0_words(spv, inputs=inputs, **kw)
    return spv, struct.unpack("<4f", struct.pack("<4I", *words))


class VgprReadWriteUnitTests(unittest.TestCase):
    def test_accumulating_dot_products_read_the_destination(self):
        for op in ("v_dot2c_i32_i16", "v_dot2c_u32_u16", "v_dot4c_u32_u8",
                   "v_dot8c_i32_i4", "v_dot8c_u32_u4", "v_dot2c_f32_f16"):
            for encoding in ("e32", "e64"):
                with self.subTest(op=op, encoding=encoding):
                    ins = [parse(f"\t{op}_{encoding} v0, v1, v2")]
                    self.assertEqual(rdna2spirv.reg_reads_writes(ins), ({0, 1, 2}, {0}))
                    self.assertEqual(rdna2spirv.read_before_write_vgprs(ins), {0, 1, 2})
                    initialized = [parse("\tv_mov_b32_e32 v0, 0")] + ins
                    self.assertEqual(rdna2spirv.read_before_write_vgprs(initialized), {1, 2})

    def test_sdwa_preserve_and_buffer_d16_and_flat_store(self):
        cases = [
            ("\tv_mov_b32_sdwa v5, v1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD", {1, 5}, {5}),
            ("\tv_mov_b32_sdwa v5, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD", {1}, {5}),
            ("\tbuffer_load_short_d16 v5, v1, s[0:3], 0 offen", {1, 5}, {5}),
            ("\tbuffer_load_format_d16_xy v5, v1, s[0:3], 0 offen", {1}, {5}),
            ("\tbuffer_atomic_fcmpswap v[0:1], v2, s[0:3], 0 offen glc", {0, 1, 2}, {0}),
            ("\tglobal_atomic_csub v5, v1, s[0:1]", {1, 5}, set()),
            ("\tglobal_atomic_csub v5, v1, v2, s[0:1] glc", {1, 2}, {5}),
            ("\tflat_store_dword v[0:1], v2", {0, 1, 2}, set()),
        ]
        for line, want_r, want_w in cases:
            with self.subTest(line=line.strip()):
                ins = [parse(line)]
                self.assertEqual(rdna2spirv.reg_reads_writes(ins), (want_r, want_w))
                self.assertEqual(rdna2spirv.read_before_write_vgprs(ins), want_r)

    def test_v_swap_reads_and_writes_both_operands(self):
        ins = [parse("\tv_swap_b32_e32 v5, v6")]
        self.assertEqual(rdna2spirv.reg_reads_writes(ins), ({5, 6}, {5, 6}))
        self.assertEqual(rdna2spirv.read_before_write_vgprs(ins), {5, 6})
        # Subsequent read of v6 is NOT read-before-write because v_swap wrote v6 too.
        seq = [parse("\tv_mov_b32_e32 v5, 1"), parse("\tv_swap_b32_e32 v5, v6"), parse("\tv_add_nc_u32_e32 v7, v6, 1")]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(seq), {6})

    def test_buffer_and_image_atomics(self):
        cases = [
            ("\tbuffer_atomic_add v0, v1, s[0:3], 0 offen", {0, 1}, set()),
            ("\tbuffer_atomic_add v0, v1, s[0:3], 0 offen glc", {0, 1}, {0}),
            ("\tbuffer_atomic_and_x2 v[0:1], v2, s[0:3], 0 offen", {0, 1, 2}, set()),
            ("\tbuffer_atomic_and_x2 v[0:1], v2, s[0:3], 0 offen glc", {0, 1, 2}, {0, 1}),
            ("\tbuffer_atomic_cmpswap v[0:1], v2, s[0:3], 0 offen", {0, 1, 2}, set()),
            ("\tbuffer_atomic_cmpswap v[0:1], v2, s[0:3], 0 offen glc", {0, 1, 2}, {0}),
            ("\timage_atomic_add v0, v[1:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D", {0, 1, 2}, set()),
            ("\timage_atomic_add v0, v[1:2], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D glc", {0, 1, 2}, {0}),
            ("\timage_atomic_cmpswap v[0:1], v[2:3], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D", {0, 1, 2, 3}, set()),
            ("\timage_atomic_cmpswap v[0:1], v[2:3], s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_2D glc", {0, 1, 2, 3}, {0}),
        ]
        for line, want_r, want_w in cases:
            with self.subTest(line=line.strip()):
                ins = [parse(line)]
                self.assertEqual(rdna2spirv.reg_reads_writes(ins), (want_r, want_w))
                self.assertEqual(rdna2spirv.read_before_write_vgprs(ins), want_r)

    def test_global_atomics_and_d16_loads(self):
        cases = [
            ("\tglobal_atomic_add v5, v1, s[0:1]", {1, 5}, set()),
            ("\tglobal_atomic_add v5, v1, v2, s[0:1] glc", {1, 2}, {5}),
            ("\tglobal_load_short_d16 v5, v1, s[0:1]", {1, 5}, {5}),
            ("\tglobal_load_ubyte_d16_hi v5, v1, s[0:1]", {1, 5}, {5}),
            ("\tglobal_load_ushort v5, v1, s[0:1]", {1}, {5}),
            ("\tds_read_u16_d16 v5, v1", {1, 5}, {5}),
            ("\tds_read_i8_d16_hi v5, v1", {1, 5}, {5}),
            ("\tds_read_u16 v5, v1", {1}, {5}),
        ]
        for line, want_r, want_w in cases:
            with self.subTest(line=line.strip()):
                ins = [parse(line)]
                self.assertEqual(rdna2spirv.reg_reads_writes(ins), (want_r, want_w))
                self.assertEqual(rdna2spirv.read_before_write_vgprs(ins), want_r)

    def test_ds_non_rtn_vs_rtn_atomics(self):
        cases = [
            ("\tds_add_u32 v0, v1", {0, 1}, set()),
            ("\tds_sub_u32 v0, v1", {0, 1}, set()),
            ("\tds_min_i32 v0, v1", {0, 1}, set()),
            ("\tds_and_b64 v0, v[1:2]", {0, 1, 2}, set()),
            ("\tds_mskor_b32 v0, v1, v2", {0, 1, 2}, set()),
            ("\tds_cmpst_b32 v0, v1, v2", {0, 1, 2}, set()),
            ("\tds_add_rtn_u32 v5, v0, v1", {0, 1}, {5}),
            ("\tds_wrxchg_rtn_b32 v5, v0, v1", {0, 1}, {5}),
            ("\tds_mskor_rtn_b32 v5, v0, v1, v2", {0, 1, 2}, {5}),
            ("\tds_cmpst_rtn_b32 v5, v0, v1, v2", {0, 1, 2}, {5}),
            ("\tds_swizzle_b32 v5, v0 offset:swizzle(QUAD_PERM,1,0,3,2)", {0}, {5}),
            ("\tds_bpermute_b32 v5, v0, v1", {0, 1}, {5}),
        ]
        for line, want_r, want_w in cases:
            with self.subTest(line=line.strip()):
                ins = [parse(line)]
                self.assertEqual(rdna2spirv.reg_reads_writes(ins), (want_r, want_w))
                self.assertEqual(rdna2spirv.read_before_write_vgprs(ins), want_r)

    def test_interp_copy_sources_respects_cmpx_atomics_and_swap(self):
        # v_cmpx and non-glc atomics do not write v0, so v10 stays a pristine copy.
        for prefix in (
            "\tv_cmpx_gt_f32_e32 v0, 0",
            "\tbuffer_atomic_add v0, v4, s[0:3], 0 offen",
            "\tglobal_atomic_add v0, v4, s[0:1]",
            "\tds_add_u32 v0, v4",
        ):
            with self.subTest(prefix=prefix.strip()):
                insns = [parse(prefix), parse("\tv_mov_b32_e32 v10, v0")]
                self.assertEqual(rdna2spirv.interp_copy_sources(insns), {10: (0, 1)})

        # v_swap_b32 overwrites its second operand: both copy target and source invalidation must fire.
        self.assertEqual(
            rdna2spirv.interp_copy_sources([parse("\tv_mov_b32_e32 v10, v0"), parse("\tv_swap_b32_e32 v11, v10")]),
            {},
        )
        self.assertEqual(
            rdna2spirv.interp_copy_sources([parse("\tv_swap_b32_e32 v11, v0"), parse("\tv_mov_b32_e32 v10, v0")]),
            {},
        )


class VgprReadWriteEndToEndTests(unittest.TestCase):
    def test_ps_swap_barycentrics_reads_both_and_swaps_weights(self):
        """v_swap_b32 v0, v1 must load BaryCoordKHR.y into v0 before swapping, yielding (2*J, 2*I)."""
        _, f = run_ps([
            "v_swap_b32_e32 v0, v1",
            "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "v_interp_p1_f32_e32 v3, v0, attr0.y", "v_interp_p2_f32_e32 v3, v1, attr0.y",
            "exp mrt0 v2, v3, v2, v3 done vm", "s_endpgm",
        ])
        self.assertEqual(f[:2], (2 * J, 2 * I))

    def test_ps_swap_writing_v0_as_second_operand_triggers_computed_ij(self):
        """v_swap_b32 v5, v0 overwrites v0 with 0.0; attr0.x (= 2*I') must become 0.0 instead of 2*I."""
        spv, f = run_ps([
            "v_mov_b32_e32 v5, 0",
            "v_swap_b32_e32 v5, v0",
            "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm",
        ])
        self.assertIn("PerVertexKHR", spv)
        self.assertEqual(f[0], 0.0)

    def test_ps_swap_invalidating_copy_uses_swapped_value(self):
        """v_swap_b32 v11, v10 overwrites a prior v_mov v10, v0 copy; attr0.x must see 0.0, not raw I."""
        spv, f = run_ps([
            "v_mov_b32_e32 v10, v0",
            "v_mov_b32_e32 v11, 0",
            "v_swap_b32_e32 v11, v10",
            "v_interp_p1_f32_e32 v2, v10, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm",
        ])
        self.assertIn("PerVertexKHR", spv)
        self.assertEqual(f[0], 0.0)

    def test_ps_buffer_atomic_reads_bary_i_without_marking_v0_written(self):
        """Non-glc buffer_atomic_add v0 must load BaryCoordKHR.y and keep v0 pristine for v_interp."""
        base = 0x10000
        spv = support.translate_lines("ps_buf_atomic_bary", [
            f"s_mov_b32 s0, {base:#x}",
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 64",
            "s_mov_b32 s3, 0x31016fac",
            "v_mov_b32_e32 v4, 0",
            "buffer_atomic_add v0, v4, s[0:3], 0 offen",
            "s_waitcnt vmcnt(0)",
            "v_interp_p1_f32_e32 v2, v0, attr0.x", "v_interp_p2_f32_e32 v2, v1, attr0.x",
            "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm",
        ])
        # v0 was not written, so attr0 stays a plain interpolated input (not PerVertexKHR).
        self.assertNotIn("PerVertexKHR", spv)
        mem = {base: 0}
        inputs = support.varyings(spv, {0: ATTR0}, BARY)
        si.run_fragment(spv, inputs=inputs, memory=mem)
        self.assertEqual(mem[base], struct.unpack("<I", struct.pack("<f", I))[0])

    def test_non_glc_atomic_reading_p1_partial_reports_read_not_rewrite(self):
        """A non-glc global/buffer atomic using the p1 partial reads it; interp_ij_analysis must say read."""
        for atomic_line in (
            "global_atomic_add v2, v4, s[0:1]",
            "buffer_atomic_add v2, v4, s[0:3], 0 offen",
        ):
            with self.subTest(atomic_line=atomic_line):
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    support.translate_lines("ps_partial_read", [
                        "v_mov_b32_e32 v0, 0",
                        "v_mov_b32_e32 v1, 0",
                        "v_mov_b32_e32 v4, 0",
                        "v_interp_p1_f32_e32 v2, v0, attr0.x",
                        atomic_line,
                        "v_interp_p2_f32_e32 v2, v1, attr0.x",
                        "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm",
                    ])
                self.assertIn("read between the ops", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
