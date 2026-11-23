"""tools/gbuffer_export_census.py on hand-written listings and containers (no game files, no GPU)."""
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "gbuffer_export_census.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import gbuffer_export_census as gec  # noqa: E402


def osg1_dxbc(masks, depth=False):
    """A DXBC container holding only an OSG1 part: one SV_Target per mask (register = index)."""
    elems = [(0, 0, i, 64, 3, i, mask) for i, mask in enumerate(masks)]
    if depth:
        elems.append((0, 0, 0, 65, 3, 0xFFFFFFFF, 1))
    body = struct.pack("<II", len(elems), 8)
    for stream, name, idx, sysv, ctype, reg, mask in elems:
        body += struct.pack("<6IBBHI", stream, name, idx, sysv, ctype, reg, mask, mask, 0, 0)
    part = b"OSG1" + struct.pack("<I", len(body)) + body
    head = b"DXBC" + bytes(16) + struct.pack("<III", 1, 36 + len(part), 1) + struct.pack("<I", 36)
    return head + part


def write_case(root, name, masks, lines):
    disasm = os.path.join(root, "disasm")
    os.makedirs(disasm, exist_ok=True)
    with open(os.path.join(root, name + ".dxbc"), "wb") as f:
        f.write(osg1_dxbc(masks))
    with open(os.path.join(disasm, name + ".s"), "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(f"\t{line:<44}; encoding: [0x00,0x00,0x00,0x00]\n" if not line.endswith(":") else line + "\n"
                        for line in lines))
    return disasm


def alpha_of(lines):
    ins = gec.parse(lines)
    ei = next(i for i, (op, ops) in enumerate(ins) if op == "exp" and ops[0] == "mrt0")
    return gec.mrt0_alpha(ins, ei)


EXPORT = ["v_cvt_pkrtz_f16_f32_e32 v1, v6, v5", "exp mrt0, v0, v0, v1, v1 compr"]


class Mrt0AlphaTest(unittest.TestCase):
    def test_byte_times_inverse_255(self):
        lines = ["v_cvt_f32_ubyte0_e32 v4, v3", "v_mul_f32_e32 v5, 0x3b808081, v4"] + EXPORT
        self.assertEqual(alpha_of(lines), ("byte 0 x 1/255", False))

    def test_other_byte_and_legacy_multiply(self):
        lines = ["v_cvt_f32_ubyte2_e32 v4, v3", "v_mul_legacy_f32_e32 v5, v4, 0x3b808081"] + EXPORT
        self.assertEqual(alpha_of(lines), ("byte 2 x 1/255", False))

    def test_u32_and_a_scale_moved_into_a_register(self):
        lines = ["s_mov_b32 s9, 0x3b808081", "v_cvt_f32_u32_e32 v4, v3", "v_mul_f32_e32 v5, s9, v4"] + EXPORT
        self.assertEqual(alpha_of(lines), ("u32 x 1/255", False))

    def test_lookalike_constant_is_a_float_product(self):
        # One ulp away from 1/255: not the scale, so not counted as a byte.
        lines = ["v_cvt_f32_ubyte0_e32 v4, v3", "v_mul_f32_e32 v5, 0x3b808082, v4"] + EXPORT
        self.assertEqual(alpha_of(lines), ("float product", False))

    def test_a_later_overwrite_of_the_scale_register_wins(self):
        lines = ["s_mov_b32 s9, 0x3b808081", "s_mov_b32 s9, 0x3f800000", "v_cvt_f32_ubyte0_e32 v4, v3",
                 "v_mul_f32_e32 v5, s9, v4"] + EXPORT
        self.assertEqual(alpha_of(lines), ("float product", False))

    def test_other_producer_is_named_and_a_label_marks_crossing(self):
        lines = ["v_cndmask_b32_e32 v5, v2, v3, vcc_lo", "BB0_1:"] + EXPORT
        self.assertEqual(alpha_of(lines), ("v_cndmask_b32", True))

    def test_alpha_slot_off_and_uncompressed(self):
        self.assertEqual(alpha_of(["exp mrt0, v0, v0, off, off compr"]), ("alpha off", False))
        self.assertEqual(alpha_of(["exp mrt0, v0, v1, v2, v3"]), ("not compr", False))

    def test_range_write_counts_as_writer(self):
        lines = ["s_buffer_load_dwordx2 s[8:9], s[0:3], 0x10", "v_mul_f32_e32 v5, s9, v4"] + EXPORT
        self.assertEqual(alpha_of(lines), ("float product", False))


BYTE = ("byte 0 x 1/255", False)
FLOAT = ("float product", False)


def alpha_with(before=(), after=()):
    """byte0 x 1/255 with the scale in s9, the byte in v4 and the product in v5, extra lines around the multiply."""
    lines = ["s_mov_b32 s9, 0x3b808081", "v_cvt_f32_ubyte0_e32 v4, v3", *before, "v_mul_f32_e32 v5, s9, v4", *after]
    return alpha_of(lines + EXPORT)


class WrittenRegistersTest(unittest.TestCase):
    """Instructions that name a register first but do not write it must not become the last writer.

    The sequences are synthetic; each is checked on the result of the alpha trace, which is what a wrong writer changes."""

    def test_the_reviewer_cases(self):
        self.assertEqual(alpha_with(), BYTE)
        self.assertEqual(alpha_with(before=["s_cmp_lg_u32 s9, 0"]), BYTE)
        self.assertEqual(alpha_with(after=["v_cmpx_gt_f32_e32 v5, 0"]), BYTE)

    def test_scalar_compares_write_only_scc(self):
        for line in ("s_cmp_lg_u32 s9, 0", "s_cmp_eq_u32 s9, s9", "s_cmp_lt_i32 s9, 0x10", "s_cmpk_lg_u32 s9, 0x1",
                     "s_cmpk_eq_i32 s9, -1", "s_bitcmp1_b32 s9, 0", "s_bitcmp0_b32 s9, 3", "s_bitcmp0_b64 s[8:9], 0",
                     "s_cmp_eq_u64 s[8:9], 0"):
            with self.subTest(line):
                self.assertEqual(alpha_with(before=[line]), BYTE)

    def test_v_cmp_e32_writes_vcc_not_its_first_operand(self):
        for line in ("v_cmp_gt_f32_e32 vcc_lo, v4, v3", "v_cmp_eq_u32_e32 vcc_lo, s9, v4", "v_cmp_lt_f32_e32 v4, v3",
                     "v_cmp_ne_u32_e32 s9, v4"):
            with self.subTest(line):
                self.assertEqual(alpha_with(before=[line]), BYTE)
        self.assertEqual(alpha_with(after=["v_cmp_gt_f32_e32 v5, v3"]), BYTE)

    def test_v_cmp_e64_writes_its_sgpr_destination(self):
        # The destination is the first operand: a pair or single SGPR (wave32 uses one), as in the translator.
        self.assertEqual(alpha_with(before=["v_cmp_gt_f32_e64 s[2:3], v3, 0"]), BYTE)
        self.assertEqual(alpha_with(before=["v_cmp_gt_f32_e64 s2, v3, 0"]), BYTE)
        self.assertEqual(alpha_with(before=["v_cmp_gt_f32_e64 s9, v3, 0"]), FLOAT)
        self.assertEqual(alpha_with(before=["v_cmp_gt_f32_e64 s[8:9], v3, 0"]), FLOAT)
        self.assertEqual(alpha_with(before=["v_cmp_class_f32_e64 s[8:9], v3, 0x1f"]), FLOAT)
        self.assertEqual(alpha_with(before=["v_cmp_gt_f32_e64 vcc_lo, v4, v3"]), BYTE)

    def test_v_cmpx_writes_exec_only_in_every_form(self):
        for line in ("v_cmpx_gt_f32_e32 v4, 0", "v_cmpx_lt_f32_e64 v4, v3", "v_cmpx_eq_u32_e32 s9, v4",
                     "v_cmpx_class_f32_e32 v4, 0x1f", "v_cmpx_ne_u32_e64 s9, 0"):
            with self.subTest(line):
                self.assertEqual(alpha_with(before=[line]), BYTE)
                self.assertEqual(alpha_with(after=[line.replace("v4", "v5").replace("s9", "v5")]), BYTE)

    def test_compares_still_do_not_hide_a_real_writer(self):
        lines = ["s_mov_b32 s9, 0x3b808081", "v_cvt_f32_ubyte0_e32 v4, v3", "v_mul_f32_e32 v5, s9, v4",
                 "v_cmpx_gt_f32_e32 v5, 0", "s_cmp_lg_u32 s9, 0", "v_cmp_gt_f32_e32 vcc_lo, v5, v3",
                 "v_cndmask_b32_e32 v5, v2, v5, vcc_lo"]
        self.assertEqual(alpha_of(lines + EXPORT), ("v_cndmask_b32", False))

    def test_second_destination_of_the_vop3b_forms(self):
        self.assertEqual(alpha_with(before=["v_add_co_u32 v6, s9, v1, v2"]), FLOAT)
        self.assertEqual(alpha_with(before=["v_add_co_u32_e64 v6, s[8:9], v1, v2"]), FLOAT)
        self.assertEqual(alpha_with(before=["v_mad_u64_u32 v[6:7], s9, v1, v2, v[8:9]"]), FLOAT)
        self.assertEqual(alpha_with(before=["v_div_scale_f32 v6, s9, v1, v2, v3"]), FLOAT)
        self.assertEqual(alpha_with(before=["v_add_co_u32_e32 v6, vcc_lo, v1, v2"]), BYTE)
        self.assertEqual(alpha_with(before=["v_add_co_ci_u32_e32 v6, vcc_lo, v1, v2, vcc_lo"]), BYTE)
        self.assertEqual(alpha_with(before=["v_add_co_u32_e64 v6, s2, v1, v2"]), BYTE)
        self.assertEqual(alpha_with(after=["v_add_co_u32_e64 v5, s2, v1, v2"]), ("v_add_co_u32", False))

    def test_lane_moves_write_their_destination(self):
        self.assertEqual(alpha_with(before=["v_readlane_b32 s9, v1, 0"]), FLOAT)
        self.assertEqual(alpha_with(before=["v_readfirstlane_b32 s9, v1"]), FLOAT)
        self.assertEqual(alpha_with(before=["v_writelane_b32 v4, s1, 0"]), ("v_writelane_b32 x 1/255", False))
        self.assertEqual(alpha_with(before=["v_readlane_b32 s2, v1, 0"]), BYTE)

    def test_stores_and_barriers_write_no_register(self):
        for line in ("buffer_store_dword v5, v0, s[0:3], 0 offen", "buffer_store_dwordx2 v[4:5], v0, s[0:3], 0 offen",
                     "tbuffer_store_format_xyzw v[4:7], v0, s[0:3], 0 format:[BUF_FMT_32_32_32_32_FLOAT]",
                     "image_store v[4:7], v[0:1], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
                     "global_store_dword v[4:5], v1, off", "flat_store_dword v[4:5], v1", "scratch_store_dword v5, v1, off",
                     "ds_write_b32 v5, v1", "ds_write2_b32 v5, v1, v2 offset1:1", "ds_add_u32 v5, v1",
                     "buffer_atomic_add v5, v0, s[0:3], 0 offen", "global_atomic_add v[4:5], v1, off",
                     "s_waitcnt lgkmcnt(0)", "s_waitcnt_vscnt null, 0x0", "s_nop 0", "s_sendmsg sendmsg(MSG_GS_DONE)",
                     "s_barrier", "s_sleep 1", "s_setprio 3", "s_inst_prefetch 0x3", "s_clause 0x1",
                     "s_dcache_inv", "s_endpgm"):
            with self.subTest(line):
                self.assertEqual(alpha_with(after=[line]), BYTE)

    def test_returning_atomics_and_ds_reads_write_their_destination(self):
        for line, name in (("buffer_atomic_add v5, v0, s[0:3], 0 offen glc", "buffer_atomic_add"),
                           ("global_atomic_add v5, v[2:3], v1, off glc", "global_atomic_add"),
                           ("ds_add_rtn_u32 v5, v1, v2", "ds_add_rtn_u32"),
                           ("ds_read_b32 v5, v1", "ds_read_b32"),
                           ("ds_swizzle_b32 v5, v1 offset:swizzle(QUAD_PERM,0,0,0,0)", "ds_swizzle_b32")):
            with self.subTest(line):
                self.assertEqual(alpha_with(after=[line]), (name, False))

    def test_branches_write_nothing_but_a_swap_does(self):
        # A branch between the byte and its multiply marks the path as crossing; a swap also writes s[8:9], the scale.
        self.assertEqual(alpha_with(before=["s_cbranch_scc0 BB0_2"]), ("byte 0 x 1/255", True))
        self.assertEqual(alpha_with(before=["s_setpc_b64 s[8:9]"]), ("byte 0 x 1/255", True))
        self.assertEqual(alpha_with(before=["s_swappc_b64 s[8:9], s[4:5]"]), FLOAT)

    def test_written_table(self):
        w = gec.written
        self.assertEqual(w("s_cmp_lg_u32", ["s9", "0"]), set())
        self.assertEqual(w("v_cmpx_gt_f32_e32", ["v5", "0"]), set())
        self.assertEqual(w("v_cmp_gt_f32_e32", ["vcc_lo", "v5", "v4"]), set())
        self.assertEqual(w("v_cmp_gt_f32_e64", ["s[2:3]", "v5", "0"]), {("s", 2), ("s", 3)})
        self.assertEqual(w("v_add_co_u32_e64", ["v6", "s[2:3]", "v1", "v2"]), {("v", 6), ("s", 2), ("s", 3)})
        self.assertEqual(w("v_mad_u64_u32", ["v[6:7]", "s9", "v1", "v2", "v[8:9]"]), {("v", 6), ("v", 7), ("s", 9)})
        self.assertEqual(w("v_mov_b32_e32", ["v1", "v2"]), {("v", 1)})
        self.assertEqual(w("s_load_dwordx4", ["s[4:7]", "s[0:1]", "0x0"]), {("s", i) for i in range(4, 8)})
        self.assertEqual(w("buffer_store_dword", ["v1", "v0", "s[0:3]", "0 offen"]), set())
        self.assertEqual(w("s_endpgm", []), set())


class CensusTest(unittest.TestCase):
    def test_counts_by_target_count_and_mrt4_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = ["v_cvt_f32_ubyte0_e32 v4, v3", "v_mul_f32_e32 v5, 0x3b808081, v4"] + EXPORT
            write_case(tmp, "PS_DeferredTextured_a", [15, 15, 15, 15, 3],
                       body + ["exp mrt4, v8, v8, off, off done compr vm"])
            write_case(tmp, ".PS_DeferredTextured_b", [15] * 6,
                       body + ["exp mrt4, v8, v8, v9, v9 compr"])
            disasm = write_case(tmp, "PS_Other_c", [15], body)
            shaders, alpha, mrt4 = gec.census(disasm, "PS_DeferredTextured")
        # The dot-named listing is counted (os.listdir, not glob); the non-matching one is not.
        self.assertEqual(dict(shaders), {5: 1, 6: 1})
        self.assertEqual(dict(alpha), {(5, "byte 0 x 1/255", "straight"): 1, (6, "byte 0 x 1/255", "straight"): 1})
        self.assertEqual(dict(mrt4), {(5, 3, "vv-- compr"): 1, (6, 15, "vvvv compr"): 1})

    def test_cli_prints_counts_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = write_case(tmp, "PS_DeferredTextured_secretname", [15, 15, 15, 15, 3],
                                EXPORT + ["exp mrt4, v8, v8, off, off compr"])
            p = subprocess.run([sys.executable, "-I", TOOL, disasm], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("secretname", p.stdout)
        self.assertIn("listings 1: 5 targets:1", p.stdout)
        self.assertIn("mrt4, 5 targets, OSG1 mask 0x3: vv-- compr 1", p.stdout)

    def test_missing_directory_is_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing")
            p = subprocess.run([sys.executable, "-I", TOOL, missing], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn(missing, p.stderr)
        self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
