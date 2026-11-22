"""test_ballot_votes.py: whole-wave mask reads (count, first lane, equality, any) through tools/rdna2spirv.py."""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
from support import translate_lines  # noqa: E402

LANES = 32
STORE = ["v_lshlrev_b32_e32 v3, 2, v0", "buffer_store_dword v2, v3, s[0:3], 0 offen", "s_endpgm"]


def run(lines, lanes=LANES):
    fssm = bytearray(0x148)
    struct.pack_into("<4I", fssm, 0x138, 0x8088, lanes, 1, 1)
    spv = translate_lines("cs_ballot_votes", lines + STORE, fssm=bytes(fssm))
    mem = si.run_workgroup(spv, lanes, 32,
                           inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                           push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={})
    return [mem.get(4 * i) for i in range(lanes)]


def scalar(lines):
    return run(lines + ["v_mov_b32_e32 v2, s5"])


class BallotVoteTests(unittest.TestCase):
    def test_count(self):
        for n in (0, 1, 10, 31, 32):
            with self.subTest(n=n):
                self.assertEqual(scalar([f"v_cmp_gt_u32_e32 vcc_lo, {n}, v0", "s_bcnt1_i32_b32 s5, vcc_lo"]),
                                 [n] * LANES)

    def test_first_set_lane(self):
        for n in (0, 6, 30):
            with self.subTest(n=n):
                self.assertEqual(scalar([f"v_cmp_lt_u32_e32 vcc_lo, {n}, v0", "s_ff1_i32_b32 s5, vcc_lo"]),
                                 [n + 1] * LANES)

    def test_first_set_lane_of_an_empty_mask(self):
        self.assertEqual(scalar(["v_cmp_eq_u32_e32 vcc_lo, 99, v0", "s_ff1_i32_b32 s5, vcc_lo"]),
                         [0xFFFFFFFF] * LANES)

    def test_mask_equals_exec(self):
        for n, want in ((32, 1), (3, 2), (0, 2)):
            with self.subTest(n=n):
                self.assertEqual(scalar([f"v_cmp_gt_u32_e32 vcc_lo, {n}, v0", "s_cmp_eq_u32 vcc_lo, exec_lo",
                                         "s_cselect_b32 s5, 1, 2"]), [want] * LANES)

    def test_any_lane(self):
        for k, want in ((7, 1), (99, 2)):
            with self.subTest(k=k):
                self.assertEqual(scalar([f"v_cmp_eq_u32_e32 vcc_lo, {k}, v0", "s_cmp_lg_u32 vcc_lo, 0",
                                         "s_cselect_b32 s5, 1, 2"]), [want] * LANES)

    def test_count_sees_only_active_lanes(self):
        got = run(["v_mov_b32_e32 v2, 0xdead", "v_cmpx_gt_u32_e32 16, v0", "v_cmp_gt_u32_e32 vcc_lo, 20, v0",
                   "s_bcnt1_i32_b32 s5, vcc_lo", "v_mov_b32_e32 v2, s5"])
        self.assertEqual(got[:16], [16] * 16)
        self.assertEqual(got[16:], [None] * 16)


if __name__ == "__main__":
    unittest.main()
