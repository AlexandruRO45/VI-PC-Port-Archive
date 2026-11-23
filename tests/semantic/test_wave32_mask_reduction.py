"""A whole-mask read (s_bcnt1, v_mbcnt) in a wave32 compute shader costs one subgroup reduction.

The lane of a wave32 compute invocation is LocalInvocationIndex mod 32, so the mask has no bits above 31 and
one OpGroupNonUniformBitwiseOr gives all of them. Until this change each read did a second reduction for lanes
32-63, which was always 0. Checked by value on a 64-thread group (two waves) with run_compute_workgroup.

    python -I tests/semantic/test_wave32_mask_reduction.py -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_loop_ballot_barrier import VSHARP, _translate_cs, run_compute_workgroup  # noqa: E402

PROGRAM = [
    "v_and_b32_e32 v5, 31, v0",
    "v_cmp_gt_u32_e32 vcc_lo, 10, v5",          # lanes 0-9 of each wave
    "s_bcnt1_i32_b32 s4, vcc_lo",               # 10
    "v_mbcnt_lo_u32_b32 v1, vcc_lo, 0",         # set lanes below this one: min(lane, 10)
    "v_mov_b32_e32 v2, s4",
    "v_lshlrev_b32_e32 v3, 2, v0",
    "buffer_store_dword v1, v3, s[0:3], 0 offen",
    "v_add_nc_u32_e32 v3, 0x100, v3",
    "buffer_store_dword v2, v3, s[0:3], 0 offen",
    "s_endpgm",
]


class Wave32MaskReduction(unittest.TestCase):
    def test_one_reduction_per_mask_read_and_exact_counts(self):
        spv, gen = _translate_cs("cs_w32_mask_reads", PROGRAM, threads=64)
        self.assertFalse(gen.wave64)
        self.assertEqual(spv.count("OpGroupNonUniformBitwiseOr"), 2, "one per read: s_bcnt1 and v_mbcnt")
        mem = run_compute_workgroup(spv, 64, 32, push_constants=VSHARP)
        self.assertEqual([mem.get(4 * t) for t in range(64)], [min(t % 32, 10) for t in range(64)])
        self.assertEqual([mem.get(0x100 + 4 * t) for t in range(64)], [10] * 64)


if __name__ == "__main__":
    unittest.main()
