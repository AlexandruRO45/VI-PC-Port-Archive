"""SHADER-715: an LDS access at a dynamic address past the RSRC2 allocation reads 0 and drops its write.

RDNA2 LDS does not wrap: an access outside the workgroup's allocation reads 0 and its write is dropped (RDNA2 ISA
3.6.1, 3.6.6; measured on gfx1036 in #3817). SHADER-309 checked that only for constant `offset:` values past the
allocation; a dynamic VGPR address past it wrapped onto dword 0 through the power-of-two array mask. Every compute
shader's array is now the allocation plus one sink dword: an out-of-range index goes to the sink and what the access
returns is 0, without a branch. When the sink does not fit, the branch check of SHADER-309 is used instead.

Hand-written wave32 compute listings (512 B allocated, dword 0 a sentinel, the address in a VGPR), translated and run
in spirv_mini_interp; no GPU or game files.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
from test_cs_shared_memory_limit import bounds_checks, lds_index_ops, out, run_one, sink_checks  # noqa: E402

rdna2spirv = support.rdna2spirv

SENTINEL = 0x5E112233
LAST = 0x7A6B5C4D          # dword 127, the last one of the 512-byte allocation
SEED = ["v_mov_b32_e32 v1, 0", f"v_mov_b32_e32 v2, {SENTINEL:#x}", "ds_write_b32 v1, v2",
        "v_mov_b32_e32 v8, 0x1fc", f"v_mov_b32_e32 v9, {LAST:#x}", "ds_write_b32 v8, v9",
        "v_mov_b32_e32 v3, 0x200",            # byte 512: the first byte past the allocation
        "s_waitcnt lgkmcnt(0)"]
CHECK = ["s_waitcnt lgkmcnt(0)", "ds_read_b32 v20, v1", "ds_read_b32 v21, v8", "s_waitcnt lgkmcnt(0)"]


def dead(*regs):
    return [f"v_mov_b32_e32 v{r}, 0xdead" for r in regs]


def words(mem, n):
    return [mem.get(4 * k) for k in range(n)]


class DynamicOutOfRangeLdsTests(unittest.TestCase):
    def test_reads_of_every_width_return_zero(self):
        spv, mem = run_one(SEED + dead(10, 11, 12, 13, 14, 15, 16, 17, 18, 19)
                           + ["ds_read_b32 v10, v3", "ds_read_u8 v11, v3", "ds_read_i8 v12, v3",
                              "ds_read_u16 v13, v3", "ds_read_i16 v14, v3", "ds_read_b128 v[15:18], v3",
                              "ds_read2_b32 v[4:5], v3 offset1:1", "s_waitcnt lgkmcnt(0)"]
                           + out(10, 11, 12, 13, 14, 15, 16, 17, 18, 4, 5), 1)
        self.assertEqual(words(mem, 11), [0] * 11)
        self.assertEqual(bounds_checks(spv), 0)
        self.assertEqual(lds_index_ops(spv), set())
        self.assertGreater(sink_checks(spv), 0)

    def test_wide_read_straddling_the_end_keeps_the_in_range_dword(self):
        # ds_read_b64 at byte 508: dword 127, then 0 for the dword past the allocation
        _, mem = run_one(SEED + dead(10, 11) + ["ds_read_b64 v[10:11], v8", "s_waitcnt lgkmcnt(0)"] + out(10, 11), 1)
        self.assertEqual(words(mem, 2), [LAST, 0])

    def test_writes_of_every_width_are_dropped_not_wrapped(self):
        # Every write past the allocation would land on dword 0 (or 1..3) if the index wrapped
        _, mem = run_one(SEED + ["v_mov_b32_e32 v4, 0xc3d4", "v_mov_b32_e32 v5, 0xa1b2c3d4",
                                 "v_mov_b32_e32 v6, 0x0f0e0d0c", "v_mov_b32_e32 v7, 0x0b0a0908",
                                 "ds_write_b32 v3, v5", "ds_write_b8 v3, v4", "ds_write_b16 v3, v4",
                                 "ds_write_b16 v3, v4 offset:1", "ds_write_b64 v3, v[4:5]",
                                 "ds_write_b128 v3, v[4:7]", "ds_write2_b32 v3, v5, v6 offset1:1",
                                 "ds_write_b16_d16_hi v3, v5"] + CHECK
                         + dead(10, 11, 12, 13) + ["ds_read_b128 v[10:13], v1", "s_waitcnt lgkmcnt(0)"]
                         + out(20, 21, 11, 12, 13), 1)
        self.assertEqual(words(mem, 5), [SENTINEL, LAST, 0, 0, 0])

    def test_wide_write_straddling_the_end_keeps_the_in_range_dword(self):
        _, mem = run_one(SEED + ["v_mov_b32_e32 v4, 0x1111", "v_mov_b32_e32 v5, 0x2222",
                                 "ds_write_b64 v8, v[4:5]"] + CHECK + out(20, 21), 1)
        self.assertEqual(words(mem, 2), [SENTINEL, 0x1111])

    def test_atomics_return_zero_and_change_nothing(self):
        # Native atomics, the compare-exchange loops (ds_add_f32, ds_inc/ds_dec/ds_rsub, ds_min/max_f32), cmpst,
        # wrxchg and mskor: each returns 0 past the allocation and leaves the dwords it would wrap to as they were
        ops = ["ds_add_rtn_u32 v10, v3, v4", "ds_or_rtn_b32 v11, v3, v4", "ds_wrxchg_rtn_b32 v12, v3, v4",
               "ds_add_rtn_f32 v13, v3, v4", "ds_inc_rtn_u32 v14, v3, v4", "ds_dec_rtn_u32 v15, v3, v4",
               "ds_rsub_rtn_u32 v16, v3, v4", "ds_max_rtn_f32 v17, v3, v4", "ds_cmpst_rtn_b32 v18, v3, v2, v4",
               "ds_mskor_rtn_b32 v19, v3, v4, v4"]
        _, mem = run_one(SEED + ["v_mov_b32_e32 v4, 0x40400000"] + dead(*range(10, 20)) + ops
                         + ["ds_add_u32 v3, v4", "ds_add_f32 v3, v4", "ds_inc_u32 v3, v4"] + CHECK
                         + out(*range(10, 20), 20, 21), 1)
        self.assertEqual(words(mem, 12), [0] * 10 + [SENTINEL, LAST])

    def test_addtid_past_the_allocation(self):
        # ds_read/write_addtid_b32 address M0[15:0] + offset + 4 * lane: all 32 lanes past the allocation
        _, mem = run_one(SEED + ["s_mov_b32 m0, 0x200", "v_mov_b32_e32 v4, 0xbad", "ds_write_addtid_b32 v4",
                                 "v_mov_b32_e32 v10, 0xdead", "ds_read_addtid_b32 v10"] + CHECK
                         + out(10, 20, 21), 1)
        self.assertEqual(words(mem, 3), [0, SENTINEL, LAST])

    def test_in_range_dynamic_accesses_are_unchanged(self):
        # The last dword of the allocation is written and read through a dynamic address as before
        _, mem = run_one(SEED + ["v_mov_b32_e32 v4, 0x600df00d", "ds_write_b32 v8, v4", "s_waitcnt lgkmcnt(0)",
                                 "v_mov_b32_e32 v10, 0xdead", "ds_read_b32 v10, v8", "ds_read_u16 v11, v8 offset:2",
                                 "s_waitcnt lgkmcnt(0)"] + out(10, 11), 1)
        self.assertEqual(words(mem, 2), [0x600DF00D, 0x600D])

    def test_without_room_for_the_sink_the_branch_check_is_used(self):
        # 64 KiB allocated, native: no room for the sink dword under the 64 KiB cap. A dynamic access at byte
        # 65,536 is still checked (a branch, SHADER-309) and does not wrap onto dword 0.
        _, mem = run_one(["v_mov_b32_e32 v1, 0", f"v_mov_b32_e32 v2, {SENTINEL:#x}", "ds_write_b32 v1, v2",
                          "v_mov_b32_e32 v3, 0x10000", "v_mov_b32_e32 v4, 0xbad", "s_waitcnt lgkmcnt(0)",
                          "ds_write_b32 v3, v4", "v_mov_b32_e32 v10, 0xdead", "ds_read_b32 v10, v3",
                          "s_waitcnt lgkmcnt(0)", "ds_read_b32 v20, v1", "s_waitcnt lgkmcnt(0)"] + out(10, 20), 128)
        gen = rdna2spirv.translate.last_gen
        self.assertEqual((gen.lds_bound, gen.lds_sink), (16384, None))
        self.assertEqual(words(mem, 2), [0, SENTINEL])


if __name__ == "__main__":
    unittest.main()
