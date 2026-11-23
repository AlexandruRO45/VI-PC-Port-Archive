"""SHADER-1265: the named ds_* paths that skipped lds_access's out-of-range guard.

ds_add_f32 / ds_add_rtn_f32, ds_cmpst_f32 and ds_read/write_addtid_b32 used to call lds_ptr directly, so with an
lds_bound set (SHADER-309: the program reaches past its RSRC2 allocation) an out-of-range access wrapped onto LDS[0]
instead of reading 0 and dropping the write. Each case below uses LDS_SIZE=1 (512 B, 128 dwords) and addresses byte
512, one dword past the allocation.

    python -m unittest discover -s tests/semantic -p test_lds_bounds_1265.py
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_lds_atomics_cas import OUT, run_cs  # noqa: E402  (puts tools/ on sys.path)

import rdna2spirv  # noqa: E402

INIT = 0x3F800000          # 1.0
ONE = 0x3F800000
TWO = 0x40000000
ANSWER = 0xDEAD


def seed(addr_reg=0, value=INIT, extra=()):
    # The 0xfc00 + offset:1024 read is what makes the program reach past its RSRC2 allocation, so the translator sets
    # lds_bound = 128 for LDS_SIZE=1; without it there is nothing to check against and lds_bound stays None.
    return ["v_mov_b32_e32 v0, 0", f"v_mov_b32_e32 v1, {value:#x}", "ds_write_b32 v0, v1",
            "s_waitcnt lgkmcnt(0)", "v_mov_b32_e32 v8, 0xfc00", "ds_read_b32 v9, v8 offset:1024",
            "s_waitcnt lgkmcnt(0)", *extra]


def read_back():
    return ["s_waitcnt lgkmcnt(0)", "ds_read_b32 v4, v0", "s_waitcnt lgkmcnt(0)"]


def store_out(value_reg, offset=0):
    return [f"v_mov_b32_e32 v6, {OUT + offset:#x}", f"global_store_dword v6, {value_reg}, s[0:1]"]


class LdsBoundsNames(unittest.TestCase):
    def test_add_f32_drops_the_update(self):
        body = seed() + ["v_mov_b32_e32 v2, 512", f"v_mov_b32_e32 v3, {ONE:#x}", "ds_add_f32 v2, v3"] \
            + read_back() + store_out("v4")
        mem = run_cs(body)
        self.assertEqual(rdna2spirv.translate.last_gen.lds_bound, 128)
        self.assertEqual(mem.get(OUT), INIT)

    def test_add_rtn_f32_returns_zero_and_drops_the_update(self):
        body = seed() + ["v_mov_b32_e32 v2, 512", f"v_mov_b32_e32 v3, {ONE:#x}", "ds_add_rtn_f32 v5, v2, v3"] \
            + store_out("v5") + read_back() + store_out("v4", 4)
        mem = run_cs(body)
        self.assertEqual(rdna2spirv.translate.last_gen.lds_bound, 128)
        self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (0, INIT))

    def test_cmpst_f32_drops_the_exchange(self):
        body = seed() + ["v_mov_b32_e32 v2, 512", f"v_mov_b32_e32 v3, {INIT:#x}",
                         f"v_mov_b32_e32 v4, {TWO:#x}", "ds_cmpst_f32 v2, v3, v4"] \
            + read_back() + store_out("v4")
        mem = run_cs(body)
        self.assertEqual(rdna2spirv.translate.last_gen.lds_bound, 128)
        self.assertEqual(mem.get(OUT), INIT)

    def test_cmpst_rtn_f32_returns_zero_and_drops_the_exchange(self):
        body = seed() + ["v_mov_b32_e32 v2, 512", f"v_mov_b32_e32 v3, {INIT:#x}",
                         f"v_mov_b32_e32 v4, {TWO:#x}", "ds_cmpst_rtn_f32 v5, v2, v3, v4"] \
            + store_out("v5") + read_back() + store_out("v4", 4)
        mem = run_cs(body)
        self.assertEqual(rdna2spirv.translate.last_gen.lds_bound, 128)
        self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (0, INIT))

    def test_read_addtid_past_the_allocation_reads_zero(self):
        body = seed(value=ANSWER) + ["s_mov_b32 m0, 512", "ds_read_addtid_b32 v5", "s_waitcnt lgkmcnt(0)"] \
            + store_out("v5")
        mem = run_cs(body)
        self.assertEqual(rdna2spirv.translate.last_gen.lds_bound, 128)
        self.assertEqual(mem.get(OUT), 0)

    def test_in_range_add_and_rtn_still_update(self):
        # The same programs at dword 0, so the guarded path's in-range branch runs: 1.0 + 1.0 = 2.0 exactly.
        body = seed() + ["v_mov_b32_e32 v2, 0", f"v_mov_b32_e32 v3, {ONE:#x}", "ds_add_f32 v2, v3"] \
            + read_back() + store_out("v4")
        mem = run_cs(body)
        self.assertEqual(mem.get(OUT), TWO)

        body = seed() + ["v_mov_b32_e32 v2, 0", f"v_mov_b32_e32 v3, {ONE:#x}", "ds_add_rtn_f32 v5, v2, v3"] \
            + store_out("v5") + read_back() + store_out("v4", 4)
        mem = run_cs(body)
        self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (INIT, TWO))

    def test_in_range_cmpst_and_rtn_still_exchange(self):
        body = seed() + ["v_mov_b32_e32 v2, 0", f"v_mov_b32_e32 v3, {INIT:#x}",
                         f"v_mov_b32_e32 v4, {TWO:#x}", "ds_cmpst_f32 v2, v3, v4"] + read_back() + store_out("v4")
        mem = run_cs(body)
        self.assertEqual(mem.get(OUT), TWO)

        body = seed() + ["v_mov_b32_e32 v2, 0", f"v_mov_b32_e32 v3, {INIT:#x}",
                         f"v_mov_b32_e32 v4, {TWO:#x}", "ds_cmpst_rtn_f32 v5, v2, v3, v4"] \
            + store_out("v5") + read_back() + store_out("v4", 4)
        mem = run_cs(body)
        self.assertEqual((mem.get(OUT), mem.get(OUT + 4)), (INIT, TWO))

    def test_in_range_addtid_still_reads_and_writes(self):
        body = seed(value=ANSWER) + ["s_mov_b32 m0, 0", "ds_read_addtid_b32 v5", "s_waitcnt lgkmcnt(0)"] \
            + store_out("v5")
        mem = run_cs(body)
        self.assertEqual(mem.get(OUT), ANSWER)

        body = seed() + ["s_mov_b32 m0, 0", f"v_mov_b32_e32 v5, {ANSWER:#x}", "ds_write_addtid_b32 v5"] \
            + read_back() + store_out("v4")
        mem = run_cs(body)
        self.assertEqual(mem.get(OUT), ANSWER)

    def test_write_addtid_past_the_allocation_is_dropped(self):
        body = seed() + ["s_mov_b32 m0, 512", f"v_mov_b32_e32 v5, {ANSWER:#x}", "ds_write_addtid_b32 v5"] \
            + read_back() + store_out("v4")
        mem = run_cs(body)
        self.assertEqual(rdna2spirv.translate.last_gen.lds_bound, 128)
        self.assertEqual(mem.get(OUT), INIT)


if __name__ == "__main__":
    unittest.main()
