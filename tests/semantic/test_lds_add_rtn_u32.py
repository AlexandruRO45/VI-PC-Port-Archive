from __future__ import annotations

"""ds_add_rtn_u32 and ds_add_u32 through the translator.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions":
ds_add_rtn_u32 vdst, vaddr, vdata [offset:byte_offset]
Computes M = (M + DATA) mod 2^32 on the 32-bit dword at (vaddr + offset) / 4 in LDS using
unsigned 32-bit modular addition, returning the previous value of M in vdst.

ds_add_u32 vaddr, vdata [offset:byte_offset]
Computes M = (M + DATA) mod 2^32 without returning the old value to a VGPR.

Each case translates a compute shader with tools/rdna2spirv.py (support.translate_lines)
and executes it in tests/semantic/spirv_mini_interp.py (see lds_run.py).
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402
import support  # noqa: E402

M32 = L.M32


class TestDsAddRtnU32HandTable(unittest.TestCase):
    #       Initial M   Data D      Expected Old (M)  Expected New (M + D mod 2^32)
    TABLE = (
        (0,          0,          0,                0),
        (0,          1,          0,                1),
        (1,          0,          1,                1),
        (1,          2,          1,                3),
        (100,        250,        100,              350),
        (0xFFFFFFFF, 1,          0xFFFFFFFF,       0),           # Modular overflow to 0
        (0xFFFFFFFF, 2,          0xFFFFFFFF,       1),
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,       0xFFFFFFFE),
        (0x7FFFFFFF, 1,          0x7FFFFFFF,       0x80000000),  # Signed overflow boundary
        (0x80000000, 0x80000000, 0x80000000,       0),           # Two's complement wraparound
        (0x12345678, 0x9ABCDEF0, 0x12345678,       0xACF13568),
        (0xAAAAAAAA, 0x55555555, 0xAAAAAAAA,       0xFFFFFFFF),
        (0xFFFFFFFF, 0x00000001, 0xFFFFFFFF,       0x00000000),
        (0x80000001, 0x7FFFFFFF, 0x80000001,       0x00000000),
        (5,          0xFFFFFFFF, 5,                4),           # Adding -1 in unsigned arithmetic
        (0,          0xFFFFFFFF, 0,                0xFFFFFFFF),
    )

    def test_single_atomic_add_and_return(self):
        for m, d, want_old, want_new in self.TABLE:
            with self.subTest(op="ds_add_rtn_u32", m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_add_rtn_u32", 0x40, 0, (d,))], {0x40: m})
                # vdst must receive previous value m; memory must hold (m + d) mod 2^32
                self.assertEqual(rtns, [want_old])
                self.assertEqual(fin, {0x40: want_new})

            with self.subTest(op="ds_add_u32", m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_add_u32", 0x40, 0, (d,))], {0x40: m})
                # non-returning form leaves preset 0xDEAD in destination; memory updated
                self.assertEqual(rtns, [0xDEAD])
                self.assertEqual(fin, {0x40: want_new})

    def test_offset_addressing(self):
        # Base 0x20 + offset 0x24 = effective LDS address 0x44
        rtns, fin = L.run_chain([("ds_add_rtn_u32", 0x20, 0x24, (15,))], {0x44: 40, 0x40: 100})
        self.assertEqual(rtns, [40])
        self.assertEqual(fin, {0x44: 55})
        # Address 0x40 should be untouched
        self.assertEqual(fin.get(0x40), None)

    def test_offset_large(self):
        # Base 0x100 + offset 0x200 = effective address 0x300
        rtns, fin = L.run_chain([("ds_add_rtn_u32", 0x100, 0x200, (0x1111,))], {0x300: 0x2222})
        self.assertEqual(rtns, [0x2222])
        self.assertEqual(fin, {0x300: 0x3333})

    def test_running_accumulator_chain(self):
        # Successive additions on a single dword starting at 100:
        # returns previous sum; memory accumulates total modulo 2^32.
        increments = [10, 25, 0xFFFFFFFF, 1, 0x80000000, 0x80000000]
        ops = [("ds_add_rtn_u32", 0x18, 0, (d,)) for d in increments]
        rtns, fin = L.run_chain(ops, {0x18: 100})
        expected_rtns = [100, 110, 135, 134, 135, 0x80000087]
        self.assertEqual(rtns, expected_rtns)
        self.assertEqual(fin, {0x18: 135})

    def test_multiple_independent_dwords(self):
        ops = [
            ("ds_add_rtn_u32", 0x00, 0, (5,)),
            ("ds_add_rtn_u32", 0x04, 0, (10,)),
            ("ds_add_rtn_u32", 0x08, 0, (100,)),
            ("ds_add_rtn_u32", 0x00, 0, (15,)),
            ("ds_add_rtn_u32", 0x04, 0, (20,)),
            ("ds_add_rtn_u32", 0x08, 0, (200,)),
        ]
        init = {0x00: 1, 0x04: 2, 0x08: 3}
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [1, 2, 3, 6, 12, 103])
        self.assertEqual(fin, {0x00: 21, 0x04: 32, 0x08: 303})

    def test_algebraic_modular_inverse(self):
        # Adding (2^32 - x) to x yields 0 mod 2^32
        for x in (1, 42, 0x12345678, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE):
            inv = (0x100000000 - x) & M32
            with self.subTest(x=hex(x), inv=hex(inv)):
                rtns, fin = L.run_chain([("ds_add_rtn_u32", 0x30, 0, (inv,))], {0x30: x})
                self.assertEqual(rtns, [x])
                self.assertEqual(fin, {0x30: 0})


class TestDsAddRtnU32MultiLane(unittest.TestCase):
    def test_32_lanes_parallel_add(self):
        # Each lane i uses its own LDS dword at i * 4.
        # Initial value in LDS is 0x1000 + i.
        # Add operand is (i + 1) * 10.
        # v5 receives old value (0x1000 + i); final value read into v6 is 0x1000 + i + (i + 1) * 10.
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",                   # v1 = i * 4
            "v_add_nc_u32_e32 v2, 0x1000, v0",              # v2 = 0x1000 + i
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v3, 1, v0",                   # v3 = i + 1
            "v_mul_lo_u32 v4, 10, v3",                      # v4 = (i + 1) * 10
            "ds_add_rtn_u32 v5, v1, v4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v5", 0) + L.store("v6", 1)

        mem = L.run(body, lanes=32)
        for i in range(32):
            returned_old = L.out(mem, i, 0)
            final_val = L.out(mem, i, 1)
            expected_old = 0x1000 + i
            expected_new = (0x1000 + i + (i + 1) * 10) & M32
            self.assertEqual(returned_old, expected_old, f"lane {i} return value")
            self.assertEqual(final_val, expected_new, f"lane {i} final value")


class TestDsAddRtnU32Chaining(unittest.TestCase):
    def test_deterministic_multi_step_chain(self):
        init = {0x10: 100, 0x20: 200, 0x30: 300}
        ops = [
            ("ds_add_rtn_u32", 0x10, 0, (15,)),
            ("ds_add_u32",     0x20, 0, (25,)),
            ("ds_add_rtn_u32", 0x30, 0, (35,)),
            ("ds_add_rtn_u32", 0x10, 0, (10,)),
            ("ds_add_u32",     0x20, 0, (5,)),
            ("ds_add_rtn_u32", 0x30, 0, (65,)),
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [100, 0xDEAD, 300, 115, 0xDEAD, 335])
        self.assertEqual(fin, {0x10: 125, 0x20: 230, 0x30: 400})


class TestDsAddRtnU32RegisterAliasing(unittest.TestCase):
    def test_alias_vdst_with_vaddr(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 50",
            "ds_add_rtn_u32 v1, v1, v2",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v1", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 100)

    def test_alias_vdst_with_vdata(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 50",
            "ds_add_rtn_u32 v2, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v2", 0) + L.store("v3", 1)
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 100)
        self.assertEqual(mem.get(4), 150)


class TestDsAddRtnU32Spirv(unittest.TestCase):
    def test_translator_emits_atomic_iadd_for_ds_add_rtn(self):
        listing = [
            "ds_add_rtn_u32 v5, v1, v4",
            "s_waitcnt lgkmcnt(0)",
        ]
        spvasm = support.translate_lines("cs_ds_add_rtn", listing + ["s_endpgm"], fssm=L.fssm())
        self.assertIn("OpAtomicIAdd", spvasm)

    def test_translator_emits_atomic_iadd_for_ds_add(self):
        listing = [
            "ds_add_u32 v1, v4",
            "s_waitcnt lgkmcnt(0)",
        ]
        spvasm = support.translate_lines("cs_ds_add", listing + ["s_endpgm"], fssm=L.fssm())
        self.assertIn("OpAtomicIAdd", spvasm)


class TestDsAddRtnU32NegativeRejection(unittest.TestCase):
    def test_outside_compute_rejection(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_test_outside_compute", ["ds_add_rtn_u32 v0, v1, v2", "s_endpgm"])

    def test_invalid_syntax_rejection(self):
        with self.assertRaises((rdna2spirv.Unsupported, IndexError, ValueError)):
            support.translate_lines("cs_test_bad_syntax", ["ds_add_rtn_u32 v1, v2", "s_endpgm"], fssm=L.fssm())


if __name__ == "__main__":
    unittest.main()
