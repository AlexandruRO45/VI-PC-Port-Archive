from __future__ import annotations

"""ds_max_rtn_u32 through the translator.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions":
ds_max_rtn_u32 vdst, vaddr, vdata [offset:byte_offset]
Computes M = max(M, DATA) on the 32-bit dword at (vaddr + offset) / 4 in LDS using
unsigned 32-bit comparison, returning the previous value of M in vdst.

Each case translates a compute shader with tools/rdna2spirv.py (support.translate_lines)
and executes it in tests/semantic/spirv_mini_interp.py (see lds_run.py).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402
import rdna2spirv  # noqa: E402
import support  # noqa: E402

OP = "ds_max_rtn_u32"
M32 = L.M32


class TestDsMaxRtnU32HandTable(unittest.TestCase):
    #       Initial M   Data D      Expected Max (unsigned)
    TABLE = (
        (0,          0,          0),
        (0,          100,        100),
        (100,        0,          100),
        (100,        100,        100),
        (50,         100,        100),
        (100,        50,         100),
        (1,          2,          2),
        (2,          1,          2),
        # Unsigned vs signed: 0x80000000 is 2^31 (unsigned) > 0x7FFFFFFF (2^31 - 1).
        # Signed comparison would yield 0x7FFFFFFF; unsigned must yield 0x80000000.
        (0x7FFFFFFF, 0x80000000, 0x80000000),
        (0x80000000, 0x7FFFFFFF, 0x80000000),
        (0x80000000, 0x80000001, 0x80000001),
        (0x80000001, 0x80000000, 0x80000001),
        (0,          0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0,          0xFFFFFFFF),
        (0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF),
        (0x80000000, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0x80000000, 0xFFFFFFFF),
        (0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
    )

    def test_single_atomic_max_and_return(self):
        for m, d, want_max in self.TABLE:
            with self.subTest(m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([(OP, 0x40, 0, (d,))], {0x40: m})
                # vdst must receive previous value m; memory must hold max(m, d)
                self.assertEqual(rtns, [m])
                self.assertEqual(fin, {0x40: want_max})

    def test_offset_addressing(self):
        # Base 0x20 + offset 0x24 = effective LDS address 0x44
        rtns, fin = L.run_chain([(OP, 0x20, 0x24, (90,))], {0x44: 40, 0x40: 100})
        self.assertEqual(rtns, [40])
        self.assertEqual(fin, {0x44: 90})
        # Address 0x40 should be untouched
        self.assertIsNone(fin.get(0x40))

    def test_offset_large(self):
        # Base 0x100 + offset 0x200 = 0x300
        rtns, fin = L.run_chain([(OP, 0x100, 0x200, (0xCAFE,))], {0x300: 0x1234})
        self.assertEqual(rtns, [0x1234])
        self.assertEqual(fin, {0x300: 0xCAFE})

    def test_running_reduction_chain(self):
        # Sequence of updates on a single dword starting at 50:
        # returns previous maximum; memory tracks running maximum.
        updates = [20, 80, 40, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF, 10]
        ops = [(OP, 0x10, 0, (d,)) for d in updates]
        rtns, fin = L.run_chain(ops, {0x10: 50})
        expected_rtns = [50, 50, 80, 80, 0x80000000, 0x80000000, 0xFFFFFFFF]
        self.assertEqual(rtns, expected_rtns)
        self.assertEqual(fin, {0x10: 0xFFFFFFFF})

    def test_multiple_independent_dwords(self):
        ops = [
            (OP, 0x00, 0, (30,)),
            (OP, 0x04, 0, (5,)),
            (OP, 0x08, 0, (0x90000000,)),
            (OP, 0x00, 0, (25,)),
            (OP, 0x04, 0, (15,)),
            (OP, 0x08, 0, (0x70000000,)),
        ]
        init = {0x00: 10, 0x04: 10, 0x08: 0x80000000}
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [10, 10, 0x80000000, 30, 10, 0x90000000])
        self.assertEqual(fin, {0x00: 30, 0x04: 15, 0x08: 0x90000000})


class TestDsMaxRtnU32RegisterAliasing(unittest.TestCase):
    """Test register reuse where destination overlaps with address or data."""

    def test_alias_vdst_with_vaddr(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 50",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 100",
            "ds_max_rtn_u32 v1, v1, v2",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v1", 0)
        mem = L.run(body, lanes=1)
        # v1 receives old value 50
        self.assertEqual(mem.get(0), 50)

    def test_alias_vdst_with_vdata(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 50",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 100",
            "ds_max_rtn_u32 v2, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v2", 0) + L.store("v3", 1)
        mem = L.run(body, lanes=1)
        # v2 receives old value 50, memory becomes 100
        self.assertEqual(mem.get(0), 50)
        self.assertEqual(mem.get(4), 100)


class TestDsMaxRtnU32MultiLane(unittest.TestCase):
    def test_32_lanes_parallel_max(self):
        # Each lane i uses its own LDS dword at i * 4.
        # Initial value in LDS is 0x1000 + i.
        # Max operand is 0x1000 + (31 - i).
        # v5 receives old value (0x1000 + i); final value read into v6 is 0x1000 + max(i, 31 - i).
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",                   # v1 = i * 4
            "v_add_nc_u32_e32 v2, 0x1000, v0",              # v2 = 0x1000 + i
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_sub_nc_u32_e32 v3, 31, v0",                  # v3 = 31 - i
            "v_add_nc_u32_e32 v4, 0x1000, v3",              # v4 = 0x1000 + (31 - i)
            "ds_max_rtn_u32 v5, v1, v4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v5", 0) + L.store("v6", 1)

        mem = L.run(body, lanes=32)
        for i in range(32):
            returned_old = L.out(mem, i, 0)
            final_val = L.out(mem, i, 1)
            self.assertEqual(returned_old, 0x1000 + i, f"lane {i} return value")
            self.assertEqual(final_val, 0x1000 + max(i, 31 - i), f"lane {i} final value")


class TestDsMaxRtnU32Spirv(unittest.TestCase):
    """Verify SPIR-V lowering emitted by rdna2spirv."""

    def test_translator_emits_op_atomic_umax(self):
        listing = [
            "ds_max_rtn_u32 v0, v1, v2",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("cs_max_spirv", listing, fssm=L.fssm())
        self.assertIn("OpAtomicUMax", spvasm)
        self.assertIn("OpCapability Shader", spvasm)


class TestDsMaxRtnU32Rejections(unittest.TestCase):
    """Test translator error rejection on invalid stages and malformed operands."""

    def test_rejection_outside_compute(self):
        listing = [
            "ds_max_rtn_u32 v0, v1, v2",
            "s_endpgm",
        ]
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_max_invalid_stage", listing)

    def test_rejection_malformed_operands(self):
        bad_cases = [
            ["ds_max_rtn_u32 v0, v1"],
            ["ds_max_rtn_u32 v0"],
        ]
        for idx, lines in enumerate(bad_cases):
            with self.subTest(idx=idx):
                with self.assertRaises(Exception):
                    support.translate_lines(f"cs_max_bad_{idx}", lines + ["s_endpgm"], fssm=L.fssm())


class TestDsMaxRtnU32MutationGuards(unittest.TestCase):
    """Verify that mutations to comparison or return semantics are caught."""

    def test_unsigned_vs_signed_max_mutation(self):
        # 0x7FFFFFFF (2^31 - 1) vs 0x80000000 (2^31)
        # Unsigned max: 0x80000000. Signed max would incorrectly produce 0x7FFFFFFF.
        rtns, fin = L.run_chain([(OP, 0x40, 0, (0x80000000,))], {0x40: 0x7FFFFFFF})
        self.assertEqual(rtns, [0x7FFFFFFF])
        self.assertEqual(fin, {0x40: 0x80000000})

        signed_mutant = 0x7FFFFFFF
        self.assertNotEqual(fin[0x40], signed_mutant)

    def test_returning_old_value_mutation(self):
        # Confirm returned value is previous value, not the newly computed maximum
        rtns, fin = L.run_chain([(OP, 0x40, 0, (100,))], {0x40: 25})
        self.assertEqual(rtns, [25])
        self.assertEqual(fin, {0x40: 100})
        # If instruction erroneously returned new value, it would equal 100
        self.assertNotEqual(rtns[0], 100)


if __name__ == "__main__":
    unittest.main()
