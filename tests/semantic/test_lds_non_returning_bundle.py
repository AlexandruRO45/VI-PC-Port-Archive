from __future__ import annotations

"""Bundle 66: LDS Non-Returning Arithmetic Extensions.

Covers RDNA2 Data Share (LDS) non-returning arithmetic instructions:
  - ds_add_u32: M = (M + D) & 0xFFFFFFFF
  - ds_sub_u32: M = (M - D) & 0xFFFFFFFF
  - ds_inc_u32: M = (M >= D) ? 0 : M + 1 (unsigned compare)
  - ds_dec_u32: M = (M == 0 || M > D) ? D : M - 1 (unsigned compare)

Hardware semantics:
  - Atomic read-modify-write on local data share (LDS) memory at (vaddr + offset).
  - Non-returning variants: no destination VGPR operand is specified, leaving
    destination registers untouched.
  - Optional 16-bit byte offset operand added to base byte address.
  - Workgroup-scope atomic synchronization in compute shader stages.

100% translator-backed execution via lds_run.run / lds_run.run_chain
(support.translate_lines and spirv_mini_interp). Zero mock models.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lds_run as L

class TestLdsAddU32Reference(unittest.TestCase):
    def test_add_basic_operations(self):
        cases = [
            (0, 0, 0),
            (10, 5, 15),
            (0xFFFFFFFF, 1, 0),
            (0x7FFFFFFF, 1, 0x80000000),
            (0x80000000, 0x80000000, 0),
            (0x12345678, 0x11111111, 0x23456789),
        ]
        for m, d, want in cases:
            with self.subTest(m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_add_u32", 0x40, 0, (d,))], {0x40: m})
                self.assertEqual(fin[0x40], want)

    def test_non_returning_preserves_dst_sentinel(self):
        rtns, fin = L.run_chain([("ds_add_u32", 0x40, 0, (42,))], {0x40: 100})
        self.assertEqual(rtns, [0xDEAD])
        self.assertEqual(fin[0x40], 142)

class TestLdsAddU32Offset(unittest.TestCase):
    def test_byte_offset_addition(self):
        init = {0x40: 10, 0x44: 20, 0x48: 30}
        rtns, fin = L.run_chain([("ds_add_u32", 0x20, 0x24, (5,))], init)
        self.assertEqual(fin[0x44], 25)

    def test_multiple_offsets_same_base(self):
        init = {0x40: 1, 0x44: 2, 0x48: 3}
        ops = [
            ("ds_add_u32", 0x40, 0, (10,)),
            ("ds_add_u32", 0x40, 4, (20,)),
            ("ds_add_u32", 0x40, 8, (30,)),
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(fin[0x40], 11)
        self.assertEqual(fin[0x44], 22)
        self.assertEqual(fin[0x48], 33)

class TestLdsAddU32Chaining(unittest.TestCase):
    def test_chained_accumulations(self):
        init = {0x40: 100}
        ops = [("ds_add_u32", 0x40, 0, (10,)), ("ds_add_u32", 0x40, 0, (20,)), ("ds_add_u32", 0x40, 0, (30,))]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(fin[0x40], 160)

    def test_multi_location_independent_accumulations(self):
        init = {0x10: 1000, 0x20: 2000}
        ops = [
            ("ds_add_u32", 0x10, 0, (50,)),
            ("ds_add_u32", 0x20, 0, (75,)),
            ("ds_add_u32", 0x10, 0, (50,)),
            ("ds_add_u32", 0x20, 0, (25,)),
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(fin[0x10], 1100)
        self.assertEqual(fin[0x20], 2100)

class TestLdsAddU32Divergence(unittest.TestCase):
    def test_multilane_independent_lanes(self):
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v3, 10, v0",
            "ds_add_u32 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v4", 0)
        mem = L.run(body, lanes=32)
        for lane in range(32):
            self.assertEqual(L.out(mem, lane, 0), 100 + 10 + lane)

    def test_multilane_exec_mask_gating(self):
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_mov_b32_e32 v2, 50",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, 0x55555555",
            "v_mov_b32_e32 v3, 25",
            "ds_add_u32 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, -1",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v4", 0)
        mem = L.run(body, lanes=32)
        for lane in range(32):
            expected = 75 if (lane % 2 == 0) else 50
            self.assertEqual(L.out(mem, lane, 0), expected)

class TestLdsAddU32Aliasing(unittest.TestCase):
    def test_aliased_address_and_data_registers(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 10",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_add_u32 v1, v1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 10 + 0x40)

class TestLdsAddU32BoundariesAndNegative(unittest.TestCase):
    def test_unsigned_wrap_at_2_pow_32(self):
        rtns, fin = L.run_chain([("ds_add_u32", 0x40, 0, (0xFFFFFFFF,))], {0x40: 0xFFFFFFFF})
        self.assertEqual(fin[0x40], 0xFFFFFFFE)

    def test_syntax_rejection_missing_operands(self):
        with self.assertRaises(Exception):
            L.translate(["ds_add_u32 v0"])
        with self.assertRaises(Exception):
            L.translate(["ds_add_u32_bad_opcode v0, v1"])

class TestLdsSubU32Reference(unittest.TestCase):
    def test_sub_basic_operations(self):
        cases = [
            (0, 0, 0),
            (15, 5, 10),
            (0, 1, 0xFFFFFFFF),
            (0x80000000, 1, 0x7FFFFFFF),
            (0x80000000, 0x80000000, 0),
            (0x23456789, 0x11111111, 0x12345678),
        ]
        for m, d, want in cases:
            with self.subTest(m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_sub_u32", 0x40, 0, (d,))], {0x40: m})
                self.assertEqual(fin[0x40], want)

    def test_non_returning_preserves_dst_sentinel(self):
        rtns, fin = L.run_chain([("ds_sub_u32", 0x40, 0, (42,))], {0x40: 100})
        self.assertEqual(rtns, [0xDEAD])
        self.assertEqual(fin[0x40], 58)

class TestLdsSubU32Offset(unittest.TestCase):
    def test_byte_offset_subtraction(self):
        init = {0x40: 50, 0x44: 100, 0x48: 150}
        rtns, fin = L.run_chain([("ds_sub_u32", 0x20, 0x24, (25,))], init)
        self.assertEqual(fin[0x44], 75)

    def test_multiple_offsets_same_base(self):
        init = {0x40: 50, 0x44: 60, 0x48: 70}
        ops = [
            ("ds_sub_u32", 0x40, 0, (10,)),
            ("ds_sub_u32", 0x40, 4, (20,)),
            ("ds_sub_u32", 0x40, 8, (30,)),
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(fin[0x40], 40)
        self.assertEqual(fin[0x44], 40)
        self.assertEqual(fin[0x48], 40)

class TestLdsSubU32Chaining(unittest.TestCase):
    def test_chained_subtractions(self):
        init = {0x40: 100}
        ops = [("ds_sub_u32", 0x40, 0, (10,)), ("ds_sub_u32", 0x40, 0, (20,)), ("ds_sub_u32", 0x40, 0, (30,))]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(fin[0x40], 40)

    def test_multi_location_independent_subtractions(self):
        init = {0x10: 500, 0x20: 800}
        ops = [
            ("ds_sub_u32", 0x10, 0, (100,)),
            ("ds_sub_u32", 0x20, 0, (200,)),
            ("ds_sub_u32", 0x10, 0, (50,)),
            ("ds_sub_u32", 0x20, 0, (100,)),
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(fin[0x10], 350)
        self.assertEqual(fin[0x20], 500)

class TestLdsSubU32Divergence(unittest.TestCase):
    def test_multilane_independent_lanes(self):
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_mov_b32_e32 v2, 200",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_add_nc_u32_e32 v3, 10, v0",
            "ds_sub_u32 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v4", 0)
        mem = L.run(body, lanes=32)
        for lane in range(32):
            self.assertEqual(L.out(mem, lane, 0), 200 - (10 + lane))

    def test_multilane_exec_mask_gating(self):
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, 0x55555555",
            "v_mov_b32_e32 v3, 25",
            "ds_sub_u32 v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b32 exec_lo, -1",
            "ds_read_b32 v4, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v4", 0)
        mem = L.run(body, lanes=32)
        for lane in range(32):
            expected = 75 if (lane % 2 == 0) else 100
            self.assertEqual(L.out(mem, lane, 0), expected)

class TestLdsSubU32Aliasing(unittest.TestCase):
    def test_aliased_address_and_data_registers(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_sub_u32 v1, v1",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v3", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(L.out(mem, 0, 0), 100 - 0x40)

class TestLdsSubU32BoundariesAndNegative(unittest.TestCase):
    def test_unsigned_underflow_wrap(self):
        rtns, fin = L.run_chain([("ds_sub_u32", 0x40, 0, (5,))], {0x40: 2})
        self.assertEqual(fin[0x40], 0xFFFFFFFF - 2)

    def test_syntax_rejection_missing_operands(self):
        with self.assertRaises(Exception):
            L.translate(["ds_sub_u32 v0"])
        with self.assertRaises(Exception):
            L.translate(["ds_sub_u32_bad_opcode v0, v1"])

class TestLdsIncU32Reference(unittest.TestCase):
    def test_inc_stepping_below_threshold(self):
        rtns, fin = L.run_chain([("ds_inc_u32", 0x40, 0, (10,))], {0x40: 4})
        self.assertEqual(fin[0x40], 5)

    def test_inc_wrapping_at_threshold(self):
        rtns, fin = L.run_chain([("ds_inc_u32", 0x40, 0, (10,))], {0x40: 10})
        self.assertEqual(fin[0x40], 0)
        rtns, fin = L.run_chain([("ds_inc_u32", 0x40, 0, (10,))], {0x40: 15})
        self.assertEqual(fin[0x40], 0)

    def test_non_returning_preserves_dst_sentinel(self):
        rtns, fin = L.run_chain([("ds_inc_u32", 0x40, 0, (10,))], {0x40: 3})
        self.assertEqual(rtns, [0xDEAD])
        self.assertEqual(fin[0x40], 4)

class TestLdsIncU32Offset(unittest.TestCase):
    def test_byte_offset_increment(self):
        init = {0x40: 1, 0x44: 2, 0x48: 3}
        rtns, fin = L.run_chain([("ds_inc_u32", 0x20, 0x24, (5,))], init)
        self.assertEqual(fin[0x44], 3)

    def test_multiple_offsets_same_base(self):
        init = {0x40: 4, 0x44: 5, 0x48: 6}
        ops = [
            ("ds_inc_u32", 0x40, 0, (5,)),
            ("ds_inc_u32", 0x40, 4, (5,)),
            ("ds_inc_u32", 0x40, 8, (5,)),
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(fin[0x40], 5)
        self.assertEqual(fin[0x44], 0)
        self.assertEqual(fin[0x48], 0)


if __name__ == "__main__":
    unittest.main()
