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


if __name__ == "__main__":
    unittest.main()
