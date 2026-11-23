"""ds_inc_rtn_u32 and ds_inc_u32 through the translator.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions":
    ds_inc_rtn_u32 vdst, vaddr, vdata [offset:N]
    ds_inc_u32 vaddr, vdata [offset:N]

Unsigned 32-bit atomic increment with modulo wrap on the dword at (vaddr + offset):
    M = LDS[(vaddr + offset) / 4]
    new_M = 0 if M >= D else (M + 1) mod 2^32
    LDS[(vaddr + offset) / 4] = new_M

The _rtn_ form returns the pre-operation value M in vdst; the non-returning form
leaves vector destination registers unmodified. Modulo wrap compares M and D as
unsigned 32-bit integers.

Each case is a compute shader translated by tools/rdna2spirv.py (via support.translate_lines)
and executed in tests/semantic/spirv_mini_interp.py (via lds_run.py).
"""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402
import support  # noqa: E402

M32 = L.M32


def inc_u32(m: int, d: int) -> int:
    """Reference ISA calculation for 32-bit unsigned wrapping atomic increment."""
    m &= M32
    d &= M32
    return 0 if m >= d else (m + 1) & M32


def model(ops, init):
    lds = dict(init)
    rtns = []
    for name, addr, off, (d,) in ops:
        a = addr + off
        old = lds[a]
        rtns.append(old if "_rtn_" in name else 0xDEAD)
        lds[a] = inc_u32(old, d)
    return rtns, {a: lds[a] for a in sorted({addr + off for _, addr, off, _ in ops})}


class TestHandTable(unittest.TestCase):
    # (Initial M, Limit D, Expected new M)
    TABLE = (
        # Normal increments below limit
        (0, 5, 1),
        (1, 5, 2),
        (2, 5, 3),
        (3, 5, 4),
        (4, 5, 5),
        # Boundary wrap when M == D
        (5, 5, 0),
        # Wrap when M > D
        (6, 5, 0),
        (100, 10, 0),
        # Immediate wrap when limit is 0
        (0, 0, 0),
        (1, 0, 0),
        (0xFFFFFFFF, 0, 0),
        # Limit 1
        (0, 1, 1),
        (1, 1, 0),
        (2, 1, 0),
        # Unsigned 32-bit upper edge cases
        (0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF, 0),
        (0x7FFFFFFF, 0x80000000, 0x80000000),
        (0x80000000, 0x7FFFFFFF, 0),
        (0x80000000, 0x80000000, 0),
        (0x80000000, 0x80000001, 0x80000001),
        (0xFFFFFFFE, 0x80000000, 0),
    )

    def test_one_operation_returning(self):
        for m, d, want in self.TABLE:
            with self.subTest(op="ds_inc_rtn_u32", m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_inc_rtn_u32", 0x40, 0, (d,))], {0x40: m})
                self.assertEqual((rtns, fin), ([m], {0x40: want}))

    def test_one_operation_non_returning(self):
        for m, d, want in self.TABLE:
            with self.subTest(op="ds_inc_u32", m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_inc_u32", 0x40, 0, (d,))], {0x40: m})
                self.assertEqual((rtns, fin), ([0xDEAD], {0x40: want}))

    def test_offset_adds_to_the_address_in_bytes(self):
        # vaddr 0x20 + offset 0x24 targets byte 0x44; adjacent dwords 0x40 and 0x48 remain unchanged.
        init = {0x40: 10, 0x44: 2, 0x48: 30}
        rtns, fin = L.run_chain([("ds_inc_rtn_u32", 0x20, 0x24, (5,))], init)
        self.assertEqual(rtns, [2])
        self.assertEqual(fin, {0x44: 3})

        # Multiple ops with distinct base addresses and offsets
        ops = [
            ("ds_inc_rtn_u32", 0x20, 0x24, (5,)),  # 0x44: was 2, now 3 (returns 2)
            ("ds_inc_u32", 0x40, 0, (10,)),        # 0x40: was 10, now 0 (returns 0xDEAD)
            ("ds_inc_rtn_u32", 0x48, 0, (50,)),    # 0x48: was 30, now 31 (returns 30)
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [2, 0xDEAD, 30])
        self.assertEqual(fin, {0x40: 0, 0x44: 3, 0x48: 31})

    def test_full_wrapping_cycle(self):
        # Count 0 -> 1 -> 2 -> 3 -> 0 -> 1 using limit 3
        ops = [("ds_inc_rtn_u32", 0x50, 0, (3,)) for _ in range(5)]
        rtns, fin = L.run_chain(ops, {0x50: 0})
        self.assertEqual(rtns, [0, 1, 2, 3, 0])
        self.assertEqual(fin, {0x50: 1})

    def test_non_returning_wrapping_cycle(self):
        # Count 0 -> 1 -> 2 -> 0 -> 1 using limit 2 with ds_inc_u32
        ops = [("ds_inc_u32", 0x60, 0, (2,)) for _ in range(4)]
        rtns, fin = L.run_chain(ops, {0x60: 0})
        self.assertEqual(rtns, [0xDEAD, 0xDEAD, 0xDEAD, 0xDEAD])
        self.assertEqual(fin, {0x60: 1})

    def test_interleaved_returning_and_non_returning(self):
        # Interleave returning and non-returning increments on the same location
        ops = [
            ("ds_inc_rtn_u32", 0x70, 0, (4,)),  # 0 -> 1, returns 0
            ("ds_inc_u32", 0x70, 0, (4,)),      # 1 -> 2, returns 0xDEAD
            ("ds_inc_rtn_u32", 0x70, 0, (4,)),  # 2 -> 3, returns 2
            ("ds_inc_u32", 0x70, 0, (4,)),      # 3 -> 4, returns 0xDEAD
            ("ds_inc_rtn_u32", 0x70, 0, (4,)),  # 4 -> 0 (wrap), returns 4
            ("ds_inc_rtn_u32", 0x70, 0, (4,)),  # 0 -> 1, returns 0
        ]
        rtns, fin = L.run_chain(ops, {0x70: 0})
        self.assertEqual(rtns, [0, 0xDEAD, 2, 0xDEAD, 4, 0])
        self.assertEqual(fin, {0x70: 1})

    def test_multiple_independent_addresses(self):
        ops = [
            ("ds_inc_rtn_u32", 0x10, 0, (5,)),  # 0x10: 4 -> 5
            ("ds_inc_rtn_u32", 0x20, 0, (2,)),  # 0x20: 2 -> 0 (wrap)
            ("ds_inc_rtn_u32", 0x30, 0, (1,)),  # 0x30: 0 -> 1
            ("ds_inc_rtn_u32", 0x10, 0, (5,)),  # 0x10: 5 -> 0 (wrap)
        ]
        init = {0x10: 4, 0x20: 2, 0x30: 0}
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [4, 2, 0, 5])
        self.assertEqual(fin, {0x10: 0, 0x20: 0, 0x30: 1})


class TestSeeded(unittest.TestCase):
    def test_random_sequences(self):
        rng = random.Random(0x13C3)
        edge = (0, 1, 2, 3, 5, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF)
        for trial in range(8):
            addrs = [0x10, 0x30, 0x7C, 0x100]
            init = {a: rng.choice((rng.getrandbits(32), *edge)) for a in addrs}
            ops = []
            for _ in range(12):
                name = "ds_inc_rtn_u32" if rng.random() < 0.7 else "ds_inc_u32"
                eff = rng.choice(addrs)
                off = rng.choice((0, 0, 0x10, 0x20))
                limit = rng.choice((rng.getrandbits(32), *edge))
                ops.append((name, eff - off, off, (limit,)))
            with self.subTest(trial=trial):
                self.assertEqual(L.run_chain(ops, init), model(ops, init))


class TestOutsideCompute(unittest.TestCase):
    def test_pixel_shader_refuses_ds_inc(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_lds_inc_rtn", ["ds_inc_rtn_u32 v5, v1, v3", "s_endpgm"])
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_lds_inc", ["ds_inc_u32 v1, v3", "s_endpgm"])


if __name__ == "__main__":
    unittest.main()
