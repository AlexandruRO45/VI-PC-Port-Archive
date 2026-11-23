"""ds_min_rtn_i32 through the translator (RDNA2 -> SPIR-V).

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions":
    ds_min_rtn_i32 vdst, vaddr, vdata [offset:N]
Atomically computes:
    effective_byte = (vaddr + offset) & 0xFFFFFFFF
    old_val = LDS[effective_byte / 4]
    new_val = min_i32(old_val, vdata)
    LDS[effective_byte / 4] = new_val
    vdst = old_val

The comparison treats operands as two's-complement signed 32-bit integers [-2^31, 2^31 - 1].
In particular:
    - 0x80000000 (-2147483648) is the minimum signed 32-bit value.
    - 0xFFFFFFFF (-1) < 0x00000000 (0).
    - 0x7FFFFFFF (+2147483647) is the maximum signed 32-bit value.

This contrasts with unsigned minimum (ds_min_u32), where 0x00000000 is the minimum
and 0xFFFFFFFF is the maximum.

The returning form (_rtn_) writes the pre-atomic value into vdst.

Each test is translated by tools/rdna2spirv.py and executed in tests/semantic/spirv_mini_interp.py
via lds_run.
"""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import lds_run as L
import rdna2spirv
import support

MASK32 = 0xFFFFFFFF
SIGN32 = 0x80000000
INT32_MIN = 0x80000000
INT32_MAX = 0x7FFFFFFF


def s32(x: int) -> int:
    """Interpret a 32-bit unsigned bit pattern as a signed integer."""
    u = x & MASK32
    return u - (1 << 32) if (u & SIGN32) else u


def u32(x: int) -> int:
    """Convert an integer to an unsigned 32-bit pattern."""
    return x & MASK32


def min_i32(m: int, d: int) -> int:
    """Two's-complement signed 32-bit minimum."""
    return m if s32(m) <= s32(d) else d


def model(ops, init):
    lds = dict(init)
    rtns = []
    for name, addr, off, (d,) in ops:
        a = addr + off
        old = lds.get(a, 0)
        rtns.append(old)
        lds[a] = min_i32(old, d)
    return rtns, {a: lds[a] for a in sorted({addr + off for _, addr, off, _ in ops})}


class TestDSMinRtnI32Translation(unittest.TestCase):
    """Verify that rdna2spirv translates ds_min_rtn_i32 to SPIR-V OpAtomicSMin."""

    def test_spirv_opcode_and_operands(self):
        spv = L.translate([
            "v_mov_b32_e32 v1, 0x20",
            "v_mov_b32_e32 v2, 0x10",
            "ds_min_rtn_i32 v5, v1, v2 offset:8",
            "s_waitcnt lgkmcnt(0)",
        ])
        self.assertIn("OpAtomicSMin", spv)
        self.assertIn("%lds", spv)
        self.assertIn("Workgroup", spv)


class TestDSMinRtnI32SignedBoundaries(unittest.TestCase):
    """Test signed 32-bit two's-complement boundaries and comparisons."""

    # Table of (M, D, expected_min_i32):
    # M = initial memory value, D = operand data, expected = result in memory.
    # The return value is always the initial M.
    CASES = (
        # Basic positive comparisons
        (10, 20, 10),
        (50, 30, 30),
        (100, 100, 100),
        (0, 0, 0),

        # Crossing zero: negative vs non-negative
        (0, 0xFFFFFFFF, 0xFFFFFFFF),       # 0 vs -1: min is -1
        (0xFFFFFFFF, 0, 0xFFFFFFFF),       # -1 vs 0: min is -1
        (10, 0xFFFFFFFF, 0xFFFFFFFF),      # 10 vs -1: min is -1
        (0xFFFFFFFF, 10, 0xFFFFFFFF),      # -1 vs 10: min is -1
        (100, u32(-500), u32(-500)),       # 100 vs -500: min is -500
        (u32(-500), 100, u32(-500)),       # -500 vs 100: min is -500

        # Negative vs negative comparisons
        (u32(-1), u32(-10), u32(-10)),     # -1 vs -10: min is -10
        (u32(-10), u32(-1), u32(-10)),     # -10 vs -1: min is -10
        (u32(-100), u32(-50), u32(-100)),  # -100 vs -50: min is -100

        # Extremes: INT32_MIN (0x80000000) and INT32_MAX (0x7FFFFFFF)
        (INT32_MAX, INT32_MIN, INT32_MIN),
        (INT32_MIN, INT32_MAX, INT32_MIN),
        (0, INT32_MIN, INT32_MIN),
        (INT32_MIN, 0, INT32_MIN),
        (u32(-1), INT32_MIN, INT32_MIN),
        (INT32_MIN, u32(-1), INT32_MIN),
        (INT32_MIN, INT32_MIN + 1, INT32_MIN),
        (INT32_MIN + 1, INT32_MIN, INT32_MIN),
        (INT32_MAX, INT32_MAX - 1, INT32_MAX - 1),
        (INT32_MAX - 1, INT32_MAX, INT32_MAX - 1),
    )

    def test_single_operation_signed_min(self):
        for m, d, expected in self.CASES:
            with self.subTest(m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_min_rtn_i32", 0x40, 0, (d,))], {0x40: m})
                self.assertEqual(rtns, [m], f"Expected return {hex(m)}, got {hex(rtns[0])}")
                self.assertEqual(fin, {0x40: expected}, f"Expected memory {hex(expected)}, got {hex(fin[0x40])}")

    def test_divergence_from_unsigned_minimum(self):
        """Signed min_i32 and unsigned min_u32 must produce opposite winners when bit 31 differs."""
        # 1 vs 0x80000000 (-2147483648):
        # min_i32(1, -2147483648) = -2147483648 (0x80000000)
        # min_u32(1, 2147483648)  = 1
        ops = [
            ("ds_min_rtn_i32", 0x10, 0, (0x80000000,)),
            ("ds_min_rtn_u32", 0x14, 0, (0x80000000,)),
        ]
        rtns, fin = L.run_chain(ops, {0x10: 1, 0x14: 1})
        self.assertEqual(rtns, [1, 1])
        self.assertEqual(fin[0x10], 0x80000000, "Signed min must choose negative 0x80000000")
        self.assertEqual(fin[0x14], 1, "Unsigned min must choose 1")

        # 0 vs 0xFFFFFFFF (-1):
        # min_i32(0, -1) = -1 (0xFFFFFFFF)
        # min_u32(0, 4294967295) = 0
        ops2 = [
            ("ds_min_rtn_i32", 0x20, 0, (0xFFFFFFFF,)),
            ("ds_min_rtn_u32", 0x24, 0, (0xFFFFFFFF,)),
        ]
        rtns2, fin2 = L.run_chain(ops2, {0x20: 0, 0x24: 0})
        self.assertEqual(rtns2, [0, 0])
        self.assertEqual(fin2[0x20], 0xFFFFFFFF, "Signed min must choose -1 (0xFFFFFFFF)")
        self.assertEqual(fin2[0x24], 0, "Unsigned min must choose 0")


class TestDSMinRtnI32Semantics(unittest.TestCase):
    """Test return values, sequential reductions, offsets, and address independence."""

    def test_byte_offset_addressing(self):
        """offset:N adds N bytes to vaddr."""
        # Address 0x30 with offset 0x14 lands at 0x44. Other locations untouched.
        ops = [("ds_min_rtn_i32", 0x30, 0x14, (u32(-20),))]
        init = {0x44: 50, 0x30: 100}
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [50])
        self.assertEqual(fin, {0x44: u32(-20)})

    def test_running_sequential_reduction(self):
        """A series of atomic mins at the same address returns each prior value."""
        values = [200, -5, 10, -50, -20, -100]
        # Start memory at 100.
        # Step 1: min(100, 200) -> 100, returns 100
        # Step 2: min(100, -5)  -> -5,  returns 100
        # Step 3: min(-5, 10)   -> -5,  returns -5
        # Step 4: min(-5, -50)  -> -50, returns -5
        # Step 5: min(-50, -20) -> -50, returns -50
        # Step 6: min(-50, -100)-> -100,returns -50
        ops = [("ds_min_rtn_i32", 0x50, 0, (u32(v),)) for v in values]
        rtns, fin = L.run_chain(ops, {0x50: 100})
        expected_rtns = [100, 100, u32(-5), u32(-5), u32(-50), u32(-50)]
        self.assertEqual(rtns, expected_rtns)
        self.assertEqual(fin, {0x50: u32(-100)})

    def test_independent_dwords_interleaved(self):
        """Atomics on distinct addresses do not interfere with each other."""
        ops = [
            ("ds_min_rtn_i32", 0x10, 0, (u32(-10),)),
            ("ds_min_rtn_i32", 0x20, 0, (u32(-20),)),
            ("ds_min_rtn_i32", 0x10, 0, (u32(-30),)),
            ("ds_min_rtn_i32", 0x20, 0, (u32(-5),)),
        ]
        init = {0x10: 0, 0x20: 0}
        rtns, fin = L.run_chain(ops, init)
        # Op 0: addr 0x10, init 0, min(0, -10) -> -10, returns 0
        # Op 1: addr 0x20, init 0, min(0, -20) -> -20, returns 0
        # Op 2: addr 0x10, curr -10, min(-10, -30) -> -30, returns -10
        # Op 3: addr 0x20, curr -20, min(-20, -5) -> -20, returns -20
        self.assertEqual(rtns, [0, 0, u32(-10), u32(-20)])
        self.assertEqual(fin, {0x10: u32(-30), 0x20: u32(-20)})


class TestDSMinRtnI32SeededFuzz(unittest.TestCase):
    """Fuzz randomized sequences against reference model."""

    def test_random_chain(self):
        rng = random.Random(0x4D494E)
        edge_values = (
            0, 1, MASK32, INT32_MIN, INT32_MAX,
            u32(-1), u32(-2), u32(-100), 100, 0x7FFFFFFE, 0x80000001,
        )
        for trial in range(10):
            addrs = [0x10, 0x20, 0x30, 0x40]
            init = {a: rng.choice(edge_values) for a in addrs}
            ops = []
            for _ in range(10):
                eff = rng.choice(addrs)
                off = rng.choice((0, 4, 8))
                base = eff - off
                d = rng.choice(edge_values)
                ops.append(("ds_min_rtn_i32", base, off, (d,)))
            with self.subTest(trial=trial):
                expected_rtns, expected_fin = model(ops, init)
                actual_rtns, actual_fin = L.run_chain(ops, init)
                self.assertEqual(actual_rtns, expected_rtns)
                self.assertEqual(actual_fin, expected_fin)


class TestDSMinRtnI32MultiLaneCompute(unittest.TestCase):
    """Test ds_min_rtn_i32 across multiple lanes in a compute shader."""

    def test_per_lane_atomic_min(self):
        # 4 lanes, each lane i operates on address 0x100 + i * 4
        # Each lane initializes LDS[i] = 10 * (i + 1), then ds_min_rtn_i32 with -5 * (i + 1).
        # Old value (10 * (i + 1)) is stored in out[lane * 8 + 0],
        # new value (-5 * (i + 1)) is read back and stored in out[lane * 8 + 1].
        body = []
        for i in range(4):
            body += [
                f"v_mov_b32_e32 v1, {0x100 + i * 4:#x}",
                f"v_mov_b32_e32 v2, {10 * (i + 1):#x}",
                "ds_write_b32 v1, v2",
            ]
        body += [
            "s_waitcnt lgkmcnt(0)",
            # lane i computes addr = 0x100 + v0 * 4
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v1, 0x100, v1",
            # data = -5 * (v0 + 1) -> we compute via signed arithmetic or load table
            # for 4 lanes, let's load data based on v0
            "v_add_nc_u32_e32 v2, 1, v0",
            # v2 = -5 * (v0 + 1)
            # multiply by 5 then negate
            "v_mul_lo_u32 v2, 5, v2",
            "v_sub_nc_u32_e32 v2, 0, v2",
            # execute ds_min_rtn_i32: v5 = old, LDS[v1] = min(old, v2)
            "ds_min_rtn_i32 v5, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            # read back final value
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
        ]
        # store v5 to slot 0 and v6 to slot 1
        body += L.store("v5", 0)
        body += L.store("v6", 1)

        mem = L.run(body, lanes=4)
        for lane in range(4):
            init_val = 10 * (lane + 1)
            data_val = u32(-5 * (lane + 1))
            old_res = L.out(mem, lane, 0)
            fin_res = L.out(mem, lane, 1)
            self.assertEqual(old_res, init_val, f"Lane {lane} expected old {init_val}, got {old_res}")
            self.assertEqual(fin_res, data_val, f"Lane {lane} expected min {data_val}, got {fin_res}")


if __name__ == "__main__":
    unittest.main()
