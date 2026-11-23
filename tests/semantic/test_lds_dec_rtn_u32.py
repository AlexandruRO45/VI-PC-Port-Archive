from __future__ import annotations

"""Semantic interpreter tests for LDS atomic unsigned decrement with wrap (ds_dec_rtn_u32).

Target:
    ds_dec_rtn_u32 vdst, vaddr, vdata [offset:N]
    Lowered in tools/rdna2spirv.py via Workgroup-scope OpAtomicCompareExchange CAS loop.

RDNA2 ISA Specification:
    Atomic unsigned decrement with wrapping limit.
    Address calculation:
        addr = (vaddr + offset) & 0xFFFFFFFF
        dword = (addr >> 2) & (lds_dwords - 1)
    Memory operation:
        cur = LDS[dword]
        LDS[dword] = vdata if (cur == 0 or cur > vdata) else (cur - 1) & 0xFFFFFFFF
        vdst = cur  (returning form returns the pre-operation memory value)

Standard library only; no GPU, no LLVM, no game files.
Run directly:
    py -I tests/semantic/test_lds_dec_rtn_u32.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402
import rdna2spirv  # noqa: E402
import support  # noqa: E402

OP = "ds_dec_rtn_u32"
MASK32 = 0xFFFFFFFF


class TestLdsDecRtnU32SingleOp(unittest.TestCase):
    """Test individual ds_dec_rtn_u32 executions with boundary and edge cases."""

    def test_in_range_decrement(self):
        """Values in range 0 < cur <= limit decrement by 1."""
        cases = [
            (5, 5, 4),
            (4, 5, 3),
            (2, 5, 1),
            (1, 5, 0),
            (100, 200, 99),
        ]
        for cur, limit, expected_new in cases:
            with self.subTest(cur=cur, limit=limit):
                rtns, finals = L.run_chain([(OP, 0x40, 0, (limit,))], {0x40: cur})
                self.assertEqual(rtns, [cur])
                self.assertEqual(finals, {0x40: expected_new})

    def test_zero_wraps_to_limit(self):
        """When cur == 0, memory wraps directly to limit."""
        for limit in (1, 5, 100, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF):
            with self.subTest(limit=hex(limit)):
                rtns, finals = L.run_chain([(OP, 0x40, 0, (limit,))], {0x40: 0})
                self.assertEqual(rtns, [0])
                self.assertEqual(finals, {0x40: limit})

    def test_above_limit_wraps_to_limit(self):
        """When cur > limit (unsigned), memory resets directly to limit."""
        cases = [
            (6, 5),
            (10, 5),
            (100, 5),
            (0x80000000, 5),
            (0xFFFFFFFF, 5),
            (0xFFFFFFFF, 0x7FFFFFFF),
        ]
        for cur, limit in cases:
            with self.subTest(cur=hex(cur), limit=hex(limit)):
                rtns, finals = L.run_chain([(OP, 0x40, 0, (limit,))], {0x40: cur})
                self.assertEqual(rtns, [cur])
                self.assertEqual(finals, {0x40: limit})

    def test_limit_zero(self):
        """When limit == 0, all values result in 0."""
        cases = [0, 1, 10, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]
        for cur in cases:
            with self.subTest(cur=hex(cur)):
                rtns, finals = L.run_chain([(OP, 0x40, 0, (0,))], {0x40: cur})
                self.assertEqual(rtns, [cur])
                self.assertEqual(finals, {0x40: 0})

    def test_limit_max_u32(self):
        """When limit == 0xFFFFFFFF, only cur == 0 wraps; all non-zero values decrement."""
        cases = [
            (0, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE),
            (0x80000000, 0xFFFFFFFF, 0x7FFFFFFF),
            (1, 0xFFFFFFFF, 0),
        ]
        for cur, limit, expected_new in cases:
            with self.subTest(cur=hex(cur), limit=hex(limit)):
                rtns, finals = L.run_chain([(OP, 0x40, 0, (limit,))], {0x40: cur})
                self.assertEqual(rtns, [cur])
                self.assertEqual(finals, {0x40: expected_new})

    def test_unsigned_high_bit_comparison(self):
        """Confirms comparison is unsigned: 0x80000000 > 0x7FFFFFFF."""
        # 0x80000000 > 0x7FFFFFFF unsigned -> wraps to 0x7FFFFFFF
        rtns, finals = L.run_chain([(OP, 0x40, 0, (0x7FFFFFFF,))], {0x40: 0x80000000})
        self.assertEqual(rtns, [0x80000000])
        self.assertEqual(finals, {0x40: 0x7FFFFFFF})

        # 0x7FFFFFFF <= 0x80000000 unsigned -> decrements to 0x7FFFFFFE
        rtns, finals = L.run_chain([(OP, 0x40, 0, (0x80000000,))], {0x40: 0x7FFFFFFF})
        self.assertEqual(rtns, [0x7FFFFFFF])
        self.assertEqual(finals, {0x40: 0x7FFFFFFE})

        # 0x80000000 <= 0x80000000 unsigned -> decrements to 0x7FFFFFFF
        rtns, finals = L.run_chain([(OP, 0x40, 0, (0x80000000,))], {0x40: 0x80000000})
        self.assertEqual(rtns, [0x80000000])
        self.assertEqual(finals, {0x40: 0x7FFFFFFF})


class TestLdsDecRtnU32Chains(unittest.TestCase):
    """Test sequential chained decrements and multi-address isolation."""

    def test_countdown_cycle(self):
        """Decrements countdown: 5 -> 4 -> 3 -> 2 -> 1 -> 0 -> 5 -> 4 -> 3."""
        limit = 5
        ops = [(OP, 0x20, 0, (limit,)) for _ in range(8)]
        rtns, finals = L.run_chain(ops, {0x20: 5})
        expected_rtns = [5, 4, 3, 2, 1, 0, 5, 4]
        self.assertEqual(rtns, expected_rtns)
        self.assertEqual(finals, {0x20: 3})

    def test_address_isolation(self):
        """Modifying one LDS dword does not affect other dwords."""
        ops = [
            (OP, 0x10, 0, (10,)),
            (OP, 0x20, 0, (20,)),
            (OP, 0x10, 0, (10,)),
        ]
        init = {0x10: 2, 0x20: 15, 0x30: 0xDEADBEEF}
        rtns, finals = L.run_chain(ops, init)
        self.assertEqual(rtns, [2, 15, 1])
        self.assertEqual(finals, {0x10: 0, 0x20: 14})

    def test_deterministic_multi_step_chain(self):
        init = {0x10: 5, 0x20: 0, 0x30: 10}
        ops = [
            (OP, 0x10, 0, (10,)),
            (OP, 0x20, 0, (15,)),
            (OP, 0x30, 0, (8,)),
            (OP, 0x10, 0, (10,)),
            (OP, 0x20, 0, (15,)),
            (OP, 0x30, 0, (8,)),
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [5, 0, 10, 4, 15, 8])
        self.assertEqual(fin, {0x10: 3, 0x20: 14, 0x30: 7})


class TestLdsDecRtnU32Offsets(unittest.TestCase):
    """Test immediate byte offset decoding and address calculation."""

    def test_offset_effective_address(self):
        """offset:N adds to base VGPR address in bytes."""
        rtns, finals = L.run_chain([(OP, 0x20, 0x24, (5,))], {0x44: 4, 0x20: 100, 0x48: 200})
        self.assertEqual(rtns, [4])
        self.assertEqual(finals, {0x44: 3})

    def test_multiple_offsets_same_base(self):
        """Different offsets from the same base address modify distinct slots."""
        ops = [
            (OP, 0x40, 0, (10,)),
            (OP, 0x40, 4, (10,)),
            (OP, 0x40, 8, (10,)),
        ]
        init = {0x40: 5, 0x44: 0, 0x48: 12}
        rtns, finals = L.run_chain(ops, init)
        self.assertEqual(rtns, [5, 0, 12])
        self.assertEqual(finals, {0x40: 4, 0x44: 10, 0x48: 10})


class TestLdsDecRtnU32RegisterAliasing(unittest.TestCase):
    """Test register reuse where destination overlaps with address or data."""

    def test_alias_vdst_with_vaddr(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 5",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 10",
            "ds_dec_rtn_u32 v1, v1, v2",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v1", 0)
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 5)

    def test_alias_vdst_with_vdata(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 5",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 10",
            "ds_dec_rtn_u32 v2, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v2", 0) + L.store("v3", 1)
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 5)
        self.assertEqual(mem.get(4), 4)


class TestLdsDecRtnU32Spirv(unittest.TestCase):
    """Verify SPIR-V lowering emitted by rdna2spirv."""

    def test_translator_emits_cas_loop_for_ds_dec_rtn(self):
        listing = [
            "ds_dec_rtn_u32 v5, v1, v3",
            "s_waitcnt lgkmcnt(0)",
        ]
        spvasm = support.translate_lines("cs_dec_cas", listing + ["s_endpgm"], fssm=L.fssm())
        self.assertIn("OpAtomicCompareExchange", spvasm)
        self.assertIn("OpPhi", spvasm)
        self.assertIn("OpLoopMerge", spvasm)
        self.assertIn("OpBranchConditional", spvasm)


class TestLdsDecRtnU32MultiLane(unittest.TestCase):
    """Test execution across all 32 lanes of a compute workgroup."""

    def test_multi_lane_independent_slots(self):
        """Each lane i decrements LDS[i * 4] from (i + 1) to i."""
        body = []
        body += [
            "v_lshlrev_b32_e32 v1, 2, v0",      # v1 = lane * 4
            "v_add_nc_u32_e32 v2, 1, v0",       # v2 = lane + 1
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "v_mov_b32_e32 v3, 100",            # limit = 100
            "v_mov_b32_e32 v5, 0xdead",
            "ds_dec_rtn_u32 v5, v1, v3",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
        ]
        body += L.store("v5", 0)
        body += L.store("v6", 1)

        mem = L.run(body, lanes=32)
        for lane in range(32):
            with self.subTest(lane=lane):
                self.assertEqual(L.out(mem, lane, 0), lane + 1)
                self.assertEqual(L.out(mem, lane, 1), lane)


class TestLdsDecRtnU32Rejections(unittest.TestCase):
    """Test translator error rejection on invalid stages and malformed operands."""

    def test_rejection_outside_compute(self):
        """ds_dec_rtn_u32 is only valid in compute stage."""
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_lds_dec", [f"{OP} v5, v1, v3", "s_endpgm"])

    def test_rejection_operand_count(self):
        """Missing or excess operands are rejected."""
        with self.assertRaises(rdna2spirv.Unsupported):
            L.translate([f"{OP} v1, v2"])
        with self.assertRaises(rdna2spirv.Unsupported):
            L.translate([f"{OP} v5, v1, v2, v3"])

    def test_rejection_non_vgpr_operands(self):
        """Destination and data operands must be VGPRs."""
        with self.assertRaises(rdna2spirv.Unsupported):
            L.translate([f"{OP} s0, v1, v2"])
        with self.assertRaises(rdna2spirv.Unsupported):
            L.translate([f"{OP} v5, v1, s2"])


if __name__ == "__main__":
    unittest.main()
