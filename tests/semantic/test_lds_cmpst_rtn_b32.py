"""ds_cmpst_rtn_b32 through the translator (tools/rdna2spirv.py) and spirv_mini_interp.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions":
ds_cmpst_rtn_b32 vdst, vaddr, vcmp, vsrc [offset:offset16]
Performs a 32-bit atomic compare-and-swap on the LDS dword at (vaddr + offset) / 4:
    tmp = M[(vaddr + offset) / 4]
    if (tmp == vcmp)
        M[(vaddr + offset) / 4] = vsrc
    vdst = tmp

The old value is always returned in vdst, whether or not the comparison matched.
Each test case is translated to SPIR-V via tools/rdna2spirv.py and executed in
tests/semantic/spirv_mini_interp.py via tests/semantic/lds_run.py.
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402
import support  # noqa: E402

OP = "ds_cmpst_rtn_b32"
M32 = L.M32


def model_cmpst(ops, init):
    """Python architectural oracle for a sequence of ds_cmpst_rtn_b32 operations.

    ops: list of (mnemonic, byte_addr, byte_offset, (vcmp, vsrc))
    init: dict of {byte_addr: initial_dword}
    Returns: ([returned_vdst_for_each_op], {final_byte_addr: final_dword})
    """
    lds = dict(init)
    rtns = []
    for _, addr, off, (vcmp, vsrc) in ops:
        eff = addr + off
        old = lds[eff] & M32
        cmp_val = vcmp & M32
        src_val = vsrc & M32
        rtns.append(old)
        if old == cmp_val:
            lds[eff] = src_val
    finals = {addr + off for _, addr, off, _ in ops}
    return rtns, {a: lds[a] & M32 for a in sorted(finals)}


class TestCompareAndSwapHandTable(unittest.TestCase):
    """Exhaustive hand-crafted verification of compare-and-swap edge cases and values."""

    # Hand-checked cases: (initial_M, vcmp, vsrc, expected_rtn, expected_M_after)
    CASES = [
        # Matching cases: memory matches comparator -> memory updated to vsrc
        (0, 0, 1, 0, 1),
        (1, 1, 0, 1, 0),
        (10, 10, 20, 10, 20),
        (0xFFFFFFFF, 0xFFFFFFFF, 0, 0xFFFFFFFF, 0),
        (0, 0, 0xFFFFFFFF, 0, 0xFFFFFFFF),
        (0x7FFFFFFF, 0x7FFFFFFF, 0x80000000, 0x7FFFFFFF, 0x80000000),
        (0x80000000, 0x80000000, 0x7FFFFFFF, 0x80000000, 0x7FFFFFFF),
        (0x12345678, 0x12345678, 0x9ABCDEF0, 0x12345678, 0x9ABCDEF0),
        (0xDEADBEEF, 0xDEADBEEF, 0xCAFEBABE, 0xDEADBEEF, 0xCAFEBABE),
        (42, 42, 42, 42, 42),  # store identical value

        # Non-matching cases: memory does NOT match comparator -> memory untouched
        (0, 1, 2, 0, 0),
        (1, 0, 2, 1, 1),
        (10, 20, 30, 10, 10),
        (0xFFFFFFFF, 0xFFFFFFFE, 0, 0xFFFFFFFF, 0xFFFFFFFF),
        (0x7FFFFFFF, 0x80000000, 1, 0x7FFFFFFF, 0x7FFFFFFF),
        (0x80000000, 0x7FFFFFFF, 1, 0x80000000, 0x80000000),
        (0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0xDEADBEEF, 0xDEADBEEF),
        (100, 101, 100, 100, 100),  # vsrc == initial, but comparator differs
        (100, 100, 100, 100, 100),  # all match
    ]

    def test_single_operation_cases(self):
        for m, cmp_val, src_val, want_rtn, want_mem in self.CASES:
            with self.subTest(m=hex(m), cmp=hex(cmp_val), src=hex(src_val)):
                rtns, fin = L.run_chain([(OP, 0x40, 0, (cmp_val, src_val))], {0x40: m})
                self.assertEqual((rtns, fin), ([want_rtn], {0x40: want_mem}))

    def test_byte_offset_calculation(self):
        # Base address 0x20 + offset 0x24 = effective address 0x44 (dword-aligned)
        # Verify targeted dword updates on match and neighboring dwords remain untouched
        init = {0x40: 100, 0x44: 200, 0x48: 300}
        rtns, fin = L.run_chain([(OP, 0x20, 0x24, (200, 999))], init)
        self.assertEqual(rtns, [200])
        self.assertEqual(fin, {0x44: 999})

        # Non-matching offset operation: returns old value, memory untouched
        rtns, fin = L.run_chain([(OP, 0x10, 0x34, (555, 777))], init)
        self.assertEqual(rtns, [200])
        self.assertEqual(fin, {0x44: 200})

    def test_immediate_offset_boundaries(self):
        # Check various valid 16-bit dword-aligned byte offsets
        for off in (0, 4, 16, 64, 0x100, 0x3FC):
            with self.subTest(offset=off):
                target = 0x40 + off
                rtns, fin = L.run_chain([(OP, 0x40, off, (0xAAAA, 0xBBBB))], {target: 0xAAAA})
                self.assertEqual(rtns, [0xAAAA])
                self.assertEqual(fin, {target: 0xBBBB})

    def test_cas_chain_progression(self):
        # Sequence of CAS operations on the same address demonstrating state transitions:
        # Initial: 10
        # 1. CAS(cmp=10, src=20) -> success: returns 10, memory becomes 20
        # 2. CAS(cmp=10, src=30) -> fail: returns 20, memory remains 20
        # 3. CAS(cmp=20, src=30) -> success: returns 20, memory becomes 30
        # 4. CAS(cmp=30, src=40) -> success: returns 30, memory becomes 40
        # 5. CAS(cmp=99, src=50) -> fail: returns 40, memory remains 40
        ops = [
            (OP, 0x30, 0, (10, 20)),
            (OP, 0x30, 0, (10, 30)),
            (OP, 0x30, 0, (20, 30)),
            (OP, 0x30, 0, (30, 40)),
            (OP, 0x30, 0, (99, 50)),
        ]
        rtns, fin = L.run_chain(ops, {0x30: 10})
        self.assertEqual(rtns, [10, 20, 20, 30, 40])
        self.assertEqual(fin, {0x30: 40})

    def test_independent_addresses_no_crosstalk(self):
        # Run CAS operations across multiple LDS addresses; verify none affect another
        ops = [
            (OP, 0x10, 0, (1, 101)),
            (OP, 0x20, 0, (2, 202)),
            (OP, 0x30, 0, (999, 303)),  # mismatch on 0x30
            (OP, 0x40, 0, (4, 404)),
        ]
        init = {0x10: 1, 0x20: 2, 0x30: 3, 0x40: 4}
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [1, 2, 3, 4])
        self.assertEqual(fin, {0x10: 101, 0x20: 202, 0x30: 3, 0x40: 404})


class TestRegisterAliasing(unittest.TestCase):
    """Test register reuse where destination overlaps with address, comparator, or source."""

    def test_alias_dst_and_vaddr(self):
        # ds_cmpst_rtn_b32 v1, v1, v3, v4 (vdst == vaddr)
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 10",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 10",
            "v_mov_b32_e32 v4, 25",
            f"{OP} v1, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v1, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v10, 0x40",
            "ds_read_b32 v6, v10",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 10)   # returned old value into v1
        self.assertEqual(mem.get(4), 25)   # LDS updated to v4

    def test_alias_dst_and_vcmp(self):
        # ds_cmpst_rtn_b32 v3, v1, v3, v4 (vdst == vcmp)
        body = [
            "v_mov_b32_e32 v1, 0x50",
            "v_mov_b32_e32 v2, 77",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 77",
            "v_mov_b32_e32 v4, 88",
            f"{OP} v3, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v3, v31, s[0:3], 0 offen",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 77)
        self.assertEqual(mem.get(4), 88)

    def test_alias_dst_and_vsrc(self):
        # ds_cmpst_rtn_b32 v4, v1, v3, v4 (vdst == vsrc)
        body = [
            "v_mov_b32_e32 v1, 0x60",
            "v_mov_b32_e32 v2, 123",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 123",
            "v_mov_b32_e32 v4, 456",
            f"{OP} v4, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v4, v31, s[0:3], 0 offen",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 123)
        self.assertEqual(mem.get(4), 456)


class TestMultiLane(unittest.TestCase):
    """Multiple SIMD lanes executing independent compare-and-swap operations."""

    def test_multi_lane_independent_slots(self):
        # 4 lanes, each operating on its own private LDS dword at 0x40 + lane * 4
        # Even lanes match comparator and update; odd lanes mismatch and keep original
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v1, 0x40, v1",
            "v_add_nc_u32_e32 v2, 100, v0",    # initial memory = 100 + lane
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_and_b32_e32 v7, 1, v0",         # lane & 1
            "v_cmp_eq_u32_e64 s[4:5], v7, 0",  # even lanes match
            "v_cndmask_b32_e64 v3, 999, v2, s[4:5]",  # vcmp = v2 if even else 999
            "v_add_nc_u32_e32 v4, 200, v0",    # vsrc = 200 + lane
            f"{OP} v5, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            *L.store("v5", 0),                 # slot 0: returned vdst
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            *L.store("v6", 1),                 # slot 1: final LDS memory
        ]
        mem = L.run(body, lanes=4)
        for lane in range(4):
            with self.subTest(lane=lane):
                rtn = L.out(mem, lane, 0)
                fin = L.out(mem, lane, 1)
                self.assertEqual(rtn, 100 + lane)
                expected_fin = (200 + lane) if (lane % 2 == 0) else (100 + lane)
                self.assertEqual(fin, expected_fin)


class TestSeededSequences(unittest.TestCase):
    """Deterministic pseudo-random sequences compared against reference oracle."""

    def test_random_cas_chains(self):
        rng = random.Random(0xCA532)
        for trial in range(10):
            addrs = [0x10, 0x24, 0x50, 0x88, 0xC0]
            init = {a: rng.getrandbits(32) for a in addrs}
            ops = []
            for _ in range(15):
                eff = rng.choice(addrs)
                off = rng.choice((0, 0, 4, 16))
                base = eff - off
                # 50% chance comparator matches current expected value in model
                _, current_state = model_cmpst(ops, init)
                current_val = current_state.get(eff, init[eff])
                if rng.random() < 0.5:
                    cmp_val = current_val
                else:
                    cmp_val = rng.getrandbits(32)
                src_val = rng.choice([rng.getrandbits(32), 0, 1, 0xFFFFFFFF, 0x80000000])
                ops.append((OP, base, off, (cmp_val, src_val)))

            with self.subTest(trial=trial):
                actual_rtns, actual_fin = L.run_chain(ops, init)
                expected_rtns, expected_fin = model_cmpst(ops, init)
                self.assertEqual(actual_rtns, expected_rtns)
                self.assertEqual(actual_fin, expected_fin)


class TestRejection(unittest.TestCase):
    """LDS operations must be rejected outside workgroup/compute shaders."""

    def test_pixel_shader_refuses_lds_cmpst(self):
        with self.assertRaises(rdna2spirv.Unsupported) as caught:
            support.translate_lines("ps_lds_cmpst", [f"{OP} v5, v1, v3, v4", "s_endpgm"])
        self.assertIn("outside compute", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
