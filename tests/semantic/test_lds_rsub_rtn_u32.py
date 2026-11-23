from __future__ import annotations

"""Semantic interpreter tests for LDS atomic reverse subtraction (ds_rsub_rtn_u32, ds_rsub_u32).

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions":
    `ds_rsub_u32 vaddr, vdata [offset:N]`:
        Atomically computes:
            addr = (vaddr + byte_offset) & ~3
            M = LDS[addr >> 2]
            LDS[addr >> 2] = (vdata - M) & 0xFFFFFFFF
        Non-returning form: the prior value at LDS[addr >> 2] is discarded.

    `ds_rsub_rtn_u32 vdst, vaddr, vdata [offset:N]`:
        Atomically computes:
            addr = (vaddr + byte_offset) & ~3
            old_val = LDS[addr >> 2]
            LDS[addr >> 2] = (vdata - old_val) & 0xFFFFFFFF
            vdst = old_val
        Returning form: writes the pre-operation value into destination VGPR vdst.

Each test is a compute shader translated by tools/rdna2spirv.py and executed in
tests/semantic/spirv_mini_interp.py.
"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lds_run as L  # noqa: E402
import rdna2spirv  # noqa: E402
import support  # noqa: E402

M32 = L.M32


class TestHandTable(unittest.TestCase):
    # (M, D, D - M mod 2^32)
    TABLE = (
        (0, 0, 0),
        (42, 42, 0),
        (3, 10, 7),
        (10, 3, 0xFFFFFFF9),           # 3 - 10 = -7 = 0xFFFFFFF9
        (1, 0, 0xFFFFFFFF),           # 0 - 1 = -1 = 0xFFFFFFFF
        (0xFFFFFFFF, 0, 1),           # 0 - 0xFFFFFFFF = -(-1) = 1
        (0xFFFFFFFF, 0xFFFFFFFF, 0),
        (0x80000000, 0, 0x80000000),   # 0 - 0x80000000 = 0x80000000
        (0x7FFFFFFF, 0x80000000, 1),  # 0x80000000 - 0x7FFFFFFF = 1
        (5, 0xFFFFFFFF, 0xFFFFFFFA),  # 0xFFFFFFFF - 5 = 0xFFFFFFFA
        (0x12345678, 0x9ABCDEF0, 0x88888878),  # 0x9ABCDEF0 - 0x12345678
        (0x9ABCDEF0, 0x12345678, 0x77777788),  # 0x12345678 - 0x9ABCDEF0 mod 2^32
    )

    def test_one_operation(self):
        for m, d, want in self.TABLE:
            with self.subTest(op="ds_rsub_rtn_u32", m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_rsub_rtn_u32", 0x40, 0, (d,))], {0x40: m})
                self.assertEqual((rtns, fin), ([m], {0x40: want}))

            with self.subTest(op="ds_rsub_u32", m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_rsub_u32", 0x40, 0, (d,))], {0x40: m})
                self.assertEqual((rtns, fin), ([0xDEAD], {0x40: want}))


class TestAddressingAndOffsets(unittest.TestCase):
    def test_offset_adds_to_the_address_in_bytes(self):
        # Base 0x20 + offset 0x24 is 0x44; adjacent dwords at 0x40 and 0x48 are untouched.
        init = {0x40: 10, 0x44: 50, 0x48: 90}
        rtns, fin = L.run_chain([("ds_rsub_rtn_u32", 0x20, 0x24, (70,))], init)
        self.assertEqual(rtns, [50])
        self.assertEqual(fin, {0x44: 20})

    def test_offset_imm_variations(self):
        # Test various immediate byte offsets.
        init = {0x100: 25, 0x104: 100, 0x110: 500}
        ops = [
            ("ds_rsub_rtn_u32", 0x100, 0, (40,)),     # 40 - 25 = 15, rtn 25
            ("ds_rsub_rtn_u32", 0x100, 4, (150,)),    # 150 - 100 = 50, rtn 100
            ("ds_rsub_rtn_u32", 0x100, 16, (200,)),   # 200 - 500 = -300 = 0xFFFFFED4, rtn 500
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [25, 100, 500])
        self.assertEqual(fin, {0x100: 15, 0x104: 50, 0x110: 0xFFFFFED4})

    def test_unaligned_byte_address_truncates_to_dword(self):
        # Byte addresses 0x40, 0x41, 0x42, 0x43 all alias to the dword at 0x40.
        init = {0x40: 12}
        ops = [
            ("ds_rsub_rtn_u32", 0x41, 0, (30,)),   # reads 12 from dword 0x40, writes 30 - 12 = 18
            ("ds_rsub_rtn_u32", 0x40, 0, (50,)),   # reads 18 from dword 0x40, writes 50 - 18 = 32
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [12, 18])
        self.assertEqual(fin, {0x40: 32, 0x41: 32})


class TestInvolutionAndChaining(unittest.TestCase):
    def test_involution_property(self):
        # D - (D - M) == M: two consecutive reverse subtractions with the same data D restore initial value.
        m = 0x1337BEEF
        d = 0xCAFEBABE
        ops = [
            ("ds_rsub_rtn_u32", 0x60, 0, (d,)),
            ("ds_rsub_rtn_u32", 0x60, 0, (d,)),
        ]
        expected_step1 = (d - m) & M32
        rtns, fin = L.run_chain(ops, {0x60: m})
        self.assertEqual(rtns, [m, expected_step1])
        self.assertEqual(fin, {0x60: m})

    def test_sequential_chaining_on_same_dword(self):
        # Three chained reverse subtractions: M0=100 -> D1(250)-100=150 -> D2(300)-150=150 -> D3(50)-150=0xFFFFFF9C
        ops = [
            ("ds_rsub_rtn_u32", 0x80, 0, (250,)),
            ("ds_rsub_rtn_u32", 0x80, 0, (300,)),
            ("ds_rsub_rtn_u32", 0x80, 0, (50,)),
        ]
        rtns, fin = L.run_chain(ops, {0x80: 100})
        self.assertEqual(rtns, [100, 150, 150])
        self.assertEqual(fin, {0x80: (50 - 150) & M32})

    def test_interleaved_rtn_and_plain_forms(self):
        # Interleave ds_rsub_rtn_u32 and ds_rsub_u32.
        ops = [
            ("ds_rsub_rtn_u32", 0x20, 0, (100,)),  # M=20 -> 100 - 20 = 80, returns 20
            ("ds_rsub_u32", 0x20, 0, (200,)),      # M=80 -> 200 - 80 = 120, returns DEAD
            ("ds_rsub_rtn_u32", 0x20, 0, (300,)),  # M=120 -> 300 - 120 = 180, returns 120
        ]
        rtns, fin = L.run_chain(ops, {0x20: 20})
        self.assertEqual(rtns, [20, 0xDEAD, 120])
        self.assertEqual(fin, {0x20: 180})

    def test_deterministic_chaining_on_multiple_dwords(self):
        init = {0x10: 100, 0x20: 200, 0x30: 50}
        ops = [
            ("ds_rsub_rtn_u32", 0x10, 0, (150,)),  # 150 - 100 = 50, ret 100
            ("ds_rsub_u32",     0x20, 0, (250,)),  # 250 - 200 = 50, ret DEAD
            ("ds_rsub_rtn_u32", 0x30, 0, (70,)),   # 70 - 50 = 20, ret 50
            ("ds_rsub_rtn_u32", 0x10, 0, (80,)),   # 80 - 50 = 30, ret 50
            ("ds_rsub_u32",     0x20, 0, (100,)),  # 100 - 50 = 50, ret DEAD
            ("ds_rsub_rtn_u32", 0x30, 0, (20,)),   # 20 - 20 = 0, ret 20
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [100, 0xDEAD, 50, 50, 0xDEAD, 20])
        self.assertEqual(fin, {0x10: 30, 0x20: 50, 0x30: 0})


class TestMultiLaneExecution(unittest.TestCase):
    def test_32_lanes_parallel_rsub(self):
        # 32 lanes each operating on their own private LDS dword (lane * 4).
        # Lane i initializes LDS[i * 4] = 1000 + i * 17.
        # Then lane i runs ds_rsub_rtn_u32 with data = 5000 + i * 31.
        # Destination register gets 1000 + i * 17; final LDS gets (5000 + i * 31) - (1000 + i * 17).
        body = [
            # v1 = lane * 4
            "v_lshlrev_b32_e32 v1, 2, v0",
            # v2 = 1000 + lane * 17
            "v_mul_u32_u24_e32 v2, 17, v0",
            "v_add_nc_u32_e32 v2, 1000, v2",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            # v3 = 5000 + lane * 31 (operand data)
            "v_mul_u32_u24_e32 v3, 31, v0",
            "v_add_nc_u32_e32 v3, 5000, v3",
            # ds_rsub_rtn_u32 v4, v1, v3
            "ds_rsub_rtn_u32 v4, v1, v3",
            "s_waitcnt lgkmcnt(0)",
            # read final LDS into v5
            "ds_read_b32 v5, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v4", 0) + L.store("v5", 1)

        mem = L.run(body, lanes=32)
        for lane in range(32):
            init_val = 1000 + lane * 17
            data_val = 5000 + lane * 31
            expected_fin = (data_val - init_val) & M32
            actual_rtn = L.out(mem, lane, 0)
            actual_fin = L.out(mem, lane, 1)
            self.assertEqual(actual_rtn, init_val, f"lane {lane} rtn mismatch")
            self.assertEqual(actual_fin, expected_fin, f"lane {lane} final mismatch")


class TestRegisterAliasing(unittest.TestCase):
    def test_alias_vdst_with_vaddr(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 150",
            "ds_rsub_rtn_u32 v1, v1, v2",
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
            "v_mov_b32_e32 v2, 150",
            "ds_rsub_rtn_u32 v2, v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v3, v1",
            "s_waitcnt lgkmcnt(0)",
        ] + L.store("v2", 0) + L.store("v3", 1)
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 100)
        self.assertEqual(mem.get(4), 50)


class TestSpirvStructure(unittest.TestCase):
    def test_translator_emits_cas_loop_for_ds_rsub_rtn(self):
        listing = [
            "ds_rsub_rtn_u32 v0, v1, v2",
            "s_waitcnt lgkmcnt(0)",
        ]
        spvasm = support.translate_lines("cs_rsub_cas", listing + ["s_endpgm"], fssm=L.fssm())
        # Verify SPIR-V contains the CAS loop primitives:
        # OpAtomicCompareExchange, OpISub, OpPhi, OpLoopMerge, OpBranchConditional
        self.assertIn("OpAtomicCompareExchange", spvasm)
        self.assertIn("OpISub", spvasm)
        self.assertIn("OpPhi", spvasm)
        self.assertIn("OpLoopMerge", spvasm)
        self.assertIn("OpBranchConditional", spvasm)
        matches = re.findall(r"OpISub\s+%uint\s+(%\w+)\s+(%\w+)", spvasm)
        self.assertTrue(matches, "Expected at least one OpISub instruction in translated SPIR-V")


class TestNegativeRejection(unittest.TestCase):
    def test_outside_compute_rejection(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_test_rsub_outside_compute", ["ds_rsub_rtn_u32 v0, v1, v2", "s_endpgm"])

    def test_invalid_syntax_rejection(self):
        with self.assertRaises((rdna2spirv.Unsupported, IndexError, ValueError)):
            support.translate_lines("cs_test_rsub_bad_syntax", ["ds_rsub_rtn_u32 v1, v2", "s_endpgm"], fssm=L.fssm())


if __name__ == "__main__":
    unittest.main()
