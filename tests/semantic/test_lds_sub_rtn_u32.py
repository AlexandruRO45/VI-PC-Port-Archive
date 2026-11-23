from __future__ import annotations

"""Semantic interpreter tests for LDS atomic subtraction: ds_sub_rtn_u32 and ds_sub_u32.

Standard library only; no GPU, no LLVM, no external tools, no game files.

Run directly:
    py -I tests/semantic/test_lds_sub_rtn_u32.py
Or via unittest discovery:
    py -m unittest discover -s tests/semantic -p "test_lds_sub_rtn_u32.py" -v

Background & Specification Rules:
  * AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
    - LDS is an on-chip, high-bandwidth software-managed scratchpad memory shared
      by all work-items (lanes/threads) in a workgroup.
    - `ds_sub_rtn_u32 vdst, vaddr, vdata [offset:N]`:
        Atomically computes:
          effective_byte_address = (vaddr + byte_offset) & 0xFFFFFFFF
          dword_index = (effective_byte_address >> 2) & (lds_dwords - 1)
          old_val = LDS[dword_index]
          LDS[dword_index] = (old_val - vdata) & 0xFFFFFFFF
          vdst = old_val
    - `ds_sub_u32 vaddr, vdata [offset:N]`:
        Non-returning form that performs the same atomic subtraction without returning
        the pre-subtraction value to a destination VGPR.
    - Byte addressing & alignment:
        Byte addresses are converted to dwords via right shift by 2, dropping
        unaligned low bits [1:0]. Out-of-range addresses wrap within the allocated
        power-of-two LDS allocation via `& (lds_dwords - 1)`.
    - Modulo 2^32 wrapping:
        All arithmetic wraps modulo 2^32 without saturation or trapping
        (e.g., 0 - 1 = 0xFFFFFFFF, 0 - 0xFFFFFFFF = 1).
  * SPIR-V Translation:
    - LDS is modeled as a Workgroup-scoped array of uints (%lds).
    - Translates to OpAtomicISub:
        %old = OpAtomicISub %uint %ptr %scope %semantics %vdata
      where %scope is Workgroup (2) and %semantics is Relaxed (0).
    - Result %old is stored into destination register vdst for returning form.
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lds_run as L  # noqa: E402
import rdna2spirv  # noqa: E402
import support  # noqa: E402

M32 = 0xFFFFFFFF


class TestLdsSubRtnU32SingleOp(unittest.TestCase):
    """Test individual ds_sub_rtn_u32 and ds_sub_u32 operations across critical numeric boundaries."""

    # (initial_memory, subtrahend, expected_returned_old, expected_final_memory)
    BOUNDARY_CASES = (
        (0, 0, 0, 0),
        (1, 1, 1, 0),
        (0, 1, 0, 0xFFFFFFFF),                     # underflow wrapping
        (0, 0xFFFFFFFF, 0, 1),                     # negative subtrahend / wrap
        (10, 4, 10, 6),                            # simple subtraction
        (100, 100, 100, 0),
        (0x80000000, 1, 0x80000000, 0x7FFFFFFF),   # sign bit transition
        (0x7FFFFFFF, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF),
        (0x80000000, 0x80000000, 0x80000000, 0),
        (0xFFFFFFFF, 1, 0xFFFFFFFF, 0xFFFFFFFE),
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0),
        (0x12345678, 0x9ABCDEF0, 0x12345678, (0x12345678 - 0x9ABCDEF0) & M32),
        (0xA5A5A5A5, 0x5A5A5A5A, 0xA5A5A5A5, 0x4B4B4B4B),
        (0x00000001, 0x00000002, 0x00000001, 0xFFFFFFFF),
    )

    def test_boundary_cases(self):
        for m, d, want_rtn, want_mem in self.BOUNDARY_CASES:
            with self.subTest(op="ds_sub_rtn_u32", m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_sub_rtn_u32", 0x40, 0, (d,))], {0x40: m})
                self.assertEqual(rtns, [want_rtn], f"Returned old value mismatch for {hex(m)} - {hex(d)}")
                self.assertEqual(fin, {0x40: want_mem}, f"Final memory value mismatch for {hex(m)} - {hex(d)}")

            with self.subTest(op="ds_sub_u32", m=hex(m), d=hex(d)):
                rtns, fin = L.run_chain([("ds_sub_u32", 0x40, 0, (d,))], {0x40: m})
                # non-returning form leaves preset 0xDEAD in destination; memory updated
                self.assertEqual(rtns, [0xDEAD])
                self.assertEqual(fin, {0x40: want_mem}, f"Final memory value mismatch for {hex(m)} - {hex(d)}")


class TestLdsSubRtnU32Offsets(unittest.TestCase):
    """Test immediate byte offset addressing for ds_sub_rtn_u32."""

    def test_byte_offsets(self):
        # Base address 0x20 with varying byte offsets
        # Each offset maps to a specific dword: (0x20 + offset)
        offsets = (0, 4, 8, 16, 28, 64, 128, 256)
        for off in offsets:
            with self.subTest(offset=off):
                eff_addr = 0x20 + off
                init = {0x1C: 0x11111111, eff_addr: 1000, eff_addr + 4: 0x22222222}
                rtns, fin = L.run_chain([("ds_sub_rtn_u32", 0x20, off, (350,))], init)
                self.assertEqual(rtns, [1000])
                self.assertEqual(fin[eff_addr], 650)
                # Verify neighbouring addresses were not corrupted
                read_all, mem_all = L.run_chain(
                    [("ds_sub_rtn_u32", 0x20, off, (0,))],
                    {0x1C: 0x11111111, eff_addr: 650, eff_addr + 4: 0x22222222}
                )
                self.assertEqual(read_all, [650])

    def test_unaligned_byte_address_truncation(self):
        # RDNA2 LDS dword addressing: byte address low 2 bits are dropped (eff_addr >> 2)
        # Therefore vaddr=0x41 with offset=0 targets dword at byte 0x40
        rtns, fin = L.run_chain([("ds_sub_rtn_u32", 0x41, 0, (15,))], {0x40: 100})
        self.assertEqual(rtns, [100])
        self.assertEqual(fin, {0x41: 85})


class TestLdsSubRtnU32Chaining(unittest.TestCase):
    """Test sequential chained atomic subtractions on the same and multiple LDS dwords."""

    def test_chain_on_same_dword(self):
        # Successive subtractions on one memory location:
        # 1000 - 100 = 900
        #  900 - 250 = 650
        #  650 -  50 = 600
        #  600 - 600 = 0
        #    0 -   1 = 0xFFFFFFFF
        ops = [
            ("ds_sub_rtn_u32", 0x30, 0, (100,)),
            ("ds_sub_rtn_u32", 0x30, 0, (250,)),
            ("ds_sub_rtn_u32", 0x30, 0, (50,)),
            ("ds_sub_rtn_u32", 0x30, 0, (600,)),
            ("ds_sub_rtn_u32", 0x30, 0, (1,)),
        ]
        rtns, fin = L.run_chain(ops, {0x30: 1000})
        self.assertEqual(rtns, [1000, 900, 650, 600, 0])
        self.assertEqual(fin, {0x30: 0xFFFFFFFF})

    def test_interleaved_dwords(self):
        # Subtractions interleaved across three distinct LDS locations mixing returning and non-returning forms
        init = {0x10: 500, 0x20: 300, 0x30: 100}
        ops = [
            ("ds_sub_rtn_u32", 0x10, 0, (50,)),    # 0x10: 500 -> 450 (returns 500)
            ("ds_sub_u32",     0x20, 0, (100,)),   # 0x20: 300 -> 200 (returns DEAD)
            ("ds_sub_rtn_u32", 0x30, 0, (30,)),    # 0x30: 100 -> 70  (returns 100)
            ("ds_sub_rtn_u32", 0x10, 0, (200,)),   # 0x10: 450 -> 250 (returns 450)
            ("ds_sub_u32",     0x20, 0, (200,)),   # 0x20: 200 -> 0   (returns DEAD)
            ("ds_sub_rtn_u32", 0x30, 0, (70,)),    # 0x30: 70  -> 0   (returns 70)
        ]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [500, 0xDEAD, 100, 450, 0xDEAD, 70])
        self.assertEqual(fin, {0x10: 250, 0x20: 0, 0x30: 0})


class TestLdsSubRtnU32RegisterAliasing(unittest.TestCase):
    """Test register aliasing between destination (vdst), address (vaddr), and data (vdata)."""

    def test_vdst_aliases_vaddr(self):
        # In ds_sub_rtn_u32 v1, v1, v3: vdst == vaddr == v1
        # The address in v1 must be read before the returned old value overwrites v1.
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 500",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 120",
            "ds_sub_rtn_u32 v1, v1, v3",            # vdst == vaddr (v1)
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v1, v31, s[0:3], 0 offen",   # store returned old value
            "v_mov_b32_e32 v4, 0x40",
            "ds_read_b32 v5, v4",                   # read back updated memory
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v5, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        # v1 returned old value: 500
        self.assertEqual(mem.get(0), 500)
        # LDS memory updated: 500 - 120 = 380
        self.assertEqual(mem.get(4), 380)

    def test_vdst_aliases_vdata(self):
        # In ds_sub_rtn_u32 v2, v1, v2: vdst == vdata == v2
        # The subtrahend in v2 must be read before the returned old value overwrites v2.
        body = [
            "v_mov_b32_e32 v1, 0x60",
            "v_mov_b32_e32 v10, 800",
            "ds_write_b32 v1, v10",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v2, 350",
            "ds_sub_rtn_u32 v2, v1, v2",            # vdst == vdata (v2)
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v2, v31, s[0:3], 0 offen",   # store returned old value
            "ds_read_b32 v6, v1",                   # read back updated memory
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        # v2 returned old value: 800
        self.assertEqual(mem.get(0), 800)
        # LDS memory updated: 800 - 350 = 450
        self.assertEqual(mem.get(4), 450)


class TestLdsSubRtnU32MultiLane(unittest.TestCase):
    """Test multi-lane execution of ds_sub_rtn_u32 across independent LDS slots."""

    def test_32_lanes_independent_slots(self):
        # Each lane operates on its own LDS slot: byte address = lane * 4
        # Lane seeds LDS[lane * 4] = (lane + 1) * 100
        # Lane performs ds_sub_rtn_u32 subtracting (lane + 1) * 25
        # Expected returned old value: (lane + 1) * 100
        # Expected final memory: (lane + 1) * 75
        body = [
            # v1 = lane * 4 (LDS address)
            "v_lshlrev_b32_e32 v1, 2, v0",
            # v2 = (lane + 1) * 100
            "v_add_nc_u32_e32 v2, 1, v0",
            "v_mul_lo_u32 v2, v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            # v3 = (lane + 1) * 25
            "v_add_nc_u32_e32 v3, 1, v0",
            "v_mul_lo_u32 v3, v3, 25",
            # ds_sub_rtn_u32 v5, v1, v3
            "v_mov_b32_e32 v5, 0xdead",
            "ds_sub_rtn_u32 v5, v1, v3",
            "s_waitcnt lgkmcnt(0)",
            # read back updated memory to v6
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            # store v5 (returned old) to out slot 0, v6 (final mem) to out slot 1
        ] + L.store("v5", 0) + L.store("v6", 1)

        mem = L.run(body, lanes=32)
        for lane in range(32):
            want_old = (lane + 1) * 100
            want_new = (lane + 1) * 75
            got_old = L.out(mem, lane, 0)
            got_new = L.out(mem, lane, 1)
            self.assertEqual(got_old, want_old, f"Lane {lane} returned value mismatch")
            self.assertEqual(got_new, want_new, f"Lane {lane} final memory mismatch")


class TestLdsSubRtnU32SpirvOutput(unittest.TestCase):
    """Verify translator SPIR-V structure for ds_sub_rtn_u32."""

    def test_spirv_op_atomic_isub(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 0x10",
            "ds_sub_rtn_u32 v0, v1, v2 offset:16",
        ]
        spv = L.translate(body)
        # Check that OpAtomicISub is generated
        self.assertIn("OpAtomicISub", spv)
        # Check that LDS Workgroup array pointer is accessed
        self.assertIn("%lds", spv)
        # Check that Workgroup storage class is used
        self.assertIn("Workgroup", spv)


class TestLdsSubRtnU32NegativeRejection(unittest.TestCase):
    """Verify negative syntax rejection and stage enforcement."""

    def test_outside_compute_rejection(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines("ps_test_sub_outside_compute", ["ds_sub_rtn_u32 v0, v1, v2", "s_endpgm"])

    def test_invalid_syntax_rejection(self):
        with self.assertRaises((rdna2spirv.Unsupported, IndexError, ValueError)):
            support.translate_lines("cs_test_sub_bad_syntax", ["ds_sub_rtn_u32 v1, v2", "s_endpgm"], fssm=L.fssm())


if __name__ == "__main__":
    unittest.main()
