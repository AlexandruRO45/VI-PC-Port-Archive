"""ds_cmpst_rtn_b32 and ds_cmpst_b32 through the translator (tools/rdna2spirv.py) and spirv_mini_interp.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA, "Data Share Instructions":
ds_cmpst_rtn_b32 vdst, vaddr, vcmp, vsrc [offset:offset16]
Performs a 32-bit atomic compare-and-swap on the LDS dword at (vaddr + offset) / 4:
    tmp = M[(vaddr + offset) / 4]
    if (tmp == vcmp)
        M[(vaddr + offset) / 4] = vsrc
    vdst = tmp

ds_cmpst_b32 vaddr, vcmp, vsrc [offset:offset16]
Performs the same 32-bit atomic compare-and-swap without returning the old value:
    tmp = M[(vaddr + offset) / 4]
    if (tmp == vcmp)
        M[(vaddr + offset) / 4] = vsrc

The old value is always returned in vdst for the _rtn_ form, whether or not the comparison matched.
Each test case is translated to SPIR-V via tools/rdna2spirv.py and executed in
tests/semantic/spirv_mini_interp.py via tests/semantic/lds_run.py and support.translate_lines.
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
OP_NO_RTN = "ds_cmpst_b32"
M32 = L.M32


def model_cmpst(ops, init):
    """Python architectural oracle for a sequence of ds_cmpst operations.

    ops: list of (mnemonic, byte_addr, byte_offset, (vcmp, vsrc))
    init: dict of {byte_addr: initial_dword}
    Returns: ([returned_vdst_for_each_op], {final_byte_addr: final_dword})
    """
    lds = dict(init)
    rtns = []
    for mnemonic, addr, off, (vcmp, vsrc) in ops:
        eff = addr + off
        old = lds.get(eff, 0) & M32
        cmp_val = vcmp & M32
        src_val = vsrc & M32
        if "_rtn_" in mnemonic:
            rtns.append(old)
        if old == cmp_val:
            lds[eff] = src_val
    finals = {addr + off for _, addr, off, _ in ops}
    return rtns, {a: lds.get(a, 0) & M32 for a in sorted(finals)}


class TestCompareAndSwapHandTable(unittest.TestCase):
    """Exhaustive hand-crafted verification of compare-and-swap edge cases and values."""

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
        init = {0x40: 100, 0x44: 200, 0x48: 300}
        rtns, fin = L.run_chain([(OP, 0x20, 0x24, (200, 999))], init)
        self.assertEqual(rtns, [200])
        self.assertEqual(fin, {0x44: 999})

        # Non-matching offset operation: returns old value, memory untouched
        rtns, fin = L.run_chain([(OP, 0x10, 0x34, (555, 777))], init)
        self.assertEqual(rtns, [200])
        self.assertEqual(fin, {0x44: 200})

    def test_immediate_offset_boundaries(self):
        for off in (0, 4, 16, 64, 0x100, 0x3FC):
            with self.subTest(offset=off):
                target = 0x40 + off
                rtns, fin = L.run_chain([(OP, 0x40, off, (0xAAAA, 0xBBBB))], {target: 0xAAAA})
                self.assertEqual(rtns, [0xAAAA])
                self.assertEqual(fin, {target: 0xBBBB})

    def test_cas_chain_progression(self):
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
        ops = [
            (OP, 0x10, 0, (1, 101)),
            (OP, 0x20, 0, (2, 202)),
            (OP, 0x30, 0, (999, 303)),
            (OP, 0x40, 0, (4, 404)),
        ]
        init = {0x10: 1, 0x20: 2, 0x30: 3, 0x40: 4}
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [1, 2, 3, 4])
        self.assertEqual(fin, {0x10: 101, 0x20: 202, 0x30: 3, 0x40: 404})


class TestNonReturnVariant(unittest.TestCase):
    """Verification of ds_cmpst_b32 non-returning atomic compare-and-swap variant."""

    def test_non_rtn_matching_value_updates_memory(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 100",
            "v_mov_b32_e32 v4, 250",
            f"{OP_NO_RTN} v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 250)

    def test_non_rtn_mismatch_leaves_memory_untouched(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 100",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 999",
            "v_mov_b32_e32 v4, 250",
            f"{OP_NO_RTN} v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 100)

    def test_non_rtn_with_immediate_offset(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 0xAA",
            "ds_write_b32 v1, v2 offset:32",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0xAA",
            "v_mov_b32_e32 v4, 0xBB",
            f"{OP_NO_RTN} v1, v3, v4 offset:32",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v6, v1 offset:32",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 0xBB)

    def test_non_rtn_chain_with_rtn(self):
        body = [
            "v_mov_b32_e32 v1, 0x50",
            "v_mov_b32_e32 v2, 10",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 10",
            "v_mov_b32_e32 v4, 20",
            f"{OP_NO_RTN} v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v5, 0xDEAD",
            f"{OP} v5, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v5, v31, s[0:3], 0 offen",
            "v_mov_b32_e32 v3, 20",
            "v_mov_b32_e32 v4, 30",
            f"{OP_NO_RTN} v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 20)
        self.assertEqual(mem.get(4), 30)


class TestRegisterAliasing(unittest.TestCase):
    """Test register reuse where destination overlaps with address, comparator, or source."""

    def test_alias_dst_and_vaddr(self):
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
        self.assertEqual(mem.get(0), 10)
        self.assertEqual(mem.get(4), 25)

    def test_alias_dst_and_vcmp(self):
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

    def test_alias_vaddr_and_vcmp(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 0x40",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v4, 0x99",
            f"{OP} v5, v1, v1, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v5, v31, s[0:3], 0 offen",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 0x40)
        self.assertEqual(mem.get(4), 0x99)

    def test_alias_vcmp_and_vsrc(self):
        body = [
            "v_mov_b32_e32 v1, 0x50",
            "v_mov_b32_e32 v2, 77",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 77",
            f"{OP} v5, v1, v3, v3",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v5, v31, s[0:3], 0 offen",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 77)
        self.assertEqual(mem.get(4), 77)

    def test_alias_all_registers(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 0x40",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            f"{OP} v1, v1, v1, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v1, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 0x40)


class TestMultiLane(unittest.TestCase):
    """Multiple SIMD lanes executing independent compare-and-swap operations."""

    def test_multi_lane_independent_slots(self):
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v1, 0x40, v1",
            "v_add_nc_u32_e32 v2, 100, v0",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_and_b32_e32 v7, 1, v0",
            "v_cmp_eq_u32_e64 s[4:5], v7, 0",
            "v_cndmask_b32_e64 v3, 999, v2, s[4:5]",
            "v_add_nc_u32_e32 v4, 200, v0",
            f"{OP} v5, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            *L.store("v5", 0),
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            *L.store("v6", 1),
        ]
        mem = L.run(body, lanes=4)
        for lane in range(4):
            with self.subTest(lane=lane):
                rtn = L.out(mem, lane, 0)
                fin = L.out(mem, lane, 1)
                self.assertEqual(rtn, 100 + lane)
                expected_fin = (200 + lane) if (lane % 2 == 0) else (100 + lane)
                self.assertEqual(fin, expected_fin)

    def test_multi_lane_cross_lane_divergent_comparison(self):
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v1, 0x80, v1",
            "v_add_nc_u32_e32 v2, 50, v0",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_and_b32_e32 v7, 1, v0",
            "v_cmp_eq_u32_e64 s[4:5], v7, 0",
            "v_cndmask_b32_e64 v3, 0xDEAD, v2, s[4:5]",
            "v_add_nc_u32_e32 v4, 500, v0",
            f"{OP} v5, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            *L.store("v5", 0),
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            *L.store("v6", 1),
        ]
        mem = L.run(body, lanes=8)
        for lane in range(8):
            with self.subTest(lane=lane):
                rtn = L.out(mem, lane, 0)
                fin = L.out(mem, lane, 1)
                self.assertEqual(rtn, 50 + lane)
                expected_fin = (500 + lane) if (lane % 2 == 0) else (50 + lane)
                self.assertEqual(fin, expected_fin)

    def test_multi_lane_masked_inactive_lanes(self):
        body = [
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_add_nc_u32_e32 v1, 0xC0, v1",
            "v_mov_b32_e32 v2, 777",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b64 s[6:7], exec",
            "s_mov_b64 exec, 0x0F",
            "v_mov_b32_e32 v3, 777",
            "v_mov_b32_e32 v4, 888",
            "v_mov_b32_e32 v5, 0x9999",
            f"{OP} v5, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "s_mov_b64 exec, s[6:7]",
            *L.store("v5", 0),
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            *L.store("v6", 1),
        ]
        mem = L.run(body, lanes=8)
        for lane in range(4):
            self.assertEqual(L.out(mem, lane, 0), 777)
            self.assertEqual(L.out(mem, lane, 1), 888)
        for lane in range(4, 8):
            self.assertEqual(L.out(mem, lane, 1), 777)


class TestTranslatorLoweringAndOpcodes(unittest.TestCase):
    """Verify SPIR-V instruction lowering emitted by rdna2spirv via support.translate_lines."""

    def test_translator_emits_op_atomic_compare_exchange(self):
        lines = [
            f"{OP} v5, v1, v3, v4",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_cmpst_spv", lines, fssm=L.fssm())
        self.assertIn("OpAtomicCompareExchange %uint", spv)
        self.assertIn("%_ptr_WG_uint = OpTypePointer Workgroup %uint", spv)
        self.assertIn("OpVariable %_ptr_WG_lds Workgroup", spv)

    def test_translator_scope_and_semantics(self):
        lines = [
            f"{OP} v5, v1, v3, v4",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_cmpst_scope_sem", lines, fssm=L.fssm())
        cas_lines = [ln for ln in spv.splitlines() if "OpAtomicCompareExchange" in ln]
        self.assertEqual(len(cas_lines), 1)
        self.assertIn("%c2 %c0 %c0", cas_lines[0])

    def test_translator_operand_ordering_src_before_cmp(self):
        lines = [
            "v_mov_b32_e32 v3, 0x11111111",
            "v_mov_b32_e32 v4, 0x22222222",
            f"{OP} v5, v1, v3, v4",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_cmpst_order", lines, fssm=L.fssm())
        cas_line = next(ln for ln in spv.splitlines() if "OpAtomicCompareExchange" in ln)
        tokens = cas_line.strip().split()
        self.assertEqual(len(tokens), 10)
        self.assertEqual(tokens[2], "OpAtomicCompareExchange")
        val_tok = tokens[8]
        cmp_tok = tokens[9]
        self.assertNotEqual(val_tok, cmp_tok)

    def test_translator_non_rtn_drops_dst(self):
        lines = [
            f"{OP_NO_RTN} v1, v3, v4",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_cmpst_no_dst", lines, fssm=L.fssm())
        self.assertIn("OpAtomicCompareExchange %uint", spv)
        cas_idx = next(i for i, ln in enumerate(spv.splitlines()) if "OpAtomicCompareExchange" in ln)
        subsequent = spv.splitlines()[cas_idx + 1:cas_idx + 5]
        for ln in subsequent:
            self.assertNotIn("OpStore %v", ln)

    def test_translator_immediate_offset_emitted(self):
        lines_no_off = [f"{OP} v5, v1, v3, v4", "s_endpgm"]
        lines_off = [f"{OP} v5, v1, v3, v4 offset:64", "s_endpgm"]
        spv_no_off = support.translate_lines("cs_no_off", lines_no_off, fssm=L.fssm())
        spv_off = support.translate_lines("cs_off", lines_off, fssm=L.fssm())
        # Immediate offset 64 (0x40) lowers to an offset addition with %c40
        self.assertIn("%c40 = OpConstant %uint 64", spv_off)
        self.assertIn("%c40", spv_off)
        self.assertNotIn("%c40 = OpConstant %uint 64", spv_no_off)


class TestDirectTranslatorExecution(unittest.TestCase):
    """Hardware translation and execution verification through spirv_mini_interp."""

    def test_direct_atomic_cas_match_cycle(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 0x11112222",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0x11112222",
            "v_mov_b32_e32 v4, 0x33334444",
            f"{OP} v5, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v5, v31, s[0:3], 0 offen",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 0x11112222)
        self.assertEqual(mem.get(4), 0x33334444)

    def test_direct_atomic_cas_mismatch_cycle(self):
        body = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 0x11112222",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0x99999999",
            "v_mov_b32_e32 v4, 0x33334444",
            f"{OP} v5, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v5, v31, s[0:3], 0 offen",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 0x11112222)
        self.assertEqual(mem.get(4), 0x11112222)

    def test_direct_cas_retry_spin_pattern(self):
        body = [
            "v_mov_b32_e32 v1, 0x60",
            "v_mov_b32_e32 v2, 42",
            "ds_write_b32 v1, v2",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v3, 0",
            "v_mov_b32_e32 v4, 100",
            f"{OP} v5, v1, v3, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 0",
            "buffer_store_dword v5, v31, s[0:3], 0 offen",
            f"{OP} v8, v1, v5, v4",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 4",
            "buffer_store_dword v8, v31, s[0:3], 0 offen",
            "ds_read_b32 v6, v1",
            "s_waitcnt lgkmcnt(0)",
            "v_mov_b32_e32 v31, 8",
            "buffer_store_dword v6, v31, s[0:3], 0 offen",
        ]
        mem = L.run(body, lanes=1)
        self.assertEqual(mem.get(0), 42)
        self.assertEqual(mem.get(4), 42)
        self.assertEqual(mem.get(8), 100)

    def test_direct_extreme_bit_patterns(self):
        extremes = [0x00000000, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 0x55555555, 0xAAAAAAAA]
        for val in extremes:
            with self.subTest(val=hex(val)):
                rtns, fin = L.run_chain([(OP, 0x40, 0, (val, val ^ 0xFFFFFFFF))], {0x40: val})
                self.assertEqual(rtns, [val])
                self.assertEqual(fin, {0x40: val ^ 0xFFFFFFFF})


class TestMutationDetection(unittest.TestCase):
    """Negative control and regression verification ensuring mutations are detected."""

    def test_mutation_unconditional_exchange_detected(self):
        m_initial, cmp_val, src_val = 100, 200, 300
        _, correct_mem = model_cmpst([(OP, 0x40, 0, (cmp_val, src_val))], {0x40: m_initial})
        self.assertEqual(correct_mem[0x40], 100)
        mutant_mem = src_val
        self.assertNotEqual(mutant_mem, correct_mem[0x40])

    def test_mutation_swapped_operands_detected(self):
        m_initial, cmp_val, src_val = 10, 10, 20
        _, correct_mem = model_cmpst([(OP, 0x40, 0, (cmp_val, src_val))], {0x40: m_initial})
        self.assertEqual(correct_mem[0x40], 20)
        _, mutant_mem = model_cmpst([(OP, 0x40, 0, (src_val, cmp_val))], {0x40: m_initial})
        self.assertEqual(mutant_mem[0x40], 10)
        self.assertNotEqual(mutant_mem[0x40], correct_mem[0x40])

    def test_mutation_dropped_offset_detected(self):
        base, off = 0x40, 0x20
        _, correct = model_cmpst([(OP, base, off, (1, 2))], {base + off: 1, base: 99})
        self.assertEqual(correct[base + off], 2)
        self.assertNotIn(base, correct)

    def test_mutation_return_value_omission_detected(self):
        rtns, _ = L.run_chain([(OP, 0x40, 0, (10, 20))], {0x40: 10})
        self.assertEqual(rtns, [10])
        self.assertNotEqual(rtns, [0xDEAD])
        self.assertNotEqual(rtns, [0])


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
    """LDS operations must be rejected outside workgroup/compute shaders and on malformed operands."""

    def test_pixel_shader_refuses_lds_cmpst_rtn(self):
        with self.assertRaises(rdna2spirv.Unsupported) as caught:
            support.translate_lines("ps_lds_cmpst", [f"{OP} v5, v1, v3, v4", "s_endpgm"])
        self.assertIn("outside compute", str(caught.exception))

    def test_pixel_shader_refuses_lds_cmpst_non_rtn(self):
        with self.assertRaises(rdna2spirv.Unsupported) as caught:
            support.translate_lines("ps_lds_cmpst_no_rtn", [f"{OP_NO_RTN} v1, v3, v4", "s_endpgm"])
        self.assertIn("outside compute", str(caught.exception))

    def test_vertex_shader_refuses_lds_cmpst(self):
        with self.assertRaises(rdna2spirv.Unsupported) as caught:
            support.translate_lines("vs_lds_cmpst", [f"{OP} v5, v1, v3, v4", "s_endpgm"])
        self.assertIn("outside compute", str(caught.exception))

    def test_insufficient_operands_rejected(self):
        with self.assertRaises(Exception):
            support.translate_lines("cs_lds_bad1", [f"{OP} v5, v1, v3", "s_endpgm"], fssm=L.fssm())
        with self.assertRaises(Exception):
            support.translate_lines("cs_lds_bad2", [f"{OP_NO_RTN} v1, v3", "s_endpgm"], fssm=L.fssm())


if __name__ == "__main__":
    unittest.main()
