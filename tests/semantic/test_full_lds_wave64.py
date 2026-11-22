"""SHADER-638. Execute the emitted module with the CPU workgroup interpreter."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_cs_shared_memory_limit import PROGRAM, fssm, support, workgroup_bytes
import test_loop_ballot_barrier as wave


class FullLdsWave64Tests(unittest.TestCase):
    def test_ds_nop_does_not_borrow_undeclared_guest_storage(self):
        program = ["s_mov_b64 exec, -1", "ds_nop",
                   "v_readfirstlane_b32 s8, v0", "s_endpgm"]
        asm, gen = wave._translate_cs("cs_full_nop", program, 64,
                                      lds_bytes=65536, emulate=True)
        self.assertFalse(gen.borrow_lds)
        self.assertFalse(gen.uses_lds)
        self.assertNotIn("%lds", asm)
        self.assertEqual(workgroup_bytes(asm), {"%x_buf": gen.x_buf_bytes()})

    def check_kernel(self, name, program, expected):
        threads = len(expected)
        asm, gen = wave._translate_cs("cs_" + name, program, threads, lds_bytes=65536, emulate=True)
        self.assertEqual(workgroup_bytes(asm), {"%lds": 65536})
        self.assertEqual(gen.lds_declared, 16384)
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                memory = wave.run_compute_workgroup(asm, threads, 32, push_constants=wave.VSHARP,
                                                     reverse_subgroups=reverse)
                self.assertEqual([memory[4 * i] for i in range(threads)], expected)

    def test_cross_half_and_consecutive_exchanges_restore_guest_words(self):
        source = Path(__file__).with_name("fixtures") / "full_lds_wave64.asm"
        program = [line for line in source.read_text(encoding="utf-8").splitlines()
                   if line and not line.startswith(";")]
        self.check_kernel("full_exchange", program, [840 + (i ^ 31) + 3 * i for i in range(64)])

    def test_waiting_and_finished_waves(self):
        program = wave.BarrierInLoopsTests()._wave_divergent_barrier_loop_cs(True)
        self.check_kernel("full_wait", program, [2] * 64 + [4] * 64)

    def test_partial_last_wave(self):
        program = ["s_mov_b64 exec, -1", "v_lshlrev_b32_e32 v1, 2, v0",
                   "ds_write_b32 v1, v0", "v_mov_b32_e32 v2, 124",
                   "ds_bpermute_b32 v3, v2, v0", "buffer_store_dword v3, v1, s[0:3], 0 offen",
                   "s_endpgm"]
        self.check_kernel("partial_last", program, [31] * 32 + [63] * 32 + [0] * 7)

    def test_quad_vote_ballot_and_dpp(self):
        # Only lanes 37 and 39 remain active. WQM expands them to quad 36..39.
        # The DPP fetch reverses this quad. Its value must survive the ballot
        # and the branch vote before adding in the count of active lanes.
        program = ["s_mov_b64 exec, -1", "v_lshlrev_b32_e32 v1, 2, v0",
                   "ds_write_b32 v1, v0", "v_mov_b32_e32 v2, 123",
                   "s_mov_b32 exec_lo, 0", "s_mov_b32 exec_hi, 160",
                   "s_wqm_b64 exec, exec",
                   "v_mov_b32_dpp v2, v0 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
                   "s_bcnt1_i32_b64 s8, exec", "s_cbranch_execz done",
                   "v_add_nc_u32_e32 v2, s8, v2", "s_mov_b64 exec, -1",
                   "buffer_store_dword v2, v1, s[0:3], 0 offen", "done:", "s_endpgm"]
        self.check_kernel("full_quad", program, [123] * 36 + [43, 42, 41, 40] + [123] * 24)

    def test_allocation_distinguishes_equal_high_offsets(self):
        program = [line.replace("ds_write_b32 v1, v0", "ds_write_b32 v1, v0 offset:65532")
                   for line in PROGRAM]
        with patch.object(support.rdna2spirv, "WAVE64_EMULATE", True):
            for units, expected in ((64, {"%lds": 32768, "%x_buf": 8604}), (128, {"%lds": 65536})):
                with self.subTest(units=units):
                    asm = support.translate_lines("cs_full_offset", program, fssm=fssm(units, 1024))
                    self.assertEqual(workgroup_bytes(asm), expected)
                    meta = support.rdna2spirv.shader_meta(support.rdna2spirv.translate.last_gen, asm)
                    self.assertEqual(meta["lds_bytes"], sum(expected.values()))


if __name__ == "__main__":
    unittest.main()
