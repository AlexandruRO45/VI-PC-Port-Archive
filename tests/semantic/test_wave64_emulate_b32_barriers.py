"""Regression tests for 32-bit lane-mask barrier detection in tools/wave64_emulate_opt.py.

Covers three wave64 --wave64-emulate codegen paths in tools/rdna2spirv.py that emit cross-lane
`x_round` / `OpControlBarrier`:
  1. Live-SCC 32-bit bitwise SALU (`scc_logic` -> `scc_any_half`/`scc_from_mask`) across all nine
     `_SOP2_B32_LOGIC` opcodes (`s_and/or/xor/andn2/orn2/nand/nor/xnor/not_b32`), including mask
     `vcc_lo`/`vcc_hi` destinations with plain immediate sources.
  2. Opposite-half `_mask_b32_src` -> `wave_mask` reads in `s_not_b32` and `MASK_SOP2_B32`
     (including `s_cselect_b32`, which never writes SCC), even when SCC is dead.
  3. Opposite-half `wave_mask` in `s_*_saveexec_b32` and `s_andn{1,2}_wrexec_b32` (high-half mask
     source or `exec_hi`/`vcc_hi` destination), even when SCC is dead.

Also verifies negative controls: plain-data 32-bit bitwise SALU (even with live SCC) and same-half
32-bit mask ops with dead SCC continue to fold in `find_local_exec_folds` and stay `sync_free` in
multi-wave `structured_hybrid` plans.

Run from the repository root:  python -I tests/semantic/test_wave64_emulate_b32_barriers.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import lane_masks  # noqa: E402
import spirv_asm  # noqa: E402
import support  # noqa: E402
import test_wave64_emulate_opt as w64opt  # noqa: E402

rdna2spirv = support.rdna2spirv
wave64_emulate_opt = rdna2spirv.wave64_emulate_opt

VSHARP = w64opt.VSHARP
PC = VSHARP + [0] * 28


def _assemble_ok(tc, spv):
    binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
    tc.assertEqual(binary[:4], b"\x03\x02\x23\x07")


def _run_both_orders(spv, lx):
    out = []
    for rev in (False, True):
        mem, _ = w64opt.run_emulated_workgroup(spv, (lx, 1, 1), push_constants=PC, reverse_order=rev)
        out.append((rev, mem))
    return out


class LiveSccB32LogicBarrierTests(unittest.TestCase):
    def test_folded_execz_rejects_all_b32_logic_ops_with_live_scc(self):
        # VCC has only lanes 10..15 set in low half; s[18:19] has lanes 0..3 (disjoint);
        # s[20:21] has lanes 10..12 (overlapping); s[22:23] has lanes 10..15 (equal to vcc);
        # s[24:25] is ~vcc. Active EXEC inside the guarded region is lanes 0..7 (none have VCC set).
        # Each 32-bit logic op reads MASK_LO operand(s) with LIVE SCC: scc_any_half('lo') must vote
        # over lanes 0..31 so SCC reflects lanes 10..15 even though only lanes 0..7 are active.
        cases = [
            ("s_and_b32_zero", "s_and_b32 s10, vcc_lo, s18", False),
            ("s_and_b32_nz", "s_and_b32 s10, vcc_lo, s20", True),
            ("s_or_b32", "s_or_b32 s10, vcc_lo, 0", True),
            ("s_xor_b32_zero", "s_xor_b32 s10, vcc_lo, s22", False),
            ("s_xor_b32_nz", "s_xor_b32 s10, vcc_lo, s20", True),
            ("s_andn2_b32_zero", "s_andn2_b32 s10, vcc_lo, s22", False),
            ("s_andn2_b32_nz", "s_andn2_b32 s10, vcc_lo, s18", True),
            ("s_orn2_b32", "s_orn2_b32 s10, vcc_lo, -1", True),
            ("s_nand_b32", "s_nand_b32 s10, vcc_lo, 0", True),
            ("s_nor_b32", "s_nor_b32 s10, vcc_lo, -1", False),
            ("s_xnor_b32_zero", "s_xnor_b32 s10, vcc_lo, s24", False),
            ("s_xnor_b32_nz", "s_xnor_b32 s10, vcc_lo, s22", True),
            ("s_not_b32", "s_not_b32 s10, vcc_lo", True),
        ]
        for tag, op_asm, want_scc in cases:
            insns = [
                ("s_mov_b64 exec, -1", 4),
                ("v_mov_b32_e32 v4, 999", 4),
                ("v_cmp_ge_u32_e64 s[14:15], v0, 10", 8),
                ("v_cmp_lt_u32_e64 s[16:17], v0, 16", 8),
                ("s_and_b64 vcc, s[14:15], s[16:17]", 4),      # lanes 10..15
                ("v_cmp_lt_u32_e64 s[18:19], v0, 4", 8),       # lanes 0..3 (disjoint)
                ("v_cmp_lt_u32_e64 s[20:21], v0, 13", 8),
                ("s_and_b64 s[20:21], s[14:15], s[20:21]", 4),  # lanes 10..12 (overlap)
                ("s_mov_b64 s[22:23], vcc", 4),                # lanes 10..15 (equal)
                ("s_not_b64 s[24:25], vcc", 4),                # ~vcc
                ("v_cmp_lt_u32_e64 s[6:7], v0, 8", 8),
                ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
                ("s_cbranch_execz 4", 4),
                (op_asm, 4),
                ("s_cselect_b32 s11, 111, 222", 4),
                ("v_mov_b32_e32 v4, s11", 4),
                ("s_cmp_eq_u32 0, 0", 4),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("v_lshlrev_b32_e32 v10, 2, v0", 4),
                ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
                ("s_endpgm", 4),
            ]
            with self.subTest(case=tag):
                spv, _ = w64opt._translate_encoded(f"cs_b32_scc_{tag}", insns, (64, 1, 1), emulate=True, rsrc2=0x20088)
                _assemble_ok(self, spv)
                want_active = 111 if want_scc else 222
                for rev, mem in _run_both_orders(spv, 64):
                    for lane in range(64):
                        want = want_active if lane < 8 else 999
                        self.assertEqual(mem[lane * 4], want, f"{tag} rev={rev} lane={lane}")

    def test_folded_execz_rejects_vcc_mask_dst_imm_srcs_live_scc(self):
        # VCC is a lane mask; destination is vcc_lo with immediate-only sources -> to_mask_dst is True,
        # mask_srcs is empty, and scc_logic calls scc_any_half(r, 'lo') -> x_round.
        cases = [
            ("and_nz", "s_and_b32 vcc_lo, 0x0f, 0x05", 0x05, True),
            ("and_zero", "s_and_b32 vcc_lo, 0x0f, 0x30", 0x00, False),
            ("not_nz", "s_not_b32 vcc_lo, 0", 0xFFFFFFFF, True),
            ("not_zero", "s_not_b32 vcc_lo, -1", 0x00000000, False),
        ]
        for tag, op_asm, want_lo_mask, want_scc in cases:
            insns = [
                ("s_mov_b64 exec, -1", 4),
                ("v_mov_b32_e32 v4, 999", 4),
                ("v_cmp_lt_u32_e64 vcc, v0, 48", 8),
                ("v_cmp_lt_u32_e64 s[6:7], v0, 8", 8),
                ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
                ("s_cbranch_execz 5", 4),
                (op_asm, 4),
                ("s_cselect_b32 s11, 100, 200", 4),
                ("v_cndmask_b32_e32 v5, 0, 1, vcc", 4),
                ("v_add_nc_u32_e32 v4, s11, v5", 4),
                ("s_cmp_eq_u32 0, 0", 4),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("v_lshlrev_b32_e32 v10, 2, v0", 4),
                ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
                ("s_endpgm", 4),
            ]
            with self.subTest(case=tag):
                spv, _ = w64opt._translate_encoded(f"cs_vcc_dst_{tag}", insns, (64, 1, 1), emulate=True, rsrc2=0x20088)
                _assemble_ok(self, spv)
                base = 100 if want_scc else 200
                for rev, mem in _run_both_orders(spv, 64):
                    for lane in range(64):
                        if lane < 8:
                            bit = (want_lo_mask >> lane) & 1
                            want = base + bit
                        else:
                            want = 999
                        self.assertEqual(mem[lane * 4], want, f"{tag} rev={rev} lane={lane}")

    def test_multiwave_structured_hybrid_routes_vcc_mask_dst_live_scc_to_mini_scheduler(self):
        # 128-thread multi-wave shader: wave 0 takes the divergent s_cbranch_scc0 guard, wave 1 skips.
        # Inside the guard, s_and/xor/not_b32 writes vcc_lo (mask dest) with immediate sources and live SCC.
        # _is_uniform_scc_safe stays True (scc_any_half is wave-uniform), so structured_hybrid engages
        # and _insn_may_barrier must route the guard to mini_scheduler rather than inline_sync_free.
        cases = [
            ("and_nz", "s_and_b32 vcc_lo, 0x0f, 0x05", 0x05, True),
            ("xor_zero", "s_xor_b32 vcc_lo, 0x2a, 0x2a", 0x00, False),
            ("not_zero", "s_not_b32 vcc_lo, -1", 0x00, False),
        ]
        for tag, op_asm, want_lo_mask, want_scc in cases:
            insns = [
                ("s_mov_b64 exec, -1", 4),
                ("v_mov_b32_e32 v4, 777", 4),
                ("v_cmp_lt_u32_e64 vcc, v0, 128", 8),
                ("v_readfirstlane_b32 s4, v0", 4),  # wave 0 -> 0, wave 1 -> 64
                ("s_cmp_lt_u32 s4, 32", 4),
                ("s_cbranch_scc0 4", 4),
                (op_asm, 4),
                ("s_cselect_b32 s11, 100, 200", 4),
                ("v_cndmask_b32_e32 v5, 0, 1, vcc", 4),
                ("v_add_nc_u32_e32 v4, s11, v5", 4),
                ("v_lshlrev_b32_e32 v10, 2, v0", 4),
                ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
                ("s_endpgm", 4),
            ]
            with self.subTest(case=tag):
                spv, gen = w64opt._translate_encoded(f"cs_mw_vcc_{tag}", insns, (128, 1, 1), emulate=True, rsrc2=0x20088)
                self.assertEqual(gen.control_flow, "emulated wave64 structured")
                self.assertEqual(gen.emulate_plan["mode"], "structured_hybrid")
                self.assertGreater(len(gen.emulate_plan["mini_blocks"]), 0)
                self.assertEqual(len(gen.emulate_plan["sync_free_blocks"]), 0)
                _assemble_ok(self, spv)
                base = 100 if want_scc else 200
                for rev, mem in _run_both_orders(spv, 128):
                    for lane in range(128):
                        if lane < 64:
                            bit = ((want_lo_mask >> lane) & 1) if lane < 32 else 1
                            want = base + bit
                        else:
                            want = 777
                        self.assertEqual(mem[lane * 4], want, f"{tag} rev={rev} lane={lane}")

    def test_folded_execz_mask_hi_sgpr_live_scc_votes_upper_half(self):
        # Only lane 40 satisfies s[4:5] (MASK_HI bit 8); active EXEC inside region is lane 5 only.
        # s_and_b32 s10, s5, 0xffffffff must vote across lanes 32..63 -> SCC = 1.
        insns = [
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v4, 999", 4),
            ("v_cmp_eq_u32_e64 s[4:5], v0, 40", 8),
            ("v_cmp_eq_u32_e64 s[6:7], v0, 5", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 4", 4),
            ("s_and_b32 s10, s5, -1", 4),
            ("s_cselect_b32 s11, 333, 444", 4),
            ("v_mov_b32_e32 v4, s11", 4),
            ("s_cmp_eq_u32 0, 0", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_lshlrev_b32_e32 v10, 2, v0", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, _ = w64opt._translate_encoded("cs_mask_hi_scc", insns, (64, 1, 1), emulate=True, rsrc2=0x20088)
        _assemble_ok(self, spv)
        for rev, mem in _run_both_orders(spv, 64):
            for lane in range(64):
                want = 333 if lane == 5 else 999
                self.assertEqual(mem[lane * 4], want, f"rev={rev} lane={lane}")


class OppositeHalfB32MaskSrcBarrierTests(unittest.TestCase):
    def test_folded_execz_opposite_half_mask_reads_with_dead_scc(self):
        # VCC has lanes 34..37 set in the high half (bits 2..5 of vcc_hi) and lanes 0..1 set in low half.
        # Inside the EXECZ region (lanes 0..15 active), an opposite-half 32-bit mask op with DEAD SCC
        # transfers vcc_hi's bits into vcc_lo (calling _mask_b32_src -> wave_mask -> x_round).
        hi_bits = sum(1 << b for b in range(2, 6))  # lanes 34..37 -> low-half bits 2..5
        cases = [
            ("cselect", "s_cselect_b32 vcc_lo, vcc_hi, 0", hi_bits),
            ("and", "s_and_b32 vcc_lo, vcc_hi, 0x0000000c", hi_bits & 0x0C),
            ("not", "s_not_b32 vcc_lo, vcc_hi", (~hi_bits) & 0xFFFFFFFF),
            ("xor", "s_xor_b32 vcc_lo, vcc_hi, 0x0000000f", hi_bits ^ 0x0F),
        ]
        for tag, op_asm, want_lo in cases:
            insns = [
                ("s_mov_b64 exec, -1", 4),
                ("v_mov_b32_e32 v4, 999", 4),
                ("v_cmp_ge_u32_e64 s[14:15], v0, 34", 8),
                ("v_cmp_lt_u32_e64 s[16:17], v0, 38", 8),
                ("s_and_b64 vcc, s[14:15], s[16:17]", 4),
                ("v_cmp_lt_u32_e64 s[6:7], v0, 16", 8),
                ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
                ("s_cmp_eq_u32 1, 1", 4),  # SCC = 1 for s_cselect_b32
                ("s_cbranch_execz 3", 4),
                (op_asm, 4),
                ("v_cndmask_b32_e32 v4, 10, 20, vcc", 4),
                ("s_mov_b64 vcc, 0", 4),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("v_lshlrev_b32_e32 v10, 2, v0", 4),
                ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
                ("s_endpgm", 4),
            ]
            with self.subTest(case=tag):
                spv, _ = w64opt._translate_encoded(f"cs_opp_fold_{tag}", insns, (64, 1, 1), emulate=True, rsrc2=0x20088)
                _assemble_ok(self, spv)
                for rev, mem in _run_both_orders(spv, 64):
                    for lane in range(64):
                        if lane < 16:
                            want = 20 if ((want_lo >> lane) & 1) else 10
                        else:
                            want = 999
                        self.assertEqual(mem[lane * 4], want, f"{tag} rev={rev} lane={lane}")

    def test_multiwave_structured_hybrid_opposite_half_mask_reads_with_dead_scc(self):
        # 128-thread multi-wave divergent guard with opposite-half _mask_b32_src reads and DEAD SCC.
        # Wave 0 (v0 in 0..63) takes the guard; wave 1 (v0 in 64..127) skips it.
        hi_bits = sum(1 << b for b in range(4, 10))  # wave-local lanes 36..41 -> low-half bits 4..9
        cases = [
            ("cselect", "s_cselect_b32 vcc_lo, vcc_hi, 0", hi_bits),
            ("and", "s_and_b32 vcc_lo, vcc_hi, -1", hi_bits),
            ("not", "s_not_b32 vcc_lo, vcc_hi", (~hi_bits) & 0xFFFFFFFF),
            ("or_sgpr_hi", "s_or_b32 vcc_lo, s15, 1", hi_bits | 1),
        ]
        for tag, op_asm, want_lo in cases:
            insns = [
                ("s_mov_b64 exec, -1", 4),
                ("v_mov_b32_e32 v4, 777", 4),
                ("v_and_b32_e32 v1, 63, v0", 4),
                ("v_cmp_ge_u32_e64 s[14:15], v1, 36", 8),
                ("v_cmp_lt_u32_e64 s[16:17], v1, 42", 8),
                ("s_and_b64 s[14:15], s[14:15], s[16:17]", 4),
                ("s_mov_b64 vcc, s[14:15]", 4),
                ("v_readfirstlane_b32 s4, v0", 4),
                ("s_cmp_lt_u32 s4, 32", 4),  # SCC = 1 for wave 0, 0 for wave 1
                ("s_cbranch_scc0 2", 4),
                (op_asm, 4),
                ("v_cndmask_b32_e32 v4, 10, 20, vcc", 4),
                ("v_lshlrev_b32_e32 v10, 2, v0", 4),
                ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
                ("s_endpgm", 4),
            ]
            with self.subTest(case=tag):
                spv, gen = w64opt._translate_encoded(f"cs_mw_opp_{tag}", insns, (128, 1, 1), emulate=True, rsrc2=0x20088)
                self.assertEqual(gen.control_flow, "emulated wave64 structured")
                self.assertEqual(gen.emulate_plan["mode"], "structured_hybrid")
                self.assertGreater(len(gen.emulate_plan["mini_blocks"]), 0)
                self.assertEqual(len(gen.emulate_plan["sync_free_blocks"]), 0)
                _assemble_ok(self, spv)
                for rev, mem in _run_both_orders(spv, 128):
                    for lane in range(128):
                        if lane < 32:
                            want = 20 if ((want_lo >> lane) & 1) else 10
                        elif lane < 64:
                            want = 20 if (36 <= lane < 42) else 10
                        else:
                            want = 777
                        self.assertEqual(mem[lane * 4], want, f"{tag} rev={rev} lane={lane}")

    def test_multiwave_exec_hi_dest_with_lo_mask_src_dead_scc(self):
        # Opposite direction (lo -> hi): s_cselect_b32 exec_hi, s14, 0 where s14 is MASK_LO (lanes 3..6).
        insns = [
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v4, 555", 4),
            ("v_and_b32_e32 v1, 63, v0", 4),
            ("v_cmp_ge_u32_e64 s[14:15], v1, 3", 8),
            ("v_cmp_lt_u32_e64 s[16:17], v1, 7", 8),
            ("s_and_b64 s[14:15], s[14:15], s[16:17]", 4),
            ("v_readfirstlane_b32 s4, v0", 4),
            ("s_cmp_lt_u32 s4, 32", 4),
            ("s_cbranch_scc0 3", 4),
            ("s_cselect_b32 exec_hi, s14, 0", 4),
            ("v_mov_b32_e32 v4, 888", 4),
            ("s_mov_b64 exec, -1", 4),
            ("v_lshlrev_b32_e32 v10, 2, v0", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = w64opt._translate_encoded("cs_mw_lo_to_hi", insns, (128, 1, 1), emulate=True, rsrc2=0x20088)
        self.assertEqual(gen.control_flow, "emulated wave64 structured")
        self.assertEqual(gen.emulate_plan["mode"], "structured_hybrid")
        self.assertGreater(len(gen.emulate_plan["mini_blocks"]), 0)
        _assemble_ok(self, spv)
        for rev, mem in _run_both_orders(spv, 128):
            for lane in range(128):
                if lane < 32:
                    want = 888
                elif lane < 64:
                    want = 888 if (3 <= (lane - 32) < 7) else 555
                else:
                    want = 555
                self.assertEqual(mem[lane * 4], want, f"rev={rev} lane={lane}")


class SaveexecWrexecB32OppositeHalfBarrierTests(unittest.TestCase):
    def test_folded_execz_saveexec_and_wrexec_b32_hi_src_dead_scc(self):
        # VCC_HI has bits 2..5 set (lanes 34..37). Outer EXECZ activates lanes 0..9 and 40..49.
        # Inside the region (dead SCC):
        #   - s_and_saveexec_b32 s10, vcc_hi: EXEC_LO &= vcc_hi (lanes 2..5 survive); EXEC_HI untouched (40..49).
        #   - s_andn1_wrexec_b32 s10, vcc_hi: EXEC_LO = ~vcc_hi & EXEC_LO (lanes 0..1, 6..9 survive); EXEC_HI untouched.
        cases = [
            ("saveexec", "s_and_saveexec_b32 s10, vcc_hi", lambda l: (2 <= l < 6) or (40 <= l < 50)),
            ("wrexec", "s_andn1_wrexec_b32 s10, vcc_hi", lambda l: (l < 10 and not (2 <= l < 6)) or (40 <= l < 50)),
        ]
        for tag, op_asm, want_pred in cases:
            insns = [
                ("s_mov_b64 exec, -1", 4),
                ("v_mov_b32_e32 v4, 999", 4),
                ("v_cmp_ge_u32_e64 s[14:15], v0, 34", 8),
                ("v_cmp_lt_u32_e64 s[16:17], v0, 38", 8),
                ("s_and_b64 vcc, s[14:15], s[16:17]", 4),
                ("v_cmp_lt_u32_e64 s[6:7], v0, 10", 8),
                ("v_cmp_ge_u32_e64 s[18:19], v0, 40", 8),
                ("v_cmp_lt_u32_e64 s[20:21], v0, 50", 8),
                ("s_and_b64 s[18:19], s[18:19], s[20:21]", 4),
                ("s_or_b64 s[6:7], s[6:7], s[18:19]", 4),
                ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
                ("s_cbranch_execz 3", 4),
                (op_asm, 4),
                ("v_mov_b32_e32 v4, 77", 4),
                ("s_mov_b32 s10, 0", 4),
                ("s_or_b64 exec, exec, s[8:9]", 4),
                ("v_lshlrev_b32_e32 v10, 2, v0", 4),
                ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
                ("s_endpgm", 4),
            ]
            with self.subTest(case=tag):
                spv, _ = w64opt._translate_encoded(f"cs_fold_{tag}_b32", insns, (64, 1, 1), emulate=True, rsrc2=0x20088)
                _assemble_ok(self, spv)
                for rev, mem in _run_both_orders(spv, 64):
                    for lane in range(64):
                        want = 77 if want_pred(lane) else 999
                        self.assertEqual(mem[lane * 4], want, f"{tag} rev={rev} lane={lane}")

    def test_multiwave_structured_hybrid_saveexec_wrexec_b32_hi_src_and_dst_dead_scc(self):
        # 128-thread multi-wave divergent guards exercising both opposite-half sources (vcc_hi) and
        # opposite-half destinations (vcc_hi / exec_hi) with dead SCC.
        # Initial wave-local mask s[14:15] = lanes 0..7 and 35..38 (lo=0x000000FF, hi=0x00000078).
        # Initial EXEC before guard has EXEC_LO = 0x00000FFF (lanes 0..11) and EXEC_HI = 0xFFFFFFFF (lanes 32..63).
        # Wave 0 enters guard, executes op, reads vcc / active lane state, then restores EXEC = -1.
        cases = [
            # s_and_saveexec_b32 s10, vcc_hi: EXEC_LO = 0x0FFF & 0x78 = 0x78 (lanes 3..6), EXEC_HI unchanged
            ("saveexec_hi_src", "s_and_saveexec_b32 s10, vcc_hi", False,
             lambda l: (3 <= l < 7) or (32 <= l < 64)),
            # s_andn1_wrexec_b32 s10, vcc_hi: EXEC_LO = ~0x78 & 0x0FFF (lanes 0..2, 7..11), EXEC_HI unchanged
            ("wrexec_hi_src", "s_andn1_wrexec_b32 s10, vcc_hi", False,
             lambda l: (l < 12 and not (3 <= l < 7)) or (32 <= l < 64)),
            # s_and_saveexec_b32 vcc_hi, s14: saves old EXEC_LO (lanes 0..11) into VCC_HI (lanes 32..43);
            # new EXEC_LO = 0x0FF & 0x0FFF = 0x0FF (lanes 0..7), EXEC_HI unchanged.
            ("saveexec_vcc_hi_dst", "s_and_saveexec_b32 vcc_hi, s14", True,
             lambda l: (l < 8) or (32 <= l < 44)),
            # s_andn2_wrexec_b32 vcc_hi, s14: EXEC_LO = s14 & ~EXEC_LO = 0x0FF & ~0x0FFF = 0;
            # writes new EXEC_LO (0) into VCC_HI (lanes 32..63 -> 0); VCC_LO untouched (lanes 0..7).
            ("wrexec_vcc_hi_dst", "s_andn2_wrexec_b32 vcc_hi, s14", True,
             lambda l: l < 8),
            # s_and_saveexec_b32 exec_hi, s14: saves old EXEC_LO (lanes 0..11) into EXEC_HI (lanes 32..43),
            # then sets EXEC_LO = 0x0FF & 0x0FFF = 0x0FF (lanes 0..7).
            ("saveexec_exec_hi_dst", "s_and_saveexec_b32 exec_hi, s14", False,
             lambda l: (l < 8) or (32 <= l < 44)),
        ]
        for tag, op_asm, check_vcc, want_pred in cases:
            if check_vcc:
                body = [
                    (op_asm, 4),
                    ("s_mov_b64 exec, -1", 4),
                    ("v_cndmask_b32_e32 v4, 555, 888, vcc", 4),
                ]
            else:
                body = [
                    (op_asm, 4),
                    ("v_mov_b32_e32 v4, 888", 4),
                    ("s_mov_b64 exec, -1", 4),
                ]
            insns = [
                ("s_mov_b64 exec, -1", 4),
                ("v_mov_b32_e32 v4, 555", 4),
                ("v_and_b32_e32 v1, 63, v0", 4),
                ("v_cmp_lt_u32_e64 s[14:15], v1, 8", 8),
                ("v_cmp_ge_u32_e64 s[16:17], v1, 35", 8),
                ("v_cmp_lt_u32_e64 s[18:19], v1, 39", 8),
                ("s_and_b64 s[16:17], s[16:17], s[18:19]", 4),
                ("s_or_b64 s[14:15], s[14:15], s[16:17]", 4),
                ("s_mov_b64 vcc, s[14:15]", 4),
                ("v_readfirstlane_b32 s4, v0", 4),
                ("s_cmp_lt_u32 s4, 32", 4),
                ("s_cbranch_scc0 5", 4),
                ("s_mov_b32 exec_lo, 0x00000fff", 8),
                *body,
                ("v_lshlrev_b32_e32 v10, 2, v0", 4),
                ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
                ("s_endpgm", 4),
            ]
            with self.subTest(case=tag):
                spv, gen = w64opt._translate_encoded(f"cs_mw_{tag}", insns, (128, 1, 1), emulate=True, rsrc2=0x20088)
                self.assertEqual(gen.control_flow, "emulated wave64 structured")
                self.assertEqual(gen.emulate_plan["mode"], "structured_hybrid")
                self.assertGreater(len(gen.emulate_plan["mini_blocks"]), 0)
                self.assertEqual(len(gen.emulate_plan["sync_free_blocks"]), 0)
                _assemble_ok(self, spv)
                for rev, mem in _run_both_orders(spv, 128):
                    for lane in range(128):
                        want = (888 if want_pred(lane) else 555) if lane < 64 else 555
                        self.assertEqual(mem[lane * 4], want, f"{tag} rev={rev} lane={lane}")


class NegativeControlFoldingPreservedTests(unittest.TestCase):
    def test_plain_data_b32_logic_live_scc_still_folds_and_stays_sync_free(self):
        # Plain scalar data 32-bit logic with live SCC (including vcc_lo when vcc is NOT a lane mask)
        # must still fold in 64-thread EXECZ and stay sync_free in 128-thread structured_hybrid.
        insns_fold = [
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v4, 999", 4),
            ("s_mov_b32 s20, 0x15", 4),
            ("s_mov_b32 s21, 0x04", 4),
            ("v_cmp_lt_u32_e64 s[6:7], v0, 16", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 5", 4),
            ("s_and_b32 s10, s20, s21", 4),
            ("s_cselect_b32 s11, 111, 222", 4),
            ("v_mov_b32_e32 v4, s11", 4),
            ("s_mov_b32 s10, 0", 4),
            ("s_cmp_eq_u32 0, 0", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_lshlrev_b32_e32 v10, 2, v0", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv, gen = w64opt._translate_encoded("cs_plain_b32_fold", insns_fold, (64, 1, 1), emulate=True, rsrc2=0x20088)
        self.assertEqual(w64opt._switch_case_count(gen), 1)
        _assemble_ok(self, spv)
        for rev, mem in _run_both_orders(spv, 64):
            for lane in range(64):
                self.assertEqual(mem[lane * 4], 111 if lane < 16 else 999, f"rev={rev} lane={lane}")

        # 128-thread structured_hybrid: vcc holds plain data (never written as lane mask).
        insns_mw = [
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v4, 777", 4),
            ("s_mov_b32 vcc_lo, 0x15", 4),
            ("v_readfirstlane_b32 s4, v0", 4),
            ("s_cmp_lt_u32 s4, 32", 4),
            ("s_cbranch_scc0 3", 4),
            ("s_and_b32 vcc_lo, 0x0f, 0x05", 4),
            ("s_cselect_b32 s11, 111, 222", 4),
            ("v_mov_b32_e32 v4, s11", 4),
            ("v_lshlrev_b32_e32 v10, 2, v0", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv_mw, gen_mw = w64opt._translate_encoded("cs_plain_b32_mw", insns_mw, (128, 1, 1), emulate=True, rsrc2=0x20088)
        self.assertEqual(gen_mw.control_flow, "emulated wave64 structured")
        self.assertEqual(gen_mw.emulate_plan["mode"], "structured_hybrid")
        self.assertEqual(len(gen_mw.emulate_plan["sync_free_blocks"]), 1)
        self.assertEqual(len(gen_mw.emulate_plan["mini_blocks"]), 0)
        _assemble_ok(self, spv_mw)
        for rev, mem in _run_both_orders(spv_mw, 128):
            for lane in range(128):
                self.assertEqual(mem[lane * 4], 111 if lane < 64 else 777, f"rev={rev} lane={lane}")

    def test_same_half_b32_mask_ops_dead_scc_still_fold_and_stay_sync_free(self):
        # Same-half 32-bit mask ops with DEAD SCC emit no wave_mask and no scc_any_half,
        # so they must continue to fold (64 threads) and stay sync_free (128 threads).
        cases = [
            ("cselect_lo", "s_cselect_b32 vcc_lo, vcc_lo, 0", lambda l: (l < 6) or (40 <= l < 44)),
            ("and_lo_imm", "s_and_b32 vcc_lo, vcc_lo, 0x0000000f", lambda l: (l < 4) or (40 <= l < 44)),
            ("not_lo", "s_not_b32 vcc_lo, vcc_lo", lambda l: (6 <= l < 32) or (40 <= l < 44)),
        ]
        for tag, op_asm, want_pred in cases:
            insns_mw = [
                ("s_mov_b64 exec, -1", 4),
                ("v_mov_b32_e32 v4, 777", 4),
                ("v_and_b32_e32 v1, 63, v0", 4),
                ("v_cmp_lt_u32_e64 s[14:15], v1, 6", 8),
                ("v_cmp_ge_u32_e64 s[16:17], v1, 40", 8),
                ("v_cmp_lt_u32_e64 s[18:19], v1, 44", 8),
                ("s_and_b64 s[16:17], s[16:17], s[18:19]", 4),
                ("s_or_b64 vcc, s[14:15], s[16:17]", 4),
                ("v_readfirstlane_b32 s4, v0", 4),
                ("s_cmp_lt_u32 s4, 32", 4),
                ("s_cbranch_scc0 2", 4),
                (op_asm, 4),
                ("v_cndmask_b32_e32 v4, 10, 20, vcc", 4),
                ("v_lshlrev_b32_e32 v10, 2, v0", 4),
                ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
                ("s_endpgm", 4),
            ]
            with self.subTest(case=tag):
                spv, gen = w64opt._translate_encoded(f"cs_same_{tag}", insns_mw, (128, 1, 1), emulate=True, rsrc2=0x20088)
                self.assertEqual(gen.control_flow, "emulated wave64 structured")
                self.assertEqual(gen.emulate_plan["mode"], "structured_hybrid")
                self.assertEqual(len(gen.emulate_plan["sync_free_blocks"]), 1)
                self.assertEqual(len(gen.emulate_plan["mini_blocks"]), 0)
                _assemble_ok(self, spv)
                for rev, mem in _run_both_orders(spv, 128):
                    for lane in range(128):
                        want = (20 if want_pred(lane) else 10) if lane < 64 else 777
                        self.assertEqual(mem[lane * 4], want, f"{tag} rev={rev} lane={lane}")

        # Same-half s_and_saveexec_b32 with low-half mask src and plain SGPR dst, dead SCC -> still folds!
        insns_se = [
            ("s_mov_b64 exec, -1", 4),
            ("v_mov_b32_e32 v4, 999", 4),
            ("v_cmp_lt_u32_e64 s[14:15], v0, 4", 8),
            ("v_cmp_lt_u32_e64 s[6:7], v0, 10", 8),
            ("s_and_saveexec_b64 s[8:9], s[6:7]", 4),
            ("s_cbranch_execz 3", 4),
            ("s_and_saveexec_b32 s10, s14", 4),
            ("v_mov_b32_e32 v4, 42", 4),
            ("s_mov_b32 s10, 0", 4),
            ("s_or_b64 exec, exec, s[8:9]", 4),
            ("v_lshlrev_b32_e32 v10, 2, v0", 4),
            ("buffer_store_dword v4, v10, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv_se, gen_se = w64opt._translate_encoded("cs_same_saveexec_b32", insns_se, (64, 1, 1), emulate=True, rsrc2=0x20088)
        self.assertEqual(w64opt._switch_case_count(gen_se), 1)
        _assemble_ok(self, spv_se)
        for rev, mem in _run_both_orders(spv_se, 64):
            for lane in range(64):
                self.assertEqual(mem[lane * 4], 42 if lane < 4 else 999, f"rev={rev} lane={lane}")

    def test_b32_mask_op_may_barrier_edge_cases(self):
        fn = wave64_emulate_opt._b32_mask_op_may_barrier
        vcc_lo = {"k": "special", "t": "vcc_lo"}
        vcc_hi = {"k": "special", "t": "vcc_hi"}
        scc_op = {"k": "special", "t": "scc"}
        s10 = {"k": "sgpr", "r": 10, "n": 1}
        s4 = {"k": "sgpr", "r": 4, "n": 1}
        s5 = {"k": "sgpr", "r": 5, "n": 1}
        lm_vcc = {"vcc": lane_masks.MASK, 4: lane_masks.MASK_LO, 5: lane_masks.MASK_HI}

        # Non-_maskable_b32 source (scc): no opposite-half _mask_b32_src when SCC dead, votes when SCC live.
        ins = {"op": "s_and_b32", "args": [vcc_lo, vcc_hi, scc_op]}
        self.assertFalse(fn(ins, "s_and_b32", ins["args"], lm_vcc, set()))
        self.assertTrue(fn(ins, "s_and_b32", ins["args"], lm_vcc, {id(ins)}))

        # Mixed halves (lo + hi) into non-mask SGPR dest with live SCC: scc_logic falls back to scc_nonzero (no vote).
        ins_mix = {"op": "s_and_b32", "args": [s10, s4, s5]}
        self.assertFalse(fn(ins_mix, "s_and_b32", ins_mix["args"], lm_vcc, {id(ins_mix)}))

        # Single half into non-mask SGPR dest: votes when SCC live, no barrier when SCC dead.
        ins_one = {"op": "s_and_b32", "args": [s10, s4, {"k": "imm", "v": 1}]}
        self.assertFalse(fn(ins_one, "s_and_b32", ins_one["args"], lm_vcc, set()))
        self.assertTrue(fn(ins_one, "s_and_b32", ins_one["args"], lm_vcc, {id(ins_one)}))

        # vcc_lo dest with plain data vcc (not in lm) and exec_hi source -> to_mask_dst is True, opposite half -> barrier!
        ins_vcc_promote = {"op": "s_and_b32", "args": [vcc_lo, {"k": "special", "t": "exec_hi"}, {"k": "imm", "v": 1}]}
        self.assertTrue(fn(ins_vcc_promote, "s_and_b32", ins_vcc_promote["args"], {}, set()))


if __name__ == "__main__":
    unittest.main()
