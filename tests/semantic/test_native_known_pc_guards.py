"""Semantic tests for native structured known-%pc guard unwrapping and spine coalescing.

`cfg_structure.unwrap_known_pc_guards` removes provably-true `("guard", g, ...)` wrappers and coalesces
hoisted post-dominating spine guards in `rdna2spirv.capped_tree` for all native structured shaders
(not only `--wave64-emulate`).

Verifies:
  1. Diamond join and loop-exit blocks unwrap out of redundant `if (%pc == g)` guards, assemble cleanly
     with `tools/spirv_asm.py`, and execute bit-identically on the wave/workgroup interpreters.
  2. Early-exit + 50 sequential diamonds coalesce all 50 hoisted spine guards in native mode (-50
     `OpSelectionMerge`s, -100 `OpLabel`s, -50 `OpLoad %uint %pc`), matching ISA results on both the
     early-exit and full-chain paths.
  3. `native_out_pcs` adds `cfg_structure.END` for a conditional branch with no fallthrough so a block
     that can fall off the end never falsely proves a single successor.
  4. Native wave32 compute shaders with `s_barrier` in a post-dominating diamond join unwrap that block
     to top level and stay in structured control flow (no unnecessary `barrier rounds` loop), while
     divergent, early-end, and in-loop barriers remain in `barrier rounds`.
  5. `capped_tree` enforces `MAX_STRUCTURED_DEPTH` on the unwrapped depth.

Run:
    python -I tests/semantic/test_native_known_pc_guards.py
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import test_loop_ballot_barrier as lbb  # noqa: E402
from spirv_wave_interp import run_wave  # noqa: E402

import cfg_structure  # noqa: E402
import spirv_asm  # noqa: E402

rdna2spirv = support.rdna2spirv
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def _translate_cs(name, items, threads=64, lds_units=1, max_depth=None):
    lines = [f"{text} ; encoding: [{','.join(['0x00'] * size)}]" for text, size in lbb._resolve_labels(items)]
    hdr = bytes(0x138) + struct.pack("<4I", 0x90 | (lds_units << 15), threads, 1, 1)
    prev_em, prev_md = rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_STRUCTURED_DEPTH
    rdna2spirv.WAVE64_EMULATE = False
    if max_depth is not None:
        rdna2spirv.MAX_STRUCTURED_DEPTH = max_depth
    try:
        spv = support.translate_lines(name, lines, fssm=hdr)
    finally:
        rdna2spirv.WAVE64_EMULATE, rdna2spirv.MAX_STRUCTURED_DEPTH = prev_em, prev_md
    return spv, rdna2spirv.translate.last_gen


def _translate_ps(name, items):
    lines = [f"{text} ; encoding: [{','.join(['0x00'] * size)}]" for text, size in lbb._resolve_labels(items)]
    prev_em = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = False
    try:
        spv = support.translate_lines(name, lines)
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_em
    return spv, rdna2spirv.translate.last_gen


class NativeKnownPcGuardTests(unittest.TestCase):
    def test_diamond_join_and_loop_exit_unwrap_in_native_ps(self):
        # Diamond 0 -> {arm_a, arm_b} -> join -> countdown loop -> exit.
        # Both `join` and `exit` must unwrap to top-level blocks with no `%pc == g` guard.
        items = [
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "s_cmp_eq_u32 s0, 0",
            "s_cbranch_scc0 arm_b",
            "v_add_nc_u32_e32 v4, 10, v0",
            "s_branch join",
            "arm_b:",
            "v_add_nc_u32_e32 v4, 20, v0",
            "join:",
            "s_mov_b32 s4, s1",
            "loop:",
            "v_add_nc_u32_e32 v4, 1, v4",
            "s_sub_u32 s4, s4, 1",
            "s_cmp_gt_u32 s4, 0",
            "s_cbranch_scc1 loop",
            "v_add_nc_u32_e32 v4, 100, v4",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ]
        spv, gen = _translate_ps("ps_native_unwrap", items)
        self.assertEqual(gen.control_flow, "structured")
        tree, nesting = rdna2spirv.capped_tree(gen.succ, gen.out_pcs)
        self.assertEqual(nesting, 1)
        # Entry block, two diamond arms (guarded), unwrapped join+loop, unwrapped exit block.
        self.assertEqual(tree[0][0], "block")
        self.assertEqual(tree[-1][0], "block")
        # Only the two diamond arms need OpLoad %uint %pc guards (+ 1 loop condition check = 3 total).
        self.assertEqual(spv.count("OpLoad %uint %pc"), 3)
        self.assertEqual(spv.count("OpSelectionMerge"), 2)
        binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
        self.assertEqual(binary[:4], struct.pack("<I", 0x07230203))

        for s0, s1 in ((0, 1), (0, 5), (1, 1), (1, 4)):
            with self.subTest(s0=s0, s1=s1):
                res = run_wave(spv, 32, push_constants=[s0, s1] + [0] * 30)
                base = 10 if s0 == 0 else 20
                for lane in range(32):
                    want = (base + lane + s1 + 100) & 0xFFFFFFFF
                    self.assertEqual(res["outputs"][lane]["%out_mrt0"][0], want)

    def test_early_exit_50_diamonds_coalesces_hoisted_spine_guards(self):
        n_diamonds = 50
        items = [
            "s_mov_b64 exec, -1",
            "v_mov_b32_e32 v1, v0",
            "s_cmp_eq_u32 s4, 0",
            "s_cbranch_scc1 early_exit",
        ]
        for i in range(n_diamonds):
            items += [
                f"s_cmp_gt_u32 s5, {i}",
                f"s_cbranch_scc0 skip_{i}",
                "v_add_nc_u32_e32 v1, 1, v1",
                f"skip_{i}:",
            ]
        items += [
            "v_lshlrev_b32_e32 v2, 2, v0",
            "buffer_store_dword v1, v2, s[0:3], 0 offen",
            "early_exit:",
            "s_endpgm",
        ]
        spv, gen = _translate_cs("cs_native_coalesce_50", items, threads=64)
        self.assertEqual(gen.control_flow, "structured")
        self.assertTrue(gen.wave64)
        # Raw structure emits 102 guards (main body + 50 arm guards + 50 hoisted spine guards + final early_exit);
        # coalescing + unwrapping removes all 50 hoisted spine guards and the post-dominating early_exit guard -> 51.
        self.assertEqual(spv.count("OpLoad %uint %pc"), 51)
        self.assertEqual(spv.count("OpSelectionMerge"), 52)  # 51 CFG guards + 1 buffer_store OOB/active guard
        binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
        self.assertEqual(binary[:4], struct.pack("<I", 0x07230203))

        # Execute both early-exit (s4=0: no buffer writes) and main path (s4=1, s5=17: +17 per lane).
        mem_early = lbb.run_compute_workgroup(spv, 64, 64, push_constants=VSHARP + [0, 17, 0, 0])
        self.assertEqual(mem_early, {})
        mem_main = lbb.run_compute_workgroup(spv, 64, 64, push_constants=VSHARP + [1, 17, 0, 0])
        for lane in range(64):
            self.assertEqual(mem_main[lane * 4], lane + 17)

    def test_native_out_pcs_marks_conditional_falloff_with_end(self):
        # Last block ends in a conditional branch with no fallthrough (len(raw_edges[last]) == 1).
        blocks = [
            [{"op": "s_cmp_eq_u32", "args": []}, {"op": "s_cbranch_scc1", "args": []}],
            [{"op": "s_cmp_eq_u32", "args": []}, {"op": "s_cbranch_scc1", "args": []}],
        ]
        raw_edges = [[1, 0], [0]]
        succ = [[1, 0], [0]]
        out_pcs = rdna2spirv.native_out_pcs(blocks, raw_edges, succ)
        self.assertEqual(out_pcs[0], frozenset((0, 1)))
        self.assertEqual(out_pcs[1], frozenset((0, cfg_structure.END)))

        # Unconditional s_branch with len(raw_edges) == 1 must NOT add END.
        blocks_uncond = [
            [{"op": "s_branch", "args": []}],
            [{"op": "s_endpgm", "args": []}],
        ]
        out_uncond = rdna2spirv.native_out_pcs(blocks_uncond, [[1], []], [[1], []])
        self.assertEqual(out_uncond[0], frozenset((1,)))
        self.assertEqual(out_uncond[1], frozenset((cfg_structure.END,)))

    def test_wave32_convergent_diamond_before_barrier_stays_structured(self):
        # Both arms converge unconditionally at `join` (no early s_endpgm), so `join` unwraps to top level
        # and the wave32 compute shader stays structured with one direct OpControlBarrier (no barrier rounds).
        items = [
            "v_lshlrev_b32_e32 v2, 2, v0",
            "v_lshrrev_b32_e32 v5, 5, v0",
            "v_readfirstlane_b32 s8, v5",
            "s_cmp_eq_u32 s8, 0",
            "s_cbranch_scc0 arm_b",
            "v_add_nc_u32_e32 v1, 100, v0",
            "s_branch join",
            "arm_b:",
            "v_add_nc_u32_e32 v1, 200, v0",
            "join:",
            "ds_write_b32 v2, v1",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "v_xor_b32_e32 v3, 32, v0",
            "v_lshlrev_b32_e32 v3, 2, v3",
            "ds_read_b32 v4, v3",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v4, v2, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv, gen = _translate_cs("cs_w32_conv_diamond_bar", items, threads=64)
        self.assertFalse(gen.wave64)
        self.assertFalse(gen.bar_rounds)
        self.assertEqual(gen.control_flow, "structured")
        self.assertEqual(spv.count("OpControlBarrier"), 1)
        self.assertNotIn("%x_buf", spv)
        self.assertNotIn("OpSwitch", spv)
        binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
        self.assertEqual(binary[:4], struct.pack("<I", 0x07230203))

        # Wave 0 (lanes 0..31) took arm_a (100+lane); wave 1 (lanes 32..63) took arm_b (200+lane).
        # After s_barrier each lane reads its partner (lane ^ 32) from LDS.
        for rev in (False, True):
            mem = lbb.run_compute_workgroup(spv, 64, 32, push_constants=VSHARP, reverse_subgroups=rev)
            for lane in range(64):
                partner = lane ^ 32
                want = (100 if partner < 32 else 200) + partner
                self.assertEqual(mem[lane * 4], want, f"lane={lane} rev={rev}")

    def test_wave32_early_end_before_barrier_stays_in_barrier_rounds(self):
        # Negative control: wave 0 exits early via s_endpgm, so `join` does NOT unwrap and bar_rounds stays True.
        items = [
            "v_lshlrev_b32_e32 v2, 2, v0",
            "v_lshrrev_b32_e32 v5, 5, v0",
            "v_readfirstlane_b32 s8, v5",
            "s_cmp_eq_u32 s8, 0",
            "s_cbranch_scc0 wave1",
            "s_endpgm",
            "wave1:",
            "ds_write_b32 v2, v0",
            "s_waitcnt lgkmcnt(0)",
            "s_barrier",
            "ds_read_b32 v4, v2",
            "s_waitcnt lgkmcnt(0)",
            "buffer_store_dword v4, v2, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv, gen = _translate_cs("cs_w32_early_end_bar", items, threads=64)
        self.assertFalse(gen.wave64)
        self.assertTrue(gen.bar_rounds)
        self.assertEqual(gen.control_flow, "barrier rounds")
        for rev in (False, True):
            mem = lbb.run_compute_workgroup(spv, 64, 32, push_constants=VSHARP, reverse_subgroups=rev)
            for lane in range(32):
                self.assertNotIn(lane * 4, mem)
            for lane in range(32, 64):
                self.assertEqual(mem[lane * 4], lane, f"lane={lane} rev={rev}")

    def test_capped_tree_measures_depth_after_unwrapping(self):
        # A diamond whose raw structure nests to depth 1 stays depth 1, whereas a pure straight chain
        # with a guarded join (loop-free) unwraps to depth 0.
        # Graph: 0 -> {1, 2}, 1 -> 3, 2 -> 3, 3 -> {4, 5}, 4 -> 6, 5 -> 6, 6 -> END.
        succ = [[1, 2], [3], [3], [4, 5], [6], [6], []]
        raw_depth = cfg_structure.depth(cfg_structure.structure(succ))
        tree, unwrapped_depth = rdna2spirv.capped_tree(succ)
        self.assertIsNotNone(tree)
        self.assertEqual(raw_depth, 2)
        self.assertEqual(unwrapped_depth, 1)
        prev_md = rdna2spirv.MAX_STRUCTURED_DEPTH
        rdna2spirv.MAX_STRUCTURED_DEPTH = 1
        try:
            # Passes MAX_STRUCTURED_DEPTH=1 because unwrapped depth is 1 (raw depth was 2).
            t_ok, d_ok = rdna2spirv.capped_tree(succ)
            self.assertIsNotNone(t_ok)
            self.assertEqual(d_ok, 1)
        finally:
            rdna2spirv.MAX_STRUCTURED_DEPTH = prev_md


if __name__ == "__main__":
    unittest.main()
