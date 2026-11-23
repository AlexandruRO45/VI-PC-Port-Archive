"""Soundness regression tests for readfirstlane/readlane uniformity in tools/wave64_halves.py.

Checks four cases where unsafe wave64 VS/PS programs used to be marked uniform (`st["u"]` / `st["ua"]`)
and accepted as Tier A or B (`readfirstlane-uniform` / `readlane-uniform`), together with clean controls
that must remain Tier B:
  1. Unsigned RMW accumulating integer dot products (`v_dot2c_u32_u16`, `v_dot4c_u32_u8`, `v_dot8c_u32_u4`).
  2. Non-subset EXEC widening (`s_wqm_b64 exec, exec`, `s_or_saveexec_b64`, `s_xor_saveexec_b64`).
  3. Launch-uniform VGPR read via `v_readfirstlane_b32` under narrowed EXEC escaping past EXEC restore.
  4. Non-X tainted scalars (`lane-count`) copied into a VGPR must not qualify for `readlane-uniform`.

Standard library only; no GPU, no game files.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
from test_wave64_halves import (  # noqa: E402
    meta_of,
    out_param,
    ps,
    run_cut,
    vs,
)


class Wave64HalvesUniformSoundnessTests(unittest.TestCase):
    def assert_valid_spirv(self, asm):
        words = spirv_asm.assemble(asm, target_env="vulkan1.2")
        self.assertGreater(len(words), 20)

    def assert_tier_c_vs_diverges(self, name, body, *, contains=None, forbidden_idiom=None):
        (tier, reasons, idioms), asm = meta_of(name, vs(body))
        w = rdna2spirv.translate.last_gen.w64g
        self.assert_valid_spirv(asm)
        self.assertEqual(tier, "C", f"{name}: expected tier C, got {tier} (reasons={reasons}, idioms={idioms})")
        if contains is not None:
            self.assertTrue(set(contains) <= set(reasons), f"{name}: reasons {reasons} lack {contains}")
        if forbidden_idiom is not None:
            self.assertNotIn(forbidden_idiom, idioms, f"{name}: unexpected idiom {idioms}")
            if forbidden_idiom == "readlane-uniform":
                self.assertEqual(w["uniform_readlane"], set(), f"{name}: unexpected uniform_readlane rewrite")
        whole = out_param(run_cut(asm, 64))
        halves = out_param(run_cut(asm, 32))
        self.assertNotEqual(whole, halves, f"{name}: expected 1x64 and 2x32 outputs to diverge")

    def assert_ps_tier_c(self, name, body, *, contains=None, forbidden_idiom=None):
        (tier, reasons, idioms), asm = meta_of(name, ps(body))
        w = rdna2spirv.translate.last_gen.w64g
        self.assert_valid_spirv(asm)
        self.assertEqual(tier, "C", f"{name}: expected tier C, got {tier} (reasons={reasons}, idioms={idioms})")
        if contains is not None:
            self.assertTrue(set(contains) <= set(reasons), f"{name}: reasons {reasons} lack {contains}")
        if forbidden_idiom is not None:
            self.assertNotIn(forbidden_idiom, idioms, f"{name}: unexpected idiom {idioms}")
            if forbidden_idiom == "readlane-uniform":
                self.assertEqual(w["uniform_readlane"], set(), f"{name}: unexpected uniform_readlane rewrite")

    def assert_ps_tier_b(self, name, body, *, expected_idioms=("readfirstlane-uniform",)):
        (tier, reasons, idioms), asm = meta_of(name, ps(body))
        self.assert_valid_spirv(asm)
        self.assertEqual(tier, "B", f"{name}: expected tier B, got {tier} (reasons={reasons}, idioms={idioms})")
        self.assertEqual(reasons, [], f"{name}: unexpected reasons {reasons}")
        self.assertEqual(sorted(idioms), sorted(expected_idioms), f"{name}: unexpected idioms {idioms}")

    def test_unsigned_dot_rmw_of_varying_vgpr_not_uniform_and_diverges(self):
        for op in ("v_dot2c_u32_u16", "v_dot4c_u32_u8", "v_dot8c_u32_u4"):
            with self.subTest(op=op):
                self.assert_tier_c_vs_diverges(
                    f"vs_{op}_rmw_rfl",
                    [
                        "v_mov_b32_e32 v4, 1",
                        "v_mov_b32_e32 v5, v0",
                        f"{op} v5, 0, v4",
                        "v_readfirstlane_b32 s6, v5",
                        "v_mov_b32_e32 v6, s6",
                    ],
                    contains=["cross-lane"],
                    forbidden_idiom="readfirstlane-uniform",
                )

    def test_unsigned_dot_rmw_preserves_lane_id_taint_and_diverges(self):
        self.assert_tier_c_vs_diverges(
            "vs_v_dot2c_u32_u16_lane_id",
            [
                "v_mov_b32_e32 v4, 1",
                "v_mbcnt_lo_u32_b32 v6, -1, 0",
                "v_mbcnt_hi_u32_b32 v6, -1, v6",
                "v_dot2c_u32_u16 v6, 0, v4",
            ],
            contains=["lane-id"],
        )

    def test_unsigned_dot_rmw_clean_control_stays_tier_b(self):
        (tier, reasons, idioms), asm = meta_of(
            "vs_v_dot2c_u32_u16_clean",
            vs([
                "v_mov_b32_e32 v4, 2",
                "v_mov_b32_e32 v5, s4",
                "v_dot2c_u32_u16 v5, 3, v4",
                "v_readfirstlane_b32 s6, v5",
                "v_mov_b32_e32 v6, s6",
            ]),
        )
        self.assert_valid_spirv(asm)
        self.assertEqual(tier, "B", f"got tier {tier} (reasons={reasons}, idioms={idioms})")
        self.assertEqual(reasons, [], reasons)
        self.assertIn("readfirstlane-uniform", idioms)
        self.assertEqual(out_param(run_cut(asm, 64)), out_param(run_cut(asm, 32)))

    def test_readfirstlane_after_s_wqm_widening_is_tier_c(self):
        self.assert_ps_tier_c(
            "ps_wqm_rfl_before",
            [
                "v_mov_b32_e32 v5, s4",
                "s_wqm_b64 exec, exec",
                "v_readfirstlane_b32 s20, v5",
                "v_mov_b32_e32 v6, s20",
            ],
            contains=["cross-lane"],
            forbidden_idiom="readfirstlane-uniform",
        )

    def test_readlane_after_s_wqm_widening_is_tier_c(self):
        self.assert_ps_tier_c(
            "ps_wqm_readlane_before",
            [
                "v_mov_b32_e32 v5, s4",
                "s_wqm_b64 exec, exec",
                "v_readlane_b32 s20, v5, 0",
                "v_mov_b32_e32 v6, s20",
            ],
            contains=["cross-lane"],
            forbidden_idiom="readlane-uniform",
        )

    def test_readfirstlane_after_non_subset_saveexec_is_tier_c(self):
        for tag, op in (("or", "s_or_saveexec_b64"), ("xor", "s_xor_saveexec_b64")):
            with self.subTest(op=op):
                self.assert_ps_tier_c(
                    f"ps_{tag}_saveexec_rfl",
                    [
                        "v_mov_b32_e32 v5, s4",
                        f"{op} s[12:13], s[10:11]",
                        "v_readfirstlane_b32 s20, v5",
                        "v_mov_b32_e32 v6, s20",
                    ],
                    contains=["cross-lane"],
                    forbidden_idiom="readfirstlane-uniform",
                )

    def test_readfirstlane_written_after_s_wqm_stays_tier_b(self):
        self.assert_ps_tier_b(
            "ps_wqm_rfl_written_after",
            [
                "s_wqm_b64 exec, exec",
                "v_mov_b32_e32 v5, s4",
                "v_readfirstlane_b32 s20, v5",
                "v_add_nc_u32_e32 v6, s20, v0",
            ],
        )

    def test_readfirstlane_launch_uniform_under_narrow_exec_escape_is_tier_c(self):
        self.assert_ps_tier_c(
            "ps_rfl_launch_uniform_narrow_escape",
            [
                "v_mov_b32_e32 v5, s4",
                "v_cmp_eq_u32_e32 vcc, 1, v0",
                "s_and_saveexec_b64 s[12:13], vcc",
                "v_readfirstlane_b32 s20, v5",
                "s_mov_b64 exec, s[10:11]",
                "v_mov_b32_e32 v6, s20",
            ],
            contains=["cross-lane"],
        )

    def test_readfirstlane_launch_uniform_used_inside_narrow_exec_stays_tier_b(self):
        self.assert_ps_tier_b(
            "ps_rfl_launch_uniform_narrow_inside",
            [
                "v_mov_b32_e32 v5, s4",
                "v_cmp_eq_u32_e32 vcc, 1, v0",
                "s_and_saveexec_b64 s[12:13], vcc",
                "v_readfirstlane_b32 s20, v5",
                "v_add_nc_u32_e32 v6, s20, v0",
                "s_mov_b64 exec, s[10:11]",
            ],
        )

    def test_tainted_scalar_vgpr_not_marked_uniform_for_readlane(self):
        # Lane 0 runs in both halves of a 32-wide wave, proving uniform_readlane stays empty and 1x64 != 2x32.
        self.assert_tier_c_vs_diverges(
            "vs_cnt_vgpr_readlane_0",
            [
                "v_cmp_lt_u32_e32 vcc, 4, v0",
                "s_bcnt1_i32_b64 s6, vcc",
                "v_mov_b32_e32 v5, s6",
                "v_readlane_b32 s7, v5, 0",
                "v_mov_b32_e32 v6, s7",
            ],
            contains=["lane-count"],
            forbidden_idiom="readlane-uniform",
        )
        # Lane 40 also must not trigger uniform_readlane (classification check; lane 40 is absent on 32-wide waves).
        (tier, reasons, idioms), asm = meta_of(
            "vs_cnt_vgpr_readlane_40",
            vs([
                "v_cmp_lt_u32_e32 vcc, 4, v0",
                "s_bcnt1_i32_b64 s6, vcc",
                "v_mov_b32_e32 v5, s6",
                "v_readlane_b32 s7, v5, 40",
                "v_mov_b32_e32 v6, s7",
            ]),
        )
        self.assert_valid_spirv(asm)
        self.assertEqual(tier, "C", f"got tier {tier} (reasons={reasons}, idioms={idioms})")
        self.assertIn("lane-count", reasons)
        self.assertNotIn("readlane-uniform", idioms)
        self.assertEqual(rdna2spirv.translate.last_gen.w64g["uniform_readlane"], set())


if __name__ == "__main__":
    unittest.main()
