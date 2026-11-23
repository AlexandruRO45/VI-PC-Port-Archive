from pathlib import Path
import sys
import unittest
from unittest.mock import patch

__author__ = "it_komarov"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import support
import rdna2spirv
from test_wave64_halves import _run_ps_cut, listing, meta_of, out_param, ps, run_cut, vs


class ScalarSpillIndexTests(unittest.TestCase):
    def exact(self, body, value=100):
        (tier, reasons, idioms), asm = meta_of("vs_spill_sgpr_index", vs(body))
        self.assertEqual((tier, reasons), ("B", []))
        self.assertIn("lane-spill", idioms)
        self.assertNotIn("OpGroupNonUniformShuffle", asm)
        push = [0, 0, 0, 0, 100, 23]
        whole = out_param(run_cut(asm, 64, push))
        halves = out_param(run_cut(asm, 32, push))
        self.assertEqual(whole, halves)
        self.assertEqual([word[0] for word in whole], [value + lane for lane in range(64)])
        return asm

    def test_each_lane_including_upper_half(self):
        for lane in (0, 7, 31, 32, 40, 63):
            with self.subTest(lane=lane):
                self.exact([
                    f"s_mov_b32 s8, {lane}", "v_writelane_b32 v30, s4, s8",
                    "v_readlane_b32 s21, v30, s8", "v_add_nc_u32_e32 v6, s21, v0",
                ])

    def test_folded_lane_selectors(self):
        cases = [
            ["s_mov_b32 s8, 8", "s_lshl_b32 s8, s8, 2"],
            ["s_mov_b32 s8, 80", "s_lshr_b32 s8, s8, 1"],
            ["s_mov_b32 s8, 5", "s_mul_i32 s8, s8, 8"],
            ["s_mov_b32 s8, 104", "s_and_b32 s8, s8, 63"],
            ["s_movk_i32 s8, -24"],
            ["s_mov_b32 s9, 40", "s_mov_b32 s8, s9"],
            ["s_mov_b64 s[8:9], 40"],
        ]
        for prefix in cases:
            with self.subTest(prefix=prefix):
                self.exact(prefix + ["v_writelane_b32 v30, s4, s8", "v_readlane_b32 s21, v30, s8",
                                     "v_add_nc_u32_e32 v6, s21, v0"])

    def test_immediate_and_sgpr_selectors_share_storage(self):
        for write, read in (("s8", "40"), ("40", "s8")):
            with self.subTest(write=write, read=read):
                self.exact(["s_mov_b32 s8, 40", f"v_writelane_b32 v30, s4, {write}",
                            f"v_readlane_b32 s21, v30, {read}", "v_add_nc_u32_e32 v6, s21, v0"])

    def test_slot_is_bound_to_instruction_before_selector_is_changed(self):
        self.exact([
            "s_mov_b32 s8, 40", "v_writelane_b32 v30, s4, s8",
            "s_mov_b32 s8, 63", "v_writelane_b32 v30, s5, s8",
            "s_mov_b32 s8, 104", "v_readlane_b32 s21, v30, s8",
            "s_movk_i32 s8, -1", "v_readlane_b32 s22, v30, s8",
            "s_add_u32 s23, s21, s22", "v_add_nc_u32_e32 v6, s23, v0",
        ], 123)

    def test_modulo_64_alias_write_updates_the_same_slot(self):
        self.exact([
            "s_mov_b32 s8, 40", "v_writelane_b32 v30, s4, s8",
            "s_mov_b32 s8, 104", "v_writelane_b32 v30, s5, s8",
            "s_movk_i32 s8, -24", "v_readlane_b32 s21, v30, s8",
            "v_add_nc_u32_e32 v6, s21, v0",
        ], 23)

    def test_spill_ignores_exec(self):
        self.exact([
            "s_mov_b64 exec, 0", "s_mov_b32 s8, 40", "v_writelane_b32 v30, s4, s8",
            "s_mov_b64 exec, s[10:11]", "v_readlane_b32 s21, v30, s8",
            "v_add_nc_u32_e32 v6, s21, v0",
        ])

    def test_tainted_scalar_stays_tier_c_and_really_differs(self):
        (tier, reasons, _), asm = meta_of("vs_spill_tainted", vs([
            "v_cmp_lt_u32_e32 vcc, 40, v0", "s_bcnt1_i32_b64 s21, vcc", "s_mov_b32 s8, 40",
            "v_writelane_b32 v30, s21, s8", "v_readlane_b32 s22, v30, s8", "v_mov_b32_e32 v6, s22",
        ]))
        self.assertEqual(tier, "C")
        self.assertIn("lane-count", reasons)
        self.assertNotEqual(out_param(run_cut(asm, 64)), out_param(run_cut(asm, 32)))

    def test_vote_controlled_write_is_stale_even_through_lane_alias(self):
        (tier, reasons, _), asm = meta_of("vs_spill_stale", vs([
            "s_mov_b32 s8, 40", "v_writelane_b32 v30, s4, s8",
            "v_cmp_eq_u32_e32 vcc, 0, v0", "s_cbranch_vccz 3",
            "s_mov_b32 s8, 104", "v_writelane_b32 v30, s5, s8",
            "s_mov_b32 s8, 40", "v_readlane_b32 s21, v30, s8", "v_mov_b32_e32 v6, s21",
        ]))
        self.assertEqual(tier, "C")
        self.assertIn("stale-scalar", reasons)
        push = [0, 0, 0, 0, 100, 23]
        self.assertNotEqual(out_param(run_cut(asm, 64, push)), out_param(run_cut(asm, 32, push)))

    def test_runtime_selector_disqualifies_the_entire_vgpr(self):
        (tier, reasons, _), _ = meta_of("ps_spill_runtime_index", ps([
            "s_mov_b32 s8, 40", "v_writelane_b32 v30, s4, s8",
            "v_writelane_b32 v30, s5, s9", "v_readlane_b32 s21, v30, s8", "v_mov_b32_e32 v6, s21",
        ]))
        self.assertEqual(tier, "C")
        self.assertIn("cross-lane", reasons)
        self.assertEqual(rdna2spirv.translate.last_gen.w64g["scalar_spills"], [])

    def test_vector_use_disqualifies_scalar_storage(self):
        (tier, _, _), _ = meta_of("ps_spill_vector_use", ps([
            "s_mov_b32 s8, 40", "v_writelane_b32 v30, s4, s8", "v_readlane_b32 s21, v30, s8",
            "v_mov_b32_e32 v6, s21", "v_mov_b32_e32 v7, v30",
        ]))
        self.assertEqual(tier, "C")
        self.assertEqual(rdna2spirv.translate.last_gen.w64g["scalar_spills"], [])

    def test_relative_vector_alias_disqualifies_scalar_storage(self):
        (tier, _, _), _ = meta_of("ps_spill_relative_alias", ps([
            "s_mov_b32 s8, 40", "v_writelane_b32 v30, s4, s8", "s_mov_b32 m0, 1",
            "v_movrels_b32 v7, v29", "v_readlane_b32 s21, v30, s8", "v_mov_b32_e32 v6, s21",
        ]))
        self.assertEqual(tier, "C")
        self.assertEqual(rdna2spirv.translate.last_gen.w64g["scalar_spills"], [])

    def test_selector_from_another_block_is_not_assumed_constant(self):
        (tier, _, _), _ = meta_of("ps_spill_index_across_blocks", ps([
            "s_mov_b32 s8, 40", "v_writelane_b32 v30, s4, s8", "s_cmp_eq_u32 s4, 0",
            "s_cbranch_scc1 1", "s_mov_b32 s8, 63", "v_readlane_b32 s21, v30, s8",
            "v_mov_b32_e32 v6, s21",
        ]))
        self.assertEqual(tier, "C")
        self.assertEqual(rdna2spirv.translate.last_gen.w64g["scalar_spills"], [])

    def test_lane_mask_is_not_scalar_storage(self):
        (tier, _, _), _ = meta_of("ps_spill_mask_value", ps([
            "s_mov_b32 s8, 40", "v_writelane_b32 v30, exec_lo, s8",
            "v_readlane_b32 s21, v30, s8", "v_mov_b32_e32 v6, s21",
        ]))
        self.assertEqual(tier, "C")
        self.assertEqual(rdna2spirv.translate.last_gen.w64g["scalar_spills"], [])

    def test_disabling_analysis_keeps_native_lane_operations(self):
        with patch.object(rdna2spirv, "WAVE64_GRAPHICS", "off"):
            (_, _, _), asm = meta_of("vs_spill_analysis_off", vs([
                "s_mov_b32 s8, 40", "v_writelane_b32 v30, s4, s8",
                "v_readlane_b32 s21, v30, s8", "v_mov_b32_e32 v6, s21",
            ]))
        self.assertIsNone(rdna2spirv.translate.last_gen.w64g)
        self.assertIn("OpGroupNonUniformShuffle", asm)

    def test_geometry_spill_is_tier_b(self):
        from test_wave64_halves import GeometryRing
        gs = GeometryRing()
        self.assertEqual(gs.tier([
            "s_mov_b32 s8, 40", "v_writelane_b32 v30, s4, s8",
            "v_readlane_b32 s21, v30, s8", "v_mov_b32_e32 v14, s21", "v_mov_b32_e32 v12, 0",
            "ds_write2_b32 v12, v14, v14 offset0:0 offset1:1",
        ]), ("B", []))

    def test_pixel_fixture_matches_all_words_at_both_widths(self):
        path = HERE.parents[1] / "runtime/test/shaders/ps_w64_b_spill_sgpr.asm"
        lines = path.read_text(encoding="utf-8").splitlines()
        (tier, reasons, _), asm = meta_of("ps_w64_b_spill_sgpr", lines)
        self.assertEqual((tier, reasons), ("B", []))
        for width in (64, 32):
            with self.subTest(width=width):
                self.assertEqual(_run_ps_cut(asm, width), [(123, 77, 200, 0)] * 64)


class VertexSystemInputTests(unittest.TestCase):
    def test_draw_bases_do_not_change_zero_based_vertex_and_instance_ids(self):
        (_, _, _), asm = meta_of("vs_zero_based_ids", vs([
            "v_mov_b32_e32 v6, v0", "v_mov_b32_e32 v7, v3",
        ]))
        import spirv_wave_interp as wi
        for width in (32, 64):
            for base_vertex, base_instance in ((0, 0), (16, 7), (-19, 0x80000001)):
                with self.subTest(width=width, base_vertex=base_vertex, base_instance=base_instance):
                    result = wi.run_wave(asm, wave=width, builtins={
                        "VertexIndex": lambda lane: (base_vertex + lane) & 0xffffffff,
                        "InstanceIndex": lambda lane: (base_instance + 3) & 0xffffffff,
                        "BaseVertex": lambda lane: base_vertex & 0xffffffff,
                        "BaseInstance": lambda lane: base_instance,
                    })
                    words = [output["%out_param0"] for output in result["outputs"]]
                    self.assertEqual([word[:2] for word in words], [(lane, 3) for lane in range(width)])


if __name__ == "__main__":
    unittest.main()
