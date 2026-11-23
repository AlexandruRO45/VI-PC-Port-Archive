"""Soundness regression tests for non-uniform bitmask operands on v_cndmask and carry-in VALU ops.

`rdna2spirv.Gen.mask_src` reads bit `wave_lane` (0..63 in one 64-lane wave vs 0..31 in two 32-lane halves) for
any 64-bit VALU mask/carry-in operand that is neither a tracked lane mask nor a wave-uniform constant
(0, -1, 0xFFFFFFFFFFFFFFFF, or null). Accepting non-uniform literals (1, 0xFFFFFFFF) or plain-data SGPR
pairs silently classifies diverging wave64 graphics shaders as Tier A or Tier B.

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
    run_cut,
    vs,
)


class Wave64HalvesValuMaskSrcTests(unittest.TestCase):
    def assert_valid_spirv(self, asm):
        words = spirv_asm.assemble(asm, target_env="vulkan1.2")
        self.assertGreater(len(words), 20)

    def assert_tier_c_and_diverges(self, name, body, *, contains=("lane-id",), forbidden_idiom=None):
        (tier, reasons, idioms), asm = meta_of(name, vs(body))
        self.assert_valid_spirv(asm)
        self.assertEqual(tier, "C", f"{name}: expected tier C, got {tier} (reasons={reasons}, idioms={idioms})")
        if contains is not None:
            self.assertTrue(set(contains) <= set(reasons), f"{name}: reasons {reasons} lack {contains}")
        if forbidden_idiom is not None:
            self.assertNotIn(forbidden_idiom, idioms, f"{name}: unexpected idiom {idioms}")
        whole = out_param(run_cut(asm, 64))
        halves = out_param(run_cut(asm, 32))
        self.assertNotEqual(whole, halves, f"{name}: expected 1x64 and 2x32 outputs to diverge")

    def assert_tier_a_and_agrees(self, name, body):
        (tier, reasons, idioms), asm = meta_of(name, vs(body))
        self.assert_valid_spirv(asm)
        self.assertEqual(tier, "A", f"{name}: expected tier A, got {tier} (reasons={reasons}, idioms={idioms})")
        self.assertEqual(reasons, [], f"{name}: unexpected reasons {reasons}")
        whole = out_param(run_cut(asm, 64))
        halves = out_param(run_cut(asm, 32))
        self.assertEqual(whole, halves, f"{name}: expected 1x64 and 2x32 outputs to agree")

    def test_v_cndmask_non_uniform_mask_src_is_tier_c_and_diverges(self):
        cases = (
            ("vs_cndmask_imm_1", ["v_cndmask_b32_e64 v6, 0, 1, 1"]),
            ("vs_cndmask_u32_all_ones", ["v_cndmask_b32_e64 v6, 0, 1, 0xFFFFFFFF"]),
            ("vs_cndmask_plain_pair", [
                "s_mov_b32 s8, 1",
                "s_mov_b32 s9, 0",
                "v_cndmask_b32_e64 v6, 0, 1, s[8:9]",
            ]),
        )
        for name, body in cases:
            with self.subTest(name=name):
                self.assert_tier_c_and_diverges(name, body)

    def test_carry_in_ops_with_non_uniform_mask_src_are_tier_c_and_diverge(self):
        for tag, op, operands in (
            ("add", "v_add_co_ci_u32_e64", "2, 3"),
            ("sub", "v_sub_co_ci_u32_e64", "10, 3"),
            ("subrev", "v_subrev_co_ci_u32_e64", "3, 10"),
        ):
            for mtag, setup, mask_src in (
                ("imm_1", [], "1"),
                ("u32_all_ones", [], "0xFFFFFFFF"),
                ("plain_pair", ["s_mov_b32 s8, 1", "s_mov_b32 s9, 0"], "s[8:9]"),
            ):
                with self.subTest(op=tag, mask=mtag):
                    self.assert_tier_c_and_diverges(
                        f"vs_{tag}_co_ci_{mtag}",
                        setup + [f"{op} v6, s[12:13], {operands}, {mask_src}"],
                    )

    def test_carry_in_non_uniform_mask_blocks_readfirstlane_uniform(self):
        self.assert_tier_c_and_diverges(
            "vs_add_co_ci_rfl_uniform",
            [
                "v_add_co_ci_u32_e64 v5, s[12:13], 0, 0, 1",
                "v_readfirstlane_b32 s6, v5",
                "v_add_nc_u32_e32 v6, s6, v5",
            ],
            contains=("lane-id",),
            forbidden_idiom="readfirstlane-uniform",
        )

    def test_uniform_and_tracked_mask_sources_stay_tier_a(self):
        cases = (
            ("vs_cndmask_0", ["v_cndmask_b32_e64 v6, 0, 1, 0"]),
            ("vs_cndmask_neg1", ["v_cndmask_b32_e64 v6, 0, 1, -1"]),
            ("vs_cndmask_null", ["v_cndmask_b32_e64 v6, 0, 1, null"]),
            ("vs_cndmask_tracked", [
                "v_cmp_lt_u32_e64 s[8:9], 4, v0",
                "v_cndmask_b32_e64 v6, 0, 1, s[8:9]",
            ]),
            ("vs_add_co_ci_0", ["v_add_co_ci_u32_e64 v6, s[12:13], 2, 3, 0"]),
            ("vs_add_co_ci_neg1", ["v_add_co_ci_u32_e64 v6, s[12:13], 2, 3, -1"]),
            ("vs_add_co_ci_tracked", [
                "v_cmp_lt_u32_e64 s[8:9], 4, v0",
                "v_add_co_ci_u32_e64 v6, s[12:13], 2, 3, s[8:9]",
            ]),
        )
        for name, body in cases:
            with self.subTest(name=name):
                self.assert_tier_a_and_agrees(name, body)


if __name__ == "__main__":
    unittest.main()
