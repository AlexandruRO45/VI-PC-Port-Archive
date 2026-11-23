"""SHADER-10: lane-id values equal across all 32 bits in wave64 and 2x32 halves (lmod == 32) at any sink.

Checks both classification in tools/wave64_halves.py and strict 1x64 vs 2x32 execution in spirv_wave_interp.py.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_wave64_halves as w64  # noqa: E402


class LmodSinksTest(unittest.TestCase):
    def check_vs(self, name, body, expect_tier, expect_equal, expect_reasons=None):
        lines = w64.vs(body)
        (tier, reasons, _), asm = w64.meta_of(name, lines)
        self.assertEqual(tier, expect_tier, f"{name}: got tier {tier}, reasons {reasons}")
        if expect_reasons is not None:
            self.assertEqual(sorted(reasons), sorted(expect_reasons), f"{name}: got reasons {reasons}")
        whole = w64.out_param(w64.run_cut(asm, 64), 0)
        halves = w64.out_param(w64.run_cut(asm, 32), 0)
        if expect_equal:
            self.assertEqual(whole, halves, f"{name}: 1x64 and 2x32 outputs differed")
        else:
            self.assertNotEqual(whole, halves, f"{name}: negative control unexpectedly matched")

    def check_tier(self, name, body, expect_tier, expect_reasons=None):
        lines = w64.ps(body)
        (tier, reasons, _), _ = w64.meta_of(name, lines)
        self.assertEqual(tier, expect_tier, f"{name}: got tier {tier}, reasons {reasons}")
        if expect_reasons is not None:
            self.assertEqual(sorted(reasons), sorted(expect_reasons), f"{name}: got reasons {reasons}")

    def test_and_masks_both_operand_orders(self):
        for mask in (0, 1, 3, 7, 31):
            with self.subTest(order="imm_first", mask=mask):
                self.check_vs(f"vs_and_imm_first_{mask}", [
                    *w64.LANE_ID,
                    f"v_and_b32_e32 v6, {mask}, v10",
                ], "A", True, [])
            with self.subTest(order="imm_second", mask=mask):
                self.check_vs(f"vs_and_imm_second_{mask}", [
                    *w64.LANE_ID,
                    f"v_and_b32_e64 v6, v10, {mask}",
                ], "A", True, [])

    def test_or_mask_and_and_or(self):
        self.check_vs("vs_or_high_ones", [
            *w64.LANE_ID,
            "v_or_b32_e32 v6, 0xffffffe0, v10",
        ], "A", True, [])
        self.check_vs("vs_and_or_mask", [
            *w64.LANE_ID,
            "v_and_or_b32 v6, v10, 3, v0",
        ], "A", True, [])

    def test_downstream_alu_cmp_and_store_sinks(self):
        self.check_vs("vs_and_downstream_alu", [
            *w64.LANE_ID,
            "v_and_b32_e32 v11, 3, v10",
            "v_mul_lo_u32 v11, v11, 7",
            "v_lshlrev_b32_e32 v11, 2, v11",
            "v_add_nc_u32_e32 v11, 100, v11",
            "v_xor_b32_e32 v11, 1, v11",
            "v_cmp_eq_u32_e32 vcc, 101, v11",
            "v_cndmask_b32_e64 v6, v11, 999, vcc",
        ], "A", True, [])
        self.check_tier("ps_and_buffer_store", [
            *w64.LANE_ID,
            "v_and_b32_e32 v11, 15, v10",
            "buffer_store_dword v11, v11, s[0:3], 0 offen offset:256",
        ], "A", [])

    def test_arith_then_mask(self):
        self.check_vs("vs_add_then_and31", [
            *w64.LANE_ID,
            "v_add_nc_u32_e32 v11, 100, v10",
            "v_and_b32_e32 v6, 31, v11",
        ], "A", True, [])
        self.check_vs("vs_mul_then_and31", [
            *w64.LANE_ID,
            "v_mul_lo_u32 v11, v10, 7",
            "v_and_b32_e32 v6, 31, v11",
        ], "A", True, [])

    def test_bfe_extracts(self):
        for name, op, off, width in (
            ("u_0_5", "v_bfe_u32", 0, 5),
            ("u_2_3", "v_bfe_u32", 2, 3),
            ("u_0_0", "v_bfe_u32", 0, 0),
            ("i_0_5", "v_bfe_i32", 0, 5),
        ):
            with self.subTest(case=name):
                self.check_vs(f"vs_bfe_{name}", [
                    *w64.LANE_ID,
                    f"{op} v6, v10, {off}, {width}",
                ], "A", True, [])

    def test_shifts_and_masked_shifts(self):
        self.check_vs("vs_shl1_and31", [
            *w64.LANE_ID,
            "v_lshlrev_b32_e32 v11, 1, v10",
            "v_and_b32_e32 v6, 31, v11",
        ], "A", True, [])
        self.check_vs("vs_shl1_and63", [
            *w64.LANE_ID,
            "v_lshlrev_b32_e32 v11, 1, v10",
            "v_and_b32_e32 v6, 63, v11",
        ], "A", True, [])
        self.check_vs("vs_shl27", [
            *w64.LANE_ID,
            "v_lshlrev_b32_e32 v6, 27, v10",
        ], "A", True, [])

    def test_literal_sgpr_mask(self):
        self.check_vs("vs_sgpr_lit_mask", [
            *w64.LANE_ID,
            "s_mov_b32 s4, 31",
            "v_and_b32_e32 v6, s4, v10",
        ], "A", True, [])

    def test_narrowed_exec_clean_dst_is_safe(self):
        self.check_vs("vs_narrow_exec_clean_dst", [
            *w64.LANE_ID,
            "v_mov_b32_e32 v11, 7",
            "v_cmp_lt_u32_e32 vcc, v0, 48",
            "s_and_saveexec_b64 s[12:13], vcc",
            "v_and_b32_e32 v11, 3, v10",
            "s_mov_b64 exec, s[10:11]",
            "v_mov_b32_e32 v6, v11",
        ], "A", True, [])

    def test_negative_controls(self):
        for name, body in (
            ("and32", [*w64.LANE_ID, "v_and_b32_e32 v6, 32, v10"]),
            ("and63", [*w64.LANE_ID, "v_and_b32_e32 v6, 63, v10"]),
            ("and_neg1", [*w64.LANE_ID, "v_and_b32_e32 v6, -1, v10"]),
            ("or_high_missing_bit5", [*w64.LANE_ID, "v_or_b32_e32 v6, 0xffffffc0, v10"]),
            ("shr5_and1", [*w64.LANE_ID, "v_lshrrev_b32_e32 v11, 5, v10", "v_and_b32_e32 v6, 1, v11"]),
            ("shr1_and16", [*w64.LANE_ID, "v_lshrrev_b32_e32 v11, 1, v10", "v_and_b32_e32 v6, 16, v11"]),
            ("shl1_and64", [*w64.LANE_ID, "v_lshlrev_b32_e32 v11, 1, v10", "v_and_b32_e32 v6, 64, v11"]),
            ("bfe_u_4_3", [*w64.LANE_ID, "v_bfe_u32 v6, v10, 4, 3"]),
            ("bfe_i_0_6", [*w64.LANE_ID, "v_bfe_i32 v6, v10, 0, 6"]),
        ):
            with self.subTest(case=name):
                self.check_vs(f"vs_neg_{name}", body, "C", False, ["lane-id"])

    def test_negative_runtime_sgpr_and_narrow_exec_inplace(self):
        self.check_tier("ps_neg_runtime_sgpr", [
            *w64.LANE_ID,
            "v_and_b32_e32 v6, s0, v10",
        ], "C", ["lane-id"])
        self.check_vs("vs_neg_narrow_exec_inplace", [
            *w64.LANE_ID,
            "v_cmp_lt_u32_e32 vcc, v0, 48",
            "s_and_saveexec_b64 s[12:13], vcc",
            "v_and_b32_e32 v10, 3, v10",
            "s_mov_b64 exec, s[10:11]",
            "v_mov_b32_e32 v6, v10",
        ], "C", False, ["lane-id"])


if __name__ == "__main__":
    unittest.main()
