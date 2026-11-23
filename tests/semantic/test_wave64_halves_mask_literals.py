"""Soundness regression tests for tools/wave64_halves.py on SALU half-mask and non-uniform 64-bit mask writes.

Covers wave64 graphics shaders where SALU mask-writing instructions diverge between one 64-lane wave and
two 32-lane halves (`run_cut(asm, 64)` vs `run_cut(asm, 32)` on `spirv_wave_interp`):
  - 32-bit half writes (`vcc_lo` / `vcc_hi`) to a live VCC lane mask whose sources are immediates or
    non-mask SGPRs (`s_mov_b32`, `s_not_b32`, `s_and_b32`, etc.).
  - Non-uniform 64-bit mask literals (including positive 32-bit `0xFFFFFFFF`, whose upper 32 bits are 0)
    or plain-data SGPR pairs written to `vcc`, `exec`, or combined with a lane mask.
  - Clean Tier A controls (`0`, `-1`, saved EXEC, plain SGPR-pair constants, and dead VCC writes) that
    must remain Tier A and produce identical 1x64 and 2x32 outputs.

Standard library only; no GPU, no game files.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for _p in (HERE, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402,F401
import spirv_asm  # noqa: E402
from test_wave64_halves import meta_of, out_param, run_cut, vs  # noqa: E402


class Wave64HalvesMaskLiteralSoundnessTests(unittest.TestCase):
    def assert_tier_c_and_diverges(self, name, body, *, contains=None):
        (tier, reasons, _), asm = meta_of(name, vs(body))
        self.assertGreater(len(spirv_asm.assemble(asm, target_env="vulkan1.2")), 20)
        self.assertEqual(tier, "C", f"{name}: expected tier C, got {tier} (reasons={reasons})")
        if contains is not None:
            self.assertTrue(set(contains) <= set(reasons), f"{name}: reasons {reasons} lack {contains}")
        whole = out_param(run_cut(asm, 64))
        halves = out_param(run_cut(asm, 32))
        self.assertNotEqual(whole, halves, f"{name}: expected 1x64 and 2x32 outputs to diverge")

    def assert_tier_a_and_agrees(self, name, body):
        (tier, reasons, _), asm = meta_of(name, vs(body))
        self.assertGreater(len(spirv_asm.assemble(asm, target_env="vulkan1.2")), 20)
        self.assertEqual(tier, "A", f"{name}: expected tier A, got {tier} (reasons={reasons})")
        self.assertEqual(reasons, [], f"{name}: unexpected reasons {reasons}")
        whole = out_param(run_cut(asm, 64))
        halves = out_param(run_cut(asm, 32))
        self.assertEqual(whole, halves, f"{name}: expected 1x64 and 2x32 outputs to agree")

    def test_vcc_half_writes_with_non_mask_sources_are_tier_c(self):
        cases = [
            ("vs_vcc_hi_mov_0", ["v_cmp_ne_u32_e64 vcc, v0, 999", "s_mov_b32 vcc_hi, 0",
                                 "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_lo_mov_0", ["v_cmp_ne_u32_e64 vcc, v0, 999", "s_mov_b32 vcc_lo, 0",
                                 "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_hi_mov_sgpr_0", ["v_cmp_ne_u32_e64 vcc, v0, 999", "s_mov_b32 s4, 0",
                                      "s_mov_b32 vcc_hi, s4", "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_lo_not_imm", ["v_cmp_ne_u32_e64 vcc, v0, 999", "s_not_b32 vcc_lo, 0x0f",
                                   "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_hi_and_imm", ["v_cmp_ne_u32_e64 vcc, v0, 999", "s_and_b32 vcc_hi, 0x0f, 0x05",
                                   "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
        ]
        for name, body in cases:
            with self.subTest(case=name):
                self.assert_tier_c_and_diverges(name, body, contains=["mask-half"])

    def test_non_uniform_b64_vcc_and_mask_literals_are_tier_c(self):
        cases = [
            ("vs_vcc_mov64_imm", ["s_mov_b64 vcc, 0x0f", "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_mov64_data_pair", ["s_mov_b64 s[8:9], 0x0f", "s_mov_b64 vcc, s[8:9]",
                                        "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_and64_imm", ["s_and_b64 vcc, 0x0f, 0x05", "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_not64_imm", ["s_not_b64 vcc, 0x0f", "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_cselect64_imm", ["s_cmp_eq_u32 s4, s4", "s_cselect_b64 vcc, 0x0f, 0",
                                      "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_wqm64_imm", ["s_wqm_b64 vcc, 0x0f", "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_wqm64_data", ["s_mov_b64 s[8:9], 0x0f", "s_wqm_b64 vcc, s[8:9]",
                                   "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_vcc_and64_u32_max", ["v_cmp_ne_u32_e64 vcc, v0, 999", "s_and_b64 vcc, vcc, 0xffffffff",
                                      "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_sgpr_and64_u32_max", ["v_cmp_ne_u32_e64 vcc, v0, 999", "s_and_b64 s[6:7], vcc, 0xffffffff",
                                       "v_cndmask_b32_e64 v6, 0, 1, s[6:7]"]),
        ]
        for name, body in cases:
            with self.subTest(case=name):
                self.assert_tier_c_and_diverges(name, body, contains=["cross-lane"])

    def test_non_uniform_b64_exec_writes_are_tier_c(self):
        cases = [
            ("vs_exec_mov64_imm", ["v_mov_b32_e32 v6, 0", "s_mov_b64 exec, 0x0f",
                                   "v_mov_b32_e32 v6, 1", "s_mov_b64 exec, -1"]),
            ("vs_exec_mov64_u32_max", ["v_mov_b32_e32 v6, 0", "s_mov_b64 exec, 0xffffffff",
                                       "v_mov_b32_e32 v6, 1", "s_mov_b64 exec, -1"]),
            ("vs_exec_and64_imm", ["v_mov_b32_e32 v6, 0", "s_and_b64 exec, exec, 0x0f",
                                   "v_mov_b32_e32 v6, 1", "s_mov_b64 exec, -1"]),
            ("vs_exec_wqm64_imm", ["v_mov_b32_e32 v6, 0", "s_wqm_b64 exec, 0x0f",
                                   "v_mov_b32_e32 v6, 1", "s_mov_b64 exec, -1"]),
            ("vs_saveexec64_imm", ["v_mov_b32_e32 v6, 0", "s_and_saveexec_b64 s[4:5], 0x0f",
                                   "v_mov_b32_e32 v6, 1", "s_mov_b64 exec, s[4:5]"]),
            ("vs_saveexec64_data_pair", ["s_mov_b64 s[8:9], 0x0f", "v_mov_b32_e32 v6, 0",
                                         "s_and_saveexec_b64 s[4:5], s[8:9]",
                                         "v_mov_b32_e32 v6, 1", "s_mov_b64 exec, s[4:5]"]),
        ]
        for name, body in cases:
            with self.subTest(case=name):
                self.assert_tier_c_and_diverges(name, body, contains=["tainted-exec"])

    def test_uniform_constants_and_clean_controls_remain_tier_a(self):
        cases = [
            ("vs_ctl_vcc_0", ["s_mov_b64 vcc, 0", "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_ctl_vcc_m1", ["s_mov_b64 vcc, -1", "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_ctl_vcc_not_0", ["s_not_b64 vcc, 0", "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_ctl_vcc_not_m1", ["s_not_b64 vcc, -1", "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_ctl_vcc_and_m1", ["v_cmp_ne_u32_e64 vcc, v0, 0", "s_and_b64 vcc, vcc, -1",
                                   "v_cndmask_b32_e64 v6, 0, 1, vcc"]),
            ("vs_ctl_exec_m1", ["v_mov_b32_e32 v6, 0", "s_and_saveexec_b64 s[4:5], -1",
                                "v_mov_b32_e32 v6, 1", "s_mov_b64 exec, s[4:5]"]),
            ("vs_ctl_sgpr_pair_imm", ["s_mov_b64 s[8:9], 0x0f", "v_add_nc_u32_e64 v6, s8, v0"]),
            ("vs_ctl_dead_vcc_imm_keeps_scc", [
                "s_cmp_eq_u32 s4, s4",
                "s_mov_b64 vcc, 0x0f",
                "s_cselect_b32 s6, 11, 22",
                "v_cmp_ne_u32_e64 vcc, v0, 0",
                "v_add_nc_u32_e64 v6, s6, v0",
            ]),
        ]
        for name, body in cases:
            with self.subTest(case=name):
                self.assert_tier_a_and_agrees(name, body)


if __name__ == "__main__":
    unittest.main()
