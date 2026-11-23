"""Soundness regression tests for tools/wave64_halves.py:
  1. Unsigned accumulating integer dot products (`v_dot2c_u32_u16`, `v_dot4c_u32_u8`, `v_dot8c_u32_u4`) in `RMW_DEST`:
     they read and update `vdst` (`D = dot(S0, S1) + D`), so they must preserve prior `vdst` taint/rank/base tokens
     and must not mark a varying `vdst` as wave-uniform (`st["u"]` / `st["ua"]`) for `v_readfirstlane_b32` or
     `v_readlane_b32` (`uniform_readlane`).
  2. Complete `sgpr_imm` scalar-fold mirror in `Analysis.literal_sgpr` (`s_abs_i32`, `s_sext_i32_i8`, `s_sext_i32_i16`,
     `s_min_i32`, `s_max_i32`, `s_min_u32`, `s_max_u32`, `s_add_u32`, `s_sub_u32`, `s_add_i32`, `s_sub_i32`,
     `s_lshl1_add_u32`..`s_lshl4_add_u32`, `s_xor_b32`, `s_andn2_b32`, `s_orn2_b32`, `s_nand_b32`, `s_nor_b32`,
     `s_xnor_b32`, `s_bfe_u32`, `s_bfe_i32`, `s_absdiff_i32`): `rdna2spirv.py` folds all 24 into `self.sgpr_imm`
     and emits compile-time `idx = wave_lane()` when V# word3 bit 23 (`ADD_TID_ENABLE`) is 1, so `literal_sgpr`
     must fold them too rather than returning `None` (`add_tid_lane -> 0`, Tier A).

Every Tier C vertex-shader regression is also executed on `spirv_wave_interp.py` as one 64-lane wave vs two 32-lane
waves to prove the outputs actually diverge (and agree for clean controls). Standard library only; no game files.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402,F401
import rdna2spirv  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402
from test_wave64_halves import meta_of, out_param, ps, run_cut, vs  # noqa: E402

UNSIGNED_DOTC_OPS = ("v_dot2c_u32_u16", "v_dot4c_u32_u8", "v_dot8c_u32_u4")
BUF_MEM = {0x1000 + 4 * k: 1000 + k for k in range(128)}


def run_store_mem(asm, wave):
    """Run a 64-lane VS storing to PSB memory as 64/wave waves of `wave` lanes sharing one buffer."""
    mem = {}
    for half in range(64 // wave):
        wi.run_wave(
            asm,
            wave=wave,
            memory=mem,
            builtins={
                "VertexIndex": lambda lane, base=half * wave: base + lane,
                "InstanceIndex": lambda lane: 0,
                "BaseVertex": lambda lane: 0,
                "BaseInstance": lambda lane: 0,
            },
        )
    return mem


class Wave64HalvesDotcAndLiteralSgprTests(unittest.TestCase):
    def test_unsigned_dotc_preserves_lane_id_taint_and_diverges(self):
        for op in UNSIGNED_DOTC_OPS:
            with self.subTest(op=op):
                lines = vs([
                    "v_mbcnt_lo_u32_b32 v6, -1, 0",
                    "v_mbcnt_hi_u32_b32 v6, -1, v6",
                    "v_mov_b32_e32 v4, 0x00020003",
                    "v_mov_b32_e32 v5, 0x00040005",
                    f"{op} v6, v4, v5",
                ])
                (tier, reasons, _), asm = meta_of(f"vs_rmw_{op}", lines)
                self.assertEqual(tier, "C", f"{op}: expected tier C, got {tier} (reasons={reasons})")
                self.assertIn("lane-id", reasons, f"{op}: reasons={reasons}")
                whole = out_param(run_cut(asm, 64))
                halves = out_param(run_cut(asm, 32))
                self.assertNotEqual(whole, halves, f"{op}: expected 1x64 and 2x32 outputs to diverge")

    def test_unsigned_dotc_on_varying_dst_is_not_uniform_for_readfirstlane_or_readlane(self):
        for op in UNSIGNED_DOTC_OPS:
            with self.subTest(op=op, reader="v_readfirstlane_b32"):
                # v8 starts as varying VertexIndex (v0); accumulating uniform 1, 2 into v8 leaves v8 varying.
                lines = vs([
                    "v_mov_b32_e32 v8, v0",
                    f"{op} v8, 1, 2",
                    "v_readfirstlane_b32 s6, v8",
                    "v_mov_b32_e32 v6, s6",
                ])
                (tier, reasons, idioms), asm = meta_of(f"vs_rfl_{op}", lines)
                self.assertEqual(tier, "C", f"{op}: expected tier C, got {tier} (reasons={reasons}, idioms={idioms})")
                self.assertIn("cross-lane", reasons)
                self.assertNotIn("readfirstlane-uniform", idioms)
                self.assertNotEqual(out_param(run_cut(asm, 64)), out_param(run_cut(asm, 32)))

            with self.subTest(op=op, reader="v_readlane_b32"):
                lines = vs([
                    "v_mov_b32_e32 v8, v0",
                    f"{op} v8, 1, 2",
                    "v_readlane_b32 s6, v8, 0",
                    "v_mov_b32_e32 v6, s6",
                ])
                (tier, reasons, idioms), asm = meta_of(f"vs_rl_{op}", lines)
                self.assertEqual(tier, "C", f"{op}: expected tier C, got {tier} (reasons={reasons}, idioms={idioms})")
                self.assertIn("cross-lane", reasons)
                self.assertNotIn("readlane-uniform", idioms)
                self.assertEqual(rdna2spirv.translate.last_gen.w64g["uniform_readlane"], set())
                self.assertNotEqual(out_param(run_cut(asm, 64)), out_param(run_cut(asm, 32)))

            with self.subTest(op=op, reader="clean_uniform_control"):
                (tier, reasons, idioms), _ = meta_of(f"ps_clean_{op}", ps([
                    "v_mov_b32_e32 v8, s4",
                    f"{op} v8, 1, 2",
                    "v_readfirstlane_b32 s6, v8",
                    "v_mov_b32_e32 v6, s6",
                ]))
                self.assertEqual((tier, reasons, idioms), ("B", [], ["readfirstlane-uniform"]))

    def test_literal_sgpr_mirrors_all_rdna2spirv_scalar_folds_for_add_tid(self):
        head = [
            "s_mov_b32 s0, 0x1000",
            "s_mov_b32 s1, 0x80040000",
            "s_mov_b32 s2, 0x1000",
        ]
        load = "buffer_load_dword v6, off, s[0:3], 0"
        # (name, instructions building s3 with ADD_TID_ENABLE bit 23 = 1, instructions building s3 with bit 23 = 0)
        cases = [
            ("s_abs_i32", ["s_abs_i32 s3, 0xce7e9054"], ["s_abs_i32 s3, 0xcefe9054"]),
            ("s_sext_i32_i8", ["s_sext_i32_i8 s3, 0x80"], ["s_sext_i32_i8 s3, 0x7f"]),
            ("s_sext_i32_i16", ["s_sext_i32_i16 s3, 0x8000"], ["s_sext_i32_i16 s3, 0x7fff"]),
            ("s_min_i32", ["s_min_i32 s3, 0x31816fac, 0x31817000"], ["s_min_i32 s3, 0x31016fac, 0x31816fac"]),
            ("s_max_i32", ["s_max_i32 s3, 0x31016fac, 0x31816fac"], ["s_max_i32 s3, 0x31016fac, 0x31000000"]),
            ("s_min_u32", ["s_min_u32 s3, 0x31816fac, 0x31817000"], ["s_min_u32 s3, 0x31016fac, 0x31816fac"]),
            ("s_max_u32", ["s_max_u32 s3, 0x31016fac, 0x31816fac"], ["s_max_u32 s3, 0x31016fac, 0x31000000"]),
            ("s_add_u32", ["s_add_u32 s3, 0x31016fac, 0x00800000"], ["s_add_u32 s3, 0x31016fac, 0x00000004"]),
            ("s_sub_u32", ["s_sub_u32 s3, 0x31816fb0, 4"], ["s_sub_u32 s3, 0x31816fac, 0x00800000"]),
            ("s_add_i32", ["s_add_i32 s3, 0x31016fac, 0x00800000"], ["s_add_i32 s3, 0x31016fac, 4"]),
            ("s_sub_i32", ["s_sub_i32 s3, 0x31816fb0, 4"], ["s_sub_i32 s3, 0x31816fac, 0x00800000"]),
            ("s_lshl1_add_u32", ["s_lshl1_add_u32 s3, 0x00400000, 0x31016fac"], ["s_lshl1_add_u32 s3, 2, 0x31016fac"]),
            ("s_lshl2_add_u32", ["s_lshl2_add_u32 s3, 0x00200000, 0x31016fac"], ["s_lshl2_add_u32 s3, 2, 0x31016fac"]),
            ("s_lshl3_add_u32", ["s_lshl3_add_u32 s3, 0x00100000, 0x31016fac"], ["s_lshl3_add_u32 s3, 2, 0x31016fac"]),
            ("s_lshl4_add_u32", ["s_lshl4_add_u32 s3, 0x00080000, 0x31016fac"], ["s_lshl4_add_u32 s3, 2, 0x31016fac"]),
            ("s_xor_b32", ["s_xor_b32 s3, 0x31016fac, 0x00800000"], ["s_xor_b32 s3, 0x31816fac, 0x00800000"]),
            ("s_andn2_b32", ["s_andn2_b32 s3, 0x31816fac, 0x000000ff"], ["s_andn2_b32 s3, 0x31816fac, 0x00800000"]),
            ("s_orn2_b32", ["s_orn2_b32 s3, 0x31016fac, 0xff7fffff"], ["s_orn2_b32 s3, 0x00000000, 0xffffffff"]),
            ("s_nand_b32", ["s_nand_b32 s3, 0xff7fffff, 0xffffffff"], ["s_nand_b32 s3, 0x00800000, 0xffffffff"]),
            ("s_nor_b32", ["s_nor_b32 s3, 0xce7e9053, 0"], ["s_nor_b32 s3, 0x00800000, 0"]),
            ("s_xnor_b32", ["s_xnor_b32 s3, 0x31816fac, 0xffffffff"], ["s_xnor_b32 s3, 0x31016fac, 0xffffffff"]),
            ("s_bfe_u32", ["s_bfe_u32 s3, 0x6302df58, 0x001f0001"], ["s_bfe_u32 s3, 0x6202df58, 0x001f0001"]),
            ("s_bfe_i32", ["s_bfe_i32 s3, 0x00000080, 0x00080000"], ["s_bfe_i32 s3, 0x0000007f, 0x00080000"]),
            ("s_absdiff_i32", ["s_absdiff_i32 s3, 0x10000000, 0x41816fac"], ["s_absdiff_i32 s3, 0x10000000, 0x41016fac"]),
        ]
        for name, build_set, build_clear in cases:
            for access in (load, "buffer_store_dword v0, off, s[0:3], 0"):
                with self.subTest(op=name, access=access.split()[0], add_tid=True):
                    lines_set = vs(head + build_set + [access, "s_waitcnt vmcnt(0)"])
                    (tier, reasons, _), asm_set = meta_of(f"vs_w3_set_{name}", lines_set)
                    self.assertEqual(tier, "C", f"{name} (set): expected tier C, got {tier} (reasons={reasons})")
                    self.assertIn("lane-id", reasons, f"{name} (set): reasons={reasons}")
                    if access.startswith("buffer_store"):
                        self.assertNotEqual(
                            run_store_mem(asm_set, 64),
                            run_store_mem(asm_set, 32),
                            f"{name} (set): expected 1x64 and 2x32 buffer_store memory to diverge",
                        )

                with self.subTest(op=name, access=access.split()[0], add_tid=False):
                    lines_clear = vs(head + build_clear + [access, "s_waitcnt vmcnt(0)"])
                    (tier, reasons, _), asm_clear = meta_of(f"vs_w3_clr_{name}", lines_clear)
                    self.assertEqual((tier, reasons), ("A", []), f"{name} (clear): got {tier}, reasons={reasons}")
                    if access.startswith("buffer_store"):
                        self.assertEqual(
                            run_store_mem(asm_clear, 64),
                            run_store_mem(asm_clear, 32),
                            f"{name} (clear): expected 1x64 and 2x32 buffer_store memory to agree",
                        )


if __name__ == "__main__":
    unittest.main()
