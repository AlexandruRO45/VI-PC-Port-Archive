"""Semantic and structural tests for RDNA2 inline source operands src_vccz (251) and src_execz (252).

Verifies:
  1. Decoder (`rdna2_decode`) and IR (`shader_ir`) recognition of encodings 0xfb (src_vccz) and 0xfc (src_execz).
  2. Whole-wave zero semantics across scalar (SOP1/SOP2/SOPC/SOPK/64-bit) and vector (VOP1/VOP2/VOP3/VOP3P/SDWA/VOPC)
     ops in wave32, wave64, compute and pixel stages (`spirv_wave_interp`, `spirv_mini_interp`, and `spirv-val`).
  3. Wave64 data VCC (`vcc_hi` non-zero with `vcc_lo == 0`), inactive-EXEC VCC bits, and 64-bit integer/mask
     zero-extension (`s_mov_b64`, `s_and_b64`, `s_cmp_eq/lg_u64`, `v_cndmask_b32_e64`, `s_and_b32 exec_lo`).
  4. `Gen.lane_local` exclusion so VALU ops reading `src_execz`/`src_vccz` execute votes/barriers outside `if (%active)`.
  5. `--wave64-emulate` scheduler/optimizer rules (`tools/wave64_emulate_opt.py`): VCC read liveness, blocking
     `s_cbranch_execz` folding, `_insn_may_barrier` in divergent multi-wave guards, and 64/128-lane barrier execution.
  6. `tools/wave64_halves.py` cross-lane Tier C classification when reading `src_execz`/`src_vccz` into live sinks.

Run from repository root:  python -I tests/semantic/test_src_execz_vccz.py
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2_decode  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402
import test_wave64_emulate_opt as weo  # noqa: E402

rdna2spirv = support.rdna2spirv
wave64_emulate_opt = rdna2spirv.wave64_emulate_opt
wave64_halves = rdna2spirv.wave64_halves

VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]
FSSM32 = bytes(0x138) + struct.pack("<4I", 0x8090, 32, 1, 1)
FSSM64 = bytes(0x138) + struct.pack("<4I", 0x8090, 64, 1, 1)
FSSM128 = bytes(0x138) + struct.pack("<4I", 0x8090, 128, 1, 1)

HAS_SPIRV_TOOLS = bool(shutil.which("spirv-as") and shutil.which("spirv-val"))


def _validate_spirv(spv_text: str) -> None:
    if not HAS_SPIRV_TOOLS:
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "mod.spvasm")
        spv = os.path.join(tmp, "mod.spv")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv_text)
        subprocess.run(["spirv-as", "--target-env", "vulkan1.2", src, "-o", spv],
                       check=True, capture_output=True, text=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", spv],
                       check=True, capture_output=True, text=True)


def _translate_cs(name: str, insns: list[tuple[str, int]], fssm: bytes, *, emulate: bool = False) -> str:
    text = f"; {name}\n" + "".join(
        f"\t{line} ; encoding: [{','.join(['0x00'] * size)}]\n" for line, size in insns
    )
    old = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = emulate
    try:
        spv = support.translate_listing(name, text.encode("utf-8"), support.STUB_DXBC, fssm)
    finally:
        rdna2spirv.WAVE64_EMULATE = old
    _validate_spirv(spv)
    return spv


def _run_cs_native(spv: str, wave: int, push_extra=()) -> list[int]:
    pc = VSHARP + list(push_extra) + [0] * max(0, 28 - len(push_extra))
    mem = si.run_workgroup(
        spv, wave, wave,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc[:32],
    )
    return [mem.get(4 * lane, 0) for lane in range(wave)]


def _run_cs_emulated(spv: str, threads: int, push_extra=()) -> list[int]:
    pc = VSHARP + list(push_extra) + [0] * max(0, 28 - len(push_extra))
    mem, _ = weo.run_emulated_workgroup(spv, (threads, 1, 1), push_constants=pc[:32])
    return [mem.get(4 * lane, 0) for lane in range(threads)]


class DecodeTests(unittest.TestCase):
    def test_decode_special_inline_encodings(self):
        # s_mov_b32 s4, src_vccz (0xfb) and s_mov_b32 s5, src_execz (0xfc)
        raw = bytes([0xFB, 0x03, 0x84, 0xBE, 0xFC, 0x03, 0x85, 0xBE])
        dec = rdna2_decode.decode_bytes(raw, wave32=True)
        self.assertEqual(dec["invalid"], 0)
        self.assertEqual(len(dec["insns"]), 2)
        ins_v, ins_e = dec["insns"]
        self.assertEqual(ins_v["op"], "s_mov_b32")
        self.assertEqual(ins_v["args"][1], {"k": "special", "t": "src_vccz", "neg": False, "abs": False})
        self.assertEqual(ins_e["op"], "s_mov_b32")
        self.assertEqual(ins_e["args"][1], {"k": "special", "t": "src_execz", "neg": False, "abs": False})

    def test_f64_source_unsupported(self):
        for op in ("src_execz", "src_vccz"):
            with self.assertRaises(rdna2spirv.Unsupported):
                support.translate_lines("cs_f64_bad", [f"v_add_f64 v[0:1], v[2:3], {op}", "s_endpgm"], fssm=FSSM32)


class ValueSemanticsTests(unittest.TestCase):
    def test_wave32_and_wave64_whole_wave_zero_rules(self):
        for wave, fssm, high_lane, emu in [
            (32, FSSM32, 29, False),
            (64, FSSM64, 45, False),
            (64, FSSM64, 45, True),
        ]:
            with self.subTest(wave=wave, emulate=emu):
                insns = ([("s_and_b64 s[12:13], exec, exec", 4)] if wave == 64 else []) + [
                    ("s_mov_b64 vcc, 0", 4),
                    ("s_mov_b32 s4, src_vccz", 4),              # bit 0: 1 (VCC == 0)
                    ("s_mov_b32 s5, src_execz", 4),             # bit 1: 0 (EXEC != 0)
                    (f"v_cmp_eq_u32_e32 vcc, {high_lane}, v0", 8),
                    ("s_mov_b32 s6, src_vccz", 4),              # bit 2: 0 (single lane set VCC != 0)
                    ("s_and_saveexec_b64 s[14:15], vcc" if wave == 64
                     else "s_and_saveexec_b32 s14, vcc_lo", 4),
                    ("s_mov_b32 s7, src_execz", 4),             # bit 3: 0 (single lane active in EXEC)
                    ("s_mov_b64 exec, 0" if wave == 64 else "s_mov_b32 exec_lo, 0", 4),
                    ("s_mov_b32 s8, src_execz", 4),             # bit 4: 1 (all lanes inactive)
                    ("s_mov_b64 vcc, -1" if wave == 64 else "s_mov_b32 vcc_lo, -1", 4),
                    ("s_mov_b32 s9, src_vccz", 4),              # bit 5: 0 (VCC != 0 even while EXEC == 0!)
                    ("s_mov_b64 exec, -1" if wave == 64 else "s_mov_b32 exec_lo, -1", 4),
                    ("v_add_nc_u32_e64 v1, src_execz, 0", 8),   # bit 6: 0 (VALU path)
                    ("s_mov_b64 vcc, 0", 4),
                    ("v_add_nc_u32_e64 v2, src_vccz, 0", 8),    # bit 7: 1 (VALU path, VCC == 0)
                    ("s_add_u32 s10, src_execz, src_vccz", 4),  # bit 8: 0 + 1 = 1 (dual inline statuses)
                    ("s_lshl_b32 s5, s5, 1", 4),
                    ("s_lshl_b32 s6, s6, 2", 4),
                    ("s_lshl_b32 s7, s7, 3", 4),
                    ("s_lshl_b32 s8, s8, 4", 4),
                    ("s_lshl_b32 s9, s9, 5", 4),
                    ("s_lshl_b32 s10, s10, 8", 4),
                    ("s_or_b32 s4, s4, s5", 4),
                    ("s_or_b32 s4, s4, s6", 4),
                    ("s_or_b32 s4, s4, s7", 4),
                    ("s_or_b32 s4, s4, s8", 4),
                    ("s_or_b32 s4, s4, s9", 4),
                    ("s_or_b32 s4, s4, s10", 4),
                    ("v_lshlrev_b32_e32 v1, 6, v1", 4),
                    ("v_lshlrev_b32_e32 v2, 7, v2", 4),
                    ("v_or_b32_e32 v1, v1, v2", 4),
                    ("v_or_b32_e32 v1, s4, v1", 4),
                    ("v_lshlrev_b32_e32 v4, 2, v0", 4),
                    ("buffer_store_dword v1, v4, s[0:3], 0 offen", 8),
                    ("s_endpgm", 4),
                ]
                spv = _translate_cs(f"cs_status_w{wave}_{emu}", insns, fssm, emulate=emu)
                out = _run_cs_emulated(spv, wave) if emu else _run_cs_native(spv, wave)
                expected = (1 << 0) | (1 << 4) | (1 << 7) | (1 << 8)
                self.assertEqual(out, [expected] * wave)

    def test_wave64_data_vcc_hi_nonzero(self):
        # Data VCC with vcc_lo == 0 and vcc_hi != 0 must make src_vccz read 0.
        for emu in (False, True):
            with self.subTest(emulate=emu):
                insns = [
                    ("s_and_b64 s[12:13], exec, exec", 4),
                    ("s_mov_b32 vcc_lo, s4", 4),
                    ("s_mov_b32 vcc_hi, s5", 4),
                    ("s_mov_b32 s6, src_vccz", 4),
                    ("v_mov_b32_e32 v1, s6", 4),
                    ("v_lshlrev_b32_e32 v4, 2, v0", 4),
                    ("buffer_store_dword v1, v4, s[0:3], 0 offen", 8),
                    ("s_endpgm", 4),
                ]
                spv = _translate_cs(f"cs_vcc_data_{emu}", insns, FSSM64, emulate=emu)
                for lo, hi, want in [(0, 0, 1), (0, 0x80000000, 0), (1, 0, 0), (0xA, 0x8000000A, 0)]:
                    out = (_run_cs_emulated(spv, 64, [lo, hi]) if emu
                           else _run_cs_native(spv, 64, [lo, hi]))
                    self.assertEqual(out, [want] * 64, f"lo={lo:#x} hi={hi:#x}")

    def test_alu_forms_modifiers_64bit_and_masks(self):
        # Cover SOP2/SOPC, VOP3 float modifiers, SDWA, VOP3 64-bit SALU/SOPC, v_cndmask_b32, and s_and_b32 exec_lo.
        insns = [
            ("s_mov_b32 vcc_lo, 0", 4),
            ("s_cmp_eq_u32 src_vccz, 1", 4),            # SCC = 1
            ("s_cselect_b32 s4, src_vccz, src_execz", 4),  # s4 = 1
            ("s_mov_b64 s[6:7], src_vccz", 4),          # s6 = 1, s7 = 0
            ("s_and_b64 s[8:9], s[6:7], src_vccz", 4),  # s8 = 1, s9 = 0
            ("s_cmp_eq_u64 s[6:7], src_vccz", 4),       # SCC = 1
            ("s_cselect_b32 s10, src_scc, 0", 4),       # s10 = 1
            ("v_mov_b32_e32 v2, 10", 4),
            ("v_mov_b32_e32 v3, 20", 4),
            ("v_cndmask_b32_e64 v1, v2, v3, src_vccz", 8),  # lane 0 gets 20, others get 10
            ("v_add_nc_u32_e64 v4, v1, src_execz", 8),      # still 20 / 10
            ("v_add_nc_u32_e32 v4, s4, v4", 4),             # 21 / 11
            ("v_add_nc_u32_e32 v4, s8, v4", 4),             # 22 / 12
            ("v_add_nc_u32_e32 v4, s9, v4", 4),             # still 22 / 12 (hi == 0)
            ("v_add_nc_u32_e32 v4, s10, v4", 4),            # 23 / 13
            ("s_and_b32 exec_lo, exec_lo, src_vccz", 4),    # leaves only lane 0 active
            ("v_add_nc_u32_e32 v4, 100, v4", 4),            # lane 0 -> 123, others stay 13
            ("s_mov_b32 exec_lo, -1", 4),
            ("v_lshlrev_b32_e32 v5, 2, v0", 4),
            ("buffer_store_dword v4, v5, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv = _translate_cs("cs_mask_64_ops", insns, FSSM32, emulate=False)
        out = _run_cs_native(spv, 32)
        self.assertEqual(out, [123] + [13] * 31)

    def test_extra_encodings_validate_and_neg_bits(self):
        # Verify f32 sign mods (-src_execz), f16/VOP3P, fma_mix, and SDWA translate and pass spirv-val.
        spv = support.translate_lines("ps_extra_forms", [
            "s_mov_b32 vcc_lo, 0",
            "v_add_f32_e64 v10, -src_vccz, 0",
            "v_add_f16_e64 v11, src_execz, v0",
            "v_pk_add_u16 v12, src_vccz, v0",
            "v_fma_mix_f32 v13, src_execz, v10, 1.0",
            "v_add_nc_u32_sdwa v14, src_vccz, v12 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0 src1_sel:DWORD",
            "exp mrt0 v10, v11, v13, v14 done vm",
            "s_endpgm",
        ], fssm=FSSM32)
        _validate_spirv(spv)
        # Also verify negative f32 bits of -src_vccz (bit 31 flipped on 0x00000001 -> 0x80000001) or 0 -> 0x80000000
        spv_neg = support.translate_lines("cs_neg_bits", [
            "s_mov_b32 vcc_lo, 0",
            "v_mov_b32_e64 v1, -src_vccz",
            "v_lshlrev_b32_e32 v2, 2, v0",
            "buffer_store_dword v1, v2, s[0:3], 0 offen",
            "s_endpgm",
        ], fssm=FSSM32)
        _validate_spirv(spv_neg)
        out = _run_cs_native(spv_neg, 32)
        self.assertEqual(out, [0x80000001] * 32)

    def test_ps_stage_wave32_and_wave64(self):
        for wave, fssm in [(32, FSSM32), (64, FSSM64)]:
            with self.subTest(wave=wave):
                spv = support.translate_lines(f"ps_status_w{wave}", [
                    "s_mov_b64 vcc, 0",
                    "s_mov_b32 s4, src_vccz",
                    "v_mov_b32_e32 v10, s4",
                    "v_add_nc_u32_e64 v11, src_execz, 5",
                    "exp mrt0 v10, v11, off, off done vm",
                    "s_endpgm",
                ], fssm=fssm)
                _validate_spirv(spv)
                res = wi.run_wave(spv, wave=wave)
                self.assertEqual(res["outputs"][0]["%out_mrt0"], (1, 5, 0, 0))

    def test_valu_with_src_execz_excluded_from_lane_local_active_branch(self):
        # Consecutive lane-local VALU ops enter `if (active)`, while the op reading src_execz stays outside.
        spv = support.translate_lines("cs_lane_local_guard", [
            "s_mov_b32 exec_lo, 0xf",
            "v_mov_b32_e32 v1, 10",
            "v_add_nc_u32_e32 v2, 20, v0",
            "v_add_nc_u32_e64 v3, src_execz, 7",
            "v_mov_b32_e32 v4, 30",
            "s_mov_b32 exec_lo, -1",
            "v_lshlrev_b32_e32 v5, 2, v0",
            "buffer_store_dword v3, v5, s[0:3], 0 offen",
            "s_endpgm",
        ], fssm=FSSM32)
        _validate_spirv(spv)
        out = _run_cs_native(spv, 32)
        self.assertEqual(out, [7] * 4 + [0] * 28)


class EmulateOptAndHalvesTests(unittest.TestCase):
    def test_emulate_opt_liveness_fold_blocking_and_barrier_flag(self):
        sgpr = lambda r, n=1: {"k": "sgpr", "r": r, "n": n}
        vgpr = lambda r, n=1: {"k": "vgpr", "r": r, "n": n}
        spec = lambda t: {"k": "special", "t": t}

        ins_vccz = {"op": "s_mov_b32", "args": [sgpr(4), spec("src_vccz")], "mods": ()}
        ins_execz = {"op": "v_add_nc_u32_e64", "args": [vgpr(1), spec("src_execz"), vgpr(0)], "mods": ()}
        for ins, want_rd in [(ins_vccz, 1 << wave64_emulate_opt._BIT_VCC), (ins_execz, 0)]:
            rd, _, _ = wave64_emulate_opt._scalar_rw(ins)
            self.assertEqual(rd, want_rd)
            self.assertTrue(wave64_emulate_opt._insn_may_barrier(ins, {}, set()))
            clean, _, _, _, _ = wave64_emulate_opt._classify_block_body([ins], {}, set(), frozenset())
            self.assertFalse(clean)

    def test_emulate_execz_fold_blocked_and_multiwave_divergent_guard(self):
        # 1) Block reading src_execz under s_cbranch_execz must NOT fold and must evaluate wave-wide in 64 & 128 threads.
        insns_fold = [
            ("s_mov_b32 s10, 0", 4),
            ("s_mov_b64 vcc, 0", 4),
            ("v_cmp_eq_u32_e32 vcc, 45, v0", 8),     # True only in wave 0 (lane 45), False in wave 1
            ("s_mov_b32 s11, src_vccz", 4),          # wave 0 -> 0, wave 1 -> 1
            ("s_and_saveexec_b64 s[6:7], vcc", 4),   # wave 0: lane 45 active; wave 1: EXEC == 0
            ("s_cbranch_execz 3", 4),
            ("s_mov_b32 s10, src_execz", 4),         # 0 when wave entered
            ("s_mov_b64 exec, 0", 4),
            ("s_mov_b32 s12, src_execz", 4),         # 1 when wave entered
            ("s_mov_b64 exec, -1", 4),
            ("s_add_u32 s10, s10, s12", 4),
            ("s_lshl_b32 s11, s11, 4", 4),
            ("s_or_b32 s10, s10, s11", 4),           # wave 0: 1, wave 1: 16
            ("v_mov_b32_e32 v1, s10", 4),
            ("v_lshlrev_b32_e32 v4, 2, v0", 4),
            ("buffer_store_dword v1, v4, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        for threads, fssm, want in [(64, FSSM64, [1] * 64), (128, FSSM128, [1] * 64 + [16] * 64)]:
            spv = _translate_cs(f"cs_nofold_{threads}", insns_fold, fssm, emulate=True)
            self.assertGreaterEqual(spv.count("OpControlBarrier"), 1)
            self.assertEqual(_run_cs_emulated(spv, threads), want)

        # 2) Divergent s_cbranch_vccz in 128-thread workgroup containing src_vccz must isolate barriers safely.
        insns_div = [
            ("s_and_b64 s[12:13], exec, exec", 4),
            ("v_cmp_lt_u32_e32 vcc, 100, v0", 8),    # wave 0: vcc == 0; wave 1: vcc != 0
            ("s_mov_b32 s10, 77", 4),
            ("s_cbranch_vccz 1", 4),
            ("s_mov_b32 s10, src_vccz", 4),          # runs only in wave 1 -> 0
            ("s_mov_b32 s11, src_execz", 4),         # 0 in both waves
            ("s_add_u32 s10, s10, s11", 4),
            ("v_mov_b32_e32 v1, s10", 4),
            ("v_lshlrev_b32_e32 v4, 2, v0", 4),
            ("buffer_store_dword v1, v4, s[0:3], 0 offen", 8),
            ("s_endpgm", 4),
        ]
        spv_div = _translate_cs("cs_div_guard_128", insns_div, FSSM128, emulate=True)
        self.assertEqual(_run_cs_emulated(spv_div, 128), [77] * 64 + [0] * 64)

    def test_wave64_halves_classifies_live_uses_as_tier_c(self):
        for name, body, tier, reasons in [
            ("ps_execz_export", [
                "s_mov_b64 s[10:11], exec",
                "s_mov_b32 s4, src_execz",
                "v_mov_b32_e32 v10, s4",
                "exp mrt0 v10, v10, v10, v10 done vm",
                "s_endpgm",
            ], "C", ["cross-lane"]),
            ("ps_vccz_branch", [
                "s_mov_b64 s[10:11], exec",
                "s_cmp_eq_u32 src_vccz, 0",
                "s_cbranch_scc0 1",
                "v_mov_b32_e32 v10, 1",
                "exp mrt0 v10, v10, v10, v10 done vm",
                "s_endpgm",
            ], "C", ["tainted-branch"]),
            ("ps_dead_execz_stays_tier_a", [
                "s_mov_b64 s[10:11], exec",
                "s_mov_b32 s4, src_execz",
                "v_mov_b32_e32 v10, 42",
                "exp mrt0 v10, v10, v10, v10 done vm",
                "s_endpgm",
            ], "A", []),
        ]:
            with self.subTest(name=name):
                text = f"; {name}\n" + "".join(f"\t{line} ; encoding: [0x00,0x00,0x00,0x00]\n" for line in body)
                support.translate_listing(name, text.encode("utf-8"), support.STUB_DXBC, FSSM64)
                w = rdna2spirv.translate.last_gen.w64g
                self.assertEqual(w["tier"], tier)
                for req in reasons:
                    self.assertIn(req, w["reasons"])


if __name__ == "__main__":
    unittest.main()
