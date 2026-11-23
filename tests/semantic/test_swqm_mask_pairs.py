"""Semantic regression tests for s_wqm_b32 / s_wqm_b64 on opposite-half and mixed/non-canonical mask pairs.

RDNA2 ISA (SOP1):
  - s_wqm_b32 D, S0: each 4-bit quad of S0[31:0] with any set bit becomes 0xF, else 0; SCC = (D != 0).
  - s_wqm_b64 D, S0: same across all 16 quads of S0[63:0]; SCC = (D != 0).

Covers:
  - wave64 opposite-half s_wqm_b32 destinations: exec_lo/vcc_lo from MASK_HI, exec_hi/vcc_hi from MASK_LO
    (both non-zero and zero cases, checking active lanes and SCC)
  - wave64 mixed mask/data s_wqm_b64 SGPR pairs into SGPR pairs, exec, and vcc
  - wave64 non-canonical mask s_wqm_b64 SGPR pairs: (MASK_HI, 0), (0, MASK_LO), (MASK_HI, MASK_LO),
    (MASK_LO, MASK_LO), and partial-wave (20-thread) (MASK_LO=0, -1) SCC
  - wave64 plain-data vcc source (preserving %vcc_hi) and null destination
  - canonical (MASK_LO, MASK_HI) fast-path preservation (no ballot), no OpBitFieldUExtract on mask-dest
    materialization, and clean --wave64-emulate translation and SPIR-V assembly.

Standard library only; CPU-only, no GPU, no game files.
"""
from __future__ import annotations

import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402
import support  # noqa: E402

VSHARP = [0, 0, 0x1000, 0x31016FAC]


def make_fssm(wave64: bool = True, threads: int = 64) -> bytes:
    h = bytearray(0x148)
    h[0:4] = b"FSSM"
    struct.pack_into("<I", h, 0xD4, 0 if wave64 else 0x20)
    struct.pack_into("<4I", h, 0x138, 0x0008, threads, 1, 1)
    return bytes(h)


def ref_wqm32(w: int) -> int:
    out = 0
    for q in range(8):
        if (w >> (4 * q)) & 0xF:
            out |= 0xF << (4 * q)
    return out


def run_vs_w64(name: str, lines: list[str], *, wave: int = 64) -> tuple[str, dict[int, int]]:
    spv = support.translate_lines("vs_" + name, lines, fssm=make_fssm(True, wave))
    magic = spirv_asm.assemble(spv, target_env="vulkan1.2")[:4]
    assert magic == b"\x03\x02\x23\x07", f"bad SPIR-V magic: {magic!r}"
    res = wi.run_wave(
        spv,
        wave=wave,
        push_constants=VSHARP + [0] * 28,
        builtins={"VertexIndex": lambda i: i, "InstanceIndex": lambda i: 0},
        memory={},
    )
    return spv, res["memory"]


class WqmOppositeHalfTests(unittest.TestCase):
    def test_w64_s_wqm_b32_opposite_half_into_exec(self):
        # Case A: exec_lo <- wqm(s15[MASK_HI]), lane 32 set -> bit 0 set -> wqm = 0xF -> exec_lo = 0..3, exec_hi unchanged.
        lines_lo = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_cmp_eq_u32_e64 s[14:15], 32, v1",
            "v_mov_b32_e32 v20, 0",
            "s_wqm_b32 exec_lo, s15",
            "s_cselect_b32 s22, 1, 0",
            "v_mov_b32_e32 v20, 1",
            "s_mov_b64 exec, -1",
            "v_mov_b32_e32 v21, s22",
            "v_mul_u32_u24_e32 v22, 8, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v23, 4, v22",
            "buffer_store_dword v21, v23, s[0:3], 0 offen",
            "s_endpgm",
        ]
        _, mem_lo = run_vs_w64("wqm_exec_lo_hi", lines_lo)
        active_lo = [l for l in range(64) if mem_lo.get(8 * l, 0) == 1]
        self.assertEqual(active_lo, list(range(0, 4)) + list(range(32, 64)))
        self.assertEqual({mem_lo[8 * l + 4] for l in range(64)}, {1})

        # Case B: exec_hi <- wqm(s14[MASK_LO]), lane 5 set -> bit 5 -> wqm = 0xF0 -> exec_hi = 36..39, exec_lo unchanged.
        lines_hi = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_cmp_eq_u32_e64 s[14:15], 5, v1",
            "v_mov_b32_e32 v20, 0",
            "s_wqm_b32 exec_hi, s14",
            "s_cselect_b32 s22, 1, 0",
            "v_mov_b32_e32 v20, 1",
            "s_mov_b64 exec, -1",
            "v_mov_b32_e32 v21, s22",
            "v_mul_u32_u24_e32 v22, 8, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v23, 4, v22",
            "buffer_store_dword v21, v23, s[0:3], 0 offen",
            "s_endpgm",
        ]
        _, mem_hi = run_vs_w64("wqm_exec_hi_lo", lines_hi)
        active_hi = [l for l in range(64) if mem_hi.get(8 * l, 0) == 1]
        self.assertEqual(active_hi, list(range(0, 32)) + list(range(36, 40)))
        self.assertEqual({mem_hi[8 * l + 4] for l in range(64)}, {1})

        # Case C: empty MASK_HI into exec_lo -> exec_lo becomes 0, exec_hi stays all-1, SCC = 0.
        lines_zero = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_cmp_eq_u32_e64 s[14:15], 5, v1",
            "v_mov_b32_e32 v20, 0",
            "s_wqm_b32 exec_lo, s15",
            "s_cselect_b32 s22, 1, 0",
            "v_mov_b32_e32 v20, 1",
            "s_mov_b64 exec, -1",
            "v_mov_b32_e32 v21, s22",
            "v_mul_u32_u24_e32 v22, 8, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v23, 4, v22",
            "buffer_store_dword v21, v23, s[0:3], 0 offen",
            "s_endpgm",
        ]
        _, mem_zero = run_vs_w64("wqm_exec_lo_hi_zero", lines_zero)
        active_zero = [l for l in range(64) if mem_zero.get(8 * l, 0) == 1]
        self.assertEqual(active_zero, list(range(32, 64)))
        self.assertEqual({mem_zero[8 * l + 4] for l in range(64)}, {0})

    def test_w64_s_wqm_b32_opposite_half_into_vcc(self):
        # vcc starts with lane 40 set; s_wqm_b32 vcc_lo, s15 (lane 32 set -> 0xF) sets lanes 0..3 and keeps lane 40.
        lines_lo = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_cmp_eq_u32_e64 vcc, 40, v1",
            "v_cmp_eq_u32_e64 s[14:15], 32, v1",
            "s_wqm_b32 vcc_lo, s15",
            "s_cselect_b32 s22, 1, 0",
            "v_cndmask_b32_e64 v20, 0, 1, vcc",
            "v_mov_b32_e32 v21, s22",
            "v_mul_u32_u24_e32 v22, 8, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v23, 4, v22",
            "buffer_store_dword v21, v23, s[0:3], 0 offen",
            "s_endpgm",
        ]
        _, mem_lo = run_vs_w64("wqm_vcc_lo_hi", lines_lo)
        self.assertEqual([l for l in range(64) if mem_lo[8 * l] == 1], [0, 1, 2, 3, 40])
        self.assertEqual({mem_lo[8 * l + 4] for l in range(64)}, {1})

        # vcc starts with lane 8 set; s_wqm_b32 vcc_hi, s14 (lane 1 set -> 0xF) sets lanes 32..35 and keeps lane 8.
        lines_hi = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_cmp_eq_u32_e64 vcc, 8, v1",
            "v_cmp_eq_u32_e64 s[14:15], 1, v1",
            "s_wqm_b32 vcc_hi, s14",
            "s_cselect_b32 s22, 1, 0",
            "v_cndmask_b32_e64 v20, 0, 1, vcc",
            "v_mov_b32_e32 v21, s22",
            "v_mul_u32_u24_e32 v22, 8, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v23, 4, v22",
            "buffer_store_dword v21, v23, s[0:3], 0 offen",
            "s_endpgm",
        ]
        _, mem_hi = run_vs_w64("wqm_vcc_hi_lo", lines_hi)
        self.assertEqual([l for l in range(64) if mem_hi[8 * l] == 1], [8, 32, 33, 34, 35])
        self.assertEqual({mem_hi[8 * l + 4] for l in range(64)}, {1})


class Wqm64MaskPairTests(unittest.TestCase):
    def test_w64_mixed_mask_data_pairs_into_sgpr_pair(self):
        # s14 = lane 5 (bit 5 -> wqm 0xF0), s15 = lane 33 (bit 1 -> wqm 0xF).
        cases = [
            ("lo_data", ["s_mov_b32 s10, s14", "s_mov_b32 s11, 1"], (ref_wqm32(1 << 5), ref_wqm32(1), 1)),
            ("data_hi", ["s_mov_b32 s10, 0x100", "s_mov_b32 s11, s15"], (ref_wqm32(0x100), ref_wqm32(1 << 1), 1)),
            ("hi_data", ["s_mov_b32 s10, s15", "s_mov_b32 s11, 0x10"], (ref_wqm32(1 << 1), ref_wqm32(0x10), 1)),
            ("data_lo", ["s_mov_b32 s10, 5", "s_mov_b32 s11, s14"], (ref_wqm32(5), ref_wqm32(1 << 5), 1)),
        ]
        for label, assign, want in cases:
            with self.subTest(case=label):
                lines = [
                    "v_mbcnt_lo_u32_b32 v1, -1, 0",
                    "v_mbcnt_hi_u32_b32 v1, -1, v1",
                    "v_cmp_eq_u32_e64 s[14:15], 5, v1",
                    "v_cmp_eq_u32_e64 s[16:17], 33, v1",
                    "s_mov_b32 s15, s17",
                    *assign,
                    "s_wqm_b64 s[20:21], s[10:11]",
                    "s_cselect_b32 s22, 1, 0",
                    "v_mov_b32_e32 v20, s20",
                    "v_mov_b32_e32 v21, s21",
                    "v_mov_b32_e32 v22, s22",
                    "v_mul_u32_u24_e32 v23, 12, v1",
                    "buffer_store_dword v20, v23, s[0:3], 0 offen",
                    "v_add_nc_u32_e32 v24, 4, v23",
                    "buffer_store_dword v21, v24, s[0:3], 0 offen",
                    "v_add_nc_u32_e32 v25, 8, v23",
                    "buffer_store_dword v22, v25, s[0:3], 0 offen",
                    "s_endpgm",
                ]
                _, mem = run_vs_w64("wqm_mixed_" + label, lines)
                got = {(mem[12 * l], mem[12 * l + 4], mem[12 * l + 8]) for l in range(64)}
                self.assertEqual(got, {want})

    def test_w64_mixed_and_swapped_pairs_into_exec_and_sgpr_mask(self):
        # Mixed (MASK_LO[lane 5], 1) -> exec: lanes 4..7 + 32..35 active.
        lines_exec_mixed = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_cmp_eq_u32_e64 s[14:15], 5, v1",
            "s_mov_b32 s10, s14",
            "s_mov_b32 s11, 1",
            "v_mov_b32_e32 v20, 0",
            "s_wqm_b64 exec, s[10:11]",
            "v_mov_b32_e32 v20, 1",
            "s_mov_b64 exec, -1",
            "v_mul_u32_u24_e32 v22, 4, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "s_endpgm",
        ]
        _, mem_em = run_vs_w64("wqm_exec_mixed", lines_exec_mixed)
        self.assertEqual([l for l in range(64) if mem_em.get(4 * l, 0) == 1], [4, 5, 6, 7, 32, 33, 34, 35])

        # Swapped (MASK_HI[lane 37 -> bit 5 -> 4..7], MASK_LO[lane 2 -> bit 2 -> 32..35]) -> exec.
        lines_exec_swap = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_cmp_eq_u32_e64 s[14:15], 2, v1",
            "v_cmp_eq_u32_e64 s[16:17], 37, v1",
            "s_mov_b32 s10, s17",
            "s_mov_b32 s11, s14",
            "v_mov_b32_e32 v20, 0",
            "s_wqm_b64 exec, s[10:11]",
            "v_mov_b32_e32 v20, 1",
            "s_mov_b64 exec, -1",
            "v_mul_u32_u24_e32 v22, 4, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "s_endpgm",
        ]
        _, mem_es = run_vs_w64("wqm_exec_swap", lines_exec_swap)
        self.assertEqual([l for l in range(64) if mem_es.get(4 * l, 0) == 1], [4, 5, 6, 7, 32, 33, 34, 35])

        # Non-canonical mask pairs into s[20:21] (tracked as canonical mask by lane_masks.py):
        # verify downstream s_bcnt1_i32_b64 and s_ff1_i32_b64 read exact bits!
        mask_cases = [
            ("hi_zero", ["s_mov_b32 s10, s17", "s_mov_b32 s11, 0"], (4, 4)),
            ("zero_lo", ["s_mov_b32 s10, 0", "s_mov_b32 s11, s14"], (4, 36)),
            ("dup_lo", ["s_mov_b32 s10, s14", "s_mov_b32 s11, s14"], (8, 4)),
        ]
        for label, assign, want in mask_cases:
            with self.subTest(case=label):
                lines = [
                    "v_mbcnt_lo_u32_b32 v1, -1, 0",
                    "v_mbcnt_hi_u32_b32 v1, -1, v1",
                    "v_cmp_eq_u32_e64 s[14:15], 5, v1",
                    "v_cmp_eq_u32_e64 s[16:17], 37, v1",
                    *assign,
                    "s_wqm_b64 s[20:21], s[10:11]",
                    "s_bcnt1_i32_b64 s24, s[20:21]",
                    "s_ff1_i32_b64 s25, s[20:21]",
                    "v_mov_b32_e32 v20, s24",
                    "v_mov_b32_e32 v21, s25",
                    "v_mul_u32_u24_e32 v22, 8, v1",
                    "buffer_store_dword v20, v22, s[0:3], 0 offen",
                    "v_add_nc_u32_e32 v23, 4, v22",
                    "buffer_store_dword v21, v23, s[0:3], 0 offen",
                    "s_endpgm",
                ]
                _, mem = run_vs_w64("wqm_nc_" + label, lines)
                got = {(mem[8 * l], mem[8 * l + 4]) for l in range(64)}
                self.assertEqual(got, {want})

    def test_partial_wave_either_minus1_sets_scc(self):
        # 20 launched threads: s10 = empty MASK_LO, s11 = -1 (EITHER) -> high word is 0xFFFFFFFF -> SCC must be 1!
        lines = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_cmp_eq_u32_e64 s[14:15], 30, v1",
            "s_mov_b32 s10, s14",
            "s_mov_b32 s11, -1",
            "s_wqm_b64 s[20:21], s[10:11]",
            "s_cselect_b32 s22, 1, 0",
            "v_mov_b32_e32 v20, s22",
            "v_mul_u32_u24_e32 v22, 4, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv = support.translate_lines("cs_wqm_partial_m1", lines, fssm=make_fssm(True, 20))
        self.assertEqual(spirv_asm.assemble(spv, target_env="vulkan1.2")[:4], b"\x03\x02\x23\x07")
        mem = si.run_workgroup(
            spv,
            20,
            64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28,
            memory={},
        )
        self.assertEqual({mem[4 * l] for l in range(20)}, {1})

    def test_wqm_mask_dest_uses_shift_and_mask_not_bitfield_uextract(self):
        # Scalar/constant WQM into EXEC avoids OpBitFieldUExtract (Intel 16-bit offset bug).
        spv = support.translate_lines(
            "ps_wqm_exec_imm",
            ["s_wqm_b32 exec_lo, 1", "v_mov_b32 v0, 0", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"],
        )
        self.assertNotIn("OpBitFieldUExtract", spv)
        self.assertEqual(spirv_asm.assemble(spv, target_env="vulkan1.2")[:4], b"\x03\x02\x23\x07")

    def test_plain_data_vcc_and_null_dest(self):
        # Plain-data vcc: vcc_lo = 0, vcc_hi = 1 -> s_wqm_b64 must produce (0, 0xF) and SCC = 1.
        lines_vcc = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "s_mov_b32 vcc_lo, 0",
            "s_mov_b32 vcc_hi, 1",
            "s_wqm_b64 s[20:21], vcc",
            "s_cselect_b32 s22, 1, 0",
            "v_mov_b32_e32 v20, s20",
            "v_mov_b32_e32 v21, s21",
            "v_mov_b32_e32 v22, s22",
            "v_mul_u32_u24_e32 v23, 12, v1",
            "buffer_store_dword v20, v23, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v24, 4, v23",
            "buffer_store_dword v21, v24, s[0:3], 0 offen",
            "v_add_nc_u32_e32 v25, 8, v23",
            "buffer_store_dword v22, v25, s[0:3], 0 offen",
            "s_endpgm",
        ]
        _, mem_vcc = run_vs_w64("wqm_plain_vcc", lines_vcc)
        self.assertEqual({(mem_vcc[12 * l], mem_vcc[12 * l + 4], mem_vcc[12 * l + 8]) for l in range(64)}, {(0, 0xF, 1)})

        # Plain-data s_wqm_b64 into null sets SCC without emitting subgroup ops.
        lines_null = [
            "s_mov_b32 s10, 0",
            "s_mov_b32 s11, 2",
            "s_wqm_b64 null, s[10:11]",
            "s_cselect_b32 s22, 1, 0",
            "v_mov_b32_e32 v20, s22",
            "v_mov_b32_e32 v22, 0",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv_null, mem_null = run_vs_w64("wqm_null_dest", lines_null)
        self.assertEqual(spv_null.count("OpGroupNonUniform"), 0)
        self.assertEqual(mem_null[0], 1)

    def test_canonical_pair_keeps_quad_fast_path_and_emulate_assembles(self):
        # Canonical wave64 (MASK_LO, MASK_HI): keeps quad_any path with zero wave_mask BitwiseOr ballot!
        lines_canon = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_cmp_eq_u32_e64 s[14:15], 5, v1",
            "v_mov_b32_e32 v20, 0",
            "s_wqm_b64 exec, s[14:15]",
            "v_mov_b32_e32 v20, 1",
            "s_mov_b64 exec, -1",
            "v_mul_u32_u24_e32 v22, 4, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "s_endpgm",
        ]
        spv_c, mem_c = run_vs_w64("wqm_canon", lines_canon)
        self.assertEqual(spv_c.count("OpGroupNonUniformBitwiseOr"), 0)
        self.assertEqual([l for l in range(64) if mem_c.get(4 * l, 0) == 1], [4, 5, 6, 7])

        # --wave64-emulate translates and assembles cleanly for both mixed b64 and opposite-half b32.
        emu_lines = [
            "v_mbcnt_lo_u32_b32 v1, -1, 0",
            "v_mbcnt_hi_u32_b32 v1, -1, v1",
            "v_cmp_eq_u32_e64 s[14:15], 32, v1",
            "s_mov_b32 s10, s14",
            "s_mov_b32 s11, 1",
            "s_wqm_b64 s[20:21], s[10:11]",
            "s_wqm_b32 exec_lo, s15",
            "v_mov_b32_e32 v20, s20",
            "v_mul_u32_u24_e32 v22, 4, v1",
            "buffer_store_dword v20, v22, s[0:3], 0 offen",
            "s_endpgm",
        ]
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = True
        try:
            spv_e = support.translate_lines("cs_w64x_wqm_mixed", emu_lines, fssm=make_fssm(True, 64))
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        self.assertGreater(spv_e.count("OpAtomicOr"), 0)
        self.assertNotIn("OpGroupNonUniform", spv_e)
        self.assertEqual(spirv_asm.assemble(spv_e, target_env="vulkan1.2")[:4], b"\x03\x02\x23\x07")


if __name__ == "__main__":
    unittest.main()
