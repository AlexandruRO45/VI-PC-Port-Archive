"""Semantic regression tests for SHADER-1263: Gen.mov64 preserving 64-bit data to/from VCC.

On RDNA2, `s_mov_b64 vcc, s[r:r+1]` and `s_mov_b64 s[r:r+1], vcc` copy all 64 bits verbatim. When `lane_masks.py`
marks `vcc` or the source SGPR pair as plain 64-bit scalar data rather than a lane mask, `Gen.mov64` must read and
write both `%vcc` (`vcc_lo`) and `%vcc_hi` (`vcc_hi`) without converting data into per-lane mask bits or duplicating
`%vcc` into `vcc_hi`. When `vcc` / `s[r:r+1]` holds a certified lane mask, existing wave32 and wave64 mask-saving and
restoring semantics stay intact.

Translated by `tools/rdna2spirv.py` and executed per-lane on `tests/semantic/spirv_mini_interp.py`; no GPU, LLVM,
SPIRV-Tools, or game files required.
"""
from __future__ import annotations

import gc
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (HERE, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from support import STUB_DXBC, mrt0_words, translate_lines  # noqa: E402

M32 = 0xFFFFFFFF
VSHARP = [0, 0, M32, 0x31016FAC]


def make_fssm(threads: int, *, wave32: bool = False) -> bytes:
    header = bytearray(0x138 + 16)
    if wave32:
        struct.pack_into("<I", header, 0xD4, 0x20)
    struct.pack_into("<4I", header, 0x138, 0x8090, threads, 1, 1)
    return bytes(header)


def translate_cs(name: str, lines: list[tuple[str, int]], threads: int = 64, *, wave32: bool = False) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", name + ".s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n")
            for text, size in lines:
                f.write(f"\t{text} ; encoding: [{','.join(['0x00'] * size)}]\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(make_fssm(threads, wave32=wave32))
        saved_w64g = dict(rdna2spirv.W64G_COUNTS)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
        finally:
            rdna2spirv.W64G_COUNTS.clear()
            rdna2spirv.W64G_COUNTS.update(saved_w64g)
        return spv


def run_cs(spv: str, width: int, s4: int = 0, s5: int = 0) -> dict[int, int]:
    pc = VSHARP + [s4 & M32, s5 & M32, 0, 0] + [0] * 24
    return si.run_workgroup(
        spv,
        width,
        width,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc,
        memory={},
    )


class Mov64VccDataTests(unittest.TestCase):
    def test_wave32_mov64_data_pair_to_and_from_vcc_preserves_both_halves(self):
        spv = translate_lines("ps_mov64_vcc_data_w32", [
            "s_mov_b32 s4, 0x11223344",
            "s_mov_b32 s5, 0x8000000a",
            "s_mov_b64 vcc, s[4:5]",
            "s_mov_b64 s[6:7], vcc",
            "v_mov_b32 v0, s6",
            "v_mov_b32 v1, s7",
            "v_mov_b32 v2, vcc_lo",
            "v_mov_b32 v3, vcc_hi",
            "exp mrt0 v0, v1, v2, v3 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpStore %vcc_hi", spv)
        self.assertIn("OpLoad %uint %vcc_hi", spv)
        self.assertNotIn("OpShiftRightLogical", spv)
        want = [0x11223344, 0x8000000A, 0x11223344, 0x8000000A]
        for lane in (0, 1, 5, 31):
            with self.subTest(lane=lane):
                self.assertEqual(mrt0_words(spv, inputs={"%sg_lane": lane}), want)

    def test_wave64_mov64_data_pair_roundtrip_and_vccz_branch(self):
        spv = translate_cs("cs_w64_mov64_vcc_data_roundtrip", [
            ("s_mov_b64 s[12:13], exec", 4),
            ("s_mov_b64 vcc, s[4:5]", 4),
            ("s_mov_b64 s[6:7], vcc", 4),
            ("s_mov_b32 s20, 7", 4),
            ("s_cbranch_vccz 1", 4),
            ("s_mov_b32 s20, 9", 4),
            ("v_lshlrev_b32_e32 v4, 2, v0", 4),
            ("v_mov_b32_e32 v1, s6", 4),
            ("v_mov_b32_e32 v2, s7", 4),
            ("v_mov_b32_e32 v3, s20", 4),
            ("buffer_store_dword v1, v4, s[0:3], 0 offen", 8),
            ("buffer_store_dword v2, v4, s[0:3], 0 offen offset:256", 8),
            ("buffer_store_dword v3, v4, s[0:3], 0 offen offset:512", 8),
            ("s_endpgm", 4),
        ], threads=64)
        for lo, hi in [(0x11223344, 0x8000000A), (0, 0x8000000A), (0, 0), (M32, 0), (0xCAFEBABE, 0xDEADBEEF)]:
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                mem = run_cs(spv, 64, lo, hi)
                want_br = 7 if (lo | hi) == 0 else 9
                self.assertEqual([mem[4 * l] for l in range(64)], [lo] * 64)
                self.assertEqual([mem[256 + 4 * l] for l in range(64)], [hi] * 64)
                self.assertEqual([mem[512 + 4 * l] for l in range(64)], [want_br] * 64)

    def test_mov64_from_vcc_after_b32_half_writes_preserves_vcc_hi(self):
        for wave32 in (True, False):
            width = 32 if wave32 else 64
            tag = "w32" if wave32 else "w64"
            prelude = [] if wave32 else [("s_mov_b64 s[12:13], exec", 4)]
            spv = translate_cs(f"cs_{tag}_mov64_from_vcc_b32_halves", prelude + [
                ("s_mov_b32 vcc_lo, s4", 4),
                ("s_mov_b32 vcc_hi, s5", 4),
                ("s_mov_b64 s[6:7], vcc", 4),
                ("v_lshlrev_b32_e32 v4, 2, v0", 4),
                ("v_mov_b32_e32 v1, s6", 4),
                ("v_mov_b32_e32 v2, s7", 4),
                ("buffer_store_dword v1, v4, s[0:3], 0 offen", 8),
                ("buffer_store_dword v2, v4, s[0:3], 0 offen offset:256", 8),
                ("s_endpgm", 4),
            ], threads=width, wave32=wave32)
            with self.subTest(wave=width):
                mem = run_cs(spv, width, 0xCAFEBABE, 0xDEADBEEF)
                self.assertEqual([mem[4 * l] for l in range(width)], [0xCAFEBABE] * width)
                self.assertEqual([mem[256 + 4 * l] for l in range(width)], [0xDEADBEEF] * width)

    def test_mov64_data_pair_overwrites_prior_vcc_lane_mask(self):
        for wave32 in (True, False):
            width = 32 if wave32 else 64
            tag = "w32" if wave32 else "w64"
            prelude = [] if wave32 else [("s_mov_b64 s[12:13], exec", 4)]
            spv = translate_cs(f"cs_{tag}_mov64_overwrite_mask_vcc", prelude + [
                ("v_cmp_lt_u32_e32 vcc, 12, v0", 8),
                ("s_mov_b64 vcc, s[4:5]", 4),
                ("s_mov_b64 s[6:7], vcc", 4),
                ("v_lshlrev_b32_e32 v4, 2, v0", 4),
                ("v_mov_b32_e32 v1, s6", 4),
                ("v_mov_b32_e32 v2, s7", 4),
                ("buffer_store_dword v1, v4, s[0:3], 0 offen", 8),
                ("buffer_store_dword v2, v4, s[0:3], 0 offen offset:256", 8),
                ("s_endpgm", 4),
            ], threads=width, wave32=wave32)
            with self.subTest(wave=width):
                mem = run_cs(spv, width, 0x00FF00FF, 0xA5A55A5A)
                self.assertEqual([mem[4 * l] for l in range(width)], [0x00FF00FF] * width)
                self.assertEqual([mem[256 + 4 * l] for l in range(width)], [0xA5A55A5A] * width)

    def test_mov64_either_constants_sgpr_pair_to_vcc(self):
        spv = translate_cs("cs_w64_mov64_either_to_vcc", [
            ("s_mov_b64 s[12:13], exec", 4),
            ("s_mov_b32 s4, 0", 4),
            ("s_mov_b32 s5, -1", 4),
            ("s_mov_b64 vcc, s[4:5]", 4),
            ("s_mov_b64 s[6:7], vcc", 4),
            ("s_mov_b32 s20, 7", 4),
            ("s_cbranch_vccz 1", 4),
            ("s_mov_b32 s20, 9", 4),
            ("v_lshlrev_b32_e32 v4, 2, v0", 4),
            ("v_mov_b32_e32 v1, s6", 4),
            ("v_mov_b32_e32 v2, s7", 4),
            ("v_mov_b32_e32 v3, s20", 4),
            ("buffer_store_dword v1, v4, s[0:3], 0 offen", 8),
            ("buffer_store_dword v2, v4, s[0:3], 0 offen offset:256", 8),
            ("buffer_store_dword v3, v4, s[0:3], 0 offen offset:512", 8),
            ("s_endpgm", 4),
        ], threads=64)
        mem = run_cs(spv, 64)
        self.assertEqual([mem[4 * l] for l in range(64)], [0] * 64)
        self.assertEqual([mem[256 + 4 * l] for l in range(64)], [M32] * 64)
        self.assertEqual([mem[512 + 4 * l] for l in range(64)], [9] * 64)

    def test_mov64_mixed_mask_and_data_pair_to_vcc_wave64(self):
        spv = translate_cs("cs_w64_mov64_mixed_pair_to_vcc", [
            ("s_mov_b64 s[12:13], exec", 4),
            ("v_cmp_lt_u32_e64 s[4:5], v0, 5", 8),
            ("s_mov_b32 s5, 0x12345678", 8),
            ("s_mov_b64 vcc, s[4:5]", 4),
            ("s_mov_b64 s[6:7], vcc", 4),
            ("v_lshlrev_b32_e32 v4, 2, v0", 4),
            ("v_mov_b32_e32 v1, s6", 4),
            ("v_mov_b32_e32 v2, s7", 4),
            ("buffer_store_dword v1, v4, s[0:3], 0 offen", 8),
            ("buffer_store_dword v2, v4, s[0:3], 0 offen offset:256", 8),
            ("s_endpgm", 4),
        ], threads=64)
        mem = run_cs(spv, 64)
        self.assertEqual([mem[4 * l] for l in range(64)], [0x0000001F] * 64)
        self.assertEqual([mem[256 + 4 * l] for l in range(64)], [0x12345678] * 64)

    def test_mov64_lane_mask_save_and_restore_preserved(self):
        for wave32 in (True, False):
            width = 32 if wave32 else 64
            tag = "w32" if wave32 else "w64"
            prelude = [] if wave32 else [("s_mov_b64 s[12:13], exec", 4)]
            spv = translate_cs(f"cs_{tag}_mov64_mask_save_restore", prelude + [
                ("v_cmp_gt_u32_e32 vcc, v0, 17", 8),
                ("s_mov_b64 s[6:7], vcc", 4),
                ("s_mov_b64 vcc, 0", 4),
                ("s_mov_b64 vcc, s[6:7]", 4),
                ("s_bcnt1_i32_b64 s8, vcc", 4),
                ("v_cndmask_b32_e64 v1, 10, 20, vcc", 8),
                ("v_mov_b32_e32 v2, s8", 4),
                ("v_lshlrev_b32_e32 v4, 2, v0", 4),
                ("buffer_store_dword v1, v4, s[0:3], 0 offen", 8),
                ("buffer_store_dword v2, v4, s[0:3], 0 offen offset:256", 8),
                ("s_endpgm", 4),
            ], threads=width, wave32=wave32)
            with self.subTest(wave=width):
                mem = run_cs(spv, width)
                self.assertEqual([mem[4 * l] for l in range(width)], [20 if l > 17 else 10 for l in range(width)])
                self.assertEqual([mem[256 + 4 * l] for l in range(width)], [width - 18] * width)


if __name__ == "__main__":
    unittest.main()
