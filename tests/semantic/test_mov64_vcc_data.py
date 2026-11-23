"""Regression tests for SHADER-1263: s_mov_b64 to/from VCC when VCC holds 64-bit scalar data."""
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
from support import translate_lines  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def _run_cs(name, body_lines, wave_size=64, emulate=False):
    prev_emu = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", name + ".s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"; {name}\n")
                for ln in body_lines:
                    f.write(f"\t{ln}\n")
                f.write("\ts_endpgm\n")
            with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
                f.write(bytes(0x138) + struct.pack("<4I", 0x8088, wave_size, 1, 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
                gc.collect()
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu
    sg_size = 32 if emulate else wave_size
    return si.run_workgroup(
        spv, wave_size, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=VSHARP + [0] * 28, memory={},
    )


class Mov64VccDataTests(unittest.TestCase):
    def test_s_mov_b64_data_pair_through_vcc_preserves_both_halves(self):
        lines = [
            "s_mov_b32 s2, 0x11223344",
            "s_mov_b32 s3, 0x55667788",
            "s_mov_b64 vcc, s[2:3]",
            "s_mov_b64 s[4:5], vcc",
            "v_mov_b32 v0, s4",
            "v_mov_b32 v1, s5",
            "exp mrt0 v0, v1, v0, v1 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mov64_vcc_data", lines)
        self.assertIn("OpStore %vcc_hi", spv)
        self.assertIn("OpLoad %uint %vcc_hi", spv)
        self.assertNotIn("OpShiftRightLogical", spv)

    def test_s_mov_b64_lane_mask_through_vcc_keeps_mask_semantics(self):
        lines = [
            "v_cmp_lt_f32 vcc, v0, v1",
            "s_mov_b64 s[2:3], vcc",
            "s_mov_b64 vcc, s[2:3]",
            "v_cndmask_b32 v2, v0, v1, vcc",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mov64_vcc_mask", lines)
        self.assertNotIn("OpLoad %uint %vcc_hi", spv)
        self.assertNotIn("OpShiftRightLogical", spv)

    def test_s_mov_b64_data_pair_into_vcc_then_mask_read_extracts_lane_bit(self):
        lines = [
            "s_mov_b32 s2, 2",
            "s_mov_b32 s3, 0",
            "s_mov_b64 vcc, s[2:3]",
            "v_mov_b32 v0, 10",
            "v_mov_b32 v1, 20",
            "v_cndmask_b32 v2, v0, v1, vcc",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mov64_vcc_data_then_cndmask", lines)
        self.assertIn("OpShiftRightLogical %uint", spv)

    def test_s_mov_b64_null_into_vcc_and_sgpr(self):
        lines = [
            "s_mov_b64 vcc, null",
            "s_mov_b64 s[2:3], null",
            "v_mov_b32 v0, s2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_mov64_null", lines)
        self.assertIn("OpStore %vcc %c0", spv)

    def test_wave64_executed_vcc_roundtrip_and_mask_reads(self):
        # 1) 64-bit data round-trip through VCC preserves both halves.
        # 2) s_and_b64 vcc, exec, s[4:5] followed by v_cndmask_b32 covers lanes 0..63.
        # 3) s_mov_b32 vcc_hi, s5; s_mov_b64 vcc, null clears vcc_hi across lanes 32..63.
        lo_val, hi_val = 0x11223344, 0x80000005
        mem = _run_cs("cs_w64_vcc_data_and_mask", [
            "s_mov_b64 exec, -1",
            f"s_mov_b32 s4, {lo_val:#x}",
            f"s_mov_b32 s5, {hi_val:#x}",
            "s_mov_b64 vcc, s[4:5]",
            "s_mov_b64 s[6:7], vcc",
            "s_and_b64 vcc, exec, s[4:5]",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 1",
            "v_cndmask_b32 v3, v1, v2, vcc",
            "s_mov_b32 vcc_hi, s5",
            "s_mov_b64 vcc, null",
            "v_cndmask_b32 v4, v1, v2, vcc",
            "v_lshlrev_b32 v5, 2, v0",
            "buffer_store_dword v3, v5, s[0:3], 0 offen",
            "buffer_store_dword v4, v5, s[0:3], 256 offen",
            "v_mov_b32 v6, s6",
            "v_mov_b32 v7, s7",
            "buffer_store_dword v6, v5, s[0:3], 512 offen",
            "buffer_store_dword v7, v5, s[0:3], 768 offen",
        ], wave_size=64)
        self.assertEqual(mem[512], lo_val)
        self.assertEqual(mem[768], hi_val)
        mask64 = (hi_val << 32) | lo_val
        for lane in range(64):
            self.assertEqual(mem[lane * 4], (mask64 >> lane) & 1, f"lane {lane} s_and_b64 vcc")
            self.assertEqual(mem[256 + lane * 4], 0, f"lane {lane} s_mov_b64 vcc, null")

    def test_wave32_executed_s_mov_b32_exec_lo_from_vcc_data(self):
        word = 0x0000F0F1
        mem = _run_cs("cs_w32_exec_from_vcc_lo", [
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32 v1, 0",
            "v_lshlrev_b32 v2, 2, v0",
            "buffer_store_dword v1, v2, s[0:3], 0 offen",
            f"s_mov_b32 s4, {word:#x}",
            "s_mov_b32 s5, 0",
            "s_mov_b64 vcc, s[4:5]",
            "s_mov_b32 exec_lo, vcc_lo",
            "v_mov_b32 v3, 1",
            "buffer_store_dword v3, v2, s[0:3], 0 offen",
        ], wave_size=32)
        for lane in range(32):
            self.assertEqual(mem[lane * 4], (word >> lane) & 1, f"lane {lane}")


if __name__ == "__main__":
    unittest.main()
