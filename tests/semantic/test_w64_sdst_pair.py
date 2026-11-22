"""A wave64 VALU lane mask written to an SGPR printed as one register is the SGPR pair.

shader_disasm.py disassembles for gfx1013 without +wavefrontsize64, so the 64-bit SDST of a wave64
v_cmp_*_e64 (or the carry-out of v_add_co_u32) prints as one register: v_cmp_gt_u32_e64 s10, 40, v4 writes
s[10:11]. lane_masks.py has to count both registers as the mask, or an s_and_b64 / s_or_b64 of two compare results
reads as 64-bit data: its SCC was this lane's bit instead of "any lane of the wave" and s_bcnt1_i32_b64 counted
data bits. GPU test: runtime/test/shaders/cs_w64_sdst_pair.asm.

Run directly:
    python -I tests/semantic/test_w64_sdst_pair.py -v
"""
import os
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_wave_interp as wi  # noqa: E402
from support import STUB_DXBC, rdna2spirv  # noqa: E402

LANE_ID = ["v_mbcnt_lo_u32_b32 v4, -1, 0", "v_mbcnt_hi_u32_b32 v4, -1, v4"]


def run_ps64(lines):
    with tempfile.TemporaryDirectory() as tmp:
        dis = os.path.join(tmp, "disasm")
        os.mkdir(dis)
        with open(os.path.join(tmp, "ps_w64_sdst.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        path = os.path.join(dis, "ps_w64_sdst.s")
        body = LANE_ID + lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("; ps_w64_sdst\n" + "".join(f"\t{ln}\n" for ln in body))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(path)
    res = wi.run_wave(spv, wave=64)
    return [o[next(iter(o))] if o is not None else None for o in res["outputs"]]


class Wave64SdstPairTests(unittest.TestCase):
    def check(self, sdst_a, sdst_b, pair):
        # lanes 0-39 and lanes 21-63: the AND is lanes 21-39, 19 lanes, so SCC = 1 in every lane.
        outs = run_ps64([
            f"v_cmp_gt_u32_e64 {sdst_a}, 40, v4",
            f"v_cmp_lt_u32_e64 {sdst_b}, 20, v4",
            f"s_and_b64 s[14:15], {pair[0]}, {pair[1]}",
            "s_cselect_b32 s7, 7, 0",
            "s_bcnt1_i32_b64 s6, s[14:15]",
            "v_cndmask_b32_e64 v2, 0, 1, s[14:15]",
            "v_mov_b32 v0, s7",
            "v_mov_b32 v1, s6",
            "v_mov_b32 v3, 0",
        ])
        for lane in range(64):
            self.assertEqual(outs[lane], (7, 19, 1 if 21 <= lane < 40 else 0, 0), f"lane={lane}")

    def test_sdst_printed_as_one_sgpr(self):
        self.check("s10", "s12", ("s[10:11]", "s[12:13]"))

    def test_sdst_printed_as_pair(self):
        self.check("s[10:11]", "s[12:13]", ("s[10:11]", "s[12:13]"))

    def test_carry_out_printed_as_one_sgpr(self):
        # v_add_co_u32 carry-out: v4 + 0xffffffd8 carries in lanes 40-63 (24 lanes); AND with lanes 0-39 is empty.
        outs = run_ps64([
            "v_add_co_u32_e64 v5, s10, v4, 0xffffffd8",
            "v_cmp_gt_u32_e64 s12, 40, v4",
            "s_or_b64 s[14:15], s[10:11], s[12:13]",
            "s_bcnt1_i32_b64 s6, s[14:15]",
            "s_and_b64 s[16:17], s[10:11], s[12:13]",
            "s_cselect_b32 s7, 7, 0",
            "v_mov_b32 v0, s7",
            "v_mov_b32 v1, s6",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ])
        for lane in range(64):
            self.assertEqual(outs[lane], (0, 64, 0, 0), f"lane={lane}")


if __name__ == "__main__":
    unittest.main()
