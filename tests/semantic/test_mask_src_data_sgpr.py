"""Gen.mask_src on an SGPR that is plain data, not a lane mask (SHADER-200, from the finding on #1218).

The translator's per-lane mask form is 0xFFFFFFFF or 0. An SGPR that lane_masks.py does not mark holds one value for
the whole wave; bit N of it is lane N's mask bit (ISA: s_and_saveexec_b32 with s4 = 0xff00ff leaves 16 lanes on, lanes
0-7 and 16-23). Until this change every lane read the raw value, so any non-zero SGPR meant "all lanes on". For a wave64
pair s[r:r+1], lanes 32-63 take their bit from r+1. A register certified as a mask is read exactly as before.

Each case translates a hand-written compute shader, runs it in tests/semantic/spirv_mini_interp.py and has every lane
store 1 when it is active after s_and_saveexec.
"""
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

TAIL = [
    "v_mov_b32 v8, 1",
    "v_lshlrev_b32 v10, 2, v0",
    "buffer_store_dword v8, v10, s[0:3], 0 offen",
]


def run(name: str, lanes: int, body):
    """Run `body` on `lanes` threads; returns the list of active flags (1 = the lane stored after saveexec)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        s_path = os.path.join(tmp, "disasm", name + ".s")
        with open(s_path, "w", encoding="utf-8") as f:
            f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in list(body) + TAIL) + "\ts_endpgm\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(bytes(0x138) + struct.pack("<4I", 0x8088, lanes, 1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(s_path)
    pc = [0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28
    mem = si.run_workgroup(
        spv, lanes, lanes,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc, memory={},
    )
    return [mem.get(i * 4, 0) for i in range(lanes)]


def bits(v: int, lo: int, n: int):
    return [(v >> (lo + i)) & 1 for i in range(n)]


class MaskSrcDataSgprTests(unittest.TestCase):
    def w32(self, value: str):
        return run("cs_mask_src_w32", 32, ["s_mov_b32 exec_lo, -1", f"s_mov_b32 s4, {value}",
                                           "s_and_saveexec_b32 s6, s4"])

    def test_wave32_partial_data(self):
        got = self.w32("0xff00ff")
        self.assertEqual(sum(got), 16)
        self.assertEqual(got, bits(0xFF00FF, 0, 32))

    def test_wave32_zero_and_all_ones(self):
        self.assertEqual(self.w32("0"), [0] * 32)
        self.assertEqual(self.w32("-1"), [1] * 32)

    def test_wave32_certified_mask_unchanged(self):
        got = run("cs_mask_src_cert32", 32, ["s_mov_b32 exec_lo, -1", "v_cmp_eq_u32_e64 s4, v0, 5",
                                             "s_and_saveexec_b32 s6, s4"])
        self.assertEqual(got, [1 if i == 5 else 0 for i in range(32)])

    def w64(self, lo: str, hi: str):
        return run("cs_mask_src_w64", 64, ["s_mov_b64 exec, -1", f"s_mov_b32 s4, {lo}", f"s_mov_b32 s5, {hi}",
                                           "s_and_saveexec_b64 s[6:7], s[4:5]"])

    def test_wave64_pair_data(self):
        got = self.w64("0xff00ff", "0xf0f0f0f0")
        self.assertEqual(got, bits(0xFF00FF, 0, 32) + bits(0xF0F0F0F0, 0, 32))
        self.assertEqual(sum(got), 16 + 16)

    def test_wave64_high_only_and_zero(self):
        self.assertEqual(self.w64("0", "1"), [0] * 32 + [1] + [0] * 31)
        self.assertEqual(self.w64("0", "0"), [0] * 64)
        self.assertEqual(self.w64("-1", "-1"), [1] * 64)

    def test_wave64_certified_mask_unchanged(self):
        got = run("cs_mask_src_cert64", 64, ["s_mov_b64 exec, -1", "v_cmp_eq_u32_e64 s[4:5], v0, 40",
                                             "s_and_saveexec_b64 s[6:7], s[4:5]"])
        self.assertEqual(got, [1 if i == 40 else 0 for i in range(64)])


if __name__ == "__main__":
    unittest.main()
