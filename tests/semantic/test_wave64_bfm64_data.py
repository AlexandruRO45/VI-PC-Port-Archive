"""s_bfm_b64 with two literal operands in wave64 code: a known constant, read as data or as a lane mask.

Standard library only; no GPU, no LLVM, no game files.

RDNA2 ISA SOP2 S_BFM_B64: D = ((1 << (S0 & 63)) - 1) << (S1 & 63), SCC not written. In wave32 code the translator
already treats D as 64-bit data (test_wave32_bfm64_data.py): the pipeline library builds the first two words of an
S# with `s_bfm_b64 s[88:89], 2, 54`. The same idiom in a wave64 program was lowered as a per-lane mask (every lane
0 or ~0 in both halves), so a data reader got the wrong words: the S# word 1 was not the constant 0x00C00000 and a
v_mov_b32 of either half stored 0 or 0xFFFFFFFF. With two literal operands D is a constant, so it is now moved like
`s_mov_b64 s[N:N+1], <value>`, which serves a mask reader (v_cndmask) as well as a data reader.

Each case writes fspx/<name>.json with the wave size (as tools/shader_library.py does), translates a compute
listing, runs all 64 invocations with run_workgroup and has each store both words at its own slot; the mask cases
run a wave64 pixel shader through spirv_wave_interp.
"""
import json
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

import lane_masks  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402
import support  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0x4000, 0x31016FAC]
RSRC2 = 0x88                    # 4 user SGPRs (the output V#), workgroup id x in s4, thread id x in v0
THREADS = 64
MASK64 = 0xFFFFFFFFFFFFFFFF
CASES = ((2, 54), (1, 63), (8, 28), (32, 8), (32, 32), (32, 0), (32, 48), (0, 5), (64, 0), (65, 66))


def bfm(size, offset):
    return (((1 << (size & 63)) - 1) << (offset & 63)) & MASK64


def translate(body, name="cs_bfm64"):
    lines = list(body) + ["s_endpgm"]
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fspx"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, "disasm", name + ".s"), "w", encoding="utf-8", newline="\n") as f:
            f.write("; " + name + "\n" + "".join(f"\t{ln}\n" for ln in lines))
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fspx", name + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"stage": "cs", "rsrc2": RSRC2, "threads": [THREADS, 1, 1], "wave32": False}, f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(os.path.join(tmp, "disasm", name + ".s"))


def words(size, offset):
    """Each invocation stores s8, s9 at byte 8 * i; returns the set of (lo, hi) over all 64 invocations."""
    spv = translate([f"s_bfm_b64 s[8:9], {size}, {offset}", "v_lshlrev_b32 v3, 3, v0",
                     "v_mov_b32 v5, s8", "v_mov_b32 v6, s9", "buffer_store_dwordx2 v[5:6], v3, s[0:3], 0 offen"])

    def inputs(i):
        return {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i}
    mem = si.run_workgroup(spv, THREADS, 64, inputs_for=inputs, push_constants=VSHARP + [0] * 28, memory={})
    return {(mem.get(8 * i), mem.get(8 * i + 4)) for i in range(THREADS)}


def mask_and_data(size, offset):
    """Wave64 pixel shader: (the lane mask v_cndmask reads from s[0:1], the set of (s0, s1) every lane reads)."""
    spv = support.translate_lines("ps_bfm64_both", [
        "s_mov_b64 exec, -1",
        f"s_bfm_b64 s[0:1], {size}, {offset}",
        "v_cndmask_b32_e64 v0, 0, 1, s[0:1]",
        "v_mov_b32 v1, s0",
        "v_mov_b32 v2, s1",
        "v_mov_b32 v3, 0",
        "exp mrt0 v0, v1, v2, v3 done vm",
        "s_endpgm",
    ])
    outs = wi.run_wave(spv, wave=64)["outputs"]
    mask = sum(out["%out_mrt0"][0] << lane for lane, out in enumerate(outs))
    return mask, {tuple(out["%out_mrt0"][1:3]) for out in outs}


class Wave64LiteralBfmTests(unittest.TestCase):
    def test_value_in_every_invocation(self):
        for size, offset in CASES:
            with self.subTest(size=size, offset=offset):
                v = bfm(size, offset)
                self.assertEqual(words(size, offset), {(v & 0xFFFFFFFF, v >> 32)})

    def test_sampler_words_are_known_constants(self):
        # The S# built in place reaches the literal samplers, as in wave32 and as one moved with s_mov_b64 does.
        spv = translate(["s_bfm_b64 s[88:89], 2, 54", "s_mov_b64 s[90:91], 0x18500000",
                         "s_load_dwordx8 s[80:87], s[0:1], 0x0", "v_mov_b32 v1, 0", "v_mov_b32 v2, 0",
                         "s_waitcnt lgkmcnt(0)",
                         "image_sample_lz v5, v[1:2], s[80:87], s[88:91] dmask:0x1 dim:SQ_RSRC_IMG_2D",
                         "s_waitcnt vmcnt(0)", "buffer_store_dword v5, v0, s[0:3], 0 offen"])
        self.assertIn("OpImageSample", spv)
        self.assertIn("OpConstant %uint 12582912", spv)         # word 1 = (3 << 54) >> 32, folded

    def test_the_same_constant_still_works_as_a_lane_mask(self):
        for size, offset in CASES:
            with self.subTest(size=size, offset=offset):
                v = bfm(size, offset)
                mask, data = mask_and_data(size, offset)
                self.assertEqual(mask, v)
                self.assertEqual(data, {(v & 0xFFFFFFFF, v >> 32)})

    def test_lane_masks_classifies_it_as_the_immediate_copy(self):
        def kinds(size, offset):
            st = {}
            lane_masks.step(st, {"op": "s_bfm_b64", "mods": [], "args": [
                {"k": "sgpr", "r": 8, "n": 2}, {"k": "imm", "v": size}, {"k": "imm", "v": offset}]})
            return st.get(8), st.get(9)
        self.assertEqual(kinds(2, 54), (None, None))
        self.assertEqual(kinds(0, 5), (lane_masks.EITHER, lane_masks.EITHER))
        self.assertEqual(kinds(64, 0), (lane_masks.EITHER, lane_masks.EITHER))


if __name__ == "__main__":
    unittest.main()
