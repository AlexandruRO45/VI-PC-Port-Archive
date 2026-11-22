"""Wave32 vertex shaders on vertex subgroups of 32 and 64 invocations (tools/rdna2spirv.py `Gen.halves`).

Vulkan cannot require a vertex subgroup size (no driver lists VERTEX in requiredSubgroupSizeStages) and the AMD
Windows driver runs the vertex stage 64-wide (measured on an RX 7600, driver 26.11.2, by
runtime/test/shaders/w32_vs_halves.py). A wave32 VS then shares one subgroup with a second wave's worth of
vertices. With halves (the default), subgroup lanes 32k..32k+31 are one wave: the reference here is an independent
model of two wave32s of 32 lanes each, written from the RDNA2 ISA (v_mbcnt_lo, s_bcnt1, s_ff1, v_readfirstlane,
v_readlane, ds_bpermute, s_cbranch_vccz, writes to exec_lo), and the translated module runs through
spirv_wave_interp at 32 and 64 invocations. With --no-wave32-halves the 32-invocation run still matches and the
64-invocation run does not, which is the defect the default fixes. No GPU.
"""
import os
import re
import tempfile
import unittest
import warnings

import support  # noqa: F401  (puts tools/ on the path)
import rdna2spirv
from spirv_wave_interp import run_wave

BASE = 1000            # vertex index of invocation 0
VOTE_VERTEX = BASE + 37  # in the second wave of a 64-invocation subgroup


def sized(lines):
    """Placeholder encodings of the right size, so branch offsets count dwords (as test_export_valid_mask)."""
    out = []
    for line in lines:
        op = line.split()[0]
        vop3 = op.startswith(("v_mbcnt", "v_readlane", "v_cndmask_b32_e64")) or op.startswith("ds_")
        literal = any(int(t, 16) > 64 for t in re.findall(r"0x([0-9a-f]+)", line))
        size = 8 if (op == "exp" or vop3 or literal) else 4
        out.append(f"{line} ; encoding: [" + ",".join(["0x00"] * size) + "]")
    return out


PROLOGUE = ["v_mbcnt_lo_u32_b32 v6, -1, 0", "v_readfirstlane_b32 s20, v0", "v_mov_b32_e32 v9, s20"]
EPILOGUE = ["v_mov_b32_e32 v5, 0", "exp pos0 v5, v5, v5, v5 done", "exp param0 v6, v7, v8, v9",
            "exp param1 v10, v11, v12, v13", "s_endpgm"]

PROGRAMS = {
    # param0 = (lane, s_bcnt1 exec_lo, 0, key); param1 = 0
    "lane": ["s_bcnt1_i32_b32 s4, exec_lo", "v_mov_b32_e32 v7, s4", "v_mov_b32_e32 v8, 0",
             "v_mov_b32_e32 v10, 0", "v_mov_b32_e32 v11, 0", "v_mov_b32_e32 v12, 0", "v_mov_b32_e32 v13, 0"],
    # VCC = v % 4 == 1. param0 = (lane, rank, count, key); param1 = (first, sel, 0, 0)
    "ballot": ["v_and_b32_e32 v4, 3, v0", "v_cmp_eq_u32_e32 vcc_lo, 1, v4", "v_mbcnt_lo_u32_b32 v7, vcc_lo, 0",
               "s_bcnt1_i32_b32 s6, vcc_lo", "v_mov_b32_e32 v8, s6", "s_ff1_i32_b32 s7, vcc_lo",
               "v_mov_b32_e32 v10, s7", "v_cndmask_b32_e64 v11, 0, 1, vcc_lo", "v_mov_b32_e32 v12, 0",
               "v_mov_b32_e32 v13, 0"],
    # param0 = (lane, readlane 5, readlane 31, key); param1 = (bpermute lane ^ 1, bpermute (lane + 1) & 31, 0, 0)
    "shuffle": ["v_readlane_b32 s6, v0, 5", "v_mov_b32_e32 v7, s6", "v_readlane_b32 s7, v0, 31",
                "v_mov_b32_e32 v8, s7", "v_xor_b32_e32 v14, 1, v6", "v_lshlrev_b32_e32 v14, 2, v14",
                "v_add_nc_u32_e32 v15, 1, v6", "v_and_b32_e32 v15, 31, v15", "v_lshlrev_b32_e32 v15, 2, v15",
                "ds_bpermute_b32 v10, v14, v0", "ds_bpermute_b32 v11, v15, v0", "s_waitcnt lgkmcnt(0)",
                "v_mov_b32_e32 v12, 0", "v_mov_b32_e32 v13, 0"],
    # param0 = (lane, s5 after s_cbranch_vccz on v == VOTE_VERTEX, low16 under exec_lo = 0xffff, key); param1 = 0
    "vote": ["s_mov_b32 s5, 0", f"v_mov_b32_e32 v4, 0x{VOTE_VERTEX:x}", "v_cmp_eq_u32_e32 vcc_lo, v4, v0",
             "s_cbranch_vccz 1", "s_mov_b32 s5, 7", "v_mov_b32_e32 v7, s5", "s_mov_b32 s10, exec_lo",
             "v_mov_b32_e32 v8, 0", "s_mov_b32 exec_lo, 0xffff", "v_mov_b32_e32 v8, 1", "s_mov_b32 exec_lo, s10",
             "v_mov_b32_e32 v10, 0", "v_mov_b32_e32 v11, 0", "v_mov_b32_e32 v12, 0", "v_mov_b32_e32 v13, 0"],
}


def model(name, invocations):
    """(param0, param1) per invocation for two (or one) independent wave32s of 32 lanes each."""
    rows = []
    for inv in range(invocations):
        wave, lane = divmod(inv, 32)
        vs = [BASE + 32 * wave + l for l in range(32)]           # the wave's vertices, by lane
        v, key = vs[lane], vs[0]
        if name == "lane":
            rows.append(((lane, 32, 0, key), (0, 0, 0, 0)))
        elif name == "ballot":
            sel = [l for l in range(32) if vs[l] % 4 == 1]
            rank = sum(1 for l in sel if l < lane)
            first = sel[0] if sel else 0xFFFFFFFF
            rows.append(((lane, rank, len(sel), key), (first, int(v % 4 == 1), 0, 0)))
        elif name == "shuffle":
            rows.append(((lane, vs[5], vs[31], key), (vs[lane ^ 1], vs[(lane + 1) & 31], 0, 0)))
        elif name == "vote":
            rows.append(((lane, 7 if VOTE_VERTEX in vs else 0, int(lane < 16), key), (0, 0, 0, 0)))
    return rows


def translate(name, halves):
    body = sized(PROLOGUE + PROGRAMS[name] + EPILOGUE)
    asm = f"; vs_w32_{name}\n" + "".join(f"\t{ln}\n" for ln in body)
    saved = rdna2spirv.WAVE32_HALVES
    rdna2spirv.WAVE32_HALVES = halves
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            src = os.path.join(tmp, "disasm", f"vs_w32_{name}.s")
            with open(src, "w", encoding="utf-8") as f:
                f.write(asm)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(src)
    finally:
        rdna2spirv.WAVE32_HALVES = saved


def run(spv, invocations):
    res = run_wave(spv, wave=invocations, builtins={"VertexIndex": lambda lane: BASE + lane,
                                                    "InstanceIndex": lambda lane: 0})
    return [(o["%out_param0"], o["%out_param1"]) for o in res["outputs"]]


class Wave32VertexHalves(unittest.TestCase):
    def check(self, name):
        spv = translate(name, halves=True)
        for n in (32, 64):
            with self.subTest(program=name, invocations=n):
                self.assertEqual(run(spv, n), model(name, n))

    def test_lane_count_and_first_lane(self):
        self.check("lane")

    def test_ballot_rank_count_first(self):
        self.check("ballot")

    def test_readlane_and_bpermute_stay_in_the_wave(self):
        self.check("shuffle")

    def test_vccz_vote_and_exec_lo_write(self):
        self.check("vote")

    def test_reads_this_waves_ballot_word(self):
        spv = translate("lane", halves=True)
        self.assertIn("OpVectorExtractDynamic", spv)


class WithoutHalves(unittest.TestCase):
    """--no-wave32-halves: what main did. Exact on 32 invocations, wrong on 64 (each program)."""

    def test_exact_on_32_wrong_on_64(self):
        for name in PROGRAMS:
            spv = translate(name, halves=False)
            with self.subTest(program=name, invocations=32):
                self.assertEqual(run(spv, 32), model(name, 32))
            with self.subTest(program=name, invocations=64):
                got, want = run(spv, 64), model(name, 64)
                self.assertNotEqual(got, want)
                wrong = sum(1 for g, w in zip(got, want) if g != w)
                self.assertGreaterEqual(wrong, 32, f"{name}: {wrong} of 64 invocations differ")


if __name__ == "__main__":
    unittest.main()
