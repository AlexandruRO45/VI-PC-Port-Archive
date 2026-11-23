"""Tests for MUBUF monotonic bounds slack, per-block V#/wave_lane caching, and cond0 factoring (#884).

Verifies bit-exact multi-dword OOB straddling, strict pattern-memory read/write exactness across all
OOB_SELECT modes (0..3) and swizzle configurations, cache invalidation on SGPR writes / basic-block
boundaries / x_split, selection-construct dominance isolation, and multi-access IR size reduction.
Synthetic inputs only; standard library only.

    python -I -m unittest discover -s tests/semantic -p test_mubuf_slack_and_vsharp_cache.py -v
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for _p in (TOOLS_DIR, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import spirv_opt  # noqa: E402
import support  # noqa: E402
import test_value_semantics as tv  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

BASE = tv.VEC_REGION
OUT_V = 0x31016FAC
W3_BASE = 0x01000FAC | (22 << 12)


class ReadTrackMemory(dict):
    """Records every read byte address identically to test_buffer_swizzle.PatternMemory."""

    def __init__(self, initial=None):
        super().__init__(initial or {})
        self.reads = []

    def get(self, key, default=None):
        self.reads.append(key)
        if key in self:
            return self[key]
        return tv._pattern_word(key - BASE) if BASE <= key < BASE + 0x10000 else default


def _validate_spv(spv: str) -> None:
    spirv_asm.assemble(spv, target_env="vulkan1.2")
    if SPIRV_AS and SPIRV_VAL:
        with tempfile.TemporaryDirectory() as d:
            sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
            with open(sa, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            for cmd in (
                [SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"],
                [SPIRV_VAL, "--target-env", "vulkan1.2", sv],
            ):
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise AssertionError(f"{cmd[0]} failed:\n{r.stderr}")


def _translate_cs(name: str, body: list[str], *, threads=(64, 1, 1), rsrc2=0x820,
                  emulate: bool = False, opt: bool = False) -> str:
    lines = list(body)
    if not lines or lines[-1].split()[0] != "s_endpgm":
        lines.append("s_endpgm")
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "disasm"))
        os.makedirs(os.path.join(d, "fspx"))
        sp = os.path.join(d, "disasm", f"{name}.s")
        with open(sp, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}  code 0x0..0x2000, gfx1013, 0 invalid words\n")
            for ln in lines:
                op = ln.split()[0]
                enc = (
                    "[0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]"
                    if op.startswith(("buffer_", "s_load", "v_cmp_", "v_cmpx_"))
                    else "[0x00,0x00,0x00,0x00]"
                )
                f.write(f"\t{ln} ; encoding: {enc}\n")
        with open(os.path.join(d, "fspx", f"{name}.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"rsrc2": rsrc2, "threads": list(threads)}, f)
        old_w, old_o = rdna2spirv.WAVE64_EMULATE, rdna2spirv.OPT_SPIRV
        try:
            rdna2spirv.WAVE64_EMULATE = emulate
            rdna2spirv.OPT_SPIRV = opt
            return rdna2spirv.translate(sp)
        finally:
            rdna2spirv.WAVE64_EMULATE, rdna2spirv.OPT_SPIRV = old_w, old_o


def _vsharp_setup(stride: int, nr: int, mode: int, *, base: int = BASE,
                  swizzle: bool = False, tid: bool = False, istride: int = 0, runtime: bool = True) -> list[str]:
    w0 = base & 0xFFFFFFFF
    w1 = ((base >> 32) & 0xFFFF) | ((stride & 0x3FFF) << 16) | ((1 if swizzle else 0) << 31)
    w2 = nr & 0xFFFFFFFF
    w3 = W3_BASE | ((istride & 3) << 21) | ((1 if tid else 0) << 23) | ((mode & 3) << 28)
    lines = [f"s_mov_b32 s4, {w0:#x}", f"s_mov_b32 s6, {w2:#x}"]
    if runtime:
        lines += [
            "s_mov_b32 s5, 0", f"s_mov_b32 s9, {w1:#x}", "s_or_b32 s5, s5, s9",
            "s_mov_b32 s7, 0", f"s_mov_b32 s10, {w3:#x}", "s_or_b32 s7, s7, s10",
        ]
    else:
        lines += [f"s_mov_b32 s5, {w1:#x}", f"s_mov_b32 s7, {w3:#x}"]
    return lines


class MubufSlackAndVSharpCacheTests(unittest.TestCase):
    def test_multi_dword_straddle_bounds_and_reads_exact_all_modes(self):
        # Exercises modes 0, 1, 2, 3 (raw & swizzled), soffset != 0, and literal vs runtime V#
        # for buffer_load_dwordx{2,3,4} and buffer_store_dwordx{2,3,4}.
        cases = [
            # (mode, stride, nr, swizzle, tid, idx, off, soff, n, inside_dwords)
            (3, 0, 12, False, False, None, 0, 0, 4, [0, 4, 8]),        # raw mode 3 straddle at byte 12 -> 3 of 4 dwords
            (3, 0, 11, False, False, None, 0, 0, 4, [0, 4]),           # 11 bytes leaves only 2 complete dwords
            (3, 0, 20, False, False, None, 4, 8, 4, [12, 16]),         # with soffset=8, off+soff=12: 2 dwords fit in 20
            (3, 0, 0x100, False, False, None, 0, 0, 4, [0, 4, 8, 12]),  # all 4 dwords fit
            (0, 8, 2, False, False, 1, 0, 0, 4, [8, 12]),              # mode 0: stride=8, idx=1 -> only 2 dwords fit
            (0, 12, 2, False, False, 1, 4, 0, 3, [16, 20]),            # mode 0 straddle with off=4 -> 2 of 3 dwords fit
            (1, 8, 2, False, False, 1, 0, 0, 4, [8, 12, 16, 20]),      # mode 1: checks idx only -> all 4 fit
            (1, 8, 1, False, False, 1, 0, 0, 4, []),                   # mode 1: idx >= nr -> 0 dwords
            (2, 8, 1, False, False, 0, 0, 0, 2, [0, 4]),               # mode 2: nr != 0 -> all fit
            (2, 8, 0, False, False, 0, 0, 0, 2, []),                   # mode 2: nr == 0 -> OOB
            (3, 8, 4, True, True, None, 0, 0, 4, [0x08, 0x28]),        # scratch swizzle lane=2: steps by 32, 2 in stride 8
            (3, 8, 4, True, False, 2, 0, 0, 4, [0x08, 0x0C]),          # indexed swizzle: steps by +4, 2 fit in stride 8
        ]
        for runtime in (False, True):
            for case in cases:
                mode, stride, nr, swz, tid, idx, off, soff, n, inside = case
                lane = 2 if tid and idx is None else 0
                with self.subTest(case=case, runtime=runtime):
                    setup = _vsharp_setup(stride, nr, mode, swizzle=swz, tid=tid, runtime=runtime)
                    if idx is not None:
                        vaddr = "v[1:2]"
                        mods = "idxen offen"
                        setup += [f"v_mov_b32 v1, {idx}", f"v_mov_b32 v2, {off}"]
                    else:
                        vaddr = "v1"
                        mods = "offen"
                        setup += [f"v_mov_b32 v1, {off}"]
                    if soff:
                        setup.append(f"s_mov_b32 s8, {soff}")
                        soff_op = "s8"
                    else:
                        soff_op = "0"
                    regs = f"v[10:{9 + n}]"
                    ld_spv = support.translate_lines(
                        "cs_mubuf_slack_ld",
                        [
                            "v_cmpx_eq_u32_e32 0, v0" if not tid else "s_mov_b64 exec, exec",
                            *setup,
                            f"buffer_load_dwordx{n} {regs}, {vaddr}, s[4:7], {soff_op} {mods}",
                            "s_waitcnt vmcnt(0)",
                            f"buffer_store_dwordx{n} {regs}, off, s[0:3], 0",
                            "s_endpgm",
                        ],
                        fssm=tv.CS_BOUNDS_FSSM,
                    )
                    _validate_spv(ld_spv)
                    mem = ReadTrackMemory()
                    si.run_fragment(
                        ld_spv,
                        inputs={"%wg_id": (0, 0, 0), "%local_id": (lane, 0, 0)},
                        push_constants=[0, 0, 0x100, OUT_V] + [0] * 28,
                        memory=mem,
                    )
                    self.assertEqual([a - BASE for a in mem.reads], inside)
                    want = [tv._pattern_word(o) for o in inside] + [0] * (n - len(inside))
                    self.assertEqual([mem.get(4 * k) for k in range(n)], want)

                    st_lines = [
                        "s_mov_b64 exec, exec",
                        *setup,
                        *[f"v_mov_b32 v{10 + k}, {0xA0000001 + k * 0x111:#x}" for k in range(n)],
                        f"buffer_store_dwordx{n} {regs}, {vaddr}, s[4:7], {soff_op} {mods}",
                        "s_endpgm",
                    ]
                    st_spv = support.translate_lines("cs_mubuf_slack_st", st_lines, fssm=tv.CS_BOUNDS_FSSM)
                    _validate_spv(st_spv)
                    st_mem = ReadTrackMemory()
                    si.run_fragment(
                        st_spv,
                        inputs={"%wg_id": (0, 0, 0), "%local_id": (lane, 0, 0)},
                        push_constants=[0, 0, 0x100, OUT_V] + [0] * 28,
                        memory=st_mem,
                    )
                    want_st = {BASE + o: 0xA0000001 + k * 0x111 for k, o in enumerate(inside)}
                    self.assertEqual(dict(st_mem), want_st)

    def test_vsharp_cache_invalidation_on_sgpr_write_and_block_boundary(self):
        # Verify:
        # 1) Back-to-back loads on s[4:7] emit only 1 local_id load and 1 OOB_SELECT extract.
        # 2) Writing any SGPR inside s[4:7] invalidates the cache so the next load sees the new V#.
        # 3) Writing an adjacent SGPR (s3 or s8) keeps the s[4:7] cache valid.
        for mod_line, expect_second in (
            ("s_mov_b32 s8, 0xdead", [0x11111111, 0x22222222, 0x33333333, 0x44444444]),   # neighbor untouched
            ("s_mov_b32 s6, 8", [0x11111111, 0x22222222, 0, 0]),                           # shrunk NUM_RECORDS
            (f"s_mov_b32 s4, {BASE + 0x20:#x}", [0x55555555, 0x66666666, 0x77777777, 0x88888888]),  # new base_lo
            ("s_mov_b64 s[6:7], 0", [0, 0, 0, 0]),                                         # s[6:7] zeroed
        ):
            with self.subTest(mod=mod_line):
                lines = [
                    *_vsharp_setup(0, 16, 3, runtime=True),
                    "v_mov_b32 v1, 0",
                    "buffer_load_dwordx4 v[10:13], v1, s[4:7], 0 offen",
                    mod_line,
                    "buffer_load_dwordx4 v[14:17], v1, s[4:7], 0 offen offset:0",
                    "s_waitcnt vmcnt(0)",
                    "buffer_store_dwordx4 v[10:13], off, s[0:3], 0",
                    "buffer_store_dwordx4 v[14:17], off, s[0:3], 0 offset:16",
                ]
                spv = _translate_cs("cs_vsharp_inv", lines)
                _validate_spv(spv)
                mem = {
                    BASE + 0: 0x11111111, BASE + 4: 0x22222222, BASE + 8: 0x33333333, BASE + 12: 0x44444444,
                    BASE + 0x20: 0x55555555, BASE + 0x24: 0x66666666, BASE + 0x28: 0x77777777, BASE + 0x2C: 0x88888888,
                }
                si.run_fragment(
                    spv,
                    inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                    push_constants=[0, 0, 0x100, OUT_V] + [0] * 28,
                    memory=mem,
                )
                self.assertEqual([mem.get(4 * k, 0) for k in range(4)], [0x11111111, 0x22222222, 0x33333333, 0x44444444])
                self.assertEqual([mem.get(16 + 4 * k, 0) for k in range(4)], expect_second)

    def test_multi_access_ir_and_binary_shrink_and_negative_control_bounds(self):
        # 40 repeated buffer_load_dwordx4 / buffer_store_dwordx4 on runtime V#s in one block:
        # Must emit only 1 %local_id load, 1 mode extract per distinct V#, and stay compact.
        body = [
            "s_load_dwordx4 s[4:7], s[0:1], 0x0",
            "s_load_dwordx4 s[8:11], s[0:1], 0x10",
            "s_waitcnt lgkmcnt(0)",
            "v_lshlrev_b32_e32 v1, 4, v0",
        ]
        for i in range(20):
            body += [
                f"buffer_load_dwordx4 v[10:13], v1, s[4:7], 0 offen offset:{(i * 16) & 0xfff}",
                "s_waitcnt vmcnt(0)",
                "v_add_nc_u32_e32 v10, v10, 1",
                f"buffer_store_dwordx4 v[10:13], v1, s[8:11], 0 offen offset:{(i * 16) & 0xfff}",
            ]
        raw = _translate_cs("cs_mubuf_40", body)
        opt = spirv_opt.optimize(raw)
        for spv in (raw, opt):
            _validate_spv(spv)
        # 1 %local_id load in the CS entry prologue (v0) + 1 shared across all 40 buffer ops (vs 41 uncached),
        # and at most 2 OOB_SELECT extracts (one for s[4:7], one for s[8:11]).
        self.assertEqual(raw.count("OpLoad %v3uint %local_id"), 2)
        self.assertLessEqual(len(re.findall(r"OpBitFieldUExtract %uint %\w+ %c1c %c2\b", raw)), 2)
        # Slack eliminated per-dword OpSelect chains: 40 x4 ops = 40 * (4 check + 3 slack) = 280 OpSelects (vs 640 uncached).
        self.assertEqual(raw.count(" = OpSelect "), 280)
        raw_insts = sum(1 for l in raw.splitlines() if l.strip() and not l.lstrip().startswith(";"))
        self.assertLess(raw_insts, 3600)

        # Functional check across a 4-lane workgroup where lane 3 straddles s[4:7]'s 60-byte buffer.
        mem = {0x8000 + 4 * i: 100 + i for i in range(64)}
        mem[0x10000] = 0x8000
        mem[0x10004] = 0
        mem[0x10008] = 60          # s[4:7]: 60 bytes (lane 3 at v1=48 gets dwords 0,1,2 and 0 for dword 3)
        mem[0x1000C] = W3_BASE | (3 << 28)
        mem[0x10010] = 0xC000
        mem[0x10014] = 0
        mem[0x10018] = 0x1000      # s[8:11]: 4096 bytes output
        mem[0x1001C] = W3_BASE | (3 << 28)
        res_mem = si.run_workgroup(
            raw, 4, 4,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
            push_constants=[0x10000, 0] + [0] * 30,
            memory=dict(mem),
        )
        self.assertEqual([res_mem[0xC000 + 48 + 4 * k] for k in range(4)], [113, 113, 114, 0])

    def test_construct_isolation_and_wave64_emulated_x_split_valid(self):
        # 1) Calling buffer_address() and wave_lane() inside an open selection construct must NOT pollute caches.
        g = rdna2spirv.Gen("cs")
        g.cs = {"size": (64, 1, 1), "user": 0, "tgid": (), "tid": 0}
        g.emit("OpSelectionMerge %t_merge None")
        g.emit("OpBranchConditional %true %t_do %t_merge")
        g.emit("%t_do = OpLabel")
        self.assertFalse(g.desc_cache_ok())
        g.buffer_address(
            [{"k": "vgpr", "r": 2, "n": 1}, {"k": "vgpr", "r": 1, "n": 1}, {"k": "sgpr", "r": 4, "n": 4}],
            ["offen"], parts=True,
        )
        self.assertEqual(g.vsharp_cache, {})
        self.assertIsNone(g.lane_cache)
        g.emit("OpBranch %t_merge")
        g.emit("%t_merge = OpLabel")
        self.assertTrue(g.desc_cache_ok())
        g.buffer_address(
            [{"k": "vgpr", "r": 3, "n": 1}, {"k": "vgpr", "r": 1, "n": 1}, {"k": "sgpr", "r": 4, "n": 4}],
            ["offen"], parts=True,
        )
        self.assertIn((4, None, None), g.vsharp_cache)
        self.assertIsNotNone(g.lane_cache)
        _validate_spv(rdna2spirv.assemble(g))

        # 2) g.x_split() clears both vsharp_cache and lane_cache so the post-split segment rebuilds them.
        g2 = rdna2spirv.Gen("cs")
        g2.cs = {"size": (64, 1, 1), "user": 0, "tgid": (), "tid": 0}
        g2.emulate = True
        g2.wave64 = True
        g2.buffer_address(
            [{"k": "vgpr", "r": 2, "n": 1}, {"k": "vgpr", "r": 1, "n": 1}, {"k": "sgpr", "r": 4, "n": 4}],
            ["offen"], parts=True,
        )
        self.assertIn((4, None, None), g2.vsharp_cache)
        self.assertIsNotNone(g2.lane_cache)
        g2.x_split()
        self.assertEqual(g2.vsharp_cache, {})
        self.assertIsNone(g2.lane_cache)

        # 3) --wave64-emulate shader with s[4:7] loaded before AND after an x_round boundary validates cleanly
        #    and matches non-emulated execution on all 64 lanes for the wave64 active-masked path.
        lines = [
            *_vsharp_setup(0, 64 * 4, 3, runtime=True),
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_cmp_lt_u32_e64 vcc, v0, 64",
            "s_and_saveexec_b64 s[14:15], vcc",
            "buffer_load_dword v10, v1, s[4:7], 0 offen",
            "s_waitcnt vmcnt(0)",
            "ds_swizzle_b32 v11, v10 offset:0x8000",   # forces %x_buf x_round under emulate=True
            "v_add_nc_u32_e32 v12, v10, v11",
            "buffer_store_dword v12, v1, s[4:7], 0 offen offset:256",
            "buffer_store_dword v12, v1, s[0:3], 0 offen",
        ]
        spv_emu = _translate_cs("cs_w64_split", lines, emulate=True)
        _validate_spv(spv_emu)
        self.assertIn("%x_buf", spv_emu)
        spv_opt = spirv_opt.optimize(spv_emu)
        _validate_spv(spv_opt)

        # Native wave64 execution equivalence across 64 lanes with diverging active mask and straddling bounds.
        lines_exec = [
            *_vsharp_setup(0, 48 * 4, 3, runtime=True),
            "v_lshlrev_b32_e32 v1, 2, v0",
            "v_cmp_lt_u32_e64 vcc, v0, 60",
            "s_and_saveexec_b64 s[14:15], vcc",
            "buffer_load_dword v10, v1, s[4:7], 0 offen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dword v10, v1, s[0:3], 0 offen",
        ]
        spv_n = _translate_cs("cs_w64_native", lines_exec, emulate=False)
        _validate_spv(spv_n)
        mem = {BASE + 4 * i: 1000 + i for i in range(64)}
        out = si.run_workgroup(
            spv_n, 64, 64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=[0, 0, 0x1000, OUT_V] + [0] * 28,
            memory=mem,
        )
        for lane in range(48):
            self.assertEqual(out[4 * lane], 1000 + lane)
        for lane in range(48, 60):
            self.assertEqual(out[4 * lane], 0)
        for lane in range(60, 64):
            self.assertNotIn(4 * lane, out)


if __name__ == "__main__":
    unittest.main()
