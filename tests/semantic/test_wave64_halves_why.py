"""SHADER-10: where a tier C reason comes from (`wave64_graphics_why`, `tools/wave64_halves.py --census --why`).

For a tier C wave64 VS/PS/GS, `wave64_halves.analyse(..., why=True)` gives one line per reason, in program order:

    reason|sink|producer@exec|source@exec|source of the source

with instruction families only (no register, no constant, no name): the family of the instruction where the reason was
found, of the instruction that first made the value depend on the partition, of the instructions that wrote the
producer's operands, and of the instructions that wrote the operands of those; `@launch` / `@narrowed` / `@both` says
if EXEC was the launch mask there. With `--wave64-why` the translator writes the lines to the index, and
`--census --why` counts them, so
that a person with the game's shaders can see which missing rule keeps the most shaders in tier C.

Each program here is a hand-written listing (no llvm-mc, GPU or game data). It is translated, and the lines in the
translator's metadata are compared with the lines the comment of the case derives. Other checks: the second analysis
run that makes the lines finds the same tier and reasons as the first one for every listing, a failure of that run
costs only the lines, tier A and B programs have none, and the census prints counts and no shader name.

    python -I tests/semantic/test_wave64_halves_why.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
import rdna2spirv  # noqa: E402
import wave64_halves  # noqa: E402
from test_wave64_halves import meta_of, ps, vs  # noqa: E402

LOAD = ["v_mov_b32_e32 v2, 8", "buffer_load_dword v4, v2, s[0:3], 0 offen"]
SKIP = ["v_cmp_gt_u32_e32 vcc, 32, v0", "s_and_saveexec_b64 s[12:13], vcc"]
# name -> (listing, tier C reasons in program order, the why lines)
CASES = {
    # the scalar is the first lane's copy of a loaded VGPR (the load's address came from a VALU result and the V#)
    "vs_why_rfl_load": (vs(LOAD + ["v_readfirstlane_b32 s20, v4", "v_add_nc_u32_e32 v6, s20, v0"]),
                        ["cross-lane|export|v_readfirstlane@launch|buffer_load@launch|input+valu"]),
    # the same scalar decides a scalar branch: the sink is the branch
    "vs_why_rfl_branch": (vs(LOAD + ["v_readfirstlane_b32 s20, v4", "s_cmp_eq_u32 s20, 3", "s_cbranch_scc1 1",
                                     "v_mov_b32_e32 v6, 1"]),
                          ["waterfall-scalar|scc_branch|v_readfirstlane@launch|buffer_load@launch|input+valu"]),
    # a VGPR written from a scalar and read back, both under a narrowed EXEC
    "vs_why_rfl_narrowed": (vs(SKIP + ["v_mov_b32_e32 v4, s4", "v_readfirstlane_b32 s20, v4",
                                       "s_or_b64 exec, exec, s[12:13]", "v_add_nc_u32_e32 v6, s20, v0"]),
                            ["cross-lane|export|v_readfirstlane@narrowed|valu@narrowed|input"]),
    # an if/else on a compare of a value that a VALU instruction wrote (the low bit of the vertex number, so the
    # compare is no uniform vote): the vote is the producer, the compare its source; the SGPR set on the two sides
    # is the second reason
    "vs_why_vote_valu": (vs(["v_and_b32_e32 v1, 1, v0", "v_cmp_lt_u32_e32 vcc, 0, v1", "s_cbranch_vccz 3",
                             "v_mov_b32_e32 v6, 7", "s_mov_b32 s20, 1", "s_branch 2", "v_mov_b32_e32 v6, 5",
                             "s_mov_b32 s20, 2", "v_add_nc_u32_e32 v7, s20, v0"]),
                         ["unsafe-vote-branch|vcc_branch|vcc_branch@launch|v_cmp@launch|valu",
                          "stale-scalar|export|valu@launch|input+salu@launch|const"]),
    # the same vote on an interpolated value
    "ps_why_vote_interp": (ps(["v_interp_p1_f32_e32 v2, v0, attr0.x", "v_cmp_lt_f32_e32 vcc, 0, v2",
                               "s_cbranch_vccz 1", "v_mov_b32_e32 v6, 7"]),
                           ["unsafe-vote-branch|vcc_branch|vcc_branch@launch|v_cmp@launch|v_interp"]),
    "vs_why_lane_id": (vs(["v_mbcnt_lo_u32_b32 v6, -1, 0", "v_mbcnt_hi_u32_b32 v6, -1, v6"]),
                       ["lane-id|export|v_mbcnt@launch|const|-"]),
    "vs_why_lane_count": (vs(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_bcnt1_i32_b64 s6, vcc", "v_mov_b32_e32 v6, s6"]),
                          ["lane-count|export|s_bcnt@launch|v_cmp@launch|input"]),
    "vs_why_readlane": (vs(["v_readlane_b32 s6, v0, 40", "v_mov_b32_e32 v6, s6"]),
                        ["cross-lane|export|v_readlane@launch|input|-"]),
    # the reason is about the instruction: sink and producer are the same
    "vs_why_writelane": (vs(["v_writelane_b32 v6, s4, 40"]), ["cross-lane|v_writelane|v_writelane@launch|input|-"]),
    "vs_why_mask_half": (vs(["s_mov_b32 exec_hi, 0"]), ["mask-half|exec_write|exec_write@launch|const|-"]),
    "vs_why_execz_else": (vs(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_and_saveexec_b64 s[12:13], vcc",
                              "s_cbranch_execz 2", "v_mov_b32_e32 v6, 7", "s_branch 1", "v_mov_b32_e32 v6, 5",
                              "s_or_b64 exec, exec, s[12:13]"]),
                          ["unsafe-execz-branch|exec_branch|exec_branch@narrowed|const|-"]),
    # an SGPR set before a skip region and again in it: the VALU read after the region is the producer, and its
    # source was written with EXEC the launch mask and with EXEC narrowed
    "vs_why_stale": (vs(["v_cmp_eq_u32_e32 vcc, 0, v0", "s_mov_b32 s5, 7", "s_and_saveexec_b64 s[12:13], vcc",
                         "s_cbranch_execz 1", "s_add_u32 s5, s5, 1", "s_or_b64 exec, exec, s[12:13]",
                         "v_mov_b32_e32 v6, s5"]),
                     ["stale-scalar|export|valu@launch|salu@both|const+salu"]),
    # the break mask of a loop, carried round the back edge: the producer is the s_or_b64 in the loop (EXEC narrowed),
    # written before the loop (launch) and in it (narrowed)
    "vs_why_loop_mask": (vs(["v_mov_b32_e32 v9, 0", "s_mov_b64 s[16:17], 0", "v_lshrrev_b32_e32 v10, 4, v0",
                             "v_add_nc_u32_e32 v9, 1, v9", "v_add_nc_u32_e32 v6, 2, v6",
                             "v_cmp_gt_u32_e32 vcc, v9, v10", "s_or_b64 s[16:17], vcc, s[16:17]",
                             "s_andn2_b64 exec, exec, s[16:17]", "s_cbranch_execnz -6",
                             "s_or_b64 exec, exec, s[16:17]"]),
                         ["tainted-exec|exec_write|salu@narrowed|salu+v_cmp@both|const+salu+v_cmp+valu"]),
    "vs_why_first_lane": (vs(["v_cmp_eq_u32_e32 vcc, 1, v0", "s_ff1_i32_b64 s6, vcc", "v_mov_b32_e32 v6, s6"]),
                          ["first-lane|export|s_ff@launch|v_cmp@launch|input"]),
}
NOT_C = {
    "vs_why_tier_a": (vs(["v_add_nc_u32_e32 v6, 1, v0"]), "A"),
    "vs_why_tier_b": (vs(["v_mov_b32_e32 v5, s4", "v_readfirstlane_b32 s20, v5", "v_add_nc_u32_e32 v6, s20, v0"]), "B"),
}


def translated(name, lines, why=True):
    """(w64g of the translator, the index entry it would write) for a hand-written listing, translated with
    --wave64-why (or without it)."""
    old = rdna2spirv.WAVE64_WHY
    rdna2spirv.WAVE64_WHY = why
    try:
        _, asm = meta_of(name, lines)
    finally:
        rdna2spirv.WAVE64_WHY = old
    g = rdna2spirv.translate.last_gen
    return g.w64g, rdna2spirv.shader_meta(g, asm)


class WhyLines(unittest.TestCase):
    def test_lines_of_each_case(self):
        for name, (lines, want) in CASES.items():
            w64g, meta = translated(name, lines)
            self.assertEqual(w64g["tier"], "C", name)
            self.assertEqual(w64g.get("why"), want, name)
            self.assertEqual(meta.get("wave64_graphics_why"), want, name)
            # one line per reason, and the reasons are the ones the tier was decided on
            self.assertEqual(sorted(x.split("|")[0] for x in want), w64g["reasons"], name)

    def test_lines_hold_families_only(self):
        families = {f for _, f in wave64_halves.WHY_FAMILY} | {"saveexec", "exec_write", "dpp", "other", "input",
                                                               "const", "-"}
        for name, (lines, _) in CASES.items():
            for line in translated(name, lines)[0]["why"]:
                reason, sink, producer, source, source2 = line.split("|")
                self.assertIn(reason, wave64_halves.REASONS, line)
                words = [sink, producer.split("@")[0]] + source.split("@")[0].split("+") + source2.split("+")
                self.assertTrue(set(words) <= families, line)
                self.assertIn(producer.split("@")[1], ("launch", "narrowed"), line)
                self.assertNotRegex(line, r"\d", line)          # no register number, constant or offset

    def test_tier_a_and_b_have_no_lines(self):
        for name, (lines, tier) in NOT_C.items():
            w64g, meta = translated(name, lines)
            self.assertEqual(w64g["tier"], tier, name)
            self.assertNotIn("why", w64g, name)
            self.assertNotIn("wave64_graphics_why", meta, name)

    def test_without_the_option_the_index_is_as_before(self):
        self.assertFalse(rdna2spirv.WAVE64_WHY)                 # the default: no second run, no new index key
        for name, (lines, _) in CASES.items():
            with_lines, without = translated(name, lines)[1], translated(name, lines, why=False)[1]
            self.assertNotIn("wave64_graphics_why", without, name)
            with_lines.pop("wave64_graphics_why")
            self.assertEqual(with_lines, without, name)

    def test_the_translator_has_the_option(self):
        import subprocess
        r = subprocess.run([sys.executable, "-I", os.path.join(support.TOOLS_DIR, "rdna2spirv.py"), "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--wave64-why", r.stdout)

    def test_family_names(self):
        def fam(op, args=(), mods=()):
            return wave64_halves.why_family({"op": op, "args": list(args), "mods": list(mods)})
        exec_ = {"k": "special", "t": "exec"}
        s4 = {"k": "sgpr", "r": 4, "n": 2}
        self.assertEqual(fam("buffer_load_format_xyzw"), "buffer_load")
        self.assertEqual(fam("s_buffer_load_dword"), "s_load")
        self.assertEqual(fam("image_sample_lz"), "image")
        self.assertEqual(fam("v_interp_p2_f32_e32"), "v_interp")
        self.assertEqual(fam("v_cmpx_gt_u32_e32"), "v_cmpx")
        self.assertEqual(fam("v_cmp_gt_u32_e64"), "v_cmp")
        self.assertEqual(fam("s_and_saveexec_b64", [s4, s4]), "saveexec")
        self.assertEqual(fam("s_andn2_b64", [exec_, exec_, s4]), "exec_write")
        self.assertEqual(fam("s_andn2_b64", [s4, exec_, s4]), "salu")
        self.assertEqual(fam("v_mov_b32_dpp", mods=["quad_perm:[0,1,2,3]"]), "dpp")
        self.assertEqual(fam("v_fma_f32_e64"), "valu")
        self.assertEqual(fam("buffer_atomic_add"), "atomic")
        self.assertEqual(fam("buffer_store_dword"), "store")
        self.assertEqual(fam("ds_bpermute_b32"), "permute")


class SecondRun(unittest.TestCase):
    """The lines come from a second run of the analysis with two more tables. It must not decide anything."""

    def recorded(self, name, lines):
        """The arguments the translator gave to analyse() for a listing."""
        seen, old = [], wave64_halves.analyse

        def spy(*a, **k):
            seen.append(a)
            self.assertEqual(k, {"why": False})         # the translator's default
            return old(*a, **k)
        wave64_halves.analyse = spy
        try:
            meta_of(name, lines)
        finally:
            wave64_halves.analyse = old
        self.assertEqual(len(seen), 1, name)
        return seen[0]

    def test_same_tier_reasons_and_sites_with_and_without_why(self):
        both = dict({k: v[0] for k, v in CASES.items()}, **{k: v[0] for k, v in NOT_C.items()})
        for name, lines in both.items():
            args = self.recorded(name, lines)
            plain, why = wave64_halves.analyse(*args), wave64_halves.analyse(*args, why=True)
            why.pop("why", None)
            self.assertEqual(plain, why, name)
            # the second run alone reports the same violations as the first, in the same order
            base = wave64_halves._fixpoint(wave64_halves.Analysis(*args))
            more = wave64_halves._fixpoint(wave64_halves.WhyAnalysis(*args))
            self.assertEqual(base["viol"], more["viol"], name)
            self.assertEqual(base["idioms"], more["idioms"], name)
            self.assertEqual(len(more.get("why", [])), len(more["viol"]), name)

    def test_a_failure_of_the_second_run_costs_only_the_lines(self):
        name, (lines, _) = "vs_why_lane_id", CASES["vs_why_lane_id"]
        old = wave64_halves.WhyAnalysis.step

        def boom(self, *a, **k):
            raise RuntimeError("boom")
        wave64_halves.WhyAnalysis.step = boom
        try:
            w64g, meta = translated(name, lines)
        finally:
            wave64_halves.WhyAnalysis.step = old
        self.assertEqual((w64g["tier"], w64g["reasons"]), ("C", ["lane-id"]))
        self.assertNotIn("why", w64g)
        self.assertNotIn("wave64_graphics_why", meta)


class Census(unittest.TestCase):
    def index(self):
        """An index made of the translated cases, under names that must not be printed."""
        index = {}
        for k, (name, (lines, _)) in enumerate(CASES.items()):
            index[f"secret_shader_{k}"] = translated(name, lines)[1]
        for k, (name, (lines, _)) in enumerate(NOT_C.items()):
            index[f"secret_clean_{k}"] = translated(name, lines)[1]
        index["secret_old"] = {"stage": "ps", "wave64": True, "wave64_graphics": "C",
                               "wave64_graphics_reasons": ["lane-id"]}           # translated before --why
        return index

    def test_counts(self):
        why = wave64_halves.census(self.index())["why"]
        self.assertEqual((why["tier_c"], why["with_why"]), (len(CASES) + 1, len(CASES)))
        self.assertEqual(sum(why["first"].values()), len(CASES))            # one first line per shader
        self.assertEqual(why["first"]["cross-lane|export|v_readfirstlane@launch|buffer_load@launch|input+valu"], 1)
        self.assertEqual(sum(why["by_reason"]["cross-lane"].values()), 4)   # four cases have that reason
        self.assertEqual(sum(why["by_reason"]["unsafe-vote-branch"].values()), 2)
        self.assertEqual(sum(why["by_reason"]["stale-scalar"].values()), 2)  # once as a second reason

    def run_cli(self, index, *flags):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "index.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(index, f)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = wave64_halves.main(["--census", path, *flags])
        return code, out.getvalue()

    def test_cli_prints_counts_and_no_name(self):
        code, out = self.run_cli(self.index(), "--why")
        self.assertEqual(code, 0)
        self.assertNotIn("secret", out)
        self.assertNotIn("vs_why", out)
        self.assertIn(f"why: {len(CASES)} of {len(CASES) + 1} tier C entries have wave64_graphics_why", out)
        self.assertIn("     1  unsafe-vote-branch|vcc_branch|vcc_branch@launch|v_cmp@launch|v_interp", out)
        self.assertIn("first reason by producer: ", out)
        self.assertIn("v_readfirstlane@launch x2", out)
        self.assertIn("unsafe-vote-branch (first site in each shader that has it, 2 shaders", out)
        # without --why the census prints what it printed before
        code, plain = self.run_cli(self.index())
        self.assertEqual(code, 0)
        self.assertNotIn("why", plain)
        self.assertTrue(out.startswith(plain))

    def test_top_limits_each_table(self):
        _, out = self.run_cli(self.index(), "--why", "--top", "2")
        head = out.split("first reason of a shader")[1].split("first reason by sink")[0]
        self.assertEqual(len([ln for ln in head.splitlines() if "|" in ln]), 2)

    def test_an_index_without_the_lines_says_so(self):
        code, out = self.run_cli({"a": {"stage": "ps", "wave64": True, "wave64_graphics": "C",
                                        "wave64_graphics_reasons": ["lane-id"]}}, "--why")
        self.assertEqual(code, 0)
        self.assertIn("why: 0 of 1 tier C entries have wave64_graphics_why (translate with rdna2spirv.py "
                      "--wave64-why to get it)", out)
        self.assertNotIn("first reason", out)


if __name__ == "__main__":
    unittest.main()
