"""SHADER-10: why the tier analysis refused a vote (`wave64_graphics_vote_why`, `wave64_halves.py --census --why`).

`tools/wave64_halves.py` accepts a wave-wide vote on a lane mask only when the side that runs on a yes vote starts by
narrowing EXEC to that mask and the other side is the join. Each refused vote is one `unsafe-vote-branch` (or, for a
vote on EXEC itself, `unsafe-execz-branch`), and the reason name does not say which condition failed. With
`--wave64-why` the translator also writes, for a tier C shader, the distinct lines

    reason|condition|run:family|other:family

of its refused votes, in program order: the condition (`no-narrowing`, `other-side-joins`, `other-side-exits`, the
two together, `one-way`), and the instruction family at the start of the side that runs on a yes vote and of the other
side (`join` when that side has no code of its own). `--census --why` counts them, so that a person with the game's
shaders can see which missing vote rule keeps the most shaders in tier C. No register, constant or name is in a line.

Each program here is a hand-written vertex listing (no llvm-mc, GPU or game data). It is translated, and the lines in
the translator's metadata are compared with the lines the comment of the case derives. The tier and the reasons of
each listing are checked too: the lines change no decision.

    python -I tests/semantic/test_wave64_halves_vote_why.py
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
from test_wave64_halves import meta_of, vs  # noqa: E402

CMP = "v_cmp_eq_u32_e32 vcc, 1, v0"
SKIP = ["s_and_saveexec_b64 s[16:17], vcc", "v_mov_b32_e32 v6, 7", "s_or_b64 exec, exec, s[16:17]"]
# name -> (listing, reasons, the refused-vote lines)
CASES = {
    # the run side starts with a VALU instruction and never narrows; the other side is the join
    "vs_vw_no_narrowing": (vs([CMP, "s_cbranch_vccz 1", "v_add_nc_u32_e32 v6, 9, v6"]), ["unsafe-vote-branch"],
                           ["unsafe-vote-branch|no-narrowing|run:valu|other:join"]),
    # the run side narrows, but a scalar load comes first
    "vs_vw_load_first": (vs([CMP, "s_cbranch_vccz 4", "s_buffer_load_dword s20, s[0:3], 0",
                             "s_and_saveexec_b64 s[16:17], vcc", "v_add_nc_u32_e32 v6, s20, v6",
                             "s_or_b64 exec, exec, s[16:17]"]), ["unsafe-vote-branch"],
                         ["unsafe-vote-branch|no-narrowing|run:s_load|other:join"]),
    # an SCC vote on a mask pair, no narrowing
    "vs_vw_scc_vote": (vs(["v_cmp_gt_u32_e32 vcc, 20, v0", "s_and_b64 s[20:21], exec, vcc", "s_cbranch_scc0 1",
                           "v_add_nc_u32_e32 v6, 3, v0"]), ["unsafe-vote-branch"],
                       ["unsafe-vote-branch|no-narrowing|run:valu|other:join"]),
    # an if/else: the run side narrows, the other side has a VALU instruction and then meets the run side
    "vs_vw_if_else": (vs([CMP, "s_cbranch_vccnz 2", "v_mov_b32_e32 v6, 5", "s_branch 3"] + SKIP),
                      ["unsafe-vote-branch"], ["unsafe-vote-branch|other-side-joins|run:saveexec|other:valu"]),
    # an if/else whose other side narrows EXEC to the lanes that voted no
    "vs_vw_two_sided": (vs([CMP, "s_cbranch_vccz 4", "s_and_saveexec_b64 s[14:15], vcc", "v_mov_b32_e32 v6, 3",
                            "s_mov_b64 exec, s[10:11]", "s_branch 3", "s_andn2_b64 exec, exec, vcc",
                            "v_mov_b32_e32 v6, 9", "s_mov_b64 exec, s[10:11]"]), ["unsafe-vote-branch"],
                        ["unsafe-vote-branch|other-side-joins|run:saveexec|other:exec_write"]),
    # an if/else with no narrowing on any side
    "vs_vw_if_else_plain": (vs([CMP, "s_cbranch_vccz 2", "v_mov_b32_e32 v6, 7", "s_branch 1", "v_mov_b32_e32 v6, 5"]),
                            ["unsafe-vote-branch"],
                            ["unsafe-vote-branch|no-narrowing+other-side-joins|run:valu|other:valu"]),
    # the other side exports and ends the program: it never meets the run side
    "vs_vw_exit": (vs([CMP, "s_cbranch_vccnz 6", "v_mov_b32_e32 v5, 0", "exp pos0 v5, v5, v5, v5 done",
                       "exp param0 v6, v7, v5, v5", "s_endpgm"] + SKIP), ["unsafe-vote-branch"],
                   ["unsafe-vote-branch|other-side-exits|run:saveexec|other:valu"]),
    # a vote on EXEC: the side that runs with no lane on writes an SGPR
    "vs_vw_execz": (vs([CMP, "s_and_saveexec_b64 s[12:13], vcc", "s_cbranch_execz 2", "v_mov_b32_e32 v6, 7",
                        "s_branch 1", "s_mov_b32 s5, 9", "s_or_b64 exec, exec, s[12:13]"]),
                    ["unsafe-execz-branch"], ["unsafe-execz-branch|other-side-joins|run:valu|other:salu"]),
    # two refused votes of different kinds, in program order; a third one of the first kind adds no line
    "vs_vw_two_votes": (vs([CMP, "s_cbranch_vccz 1", "v_add_nc_u32_e32 v6, 9, v6", CMP, "s_cbranch_vccz 2",
                            "v_mov_b32_e32 v7, 7", "s_branch 1", "v_mov_b32_e32 v7, 5", CMP, "s_cbranch_vccz 1",
                            "v_add_nc_u32_e32 v6, 1, v6"]), ["unsafe-vote-branch"],
                        ["unsafe-vote-branch|no-narrowing|run:valu|other:join",
                         "unsafe-vote-branch|no-narrowing+other-side-joins|run:valu|other:valu"]),
}
# no refused vote: an accepted skip region (tier A), and a tier C program whose reason is no vote
NO_LINES = {
    "vs_vw_accepted": (vs([CMP, "s_cbranch_vccz 3"] + SKIP), "A"),
    "vs_vw_lane_id": (vs(["v_mbcnt_lo_u32_b32 v6, -1, 0", "v_mbcnt_hi_u32_b32 v6, -1, v6"]), "C"),
}


def translated(name, lines, why=True):
    """(w64g of the translator, the index entry it would write) for a listing translated with --wave64-why or not."""
    old = rdna2spirv.WAVE64_WHY
    rdna2spirv.WAVE64_WHY = why
    try:
        _, asm = meta_of(name, lines)
    finally:
        rdna2spirv.WAVE64_WHY = old
    g = rdna2spirv.translate.last_gen
    return g.w64g, rdna2spirv.shader_meta(g, asm)


class VoteLines(unittest.TestCase):
    def test_lines_of_each_case(self):
        for name, (lines, reasons, want) in CASES.items():
            w64g, meta = translated(name, lines)
            self.assertEqual((w64g["tier"], [r for r in w64g["reasons"] if r.startswith("unsafe-")]),
                             ("C", reasons), name)
            self.assertEqual(w64g.get("vote_why"), want, name)
            self.assertEqual(meta.get("wave64_graphics_vote_why"), want, name)

    def test_lines_hold_families_and_conditions_only(self):
        families = {f for _, f in wave64_halves.WHY_FAMILY} | {"saveexec", "exec_write", "dpp", "other", "join",
                                                               "empty", "none"}
        conditions = {"no-narrowing", "other-side-joins", "other-side-exits", "no-narrowing+other-side-joins",
                      "no-narrowing+other-side-exits", "one-way"}
        for name, (lines, _, _) in CASES.items():
            for line in translated(name, lines)[0]["vote_why"]:
                reason, cond, run, other = line.split("|")
                self.assertIn(reason, ("unsafe-vote-branch", "unsafe-execz-branch"), line)
                self.assertIn(cond, conditions, line)
                self.assertTrue(run.startswith("run:") and run[4:] in families, line)
                self.assertTrue(other.startswith("other:") and other[6:] in families, line)
                self.assertNotRegex(line.replace("s_load", ""), r"\d", line)       # no register, constant or offset

    def test_no_refused_vote_no_lines(self):
        for name, (lines, tier) in NO_LINES.items():
            w64g, meta = translated(name, lines)
            self.assertEqual(w64g["tier"], tier, name)
            self.assertNotIn("vote_why", w64g, name)
            self.assertNotIn("wave64_graphics_vote_why", meta, name)

    def test_the_lines_change_no_decision_and_need_the_option(self):
        for name, (lines, _, _) in CASES.items():
            with_lines, without = translated(name, lines), translated(name, lines, why=False)
            self.assertNotIn("wave64_graphics_vote_why", without[1], name)
            self.assertEqual((with_lines[0]["tier"], with_lines[0]["reasons"], with_lines[0]["idioms"]),
                             (without[0]["tier"], without[0]["reasons"], without[0]["idioms"]), name)
            entry = dict(with_lines[1])
            entry.pop("wave64_graphics_vote_why")
            entry.pop("wave64_graphics_why")
            self.assertEqual(entry, without[1], name)

    def test_at_most_eight_lines(self):
        self.assertEqual(wave64_halves.VOTE_WHY_MAX, 8)
        for name, (lines, _, _) in CASES.items():
            self.assertLessEqual(len(translated(name, lines)[0]["vote_why"]), 8, name)


class Census(unittest.TestCase):
    def index(self):
        index = {}
        for k, (name, (lines, _, _)) in enumerate(CASES.items()):
            index[f"secret_shader_{k}"] = translated(name, lines)[1]
        for k, (name, (lines, _)) in enumerate(NO_LINES.items()):
            index[f"secret_clean_{k}"] = translated(name, lines)[1]
        return index

    def test_counts(self):
        why = wave64_halves.census(self.index())["why"]
        self.assertEqual((why["tier_c"], why["with_votes"]), (len(CASES) + 1, len(CASES)))
        self.assertEqual(sum(why["vote_first"].values()), len(CASES))           # one first line per shader
        self.assertEqual(why["vote_first"]["unsafe-vote-branch|no-narrowing|run:valu|other:join"], 3)
        # the last case has two lines: its second one counts in "any" only
        self.assertEqual(why["vote_first"]["unsafe-vote-branch|no-narrowing+other-side-joins|run:valu|other:valu"], 1)
        self.assertEqual(why["vote_any"]["unsafe-vote-branch|no-narrowing+other-side-joins|run:valu|other:valu"], 2)
        self.assertEqual(why["vote_condition"], {
            "unsafe-vote-branch|no-narrowing": 4, "unsafe-vote-branch|other-side-joins": 2,
            "unsafe-vote-branch|no-narrowing+other-side-joins": 2, "unsafe-vote-branch|other-side-exits": 1,
            "unsafe-execz-branch|other-side-joins": 1})

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
        self.assertNotIn("vs_vw", out)
        self.assertIn(f"refused votes: {len(CASES)} of {len(CASES) + 1} tier C entries have wave64_graphics_vote_why",
                      out)
        self.assertIn("unsafe-vote-branch|no-narrowing x4", out)
        self.assertIn("     3  unsafe-vote-branch|no-narrowing|run:valu|other:join", out)
        self.assertIn("     1  unsafe-vote-branch|other-side-exits|run:saveexec|other:valu", out)
        # without --why the census prints what it printed before
        code, plain = self.run_cli(self.index())
        self.assertEqual(code, 0)
        self.assertNotIn("refused", plain)
        self.assertTrue(out.startswith(plain))

    def test_an_index_without_the_lines_says_so(self):
        code, out = self.run_cli({"a": {"stage": "ps", "wave64": True, "wave64_graphics": "C",
                                        "wave64_graphics_reasons": ["unsafe-vote-branch"],
                                        "wave64_graphics_why": ["unsafe-vote-branch|vcc_branch|vcc_branch@launch|"
                                                                "v_cmp@launch|valu"]}}, "--why")
        self.assertEqual(code, 0)
        self.assertIn("refused votes: 0 of 1 tier C entries have wave64_graphics_vote_why", out)
        self.assertNotIn("first refused vote", out)


if __name__ == "__main__":
    unittest.main()
