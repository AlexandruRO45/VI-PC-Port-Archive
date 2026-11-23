"""SHADER-10: a vote on a compare that is uniform in the active lanes of a narrowed EXEC.

    python -I tests/semantic/test_wave64_uniform_vote_narrowed.py

tools/wave64_halves.py accepts s_cbranch_vccz/vccnz on a v_cmp whose sources are the same in every active lane, made
while EXEC is narrowed, when the two sides of the branch meet again and no side makes EXEC wider than it was at the
vote (Analysis.uniform_region). A half that has an active lane votes as the wave does. A half that has none has an
empty mask: it takes the no side, whatever the wave does, with all its lanes off, and is back on the path of the wave
at the join.

All listings are hand-written vertex shaders. Each is translated, and run on the strict wave interpreter
(spirv_wave_interp.py) as one wave of 64 lanes and as two waves of 32, with s4 = 9 (the compare 5 < s4 says yes) and
s4 = 1 (no). EXEC is narrowed to lanes 0..47 (the two halves have active lanes) and to lanes 0..31 (the second half
has none: the case that the rule is about).

  * Accepted shapes are tier A, export the same words both ways, and export the words that the test derives.
  * Near misses are tier C, and the test shows a lane whose words differ between the two runs: the rule is right to
    refuse them.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import support  # noqa: E402,F401  (sets the tools path)
from test_wave64_halves import meta_of, out_param, run_cut, size, vs  # noqa: E402

PUSH = [[0, 0, 0, 0, 9], [0, 0, 0, 0, 1]]      # s4 = 9: 5 < s4 is yes; s4 = 1: no
N48 = ["v_cmp_gt_u32_e32 vcc, 48, v0", "s_and_saveexec_b64 s[12:13], vcc"]     # lanes 0..47
N32 = ["v_cmp_gt_u32_e32 vcc, 32, v0", "s_and_saveexec_b64 s[12:13], vcc"]     # lanes 0..31: the second half is off
RESTORE = ["s_or_b64 exec, exec, s[12:13]"]
LAUNCH = ["s_mov_b64 exec, s[10:11]"]          # s[10:11] is the launch mask (the prologue of vs())
UCMP = ["v_mov_b32_e32 v1, s4", "v_cmp_lt_u32_e32 vcc, 5, v1"]                 # uniform in the active lanes
ADD7, ADD5 = ["v_add_nc_u32_e32 v6, 7, v6"], ["v_add_nc_u32_e32 v6, 5, v6"]


def ifelse(yes, no, tag=""):
    """s_cbranch_vccz if/else: `yes` runs when some lane voted yes, `no` when none did."""
    return [f"s_cbranch_vccz @else{tag}"] + yes + [f"s_branch @join{tag}", f"@else{tag}:"] + no + [f"@join{tag}:"]


def resolve(lines):
    """Replace "@name:" markers and "@name" branch targets by offsets in words, with the sizes of the fake encoding."""
    addr, labels, placed = 0, {}, []
    for text in lines:
        if text.startswith("@") and text.endswith(":"):
            assert text[1:-1] not in labels, text
            labels[text[1:-1]] = addr
            continue
        placed.append((addr, text))
        addr += size(text.split("@")[0] + "0" if "@" in text else text)
    out = []
    for at, text in placed:
        if "@" in text:
            head, label = text.split("@")
            text = f"{head}{(labels[label.strip()] - (at + 4)) // 4}"
        out.append(text)
    return out


class Base(unittest.TestCase):
    def run_all(self, name, body):
        """(tier, reasons, [(push, words of 64 lanes, words of 2 x 32 lanes)]) of one vertex listing."""
        lines = resolve(vs(body))
        (tier, reasons, _), asm = meta_of(name, lines)
        runs = [(push, out_param(run_cut(asm, 64, push)), out_param(run_cut(asm, 32, push))) for push in PUSH]
        return tier, reasons, runs

    def exact(self, name, body, want):
        """Tier A, the same words both ways, and the words want(s4, lane) -> (param0.x, param0.y)."""
        tier, reasons, runs = self.run_all(name, body)
        self.assertEqual((tier, reasons), ("A", []), name)
        for push, whole, halves in runs:
            self.assertEqual(whole, halves, (name, push))
            self.assertEqual([w[:2] for w in whole], [want(push[4], k) for k in range(64)], (name, push))

    def refused(self, name, body, reason, lane=32):
        """Tier C with `reason`, and the words of `lane` differ between the two runs for some s4."""
        tier, reasons, runs = self.run_all(name, body)
        self.assertEqual(tier, "C", name)
        self.assertIn(reason, reasons, name)
        self.assertTrue(any(whole[lane] != halves[lane] for _, whole, halves in runs), name)


class Accepted(Base):
    def test_lane_work_on_the_two_sides(self):
        # the census shape "run:valu, other:valu"
        for tag, narrow, on in (("48", N48, 48), ("32", N32, 32)):
            self.exact(f"vs_uvn_valu_{tag}", narrow + UCMP + ifelse(ADD7, ADD5) + RESTORE,
                       lambda s4, k, on=on: ((7 if s4 > 5 else 5) if k < on else 0, 0))

    def test_each_side_loads_a_scalar_and_uses_it_in_the_side(self):
        # the census shape "run:s_load, other:s_load"; the buffer of the interpreter reads as 0
        for tag, narrow, on in (("48", N48, 48), ("32", N32, 32)):
            body = narrow + UCMP + ifelse(
                ["s_buffer_load_dword s20, s[0:3], 0", "v_add_nc_u32_e32 v6, s20, v6"] + ADD7,
                ["s_buffer_load_dword s20, s[0:3], 4", "v_add_nc_u32_e32 v6, s20, v6"] + ADD5) + RESTORE
            self.exact(f"vs_uvn_sload_{tag}", body, lambda s4, k, on=on: ((7 if s4 > 5 else 5) if k < on else 0, 0))

    def test_skip_with_no_other_side(self):
        self.exact("vs_uvn_skip", N32 + UCMP + ["s_cbranch_vccz @join"] + ADD7 + ["@join:"] + RESTORE,
                   lambda s4, k: (7 if s4 > 5 and k < 32 else 0, 0))

    def test_llvm_form_and_with_exec(self):
        body = N32 + ["v_mov_b32_e32 v1, s4", "v_cmp_lt_u32_e64 s[20:21], 5, v1", "s_and_b64 vcc, exec, s[20:21]"] + \
            ifelse(ADD7, ADD5) + RESTORE
        self.exact("vs_uvn_and_exec", body, lambda s4, k: ((7 if s4 > 5 else 5) if k < 32 else 0, 0))

    def test_scalar_against_an_immediate(self):
        self.exact("vs_uvn_sgpr", N32 + ["v_cmp_lt_u32_e64 vcc, 5, s4"] + ifelse(ADD7, ADD5) + RESTORE,
                   lambda s4, k: ((7 if s4 > 5 else 5) if k < 32 else 0, 0))

    def test_a_side_with_an_if_else_of_its_own(self):
        yes = ["v_and_b32_e32 v4, 1, v0", "v_cmp_eq_u32_e32 vcc, 1, v4", "s_and_saveexec_b64 s[14:15], vcc",
               "s_xor_b64 s[14:15], exec, s[14:15]", "s_cbranch_execz @k", "v_add_nc_u32_e32 v6, 7, v6", "@k:",
               "s_or_saveexec_b64 s[16:17], s[14:15]", "s_xor_b64 exec, exec, s[16:17]", "s_cbranch_execz @m",
               "v_add_nc_u32_e32 v6, 3, v6", "@m:", "s_or_b64 exec, exec, s[16:17]"]
        self.exact("vs_uvn_inner_if_else", N32 + UCMP + ifelse(yes, ADD5) + RESTORE,
                   lambda s4, k: (((7 if k % 2 else 3) if s4 > 5 else 5) if k < 32 else 0, 0))

    def test_a_side_with_a_loop_of_its_own(self):
        yes = ["v_mov_b32_e32 v9, 0", "v_and_b32_e32 v8, 3, v0", "s_mov_b64 s[18:19], exec", "@loop:",
               "v_add_nc_u32_e32 v9, 1, v9", "v_add_nc_u32_e32 v6, 2, v6", "v_cmp_le_u32_e32 vcc, v9, v8",
               "s_and_b64 exec, exec, vcc", "s_cbranch_execnz @loop", "s_mov_b64 exec, s[18:19]"]
        self.exact("vs_uvn_inner_loop", N32 + UCMP + ifelse(yes, ADD5) + RESTORE,
                   lambda s4, k: ((2 * (k % 4 + 1) if s4 > 5 else 5) if k < 32 else 0, 0))

    def test_a_second_vote_in_a_side_and_one_after_the_join(self):
        inner = ["v_cmp_gt_u32_e32 vcc, 3, v1"] + ifelse(["v_add_nc_u32_e32 v6, 1, v6"], ["v_add_nc_u32_e32 v6, 2, v6"],
                                                         "2")
        self.exact("vs_uvn_nested", N32 + UCMP + ifelse(inner, ADD5) + RESTORE,
                   lambda s4, k: ((2 if s4 > 5 else 5) if k < 32 else 0, 0))
        after = ["v_cmp_gt_u32_e32 vcc, 3, v1"] + ifelse(["v_add_nc_u32_e32 v7, 1, v7"], ["v_add_nc_u32_e32 v7, 2, v7"],
                                                         "2")
        self.exact("vs_uvn_second", N32 + UCMP + ifelse(ADD7, ADD5) + after + RESTORE,
                   lambda s4, k: ((7, 2) if s4 > 5 else (5, 1)) if k < 32 else (0, 0))

    def test_the_vote_mask_read_as_a_mask_after_the_restore(self):
        # the mask is inside the EXEC of the compare: 0 in the lanes of the half that is off, in the wave too
        body = N32 + UCMP + ifelse(ADD7, ADD5) + LAUNCH + ["v_mov_b32_e32 v10, 1", "v_cndmask_b32_e32 v7, v7, v10, vcc"]
        self.exact("vs_uvn_mask_after", body,
                   lambda s4, k: ((7, 1) if s4 > 5 else (5, 0)) if k < 32 else (0, 0))


class NearMisses(Base):
    def test_scalar_of_a_side_read_after_exec_is_restored(self):
        body = N32 + UCMP + ifelse(["s_mov_b32 s20, 7"], ["s_mov_b32 s20, 5"]) + RESTORE + \
            ["v_add_nc_u32_e32 v7, s20, v0"]
        self.refused("vs_uvn_c_scalar", body, "stale-scalar")

    def test_scc_of_a_side_read_after_the_join(self):
        body = N32 + UCMP + ifelse(["s_cmp_eq_u32 s4, s4"] + ADD7, ["s_cmp_lg_u32 s4, s4"] + ADD5) + LAUNCH + \
            ["s_cbranch_scc0 @q", "v_add_nc_u32_e32 v7, 3, v7", "@q:"]
        self.refused("vs_uvn_c_scc", body, "stale-scalar")

    def test_a_side_makes_exec_wider(self):
        wider = {
            "launch": ["s_mov_b64 exec, s[10:11]"],
            "or_outer": ["s_or_b64 exec, exec, s[12:13]"],
            "xor_outer": ["s_xor_b64 exec, exec, s[12:13]"],
            "or_saveexec": ["s_or_saveexec_b64 s[14:15], s[12:13]"],
            "andn2_saveexec": ["s_andn2_saveexec_b64 s[14:15], s[12:13]"],
            "andn2_wrexec": ["s_andn2_wrexec_b64 s[14:15], s[12:13]"],
            "andn2_exec_last": ["s_andn2_b64 exec, s[12:13], exec"],
            "not": ["s_not_b64 exec, exec"],
            "orn2": ["s_orn2_b64 exec, exec, exec"],
            "all": ["s_mov_b64 exec, -1"],
            "wqm": ["s_wqm_b64 exec, s[12:13]"],
            "nand_saveexec": ["s_mov_b64 s[18:19], exec", "s_nand_saveexec_b64 s[14:15], s[18:19]"],
            "orn2_saveexec": ["s_mov_b64 s[18:19], exec", "s_orn2_saveexec_b64 s[14:15], s[18:19]"],
            "copy_rewritten": ["s_mov_b64 s[18:19], exec", "s_mov_b64 s[18:19], s[10:11]", "s_mov_b64 exec, s[18:19]"],
            "copy_half_rewritten": ["s_mov_b64 s[18:19], exec", "s_mov_b32 s19, -1", "s_mov_b64 exec, s[18:19]"],
        }
        for tag, w in wider.items():
            self.refused(f"vs_uvn_c_wide_yes_{tag}", N32 + UCMP + ifelse(w + ADD7, ADD5) + LAUNCH,
                         "unsafe-vote-branch")
            self.refused(f"vs_uvn_c_wide_no_{tag}", N32 + UCMP + ifelse(ADD7, w + ADD5) + LAUNCH,
                         "unsafe-vote-branch")

    def test_the_other_side_ends_the_program(self):
        end = ["v_mov_b32_e32 v5, 0", "exp pos0 v5, v5, v5, v5 done", "exp param0 v6, v7, v5, v5", "s_endpgm"]
        lines = ["s_mov_b64 s[10:11], exec", "v_mov_b32_e32 v6, 0", "v_mov_b32_e32 v7, 0"] + N32 + UCMP + \
            ["s_cbranch_vccz @else"] + ADD7 + RESTORE + ["v_add_nc_u32_e32 v7, 1, v7"] + end + \
            ["@else:"] + ADD5 + RESTORE + end
        (tier, reasons, _), asm = meta_of("vs_uvn_c_exit", resolve(lines))
        self.assertEqual(tier, "C")
        self.assertIn("unsafe-vote-branch", reasons)
        whole, halves = out_param(run_cut(asm, 64, PUSH[0])), out_param(run_cut(asm, 32, PUSH[0]))
        self.assertNotEqual(whole[32], halves[32])

    def test_the_branch_comes_after_exec_is_restored(self):
        self.refused("vs_uvn_c_branch_late", N32 + UCMP + RESTORE + ifelse(ADD7, ADD5), "unsafe-vote-branch")
        self.refused("vs_uvn_c_and_exec_late", N32 + ["v_mov_b32_e32 v1, s4", "v_cmp_lt_u32_e64 s[20:21], 5, v1"]
                     + RESTORE + ["s_and_b64 vcc, exec, s[20:21]"] + ifelse(ADD7, ADD5), "unsafe-vote-branch")

    def test_the_same_mask_voted_again_after_the_restore(self):
        again = ifelse(["v_add_nc_u32_e32 v7, 1, v7"], ["v_add_nc_u32_e32 v7, 2, v7"], "2")
        self.refused("vs_uvn_c_again", N32 + UCMP + ifelse(ADD7, ADD5) + LAUNCH + again, "unsafe-vote-branch")

    def test_the_vgpr_was_made_uniform_under_another_exec(self):
        # v1 is uniform in lanes 0..31 only; the compare runs under lanes 16..63
        body = N32 + ["v_mov_b32_e32 v1, s4"] + RESTORE + \
            ["v_cmp_le_u32_e32 vcc, 16, v0", "s_and_saveexec_b64 s[14:15], vcc", "v_cmp_lt_u32_e32 vcc, 5, v1"] + \
            ifelse(ADD7, ADD5) + LAUNCH
        self.refused("vs_uvn_c_other_exec", body, "unsafe-vote-branch")

    def test_a_mask_register_of_a_side_becomes_exec_after_the_join(self):
        body = ["s_mov_b64 s[14:15], s[10:11]"] + N32 + UCMP + ifelse(["s_mov_b64 s[14:15], exec"] + ADD7, ADD5) + \
            ["s_mov_b64 exec, s[14:15]", "v_add_nc_u32_e32 v7, 1, v7"] + LAUNCH
        tier, _, runs = self.run_all("vs_uvn_c_mask_reg", body)
        self.assertEqual(tier, "C")
        self.assertTrue(any(whole[32] != halves[32] for _, whole, halves in runs))

    def test_a_varying_compare_is_no_uniform_vote(self):
        body = N48 + ["v_and_b32_e32 v1, 32, v0", "v_cmp_lt_u32_e32 vcc, 0, v1"] + ifelse(ADD7, ADD5) + RESTORE
        self.refused("vs_uvn_c_varying", body, "unsafe-vote-branch", lane=0)


if __name__ == "__main__":
    unittest.main()
