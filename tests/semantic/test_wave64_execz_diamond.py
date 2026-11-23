"""EXECZ diamonds: hand-written integer shaders, one wave64 against two wave32s. No game data."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_wave64_halves import LANE_ID, meta_of, out_param, run_cut, size, vs


def resolve(lines):
    """Resolve synthetic labels using the same fake instruction sizes as the translator tests."""
    labels, pc = {}, 0
    for line in lines:
        if line.endswith(":"):
            labels[line[:-1]] = pc
        else:
            pc += size(line)
    result, pc = [], 0
    for line in lines:
        if line.endswith(":"):
            continue
        length = size(line)
        if " @" in line:
            op, label = line.split(" @")
            line = f"{op} {(labels[label] - pc - length) // 4}"
        result.append(line)
        pc += length
    return result


def diamond(yes, no, tail=(), vote="s_cbranch_execz", limit=1):
    if vote.endswith("nz"):
        yes, no = no, yes
    return vs(resolve([
        "v_mov_b32_e32 v6, 77", "v_mov_b32_e32 v7, 88",
        f"v_cmp_lt_u32_e32 vcc, v0, {limit}",
        "s_and_saveexec_b64 s[12:13], vcc", f"{vote} @other",
        *yes, "s_branch @join", "other:", *no, "join:",
        "s_mov_b64 exec, s[10:11]", *tail]))


class ExecDiamond(unittest.TestCase):
    def check_case(self, lines, tier, differs=False, different_lanes=32):
        (actual, reasons, _), asm = meta_of("vs_execz_diamond", lines)
        self.assertEqual(actual, tier, reasons)
        whole, halves = out_param(run_cut(asm, 64)), out_param(run_cut(asm, 32))
        if differs:
            self.assertNotEqual(whole, halves)
            self.assertEqual(sum(a != b for a, b in zip(whole, halves)), different_lanes)
        else:
            self.assertEqual(whole, halves)
        return whole

    def test_accepted_twelve_listings(self):
        for vote in ("s_cbranch_execz", "s_cbranch_execnz"):
            for limit in (1, 32, 48):
                for scalar in (False, True):
                    with self.subTest(vote=vote, limit=limit, scalar=scalar):
                        yes = ["v_add_nc_u32_e32 v6, 3, v0"]
                        no = ["v_add_nc_u32_e32 v7, 9, v0"]
                        if scalar:
                            yes = ["s_mov_b32 s5, 3", "s_add_u32 s5, s5, 2",
                                   "v_add_nc_u32_e32 v6, s5, v0"]
                            no = ["s_mov_b32 s5, 9", "s_mul_i32 s5, s5, 2",
                                  "v_add_nc_u32_e32 v7, s5, v0"]
                        words = self.check_case(diamond(yes, no, vote=vote, limit=limit), "A")
                        self.assertEqual([w[:2] for w in words],
                                         [(i + (5 if scalar else 3) if i < limit else 77, 88)
                                          for i in range(64)])

    def test_twenty_near_misses_really_differ(self):
        for vote in ("s_cbranch_execz", "s_cbranch_execnz"):
            cases = [
                (["s_mov_b32 s5, 3"], ["s_mov_b32 s5, 9"], ["v_mov_b32_e32 v6, s5"]),
                (["s_add_u32 s5, 0, 0"], ["s_add_u32 s5, -1, 1"],
                 ["s_cselect_b32 s6, 3, 9", "v_mov_b32_e32 v6, s6"]),
                (["v_mov_b32_e32 v6, 3"], ["s_endpgm"], []),
                (["s_mov_b32 s14, 3"], ["s_mov_b32 s14, 9"], ["v_mov_b32_e32 v7, s14"]),
            ]
            for restore in ("s_mov_b64 exec, s[10:11]", "s_or_saveexec_b64 s[14:15], s[10:11]",
                            "s_andn2_saveexec_b64 s[14:15], s[10:11]"):
                cases.extend([
                    ([restore, "v_mov_b32_e32 v6, 3"], ["v_mov_b32_e32 v6, 9"], []),
                    (["v_mov_b32_e32 v6, 3"], [restore, "v_mov_b32_e32 v6, 9"], []),
                ])
            for i, (yes, no, tail) in enumerate(cases):
                with self.subTest(vote=vote, case=i):
                    self.check_case(diamond(yes, no, tail, vote=vote), "C", differs=True)

    def test_six_preexisting_taint_near_misses(self):
        # A narrowed overwrite must not hide the lane id still held by inactive lanes (#4619).
        for vote in ("s_cbranch_execz", "s_cbranch_execnz"):
            initializers = [
                [*LANE_ID, "v_mov_b32_e32 v6, v10"],
                # Wave-uniform is not enough: a lane count is uniform but partition-dependent.
                ["s_bcnt1_i32_b64 s8, exec", "v_mov_b32_e32 v6, s8"],
                # Current taint alone is not enough: an earlier narrowed write has already hidden it.
                [*LANE_ID, "v_mov_b32_e32 v6, v10", "v_cmp_eq_u32_e32 vcc, 1, v0",
                 "s_and_saveexec_b64 s[20:21], vcc", "v_mov_b32_e32 v6, 0",
                 "s_mov_b64 exec, s[10:11]"],
            ]
            for i, initial in enumerate(initializers):
                with self.subTest(vote=vote, initial=i):
                    lines = diamond(["v_mov_b32_e32 v6, 0"], ["v_mov_b32_e32 v6, 0"], vote=vote)
                    before_compare = next(i for i, line in enumerate(lines) if line.startswith("v_cmp_lt"))
                    lines[before_compare:before_compare] = initial
                    # A full-wave count differs in 63 inactive lanes, not only in the upper half.
                    self.check_case(lines, "C", differs=True, different_lanes=63 if i == 1 else 32)


if __name__ == "__main__":
    unittest.main()
