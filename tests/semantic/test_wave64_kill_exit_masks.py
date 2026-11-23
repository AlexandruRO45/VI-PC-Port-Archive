"""SHADER-1155: synthetic kill exits through mask arithmetic, and exits that really change outputs."""
import unittest

from test_wave64_halves import (KILL, RUN_WQM, EXIT_PS, kill, meta_of, run_cut, out_param)


ACCEPTED = {
    "wqm_exec": ["s_wqm_b64 exec, s[20:21]"],
    "saveexec": ["s_and_saveexec_b64 s[30:31], s[20:21]"],
    "andn2_saveexec": ["s_andn2_saveexec_b64 s[30:31], s[20:21]"],
    "wqm_copy": ["s_wqm_b64 s[24:25], s[20:21]", "s_mov_b64 exec, s[24:25]"],
    "copy_and": ["s_mov_b64 s[24:25], s[20:21]", "s_and_b64 exec, exec, s[24:25]"],
    "copy_union": ["s_mov_b64 s[24:25], s[20:21]", "s_wqm_b64 s[26:27], s[20:21]",
                   "s_or_b64 s[24:25], s[24:25], s[26:27]", "s_mov_b64 exec, s[24:25]"],
}
REFUSED = {
    "other_mask": ["s_wqm_b64 exec, s[10:11]"],
    "saved_launch": ["s_and_saveexec_b64 s[30:31], s[20:21]", "s_mov_b64 exec, s[30:31]"],
    "copy_rewritten": ["s_mov_b64 s[24:25], s[20:21]", "s_mov_b64 s[24:25], s[10:11]",
                       "s_mov_b64 exec, s[24:25]"],
    "half_rewritten": ["s_mov_b64 s[24:25], s[20:21]", "s_mov_b32 s24, -1",
                       "s_mov_b64 exec, s[24:25]"],
    "union_launch": ["s_wqm_b64 s[24:25], s[20:21]", "s_or_b64 exec, s[24:25], s[10:11]"],
    "export_first": ["exp param0 v7, v7, v7, v7", "s_wqm_b64 exec, s[20:21]"],
}
VS_EXIT = ["exp param0 v7, v7, v7, v7", "s_endpgm"]
HEAD = ["v_mov_b32_e32 v7, 77"] + KILL


class KillExitMasks(unittest.TestCase):
    def test_accepted(self):
        for name, exit_head in ACCEPTED.items():
            with self.subTest(name=name):
                for stage, tail in (("vs", VS_EXIT), ("ps", EXIT_PS[1:])):
                    lines = kill(HEAD, RUN_WQM + ["v_add_nc_u32_e32 v6, 3, v0"], exit_head + tail, stage)
                    meta, asm = meta_of(stage + "_kill_exit_" + name, lines)
                    self.assertEqual(meta, ("A", [], []))
                    if stage == "vs":
                        whole, halves = run_cut(asm, 64), run_cut(asm, 32)
                        self.assertEqual(whole, halves)
                        self.assertEqual([w[0] for w in out_param(whole)], [0] * 40 + list(range(43, 67)))

    def test_refused_and_different(self):
        for name, exit_head in REFUSED.items():
            with self.subTest(name=name):
                lines = kill(HEAD, RUN_WQM + ["v_add_nc_u32_e32 v6, 3, v0"], exit_head + VS_EXIT, "vs")
                meta, asm = meta_of("vs_kill_exit_bad_" + name, lines)
                self.assertEqual(meta[0], "C")
                self.assertNotEqual(run_cut(asm, 64), run_cut(asm, 32))


if __name__ == "__main__":
    unittest.main()
