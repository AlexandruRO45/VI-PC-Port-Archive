"""tools/wave_diff.py: its reference for the cross-lane instructions, its canaries and a short seeded run.

    python -I tests/test_wave_diff.py

The known-answer cases derive each value by hand from the RDNA2 rule written next to the instruction in
wave_diff.WaveRef, with lane values chosen so that a wrong lane, a wrong half or an ignored EXEC bit gives a different
number. The seeded runs execute the repository's translator (tools/rdna2spirv.py) on tests/semantic/spirv_mini_interp.py
across a whole wave. Standard library only: no llvm-mc, SPIR-V tools, GPU or game data.
"""
import contextlib
import io
import os
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import cf_ref  # noqa: E402
import wave_diff  # noqa: E402
from shader_ir import parse_file  # noqa: E402

LANE64 = ["v_mbcnt_lo_u32_b32 v1, -1, 0", "v_mbcnt_hi_u32_b32 v1, -1, v1"]
LANE32 = ["v_mbcnt_lo_u32_b32 v1, -1, 0"]


def machine(body, wave=64):
    """The reference after `body` (with the lane in v1 first), for direct register reads."""
    lines = (LANE64 if wave == 64 else LANE32) + body
    with tempfile.TemporaryDirectory(prefix="wave-ref-") as tmp:
        path = os.path.join(tmp, "t.s")
        with open(path, "w", encoding="utf-8") as f:
            f.write("; t  code 0x0..0x1000\n" + "".join(f"\t{x}\n" for x in lines) + "\ts_endpgm\n")
        m = wave_diff.WaveRef(parse_file(path), wave)
        m.run()
    return m


def vgpr(m, r):
    return [m._vgpr(r, lane) for lane in range(m.wave)]


class Reference(unittest.TestCase):
    def test_readlane_ignores_exec_and_takes_the_lane_mod_the_wave_size(self):
        m = machine(["v_mad_u32_u24 v2, v1, 3, 1", "s_mov_b64 exec, 1", "s_mov_b32 s6, 105",
                     "v_readlane_b32 s7, v2, s6", "v_readlane_b32 s8, v2, 63"])
        self.assertEqual((m.sgpr[7], m.sgpr[8]), (3 * 41 + 1, 3 * 63 + 1))     # 105 & 63 = 41, a lane that is off
        m = machine(["v_mad_u32_u24 v2, v1, 3, 1", "v_readlane_b32 s7, v2, 40"], wave=32)
        self.assertEqual(m.sgpr[7], 3 * 8 + 1)                                 # 40 & 31 = 8

    def test_readfirstlane_takes_the_lowest_active_lane_or_lane_0(self):
        m = machine(["v_mad_u32_u24 v2, v1, 5, 2", "s_mov_b64 exec, -8", "v_readfirstlane_b32 s6, v2",
                     "s_mov_b64 exec, 0", "v_readfirstlane_b32 s7, v2"])
        self.assertEqual((m.sgpr[6], m.sgpr[7]), (5 * 3 + 2, 2))

    def test_writelane_ignores_exec_and_keeps_the_other_lanes(self):
        m = machine(["v_mov_b32 v2, 7", "s_mov_b64 exec, 0", "s_mov_b32 s6, 99", "v_writelane_b32 v2, s6, 70"])
        want = [7] * 64
        want[70 & 63] = 99
        self.assertEqual(vgpr(m, 2), want)

    def test_bpermute_stays_in_its_half_and_reads_zero_from_an_off_lane(self):
        m = machine(["v_mad_u32_u24 v2, v1, 10, 1", "v_sub_nc_u32 v3, 0, v1", "v_lshlrev_b32 v3, 2, v3",
                     "v_mov_b32 v4, 555", "s_mov_b64 exec, -2", "ds_bpermute_b32 v4, v3, v2 offset:4"])
        out = vgpr(m, 4)
        # lane l reads lane (1 - l) mod 32 of its half: lane 1 reads lane 0 (off) -> 0; lane 0 is off -> kept.
        self.assertEqual([out[0], out[1], out[2], out[33], out[34]], [555, 0, 10 * 31 + 1, 10 * 32 + 1, 10 * 63 + 1])

    def test_bpermute_reads_every_source_before_writing(self):
        m = machine(["v_lshlrev_b32 v2, 2, v1", "v_xor_b32 v2, 4, v2", "ds_bpermute_b32 v2, v2, v2"])
        self.assertEqual(vgpr(m, 2)[:4], [0, 4, 8, 12])                        # lane l reads (l ^ 1)'s word 4 * l

    def test_permlane16_and_x16(self):
        m = machine(["v_mad_u32_u24 v2, v1, 2, 1", "s_mov_b32 s6, 0x76543210", "s_mov_b32 s7, 0xfedcba98",
                     "s_mov_b32 s8, 0x01234567", "s_mov_b32 s9, 0x89abcdef",
                     "v_permlane16_b32 v3, v2, s6, s7", "v_permlanex16_b32 v4, v2, s8, s9"])
        self.assertEqual(vgpr(m, 3), vgpr(m, 2))                               # identity selectors
        # x16 reads the other row of the 32-lane half. Lane 0: nibble 0 of s8 = 7 of row 16..31 -> 23; lane 40:
        # nibble 0 of s9 = 0xf of row 48..63 -> 63; lane 21: nibble 5 of s8 = 2 of row 0..15 -> 2.
        x = vgpr(m, 4)
        self.assertEqual([x[0], x[40], x[21]], [2 * 23 + 1, 2 * 63 + 1, 2 * 2 + 1])

    def test_permlane_from_a_lane_that_is_off_leaves_the_destination(self):
        m = machine(["v_mov_b32 v2, 9", "s_mov_b32 s6, 0", "s_mov_b64 exec, -2", "v_permlane16_b32 v2, v1, s6, s6"])
        self.assertEqual(vgpr(m, 2)[:3], [9, 9, 9])                            # every lane of row 0 reads lane 0
        self.assertEqual(vgpr(m, 2)[17], 16)                                   # row 16 reads lane 16, which is on

    def test_bit_counts_first_bits_and_not(self):
        m = machine(["s_mov_b64 s[6:7], 0", "s_bcnt1_i32_b64 s10, -16", "s_bcnt0_i32_b64 s11, -16",
                     "s_ff1_i32_b64 s12, -16", "s_ff0_i32_b64 s13, -1", "s_ff1_i32_b64 s14, s[6:7]",
                     "s_not_b64 s[8:9], s[6:7]", "s_bcnt1_i32_b64 s15, s[8:9]"])
        self.assertEqual([m.sgpr[i] for i in range(10, 16)], [60, 4, 4, 0xFFFFFFFF, 0xFFFFFFFF, 64])
        m = machine(["s_bcnt1_i32_b32 s10, -1", "s_ff0_i32_b32 s11, -1", "v_cmp_gt_u32_e64 s6, 5, v1",
                     "s_bcnt1_i32_b32 s12, s6"], wave=32)
        self.assertEqual([m.sgpr[10], m.sgpr[11], m.sgpr[12]], [32, 0xFFFFFFFF, 5])

    def test_an_op_sel_permlane_is_not_guessed(self):
        with self.assertRaises(cf_ref.Unsupported):
            machine(["v_permlane16_b32 v2, v1, s6, s6 op_sel:[1,0]"])


class Fuzzer(unittest.TestCase):
    def test_canaries(self):
        self.assertTrue(wave_diff.self_test())

    def test_a_reference_program_error_is_a_harness_error_not_a_skip(self):
        def broken(lines, wave):
            raise cf_ref.ProgramError("reference broke")
        real = wave_diff.run_model
        wave_diff.run_model = broken
        try:
            counts, first = wave_diff.run_suite(programs=1, seed=1, max_ops=2)
            self.assertEqual(wave_diff.main(["--programs", "1", "--max-ops", "2"]), 2)
        finally:
            wave_diff.run_model = real
        self.assertEqual((counts["harness-error"], counts["skip"]), (1, {}))
        self.assertIn("reference broke", first)

    def test_seeded_run_on_both_wave_sizes(self):
        counts, first = wave_diff.run_suite(programs=8, seed=1, max_ops=10, waves=(32, 64))
        self.assertEqual(counts["mismatch"], 0, first)
        self.assertEqual(counts["harness-error"], 0, first)
        self.assertEqual(counts["pass"], 8, counts["skip"])

    def test_replay_compares_only_case_five_of_the_same_mixed_wave_batch(self):
        options = ["--seed", "1", "--max-ops", "10", "--wave", "both"]
        with patch.object(wave_diff, "run_one", wraps=wave_diff.run_one) as runner:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(wave_diff.main(["--programs", "6", *options]), 0)
            self.assertEqual([call.args[2] for call in runner.call_args_list], [32, 64, 32, 64, 32, 64])
            case_five = runner.call_args_list[5]
            self.assertEqual(case_five.args[0], "cs_wdiff_0005")
            runner.reset_mock()
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(wave_diff.main(["--program-index", "5", *options]), 0)
            runner.assert_called_once_with(*case_five.args)
            self.assertIn("1 pass, 0 mismatch, 0 skipped, 0 harness errors", output.getvalue())

    def test_failure_commands_replay_the_reported_cases(self):
        options = ["--seed", "37", "--max-ops", "9", "--wave", "32", "--masks-as-data", "--permlane-partial"]
        for status, code, reporting in [("mismatch", 1, []), ("error", 2, ["--report-all"])]:
            with self.subTest(status=status), patch.object(wave_diff, "run_one", return_value=(status, "test")) as runner:
                with contextlib.chdir(ROOT), contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(wave_diff.main(["--programs", "2", *options, *reporting]), code)
                batch = runner.call_args_list[:]
                commands = [line.removeprefix("Replay: ").split() for line in output.getvalue().splitlines()
                            if line.startswith("Replay: ")]
                self.assertEqual(len(commands), 2 if reporting else 1)
                for i, command in enumerate(commands):
                    self.assertEqual(command[:3], ["python", "-I", "tools/wave_diff.py"])
                    self.assertEqual(command[3:], ["--program-index", str(i), *options])
                    runner.reset_mock()
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(wave_diff.main(command[3:]), code)
                    runner.assert_called_once_with(*batch[i].args)

    def test_replay_command_from_another_directory_locates_the_script(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            with patch.object(wave_diff, "run_one", return_value=("mismatch", "test")):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(wave_diff.main(["--program-index", "0"]), 1)
            line = next(line for line in output.getvalue().splitlines() if line.startswith("Replay: "))
            prefix, options = line.removeprefix("Replay: python -I ").split(" --program-index ", 1)
            if os.name == "nt" and prefix.startswith("'"):
                self.assertTrue(prefix.endswith("'"))
                script = prefix[1:-1].replace("''", "'")
            elif os.name == "nt":
                self.assertTrue(prefix.startswith('"') and prefix.endswith('"'))
                script = prefix[1:-1]
            else:
                script, = shlex.split(prefix)
            self.assertTrue(os.path.samefile(script, wave_diff.__file__))
            self.assertTrue(options.startswith("0 "))

    def test_existing_batch_abbreviations_keep_their_meaning(self):
        for args in (["--pr=2"], ["--program", "2"]):
            with self.subTest(args=args), patch.object(wave_diff, "run_one", return_value=("match", "")) as runner:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(wave_diff.main(args), 0)
                self.assertEqual([call.args[0] for call in runner.call_args_list], ["cs_wdiff_0000", "cs_wdiff_0001"])

    def test_invalid_replay_options_fail_before_running_a_case(self):
        for args in (["--program-index", "-1"], ["--program-index", "0", "--programs", "1"],
                     ["--program-index", "0", "--self-test"]):
            with self.subTest(args=args), patch.object(wave_diff, "run_one") as runner:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    wave_diff.main(args)
                self.assertEqual(error.exception.code, 2)
                runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
