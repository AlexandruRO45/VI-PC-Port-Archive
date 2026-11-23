"""tools/vk_fuzz_run.py (the compute-program wrapper behind the --oracle hw,vk modes of the differential fuzzers) and the
--oracle vk guard of each fuzzer that has the mode. Standard library only; no GPU, no llvm-mc.

    python -I tests/test_vk_fuzz_run.py
"""
import contextlib
import importlib
import io
import subprocess
import sys
import unittest
import unittest.mock as mock
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.dont_write_bytecode = True

import hw_oracle.runner  # noqa: E402
import vk_fuzz_run as R  # noqa: E402


class ProgramText(unittest.TestCase):
    def test_wave32_shell(self):
        text = R.program_text("cs_x", ["v_mov_b32 v1, 5"], 4, 32).splitlines()
        self.assertEqual(text[:5], ["; cs_x", "; group: 32,1,1", "; rsrc2: 0x8088", "s_mov_b32 exec_lo, -1",
                                    "v_mov_b32 v20, v0"])
        self.assertIn("v_lshlrev_b32 v21, 4, v20", text)
        self.assertEqual(text[-1], "s_endpgm")
        self.assertEqual(sum(ln.startswith("buffer_store_dword") for ln in text), 4)

    def test_store_stride_follows_the_word_count(self):
        for words, shift in ((1, 2), (2, 3), (4, 4), (8, 5)):
            self.assertIn(f"v_lshlrev_b32 v21, {shift}, v20", R.program_text("c", [], words, 32).splitlines())

    def test_wave64_rewrites_masks_and_vcc_data(self):
        w = R.wave64_text
        self.assertEqual(w("s_mov_b32 vcc_lo, 1"), "s_mov_b64 vcc, 1")
        self.assertEqual(w("s_mov_b32 vcc_lo, 0"), "s_mov_b64 vcc, 0")
        self.assertEqual(w("v_cndmask_b32_e64 v11, 0, 1, vcc_lo"), "v_cndmask_b32_e64 v11, 0, 1, vcc")
        self.assertEqual(w("v_cmp_lt_u32_e64 s20, v1, v2"), "v_cmp_lt_u32_e64 s[20:21], v1, v2")
        self.assertEqual(w("v_cndmask_b32_e64 v1, 0, 1, s21"), "v_cndmask_b32_e64 v1, 0, 1, s[22:23]")
        self.assertEqual(w("s_mov_b32 s4, 7"), "s_mov_b32 s4, 7")

    def test_rewrite_false_keeps_lines_that_already_spell_wave64(self):
        lines = ["v_cmp_lt_u32_e64 s[20:21], v1, v2", "s_mov_b32 s20, 3"]
        text = R.program_text("c", lines, 4, 64, rewrite=False).splitlines()
        self.assertIn("; mattr: +wavefrontsize64,-wavefrontsize32", text)
        self.assertIn("s_mov_b64 exec, -1", text)
        self.assertIn("s_mov_b32 s20, 3", text)
        self.assertIn("v_cmp_lt_u32_e64 s[20:21], v1, v2", text)

    def test_opaque_seeds_compute_the_same_values_at_run_time(self):
        self.assertEqual(R.opaque_seed("v_mov_b32_e32 v3, 0x3f800000"), "v_xor_b32_e32 v3, 0xd16eeeee, v22")
        self.assertEqual(R.opaque_seed("v_mov_b32 v1, 5"), "v_xor_b32_e32 v1, 0xeeeeeeeb, v22")
        for line in ("s_mov_b32 s4, 1", "v_mov_b32 v1, v2", "v_mov_b32 v20, 3", "v_add_f32 v1, v2, v3"):
            self.assertEqual(R.opaque_seed(line), line)
        text = R.program_text("c", ["v_mov_b32 v1, 5"], 4, 32, opaque=True).splitlines()
        self.assertIn("buffer_load_dword v22, v21, s[0:3], 0 offen", text)
        self.assertLess(text.index("s_waitcnt vmcnt(0)"), text.index("v_xor_b32_e32 v1, 0xeeeeeeeb, v22"))
        for k in (0x0, 0x1234, 0xFFFFFFFF, 0x7FC00000):
            self.assertEqual(k ^ R.FILL ^ R.FILL, k)

    def test_exact_verdict(self):
        lines = ["v_add_u32 v1, v2, v3", "v_mov_b32 v8, v1", "v_mov_b32 v9, v2", "v_mov_b32 v10, v3", "v_mov_b32 v11, v4"]
        self.assertEqual(R.exact_verdict(lines, [1, 2, 3, 4], [1, 2, 3, 4]), ("match", ""))
        klass, why = R.exact_verdict(lines, [1, 2, 3, 4], [1, 9, 3, 4])
        self.assertEqual(klass, "mismatch")
        self.assertIn("word 1 (v2)", why)
        klass, why = R.exact_verdict(["s_nop 0"], [1, 2], [1, 3])
        self.assertEqual(klass, "mismatch")
        self.assertIn("word 1 (word 1)", why)
        klass, why = R.exact_verdict(lines, [1, 2], [1])
        self.assertEqual(klass, "mismatch")
        self.assertIn("word count differs", why)

    def test_run_programs_rejects_invalid_wave_and_cleans_up_on_error(self):
        import os
        import tempfile
        import unittest.mock as mock
        with self.assertRaises(ValueError):
            R.run_programs([], words=4, wave=16)
        with tempfile.TemporaryDirectory() as tmp:
            fake_exe = Path(tmp) / ("vk_compute_test.exe" if os.name == "nt" else "vk_compute_test")
            fake_exe.write_bytes(b"")
            with mock.patch("mc_command.mc_command", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    R.run_programs([["s_nop 0"]], words=4, wave=32, build=tmp, work_parent=tmp)
            leftovers = [p for p in Path(tmp).iterdir() if p.name.startswith("vk_fuzz_")]
            self.assertEqual(leftovers, [])

    def test_judge_counts_skips_and_mismatches(self):
        import unittest.mock as mock
        obs = ["v_mov_b32 v8, v1", "v_mov_b32 v9, v2", "v_mov_b32 v10, v3", "v_mov_b32 v11, v4"]
        progs = [["a"] + obs, ["b"] + obs, ["c"] + obs]
        with mock.patch.object(R, "run_programs", return_value=[[1, 2, 3, 4], [1, 2, 3, 5], "llvm-mc: bad"]):
            counts, bad = R.judge(progs, [[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], R.exact_verdict)
        self.assertEqual(counts, {"match": 1, "mismatch": 1, "skip: llvm-mc: bad": 1})
        self.assertEqual([b[0] for b in bad], [1])

    def test_judge_counts_uniform_lane_divergence_as_mismatch(self):
        import unittest.mock as mock
        divergence = R.UniformLaneDivergence("lanes [1, 2, 3] differ from lane 0 on identical inputs")
        progs = [["a"], ["b"], ["c"], ["d"]]
        hardware = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], "unsupported"]
        results = [divergence, "llvm-mc: bad", [1, 2, 3, 4], [1, 2, 3, 4]]
        with mock.patch.object(R, "run_programs", return_value=results):
            counts, bad = R.judge(progs, hardware, R.exact_verdict)
        self.assertEqual(counts, {"mismatch": 1, "skip: llvm-mc: bad": 1, "match": 1,
                                  "skip: hardware: unsupported": 1})
        self.assertEqual(bad[0][0], 0)
        self.assertEqual(bad[0][1], f"Vulkan uniform-input result diverged: {divergence}")
        self.assertEqual(bad[0][2], progs[0])

    def test_run_programs_types_lane_divergence_from_synthetic_readback(self):
        import json
        import os
        import struct
        import tempfile
        import unittest.mock as mock
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / ("vk_compute_test.exe" if os.name == "nt" else "vk_compute_test")
            exe.write_bytes(b"")

            def run_child(args, **kwargs):
                if len(args) > 2 and args[2].endswith("rdna2spirv.py"):
                    spirv = Path(args[-1])
                    spirv.mkdir()
                    (spirv / "index.json").write_text(
                        json.dumps({"cs_vkfuzz_0": {}}), encoding="utf-8", newline="\n")
                    (spirv / "cs_vkfuzz_0.spv").write_bytes(b"synthetic")
                else:
                    dump = Path(kwargs["env"]["VKC_DUMP"])
                    values = [0x12345678] * 128
                    values[4] = 0x87654321
                    dump.write_bytes(struct.pack("<128I", *values))
                return subprocess.CompletedProcess(args, 0, "", "")

            assembler = subprocess.CompletedProcess([], 0, ".text\ns_endpgm\n", "")
            with mock.patch("mc_command.mc_command", return_value=["synthetic-mc"]), \
                 mock.patch("mc_command.run", return_value=assembler), \
                 mock.patch.object(R.subprocess, "run", side_effect=run_child):
                result = R.run_programs([[]], build=tmp, work_parent=tmp, opaque=False)
                per_lane = R.run_programs([[]], build=tmp, work_parent=tmp, opaque=False, per_lane=True)
                lane_zero = R.run_programs([[]], build=tmp, work_parent=tmp, opaque=False, uniform=False)
                counts, bad = R.judge([[]], [[0x12345678] * 4], R.exact_verdict,
                                      build=tmp, work_parent=tmp, opaque=False)

        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)
        self.assertIs(type(result[0]), R.UniformLaneDivergence)
        self.assertEqual(str(result[0]), "lanes [1] differ from lane 0 on identical inputs")
        self.assertEqual(len(per_lane[0]), 32)
        self.assertEqual(per_lane[0][1][0], 0x87654321)
        self.assertEqual(lane_zero, [[0x12345678] * 4])
        self.assertEqual(counts, {"mismatch": 1})
        self.assertEqual(bad[0][0], 0)

    def test_judge_nonuniform_mode_compares_lane_zero(self):
        import unittest.mock as mock
        progs = [["a"]]
        with mock.patch.object(R, "run_programs", return_value=[[1, 2, 3, 4]]) as run:
            counts, bad = R.judge(progs, [[1, 2, 3, 4]], R.exact_verdict, uniform=False)
        run.assert_called_once_with(progs, 4, 32, None, None, opaque=True, uniform=False)
        self.assertEqual(counts, {"match": 1})
        self.assertEqual(bad, [])

    def test_all_skipped(self):
        self.assertTrue(R.all_skipped({}))
        self.assertTrue(R.all_skipped({"skip: vk_compute_test: unsupported": 3}))
        self.assertTrue(R.all_skipped({"skip: hardware: no": 2, "match": 0}))
        self.assertFalse(R.all_skipped({"match": 1, "skip: x": 2}))
        self.assertFalse(R.all_skipped({"denormal": 1, "skip: x": 2}))


class Guard(unittest.TestCase):
    FUZZERS = ("pk16_diff", "vint_diff", "sdwa_diff", "wint_diff", "fmisc_diff")
    UNIFORM_FUZZERS = ("pk16_diff", "vint_diff", "fmisc_diff")

    def test_vk_without_hw_is_a_usage_error(self):
        for tool in self.FUZZERS:
            with self.subTest(tool=tool):
                r = subprocess.run([sys.executable, "-I", str(TOOLS / f"{tool}.py"), "--oracle", "vk", "--programs", "1"],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
                self.assertIn("--oracle hw,vk", r.stderr)

    def test_hw_vk_exits_two_when_all_vk_programs_skip(self):
        for tool in self.FUZZERS:
            mod = importlib.import_module(tool)
            for vk_counts, expected_rc in (
                ({"skip: vk_compute_test: unsupported": 2}, 2),
                ({"match": 1, "skip: vk_compute_test: unsupported": 1}, 0),
            ):
                oracle = mock.Mock(device="mock", arch="gfx1030")
                counts = {"pass": 2, "mismatch": 0, "skip": {}, "vk": dict(vk_counts), "vk_failures": []}
                out = io.StringIO()
                with self.subTest(tool=tool, expected_rc=expected_rc):
                    with mock.patch.object(hw_oracle.runner, "open_oracle", return_value=oracle), \
                         mock.patch.object(mod, "run_suite", return_value=(counts, [])), \
                         contextlib.redirect_stdout(out):
                        rc = mod.main(["--oracle", "hw,vk", "--programs", "2"])
                    self.assertEqual(rc, expected_rc, out.getvalue())
                    if expected_rc == 2:
                        self.assertIn("no Vulkan program ran: every one was skipped", out.getvalue())
                    else:
                        self.assertNotIn("no Vulkan program ran", out.getvalue())

    def test_uniform_lane_divergence_makes_fuzzer_campaign_fail(self):
        for tool in self.UNIFORM_FUZZERS:
            mod = importlib.import_module(tool)
            counts = {"pass": 2, "mismatch": 0, "skip": {},
                      "vk": {"mismatch": 1},
                      "vk_failures": [(0, "Vulkan uniform-input result diverged", ["v_add_u32 v1, v2, v3"])]}
            oracle = mock.Mock(device="mock", arch="gfx1030")
            out = io.StringIO()
            with self.subTest(tool=tool):
                with mock.patch.object(hw_oracle.runner, "open_oracle", return_value=oracle), \
                     mock.patch.object(mod, "run_suite", return_value=(counts.copy(), [])), \
                     contextlib.redirect_stdout(out):
                    rc = mod.main(["--oracle", "hw,vk", "--programs", "2"])
                self.assertEqual(rc, 1, out.getvalue())
                self.assertIn("VK MISMATCH program 0", out.getvalue())


if __name__ == "__main__":
    unittest.main()
