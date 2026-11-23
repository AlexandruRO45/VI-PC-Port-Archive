"""make_game.py: the pipelines stage checks every module index under shaders/spirv, not only the top one (BUILD-1842).

vk_pipeline_check reads one folder's index.json and does not descend, so the FidelityFX programs the shaders stage
translates into spirv/fspx were never created on the driver by the build. Standard library only; no GPU, Vulkan SDK
or game data: the checker is scripted.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import make_game  # noqa: E402


class PipelineFoldersTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="pipeline folders ")
        self.addCleanup(tmp.cleanup)
        self.work = os.path.join(tmp.name, "work")
        self.out = os.path.join(tmp.name, "out")
        self.spirv = os.path.join(self.work, "shaders", "spirv")
        self.fspx = os.path.join(self.spirv, "fspx")
        self.write_index(self.spirv, ["b_mod", "a_mod", "c_mod"])
        self.write_index(self.fspx, ["fsr_0", "fsr_1"])
        self.calls = []

    @staticmethod
    def write_index(folder, names):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({n: {} for n in names}, f)

    def stage(self, script=lambda args: 0, supports=True):
        """script(args) -> exit code, as vk_pipeline_check would return it for that invocation."""
        def fake_run(args, hint="", check=True, **kw):
            args = [str(a) for a in args]
            self.calls.append(args)
            code = script(args)
            if code in (0, 1) and "--inflight-file" in args:
                # A native checker that finishes its shader pass clears this report, even on validation failure.
                with open(args[args.index("--inflight-file") + 1], "w", encoding="utf-8", newline="\n") as f:
                    f.write("")
            if check and code != 0:   # as run() does
                raise SystemExit(1)
            return code
        logs, err = [], io.StringIO()
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="vk_pipeline_check.exe"), \
                mock.patch.object(make_game, "checker_supports", return_value=supports), \
                mock.patch.object(make_game, "run", side_effect=fake_run), \
                mock.patch.object(make_game, "log", logs.append), contextlib.redirect_stderr(err):
            try:
                make_game.pipelines(self.work, self.out, 2)
                return None, logs, err.getvalue()
            except SystemExit as e:
                return e.code, logs, err.getvalue()

    def folders_checked(self):
        return [c[1] for c in self.calls]

    def test_the_fspx_index_is_checked_after_the_main_one(self):
        code, logs, err = self.stage()
        self.assertIsNone(code, err)
        self.assertEqual(self.folders_checked(), [self.spirv, self.fspx])
        cache = os.path.join(self.out, "pipeline_cache.bin")
        self.assertTrue(all(c[2] == cache for c in self.calls), self.calls)   # one cache, as the runtime loads
        self.assertTrue(all(c[c.index("--jobs") + 1] == "2" for c in self.calls), self.calls)
        summary = [m for m in logs if "per folder" in m]
        self.assertEqual(len(summary), 1, logs)
        self.assertIn(f"{os.path.join('shaders', 'spirv')} 3", summary[0])
        self.assertIn(f"{os.path.join('shaders', 'spirv', 'fspx')} 2", summary[0])
        self.assertEqual(logs[-1], "pipelines: done")

    def test_an_older_checker_also_gets_every_folder(self):
        code, _, err = self.stage(supports=False)
        self.assertIsNone(code, err)
        self.assertEqual(self.folders_checked(), [self.spirv, self.fspx])

    def test_folders_without_an_index_are_not_passed(self):
        os.makedirs(os.path.join(self.spirv, "scratch"))
        code, _, err = self.stage()
        self.assertIsNone(code, err)
        self.assertEqual(self.folders_checked(), [self.spirv, self.fspx])

    def test_every_index_folder_is_found_by_name(self):
        bvh = os.path.join(self.spirv, "bvh")   # #5680 translates XboxBvhBuilder.dll's programs there
        self.write_index(bvh, ["LeafRehydration"])
        code, _, err = self.stage()
        self.assertIsNone(code, err)
        self.assertEqual(self.folders_checked(), [self.spirv, bvh, self.fspx])

    def test_a_failure_in_the_main_folder_still_checks_fspx_and_fails_the_stage(self):
        code, _, err = self.stage(lambda args: 1 if args[1] == self.spirv else 0)
        self.assertEqual(code, 1)
        self.assertEqual(self.folders_checked(), [self.spirv, self.fspx])
        self.assertIn(os.path.join("shaders", "spirv") + ": exited with 1", err)
        self.assertNotIn("fspx", err)

    def test_a_rejected_fspx_module_fails_the_stage(self):
        code, _, err = self.stage(lambda args: 1 if args[1] == self.fspx else 0)
        self.assertEqual(code, 1)
        self.assertIn(os.path.join("shaders", "spirv", "fspx") + ": exited with 1", err)

    def test_a_crashing_fspx_module_is_named_as_the_runtime_names_it(self):
        inflight = os.path.join(self.out, "pipeline_inflight.txt")

        def script(args):
            if args[1] != self.fspx:
                return 0
            if "--skip-file" in args:
                return 1   # the checker counts a skipped module as failed
            with open(inflight, "w", encoding="utf-8", newline="\n") as f:
                f.write("fsr_1\tpipeline\n")   # the batch and the probe of fsr_1 alone both die in it
            return -11
        code, logs, err = self.stage(script)
        self.assertEqual(code, 1, logs)
        self.assertIn("fspx/fsr_1", err)
        self.assertEqual(self.calls[0][1], self.spirv)

    def test_a_setup_error_stops_before_the_next_folder(self):
        code, _, _ = self.stage(lambda args: 2)
        self.assertEqual(code, 1)
        self.assertEqual(self.folders_checked(), [self.spirv])

    def test_a_single_folder_keeps_its_message_and_log(self):
        os.remove(os.path.join(self.fspx, "index.json"))
        code, logs, err = self.stage()
        self.assertIsNone(code, err)
        self.assertEqual(self.folders_checked(), [self.spirv])
        self.assertFalse(any("per folder" in m for m in logs), logs)
        code, _, err = self.stage(lambda args: 1)
        self.assertEqual(code, 1)
        self.assertIn("vk_pipeline_check exited with 1", err)


if __name__ == "__main__":
    unittest.main()
