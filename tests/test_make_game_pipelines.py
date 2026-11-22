"""make_game.py: vk_pipeline_check selection and arguments in the pipelines stage.

Standard library only; no GPU, no Vulkan SDK, no game data: the checker is mocked.
"""
import hashlib
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


class PipelinesJobsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.game = os.path.join(self.root, "game")
        self.work = os.path.join(self.root, "work")
        self.out = os.path.join(self.root, "out")
        spirv = os.path.join(self.work, "shaders", "spirv")
        os.makedirs(self.game)
        os.makedirs(spirv)
        with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8") as f:
            f.write("{}")

    def checker_args(self, call):
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="vk_pipeline_check.exe"), \
                mock.patch.object(make_game, "run", return_value=0) as run, \
                mock.patch.object(make_game, "log", lambda *a: None):
            call()
        self.assertEqual(run.call_count, 1)
        return [str(a) for a in run.call_args.args[0]]

    def jobs_of(self, args):
        self.assertEqual(args.count("--jobs"), 1, args)
        return args[args.index("--jobs") + 1]

    def main_with(self, *extra):
        argv = ["make_game.py", "--game", self.game, "--work", self.work, "--out", self.out,
                "--stages", "pipelines", *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(make_game, "below_normal"), \
                mock.patch.object(make_game, "check", return_value={"expected_exe": "g.exe"}), \
                mock.patch.object(make_game, "preflight"):
            self.assertEqual(make_game.main(), 0)

    def test_the_stage_passes_its_jobs(self):
        args = self.checker_args(lambda: make_game.pipelines(self.work, self.out, 7))
        self.assertEqual(self.jobs_of(args), "7")
        self.assertEqual(args[1:3], [os.path.join(self.work, "shaders", "spirv"),
                                     os.path.join(self.out, "pipeline_cache.bin")])

    def test_jobs_on_the_command_line_reach_the_checker(self):
        self.assertEqual(self.jobs_of(self.checker_args(lambda: self.main_with("--jobs", "5"))), "5")

    def test_windows_reuses_the_checker_from_the_supplied_runtime_build(self):
        supplied = os.path.join(self.root, "unpacked runtime")
        os.makedirs(supplied)
        checker = os.path.join(supplied, "vk_pipeline_check.exe")
        for name in (*make_game.RUNTIME_FILES, "vk_pipeline_check.exe"):
            with open(os.path.join(supplied, name), "wb"):
                pass
        argv = ["make_game.py", "--game", self.game, "--work", self.work, "--out", self.out,
                "--stages", "runtime,pipelines", "--runtime-build", supplied, "--jobs", "3"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(make_game, "ROOT", self.root), \
                mock.patch("os.name", "nt"), mock.patch.object(make_game, "below_normal"), \
                mock.patch.object(make_game, "check", return_value={"expected_exe": "g.exe"}), \
                mock.patch.object(make_game, "log"), mock.patch.object(make_game, "run") as run:
            self.assertEqual(make_game.main(), 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][0], checker)
        self.assertEqual(self.jobs_of(run.call_args.args[0]), "3")

    def test_the_default_matches_the_checker_default(self):
        # vk_pipeline_check's own default is half the hardware threads, as is make_game.py's --jobs.
        want = str(max(1, (os.cpu_count() or 2) // 2))
        self.assertEqual(self.jobs_of(self.checker_args(lambda: self.main_with())), want)

    def test_the_stage_log_names_compute_and_graphics_pipelines(self):
        # The stage runs vk_pipeline_check without --stage, and its default is all: cs become compute pipelines,
        # ps/vs/ms/gs become graphics pipelines with partner stubs, so the announce line must not claim compute only.
        logs = []
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="vk_pipeline_check.exe"), \
                mock.patch.object(make_game, "checker_supports", return_value=False), \
                mock.patch.object(make_game, "run", return_value=0), \
                mock.patch.object(make_game, "log", logs.append):
            make_game.pipelines(self.work, self.out)
        announce = [m for m in logs if "creating" in m]
        self.assertEqual(len(announce), 1, logs)
        self.assertIn("graphics", announce[0])
        self.assertIn("compute", announce[0])


ABORT = 0x80000003   # how Windows reports the driver's LLVM abort (#884)


class PipelinesCrashRetryTest(unittest.TestCase):
    """A module that takes vk_pipeline_check down is skipped and the rest is checked; the checker is scripted."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = os.path.join(tmp.name, "work")
        self.out = os.path.join(tmp.name, "out")
        self.spirv = os.path.join(self.work, "shaders", "spirv")
        os.makedirs(self.spirv)
        self.b_hash = "ab" * 32
        self.c_spirv = b"synthetic c_mod"
        with open(os.path.join(self.spirv, "index.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({"b_mod": {"fssm_sha256": self.b_hash}, "a_mod": {}, "c_mod": {}}, f)
        for name, data in (("a_mod", b"synthetic a_mod"), ("c_mod", self.c_spirv)):
            with open(os.path.join(self.spirv, name + ".spv"), "wb") as f:
                f.write(data)
        self.inflight = os.path.join(self.out, "pipeline_inflight.txt")
        self.skip = os.path.join(self.out, "pipeline_crashes.txt")
        self.report = os.path.join(self.out, "gpu.json")
        self.calls = []

    def stage(self, script, supports=True):
        """Return an exit code or (code, stdout) to override the range protocol."""
        def fake_run(args, hint="", check=True, **kw):
            args = [str(a) for a in args]
            self.calls.append(args)
            code = script(args, len(self.calls))
            if isinstance(code, tuple):
                code, text = code
                kw["stdout"].write(text)
            elif "--isolate-child" in args:
                kw["stdout"].write("ISO-READY\n")
                if code == 0:
                    kw["stdout"].write("ISO-DONE cache_ok=1\n")
            if check and code != 0:   # as run() does
                raise SystemExit(1)
            return code
        logs = []
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="vk_pipeline_check.exe"), \
                mock.patch.object(make_game, "checker_supports", return_value=supports), \
                mock.patch.object(make_game, "run", side_effect=fake_run), \
                mock.patch.object(make_game, "log", logs.append):
            try:
                make_game.pipelines(self.work, self.out)
                return None, logs
            except SystemExit:
                return sys.stderr, logs

    def inflight_is(self, *names):
        """As vk_pipeline_check writes it: name<TAB>phase, where a module can die in either phase."""
        with open(self.inflight, "w", encoding="utf-8") as f:
            f.write("".join(f"{n}\t{'module' if i % 2 else 'pipeline'}\n" for i, n in enumerate(names)))

    def skipped(self, args):
        return args[args.index("--skip-file") + 1] if "--skip-file" in args else None

    def selected_range(self, args):
        start = int(args[args.index("--start") + 1])
        count = int(args[args.index("--count") + 1])
        return range(start, start + count)

    def crash_report(self):
        with open(self.report, encoding="utf-8") as f:
            return json.load(f)["pipeline_crashes"]

    def test_a_named_module_is_skipped_and_the_rest_checked(self):
        def script(args, n):
            if n == 1:
                self.inflight_is("b_mod")
                return ABORT
            if "--isolate-child" in args:
                self.assertEqual(self.selected_range(args), range(1, 2))
                return ABORT
            self.assertEqual(self.skipped(args), self.skip)
            with open(self.skip, encoding="utf-8") as f:
                self.assertEqual(f.read(), "b_mod\n")
            return 1   # the skipped module counts as failed
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            failed, logs = self.stage(script)
        self.assertIsNotNone(failed)
        self.assertEqual(len(self.calls), 3)
        self.assertIn("--inflight-file", self.calls[0])
        self.assertIsNone(self.skipped(self.calls[0]))
        self.assertTrue(any("b_mod took vk_pipeline_check down" in m and "0x80000003" in m for m in logs), logs)
        self.assertIn("1 module(s) took the driver down", err.getvalue())
        self.assertIn("b_mod", err.getvalue())
        self.assertEqual(self.crash_report(), [{"name": "b_mod", "exit_code": ABORT, "fssm_sha256": self.b_hash}])
        self.assertFalse(os.path.exists(self.inflight))

    def test_several_in_flight_are_tried_alone_and_only_the_crasher_is_skipped(self):
        def script(args, n):
            if "--start" in args:   # the index is name-sorted, so a_mod=0, b_mod=1, c_mod=2
                self.assertEqual(args[args.index("--count") + 1], "1")
                return ABORT if args[args.index("--start") + 1] == "2" else 0
            if n == 1:
                self.inflight_is("a_mod", "c_mod")
                return ABORT
            with open(self.skip, encoding="utf-8") as f:
                self.assertEqual(f.read(), "c_mod\n")
            return 1
        with mock.patch("sys.stderr", io.StringIO()):
            failed, _ = self.stage(script)
        self.assertIsNotNone(failed)
        starts = [c[c.index("--start") + 1] for c in self.calls if "--start" in c]
        self.assertEqual(starts, ["0", "2"])

    def test_a_crash_without_a_snapshot_bisects_and_checks_both_sides(self):
        def script(args, n):
            if "--isolate-child" not in args:
                return ABORT
            return ABORT if 2 in self.selected_range(args) else 0
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            failed, _ = self.stage(script)
        self.assertIsNotNone(failed)
        self.assertEqual([self.selected_range(args) for args in self.calls[1:]],
                         [range(0, 3), range(0, 1), range(1, 3), range(1, 2), range(2, 3)])
        self.assertEqual(self.crash_report(), [
            {"name": "c_mod", "exit_code": ABORT, "spirv_sha256": hashlib.sha256(self.c_spirv).hexdigest()}])
        self.assertIn("pipeline ranges checked", err.getvalue())

    def test_a_stale_snapshot_or_an_unreproduced_crash_does_not_name_a_module(self):
        def script(args, n):
            if n == 1:
                self.inflight_is("a_mod")
                return ABORT
            return ABORT if self.selected_range(args) == range(0, 3) else 0
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            failed, logs = self.stage(script)
        self.assertIsNotNone(failed)
        self.assertFalse(os.path.exists(self.skip))
        self.assertFalse(os.path.exists(self.report))
        self.assertIn("could not be attributed", err.getvalue())
        self.assertTrue(any("did not recur after splitting" in m for m in logs), logs)

    def test_setup_shutdown_and_interruption_do_not_blame_a_module(self):
        for result in ((ABORT, ""), (0xC000013A, "ISO-READY\n"),
                       (ABORT, "ISO-READY\nISO-RESULT cs 1 0 1 0 0 0 0 0 0\n"),
                       (1, "ISO-READY\nISO-DONE cache_ok=0\n")):
            with self.subTest(result=result):
                self.calls.clear()
                err = io.StringIO()
                with mock.patch("sys.stderr", err):
                    failed, _ = self.stage(lambda args, n: ABORT if n == 1 else result)
                self.assertIsNotNone(failed)
                self.assertEqual(len(self.calls), 2)
                self.assertFalse(os.path.exists(self.report))
                self.assertIn("stopped during range checking", err.getvalue())

    def test_ctrl_c_is_not_retried(self):
        def script(args, n):
            self.inflight_is("b_mod")
            return 0xC000013A
        with mock.patch("sys.stderr", io.StringIO()):
            failed, _ = self.stage(script)
        self.assertIsNotNone(failed)
        self.assertEqual(len(self.calls), 1)

    def test_exhausting_full_restarts_switches_to_ranges_with_the_skip_list(self):
        initial = iter(["a_mod", "b_mod"])
        def script(args, n):
            if "--isolate-child" not in args:
                self.inflight_is(next(initial))
                return ABORT
            if n <= 4:
                return ABORT
            self.assertEqual(self.skipped(args), self.skip)
            with open(self.skip, encoding="utf-8") as f:
                self.assertEqual(f.read(), "a_mod\nb_mod\n")
            return 0
        err = io.StringIO()
        with mock.patch("sys.stderr", err), mock.patch.object(make_game, "PIPELINE_CRASH_RETRIES", 1):
            failed, _ = self.stage(script)
        self.assertIsNotNone(failed)
        self.assertEqual(len(self.calls), 5)
        self.assertIn("2 module(s) took the driver down", err.getvalue())

    def test_isolation_probes_have_a_budget(self):
        err = io.StringIO()
        with mock.patch("sys.stderr", err), mock.patch.object(make_game, "PIPELINE_CRASH_PROBES", 2):
            failed, _ = self.stage(lambda args, n: ABORT)
        self.assertIsNotNone(failed)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual([item["name"] for item in self.crash_report()], ["a_mod"])
        self.assertIn("reached 2 probes", err.getvalue())

    def test_a_clean_run_is_one_call_and_an_old_skip_list_is_dropped(self):
        os.makedirs(self.out)
        with open(self.skip, "w", encoding="utf-8") as f:
            f.write("b_mod\n")   # from an earlier build: every module gets a fresh chance
        with open(self.report, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"device": "external data", "pipeline_crashes": [{"name": "b_mod"}]}, f)
        failed, logs = self.stage(lambda args, n: 0)
        self.assertIsNone(failed)
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.skipped(self.calls[0]))
        self.assertFalse(os.path.exists(self.skip))
        with open(self.report, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"device": "external data"})
        self.assertIn("pipelines: done", logs)


    def test_an_older_checker_gets_the_old_invocation(self):
        # Before --inflight-file the checker took unknown arguments as the cache path (it would write the cache into
        # pipeline_inflight.txt), so an older build runs as before: one call, no new flags, a crash stops the stage.
        def script(args, n):
            self.inflight_is("b_mod")
            return ABORT
        with mock.patch("sys.stderr", io.StringIO()):
            failed, logs = self.stage(script, supports=False)
        self.assertIsNotNone(failed)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("--inflight-file", self.calls[0])
        self.assertNotIn("--skip-file", self.calls[0])
        self.assertTrue(any("has no --inflight-file" in m for m in logs), logs)

    def test_support_is_read_from_the_usage_line(self):
        new = "usage: vk_pipeline_check <spirv dir> ... [--inflight-file F] [--skip-file F]\n"
        old = "usage: vk_pipeline_check <spirv dir> [cache.bin] [--jobs N] [--isolate [--isolate-batch N]]\n"
        for usage, want in ((new, True), (old, False)):
            done = mock.Mock(stdout="", stderr=usage, returncode=2)
            with mock.patch.object(make_game.subprocess, "run", return_value=done) as sp:
                self.assertEqual(make_game.checker_supports("vk.exe", "--inflight-file", "--skip-file"), want)
            self.assertEqual(sp.call_args.args[0], ["vk.exe"])   # no arguments: usage only, no Vulkan
        with mock.patch.object(make_game.subprocess, "run", side_effect=OSError("not an executable")):
            self.assertFalse(make_game.checker_supports("vk.exe", "--inflight-file"))


if __name__ == "__main__":
    unittest.main()
