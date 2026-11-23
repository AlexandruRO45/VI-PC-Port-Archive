"""tools/progress_report.py and tools/launcher_probe.py, the two helpers of Leonida.exe (launcher-wpf/).

The progress reports must change nothing unless LEONIDA_PROGRESS_DIR is set (Build.bat runs exactly as before), and
when it is set a reader must only ever see whole reports. The probe must answer with make_game.py's own defaults, so
the launcher never keeps a second copy of them.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
sys.path.insert(0, ROOT)

import progress_report  # noqa: E402


class ProgressReportTest(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.pop(progress_report.ENV, None)

    def tearDown(self):
        os.environ.pop(progress_report.ENV, None)
        if self.saved is not None:
            os.environ[progress_report.ENV] = self.saved

    def test_off_without_the_variable(self):
        with tempfile.TemporaryDirectory() as d:
            p = progress_report.Progress("translate.shard0")
            self.assertFalse(p)
            p.update(force=True, total=3, done=3)
            self.assertEqual(os.listdir(d), [])

    def test_writes_whole_reports_and_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ[progress_report.ENV] = d
            p = progress_report.Progress("disasm")
            p.update(total=10, done=1)
            p.update(total=10, done=2)          # within the interval: skipped
            p.update(force=True, total=10, done=10, finished=True)
            self.assertEqual(os.listdir(d), ["disasm.json"])
            with open(os.path.join(d, "disasm.json"), encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual((report["total"], report["done"], report["finished"]), (10, 10, True))
            self.assertIn("started", report)

    def test_a_missing_folder_is_not_an_error(self):
        os.environ[progress_report.ENV] = os.path.join(tempfile.gettempdir(), "leonida-no-such-folder-xyz")
        progress_report.Progress("x").update(force=True, done=1)   # must not raise

    def test_the_shader_tools_import_it(self):
        for tool in ("rdna2spirv.py", "shader_disasm.py"):
            with open(os.path.join(TOOLS, tool), encoding="utf-8") as f:
                self.assertIn("from progress_report import Progress", f.read(), tool)


class LauncherProbeTest(unittest.TestCase):
    def test_answers_with_make_games_defaults(self):
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "launcher_probe.py")],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        info = json.loads(r.stdout.strip().splitlines()[-1])
        import make_game
        self.assertEqual(info["stages"], make_game.STAGES)
        self.assertEqual(info["wave64_modes"], list(make_game.WAVE64_MODES))
        self.assertEqual(info["min_free_gb"], make_game.MIN_FREE_GB)
        self.assertEqual(os.path.normcase(info["default_out"]), os.path.normcase(make_game.default_out()))
        self.assertGreaterEqual(info["default_jobs"], 1)
        names = {t["name"] for t in info["tools"]}
        self.assertTrue({"Python", "llvm-mc", "spirv-as", "spirv-val", ".NET 8 SDK"} <= names, names)
        self.assertTrue(all(isinstance(t["ok"], bool) for t in info["tools"]))
        self.assertTrue(any(v["file"].endswith(".json") for v in info["versions"]))

    def test_free_space(self):
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "launcher_probe.py"), "--free",
                            os.path.join(tempfile.gettempdir(), "not", "made", "yet")],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertGreater(json.loads(r.stdout.strip().splitlines()[-1])["free_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
