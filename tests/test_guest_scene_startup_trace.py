"""CPU-only coverage of startup traces at the guest-scene loader boundary."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_guest_scene.py"
DEV = ROOT / "scripts" / "dev.py"
spec = importlib.util.spec_from_file_location("run_guest_scene", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def guest_path(path):
    return str(path) if os.name == "nt" else "Z:" + str(path).replace("/", "\\")


class GuestSceneStartupTraceTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="scene trace ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.loader = self.root / "loader.py"
        self.loader.write_text("raise SystemExit(7)\n", encoding="utf-8", newline="\n")
        self.guest = self.root / "guest.exe"
        self.guest.touch()
        (self.root / "spirv").mkdir()
        self.options = ["--loader", str(self.loader), "--guest", str(self.guest),
                        "--shaders", str(self.root)]

    def check_runs(self, enabled):
        out = self.root / "out"
        options = [str(RUNNER), *self.options, "--out", str(out)]
        if os.name != "nt":
            options.extend(["--wine", sys.executable])
        if enabled:
            options.append("--startup-trace")
        with mock.patch.object(sys, "argv", options):
            args = runner.parse_args()
        session = runner.Session(args, out)
        session.logs.mkdir(parents=True)
        package = self.root / "package"
        cases = [("full", "full", 0), ("save", "save", 0), ("resume", "resume", 0),
                 ("bad-scene", "full", 1), ("bad-checkpoint", "resume", 1)]
        expected_files = set()
        for name, mode, code in cases:
            with self.subTest(name=name):
                overlay = out / name
                path = session.logs / f"{name}.startup.jsonl"
                expected = ([args.wine] if args.wine else []) + [
                    str(self.loader), "--run", guest_path(self.guest), "--game-dir", guest_path(package),
                    "--overlay", guest_path(overlay)]
                if enabled:
                    expected.extend(["--trace", guest_path(path)])
                expected.extend(["--", guest_path(self.root)])

                def loader(cmd, **kwargs):
                    self.assertEqual(cmd, expected)
                    self.assertEqual(kwargs["cwd"], overlay)
                    self.assertEqual(kwargs["env"]["LEONIDA_SCENE_MODE"], mode)
                    if "--trace" in cmd:
                        events = [{"kind": "trace_open", "pid": 42}, {"kind": "exit", "code": code}]
                        path.write_text("".join(json.dumps(e) + "\n" for e in events),
                                        encoding="utf-8", newline="\n")
                    return subprocess.CompletedProcess(cmd, code)

                with mock.patch.object(runner.subprocess, "run", side_effect=loader):
                    rc, _, _ = session.run(name, mode, package, overlay)
                self.assertEqual(rc, code)
                record = json.loads((session.logs / f"{name}.json").read_bytes())
                self.assertEqual(record["cmd"], expected)
                expected_files.update([f"{name}.json", f"{name}.stdout.log", f"{name}.stderr.log"])
                if enabled:
                    expected_files.add(path.name)
                    self.assertEqual(Path(record["startup_trace"]), path)
                    result = subprocess.run([sys.executable, "-I", str(DEV), "boot-trace",
                                             str(path.relative_to(out)), "--json"],
                                            cwd=out, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    startup = json.loads(result.stdout)["startup"]
                    self.assertEqual((startup["events"], startup["bad_lines"], startup["exit"]), (2, 0, code))
                else:
                    self.assertNotIn("startup_trace", record)
        self.assertEqual({p.name for p in session.logs.iterdir()}, expected_files)

    def test_default_loader_arguments_and_logs(self):
        self.check_runs(False)

    def test_opt_in_traces_are_separate_and_readable_by_boot_trace(self):
        self.check_runs(True)

    def test_runner_does_not_report_a_trace_the_loader_never_created(self):
        out = self.root / "failed"
        result = subprocess.run([sys.executable, "-I", str(RUNNER), *self.options,
                                 "--wine", sys.executable, "--out", str(out), "--startup-trace"],
                                cwd=self.root, capture_output=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        record = json.loads((out / "report.json").read_bytes())["runs"]["full"]
        self.assertEqual(record["returncode"], 7)
        self.assertNotIn("startup_trace", record)
        self.assertFalse((out / "logs" / "full.startup.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
