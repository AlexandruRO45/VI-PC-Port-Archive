"""Fixed reference points for the synthetic guest's renderer and input trace (no GPU)."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


path = Path(__file__).resolve().parents[1] / "scripts" / "run_guest_scene.py"
spec = importlib.util.spec_from_file_location("run_guest_scene", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GuestSceneReferenceTest(unittest.TestCase):
    def test_trace_and_pixel_digest_at_scene_boundaries(self):
        # These records were read back independently on the RTX 5090 and Apple M3 Max.
        anchors = [
            (0, 512, 9, 8, 18, 1, 376597285, 314582078),
            (137, 132, 44, 32, 153, 3, 3593765925, 2667449936),
            (299, 320, 8, 58, 169, 2, 1840220197, 1732923390),
            (300, 64, 8, 57, 177, 2, 1840220197, 1732923390),
            (422, 4, 58, 8, 228, 1, 1662446373, 2331208379),
            (599, 576, 58, 8, 30, 1, 1662446373, 2331208379),
        ]
        steps = runner.trace()
        for frame, buttons, x, y, scene_byte, bias, low, high in anchors:
            with self.subTest(frame=frame):
                self.assertEqual(steps[frame], (buttons, x, y, scene_byte, bias))
                digest, _ = runner.predicted_image(x % 8, bias)
                self.assertEqual(digest, (high << 32) | low)

    def test_pcm_envelope_and_period(self):
        samples = struct.unpack("<11025h", runner.expected_pcm(9000))
        self.assertEqual(samples[0], 0)
        self.assertEqual(samples[500:550], (9000,) * 50)
        self.assertEqual(samples[550:600], (-9000,) * 50)
        self.assertEqual(samples[-1], 0)


    def test_redirected_cli_preserves_unicode_paths_and_guest_diagnostic(self):
        # Force strict redirected text streams even on UTF-8 hosts. Invoke the real CLI with a
        # small stand-in loader that produces a diagnostic and exits before rendering.
        entry = (
            "import runpy, sys; "
            "encoding, script = sys.argv[1:3]; del sys.argv[1:3]; "
            "sys.stdout.reconfigure(encoding=encoding, errors='strict'); "
            "sys.stderr.reconfigure(encoding=encoding, errors='strict'); "
            "runpy.run_path(script, run_name='__main__')"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "spirv").mkdir()
            loader = root / "loader.py"
            diagnostic = "guest diagnostic: 測試"
            loader.write_text(f"import os\nos.write(2, {diagnostic.encode()!r})\nraise SystemExit(7)\n",
                              encoding="ascii")
            for encoding in ("cp1252", "utf-8"):
                with self.subTest(encoding=encoding):
                    out = root / f"測試 café {encoding}"
                    result = subprocess.run(
                        [sys.executable, "-I", "-c", entry, encoding, str(path), "--wine", sys.executable,
                         "--loader", str(loader), "--guest", str(loader), "--shaders", str(root), "--out", str(out)],
                        capture_output=True,
                    )
                    self.assertEqual(result.returncode, 1)
                    report = out / "report.json"
                    summary = json.loads(report.read_text(encoding="utf-8"))
                    self.assertEqual(summary["error"], "full: exit code 7")
                    self.assertEqual(summary["runs"]["full"]["outcome"], "exited")
                    self.assertEqual(summary["runs"]["full"]["returncode"], 7)
                    self.assertIn(str(report).encode(encoding, "backslashreplace"), result.stdout)
                    self.assertIn(diagnostic.encode(encoding, "backslashreplace"), result.stderr)
                    self.assertNotIn(b"UnicodeEncodeError", result.stderr)

    def test_launch_failure_preserves_run_record_and_logs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "spirv").mkdir()
            bad_loader = root / "invalid.exe"
            bad_loader.write_bytes(b"synthetic input that is not an executable")
            bad_loader.chmod(0o700)
            out = root / "output"
            cmd = [sys.executable, "-I", str(path), "--loader", str(bad_loader), "--guest", str(bad_loader),
                   "--shaders", str(root), "--out", str(out)]
            if sys.platform != "win32":
                cmd += ["--wine", str(bad_loader)]
            result = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=20)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertIn("could not launch", report["error"])
            record = json.loads((out / "logs" / "full.json").read_text(encoding="utf-8"))
            self.assertEqual(record, report["runs"]["full"])
            self.assertEqual(record["outcome"], "launch_failed")
            self.assertIsNone(record["returncode"])
            self.assertEqual(record["cmd"][0], str(bad_loader))
            self.assertTrue(record["launch_error"])
            for stream in ("stdout", "stderr"):
                contents = (out / "logs" / f"full.{stream}.log").read_bytes()
                self.assertEqual(record[f"{stream}_sha256"], runner.sha256(contents))

    def test_timeout_has_its_own_run_outcome(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = argparse.Namespace(wine=None, loader=root / "loader.exe", guest=root / "guest.exe",
                                      shaders=root, play_audio=False)
            session = runner.Session(args, root)
            session.logs.mkdir()
            with mock.patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("loader", 10)), \
                    self.assertRaisesRegex(runner.Fail, "timed out"):
                session.run("full", "full", root, root)
            self.assertEqual(session.runs["full"]["outcome"], "timed_out")
            self.assertIsNone(session.runs["full"]["returncode"])
            self.assertNotIn("launch_error", session.runs["full"])

    def test_wine_accepts_a_command_and_the_wine_environment_variable(self):
        # --wine is a binary or a shell-split command (umu-run, "proton run"); outside Windows $WINE is the default.
        # Only the first word is looked up and resolved. A synthetic stand-in records the argv it is given.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "spirv").mkdir()
            (root / "loader.exe").write_bytes(b"synthetic")
            spaced = root / "wine dir"
            spaced.mkdir()
            wrapper = spaced / "fake wine"
            wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
            wrapper.chmod(0o755)
            base = ["run_guest_scene.py", "--loader", str(root / "loader.exe"), "--guest", str(root / "loader.exe"),
                    "--shaders", str(root)]
            cases = [
                ({}, ["--wine", f"{shlex.quote(str(wrapper))} run --verbose"], [str(wrapper.resolve()), "run", "--verbose"]),
                ({}, ["--wine", str(wrapper)], [str(wrapper.resolve())]),
                ({"WINE": f"{shlex.quote(str(wrapper))} run"}, [], [str(wrapper.resolve()), "run"]),
                ({"WINE": "ignored-when-given"}, ["--wine", str(wrapper)], [str(wrapper.resolve())]),
            ]
            for env, extra, expected in cases:
                with self.subTest(env=env, extra=extra), mock.patch.object(sys, "argv", base + extra), \
                        mock.patch.dict(runner.os.environ, env, clear=False), mock.patch.object(runner.os, "name", "posix"):
                    if "WINE" not in env:
                        runner.os.environ.pop("WINE", None)
                    self.assertEqual(runner.parse_args().wine, expected)
            session = runner.Session(argparse.Namespace(wine=[str(wrapper), "run"], loader=root / "loader.exe",
                                                        guest=root / "loader.exe", shaders=root, play_audio=False), root)
            session.logs.mkdir()
            with mock.patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
                session.run("full", "full", root, root)
            self.assertEqual(run.call_args.args[0][:3], [str(wrapper), "run", str(root / "loader.exe")])

    def test_wine_command_errors_exit_with_code_2(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "spirv").mkdir()
            (root / "loader.exe").write_bytes(b"synthetic")
            base = ["run_guest_scene.py", "--loader", str(root / "loader.exe"), "--guest", str(root / "loader.exe"),
                    "--shaders", str(root)]
            for extra, env in ((["--wine", "no-such-wine-binary-xyz run"], {}), (["--wine", "'unterminated"], {}),
                               ([], {})):
                with self.subTest(extra=extra), mock.patch.object(sys, "argv", base + extra), \
                        mock.patch.dict(runner.os.environ, env), mock.patch.object(runner.os, "name", "posix"), \
                        mock.patch("sys.stderr"), self.assertRaises(SystemExit) as exit_:
                    runner.os.environ.pop("WINE", None)
                    runner.parse_args()
                self.assertEqual(exit_.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
