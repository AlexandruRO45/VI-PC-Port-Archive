"""Tests for scripts/run_wine_session.sh: which Wine boots the prefix and which wineserver the exit trap stops.

    python -I tests/test_run_wine_session.py

No Wine: `wine` and `wineserver` are tiny shell scripts that append what they were called with to a log.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wine_session.sh"
BASH = shutil.which("bash")


@unittest.skipIf(sys.platform == "win32" or not BASH or not shutil.which("timeout"), "needs bash and timeout")
class RunWineSessionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="wine-session-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.log = self.root / "calls.log"
        self.path_bin = self.stub_dir("path-bin")

    def stub_dir(self, name: str, *, server: bool = True) -> Path:
        folder = self.root / name
        folder.mkdir()
        for tool in ("wine", "wineserver") if server else ("wine",):
            stub = folder / tool
            stub.write_text(f'#!/bin/sh\nprintf \'%s\\n\' "{name}/{tool} $*" >> "$CALLS_LOG"\n')
            stub.chmod(0o755)
        return folder

    def run_session(self, *command: str, **env: str) -> tuple[subprocess.CompletedProcess, list[str]]:
        environment = {
            "PATH": os.pathsep.join((str(self.path_bin), os.environ.get("PATH", os.defpath))),
            "CALLS_LOG": str(self.log), "WINEPREFIX": str(self.root / "prefix"), **env,
        }
        for name in ("WINE", "WINESERVER"):
            if name not in env:
                environment.pop(name, None)
        result = subprocess.run([BASH, str(SCRIPT), *command], env=environment, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=60)
        calls = self.log.read_text().splitlines() if self.log.exists() else []
        return result, calls

    def test_without_wine_set_it_uses_the_wine_on_path(self):
        result, calls = self.run_session("sh", "-c", 'echo "tests $*" >> "$CALLS_LOG"', "x", "--a")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["path-bin/wine wineboot --init", "tests --a", "path-bin/wineserver -k"])

    def test_wine_outside_path_boots_and_stops_with_its_own_wineserver(self):
        proton = self.stub_dir("proton-bin")
        result, calls = self.run_session("true", WINE=str(proton / "wine"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["proton-bin/wine wineboot --init", "proton-bin/wineserver -k"])

    def test_wine_command_with_arguments_is_split(self):
        self.stub_dir("umu-bin", server=False)
        wrapper = self.root / "umu-bin" / "umu-run"
        (self.root / "umu-bin" / "wine").rename(wrapper)
        result, calls = self.run_session("true", WINE=f"{wrapper} --verbose")
        self.assertEqual(result.returncode, 0, result.stderr)
        # umu-bin has no wineserver of its own, so the one on PATH is stopped.
        self.assertEqual(calls, ["umu-bin/wine --verbose wineboot --init", "path-bin/wineserver -k"])

    def test_wineserver_overrides_the_lookup(self):
        proton = self.stub_dir("proton-bin")
        other = self.stub_dir("other-bin")
        result, calls = self.run_session("true", WINE=str(proton / "wine"), WINESERVER=str(other / "wineserver"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["proton-bin/wine wineboot --init", "other-bin/wineserver -k"])

    def test_a_failing_command_still_stops_the_server_and_keeps_its_status(self):
        result, calls = self.run_session("sh", "-c", "exit 7")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(calls[-1], "path-bin/wineserver -k")

    def test_blank_wine_is_refused(self):
        result, calls = self.run_session("true", WINE="  ")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("WINE is blank", result.stderr)
        self.assertNotIn("path-bin/wine wineboot --init", calls)


if __name__ == "__main__":
    unittest.main()
