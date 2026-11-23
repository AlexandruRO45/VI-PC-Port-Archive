"""Verify Python 3.10+ requirement in runtime/CMakeLists.txt and Python3_EXECUTABLE forwarding."""
import base64
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import make_game  # noqa: E402
sys.path.insert(0, str(ROOT / "tests"))
import test_build_runtime_cmake as cmake_test  # noqa: E402

CMAKELISTS = ROOT / "runtime" / "CMakeLists.txt"
BUILD_RUNTIME_PS1 = ROOT / "scripts" / "build_runtime.ps1"


class CMakePythonVersionTest(unittest.TestCase):
    def test_cmakelists_requires_python3_min_version(self):
        src = CMAKELISTS.read_text(encoding="utf-8")
        calls = re.findall(r"find_package\s*\(\s*Python3\b([^)]*)\)", src)
        self.assertGreaterEqual(len(calls), 5)
        for args in calls:
            m = re.match(r"^\s*(\d+)\.(\d+)(?:\.\d+)?\b", args)
            self.assertIsNotNone(m, f"unversioned find_package(Python3{args})")
            ver = (int(m.group(1)), int(m.group(2)))
            self.assertGreaterEqual(ver, (3, 10), f"Python3 minimum {ver} < (3, 10)")

    def test_make_game_forwards_python3_executable(self):
        recorded = []
        with tempfile.TemporaryDirectory(prefix="leonida_mg_py_") as tmp:
            with mock.patch.object(make_game, "run", side_effect=lambda cmd, **kw: recorded.append(cmd)):
                make_game.cmake_configure(tmp, "-DTEST=1")
                self.assertEqual(len(recorded), 1)
                self.assertIn(f"-DPython3_EXECUTABLE={sys.executable}", recorded[0])

                recorded.clear()
                custom = r"C:\custom\python.exe"
                make_game.cmake_configure(tmp, f"-DPython3_EXECUTABLE={custom}")
                py_args = [a for a in recorded[0] if a.startswith("-DPython3_EXECUTABLE=")]
                self.assertEqual(py_args, [f"-DPython3_EXECUTABLE={custom}"])

            if os.name == "nt":
                out_dir = Path(tmp) / "out"
                built_dir = ROOT / "runtime" / "build"
                fake_bins = [built_dir / name for name in make_game.RUNTIME_FILES]
                existed = [p.exists() for p in fake_bins]
                try:
                    built_dir.mkdir(parents=True, exist_ok=True)
                    for p in fake_bins:
                        if not p.exists():
                            p.write_bytes(b"")
                    recorded.clear()
                    with mock.patch.object(make_game, "run", side_effect=lambda cmd, **kw: recorded.append(cmd)):
                        make_game.runtime("game", str(out_dir), {"expected_exe": "fake.exe"}, jobs=2)
                    self.assertEqual(len(recorded), 1)
                    cmd = recorded[0]
                    self.assertIn("-Python", cmd)
                    self.assertEqual(cmd[cmd.index("-Python") + 1], sys.executable)
                finally:
                    for p, was in zip(fake_bins, existed):
                        if not was and p.exists():
                            p.unlink()


@unittest.skipUnless(os.name == "nt" and cmake_test.CSC.is_file(),
                     "requires Windows PowerShell and .NET Framework csc")
class BuildRuntimePs1PythonArgTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="leonida_ps1_py_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        source = cls.base / "recorder.cs"
        source.write_text(cmake_test.RECORDER, encoding="utf-8", newline="\n")
        cls.recorder = cls.base / "recorder.exe"
        subprocess.run([str(cmake_test.CSC), "/nologo", "/target:exe", "/out:" + str(cls.recorder), str(source)],
                       check=True, capture_output=True, timeout=30)

    def test_build_runtime_ps1_forwards_python_arg(self):
        text = BUILD_RUNTIME_PS1.read_text(encoding="utf-8-sig")
        start = text.index("$cfg = if ($Debug)")
        end = text.index("} else {", text.index("if ($cmake -and $ninja)", start))
        branch = text[start:end] + "}\n"
        writer = text[text.index('$bat = Join-Path $out "_build.cmd"'):]
        vcvars = self.base / "vcvars64.bat"
        vcvars.write_text("@exit /b 0\n", encoding="ascii", newline="\r\n")

        for extension in (".exe", ".cmd", ".bat"):
            with self.subTest(extension=extension):
                with tempfile.TemporaryDirectory(dir=self.base) as tmp:
                    folder = Path(tmp) / "пробел ^ & 東京"
                    folder.mkdir()
                    root = folder / "checkout"
                    out = root / "runtime" / "build"
                    out.mkdir(parents=True)
                    tools = folder / "tools ^ & 東京"
                    tools.mkdir()
                    backend = tools / "recorder.exe"
                    shutil.copyfile(self.recorder, backend)
                    cmake = tools / ("cmake" + extension)
                    if extension == ".exe":
                        shutil.copyfile(self.recorder, cmake)
                    else:
                        cmake.write_text('@echo off\nchcp 65001 >nul\n"' + str(backend) + '" %*\nexit /b %errorlevel%\n',
                                         encoding="utf-8", newline="\r\n")
                    py_exe = folder / "py ^ & 東京" / "python.exe"
                    ninja = tools / "ninja.exe"
                    command = (
                        "$ErrorActionPreference = 'Stop'\n"
                        "$root = $env:LEONIDA_CMAKE_ROOT\n"
                        "$src = Join-Path $root 'runtime'\n"
                        "$out = Join-Path $src 'build'\n"
                        "$installer = $env:LEONIDA_CMAKE_INSTALLER\n"
                        "$vcvars = Join-Path $installer 'vcvars64.bat'\n"
                        "$cmake = $env:LEONIDA_CMAKE_EXE\n"
                        "$ninja = $env:LEONIDA_CMAKE_NINJA\n"
                        "$Python = $env:LEONIDA_CMAKE_PYTHON\n"
                        "$Debug = $false\n$FetchSdl3 = $false\n$jobs = 2\n"
                        + branch + writer
                    )
                    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
                    res = subprocess.run(
                        ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                        env=dict(os.environ, LEONIDA_CMAKE_ROOT=str(root), LEONIDA_CMAKE_LOG=str(folder),
                                 LEONIDA_CMAKE_INSTALLER=str(self.base), LEONIDA_CMAKE_EXE=str(cmake),
                                 LEONIDA_CMAKE_NINJA=str(ninja), LEONIDA_CMAKE_PYTHON=str(py_exe),
                                 LEONIDA_CMAKE_FAIL=""),
                        capture_output=True, timeout=45,
                    )
                    self.assertEqual(res.returncode, 0, (res.stdout + res.stderr).decode(errors="replace"))
                    configure = (folder / "configure.txt").read_text(encoding="utf-8-sig").splitlines()
                    self.assertIn(f"-DPython3_EXECUTABLE={py_exe}", configure)


if __name__ == "__main__":
    unittest.main()
