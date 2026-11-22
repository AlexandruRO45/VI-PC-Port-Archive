"""Tests for runtime/test/check_texture_readback_sync.cmake.

    python -I tests/test_check_texture_readback_sync_cmake.py

No GPU, Vulkan loader or game data: exercises the CMake script with synthetic
fake tools and runners to verify optional -DRUNNER passthrough, environment
variables, arguments, working directory and validation-log failure checks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "runtime" / "test" / "check_texture_readback_sync.cmake"

WINDOWS_CMAKE_CANDIDATES = (
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
         r"\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community"
         r"\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional"
         r"\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
         r"\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"),
    Path(r"C:\Program Files\CMake\bin\cmake.exe"),
)

FAKE_TOOL_SOURCE = """\
import json
import os
import sys

record_path = os.environ.get("TEST_RECORD_PATH")
if record_path:
    payload = {
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "isolated": sys.flags.isolated,
        "env": {
            "VK_INSTANCE_LAYERS": os.environ.get("VK_INSTANCE_LAYERS"),
            "VK_LOADER_DEBUG": os.environ.get("VK_LOADER_DEBUG"),
            "VK_LAYER_ENABLES": os.environ.get("VK_LAYER_ENABLES"),
            "TEST_RUNNER_MARKER": os.environ.get("TEST_RUNNER_MARKER"),
        },
    }
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

stdout_text = os.environ.get("TEST_STDOUT", "")
stderr_text = os.environ.get("TEST_STDERR", "")
if stdout_text:
    sys.stdout.write(stdout_text)
    sys.stdout.flush()
if stderr_text:
    sys.stderr.write(stderr_text)
    sys.stderr.flush()

sys.exit(int(os.environ.get("TEST_EXIT_CODE", "0")))
"""

FAKE_RUNNER_PY_SOURCE = """\
import json
import os
import subprocess
import sys

record_path = os.environ.get("TEST_RUNNER_RECORD_PATH")
if record_path:
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump({"runner_argv": sys.argv[1:]}, handle)

env = os.environ.copy()
env["TEST_RUNNER_MARKER"] = "from_fake_runner"
result = subprocess.run([sys.executable, "-I", *sys.argv[1:]], env=env)
sys.exit(result.returncode)
"""

PASSING_STDOUT = "vk_texture_test cube: PASSED\n"
PASSING_STDERR = 'LAYER: Insert instance layer "VK_LAYER_KHRONOS_validation"\n'


def find_cmake() -> str | None:
    found = shutil.which("cmake")
    if found:
        return found
    if os.name == "nt":
        for candidate in WINDOWS_CMAKE_CANDIDATES:
            if candidate.is_file():
                return str(candidate)
    return None


def norm_realpath(path: str | Path) -> str:
    return os.path.normcase(os.path.realpath(str(path)))


CMAKE = find_cmake()


@unittest.skipUnless(CMAKE, "cmake not available on PATH or in standard Visual Studio BuildTools locations")
class CheckTextureReadbackSyncCMakeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.spirv_dir = self.root / "spirv"
        self.spirv_dir.mkdir()
        self.output_file = self.root / "readback out.bin"
        self.record_file = self.root / "record.json"
        self.runner_record_file = self.root / "runner_record.json"

        self.fake_tool_py = self.root / "fake_tool.py"
        self.fake_tool_py.write_text(FAKE_TOOL_SOURCE, encoding="utf-8")

        self.fake_runner_py = self.root / "fake_runner.py"
        self.fake_runner_py.write_text(FAKE_RUNNER_PY_SOURCE, encoding="utf-8")

        self.direct_tool = self._make_direct_executable(self.root / "fake_vk_texture_test", self.fake_tool_py)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_direct_executable(self, stem: Path, target_py: Path) -> Path:
        if os.name == "nt":
            wrapper = stem.with_suffix(".cmd")
            wrapper.write_text(
                f'@echo off\r\n"{sys.executable}" -I "{target_py}" %*\r\nexit /b %ERRORLEVEL%\r\n',
                encoding="utf-8",
            )
            return wrapper
        wrapper = stem.with_suffix(".sh")
        wrapper.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" -I "{target_py}" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return wrapper

    def _run_script(
        self,
        *,
        tool: Path,
        runner: str | None = None,
        output_arg: str | Path | None = None,
        stdout_text: str = PASSING_STDOUT,
        stderr_text: str = PASSING_STDERR,
        exit_code: int = 0,
    ) -> subprocess.CompletedProcess[str]:
        output_value = str(self.output_file if output_arg is None else output_arg)
        cmd = [
            CMAKE,
            f"-DTOOL={tool}",
            f"-DSPIRV={self.spirv_dir}",
            f"-DOUTPUT={output_value}",
        ]
        if runner is not None:
            cmd.append(f"-DRUNNER={runner}")
        cmd.extend(["-P", str(SCRIPT)])

        env = os.environ.copy()
        env["TEST_RECORD_PATH"] = str(self.record_file)
        env["TEST_RUNNER_RECORD_PATH"] = str(self.runner_record_file)
        env["TEST_STDOUT"] = stdout_text
        env["TEST_STDERR"] = stderr_text
        env["TEST_EXIT_CODE"] = str(exit_code)
        env.pop("TEST_RUNNER_MARKER", None)
        # Clear Vulkan vars beforehand so we confirm the CMake script sets them.
        for key in ("VK_INSTANCE_LAYERS", "VK_LOADER_DEBUG", "VK_LAYER_ENABLES"):
            env.pop(key, None)

        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env=env, timeout=30)

    def _assert_common_invocation(self, record: dict, expected_output: str | Path | None = None) -> None:
        output_value = str(self.output_file if expected_output is None else expected_output)
        self.assertEqual(record["argv"], ["cube", output_value])
        self.assertEqual(norm_realpath(record["cwd"]), norm_realpath(self.spirv_dir))
        self.assertEqual(record["env"]["VK_INSTANCE_LAYERS"], "VK_LAYER_KHRONOS_validation")
        self.assertEqual(record["env"]["VK_LOADER_DEBUG"], "layer")
        self.assertEqual(
            record["env"]["VK_LAYER_ENABLES"],
            "VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT",
        )

    def test_runner_omitted_invokes_tool_directly(self):
        # Use a space-free output name here so the Windows .cmd wrapper stays
        # immune to cmd.exe /c outer-quote stripping even if %TEMP% has spaces.
        result = self._run_script(tool=self.direct_tool, runner=None, output_arg="readback.bin")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self._assert_common_invocation(record, expected_output="readback.bin")
        self.assertIsNone(record["env"]["TEST_RUNNER_MARKER"])

    def test_runner_empty_invokes_tool_directly(self):
        result = self._run_script(tool=self.direct_tool, runner="", output_arg="readback.bin")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self._assert_common_invocation(record, expected_output="readback.bin")
        self.assertIsNone(record["env"]["TEST_RUNNER_MARKER"])

    def test_single_element_runner_runs_before_tool(self):
        result = self._run_script(tool=self.fake_tool_py, runner=sys.executable)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self._assert_common_invocation(record)

    def test_runner_paths_with_spaces_are_not_split(self):
        spaced_dir = self.root / "runner with spaces"
        spaced_dir.mkdir()
        spaced_runner_py = spaced_dir / "fake_runner.py"
        spaced_runner_py.write_text(FAKE_RUNNER_PY_SOURCE, encoding="utf-8")

        # CMake list expansion splits ${RUNNER} on ';' only, preserving spaces in each item.
        runner_list = f"{sys.executable};-I;{spaced_runner_py}"
        result = self._run_script(tool=self.fake_tool_py, runner=runner_list)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self._assert_common_invocation(record)
        self.assertEqual(record["env"]["TEST_RUNNER_MARKER"], "from_fake_runner")

        runner_record = json.loads(self.runner_record_file.read_text(encoding="utf-8"))
        self.assertEqual(runner_record["runner_argv"], [str(self.fake_tool_py), "cube", str(self.output_file)])

        if os.name != "nt":
            self.record_file.unlink()
            self.runner_record_file.unlink()
            runner_exe = self._make_direct_executable(spaced_dir / "emu_runner", spaced_runner_py)
            result = self._run_script(tool=self.fake_tool_py, runner=str(runner_exe))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            record = json.loads(self.record_file.read_text(encoding="utf-8"))
            self._assert_common_invocation(record)
            self.assertEqual(record["env"]["TEST_RUNNER_MARKER"], "from_fake_runner")

    def test_multi_element_semicolon_runner_preserves_arguments_and_order(self):
        # First verify python;-I sets sys.flags.isolated == 1 directly on the tool.
        result = self._run_script(tool=self.fake_tool_py, runner=f"{sys.executable};-I")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self._assert_common_invocation(record)
        self.assertEqual(record["isolated"], 1)

        # Then verify a 3-element runner list prepends all elements before "${TOOL}" cube "${OUTPUT}".
        self.record_file.unlink()
        runner_list = f"{sys.executable};-I;{self.fake_runner_py}"
        result = self._run_script(tool=self.fake_tool_py, runner=runner_list)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        record = json.loads(self.record_file.read_text(encoding="utf-8"))
        self._assert_common_invocation(record)
        self.assertEqual(record["env"]["TEST_RUNNER_MARKER"], "from_fake_runner")
        runner_record = json.loads(self.runner_record_file.read_text(encoding="utf-8"))
        self.assertEqual(runner_record["runner_argv"], [str(self.fake_tool_py), "cube", str(self.output_file)])

    def test_failure_detection(self):
        cases = (
            ("nonzero_exit", PASSING_STDOUT, PASSING_STDERR, 1),
            ("missing_passed_line", "vk_texture_test cube: something else\n", PASSING_STDERR, 0),
            ("missing_validation_layer_line", PASSING_STDOUT, "LAYER: some other layer\n", 0),
            ("sync_hazard_on_stderr", PASSING_STDOUT,
             PASSING_STDERR + "SYNC-HAZARD-READ-AFTER-WRITE detected\n", 0),
            ("validation_error_on_stdout",
             PASSING_STDOUT + "Validation Error: [ VUID-test ] broken barrier\n", PASSING_STDERR, 0),
        )
        for name, stdout_text, stderr_text, exit_code in cases:
            with self.subTest(case=name):
                result = self._run_script(
                    tool=self.fake_tool_py,
                    runner=f"{sys.executable};-I",
                    stdout_text=stdout_text,
                    stderr_text=stderr_text,
                    exit_code=exit_code,
                )
                self.assertNotEqual(result.returncode, 0, f"expected failure for {name}")
                self.assertIn("Texture draw with synchronization validation returned",
                              result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
