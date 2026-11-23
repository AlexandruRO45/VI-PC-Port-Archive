"""Synthetic tests for runtime/test/check_sparse_queue_family.py (TEST-1089).

The contract checker runs the GPU test twice, once under VK_LAYER_LUNARG_api_dump and once under
VK_LAYER_KHRONOS_validation, because the two together crash in api_dump's vkDestroyDevice on the AMD
Windows driver. These tests check that each run loads exactly one of the two layers and that a crash,
a skip or a missing case is still reported. No GPU or game data: the "GPU test" is a stub script.

Run:
    python -I tests/test_check_sparse_queue_family.py
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_sparse_queue_family",
                                              ROOT / "runtime" / "test" / "check_sparse_queue_family.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

PASS_OUT = "".join(f"RESULT LAYER_MAP_{kind}_COPY_COMPUTE PASS mapped_mismatch=0/65536 unmapped_nonzero=0/65536\n"
                   for kind in ("DIRECT", "COPY", "COMPUTE")) + "LAYER_FINAL PASS\n"


class Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class PassEnvsTest(unittest.TestCase):
    def test_each_run_loads_one_layer(self) -> None:
        base = {"PATH": "x", "VK_INSTANCE_LAYERS": "VK_LAYER_LUNARG_api_dump", "LEONIDA_VK_VALIDATION": "1"}
        dump, validation = checker.pass_envs(base, Path("settings"))
        self.assertEqual(dump["VK_INSTANCE_LAYERS"], "VK_LAYER_LUNARG_api_dump")
        self.assertEqual(dump["LEONIDA_VK_VALIDATION"], "0")
        self.assertEqual(dump["VK_LAYER_SETTINGS_PATH"], "settings")
        self.assertNotIn("VK_INSTANCE_LAYERS", validation)
        self.assertEqual(validation["LEONIDA_VK_VALIDATION"], "1")
        self.assertEqual(dump["PATH"], "x")
        self.assertEqual(validation["PATH"], "x")

    def test_base_environment_is_not_changed(self) -> None:
        base = {"LEONIDA_VK_VALIDATION": "1"}
        checker.pass_envs(base, Path("s"))
        self.assertEqual(base, {"LEONIDA_VK_VALIDATION": "1"})


class RunPassTest(unittest.TestCase):
    def run_with(self, result: Completed) -> tuple[str, int]:
        with mock.patch.object(checker.subprocess, "run", return_value=result):
            return checker.run_pass(Path("t.exe"), {}, Path("."))

    def test_all_cases_pass(self) -> None:
        self.assertEqual(self.run_with(Completed(0, PASS_OUT))[1], 0)

    def test_skip_code_is_a_skip(self) -> None:
        self.assertEqual(self.run_with(Completed(77))[1], 77)

    def test_missing_layer_is_a_skip(self) -> None:
        self.assertEqual(self.run_with(Completed(1, stderr="[rt] d3d12: vkCreateInstance (Vulkan 1.3) failed"))[1], 77)

    def test_access_violation_fails(self) -> None:
        with self.assertRaises(AssertionError):
            self.run_with(Completed(0xC0000005, PASS_OUT))

    def test_missing_case_fails(self) -> None:
        with self.assertRaises(AssertionError):
            self.run_with(Completed(0, PASS_OUT.replace("LAYER_MAP_COPY", "LAYER_MAP_OTHER")))


class MainTest(unittest.TestCase):
    def test_validation_errors_fail_the_check(self) -> None:
        calls = []

        def fake_run(argv, cwd, env, **kwargs):
            calls.append(dict(env))
            if env.get("LEONIDA_VK_VALIDATION") == "1":
                return Completed(0, PASS_OUT + "Validation Error: [ VUID-vkCmdDraw-None-02699 ] synthetic\n")
            return Completed(0, PASS_OUT)

        with tempfile.TemporaryDirectory() as td:
            exe = Path(td) / "t.exe"
            exe.write_bytes(b"")
            with mock.patch.object(checker.subprocess, "run", side_effect=fake_run), \
                 mock.patch.object(sys, "argv", ["check", str(exe)]):
                with self.assertRaises(AssertionError):
                    checker.main()
        self.assertEqual(len(calls), 2)
        self.assertEqual([c.get("VK_INSTANCE_LAYERS") for c in calls], ["VK_LAYER_LUNARG_api_dump", None])


if __name__ == "__main__":
    unittest.main()
