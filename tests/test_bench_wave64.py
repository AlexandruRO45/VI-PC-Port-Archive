"""bench_wave64.py: a configuration that fails after its first process is a failure, not a configuration that ran.
The builds and GPU runs are replaced by stubs, so this needs no GPU, LLVM or Vulkan SDK."""
import contextlib
import fnmatch
import io
import os
from pathlib import Path
from subprocess import CompletedProcess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tools")]
import bench_wave64  # noqa: E402
import rdna2spirv  # noqa: E402


def bench(fail):
    """Run main() with 2 processes per configuration; `fail` maps a configuration name to the process that fails."""
    calls = {}

    def run(tools, spv, groups, reps, driver, emulate, **kwargs):
        name = next(n for n, d, e in bench_wave64.CONFIGS if d == driver and e == emulate)
        calls[name] = calls.get(name, 0) + 1
        if fail.get(name) == calls[name]:
            return None, "exit 1: device lost", None
        return (1.0, 0.9), "same", "stub GPU"

    out = io.StringIO()
    with patch.object(bench_wave64, "build", lambda work, name, emulate, llvm_mc: f"{name}-{emulate}.spv"), \
            patch.object(bench_wave64, "segments", lambda spv: None), \
            patch.object(bench_wave64, "run", run), \
            patch.object(sys, "argv", ["bench_wave64.py", "--procs", "2"]), \
            contextlib.redirect_stdout(out):
        code = bench_wave64.main()
    return code, out.getvalue()


class PartialFailure(unittest.TestCase):
    def test_every_process_ran(self):
        code, out = bench({})
        self.assertEqual(code, 0, out)
        self.assertIn("65535 workgroups", out)
        self.assertIn("emulated output equals the reference", out)

    def test_emulated_config_fails_on_its_second_process(self):
        code, out = bench({"NVIDIA --wave64-emulate": 2})
        self.assertEqual(code, 1, out)
        self.assertIn("NVIDIA --wave64-emulate: failed or unsupported", out)
        self.assertIn("incomplete comparison: no complete result from NVIDIA --wave64-emulate", out)
        self.assertNotIn("emulated output equals the reference", out)

    def test_reference_fails_on_its_second_process(self):
        code, out = bench({"AMD native": 2})
        self.assertEqual(code, 1, out)
        self.assertIn("no reference output from AMD native", out)


class HarnessError(unittest.TestCase):
    def test_harness_is_found_in_a_cmake_configuration_subdirectory(self):
        executable = "vk_compute_test.exe" if os.name == "nt" else "vk_compute_test"
        output = "bench ms: median=1.0 min=0.9 max=1.1\nbench out fnv1a: abc\n4 groups x 64: 0 mismatches\n"
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "RelWithDebInfo" / executable
            harness.parent.mkdir()
            harness.touch()
            argv = ["bench_wave64.py", "--build", tmp, "--procs", "1"]
            with patch.object(bench_wave64, "build", return_value="test.spv"), \
                    patch.object(bench_wave64, "segments", return_value=None), \
                    patch.object(bench_wave64.subprocess, "run") as launch, \
                    patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                launch.return_value.stdout = output
                launch.return_value.stderr = ""
                launch.return_value.returncode = 0
                self.assertEqual(bench_wave64.main(), 0)
            self.assertEqual([call.args[0][0] for call in launch.call_args_list], [str(harness)] * 3)

    def test_missing_binary_reports_failure(self):
        with patch.object(bench_wave64, "build", lambda work, name, emulate, llvm_mc: "x.spv"), \
                patch.object(bench_wave64, "segments", lambda spv: None), \
                patch.object(bench_wave64.subprocess, "run", side_effect=FileNotFoundError(2, "No such file")), \
                patch.object(sys, "argv", ["bench_wave64.py", "--procs", "2"]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = bench_wave64.main()
        self.assertEqual(code, 1)
        self.assertIn("failed or unsupported", out.getvalue())


class ArgumentValidation(unittest.TestCase):
    def test_positive_only(self):
        for option in ("--groups", "--reps", "--procs"):
            for value in ("0", "-1"):
                with self.subTest(option=option, value=value), \
                        patch.object(sys, "argv", ["bench_wave64.py", option, value]), \
                        patch.object(bench_wave64, "build") as build:
                    with self.assertRaises(SystemExit) as e:
                        bench_wave64.main()
                    self.assertEqual(e.exception.code, 2)
                    build.assert_not_called()


class DriverSelectors(unittest.TestCase):
    def run_with_device_filter(self, requested, *, amd_error=None):
        calls = []

        def launch(args, *, env, **kwargs):
            driver = env["VK_LOADER_DRIVERS_SELECT"]
            device_filter = env.get("LEONIDA_VK_DEVICE")
            native = "--require-wave64" in args
            is_amd = driver == "*amd*,*radeon*"
            calls.append((driver, native, device_filter))
            devices = ("AMD discrete", "AMD integrated") if is_amd else ("NVIDIA discrete",)
            device = device_filter if device_filter else devices[0]
            if device not in devices:
                return CompletedProcess(args, 1, "", f"LEONIDA_VK_DEVICE={device.lower()}: no Vulkan physical device matches\n")
            if amd_error and is_amd:
                return CompletedProcess(args, 1, "", amd_error + "\n")
            output = (f"device: {device} (subgroup 32..64)\n"
                      "bench ms: median=1.0 min=0.9 max=1.1\nbench out fnv1a: abc\n0 mismatches\n")
            return CompletedProcess(args, 0, output, "")

        out = io.StringIO()
        with patch.dict(os.environ, {"LEONIDA_VK_DEVICE": requested}, clear=True), \
                patch.object(bench_wave64, "build", return_value="test.spv"), \
                patch.object(bench_wave64, "segments", return_value=None), \
                patch.object(bench_wave64.subprocess, "run", side_effect=launch), \
                patch.object(sys, "argv", ["bench_wave64.py", "--procs", "2"]), \
                contextlib.redirect_stdout(out):
            code = bench_wave64.main()
            self.assertEqual(os.environ["LEONIDA_VK_DEVICE"], requested)
        return code, calls, out.getvalue()

    def test_opposite_vendor_retries_without_dropping_valid_same_vendor_filter(self):
        code, calls, out = self.run_with_device_filter("AMD integrated")
        self.assertEqual(code, 0, out)
        amd = "*amd*,*radeon*"
        self.assertEqual(calls, [(amd, True, "AMD integrated")] * 2
                         + [(amd, False, "AMD integrated")] * 2
                         + [("*nv*", False, "AMD integrated"), ("*nv*", False, None), ("*nv*", False, None)])
        self.assertIn("NVIDIA --wave64-emulate: no match for LEONIDA_VK_DEVICE", out)
        self.assertIn("AMD native                 AMD integrated", out)
        self.assertIn("AMD --wave64-emulate       AMD integrated", out)
        self.assertIn("emulated output equals the reference", out)

    def test_filter_that_matches_no_configuration_is_an_error(self):
        code, _, out = self.run_with_device_filter("missing device")
        self.assertEqual(code, 1, out)
        self.assertIn("no configuration completed using LEONIDA_VK_DEVICE", out)
        self.assertNotIn("emulated output equals the reference", out)

    def test_other_errors_do_not_retry_without_the_filter(self):
        code, calls, out = self.run_with_device_filter("AMD integrated", amd_error="device lost")
        self.assertEqual(code, 1, out)
        self.assertIn("device lost", out)
        self.assertEqual([f for d, _, f in calls if d == "*amd*,*radeon*"], ["AMD integrated"] * 2)

    @staticmethod
    def _matches(selector, manifest):
        return any(fnmatch.fnmatch(manifest, pat) for pat in selector.split(","))

    def test_amd_selector_matches_windows_amdvlk_and_linux_radv(self):
        amd_selectors = [driver for name, driver, _ in bench_wave64.CONFIGS if name.startswith("AMD")]
        self.assertEqual(len(amd_selectors), 2)
        for selector in amd_selectors:
            for manifest in ("amd-vulkan64.json", "amd_icd64.json", "radeon_icd.x86_64.json"):
                self.assertTrue(self._matches(selector, manifest), f"{selector!r} should match {manifest!r}")
            for other in ("nv-vk64.json", "nvidia_icd.json", "igvk64.json", "intel_icd.x86_64.json"):
                self.assertFalse(self._matches(selector, other), f"{selector!r} must not match {other!r}")

    def test_nvidia_selector_matches_windows_and_linux(self):
        nv_selectors = [driver for name, driver, _ in bench_wave64.CONFIGS if name.startswith("NVIDIA")]
        self.assertEqual(len(nv_selectors), 1)
        selector = nv_selectors[0]
        for manifest in ("nv-vk64.json", "nvidia_icd.json"):
            self.assertTrue(self._matches(selector, manifest), f"{selector!r} should match {manifest!r}")
        for other in ("amd-vulkan64.json", "radeon_icd.x86_64.json", "igvk64.json", "intel_icd.x86_64.json"):
            self.assertFalse(self._matches(selector, other), f"{selector!r} must not match {other!r}")


class SchedulerSegments(unittest.TestCase):
    def setUp(self) -> None:
        self.spirv = Path(self.enterContext(tempfile.TemporaryDirectory())) / "shader.spv"

    def measure(self, assembly: str) -> tuple[int, int] | None:
        self.spirv.with_suffix(".spvasm").write_text(assembly, encoding="utf-8", newline="\n")
        return bench_wave64.segments(str(self.spirv))

    def test_unpartitioned_scheduler(self) -> None:
        assembly = "OpSwitch %pcv %sw_default 0 %blk0 1 %blk1 3 %blk3\n"
        self.assertEqual(self.measure(assembly), (3, 1))

    def test_partitioned_scheduler(self) -> None:
        cases = " ".join(f"{case} %blk{case}" for case in (*range(16_383), 16_384))
        assembly = rdna2spirv.split_big_switch(
            "%zero = OpConstant %uint 0\n"
            "OpSelectionMerge %sw_merge None\n"
            f"OpSwitch %pcv %sw_default {cases}\n"
            "%sw_default = OpLabel\n"
            "OpBranch %sw_merge\n"
        )
        self.assertIn("OpSwitch %pcc %sw_default", assembly)
        self.assertEqual(assembly.count("OpSwitch %pcv %swm"), 2)
        self.assertEqual(self.measure(assembly), (16_384, 1))


if __name__ == "__main__":
    unittest.main()
