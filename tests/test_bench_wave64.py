"""bench_wave64.py: a configuration that fails after its first process is a failure, not a configuration that ran.
The builds and GPU runs are replaced by stubs, so this needs no GPU, LLVM or Vulkan SDK."""
import contextlib
import fnmatch
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tools")]
import bench_wave64  # noqa: E402
import rdna2spirv  # noqa: E402


def bench(fail):
    """Run main() with 2 processes per configuration; `fail` maps a configuration name to the process that fails."""
    calls = {}

    def run(tools, spv, groups, reps, driver, emulate):
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
    def test_missing_binary_reports_failure(self):
        with patch.object(bench_wave64, "build", lambda work, name, emulate, llvm_mc: "x.spv"), \
                patch.object(bench_wave64, "segments", lambda spv: None), \
                patch.object(bench_wave64.subprocess, "run", side_effect=FileNotFoundError(2, "No such file")), \
                patch.object(sys, "argv", ["bench_wave64.py", "--procs", "2"]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            code = bench_wave64.main()
        self.assertEqual(code, 1)
        self.assertIn("failed or unsupported", out.getvalue())


class MismatchCount(unittest.TestCase):
    """run() reads vk_compute_test's mismatch count as a number: a count that ends in 0 is not zero."""
    BENCH = ("device: stub GPU (subgroup 32..64)\nbench ms: median=1.500 min=1.250 max=2.000\n"
             "bench out fnv1a: 1a2b3c4d\n")

    def run_with(self, tail, returncode=0):
        done = bench_wave64.subprocess.CompletedProcess(["vk_compute_test"], returncode, self.BENCH + tail, "")
        with patch.object(bench_wave64.subprocess, "run", return_value=done):
            return bench_wave64.run("build", "x.spv", 65536, 30, "*nv*", True)

    def test_zero_is_a_result(self):
        self.assertEqual(self.run_with("65536 groups x 64: 0 mismatches\nout[0..7] = 1 2 3 4 5 6 7 8\n"),
                         ((1.5, 1.25), "1a2b3c4d", "stub GPU"))

    def test_counts_ending_in_zero_are_failures(self):
        for count in (10, 20, 100, 4194300):
            with self.subTest(count=count):
                tail = (f"65536 groups x 64: {count} mismatches (first at 64: got 0x00000000)\n"
                        "out[0..7] = 0 0 0 0 0 0 0 0\n")
                timing, why, device = self.run_with(tail)
                self.assertIsNone(timing)
                self.assertIsNone(device)
                self.assertTrue(why.startswith(f"{count} mismatches: "), why)

    def test_other_counts_are_failures(self):
        for count in (1, 7, 65):
            with self.subTest(count=count):
                timing, why, _ = self.run_with(f"65536 groups x 64: {count} mismatches (first at 0: got 0x00000001)\n")
                self.assertIsNone(timing)
                self.assertTrue(why.startswith(f"{count} mismatches: "), why)

    def test_every_count_line_is_read(self):
        # the 2x3x4 workgroup line comes before the per-group line; a clean second line must not hide the first
        timing, why, _ = self.run_with("2x3x4 workgroup Y/Z: 30 mismatches\n65536 groups x 64: 0 mismatches\n")
        self.assertIsNone(timing)
        self.assertTrue(why.startswith("30 mismatches: "), why)
        self.assertEqual(self.run_with("2x3x4 workgroup Y/Z: 0 mismatches\n65536 groups x 64: 0 mismatches\n")[0],
                         (1.5, 1.25))

    def test_no_count_line_is_a_failure(self):
        timing, why, _ = self.run_with("out[0..7] = 1 2 3 4 5 6 7 8\n")
        self.assertIsNone(timing)
        self.assertTrue(why.startswith("no 'mismatches' line: "), why)

    def test_exit_status_comes_first(self):
        timing, why, _ = self.run_with("65536 groups x 64: 10 mismatches (first at 64: got 0x00000000)\n", 1)
        self.assertIsNone(timing)
        self.assertTrue(why.startswith("exit 1: "), why)

    def test_counts(self):
        self.assertEqual(bench_wave64.mismatch_counts("a: 0 mismatches\nb: 10 mismatches (first at 3)\n"), [0, 10])
        self.assertEqual(bench_wave64.mismatch_counts("no count here, mismatches or not\n"), [])


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
