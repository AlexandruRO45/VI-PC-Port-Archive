"""A missing or empty SPIR-V path is one ERROR line from vk_harness, not SIGSEGV.

    python -I tests/test_vk_harness_missing_spv.py

The harness must have been built; set LEONIDA_VK_HARNESS to its path or put it in one of
the usual build folders. The test skips when no binary is available. No game files, and
the bad paths are hand-written.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def harness():
    exe = ".exe" if os.name == "nt" else ""
    for path in (
        os.environ.get("LEONIDA_VK_HARNESS"),
        ROOT / "runtime" / "build" / "linux" / f"vk_harness{exe}",
        ROOT / "runtime" / "build" / "windows" / f"vk_harness{exe}",
        ROOT / "runtime" / "build" / f"vk_harness{exe}",
    ):
        if path and Path(path).is_file():
            return str(path)
    return None


class MissingSpvTests(unittest.TestCase):
    def setUp(self):
        self.exe = harness()
        if not self.exe:
            self.skipTest("no built vk_harness (set LEONIDA_VK_HARNESS)")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = os.path.join(self.tmp.name, "out.raw")
        self.empty = os.path.join(self.tmp.name, "empty.spv")
        Path(self.empty).write_bytes(b"")

    def run_harness(self, vs, fs):
        return subprocess.run([self.exe, vs, fs, self.out], capture_output=True,
                              text=True, errors="replace", timeout=60)

    def test_missing_path_is_an_error_not_a_crash(self):
        missing = os.path.join(self.tmp.name, "no-such.spv")
        result = self.run_harness(missing, missing)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("ERROR: cannot read SPIR-V module:", result.stderr)
        self.assertIn("no-such.spv", result.stderr)
        self.assertFalse(Path(self.out).exists())

    def test_empty_module_is_an_error_not_a_crash(self):
        result = self.run_harness(self.empty, self.empty)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("ERROR: cannot read SPIR-V module:", result.stderr)


if __name__ == "__main__":
    unittest.main()
