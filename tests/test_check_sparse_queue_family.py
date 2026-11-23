"""Tests for runtime/test/check_sparse_queue_family.py (no GPU or Vulkan SDK required)."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "runtime" / "test" / "check_sparse_queue_family.py"

SPEC = importlib.util.spec_from_file_location("check_sparse_queue_family", SCRIPT_PATH)
csqf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(csqf)


def make_trace(concurrent=True, count=6):
    mode = "VK_SHARING_MODE_CONCURRENT" if concurrent else "VK_SHARING_MODE_EXCLUSIVE"
    lines = []
    for _ in range(count):
        lines.append(csqf.SPARSE_FLAGS)
        lines.append(f"    sharingMode: VkSharingMode = {mode} (1)")
        if concurrent:
            lines.append("    queueFamilyIndexCount: uint32_t = 2")
            lines.append("    pQueueFamilyIndices[0]: const uint32_t = 0")
            lines.append("    pQueueFamilyIndices[1]: const uint32_t = 1")
        else:
            lines.append("    queueFamilyIndexCount: uint32_t = 0")
        lines.extend(["    padding: uint32_t = 0"] * 12)
    return "\n".join(lines)


class CheckSparseQueueFamilyTests(unittest.TestCase):
    def test_check_trace_accepts_valid_concurrent_records(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "trace.log"
            log.write_text(make_trace(concurrent=True, count=6), encoding="utf-8")
            calls, infos = csqf.check_trace(log, expect_exclusive=False)
            self.assertEqual(calls, 3)
            self.assertEqual(infos, 6)

    def test_check_trace_accepts_exclusive_in_negative_control_mode(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "trace.log"
            log.write_text(make_trace(concurrent=False, count=6), encoding="utf-8")
            calls, infos = csqf.check_trace(log, expect_exclusive=True)
            self.assertEqual(calls, 3)
            self.assertEqual(infos, 6)

    def test_check_trace_rejects_exclusive_in_normal_mode(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "trace.log"
            log.write_text(make_trace(concurrent=False, count=6), encoding="utf-8")
            with self.assertRaises(AssertionError) as ctx:
                csqf.check_trace(log, expect_exclusive=False)
            self.assertIn("not all sparse buffers share active families", str(ctx.exception))

    def test_check_trace_rejects_insufficient_records(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "trace.log"
            log.write_text(make_trace(concurrent=True, count=4), encoding="utf-8")
            with self.assertRaises(AssertionError) as ctx:
                csqf.check_trace(log, expect_exclusive=False)
            self.assertIn("insufficient sparse VkBuffer create-info records", str(ctx.exception))

    def test_missing_executable_exits_2(self):
        missing = Path(ROOT) / "nonexistent_executable.exe"
        r = subprocess.run([sys.executable, "-I", str(SCRIPT_PATH), str(missing)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("ERROR: missing executable", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_instance_layers_order_puts_validation_ahead_of_api_dump(self):
        # Issue #5804: VK_LAYER_KHRONOS_validation must precede VK_LAYER_LUNARG_api_dump
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        expected_layer_spec = 'os.pathsep.join(["VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_api_dump"])'
        self.assertIn(expected_layer_spec, source)


if __name__ == "__main__":
    unittest.main()
