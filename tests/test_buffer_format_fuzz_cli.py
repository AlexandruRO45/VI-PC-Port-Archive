"""CPU-only tests for tools/buffer_format_fuzz.py's command line: an explicit --formats is checked before the GPU.

    python -I tests/test_buffer_format_fuzz_cli.py

A typo in --formats used to be found only after runner.open_oracle(), so a machine without the oracle answered
"hardware oracle unavailable" (77) and one with it raised SystemExit (1). It is a usage error (2) before the oracle.
"""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "buffer_format_fuzz.py")


def run(*args):
    return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, check=False)


def load_tool():
    spec = importlib.util.spec_from_file_location("buffer_format_fuzz", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BufferFormatFuzzFormatsTests(unittest.TestCase):
    def test_unknown_format_is_a_usage_error_before_the_oracle(self):
        proc = run("--oracle", "hw", "--formats", "8_8_8_8_UNORM_typo")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("unknown format '8_8_8_8_UNORM_typo'", proc.stderr)
        self.assertNotIn("hardware oracle", proc.stdout + proc.stderr)

    def test_a_code_outside_the_table_is_refused_too(self):
        proc = run("--oracle", "hw", "--formats", "0,8_8_8_8_UNORM")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("unknown format '0'", proc.stderr)

    def test_known_names_and_codes_map_to_the_table(self):
        tool = load_tool()
        pairs = tool.select_formats("8_8_8_8_UNORM,56")
        self.assertEqual([n for _, n in pairs], ["8_8_8_8_UNORM", tool.GFX10_NAMES[56]])
        with self.assertRaises(ValueError):
            tool.select_formats("not_a_format")


class HardwareFormatSelectionTests(unittest.TestCase):
    def compare(self, arch, *args):
        tool = load_tool()
        oracle = Mock(arch=arch, device="synthetic GPU")
        # A nonempty batch yields a visible failure for every kernel.
        oracle.run.side_effect = lambda kernels, words: ["synthetic mismatch"] * len(kernels)
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = [TOOL, "--oracle", "hw", "--mode", "load", "--path", "mubuf", "--budget", "0", *args]
        with patch("hw_oracle.runner.open_oracle", return_value=oracle), patch.object(sys, "argv", argv), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            status = tool.main()
        oracle.close.assert_called_once_with()
        return status, stdout.getvalue(), stderr.getvalue(), oracle.run

    def test_gfx11_skips_an_entirely_unsupported_selection(self):
        status, out, err, dispatch = self.compare("gfx1100", "--formats", "30")
        self.assertEqual(status, 77)
        dispatch.assert_not_called()
        self.assertIn("gfx11", err.lower())
        self.assertIn("30", err)
        self.assertNotIn("validated elements", out)

    def test_mixed_gfx11_selection_runs_only_the_supported_format(self):
        status, out, _, dispatch = self.compare("gfx1100", "--formats", "30,56")
        self.assertEqual(status, 1)
        dispatch.assert_called_once()
        rows = [line.split() for line in out.splitlines() if line.strip().startswith("load mubuf")]
        self.assertEqual([row[2] for row in rows], ["56"])
        self.assertIn("synthetic mismatch", out)

    def test_negative_control_uses_its_own_formats(self):
        status, out, _, dispatch = self.compare("gfx1100", "--formats", "30", "--negative-control")
        self.assertEqual(status, 0)
        dispatch.assert_called_once()
        self.assertIn("wrong V# format", out)
        self.assertNotIn("NOT CAUGHT", out)


if __name__ == "__main__":
    unittest.main()
