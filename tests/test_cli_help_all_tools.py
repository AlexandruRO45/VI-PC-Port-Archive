"""Every tool with an argparse CLI starts with --help and with -h, under `python -I`.

    python -I tests/test_cli_help_all_tools.py

The tool list is derived from the sources, so a new CLI tool is covered automatically. Tools that need numpy are
skipped where it is not installed (the texture tools are allowed to use it); a tool that fails for any other reason,
or prints a traceback, is a failure. No game data or GPU.
"""
import glob
import importlib.util
import os
import subprocess
import tempfile
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
NUMERIC = {"amd_swizzle", "calibrated_texture", "swizzle_search", "tex_export", "tex_mode_probe", "tile_probe"}
TIMEOUT = 120


def cli_tools():
    found = []
    for path in sorted(glob.glob(os.path.join(TOOLS, "*.py"))):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        if "argparse.ArgumentParser" in text and "__main__" in text:
            found.append(path)
    return found


class CliHelpTests(unittest.TestCase):
    def run_tool(self, path, spelling):
        return subprocess.run([sys.executable, "-I", path, spelling], capture_output=True, text=True,
                              timeout=TIMEOUT, errors="replace")

    def loop(self, spelling):
        paths = cli_tools()
        self.assertGreater(len(paths), 40, "the CLI scan found almost nothing")
        have_numpy = importlib.util.find_spec("numpy") is not None
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if name in NUMERIC and not have_numpy:
                continue
            with self.subTest(tool=name):
                proc = self.run_tool(path, spelling)
                self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("usage:", proc.stdout.lower())

    def test_every_cli_tool_answers_dash_dash_help(self):
        self.loop("--help")

    def loop_from_tempdir(self, spelling):
        paths = cli_tools()
        self.assertGreater(len(paths), 40, "the CLI scan found almost nothing")
        have_numpy = importlib.util.find_spec("numpy") is not None
        with tempfile.TemporaryDirectory() as tmp:
            for path in paths:
                name = os.path.splitext(os.path.basename(path))[0]
                if name in NUMERIC and not have_numpy:
                    continue
                with self.subTest(tool=name):
                    proc = subprocess.run([sys.executable, "-I", path, spelling], capture_output=True,
                                          text=True, timeout=TIMEOUT, errors="replace", cwd=tmp)
                    self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
                    self.assertNotIn("Traceback", proc.stderr)
                    self.assertIn("usage:", proc.stdout.lower())

    def test_help_works_from_an_unrelated_cwd(self):
        self.loop_from_tempdir("--help")

    def test_every_cli_tool_answers_dash_h(self):
        self.loop("-h")



if __name__ == "__main__":
    unittest.main()
