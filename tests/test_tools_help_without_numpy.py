"""Tools whose decode needs numpy but whose --help must work without it.

    python -I tests/test_tools_help_without_numpy.py

The texture tools import numpy at module level today, so on a stdlib-only checkout (CI's standard-library job) even
`--help` dies with ModuleNotFoundError instead of printing usage. Run the tools with -S here so the test also proves
it on a machine that has numpy installed. No game data; --help reads no texture.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = ("amd_swizzle", "tex_mode_probe")
MODULES = ("bc_feature",)


class HelpWithoutNumpyTests(unittest.TestCase):
    def test_help_works_when_site_packages_are_unavailable(self):
        for tool in TOOLS:
            with self.subTest(tool=tool):
                path = os.path.join(ROOT, "tools", tool + ".py")
                p = subprocess.run([sys.executable, "-I", "-S", path, "--help"],
                                   capture_output=True, text=True, timeout=60)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertNotIn("Traceback", p.stderr, p.stderr)
                self.assertIn("usage:", p.stdout)

    def test_the_module_imports_when_site_packages_are_unavailable(self):
        tools = repr(os.path.join(ROOT, "tools"))
        code = f"import sys; sys.path.insert(0, {tools}); import bc_feature; print('ok')"
        p = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("Traceback", p.stderr, p.stderr)
        self.assertEqual(p.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
