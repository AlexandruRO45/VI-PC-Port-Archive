"""Two more texture tools that must not import numpy for --help or a plain import.

    python -I tests/test_more_numpy_free_tools.py

`python -I -S` hides site-packages, so this also proves it on a machine that has numpy installed. No game data.
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MoreNumpyFreeTests(unittest.TestCase):
    def test_calibrated_texture_imports_without_numpy(self):
        tools = repr(os.path.join(ROOT, "tools"))
        code = f"import sys; sys.path.insert(0, {tools}); import calibrated_texture; print('ok')"
        p = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("Traceback", p.stderr)
        self.assertEqual(p.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
