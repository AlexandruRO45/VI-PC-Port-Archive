"""Exercise the synthetic scene runner's output setup before a guest is launched."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_guest_scene.py"


class OutputSetup(unittest.TestCase):
    def test_uncreatable_output_and_occupied_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "spirv").mkdir()
            sentinel = root / "report.json"
            sentinel.write_bytes(b"previous run\n")
            command = [sys.executable, "-I", str(RUNNER), "--loader", str(sentinel),
                       "--guest", str(sentinel), "--shaders", str(root), "--wine", sys.executable]
            for destination, expected_rc in ((sentinel / "new output", 1), (root, 2)):
                with self.subTest(destination=destination):
                    result = subprocess.run(command + ["--out", str(destination)], cwd=root,
                                            capture_output=True, timeout=20)
                    self.assertEqual(result.returncode, expected_rc, result.stderr)
                    self.assertEqual(result.stdout, b"")
                    self.assertIn(b"--out", result.stderr)
                    self.assertNotIn(b"Traceback", result.stderr)
                    self.assertEqual(sentinel.read_bytes(), b"previous run\n")
                    self.assertEqual(set(root.iterdir()), {sentinel, root / "spirv"})


if __name__ == "__main__":
    unittest.main()
