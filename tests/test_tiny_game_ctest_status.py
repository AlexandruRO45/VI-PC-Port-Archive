"""Exercise the tiny_game CTest script with a synthetic loader process."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


CHECK = Path(__file__).resolve().parents[1] / "runtime/test/check_tiny_game.cmake"
CMAKE = shutil.which("cmake")


@unittest.skipUnless(CMAKE, "CMake is required to exercise the tiny_game check")
class TinyGameCTestStatusTests(unittest.TestCase):
    def test_requires_successful_loader_and_guest(self):
        success = "tiny_game: OK\ngame exited with code 0\n"
        for name, output, exit_code, passed in (
            ("success", success, 0, True),
            ("missing guest verdict", "game exited with code 0\n", 0, False),
            ("failed guest", "tiny_game: FAILED\ngame exited with code 1\n", 0, False),
            ("nonzero exit after success marker", success, 1, False),
            ("missing loader verdict", "tiny_game: OK\n", 0, False),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                loader = root / "loader.py"
                loader.write_text(
                    "import sys\n"
                    "assert sys.argv[1:] == ['--run', 'tiny_game.exe', '--', 'hello', '42']\n"
                    f"sys.stdout.write({output!r})\n"
                    f"raise SystemExit({exit_code})\n",
                    encoding="utf-8",
                    newline="\n",
                )
                result = subprocess.run(
                    [CMAKE, f"-DRUNNER={sys.executable}", f"-DLOADER={loader}",
                     f"-DGUEST={root / 'tiny_game.exe'}", "-P", str(CHECK)],
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode == 0, passed, result.stdout + result.stderr)
                if not passed:
                    self.assertIn(f"loader exit {exit_code}", result.stderr)


if __name__ == "__main__":
    unittest.main()
